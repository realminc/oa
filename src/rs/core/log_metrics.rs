//! Structured JSONL scalar metrics sink.
//!
//! Port provenance: `oa/core/log.h` (`LogMetrics` class).
//!
//! Appends newline-delimited JSON records to `{dir}/events.jsonl` for
//! consumption by the OA plot viewer and external analysis tools.
//!
//! # Record format
//!
//! ```json
//! {"tag":"loss","step":100,"value":2.345678,"wall_time":4.321}
//! ```
//!
//! # Usage
//!
//! ```rust,ignore
//! let mut metrics = oa::LogMetrics::new();
//! metrics.open(&oa::Path::var_rel("run/log"))?;
//! metrics.log_scalar("loss", step, loss as f64);
//! metrics.flush();
//! ```

use std::collections::HashMap;
use std::sync::{
	Arc, Mutex,
	atomic::{AtomicBool, Ordering},
};
use std::time::Instant;

use crate::core::filesystem::Filesystem;
use crate::{Path, Result};

// ─── LogMetrics ──────────────────────────────────────────────────────────────

/// Structured JSONL scalar-metrics sink.
///
/// Thread-safe: `log_scalar` and `flush` serialize through an internal mutex.
/// The open flag is an atomic for cheap `is_open` checks on the hot path.
pub struct LogMetrics {
	inner: Arc<Inner>,
}

struct Inner {
	state: Mutex<State>,
	is_open: AtomicBool,
}

struct State {
	log_dir: String,
	events_path: String,
	start: Instant,
	buffer: String,
	buffer_count: usize,
	flush_interval: usize,
}

impl Default for State {
	fn default() -> Self {
		Self {
			log_dir: String::new(),
			events_path: String::new(),
			start: Instant::now(),
			buffer: String::new(),
			buffer_count: 0,
			flush_interval: 16,
		}
	}
}

impl Default for LogMetrics {
	fn default() -> Self {
		Self::new()
	}
}

impl LogMetrics {
	/// Create a closed metrics sink.
	pub fn new() -> Self {
		Self {
			inner: Arc::new(Inner {
				state: Mutex::new(State::default()),
				is_open: AtomicBool::new(false),
			}),
		}
	}

	/// Open (or reopen) the sink, creating `{dir}/events.jsonl`.
	pub fn open(&mut self, dir: &Path) -> Result<()> {
		let mut state = self.inner.state.lock().unwrap();
		// Flush any buffered data from a previous open.
		flush_unlocked(&mut state)?;
		let dir_str = dir.display().to_string();
		Filesystem::create_directories(dir)?;
		state.log_dir = dir_str.clone();
		state.events_path = format!("{dir_str}/events.jsonl");
		state.start = Instant::now();
		self.inner.is_open.store(true, Ordering::Release);
		Ok(())
	}

	/// True while the sink is open.
	pub fn is_open(&self) -> bool {
		self.inner.is_open.load(Ordering::Acquire)
	}

	/// Directory passed to the last successful `open`.
	pub fn log_dir(&self) -> String {
		self.inner.state.lock().unwrap().log_dir.clone()
	}

	/// Append one scalar record.
	///
	/// No-op when the sink is closed.
	pub fn log_scalar(&self, tag: &str, step: i64, value: f64) {
		if !self.is_open() {
			return;
		}
		let mut state = self.inner.state.lock().unwrap();
		if !self.inner.is_open.load(Ordering::Relaxed) {
			return;
		}

		let wall = state.start.elapsed().as_secs_f64();
		let tag_json = json_string(tag);
		let value_str = json_number(value, 6);
		let wall_str = json_number(wall, 3);

		state.buffer.push_str(&format!(
			"{{\"tag\":{tag_json},\"step\":{step},\"value\":{value_str},\"wall_time\":{wall_str}}}\n"
		));
		state.buffer_count += 1;

		let interval = state.flush_interval;
		if state.buffer_count >= interval {
			let _ = flush_unlocked(&mut state);
		}
	}

	/// Append one scalar record per entry in `values`.
	///
	/// Each tag is `"{base_tag}/{name}"`.
	pub fn log_scalars(&self, base_tag: &str, step: i64, values: &HashMap<String, f64>) {
		for (name, &value) in values {
			self.log_scalar(&format!("{base_tag}/{name}"), step, value);
		}
	}

	/// Flush buffered records to disk.
	pub fn flush(&self) -> Result<()> {
		let mut state = self.inner.state.lock().unwrap();
		flush_unlocked(&mut state)
	}

	/// Flush and mark the sink closed.
	pub fn close(&self) -> Result<()> {
		let mut state = self.inner.state.lock().unwrap();
		flush_unlocked(&mut state)?;
		self.inner.is_open.store(false, Ordering::Release);
		Ok(())
	}

	/// Set the auto-flush interval (records). Default: 16.
	pub fn set_flush_interval(&self, n: usize) {
		self.inner.state.lock().unwrap().flush_interval = n.max(1);
	}
}

impl Drop for LogMetrics {
	fn drop(&mut self) {
		if self.is_open() {
			let _ = self.close();
		}
	}
}

// ─── helpers ─────────────────────────────────────────────────────────────────

fn flush_unlocked(state: &mut State) -> Result<()> {
	if state.buffer.is_empty() || state.events_path.is_empty() {
		return Ok(());
	}
	Filesystem::append_text(&Path::from(state.events_path.as_str()), &state.buffer)?;
	state.buffer.clear();
	state.buffer_count = 0;
	Ok(())
}

fn json_string(text: &str) -> String {
	let mut out = String::with_capacity(text.len() + 2);
	out.push('"');
	for byte in text.bytes() {
		match byte {
			b'"' => out.push_str("\\\""),
			b'\\' => out.push_str("\\\\"),
			b'\x08' => out.push_str("\\b"),
			b'\x0c' => out.push_str("\\f"),
			b'\n' => out.push_str("\\n"),
			b'\r' => out.push_str("\\r"),
			b'\t' => out.push_str("\\t"),
			b if b < 0x20 => {
				out.push_str(&format!("\\u{b:04x}"));
			}
			b => out.push(b as char),
		}
	}
	out.push('"');
	out
}

fn json_number(value: f64, precision: usize) -> String {
	if value.is_finite() {
		if precision == 6 {
			// %g-style: use exponential when value is very large/small.
			format!("{value:.6e}")
				.parse::<f64>()
				.map(|_| format!("{value:.6}"))
				.unwrap_or_else(|_| "null".to_owned())
		} else {
			format!("{value:.3}")
		}
	} else {
		"null".to_owned()
	}
}
