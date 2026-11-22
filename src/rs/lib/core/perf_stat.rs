//! Rolling-window performance statistics accumulator.
//!
//! Port provenance: `oa/core/perfStat.h`.
//!
//! Accumulates a configurable sliding window of `f64` samples and computes
//! mean, stddev, min, max, and percentiles. A warmup period discards the first
//! N samples (GPU boost-clock / JIT cache warm-up effects).
//!
//! # Usage
//!
//! ```rust,ignore
//! let mut stat = oa::PerfStat::new("step_ms", 200, 20);
//! stat.push(gpu_ms);
//! if stat.is_ready() {
//!     println!("mean={:.3}ms p95={:.3}ms", stat.mean(), stat.p95());
//! }
//! ```

/// Rolling-window performance statistics.
///
/// No GPU dependency; pure CPU value struct.
pub struct PerfStat {
	name: String,
	window: usize,
	warmup: usize,
	total_count: u64,
	ring: Vec<f64>,
	head: usize,
	filled: usize,
	sum: f64,
	sum_sq: f64,
	last_val: f64,
}

impl PerfStat {
	/// Create a new accumulator.
	///
	/// - `name`   — human-readable label used in diagnostics.
	/// - `window` — rolling window capacity (clamped to 1).
	/// - `warmup` — samples discarded before statistics are valid.
	pub fn new(name: &str, window: usize, warmup: usize) -> Self {
		let window = window.max(1);
		Self {
			name: name.to_owned(),
			window,
			warmup,
			total_count: 0,
			ring: vec![0.0; window],
			head: 0,
			filled: 0,
			sum: 0.0,
			sum_sq: 0.0,
			last_val: 0.0,
		}
	}

	/// Push one sample.
	pub fn push(&mut self, value: f64) {
		self.last_val = value;
		self.total_count += 1;

		if self.total_count <= self.warmup as u64 {
			return; // discard warmup samples
		}

		let slot = self.head % self.window;
		// Evict old sample if the window is full.
		if self.filled == self.window {
			let old = self.ring[slot];
			self.sum -= old;
			self.sum_sq -= old * old;
		} else {
			self.filled += 1;
		}

		self.ring[slot] = value;
		self.sum += value;
		self.sum_sq += value * value;
		self.head = slot + 1;
	}

	/// True when warmup is done and at least one window sample is available.
	pub fn is_ready(&self) -> bool {
		self.filled > 0
	}

	/// Mean of the current window. Valid only when `is_ready()`.
	pub fn mean(&self) -> f64 {
		if self.filled == 0 {
			return 0.0;
		}
		self.sum / self.filled as f64
	}

	/// Population standard deviation of the current window.
	pub fn stddev(&self) -> f64 {
		if self.filled < 2 {
			return 0.0;
		}
		let n = self.filled as f64;
		let var = (self.sum_sq / n) - (self.sum / n).powi(2);
		var.max(0.0).sqrt()
	}

	/// Minimum value in the current window.
	pub fn min(&self) -> f64 {
		self
			.window_slice()
			.iter()
			.cloned()
			.fold(f64::INFINITY, f64::min)
	}

	/// Maximum value in the current window.
	pub fn max(&self) -> f64 {
		self
			.window_slice()
			.iter()
			.cloned()
			.fold(f64::NEG_INFINITY, f64::max)
	}

	/// 50th percentile (median) of the current window.
	pub fn p50(&self) -> f64 {
		self.percentile(0.50)
	}

	/// 95th percentile of the current window.
	pub fn p95(&self) -> f64 {
		self.percentile(0.95)
	}

	/// 99th percentile of the current window.
	pub fn p99(&self) -> f64 {
		self.percentile(0.99)
	}

	/// Last pushed value regardless of warmup.
	pub fn last(&self) -> f64 {
		self.last_val
	}

	/// Total samples pushed (including warmup).
	pub fn count(&self) -> u64 {
		self.total_count
	}

	/// Human-readable name supplied at construction.
	pub fn name(&self) -> &str {
		&self.name
	}

	/// Reset all state to initial.
	pub fn reset(&mut self) {
		self.total_count = 0;
		self.filled = 0;
		self.head = 0;
		self.sum = 0.0;
		self.sum_sq = 0.0;
		self.last_val = 0.0;
		self.ring.fill(0.0);
	}

	// ─── helpers ─────────────────────────────────────────────────────────

	/// Return the live window samples in push order.
	fn window_slice(&self) -> Vec<f64> {
		if self.filled == 0 {
			return Vec::new();
		}
		let mut out = Vec::with_capacity(self.filled);
		// Oldest sample is at (head % window) when window is full; at 0 when not.
		let start = if self.filled == self.window {
			self.head % self.window
		} else {
			0
		};
		for i in 0..self.filled {
			out.push(self.ring[(start + i) % self.window]);
		}
		out
	}

	fn percentile(&self, p: f64) -> f64 {
		if self.filled == 0 {
			return 0.0;
		}
		let mut sorted = self.window_slice();
		sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
		let idx = ((p * (sorted.len() - 1) as f64).round() as usize).min(sorted.len() - 1);
		sorted[idx]
	}
}
