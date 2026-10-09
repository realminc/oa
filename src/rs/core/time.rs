//! Nanosecond-precision host timing.
//!
//! Port provenance: `oa/core/time.h`.
//!
//! Provides [`Timestamp`], [`Stopwatch`], [`ScopedTimer`], and [`Datetime`] as
//! thin, zero-cost wrappers over [`std::time::Instant`] and
//! [`std::time::SystemTime`]. No external crate is required.
//!
//! # Usage
//!
//! ```rust,ignore
//! // Stopwatch
//! let mut sw = oa::Stopwatch::new();
//! sw.start();
//! // ... work ...
//! println!("elapsed: {:.3}ms", sw.elapsed_ms());
//!
//! // ScopedTimer
//! let mut elapsed = 0.0_f64;
//! {
//!     let _t = oa::ScopedTimer::new(&mut elapsed);
//!     // ... work ...
//! } // writes elapsed seconds on drop
//! ```

use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

// ─── Timestamp ───────────────────────────────────────────────────────────────

/// Nanosecond-precision monotonic timestamp.
///
/// Internally stores nanoseconds elapsed since an arbitrary monotonic origin.
/// Two `Timestamp` values from the same process can be subtracted; the result
/// is meaningful as a [`Duration`]. Do not compare values across processes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Timestamp {
	nanos: i64,
}

/// The process-global monotonic origin for [`Timestamp::now()`].
fn origin() -> Instant {
	use std::sync::OnceLock;
	static ORIGIN: OnceLock<Instant> = OnceLock::new();
	*ORIGIN.get_or_init(Instant::now)
}

impl Timestamp {
	/// Capture the current monotonic time.
	pub fn now() -> Self {
		let elapsed = origin().elapsed();
		Self {
			nanos: elapsed.as_nanos().min(i64::MAX as u128) as i64,
		}
	}

	/// Zero origin timestamp.
	pub const fn zero() -> Self {
		Self { nanos: 0 }
	}

	/// Construct from nanoseconds.
	pub const fn from_nanos(n: i64) -> Self {
		Self { nanos: n }
	}

	/// Construct from seconds (f64).
	pub fn from_secs_f64(s: f64) -> Self {
		Self {
			nanos: (s * 1e9) as i64,
		}
	}

	/// Nanoseconds since origin.
	pub const fn as_nanos(self) -> i64 {
		self.nanos
	}

	/// Elapsed time as seconds (f64).
	pub fn to_secs(self) -> f64 {
		self.nanos as f64 / 1e9
	}

	/// Elapsed time as milliseconds (f64).
	pub fn to_ms(self) -> f64 {
		self.nanos as f64 / 1e6
	}

	/// True when the stored value is positive (constructed by `now()` or explicit
	/// positive nanos).
	pub fn is_valid(self) -> bool {
		self.nanos > 0
	}
}

impl std::ops::Sub for Timestamp {
	type Output = Timestamp;
	fn sub(self, rhs: Self) -> Self {
		Self {
			nanos: self.nanos.saturating_sub(rhs.nanos),
		}
	}
}

impl std::ops::Add for Timestamp {
	type Output = Timestamp;
	fn add(self, rhs: Self) -> Self {
		Self {
			nanos: self.nanos.saturating_add(rhs.nanos),
		}
	}
}

impl std::ops::SubAssign for Timestamp {
	fn sub_assign(&mut self, rhs: Self) {
		self.nanos = self.nanos.saturating_sub(rhs.nanos);
	}
}

impl std::ops::AddAssign for Timestamp {
	fn add_assign(&mut self, rhs: Self) {
		self.nanos = self.nanos.saturating_add(rhs.nanos);
	}
}

// ─── Stopwatch ───────────────────────────────────────────────────────────────

/// Accumulating stopwatch.
///
/// ```rust,ignore
/// let mut sw = oa::Stopwatch::new();
/// sw.start();
/// // ... work ...
/// sw.stop();
/// println!("{:.3}ms", sw.elapsed_ms());
/// ```
pub struct Stopwatch {
	start: Timestamp,
	elapsed: Timestamp,
	running: bool,
}

impl Default for Stopwatch {
	fn default() -> Self {
		Self::new()
	}
}

impl Stopwatch {
	/// Create a stopped stopwatch with zero elapsed time.
	pub fn new() -> Self {
		Self {
			start: Timestamp::zero(),
			elapsed: Timestamp::zero(),
			running: false,
		}
	}

	/// Start accumulating. No-op if already running.
	pub fn start(&mut self) {
		if !self.running {
			self.start = Timestamp::now();
			self.running = true;
		}
	}

	/// Stop accumulating. No-op if already stopped.
	pub fn stop(&mut self) {
		if self.running {
			self.elapsed += Timestamp::now() - self.start;
			self.running = false;
		}
	}

	/// Reset elapsed to zero and stop.
	pub fn reset(&mut self) {
		self.elapsed = Timestamp::zero();
		self.running = false;
	}

	/// Reset and immediately start.
	pub fn restart(&mut self) {
		self.reset();
		self.start();
	}

	/// Current elapsed duration (includes running segment if active).
	pub fn elapsed(&self) -> Timestamp {
		if self.running {
			self.elapsed + (Timestamp::now() - self.start)
		} else {
			self.elapsed
		}
	}

	/// Elapsed in seconds.
	pub fn elapsed_sec(&self) -> f64 {
		self.elapsed().to_secs()
	}

	/// Elapsed in milliseconds.
	pub fn elapsed_ms(&self) -> f64 {
		self.elapsed().to_ms()
	}

	/// True while the stopwatch is running.
	pub fn is_running(&self) -> bool {
		self.running
	}
}

// ─── ScopedTimer ─────────────────────────────────────────────────────────────

/// RAII scoped timer.
///
/// Borrows a `f64` and writes elapsed seconds on drop.
///
/// ```rust,ignore
/// let mut elapsed = 0.0_f64;
/// {
///     let _t = oa::ScopedTimer::new(&mut elapsed);
///     expensive_work();
/// }
/// println!("took {elapsed:.3}s");
/// ```
pub struct ScopedTimer<'a> {
	output: &'a mut f64,
	start: Timestamp,
}

impl<'a> ScopedTimer<'a> {
	/// Create a scoped timer that writes to `output` on drop.
	pub fn new(output: &'a mut f64) -> Self {
		Self {
			output,
			start: Timestamp::now(),
		}
	}
}

impl Drop for ScopedTimer<'_> {
	fn drop(&mut self) {
		*self.output = (Timestamp::now() - self.start).to_secs();
	}
}

// ─── Datetime ────────────────────────────────────────────────────────────────

/// UTC human-readable date/time.
///
/// For log display and file naming only — never for consensus or deterministic
/// math.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Datetime {
	unix_secs: i64,
}

impl Datetime {
	/// Capture the current UTC time.
	pub fn now() -> Self {
		let secs = SystemTime::now()
			.duration_since(UNIX_EPOCH)
			.unwrap_or(Duration::ZERO)
			.as_secs();
		Self {
			unix_secs: secs as i64,
		}
	}

	/// Construct from Unix timestamp in seconds.
	pub const fn from_unix_secs(secs: i64) -> Self {
		Self { unix_secs: secs }
	}

	/// Unix timestamp in seconds.
	pub const fn as_unix_secs(self) -> i64 {
		self.unix_secs
	}
}

impl std::fmt::Display for Datetime {
	/// Format as `"YYYY-MM-DD HH:MM:SS"` (UTC).
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		let s = self.unix_secs.unsigned_abs();
		let secs = s % 60;
		let mins = (s / 60) % 60;
		let hours = (s / 3600) % 24;
		let days = s / 86400;
		// Civil date from day count (proleptic Gregorian, unsigned).
		let (y, m, d) = civil_from_days(days);
		write!(f, "{y:04}-{m:02}-{d:02} {hours:02}:{mins:02}:{secs:02}")
	}
}

/// Convert days since Unix epoch to (year, month, day).
/// Implements the Gregorian calendar algorithm from H. F. Hinnant.
fn civil_from_days(z: u64) -> (u32, u32, u32) {
	let z = z as i64 + 719_468;
	let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
	let doe = (z - era * 146_097) as u32;
	let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
	let y = yoe as i64 + era * 400;
	let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
	let mp = (5 * doy + 2) / 153;
	let d = doy - (153 * mp + 2) / 5 + 1;
	let m = if mp < 10 { mp + 3 } else { mp - 9 };
	let y = if m <= 2 { y + 1 } else { y };
	(y as u32, m, d)
}
