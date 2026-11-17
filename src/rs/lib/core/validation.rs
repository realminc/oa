//! Debug validation layer for OA compute operations.
//!
//! Port provenance: `oa/core/validation.h`.
//!
//! # Build model
//!
//! | Build | Behaviour |
//! |---|---|
//! | `debug_assertions` on (dev) | [`oa_validate!`](macro@crate::oa_validate) always active; counters compiled in |
//! | `debug_assertions` off (release) | [`oa_validate!`](macro@crate::oa_validate) compiled out — zero binary size |
//! | release + `OA_ENABLE_VALIDATION` feature | [`oa_validate!`](macro@crate::oa_validate) compiled in, runtime-gated by `Validation::is_enabled()` |
//!
//! # Usage
//!
//! ```rust,ignore
//! use oa::{oa_validate, ValidationSeverity};
//!
//! fn my_op(m: usize) -> oa::Result<()> {
//!     oa_validate!(m > 0, ValidationSeverity::Error, "my_op: M must be > 0, got {m}");
//!     Ok(())
//! }
//! ```

#[cfg(debug_assertions)]
use std::collections::HashMap;
#[cfg(debug_assertions)]
use std::sync::atomic::AtomicU64;
use std::sync::{
	Mutex, OnceLock,
	atomic::{AtomicBool, Ordering},
};

// ─── ValidationSeverity ──────────────────────────────────────────────────────

/// Severity levels for validation messages.
///
/// Port provenance: `oa::ValidationSeverity`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
pub enum ValidationSeverity {
	/// Internal state dumps; opt-in.
	Verbose = 0,
	/// Routing decisions, kernel selection, shape reports.
	Info = 1,
	/// Performance issues (naive fallback, misaligned tiles).
	Warning = 2,
	/// Contract violations producing wrong results.
	Error = 3,
	/// Invariant broken; cannot continue safely.
	Fatal = 4,
}

impl ValidationSeverity {
	/// Short uppercase name for log output.
	pub fn as_str(self) -> &'static str {
		match self {
			Self::Verbose => "VERBOSE",
			Self::Info => "INFO",
			Self::Warning => "WARNING",
			Self::Error => "ERROR",
			Self::Fatal => "FATAL",
		}
	}
}

// ─── Validation ──────────────────────────────────────────────────────────────

/// Optional custom callback type for validation messages.
pub type ValidationCallback = fn(ValidationSeverity, &str, &str);

struct ValidationState {
	enabled: AtomicBool,
	min_severity: Mutex<ValidationSeverity>,
	callback: Mutex<Option<ValidationCallback>>,
	#[cfg(debug_assertions)]
	counters: Mutex<HashMap<String, AtomicU64>>,
}

static STATE: OnceLock<ValidationState> = OnceLock::new();

fn state() -> &'static ValidationState {
	STATE.get_or_init(|| ValidationState {
		#[cfg(debug_assertions)]
		enabled: AtomicBool::new(true),
		#[cfg(not(debug_assertions))]
		enabled: AtomicBool::new(false),
		min_severity: Mutex::new(ValidationSeverity::Verbose),
		callback: Mutex::new(None),
		#[cfg(debug_assertions)]
		counters: Mutex::new(HashMap::new()),
	})
}

/// Global validation controller.
///
/// Port provenance: `oa::Validation`.
pub struct Validation;

impl Validation {
	/// Enable or disable the validation layer globally.
	pub fn enable(on: bool) {
		state().enabled.store(on, Ordering::Release);
	}

	/// Return `true` when the validation layer is enabled.
	pub fn is_enabled() -> bool {
		state().enabled.load(Ordering::Acquire)
	}

	/// Read `OA_VK_VALIDATION` and `OA_VALIDATION_SEVERITY` from the process
	/// environment and apply them. Call once at engine init.
	pub fn init_from_env() {
		if std::env::var("OA_VK_VALIDATION")
			.map(|v| !matches!(v.as_str(), "" | "0" | "false" | "no" | "off"))
			.unwrap_or(false)
		{
			Self::enable(true);
		}
		if let Ok(sev) = std::env::var("OA_VALIDATION_SEVERITY") {
			let parsed = match sev.to_ascii_uppercase().as_str() {
				"VERBOSE" => Some(ValidationSeverity::Verbose),
				"INFO" => Some(ValidationSeverity::Info),
				"WARNING" | "WARN" => Some(ValidationSeverity::Warning),
				"ERROR" => Some(ValidationSeverity::Error),
				"FATAL" => Some(ValidationSeverity::Fatal),
				_ => None,
			};
			if let Some(s) = parsed {
				Self::set_min_severity(s);
			}
		}
	}

	/// Override the minimum severity for emitting messages. Default: `Verbose`
	/// in debug, `Warning` in release.
	pub fn set_min_severity(sev: ValidationSeverity) {
		*state().min_severity.lock().unwrap() = sev;
	}

	/// Return the current minimum severity.
	pub fn get_min_severity() -> ValidationSeverity {
		*state().min_severity.lock().unwrap()
	}

	/// Set an optional custom callback; `None` restores the default stderr
	/// writer.
	pub fn set_callback(cb: Option<ValidationCallback>) {
		*state().callback.lock().unwrap() = cb;
	}

	/// Emit a validation message. Returns `Err` on `Error` or `Fatal`
	/// severity; `Ok` otherwise.
	///
	/// Called by [`oa_validate!`](macro@crate::oa_validate) when the condition fails.
	pub fn report(sev: ValidationSeverity, component: &str, message: &str) -> crate::Result<()> {
		let s = state();
		if !s.enabled.load(Ordering::Acquire) {
			return Ok(());
		}
		if sev < *s.min_severity.lock().unwrap() {
			return Ok(());
		}
		let cb = *s.callback.lock().unwrap();
		if let Some(f) = cb {
			f(sev, component, message);
		} else {
			eprintln!("[OA {}] {}: {}", sev.as_str(), component, message);
		}
		if sev >= ValidationSeverity::Error {
			return Err(crate::core::error::Error::invalid_argument(format!(
				"validation error [{component}]: {message}"
			)));
		}
		Ok(())
	}

	// ── Debug counters (compiled out in release) ──────────────────────────

	/// Increment a named debug counter (no-op in release builds).
	#[cfg(debug_assertions)]
	pub fn incr_counter(name: &str) {
		let s = state();
		let mut map = s.counters.lock().unwrap();
		map
			.entry(name.to_owned())
			.or_insert_with(|| AtomicU64::new(0))
			.fetch_add(1, Ordering::Relaxed);
	}

	#[cfg(not(debug_assertions))]
	pub fn incr_counter(_name: &str) {}

	/// Return the current value of a named debug counter (0 in release builds).
	#[cfg(debug_assertions)]
	pub fn get_counter(name: &str) -> u64 {
		state()
			.counters
			.lock()
			.unwrap()
			.get(name)
			.map(|c| c.load(Ordering::Relaxed))
			.unwrap_or(0)
	}

	#[cfg(not(debug_assertions))]
	pub fn get_counter(_name: &str) -> u64 {
		0
	}

	/// Reset all debug counters to zero (no-op in release builds).
	#[cfg(debug_assertions)]
	pub fn reset_counters() {
		state().counters.lock().unwrap().clear();
	}

	#[cfg(not(debug_assertions))]
	pub fn reset_counters() {}
}

// ─── oa_validate! macro ──────────────────────────────────────────────────────

/// Validate a condition and report via the OA validation layer.
///
/// In debug builds the check is always active. In release builds the check
/// compiles out unless the `oa-enable-validation` feature is set.
///
/// ```rust,ignore
/// // In a function returning oa::Result<T>:
/// oa_validate!(m > 0, oa::ValidationSeverity::Error, "Core", "M must be > 0, got {m}");
/// ```
///
/// - `Error` / `Fatal` → logs and returns `Err(...)` from the enclosing function.
/// - `Warning` / `Info` / `Verbose` → logs only; does not return.
///
/// Port provenance: `OA_VALIDATE` macro in `oa/core/validation.h`.
#[macro_export]
macro_rules! oa_validate {
	($cond:expr, $sev:expr, $comp:expr, $($arg:tt)*) => {
		#[cfg(any(debug_assertions, feature = "oa-enable-validation"))]
		{
			if $crate::Validation::is_enabled() && !($cond) {
				let msg = format!($($arg)*);
				$crate::Validation::report($sev, $comp, &msg)?;
			}
		}
	};
}

/// Performance warning: logs at `Warning` severity when `cond` is true.
/// Does **not** return an error. Compiled out in release without the feature.
///
/// Port provenance: `OA_WARN_PERF` in `oa/core/validation.h`.
#[macro_export]
macro_rules! oa_warn_perf {
	($cond:expr, $($arg:tt)*) => {
		#[cfg(any(debug_assertions, feature = "oa-enable-validation"))]
		{
			if $crate::Validation::is_enabled() && ($cond) {
				let msg = format!($($arg)*);
				let _ = $crate::Validation::report(
					$crate::ValidationSeverity::Warning, "Core", &msg);
			}
		}
	};
}
