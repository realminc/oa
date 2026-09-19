//! Tests for `oa::ValidationSeverity`, `oa::Validation`, macros.

use oa::{Validation, ValidationSeverity};

// ── ValidationSeverity ordering ───────────────────────────────────────────────

#[test]
fn severity_ordering() {
	assert!(ValidationSeverity::Verbose < ValidationSeverity::Info);
	assert!(ValidationSeverity::Info < ValidationSeverity::Warning);
	assert!(ValidationSeverity::Warning < ValidationSeverity::Error);
	assert!(ValidationSeverity::Error < ValidationSeverity::Fatal);
}

#[test]
fn severity_as_str() {
	assert_eq!(ValidationSeverity::Verbose.as_str(), "VERBOSE");
	assert_eq!(ValidationSeverity::Info.as_str(), "INFO");
	assert_eq!(ValidationSeverity::Warning.as_str(), "WARNING");
	assert_eq!(ValidationSeverity::Error.as_str(), "ERROR");
	assert_eq!(ValidationSeverity::Fatal.as_str(), "FATAL");
}

// ── enable / disable toggle ───────────────────────────────────────────────────

#[test]
fn enable_disable_roundtrip() {
	let was = Validation::is_enabled();
	Validation::enable(true);
	assert!(Validation::is_enabled());
	Validation::enable(false);
	assert!(!Validation::is_enabled());
	// Restore original state.
	Validation::enable(was);
}

// ── min severity filter ───────────────────────────────────────────────────────

#[test]
fn set_get_min_severity() {
	let orig = Validation::get_min_severity();
	Validation::set_min_severity(ValidationSeverity::Warning);
	assert_eq!(Validation::get_min_severity(), ValidationSeverity::Warning);
	Validation::set_min_severity(orig);
}

// ── report returns Err on Error severity ─────────────────────────────────────

#[test]
fn report_error_returns_err() {
	Validation::enable(true);
	Validation::set_min_severity(ValidationSeverity::Verbose);
	// Set a no-op callback to suppress stderr output during tests.
	Validation::set_callback(Some(|_sev, _comp, _msg| {}));

	let r = Validation::report(ValidationSeverity::Error, "Test", "contract violated");
	assert!(r.is_err(), "Error severity must return Err");

	Validation::set_callback(None);
}

#[test]
fn report_warning_returns_ok() {
	Validation::enable(true);
	Validation::set_min_severity(ValidationSeverity::Verbose);
	Validation::set_callback(Some(|_sev, _comp, _msg| {}));

	let r = Validation::report(ValidationSeverity::Warning, "Test", "perf note");
	assert!(r.is_ok(), "Warning severity must return Ok");

	Validation::set_callback(None);
}

// ── debug counter API ─────────────────────────────────────────────────────────

#[test]
fn debug_counter_incr_and_get() {
	Validation::reset_counters();
	Validation::incr_counter("test_counter");
	Validation::incr_counter("test_counter");
	#[cfg(debug_assertions)]
	assert_eq!(Validation::get_counter("test_counter"), 2);
	#[cfg(not(debug_assertions))]
	assert_eq!(Validation::get_counter("test_counter"), 0);
	Validation::reset_counters();
}

// ── disabled layer skips all reporting ───────────────────────────────────────

#[test]
fn disabled_layer_skips_error() {
	Validation::enable(false);
	// Even Error severity must return Ok when disabled.
	let r = Validation::report(ValidationSeverity::Error, "Test", "should be skipped");
	assert!(r.is_ok(), "disabled layer must not return Err");
	// Restore.
	#[cfg(debug_assertions)]
	Validation::enable(true);
}
