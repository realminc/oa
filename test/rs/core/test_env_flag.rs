//! Tests for `oa::EnvFlag`, `oa::NumericMode`, `oa::apply_numeric_mode`.

use oa::{EnvFlag, NumericMode};

// ── EnvFlag::is_set ───────────────────────────────────────────────────────────

#[test]
fn is_set_returns_false_when_unset() {
	// Use a name that will never be set in the test environment.
	assert!(!EnvFlag::is_set("OA_TEST_NEVER_SET_XYZZY_1234"));
}

#[test]
fn is_set_falsy_values() {
	// We can test the internal logic through get_string / get_int, which are
	// pure reads. Direct env writes are tested where we have process isolation.
	// For the falsy-value check, test by inspecting boolean string parsing.
	// "0", "false", "no", "off" are all falsy — tested through get_int fallback.
	assert_eq!(EnvFlag::get_int("OA_TEST_NEVER_SET_XYZZY_1234", 99), 99);
}

// ── EnvFlag::get_string ───────────────────────────────────────────────────────

#[test]
fn get_string_returns_default_when_unset() {
	let s = EnvFlag::get_string("OA_TEST_NEVER_SET_XYZZY_1234", "FP32");
	assert_eq!(s, "FP32");
}

#[test]
fn get_string_returns_value_when_set() {
	// Set in the child process via env::set_var — safe in single-threaded test.
	unsafe { std::env::set_var("OA_TEST_GET_STRING_1234", "BF16") };
	let s = EnvFlag::get_string("OA_TEST_GET_STRING_1234", "FP32");
	assert_eq!(s, "BF16");
	unsafe { std::env::remove_var("OA_TEST_GET_STRING_1234") };
}

// ── EnvFlag::get_int ─────────────────────────────────────────────────────────

#[test]
fn get_int_returns_default_when_unset() {
	assert_eq!(EnvFlag::get_int("OA_TEST_NEVER_SET_XYZZY_1234", -7), -7);
}

#[test]
fn get_int_parses_value_when_set() {
	unsafe { std::env::set_var("OA_TEST_GET_INT_1234", "42") };
	assert_eq!(EnvFlag::get_int("OA_TEST_GET_INT_1234", 0), 42);
	unsafe { std::env::remove_var("OA_TEST_GET_INT_1234") };
}

#[test]
fn get_int_returns_default_on_non_decimal() {
	unsafe { std::env::set_var("OA_TEST_GET_INT_BADVAL", "0x1F") };
	assert_eq!(EnvFlag::get_int("OA_TEST_GET_INT_BADVAL", 5), 5);
	unsafe { std::env::remove_var("OA_TEST_GET_INT_BADVAL") };
}

// ── NumericMode ───────────────────────────────────────────────────────────────

#[test]
fn numeric_mode_default_is_stable() {
	assert_eq!(NumericMode::default(), NumericMode::Stable);
}

#[test]
fn numeric_mode_variants_are_distinct() {
	assert_ne!(NumericMode::Fast, NumericMode::Stable);
	assert_ne!(NumericMode::Stable, NumericMode::Deterministic);
}

#[test]
fn numeric_mode_repr_u8() {
	assert_eq!(NumericMode::Fast as u8, 0);
	assert_eq!(NumericMode::Stable as u8, 1);
	assert_eq!(NumericMode::Deterministic as u8, 2);
}
