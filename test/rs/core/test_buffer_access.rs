//! Tests for `oa::BufferAccess` and miscellaneous op-contract types.

use oa::BufferAccess;

// ── BufferAccess variants ─────────────────────────────────────────────────────

#[test]
fn buffer_access_variants_are_distinct() {
	assert_ne!(BufferAccess::Read, BufferAccess::Write);
	assert_ne!(BufferAccess::Write, BufferAccess::ReadWrite);
	assert_ne!(BufferAccess::Read, BufferAccess::ReadWrite);
}

#[test]
fn buffer_access_copy_and_debug() {
	let a = BufferAccess::ReadWrite;
	let b = a; // Copy
	assert_eq!(a, b);
	let _ = format!("{a:?}"); // Debug
}
