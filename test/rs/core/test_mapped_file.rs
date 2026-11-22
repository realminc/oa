//! Tests for `oa::MappedFile`.

use oa::{MappedFile, Path};

// ── default / empty state ─────────────────────────────────────────────────────

#[test]
fn default_is_closed() {
	let mf = MappedFile::default();
	assert!(!mf.is_open());
	assert_eq!(mf.size(), 0);
	assert!(mf.bytes().is_empty());
}

// ── open and read a real file ─────────────────────────────────────────────────

#[test]
fn open_and_read_cargo_toml() {
	let mut mf = MappedFile::default();
	// Use Cargo.toml which is always present in the workspace root.
	mf.open_read_only(&Path::from("Cargo.toml"))
		.expect("should open Cargo.toml");
	assert!(mf.is_open());
	assert!(mf.size() > 0);
	// File should start with "[package]" (TOML) or a comment.
	let head = &mf.bytes()[..7.min(mf.size())];
	assert!(!head.is_empty());
}

// ── slice ─────────────────────────────────────────────────────────────────────

#[test]
fn slice_valid_range() {
	let mut mf = MappedFile::default();
	mf.open_read_only(&Path::from("Cargo.toml")).unwrap();
	let s = mf.slice(0, 4).expect("slice [0,4) must succeed");
	assert_eq!(s.len(), 4);
}

#[test]
fn slice_out_of_range_returns_err() {
	let mut mf = MappedFile::default();
	mf.open_read_only(&Path::from("Cargo.toml")).unwrap();
	let result = mf.slice(0, u64::MAX);
	assert!(result.is_err());
}

// ── close ────────────────────────────────────────────────────────────────────

#[test]
fn close_resets_state() {
	let mut mf = MappedFile::default();
	mf.open_read_only(&Path::from("Cargo.toml")).unwrap();
	assert!(mf.is_open());
	mf.close();
	assert!(!mf.is_open());
	assert_eq!(mf.size(), 0);
}

// ── missing file returns error ────────────────────────────────────────────────

#[test]
fn open_missing_file_returns_err() {
	let mut mf = MappedFile::default();
	let r = mf.open_read_only(&Path::from("__no_such_file_xyzzy__.bin"));
	assert!(r.is_err());
	assert!(!mf.is_open());
}
