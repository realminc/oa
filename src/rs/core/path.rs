//! Named OA filesystem locations and the `oa::Path` value type.
//!
//! # `oa::Path`
//!
//! A thin newtype over [`std::path::PathBuf`] that:
//! - implements `Deref<Target = std::path::Path>` so it works everywhere a
//!   `&std::path::Path` is expected (all `std::fs` calls, `Filesystem` methods)
//! - carries the named OA location methods (`asset`, `var`, `data`, `home`,
//!   `temp`) directly, so call sites read `oa::Path::asset_rel("audio/x.wav")`
//! - exposes `.join()`, `.parent()`, etc. returning `oa::Path`
//! - implements common `From`/`Into`/`AsRef`/`Display`
//!
//! # Resolution order
//!
//! - `asset` → `OA_ASSET` env var → nearest ancestor `sdk/asset` → `./asset`
//! - `var`   → `OA_VAR` env var   → nearest ancestor `var`       → `./var`
//! - `data`  → `OA_DATA` env var  → `<var>/data`
//! - `home`  → `HOME` (Unix) / `USERPROFILE` (Windows)
//! - `temp`  → `TMPDIR` / `TEMP` / `TMP` → `/tmp` (Unix) / current dir

use std::ffi::OsStr;
use std::fmt;
use std::ops::Deref;
use std::path::{self, PathBuf};

// ─── oa::Path ─────────────────────────────────────────────────────────────────

/// An owned OA path value.
///
/// Wraps [`std::path::PathBuf`] and adds the canonical OA named-location
/// resolvers as associated functions. All `std::fs` calls and
/// [`crate::Filesystem`] methods accept `&oa::Path` directly through `Deref`.
///
/// ```ignore
/// let wav  = oa::Path::asset_rel("audio/oaNarration.wav");
/// let out  = oa::Path::var_rel("example/audio/out.wav");
/// let ckpt = oa::Path::temp().join("model.oam");
/// ```
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct Path(PathBuf);

impl Path {
	// ─── Constructors ─────────────────────────────────────────────────────

	/// Create an empty path.
	pub fn empty() -> Self {
		Self(PathBuf::new())
	}

	/// Wrap an existing [`PathBuf`].
	pub fn from_path_buf(p: PathBuf) -> Self {
		Self(p)
	}

	/// Consume this path and return the inner [`PathBuf`].
	pub fn into_path_buf(self) -> PathBuf {
		self.0
	}

	// ─── Path manipulation ────────────────────────────────────────────────

	/// Extend this path with `component`, returning a new `oa::Path`.
	pub fn join(&self, component: impl AsRef<path::Path>) -> Self {
		Self(self.0.join(component))
	}

	/// Return the parent directory path, or `None` if this is a root or
	/// prefix path.
	pub fn parent(&self) -> Option<Self> {
		self.0.parent().map(|p| Self(p.to_owned()))
	}

	/// Return the final filename component, or `None`.
	pub fn file_name(&self) -> Option<&OsStr> {
		self.0.file_name()
	}

	/// Return the filename without extension, or `None`.
	pub fn file_stem(&self) -> Option<&OsStr> {
		self.0.file_stem()
	}

	/// Return the extension, or `None`.
	pub fn extension(&self) -> Option<&OsStr> {
		self.0.extension()
	}

	/// Return a new path with the extension replaced.
	pub fn with_extension(&self, ext: impl AsRef<OsStr>) -> Self {
		Self(self.0.with_extension(ext))
	}

	/// Return `true` if the path is absolute.
	pub fn is_absolute(&self) -> bool {
		self.0.is_absolute()
	}

	/// Return `true` if the path is relative.
	pub fn is_relative(&self) -> bool {
		self.0.is_relative()
	}

	/// Return `true` if the path points to an existing regular file.
	pub fn is_file(&self) -> bool {
		self.0.is_file()
	}

	/// Return `true` if the path points to an existing directory.
	pub fn is_dir(&self) -> bool {
		self.0.is_dir()
	}

	/// Return `true` if the path exists.
	pub fn exists(&self) -> bool {
		self.0.exists()
	}

	/// Return a displayable representation.
	pub fn display(&self) -> path::Display<'_> {
		self.0.display()
	}

	// ─── Named OA locations ───────────────────────────────────────────────

	/// Root of the OA asset tree.
	///
	/// Resolution: `OA_ASSET` env var → nearest ancestor `sdk/asset` dir →
	/// `./asset`.
	pub fn asset() -> Self {
		if let Ok(v) = std::env::var("OA_ASSET") {
			return Self(PathBuf::from(v));
		}
		if let Some(root) = find_source_root() {
			let candidate = root.join("sdk/asset");
			if candidate.is_dir() {
				return Self(candidate);
			}
		}
		Self(PathBuf::from("asset"))
	}

	/// Asset path joined with a relative component.
	pub fn asset_rel(relative: impl AsRef<path::Path>) -> Self {
		Self::asset().join(relative)
	}

	/// Root of the OA variable-data tree (outputs, checkpoints, logs).
	///
	/// Resolution: `OA_VAR` env var → nearest ancestor `var` dir → `./var`.
	pub fn var() -> Self {
		if let Ok(v) = std::env::var("OA_VAR") {
			return Self(PathBuf::from(v));
		}
		if let Some(root) = find_source_root() {
			let candidate = root.join("var");
			if candidate.is_dir() {
				return Self(candidate);
			}
		}
		Self(PathBuf::from("var"))
	}

	/// Var path joined with a relative component.
	pub fn var_rel(relative: impl AsRef<path::Path>) -> Self {
		Self::var().join(relative)
	}

	/// Root of the optional dataset tree.
	///
	/// Resolution: `OA_DATA` env var → `<var>/data`.
	///
	/// Datasets are never downloaded implicitly.
	pub fn data() -> Self {
		if let Ok(v) = std::env::var("OA_DATA") {
			return Self(PathBuf::from(v));
		}
		Self::var().join("data")
	}

	/// Data path joined with a relative component.
	pub fn data_rel(relative: impl AsRef<path::Path>) -> Self {
		Self::data().join(relative)
	}

	/// User home directory.
	///
	/// Resolution: `HOME` (Unix) → `USERPROFILE` (Windows) → current dir.
	pub fn home() -> Self {
		if let Ok(v) = std::env::var("HOME") {
			return Self(PathBuf::from(v));
		}
		if let Ok(v) = std::env::var("USERPROFILE") {
			return Self(PathBuf::from(v));
		}
		Self(std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")))
	}

	/// System temporary directory.
	///
	/// Resolution: `TMPDIR` → `TEMP` → `TMP` → `/tmp` (Unix) → current dir.
	pub fn temp() -> Self {
		for var in ["TMPDIR", "TEMP", "TMP"] {
			if let Ok(v) = std::env::var(var) {
				return Self(PathBuf::from(v));
			}
		}
		#[cfg(unix)]
		return Self(PathBuf::from("/tmp"));
		#[cfg(not(unix))]
		Self(std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")))
	}
}

// ─── Deref / AsRef / Borrow ───────────────────────────────────────────────────

impl Deref for Path {
	type Target = path::Path;
	fn deref(&self) -> &path::Path {
		&self.0
	}
}

impl AsRef<path::Path> for Path {
	fn as_ref(&self) -> &path::Path {
		&self.0
	}
}

impl std::borrow::Borrow<path::Path> for Path {
	fn borrow(&self) -> &path::Path {
		&self.0
	}
}

// ─── Conversions ──────────────────────────────────────────────────────────────

impl From<PathBuf> for Path {
	fn from(p: PathBuf) -> Self {
		Self(p)
	}
}

impl From<Path> for PathBuf {
	fn from(p: Path) -> PathBuf {
		p.0
	}
}

impl From<&path::Path> for Path {
	fn from(p: &path::Path) -> Self {
		Self(p.to_owned())
	}
}

impl From<&str> for Path {
	fn from(s: &str) -> Self {
		Self(PathBuf::from(s))
	}
}

impl From<String> for Path {
	fn from(s: String) -> Self {
		Self(PathBuf::from(s))
	}
}

impl AsRef<OsStr> for Path {
	fn as_ref(&self) -> &OsStr {
		self.0.as_os_str()
	}
}

// ─── Formatting ───────────────────────────────────────────────────────────────

impl fmt::Debug for Path {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		fmt::Debug::fmt(&self.0, f)
	}
}

impl fmt::Display for Path {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		fmt::Display::fmt(&self.0.display(), f)
	}
}

// ─── find_source_root ─────────────────────────────────────────────────────────

fn find_source_root() -> Option<PathBuf> {
	let cwd = std::env::current_dir().ok()?;
	let mut dir: &path::Path = &cwd;
	loop {
		if dir.join("sdk/asset").is_dir() || dir.join("Cargo.toml").is_file() {
			return Some(dir.to_owned());
		}
		dir = dir.parent()?;
	}
}
