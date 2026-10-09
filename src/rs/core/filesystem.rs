//! Stateless host-filesystem operations.
//!
//! [`Filesystem`] is the direct port of `oa::Filesystem` from C++. It has no
//! state of its own; every method is a free function on [`std::path::Path`].
//!
//! # Path type
//!
//! The Rust port uses [`std::path::Path`] / [`std::path::PathBuf`] directly.
//! Those types already normalise `/` and `\` on every platform, so there is no
//! need to port `oa::Path`'s lexical layer. Call `.join()`, `.parent()`,
//! `.file_name()`, `.extension()`, and `std::path::absolute()` directly for
//! path manipulation.
//!
//! # Error handling
//!
//! All fallible methods return [`crate::Result`]. I/O failures are wrapped
//! through [`crate::Error`] with [`crate::ErrorKind::Io`] so callers do not
//! depend on `std::io::Error` directly.

use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

use crate::{Error, Result};

// ─── helpers ─────────────────────────────────────────────────────────────────

fn io_err(op: &'static str, source: std::io::Error) -> Error {
	Error::io(op, source)
}

/// Sort paths lexicographically for deterministic output, matching C++
/// `sortPaths` which sorts by `genericString()`.
fn sort_paths(paths: &mut [PathBuf]) {
	paths.sort_unstable();
}

// ─── glob ────────────────────────────────────────────────────────────────────

/// Match a filename against a `*` / `?` wildcard pattern.
///
/// Direct port of `globMatch` in `filesystem.cpp`. The pattern is matched
/// against the *filename only*, not the full path.
fn glob_match(pattern: &[u8], name: &[u8]) -> bool {
	let mut p = 0usize;
	let mut n = 0usize;
	let mut star_p = usize::MAX;
	let mut star_n = 0usize;

	while n < name.len() {
		if p < pattern.len() && (pattern[p] == name[n] || pattern[p] == b'?') {
			p += 1;
			n += 1;
		} else if p < pattern.len() && pattern[p] == b'*' {
			star_p = p;
			p += 1;
			star_n = n;
		} else if star_p != usize::MAX {
			p = star_p + 1;
			star_n += 1;
			n = star_n;
		} else {
			return false;
		}
	}
	while p < pattern.len() && pattern[p] == b'*' {
		p += 1;
	}
	p == pattern.len()
}

// ─── Filesystem ──────────────────────────────────────────────────────────────

/// Stateless host-filesystem operations.
///
/// All methods are free functions on `&Path`. There is no instance state.
///
/// Matches the surface of `oa::Filesystem` in C++ and its Python bindings so
/// that cross-language code reads identically:
///
/// ```rust,ignore
/// Filesystem::create_directories(output.parent().unwrap())?;
/// let text = Filesystem::read_text(&config_path)?;
/// ```
pub struct Filesystem;

impl Filesystem {
	// ─── Existence & Info ─────────────────────────────────────────────────

	/// Return `true` if `path` exists (file, directory, or symlink target).
	pub fn exists(path: &Path) -> bool {
		path.exists()
	}

	/// Return `true` if `path` is a regular file.
	pub fn is_file(path: &Path) -> bool {
		path.is_file()
	}

	/// Return `true` if `path` is a directory.
	pub fn is_directory(path: &Path) -> bool {
		path.is_dir()
	}

	/// Return the byte size of the file at `path`.
	pub fn get_file_size(path: &Path) -> Result<u64> {
		let meta = std::fs::metadata(path).map_err(|e| io_err("get_file_size", e))?;
		Ok(meta.len())
	}

	/// Return the last-modified time as seconds since the Unix epoch.
	///
	/// Matches `oa::Filesystem::getLastModified` which returns an `i64` Unix
	/// timestamp.
	pub fn get_last_modified(path: &Path) -> Result<i64> {
		let meta = std::fs::metadata(path).map_err(|e| io_err("get_last_modified", e))?;
		let modified = meta
			.modified()
			.map_err(|e| io_err("get_last_modified", e))?;
		let secs = modified
			.duration_since(UNIX_EPOCH)
			.map(|d| d.as_secs() as i64)
			.unwrap_or_else(|e| -(e.duration().as_secs() as i64));
		Ok(secs)
	}

	// ─── Directory operations ─────────────────────────────────────────────

	/// Create a single directory. Succeeds silently if it already exists.
	pub fn create_directory(path: &Path) -> Result<()> {
		match std::fs::create_dir(path) {
			Ok(()) => Ok(()),
			Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists && path.is_dir() => Ok(()),
			Err(e) => Err(io_err("create_directory", e)),
		}
	}

	/// Create a directory and all missing ancestors. Succeeds silently if the
	/// full path already exists as a directory.
	pub fn create_directories(path: &Path) -> Result<()> {
		std::fs::create_dir_all(path).map_err(|e| io_err("create_directories", e))
	}

	/// Remove a file. Succeeds silently if the file does not exist.
	pub fn remove_file(path: &Path) -> Result<()> {
		match std::fs::remove_file(path) {
			Ok(()) => Ok(()),
			Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
			Err(e) => Err(io_err("remove_file", e)),
		}
	}

	/// Remove a directory.
	///
	/// When `recursive` is `false` the directory must be empty. When `true`
	/// the entire subtree is removed. Succeeds silently if the path does not
	/// exist.
	pub fn remove_directory(path: &Path, recursive: bool) -> Result<()> {
		if !path.exists() {
			return Ok(());
		}
		if recursive {
			std::fs::remove_dir_all(path).map_err(|e| io_err("remove_directory", e))
		} else {
			std::fs::remove_dir(path).map_err(|e| io_err("remove_directory", e))
		}
	}

	/// Copy a file or directory.
	///
	/// For a directory, creates the destination directory (non-recursive).
	/// For a file, reads the source and writes it to the destination, creating
	/// parent directories as needed.
	pub fn copy(from: &Path, to: &Path) -> Result<()> {
		if from.is_dir() {
			return Self::create_directories(to);
		}
		if let Some(parent) = to.parent()
			&& !parent.as_os_str().is_empty()
		{
			Self::create_directories(parent)?;
		}
		std::fs::copy(from, to)
			.map(|_| ())
			.map_err(|e| io_err("copy", e))
	}

	/// Move (rename) a path. Creates parent directories of the destination.
	///
	/// Note: cannot be named `move` in Rust (reserved keyword).
	pub fn move_path(from: &Path, to: &Path) -> Result<()> {
		if let Some(parent) = to.parent()
			&& !parent.as_os_str().is_empty()
		{
			Self::create_directories(parent)?;
		}
		std::fs::rename(from, to).map_err(|e| io_err("move_path", e))
	}

	// ─── Listing ──────────────────────────────────────────────────────────

	/// List regular files in `dir`, optionally filtered by extension.
	///
	/// `extension` is matched against the full extension string including the
	/// leading `.` (e.g. `".png"`). Pass `""` to return all files.
	/// Results are sorted lexicographically.
	pub fn list_files(dir: &Path, extension: &str) -> Result<Vec<PathBuf>> {
		if !dir.is_dir() {
			return Err(Error::not_found(format!(
				"directory does not exist: {}",
				dir.display()
			)));
		}
		let mut files = Vec::new();
		for entry in std::fs::read_dir(dir).map_err(|e| io_err("list_files", e))? {
			let entry = entry.map_err(|e| io_err("list_files", e))?;
			let path = entry.path();
			if !path.is_file() {
				continue;
			}
			if extension.is_empty()
				|| path
					.extension()
					.and_then(|e| e.to_str())
					.map(|e| format!(".{e}") == extension)
					.unwrap_or(false)
			{
				files.push(path);
			}
		}
		sort_paths(&mut files);
		Ok(files)
	}

	/// List immediate sub-directories of `dir`. Symlinks are not followed.
	/// Results are sorted lexicographically.
	pub fn list_directories(dir: &Path) -> Result<Vec<PathBuf>> {
		if !dir.is_dir() {
			return Err(Error::not_found(format!(
				"directory does not exist: {}",
				dir.display()
			)));
		}
		let mut dirs = Vec::new();
		for entry in std::fs::read_dir(dir).map_err(|e| io_err("list_directories", e))? {
			let entry = entry.map_err(|e| io_err("list_directories", e))?;
			// Use symlink_metadata so we do not follow symlinks (matches C++).
			let meta = entry
				.metadata()
				.map_err(|e| io_err("list_directories", e))?;
			if meta.is_dir() && !meta.is_symlink() {
				dirs.push(entry.path());
			}
		}
		sort_paths(&mut dirs);
		Ok(dirs)
	}

	/// List all entries (files and directories) in `dir`.
	///
	/// When `recursive` is `true`, descends into sub-directories but does not
	/// follow symlinks. Results are sorted lexicographically.
	pub fn list_all(dir: &Path, recursive: bool) -> Result<Vec<PathBuf>> {
		if !dir.is_dir() {
			return Err(Error::not_found(format!(
				"directory does not exist: {}",
				dir.display()
			)));
		}
		let mut entries = Vec::new();
		list_all_impl(dir, recursive, &mut entries)?;
		sort_paths(&mut entries);
		Ok(entries)
	}

	// ─── Text file operations ─────────────────────────────────────────────

	/// Read the entire file as a UTF-8 string.
	pub fn read_text(path: &Path) -> Result<String> {
		std::fs::read_to_string(path).map_err(|e| io_err("read_text", e))
	}

	/// Write `content` to a file, creating parent directories as needed.
	/// Truncates any existing content.
	pub fn write_text(path: &Path, content: &str) -> Result<()> {
		if let Some(parent) = path.parent()
			&& !parent.as_os_str().is_empty()
		{
			Self::create_directories(parent)?;
		}
		std::fs::write(path, content.as_bytes()).map_err(|e| io_err("write_text", e))
	}

	/// Append `content` to a file, creating parent directories as needed.
	pub fn append_text(path: &Path, content: &str) -> Result<()> {
		if let Some(parent) = path.parent()
			&& !parent.as_os_str().is_empty()
		{
			Self::create_directories(parent)?;
		}
		use std::io::Write as _;
		let mut file = std::fs::OpenOptions::new()
			.create(true)
			.append(true)
			.open(path)
			.map_err(|e| io_err("append_text", e))?;
		file
			.write_all(content.as_bytes())
			.map_err(|e| io_err("append_text", e))
	}

	/// Read lines, stripping `\r\n` and `\n` line endings. The final line is
	/// included even if it lacks a trailing newline. Empty files return an
	/// empty `Vec`.
	pub fn read_lines(path: &Path) -> Result<Vec<String>> {
		let text = Self::read_text(path)?;
		if text.is_empty() {
			return Ok(Vec::new());
		}
		let mut lines = Vec::new();
		let mut begin = 0usize;
		let bytes = text.as_bytes();
		for i in 0..bytes.len() {
			if bytes[i] == b'\n' {
				let end = if i > begin && bytes[i - 1] == b'\r' {
					i - 1
				} else {
					i
				};
				lines.push(text[begin..end].to_owned());
				begin = i + 1;
			}
		}
		if begin < text.len() {
			lines.push(text[begin..].to_owned());
		}
		Ok(lines)
	}

	// ─── Binary file operations ───────────────────────────────────────────

	/// Read the entire file as raw bytes.
	pub fn read_binary(path: &Path) -> Result<Vec<u8>> {
		std::fs::read(path).map_err(|e| io_err("read_binary", e))
	}

	/// Write `data` to a file, creating parent directories as needed.
	/// Truncates any existing content.
	pub fn write_binary(path: &Path, data: &[u8]) -> Result<()> {
		if let Some(parent) = path.parent()
			&& !parent.as_os_str().is_empty()
		{
			Self::create_directories(parent)?;
		}
		std::fs::write(path, data).map_err(|e| io_err("write_binary", e))
	}

	// ─── Resolution ───────────────────────────────────────────────────────

	/// Resolve a path to an absolute path against the current working
	/// directory. Does not require the path to exist.
	pub fn absolute(path: &Path) -> Result<PathBuf> {
		std::path::absolute(path).map_err(|e| io_err("absolute", e))
	}

	// ─── Glob ─────────────────────────────────────────────────────────────

	/// Return the files in `dir` whose filename matches `pattern`.
	///
	/// `*` matches any sequence of characters; `?` matches exactly one
	/// character. Only the filename component is matched, not the full path.
	/// Results are sorted lexicographically.
	pub fn glob(dir: &Path, pattern: &str) -> Result<Vec<PathBuf>> {
		if !dir.is_dir() {
			return Err(Error::not_found(format!(
				"directory does not exist: {}",
				dir.display()
			)));
		}
		let pat = pattern.as_bytes();
		let mut matches = Vec::new();
		for entry in std::fs::read_dir(dir).map_err(|e| io_err("glob", e))? {
			let entry = entry.map_err(|e| io_err("glob", e))?;
			let path = entry.path();
			let name = path
				.file_name()
				.and_then(|n| n.to_str())
				.unwrap_or_default();
			if glob_match(pat, name.as_bytes()) {
				matches.push(path);
			}
		}
		sort_paths(&mut matches);
		Ok(matches)
	}
}

// ─── list_all recursive helper ───────────────────────────────────────────────

fn list_all_impl(dir: &Path, recursive: bool, out: &mut Vec<PathBuf>) -> Result<()> {
	for entry in std::fs::read_dir(dir).map_err(|e| io_err("list_all", e))? {
		let entry = entry.map_err(|e| io_err("list_all", e))?;
		let path = entry.path();
		let meta = entry.metadata().map_err(|e| io_err("list_all", e))?;
		out.push(path.clone());
		if recursive && meta.is_dir() && !meta.is_symlink() {
			list_all_impl(&path, recursive, out)?;
		}
	}
	Ok(())
}
