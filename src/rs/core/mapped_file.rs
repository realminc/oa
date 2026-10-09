//! Read-only whole-file memory mapping with RAII ownership.
//!
//! Port provenance: `oa/core/mappedFile.h`.
//!
//! On Linux/macOS/Windows uses [`memmap2`] for true kernel-page-backed
//! access. Other platforms fall back to a heap-owned `Vec<u8>` until a
//! native mapping implementation is added.
//!
//! # Usage
//!
//! ```rust,ignore
//! use oa::{MappedFile, Path};
//!
//! let mut mf = MappedFile::default();
//! mf.open_read_only(&Path::from("weights.safetensors"))?;
//! let data: &[u8] = mf.bytes();
//! // file is unmapped when `mf` is dropped
//! ```

use crate::core::error::Error;
use crate::{Path, Result};

// ─── MappedFile ───────────────────────────────────────────────────────────────

/// Read-only whole-file mapping with RAII lifetime.
///
/// Port provenance: `oa::MappedFile`.
#[derive(Default)]
pub struct MappedFile {
	path: Path,
	inner: Option<MappedInner>,
}

enum MappedInner {
	/// `memmap2`-backed true OS mapping.
	#[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
	Mmap(memmap2::Mmap),
	/// Heap fallback for platforms without a native mmap implementation.
	#[allow(dead_code)]
	Owned(Vec<u8>),
}

impl MappedFile {
	/// Open `path` for read-only access.
	///
	/// On supported platforms this performs a true OS-level memory mapping;
	/// elsewhere it reads the file into a heap buffer.
	///
	/// Port provenance: `oa::MappedFile::openReadOnly`.
	pub fn open_read_only(&mut self, path: &Path) -> Result<()> {
		self.close();
		self.path = path.clone();
		self.inner = Some(Self::map_file(path)?);
		Ok(())
	}

	#[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
	fn map_file(path: &Path) -> Result<MappedInner> {
		let file = std::fs::File::open(std::ops::Deref::deref(path))
			.map_err(|e| Error::io("MappedFile::open", e))?;
		// SAFETY: the mapping is read-only; the file stays open while the
		// `Mmap` is alive and both are owned by `MappedInner`.
		let mmap =
			unsafe { memmap2::Mmap::map(&file) }.map_err(|e| Error::io("MappedFile::mmap", e))?;
		Ok(MappedInner::Mmap(mmap))
	}

	#[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
	fn map_file(path: &Path) -> Result<MappedInner> {
		let data = std::fs::read(path).map_err(|e| Error::io("MappedFile::read", e))?;
		Ok(MappedInner::Owned(data))
	}

	/// Unmap and release the underlying resource.
	pub fn close(&mut self) {
		self.inner = None;
		self.path = Path::default();
	}

	/// `true` when a file is currently open.
	pub fn is_open(&self) -> bool {
		self.inner.is_some()
	}

	/// The path this file was opened from.
	pub fn path(&self) -> &Path {
		&self.path
	}

	/// Raw byte slice over the mapped data. Empty when not open.
	pub fn bytes(&self) -> &[u8] {
		match &self.inner {
			None => &[],
			#[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
			Some(MappedInner::Mmap(m)) => m,
			Some(MappedInner::Owned(v)) => v,
		}
	}

	/// Length in bytes. Zero when not open.
	pub fn size(&self) -> usize {
		self.bytes().len()
	}

	/// Return a sub-slice `[offset, offset+len)` or an error if out of range.
	///
	/// Port provenance: `oa::MappedFile::slice`.
	pub fn slice(&self, offset: u64, len: u64) -> Result<&[u8]> {
		let data = self.bytes();
		let start = offset as usize;
		let end = start
			.checked_add(len as usize)
			.filter(|&e| e <= data.len())
			.ok_or_else(|| {
				Error::out_of_range(format!(
					"MappedFile::slice: [{offset}, {}) out of bounds (size={})",
					offset + len,
					data.len()
				))
			})?;
		Ok(&data[start..end])
	}
}
