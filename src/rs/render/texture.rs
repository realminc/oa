//! Packed buffer-backed texture values and explicit host boundaries.

use std::{path::Path, rc::Rc};

use crate::{DType, Engine, Error, Matrix, Result};

/// Packed RGBA8 texture storage reusable by Render, UI, Image, and presentation.
///
/// This first backing is a checked U8 Matrix shaped `[height, width, 4]`. The
/// Texture retains distinct semantic identity and does not pretend to be an
/// Image. Native sampled/storage images remain a private future backing.
#[derive(Clone)]
pub struct Texture {
	data: Matrix,
	_semantic: Rc<()>,
}

impl Texture {
	/// Return the texture width.
	pub fn width(&self) -> usize {
		self.data.shape()[1]
	}

	/// Return the texture height.
	pub fn height(&self) -> usize {
		self.data.shape()[0]
	}

	/// Return the packed byte representation.
	pub const fn dtype(&self) -> DType {
		self.data.dtype()
	}

	/// Read packed row-major RGBA8 pixels to host memory.
	///
	/// This is an explicit host-observation boundary that flushes and waits for
	/// the exact Matrix producer when necessary.
	///
	/// # Errors
	///
	/// Returns an error when submission, completion, mapping, or cache
	/// invalidation fails.
	pub fn read_rgba8(&self) -> Result<Vec<u8>> {
		self.data.read::<u8>()
	}
}

/// Upload one exact packed row-major RGBA8 host image as a Texture.
///
/// # Errors
///
/// Returns an error for zero or overflowing dimensions, a mismatched byte
/// count, exhausted semantic identity, or device allocation/upload failure.
pub fn texture_from_rgba8(
	engine: &Engine,
	rgba: &[u8],
	width: usize,
	height: usize,
) -> Result<Texture> {
	if width == 0 || height == 0 {
		return Err(Error::invalid_argument(
			"render::texture_from_rgba8 dimensions must be non-zero",
		));
	}
	let expected = width
		.checked_mul(height)
		.and_then(|pixels| pixels.checked_mul(4))
		.ok_or_else(|| Error::out_of_range("texture byte count overflows usize"))?;
	if rgba.len() != expected {
		return Err(Error::invalid_argument(
			"render::texture_from_rgba8 requires exactly width * height * 4 bytes",
		));
	}
	let data = Matrix::from_slice(engine, [height, width, 4], rgba)?;
	Ok(Texture {
		data,
		_semantic: Rc::new(()),
	})
}

/// Read and save one packed RGBA8 Texture using its path extension.
///
/// Supported extensions are `.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp`, and
/// `.tga`. This is an explicit blocking readback and filesystem boundary.
///
/// # Errors
///
/// Returns an error for unsupported paths, failed device observation, codec
/// failure, or filesystem failure.
pub fn save_texture_file(texture: &Texture, path: impl AsRef<Path>, quality: u32) -> Result<()> {
	let width = u32::try_from(texture.width())
		.map_err(|_| Error::out_of_range("texture width exceeds codec limits"))?;
	let height = u32::try_from(texture.height())
		.map_err(|_| Error::out_of_range("texture height exceeds codec limits"))?;
	let rgba = texture.read_rgba8()?;
	crate::image::save_rgba_file(&rgba, width, height, path, quality)
}
