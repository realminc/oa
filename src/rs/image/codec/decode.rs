//! Synchronous CPU decoding into semantic [`crate::Image`] values.

use std::{fs, path::Path};

use crate::{Engine, Error, Image, ImageFormat, ImageLayout, Matrix, Result};

use super::detect_codec;

/// Decode a JPEG, PNG, WebP, BMP, or TGA file and upload normalized planar FP32 pixels.
///
/// The result is one NCHW image shaped `[1, channels, height, width]`. This is
/// a synchronous host codec boundary followed by device upload.
///
/// # Errors
///
/// Returns an error for an empty path, filesystem failure, unsupported or
/// malformed media, unsupported requested format, checked-size overflow, or
/// device allocation/upload failure.
pub fn decode_file(engine: &Engine, path: impl AsRef<Path>, format: ImageFormat) -> Result<Image> {
	let path = path.as_ref();
	if path.as_os_str().is_empty() {
		return Err(Error::invalid_argument("image::decode_file path is empty"));
	}
	let encoded = fs::read(path).map_err(|source| Error::io("image file read", source))?;
	decode_memory(engine, &encoded, format)
}

/// Decode JPEG, PNG, WebP, BMP, or TGA bytes and upload normalized planar FP32 pixels.
///
/// The requested output format may be Gray, Rgb, or Rgba. The result is one
/// NCHW image shaped `[1, channels, height, width]`.
///
/// # Errors
///
/// Returns an error for empty, unsupported, or malformed media, unsupported
/// requested format, checked-size overflow, or device allocation/upload
/// failure.
pub fn decode_memory(engine: &Engine, encoded: &[u8], format: ImageFormat) -> Result<Image> {
	if encoded.is_empty() {
		return Err(Error::invalid_argument(
			"image::decode_memory input buffer is empty",
		));
	}
	if !matches!(
		format,
		ImageFormat::Gray | ImageFormat::Rgb | ImageFormat::Rgba
	) {
		return Err(Error::invalid_argument(
			"image::decode_memory output format must be Gray, Rgb, or Rgba",
		));
	}
	let codec = detect_codec(encoded).ok_or_else(|| {
		Error::invalid_argument("image::decode_memory bitstream is unsupported or unrecognized")
	})?;
	let external = super::external_format(codec)
		.ok_or_else(|| Error::internal("detected image codec has no decoder format"))?;
	let decoded = image_codec::load_from_memory_with_format(encoded, external)
		.map_err(|source| Error::backend_failure("image", "decode", source))?;
	let width = usize::try_from(decoded.width())
		.map_err(|_| Error::out_of_range("decoded image width exceeds usize"))?;
	let height = usize::try_from(decoded.height())
		.map_err(|_| Error::out_of_range("decoded image height exceeds usize"))?;
	if width == 0 || height == 0 {
		return Err(Error::invalid_argument(
			"image::decode_memory decoded an empty image",
		));
	}

	let packed = match format {
		ImageFormat::Gray => decoded.to_luma8().into_raw(),
		ImageFormat::Rgb => decoded.to_rgb8().into_raw(),
		ImageFormat::Rgba => decoded.to_rgba8().into_raw(),
		_ => unreachable!("requested output format was validated"),
	};
	upload_packed(engine, &packed, width, height, format)
}

fn upload_packed(
	engine: &Engine,
	packed: &[u8],
	width: usize,
	height: usize,
	format: ImageFormat,
) -> Result<Image> {
	let channels = format.channels();
	let pixels = width
		.checked_mul(height)
		.ok_or_else(|| Error::out_of_range("decoded image pixel count overflows usize"))?;
	let elements = pixels
		.checked_mul(channels)
		.ok_or_else(|| Error::out_of_range("decoded image element count overflows usize"))?;
	if packed.len() != elements {
		return Err(Error::internal(
			"decoded image pixel count does not match its metadata",
		));
	}
	let mut planar = Vec::new();
	planar
		.try_reserve_exact(elements)
		.map_err(|_| Error::resource_exhausted("decoded image allocation failed"))?;
	planar.resize(elements, 0.0_f32);
	for pixel in 0..pixels {
		for channel in 0..channels {
			planar[channel * pixels + pixel] = f32::from(packed[pixel * channels + channel]) / 255.0;
		}
	}
	let matrix = Matrix::from_slice(engine, [1, channels, height, width], &planar)?;
	Image::new(matrix, ImageLayout::Nchw, format)
}
