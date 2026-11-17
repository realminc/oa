//! Synchronous encoding from semantic [`crate::Image`] values or packed host pixels.

use std::{io::Cursor, path::Path};

use image_codec::{DynamicImage, ImageBuffer};

use crate::{DType, Error, Image, ImageFormat, ImageLayout, Result};

use super::{ImageCodec, codec_from_path, external_format};

/// Read normalized planar FP32 pixels and encode one still-image bitstream.
///
/// `quality` is used by the lossy JPEG and WebP encoders and must be in
/// `[1, 100]` for those codecs. This is an explicit blocking host-observation
/// boundary.
///
/// # Errors
///
/// Returns an error for Auto, unsupported image metadata, invalid quality,
/// readback failure, checked-size overflow, or codec failure.
pub fn encode(image: &Image, codec: ImageCodec, quality: u32) -> Result<Vec<u8>> {
	if codec == ImageCodec::Auto {
		return Err(Error::invalid_argument(
			"image::encode requires an explicit codec",
		));
	}
	let packed = read_packed_pixels(image, codec)?;
	encode_packed(&packed, codec, quality)
}

/// Encode an Image using the codec inferred from the output extension and save it.
///
/// Supported extensions are `.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp`, and
/// `.tga`. This is an explicit blocking device-readback and filesystem boundary.
///
/// # Errors
///
/// Returns an error for an empty or unsupported path, invalid image or quality,
/// readback or encoding failure, or a host filesystem failure.
pub fn save_file(path: impl AsRef<Path>, image: &Image, quality: u32) -> Result<()> {
	let path = path.as_ref();
	if path.as_os_str().is_empty() {
		return Err(Error::invalid_argument("image::save_file path is empty"));
	}
	let codec = codec_from_path(path).ok_or_else(|| {
		Error::invalid_argument("image::save_file expected .jpg, .jpeg, .png, .webp, .bmp, or .tga")
	})?;
	let encoded = encode(image, codec, quality)?;
	std::fs::write(path, encoded).map_err(|source| Error::io("image file write", source))
}

/// Encode and save one exact packed RGBA8 host image.
///
/// This is the host sink used by render sessions after their own explicit
/// completion and readback protocol has produced packed bytes.
///
/// # Errors
///
/// Returns an error for an empty or unsupported path, zero or overflowing
/// dimensions, a mismatched byte count, invalid quality, codec failure, or a
/// host filesystem failure.
pub fn save_rgba_file(
	rgba: &[u8],
	width: u32,
	height: u32,
	path: impl AsRef<Path>,
	quality: u32,
) -> Result<()> {
	let path = path.as_ref();
	if path.as_os_str().is_empty() {
		return Err(Error::invalid_argument(
			"image::save_rgba_file path is empty",
		));
	}
	let codec = codec_from_path(path).ok_or_else(|| {
		Error::invalid_argument(
			"image::save_rgba_file expected .jpg, .jpeg, .png, .webp, .bmp, or .tga",
		)
	})?;
	let expected = usize::try_from(width)
		.ok()
		.and_then(|value| value.checked_mul(usize::try_from(height).ok()?))
		.and_then(|value| value.checked_mul(4))
		.ok_or_else(|| Error::out_of_range("RGBA image byte count overflows usize"))?;
	if width == 0 || height == 0 || rgba.len() != expected {
		return Err(Error::invalid_argument(
			"image::save_rgba_file byte count must equal width * height * 4 for non-zero dimensions",
		));
	}
	let packed = PackedPixels {
		data: rgba.to_vec(),
		width,
		height,
		format: ImageFormat::Rgba,
	};
	let encoded = encode_packed(&packed, codec, quality)?;
	std::fs::write(path, encoded).map_err(|source| Error::io("RGBA image file write", source))
}

struct PackedPixels {
	data: Vec<u8>,
	width: u32,
	height: u32,
	format: ImageFormat,
}

fn read_packed_pixels(image: &Image, codec: ImageCodec) -> Result<PackedPixels> {
	if !matches!(image.layout(), ImageLayout::Nchw | ImageLayout::Chw) {
		return Err(Error::invalid_argument(
			"image::encode requires Nchw or Chw layout",
		));
	}
	if image.batch_size() != 1 {
		return Err(Error::invalid_argument(
			"image::encode requires one image; select a batch item explicitly",
		));
	}
	if image.dtype() != DType::F32 {
		return Err(Error::invalid_argument(
			"image::encode requires FP32 image data",
		));
	}
	let width = u32::try_from(image.width())
		.map_err(|_| Error::out_of_range("image width exceeds codec limits"))?;
	let height = u32::try_from(image.height())
		.map_err(|_| Error::out_of_range("image height exceeds codec limits"))?;
	if width == 0 || height == 0 {
		return Err(Error::invalid_argument(
			"image::encode requires non-zero dimensions",
		));
	}
	let planar = image.as_matrix().read_f32()?;
	pack_planar(&planar, width, height, image.format(), codec)
}

fn pack_planar(
	planar: &[f32],
	width: u32,
	height: u32,
	input_format: ImageFormat,
	codec: ImageCodec,
) -> Result<PackedPixels> {
	let pixels = usize::try_from(width)
		.ok()
		.and_then(|value| value.checked_mul(usize::try_from(height).ok()?))
		.ok_or_else(|| Error::out_of_range("image pixel count overflows usize"))?;
	let expected = pixels
		.checked_mul(input_format.channels())
		.ok_or_else(|| Error::out_of_range("image element count overflows usize"))?;
	if planar.len() != expected {
		return Err(Error::internal(
			"image storage length does not match its semantic metadata",
		));
	}

	let output_format = match (codec, input_format) {
		(ImageCodec::Jpeg | ImageCodec::Webp, ImageFormat::GrayAlpha) => ImageFormat::Rgb,
		(ImageCodec::Webp, ImageFormat::Gray) => ImageFormat::Rgb,
		(_, ImageFormat::Bgr) => ImageFormat::Rgb,
		(_, ImageFormat::Bgra) => ImageFormat::Rgba,
		(_, format) => format,
	};
	let output_channels = output_format.channels();
	let output_len = pixels
		.checked_mul(output_channels)
		.ok_or_else(|| Error::out_of_range("packed image byte count overflows usize"))?;
	let mut data = Vec::new();
	data
		.try_reserve_exact(output_len)
		.map_err(|_| Error::resource_exhausted("packed image allocation failed"))?;
	data.resize(output_len, 0);
	let sample = |pixel: usize, channel: usize| quantize(planar[channel * pixels + pixel]);
	for pixel in 0..pixels {
		let destination = &mut data[pixel * output_channels..(pixel + 1) * output_channels];
		match input_format {
			ImageFormat::Gray => {
				let gray = sample(pixel, 0);
				destination.fill(gray);
			}
			ImageFormat::GrayAlpha if output_channels == 2 => {
				destination[0] = sample(pixel, 0);
				destination[1] = sample(pixel, 1);
			}
			ImageFormat::GrayAlpha => {
				let gray = sample(pixel, 0);
				destination.fill(gray);
			}
			ImageFormat::Rgb | ImageFormat::Rgba => {
				for (channel, value) in destination.iter_mut().enumerate() {
					*value = sample(pixel, channel);
				}
			}
			ImageFormat::Bgr | ImageFormat::Bgra => {
				destination[0] = sample(pixel, 2);
				destination[1] = sample(pixel, 1);
				destination[2] = sample(pixel, 0);
				if output_channels == 4 {
					destination[3] = sample(pixel, 3);
				}
			}
		}
	}
	Ok(PackedPixels {
		data,
		width,
		height,
		format: output_format,
	})
}

fn quantize(value: f32) -> u8 {
	if !value.is_finite() {
		return 0;
	}
	(value.clamp(0.0, 1.0).mul_add(255.0, 0.5)) as u8
}

fn encode_packed(pixels: &PackedPixels, codec: ImageCodec, quality: u32) -> Result<Vec<u8>> {
	if matches!(codec, ImageCodec::Jpeg | ImageCodec::Webp) && !(1..=100).contains(&quality) {
		return Err(Error::invalid_argument(
			"image::encode JPEG and WebP quality must be in [1, 100]",
		));
	}
	if codec == ImageCodec::Webp {
		let quality = u8::try_from(quality)
			.map_err(|_| Error::invalid_argument("image::encode quality exceeds u8"))?;
		let encoder = match pixels.format {
			ImageFormat::Rgb => webp_codec::Encoder::from_rgb(&pixels.data, pixels.width, pixels.height),
			ImageFormat::Rgba => {
				webp_codec::Encoder::from_rgba(&pixels.data, pixels.width, pixels.height)
			}
			_ => {
				return Err(Error::internal(
					"WebP packed conversion did not produce RGB or RGBA pixels",
				));
			}
		};
		let memory = encoder
			.encode_simple(false, f32::from(quality))
			.map_err(|source| Error::internal(format!("WebP encode failed: {source:?}")))?;
		return Ok(memory.to_vec());
	}

	let dynamic = dynamic_image(pixels)?;
	let mut encoded = Vec::new();
	if codec == ImageCodec::Jpeg {
		let quality = u8::try_from(quality)
			.map_err(|_| Error::invalid_argument("image::encode quality exceeds u8"))?;
		let encoder = image_codec::codecs::jpeg::JpegEncoder::new_with_quality(&mut encoded, quality);
		dynamic
			.write_with_encoder(encoder)
			.map_err(|source| Error::backend_failure("image", "JPEG encode", source))?;
	} else {
		let format = external_format(codec)
			.ok_or_else(|| Error::invalid_argument("image::encode requires an explicit codec"))?;
		dynamic
			.write_to(&mut Cursor::new(&mut encoded), format)
			.map_err(|source| Error::backend_failure("image", "encode", source))?;
	}
	Ok(encoded)
}

fn dynamic_image(pixels: &PackedPixels) -> Result<DynamicImage> {
	macro_rules! buffer {
		($pixel:ty, $variant:ident) => {{
			let image =
				ImageBuffer::<$pixel, _>::from_raw(pixels.width, pixels.height, pixels.data.clone())
					.ok_or_else(|| Error::internal("packed image metadata is inconsistent"))?;
			DynamicImage::$variant(image)
		}};
	}
	Ok(match pixels.format {
		ImageFormat::Gray => buffer!(image_codec::Luma<u8>, ImageLuma8),
		ImageFormat::GrayAlpha => buffer!(image_codec::LumaA<u8>, ImageLumaA8),
		ImageFormat::Rgb => buffer!(image_codec::Rgb<u8>, ImageRgb8),
		ImageFormat::Rgba => buffer!(image_codec::Rgba<u8>, ImageRgba8),
		ImageFormat::Bgr | ImageFormat::Bgra => {
			return Err(Error::internal(
				"packed image conversion retained a BGR channel order",
			));
		}
	})
}
