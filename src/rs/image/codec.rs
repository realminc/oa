//! One-shot host codec operations exported through [`crate::image`].

mod decode;
mod encode;

pub use decode::{decode_file, decode_memory};
pub use encode::{encode, save_file, save_rgba_file};

/// Still-image bitstream codec.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ImageCodec {
	/// Infer a codec from an input bitstream or output path.
	Auto,
	/// Joint Photographic Experts Group image.
	Jpeg,
	/// Portable Network Graphics image.
	Png,
	/// WebP image.
	Webp,
	/// Windows bitmap image.
	Bmp,
	/// Truevision TGA image.
	Tga,
}

/// Return whether this build can decode `codec`.
pub const fn can_decode(codec: ImageCodec) -> bool {
	matches!(
		codec,
		ImageCodec::Jpeg | ImageCodec::Png | ImageCodec::Webp | ImageCodec::Bmp | ImageCodec::Tga
	)
}

/// Return whether this build can encode `codec`.
pub const fn can_encode(codec: ImageCodec) -> bool {
	can_decode(codec)
}

pub(super) fn codec_from_path(path: &std::path::Path) -> Option<ImageCodec> {
	match path.extension()?.to_str()?.to_ascii_lowercase().as_str() {
		"jpg" | "jpeg" => Some(ImageCodec::Jpeg),
		"png" => Some(ImageCodec::Png),
		"webp" => Some(ImageCodec::Webp),
		"bmp" => Some(ImageCodec::Bmp),
		"tga" => Some(ImageCodec::Tga),
		_ => None,
	}
}

pub(super) const fn external_format(codec: ImageCodec) -> Option<image_codec::ImageFormat> {
	match codec {
		ImageCodec::Auto => None,
		ImageCodec::Jpeg => Some(image_codec::ImageFormat::Jpeg),
		ImageCodec::Png => Some(image_codec::ImageFormat::Png),
		ImageCodec::Webp => Some(image_codec::ImageFormat::WebP),
		ImageCodec::Bmp => Some(image_codec::ImageFormat::Bmp),
		ImageCodec::Tga => Some(image_codec::ImageFormat::Tga),
	}
}

pub(super) fn detect_codec(encoded: &[u8]) -> Option<ImageCodec> {
	match image_codec::guess_format(encoded).ok() {
		Some(image_codec::ImageFormat::Jpeg) => Some(ImageCodec::Jpeg),
		Some(image_codec::ImageFormat::Png) => Some(ImageCodec::Png),
		Some(image_codec::ImageFormat::WebP) => Some(ImageCodec::Webp),
		Some(image_codec::ImageFormat::Bmp) => Some(ImageCodec::Bmp),
		Some(image_codec::ImageFormat::Tga) => Some(ImageCodec::Tga),
		_ if looks_like_tga(encoded) => Some(ImageCodec::Tga),
		_ => None,
	}
}

fn looks_like_tga(encoded: &[u8]) -> bool {
	if encoded.len() < 18 {
		return false;
	}
	let color_map_type = encoded[1];
	let image_type = encoded[2];
	let width = u16::from_le_bytes([encoded[12], encoded[13]]);
	let height = u16::from_le_bytes([encoded[14], encoded[15]]);
	let depth = encoded[16];
	color_map_type <= 1
		&& matches!(image_type, 1 | 2 | 3 | 9 | 10 | 11)
		&& matches!(depth, 8 | 15 | 16 | 24 | 32)
		&& width > 0
		&& height > 0
}
