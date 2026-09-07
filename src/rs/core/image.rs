//! Image - matrix composition plus layout and pixel-format meaning

use super::Result;

/// Image represents a pixel buffer with format and layout semantics
pub struct Image {
	// TODO: Add matrix composition, layout, pixel format
}

impl Image {
	/// Create an image filled with zeros
	pub fn zeros(_shape: impl Into<[usize; 2]>, _format: Format) -> Result<Self> {
		todo!("Image::zeros")
	}
}

/// Pixel format
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Format {
	Rgba8,
	Rgb8,
	// TODO: Add more formats
}
