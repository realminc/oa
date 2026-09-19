//! Semantic image values composed over [`Matrix`].

use std::{
	rc::Rc,
	sync::atomic::{AtomicU64, Ordering},
};

use crate::runtime::{EngineHandle, Storage};

use super::{DType, Error, Matrix, Result};

/// Axis order carried by an [`Image`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ImageLayout {
	/// Batched channel-first layout: `[batch, channels, height, width]`.
	Nchw,
	/// Batched channel-last layout: `[batch, height, width, channels]`.
	Nhwc,
	/// Unbatched channel-first layout: `[channels, height, width]`.
	Chw,
	/// Unbatched channel-last layout: `[height, width, channels]`.
	Hwc,
	/// Unbatched single-channel layout: `[height, width]`.
	Hw,
}

impl ImageLayout {
	/// Return the matrix rank required by this layout.
	pub const fn rank(self) -> usize {
		match self {
			Self::Nchw | Self::Nhwc => 4,
			Self::Chw | Self::Hwc => 3,
			Self::Hw => 2,
		}
	}

	const fn axes(self) -> ImageAxes {
		match self {
			Self::Nchw => ImageAxes {
				batch: Some(0),
				channel: Some(1),
				height: 2,
				width: 3,
			},
			Self::Nhwc => ImageAxes {
				batch: Some(0),
				channel: Some(3),
				height: 1,
				width: 2,
			},
			Self::Chw => ImageAxes {
				batch: None,
				channel: Some(0),
				height: 1,
				width: 2,
			},
			Self::Hwc => ImageAxes {
				batch: None,
				channel: Some(2),
				height: 0,
				width: 1,
			},
			Self::Hw => ImageAxes {
				batch: None,
				channel: None,
				height: 0,
				width: 1,
			},
		}
	}
}

/// Logical channel order carried by an [`Image`].
///
/// Scalar width and numeric representation remain properties of the backing
/// [`Matrix`]; this format describes channel meaning only.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ImageFormat {
	/// One luminance channel.
	Gray,
	/// Luminance followed by alpha.
	GrayAlpha,
	/// Red, green, and blue.
	Rgb,
	/// Red, green, blue, and alpha.
	Rgba,
	/// Blue, green, and red.
	Bgr,
	/// Blue, green, red, and alpha.
	Bgra,
}

impl ImageFormat {
	/// Return the channel count required by this format.
	pub const fn channels(self) -> usize {
		match self {
			Self::Gray => 1,
			Self::GrayAlpha => 2,
			Self::Rgb | Self::Bgr => 3,
			Self::Rgba | Self::Bgra => 4,
		}
	}
}

/// Dense pixels with explicit axis-order and channel-order semantics.
///
/// `Image` composes a [`Matrix`] instead of erasing its numeric dtype and
/// storage behavior. Construction validates all metadata once; the matrix is
/// thereafter exposed immutably so the image contract cannot be invalidated.
#[derive(Clone)]
pub struct Image {
	data: Matrix,
	layout: ImageLayout,
	format: ImageFormat,
	semantic: Rc<ImageSemantic>,
}

/// Persistent semantic identity shared by clones of one Image value.
pub(crate) struct ImageSemantic {
	id: u64,
}

impl Image {
	/// Attach checked image semantics to a dense matrix.
	///
	/// Zero-sized dimensions are legal. They remain useful for shape propagation
	/// and carry no special allocation promise beyond the backing matrix contract.
	///
	/// # Errors
	///
	/// Returns an error when the matrix rank does not match `layout`, the channel
	/// axis does not match `format`, or an `Hw` image is assigned a non-gray
	/// format.
	pub fn new(data: Matrix, layout: ImageLayout, format: ImageFormat) -> Result<Self> {
		validate_image(&data, layout, format)?;
		Ok(Self {
			data,
			layout,
			format,
			semantic: Rc::new(ImageSemantic {
				id: next_image_value_id()?,
			}),
		})
	}

	/// Borrow the backing dense matrix without copying storage.
	pub const fn as_matrix(&self) -> &Matrix {
		&self.data
	}

	/// Remove image semantics and return the backing matrix.
	pub fn into_matrix(self) -> Matrix {
		self.data
	}

	/// Return the image width.
	pub fn width(&self) -> usize {
		self.data.shape()[self.layout.axes().width]
	}

	/// Return the image height.
	pub fn height(&self) -> usize {
		self.data.shape()[self.layout.axes().height]
	}

	/// Return the number of logical channels.
	pub const fn channels(&self) -> usize {
		self.format.channels()
	}

	/// Return the batch extent, or one for an unbatched layout.
	pub fn batch_size(&self) -> usize {
		self
			.layout
			.axes()
			.batch
			.map_or(1, |axis| self.data.shape()[axis])
	}

	/// Return the image axis order.
	pub const fn layout(&self) -> ImageLayout {
		self.layout
	}

	/// Return the logical channel order.
	pub const fn format(&self) -> ImageFormat {
		self.format
	}

	/// Return the scalar representation of the backing matrix.
	pub const fn dtype(&self) -> DType {
		self.data.dtype()
	}

	pub(crate) const fn engine_handle(&self) -> &EngineHandle {
		self.data.engine_handle()
	}

	pub(crate) fn storage(&self) -> &Storage {
		self.data.storage()
	}

	pub(crate) fn value_id(&self) -> u64 {
		self.semantic.id
	}
}

fn next_image_value_id() -> Result<u64> {
	static NEXT_IMAGE_VALUE_ID: AtomicU64 = AtomicU64::new(1);
	let id = NEXT_IMAGE_VALUE_ID.fetch_add(1, Ordering::Relaxed);
	if id == u64::MAX {
		return Err(Error::resource_exhausted(
			"image semantic value identity space is exhausted",
		));
	}
	Ok(id)
}

#[derive(Clone, Copy)]
struct ImageAxes {
	batch: Option<usize>,
	channel: Option<usize>,
	height: usize,
	width: usize,
}

fn validate_image(data: &Matrix, layout: ImageLayout, format: ImageFormat) -> Result<()> {
	if data.shape().len() != layout.rank() {
		return Err(Error::invalid_argument(format!(
			"image layout {layout:?} requires rank {}, but the matrix has rank {}",
			layout.rank(),
			data.shape().len()
		)));
	}

	let axes = layout.axes();
	if let Some(channel_axis) = axes.channel {
		let actual = data.shape()[channel_axis];
		let required = format.channels();
		if actual != required {
			return Err(Error::invalid_argument(format!(
				"image format {format:?} requires {required} channels, but axis {channel_axis} has {actual}"
			)));
		}
	} else if format != ImageFormat::Gray {
		return Err(Error::invalid_argument(format!(
			"image layout {layout:?} has no channel axis and requires Gray format"
		)));
	}

	Ok(())
}
