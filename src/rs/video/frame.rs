//! Timestamped video frames with retained semantic backing.

use std::time::Duration;

use crate::{Error, Image, Result};

/// Matrix coefficients associated with a frame's source color conversion.
///
/// This is deliberately narrower than complete color management. Primaries,
/// transfer function, chromatic adaptation, and mastering metadata are not
/// inferred from this value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VideoColorMatrix {
	/// No matrix coefficients were supplied by the producer.
	Unspecified,
	/// ITU-R BT.601 matrix coefficients.
	Bt601,
	/// ITU-R BT.709 matrix coefficients.
	Bt709,
	/// ITU-R BT.2020 non-constant-luminance matrix coefficients.
	Bt2020,
}

/// Encoded or converted component range associated with a video frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VideoColorRange {
	/// The producer did not specify a component range.
	Unspecified,
	/// Video-range or narrow-range components.
	Limited,
	/// Full-range components.
	Full,
}

/// Source color metadata retained across frame transformations.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VideoColorInfo {
	matrix: VideoColorMatrix,
	range: VideoColorRange,
}

impl VideoColorInfo {
	/// Construct explicit source color metadata.
	pub const fn new(matrix: VideoColorMatrix, range: VideoColorRange) -> Self {
		Self { matrix, range }
	}

	/// Construct metadata for a producer that did not specify color properties.
	pub const fn unspecified() -> Self {
		Self::new(VideoColorMatrix::Unspecified, VideoColorRange::Unspecified)
	}

	/// Return the source matrix coefficients.
	pub const fn matrix(self) -> VideoColorMatrix {
		self.matrix
	}

	/// Return the source component range.
	pub const fn range(self) -> VideoColorRange {
		self.range
	}
}

/// Presentation timing for one decoded, captured, or generated frame.
///
/// The presentation timestamp is a non-negative offset from a stream-defined
/// epoch. Duration is absent when the producer cannot determine it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VideoFrameTiming {
	presentation_timestamp: Duration,
	duration: Option<Duration>,
}

impl VideoFrameTiming {
	/// Construct checked frame timing.
	///
	/// # Errors
	///
	/// Returns an error when a known frame duration is zero.
	pub fn new(presentation_timestamp: Duration, duration: Option<Duration>) -> Result<Self> {
		if duration == Some(Duration::ZERO) {
			return Err(Error::invalid_argument(
				"video frame duration must be non-zero when specified",
			));
		}
		Ok(Self {
			presentation_timestamp,
			duration,
		})
	}

	/// Construct checked microsecond timing used by OA's current media boundary.
	///
	/// A zero `duration_microseconds` means that duration is unknown.
	pub fn from_microseconds(
		presentation_timestamp_microseconds: u64,
		duration_microseconds: u64,
	) -> Self {
		Self {
			presentation_timestamp: Duration::from_micros(presentation_timestamp_microseconds),
			duration: (duration_microseconds != 0)
				.then_some(Duration::from_micros(duration_microseconds)),
		}
	}

	/// Return the presentation timestamp relative to the producer's epoch.
	pub const fn presentation_timestamp(self) -> Duration {
		self.presentation_timestamp
	}

	/// Return the known frame duration.
	pub const fn duration(self) -> Option<Duration> {
		self.duration
	}
}

/// One timestamped video frame with retained storage and readiness semantics.
///
/// The admitted packed-image path retains the backing [`Image`], which in turn
/// retains its [`crate::Matrix`] storage and exact producer readiness. Cloning
/// a frame shares that retained value; it does not copy pixels or manufacture
/// a second completion event.
#[derive(Clone)]
pub struct VideoFrame {
	backing: VideoFrameBacking,
	timing: VideoFrameTiming,
	color: VideoColorInfo,
}

impl VideoFrame {
	/// Attach single-frame timing and color semantics to a packed image.
	///
	/// Batched image layouts are accepted only when their batch extent is one.
	/// Zero-width and zero-height Images remain useful for shape propagation but
	/// are not valid produced video frames.
	///
	/// # Errors
	///
	/// Returns an error when `image` has a batch extent other than one or a zero
	/// spatial extent.
	pub fn from_image(
		image: Image,
		timing: VideoFrameTiming,
		color: VideoColorInfo,
	) -> Result<Self> {
		if image.batch_size() != 1 {
			return Err(Error::invalid_argument(format!(
				"video frame requires one image, but the batch extent is {}",
				image.batch_size()
			)));
		}
		if image.width() == 0 || image.height() == 0 {
			return Err(Error::invalid_argument(
				"video frame width and height must be non-zero",
			));
		}
		Ok(Self {
			backing: VideoFrameBacking::PackedImage(image),
			timing,
			color,
		})
	}

	/// Return the visible frame width.
	pub fn width(&self) -> usize {
		self.backing.width()
	}

	/// Return the visible frame height.
	pub fn height(&self) -> usize {
		self.backing.height()
	}

	/// Return the presentation timing.
	pub const fn timing(&self) -> VideoFrameTiming {
		self.timing
	}

	/// Return the retained source color metadata.
	pub const fn color_info(&self) -> VideoColorInfo {
		self.color
	}

	/// Borrow the packed Image backing when this frame has one.
	///
	/// Future native multi-plane decoder output will return `None` rather than
	/// pretending that a Vulkan image or several planes are one dense Image.
	pub const fn as_image(&self) -> Option<&Image> {
		match &self.backing {
			VideoFrameBacking::PackedImage(image) => Some(image),
		}
	}

	/// Consume the frame and return its packed Image backing when present.
	pub fn into_image(self) -> Option<Image> {
		match self.backing {
			VideoFrameBacking::PackedImage(image) => Some(image),
		}
	}
}

#[derive(Clone)]
enum VideoFrameBacking {
	PackedImage(Image),
}

impl VideoFrameBacking {
	fn width(&self) -> usize {
		match self {
			Self::PackedImage(image) => image.width(),
		}
	}

	fn height(&self) -> usize {
		match self {
			Self::PackedImage(image) => image.height(),
		}
	}
}
