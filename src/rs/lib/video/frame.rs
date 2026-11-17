//! Timestamped video frames with retained semantic backing.

use std::{rc::Rc, sync::Arc, time::Duration};

use crate::{Engine, Error, Event, Image, Matrix, Result, Texture, runtime::NativeDecodedFrame};

use super::{
	VideoPixelFormat,
	plane::{DevicePlanar, plane_count, plane_extents},
};
pub use crate::video::types::VideoColorInfo;

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
	pub fn new_from_microseconds(
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
	pub(crate) const fn native_backing(&self) -> Option<&NativeDecodedFrame> {
		match &self.backing {
			VideoFrameBacking::Native(frame) => Some(frame),
			_ => None,
		}
	}

	pub(crate) fn from_native(
		frame: NativeDecodedFrame,
		timing: VideoFrameTiming,
		color: VideoColorInfo,
	) -> Self {
		Self {
			backing: VideoFrameBacking::Native(frame),
			timing,
			color,
		}
	}

	pub(crate) fn from_texture(
		texture: Texture,
		timing: VideoFrameTiming,
		color: VideoColorInfo,
		ready: Option<Event>,
	) -> Self {
		Self {
			backing: VideoFrameBacking::Texture { texture, ready },
			timing,
			color,
		}
	}

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
	pub fn new_from_image(
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

	/// Upload tightly packed plane bytes into device-resident planar storage.
	///
	/// `planes` must match the format's plane count and each plane's tightly
	/// packed `width * height` `u8` extent. `width` and `height` are the
	/// visible luma extent. The returned frame carries the same timing and
	/// color metadata the caller supplies; this is the host-to-device staging
	/// step for native frame operations and `oa::video::to_matrix`.
	///
	/// # Errors
	///
	/// Returns an error for a zero extent, a wrong plane count, mismatched
	/// plane sizes, a bit depth that has no admitted storage, or a storage
	/// allocation or upload failure.
	pub fn upload_planes(
		engine: &Engine,
		format: VideoPixelFormat,
		planes: Vec<Vec<u8>>,
		width: usize,
		height: usize,
		timing: VideoFrameTiming,
		color: VideoColorInfo,
	) -> Result<Self> {
		if planes.len() != plane_count(format) {
			return Err(Error::invalid_argument(format!(
				"{format:?} requires {} planes, but {} were provided",
				plane_count(format),
				planes.len()
			)));
		}
		let extents = plane_extents(format, width, height)?;
		let mut matrices = Vec::with_capacity(planes.len());
		for (bytes, extent) in planes.iter().zip(extents.iter()) {
			let expected = extent
				.width
				.checked_mul(extent.height)
				.ok_or_else(|| Error::out_of_range("plane byte extent overflows usize"))?;
			if bytes.len() != expected {
				return Err(Error::invalid_argument(format!(
					"plane requires {expected} bytes, but {} were provided",
					bytes.len()
				)));
			}
			matrices.push(Matrix::new_from_slice(
				engine,
				[extent.height, extent.width],
				bytes,
			)?);
		}
		Ok(Self {
			backing: VideoFrameBacking::DevicePlanar(DevicePlanar {
				planes: Rc::new(matrices),
				format,
				width,
				height,
			}),
			timing,
			color,
		})
	}

	/// Stage a host-planar frame into device-resident planar storage.
	///
	/// A device-planar frame is returned unchanged. A tightly packed
	/// YUV420 host frame uploads as three `Yuv420Planar8` planes with its
	/// timing and color metadata preserved. Packed images, textures, and
	/// native decoder frames have no upload route in this checkpoint.
	///
	/// # Errors
	///
	/// Returns an error for an ineligible backing or an upload failure.
	pub fn upload(&self, engine: &Engine) -> Result<VideoFrame> {
		match &self.backing {
			VideoFrameBacking::DevicePlanar(_) => Ok(self.clone()),
			VideoFrameBacking::PlanarYuv420 {
				bytes,
				width,
				height,
			} => {
				let width = *width;
				let height = *height;
				let luma = width * height;
				let chroma = luma / 4;
				let planes = vec![
					bytes[..luma].to_vec(),
					bytes[luma..luma + chroma].to_vec(),
					bytes[luma + chroma..].to_vec(),
				];
				Self::upload_planes(
					engine,
					VideoPixelFormat::Yuv420Planar8,
					planes,
					width,
					height,
					self.timing,
					self.color,
				)
			}
			VideoFrameBacking::PackedImage(_)
			| VideoFrameBacking::Texture { .. }
			| VideoFrameBacking::PlanarYuv420P10 { .. }
			| VideoFrameBacking::Native(_) => Err(Error::failed_precondition(
				"upload applies to 8-bit host-planar or device-planar video frames",
			)),
		}
	}

	/// Wrap a device-planar backing in frame semantics for operation outputs.
	pub(crate) fn from_device_planar(
		planar: DevicePlanar,
		timing: VideoFrameTiming,
		color: VideoColorInfo,
	) -> Self {
		Self {
			backing: VideoFrameBacking::DevicePlanar(planar),
			timing,
			color,
		}
	}

	/// Attach frame semantics to tightly packed planar 8-bit YUV 4:2:0 bytes.
	///
	/// Planes are stored as full-resolution Y followed by quarter-resolution U
	/// and V. The owned `Vec` allocation is retained without copying pixels;
	/// cloning the frame shares that allocation. Width and height must both be
	/// non-zero and even.
	///
	/// # Errors
	///
	/// Returns an error for invalid 4:2:0 geometry, arithmetic overflow, or a
	/// byte count other than `width * height * 3 / 2`.
	pub fn new_from_yuv420p(
		bytes: Vec<u8>,
		width: usize,
		height: usize,
		timing: VideoFrameTiming,
		color: VideoColorInfo,
	) -> Result<Self> {
		if width == 0 || height == 0 || !width.is_multiple_of(2) || !height.is_multiple_of(2) {
			return Err(Error::invalid_argument(
				"planar YUV420 video frames require non-zero even dimensions",
			));
		}
		let expected = width
			.checked_mul(height)
			.and_then(|luma| luma.checked_mul(3))
			.and_then(|samples| samples.checked_div(2))
			.ok_or_else(|| Error::out_of_range("planar YUV420 frame size overflows usize"))?;
		if bytes.len() != expected {
			return Err(Error::invalid_argument(format!(
				"planar YUV420 frame requires {expected} bytes, but received {}",
				bytes.len()
			)));
		}
		Ok(Self {
			backing: VideoFrameBacking::PlanarYuv420 {
				bytes: Arc::new(bytes),
				width,
				height,
			},
			timing,
			color,
		})
	}

	/// Attach frame semantics to tightly packed planar 10-bit YUV 4:2:0
	/// samples.
	///
	/// Samples are little-endian `u16` lanes with the ten significant bits
	/// right-justified (`yuv420p10le` layout). Geometry and plane order match
	/// [`new_from_yuv420p`](Self::new_from_yuv420p).
	///
	/// # Errors
	///
	/// Returns an error for invalid 4:2:0 geometry, arithmetic overflow, or a
	/// sample count other than `width * height * 3 / 2`.
	pub fn new_from_yuv420p10(
		bytes: Vec<u16>,
		width: usize,
		height: usize,
		timing: VideoFrameTiming,
		color: VideoColorInfo,
	) -> Result<Self> {
		if width == 0 || height == 0 || !width.is_multiple_of(2) || !height.is_multiple_of(2) {
			return Err(Error::invalid_argument(
				"planar YUV420 video frames require non-zero even dimensions",
			));
		}
		let expected = width
			.checked_mul(height)
			.and_then(|luma| luma.checked_mul(3))
			.and_then(|samples| samples.checked_div(2))
			.ok_or_else(|| Error::out_of_range("planar YUV420 frame size overflows usize"))?;
		if bytes.len() != expected {
			return Err(Error::invalid_argument(format!(
				"planar YUV420 10-bit frame requires {expected} samples, but received {}",
				bytes.len()
			)));
		}
		Ok(Self {
			backing: VideoFrameBacking::PlanarYuv420P10 {
				bytes: Arc::new(bytes),
				width,
				height,
			},
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
			VideoFrameBacking::Texture { .. }
			| VideoFrameBacking::Native(_)
			| VideoFrameBacking::DevicePlanar(_)
			| VideoFrameBacking::PlanarYuv420 { .. }
			| VideoFrameBacking::PlanarYuv420P10 { .. } => None,
		}
	}

	/// Borrow the retained render Texture when this frame has one.
	///
	/// Texture backing remains semantically distinct from packed Image and
	/// planar codec output. Its Matrix producer readiness remains attached to the
	/// retained Texture.
	pub const fn as_texture(&self) -> Option<&Texture> {
		match &self.backing {
			VideoFrameBacking::Texture { texture, .. } => Some(texture),
			VideoFrameBacking::PackedImage(_)
			| VideoFrameBacking::Native(_)
			| VideoFrameBacking::DevicePlanar(_)
			| VideoFrameBacking::PlanarYuv420 { .. }
			| VideoFrameBacking::PlanarYuv420P10 { .. } => None,
		}
	}

	/// Borrow tightly packed planar 8-bit YUV 4:2:0 bytes when present.
	pub fn as_yuv420p(&self) -> Option<&[u8]> {
		match &self.backing {
			VideoFrameBacking::PlanarYuv420 { bytes, .. } => Some(bytes.as_slice()),
			VideoFrameBacking::PackedImage(_)
			| VideoFrameBacking::Texture { .. }
			| VideoFrameBacking::DevicePlanar(_)
			| VideoFrameBacking::PlanarYuv420P10 { .. }
			| VideoFrameBacking::Native(_) => None,
		}
	}

	/// Borrow tightly packed planar 10-bit YUV 4:2:0 samples when present.
	///
	/// Samples are stored little-endian in `u16` lanes with the ten
	/// significant bits right-justified, matching FFmpeg's `yuv420p10le`.
	pub fn as_yuv420p10(&self) -> Option<&[u16]> {
		match &self.backing {
			VideoFrameBacking::PlanarYuv420P10 { bytes, .. } => Some(bytes.as_slice()),
			VideoFrameBacking::PackedImage(_)
			| VideoFrameBacking::Texture { .. }
			| VideoFrameBacking::DevicePlanar(_)
			| VideoFrameBacking::PlanarYuv420 { .. }
			| VideoFrameBacking::Native(_) => None,
		}
	}

	/// Return the native decoded-plane format when this frame retains one.
	pub const fn native_format(&self) -> Option<VideoPixelFormat> {
		match &self.backing {
			VideoFrameBacking::Native(frame) => Some(frame.format()),
			_ => None,
		}
	}

	/// Return the frame's plane format when it has a planar or native backing.
	pub fn pixel_format(&self) -> Option<VideoPixelFormat> {
		match &self.backing {
			VideoFrameBacking::DevicePlanar(planar) => Some(planar.format),
			VideoFrameBacking::Native(frame) => Some(frame.format()),
			VideoFrameBacking::PlanarYuv420 { .. } => Some(VideoPixelFormat::Yuv420Planar8),
			VideoFrameBacking::PlanarYuv420P10 { .. } => Some(VideoPixelFormat::Yuv420Planar10),
			VideoFrameBacking::PackedImage(_) | VideoFrameBacking::Texture { .. } => None,
		}
	}

	/// Borrow the frame's device plane matrices when it has a device-planar
	/// backing, in the format's plane order.
	///
	/// The matrices are `U8` planes shaped `[plane_height, plane_width]` in
	/// tightly packed row order; NV12 carries the chroma pair plane second.
	/// Returns `None` for packed, texture, native-decoded, and host-planar
	/// frames.
	pub fn planes(&self) -> Option<&[Matrix]> {
		match &self.backing {
			VideoFrameBacking::DevicePlanar(planar) => Some(planar.planes.as_slice()),
			_ => None,
		}
	}

	/// Borrow the device-planar backing when this frame has one.
	pub(crate) fn device_planar(&self) -> Option<&DevicePlanar> {
		match &self.backing {
			VideoFrameBacking::DevicePlanar(planar) => Some(planar),
			_ => None,
		}
	}

	/// Return the exact producer completion when the frame carries one.
	///
	/// Native decode and render-target adapters retain their producer event.
	/// The current native decoder waits before publishing, while a render target
	/// may still be in flight. Consumers must honor this dependency.
	pub const fn ready_event(&self) -> Option<&Event> {
		match &self.backing {
			VideoFrameBacking::Native(frame) => Some(frame.ready()),
			VideoFrameBacking::Texture { ready, .. } => ready.as_ref(),
			_ => None,
		}
	}

	/// Retain a native decoded or render-target frame until a GPU consumer completes.
	///
	/// Multiple consumers may register events from the same Engine; the latest
	/// timeline point controls slot reuse. Dropping every frame clone declares
	/// immediate reuse only when no consumer event was registered.
	///
	/// # Errors
	///
	/// Returns an error for unsupported backing, a foreign or out-of-order
	/// event, or a frame whose producer slot is no longer active.
	pub fn mark_consumed(&self, event: &Event) -> Result<()> {
		match &self.backing {
			VideoFrameBacking::Native(frame) => frame.mark_consumed(event),
			VideoFrameBacking::Texture { texture, .. } => texture.mark_consumed(event),
			_ => Err(Error::failed_precondition(
				"consumer completion applies only to native decoded or render-target video frames",
			)),
		}
	}

	/// Consume the frame and return its packed Image backing when present.
	pub fn into_image(self) -> Option<Image> {
		match self.backing {
			VideoFrameBacking::PackedImage(image) => Some(image),
			VideoFrameBacking::Texture { .. }
			| VideoFrameBacking::Native(_)
			| VideoFrameBacking::DevicePlanar(_)
			| VideoFrameBacking::PlanarYuv420 { .. }
			| VideoFrameBacking::PlanarYuv420P10 { .. } => None,
		}
	}

	/// Consume the frame and return its render Texture backing when present.
	pub fn into_texture(self) -> Option<Texture> {
		match self.backing {
			VideoFrameBacking::Texture { texture, .. } => Some(texture),
			VideoFrameBacking::PackedImage(_)
			| VideoFrameBacking::Native(_)
			| VideoFrameBacking::DevicePlanar(_)
			| VideoFrameBacking::PlanarYuv420 { .. }
			| VideoFrameBacking::PlanarYuv420P10 { .. } => None,
		}
	}
}

/// Host-visible decoded planes returned by the hardware readback path.
///
/// Eight-bit streams yield packed `yuv420p` bytes; ten- and twelve-bit
/// streams yield little-endian `u16` lanes with significant bits
/// right-justified (`yuv420p10le`/`yuv420p12le` layout).
#[derive(Clone, Debug)]
pub(crate) enum VideoHostPlanes {
	U8(Vec<u8>),
	U16(Vec<u16>),
}

#[derive(Clone)]
enum VideoFrameBacking {
	PackedImage(Image),
	Texture {
		texture: Texture,
		ready: Option<Event>,
	},
	Native(NativeDecodedFrame),
	DevicePlanar(DevicePlanar),
	PlanarYuv420 {
		bytes: Arc<Vec<u8>>,
		width: usize,
		height: usize,
	},
	PlanarYuv420P10 {
		bytes: Arc<Vec<u16>>,
		width: usize,
		height: usize,
	},
}

impl VideoFrameBacking {
	fn width(&self) -> usize {
		match self {
			Self::PackedImage(image) => image.width(),
			Self::Texture { texture, .. } => texture.width(),
			Self::Native(frame) => frame.extent().width as usize,
			Self::DevicePlanar(planar) => planar.width,
			Self::PlanarYuv420 { width, .. } | Self::PlanarYuv420P10 { width, .. } => *width,
		}
	}

	fn height(&self) -> usize {
		match self {
			Self::PackedImage(image) => image.height(),
			Self::Texture { texture, .. } => texture.height(),
			Self::Native(frame) => frame.extent().height as usize,
			Self::DevicePlanar(planar) => planar.height,
			Self::PlanarYuv420 { height, .. } | Self::PlanarYuv420P10 { height, .. } => *height,
		}
	}
}
