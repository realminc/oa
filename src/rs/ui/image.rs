//! Planar multi-dtype images for `Ui::image_planar`/`Ui::image_plane`.
//!
//! Donor: `oa/ui/image.h` and `oa/ui/image.cpp`.
//!
//! [`ImagePlanes`] stores one device buffer per channel, each in an
//! independent [`ImageDtype`]. Plane buffers are retained `Matrix` values —
//! display consumption through `Ui` retains them automatically until frame
//! completion, so the donor's manual `markConsumed` contract has no owner
//! here.
//!
//! Typical flows:
//! ```ignore
//! let planes = ImagePlanes::load_file(&engine, "frame.png")?;       // planar U8
//! let planes = ImagePlanes::from_planes(&engine, &[&r, &d],         // RGB + U16 depth
//!     &[ImageDtype::U8, ImageDtype::U16], 640, 480)?;
//! ui.image_planar(&planes, 0, 0)?;
//! ui.image_plane(&planes, 1, 0, 0)?; // depth channel as grayscale
//! ```

use std::path::Path;

use crate::runtime::EngineHandle;
use crate::{Engine, Error, Matrix, Result};

/// Per-channel storage type; selects the `blit_planar` normalization path.
///
/// Donor: `oa::ImageDtype`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u8)]
pub enum ImageDtype {
	/// `uint8` `[0, 255]` normalized to `[0.0, 1.0]`.
	#[default]
	U8 = 0,
	/// `uint16` `[0, 65535]` normalized to `[0.0, 1.0]`.
	U16 = 1,
	/// `float32` passed through as-is.
	F32 = 2,
	/// `bfloat16` decoded to float32, passed through as-is.
	Bf16 = 3,
}

impl ImageDtype {
	const fn scalar_bytes(self) -> usize {
		match self {
			Self::U8 => 1,
			Self::U16 | Self::Bf16 => 2,
			Self::F32 => 4,
		}
	}
}

/// Maximum channel/plane count: 1=gray, 2=gray+alpha, 3=RGB, 4=RGBA/RGB+depth.
///
/// Donor: `oa::kImageMaxPlanes`.
pub const IMAGE_MAX_PLANES: usize = 4;

/// Planar image: one retained device buffer per channel, independent dtype.
///
/// Each plane is a flat row-major array of `width * height` elements. Missing
/// alpha (fewer than 4 channels) defaults to `1.0` in the blit shader. The
/// value owns its plane storage; cloning plane buffers for frame retention is
/// internal to `Ui`.
///
/// Donor: `oa::ImagePlanes`.
pub struct ImagePlanes {
	engine: EngineHandle,
	planes: Vec<Matrix>,
	dtypes: Vec<ImageDtype>,
	width: i32,
	height: i32,
}

impl ImagePlanes {
	/// Return the dtype stored for `channel` (`U8` when out of range).
	#[must_use]
	pub fn plane_dtype(&self, channel: usize) -> ImageDtype {
		self.dtypes.get(channel).copied().unwrap_or_default()
	}

	/// Return the image width in pixels.
	#[must_use]
	pub const fn width(&self) -> i32 {
		self.width
	}

	/// Return the image height in pixels.
	#[must_use]
	pub const fn height(&self) -> i32 {
		self.height
	}

	/// Return the number of planes (1–4).
	#[must_use]
	pub fn channel_count(&self) -> usize {
		self.planes.len()
	}

	/// Return whether the image has positive dimensions and live planes.
	#[must_use]
	pub fn is_valid(&self) -> bool {
		self.width > 0 && self.height > 0 && !self.planes.is_empty()
	}

	/// Bindless buffer index for `channel`, or `u32::MAX` when absent.
	pub(crate) fn plane_index(&self, channel: usize) -> Result<u32> {
		if channel >= self.planes.len() {
			return Ok(u32::MAX);
		}
		self
			.engine
			.storage_descriptor_index(self.planes[channel].storage())
	}

	pub(crate) const fn engine_handle(&self) -> &EngineHandle {
		&self.engine
	}

	/// Clone the retained plane storages for frame-scope retention.
	pub(crate) fn retain_planes(&self, frame_matrices: &mut Vec<Matrix>) {
		frame_matrices.extend(self.planes.iter().cloned());
	}

	/// Decode a supported image file into planar channels.
	///
	/// Preserves the file's channel count: 1=gray, 2=gray+alpha, 3=RGB,
	/// 4=RGBA. 8-bit files upload `U8` planes, 16-bit files `U16`, and
	/// floating-point (HDR) files `F32`. This is a synchronous host codec
	/// boundary followed by device uploads.
	///
	/// # Errors
	///
	/// Returns an error for an unreadable, unsupported, or malformed file,
	/// checked-size overflow, or device upload failure.
	pub fn load_file(engine: &Engine, path: impl AsRef<Path>) -> Result<Self> {
		let path = path.as_ref();
		if path.as_os_str().is_empty() {
			return Err(Error::invalid_argument(
				"ImagePlanes::load_file path is empty",
			));
		}
		let encoded = std::fs::read(path).map_err(|source| Error::io("image file read", source))?;
		let decoded = image_codec::load_from_memory(&encoded)
			.map_err(|source| Error::backend_failure("image", "decode", source))?;
		let (channels, dtype) = match &decoded {
			image_codec::DynamicImage::ImageLuma8(_) => (1_usize, ImageDtype::U8),
			image_codec::DynamicImage::ImageLumaA8(_) => (2, ImageDtype::U8),
			image_codec::DynamicImage::ImageRgb8(_) => (3, ImageDtype::U8),
			image_codec::DynamicImage::ImageRgba8(_) => (4, ImageDtype::U8),
			image_codec::DynamicImage::ImageLuma16(_) => (1, ImageDtype::U16),
			image_codec::DynamicImage::ImageLumaA16(_) => (2, ImageDtype::U16),
			image_codec::DynamicImage::ImageRgb16(_) => (3, ImageDtype::U16),
			image_codec::DynamicImage::ImageRgba16(_) => (4, ImageDtype::U16),
			image_codec::DynamicImage::ImageRgb32F(_) => (3, ImageDtype::F32),
			image_codec::DynamicImage::ImageRgba32F(_) => (4, ImageDtype::F32),
			_ => {
				return Err(Error::invalid_argument(
					"ImagePlanes::load_file decoded an unsupported pixel format",
				));
			}
		};
		let width = usize::try_from(decoded.width())
			.map_err(|_| Error::out_of_range("decoded image width exceeds usize"))?;
		let height = usize::try_from(decoded.height())
			.map_err(|_| Error::out_of_range("decoded image height exceeds usize"))?;
		let bytes = decoded.into_bytes();
		let scalar = dtype.scalar_bytes();
		let stride = channels * scalar;
		if bytes.len() != width * height * stride {
			return Err(Error::internal(
				"decoded plane byte count does not match extent",
			));
		}
		let plane_bytes = (0..channels)
			.map(|channel| {
				bytes
					.chunks_exact(stride)
					.flat_map(|pixel| &pixel[channel * scalar..(channel + 1) * scalar])
					.copied()
					.collect::<Vec<u8>>()
			})
			.collect::<Vec<_>>();
		let spans = plane_bytes.iter().map(Vec::as_slice).collect::<Vec<_>>();
		let dtypes = vec![dtype; channels];
		Self::from_planes(
			engine,
			&spans,
			&dtypes,
			i32::try_from(width).map_err(|_| Error::out_of_range("image width exceeds i32"))?,
			i32::try_from(height).map_err(|_| Error::out_of_range("image height exceeds i32"))?,
		)
	}

	/// Upload caller-provided planes.
	///
	/// `planes` and `dtypes` must have the same length (1–4); each span holds
	/// exactly `width * height * sizeof(dtype)` bytes. Synchronous.
	///
	/// # Errors
	///
	/// Returns an error for a channel count outside 1–4, a planes/dtypes
	/// length mismatch, non-positive or overflowing dimensions, a plane byte
	/// size that does not match its dtype, or device upload failure.
	pub fn from_planes(
		engine: &Engine,
		planes: &[&[u8]],
		dtypes: &[ImageDtype],
		width: i32,
		height: i32,
	) -> Result<Self> {
		if planes.is_empty() || planes.len() > IMAGE_MAX_PLANES {
			return Err(Error::invalid_argument(
				"ImagePlanes::from_planes channel count must be 1-4",
			));
		}
		if planes.len() != dtypes.len() {
			return Err(Error::invalid_argument(
				"ImagePlanes::from_planes planes/dtypes size mismatch",
			));
		}
		if width <= 0 || height <= 0 {
			return Err(Error::invalid_argument(
				"ImagePlanes::from_planes dimensions must be positive",
			));
		}
		let pixels = (width as u64)
			.checked_mul(height as u64)
			.ok_or_else(|| Error::invalid_argument("ImagePlanes::from_planes pixel count overflows"))?;
		let mut uploaded = Vec::with_capacity(planes.len());
		for (index, (plane, dtype)) in planes.iter().zip(dtypes).enumerate() {
			let expected = usize::try_from(pixels.checked_mul(dtype.scalar_bytes() as u64).ok_or_else(
				|| Error::invalid_argument("ImagePlanes::from_planes plane byte size overflows"),
			)?)
			.map_err(|_| Error::invalid_argument("ImagePlanes::from_planes plane byte size overflows"))?;
			if plane.len() != expected {
				return Err(Error::invalid_argument(format!(
					"ImagePlanes::from_planes plane {index} byte size does not match dimensions and dtype",
				)));
			}
			uploaded.push(Matrix::new_from_slice(engine, [expected], plane)?);
		}
		Ok(Self {
			engine: engine.handle(),
			planes: uploaded,
			dtypes: dtypes.to_vec(),
			width,
			height,
		})
	}
}
