//! Device-resident planar frame storage for `oa::video` operations.
//!
//! `DevicePlanar` is the private backing used by transformed frames: one
//! `U8` Matrix per plane with explicit format and visible-geometry metadata.
//! Plane extents follow the pixel-format rules rather than `width / 2`, so
//! odd luma dimensions round chroma extents up consistently with the
//! kernels and the host oracle.

use std::rc::Rc;

use crate::{Error, Matrix, Result};

use super::VideoPixelFormat;

/// Plane dimensions with row width measured in stored bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PlaneExtent {
	pub(crate) width: usize,
	pub(crate) height: usize,
}

/// Checked plane count for one admitted pixel format.
pub(crate) fn plane_count(format: VideoPixelFormat) -> usize {
	match format {
		VideoPixelFormat::Yuv420Planar8 | VideoPixelFormat::Yuv420Planar10 => 3,
		VideoPixelFormat::Nv12 | VideoPixelFormat::P010 | VideoPixelFormat::P012 => 2,
	}
}

/// Per-plane byte extents for a visible luma extent.
/// P010 stores the 10 sample bits in the high bits of each little-endian lane.
pub(crate) fn plane_extents(
	format: VideoPixelFormat,
	width: usize,
	height: usize,
) -> Result<Vec<PlaneExtent>> {
	if width == 0 || height == 0 {
		return Err(Error::invalid_argument(
			"device-planar frames require a non-zero luma extent",
		));
	}
	let chroma_width = width.div_ceil(2);
	let chroma_height = height.div_ceil(2);
	let extents = match format {
		VideoPixelFormat::Yuv420Planar8 => vec![
			PlaneExtent { width, height },
			PlaneExtent {
				width: chroma_width,
				height: chroma_height,
			},
			PlaneExtent {
				width: chroma_width,
				height: chroma_height,
			},
		],
		VideoPixelFormat::Nv12 | VideoPixelFormat::P010 => {
			let sample_bytes = if format == VideoPixelFormat::P010 {
				2
			} else {
				1
			};
			let row_bytes = |sites: usize, components: usize| {
				sites
					.checked_mul(components * sample_bytes)
					.ok_or_else(|| Error::out_of_range("video plane row size overflows usize"))
			};
			vec![
				PlaneExtent {
					width: row_bytes(width, 1)?,
					height,
				},
				PlaneExtent {
					width: row_bytes(chroma_width, 2)?,
					height: chroma_height,
				},
			]
		}
		VideoPixelFormat::Yuv420Planar10 | VideoPixelFormat::P012 => {
			return Err(Error::missing_capability(
				"device-planar video supports I420, NV12, and P010",
			));
		}
	};
	Ok(extents)
}

/// OA-owned device plane storage shared by cloned frames.
///
/// Planes are `U8` matrices shaped `[plane_height, plane_width]` in tightly
/// packed row-major order. The visible region is always the full luma
/// extent in this checkpoint; coded-versus-visible metadata arrives with the
/// native decoder view work.
#[derive(Clone)]
pub(crate) struct DevicePlanar {
	pub(crate) planes: Rc<Vec<Matrix>>,
	pub(crate) format: VideoPixelFormat,
	pub(crate) width: usize,
	pub(crate) height: usize,
}
