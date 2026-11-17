//! Tier-1 native-YUV frame geometry operations.
//!
//! Every operation keeps frames device-planar and per-plane: luma, planar
//! chroma, and interleaved NV12 chroma pairs are each resampled with one
//! `resample` dispatch. Ops express their transform in luma-space map
//! parameters; chroma-space parameters derive from the centered 4:2:0 siting
//! rule, so kernel count stays flat across formats.

use std::rc::Rc;

use crate::{
	DType, Error, Matrix, OpAttribute, Result,
	runtime::{BufferBinding, ComputeDispatch, KernelId, PushConstant, SemanticDispatch},
};

use super::{
	VideoColorInfo, VideoFrame, VideoInterpolation, VideoPixelFormat,
	plane::{DevicePlanar, plane_extents},
};

/// Axis along which [`flip`] mirrors or [`stack`] joins frames.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VideoAxis {
	/// Left-right direction (x).
	Horizontal,
	/// Top-bottom direction (y).
	Vertical,
}

impl VideoAxis {
	fn token(self) -> &'static str {
		match self {
			Self::Horizontal => "horizontal",
			Self::Vertical => "vertical",
		}
	}
}

/// Luma-space resampling description shared by all tier-1 ops.
///
/// `sx = (dst + 0.5) * scale + off` in each axis; `swap` transposes the
/// coordinate assignment for quarter turns. `valid` bounds the destination
/// window that receives sampled content; outside pixels take the plane fill.
#[derive(Clone, Copy)]
struct PlaneMap {
	scale_x: f64,
	scale_y: f64,
	off_x: f64,
	off_y: f64,
	valid: [f64; 4],
	swap: bool,
	dst_off_x: usize,
	dst_off_y: usize,
	local_w: usize,
	local_h: usize,
	fill: [u8; 3],
}

impl PlaneMap {
	fn identity(fill: [u8; 3]) -> Self {
		Self {
			scale_x: 1.0,
			scale_y: 1.0,
			off_x: -0.5,
			off_y: -0.5,
			valid: [0.0; 4],
			swap: false,
			dst_off_x: 0,
			dst_off_y: 0,
			local_w: 0,
			local_h: 0,
			fill,
		}
	}

	/// Map parameters for a plane at resolution `factor` of luma (0.5 chroma).
	fn at(&self, factor: f64) -> Self {
		let shift = (factor - 1.0) * 0.5;
		Self {
			scale_x: self.scale_x,
			scale_y: self.scale_y,
			off_x: factor * self.off_x + shift,
			off_y: factor * self.off_y + shift,
			valid: [
				factor * self.valid[0],
				factor * self.valid[1],
				factor * self.valid[2],
				factor * self.valid[3],
			],
			swap: self.swap,
			dst_off_x: (self.dst_off_x as f64 * factor).floor() as usize,
			dst_off_y: (self.dst_off_y as f64 * factor).floor() as usize,
			local_w: (self.local_w as f64 * factor).ceil() as usize,
			local_h: (self.local_h as f64 * factor).ceil() as usize,
			fill: self.fill,
		}
	}
}

fn require_device_planar<'a>(frame: &'a VideoFrame, operation: &str) -> Result<&'a DevicePlanar> {
	frame.device_planar().ok_or_else(|| {
		Error::invalid_argument(format!(
			"{operation} requires device-planar frames; stage host frames with VideoFrame::upload"
		))
	})
}

fn u32_field(value: usize, operation: &str, what: &str) -> Result<u32> {
	u32::try_from(value).map_err(|_| Error::out_of_range(format!("{operation} {what} exceeds u32")))
}

fn require_even(value: usize, operation: &str, what: &str) -> Result<()> {
	if !value.is_multiple_of(2) {
		return Err(Error::invalid_argument(format!(
			"{operation} requires even {what} for 4:2:0 chroma alignment"
		)));
	}
	Ok(())
}

/// Semantic-recording fields for one `resample` operation.
struct ResampleOp {
	mode: &'static str,
	params: String,
	interpolation: VideoInterpolation,
	operation: &'static str,
}

/// One op's per-plane resample dispatches under the `resample` contract,
/// producing a new device-planar frame.
///
/// `frames` are the source planes in order; `maps` holds one luma-space map
/// per frame; `fills` holds each frame's YUV code fill. `out_width` and
/// `out_height` are the destination luma extent. Each plane's output extent,
/// stride and element grouping follow the format's plane rules.
fn run_resample(
	frames: &[&VideoFrame],
	maps: &[PlaneMap],
	out_width: usize,
	out_height: usize,
	op: ResampleOp,
) -> Result<VideoFrame> {
	let ResampleOp {
		mode,
		params,
		interpolation,
		operation,
	} = op;
	let first = require_device_planar(frames[0], operation)?;
	let format = first.format;
	let engine = first.planes[0].engine_handle();
	let timing = frames[0].timing();
	let color = frames[0].color_info();
	let mut planars = Vec::with_capacity(frames.len());
	for frame in frames {
		let planar = require_device_planar(frame, operation)?;
		if planar.format != format {
			return Err(Error::invalid_argument(format!(
				"{operation} requires frames of one pixel format"
			)));
		}
		if !planar.planes[0].engine_handle().same_as(engine) {
			return Err(Error::invalid_argument(format!(
				"{operation} requires frames from one engine"
			)));
		}
		planars.push(planar);
	}
	let extents = plane_extents(format, out_width, out_height)?;
	let mut output_planes = Vec::with_capacity(extents.len());
	for extent in &extents {
		let elements = extent
			.width
			.checked_mul(extent.height)
			.ok_or_else(|| Error::out_of_range(format!("{operation} plane extent overflows")))?;
		output_planes.push(Matrix::allocate(
			engine,
			vec![extent.height, extent.width],
			elements,
			DType::U8,
		)?);
	}
	let mut semantic_inputs: Vec<&Matrix> = Vec::new();
	let mut semantic_outputs: Vec<&Matrix> = Vec::new();
	let mut bindings: Vec<Vec<BufferBinding>> = Vec::new();
	let mut pushes = Vec::new();
	for (planar, map) in planars.iter().zip(maps.iter()) {
		for (plane_index, src) in planar.planes.iter().enumerate() {
			let factor = if plane_index == 0 { 1.0 } else { 0.5 };
			let plane_map = map.at(factor);
			let nv12_uv = format == VideoPixelFormat::Nv12 && plane_index == 1;
			let elem_stride = if nv12_uv { 2usize } else { 1usize };
			let src_component = 0usize;
			let (src_w, src_h) = if plane_index == 0 {
				(planar.width, planar.height)
			} else {
				(planar.width.div_ceil(2), planar.height.div_ceil(2))
			};
			let (local_w, local_h) = if map.local_w == 0 {
				(
					(out_width as f64 * factor).ceil() as usize,
					(out_height as f64 * factor).ceil() as usize,
				)
			} else {
				(plane_map.local_w, plane_map.local_h)
			};
			let dst_row_stride = if nv12_uv {
				extents[plane_index].width / 2
			} else {
				extents[plane_index].width
			};
			let valid = if map.valid == [0.0; 4] {
				[0.0, 0.0, local_w as f64, local_h as f64]
			} else {
				plane_map.valid
			};
			let fill = plane_map.fill[plane_index.min(2)];
			bindings.push(vec![
				BufferBinding::read(src.storage()),
				BufferBinding::write(output_planes[plane_index].storage()),
			]);
			pushes.push((
				[
					PushConstant::U32(u32_field(src_w, operation, "source width")?),
					PushConstant::U32(u32_field(src_h, operation, "source height")?),
					PushConstant::U32(elem_stride as u32),
					PushConstant::U32(elem_stride as u32),
					PushConstant::U32(src_component as u32),
					PushConstant::U32(0),
					PushConstant::U32(elem_stride as u32),
					PushConstant::U32(u32_field(local_w, operation, "output width")?),
					PushConstant::U32(u32_field(local_h, operation, "output height")?),
					PushConstant::U32(u32_field(dst_row_stride, operation, "output stride")?),
					PushConstant::U32(u32_field(plane_map.dst_off_x, operation, "offset")?),
					PushConstant::U32(u32_field(plane_map.dst_off_y, operation, "offset")?),
					PushConstant::U32(u32::from(plane_map.swap)),
					PushConstant::U32(interpolation.mode()),
					PushConstant::U32(u32::from(fill)),
					PushConstant::F32(plane_map.off_x as f32),
					PushConstant::F32(plane_map.off_y as f32),
					PushConstant::F32(plane_map.scale_x as f32),
					PushConstant::F32(plane_map.scale_y as f32),
					PushConstant::F32(valid[0] as f32),
					PushConstant::F32(valid[1] as f32),
					PushConstant::F32(valid[2] as f32),
					PushConstant::F32(valid[3] as f32),
				],
				u32_field(local_w, operation, "dispatch")?.div_ceil(16),
				u32_field(local_h, operation, "dispatch")?.div_ceil(16),
			));
			semantic_inputs.push(src);
			semantic_outputs.push(&output_planes[plane_index]);
		}
	}
	let dispatch_list: Vec<ComputeDispatch> = pushes
		.iter()
		.enumerate()
		.map(|(index, (push, wg_x, wg_y))| ComputeDispatch {
			kernel: KernelId::VideoResample,
			buffers: &bindings[index],
			push_constants: push,
			workgroups: [*wg_x, *wg_y, 1],
			indirect: None,
		})
		.collect();
	let attributes = [
		OpAttribute::Enum {
			name: "mode".into(),
			value: mode.into(),
		},
		OpAttribute::Enum {
			name: "interpolation".into(),
			value: interpolation.token().into(),
		},
		OpAttribute::UnsignedInteger {
			name: "out_width".into(),
			value: u64::from(u32_field(out_width, operation, "output width")?),
		},
		OpAttribute::UnsignedInteger {
			name: "out_height".into(),
			value: u64::from(u32_field(out_height, operation, "output height")?),
		},
		OpAttribute::String {
			name: "params".into(),
			value: params,
		},
	];
	engine.record_split_semantic(
		&dispatch_list,
		SemanticDispatch {
			contract: crate::core::operation::video::RESAMPLE,
			inputs: &semantic_inputs,
			outputs: &semantic_outputs,
			attributes: &attributes,
		},
	)?;
	Ok(VideoFrame::from_device_planar(
		DevicePlanar {
			planes: Rc::new(output_planes),
			format,
			width: out_width,
			height: out_height,
		},
		timing,
		color,
	))
}

/// Extract a luma-space region; chroma extents follow the format's rules.
///
/// `x` and `y` must be even so chroma stays subsample-aligned.
///
/// # Errors
///
/// Returns an error for a non-device-planar frame, odd offsets, or a region
/// outside the visible extent.
pub fn crop(frame: &VideoFrame, region: super::VideoRegion) -> Result<VideoFrame> {
	const OPERATION: &str = "video::crop";
	let planar = require_device_planar(frame, OPERATION)?;
	require_even(region.x as usize, OPERATION, "crop x")?;
	require_even(region.y as usize, OPERATION, "crop y")?;
	let right = u64::from(region.x) + u64::from(region.width);
	let bottom = u64::from(region.y) + u64::from(region.height);
	if region.width == 0
		|| region.height == 0
		|| right > planar.width as u64
		|| bottom > planar.height as u64
	{
		return Err(Error::invalid_argument(format!(
			"{OPERATION} region must be a non-empty area inside the visible extent"
		)));
	}
	let mut map = PlaneMap::identity([0, 128, 128]);
	map.off_x = f64::from(region.x) - 0.5;
	map.off_y = f64::from(region.y) - 0.5;
	run_resample(
		&[frame],
		&[map],
		region.width as usize,
		region.height as usize,
		ResampleOp {
			mode: "crop",
			params: format!(
				"{},{},{},{}",
				region.x, region.y, region.width, region.height
			),
			interpolation: VideoInterpolation::Nearest,
			operation: OPERATION,
		},
	)
}

/// Stretch a frame to a new extent in luma space; chroma planes follow.
///
/// # Errors
///
/// Returns an error for a non-device-planar frame or a zero extent.
pub fn resize(
	frame: &VideoFrame,
	width: u32,
	height: u32,
	interpolation: VideoInterpolation,
) -> Result<VideoFrame> {
	const OPERATION: &str = "video::resize";
	let planar = require_device_planar(frame, OPERATION)?;
	if width == 0 || height == 0 {
		return Err(Error::invalid_argument(format!(
			"{OPERATION} requires a non-zero extent"
		)));
	}
	let mut map = PlaneMap::identity([0, 128, 128]);
	map.scale_x = planar.width as f64 / f64::from(width);
	map.scale_y = planar.height as f64 / f64::from(height);
	run_resample(
		&[frame],
		&[map],
		width as usize,
		height as usize,
		ResampleOp {
			mode: "resize",
			params: format!("{width}x{height}"),
			interpolation,
			operation: OPERATION,
		},
	)
}

/// Mirror a frame along one axis in place-size output.
///
/// # Errors
///
/// Returns an error for a non-device-planar frame.
pub fn flip(frame: &VideoFrame, axis: VideoAxis) -> Result<VideoFrame> {
	const OPERATION: &str = "video::flip";
	let planar = require_device_planar(frame, OPERATION)?;
	let mut map = PlaneMap::identity([0, 128, 128]);
	match axis {
		VideoAxis::Horizontal => {
			map.scale_x = -1.0;
			map.off_x = planar.width as f64 - 0.5;
		}
		VideoAxis::Vertical => {
			map.scale_y = -1.0;
			map.off_y = planar.height as f64 - 0.5;
		}
	}
	run_resample(
		&[frame],
		&[map],
		planar.width,
		planar.height,
		ResampleOp {
			mode: "flip",
			params: axis.token().to_string(),
			interpolation: VideoInterpolation::Nearest,
			operation: OPERATION,
		},
	)
}

/// Rotate a frame ninety degrees clockwise.
///
/// # Errors
///
/// Returns an error for a non-device-planar frame.
pub fn rotate90(frame: &VideoFrame) -> Result<VideoFrame> {
	const OPERATION: &str = "video::rotate90";
	let planar = require_device_planar(frame, OPERATION)?;
	let mut map = PlaneMap::identity([0, 128, 128]);
	// Clockwise: dst (x, y) samples src (y, h-1-x) via the swapped mapping.
	map.swap = true;
	map.off_y = -0.5;
	map.off_x = planar.height as f64 - 0.5;
	map.scale_x = -1.0;
	map.scale_y = 1.0;
	run_resample(
		&[frame],
		&[map],
		planar.height,
		planar.width,
		ResampleOp {
			mode: "rotate90",
			params: "clockwise".to_string(),
			interpolation: VideoInterpolation::Nearest,
			operation: OPERATION,
		},
	)
}

/// Border extents for [`pad`]: left, top, right, bottom in luma pixels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VideoPadding {
	pub left: u32,
	pub top: u32,
	pub right: u32,
	pub bottom: u32,
}

/// Extend a frame with a constant border. `fill` is a YUV code triplet in
/// the frame's component range. `left` and `top` must be even so chroma
/// stays subsample-aligned.
///
/// # Errors
///
/// Returns an error for a non-device-planar frame, odd offsets, or extent
/// overflow.
pub fn pad(frame: &VideoFrame, padding: VideoPadding, fill: [u8; 3]) -> Result<VideoFrame> {
	const OPERATION: &str = "video::pad";
	let planar = require_device_planar(frame, OPERATION)?;
	require_even(padding.left as usize, OPERATION, "left padding")?;
	require_even(padding.top as usize, OPERATION, "top padding")?;
	let out_width = (planar.width as u64)
		.checked_add(u64::from(padding.left) + u64::from(padding.right))
		.ok_or_else(|| Error::out_of_range(format!("{OPERATION} output width overflows")))?;
	let out_height = (planar.height as u64)
		.checked_add(u64::from(padding.top) + u64::from(padding.bottom))
		.ok_or_else(|| Error::out_of_range(format!("{OPERATION} output height overflows")))?;
	if out_width == 0
		|| out_height == 0
		|| out_width > u64::from(u32::MAX)
		|| out_height > u64::from(u32::MAX)
	{
		return Err(Error::invalid_argument(format!(
			"{OPERATION} requires a positive output extent"
		)));
	}
	let mut map = PlaneMap::identity(fill);
	map.off_x = -f64::from(padding.left) - 0.5;
	map.off_y = -f64::from(padding.top) - 0.5;
	map.valid = [
		f64::from(padding.left),
		f64::from(padding.top),
		f64::from(padding.left) + planar.width as f64,
		f64::from(padding.top) + planar.height as f64,
	];
	run_resample(
		&[frame],
		&[map],
		out_width as usize,
		out_height as usize,
		ResampleOp {
			mode: "pad",
			params: format!(
				"{},{},{},{}",
				padding.left, padding.top, padding.right, padding.bottom
			),
			interpolation: VideoInterpolation::Nearest,
			operation: OPERATION,
		},
	)
}

/// Aspect-preserving resize into a fixed extent with a constant fill. `fill`
/// is a YUV code triplet in the frame's component range.
///
/// # Errors
///
/// Returns an error for a non-device-planar frame or a zero extent.
pub fn letterbox(
	frame: &VideoFrame,
	width: u32,
	height: u32,
	fill: [u8; 3],
	interpolation: VideoInterpolation,
) -> Result<VideoFrame> {
	const OPERATION: &str = "video::letterbox";
	let planar = require_device_planar(frame, OPERATION)?;
	if width == 0 || height == 0 {
		return Err(Error::invalid_argument(format!(
			"{OPERATION} requires a non-zero extent"
		)));
	}
	let scale =
		(f64::from(width) / planar.width as f64).min(f64::from(height) / planar.height as f64);
	let map_w = planar.width as f64 * scale;
	let map_h = planar.height as f64 * scale;
	let ix0 = (f64::from(width) - map_w) * 0.5;
	let iy0 = (f64::from(height) - map_h) * 0.5;
	let mut map = PlaneMap::identity(fill);
	map.scale_x = planar.width as f64 / map_w;
	map.scale_y = planar.height as f64 / map_h;
	map.off_x = -ix0 * map.scale_x - 0.5;
	map.off_y = -iy0 * map.scale_y - 0.5;
	map.valid = [ix0, iy0, ix0 + map_w, iy0 + map_h];
	run_resample(
		&[frame],
		&[map],
		width as usize,
		height as usize,
		ResampleOp {
			mode: "letterbox",
			params: format!("{width}x{height}"),
			interpolation,
			operation: OPERATION,
		},
	)
}

/// Convert a frame between `Nv12` and `Yuv420Planar8` plane layouts without
/// changing content, extent, or component ranges.
///
/// Each plane is repacked by a `resample` dispatch at identity mapping;
/// NV12's interleaved chroma pair splits into two dispatches (or merges from
/// two) through per-component stride selection.
///
/// # Errors
///
/// Returns an error for a non-device-planar frame, a same-format or
/// non-admitted target, or a runtime recording failure.
pub fn convert_format(frame: &VideoFrame, format: VideoPixelFormat) -> Result<VideoFrame> {
	const OPERATION: &str = "video::convert_format";
	let planar = require_device_planar(frame, OPERATION)?;
	if planar.format == format {
		return Err(Error::invalid_argument(format!(
			"{OPERATION} requires a different target format"
		)));
	}
	let (interleaved_in, interleaved_out) = match (planar.format, format) {
		(VideoPixelFormat::Nv12, VideoPixelFormat::Yuv420Planar8) => (true, false),
		(VideoPixelFormat::Yuv420Planar8, VideoPixelFormat::Nv12) => (false, true),
		_ => {
			return Err(Error::missing_capability(format!(
				"{OPERATION} supports only NV12 <-> I420 repacking before VNV-07 P010 work"
			)));
		}
	};
	let engine = planar.planes[0].engine_handle();
	let extents = plane_extents(format, planar.width, planar.height)?;
	let mut output_planes = Vec::with_capacity(extents.len());
	for extent in &extents {
		let elements = extent
			.width
			.checked_mul(extent.height)
			.ok_or_else(|| Error::out_of_range(format!("{OPERATION} plane extent overflows")))?;
		output_planes.push(Matrix::allocate(
			engine,
			vec![extent.height, extent.width],
			elements,
			DType::U8,
		)?);
	}
	let chroma_w = planar.width.div_ceil(2);
	let chroma_h = planar.height.div_ceil(2);
	// (src_plane, src_stride, src_component, dst_plane, dst_stride, groups_w)
	let mut jobs = vec![(
		0usize,
		1usize,
		0usize,
		0usize,
		1usize,
		0usize,
		1usize,
		planar.width,
	)];
	if interleaved_in {
		jobs.push((1, 2, 0, 1, 1, 0, 1, chroma_w));
		jobs.push((1, 2, 1, 2, 1, 0, 1, chroma_w));
	} else {
		jobs.push((1, 1, 0, 1, 2, 0, 1, chroma_w));
		jobs.push((2, 1, 0, 1, 2, 1, 1, chroma_w));
	}
	let mut semantic_inputs: Vec<&Matrix> = Vec::new();
	let mut semantic_outputs: Vec<&Matrix> = Vec::new();
	let mut bindings: Vec<Vec<BufferBinding>> = Vec::new();
	let mut pushes = Vec::new();
	for (
		src_plane,
		src_stride,
		src_component,
		dst_plane,
		dst_stride,
		dst_component,
		elem_count,
		groups_w,
	) in jobs
	{
		let src = &planar.planes[src_plane];
		let dst = &output_planes[dst_plane];
		let src_h = if src_plane == 0 {
			planar.height
		} else {
			chroma_h
		};
		let dst_h = if dst_plane == 0 {
			planar.height
		} else {
			chroma_h
		};
		let stride = if dst_stride == 2 {
			groups_w
		} else {
			extents[dst_plane].width
		};
		bindings.push(vec![
			BufferBinding::read(src.storage()),
			BufferBinding::write(dst.storage()),
		]);
		pushes.push((
			[
				PushConstant::U32(u32_field(groups_w, OPERATION, "source width")?),
				PushConstant::U32(u32_field(src_h, OPERATION, "source height")?),
				PushConstant::U32(src_stride as u32),
				PushConstant::U32(dst_stride as u32),
				PushConstant::U32(src_component as u32),
				PushConstant::U32(dst_component as u32),
				PushConstant::U32(elem_count as u32),
				PushConstant::U32(u32_field(groups_w, OPERATION, "output width")?),
				PushConstant::U32(u32_field(dst_h, OPERATION, "output height")?),
				PushConstant::U32(u32_field(stride, OPERATION, "output stride")?),
				PushConstant::U32(0),
				PushConstant::U32(0),
				PushConstant::U32(0),
				PushConstant::U32(0),
				PushConstant::U32(0),
				PushConstant::F32(-0.5),
				PushConstant::F32(-0.5),
				PushConstant::F32(1.0),
				PushConstant::F32(1.0),
				PushConstant::F32(0.0),
				PushConstant::F32(0.0),
				PushConstant::F32(groups_w as f32),
				PushConstant::F32(dst_h as f32),
			],
			u32_field(groups_w, OPERATION, "dispatch")?.div_ceil(16),
			u32_field(dst_h, OPERATION, "dispatch")?.div_ceil(16),
		));
		semantic_inputs.push(src);
		if !(interleaved_out && dst_plane == 1 && dst_component == 1) {
			semantic_outputs.push(dst);
		}
	}
	let dispatch_list: Vec<ComputeDispatch> = pushes
		.iter()
		.enumerate()
		.map(|(index, (push, wg_x, wg_y))| ComputeDispatch {
			kernel: KernelId::VideoResample,
			buffers: &bindings[index],
			push_constants: push,
			workgroups: [*wg_x, *wg_y, 1],
			indirect: None,
		})
		.collect();
	let attributes = [
		OpAttribute::Enum {
			name: "mode".into(),
			value: "convert_format".into(),
		},
		OpAttribute::Enum {
			name: "interpolation".into(),
			value: "nearest".into(),
		},
		OpAttribute::UnsignedInteger {
			name: "out_width".into(),
			value: u64::from(u32_field(planar.width, OPERATION, "width")?),
		},
		OpAttribute::UnsignedInteger {
			name: "out_height".into(),
			value: u64::from(u32_field(planar.height, OPERATION, "height")?),
		},
		OpAttribute::String {
			name: "params".into(),
			value: format!("{format:?}"),
		},
	];
	engine.record_split_semantic(
		&dispatch_list,
		SemanticDispatch {
			contract: crate::core::operation::video::RESAMPLE,
			inputs: &semantic_inputs,
			outputs: &semantic_outputs,
			attributes: &attributes,
		},
	)?;
	Ok(VideoFrame::from_device_planar(
		DevicePlanar {
			planes: Rc::new(output_planes),
			format,
			width: planar.width,
			height: planar.height,
		},
		frame.timing(),
		frame.color_info(),
	))
}

/// One filled rectangle for [`overlay_boxes`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VideoBox {
	/// Luma-space bounds; chroma edges scale down with centered siting.
	pub region: super::VideoRegion,
	/// YUV code triplet in the frame's component range.
	pub fill: [u8; 3],
}

/// Draw rectangles into a new device-planar frame.
///
/// The base frame is copied once at identity; each box then writes its
/// scaled region on luma and each chroma component with exclusive
/// single-component dispatches. `border` draws a hollow ring of that width
/// in luma pixels instead of filling.
///
/// # Errors
///
/// Returns an error for a non-device-planar frame, a box outside the
/// visible extent, or a runtime recording failure.
pub fn overlay_boxes(
	frame: &VideoFrame,
	boxes: &[VideoBox],
	border: Option<u32>,
) -> Result<VideoFrame> {
	const OPERATION: &str = "video::overlay_boxes";
	let planar = require_device_planar(frame, OPERATION)?;
	let engine = planar.planes[0].engine_handle();
	for b in boxes {
		let right = u64::from(b.region.x) + u64::from(b.region.width);
		let bottom = u64::from(b.region.y) + u64::from(b.region.height);
		if b.region.width == 0
			|| b.region.height == 0
			|| right > planar.width as u64
			|| bottom > planar.height as u64
		{
			return Err(Error::invalid_argument(format!(
				"{OPERATION} boxes must be non-empty regions inside the visible extent"
			)));
		}
	}
	// Identity copy into fresh output planes, then per-box writes.
	let extents = plane_extents(planar.format, planar.width, planar.height)?;
	let mut output_planes = Vec::with_capacity(extents.len());
	for extent in &extents {
		let elements = extent
			.width
			.checked_mul(extent.height)
			.ok_or_else(|| Error::out_of_range(format!("{OPERATION} plane extent overflows")))?;
		output_planes.push(Matrix::allocate(
			engine,
			vec![extent.height, extent.width],
			elements,
			DType::U8,
		)?);
	}
	let mut semantic_inputs: Vec<&Matrix> = Vec::new();
	let mut semantic_outputs: Vec<&Matrix> = Vec::new();
	let mut bindings: Vec<Vec<BufferBinding>> = Vec::new();
	let mut pushes: Vec<([PushConstant; 11], u32, u32)> = Vec::new();
	// (src storage, dst storage, plane_w, plane_h, elem_stride, src_comp, dst_comp, rect, fill, border)
	struct Job<'a> {
		src: &'a crate::runtime::Storage,
		dst: &'a crate::runtime::Storage,
		width: usize,
		height: usize,
		stride: u32,
		src_comp: u32,
		dst_comp: u32,
	}
	let mut jobs: Vec<Job> = Vec::new();
	for (plane_index, src) in planar.planes.iter().enumerate() {
		let stride = if planar.format == VideoPixelFormat::Nv12 && plane_index == 1 {
			2u32
		} else {
			1
		};
		let (w, h) = if plane_index == 0 {
			(planar.width, planar.height)
		} else {
			(planar.width.div_ceil(2), planar.height.div_ceil(2))
		};
		for comp in 0..stride {
			jobs.push(Job {
				src: src.storage(),
				dst: output_planes[plane_index].storage(),
				width: w,
				height: h,
				stride,
				src_comp: comp,
				dst_comp: comp,
			});
		}
	}
	for plane in planar.planes.iter() {
		semantic_inputs.push(plane);
	}
	for plane in output_planes.iter() {
		semantic_outputs.push(plane);
	}
	// First pass per plane-component: identity copy (empty rect).
	for job in &jobs {
		bindings.push(vec![
			BufferBinding::read(job.src),
			BufferBinding::write(job.dst),
		]);
		pushes.push((
			[
				PushConstant::U32(u32_field(job.width, OPERATION, "plane")?),
				PushConstant::U32(u32_field(job.height, OPERATION, "plane")?),
				PushConstant::U32(0),
				PushConstant::U32(0),
				PushConstant::U32(0),
				PushConstant::U32(0),
				PushConstant::U32(job.stride),
				PushConstant::U32(job.src_comp),
				PushConstant::U32(job.dst_comp),
				PushConstant::U32(0),
				PushConstant::U32(0),
			],
			u32_field(job.width, OPERATION, "dispatch")?.div_ceil(16),
			u32_field(job.height, OPERATION, "dispatch")?.div_ceil(16),
		));
	}
	// Per-box passes on each plane component.
	for (plane_index, _src) in planar.planes.iter().enumerate() {
		let factor = if plane_index == 0 { 1.0f64 } else { 0.5 };
		let stride = if planar.format == VideoPixelFormat::Nv12 && plane_index == 1 {
			2u32
		} else {
			1
		};
		let (w, h) = if plane_index == 0 {
			(planar.width, planar.height)
		} else {
			(planar.width.div_ceil(2), planar.height.div_ceil(2))
		};
		for comp in 0..stride {
			for b in boxes {
				let rect = [
					(b.region.x as f64 * factor) as u32,
					(b.region.y as f64 * factor) as u32,
					(b.region.width as f64 * factor).ceil() as u32,
					(b.region.height as f64 * factor).ceil() as u32,
				];
				let fill = u32::from(if plane_index == 0 {
					b.fill[0]
				} else if plane_index == 1 {
					b.fill[1]
				} else {
					b.fill[2]
				});
				let border_w = border.map(|w| (w as f64 * factor) as u32).unwrap_or(0);
				bindings.push(vec![
					BufferBinding::read(output_planes[plane_index].storage()),
					BufferBinding::write(output_planes[plane_index].storage()),
				]);
				pushes.push((
					[
						PushConstant::U32(u32_field(w, OPERATION, "plane")?),
						PushConstant::U32(u32_field(h, OPERATION, "plane")?),
						PushConstant::U32(rect[0]),
						PushConstant::U32(rect[1]),
						PushConstant::U32(rect[2]),
						PushConstant::U32(rect[3]),
						PushConstant::U32(stride),
						PushConstant::U32(comp),
						PushConstant::U32(comp),
						PushConstant::U32(fill),
						PushConstant::U32(border_w),
					],
					u32_field(w, OPERATION, "dispatch")?.div_ceil(16),
					u32_field(h, OPERATION, "dispatch")?.div_ceil(16),
				));
			}
		}
	}
	let dispatch_list: Vec<ComputeDispatch> = pushes
		.iter()
		.enumerate()
		.map(|(index, (push, wg_x, wg_y))| ComputeDispatch {
			kernel: KernelId::VideoOverlayRect,
			buffers: &bindings[index],
			push_constants: push,
			workgroups: [*wg_x, *wg_y, 1],
			indirect: None,
		})
		.collect();
	let attributes = [
		OpAttribute::Enum {
			name: "mode".into(),
			value: "overlay_boxes".into(),
		},
		OpAttribute::UnsignedInteger {
			name: "box_count".into(),
			value: boxes.len() as u64,
		},
		OpAttribute::UnsignedInteger {
			name: "border_width".into(),
			value: u64::from(border.unwrap_or(0)),
		},
		OpAttribute::String {
			name: "params".into(),
			value: String::new(),
		},
	];
	engine.record_split_semantic(
		&dispatch_list,
		SemanticDispatch {
			contract: crate::core::operation::video::OVERLAY,
			inputs: &semantic_inputs,
			outputs: &semantic_outputs,
			attributes: &attributes,
		},
	)?;
	Ok(VideoFrame::from_device_planar(
		DevicePlanar {
			planes: output_planes.clone().into(),
			format: planar.format,
			width: planar.width,
			height: planar.height,
		},
		frame.timing(),
		frame.color_info(),
	))
}

/// Join frames of matching cross-extent along one axis.
///
/// Along-axis extents of all but the last frame must be even so chroma
/// offsets stay subsample-aligned.
///
/// # Errors
///
/// Returns an error for an empty batch, non-device-planar frames, mixed
/// formats or cross-extents, odd alignment, or extent overflow.
pub fn stack(frames: &[VideoFrame], axis: VideoAxis) -> Result<VideoFrame> {
	const OPERATION: &str = "video::stack";
	if frames.is_empty() {
		return Err(Error::invalid_argument(format!(
			"{OPERATION} requires at least one frame"
		)));
	}
	let first = require_device_planar(&frames[0], OPERATION)?;
	let (cross, mut along) = match axis {
		VideoAxis::Horizontal => (first.height, first.width),
		VideoAxis::Vertical => (first.width, first.height),
	};
	for frame in &frames[1..] {
		let planar = require_device_planar(frame, OPERATION)?;
		let (frame_cross, frame_along) = match axis {
			VideoAxis::Horizontal => (planar.height, planar.width),
			VideoAxis::Vertical => (planar.width, planar.height),
		};
		if frame_cross != cross {
			return Err(Error::invalid_argument(format!(
				"{OPERATION} requires matching cross-axis extents"
			)));
		}
		require_even(along, OPERATION, "stacked extent")?;
		along = along
			.checked_add(frame_along)
			.ok_or_else(|| Error::out_of_range(format!("{OPERATION} extent overflows")))?;
	}
	let (out_width, out_height) = match axis {
		VideoAxis::Horizontal => (along, cross),
		VideoAxis::Vertical => (cross, along),
	};
	let mut maps = Vec::with_capacity(frames.len());
	let mut offset = 0usize;
	for frame in frames {
		let planar = require_device_planar(frame, OPERATION)?;
		let mut map = PlaneMap::identity([0, 128, 128]);
		map.local_w = planar.width;
		map.local_h = planar.height;
		match axis {
			VideoAxis::Horizontal => {
				map.dst_off_x = offset;
				offset += planar.width;
			}
			VideoAxis::Vertical => {
				map.dst_off_y = offset;
				offset += planar.height;
			}
		}
		maps.push(map);
	}
	let frame_refs: Vec<&VideoFrame> = frames.iter().collect();
	run_resample(
		&frame_refs,
		&maps,
		out_width,
		out_height,
		ResampleOp {
			mode: "stack",
			params: format!("{},{}", axis.token(), frames.len()),
			interpolation: VideoInterpolation::Nearest,
			operation: OPERATION,
		},
	)
}

/// Per-plane dispatch plumbing shared by the single-input plane ops.
fn alloc_output(
	planar: &DevicePlanar,
	format: VideoPixelFormat,
	operation: &str,
) -> Result<(Rc<Vec<Matrix>>, Vec<super::plane::PlaneExtent>)> {
	let engine = planar.planes[0].engine_handle();
	let extents = plane_extents(format, planar.width, planar.height)?;
	let mut planes = Vec::with_capacity(extents.len());
	for extent in &extents {
		let elements = extent
			.width
			.checked_mul(extent.height)
			.ok_or_else(|| Error::out_of_range(format!("{operation} plane extent overflows")))?;
		planes.push(Matrix::allocate(
			engine,
			vec![extent.height, extent.width],
			elements,
			DType::U8,
		)?);
	}
	Ok((Rc::new(planes), extents))
}

fn record_frame(
	planar: &DevicePlanar,
	format: VideoPixelFormat,
	planes: Rc<Vec<Matrix>>,
	dispatch_list: &[ComputeDispatch<'_>],
	semantic: SemanticDispatch<'_>,
	timing: super::VideoFrameTiming,
	color: super::VideoColorInfo,
) -> Result<VideoFrame> {
	let engine = planar.planes[0].engine_handle();
	engine.record_split_semantic(dispatch_list, semantic)?;
	Ok(VideoFrame::from_device_planar(
		DevicePlanar {
			planes,
			format,
			width: planar.width,
			height: planar.height,
		},
		timing,
		color,
	))
}

/// Stencil `fill` onto a frame wherever the mask's luma byte is nonzero.
///
/// `mask` must be a device-planar frame of the same luma extent whose luma
/// plane is the stencil. Chroma components apply the mask at chroma
/// resolution; edge columns under odd extents follow the chroma rounding.
///
/// # Errors
///
/// Returns an error for non-device-planar inputs, mismatched extents, or a
/// runtime recording failure.
pub fn overlay_mask(frame: &VideoFrame, mask: &VideoFrame, fill: [u8; 3]) -> Result<VideoFrame> {
	const OPERATION: &str = "video::overlay_mask";
	let planar = require_device_planar(frame, OPERATION)?;
	let mask_planar = require_device_planar(mask, OPERATION)?;
	if mask_planar.width != planar.width || mask_planar.height != planar.height {
		return Err(Error::invalid_argument(format!(
			"{OPERATION} requires a mask of matching luma extent"
		)));
	}
	let engine = planar.planes[0].engine_handle();
	if !mask_planar.planes[0].engine_handle().same_as(engine) {
		return Err(Error::invalid_argument(format!(
			"{OPERATION} requires one engine"
		)));
	}
	let (output_planes, _) = alloc_output(planar, planar.format, OPERATION)?;
	let mut semantic_inputs: Vec<&Matrix> = Vec::new();
	let mut semantic_outputs: Vec<&Matrix> = Vec::new();
	let mut bindings: Vec<Vec<BufferBinding>> = Vec::new();
	let mut pushes = Vec::new();
	for (plane_index, src) in planar.planes.iter().enumerate() {
		let stride = if planar.format == VideoPixelFormat::Nv12 && plane_index == 1 {
			2u32
		} else {
			1
		};
		let (w, h) = if plane_index == 0 {
			(planar.width, planar.height)
		} else {
			(planar.width.div_ceil(2), planar.height.div_ceil(2))
		};
		for comp in 0..stride {
			let fill_code = u32::from(if plane_index == 0 {
				fill[0]
			} else if comp == 0 {
				fill[1]
			} else {
				fill[2]
			});
			bindings.push(vec![
				BufferBinding::read(src.storage()),
				BufferBinding::read(mask_planar.planes[0].storage()),
				BufferBinding::write(output_planes[plane_index].storage()),
			]);
			pushes.push((
				[
					PushConstant::U32(u32_field(w, OPERATION, "plane")?),
					PushConstant::U32(u32_field(h, OPERATION, "plane")?),
					PushConstant::U32(stride),
					PushConstant::U32(comp),
					PushConstant::U32(0),
					PushConstant::U32(0),
					PushConstant::U32(fill_code),
					PushConstant::U32(u32_field(mask_planar.width, OPERATION, "mask")?),
					PushConstant::U32(u32_field(mask_planar.height, OPERATION, "mask")?),
					PushConstant::U32(if plane_index == 0 { 0 } else { 2 }),
				],
				u32_field(w, OPERATION, "dispatch")?.div_ceil(16),
				u32_field(h, OPERATION, "dispatch")?.div_ceil(16),
			));
		}
		semantic_inputs.push(src);
		semantic_inputs.push(&mask_planar.planes[0]);
		semantic_outputs.push(&output_planes[plane_index]);
	}
	let dispatch_list: Vec<ComputeDispatch> = pushes
		.iter()
		.enumerate()
		.map(|(index, (push, wg_x, wg_y))| ComputeDispatch {
			kernel: KernelId::VideoBlend,
			buffers: &bindings[index],
			push_constants: push,
			workgroups: [*wg_x, *wg_y, 1],
			indirect: None,
		})
		.collect();
	let attributes = [
		OpAttribute::Enum {
			name: "mode".into(),
			value: "stencil".into(),
		},
		OpAttribute::UnsignedInteger {
			name: "alpha".into(),
			value: 0,
		},
		OpAttribute::UnsignedInteger {
			name: "fill".into(),
			value: u64::from(fill[0]) << 16 | u64::from(fill[1]) << 8 | u64::from(fill[2]),
		},
		OpAttribute::String {
			name: "params".into(),
			value: String::new(),
		},
	];
	record_frame(
		planar,
		planar.format,
		output_planes.clone(),
		&dispatch_list,
		SemanticDispatch {
			contract: crate::core::operation::video::BLEND,
			inputs: &semantic_inputs,
			outputs: &semantic_outputs,
			attributes: &attributes,
		},
		frame.timing(),
		frame.color_info(),
	)
}

/// Alpha-composite `front` over `back` plane-by-plane; `alpha` is 0-255.
///
/// Both frames must share format, extent, and engine. Each component blends
/// with rounding `(front*a + back*(255-a) + 127) / 255`.
///
/// # Errors
///
/// Returns an error for non-device-planar inputs, mismatched frames, or a
/// runtime recording failure.
pub fn composite(back: &VideoFrame, front: &VideoFrame, alpha: u8) -> Result<VideoFrame> {
	const OPERATION: &str = "video::composite";
	let planar = require_device_planar(back, OPERATION)?;
	let front_planar = require_device_planar(front, OPERATION)?;
	if front_planar.format != planar.format
		|| front_planar.width != planar.width
		|| front_planar.height != planar.height
	{
		return Err(Error::invalid_argument(format!(
			"{OPERATION} requires matching format and extent"
		)));
	}
	let engine = planar.planes[0].engine_handle();
	if !front_planar.planes[0].engine_handle().same_as(engine) {
		return Err(Error::invalid_argument(format!(
			"{OPERATION} requires one engine"
		)));
	}
	let (output_planes, _) = alloc_output(planar, planar.format, OPERATION)?;
	let mut semantic_inputs: Vec<&Matrix> = Vec::new();
	let mut semantic_outputs: Vec<&Matrix> = Vec::new();
	let mut bindings: Vec<Vec<BufferBinding>> = Vec::new();
	let mut pushes = Vec::new();
	for (plane_index, src) in planar.planes.iter().enumerate() {
		let stride = if planar.format == VideoPixelFormat::Nv12 && plane_index == 1 {
			2u32
		} else {
			1
		};
		let (w, h) = if plane_index == 0 {
			(planar.width, planar.height)
		} else {
			(planar.width.div_ceil(2), planar.height.div_ceil(2))
		};
		for comp in 0..stride {
			bindings.push(vec![
				BufferBinding::read(src.storage()),
				BufferBinding::read(front_planar.planes[plane_index].storage()),
				BufferBinding::write(output_planes[plane_index].storage()),
			]);
			pushes.push((
				[
					PushConstant::U32(u32_field(w, OPERATION, "plane")?),
					PushConstant::U32(u32_field(h, OPERATION, "plane")?),
					PushConstant::U32(stride),
					PushConstant::U32(comp),
					PushConstant::U32(1),
					PushConstant::U32(u32::from(alpha)),
					PushConstant::U32(0),
					PushConstant::U32(0),
					PushConstant::U32(0),
					PushConstant::U32(0),
				],
				u32_field(w, OPERATION, "dispatch")?.div_ceil(16),
				u32_field(h, OPERATION, "dispatch")?.div_ceil(16),
			));
		}
		semantic_inputs.push(src);
		semantic_inputs.push(&front_planar.planes[plane_index]);
		semantic_outputs.push(&output_planes[plane_index]);
	}
	let dispatch_list: Vec<ComputeDispatch> = pushes
		.iter()
		.enumerate()
		.map(|(index, (push, wg_x, wg_y))| ComputeDispatch {
			kernel: KernelId::VideoBlend,
			buffers: &bindings[index],
			push_constants: push,
			workgroups: [*wg_x, *wg_y, 1],
			indirect: None,
		})
		.collect();
	let attributes = [
		OpAttribute::Enum {
			name: "mode".into(),
			value: "alpha".into(),
		},
		OpAttribute::UnsignedInteger {
			name: "alpha".into(),
			value: u64::from(alpha),
		},
		OpAttribute::UnsignedInteger {
			name: "fill".into(),
			value: 0,
		},
		OpAttribute::String {
			name: "params".into(),
			value: String::new(),
		},
	];
	record_frame(
		planar,
		planar.format,
		output_planes.clone(),
		&dispatch_list,
		SemanticDispatch {
			contract: crate::core::operation::video::BLEND,
			inputs: &semantic_inputs,
			outputs: &semantic_outputs,
			attributes: &attributes,
		},
		back.timing(),
		back.color_info(),
	)
}

/// Convert a frame's component range between limited and full.
///
/// The frame's color metadata is updated to the target range; `Unspecified`
/// input uses BT.709 limited as the source. Same-range requests return a
/// copy at identity.
///
/// # Errors
///
/// Returns an error for a non-device-planar frame or a runtime recording
/// failure.
pub fn convert_range(frame: &VideoFrame, range: super::VideoColorRange) -> Result<VideoFrame> {
	const OPERATION: &str = "video::convert_range";
	let planar = require_device_planar(frame, OPERATION)?;
	let source = match frame.color_info().range() {
		super::VideoColorRange::Unspecified => super::VideoColorRange::Limited,
		explicit => explicit,
	};
	let direction = match (source, range) {
		(super::VideoColorRange::Limited, super::VideoColorRange::Full) => 0u32,
		(super::VideoColorRange::Full, super::VideoColorRange::Limited) => 1u32,
		(super::VideoColorRange::Unspecified, _) | (_, super::VideoColorRange::Unspecified) => {
			return Err(Error::invalid_argument(format!(
				"{OPERATION} requires resolved source and target ranges"
			)));
		}
		_ => u32::MAX,
	};
	let (output_planes, _) = alloc_output(planar, planar.format, OPERATION)?;
	let mut semantic_inputs: Vec<&Matrix> = Vec::new();
	let mut semantic_outputs: Vec<&Matrix> = Vec::new();
	let mut bindings: Vec<Vec<BufferBinding>> = Vec::new();
	let mut pushes = Vec::new();
	for (plane_index, src) in planar.planes.iter().enumerate() {
		let stride = if planar.format == VideoPixelFormat::Nv12 && plane_index == 1 {
			2u32
		} else {
			1
		};
		let (w, h) = if plane_index == 0 {
			(planar.width, planar.height)
		} else {
			(planar.width.div_ceil(2), planar.height.div_ceil(2))
		};
		let kind = if plane_index == 0 { 0u32 } else { 1u32 };
		for comp in 0..stride {
			bindings.push(vec![
				BufferBinding::read(src.storage()),
				BufferBinding::write(output_planes[plane_index].storage()),
			]);
			pushes.push((
				[
					PushConstant::U32(u32_field(w, OPERATION, "plane")?),
					PushConstant::U32(u32_field(h, OPERATION, "plane")?),
					PushConstant::U32(stride),
					PushConstant::U32(comp),
					PushConstant::U32(kind),
					PushConstant::U32(if direction == u32::MAX { 2 } else { direction }),
				],
				u32_field(w, OPERATION, "dispatch")?.div_ceil(16),
				u32_field(h, OPERATION, "dispatch")?.div_ceil(16),
			));
		}
		semantic_inputs.push(src);
		semantic_outputs.push(&output_planes[plane_index]);
	}
	let dispatch_list: Vec<ComputeDispatch> = pushes
		.iter()
		.enumerate()
		.map(|(index, (push, wg_x, wg_y))| ComputeDispatch {
			kernel: KernelId::VideoRangeConvert,
			buffers: &bindings[index],
			push_constants: push,
			workgroups: [*wg_x, *wg_y, 1],
			indirect: None,
		})
		.collect();
	let attributes = [
		OpAttribute::Enum {
			name: "mode".into(),
			value: if direction == 0 {
				"limited_to_full"
			} else if direction == 1 {
				"full_to_limited"
			} else {
				"copy"
			}
			.into(),
		},
		OpAttribute::Enum {
			name: "source_range".into(),
			value: match source {
				super::VideoColorRange::Full => "full",
				_ => "limited",
			}
			.into(),
		},
		OpAttribute::Enum {
			name: "target_range".into(),
			value: match range {
				super::VideoColorRange::Full => "full",
				_ => "limited",
			}
			.into(),
		},
		OpAttribute::String {
			name: "params".into(),
			value: String::new(),
		},
	];
	let result = record_frame(
		planar,
		planar.format,
		output_planes.clone(),
		&dispatch_list,
		SemanticDispatch {
			contract: crate::core::operation::video::RANGE_CONVERT,
			inputs: &semantic_inputs,
			outputs: &semantic_outputs,
			attributes: &attributes,
		},
		frame.timing(),
		VideoColorInfo::new(frame.color_info().matrix(), range),
	)?;
	Ok(result)
}

/// Threshold a frame's luma plane; chroma planes are copied unchanged.
///
/// `value >= threshold` maps to `high`, otherwise `low`.
///
/// # Errors
///
/// Returns an error for a non-device-planar frame or a runtime recording
/// failure.
pub fn threshold(frame: &VideoFrame, threshold: u8, low: u8, high: u8) -> Result<VideoFrame> {
	const OPERATION: &str = "video::threshold";
	let planar = require_device_planar(frame, OPERATION)?;
	let (output_planes, _) = alloc_output(planar, planar.format, OPERATION)?;
	let mut semantic_inputs: Vec<&Matrix> = Vec::new();
	let mut semantic_outputs: Vec<&Matrix> = Vec::new();
	let mut bindings: Vec<Vec<BufferBinding>> = Vec::new();
	let mut pushes = Vec::new();
	for (plane_index, src) in planar.planes.iter().enumerate() {
		let stride = if planar.format == VideoPixelFormat::Nv12 && plane_index == 1 {
			2u32
		} else {
			1
		};
		let (w, h) = if plane_index == 0 {
			(planar.width, planar.height)
		} else {
			(planar.width.div_ceil(2), planar.height.div_ceil(2))
		};
		for comp in 0..stride {
			let is_luma = plane_index == 0;
			bindings.push(vec![
				BufferBinding::read(src.storage()),
				BufferBinding::write(output_planes[plane_index].storage()),
			]);
			pushes.push((
				[
					PushConstant::U32(u32_field(w, OPERATION, "plane")?),
					PushConstant::U32(u32_field(h, OPERATION, "plane")?),
					PushConstant::U32(stride),
					PushConstant::U32(comp),
					PushConstant::U32(if is_luma { 0 } else { 3 }),
					PushConstant::U32(u32::from(threshold)),
					PushConstant::U32(u32::from(low)),
					PushConstant::U32(u32::from(high)),
				],
				u32_field(w, OPERATION, "dispatch")?.div_ceil(16),
				u32_field(h, OPERATION, "dispatch")?.div_ceil(16),
			));
		}
		semantic_inputs.push(src);
		semantic_outputs.push(&output_planes[plane_index]);
	}
	let dispatch_list: Vec<ComputeDispatch> = pushes
		.iter()
		.enumerate()
		.map(|(index, (push, wg_x, wg_y))| ComputeDispatch {
			kernel: KernelId::VideoLumaMap,
			buffers: &bindings[index],
			push_constants: push,
			workgroups: [*wg_x, *wg_y, 1],
			indirect: None,
		})
		.collect();
	let attributes = [
		OpAttribute::Enum {
			name: "mode".into(),
			value: "threshold".into(),
		},
		OpAttribute::UnsignedInteger {
			name: "threshold".into(),
			value: u64::from(threshold),
		},
		OpAttribute::UnsignedInteger {
			name: "low".into(),
			value: u64::from(low),
		},
		OpAttribute::UnsignedInteger {
			name: "high".into(),
			value: u64::from(high),
		},
	];
	record_frame(
		planar,
		planar.format,
		output_planes.clone(),
		&dispatch_list,
		SemanticDispatch {
			contract: crate::core::operation::video::LUMA_MAP,
			inputs: &semantic_inputs,
			outputs: &semantic_outputs,
			attributes: &attributes,
		},
		frame.timing(),
		frame.color_info(),
	)
}
