//! The single colorimetric YUV-to-Matrix boundary.
//!
//! `oa::video::to_matrix` fuses crop, resize, color conversion,
//! normalization, layout and batch packing into one recorded conversion per
//! frame. No second "convert", "to RGB" or "decode-resize-normalize" entry
//! point exists beside it. The fused kernels must match the reference order
//! — crop, per-plane resize, color conversion, normalization, then layout —
//! within the family tolerance checked against the FP64 host oracle in
//! `test/rs/video/test_convert.rs`.

use crate::{
	DType, Error, ImageLayout, Matrix, OpAttribute, Result,
	runtime::{BufferBinding, ComputeDispatch, KernelId, PushConstant, SemanticDispatch},
};

use super::{VideoColorInfo, VideoColorMatrix, VideoColorRange, VideoFrame, VideoPixelFormat};

/// Output channel selection for [`to_matrix`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MatrixChannels {
	/// Nonlinear R′G′B′ in [0, 1] from the frame's matrix coefficients.
	Rgb,
	/// Channel-swapped R′G′B′.
	Bgr,
	/// Range-scaled luma only; no colorimetric conversion.
	Luma,
	/// Luma plus chroma upsampled to luma resolution, range-scaled.
	Yuv,
}

impl MatrixChannels {
	fn count(self) -> usize {
		match self {
			Self::Luma => 1,
			_ => 3,
		}
	}

	fn mode(self) -> u32 {
		match self {
			Self::Luma => 0,
			Self::Rgb => 1,
			Self::Bgr => 2,
			Self::Yuv => 3,
		}
	}

	fn token(self) -> &'static str {
		match self {
			Self::Rgb => "rgb",
			Self::Bgr => "bgr",
			Self::Luma => "luma",
			Self::Yuv => "yuv",
		}
	}
}

/// Sampling interpolation for resize inside [`to_matrix`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VideoInterpolation {
	/// Half-pixel bilinear sampling with edge clamping.
	Bilinear,
	/// Nearest source pixel center.
	Nearest,
}

impl VideoInterpolation {
	pub(crate) fn mode(self) -> u32 {
		match self {
			Self::Bilinear => 0,
			Self::Nearest => 1,
		}
	}

	pub(crate) fn token(self) -> &'static str {
		match self {
			Self::Bilinear => "bilinear",
			Self::Nearest => "nearest",
		}
	}
}

/// How a cropped source region maps to the requested output extent.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum VideoResize {
	/// Stretch the source region to the output extent.
	Stretch,
	/// Preserve aspect with a centered letterbox; the fill is given in
	/// output-channel units before normalization.
	Letterbox { fill: [f32; 3] },
	/// Preserve aspect and center-crop the overflow.
	CenterCrop,
}

impl VideoResize {
	fn token(self) -> &'static str {
		match self {
			Self::Stretch => "stretch",
			Self::Letterbox { .. } => "letterbox",
			Self::CenterCrop => "center_crop",
		}
	}
}

/// Integer luma-space region of a frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VideoRegion {
	pub x: u32,
	pub y: u32,
	pub width: u32,
	pub height: u32,
}

/// Conversion parameters for [`to_matrix`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VideoMatrixConfig {
	/// Output channels; `Rgb` by default.
	pub channels: MatrixChannels,
	/// Output matrix layout; `Nchw` by default.
	pub layout: ImageLayout,
	/// Source luma-space region applied before resize; `None` uses the whole
	/// visible extent.
	pub crop: Option<VideoRegion>,
	/// Output width and height; `None` keeps the crop extent.
	pub size: Option<(u32, u32)>,
	/// Source-to-output mapping; `Stretch` by default.
	pub resize: VideoResize,
	/// Sampling filter; `Bilinear` by default.
	pub interpolation: VideoInterpolation,
	/// Per-output-channel mean applied after scaling to [0, 1].
	pub mean: [f32; 3],
	/// Per-output-channel standard deviation.
	pub std: [f32; 3],
	/// Color metadata used only when a frame reports `Unspecified`; defaults
	/// to BT.709 limited range.
	pub unspecified_color: VideoColorInfo,
}

impl Default for VideoMatrixConfig {
	fn default() -> Self {
		Self {
			channels: MatrixChannels::Rgb,
			layout: ImageLayout::Nchw,
			crop: None,
			size: None,
			resize: VideoResize::Stretch,
			interpolation: VideoInterpolation::Bilinear,
			mean: [0.0; 3],
			std: [1.0; 3],
			unspecified_color: VideoColorInfo::new(VideoColorMatrix::Bt709, VideoColorRange::Limited),
		}
	}
}

impl VideoMatrixConfig {
	/// ImageNet normalization preset (`mean`/`std` per RGB channel).
	pub fn imagenet() -> Self {
		Self {
			mean: [0.485, 0.456, 0.406],
			std: [0.229, 0.224, 0.225],
			..Self::default()
		}
	}
}

/// Resolved frame color after applying the unspecified-color policy.
#[derive(Clone, Copy, PartialEq, Eq)]
struct ResolvedColor {
	matrix: VideoColorMatrix,
	range: VideoColorRange,
}

fn resolve_color(frame: &VideoFrame, fallback: VideoColorInfo) -> Result<ResolvedColor> {
	let color = frame.color_info();
	let matrix = match color.matrix() {
		VideoColorMatrix::Unspecified => fallback.matrix(),
		explicit => explicit,
	};
	let range = match color.range() {
		VideoColorRange::Unspecified => fallback.range(),
		explicit => explicit,
	};
	let matrix = match matrix {
		VideoColorMatrix::Unspecified => {
			return Err(Error::invalid_argument(
				"to_matrix requires resolved color metadata",
			));
		}
		explicit => explicit,
	};
	let range = match range {
		VideoColorRange::Unspecified => {
			return Err(Error::invalid_argument(
				"to_matrix requires a resolved component range",
			));
		}
		explicit => explicit,
	};
	Ok(ResolvedColor { matrix, range })
}

fn color_space_index(matrix: VideoColorMatrix) -> u32 {
	match matrix {
		VideoColorMatrix::Bt601 => 0,
		VideoColorMatrix::Bt709 => 1,
		VideoColorMatrix::Bt2020 => 2,
		VideoColorMatrix::Unspecified => 1,
	}
}

fn color_token(matrix: VideoColorMatrix) -> &'static str {
	match matrix {
		VideoColorMatrix::Bt601 => "bt601",
		VideoColorMatrix::Bt709 => "bt709",
		VideoColorMatrix::Bt2020 => "bt2020",
		VideoColorMatrix::Unspecified => "unspecified",
	}
}

fn u32_field(value: usize, operation: &str, what: &str) -> Result<u32> {
	u32::try_from(value).map_err(|_| Error::out_of_range(format!("{operation} {what} exceeds u32")))
}

/// Convert one or more video frames into a single batched F32 [`Matrix`].
///
/// This is the only colorimetric YUV→RGB boundary in `oa::video`. Inputs
/// must be device-planar frames with identical format, visible extent and
/// resolved color metadata. I420, NV12, and P010 are admitted. Stage host
/// planes with [`VideoFrame::upload_planes`] or [`VideoFrame::upload`].
/// The call records asynchronously and returns the output immediately.
/// Reading it is an explicit submission and observation point.
///
/// # Errors
///
/// Returns an error for an empty or mixed-geometry batch, an ineligible
/// backing, a crop outside the visible region, zero output extents, an
/// unsupported layout or bit depth, or a runtime recording failure.
pub fn to_matrix(frames: &[VideoFrame], config: &VideoMatrixConfig) -> Result<Matrix> {
	const OPERATION: &str = "video::to_matrix";
	if frames.is_empty() {
		return Err(Error::invalid_argument(format!(
			"{OPERATION} requires at least one frame"
		)));
	}
	let first = frames[0].device_planar().ok_or_else(|| {
		Error::invalid_argument(format!(
			"{OPERATION} requires device-planar frames; stage host frames with VideoFrame::upload"
		))
	})?;
	let engine = first.planes[0].engine_handle();
	let format = first.format;
	let frame_width = first.width;
	let frame_height = first.height;
	let color = resolve_color(&frames[0], config.unspecified_color)?;
	let mut plane_inputs: Vec<&Matrix> = Vec::new();
	let mut planars = Vec::with_capacity(frames.len());
	for frame in frames {
		let planar = frame.device_planar().ok_or_else(|| {
			Error::invalid_argument(format!(
				"{OPERATION} requires every frame to be device-planar"
			))
		})?;
		if planar.format != format || planar.width != frame_width || planar.height != frame_height {
			return Err(Error::invalid_argument(format!(
				"{OPERATION} requires identical format and extent in one batch"
			)));
		}
		if !planar.planes[0].engine_handle().same_as(engine) {
			return Err(Error::invalid_argument(format!(
				"{OPERATION} requires frames from one engine"
			)));
		}
		if resolve_color(frame, config.unspecified_color)? != color {
			return Err(Error::invalid_argument(format!(
				"{OPERATION} requires identical resolved color metadata in one batch"
			)));
		}
		for plane in planar.planes.iter() {
			u32_field(plane.num_elements(), OPERATION, "plane byte length")?;
		}
		plane_inputs.extend(planar.planes.iter());
		planars.push(planar);
	}

	let crop = config.crop.unwrap_or(VideoRegion {
		x: 0,
		y: 0,
		width: u32_field(frame_width, OPERATION, "width")?,
		height: u32_field(frame_height, OPERATION, "height")?,
	});
	{
		let right = u64::from(crop.x) + u64::from(crop.width);
		let bottom = u64::from(crop.y) + u64::from(crop.height);
		if crop.width == 0
			|| crop.height == 0
			|| right > frame_width as u64
			|| bottom > frame_height as u64
		{
			return Err(Error::invalid_argument(format!(
				"{OPERATION} crop must be a non-empty region inside the visible extent"
			)));
		}
	}
	let (out_width, out_height) = config.size.unwrap_or((crop.width, crop.height));
	if out_width == 0 || out_height == 0 {
		return Err(Error::invalid_argument(format!(
			"{OPERATION} requires a non-zero output extent"
		)));
	}
	let batch = frames.len();
	if matches!(config.layout, ImageLayout::Chw | ImageLayout::Hwc) && batch != 1 {
		return Err(Error::invalid_argument(format!(
			"{OPERATION} unbatched layouts require exactly one frame"
		)));
	}
	if !matches!(
		config.layout,
		ImageLayout::Nchw | ImageLayout::Nhwc | ImageLayout::Chw | ImageLayout::Hwc
	) {
		return Err(Error::invalid_argument(format!(
			"{OPERATION} requires an NCHW, NHWC, CHW or HWC layout"
		)));
	}
	let channel_count = config.channels.count();
	let layout_packed = matches!(config.layout, ImageLayout::Nhwc | ImageLayout::Hwc);
	let shape = match config.layout {
		ImageLayout::Nchw => vec![
			batch,
			channel_count,
			out_height as usize,
			out_width as usize,
		],
		ImageLayout::Nhwc => vec![
			batch,
			out_height as usize,
			out_width as usize,
			channel_count,
		],
		ImageLayout::Chw => vec![channel_count, out_height as usize, out_width as usize],
		ImageLayout::Hwc => vec![out_height as usize, out_width as usize, channel_count],
		ImageLayout::Hw => unreachable!("layout checked above"),
	};
	let elements = shape.iter().try_fold(1usize, |total, axis| {
		total
			.checked_mul(*axis)
			.ok_or_else(|| Error::out_of_range(format!("{OPERATION} output size overflows usize")))
	})?;
	let output = Matrix::allocate(engine, shape, elements, DType::F32)?;

	// Host-side mapping: continuous luma coordinates
	//   sx = (out_x + 0.5) * scale_x + off_x
	// matching the kernel and the FP64 test oracle exactly.
	let crop_w = f64::from(crop.width);
	let crop_h = f64::from(crop.height);
	let out_w = f64::from(out_width);
	let out_h = f64::from(out_height);
	let (scale_x, scale_y, off_x, off_y, valid, fill) = match config.resize {
		VideoResize::Stretch => (
			crop_w / out_w,
			crop_h / out_h,
			f64::from(crop.x) - 0.5,
			f64::from(crop.y) - 0.5,
			(0.0f64, 0.0f64, out_w, out_h),
			[0.0f32; 3],
		),
		VideoResize::CenterCrop => {
			let scale = (out_w / crop_w).max(out_h / crop_h);
			let src_w = out_w / scale;
			let src_h = out_h / scale;
			let src_x = f64::from(crop.x) + (crop_w - src_w) * 0.5;
			let src_y = f64::from(crop.y) + (crop_h - src_h) * 0.5;
			(
				src_w / out_w,
				src_h / out_h,
				src_x - 0.5,
				src_y - 0.5,
				(0.0f64, 0.0f64, out_w, out_h),
				[0.0f32; 3],
			)
		}
		VideoResize::Letterbox { fill } => {
			let scale = (out_w / crop_w).min(out_h / crop_h);
			let map_w = crop_w * scale;
			let map_h = crop_h * scale;
			let ix0 = (out_w - map_w) * 0.5;
			let iy0 = (out_h - map_h) * 0.5;
			(
				crop_w / map_w,
				crop_h / map_h,
				f64::from(crop.x) - ix0 * (crop_w / map_w) - 0.5,
				f64::from(crop.y) - iy0 * (crop_h / map_h) - 0.5,
				(ix0, iy0, ix0 + map_w, iy0 + map_h),
				fill,
			)
		}
	};

	let kernel = match format {
		VideoPixelFormat::Yuv420Planar8 => KernelId::VideoToMatrixI420,
		VideoPixelFormat::Nv12 => KernelId::VideoToMatrixNv12,
		VideoPixelFormat::P010 => KernelId::VideoToMatrixP010,
		VideoPixelFormat::Yuv420Planar10 | VideoPixelFormat::P012 => {
			return Err(Error::missing_capability(format!(
				"{OPERATION} supports I420, NV12, and P010"
			)));
		}
	};
	let color_mode =
		color_space_index(color.matrix) * 2 + u32::from(matches!(color.range, VideoColorRange::Full));
	let luma_width = u32_field(frame_width, OPERATION, "luma width")?;
	let luma_height = u32_field(frame_height, OPERATION, "luma height")?;
	let chroma_width = u32_field(frame_width.div_ceil(2), OPERATION, "chroma width")?;
	let chroma_height = u32_field(frame_height.div_ceil(2), OPERATION, "chroma height")?;
	let workgroups = [
		out_width.div_ceil(16),
		out_height.div_ceil(16),
		channel_count as u32,
	];
	let frame_span = channel_count
		.checked_mul(out_height as usize)
		.and_then(|span| span.checked_mul(out_width as usize))
		.ok_or_else(|| Error::out_of_range(format!("{OPERATION} frame span overflows usize")))?;

	let mut dispatches = Vec::with_capacity(batch);
	let mut binding_store: Vec<Vec<BufferBinding>> = Vec::with_capacity(batch);
	for (index, planar) in planars.iter().enumerate() {
		let mut bindings = Vec::with_capacity(planar.planes.len() + 1);
		for plane in planar.planes.iter() {
			bindings.push(BufferBinding::read(plane.storage()));
		}
		bindings.push(BufferBinding::write(output.storage()));
		binding_store.push(bindings);
		let push = [
			PushConstant::U32(out_width),
			PushConstant::U32(out_height),
			PushConstant::U32(u32_field(index * frame_span, OPERATION, "batch offset")?),
			PushConstant::U32(config.channels.mode()),
			PushConstant::U32(u32::from(layout_packed)),
			PushConstant::U32(config.interpolation.mode()),
			PushConstant::U32(luma_width),
			PushConstant::U32(luma_height),
			PushConstant::U32(chroma_width),
			PushConstant::U32(chroma_height),
			PushConstant::U32(color_mode),
			PushConstant::F32(off_x as f32),
			PushConstant::F32(off_y as f32),
			PushConstant::F32(scale_x as f32),
			PushConstant::F32(scale_y as f32),
			PushConstant::F32(valid.0 as f32),
			PushConstant::F32(valid.1 as f32),
			PushConstant::F32(valid.2 as f32),
			PushConstant::F32(valid.3 as f32),
			PushConstant::F32(fill[0]),
			PushConstant::F32(fill[1]),
			PushConstant::F32(fill[2]),
			PushConstant::F32(config.mean[0]),
			PushConstant::F32(config.mean[1]),
			PushConstant::F32(config.mean[2]),
			PushConstant::F32(config.std[0]),
			PushConstant::F32(config.std[1]),
			PushConstant::F32(config.std[2]),
		];
		dispatches.push((binding_store.len() - 1, push));
	}
	let dispatch_list: Vec<ComputeDispatch> = dispatches
		.iter()
		.map(|(binding_index, push)| ComputeDispatch {
			kernel,
			buffers: &binding_store[*binding_index],
			push_constants: push,
			workgroups,
			indirect: None,
		})
		.collect();
	let outputs = [&output];
	let attributes = [
		OpAttribute::Enum {
			name: "channels".into(),
			value: config.channels.token().into(),
		},
		OpAttribute::Enum {
			name: "layout".into(),
			value: if layout_packed { "nhwc" } else { "nchw" }.into(),
		},
		OpAttribute::Enum {
			name: "interpolation".into(),
			value: config.interpolation.token().into(),
		},
		OpAttribute::Enum {
			name: "resize".into(),
			value: config.resize.token().into(),
		},
		OpAttribute::Enum {
			name: "color_space".into(),
			value: color_token(color.matrix).into(),
		},
		OpAttribute::Boolean {
			name: "full_range".into(),
			value: matches!(color.range, VideoColorRange::Full),
		},
		OpAttribute::String {
			name: "normalization".into(),
			value: format!(
				"{},{},{};{},{},{}",
				config.mean[0], config.mean[1], config.mean[2], config.std[0], config.std[1], config.std[2]
			),
		},
		OpAttribute::String {
			name: "crop".into(),
			value: format!("{},{},{},{}", crop.x, crop.y, crop.width, crop.height),
		},
	];
	let semantic = SemanticDispatch {
		contract: crate::core::operation::video::TO_MATRIX,
		inputs: &plane_inputs,
		outputs: &outputs,
		attributes: &attributes,
	};
	if dispatches.len() == 1 {
		let (binding_index, push) = &dispatches[0];
		engine.record_semantic(
			ComputeDispatch {
				kernel,
				buffers: &binding_store[*binding_index],
				push_constants: push,
				workgroups,
				indirect: None,
			},
			semantic,
		)?;
	} else {
		engine.record_split_semantic(&dispatch_list, semantic)?;
	}
	Ok(output)
}

impl VideoFrame {
	/// Convenience form of [`to_matrix`] for one frame.
	///
	/// # Errors
	///
	/// Returns the same errors as [`to_matrix`].
	pub fn to_matrix(&self, config: &VideoMatrixConfig) -> Result<Matrix> {
		to_matrix(std::slice::from_ref(self), config)
	}
}
