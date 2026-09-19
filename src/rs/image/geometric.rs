//! Stateless geometric image transformations.

use crate::{
	DType, Error, Image, ImageLayout, Matrix, OpAttribute, Result,
	runtime::{
		BufferBinding, ComputeDispatch, ImageSemanticDispatch, ImageSemanticInput, KernelId,
		PushConstant,
	},
};

/// Sampling rule used by [`resize`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum InterpolationMode {
	/// Select the source pixel containing the scaled output coordinate.
	Nearest,
	/// Use half-pixel-center bilinear sampling with `align_corners = false`.
	#[default]
	Bilinear,
}

impl InterpolationMode {
	pub(super) const fn token(self) -> &'static str {
		match self {
			Self::Nearest => "nearest",
			Self::Bilinear => "bilinear",
		}
	}

	const fn kernel(self) -> KernelId {
		match self {
			Self::Nearest => KernelId::ImageResizeNearestF32,
			Self::Bilinear => KernelId::ImageResizeBilinearF32,
		}
	}

	pub(super) const fn code(self) -> u32 {
		match self {
			Self::Nearest => 0,
			Self::Bilinear => 1,
		}
	}
}

/// Sampling behavior for coordinates outside an image.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum BorderMode {
	/// Return the caller-provided constant value.
	#[default]
	Constant,
	/// Extend the nearest edge pixel.
	Replicate,
	/// Reflect coordinates while repeating the edge pixel.
	Reflect,
	/// Reflect coordinates without repeating the edge pixel.
	Reflect101,
	/// Wrap coordinates periodically.
	Wrap,
}

impl BorderMode {
	pub(super) const fn token(self) -> &'static str {
		match self {
			Self::Constant => "constant",
			Self::Replicate => "replicate",
			Self::Reflect => "reflect",
			Self::Reflect101 => "reflect_101",
			Self::Wrap => "wrap",
		}
	}

	pub(super) const fn code(self) -> u32 {
		match self {
			Self::Constant => 0,
			Self::Replicate => 1,
			Self::Reflect => 2,
			Self::Reflect101 => 3,
			Self::Wrap => 4,
		}
	}
}

/// Resize an FP32 channel-first image while preserving its layout and format.
///
/// NCHW and CHW storage are supported. Nearest sampling follows the donor OA
/// floor-coordinate rule. Bilinear sampling uses half-pixel centers with
/// `align_corners = false`.
///
/// The call records work asynchronously and returns an Image immediately.
/// Reading its backing matrix is an explicit submission and observation point.
///
/// # Errors
///
/// Returns an error for a zero target extent, an empty source extent, a dtype or
/// layout outside the admitted checkpoint, checked-size overflow, or a runtime
/// recording failure.
pub fn resize(
	input: &Image,
	target_width: u32,
	target_height: u32,
	interpolation: InterpolationMode,
) -> Result<Image> {
	if target_width == 0 || target_height == 0 {
		return Err(Error::invalid_argument(
			"image::resize target width and height must be non-zero",
		));
	}
	if input.dtype() != DType::F32 {
		return Err(Error::invalid_argument(format!(
			"image::resize requires f32 storage; received {}",
			input.dtype().token()
		)));
	}
	if !matches!(input.layout(), ImageLayout::Nchw | ImageLayout::Chw) {
		return Err(Error::invalid_argument(format!(
			"image::resize currently supports Nchw and Chw layouts; received {:?}",
			input.layout()
		)));
	}
	if input.batch_size() == 0 || input.width() == 0 || input.height() == 0 {
		return Err(Error::invalid_argument(
			"image::resize source batch, width, and height must be non-zero",
		));
	}

	let batch = u32_extent(input.batch_size(), "image::resize batch")?;
	let channels = u32_extent(input.channels(), "image::resize channels")?;
	let input_height = u32_extent(input.height(), "image::resize input height")?;
	let input_width = u32_extent(input.width(), "image::resize input width")?;
	u32_extent(
		input.as_matrix().num_elements(),
		"image::resize input element count",
	)?;
	let batch_channels = batch
		.checked_mul(channels)
		.ok_or_else(|| Error::out_of_range("image::resize batch-channel dispatch exceeds u32"))?;
	let target_width_usize = usize::try_from(target_width)
		.map_err(|_| Error::out_of_range("image::resize target width exceeds usize"))?;
	let target_height_usize = usize::try_from(target_height)
		.map_err(|_| Error::out_of_range("image::resize target height exceeds usize"))?;
	let output_shape = match input.layout() {
		ImageLayout::Nchw => vec![
			input.batch_size(),
			input.channels(),
			target_height_usize,
			target_width_usize,
		],
		ImageLayout::Chw => vec![input.channels(), target_height_usize, target_width_usize],
		_ => unreachable!("unsupported layouts returned above"),
	};
	let output_elements = output_shape
		.iter()
		.copied()
		.try_fold(1_usize, |count, extent| {
			count
				.checked_mul(extent)
				.ok_or_else(|| Error::out_of_range("image::resize output element count exceeds usize"))
		})?;
	u32_extent(output_elements, "image::resize output element count")?;
	let output_matrix = Matrix::allocate(
		input.engine_handle(),
		output_shape,
		output_elements,
		DType::F32,
	)?;
	let output = Image::new(output_matrix, input.layout(), input.format())?;
	let kernel = interpolation.kernel();
	let buffers = [
		BufferBinding::read(input.storage()),
		BufferBinding::write(output.storage()),
	];
	let push_constants = [
		PushConstant::U32(batch),
		PushConstant::U32(channels),
		PushConstant::U32(input_height),
		PushConstant::U32(input_width),
		PushConstant::U32(target_height),
		PushConstant::U32(target_width),
	];
	let mut workgroups = kernel.output_workgroups(target_width, target_height);
	workgroups[2] = batch_channels;
	let attributes = [
		OpAttribute::UnsignedInteger {
			name: "target_width".into(),
			value: u64::from(target_width),
		},
		OpAttribute::UnsignedInteger {
			name: "target_height".into(),
			value: u64::from(target_height),
		},
		OpAttribute::Enum {
			name: "interpolation".into(),
			value: interpolation.token().into(),
		},
	];
	input.engine_handle().record_image_semantic(
		ComputeDispatch {
			kernel,
			buffers: &buffers,
			push_constants: &push_constants,
			workgroups,
		},
		ImageSemanticDispatch {
			contract: crate::core::operation::image::RESIZE,
			inputs: &[ImageSemanticInput::Image(input)],
			outputs: &[&output],
			attributes: &attributes,
		},
	)?;
	Ok(output)
}

/// Extract a region, clipping its right and bottom edges to the source extent.
pub fn crop(input: &Image, x: u32, y: u32, width: u32, height: u32) -> Result<Image> {
	let extent = validate_image(input, "image::crop")?;
	if width == 0 || height == 0 || x >= extent.width || y >= extent.height {
		return Err(Error::invalid_argument(
			"image::crop requires a nonzero region whose origin is inside the image",
		));
	}
	let output_width = width.min(extent.width - x);
	let output_height = height.min(extent.height - y);
	let output = allocate_image(input, output_width, output_height, "image::crop")?;
	record_image(
		input,
		&output,
		KernelId::ImageCropF32,
		&[
			PushConstant::U32(extent.batch),
			PushConstant::U32(extent.channels),
			PushConstant::U32(extent.height),
			PushConstant::U32(extent.width),
			PushConstant::U32(x),
			PushConstant::U32(y),
			PushConstant::U32(output_height),
			PushConstant::U32(output_width),
		],
		crate::core::operation::image::CROP,
		&[
			unsigned("x", x),
			unsigned("y", y),
			unsigned("width", width),
			unsigned("height", height),
		],
	)?;
	Ok(output)
}

/// Flip an image horizontally, vertically, or along both axes.
pub fn flip(input: &Image, horizontal: bool, vertical: bool) -> Result<Image> {
	let extent = validate_image(input, "image::flip")?;
	let output = allocate_image(input, extent.width, extent.height, "image::flip")?;
	record_image(
		input,
		&output,
		KernelId::ImageFlipF32,
		&[
			PushConstant::U32(extent.batch),
			PushConstant::U32(extent.channels),
			PushConstant::U32(extent.height),
			PushConstant::U32(extent.width),
			PushConstant::U32(u32::from(horizontal)),
			PushConstant::U32(u32::from(vertical)),
		],
		crate::core::operation::image::FLIP,
		&[
			boolean("horizontal", horizontal),
			boolean("vertical", vertical),
		],
	)?;
	Ok(output)
}

/// Rotate clockwise by a multiple of 90 degrees.
pub fn rotate(input: &Image, degrees: u32) -> Result<Image> {
	let extent = validate_image(input, "image::rotate")?;
	let normalized = degrees % 360;
	if !matches!(normalized, 0 | 90 | 180 | 270) {
		return Err(Error::invalid_argument(
			"image::rotate degrees must be a multiple of 90",
		));
	}
	let (output_width, output_height) = if matches!(normalized, 90 | 270) {
		(extent.height, extent.width)
	} else {
		(extent.width, extent.height)
	};
	let output = allocate_image(input, output_width, output_height, "image::rotate")?;
	record_image(
		input,
		&output,
		KernelId::ImageRotateF32,
		&[
			PushConstant::U32(extent.batch),
			PushConstant::U32(extent.channels),
			PushConstant::U32(extent.height),
			PushConstant::U32(extent.width),
			PushConstant::U32(output_height),
			PushConstant::U32(output_width),
			PushConstant::U32(normalized),
		],
		crate::core::operation::image::ROTATE,
		&[unsigned("degrees", degrees)],
	)?;
	Ok(output)
}

/// Add asymmetric padding with an explicit border policy.
#[allow(clippy::too_many_arguments)]
pub fn pad(
	input: &Image,
	left: u32,
	right: u32,
	top: u32,
	bottom: u32,
	border: BorderMode,
	border_value: f32,
) -> Result<Image> {
	let extent = validate_image(input, "image::pad")?;
	if !border_value.is_finite() {
		return Err(Error::invalid_argument(
			"image::pad border value must be finite",
		));
	}
	let output_width = extent
		.width
		.checked_add(left)
		.and_then(|value| value.checked_add(right))
		.ok_or_else(|| Error::out_of_range("image::pad output width exceeds u32"))?;
	let output_height = extent
		.height
		.checked_add(top)
		.and_then(|value| value.checked_add(bottom))
		.ok_or_else(|| Error::out_of_range("image::pad output height exceeds u32"))?;
	let output = allocate_image(input, output_width, output_height, "image::pad")?;
	record_image(
		input,
		&output,
		KernelId::ImagePadF32,
		&[
			PushConstant::U32(extent.batch),
			PushConstant::U32(extent.channels),
			PushConstant::U32(extent.height),
			PushConstant::U32(extent.width),
			PushConstant::U32(output_height),
			PushConstant::U32(output_width),
			PushConstant::U32(top),
			PushConstant::U32(left),
			PushConstant::U32(border.code()),
			PushConstant::F32(border_value),
		],
		crate::core::operation::image::PAD,
		&[
			unsigned("left", left),
			unsigned("right", right),
			unsigned("top", top),
			unsigned("bottom", bottom),
			enumeration("border", border.token()),
			float("border_value", border_value),
		],
	)?;
	Ok(output)
}

/// Extract a centered region with the requested exact extent.
pub fn center_crop(input: &Image, width: u32, height: u32) -> Result<Image> {
	let extent = validate_image(input, "image::center_crop")?;
	if width == 0 || height == 0 || width > extent.width || height > extent.height {
		return Err(Error::invalid_argument(
			"image::center_crop extent must be nonzero and fit inside the image",
		));
	}
	let x = (extent.width - width) / 2;
	let y = (extent.height - height) / 2;
	let output = allocate_image(input, width, height, "image::center_crop")?;
	record_image(
		input,
		&output,
		KernelId::ImageCenterCropF32,
		&[
			PushConstant::U32(extent.batch),
			PushConstant::U32(extent.channels),
			PushConstant::U32(extent.height),
			PushConstant::U32(extent.width),
			PushConstant::U32(x),
			PushConstant::U32(y),
			PushConstant::U32(height),
			PushConstant::U32(width),
		],
		crate::core::operation::image::CENTER_CROP,
		&[unsigned("width", width), unsigned("height", height)],
	)?;
	Ok(output)
}

/// Resample an image from an FP32 absolute-coordinate map shaped
/// `[1|batch, 2, output_height, output_width]`.
pub fn remap(
	input: &Image,
	map: &Matrix,
	interpolation: InterpolationMode,
	border: BorderMode,
	border_value: f32,
) -> Result<Image> {
	let extent = validate_warp_inputs(input, map, "image::remap", border_value)?;
	if map.shape().len() != 4
		|| map.shape()[1] != 2
		|| map.shape()[2] == 0
		|| map.shape()[3] == 0
		|| !matches!(map.shape()[0], 1) && map.shape()[0] != input.batch_size()
	{
		return Err(Error::invalid_argument(
			"image::remap map must have shape [1|batch, 2, output_height, output_width]",
		));
	}
	let map_batch = u32_extent(map.shape()[0], "image::remap map batch")?;
	let output_height = u32_extent(map.shape()[2], "image::remap output height")?;
	let output_width = u32_extent(map.shape()[3], "image::remap output width")?;
	let output = allocate_image(input, output_width, output_height, "image::remap")?;
	record_warp(
		input,
		map,
		&output,
		KernelId::ImageRemapF32,
		&[
			PushConstant::U32(extent.batch),
			PushConstant::U32(extent.channels),
			PushConstant::U32(extent.height),
			PushConstant::U32(extent.width),
			PushConstant::U32(output_height),
			PushConstant::U32(output_width),
			PushConstant::U32(interpolation.code()),
			PushConstant::U32(border.code()),
			PushConstant::U32(map_batch),
			PushConstant::F32(border_value),
		],
		crate::core::operation::image::REMAP,
		&[
			enumeration("interpolation", interpolation.token()),
			enumeration("border", border.token()),
			float("border_value", border_value),
		],
	)?;
	Ok(output)
}

/// Apply an inverse FP32 2x3 affine transform to an image.
#[allow(clippy::too_many_arguments)]
pub fn warp_affine(
	input: &Image,
	transform: &Matrix,
	width: u32,
	height: u32,
	interpolation: InterpolationMode,
	border: BorderMode,
	border_value: f32,
) -> Result<Image> {
	if transform.shape() != [2, 3] {
		return Err(Error::invalid_argument(
			"image::warp_affine transform must have shape [2, 3]",
		));
	}
	warp_transform(
		input,
		transform,
		width,
		height,
		interpolation,
		border,
		border_value,
		KernelId::ImageWarpAffineF32,
		crate::core::operation::image::WARP_AFFINE,
		"image::warp_affine",
	)
}

/// Apply an inverse FP32 3x3 perspective transform to an image.
#[allow(clippy::too_many_arguments)]
pub fn warp_perspective(
	input: &Image,
	transform: &Matrix,
	width: u32,
	height: u32,
	interpolation: InterpolationMode,
	border: BorderMode,
	border_value: f32,
) -> Result<Image> {
	if transform.shape() != [3, 3] {
		return Err(Error::invalid_argument(
			"image::warp_perspective transform must have shape [3, 3]",
		));
	}
	warp_transform(
		input,
		transform,
		width,
		height,
		interpolation,
		border,
		border_value,
		KernelId::ImageWarpPerspectiveF32,
		crate::core::operation::image::WARP_PERSPECTIVE,
		"image::warp_perspective",
	)
}

#[derive(Clone, Copy)]
struct ImageExtent {
	batch: u32,
	channels: u32,
	height: u32,
	width: u32,
}

fn validate_image(input: &Image, operation: &str) -> Result<ImageExtent> {
	if input.dtype() != DType::F32 {
		return Err(Error::invalid_argument(format!(
			"{operation} requires f32 storage; received {}",
			input.dtype().token()
		)));
	}
	if !matches!(input.layout(), ImageLayout::Nchw | ImageLayout::Chw) {
		return Err(Error::invalid_argument(format!(
			"{operation} supports Nchw and Chw layouts; received {:?}",
			input.layout()
		)));
	}
	if input.batch_size() == 0 || input.channels() == 0 || input.height() == 0 || input.width() == 0 {
		return Err(Error::invalid_argument(format!(
			"{operation} requires a nonempty image"
		)));
	}
	Ok(ImageExtent {
		batch: u32_extent(input.batch_size(), &format!("{operation} batch"))?,
		channels: u32_extent(input.channels(), &format!("{operation} channels"))?,
		height: u32_extent(input.height(), &format!("{operation} height"))?,
		width: u32_extent(input.width(), &format!("{operation} width"))?,
	})
}

fn validate_warp_inputs(
	input: &Image,
	coordinates: &Matrix,
	operation: &str,
	border_value: f32,
) -> Result<ImageExtent> {
	let extent = validate_image(input, operation)?;
	if coordinates.dtype() != DType::F32 {
		return Err(Error::invalid_argument(format!(
			"{operation} coordinates must use f32 storage"
		)));
	}
	if !input.engine_handle().same_as(coordinates.engine_handle()) {
		return Err(Error::invalid_argument(format!(
			"{operation} inputs must belong to the same engine"
		)));
	}
	if !border_value.is_finite() {
		return Err(Error::invalid_argument(format!(
			"{operation} border value must be finite"
		)));
	}
	u32_extent(
		coordinates.num_elements(),
		&format!("{operation} coordinate element count"),
	)?;
	Ok(extent)
}

#[allow(clippy::too_many_arguments)]
fn warp_transform(
	input: &Image,
	transform: &Matrix,
	width: u32,
	height: u32,
	interpolation: InterpolationMode,
	border: BorderMode,
	border_value: f32,
	kernel: KernelId,
	contract: crate::OperationContract,
	operation: &str,
) -> Result<Image> {
	if width == 0 || height == 0 {
		return Err(Error::invalid_argument(format!(
			"{operation} output width and height must be nonzero"
		)));
	}
	let extent = validate_warp_inputs(input, transform, operation, border_value)?;
	let output = allocate_image(input, width, height, operation)?;
	record_warp(
		input,
		transform,
		&output,
		kernel,
		&[
			PushConstant::U32(extent.batch),
			PushConstant::U32(extent.channels),
			PushConstant::U32(extent.height),
			PushConstant::U32(extent.width),
			PushConstant::U32(height),
			PushConstant::U32(width),
			PushConstant::U32(interpolation.code()),
			PushConstant::U32(border.code()),
			PushConstant::F32(border_value),
		],
		contract,
		&[
			unsigned("width", width),
			unsigned("height", height),
			enumeration("interpolation", interpolation.token()),
			enumeration("border", border.token()),
			float("border_value", border_value),
		],
	)?;
	Ok(output)
}

fn allocate_image(input: &Image, width: u32, height: u32, operation: &str) -> Result<Image> {
	let width = usize::try_from(width)
		.map_err(|_| Error::out_of_range(format!("{operation} width exceeds usize")))?;
	let height = usize::try_from(height)
		.map_err(|_| Error::out_of_range(format!("{operation} height exceeds usize")))?;
	let shape = match input.layout() {
		ImageLayout::Nchw => vec![input.batch_size(), input.channels(), height, width],
		ImageLayout::Chw => vec![input.channels(), height, width],
		_ => unreachable!("layout was validated before allocation"),
	};
	let elements = shape.iter().copied().try_fold(1_usize, |count, extent| {
		count
			.checked_mul(extent)
			.ok_or_else(|| Error::out_of_range(format!("{operation} output size exceeds usize")))
	})?;
	u32_extent(elements, &format!("{operation} output element count"))?;
	let matrix = Matrix::allocate(input.engine_handle(), shape, elements, DType::F32)?;
	Image::new(matrix, input.layout(), input.format())
}

fn record_image(
	input: &Image,
	output: &Image,
	kernel: KernelId,
	push_constants: &[PushConstant],
	contract: crate::OperationContract,
	attributes: &[OpAttribute],
) -> Result<()> {
	let width = u32_extent(output.width(), "image operation output width")?;
	let height = u32_extent(output.height(), "image operation output height")?;
	let planes = u32_extent(
		output
			.batch_size()
			.checked_mul(output.channels())
			.ok_or_else(|| Error::out_of_range("image operation batch-channel dispatch exceeds usize"))?,
		"image operation batch-channel dispatch",
	)?;
	let mut workgroups = kernel.output_workgroups(width, height);
	workgroups[2] = planes;
	input.engine_handle().record_image_semantic(
		ComputeDispatch {
			kernel,
			buffers: &[
				BufferBinding::read(input.storage()),
				BufferBinding::write(output.storage()),
			],
			push_constants,
			workgroups,
		},
		ImageSemanticDispatch {
			contract,
			inputs: &[ImageSemanticInput::Image(input)],
			outputs: &[output],
			attributes,
		},
	)
}

fn record_warp(
	input: &Image,
	coordinates: &Matrix,
	output: &Image,
	kernel: KernelId,
	push_constants: &[PushConstant],
	contract: crate::OperationContract,
	attributes: &[OpAttribute],
) -> Result<()> {
	let width = u32_extent(output.width(), "image warp output width")?;
	let height = u32_extent(output.height(), "image warp output height")?;
	let planes = u32_extent(
		output
			.batch_size()
			.checked_mul(output.channels())
			.ok_or_else(|| Error::out_of_range("image warp batch-channel dispatch exceeds usize"))?,
		"image warp batch-channel dispatch",
	)?;
	let mut workgroups = kernel.output_workgroups(width, height);
	workgroups[2] = planes;
	input.engine_handle().record_image_semantic(
		ComputeDispatch {
			kernel,
			buffers: &[
				BufferBinding::read(input.storage()),
				BufferBinding::read(coordinates.storage()),
				BufferBinding::write(output.storage()),
			],
			push_constants,
			workgroups,
		},
		ImageSemanticDispatch {
			contract,
			inputs: &[
				ImageSemanticInput::Image(input),
				ImageSemanticInput::Matrix(coordinates),
			],
			outputs: &[output],
			attributes,
		},
	)
}

fn unsigned(name: &str, value: u32) -> OpAttribute {
	OpAttribute::UnsignedInteger {
		name: name.into(),
		value: u64::from(value),
	}
}

fn boolean(name: &str, value: bool) -> OpAttribute {
	OpAttribute::Boolean {
		name: name.into(),
		value,
	}
}

fn enumeration(name: &str, value: &str) -> OpAttribute {
	OpAttribute::Enum {
		name: name.into(),
		value: value.into(),
	}
}

fn float(name: &str, value: f32) -> OpAttribute {
	OpAttribute::Float {
		name: name.into(),
		value: f64::from(value),
	}
}

fn u32_extent(value: usize, label: &str) -> Result<u32> {
	u32::try_from(value).map_err(|_| Error::out_of_range(format!("{label} exceeds u32")))
}
