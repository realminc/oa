//! Same-shape image filters and morphology.

use crate::{
	DType, Error, Image, ImageLayout, Matrix, OpAttribute, OperationContract, Result,
	runtime::{
		BufferBinding, ComputeDispatch, ImageSemanticDispatch, ImageSemanticInput, KernelId,
		PushConstant,
	},
};

use super::BorderMode;

#[derive(Clone, Copy)]
struct FilterSpec {
	kernel: KernelId,
	contract: OperationContract,
	operation: u32,
	border: BorderMode,
	kernel_width: u32,
	kernel_height: u32,
	kernel_size: u32,
	dx: u32,
	dy: u32,
	parameters: [f32; 4],
	border_value: f32,
}

/// Apply same-shape 2D cross-correlation with an odd FP32 kernel up to 31×31.
pub fn convolve_2d(
	input: &Image,
	kernel: &Matrix,
	border: BorderMode,
	border_value: f32,
) -> Result<Image> {
	validate_kernel(input, kernel, false, "image::convolve_2d")?;
	finite(border_value, "image::convolve_2d border value")?;
	dispatch(
		input,
		&[kernel],
		spec(
			KernelId::ImageConvolve2dF32,
			crate::core::operation::image::CONVOLVE_2D,
			0,
			border,
		)
		.with_kernel(kernel.shape()[1] as u32, kernel.shape()[0] as u32),
		&[
			enumeration("border", border.token()),
			float("border_value", border_value),
		],
		border_value,
	)
}

/// Apply horizontal and vertical odd FP32 vector kernels up to length 31.
pub fn separable_convolve_2d(
	input: &Image,
	kernel_x: &Matrix,
	kernel_y: &Matrix,
	border: BorderMode,
	border_value: f32,
) -> Result<Image> {
	validate_kernel(input, kernel_x, true, "image::separable_convolve_2d")?;
	validate_kernel(input, kernel_y, true, "image::separable_convolve_2d")?;
	finite(border_value, "image::separable_convolve_2d border value")?;
	dispatch(
		input,
		&[kernel_x, kernel_y],
		spec(
			KernelId::ImageSeparableConvolve2dF32,
			crate::core::operation::image::SEPARABLE_CONVOLVE_2D,
			1,
			border,
		)
		.with_kernel(
			kernel_x.num_elements() as u32,
			kernel_y.num_elements() as u32,
		),
		&[
			enumeration("border", border.token()),
			float("border_value", border_value),
		],
		border_value,
	)
}

/// Average an odd rectangular neighborhood up to 31×31.
pub fn average_blur(input: &Image, width: u32, height: u32, border: BorderMode) -> Result<Image> {
	validate_morphology(width, height, 31, "image::average_blur")?;
	dispatch(
		input,
		&[],
		spec(
			KernelId::ImageAverageBlurF32,
			crate::core::operation::image::AVERAGE_BLUR,
			2,
			border,
		)
		.with_kernel(width, height),
		&[
			unsigned("kernel_width", width),
			unsigned("kernel_height", height),
			enumeration("border", border.token()),
		],
		0.0,
	)
}

/// Compute one first-order Sobel derivative.
pub fn sobel(input: &Image, dx: u32, dy: u32, border: BorderMode) -> Result<Image> {
	validate_derivative(dx, dy, "image::sobel")?;
	dispatch(
		input,
		&[],
		spec(
			KernelId::ImageSobelF32,
			crate::core::operation::image::SOBEL,
			3,
			border,
		)
		.with_derivative(dx, dy),
		&[
			unsigned("dx", dx),
			unsigned("dy", dy),
			enumeration("border", border.token()),
		],
		0.0,
	)
}

/// Compute one first-order Scharr derivative.
pub fn scharr(input: &Image, dx: u32, dy: u32, border: BorderMode) -> Result<Image> {
	validate_derivative(dx, dy, "image::scharr")?;
	dispatch(
		input,
		&[],
		spec(
			KernelId::ImageScharrF32,
			crate::core::operation::image::SCHARR,
			4,
			border,
		)
		.with_derivative(dx, dy),
		&[
			unsigned("dx", dx),
			unsigned("dy", dy),
			enumeration("border", border.token()),
		],
		0.0,
	)
}

/// Apply the four-neighbor discrete Laplacian.
pub fn laplacian(input: &Image, border: BorderMode) -> Result<Image> {
	dispatch(
		input,
		&[],
		spec(
			KernelId::ImageLaplacianF32,
			crate::core::operation::image::LAPLACIAN,
			5,
			border,
		),
		&[enumeration("border", border.token())],
		0.0,
	)
}

/// Apply a nonnegative four-neighbor sharpening amount.
pub fn sharpen(input: &Image, amount: f32, border: BorderMode) -> Result<Image> {
	finite(amount, "image::sharpen amount")?;
	if amount < 0.0 {
		return Err(Error::invalid_argument(
			"image::sharpen amount must be nonnegative",
		));
	}
	dispatch(
		input,
		&[],
		spec(
			KernelId::ImageSharpenF32,
			crate::core::operation::image::SHARPEN,
			12,
			border,
		)
		.with_parameters([amount, 0.0, 0.0, 0.0]),
		&[
			float("amount", amount),
			enumeration("border", border.token()),
		],
		0.0,
	)
}

/// Erode with an odd rectangular structuring element up to 31×31.
pub fn erode(
	input: &Image,
	width: u32,
	height: u32,
	border: BorderMode,
	border_value: f32,
) -> Result<Image> {
	morphology(
		input,
		width,
		height,
		border,
		border_value,
		MorphologySpec::new(
			KernelId::ImageErodeF32,
			crate::core::operation::image::ERODE,
			6,
		),
	)
}

/// Dilate with an odd rectangular structuring element up to 31×31.
pub fn dilate(
	input: &Image,
	width: u32,
	height: u32,
	border: BorderMode,
	border_value: f32,
) -> Result<Image> {
	morphology(
		input,
		width,
		height,
		border,
		border_value,
		MorphologySpec::new(
			KernelId::ImageDilateF32,
			crate::core::operation::image::DILATE,
			7,
		),
	)
}

/// Erode and then dilate with the same rectangular element.
pub fn morphology_open(
	input: &Image,
	width: u32,
	height: u32,
	border: BorderMode,
	border_value: f32,
) -> Result<Image> {
	morphology(
		input,
		width,
		height,
		border,
		border_value,
		MorphologySpec::new(
			KernelId::ImageMorphologyOpenF32,
			crate::core::operation::image::MORPHOLOGY_OPEN,
			8,
		),
	)
}

/// Dilate and then erode with the same rectangular element.
pub fn morphology_close(
	input: &Image,
	width: u32,
	height: u32,
	border: BorderMode,
	border_value: f32,
) -> Result<Image> {
	morphology(
		input,
		width,
		height,
		border,
		border_value,
		MorphologySpec::new(
			KernelId::ImageMorphologyCloseF32,
			crate::core::operation::image::MORPHOLOGY_CLOSE,
			9,
		),
	)
}

/// Return dilation minus erosion.
pub fn morphology_gradient(
	input: &Image,
	width: u32,
	height: u32,
	border: BorderMode,
	border_value: f32,
) -> Result<Image> {
	morphology(
		input,
		width,
		height,
		border,
		border_value,
		MorphologySpec::new(
			KernelId::ImageMorphologyGradientF32,
			crate::core::operation::image::MORPHOLOGY_GRADIENT,
			10,
		),
	)
}

/// Return the source minus its morphological opening.
pub fn morphology_top_hat(
	input: &Image,
	width: u32,
	height: u32,
	border: BorderMode,
	border_value: f32,
) -> Result<Image> {
	morphology(
		input,
		width,
		height,
		border,
		border_value,
		MorphologySpec::new(
			KernelId::ImageMorphologyTopHatF32,
			crate::core::operation::image::MORPHOLOGY_TOP_HAT,
			16,
		),
	)
}

/// Return the morphological closing minus the source.
pub fn morphology_black_hat(
	input: &Image,
	width: u32,
	height: u32,
	border: BorderMode,
	border_value: f32,
) -> Result<Image> {
	morphology(
		input,
		width,
		height,
		border,
		border_value,
		MorphologySpec::new(
			KernelId::ImageMorphologyBlackHatF32,
			crate::core::operation::image::MORPHOLOGY_BLACK_HAT,
			17,
		),
	)
}

/// Apply a normalized Gaussian blur with replicate borders.
pub fn gaussian_blur(input: &Image, sigma: f32, kernel_size: u32) -> Result<Image> {
	finite(sigma, "image::gaussian_blur sigma")?;
	if sigma <= 0.0 || kernel_size != 0 && (kernel_size > 31 || kernel_size.is_multiple_of(2)) {
		return Err(Error::invalid_argument(
			"image::gaussian_blur requires positive sigma and odd kernel <=31 or zero",
		));
	}
	let size = if kernel_size == 0 {
		((3.0 * sigma).ceil() as u32).min(15) * 2 + 1
	} else {
		kernel_size
	};
	dispatch(
		input,
		&[],
		spec(
			KernelId::ImageGaussianBlurF32,
			crate::core::operation::image::GAUSSIAN_BLUR,
			11,
			BorderMode::Replicate,
		)
		.with_size(size)
		.with_parameters([sigma, 0.0, 0.0, 0.0]),
		&[float("sigma", sigma), unsigned("kernel_size", kernel_size)],
		0.0,
	)
}

/// Apply an unsharp mask using a Gaussian detail estimate.
pub fn unsharp_mask(input: &Image, sigma: f32, amount: f32, kernel_size: u32) -> Result<Image> {
	finite(sigma, "image::unsharp_mask sigma")?;
	finite(amount, "image::unsharp_mask amount")?;
	if sigma <= 0.0 || kernel_size != 0 && (kernel_size > 31 || kernel_size.is_multiple_of(2)) {
		return Err(Error::invalid_argument(
			"image::unsharp_mask requires positive sigma and odd kernel <=31 or zero",
		));
	}
	let size = if kernel_size == 0 {
		((3.0 * sigma).ceil() as u32).min(15) * 2 + 1
	} else {
		kernel_size
	};
	dispatch(
		input,
		&[],
		spec(
			KernelId::ImageUnsharpMaskF32,
			crate::core::operation::image::UNSHARP_MASK,
			15,
			BorderMode::Replicate,
		)
		.with_size(size)
		.with_parameters([sigma, amount, 0.0, 0.0]),
		&[
			float("sigma", sigma),
			float("amount", amount),
			unsigned("kernel_size", kernel_size),
		],
		0.0,
	)
}

/// Select the median of an odd square neighborhood up to 15×15.
pub fn median_blur(input: &Image, kernel_size: u32, border: BorderMode) -> Result<Image> {
	validate_neighborhood(kernel_size, "image::median_blur")?;
	dispatch(
		input,
		&[],
		spec(
			KernelId::ImageMedianBlurF32,
			crate::core::operation::image::MEDIAN_BLUR,
			13,
			border,
		)
		.with_size(kernel_size),
		&[
			unsigned("kernel_size", kernel_size),
			enumeration("border", border.token()),
		],
		0.0,
	)
}

/// Apply a bilateral neighborhood filter.
pub fn bilateral_filter(
	input: &Image,
	kernel_size: u32,
	sigma_color: f32,
	sigma_space: f32,
	border: BorderMode,
) -> Result<Image> {
	validate_neighborhood(kernel_size, "image::bilateral_filter")?;
	finite(sigma_color, "image::bilateral_filter sigma_color")?;
	finite(sigma_space, "image::bilateral_filter sigma_space")?;
	if sigma_color <= 0.0 || sigma_space <= 0.0 {
		return Err(Error::invalid_argument(
			"image::bilateral_filter sigmas must be positive",
		));
	}
	dispatch(
		input,
		&[],
		spec(
			KernelId::ImageBilateralFilterF32,
			crate::core::operation::image::BILATERAL_FILTER,
			14,
			border,
		)
		.with_size(kernel_size)
		.with_parameters([sigma_color, sigma_space, 0.0, 0.0]),
		&[
			unsigned("kernel_size", kernel_size),
			float("sigma_color", sigma_color),
			float("sigma_space", sigma_space),
			enumeration("border", border.token()),
		],
		0.0,
	)
}

/// Threshold against the local arithmetic mean minus `c`.
pub fn adaptive_threshold_mean(
	input: &Image,
	kernel_size: u32,
	c: f32,
	max_value: f32,
	border: BorderMode,
) -> Result<Image> {
	validate_neighborhood(kernel_size, "image::adaptive_threshold_mean")?;
	finite(c, "image::adaptive_threshold_mean c")?;
	finite(max_value, "image::adaptive_threshold_mean max_value")?;
	dispatch(
		input,
		&[],
		spec(
			KernelId::ImageAdaptiveThresholdMeanF32,
			crate::core::operation::image::ADAPTIVE_THRESHOLD_MEAN,
			18,
			border,
		)
		.with_size(kernel_size)
		.with_parameters([c, max_value, 0.0, 0.0]),
		&[
			unsigned("kernel_size", kernel_size),
			float("c", c),
			float("max_value", max_value),
			enumeration("border", border.token()),
		],
		0.0,
	)
}

/// Threshold against a local Gaussian-weighted mean minus `c`.
pub fn adaptive_threshold_gaussian(
	input: &Image,
	kernel_size: u32,
	c: f32,
	max_value: f32,
	sigma: f32,
	border: BorderMode,
) -> Result<Image> {
	validate_neighborhood(kernel_size, "image::adaptive_threshold_gaussian")?;
	for (value, name) in [(c, "c"), (max_value, "max_value"), (sigma, "sigma")] {
		finite(value, name)?;
	}
	if sigma < 0.0 {
		return Err(Error::invalid_argument(
			"image::adaptive_threshold_gaussian sigma must be nonnegative",
		));
	}
	let resolved_sigma = if sigma > 0.0 {
		sigma
	} else {
		0.3 * ((kernel_size as f32 - 1.0) * 0.5 - 1.0) + 0.8
	};
	dispatch(
		input,
		&[],
		spec(
			KernelId::ImageAdaptiveThresholdGaussianF32,
			crate::core::operation::image::ADAPTIVE_THRESHOLD_GAUSSIAN,
			19,
			border,
		)
		.with_size(kernel_size)
		.with_parameters([c, max_value, resolved_sigma, 0.0]),
		&[
			unsigned("kernel_size", kernel_size),
			float("c", c),
			float("max_value", max_value),
			float("sigma", sigma),
			enumeration("border", border.token()),
		],
		0.0,
	)
}

fn morphology(
	input: &Image,
	width: u32,
	height: u32,
	border: BorderMode,
	border_value: f32,
	specification: MorphologySpec,
) -> Result<Image> {
	validate_morphology(width, height, 31, specification.contract.name())?;
	finite(border_value, "morphology border value")?;
	dispatch(
		input,
		&[],
		spec(
			specification.kernel,
			specification.contract,
			specification.operation,
			border,
		)
		.with_kernel(width, height),
		&[
			unsigned("kernel_width", width),
			unsigned("kernel_height", height),
			enumeration("border", border.token()),
			float("border_value", border_value),
		],
		border_value,
	)
}

struct MorphologySpec {
	kernel: KernelId,
	contract: OperationContract,
	operation: u32,
}

impl MorphologySpec {
	const fn new(kernel: KernelId, contract: OperationContract, operation: u32) -> Self {
		Self {
			kernel,
			contract,
			operation,
		}
	}
}

fn spec(
	kernel: KernelId,
	contract: OperationContract,
	operation: u32,
	border: BorderMode,
) -> FilterSpec {
	FilterSpec {
		kernel,
		contract,
		operation,
		border,
		kernel_width: 1,
		kernel_height: 1,
		kernel_size: 1,
		dx: 0,
		dy: 0,
		parameters: [0.0; 4],
		border_value: 0.0,
	}
}

impl FilterSpec {
	fn with_kernel(mut self, width: u32, height: u32) -> Self {
		self.kernel_width = width;
		self.kernel_height = height;
		self
	}
	fn with_size(mut self, size: u32) -> Self {
		self.kernel_size = size;
		self
	}
	fn with_derivative(mut self, dx: u32, dy: u32) -> Self {
		self.dx = dx;
		self.dy = dy;
		self
	}
	fn with_parameters(mut self, parameters: [f32; 4]) -> Self {
		self.parameters = parameters;
		self
	}
}

fn dispatch(
	input: &Image,
	matrices: &[&Matrix],
	mut spec: FilterSpec,
	attributes: &[OpAttribute],
	border_value: f32,
) -> Result<Image> {
	let (batch, channels, height, width) = image_extent(input, spec.contract.name())?;
	spec.border_value = border_value;
	let output = allocate_like(input, spec.contract.name())?;
	let kernel_x = matrices
		.first()
		.copied()
		.map_or(input.storage(), Matrix::storage);
	let kernel_y = matrices.get(1).copied().map_or(kernel_x, Matrix::storage);
	let buffers = [
		BufferBinding::read(input.storage()),
		BufferBinding::read(kernel_x),
		BufferBinding::read(kernel_y),
		BufferBinding::write(output.storage()),
	];
	let push_constants = [
		PushConstant::U32(batch),
		PushConstant::U32(channels),
		PushConstant::U32(height),
		PushConstant::U32(width),
		PushConstant::U32(spec.operation),
		PushConstant::U32(spec.border.code()),
		PushConstant::U32(spec.kernel_width),
		PushConstant::U32(spec.kernel_height),
		PushConstant::U32(spec.kernel_size),
		PushConstant::U32(spec.dx),
		PushConstant::U32(spec.dy),
		PushConstant::F32(spec.parameters[0]),
		PushConstant::F32(spec.parameters[1]),
		PushConstant::F32(spec.parameters[2]),
		PushConstant::F32(spec.parameters[3]),
		PushConstant::F32(spec.border_value),
	];
	let mut semantic_inputs = Vec::with_capacity(1 + matrices.len());
	semantic_inputs.push(ImageSemanticInput::Image(input));
	semantic_inputs.extend(
		matrices
			.iter()
			.map(|matrix| ImageSemanticInput::Matrix(matrix)),
	);
	input.engine_handle().record_image_semantic(
		ComputeDispatch {
			kernel: spec.kernel,
			buffers: &buffers,
			push_constants: &push_constants,
			workgroups: [width.div_ceil(16), height.div_ceil(16), batch * channels],
		},
		ImageSemanticDispatch {
			contract: spec.contract,
			inputs: &semantic_inputs,
			outputs: &[&output],
			attributes,
		},
	)?;
	Ok(output)
}

fn validate_kernel(input: &Image, kernel: &Matrix, vector: bool, operation: &str) -> Result<()> {
	image_extent(input, operation)?;
	let valid_shape = if vector {
		kernel.shape().len() == 1
			|| kernel.shape().len() == 2 && (kernel.shape()[0] == 1 || kernel.shape()[1] == 1)
	} else {
		kernel.shape().len() == 2
	};
	let elements = kernel.num_elements();
	let valid_extent = if vector {
		elements > 0 && elements <= 31 && elements % 2 == 1
	} else {
		kernel.shape()[0] > 0
			&& kernel.shape()[1] > 0
			&& kernel.shape()[0] <= 31
			&& kernel.shape()[1] <= 31
			&& kernel.shape()[0] % 2 == 1
			&& kernel.shape()[1] % 2 == 1
	};
	if !valid_shape
		|| !valid_extent
		|| kernel.dtype() != DType::F32
		|| !input.engine_handle().same_as(kernel.engine_handle())
	{
		return Err(Error::invalid_argument(format!(
			"{operation} requires a same-engine odd FP32 kernel with each extent <=31"
		)));
	}
	Ok(())
}

fn validate_derivative(dx: u32, dy: u32, operation: &str) -> Result<()> {
	if matches!((dx, dy), (1, 0) | (0, 1)) {
		Ok(())
	} else {
		Err(Error::invalid_argument(format!(
			"{operation} derivative must be (1,0) or (0,1)"
		)))
	}
}

fn validate_morphology(width: u32, height: u32, maximum: u32, operation: &str) -> Result<()> {
	if width > 0
		&& height > 0
		&& width <= maximum
		&& height <= maximum
		&& width % 2 == 1
		&& height % 2 == 1
	{
		Ok(())
	} else {
		Err(Error::invalid_argument(format!(
			"{operation} kernel dimensions must be odd and in 1..={maximum}"
		)))
	}
}

fn validate_neighborhood(size: u32, operation: &str) -> Result<()> {
	validate_morphology(size, size, 15, operation)
}

fn image_extent(input: &Image, operation: &str) -> Result<(u32, u32, u32, u32)> {
	if input.dtype() != DType::F32 || !matches!(input.layout(), ImageLayout::Nchw | ImageLayout::Chw)
	{
		return Err(Error::invalid_argument(format!(
			"{operation} requires an FP32 NCHW or CHW image"
		)));
	}
	let values = [
		input.batch_size(),
		input.channels(),
		input.height(),
		input.width(),
	];
	if values.contains(&0) {
		return Err(Error::invalid_argument(format!(
			"{operation} requires a nonempty image"
		)));
	}
	Ok((
		u32::try_from(values[0]).map_err(|_| Error::out_of_range("image batch exceeds u32"))?,
		u32::try_from(values[1]).map_err(|_| Error::out_of_range("image channels exceed u32"))?,
		u32::try_from(values[2]).map_err(|_| Error::out_of_range("image height exceeds u32"))?,
		u32::try_from(values[3]).map_err(|_| Error::out_of_range("image width exceeds u32"))?,
	))
}

fn allocate_like(input: &Image, operation: &str) -> Result<Image> {
	let elements = input.as_matrix().num_elements();
	u32::try_from(elements)
		.map_err(|_| Error::out_of_range(format!("{operation} output size exceeds u32")))?;
	let matrix = Matrix::allocate(
		input.engine_handle(),
		input.as_matrix().shape().to_vec(),
		elements,
		DType::F32,
	)?;
	Image::new(matrix, input.layout(), input.format())
}

fn finite(value: f32, label: &str) -> Result<()> {
	if value.is_finite() {
		Ok(())
	} else {
		Err(Error::invalid_argument(format!("{label} must be finite")))
	}
}
fn float(name: &str, value: f32) -> OpAttribute {
	OpAttribute::Float {
		name: name.into(),
		value: f64::from(value),
	}
}
fn unsigned(name: &str, value: u32) -> OpAttribute {
	OpAttribute::UnsignedInteger {
		name: name.into(),
		value: u64::from(value),
	}
}
fn enumeration(name: &str, value: &str) -> OpAttribute {
	OpAttribute::Enum {
		name: name.into(),
		value: value.into(),
	}
}
