//! Stateless per-element image transformations.

use crate::{
	DType, Error, Image, ImageFormat, ImageLayout, Matrix, OpAttribute, OperationContract, Result,
	runtime::{
		BufferBinding, ComputeDispatch, ImageSemanticDispatch, ImageSemanticInput, KernelId,
		PushConstant,
	},
};

#[derive(Clone, Copy)]
struct Pointwise {
	kernel: KernelId,
	contract: OperationContract,
	code: u32,
}

/// Replace values above `threshold` with `max_value` and all others with zero.
pub fn threshold_binary(input: &Image, threshold: f32, max_value: f32) -> Result<Image> {
	finite(&[threshold, max_value], "image::threshold_binary")?;
	pointwise(
		input,
		Pointwise {
			kernel: KernelId::ImageThresholdBinaryF32,
			contract: crate::core::operation::image::THRESHOLD_BINARY,
			code: 0,
		},
		[threshold, max_value, 0.0, 0.0],
		&[float("threshold", threshold), float("max_value", max_value)],
	)
}

/// Replace values at or below `threshold` with `max_value` and all others with zero.
pub fn threshold_binary_inv(input: &Image, threshold: f32, max_value: f32) -> Result<Image> {
	finite(&[threshold, max_value], "image::threshold_binary_inv")?;
	pointwise(
		input,
		Pointwise {
			kernel: KernelId::ImageThresholdBinaryInvF32,
			contract: crate::core::operation::image::THRESHOLD_BINARY_INV,
			code: 1,
		},
		[threshold, max_value, 0.0, 0.0],
		&[float("threshold", threshold), float("max_value", max_value)],
	)
}

/// Truncate values above `threshold`.
pub fn threshold_truncate(input: &Image, threshold: f32) -> Result<Image> {
	finite(&[threshold], "image::threshold_truncate")?;
	pointwise(
		input,
		Pointwise {
			kernel: KernelId::ImageThresholdTruncateF32,
			contract: crate::core::operation::image::THRESHOLD_TRUNCATE,
			code: 2,
		},
		[threshold, 0.0, 0.0, 0.0],
		&[float("threshold", threshold)],
	)
}

/// Replace values at or below `threshold` with zero.
pub fn threshold_to_zero(input: &Image, threshold: f32) -> Result<Image> {
	finite(&[threshold], "image::threshold_to_zero")?;
	pointwise(
		input,
		Pointwise {
			kernel: KernelId::ImageThresholdToZeroF32,
			contract: crate::core::operation::image::THRESHOLD_TO_ZERO,
			code: 3,
		},
		[threshold, 0.0, 0.0, 0.0],
		&[float("threshold", threshold)],
	)
}

/// Replace values above `threshold` with zero.
pub fn threshold_to_zero_inv(input: &Image, threshold: f32) -> Result<Image> {
	finite(&[threshold], "image::threshold_to_zero_inv")?;
	pointwise(
		input,
		Pointwise {
			kernel: KernelId::ImageThresholdToZeroInvF32,
			contract: crate::core::operation::image::THRESHOLD_TO_ZERO_INV,
			code: 4,
		},
		[threshold, 0.0, 0.0, 0.0],
		&[float("threshold", threshold)],
	)
}

/// Produce `true_value` for values inside the inclusive interval and zero outside it.
pub fn in_range(input: &Image, low: f32, high: f32, true_value: f32) -> Result<Image> {
	finite(&[low, high, true_value], "image::in_range")?;
	if low > high {
		return Err(Error::invalid_argument(
			"image::in_range low must not exceed high",
		));
	}
	pointwise(
		input,
		Pointwise {
			kernel: KernelId::ImageInRangeF32,
			contract: crate::core::operation::image::IN_RANGE,
			code: 5,
		},
		[low, high, true_value, 0.0],
		&[
			float("low", low),
			float("high", high),
			float("true_value", true_value),
		],
	)
}

/// Clamp every pixel to the inclusive interval.
pub fn clamp(input: &Image, low: f32, high: f32) -> Result<Image> {
	finite(&[low, high], "image::clamp")?;
	if low > high {
		return Err(Error::invalid_argument(
			"image::clamp low must not exceed high",
		));
	}
	pointwise(
		input,
		Pointwise {
			kernel: KernelId::ImageClampF32,
			contract: crate::core::operation::image::CLAMP,
			code: 6,
		},
		[low, high, 0.0, 0.0],
		&[float("low", low), float("high", high)],
	)
}

/// Invert every value around `max_value`.
pub fn invert(input: &Image, max_value: f32) -> Result<Image> {
	finite(&[max_value], "image::invert")?;
	pointwise(
		input,
		Pointwise {
			kernel: KernelId::ImageInvertF32,
			contract: crate::core::operation::image::INVERT,
			code: 7,
		},
		[max_value, 0.0, 0.0, 0.0],
		&[float("max_value", max_value)],
	)
}

/// Apply `value * contrast + brightness`.
pub fn brightness_contrast(input: &Image, brightness: f32, contrast: f32) -> Result<Image> {
	finite(&[brightness, contrast], "image::brightness_contrast")?;
	pointwise(
		input,
		Pointwise {
			kernel: KernelId::ImageBrightnessContrastF32,
			contract: crate::core::operation::image::BRIGHTNESS_CONTRAST,
			code: 8,
		},
		[contrast, brightness, 0.0, 0.0],
		&[float("brightness", brightness), float("contrast", contrast)],
	)
}

/// Apply `max(value, 0)^gamma * gain`.
pub fn gamma_contrast(input: &Image, gamma: f32, gain: f32) -> Result<Image> {
	finite(&[gamma, gain], "image::gamma_contrast")?;
	if gamma <= 0.0 {
		return Err(Error::invalid_argument(
			"image::gamma_contrast gamma must be positive",
		));
	}
	pointwise(
		input,
		Pointwise {
			kernel: KernelId::ImageGammaContrastF32,
			contract: crate::core::operation::image::GAMMA_CONTRAST,
			code: 9,
		},
		[gamma, gain, 0.0, 0.0],
		&[float("gamma", gamma), float("gain", gain)],
	)
}

/// Invert values at or above `threshold` around `max_value`.
pub fn solarize(input: &Image, threshold: f32, max_value: f32) -> Result<Image> {
	finite(&[threshold, max_value], "image::solarize")?;
	pointwise(
		input,
		Pointwise {
			kernel: KernelId::ImageSolarizeF32,
			contract: crate::core::operation::image::SOLARIZE,
			code: 10,
		},
		[threshold, max_value, 0.0, 0.0],
		&[float("threshold", threshold), float("max_value", max_value)],
	)
}

/// Quantize `[low, high]` to `levels` evenly spaced values using round-half-up.
pub fn posterize(input: &Image, levels: u32, low: f32, high: f32) -> Result<Image> {
	finite(&[low, high], "image::posterize")?;
	if !(2..=65_536).contains(&levels) || low >= high {
		return Err(Error::invalid_argument(
			"image::posterize requires 2..=65536 levels and low < high",
		));
	}
	pointwise(
		input,
		Pointwise {
			kernel: KernelId::ImagePosterizeF32,
			contract: crate::core::operation::image::POSTERIZE,
			code: 11,
		},
		[levels as f32, low, high, 0.0],
		&[
			unsigned("levels", levels),
			float("low", low),
			float("high", high),
		],
	)
}

/// Convert an RGB or RGBA image to Rec.709 grayscale.
pub fn grayscale(input: &Image) -> Result<Image> {
	let (batch, channels, height, width) = image_extent(input, "image::grayscale")?;
	if !matches!(input.format(), ImageFormat::Rgb | ImageFormat::Rgba) {
		return Err(Error::invalid_argument(
			"image::grayscale requires RGB or RGBA channel semantics",
		));
	}
	let shape = match input.layout() {
		ImageLayout::Nchw => vec![batch as usize, 1, height as usize, width as usize],
		ImageLayout::Chw => vec![1, height as usize, width as usize],
		_ => unreachable!("image extent admitted an unsupported layout"),
	};
	let output = allocate(input, shape, ImageFormat::Gray, "image::grayscale")?;
	record_images(
		input,
		&[],
		&output,
		KernelId::ImageGrayscaleF32,
		&[
			PushConstant::U32(batch),
			PushConstant::U32(channels),
			PushConstant::U32(height),
			PushConstant::U32(width),
		],
		crate::core::operation::image::GRAYSCALE,
		&[],
		&[
			BufferBinding::read(input.storage()),
			BufferBinding::write(output.storage()),
		],
		[width.div_ceil(16), height.div_ceil(16), batch],
	)?;
	Ok(output)
}

/// Blend two equal semantic images using `a * (1-alpha) + b * alpha`.
pub fn alpha_blend(a: &Image, b: &Image, alpha: f32) -> Result<Image> {
	finite(&[alpha], "image::alpha_blend")?;
	if !(0.0..=1.0).contains(&alpha) {
		return Err(Error::invalid_argument(
			"image::alpha_blend alpha must be in [0, 1]",
		));
	}
	validate_matching_images(a, b, "image::alpha_blend")?;
	composite_dispatch(
		a,
		b,
		a,
		0,
		alpha,
		[0; 4],
		0.0,
		crate::core::operation::image::ALPHA_BLEND,
		&[float("alpha", alpha)],
		KernelId::ImageAlphaBlendF32,
	)
}

/// Composite two equal images with a clamped Gray or per-channel mask.
pub fn composite(a: &Image, b: &Image, mask: &Image) -> Result<Image> {
	let (batch, channels, height, width) = image_extent(a, "image::composite")?;
	validate_matching_images(a, b, "image::composite")?;
	let (mask_batch, mask_channels, mask_height, mask_width) =
		image_extent(mask, "image::composite")?;
	if !a.engine_handle().same_as(mask.engine_handle())
		|| mask_batch != batch
		|| mask_height != height
		|| mask_width != width
		|| !matches!(mask_channels, 1) && mask_channels != channels
	{
		return Err(Error::invalid_argument(
			"image::composite mask must share batch/spatial extents and have one or matching channels",
		));
	}
	composite_dispatch(
		a,
		b,
		mask,
		1,
		0.0,
		[0; 4],
		0.0,
		crate::core::operation::image::COMPOSITE,
		&[],
		KernelId::ImageCompositeF32,
	)
}

/// Fill the intersection of a rectangle and the image with one finite value.
pub fn erase(input: &Image, x: u32, y: u32, width: u32, height: u32, value: f32) -> Result<Image> {
	finite(&[value], "image::erase")?;
	let (_, _, image_height, image_width) = image_extent(input, "image::erase")?;
	if width == 0 || height == 0 || x >= image_width || y >= image_height {
		return Err(Error::invalid_argument(
			"image::erase requires a nonzero rectangle with an in-bounds origin",
		));
	}
	composite_dispatch(
		input,
		input,
		input,
		2,
		0.0,
		[x, y, width, height],
		value,
		crate::core::operation::image::ERASE,
		&[
			unsigned("x", x),
			unsigned("y", y),
			unsigned("width", width),
			unsigned("height", height),
			float("value", value),
		],
		KernelId::ImageEraseF32,
	)
}

/// Apply one shared FP32 `[3, 4]` affine color transform to RGB channels.
pub fn color_twist(input: &Image, transform: &Matrix) -> Result<Image> {
	let (batch, channels, height, width) = image_extent(input, "image::color_twist")?;
	if !matches!(input.format(), ImageFormat::Rgb | ImageFormat::Rgba)
		|| transform.dtype() != DType::F32
		|| transform.shape() != [3, 4]
		|| !input.engine_handle().same_as(transform.engine_handle())
	{
		return Err(Error::invalid_argument(
			"image::color_twist requires RGB/RGBA input and a same-engine FP32 [3, 4] transform",
		));
	}
	let output = allocate(
		input,
		input.as_matrix().shape().to_vec(),
		input.format(),
		"image::color_twist",
	)?;
	let workgroups = [width.div_ceil(16), height.div_ceil(16), batch * channels];
	input.engine_handle().record_image_semantic(
		ComputeDispatch {
			kernel: KernelId::ImageColorTwistF32,
			buffers: &[
				BufferBinding::read(input.storage()),
				BufferBinding::read(transform.storage()),
				BufferBinding::write(output.storage()),
			],
			push_constants: &[
				PushConstant::U32(batch),
				PushConstant::U32(channels),
				PushConstant::U32(height),
				PushConstant::U32(width),
			],
			workgroups,
		},
		ImageSemanticDispatch {
			contract: crate::core::operation::image::COLOR_TWIST,
			inputs: &[
				ImageSemanticInput::Image(input),
				ImageSemanticInput::Matrix(transform),
			],
			outputs: &[&output],
			attributes: &[],
		},
	)?;
	Ok(output)
}

/// Reorder one to four channels and explicitly assign the resulting format.
///
/// `order[c]` selects the source channel for output channel `c`. The output
/// format must have the same channel count as the input so pixel data and
/// semantic metadata cannot diverge.
pub fn channel_reorder(
	input: &Image,
	order: [u32; 4],
	output_format: ImageFormat,
) -> Result<Image> {
	let (batch, channels, height, width) = image_extent(input, "image::channel_reorder")?;
	if output_format.channels() != input.channels()
		|| order[..channels as usize]
			.iter()
			.any(|channel| *channel >= channels)
	{
		return Err(Error::invalid_argument(
			"image::channel_reorder requires in-range source channels and an equal-width output format",
		));
	}
	let output = allocate(
		input,
		input.as_matrix().shape().to_vec(),
		output_format,
		"image::channel_reorder",
	)?;
	input.engine_handle().record_image_semantic(
		ComputeDispatch {
			kernel: KernelId::ImageChannelReorderF32,
			buffers: &[
				BufferBinding::read(input.storage()),
				BufferBinding::write(output.storage()),
			],
			push_constants: &[
				PushConstant::U32(batch),
				PushConstant::U32(channels),
				PushConstant::U32(height),
				PushConstant::U32(width),
				PushConstant::U32(order[0]),
				PushConstant::U32(order[1]),
				PushConstant::U32(order[2]),
				PushConstant::U32(order[3]),
			],
			workgroups: [width.div_ceil(16), height.div_ceil(16), batch * channels],
		},
		ImageSemanticDispatch {
			contract: crate::core::operation::image::CHANNEL_REORDER,
			inputs: &[ImageSemanticInput::Image(input)],
			outputs: &[&output],
			attributes: &[
				unsigned("channel_0", order[0]),
				unsigned("channel_1", order[1]),
				unsigned("channel_2", order[2]),
				unsigned("channel_3", order[3]),
				enumeration("output_format", image_format_token(output_format)),
			],
		},
	)?;
	Ok(output)
}

/// Add explicit-seed Philox Gaussian noise to every pixel.
pub fn gaussian_noise(input: &Image, mean: f32, stddev: f32, seed: u64) -> Result<Image> {
	finite(&[mean, stddev], "image::gaussian_noise")?;
	if stddev < 0.0 {
		return Err(Error::invalid_argument(
			"image::gaussian_noise standard deviation must be nonnegative",
		));
	}
	noise(
		input,
		KernelId::ImageGaussianNoiseF32,
		crate::core::operation::image::GAUSSIAN_NOISE,
		&[
			PushConstant::U32(seed as u32),
			PushConstant::U32((seed >> 32) as u32),
			PushConstant::F32(mean),
			PushConstant::F32(stddev),
		],
		&[
			float("mean", mean),
			float("stddev", stddev),
			unsigned64("seed", seed),
		],
	)
}

/// Replace a seeded fraction of pixels with salt or pepper values.
pub fn salt_pepper_noise(
	input: &Image,
	probability: f32,
	salt_value: f32,
	pepper_value: f32,
	seed: u64,
) -> Result<Image> {
	finite(
		&[probability, salt_value, pepper_value],
		"image::salt_pepper_noise",
	)?;
	if !(0.0..=1.0).contains(&probability) {
		return Err(Error::invalid_argument(
			"image::salt_pepper_noise probability must be in [0, 1]",
		));
	}
	noise(
		input,
		KernelId::ImageSaltPepperNoiseF32,
		crate::core::operation::image::SALT_PEPPER_NOISE,
		&[
			PushConstant::U32(seed as u32),
			PushConstant::U32((seed >> 32) as u32),
			PushConstant::F32(probability),
			PushConstant::F32(salt_value),
			PushConstant::F32(pepper_value),
		],
		&[
			float("probability", probability),
			float("salt_value", salt_value),
			float("pepper_value", pepper_value),
			unsigned64("seed", seed),
		],
	)
}

fn noise(
	input: &Image,
	kernel: KernelId,
	contract: OperationContract,
	kernel_parameters: &[PushConstant],
	attributes: &[OpAttribute],
) -> Result<Image> {
	image_extent(input, contract.name())?;
	let element_count = u32::try_from(input.as_matrix().num_elements())
		.map_err(|_| Error::out_of_range(format!("{} element count exceeds u32", contract.name())))?;
	let output = allocate(
		input,
		input.as_matrix().shape().to_vec(),
		input.format(),
		contract.name(),
	)?;
	let mut push_constants = Vec::with_capacity(1 + kernel_parameters.len());
	push_constants.push(PushConstant::U32(element_count));
	push_constants.extend_from_slice(kernel_parameters);
	input.engine_handle().record_image_semantic(
		ComputeDispatch {
			kernel,
			buffers: &[
				BufferBinding::read(input.storage()),
				BufferBinding::write(output.storage()),
			],
			push_constants: &push_constants,
			workgroups: kernel.linear_workgroups(element_count),
		},
		ImageSemanticDispatch {
			contract,
			inputs: &[ImageSemanticInput::Image(input)],
			outputs: &[&output],
			attributes,
		},
	)?;
	Ok(output)
}

#[allow(clippy::too_many_arguments)]
fn composite_dispatch(
	a: &Image,
	b: &Image,
	mask: &Image,
	operation: u32,
	alpha: f32,
	rectangle: [u32; 4],
	value: f32,
	contract: OperationContract,
	attributes: &[OpAttribute],
	kernel: KernelId,
) -> Result<Image> {
	let (batch, channels, height, width) = image_extent(a, contract.name())?;
	let mask_channels = u32::try_from(mask.channels())
		.map_err(|_| Error::out_of_range(format!("{} mask channels exceed u32", contract.name())))?;
	let output = allocate(
		a,
		a.as_matrix().shape().to_vec(),
		a.format(),
		contract.name(),
	)?;
	let buffers = [
		BufferBinding::read(a.storage()),
		BufferBinding::read(b.storage()),
		BufferBinding::read(mask.storage()),
		BufferBinding::write(output.storage()),
	];
	let push_constants = [
		PushConstant::U32(batch),
		PushConstant::U32(channels),
		PushConstant::U32(height),
		PushConstant::U32(width),
		PushConstant::U32(operation),
		PushConstant::U32(mask_channels),
		PushConstant::U32(rectangle[0]),
		PushConstant::U32(rectangle[1]),
		PushConstant::U32(rectangle[2]),
		PushConstant::U32(rectangle[3]),
		PushConstant::F32(alpha),
		PushConstant::F32(value),
	];
	let workgroups = [width.div_ceil(16), height.div_ceil(16), batch * channels];
	let semantic_inputs = match operation {
		0 => vec![ImageSemanticInput::Image(a), ImageSemanticInput::Image(b)],
		1 => vec![
			ImageSemanticInput::Image(a),
			ImageSemanticInput::Image(b),
			ImageSemanticInput::Image(mask),
		],
		_ => vec![ImageSemanticInput::Image(a)],
	};
	a.engine_handle().record_image_semantic(
		ComputeDispatch {
			kernel,
			buffers: &buffers,
			push_constants: &push_constants,
			workgroups,
		},
		ImageSemanticDispatch {
			contract,
			inputs: &semantic_inputs,
			outputs: &[&output],
			attributes,
		},
	)?;
	Ok(output)
}

fn validate_matching_images(a: &Image, b: &Image, operation: &str) -> Result<()> {
	image_extent(a, operation)?;
	if a.as_matrix().shape() != b.as_matrix().shape()
		|| a.dtype() != b.dtype()
		|| a.layout() != b.layout()
		|| a.format() != b.format()
		|| !a.engine_handle().same_as(b.engine_handle())
	{
		return Err(Error::invalid_argument(format!(
			"{operation} images must have identical shape, dtype, layout, format, and engine"
		)));
	}
	Ok(())
}

fn image_extent(input: &Image, operation: &str) -> Result<(u32, u32, u32, u32)> {
	if input.dtype() != DType::F32 || !matches!(input.layout(), ImageLayout::Nchw | ImageLayout::Chw)
	{
		return Err(Error::invalid_argument(format!(
			"{operation} requires an FP32 NCHW or CHW image"
		)));
	}
	let extent = (
		u32::try_from(input.batch_size()),
		u32::try_from(input.channels()),
		u32::try_from(input.height()),
		u32::try_from(input.width()),
	);
	let (batch, channels, height, width) = match extent {
		(Ok(batch), Ok(channels), Ok(height), Ok(width)) => (batch, channels, height, width),
		_ => {
			return Err(Error::out_of_range(format!(
				"{operation} extent exceeds u32"
			)));
		}
	};
	if batch == 0 || channels == 0 || height == 0 || width == 0 {
		return Err(Error::invalid_argument(format!(
			"{operation} requires a nonempty image"
		)));
	}
	Ok((batch, channels, height, width))
}

fn allocate(
	input: &Image,
	shape: Vec<usize>,
	format: ImageFormat,
	operation: &str,
) -> Result<Image> {
	let elements = shape.iter().copied().try_fold(1_usize, |count, extent| {
		count
			.checked_mul(extent)
			.ok_or_else(|| Error::out_of_range(format!("{operation} output size exceeds usize")))
	})?;
	u32::try_from(elements)
		.map_err(|_| Error::out_of_range(format!("{operation} output size exceeds u32")))?;
	let matrix = Matrix::allocate(input.engine_handle(), shape, elements, DType::F32)?;
	Image::new(matrix, input.layout(), format)
}

#[allow(clippy::too_many_arguments)]
fn record_images(
	input: &Image,
	extra_inputs: &[&Image],
	output: &Image,
	kernel: KernelId,
	push_constants: &[PushConstant],
	contract: OperationContract,
	attributes: &[OpAttribute],
	buffers: &[BufferBinding<'_>],
	workgroups: [u32; 3],
) -> Result<()> {
	let mut inputs = Vec::with_capacity(1 + extra_inputs.len());
	inputs.push(ImageSemanticInput::Image(input));
	inputs.extend(
		extra_inputs
			.iter()
			.map(|image| ImageSemanticInput::Image(image)),
	);
	input.engine_handle().record_image_semantic(
		ComputeDispatch {
			kernel,
			buffers,
			push_constants,
			workgroups,
		},
		ImageSemanticDispatch {
			contract,
			inputs: &inputs,
			outputs: &[output],
			attributes,
		},
	)
}

fn pointwise(
	input: &Image,
	op: Pointwise,
	parameters: [f32; 4],
	attributes: &[OpAttribute],
) -> Result<Image> {
	if input.dtype() != DType::F32 {
		return Err(Error::invalid_argument(format!(
			"{} requires f32 storage",
			op.contract.name()
		)));
	}
	let element_count = u32::try_from(input.as_matrix().num_elements()).map_err(|_| {
		Error::out_of_range(format!("{} element count exceeds u32", op.contract.name()))
	})?;
	if element_count == 0 {
		return Err(Error::invalid_argument(format!(
			"{} requires a nonempty image",
			op.contract.name()
		)));
	}
	let matrix = crate::Matrix::allocate(
		input.engine_handle(),
		input.as_matrix().shape().to_vec(),
		input.as_matrix().num_elements(),
		DType::F32,
	)?;
	let output = Image::new(matrix, input.layout(), input.format())?;
	input.engine_handle().record_image_semantic(
		ComputeDispatch {
			kernel: op.kernel,
			buffers: &[
				BufferBinding::read(input.storage()),
				BufferBinding::write(output.storage()),
			],
			push_constants: &[
				PushConstant::U32(element_count),
				PushConstant::U32(op.code),
				PushConstant::F32(parameters[0]),
				PushConstant::F32(parameters[1]),
				PushConstant::F32(parameters[2]),
				PushConstant::F32(parameters[3]),
			],
			workgroups: op.kernel.linear_workgroups(element_count),
		},
		ImageSemanticDispatch {
			contract: op.contract,
			inputs: &[ImageSemanticInput::Image(input)],
			outputs: &[&output],
			attributes,
		},
	)?;
	Ok(output)
}

fn finite(values: &[f32], operation: &str) -> Result<()> {
	if values.iter().all(|value| value.is_finite()) {
		Ok(())
	} else {
		Err(Error::invalid_argument(format!(
			"{operation} parameters must be finite"
		)))
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

fn unsigned64(name: &str, value: u64) -> OpAttribute {
	OpAttribute::UnsignedInteger {
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

const fn image_format_token(format: ImageFormat) -> &'static str {
	match format {
		ImageFormat::Gray => "gray",
		ImageFormat::GrayAlpha => "gray_alpha",
		ImageFormat::Rgb => "rgb",
		ImageFormat::Rgba => "rgba",
		ImageFormat::Bgr => "bgr",
		ImageFormat::Bgra => "bgra",
	}
}
