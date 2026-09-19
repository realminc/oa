//! Per-channel image color transforms.

use crate::{
	DType, Error, Image, ImageLayout, Matrix, OpAttribute, Result,
	runtime::{
		BufferBinding, ComputeDispatch, ImageSemanticDispatch, ImageSemanticInput, KernelId,
		PushConstant,
	},
};

/// Three-channel normalization parameters.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NormalizationParams {
	pub mean: [f32; 3],
	pub std: [f32; 3],
}

impl Default for NormalizationParams {
	fn default() -> Self {
		Self {
			mean: [0.0; 3],
			std: [1.0; 3],
		}
	}
}

/// Apply `(pixel - mean[channel]) / std[channel]` to up to three channels.
pub fn normalize(input: &Image, parameters: NormalizationParams) -> Result<Image> {
	if input.dtype() != DType::F32 || !matches!(input.layout(), ImageLayout::Nchw | ImageLayout::Chw)
	{
		return Err(Error::invalid_argument(
			"image::normalize requires an FP32 NCHW or CHW image",
		));
	}
	if input.channels() > 3 || input.batch_size() == 0 || input.height() == 0 || input.width() == 0 {
		return Err(Error::invalid_argument(
			"image::normalize requires a nonempty image with at most three channels",
		));
	}
	for channel in 0..input.channels() {
		if !parameters.mean[channel].is_finite()
			|| !parameters.std[channel].is_finite()
			|| parameters.std[channel] <= 0.0
		{
			return Err(Error::invalid_argument(
				"image::normalize requires finite means and positive finite standard deviations",
			));
		}
	}
	let batch = u32::try_from(input.batch_size())
		.map_err(|_| Error::out_of_range("image::normalize batch exceeds u32"))?;
	let channels = u32::try_from(input.channels())
		.map_err(|_| Error::out_of_range("image::normalize channels exceed u32"))?;
	let height = u32::try_from(input.height())
		.map_err(|_| Error::out_of_range("image::normalize height exceeds u32"))?;
	let width = u32::try_from(input.width())
		.map_err(|_| Error::out_of_range("image::normalize width exceeds u32"))?;
	let elements = input.as_matrix().num_elements();
	let matrix = Matrix::allocate(
		input.engine_handle(),
		input.as_matrix().shape().to_vec(),
		elements,
		DType::F32,
	)?;
	let output = Image::new(matrix, input.layout(), input.format())?;
	let attributes = [
		float("mean_0", parameters.mean[0]),
		float("mean_1", parameters.mean[1]),
		float("mean_2", parameters.mean[2]),
		float("std_0", parameters.std[0]),
		float("std_1", parameters.std[1]),
		float("std_2", parameters.std[2]),
	];
	input.engine_handle().record_image_semantic(
		ComputeDispatch {
			kernel: KernelId::ImageNormalizeF32,
			buffers: &[
				BufferBinding::read(input.storage()),
				BufferBinding::write(output.storage()),
			],
			push_constants: &[
				PushConstant::U32(batch),
				PushConstant::U32(channels),
				PushConstant::U32(height),
				PushConstant::U32(width),
				PushConstant::F32(parameters.mean[0]),
				PushConstant::F32(parameters.mean[1]),
				PushConstant::F32(parameters.mean[2]),
				PushConstant::F32(parameters.std[0]),
				PushConstant::F32(parameters.std[1]),
				PushConstant::F32(parameters.std[2]),
			],
			workgroups: [width.div_ceil(16), height.div_ceil(16), batch * channels],
		},
		ImageSemanticDispatch {
			contract: crate::core::operation::image::NORMALIZE,
			inputs: &[ImageSemanticInput::Image(input)],
			outputs: &[&output],
			attributes: &attributes,
		},
	)?;
	Ok(output)
}

/// Convert between three-channel RGB and BGR semantics, including identity.
pub fn convert_color(input: &Image, destination_format: crate::ImageFormat) -> Result<Image> {
	if !matches!(
		input.format(),
		crate::ImageFormat::Rgb | crate::ImageFormat::Bgr
	) || !matches!(
		destination_format,
		crate::ImageFormat::Rgb | crate::ImageFormat::Bgr
	) || input.dtype() != DType::F32
		|| !matches!(input.layout(), ImageLayout::Nchw | ImageLayout::Chw)
	{
		return Err(Error::invalid_argument(
			"image::convert_color currently admits FP32 NCHW/CHW RGB and BGR images",
		));
	}
	let batch = extent(input.batch_size(), "batch")?;
	let height = extent(input.height(), "height")?;
	let width = extent(input.width(), "width")?;
	if batch == 0 || height == 0 || width == 0 {
		return Err(Error::invalid_argument(
			"image::convert_color requires a nonempty image",
		));
	}
	let output = allocate(
		input,
		input.as_matrix().shape().to_vec(),
		destination_format,
	)?;
	let swap = u32::from(input.format() != destination_format);
	record_image(
		input,
		&output,
		KernelId::ImageConvertColorF32,
		&[
			PushConstant::U32(batch),
			PushConstant::U32(height),
			PushConstant::U32(width),
			PushConstant::U32(swap),
		],
		crate::core::operation::image::CONVERT_COLOR,
		&[enumeration(
			"destination_format",
			format_token(destination_format),
		)],
		[width.div_ceil(16), height.div_ceil(16), batch * 3],
	)
}

/// Bilinearly resize and normalize in one semantic and executable operation.
pub fn resize_normalize(
	input: &Image,
	width: u32,
	height: u32,
	parameters: NormalizationParams,
) -> Result<Image> {
	validate_normalization(input, parameters)?;
	if width == 0 || height == 0 {
		return Err(Error::invalid_argument(
			"image::resize_normalize target extent must be nonzero",
		));
	}
	let batch = extent(input.batch_size(), "batch")?;
	let channels = extent(input.channels(), "channels")?;
	let input_height = extent(input.height(), "input height")?;
	let input_width = extent(input.width(), "input width")?;
	let shape = match input.layout() {
		ImageLayout::Nchw => vec![
			batch as usize,
			channels as usize,
			height as usize,
			width as usize,
		],
		ImageLayout::Chw => vec![channels as usize, height as usize, width as usize],
		_ => unreachable!("normalization validation admitted unsupported layout"),
	};
	let output = allocate(input, shape, input.format())?;
	let attributes = [
		unsigned("target_width", width),
		unsigned("target_height", height),
		float("mean_0", parameters.mean[0]),
		float("mean_1", parameters.mean[1]),
		float("mean_2", parameters.mean[2]),
		float("std_0", parameters.std[0]),
		float("std_1", parameters.std[1]),
		float("std_2", parameters.std[2]),
	];
	record_image(
		input,
		&output,
		KernelId::ImageResizeNormalizeF32,
		&[
			PushConstant::U32(batch),
			PushConstant::U32(channels),
			PushConstant::U32(input_height),
			PushConstant::U32(input_width),
			PushConstant::U32(height),
			PushConstant::U32(width),
			PushConstant::F32(parameters.mean[0]),
			PushConstant::F32(parameters.mean[1]),
			PushConstant::F32(parameters.mean[2]),
			PushConstant::F32(parameters.std[0]),
			PushConstant::F32(parameters.std[1]),
			PushConstant::F32(parameters.std[2]),
		],
		crate::core::operation::image::RESIZE_NORMALIZE,
		&attributes,
		[width.div_ceil(16), height.div_ceil(16), batch * channels],
	)
}

/// Blend a class palette over an RGB/RGBA image using an I32 label map.
pub fn segmentation_overlay(
	input: &Image,
	mask: &Matrix,
	palette: &Matrix,
	alpha: f32,
) -> Result<Image> {
	if input.dtype() != DType::F32
		|| input.layout() != ImageLayout::Nchw
		|| !matches!(
			input.format(),
			crate::ImageFormat::Rgb | crate::ImageFormat::Rgba
		) {
		return Err(Error::invalid_argument(
			"image::segmentation_overlay requires an FP32 NCHW RGB/RGBA image",
		));
	}
	if !alpha.is_finite() || !(0.0..=1.0).contains(&alpha) {
		return Err(Error::invalid_argument(
			"image::segmentation_overlay alpha must be finite and in [0,1]",
		));
	}
	let batch = input.batch_size();
	let height = input.height();
	let width = input.width();
	let valid_mask = mask.dtype() == DType::I32
		&& (mask.shape() == [batch, height, width] || mask.shape() == [batch, 1, height, width]);
	let valid_palette = palette.dtype() == DType::F32
		&& palette.shape().len() == 2
		&& palette.shape()[0] > 0
		&& palette.shape()[1] == 3;
	if !valid_mask
		|| !valid_palette
		|| !input.engine_handle().same_as(mask.engine_handle())
		|| !input.engine_handle().same_as(palette.engine_handle())
	{
		return Err(Error::invalid_argument(
			"image::segmentation_overlay requires same-engine I32 [B,H,W]/[B,1,H,W] labels and FP32 [K,3] palette",
		));
	}
	let batch_u32 = extent(batch, "batch")?;
	let channels = extent(input.channels(), "channels")?;
	let height_u32 = extent(height, "height")?;
	let width_u32 = extent(width, "width")?;
	let classes = extent(palette.shape()[0], "class count")?;
	let output = allocate(input, input.as_matrix().shape().to_vec(), input.format())?;
	input.engine_handle().record_image_semantic(
		ComputeDispatch {
			kernel: KernelId::ImageSegmentationOverlayF32,
			buffers: &[
				BufferBinding::read(input.storage()),
				BufferBinding::read(mask.storage()),
				BufferBinding::read(palette.storage()),
				BufferBinding::write(output.storage()),
			],
			push_constants: &[
				PushConstant::U32(batch_u32),
				PushConstant::U32(channels),
				PushConstant::U32(height_u32),
				PushConstant::U32(width_u32),
				PushConstant::U32(classes),
				PushConstant::F32(alpha),
			],
			workgroups: [
				width_u32.div_ceil(16),
				height_u32.div_ceil(16),
				batch_u32 * channels,
			],
		},
		ImageSemanticDispatch {
			contract: crate::core::operation::image::SEGMENTATION_OVERLAY,
			inputs: &[
				ImageSemanticInput::Image(input),
				ImageSemanticInput::Matrix(mask),
				ImageSemanticInput::Matrix(palette),
			],
			outputs: &[&output],
			attributes: &[float("alpha", alpha)],
		},
	)?;
	Ok(output)
}

fn validate_normalization(input: &Image, parameters: NormalizationParams) -> Result<()> {
	if input.dtype() != DType::F32
		|| !matches!(input.layout(), ImageLayout::Nchw | ImageLayout::Chw)
		|| input.channels() > 3
		|| input.batch_size() == 0
		|| input.height() == 0
		|| input.width() == 0
	{
		return Err(Error::invalid_argument(
			"image normalization requires a nonempty FP32 NCHW/CHW image with at most three channels",
		));
	}
	for channel in 0..input.channels() {
		if !parameters.mean[channel].is_finite()
			|| !parameters.std[channel].is_finite()
			|| parameters.std[channel] <= 0.0
		{
			return Err(Error::invalid_argument(
				"image normalization requires finite means and positive finite standard deviations",
			));
		}
	}
	Ok(())
}

fn allocate(input: &Image, shape: Vec<usize>, format: crate::ImageFormat) -> Result<Image> {
	let elements = shape.iter().copied().try_fold(1usize, |count, value| {
		count
			.checked_mul(value)
			.ok_or_else(|| Error::out_of_range("image output size exceeds usize"))
	})?;
	u32::try_from(elements).map_err(|_| Error::out_of_range("image output size exceeds u32"))?;
	Image::new(
		Matrix::allocate(input.engine_handle(), shape, elements, DType::F32)?,
		input.layout(),
		format,
	)
}

fn record_image(
	input: &Image,
	output: &Image,
	kernel: KernelId,
	push_constants: &[PushConstant],
	contract: crate::OperationContract,
	attributes: &[OpAttribute],
	workgroups: [u32; 3],
) -> Result<Image> {
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
	)?;
	Ok(output.clone())
}

fn extent(value: usize, label: &str) -> Result<u32> {
	u32::try_from(value).map_err(|_| Error::out_of_range(format!("image {label} exceeds u32")))
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
const fn format_token(format: crate::ImageFormat) -> &'static str {
	match format {
		crate::ImageFormat::Gray => "gray",
		crate::ImageFormat::GrayAlpha => "gray_alpha",
		crate::ImageFormat::Rgb => "rgb",
		crate::ImageFormat::Rgba => "rgba",
		crate::ImageFormat::Bgr => "bgr",
		crate::ImageFormat::Bgra => "bgra",
	}
}

fn float(name: &str, value: f32) -> OpAttribute {
	OpAttribute::Float {
		name: name.into(),
		value: f64::from(value),
	}
}
