use crate::runtime::{BufferBinding, KernelId, PushConstant};
use crate::{DType, Error, Matrix, OpAttribute, Result};

use super::common::{record_semantic, shader_u32, validate_f32_same_engine};
use crate::ml::matrix::UpsampleMode;

#[derive(Clone, Copy)]
struct Upsample2dGeometry {
	batch_size: u32,
	channels: u32,
	input_height: u32,
	input_width: u32,
	output_height: u32,
	output_width: u32,
	scale_factor: u32,
}

impl Upsample2dGeometry {
	fn resolve(input: &Matrix, scale_factor: usize, operation: &'static str) -> Result<Self> {
		let [batch_size, channels, input_height, input_width] = input.shape() else {
			return Err(Error::invalid_argument(format!(
				"{operation} input must have NCHW rank four; found {:?}",
				input.shape()
			)));
		};
		if [*batch_size, *channels, *input_height, *input_width]
			.into_iter()
			.any(|extent| extent == 0)
			|| scale_factor == 0
		{
			return Err(Error::invalid_argument(format!(
				"{operation} input extents and scale factor must be nonzero"
			)));
		}
		let output_height = input_height.checked_mul(scale_factor).ok_or_else(|| {
			Error::invalid_argument(format!("{operation} output height overflows usize"))
		})?;
		let output_width = input_width.checked_mul(scale_factor).ok_or_else(|| {
			Error::invalid_argument(format!("{operation} output width overflows usize"))
		})?;
		batch_size
			.checked_mul(*channels)
			.and_then(|value| value.checked_mul(output_height))
			.and_then(|value| value.checked_mul(output_width))
			.ok_or_else(|| Error::invalid_argument(format!("{operation} output size overflows usize")))?;
		let batch_size = shader_u32(*batch_size, "batch size", operation)?;
		let channels = shader_u32(*channels, "channel count", operation)?;
		batch_size.checked_mul(channels).ok_or_else(|| {
			Error::invalid_argument(format!("{operation} batch-channel count exceeds u32"))
		})?;
		Ok(Self {
			batch_size,
			channels,
			input_height: shader_u32(*input_height, "input height", operation)?,
			input_width: shader_u32(*input_width, "input width", operation)?,
			output_height: shader_u32(output_height, "output height", operation)?,
			output_width: shader_u32(output_width, "output width", operation)?,
			scale_factor: shader_u32(scale_factor, "scale factor", operation)?,
		})
	}

	const fn push_constants(self) -> [PushConstant; 7] {
		[
			PushConstant::U32(self.batch_size),
			PushConstant::U32(self.channels),
			PushConstant::U32(self.input_height),
			PushConstant::U32(self.input_width),
			PushConstant::U32(self.output_height),
			PushConstant::U32(self.output_width),
			PushConstant::U32(self.scale_factor),
		]
	}

	const fn forward_workgroups(self) -> [u32; 3] {
		[
			self.output_width.div_ceil(16),
			self.output_height.div_ceil(16),
			self.batch_size * self.channels,
		]
	}

	const fn backward_workgroups(self) -> [u32; 3] {
		[
			self.input_width.div_ceil(16),
			self.input_height.div_ceil(16),
			self.batch_size * self.channels,
		]
	}

	fn output_shape(self) -> Vec<usize> {
		vec![
			self.batch_size as usize,
			self.channels as usize,
			self.output_height as usize,
			self.output_width as usize,
		]
	}
}

fn attributes(scale_factor: usize, mode: UpsampleMode) -> Result<[OpAttribute; 2]> {
	Ok([
		OpAttribute::UnsignedInteger {
			name: "scale_factor".into(),
			value: u64::try_from(scale_factor)
				.map_err(|_| Error::invalid_argument("upsample scale factor exceeds u64"))?,
		},
		OpAttribute::Enum {
			name: "mode".into(),
			value: mode.token().into(),
		},
	])
}

pub(in crate::ml) fn upsample_2d(
	input: &Matrix,
	scale_factor: usize,
	mode: UpsampleMode,
) -> Result<Matrix> {
	let operation = crate::core::operation::ml::UPSAMPLE_2D.name();
	validate_f32_same_engine(operation, &[input])?;
	let geometry = Upsample2dGeometry::resolve(input, scale_factor, operation)?;
	let output_shape = geometry.output_shape();
	let output_count = output_shape
		.iter()
		.try_fold(1_usize, |product, extent| product.checked_mul(*extent))
		.ok_or_else(|| Error::invalid_argument(format!("{operation} output size overflows usize")))?;
	let output = Matrix::allocate(
		input.engine_handle(),
		output_shape,
		output_count,
		DType::F32,
	)?;
	let kernel = match mode {
		UpsampleMode::Nearest => KernelId::MlUpsample2dNearestF32,
		UpsampleMode::Bilinear => KernelId::MlUpsample2dBilinearF32,
	};
	let buffers = [
		BufferBinding::read(input.storage()),
		BufferBinding::write(output.storage()),
	];
	let push_constants = geometry.push_constants();
	let attributes = attributes(scale_factor, mode)?;
	record_semantic(
		&[input],
		&[&output],
		&attributes,
		kernel,
		&buffers,
		&push_constants,
		geometry.forward_workgroups(),
	)?;
	Ok(output)
}

pub(in crate::ml) fn upsample_2d_backward(
	input: &Matrix,
	output_gradient: &Matrix,
	scale_factor: usize,
	mode: UpsampleMode,
) -> Result<Matrix> {
	let operation = crate::core::operation::ml::UPSAMPLE_2D_BACKWARD.name();
	validate_f32_same_engine(operation, &[input, output_gradient])?;
	let geometry = Upsample2dGeometry::resolve(input, scale_factor, operation)?;
	if output_gradient.shape() != geometry.output_shape() {
		return Err(Error::invalid_argument(format!(
			"{operation} output gradient shape {:?} does not match {:?}",
			output_gradient.shape(),
			geometry.output_shape()
		)));
	}
	let input_gradient = Matrix::allocate(
		input.engine_handle(),
		input.shape().to_vec(),
		input.num_elements(),
		DType::F32,
	)?;
	let kernel = match mode {
		UpsampleMode::Nearest => KernelId::MlUpsample2dNearestBackwardF32,
		UpsampleMode::Bilinear => KernelId::MlUpsample2dBilinearBackwardF32,
	};
	let buffers = [
		BufferBinding::read(output_gradient.storage()),
		BufferBinding::write(input_gradient.storage()),
	];
	let push_constants = geometry.push_constants();
	let attributes = attributes(scale_factor, mode)?;
	record_semantic(
		&[input, output_gradient],
		&[&input_gradient],
		&attributes,
		kernel,
		&buffers,
		&push_constants,
		geometry.backward_workgroups(),
	)?;
	Ok(input_gradient)
}
