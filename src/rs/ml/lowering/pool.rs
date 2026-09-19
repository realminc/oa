//! Private lowering for ML pooling operations.

use crate::{
	DType, Error, Matrix, OpAttribute, Result,
	runtime::{BufferBinding, ComputeDispatch, KernelId, PushConstant, SemanticDispatch},
};

use super::common::{shader_u32, validate_f32_same_engine};

#[derive(Clone, Copy)]
struct Pool2dGeometry {
	batch: u32,
	channels: u32,
	input_height: u32,
	input_width: u32,
	output_height: u32,
	output_width: u32,
	kernel_size: u32,
	stride: u32,
	padding: u32,
}

impl Pool2dGeometry {
	fn resolve(
		input: &Matrix,
		kernel_size: usize,
		stride: usize,
		padding: usize,
		operation: &'static str,
	) -> Result<Self> {
		let [batch, channels, input_height, input_width] = input.shape() else {
			return Err(Error::invalid_argument(format!(
				"{operation} requires NCHW rank-four input; found {:?}",
				input.shape()
			)));
		};
		if *batch == 0
			|| *channels == 0
			|| *input_height == 0
			|| *input_width == 0
			|| kernel_size == 0
			|| stride == 0
		{
			return Err(Error::invalid_argument(format!(
				"{operation} requires nonzero NCHW extents, kernel size, and stride"
			)));
		}
		shader_u32(input.num_elements(), "input element count", operation)?;
		let doubled_padding = padding.checked_mul(2).ok_or_else(|| {
			Error::invalid_argument(format!("{operation} doubled padding overflows usize"))
		})?;
		let padded_height = input_height.checked_add(doubled_padding).ok_or_else(|| {
			Error::invalid_argument(format!("{operation} padded height overflows usize"))
		})?;
		let padded_width = input_width.checked_add(doubled_padding).ok_or_else(|| {
			Error::invalid_argument(format!("{operation} padded width overflows usize"))
		})?;
		if padded_height < kernel_size || padded_width < kernel_size {
			return Err(Error::invalid_argument(format!(
				"{operation} kernel does not fit the padded input"
			)));
		}
		let output_height = (padded_height - kernel_size) / stride + 1;
		let output_width = (padded_width - kernel_size) / stride + 1;
		let last_y = output_height
			.saturating_sub(1)
			.checked_mul(stride)
			.ok_or_else(|| Error::invalid_argument(format!("{operation} Y origin overflows usize")))?;
		let last_x = output_width
			.saturating_sub(1)
			.checked_mul(stride)
			.ok_or_else(|| Error::invalid_argument(format!("{operation} X origin overflows usize")))?;
		let last_y_end = last_y.checked_add(kernel_size).ok_or_else(|| {
			Error::invalid_argument(format!("{operation} final Y window overflows usize"))
		})?;
		let last_x_end = last_x.checked_add(kernel_size).ok_or_else(|| {
			Error::invalid_argument(format!("{operation} final X window overflows usize"))
		})?;
		for (label, value) in [
			("input height", *input_height),
			("input width", *input_width),
			("kernel size", kernel_size),
			("padding", padding),
			("last Y origin", last_y),
			("last X origin", last_x),
			("final Y window end", last_y_end),
			("final X window end", last_x_end),
		] {
			if value > i32::MAX as usize {
				return Err(Error::invalid_argument(format!(
					"{operation} {label} exceeds the signed shader coordinate ABI"
				)));
			}
		}
		let batch = shader_u32(*batch, "batch size", operation)?;
		let channels = shader_u32(*channels, "channel count", operation)?;
		batch.checked_mul(channels).ok_or_else(|| {
			Error::invalid_argument(format!("{operation} batch-channel count exceeds u32"))
		})?;
		Ok(Self {
			batch,
			channels,
			input_height: shader_u32(*input_height, "input height", operation)?,
			input_width: shader_u32(*input_width, "input width", operation)?,
			output_height: shader_u32(output_height, "output height", operation)?,
			output_width: shader_u32(output_width, "output width", operation)?,
			kernel_size: shader_u32(kernel_size, "kernel size", operation)?,
			stride: shader_u32(stride, "stride", operation)?,
			padding: shader_u32(padding, "padding", operation)?,
		})
	}

	fn push_constants(self) -> [PushConstant; 9] {
		[
			PushConstant::U32(self.batch),
			PushConstant::U32(self.channels),
			PushConstant::U32(self.input_height),
			PushConstant::U32(self.input_width),
			PushConstant::U32(self.output_height),
			PushConstant::U32(self.output_width),
			PushConstant::U32(self.kernel_size),
			PushConstant::U32(self.stride),
			PushConstant::U32(self.padding),
		]
	}

	fn attributes(self) -> [OpAttribute; 3] {
		[
			OpAttribute::UnsignedInteger {
				name: "kernel_size".into(),
				value: u64::from(self.kernel_size),
			},
			OpAttribute::UnsignedInteger {
				name: "stride".into(),
				value: u64::from(self.stride),
			},
			OpAttribute::UnsignedInteger {
				name: "padding".into(),
				value: u64::from(self.padding),
			},
		]
	}
}

pub(in crate::ml) fn avg_pool_2d(
	input: &Matrix,
	kernel_size: usize,
	stride: usize,
	padding: usize,
) -> Result<Matrix> {
	let contract = crate::core::operation::ml::AVG_POOL_2D;
	let operation = contract.name();
	validate_f32_same_engine(operation, &[input])?;
	let geometry = Pool2dGeometry::resolve(input, kernel_size, stride, padding, operation)?;
	let output_shape = [
		geometry.batch as usize,
		geometry.channels as usize,
		geometry.output_height as usize,
		geometry.output_width as usize,
	];
	let output_count = output_shape
		.iter()
		.try_fold(1_usize, |product, extent| product.checked_mul(*extent))
		.ok_or_else(|| Error::invalid_argument(format!("{operation} output size overflows usize")))?;
	shader_u32(output_count, "output element count", operation)?;
	let output = Matrix::allocate(
		input.engine_handle(),
		output_shape.to_vec(),
		output_count,
		DType::F32,
	)?;
	let buffers = [
		BufferBinding::read(input.storage()),
		BufferBinding::write(output.storage()),
	];
	let push_constants = geometry.push_constants();
	let attributes = geometry.attributes();
	let inputs = [input];
	let outputs = [&output];
	input.engine_handle().record_semantic(
		ComputeDispatch {
			kernel: KernelId::MlAvgPool2dF32,
			buffers: &buffers,
			push_constants: &push_constants,
			workgroups: [
				geometry.output_height.div_ceil(16),
				geometry.output_width.div_ceil(16),
				geometry.batch * geometry.channels,
			],
		},
		SemanticDispatch {
			contract,
			inputs: &inputs,
			outputs: &outputs,
			attributes: &attributes,
		},
	)?;
	Ok(output)
}

pub(in crate::ml) fn avg_pool_2d_backward(
	input: &Matrix,
	output_gradient: &Matrix,
	kernel_size: usize,
	stride: usize,
	padding: usize,
) -> Result<Matrix> {
	let contract = crate::core::operation::ml::AVG_POOL_2D_BACKWARD;
	let operation = contract.name();
	validate_f32_same_engine(operation, &[input, output_gradient])?;
	let geometry = Pool2dGeometry::resolve(input, kernel_size, stride, padding, operation)?;
	let expected_shape = [
		geometry.batch as usize,
		geometry.channels as usize,
		geometry.output_height as usize,
		geometry.output_width as usize,
	];
	if output_gradient.shape() != expected_shape {
		return Err(Error::invalid_argument(format!(
			"{operation} output gradient must have shape {expected_shape:?}; found {:?}",
			output_gradient.shape()
		)));
	}
	let input_gradient = Matrix::allocate(
		input.engine_handle(),
		input.shape().to_vec(),
		input.num_elements(),
		DType::F32,
	)?;
	let buffers = [
		BufferBinding::read(output_gradient.storage()),
		BufferBinding::write(input_gradient.storage()),
	];
	let push_constants = geometry.push_constants();
	let attributes = geometry.attributes();
	let inputs = [input, output_gradient];
	let outputs = [&input_gradient];
	input.engine_handle().record_semantic(
		ComputeDispatch {
			kernel: KernelId::MlAvgPool2dBackwardF32,
			buffers: &buffers,
			push_constants: &push_constants,
			workgroups: [
				geometry.input_height.div_ceil(16),
				geometry.input_width.div_ceil(16),
				geometry.batch * geometry.channels,
			],
		},
		SemanticDispatch {
			contract,
			inputs: &inputs,
			outputs: &outputs,
			attributes: &attributes,
		},
	)?;
	Ok(input_gradient)
}

pub(in crate::ml) struct MaxPool2dOutput {
	pub(in crate::ml) output: Matrix,
	pub(in crate::ml) indices: Matrix,
}

pub(in crate::ml) fn max_pool_2d(
	input: &Matrix,
	kernel_size: usize,
	stride: usize,
	padding: usize,
) -> Result<MaxPool2dOutput> {
	let contract = crate::core::operation::ml::MAX_POOL_2D;
	let operation = contract.name();
	validate_f32_same_engine(operation, &[input])?;
	let geometry = Pool2dGeometry::resolve(input, kernel_size, stride, padding, operation)?;
	let output_shape = [
		geometry.batch as usize,
		geometry.channels as usize,
		geometry.output_height as usize,
		geometry.output_width as usize,
	];
	let output_count = output_shape
		.iter()
		.try_fold(1_usize, |product, extent| product.checked_mul(*extent))
		.ok_or_else(|| Error::invalid_argument(format!("{operation} output size overflows usize")))?;
	shader_u32(output_count, "output element count", operation)?;
	let output = Matrix::allocate(
		input.engine_handle(),
		output_shape.to_vec(),
		output_count,
		DType::F32,
	)?;
	let indices = Matrix::allocate(
		input.engine_handle(),
		output_shape.to_vec(),
		output_count,
		DType::U32,
	)?;
	let buffers = [
		BufferBinding::read(input.storage()),
		BufferBinding::write(output.storage()),
		BufferBinding::write(indices.storage()),
	];
	let push_constants = geometry.push_constants();
	let attributes = geometry.attributes();
	let inputs = [input];
	let outputs = [&output, &indices];
	input.engine_handle().record_semantic(
		ComputeDispatch {
			kernel: KernelId::MlMaxPool2dF32,
			buffers: &buffers,
			push_constants: &push_constants,
			workgroups: [
				geometry.output_height.div_ceil(16),
				geometry.output_width.div_ceil(16),
				geometry.batch * geometry.channels,
			],
		},
		SemanticDispatch {
			contract,
			inputs: &inputs,
			outputs: &outputs,
			attributes: &attributes,
		},
	)?;
	Ok(MaxPool2dOutput { output, indices })
}

pub(in crate::ml) fn max_pool_2d_backward(
	input: &Matrix,
	indices: &Matrix,
	output_gradient: &Matrix,
	kernel_size: usize,
	stride: usize,
	padding: usize,
) -> Result<Matrix> {
	let contract = crate::core::operation::ml::MAX_POOL_2D_BACKWARD;
	let operation = contract.name();
	validate_f32_same_engine(operation, &[input, output_gradient])?;
	if indices.dtype() != DType::U32 {
		return Err(Error::invalid_argument(format!(
			"{operation} indices must use U32; found {}",
			indices.dtype().token()
		)));
	}
	if !input.engine_handle().same_as(indices.engine_handle()) {
		return Err(Error::invalid_argument(format!(
			"{operation} inputs must belong to the same engine"
		)));
	}
	let geometry = Pool2dGeometry::resolve(input, kernel_size, stride, padding, operation)?;
	let expected_shape = [
		geometry.batch as usize,
		geometry.channels as usize,
		geometry.output_height as usize,
		geometry.output_width as usize,
	];
	if indices.shape() != expected_shape || output_gradient.shape() != expected_shape {
		return Err(Error::invalid_argument(format!(
			"{operation} indices and output gradient must have shape {expected_shape:?}; found {:?} and {:?}",
			indices.shape(),
			output_gradient.shape()
		)));
	}
	let input_gradient = Matrix::allocate(
		input.engine_handle(),
		input.shape().to_vec(),
		input.num_elements(),
		DType::F32,
	)?;
	let buffers = [
		BufferBinding::read(indices.storage()),
		BufferBinding::read(output_gradient.storage()),
		BufferBinding::write(input_gradient.storage()),
	];
	let push_constants = geometry.push_constants();
	let attributes = geometry.attributes();
	let inputs = [input, indices, output_gradient];
	let outputs = [&input_gradient];
	input.engine_handle().record_semantic(
		ComputeDispatch {
			kernel: KernelId::MlMaxPool2dBackwardF32,
			buffers: &buffers,
			push_constants: &push_constants,
			workgroups: [
				geometry.input_height.div_ceil(16),
				geometry.input_width.div_ceil(16),
				geometry.batch * geometry.channels,
			],
		},
		SemanticDispatch {
			contract,
			inputs: &inputs,
			outputs: &outputs,
			attributes: &attributes,
		},
	)?;
	Ok(input_gradient)
}

#[derive(Clone, Copy)]
struct AdaptivePool2dGeometry {
	batch: u32,
	channels: u32,
	input_height: u32,
	input_width: u32,
	output_height: u32,
	output_width: u32,
}

impl AdaptivePool2dGeometry {
	fn resolve(
		input: &Matrix,
		output_height: usize,
		output_width: usize,
		operation: &'static str,
	) -> Result<Self> {
		let [batch, channels, input_height, input_width] = input.shape() else {
			return Err(Error::invalid_argument(format!(
				"{operation} requires NCHW rank-four input; found {:?}",
				input.shape()
			)));
		};
		if [
			*batch,
			*channels,
			*input_height,
			*input_width,
			output_height,
			output_width,
		]
		.contains(&0)
		{
			return Err(Error::invalid_argument(format!(
				"{operation} requires nonzero input and output extents"
			)));
		}
		shader_u32(input.num_elements(), "input element count", operation)?;
		let output_count = batch
			.checked_mul(*channels)
			.and_then(|value| value.checked_mul(output_height))
			.and_then(|value| value.checked_mul(output_width))
			.ok_or_else(|| Error::invalid_argument(format!("{operation} output size overflows usize")))?;
		shader_u32(output_count, "output element count", operation)?;
		for (label, output_extent, input_extent) in [
			("height", output_height, *input_height),
			("width", output_width, *input_width),
		] {
			output_extent
				.checked_mul(input_extent)
				.and_then(|value| value.checked_add(output_extent - 1))
				.and_then(|value| u32::try_from(value).ok())
				.ok_or_else(|| {
					Error::invalid_argument(format!(
						"{operation} adaptive {label} geometry exceeds the u32 shader ABI"
					))
				})?;
		}
		let batch = shader_u32(*batch, "batch size", operation)?;
		let channels = shader_u32(*channels, "channel count", operation)?;
		batch.checked_mul(channels).ok_or_else(|| {
			Error::invalid_argument(format!("{operation} batch-channel count exceeds u32"))
		})?;
		Ok(Self {
			batch,
			channels,
			input_height: shader_u32(*input_height, "input height", operation)?,
			input_width: shader_u32(*input_width, "input width", operation)?,
			output_height: shader_u32(output_height, "output height", operation)?,
			output_width: shader_u32(output_width, "output width", operation)?,
		})
	}

	fn push_constants(self) -> [PushConstant; 6] {
		[
			PushConstant::U32(self.batch),
			PushConstant::U32(self.channels),
			PushConstant::U32(self.input_height),
			PushConstant::U32(self.input_width),
			PushConstant::U32(self.output_height),
			PushConstant::U32(self.output_width),
		]
	}

	fn attributes(self) -> [OpAttribute; 2] {
		[
			OpAttribute::UnsignedInteger {
				name: "output_height".into(),
				value: u64::from(self.output_height),
			},
			OpAttribute::UnsignedInteger {
				name: "output_width".into(),
				value: u64::from(self.output_width),
			},
		]
	}
}

pub(in crate::ml) fn adaptive_avg_pool_2d(
	input: &Matrix,
	output_height: usize,
	output_width: usize,
) -> Result<Matrix> {
	let contract = crate::core::operation::ml::ADAPTIVE_AVG_POOL_2D;
	let operation = contract.name();
	validate_f32_same_engine(operation, &[input])?;
	let geometry = AdaptivePool2dGeometry::resolve(input, output_height, output_width, operation)?;
	let output_shape = [
		geometry.batch as usize,
		geometry.channels as usize,
		output_height,
		output_width,
	];
	let output_count = output_shape.iter().product();
	let output = Matrix::allocate(
		input.engine_handle(),
		output_shape.to_vec(),
		output_count,
		DType::F32,
	)?;
	let buffers = [
		BufferBinding::read(input.storage()),
		BufferBinding::write(output.storage()),
	];
	let push_constants = geometry.push_constants();
	let attributes = geometry.attributes();
	input.engine_handle().record_semantic(
		ComputeDispatch {
			kernel: KernelId::MlAdaptiveAvgPool2dF32,
			buffers: &buffers,
			push_constants: &push_constants,
			workgroups: [
				geometry.output_height.div_ceil(16),
				geometry.output_width.div_ceil(16),
				geometry.batch * geometry.channels,
			],
		},
		SemanticDispatch {
			contract,
			inputs: &[input],
			outputs: &[&output],
			attributes: &attributes,
		},
	)?;
	Ok(output)
}

pub(in crate::ml) fn adaptive_avg_pool_2d_backward(
	input: &Matrix,
	output_gradient: &Matrix,
	output_height: usize,
	output_width: usize,
) -> Result<Matrix> {
	let contract = crate::core::operation::ml::ADAPTIVE_AVG_POOL_2D_BACKWARD;
	let operation = contract.name();
	validate_f32_same_engine(operation, &[input, output_gradient])?;
	let geometry = AdaptivePool2dGeometry::resolve(input, output_height, output_width, operation)?;
	let expected_shape = [
		geometry.batch as usize,
		geometry.channels as usize,
		output_height,
		output_width,
	];
	if output_gradient.shape() != expected_shape {
		return Err(Error::invalid_argument(format!(
			"{operation} output gradient must have shape {expected_shape:?}; found {:?}",
			output_gradient.shape()
		)));
	}
	let input_gradient = Matrix::allocate(
		input.engine_handle(),
		input.shape().to_vec(),
		input.num_elements(),
		DType::F32,
	)?;
	let buffers = [
		BufferBinding::read(output_gradient.storage()),
		BufferBinding::write(input_gradient.storage()),
	];
	let push_constants = geometry.push_constants();
	let attributes = geometry.attributes();
	input.engine_handle().record_semantic(
		ComputeDispatch {
			kernel: KernelId::MlAdaptiveAvgPool2dBackwardF32,
			buffers: &buffers,
			push_constants: &push_constants,
			workgroups: [
				geometry.input_height.div_ceil(16),
				geometry.input_width.div_ceil(16),
				geometry.batch * geometry.channels,
			],
		},
		SemanticDispatch {
			contract,
			inputs: &[input, output_gradient],
			outputs: &[&input_gradient],
			attributes: &attributes,
		},
	)?;
	Ok(input_gradient)
}
