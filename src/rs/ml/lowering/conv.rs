//! Private lowering for ML convolution operations.

use crate::{
	DType, Error, Matrix, OpAttribute, Result,
	runtime::{BufferBinding, ComputeDispatch, KernelId, PushConstant, SemanticDispatch},
};

use super::common::{record_semantic, shader_u32, validate_f32_same_engine};

#[derive(Clone, Copy)]
struct Conv1dGeometry {
	batch_size: u32,
	input_channels: u32,
	output_channels: u32,
	input_length: u32,
	kernel_size: u32,
	stride: u32,
	padding: u32,
	dilation: u32,
	output_length: u32,
	columns_count: u32,
	output_count: u32,
	weight_count: u32,
	parameter_count: u32,
}

impl Conv1dGeometry {
	fn resolve(
		input: &Matrix,
		weight: &Matrix,
		bias: Option<&Matrix>,
		stride: usize,
		padding: usize,
		dilation: usize,
		operation: &'static str,
	) -> Result<Self> {
		let [batch_size, input_channels, input_length] = input.shape() else {
			return Err(Error::invalid_argument(format!(
				"{operation} input must have NCL rank three; found {:?}",
				input.shape()
			)));
		};
		let [output_channels, weight_input_channels, kernel_size] = weight.shape() else {
			return Err(Error::invalid_argument(format!(
				"{operation} weight must have OIK rank three; found {:?}",
				weight.shape()
			)));
		};
		if *batch_size == 0
			|| *input_channels == 0
			|| *output_channels == 0
			|| *input_length == 0
			|| *kernel_size == 0
			|| *weight_input_channels != *input_channels
			|| stride == 0
			|| dilation == 0
		{
			return Err(Error::invalid_argument(format!(
				"{operation} requires nonzero NCL input, matching nonzero OIK weight, stride, and dilation"
			)));
		}
		if let Some(bias) = bias
			&& bias.shape() != [*output_channels]
		{
			return Err(Error::invalid_argument(format!(
				"{operation} bias must have shape [{output_channels}]; found {:?}",
				bias.shape()
			)));
		}
		let effective_kernel = kernel_size
			.saturating_sub(1)
			.checked_mul(dilation)
			.and_then(|extent| extent.checked_add(1))
			.ok_or_else(|| {
				Error::invalid_argument(format!("{operation} effective kernel overflows usize"))
			})?;
		let doubled_padding = padding.checked_mul(2).ok_or_else(|| {
			Error::invalid_argument(format!("{operation} doubled padding overflows usize"))
		})?;
		let padded_length = input_length.checked_add(doubled_padding).ok_or_else(|| {
			Error::invalid_argument(format!("{operation} padded length overflows usize"))
		})?;
		if padded_length < effective_kernel {
			return Err(Error::invalid_argument(format!(
				"{operation} effective kernel does not fit the padded input"
			)));
		}
		let output_length = (padded_length - effective_kernel) / stride + 1;
		let rows = batch_size.checked_mul(output_length).ok_or_else(|| {
			Error::invalid_argument(format!("{operation} unfolded row count overflows usize"))
		})?;
		let inner_size = input_channels.checked_mul(*kernel_size).ok_or_else(|| {
			Error::invalid_argument(format!("{operation} unfolded width overflows usize"))
		})?;
		let columns_count = rows.checked_mul(inner_size).ok_or_else(|| {
			Error::invalid_argument(format!("{operation} column storage size overflows usize"))
		})?;
		let output_count = rows.checked_mul(*output_channels).ok_or_else(|| {
			Error::invalid_argument(format!("{operation} output size overflows usize"))
		})?;
		let weight_count = output_channels
			.checked_mul(*input_channels)
			.and_then(|count| count.checked_mul(*kernel_size))
			.ok_or_else(|| {
				Error::invalid_argument(format!("{operation} weight size overflows usize"))
			})?;
		if weight_count != weight.num_elements() {
			return Err(Error::internal(format!(
				"{operation} validated weight shape disagrees with its storage extent"
			)));
		}
		let parameter_count = weight_count.checked_add(*output_channels).ok_or_else(|| {
			Error::invalid_argument(format!("{operation} parameter count overflows usize"))
		})?;
		let last_coordinate = output_length
			.saturating_sub(1)
			.checked_mul(stride)
			.and_then(|coordinate| {
				kernel_size
					.saturating_sub(1)
					.checked_mul(dilation)
					.and_then(|kernel| coordinate.checked_add(kernel))
			})
			.ok_or_else(|| {
				Error::invalid_argument(format!("{operation} shader coordinate overflows usize"))
			})?;
		for (label, value) in [
			("input length", *input_length),
			("kernel size", *kernel_size),
			("stride", stride),
			("padding", padding),
			("dilation", dilation),
			("last input coordinate", last_coordinate),
		] {
			if value > i32::MAX as usize {
				return Err(Error::invalid_argument(format!(
					"{operation} {label} exceeds the signed shader coordinate ABI"
				)));
			}
		}
		Ok(Self {
			batch_size: shader_u32(*batch_size, "batch size", operation)?,
			input_channels: shader_u32(*input_channels, "input channel count", operation)?,
			output_channels: shader_u32(*output_channels, "output channel count", operation)?,
			input_length: shader_u32(*input_length, "input length", operation)?,
			kernel_size: shader_u32(*kernel_size, "kernel size", operation)?,
			stride: shader_u32(stride, "stride", operation)?,
			padding: shader_u32(padding, "padding", operation)?,
			dilation: shader_u32(dilation, "dilation", operation)?,
			output_length: shader_u32(output_length, "output length", operation)?,
			columns_count: shader_u32(columns_count, "column element count", operation)?,
			output_count: shader_u32(output_count, "output element count", operation)?,
			weight_count: shader_u32(weight_count, "weight element count", operation)?,
			parameter_count: shader_u32(parameter_count, "parameter count", operation)?,
		})
	}

	fn dimensions(self) -> [PushConstant; 9] {
		[
			PushConstant::U32(self.batch_size),
			PushConstant::U32(self.input_channels),
			PushConstant::U32(self.output_channels),
			PushConstant::U32(self.input_length),
			PushConstant::U32(self.kernel_size),
			PushConstant::U32(self.stride),
			PushConstant::U32(self.padding),
			PushConstant::U32(self.dilation),
			PushConstant::U32(self.output_length),
		]
	}

	fn attributes(self) -> [OpAttribute; 3] {
		[
			OpAttribute::UnsignedInteger {
				name: "stride".into(),
				value: u64::from(self.stride),
			},
			OpAttribute::UnsignedInteger {
				name: "padding".into(),
				value: u64::from(self.padding),
			},
			OpAttribute::UnsignedInteger {
				name: "dilation".into(),
				value: u64::from(self.dilation),
			},
		]
	}
}

pub(in crate::ml) fn conv_1d(
	input: &Matrix,
	weight: &Matrix,
	bias: &Matrix,
	stride: usize,
	padding: usize,
	dilation: usize,
) -> Result<Matrix> {
	let contract = crate::core::operation::ml::CONV_1D;
	let operation = contract.name();
	validate_f32_same_engine(operation, &[input, weight, bias])?;
	let geometry = Conv1dGeometry::resolve(
		input,
		weight,
		Some(bias),
		stride,
		padding,
		dilation,
		operation,
	)?;
	let rows = geometry.batch_size * geometry.output_length;
	let inner_size = geometry.input_channels * geometry.kernel_size;
	let columns = Matrix::allocate(
		input.engine_handle(),
		vec![rows as usize, inner_size as usize],
		geometry.columns_count as usize,
		DType::F32,
	)?;
	let projected = Matrix::allocate(
		input.engine_handle(),
		vec![rows as usize, geometry.output_channels as usize],
		geometry.output_count as usize,
		DType::F32,
	)?;
	let output = Matrix::allocate(
		input.engine_handle(),
		vec![
			geometry.batch_size as usize,
			geometry.output_channels as usize,
			geometry.output_length as usize,
		],
		geometry.output_count as usize,
		DType::F32,
	)?;

	let unfold_buffers = [
		BufferBinding::read(input.storage()),
		BufferBinding::write(columns.storage()),
	];
	let unfold_constants = [
		PushConstant::U32(geometry.batch_size),
		PushConstant::U32(geometry.input_channels),
		PushConstant::U32(geometry.input_length),
		PushConstant::U32(geometry.kernel_size),
		PushConstant::U32(geometry.stride),
		PushConstant::U32(geometry.padding),
		PushConstant::U32(geometry.dilation),
		PushConstant::U32(geometry.output_length),
	];
	let gemm_buffers = [
		BufferBinding::read(columns.storage()),
		BufferBinding::read(weight.storage()),
		BufferBinding::write(projected.storage()),
	];
	let gemm_constants = [
		PushConstant::U32(rows),
		PushConstant::U32(geometry.output_channels),
		PushConstant::U32(inner_size),
	];
	let bias_buffers = [
		BufferBinding::read_write(projected.storage()),
		BufferBinding::read(bias.storage()),
	];
	let bias_constants = [
		PushConstant::U32(rows),
		PushConstant::U32(geometry.output_channels),
	];
	let transpose_buffers = [
		BufferBinding::read(projected.storage()),
		BufferBinding::write(output.storage()),
	];
	let transpose_constants = [
		PushConstant::U32(geometry.batch_size),
		PushConstant::U32(geometry.output_length),
		PushConstant::U32(geometry.output_channels),
	];
	let dispatches = [
		ComputeDispatch {
			kernel: KernelId::MlConv1dIm2ColF32,
			buffers: &unfold_buffers,
			push_constants: &unfold_constants,
			workgroups: KernelId::MlConv1dIm2ColF32.linear_workgroups(geometry.columns_count),
		},
		ComputeDispatch {
			kernel: KernelId::MatrixMatMulNtTiledF32,
			buffers: &gemm_buffers,
			push_constants: &gemm_constants,
			workgroups: KernelId::MatrixMatMulNtTiledF32
				.output_workgroups(rows, geometry.output_channels),
		},
		ComputeDispatch {
			kernel: KernelId::MlConv1dBiasAddF32,
			buffers: &bias_buffers,
			push_constants: &bias_constants,
			workgroups: KernelId::MlConv1dBiasAddF32.linear_workgroups(geometry.output_count),
		},
		ComputeDispatch {
			kernel: KernelId::MlConv1dTransposeF32,
			buffers: &transpose_buffers,
			push_constants: &transpose_constants,
			workgroups: [
				geometry.output_channels.div_ceil(32),
				geometry.output_length.div_ceil(32),
				geometry.batch_size,
			],
		},
	];
	let attributes = geometry.attributes();
	let inputs = [input, weight, bias];
	let outputs = [&output];
	input.engine_handle().record_split_semantic(
		&dispatches,
		SemanticDispatch {
			contract,
			inputs: &inputs,
			outputs: &outputs,
			attributes: &attributes,
		},
	)?;
	Ok(output)
}

pub(in crate::ml) fn conv_1d_backward(
	input: &Matrix,
	weight: &Matrix,
	output_gradient: &Matrix,
	stride: usize,
	padding: usize,
	dilation: usize,
) -> Result<(Matrix, Matrix, Matrix)> {
	let contract = crate::core::operation::ml::CONV_1D_BACKWARD;
	let operation = contract.name();
	validate_f32_same_engine(operation, &[input, weight, output_gradient])?;
	let geometry =
		Conv1dGeometry::resolve(input, weight, None, stride, padding, dilation, operation)?;
	let expected_output_shape = [
		geometry.batch_size as usize,
		geometry.output_channels as usize,
		geometry.output_length as usize,
	];
	if output_gradient.shape() != expected_output_shape {
		return Err(Error::invalid_argument(format!(
			"{operation} output gradient must have shape {expected_output_shape:?}; found {:?}",
			output_gradient.shape()
		)));
	}
	let input_gradient = Matrix::allocate(
		input.engine_handle(),
		input.shape().to_vec(),
		input.num_elements(),
		DType::F32,
	)?;
	let weight_gradient = Matrix::allocate(
		input.engine_handle(),
		weight.shape().to_vec(),
		geometry.weight_count as usize,
		DType::F32,
	)?;
	let bias_gradient = Matrix::allocate(
		input.engine_handle(),
		vec![geometry.output_channels as usize],
		geometry.output_channels as usize,
		DType::F32,
	)?;
	let input_buffers = [
		BufferBinding::read(output_gradient.storage()),
		BufferBinding::read(weight.storage()),
		BufferBinding::write(input_gradient.storage()),
	];
	let input_constants = geometry.dimensions();
	let parameter_buffers = [
		BufferBinding::read(input.storage()),
		BufferBinding::read(output_gradient.storage()),
		BufferBinding::write(weight_gradient.storage()),
		BufferBinding::write(bias_gradient.storage()),
	];
	let mut parameter_constants = input_constants.to_vec();
	parameter_constants.push(PushConstant::U32(geometry.parameter_count));
	let input_kernel = KernelId::MlConv1dBackwardF32;
	let parameter_kernel = KernelId::MlConv1dParameterBackwardF32;
	let dispatches = [
		ComputeDispatch {
			kernel: input_kernel,
			buffers: &input_buffers,
			push_constants: &input_constants,
			workgroups: input_kernel.linear_workgroups(shader_u32(
				input.num_elements(),
				"input element count",
				operation,
			)?),
		},
		ComputeDispatch {
			kernel: parameter_kernel,
			buffers: &parameter_buffers,
			push_constants: &parameter_constants,
			workgroups: parameter_kernel.linear_workgroups(geometry.parameter_count),
		},
	];
	let attributes = geometry.attributes();
	let inputs = [input, weight, output_gradient];
	let outputs = [&input_gradient, &weight_gradient, &bias_gradient];
	input.engine_handle().record_split_semantic(
		&dispatches,
		SemanticDispatch {
			contract,
			inputs: &inputs,
			outputs: &outputs,
			attributes: &attributes,
		},
	)?;
	Ok((input_gradient, weight_gradient, bias_gradient))
}

#[derive(Clone, Copy)]
struct ConvTranspose1dGeometry {
	batch_size: u32,
	input_channels: u32,
	output_channels: u32,
	input_length: u32,
	kernel_size: u32,
	stride: u32,
	padding: u32,
	dilation: u32,
	output_length: u32,
	output_count: u32,
	weight_count: u32,
}

impl ConvTranspose1dGeometry {
	fn resolve(
		input: &Matrix,
		weight: &Matrix,
		stride: usize,
		padding: usize,
		dilation: usize,
		operation: &'static str,
	) -> Result<Self> {
		let [batch_size, input_channels, input_length] = input.shape() else {
			return Err(Error::invalid_argument(format!(
				"{operation} input must have NCL rank three; found {:?}",
				input.shape()
			)));
		};
		let [weight_input_channels, output_channels, kernel_size] = weight.shape() else {
			return Err(Error::invalid_argument(format!(
				"{operation} weight must have IOK rank three; found {:?}",
				weight.shape()
			)));
		};
		if *batch_size == 0
			|| *input_channels == 0
			|| *output_channels == 0
			|| *input_length == 0
			|| *kernel_size == 0
			|| *weight_input_channels != *input_channels
			|| stride == 0
			|| dilation == 0
		{
			return Err(Error::invalid_argument(format!(
				"{operation} requires nonzero NCL input, matching nonzero IOK weight, stride, and dilation"
			)));
		}
		let effective_kernel = kernel_size
			.saturating_sub(1)
			.checked_mul(dilation)
			.and_then(|extent| extent.checked_add(1))
			.ok_or_else(|| {
				Error::invalid_argument(format!("{operation} effective kernel overflows usize"))
			})?;
		let expanded_length = input_length
			.saturating_sub(1)
			.checked_mul(stride)
			.and_then(|extent| extent.checked_add(effective_kernel))
			.ok_or_else(|| {
				Error::invalid_argument(format!("{operation} expanded length overflows usize"))
			})?;
		let doubled_padding = padding.checked_mul(2).ok_or_else(|| {
			Error::invalid_argument(format!("{operation} doubled padding overflows usize"))
		})?;
		if expanded_length <= doubled_padding {
			return Err(Error::invalid_argument(format!(
				"{operation} padding removes the complete transposed output"
			)));
		}
		let output_length = expanded_length - doubled_padding;
		let output_count = batch_size
			.checked_mul(*output_channels)
			.and_then(|count| count.checked_mul(output_length))
			.ok_or_else(|| {
				Error::invalid_argument(format!("{operation} output size overflows usize"))
			})?;
		let weight_count = input_channels
			.checked_mul(*output_channels)
			.and_then(|count| count.checked_mul(*kernel_size))
			.ok_or_else(|| {
				Error::invalid_argument(format!("{operation} weight size overflows usize"))
			})?;
		if weight_count != weight.num_elements() {
			return Err(Error::internal(format!(
				"{operation} validated weight shape disagrees with its storage extent"
			)));
		}
		for (label, value) in [
			("input length", *input_length),
			("output length", output_length),
			("kernel size", *kernel_size),
			("stride", stride),
			("padding", padding),
			("dilation", dilation),
		] {
			if value > i32::MAX as usize {
				return Err(Error::invalid_argument(format!(
					"{operation} {label} exceeds the signed shader coordinate ABI"
				)));
			}
		}
		Ok(Self {
			batch_size: shader_u32(*batch_size, "batch size", operation)?,
			input_channels: shader_u32(*input_channels, "input channel count", operation)?,
			output_channels: shader_u32(*output_channels, "output channel count", operation)?,
			input_length: shader_u32(*input_length, "input length", operation)?,
			kernel_size: shader_u32(*kernel_size, "kernel size", operation)?,
			stride: shader_u32(stride, "stride", operation)?,
			padding: shader_u32(padding, "padding", operation)?,
			dilation: shader_u32(dilation, "dilation", operation)?,
			output_length: shader_u32(output_length, "output length", operation)?,
			output_count: shader_u32(output_count, "output element count", operation)?,
			weight_count: shader_u32(weight_count, "weight element count", operation)?,
		})
	}

	fn dimensions(self) -> [PushConstant; 9] {
		[
			PushConstant::U32(self.batch_size),
			PushConstant::U32(self.input_channels),
			PushConstant::U32(self.output_channels),
			PushConstant::U32(self.input_length),
			PushConstant::U32(self.kernel_size),
			PushConstant::U32(self.stride),
			PushConstant::U32(self.padding),
			PushConstant::U32(self.dilation),
			PushConstant::U32(self.output_length),
		]
	}

	fn attributes(self) -> [OpAttribute; 3] {
		[
			OpAttribute::UnsignedInteger {
				name: "stride".into(),
				value: u64::from(self.stride),
			},
			OpAttribute::UnsignedInteger {
				name: "padding".into(),
				value: u64::from(self.padding),
			},
			OpAttribute::UnsignedInteger {
				name: "dilation".into(),
				value: u64::from(self.dilation),
			},
		]
	}
}

pub(in crate::ml) fn conv_transpose_1d(
	input: &Matrix,
	weight: &Matrix,
	stride: usize,
	padding: usize,
	dilation: usize,
) -> Result<Matrix> {
	let contract = crate::core::operation::ml::CONV_TRANSPOSE_1D;
	let operation = contract.name();
	validate_f32_same_engine(operation, &[input, weight])?;
	let geometry =
		ConvTranspose1dGeometry::resolve(input, weight, stride, padding, dilation, operation)?;
	let output = Matrix::allocate(
		input.engine_handle(),
		vec![
			geometry.batch_size as usize,
			geometry.output_channels as usize,
			geometry.output_length as usize,
		],
		geometry.output_count as usize,
		DType::F32,
	)?;
	let buffers = [
		BufferBinding::read(input.storage()),
		BufferBinding::read(weight.storage()),
		BufferBinding::write(output.storage()),
	];
	let dimensions = geometry.dimensions();
	let attributes = geometry.attributes();
	let kernel = KernelId::MlConvTranspose1dF32;
	record_semantic(
		&[input, weight],
		&[&output],
		&attributes,
		kernel,
		&buffers,
		&dimensions,
		kernel.linear_workgroups(geometry.output_count),
	)?;
	Ok(output)
}

pub(in crate::ml) fn conv_transpose_1d_backward(
	input: &Matrix,
	weight: &Matrix,
	output_gradient: &Matrix,
	stride: usize,
	padding: usize,
	dilation: usize,
) -> Result<(Matrix, Matrix)> {
	let contract = crate::core::operation::ml::CONV_TRANSPOSE_1D_BACKWARD;
	let operation = contract.name();
	validate_f32_same_engine(operation, &[input, weight, output_gradient])?;
	let transposed =
		ConvTranspose1dGeometry::resolve(input, weight, stride, padding, dilation, operation)?;
	let expected_output_shape = [
		transposed.batch_size as usize,
		transposed.output_channels as usize,
		transposed.output_length as usize,
	];
	if output_gradient.shape() != expected_output_shape {
		return Err(Error::invalid_argument(format!(
			"{operation} output gradient must have shape {expected_output_shape:?}; found {:?}",
			output_gradient.shape()
		)));
	}

	// The input adjoint is an ordinary bias-free Conv1d. Resolve that geometry
	// independently so the donor's exact inverse shape relation is checked.
	let forward = Conv1dGeometry::resolve(
		output_gradient,
		weight,
		None,
		stride,
		padding,
		dilation,
		operation,
	)?;
	if forward.batch_size != transposed.batch_size
		|| forward.output_channels != transposed.input_channels
		|| forward.output_length != transposed.input_length
	{
		return Err(Error::internal(format!(
			"{operation} forward and adjoint geometry disagree"
		)));
	}
	let rows = forward.batch_size * forward.output_length;
	let inner_size = forward.input_channels * forward.kernel_size;
	let columns = Matrix::allocate(
		input.engine_handle(),
		vec![rows as usize, inner_size as usize],
		forward.columns_count as usize,
		DType::F32,
	)?;
	let projected = Matrix::allocate(
		input.engine_handle(),
		vec![rows as usize, forward.output_channels as usize],
		input.num_elements(),
		DType::F32,
	)?;
	let input_gradient = Matrix::allocate(
		input.engine_handle(),
		input.shape().to_vec(),
		input.num_elements(),
		DType::F32,
	)?;
	let weight_gradient = Matrix::allocate(
		input.engine_handle(),
		weight.shape().to_vec(),
		transposed.weight_count as usize,
		DType::F32,
	)?;
	let discarded_bias_gradient = Matrix::allocate(
		input.engine_handle(),
		vec![forward.output_channels as usize],
		forward.output_channels as usize,
		DType::F32,
	)?;

	let unfold_buffers = [
		BufferBinding::read(output_gradient.storage()),
		BufferBinding::write(columns.storage()),
	];
	let unfold_constants = [
		PushConstant::U32(forward.batch_size),
		PushConstant::U32(forward.input_channels),
		PushConstant::U32(forward.input_length),
		PushConstant::U32(forward.kernel_size),
		PushConstant::U32(forward.stride),
		PushConstant::U32(forward.padding),
		PushConstant::U32(forward.dilation),
		PushConstant::U32(forward.output_length),
	];
	let gemm_buffers = [
		BufferBinding::read(columns.storage()),
		BufferBinding::read(weight.storage()),
		BufferBinding::write(projected.storage()),
	];
	let gemm_constants = [
		PushConstant::U32(rows),
		PushConstant::U32(forward.output_channels),
		PushConstant::U32(inner_size),
	];
	let transpose_buffers = [
		BufferBinding::read(projected.storage()),
		BufferBinding::write(input_gradient.storage()),
	];
	let transpose_constants = [
		PushConstant::U32(forward.batch_size),
		PushConstant::U32(forward.output_length),
		PushConstant::U32(forward.output_channels),
	];
	let parameter_buffers = [
		BufferBinding::read(output_gradient.storage()),
		BufferBinding::read(input.storage()),
		BufferBinding::write(weight_gradient.storage()),
		BufferBinding::write(discarded_bias_gradient.storage()),
	];
	let mut parameter_constants = forward.dimensions().to_vec();
	parameter_constants.push(PushConstant::U32(forward.parameter_count));
	let dispatches = [
		ComputeDispatch {
			kernel: KernelId::MlConvTranspose1dBackwardF32,
			buffers: &unfold_buffers,
			push_constants: &unfold_constants,
			workgroups: KernelId::MlConvTranspose1dBackwardF32
				.linear_workgroups(forward.columns_count),
		},
		ComputeDispatch {
			kernel: KernelId::MatrixMatMulNtTiledF32,
			buffers: &gemm_buffers,
			push_constants: &gemm_constants,
			workgroups: KernelId::MatrixMatMulNtTiledF32
				.output_workgroups(rows, forward.output_channels),
		},
		ComputeDispatch {
			kernel: KernelId::MlConv1dTransposeF32,
			buffers: &transpose_buffers,
			push_constants: &transpose_constants,
			workgroups: [
				forward.output_channels.div_ceil(32),
				forward.output_length.div_ceil(32),
				forward.batch_size,
			],
		},
		ComputeDispatch {
			kernel: KernelId::MlConv1dParameterBackwardF32,
			buffers: &parameter_buffers,
			push_constants: &parameter_constants,
			workgroups: KernelId::MlConv1dParameterBackwardF32
				.linear_workgroups(forward.parameter_count),
		},
	];
	let attributes = transposed.attributes();
	let inputs = [input, weight, output_gradient];
	let outputs = [&input_gradient, &weight_gradient];
	input.engine_handle().record_split_semantic(
		&dispatches,
		SemanticDispatch {
			contract,
			inputs: &inputs,
			outputs: &outputs,
			attributes: &attributes,
		},
	)?;
	Ok((input_gradient, weight_gradient))
}

#[derive(Clone, Copy)]
struct ConvTranspose2dGeometry {
	batch_size: u32,
	input_channels: u32,
	output_channels: u32,
	input_height: u32,
	input_width: u32,
	kernel_size: u32,
	stride: u32,
	padding: u32,
	output_height: u32,
	output_width: u32,
	output_count: u32,
	weight_count: u32,
	parameter_count: u32,
}

impl ConvTranspose2dGeometry {
	fn resolve(
		input: &Matrix,
		weight: &Matrix,
		bias: Option<&Matrix>,
		stride: usize,
		padding: usize,
		operation: &'static str,
	) -> Result<Self> {
		let [batch_size, input_channels, input_height, input_width] = input.shape() else {
			return Err(Error::invalid_argument(format!(
				"{operation} input must have NCHW rank four; found {:?}",
				input.shape()
			)));
		};
		let [
			weight_input_channels,
			output_channels,
			kernel_height,
			kernel_width,
		] = weight.shape()
		else {
			return Err(Error::invalid_argument(format!(
				"{operation} weight must have IOKK rank four; found {:?}",
				weight.shape()
			)));
		};
		if *batch_size == 0
			|| *input_channels == 0
			|| *output_channels == 0
			|| *input_height == 0
			|| *input_width == 0
			|| *kernel_height == 0
			|| kernel_height != kernel_width
			|| *weight_input_channels != *input_channels
			|| stride == 0
		{
			return Err(Error::invalid_argument(format!(
				"{operation} requires nonzero NCHW input, matching nonzero square IOKK weight, and nonzero stride"
			)));
		}
		if let Some(bias) = bias
			&& bias.shape() != [*output_channels]
		{
			return Err(Error::invalid_argument(format!(
				"{operation} bias must have shape [{output_channels}]; found {:?}",
				bias.shape()
			)));
		}
		let expanded_height = input_height
			.saturating_sub(1)
			.checked_mul(stride)
			.and_then(|extent| extent.checked_add(*kernel_height))
			.ok_or_else(|| {
				Error::invalid_argument(format!("{operation} expanded height overflows usize"))
			})?;
		let expanded_width = input_width
			.saturating_sub(1)
			.checked_mul(stride)
			.and_then(|extent| extent.checked_add(*kernel_width))
			.ok_or_else(|| {
				Error::invalid_argument(format!("{operation} expanded width overflows usize"))
			})?;
		let doubled_padding = padding.checked_mul(2).ok_or_else(|| {
			Error::invalid_argument(format!("{operation} doubled padding overflows usize"))
		})?;
		if expanded_height <= doubled_padding || expanded_width <= doubled_padding {
			return Err(Error::invalid_argument(format!(
				"{operation} padding removes the complete transposed output"
			)));
		}
		let output_height = expanded_height - doubled_padding;
		let output_width = expanded_width - doubled_padding;
		let output_count = batch_size
			.checked_mul(*output_channels)
			.and_then(|count| count.checked_mul(output_height))
			.and_then(|count| count.checked_mul(output_width))
			.ok_or_else(|| {
				Error::invalid_argument(format!("{operation} output size overflows usize"))
			})?;
		let weight_count = input_channels
			.checked_mul(*output_channels)
			.and_then(|count| count.checked_mul(*kernel_height))
			.and_then(|count| count.checked_mul(*kernel_width))
			.ok_or_else(|| {
				Error::invalid_argument(format!("{operation} weight size overflows usize"))
			})?;
		if weight_count != weight.num_elements() {
			return Err(Error::internal(format!(
				"{operation} validated weight shape disagrees with its storage extent"
			)));
		}
		let parameter_count = weight_count.checked_add(*output_channels).ok_or_else(|| {
			Error::invalid_argument(format!("{operation} parameter count overflows usize"))
		})?;
		let last_y = input_height
			.saturating_sub(1)
			.checked_mul(stride)
			.and_then(|coordinate| coordinate.checked_add(kernel_height.saturating_sub(1)))
			.ok_or_else(|| {
				Error::invalid_argument(format!("{operation} Y coordinate overflows usize"))
			})?;
		let last_x = input_width
			.saturating_sub(1)
			.checked_mul(stride)
			.and_then(|coordinate| coordinate.checked_add(kernel_width.saturating_sub(1)))
			.ok_or_else(|| {
				Error::invalid_argument(format!("{operation} X coordinate overflows usize"))
			})?;
		for (label, value) in [
			("input height", *input_height),
			("input width", *input_width),
			("output height", output_height),
			("output width", output_width),
			("kernel size", *kernel_height),
			("stride", stride),
			("padding", padding),
			("last Y coordinate", last_y),
			("last X coordinate", last_x),
		] {
			if value > i32::MAX as usize {
				return Err(Error::invalid_argument(format!(
					"{operation} {label} exceeds the signed shader coordinate ABI"
				)));
			}
		}
		Ok(Self {
			batch_size: shader_u32(*batch_size, "batch size", operation)?,
			input_channels: shader_u32(*input_channels, "input channel count", operation)?,
			output_channels: shader_u32(*output_channels, "output channel count", operation)?,
			input_height: shader_u32(*input_height, "input height", operation)?,
			input_width: shader_u32(*input_width, "input width", operation)?,
			kernel_size: shader_u32(*kernel_height, "kernel size", operation)?,
			stride: shader_u32(stride, "stride", operation)?,
			padding: shader_u32(padding, "padding", operation)?,
			output_height: shader_u32(output_height, "output height", operation)?,
			output_width: shader_u32(output_width, "output width", operation)?,
			output_count: shader_u32(output_count, "output element count", operation)?,
			weight_count: shader_u32(weight_count, "weight element count", operation)?,
			parameter_count: shader_u32(parameter_count, "parameter count", operation)?,
		})
	}

	fn dimensions(self) -> [PushConstant; 10] {
		[
			PushConstant::U32(self.batch_size),
			PushConstant::U32(self.input_channels),
			PushConstant::U32(self.output_channels),
			PushConstant::U32(self.input_height),
			PushConstant::U32(self.input_width),
			PushConstant::U32(self.kernel_size),
			PushConstant::U32(self.stride),
			PushConstant::U32(self.padding),
			PushConstant::U32(self.output_height),
			PushConstant::U32(self.output_width),
		]
	}

	fn attributes(self) -> [OpAttribute; 2] {
		[
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

pub(in crate::ml) fn conv_transpose_2d(
	input: &Matrix,
	weight: &Matrix,
	bias: &Matrix,
	stride: usize,
	padding: usize,
) -> Result<Matrix> {
	let contract = crate::core::operation::ml::CONV_TRANSPOSE_2D;
	let operation = contract.name();
	validate_f32_same_engine(operation, &[input, weight, bias])?;
	let geometry =
		ConvTranspose2dGeometry::resolve(input, weight, Some(bias), stride, padding, operation)?;
	let output = Matrix::allocate(
		input.engine_handle(),
		vec![
			geometry.batch_size as usize,
			geometry.output_channels as usize,
			geometry.output_height as usize,
			geometry.output_width as usize,
		],
		geometry.output_count as usize,
		DType::F32,
	)?;
	let convolution_buffers = [
		BufferBinding::read(input.storage()),
		BufferBinding::read(weight.storage()),
		BufferBinding::write(output.storage()),
	];
	let dimensions = geometry.dimensions();
	let bias_buffers = [
		BufferBinding::read_write(output.storage()),
		BufferBinding::read(bias.storage()),
	];
	let bias_constants = [
		PushConstant::U32(geometry.batch_size),
		PushConstant::U32(geometry.output_channels),
		PushConstant::U32(geometry.output_height),
		PushConstant::U32(geometry.output_width),
	];
	let dispatches = [
		ComputeDispatch {
			kernel: KernelId::MlConvTranspose2dF32,
			buffers: &convolution_buffers,
			push_constants: &dimensions,
			workgroups: KernelId::MlConvTranspose2dF32.linear_workgroups(geometry.output_count),
		},
		ComputeDispatch {
			kernel: KernelId::MlConvTranspose2dBiasAddF32,
			buffers: &bias_buffers,
			push_constants: &bias_constants,
			workgroups: KernelId::MlConvTranspose2dBiasAddF32
				.linear_workgroups(geometry.output_count),
		},
	];
	let attributes = geometry.attributes();
	let inputs = [input, weight, bias];
	let outputs = [&output];
	input.engine_handle().record_split_semantic(
		&dispatches,
		SemanticDispatch {
			contract,
			inputs: &inputs,
			outputs: &outputs,
			attributes: &attributes,
		},
	)?;
	Ok(output)
}

pub(in crate::ml) fn conv_transpose_2d_backward(
	input: &Matrix,
	weight: &Matrix,
	output_gradient: &Matrix,
	stride: usize,
	padding: usize,
) -> Result<(Matrix, Matrix, Matrix)> {
	let contract = crate::core::operation::ml::CONV_TRANSPOSE_2D_BACKWARD;
	let operation = contract.name();
	validate_f32_same_engine(operation, &[input, weight, output_gradient])?;
	let geometry =
		ConvTranspose2dGeometry::resolve(input, weight, None, stride, padding, operation)?;
	let expected_output_shape = [
		geometry.batch_size as usize,
		geometry.output_channels as usize,
		geometry.output_height as usize,
		geometry.output_width as usize,
	];
	if output_gradient.shape() != expected_output_shape {
		return Err(Error::invalid_argument(format!(
			"{operation} output gradient must have shape {expected_output_shape:?}; found {:?}",
			output_gradient.shape()
		)));
	}
	let input_gradient = Matrix::allocate(
		input.engine_handle(),
		input.shape().to_vec(),
		input.num_elements(),
		DType::F32,
	)?;
	let weight_gradient = Matrix::allocate(
		input.engine_handle(),
		weight.shape().to_vec(),
		geometry.weight_count as usize,
		DType::F32,
	)?;
	let bias_gradient = Matrix::allocate(
		input.engine_handle(),
		vec![geometry.output_channels as usize],
		geometry.output_channels as usize,
		DType::F32,
	)?;
	let dimensions = geometry.dimensions();
	let input_buffers = [
		BufferBinding::read(output_gradient.storage()),
		BufferBinding::read(weight.storage()),
		BufferBinding::write(input_gradient.storage()),
	];
	let parameter_buffers = [
		BufferBinding::read(input.storage()),
		BufferBinding::read(output_gradient.storage()),
		BufferBinding::write(weight_gradient.storage()),
		BufferBinding::write(bias_gradient.storage()),
	];
	let mut parameter_constants = dimensions.to_vec();
	parameter_constants.push(PushConstant::U32(geometry.parameter_count));
	let dispatches = [
		ComputeDispatch {
			kernel: KernelId::MlConvTranspose2dBackwardF32,
			buffers: &input_buffers,
			push_constants: &dimensions,
			workgroups: KernelId::MlConvTranspose2dBackwardF32.linear_workgroups(shader_u32(
				input.num_elements(),
				"input element count",
				operation,
			)?),
		},
		ComputeDispatch {
			kernel: KernelId::MlConvTranspose2dParameterBackwardF32,
			buffers: &parameter_buffers,
			push_constants: &parameter_constants,
			workgroups: KernelId::MlConvTranspose2dParameterBackwardF32
				.linear_workgroups(geometry.parameter_count),
		},
	];
	let attributes = geometry.attributes();
	let inputs = [input, weight, output_gradient];
	let outputs = [&input_gradient, &weight_gradient, &bias_gradient];
	input.engine_handle().record_split_semantic(
		&dispatches,
		SemanticDispatch {
			contract,
			inputs: &inputs,
			outputs: &outputs,
			attributes: &attributes,
		},
	)?;
	Ok((input_gradient, weight_gradient, bias_gradient))
}

#[derive(Clone, Copy)]
struct Conv2dGeometry {
	batch_size: u32,
	input_channels: u32,
	output_channels: u32,
	input_height: u32,
	input_width: u32,
	kernel_size: u32,
	stride: u32,
	padding: u32,
	output_height: u32,
	output_width: u32,
	groups: u32,
	output_count: u32,
	weight_count: u32,
	parameter_count: u32,
}

impl Conv2dGeometry {
	fn resolve(
		input: &Matrix,
		weight: &Matrix,
		bias: Option<&Matrix>,
		stride: usize,
		padding: usize,
		groups: usize,
		operation: &'static str,
	) -> Result<Self> {
		let [batch_size, input_channels, input_height, input_width] = input.shape() else {
			return Err(Error::invalid_argument(format!(
				"{operation} input must have NCHW rank four; found {:?}",
				input.shape()
			)));
		};
		let [
			output_channels,
			weight_input_channels,
			kernel_height,
			kernel_width,
		] = weight.shape()
		else {
			return Err(Error::invalid_argument(format!(
				"{operation} weight must have grouped OIHW rank four; found {:?}",
				weight.shape()
			)));
		};
		if *batch_size == 0
			|| *input_channels == 0
			|| *output_channels == 0
			|| *input_height == 0
			|| *input_width == 0
			|| *kernel_height == 0
			|| kernel_height != kernel_width
			|| stride == 0
			|| groups == 0
			|| !input_channels.is_multiple_of(groups)
			|| !output_channels.is_multiple_of(groups)
			|| *weight_input_channels != input_channels / groups
		{
			return Err(Error::invalid_argument(format!(
				"{operation} requires nonzero NCHW input, square OIHW/group weight, nonzero stride/groups, and channels divisible by groups"
			)));
		}
		if let Some(bias) = bias
			&& bias.shape() != [*output_channels]
		{
			return Err(Error::invalid_argument(format!(
				"{operation} bias must have shape [{output_channels}]; found {:?}",
				bias.shape()
			)));
		}

		let doubled_padding = padding.checked_mul(2).ok_or_else(|| {
			Error::invalid_argument(format!("{operation} doubled padding overflows usize"))
		})?;
		let padded_height = input_height.checked_add(doubled_padding).ok_or_else(|| {
			Error::invalid_argument(format!("{operation} padded height overflows usize"))
		})?;
		let padded_width = input_width.checked_add(doubled_padding).ok_or_else(|| {
			Error::invalid_argument(format!("{operation} padded width overflows usize"))
		})?;
		if padded_height < *kernel_height || padded_width < *kernel_width {
			return Err(Error::invalid_argument(format!(
				"{operation} kernel does not fit the padded input"
			)));
		}
		let output_height = (padded_height - kernel_height) / stride + 1;
		let output_width = (padded_width - kernel_width) / stride + 1;
		let output_count = batch_size
			.checked_mul(*output_channels)
			.and_then(|count| count.checked_mul(output_height))
			.and_then(|count| count.checked_mul(output_width))
			.ok_or_else(|| {
				Error::invalid_argument(format!("{operation} output size overflows usize"))
			})?;
		let weight_count = output_channels
			.checked_mul(*weight_input_channels)
			.and_then(|count| count.checked_mul(*kernel_height))
			.and_then(|count| count.checked_mul(*kernel_width))
			.ok_or_else(|| {
				Error::invalid_argument(format!("{operation} weight size overflows usize"))
			})?;
		if weight_count != weight.num_elements() {
			return Err(Error::internal(format!(
				"{operation} validated weight shape disagrees with its storage extent"
			)));
		}
		let parameter_count = weight_count.checked_add(*output_channels).ok_or_else(|| {
			Error::invalid_argument(format!("{operation} parameter count overflows usize"))
		})?;
		let last_input_coordinate = output_height
			.saturating_sub(1)
			.checked_mul(stride)
			.and_then(|value| value.checked_add(kernel_height.saturating_sub(1)))
			.ok_or_else(|| {
				Error::invalid_argument(format!("{operation} shader coordinate overflows usize"))
			})?;
		let last_input_x_coordinate = output_width
			.saturating_sub(1)
			.checked_mul(stride)
			.and_then(|value| value.checked_add(kernel_width.saturating_sub(1)))
			.ok_or_else(|| {
				Error::invalid_argument(format!("{operation} shader coordinate overflows usize"))
			})?;
		for (label, value) in [
			("input height", *input_height),
			("input width", *input_width),
			("kernel size", *kernel_height),
			("stride", stride),
			("padding", padding),
			("last input Y coordinate", last_input_coordinate),
			("last input X coordinate", last_input_x_coordinate),
		] {
			if value > i32::MAX as usize {
				return Err(Error::invalid_argument(format!(
					"{operation} {label} exceeds the signed shader coordinate ABI"
				)));
			}
		}

		Ok(Self {
			batch_size: shader_u32(*batch_size, "batch size", operation)?,
			input_channels: shader_u32(*input_channels, "input channel count", operation)?,
			output_channels: shader_u32(*output_channels, "output channel count", operation)?,
			input_height: shader_u32(*input_height, "input height", operation)?,
			input_width: shader_u32(*input_width, "input width", operation)?,
			kernel_size: shader_u32(*kernel_height, "kernel size", operation)?,
			stride: shader_u32(stride, "stride", operation)?,
			padding: shader_u32(padding, "padding", operation)?,
			output_height: shader_u32(output_height, "output height", operation)?,
			output_width: shader_u32(output_width, "output width", operation)?,
			groups: shader_u32(groups, "group count", operation)?,
			output_count: shader_u32(output_count, "output element count", operation)?,
			weight_count: shader_u32(weight_count, "weight element count", operation)?,
			parameter_count: shader_u32(parameter_count, "parameter count", operation)?,
		})
	}

	fn dimensions(self) -> [PushConstant; 11] {
		[
			PushConstant::U32(self.batch_size),
			PushConstant::U32(self.input_channels),
			PushConstant::U32(self.output_channels),
			PushConstant::U32(self.input_height),
			PushConstant::U32(self.input_width),
			PushConstant::U32(self.kernel_size),
			PushConstant::U32(self.stride),
			PushConstant::U32(self.padding),
			PushConstant::U32(self.output_height),
			PushConstant::U32(self.output_width),
			PushConstant::U32(self.groups),
		]
	}

	fn attributes(self) -> [OpAttribute; 3] {
		[
			OpAttribute::UnsignedInteger {
				name: "stride".into(),
				value: u64::from(self.stride),
			},
			OpAttribute::UnsignedInteger {
				name: "padding".into(),
				value: u64::from(self.padding),
			},
			OpAttribute::UnsignedInteger {
				name: "groups".into(),
				value: u64::from(self.groups),
			},
		]
	}
}

pub(in crate::ml) fn conv_2d(
	input: &Matrix,
	weight: &Matrix,
	bias: &Matrix,
	stride: usize,
	padding: usize,
	groups: usize,
) -> Result<Matrix> {
	let contract = crate::core::operation::ml::CONV_2D;
	let operation = contract.name();
	validate_f32_same_engine(operation, &[input, weight, bias])?;
	let geometry = Conv2dGeometry::resolve(
		input,
		weight,
		Some(bias),
		stride,
		padding,
		groups,
		operation,
	)?;
	let output = Matrix::allocate(
		input.engine_handle(),
		vec![
			geometry.batch_size as usize,
			geometry.output_channels as usize,
			geometry.output_height as usize,
			geometry.output_width as usize,
		],
		geometry.output_count as usize,
		DType::F32,
	)?;
	let buffers = [
		BufferBinding::read(input.storage()),
		BufferBinding::read(weight.storage()),
		BufferBinding::read(bias.storage()),
		BufferBinding::write(output.storage()),
	];
	let dimensions = geometry.dimensions();
	let attributes = geometry.attributes();
	let kernel = KernelId::MlConv2dF32;
	record_semantic(
		&[input, weight, bias],
		&[&output],
		&attributes,
		kernel,
		&buffers,
		&dimensions,
		kernel.linear_workgroups(geometry.output_count),
	)?;
	Ok(output)
}

pub(in crate::ml) fn conv_2d_backward(
	input: &Matrix,
	weight: &Matrix,
	output_gradient: &Matrix,
	stride: usize,
	padding: usize,
	groups: usize,
) -> Result<(Matrix, Matrix, Matrix)> {
	let contract = crate::core::operation::ml::CONV_2D_BACKWARD;
	let operation = contract.name();
	validate_f32_same_engine(operation, &[input, weight, output_gradient])?;
	let geometry =
		Conv2dGeometry::resolve(input, weight, None, stride, padding, groups, operation)?;
	let expected_output_shape = [
		geometry.batch_size as usize,
		geometry.output_channels as usize,
		geometry.output_height as usize,
		geometry.output_width as usize,
	];
	if output_gradient.shape() != expected_output_shape {
		return Err(Error::invalid_argument(format!(
			"{operation} output gradient must have shape {expected_output_shape:?}; found {:?}",
			output_gradient.shape()
		)));
	}
	let input_gradient = Matrix::allocate(
		input.engine_handle(),
		input.shape().to_vec(),
		input.num_elements(),
		DType::F32,
	)?;
	let weight_gradient = Matrix::allocate(
		input.engine_handle(),
		weight.shape().to_vec(),
		geometry.weight_count as usize,
		DType::F32,
	)?;
	let bias_gradient = Matrix::allocate(
		input.engine_handle(),
		vec![geometry.output_channels as usize],
		geometry.output_channels as usize,
		DType::F32,
	)?;
	let dimensions = geometry.dimensions();
	let input_buffers = [
		BufferBinding::read(output_gradient.storage()),
		BufferBinding::read(weight.storage()),
		BufferBinding::write(input_gradient.storage()),
	];
	let parameter_buffers = [
		BufferBinding::read(input.storage()),
		BufferBinding::read(output_gradient.storage()),
		BufferBinding::write(weight_gradient.storage()),
		BufferBinding::write(bias_gradient.storage()),
	];
	let mut parameter_dimensions = dimensions.to_vec();
	parameter_dimensions.push(PushConstant::U32(geometry.parameter_count));
	let input_kernel = KernelId::MlConv2dBackwardF32;
	let parameter_kernel = KernelId::MlConv2dParameterBackwardF32;
	let dispatches = [
		ComputeDispatch {
			kernel: input_kernel,
			buffers: &input_buffers,
			push_constants: &dimensions,
			workgroups: input_kernel.linear_workgroups(shader_u32(
				input.num_elements(),
				"input element count",
				operation,
			)?),
		},
		ComputeDispatch {
			kernel: parameter_kernel,
			buffers: &parameter_buffers,
			push_constants: &parameter_dimensions,
			workgroups: parameter_kernel.linear_workgroups(geometry.parameter_count),
		},
	];
	let attributes = geometry.attributes();
	let inputs = [input, weight, output_gradient];
	let outputs = [&input_gradient, &weight_gradient, &bias_gradient];
	input.engine_handle().record_split_semantic(
		&dispatches,
		SemanticDispatch {
			contract,
			inputs: &inputs,
			outputs: &outputs,
			attributes: &attributes,
		},
	)?;
	Ok((input_gradient, weight_gradient, bias_gradient))
}
