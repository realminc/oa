//! Channel norm Matrix operation implementations.

use crate::ml::autograd;
use crate::ml::validation::{shader_u32, validate_f32_same_engine};
use crate::{
	DType, Error, Matrix, OpAttribute, Result,
	runtime::{BufferBinding, ComputeDispatch, KernelId, PushConstant, SemanticDispatch},
};

/// Complete adjoints of channel-axis normalization over BCT storage.
pub struct ChannelNormBackward {
	/// Gradient of the BCT input.
	pub input: Matrix,
	/// Gradient of the channel affine weight.
	pub weight: Matrix,
	/// Gradient of the channel affine bias.
	pub bias: Matrix,
}

pub(in crate::ml) fn channel_norm_tracked(
	input: &Matrix,
	weight: &Matrix,
	bias: &Matrix,
	epsilon: f32,
	relu: bool,
) -> Result<Matrix> {
	let output = channel_norm_forward(input, weight, bias, epsilon, relu)?;
	autograd::record_channel_norm(
		input,
		&output,
		None,
		weight.clone(),
		None,
		bias.clone(),
		epsilon,
		relu,
	)?;
	Ok(output)
}

/// Normalize the channel axis of a nonempty FP32 `[B, C, T]` Matrix.
///
/// # Errors
///
/// Returns an error unless the affine vectors match `C`, all values belong to
/// one Engine, epsilon is finite and positive, `C <= 1024`, or recording fails.
pub fn channel_norm(
	input: &Matrix,
	weight: &Matrix,
	bias: &Matrix,
	epsilon: f32,
) -> Result<Matrix> {
	channel_norm_tracked(input, weight, bias, epsilon, false)
}

/// Normalize the channel axis of `[B, C, T]` and fuse the following ReLU.
///
/// # Errors
///
/// Returns an error under the same conditions as [`channel_norm`], or when
/// runtime recording fails.
pub fn channel_norm_relu(
	input: &Matrix,
	weight: &Matrix,
	bias: &Matrix,
	epsilon: f32,
) -> Result<Matrix> {
	channel_norm_tracked(input, weight, bias, epsilon, true)
}

/// Compute complete explicit channel-normalization adjoints.
///
/// # Errors
///
/// Returns an error when the forward contract or output gradient is invalid,
/// or runtime recording fails.
pub fn channel_norm_backward(
	input: &Matrix,
	weight: &Matrix,
	output_gradient: &Matrix,
	epsilon: f32,
) -> Result<ChannelNormBackward> {
	channel_norm_backward_saved(input, weight, None, output_gradient, epsilon)
}

/// Compute complete adjoints of fused channel normalization and ReLU.
///
/// # Errors
///
/// Returns an error when saved output or gradient does not match the forward
/// contract, or runtime recording fails.
pub fn channel_norm_relu_backward(
	input: &Matrix,
	weight: &Matrix,
	forward_output: &Matrix,
	output_gradient: &Matrix,
	epsilon: f32,
) -> Result<ChannelNormBackward> {
	channel_norm_backward_saved(
		input,
		weight,
		Some(forward_output),
		output_gradient,
		epsilon,
	)
}
fn channel_norm_geometry(
	input: &Matrix,
	weight: &Matrix,
	bias: Option<&Matrix>,
	epsilon: f32,
	operation: &'static str,
) -> Result<(u32, u32, u32)> {
	let [batch, channels, sequence_length] = input.shape() else {
		return Err(Error::invalid_argument(format!(
			"{operation} input must have BCT rank three; found {:?}",
			input.shape()
		)));
	};
	if *batch == 0 || *channels == 0 || *sequence_length == 0 {
		return Err(Error::invalid_argument(format!(
			"{operation} BCT extents must be nonzero"
		)));
	}
	if *channels > 1024 {
		return Err(Error::invalid_argument(format!(
			"{operation} currently supports at most 1024 channels; found {channels}"
		)));
	}
	if weight.shape() != [*channels] || bias.is_some_and(|value| value.shape() != [*channels]) {
		return Err(Error::invalid_argument(format!(
			"{operation} requires weight and bias vectors matching C={channels}"
		)));
	}
	if !epsilon.is_finite() || epsilon <= 0.0 {
		return Err(Error::invalid_argument(format!(
			"{operation} epsilon must be finite and positive"
		)));
	}
	let mut matrices = vec![input, weight];
	if let Some(bias) = bias {
		matrices.push(bias);
	}
	validate_f32_same_engine(operation, &matrices)?;
	Ok((
		shader_u32(*batch, "batch", operation)?,
		shader_u32(*channels, "channel count", operation)?,
		shader_u32(*sequence_length, "sequence length", operation)?,
	))
}

pub(in crate::ml) fn channel_norm_forward(
	input: &Matrix,
	weight: &Matrix,
	bias: &Matrix,
	epsilon: f32,
	relu: bool,
) -> Result<Matrix> {
	let contract = if relu {
		crate::core::operation::ml::CHANNEL_NORM_RELU
	} else {
		crate::core::operation::ml::CHANNEL_NORM
	};
	let operation = contract.name();
	let (batch, channels, sequence_length) =
		channel_norm_geometry(input, weight, Some(bias), epsilon, operation)?;
	let output = Matrix::allocate(
		input.engine_handle(),
		input.shape().to_vec(),
		input.num_elements(),
		DType::F32,
	)?;
	let buffers = [
		BufferBinding::read(input.storage()),
		BufferBinding::read(weight.storage()),
		BufferBinding::read(bias.storage()),
		BufferBinding::write(output.storage()),
	];
	let push_constants = [
		PushConstant::U32(batch),
		PushConstant::U32(channels),
		PushConstant::U32(sequence_length),
		PushConstant::F32(epsilon),
	];
	let attributes = [
		OpAttribute::UnsignedInteger {
			name: "batch".into(),
			value: u64::from(batch),
		},
		OpAttribute::UnsignedInteger {
			name: "channels".into(),
			value: u64::from(channels),
		},
		OpAttribute::UnsignedInteger {
			name: "sequence_length".into(),
			value: u64::from(sequence_length),
		},
		OpAttribute::Float {
			name: "epsilon".into(),
			value: f64::from(epsilon),
		},
	];
	let kernel = if relu {
		KernelId::MlChannelNormReluF32
	} else {
		KernelId::MlChannelNormF32
	};
	{
		let inputs: &[&Matrix] = &[input, weight, bias];
		let outputs: &[&Matrix] = &[&output];
		let attributes: &[OpAttribute] = &attributes;
		let dispatch = ComputeDispatch {
			kernel,
			buffers: &buffers,
			push_constants: &push_constants,
			workgroups: [
				batch
					.checked_mul(sequence_length)
					.ok_or_else(|| Error::invalid_argument(format!("{operation} row count exceeds u32")))?,
				1,
				1,
			],
		};
		let contract = dispatch.kernel.semantic_contract().ok_or_else(|| {
			Error::internal(format!(
				"lowering-only kernel {} cannot own one semantic ML operation",
				dispatch.kernel.report_name()
			))
		})?;
		let engine = inputs
			.first()
			.ok_or_else(|| Error::internal("semantic ML dispatch requires an input"))?
			.engine_handle();
		engine.record_semantic(
			dispatch,
			SemanticDispatch {
				contract,
				inputs,
				outputs,
				attributes,
			},
		)
	}?;
	Ok(output)
}

#[allow(
	clippy::too_many_arguments,
	reason = "the fused adjoint consumes its complete saved forward state"
)]
pub(in crate::ml) fn channel_norm_backward_saved(
	input: &Matrix,
	weight: &Matrix,
	forward_output: Option<&Matrix>,
	output_gradient: &Matrix,
	epsilon: f32,
) -> Result<ChannelNormBackward> {
	let contract = if forward_output.is_some() {
		crate::core::operation::ml::CHANNEL_NORM_RELU_BACKWARD
	} else {
		crate::core::operation::ml::CHANNEL_NORM_BACKWARD
	};
	let operation = contract.name();
	let (batch, channels, sequence_length) =
		channel_norm_geometry(input, weight, None, epsilon, operation)?;
	if output_gradient.shape() != input.shape()
		|| forward_output.is_some_and(|value| value.shape() != input.shape())
	{
		return Err(Error::invalid_argument(format!(
			"{operation} saved output and output gradient must match the input shape"
		)));
	}
	let mut matrices = vec![output_gradient];
	if let Some(output) = forward_output {
		matrices.push(output);
	}
	validate_f32_same_engine(operation, &matrices)?;
	let rows = batch
		.checked_mul(sequence_length)
		.ok_or_else(|| Error::invalid_argument(format!("{operation} row count exceeds u32")))?;
	let input_gradient = Matrix::allocate(
		input.engine_handle(),
		input.shape().to_vec(),
		input.num_elements(),
		DType::F32,
	)?;
	let contribution_count = (rows as usize)
		.checked_mul(channels as usize)
		.ok_or_else(|| Error::invalid_argument(format!("{operation} contribution size overflows")))?;
	let weight_contribution = Matrix::allocate(
		input.engine_handle(),
		vec![rows as usize, channels as usize],
		contribution_count,
		DType::F32,
	)?;
	let bias_contribution = Matrix::allocate(
		input.engine_handle(),
		vec![rows as usize, channels as usize],
		contribution_count,
		DType::F32,
	)?;
	let weight_gradient = Matrix::allocate(
		input.engine_handle(),
		weight.shape().to_vec(),
		weight.num_elements(),
		DType::F32,
	)?;
	let bias_gradient = Matrix::allocate(
		input.engine_handle(),
		weight.shape().to_vec(),
		weight.num_elements(),
		DType::F32,
	)?;
	let push_constants = [
		PushConstant::U32(batch),
		PushConstant::U32(channels),
		PushConstant::U32(sequence_length),
		PushConstant::F32(epsilon),
	];
	let plain_buffers = [
		BufferBinding::read(input.storage()),
		BufferBinding::read(weight.storage()),
		BufferBinding::read(output_gradient.storage()),
		BufferBinding::write(input_gradient.storage()),
		BufferBinding::write(weight_contribution.storage()),
		BufferBinding::write(bias_contribution.storage()),
	];
	let relu_buffers = forward_output.map(|output| {
		[
			BufferBinding::read(input.storage()),
			BufferBinding::read(weight.storage()),
			BufferBinding::read(output.storage()),
			BufferBinding::read(output_gradient.storage()),
			BufferBinding::write(input_gradient.storage()),
			BufferBinding::write(weight_contribution.storage()),
			BufferBinding::write(bias_contribution.storage()),
		]
	});
	let sum_push_constants = [
		PushConstant::U32(1),
		PushConstant::U32(rows),
		PushConstant::U32(channels),
	];
	let weight_buffers = [
		BufferBinding::read(weight_contribution.storage()),
		BufferBinding::write(weight_gradient.storage()),
	];
	let bias_buffers = [
		BufferBinding::read(bias_contribution.storage()),
		BufferBinding::write(bias_gradient.storage()),
	];
	let primary = ComputeDispatch {
		kernel: if forward_output.is_some() {
			KernelId::MlChannelNormReluBackwardF32
		} else {
			KernelId::MlChannelNormBackwardF32
		},
		buffers: relu_buffers
			.as_ref()
			.map_or(&plain_buffers[..], |buffers| &buffers[..]),
		push_constants: &push_constants,
		workgroups: [rows, 1, 1],
	};
	let sum_workgroups = KernelId::MatrixSumAxisF32.linear_workgroups(channels);
	let dispatches = [
		primary,
		ComputeDispatch {
			kernel: KernelId::MatrixSumAxisF32,
			buffers: &weight_buffers,
			push_constants: &sum_push_constants,
			workgroups: sum_workgroups,
		},
		ComputeDispatch {
			kernel: KernelId::MatrixSumAxisF32,
			buffers: &bias_buffers,
			push_constants: &sum_push_constants,
			workgroups: sum_workgroups,
		},
	];
	let mut inputs = vec![input, weight];
	if let Some(output) = forward_output {
		inputs.push(output);
	}
	inputs.push(output_gradient);
	let outputs = [&input_gradient, &weight_gradient, &bias_gradient];
	let attributes = [
		OpAttribute::UnsignedInteger {
			name: "batch".into(),
			value: u64::from(batch),
		},
		OpAttribute::UnsignedInteger {
			name: "channels".into(),
			value: u64::from(channels),
		},
		OpAttribute::UnsignedInteger {
			name: "sequence_length".into(),
			value: u64::from(sequence_length),
		},
		OpAttribute::Float {
			name: "epsilon".into(),
			value: f64::from(epsilon),
		},
	];
	input.engine_handle().record_split_semantic(
		&dispatches,
		SemanticDispatch {
			contract,
			inputs: &inputs,
			outputs: &outputs,
			attributes: &attributes,
		},
	)?;
	Ok(ChannelNormBackward {
		input: input_gradient,
		weight: weight_gradient,
		bias: bias_gradient,
	})
}
