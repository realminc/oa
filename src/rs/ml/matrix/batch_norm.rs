//! Batch norm Matrix operation implementations.

use crate::ml::autograd;
use crate::ml::validation::{shader_u32, validate_f32_same_engine};
use crate::{
	DType, Error, Matrix, OpAttribute, Result,
	runtime::{BufferBinding, ComputeDispatch, KernelId, PushConstant, SemanticDispatch},
};

/// Batch-normalized values and the per-channel statistics saved for backward.
#[must_use]
pub struct BatchNorm2dResult {
	/// Normalized and affine-transformed NCHW values.
	pub output: Matrix,
	/// Population mean for each channel.
	pub mean: Matrix,
	/// Population variance for each channel.
	pub variance: Matrix,
}
/// Normalize an NCHW Matrix using statistics computed from the current batch.
///
/// # Errors
///
/// Returns an error unless `input` is nonempty rank-four F32, `weight` and
/// `bias` are same-engine F32 channel vectors, `epsilon` is finite and
/// positive, or runtime recording fails.
pub fn batch_norm_2d(
	input: &Matrix,
	weight: &Matrix,
	bias: &Matrix,
	epsilon: f32,
) -> Result<BatchNorm2dResult> {
	let result = batch_norm_2d_forward(input, weight, bias, epsilon)?;
	autograd::record_batch_norm_2d(
		input,
		&result.output,
		result.mean.clone(),
		result.variance.clone(),
		None,
		weight.clone(),
		None,
		bias.clone(),
		epsilon,
		true,
	)?;
	Ok(result)
}
/// Normalize an NCHW Matrix using explicit per-channel statistics.
///
/// This is the stateless inference form used by [`crate::ml::nn::BatchNorm2d`]
/// in evaluation mode.
///
/// # Errors
///
/// Returns an error unless every value is same-engine F32, the channel-vector
/// shapes match the input, `epsilon` is finite and positive, or runtime
/// recording fails.
pub fn batch_norm_2d_with_stats(
	input: &Matrix,
	mean: &Matrix,
	variance: &Matrix,
	weight: &Matrix,
	bias: &Matrix,
	epsilon: f32,
) -> Result<Matrix> {
	let output = batch_norm_2d_with_stats_forward(input, mean, variance, weight, bias, epsilon)?;
	autograd::record_batch_norm_2d(
		input,
		&output,
		mean.clone(),
		variance.clone(),
		None,
		weight.clone(),
		None,
		bias.clone(),
		epsilon,
		false,
	)?;
	Ok(output)
}
#[derive(Clone, Copy)]
struct BatchNorm2dGeometry {
	batch_size: u32,
	channels: u32,
	height: u32,
	width: u32,
	element_count: u32,
}

impl BatchNorm2dGeometry {
	fn resolve(input: &Matrix, operation: &'static str) -> Result<Self> {
		let [batch_size, channels, height, width] = input.shape() else {
			return Err(Error::invalid_argument(format!(
				"{operation} input must have NCHW rank four; found {:?}",
				input.shape()
			)));
		};
		if [*batch_size, *channels, *height, *width]
			.into_iter()
			.any(|extent| extent == 0)
		{
			return Err(Error::invalid_argument(format!(
				"{operation} input extents must be nonzero"
			)));
		}
		let batch_size = shader_u32(*batch_size, "batch size", operation)?;
		let channels = shader_u32(*channels, "channel count", operation)?;
		let height = shader_u32(*height, "height", operation)?;
		let width = shader_u32(*width, "width", operation)?;
		let element_count = shader_u32(input.num_elements(), "element count", operation)?;
		batch_size.checked_mul(channels).ok_or_else(|| {
			Error::invalid_argument(format!("{operation} batch-channel count exceeds u32"))
		})?;
		Ok(Self {
			batch_size,
			channels,
			height,
			width,
			element_count,
		})
	}

	const fn dimensions(self) -> [PushConstant; 4] {
		[
			PushConstant::U32(self.batch_size),
			PushConstant::U32(self.channels),
			PushConstant::U32(self.height),
			PushConstant::U32(self.width),
		]
	}

	const fn normalization_workgroups(self) -> [u32; 3] {
		[
			self.width.div_ceil(16),
			self.height.div_ceil(16),
			self.batch_size * self.channels,
		]
	}
}

fn validate_batch_norm_affine(
	input: &Matrix,
	weight: &Matrix,
	bias: &Matrix,
	epsilon: f32,
	operation: &'static str,
) -> Result<BatchNorm2dGeometry> {
	let geometry = BatchNorm2dGeometry::resolve(input, operation)?;
	if weight.shape() != [geometry.channels as usize] || bias.shape() != [geometry.channels as usize]
	{
		return Err(Error::invalid_argument(format!(
			"{operation} requires weight and bias [{}]; found {:?} and {:?}",
			geometry.channels,
			weight.shape(),
			bias.shape()
		)));
	}
	if !epsilon.is_finite() || epsilon <= 0.0 {
		return Err(Error::invalid_argument(format!(
			"{operation} epsilon must be finite and positive"
		)));
	}
	validate_f32_same_engine(operation, &[input, weight, bias])?;
	Ok(geometry)
}

pub(in crate::ml) fn batch_norm_2d_forward(
	input: &Matrix,
	weight: &Matrix,
	bias: &Matrix,
	epsilon: f32,
) -> Result<BatchNorm2dResult> {
	let contract = crate::core::operation::ml::BATCH_NORM_2D;
	let operation = contract.name();
	let geometry = validate_batch_norm_affine(input, weight, bias, epsilon, operation)?;
	let output = Matrix::allocate(
		input.engine_handle(),
		input.shape().to_vec(),
		input.num_elements(),
		DType::F32,
	)?;
	let channel_shape = vec![geometry.channels as usize];
	let mean = Matrix::allocate(
		input.engine_handle(),
		channel_shape.clone(),
		geometry.channels as usize,
		DType::F32,
	)?;
	let variance = Matrix::allocate(
		input.engine_handle(),
		channel_shape,
		geometry.channels as usize,
		DType::F32,
	)?;
	let stats_buffers = [
		BufferBinding::read(input.storage()),
		BufferBinding::write(mean.storage()),
		BufferBinding::write(variance.storage()),
	];
	let stats_push_constants = geometry.dimensions();
	let normalize_buffers = [
		BufferBinding::read(input.storage()),
		BufferBinding::read(mean.storage()),
		BufferBinding::read(variance.storage()),
		BufferBinding::read(weight.storage()),
		BufferBinding::read(bias.storage()),
		BufferBinding::write(output.storage()),
	];
	let normalize_push_constants = [
		PushConstant::U32(geometry.batch_size),
		PushConstant::U32(geometry.channels),
		PushConstant::U32(geometry.height),
		PushConstant::U32(geometry.width),
		PushConstant::F32(epsilon),
	];
	let dispatches = [
		ComputeDispatch {
			kernel: KernelId::MlBatchNorm2dF32,
			buffers: &stats_buffers,
			push_constants: &stats_push_constants,
			workgroups: [geometry.channels, 1, 1],
		},
		ComputeDispatch {
			kernel: KernelId::MlBatchNorm2dWithStatsF32,
			buffers: &normalize_buffers,
			push_constants: &normalize_push_constants,
			workgroups: geometry.normalization_workgroups(),
		},
	];
	let inputs = [input, weight, bias];
	let outputs = [&output, &mean, &variance];
	let attributes = [OpAttribute::Float {
		name: "epsilon".into(),
		value: f64::from(epsilon),
	}];
	input.engine_handle().record_split_semantic(
		&dispatches,
		SemanticDispatch {
			contract,
			inputs: &inputs,
			outputs: &outputs,
			attributes: &attributes,
		},
	)?;
	Ok(BatchNorm2dResult {
		output,
		mean,
		variance,
	})
}

pub(in crate::ml) fn batch_norm_2d_with_stats_forward(
	input: &Matrix,
	mean: &Matrix,
	variance: &Matrix,
	weight: &Matrix,
	bias: &Matrix,
	epsilon: f32,
) -> Result<Matrix> {
	let contract = crate::core::operation::ml::BATCH_NORM_2D_WITH_STATS;
	let operation = contract.name();
	let geometry = validate_batch_norm_affine(input, weight, bias, epsilon, operation)?;
	if mean.shape() != [geometry.channels as usize]
		|| variance.shape() != [geometry.channels as usize]
	{
		return Err(Error::invalid_argument(format!(
			"{operation} requires mean and variance [{}]; found {:?} and {:?}",
			geometry.channels,
			mean.shape(),
			variance.shape()
		)));
	}
	validate_f32_same_engine(operation, &[input, mean, variance, weight, bias])?;
	let output = Matrix::allocate(
		input.engine_handle(),
		input.shape().to_vec(),
		input.num_elements(),
		DType::F32,
	)?;
	let buffers = [
		BufferBinding::read(input.storage()),
		BufferBinding::read(mean.storage()),
		BufferBinding::read(variance.storage()),
		BufferBinding::read(weight.storage()),
		BufferBinding::read(bias.storage()),
		BufferBinding::write(output.storage()),
	];
	let push_constants = [
		PushConstant::U32(geometry.batch_size),
		PushConstant::U32(geometry.channels),
		PushConstant::U32(geometry.height),
		PushConstant::U32(geometry.width),
		PushConstant::F32(epsilon),
	];
	let attributes = [OpAttribute::Float {
		name: "epsilon".into(),
		value: f64::from(epsilon),
	}];
	{
		let inputs: &[&Matrix] = &[input, mean, variance, weight, bias];
		let outputs: &[&Matrix] = &[&output];
		let attributes: &[OpAttribute] = &attributes;
		let dispatch = ComputeDispatch {
			kernel: KernelId::MlBatchNorm2dWithStatsF32,
			buffers: &buffers,
			push_constants: &push_constants,
			workgroups: geometry.normalization_workgroups(),
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

pub(in crate::ml) fn batch_norm_2d_running_update(
	running_mean: &Matrix,
	running_variance: &Matrix,
	batch_mean: &Matrix,
	batch_variance: &Matrix,
	momentum: f32,
) -> Result<(Matrix, Matrix)> {
	let contract = crate::core::operation::ml::BATCH_NORM_2D_RUNNING_UPDATE;
	let operation = contract.name();
	let [channels] = running_mean.shape() else {
		return Err(Error::invalid_argument(format!(
			"{operation} running mean must be rank one"
		)));
	};
	if *channels == 0
		|| running_variance.shape() != [*channels]
		|| batch_mean.shape() != [*channels]
		|| batch_variance.shape() != [*channels]
	{
		return Err(Error::invalid_argument(format!(
			"{operation} requires four equal nonempty channel vectors"
		)));
	}
	if !momentum.is_finite() || !(0.0..=1.0).contains(&momentum) {
		return Err(Error::invalid_argument(format!(
			"{operation} momentum must be finite and in [0, 1]"
		)));
	}
	validate_f32_same_engine(
		operation,
		&[running_mean, running_variance, batch_mean, batch_variance],
	)?;
	let channels = shader_u32(*channels, "channel count", operation)?;
	let updated_mean = Matrix::allocate(
		running_mean.engine_handle(),
		running_mean.shape().to_vec(),
		running_mean.num_elements(),
		DType::F32,
	)?;
	let updated_variance = Matrix::allocate(
		running_mean.engine_handle(),
		running_mean.shape().to_vec(),
		running_mean.num_elements(),
		DType::F32,
	)?;
	let buffers = [
		BufferBinding::read(running_mean.storage()),
		BufferBinding::read(running_variance.storage()),
		BufferBinding::read(batch_mean.storage()),
		BufferBinding::read(batch_variance.storage()),
		BufferBinding::write(updated_mean.storage()),
		BufferBinding::write(updated_variance.storage()),
	];
	let push_constants = [PushConstant::U32(channels), PushConstant::F32(momentum)];
	let attributes = [OpAttribute::Float {
		name: "momentum".into(),
		value: f64::from(momentum),
	}];
	{
		let inputs: &[&Matrix] = &[running_mean, running_variance, batch_mean, batch_variance];
		let outputs: &[&Matrix] = &[&updated_mean, &updated_variance];
		let attributes: &[OpAttribute] = &attributes;
		let dispatch = ComputeDispatch {
			kernel: KernelId::MlBatchNorm2dRunningUpdateF32,
			buffers: &buffers,
			push_constants: &push_constants,
			workgroups: KernelId::MlBatchNorm2dRunningUpdateF32.linear_workgroups(channels),
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
	Ok((updated_mean, updated_variance))
}

#[allow(
	clippy::too_many_arguments,
	reason = "the adjoint consumes the exact saved BatchNorm state explicitly"
)]
pub(in crate::ml) fn batch_norm_2d_backward(
	input: &Matrix,
	weight: &Matrix,
	mean: &Matrix,
	variance: &Matrix,
	output_gradient: &Matrix,
	epsilon: f32,
	training: bool,
) -> Result<(Matrix, Matrix, Matrix)> {
	let contract = crate::core::operation::ml::BATCH_NORM_2D_BACKWARD;
	let operation = contract.name();
	let geometry = BatchNorm2dGeometry::resolve(input, operation)?;
	if weight.shape() != [geometry.channels as usize]
		|| mean.shape() != [geometry.channels as usize]
		|| variance.shape() != [geometry.channels as usize]
		|| output_gradient.shape() != input.shape()
	{
		return Err(Error::invalid_argument(format!(
			"{operation} requires input/output gradient NCHW and matching channel vectors"
		)));
	}
	if !epsilon.is_finite() || epsilon <= 0.0 {
		return Err(Error::invalid_argument(format!(
			"{operation} epsilon must be finite and positive"
		)));
	}
	validate_f32_same_engine(operation, &[input, weight, mean, variance, output_gradient])?;
	let input_gradient = Matrix::allocate(
		input.engine_handle(),
		input.shape().to_vec(),
		input.num_elements(),
		DType::F32,
	)?;
	let channel_shape = vec![geometry.channels as usize];
	let weight_gradient = Matrix::allocate(
		input.engine_handle(),
		channel_shape.clone(),
		geometry.channels as usize,
		DType::F32,
	)?;
	let bias_gradient = Matrix::allocate(
		input.engine_handle(),
		channel_shape,
		geometry.channels as usize,
		DType::F32,
	)?;
	let stats_buffers = [
		BufferBinding::read(input.storage()),
		BufferBinding::read(mean.storage()),
		BufferBinding::read(variance.storage()),
		BufferBinding::read(output_gradient.storage()),
		BufferBinding::write(weight_gradient.storage()),
		BufferBinding::write(bias_gradient.storage()),
	];
	let stats_push_constants = [
		PushConstant::U32(geometry.batch_size),
		PushConstant::U32(geometry.channels),
		PushConstant::U32(geometry.height),
		PushConstant::U32(geometry.width),
		PushConstant::F32(epsilon),
	];
	let input_buffers = [
		BufferBinding::read(input.storage()),
		BufferBinding::read(mean.storage()),
		BufferBinding::read(variance.storage()),
		BufferBinding::read(weight.storage()),
		BufferBinding::read(output_gradient.storage()),
		BufferBinding::read(weight_gradient.storage()),
		BufferBinding::read(bias_gradient.storage()),
		BufferBinding::write(input_gradient.storage()),
	];
	let input_push_constants = [
		PushConstant::U32(geometry.element_count),
		PushConstant::U32(geometry.channels),
		PushConstant::U32(geometry.height),
		PushConstant::U32(geometry.width),
		PushConstant::F32(epsilon),
		PushConstant::U32(if training { 1 } else { 0 }),
	];
	let dispatches = [
		ComputeDispatch {
			kernel: KernelId::MlBatchNorm2dBackwardF32,
			buffers: &stats_buffers,
			push_constants: &stats_push_constants,
			workgroups: [geometry.channels, 1, 1],
		},
		ComputeDispatch {
			kernel: KernelId::MlBatchNorm2dInputBackwardF32,
			buffers: &input_buffers,
			push_constants: &input_push_constants,
			workgroups: KernelId::MlBatchNorm2dInputBackwardF32.linear_workgroups(geometry.element_count),
		},
	];
	let inputs = [input, weight, mean, variance, output_gradient];
	let outputs = [&input_gradient, &weight_gradient, &bias_gradient];
	let attributes = [
		OpAttribute::Float {
			name: "epsilon".into(),
			value: f64::from(epsilon),
		},
		OpAttribute::Boolean {
			name: "training".into(),
			value: training,
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
	Ok((input_gradient, weight_gradient, bias_gradient))
}
