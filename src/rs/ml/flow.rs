use crate::runtime::ComputeDispatch;
use crate::runtime::SemanticDispatch;
use crate::runtime::{BufferBinding, KernelId, PushConstant};
use crate::{DType, Error, Matrix, OpAttribute, Result};

use super::{
	autograd,
	validation::{shader_u32, validate_f32_same_engine},
};

/// Linear flow state and its constant path velocity.
pub struct FlowMatchBatch {
	/// Interpolated state `x(t)`.
	pub state: Matrix,
	/// Constant linear-path velocity `noise - clean`.
	pub velocity: Matrix,
}

/// Construct `x(t) = clean + t * (noise - clean)` and `v = noise - clean`.
///
/// A rank-one time vector matching the batch dimension expands across every
/// non-batch axis. Scalars and ordinarily broadcastable time shapes are also
/// accepted.
///
/// # Errors
///
/// Returns an error unless every input is same-engine F32, clean and noise are
/// matching and nonempty, time broadcasts to them, and recording succeeds.
pub fn linear_match(clean: &Matrix, noise: &Matrix, time: &Matrix) -> Result<FlowMatchBatch> {
	let (state, velocity) = linear_match_dispatch(clean, noise, time)?;
	autograd::record_flow_linear_match(clean, noise, time, &state, &velocity)?;
	Ok(FlowMatchBatch { state, velocity })
}

/// Advance one explicit Euler step: `state + velocity * delta_time`.
///
/// # Errors
///
/// Returns an error unless state and velocity are matching nonempty
/// same-engine F32 matrices, `delta_time` is finite, and recording succeeds.
pub fn euler_step(state: &Matrix, velocity: &Matrix, delta_time: f32) -> Result<Matrix> {
	let output = euler_step_dispatch(state, velocity, delta_time)?;
	autograd::record_flow_euler_step(state, velocity, delta_time, &output)?;
	Ok(output)
}

/// Compute mean squared error over broadcast-selected valid elements.
///
/// The mask contributes its values to the denominator. An all-zero mask
/// returns exact zero by clamping the denominator to one. Target and mask are
/// detached; reverse mode differentiates only prediction.
///
/// # Errors
///
/// Returns an error unless prediction and target are matching nonempty
/// same-engine F32 matrices, mask is same-engine F32 and broadcastable to
/// prediction, and recording succeeds.
pub fn masked_mse(prediction: &Matrix, target: &Matrix, mask: &Matrix) -> Result<Matrix> {
	let (loss, denominator) = masked_mse_dispatch(prediction, target, mask)?;
	autograd::record_flow_masked_mse(prediction, target, mask, &denominator, &loss)?;
	Ok(loss)
}

// ---------------------------------------------------------------------------
// Private dispatch helpers
// ---------------------------------------------------------------------------

fn linear_match_dispatch(
	clean: &Matrix,
	noise: &Matrix,
	time: &Matrix,
) -> Result<(Matrix, Matrix)> {
	const OPERATION: &str = "oa::ml::flow::linear_match";
	validate_f32_same_engine(OPERATION, &[clean, noise, time])?;
	if clean.shape() != noise.shape() || clean.num_elements() == 0 || clean.shape().len() > 8 {
		return Err(Error::invalid_argument(
			"flow linear_match requires matching nonempty clean/noise shapes with rank at most eight",
		));
	}
	let (time_dims, time_strides) = broadcast_metadata(time.shape(), clean.shape(), true)?;
	let state_strides = dense_strides(clean.shape())?;
	let element_count = shader_u32(clean.num_elements(), "element count", OPERATION)?;
	let rank = shader_u32(clean.shape().len(), "rank", OPERATION)?;
	let state = Matrix::allocate(
		clean.engine_handle(),
		clean.shape().to_vec(),
		clean.num_elements(),
		DType::F32,
	)?;
	let velocity = Matrix::allocate(
		clean.engine_handle(),
		clean.shape().to_vec(),
		clean.num_elements(),
		DType::F32,
	)?;
	let buffers = [
		BufferBinding::read(clean.storage()),
		BufferBinding::read(noise.storage()),
		BufferBinding::read(time.storage()),
		BufferBinding::write(state.storage()),
		BufferBinding::write(velocity.storage()),
	];
	let mut push_constants = Vec::with_capacity(26);
	push_constants.push(PushConstant::U32(element_count));
	push_constants.push(PushConstant::U32(rank));
	extend_u32(&mut push_constants, state_strides);
	extend_u32(&mut push_constants, time_dims);
	extend_u32(&mut push_constants, time_strides);
	{
		let inputs: &[&Matrix] = &[clean, noise, time];
		let outputs: &[&Matrix] = &[&state, &velocity];
		let attributes: &[OpAttribute] = &[];
		let dispatch = ComputeDispatch {
			kernel: KernelId::MlFlowLinearMatchF32,
			buffers: &buffers,
			push_constants: &push_constants,
			workgroups: KernelId::MlFlowLinearMatchF32.linear_workgroups(element_count),
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
	Ok((state, velocity))
}

fn euler_step_dispatch(state: &Matrix, velocity: &Matrix, delta_time: f32) -> Result<Matrix> {
	const OPERATION: &str = "oa::ml::flow::euler_step";
	validate_f32_same_engine(OPERATION, &[state, velocity])?;
	if state.shape() != velocity.shape() || state.num_elements() == 0 || !delta_time.is_finite() {
		return Err(Error::invalid_argument(
			"flow euler_step requires matching nonempty F32 state/velocity and finite delta_time",
		));
	}
	let element_count = shader_u32(state.num_elements(), "element count", OPERATION)?;
	let output = Matrix::allocate(
		state.engine_handle(),
		state.shape().to_vec(),
		state.num_elements(),
		DType::F32,
	)?;
	let buffers = [
		BufferBinding::read(state.storage()),
		BufferBinding::read(velocity.storage()),
		BufferBinding::write(output.storage()),
	];
	let push_constants = [
		PushConstant::U32(element_count),
		PushConstant::F32(delta_time),
	];
	let attributes = [OpAttribute::Float {
		name: "delta_time".to_owned(),
		value: f64::from(delta_time),
	}];
	{
		let inputs: &[&Matrix] = &[state, velocity];
		let outputs: &[&Matrix] = &[&output];
		let attributes: &[OpAttribute] = &attributes;
		let dispatch = ComputeDispatch {
			kernel: KernelId::MlFlowEulerStepF32,
			buffers: &buffers,
			push_constants: &push_constants,
			workgroups: KernelId::MlFlowEulerStepF32.linear_workgroups(element_count),
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

fn masked_mse_dispatch(
	prediction: &Matrix,
	target: &Matrix,
	mask: &Matrix,
) -> Result<(Matrix, Matrix)> {
	const OPERATION: &str = "oa::ml::flow::masked_mse";
	validate_f32_same_engine(OPERATION, &[prediction, target, mask])?;
	if prediction.shape() != target.shape()
		|| prediction.num_elements() == 0
		|| prediction.shape().len() > 8
	{
		return Err(Error::invalid_argument(
			"flow masked_mse requires matching nonempty prediction/target shapes with rank at most eight",
		));
	}
	let (mask_dims, mask_strides) = broadcast_metadata(mask.shape(), prediction.shape(), false)?;
	let prediction_strides = dense_strides(prediction.shape())?;
	let element_count = shader_u32(prediction.num_elements(), "element count", OPERATION)?;
	let rank = shader_u32(prediction.shape().len(), "rank", OPERATION)?;
	let loss = Matrix::allocate(prediction.engine_handle(), Vec::new(), 1, DType::F32)?;
	let denominator = Matrix::allocate(prediction.engine_handle(), Vec::new(), 1, DType::F32)?;
	let buffers = [
		BufferBinding::read(prediction.storage()),
		BufferBinding::read(target.storage()),
		BufferBinding::read(mask.storage()),
		BufferBinding::write(loss.storage()),
		BufferBinding::write(denominator.storage()),
	];
	let push_constants = shape_push_constants(
		element_count,
		rank,
		prediction_strides,
		mask_dims,
		mask_strides,
	);
	{
		let inputs: &[&Matrix] = &[prediction, target, mask];
		let outputs: &[&Matrix] = &[&loss];
		let attributes: &[OpAttribute] = &[];
		let dispatch = ComputeDispatch {
			kernel: KernelId::MlFlowMaskedMseF32,
			buffers: &buffers,
			push_constants: &push_constants,
			workgroups: [1, 1, 1],
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
	Ok((loss, denominator))
}

pub(in crate::ml) fn masked_mse_backward(
	prediction: &Matrix,
	target: &Matrix,
	mask: &Matrix,
	denominator: &Matrix,
	upstream: &Matrix,
) -> Result<Matrix> {
	const OPERATION: &str = "oa::ml::flow::masked_mse_backward";
	validate_f32_same_engine(
		OPERATION,
		&[prediction, target, mask, denominator, upstream],
	)?;
	if prediction.shape() != target.shape()
		|| prediction.num_elements() == 0
		|| prediction.shape().len() > 8
		|| !denominator.shape().is_empty()
		|| !upstream.shape().is_empty()
	{
		return Err(Error::invalid_argument(
			"flow masked_mse backward received invalid saved values or scalar upstream",
		));
	}
	let (mask_dims, mask_strides) = broadcast_metadata(mask.shape(), prediction.shape(), false)?;
	let prediction_strides = dense_strides(prediction.shape())?;
	let element_count = shader_u32(prediction.num_elements(), "element count", OPERATION)?;
	let rank = shader_u32(prediction.shape().len(), "rank", OPERATION)?;
	let gradient = Matrix::allocate(
		prediction.engine_handle(),
		prediction.shape().to_vec(),
		prediction.num_elements(),
		DType::F32,
	)?;
	let buffers = [
		BufferBinding::read(prediction.storage()),
		BufferBinding::read(target.storage()),
		BufferBinding::read(mask.storage()),
		BufferBinding::read(denominator.storage()),
		BufferBinding::read(upstream.storage()),
		BufferBinding::write(gradient.storage()),
	];
	let push_constants = shape_push_constants(
		element_count,
		rank,
		prediction_strides,
		mask_dims,
		mask_strides,
	);
	{
		let inputs: &[&Matrix] = &[prediction, target, mask, denominator, upstream];
		let outputs: &[&Matrix] = &[&gradient];
		let attributes: &[OpAttribute] = &[];
		let dispatch = ComputeDispatch {
			kernel: KernelId::MlFlowMaskedMseBackwardF32,
			buffers: &buffers,
			push_constants: &push_constants,
			workgroups: KernelId::MlFlowMaskedMseBackwardF32.linear_workgroups(element_count),
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
	Ok(gradient)
}

// ---------------------------------------------------------------------------
// Geometry helpers
// ---------------------------------------------------------------------------

fn shape_push_constants(
	element_count: u32,
	rank: u32,
	first_strides: [u32; 8],
	second_dims: [u32; 8],
	second_strides: [u32; 8],
) -> Vec<PushConstant> {
	let mut push_constants = Vec::with_capacity(26);
	push_constants.push(PushConstant::U32(element_count));
	push_constants.push(PushConstant::U32(rank));
	extend_u32(&mut push_constants, first_strides);
	extend_u32(&mut push_constants, second_dims);
	extend_u32(&mut push_constants, second_strides);
	push_constants
}

fn extend_u32(output: &mut Vec<PushConstant>, values: [u32; 8]) {
	output.extend(values.into_iter().map(PushConstant::U32));
}

fn dense_strides(shape: &[usize]) -> Result<[u32; 8]> {
	let mut output = [1_u32; 8];
	let mut stride = 1_usize;
	for axis in (0..shape.len()).rev() {
		output[axis] = u32::try_from(stride)
			.map_err(|_| Error::invalid_argument("flow stride exceeds shader u32 ABI"))?;
		stride = stride
			.checked_mul(shape[axis])
			.ok_or_else(|| Error::invalid_argument("flow shape stride overflows usize"))?;
	}
	Ok(output)
}

fn broadcast_metadata(
	input_shape: &[usize],
	output_shape: &[usize],
	batch_vector_special_case: bool,
) -> Result<([u32; 8], [u32; 8])> {
	if input_shape.len() > output_shape.len() || output_shape.len() > 8 {
		return Err(Error::invalid_argument(
			"flow input is not broadcastable to the output shape",
		));
	}
	let mut expanded = [1_usize; 8];
	if batch_vector_special_case
		&& input_shape.len() == 1
		&& output_shape.len() > 1
		&& (input_shape[0] == 1 || input_shape[0] == output_shape[0])
	{
		expanded[0] = input_shape[0];
	} else {
		let offset = output_shape.len() - input_shape.len();
		for (axis, extent) in input_shape.iter().copied().enumerate() {
			expanded[offset + axis] = extent;
		}
	}
	for axis in 0..output_shape.len() {
		if expanded[axis] != 1 && expanded[axis] != output_shape[axis] {
			return Err(Error::invalid_argument(
				"flow input is not broadcastable to the output shape",
			));
		}
	}
	let strides = dense_strides(&expanded[..output_shape.len()])?;
	let mut dims = [1_u32; 8];
	for axis in 0..output_shape.len() {
		dims[axis] = shader_u32(expanded[axis], "broadcast extent", "flow broadcast")?;
	}
	Ok((dims, strides))
}
