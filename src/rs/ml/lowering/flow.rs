use crate::runtime::{BufferBinding, KernelId, PushConstant};
use crate::{DType, Error, Matrix, OpAttribute, Result};

use super::common::{record_semantic, shader_u32, validate_f32_same_engine};

pub(in crate::ml) struct LinearMatchOutput {
	pub(in crate::ml) state: Matrix,
	pub(in crate::ml) velocity: Matrix,
}

pub(in crate::ml) struct MaskedMseOutput {
	pub(in crate::ml) loss: Matrix,
	pub(in crate::ml) denominator: Matrix,
}

pub(in crate::ml) fn linear_match(
	clean: &Matrix,
	noise: &Matrix,
	time: &Matrix,
) -> Result<LinearMatchOutput> {
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
	record_semantic(
		&[clean, noise, time],
		&[&state, &velocity],
		&[],
		KernelId::MlFlowLinearMatchF32,
		&buffers,
		&push_constants,
		KernelId::MlFlowLinearMatchF32.linear_workgroups(element_count),
	)?;
	Ok(LinearMatchOutput { state, velocity })
}

pub(in crate::ml) fn euler_step(
	state: &Matrix,
	velocity: &Matrix,
	delta_time: f32,
) -> Result<Matrix> {
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
	record_semantic(
		&[state, velocity],
		&[&output],
		&attributes,
		KernelId::MlFlowEulerStepF32,
		&buffers,
		&push_constants,
		KernelId::MlFlowEulerStepF32.linear_workgroups(element_count),
	)?;
	Ok(output)
}

pub(in crate::ml) fn masked_mse(
	prediction: &Matrix,
	target: &Matrix,
	mask: &Matrix,
) -> Result<MaskedMseOutput> {
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
	record_semantic(
		&[prediction, target, mask],
		&[&loss],
		&[],
		KernelId::MlFlowMaskedMseF32,
		&buffers,
		&push_constants,
		[1, 1, 1],
	)?;
	Ok(MaskedMseOutput { loss, denominator })
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
	record_semantic(
		&[prediction, target, mask, denominator, upstream],
		&[&gradient],
		&[],
		KernelId::MlFlowMaskedMseBackwardF32,
		&buffers,
		&push_constants,
		KernelId::MlFlowMaskedMseBackwardF32.linear_workgroups(element_count),
	)?;
	Ok(gradient)
}

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
