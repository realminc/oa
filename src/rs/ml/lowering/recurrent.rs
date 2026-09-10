//! Private lowering for recurrent ML operations.

use crate::runtime::{BufferBinding, ComputeDispatch, KernelId, PushConstant, SemanticDispatch};
use crate::{DType, Error, Matrix, OpAttribute, Result};

use super::common::{record_semantic, shader_u32, validate_f32_same_engine};

const MAX_HIDDEN_SIZE: usize = 1024;

pub(in crate::ml) struct RnnCellOutput {
	pub(in crate::ml) output: Matrix,
	pub(in crate::ml) gates_h: Matrix,
}

pub(in crate::ml) fn rnn_cell(
	gates_i: &Matrix,
	hidden: &Matrix,
	weight_hh: &Matrix,
	bias_hh: &Matrix,
	has_bias: bool,
) -> Result<RnnCellOutput> {
	const OPERATION: &str = crate::core::operation::ml::RNN_CELL.name();
	let [batch, hidden_size] = gates_i.shape() else {
		return Err(Error::invalid_argument(format!(
			"{OPERATION} input gates must have shape [B, H]"
		)));
	};
	if *batch == 0
		|| *hidden_size == 0
		|| *hidden_size > MAX_HIDDEN_SIZE
		|| hidden.shape() != gates_i.shape()
		|| weight_hh.shape() != [*hidden_size, *hidden_size]
		|| bias_hh.shape() != [*hidden_size]
	{
		return Err(Error::invalid_argument(format!(
			"{OPERATION} requires gates and hidden [B, H], recurrent weight [H, H], bias storage [H], and H <= {MAX_HIDDEN_SIZE}"
		)));
	}
	validate_f32_same_engine(OPERATION, &[gates_i, hidden, weight_hh, bias_hh])?;
	let output = Matrix::allocate(
		gates_i.engine_handle(),
		gates_i.shape().to_vec(),
		gates_i.num_elements(),
		DType::F32,
	)?;
	let gates_h = Matrix::allocate(
		gates_i.engine_handle(),
		gates_i.shape().to_vec(),
		gates_i.num_elements(),
		DType::F32,
	)?;
	let batch = shader_u32(*batch, "batch", OPERATION)?;
	let hidden_size = shader_u32(*hidden_size, "hidden size", OPERATION)?;
	let buffers = [
		BufferBinding::read(hidden.storage()),
		BufferBinding::read(weight_hh.storage()),
		BufferBinding::read(bias_hh.storage()),
		BufferBinding::read(gates_i.storage()),
		BufferBinding::write(output.storage()),
		BufferBinding::write(gates_h.storage()),
	];
	let push_constants = [
		PushConstant::U32(hidden_size),
		PushConstant::U32(batch),
		PushConstant::U32(u32::from(has_bias)),
	];
	let attributes = [OpAttribute::Boolean {
		name: "has_bias".into(),
		value: has_bias,
	}];
	record_semantic(
		&[gates_i, hidden, weight_hh, bias_hh],
		&[&output, &gates_h],
		&attributes,
		KernelId::MlRnnCellF32,
		&buffers,
		&push_constants,
		[batch, 1, 1],
	)?;
	Ok(RnnCellOutput { output, gates_h })
}

pub(in crate::ml) fn rnn_cell_backward(
	gates_i: &Matrix,
	gates_h: &Matrix,
	hidden: &Matrix,
	output_gradient: &Matrix,
	weight_hh: &Matrix,
	bias_hh: &Matrix,
	has_bias: bool,
) -> Result<(Matrix, Matrix, Matrix, Matrix)> {
	const OPERATION: &str = crate::core::operation::ml::RNN_CELL_BACKWARD.name();
	let [batch, hidden_size] = gates_i.shape() else {
		return Err(Error::invalid_argument(format!(
			"{OPERATION} input gates must have shape [B, H]"
		)));
	};
	if *batch == 0
		|| *hidden_size == 0
		|| *hidden_size > MAX_HIDDEN_SIZE
		|| gates_h.shape() != gates_i.shape()
		|| hidden.shape() != gates_i.shape()
		|| output_gradient.shape() != gates_i.shape()
		|| weight_hh.shape() != [*hidden_size, *hidden_size]
		|| bias_hh.shape() != [*hidden_size]
	{
		return Err(Error::invalid_argument(format!(
			"{OPERATION} saved values and output gradient do not match the RNN cell contract"
		)));
	}
	validate_f32_same_engine(
		OPERATION,
		&[
			gates_i,
			gates_h,
			hidden,
			output_gradient,
			weight_hh,
			bias_hh,
		],
	)?;
	let gates_i_gradient = Matrix::allocate(
		gates_i.engine_handle(),
		gates_i.shape().to_vec(),
		gates_i.num_elements(),
		DType::F32,
	)?;
	let gates_h_gradient = Matrix::allocate(
		gates_i.engine_handle(),
		gates_i.shape().to_vec(),
		gates_i.num_elements(),
		DType::F32,
	)?;
	let hidden_gradient = Matrix::allocate(
		gates_i.engine_handle(),
		hidden.shape().to_vec(),
		hidden.num_elements(),
		DType::F32,
	)?;
	let weight_gradient = Matrix::allocate(
		gates_i.engine_handle(),
		weight_hh.shape().to_vec(),
		weight_hh.num_elements(),
		DType::F32,
	)?;
	let bias_gradient = Matrix::allocate(
		gates_i.engine_handle(),
		bias_hh.shape().to_vec(),
		bias_hh.num_elements(),
		DType::F32,
	)?;
	let batch = shader_u32(*batch, "batch", OPERATION)?;
	let hidden_size = shader_u32(*hidden_size, "hidden size", OPERATION)?;
	let element_count = shader_u32(gates_i.num_elements(), "element count", OPERATION)?;
	let pointwise_buffers = [
		BufferBinding::read(gates_i.storage()),
		BufferBinding::read(gates_h.storage()),
		BufferBinding::read(output_gradient.storage()),
		BufferBinding::write(gates_i_gradient.storage()),
		BufferBinding::write(gates_h_gradient.storage()),
	];
	let pointwise_push_constants = [PushConstant::U32(element_count)];
	let input_buffers = [
		BufferBinding::read(weight_hh.storage()),
		BufferBinding::read(gates_h_gradient.storage()),
		BufferBinding::write(hidden_gradient.storage()),
	];
	let linear_push_constants = [
		PushConstant::U32(batch),
		PushConstant::U32(hidden_size),
		PushConstant::U32(hidden_size),
	];
	let parameter_buffers = [
		BufferBinding::read(hidden.storage()),
		BufferBinding::read(gates_h_gradient.storage()),
		BufferBinding::write(weight_gradient.storage()),
		BufferBinding::write(bias_gradient.storage()),
	];
	let dispatches = [
		ComputeDispatch {
			kernel: KernelId::MlRnnCellBackwardF32,
			buffers: &pointwise_buffers,
			push_constants: &pointwise_push_constants,
			workgroups: KernelId::MlRnnCellBackwardF32.linear_workgroups(element_count),
		},
		ComputeDispatch {
			kernel: KernelId::MlLinearBackwardF32,
			buffers: &input_buffers,
			push_constants: &linear_push_constants,
			workgroups: KernelId::MlLinearBackwardF32.output_workgroups(batch, hidden_size),
		},
		ComputeDispatch {
			kernel: KernelId::MlLinearParameterBackwardF32,
			buffers: &parameter_buffers,
			push_constants: &linear_push_constants,
			workgroups: [hidden_size, hidden_size.div_ceil(32), 1],
		},
	];
	let inputs = [
		gates_i,
		gates_h,
		hidden,
		output_gradient,
		weight_hh,
		bias_hh,
	];
	let outputs = [
		&gates_i_gradient,
		&hidden_gradient,
		&weight_gradient,
		&bias_gradient,
	];
	let attributes = [OpAttribute::Boolean {
		name: "has_bias".into(),
		value: has_bias,
	}];
	gates_i.engine_handle().record_split_semantic(
		&dispatches,
		SemanticDispatch {
			contract: crate::core::operation::ml::RNN_CELL_BACKWARD,
			inputs: &inputs,
			outputs: &outputs,
			attributes: &attributes,
		},
	)?;
	Ok((
		gates_i_gradient,
		hidden_gradient,
		weight_gradient,
		bias_gradient,
	))
}

pub(in crate::ml) struct RnnScanOutput {
	pub(in crate::ml) output: Matrix,
	pub(in crate::ml) hidden_previous: Matrix,
}

pub(in crate::ml) fn rnn_scan(
	gates_i: &Matrix,
	weight_hh: &Matrix,
	bias_hh: &Matrix,
	has_bias: bool,
) -> Result<RnnScanOutput> {
	const OPERATION: &str = crate::core::operation::ml::RNN_SCAN.name();
	let [batch, sequence_length, hidden_size] = gates_i.shape() else {
		return Err(Error::invalid_argument(format!(
			"{OPERATION} input gates must have shape [B, S, H]"
		)));
	};
	if *batch == 0
		|| *sequence_length == 0
		|| *hidden_size == 0
		|| *hidden_size > MAX_HIDDEN_SIZE
		|| weight_hh.shape() != [*hidden_size, *hidden_size]
		|| bias_hh.shape() != [*hidden_size]
	{
		return Err(Error::invalid_argument(format!(
			"{OPERATION} requires nonempty gates [B, S, H], recurrent weight [H, H], bias storage [H], and H <= {MAX_HIDDEN_SIZE}"
		)));
	}
	validate_f32_same_engine(OPERATION, &[gates_i, weight_hh, bias_hh])?;
	let output = Matrix::allocate(
		gates_i.engine_handle(),
		gates_i.shape().to_vec(),
		gates_i.num_elements(),
		DType::F32,
	)?;
	let hidden_previous = Matrix::allocate(
		gates_i.engine_handle(),
		gates_i.shape().to_vec(),
		gates_i.num_elements(),
		DType::F32,
	)?;
	let batch = shader_u32(*batch, "batch", OPERATION)?;
	let sequence_length = shader_u32(*sequence_length, "sequence length", OPERATION)?;
	let hidden_size = shader_u32(*hidden_size, "hidden size", OPERATION)?;
	let buffers = [
		BufferBinding::read(gates_i.storage()),
		BufferBinding::read(weight_hh.storage()),
		BufferBinding::read(bias_hh.storage()),
		BufferBinding::write(output.storage()),
		BufferBinding::write(hidden_previous.storage()),
	];
	let push_constants = [
		PushConstant::U32(hidden_size),
		PushConstant::U32(sequence_length),
		PushConstant::U32(batch),
		PushConstant::U32(u32::from(has_bias)),
	];
	let attributes = [OpAttribute::Boolean {
		name: "has_bias".into(),
		value: has_bias,
	}];
	record_semantic(
		&[gates_i, weight_hh, bias_hh],
		&[&output, &hidden_previous],
		&attributes,
		KernelId::MlRnnScanF32,
		&buffers,
		&push_constants,
		[batch, 1, 1],
	)?;
	Ok(RnnScanOutput {
		output,
		hidden_previous,
	})
}

pub(in crate::ml) fn rnn_scan_backward(
	output_gradient: &Matrix,
	gates_i: &Matrix,
	hidden_previous: &Matrix,
	weight_hh: &Matrix,
	bias_hh: &Matrix,
	has_bias: bool,
) -> Result<(Matrix, Matrix, Matrix)> {
	const OPERATION: &str = crate::core::operation::ml::RNN_SCAN_BACKWARD.name();
	let [batch, sequence_length, hidden_size] = gates_i.shape() else {
		return Err(Error::invalid_argument(format!(
			"{OPERATION} input gates must have shape [B, S, H]"
		)));
	};
	if *batch == 0
		|| *sequence_length == 0
		|| *hidden_size == 0
		|| *hidden_size > MAX_HIDDEN_SIZE
		|| hidden_previous.shape() != gates_i.shape()
		|| output_gradient.shape() != gates_i.shape()
		|| weight_hh.shape() != [*hidden_size, *hidden_size]
		|| bias_hh.shape() != [*hidden_size]
	{
		return Err(Error::invalid_argument(format!(
			"{OPERATION} saved values and output gradient do not match the RNN scan contract"
		)));
	}
	validate_f32_same_engine(
		OPERATION,
		&[
			output_gradient,
			gates_i,
			hidden_previous,
			weight_hh,
			bias_hh,
		],
	)?;
	let gates_i_gradient = Matrix::allocate(
		gates_i.engine_handle(),
		gates_i.shape().to_vec(),
		gates_i.num_elements(),
		DType::F32,
	)?;
	let gates_h_gradient = Matrix::allocate(
		gates_i.engine_handle(),
		gates_i.shape().to_vec(),
		gates_i.num_elements(),
		DType::F32,
	)?;
	let weight_gradient = Matrix::allocate(
		gates_i.engine_handle(),
		weight_hh.shape().to_vec(),
		weight_hh.num_elements(),
		DType::F32,
	)?;
	let bias_gradient = Matrix::allocate(
		gates_i.engine_handle(),
		bias_hh.shape().to_vec(),
		bias_hh.num_elements(),
		DType::F32,
	)?;
	let rows = batch
		.checked_mul(*sequence_length)
		.ok_or_else(|| Error::invalid_argument(format!("{OPERATION} row count overflows usize")))?;
	let batch = shader_u32(*batch, "batch", OPERATION)?;
	let sequence_length = shader_u32(*sequence_length, "sequence length", OPERATION)?;
	let hidden_size = shader_u32(*hidden_size, "hidden size", OPERATION)?;
	let rows = shader_u32(rows, "flattened row count", OPERATION)?;
	let scan_buffers = [
		BufferBinding::read(output_gradient.storage()),
		BufferBinding::read(gates_i.storage()),
		BufferBinding::read(hidden_previous.storage()),
		BufferBinding::read(weight_hh.storage()),
		BufferBinding::read(bias_hh.storage()),
		BufferBinding::write(gates_i_gradient.storage()),
		BufferBinding::write(gates_h_gradient.storage()),
	];
	let scan_push_constants = [
		PushConstant::U32(hidden_size),
		PushConstant::U32(sequence_length),
		PushConstant::U32(batch),
		PushConstant::U32(u32::from(has_bias)),
	];
	let parameter_buffers = [
		BufferBinding::read(hidden_previous.storage()),
		BufferBinding::read(gates_h_gradient.storage()),
		BufferBinding::write(weight_gradient.storage()),
		BufferBinding::write(bias_gradient.storage()),
	];
	let parameter_push_constants = [
		PushConstant::U32(rows),
		PushConstant::U32(hidden_size),
		PushConstant::U32(hidden_size),
	];
	let dispatches = [
		ComputeDispatch {
			kernel: KernelId::MlRnnScanBackwardF32,
			buffers: &scan_buffers,
			push_constants: &scan_push_constants,
			workgroups: [batch, 1, 1],
		},
		ComputeDispatch {
			kernel: KernelId::MlLinearParameterBackwardF32,
			buffers: &parameter_buffers,
			push_constants: &parameter_push_constants,
			workgroups: [hidden_size, hidden_size.div_ceil(32), 1],
		},
	];
	let inputs = [
		output_gradient,
		gates_i,
		hidden_previous,
		weight_hh,
		bias_hh,
	];
	let outputs = [&gates_i_gradient, &weight_gradient, &bias_gradient];
	let attributes = [OpAttribute::Boolean {
		name: "has_bias".into(),
		value: has_bias,
	}];
	gates_i.engine_handle().record_split_semantic(
		&dispatches,
		SemanticDispatch {
			contract: crate::core::operation::ml::RNN_SCAN_BACKWARD,
			inputs: &inputs,
			outputs: &outputs,
			attributes: &attributes,
		},
	)?;
	Ok((gates_i_gradient, weight_gradient, bias_gradient))
}

pub(in crate::ml) struct GruScanOutput {
	pub(in crate::ml) output: Matrix,
	pub(in crate::ml) hidden_previous: Matrix,
}

pub(in crate::ml) fn gru_scan(
	gates_i: &Matrix,
	weight_hh: &Matrix,
	bias_hh: &Matrix,
	has_bias: bool,
) -> Result<GruScanOutput> {
	const OPERATION: &str = crate::core::operation::ml::GRU_SCAN.name();
	let [batch, sequence_length, gate_count] = gates_i.shape() else {
		return Err(Error::invalid_argument(format!(
			"{OPERATION} input gates must have shape [B, S, 3H]"
		)));
	};
	let [weight_gate_count, hidden_size] = weight_hh.shape() else {
		return Err(Error::invalid_argument(format!(
			"{OPERATION} recurrent weight must have shape [3H, H]"
		)));
	};
	let expected_gate_count = hidden_size
		.checked_mul(3)
		.ok_or_else(|| Error::invalid_argument(format!("{OPERATION} gate size overflows usize")))?;
	if *batch == 0
		|| *sequence_length == 0
		|| *hidden_size == 0
		|| *hidden_size > MAX_HIDDEN_SIZE
		|| *gate_count != expected_gate_count
		|| *weight_gate_count != expected_gate_count
		|| bias_hh.shape() != [expected_gate_count]
	{
		return Err(Error::invalid_argument(format!(
			"{OPERATION} requires nonempty gates [B, S, 3H], recurrent weight [3H, H], bias storage [3H], and H <= {MAX_HIDDEN_SIZE}"
		)));
	}
	validate_f32_same_engine(OPERATION, &[gates_i, weight_hh, bias_hh])?;
	let output_count = batch
		.checked_mul(*sequence_length)
		.and_then(|count| count.checked_mul(*hidden_size))
		.ok_or_else(|| {
			Error::invalid_argument(format!("{OPERATION} output size overflows usize"))
		})?;
	let output_shape = vec![*batch, *sequence_length, *hidden_size];
	let output = Matrix::allocate(
		gates_i.engine_handle(),
		output_shape.clone(),
		output_count,
		DType::F32,
	)?;
	let hidden_previous = Matrix::allocate(
		gates_i.engine_handle(),
		output_shape,
		output_count,
		DType::F32,
	)?;
	let batch = shader_u32(*batch, "batch", OPERATION)?;
	let sequence_length = shader_u32(*sequence_length, "sequence length", OPERATION)?;
	let hidden_size = shader_u32(*hidden_size, "hidden size", OPERATION)?;
	let buffers = [
		BufferBinding::read(gates_i.storage()),
		BufferBinding::read(weight_hh.storage()),
		BufferBinding::read(bias_hh.storage()),
		BufferBinding::write(output.storage()),
		BufferBinding::write(hidden_previous.storage()),
	];
	let push_constants = [
		PushConstant::U32(hidden_size),
		PushConstant::U32(sequence_length),
		PushConstant::U32(batch),
		PushConstant::U32(u32::from(has_bias)),
	];
	let attributes = [OpAttribute::Boolean {
		name: "has_bias".into(),
		value: has_bias,
	}];
	record_semantic(
		&[gates_i, weight_hh, bias_hh],
		&[&output, &hidden_previous],
		&attributes,
		KernelId::MlGruScanF32,
		&buffers,
		&push_constants,
		[batch, 1, 1],
	)?;
	Ok(GruScanOutput {
		output,
		hidden_previous,
	})
}

pub(in crate::ml) fn gru_scan_backward(
	output_gradient: &Matrix,
	gates_i: &Matrix,
	hidden_previous: &Matrix,
	weight_hh: &Matrix,
	bias_hh: &Matrix,
	has_bias: bool,
) -> Result<(Matrix, Matrix, Matrix)> {
	const OPERATION: &str = crate::core::operation::ml::GRU_SCAN_BACKWARD.name();
	let [batch, sequence_length, gate_count] = gates_i.shape() else {
		return Err(Error::invalid_argument(format!(
			"{OPERATION} input gates must have shape [B, S, 3H]"
		)));
	};
	let [weight_gate_count, hidden_size] = weight_hh.shape() else {
		return Err(Error::invalid_argument(format!(
			"{OPERATION} recurrent weight must have shape [3H, H]"
		)));
	};
	let expected_gate_count = hidden_size
		.checked_mul(3)
		.ok_or_else(|| Error::invalid_argument(format!("{OPERATION} gate size overflows usize")))?;
	let output_shape = [*batch, *sequence_length, *hidden_size];
	if *batch == 0
		|| *sequence_length == 0
		|| *hidden_size == 0
		|| *hidden_size > MAX_HIDDEN_SIZE
		|| *gate_count != expected_gate_count
		|| *weight_gate_count != expected_gate_count
		|| bias_hh.shape() != [expected_gate_count]
		|| hidden_previous.shape() != output_shape
		|| output_gradient.shape() != output_shape
	{
		return Err(Error::invalid_argument(format!(
			"{OPERATION} saved values and output gradient do not match the GRU scan contract"
		)));
	}
	validate_f32_same_engine(
		OPERATION,
		&[
			output_gradient,
			gates_i,
			hidden_previous,
			weight_hh,
			bias_hh,
		],
	)?;
	let gates_i_gradient = Matrix::allocate(
		gates_i.engine_handle(),
		gates_i.shape().to_vec(),
		gates_i.num_elements(),
		DType::F32,
	)?;
	let gates_h_gradient = Matrix::allocate(
		gates_i.engine_handle(),
		gates_i.shape().to_vec(),
		gates_i.num_elements(),
		DType::F32,
	)?;
	let weight_gradient = Matrix::allocate(
		gates_i.engine_handle(),
		weight_hh.shape().to_vec(),
		weight_hh.num_elements(),
		DType::F32,
	)?;
	let bias_gradient = Matrix::allocate(
		gates_i.engine_handle(),
		bias_hh.shape().to_vec(),
		bias_hh.num_elements(),
		DType::F32,
	)?;
	let rows = batch
		.checked_mul(*sequence_length)
		.ok_or_else(|| Error::invalid_argument(format!("{OPERATION} row count overflows usize")))?;
	let batch = shader_u32(*batch, "batch", OPERATION)?;
	let sequence_length = shader_u32(*sequence_length, "sequence length", OPERATION)?;
	let hidden_size = shader_u32(*hidden_size, "hidden size", OPERATION)?;
	let gate_count = shader_u32(expected_gate_count, "gate count", OPERATION)?;
	let rows = shader_u32(rows, "flattened row count", OPERATION)?;
	let scan_buffers = [
		BufferBinding::read(output_gradient.storage()),
		BufferBinding::read(gates_i.storage()),
		BufferBinding::read(hidden_previous.storage()),
		BufferBinding::read(weight_hh.storage()),
		BufferBinding::read(bias_hh.storage()),
		BufferBinding::write(gates_i_gradient.storage()),
		BufferBinding::write(gates_h_gradient.storage()),
	];
	let scan_push_constants = [
		PushConstant::U32(hidden_size),
		PushConstant::U32(sequence_length),
		PushConstant::U32(batch),
		PushConstant::U32(u32::from(has_bias)),
	];
	let parameter_buffers = [
		BufferBinding::read(hidden_previous.storage()),
		BufferBinding::read(gates_h_gradient.storage()),
		BufferBinding::write(weight_gradient.storage()),
		BufferBinding::write(bias_gradient.storage()),
	];
	let parameter_push_constants = [
		PushConstant::U32(rows),
		PushConstant::U32(hidden_size),
		PushConstant::U32(gate_count),
	];
	let dispatches = [
		ComputeDispatch {
			kernel: KernelId::MlGruScanBackwardF32,
			buffers: &scan_buffers,
			push_constants: &scan_push_constants,
			workgroups: [batch, 1, 1],
		},
		ComputeDispatch {
			kernel: KernelId::MlLinearParameterBackwardF32,
			buffers: &parameter_buffers,
			push_constants: &parameter_push_constants,
			workgroups: [gate_count, hidden_size.div_ceil(32), 1],
		},
	];
	let inputs = [
		output_gradient,
		gates_i,
		hidden_previous,
		weight_hh,
		bias_hh,
	];
	let outputs = [&gates_i_gradient, &weight_gradient, &bias_gradient];
	let attributes = [OpAttribute::Boolean {
		name: "has_bias".into(),
		value: has_bias,
	}];
	gates_i.engine_handle().record_split_semantic(
		&dispatches,
		SemanticDispatch {
			contract: crate::core::operation::ml::GRU_SCAN_BACKWARD,
			inputs: &inputs,
			outputs: &outputs,
			attributes: &attributes,
		},
	)?;
	Ok((gates_i_gradient, weight_gradient, bias_gradient))
}

pub(in crate::ml) struct GruCellOutput {
	pub(in crate::ml) output: Matrix,
	pub(in crate::ml) gates_h: Matrix,
}

pub(in crate::ml) fn gru_cell(
	gates_i: &Matrix,
	hidden: &Matrix,
	weight_hh: &Matrix,
	bias_hh: &Matrix,
	has_bias: bool,
) -> Result<GruCellOutput> {
	const OPERATION: &str = crate::core::operation::ml::GRU_CELL.name();
	let [batch, gate_count] = gates_i.shape() else {
		return Err(Error::invalid_argument(format!(
			"{OPERATION} input gates must have shape [B, 3H]"
		)));
	};
	let [hidden_batch, hidden_size] = hidden.shape() else {
		return Err(Error::invalid_argument(format!(
			"{OPERATION} hidden state must have shape [B, H]"
		)));
	};
	let [weight_gate_count, weight_hidden_size] = weight_hh.shape() else {
		return Err(Error::invalid_argument(format!(
			"{OPERATION} recurrent weight must have shape [3H, H]"
		)));
	};
	let expected_gate_count = hidden_size
		.checked_mul(3)
		.ok_or_else(|| Error::invalid_argument(format!("{OPERATION} gate size overflows usize")))?;
	if *batch == 0
		|| *hidden_size == 0
		|| *hidden_size > MAX_HIDDEN_SIZE
		|| batch != hidden_batch
		|| *gate_count != expected_gate_count
		|| *weight_gate_count != expected_gate_count
		|| weight_hidden_size != hidden_size
		|| bias_hh.shape() != [expected_gate_count]
	{
		return Err(Error::invalid_argument(format!(
			"{OPERATION} requires gates [B, 3H], hidden [B, H], recurrent weight [3H, H], bias storage [3H], and H <= {MAX_HIDDEN_SIZE}"
		)));
	}
	validate_f32_same_engine(OPERATION, &[gates_i, hidden, weight_hh, bias_hh])?;
	let output = Matrix::allocate(
		gates_i.engine_handle(),
		hidden.shape().to_vec(),
		hidden.num_elements(),
		DType::F32,
	)?;
	let gates_h = Matrix::allocate(
		gates_i.engine_handle(),
		gates_i.shape().to_vec(),
		gates_i.num_elements(),
		DType::F32,
	)?;
	let batch = shader_u32(*batch, "batch", OPERATION)?;
	let hidden_size = shader_u32(*hidden_size, "hidden size", OPERATION)?;
	let buffers = [
		BufferBinding::read(hidden.storage()),
		BufferBinding::read(weight_hh.storage()),
		BufferBinding::read(bias_hh.storage()),
		BufferBinding::read(gates_i.storage()),
		BufferBinding::write(output.storage()),
		BufferBinding::write(gates_h.storage()),
	];
	let push_constants = [
		PushConstant::U32(hidden_size),
		PushConstant::U32(batch),
		PushConstant::U32(u32::from(has_bias)),
	];
	let attributes = [OpAttribute::Boolean {
		name: "has_bias".into(),
		value: has_bias,
	}];
	record_semantic(
		&[gates_i, hidden, weight_hh, bias_hh],
		&[&output, &gates_h],
		&attributes,
		KernelId::MlGruCellF32,
		&buffers,
		&push_constants,
		[batch, 1, 1],
	)?;
	Ok(GruCellOutput { output, gates_h })
}

pub(in crate::ml) fn gru_cell_backward(
	gates_i: &Matrix,
	gates_h: &Matrix,
	hidden: &Matrix,
	output_gradient: &Matrix,
	weight_hh: &Matrix,
	bias_hh: &Matrix,
	has_bias: bool,
) -> Result<(Matrix, Matrix, Matrix, Matrix)> {
	const OPERATION: &str = crate::core::operation::ml::GRU_CELL_BACKWARD.name();
	let [batch, gate_count] = gates_i.shape() else {
		return Err(Error::invalid_argument(format!(
			"{OPERATION} input gates must have shape [B, 3H]"
		)));
	};
	let [hidden_batch, hidden_size] = hidden.shape() else {
		return Err(Error::invalid_argument(format!(
			"{OPERATION} hidden state must have shape [B, H]"
		)));
	};
	let expected_gate_count = hidden_size
		.checked_mul(3)
		.ok_or_else(|| Error::invalid_argument(format!("{OPERATION} gate size overflows usize")))?;
	if *batch == 0
		|| *hidden_size == 0
		|| *hidden_size > MAX_HIDDEN_SIZE
		|| batch != hidden_batch
		|| *gate_count != expected_gate_count
		|| gates_h.shape() != gates_i.shape()
		|| output_gradient.shape() != hidden.shape()
		|| weight_hh.shape() != [expected_gate_count, *hidden_size]
		|| bias_hh.shape() != [expected_gate_count]
	{
		return Err(Error::invalid_argument(format!(
			"{OPERATION} saved values and output gradient do not match the GRU cell contract"
		)));
	}
	validate_f32_same_engine(
		OPERATION,
		&[
			gates_i,
			gates_h,
			hidden,
			output_gradient,
			weight_hh,
			bias_hh,
		],
	)?;
	let gates_i_gradient = Matrix::allocate(
		gates_i.engine_handle(),
		gates_i.shape().to_vec(),
		gates_i.num_elements(),
		DType::F32,
	)?;
	let gates_h_gradient = Matrix::allocate(
		gates_i.engine_handle(),
		gates_i.shape().to_vec(),
		gates_i.num_elements(),
		DType::F32,
	)?;
	let direct_hidden_gradient = Matrix::allocate(
		gates_i.engine_handle(),
		hidden.shape().to_vec(),
		hidden.num_elements(),
		DType::F32,
	)?;
	let recurrent_hidden_gradient = Matrix::allocate(
		gates_i.engine_handle(),
		hidden.shape().to_vec(),
		hidden.num_elements(),
		DType::F32,
	)?;
	let hidden_gradient = Matrix::allocate(
		gates_i.engine_handle(),
		hidden.shape().to_vec(),
		hidden.num_elements(),
		DType::F32,
	)?;
	let weight_gradient = Matrix::allocate(
		gates_i.engine_handle(),
		weight_hh.shape().to_vec(),
		weight_hh.num_elements(),
		DType::F32,
	)?;
	let bias_gradient = Matrix::allocate(
		gates_i.engine_handle(),
		bias_hh.shape().to_vec(),
		bias_hh.num_elements(),
		DType::F32,
	)?;
	let batch = shader_u32(*batch, "batch", OPERATION)?;
	let hidden_size = shader_u32(*hidden_size, "hidden size", OPERATION)?;
	let gate_count = shader_u32(expected_gate_count, "gate count", OPERATION)?;
	let element_count = shader_u32(hidden.num_elements(), "hidden element count", OPERATION)?;
	let pointwise_buffers = [
		BufferBinding::read(gates_i.storage()),
		BufferBinding::read(gates_h.storage()),
		BufferBinding::read(hidden.storage()),
		BufferBinding::read(output_gradient.storage()),
		BufferBinding::write(gates_i_gradient.storage()),
		BufferBinding::write(gates_h_gradient.storage()),
		BufferBinding::write(direct_hidden_gradient.storage()),
	];
	let pointwise_push_constants = [
		PushConstant::U32(hidden_size),
		PushConstant::U32(element_count),
	];
	let input_buffers = [
		BufferBinding::read(weight_hh.storage()),
		BufferBinding::read(gates_h_gradient.storage()),
		BufferBinding::write(recurrent_hidden_gradient.storage()),
	];
	let linear_push_constants = [
		PushConstant::U32(batch),
		PushConstant::U32(hidden_size),
		PushConstant::U32(gate_count),
	];
	let parameter_buffers = [
		BufferBinding::read(hidden.storage()),
		BufferBinding::read(gates_h_gradient.storage()),
		BufferBinding::write(weight_gradient.storage()),
		BufferBinding::write(bias_gradient.storage()),
	];
	let add_buffers = [
		BufferBinding::read(direct_hidden_gradient.storage()),
		BufferBinding::read(recurrent_hidden_gradient.storage()),
		BufferBinding::write(hidden_gradient.storage()),
	];
	let add_push_constants = [PushConstant::U32(element_count)];
	let dispatches = [
		ComputeDispatch {
			kernel: KernelId::MlGruCellBackwardF32,
			buffers: &pointwise_buffers,
			push_constants: &pointwise_push_constants,
			workgroups: KernelId::MlGruCellBackwardF32.linear_workgroups(element_count),
		},
		ComputeDispatch {
			kernel: KernelId::MlLinearBackwardF32,
			buffers: &input_buffers,
			push_constants: &linear_push_constants,
			workgroups: KernelId::MlLinearBackwardF32.output_workgroups(batch, hidden_size),
		},
		ComputeDispatch {
			kernel: KernelId::MlLinearParameterBackwardF32,
			buffers: &parameter_buffers,
			push_constants: &linear_push_constants,
			workgroups: [gate_count, hidden_size.div_ceil(32), 1],
		},
		ComputeDispatch {
			kernel: KernelId::MatrixAddF32,
			buffers: &add_buffers,
			push_constants: &add_push_constants,
			workgroups: KernelId::MatrixAddF32.linear_workgroups(element_count),
		},
	];
	let inputs = [
		gates_i,
		gates_h,
		hidden,
		output_gradient,
		weight_hh,
		bias_hh,
	];
	let outputs = [
		&gates_i_gradient,
		&hidden_gradient,
		&weight_gradient,
		&bias_gradient,
	];
	let attributes = [OpAttribute::Boolean {
		name: "has_bias".into(),
		value: has_bias,
	}];
	gates_i.engine_handle().record_split_semantic(
		&dispatches,
		SemanticDispatch {
			contract: crate::core::operation::ml::GRU_CELL_BACKWARD,
			inputs: &inputs,
			outputs: &outputs,
			attributes: &attributes,
		},
	)?;
	Ok((
		gates_i_gradient,
		hidden_gradient,
		weight_gradient,
		bias_gradient,
	))
}
