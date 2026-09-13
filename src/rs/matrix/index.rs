//! Matrix selection and sparse-routing operations.

use crate::{
	DType, Error, Matrix, OpAttribute, OperationContract, Result,
	core::autograd::{self, MatrixNode},
	runtime::{BufferBinding, ComputeDispatch, KernelId, PushConstant, SemanticDispatch},
};

const TOP_K_LIMIT: usize = 1024;
const EXPERT_LIMIT: usize = 256;

/// Gather rows from an FP32 table using arbitrary-shape integer indices.
///
/// The table must be `[rows,width]`; indices are flattened and output is
/// `[num_indices,width]`, matching OA C++ `FnMatrix::gather`. U8, U32, and I32
/// indices use one bounded kernel. Invalid rows produce NaN without reading
/// outside the table.
///
/// # Errors
///
/// Returns an error for incompatible rank, dtype, Engine ownership, shader ABI,
/// allocation, or recording.
pub fn gather(input: &Matrix, indices: &Matrix) -> Result<Matrix> {
	const CONTRACT: OperationContract = crate::core::operation::matrix::GATHER;
	let operation = CONTRACT.name();
	let [rows, width] = input.shape() else {
		return Err(Error::invalid_argument(format!(
			"{operation} input must have shape [rows,width]"
		)));
	};
	if *rows == 0
		|| *width == 0
		|| input.dtype() != DType::F32
		|| indices.shape().is_empty()
		|| !matches!(indices.dtype(), DType::U8 | DType::U32 | DType::I32)
		|| !input.engine_handle().same_as(indices.engine_handle())
	{
		return Err(Error::invalid_argument(format!(
			"{operation} requires a nonempty FP32 [rows,width] table and same-engine U8/U32/I32 indices"
		)));
	}
	let count = indices.num_elements();
	let output_count = count.checked_mul(*width).ok_or_else(|| {
		Error::invalid_argument(format!("{operation} output size overflows usize"))
	})?;
	let output = Matrix::allocate(
		input.engine_handle(),
		vec![count, *width],
		output_count,
		DType::F32,
	)?;
	let output_count_u32 = as_u32(output_count, operation, "output element count")?;
	if output_count_u32 != 0 {
		let buffers = [
			BufferBinding::read(input.storage()),
			BufferBinding::read(indices.storage()),
			BufferBinding::write(output.storage()),
		];
		let push_constants = [
			PushConstant::U32(as_u32(count, operation, "index count")?),
			PushConstant::U32(as_u32(*rows, operation, "table row count")?),
			PushConstant::U32(as_u32(*width, operation, "table row width")?),
			PushConstant::U32(u32::from(indices.dtype() != DType::U8)),
		];
		input.engine_handle().record_semantic(
			ComputeDispatch {
				kernel: KernelId::MatrixGatherF32,
				buffers: &buffers,
				push_constants: &push_constants,
				workgroups: KernelId::MatrixGatherF32.linear_workgroups(output_count_u32),
			},
			SemanticDispatch {
				contract: CONTRACT,
				inputs: &[input, indices],
				outputs: &[&output],
				attributes: &[],
			},
		)?;
	}
	autograd::record(MatrixNode::Gather {
		input: input.clone(),
		indices: indices.clone(),
		output_id: output.value_id(),
	})?;
	Ok(output)
}

pub(crate) fn gather_backward(
	input_shape: &[usize],
	indices: &Matrix,
	output_gradient: &Matrix,
) -> Result<Matrix> {
	const CONTRACT: OperationContract = crate::core::operation::matrix::GATHER_BACKWARD;
	let operation = CONTRACT.name();
	let [rows, width] = input_shape else {
		return Err(Error::invalid_argument(format!(
			"{operation} input shape must be [rows,width]"
		)));
	};
	let count = indices.num_elements();
	if *rows == 0
		|| *width == 0
		|| indices.shape().is_empty()
		|| !matches!(indices.dtype(), DType::U8 | DType::U32 | DType::I32)
		|| output_gradient.dtype() != DType::F32
		|| output_gradient.shape() != [count, *width]
		|| !indices
			.engine_handle()
			.same_as(output_gradient.engine_handle())
	{
		return Err(Error::invalid_argument(format!(
			"{operation} requires integer indices and matching FP32 [num_indices,width] gradient"
		)));
	}
	let gradient = Matrix::allocate(
		output_gradient.engine_handle(),
		input_shape.to_vec(),
		rows.checked_mul(*width)
			.ok_or_else(|| Error::invalid_argument(format!("{operation} size overflows usize")))?,
		DType::F32,
	)?;
	let rows_u32 = as_u32(*rows, operation, "table row count")?;
	let buffers = [
		BufferBinding::read(indices.storage()),
		BufferBinding::read(output_gradient.storage()),
		BufferBinding::write(gradient.storage()),
	];
	let push_constants = [
		PushConstant::U32(as_u32(count, operation, "index count")?),
		PushConstant::U32(rows_u32),
		PushConstant::U32(as_u32(*width, operation, "table row width")?),
		PushConstant::U32(u32::from(indices.dtype() != DType::U8)),
	];
	let attributes = [OpAttribute::Shape {
		name: "input_shape".into(),
		value: input_shape.to_vec(),
	}];
	output_gradient.engine_handle().record_semantic(
		ComputeDispatch {
			kernel: KernelId::MatrixGatherBackwardF32,
			buffers: &buffers,
			push_constants: &push_constants,
			workgroups: [rows_u32, 1, 1],
		},
		SemanticDispatch {
			contract: CONTRACT,
			inputs: &[indices, output_gradient],
			outputs: &[&gradient],
			attributes: &attributes,
		},
	)?;
	Ok(gradient)
}

/// Largest values and exact I32 source indices returned by [`top_k`].
#[must_use]
pub struct TopKResult {
	/// Values in descending order along the selected last axis.
	pub values: Matrix,
	/// I32 source indices. Equal values select the lower source index first.
	pub indices: Matrix,
}

/// Device-resident stable dropless route plan returned by [`moe_expert_plan`].
#[must_use]
pub struct MoeExpertPlan {
	/// U32 route count for each expert, shaped `[E]`.
	pub counts: Matrix,
	/// U32 exclusive expert offsets, shaped `[E + 1]`.
	pub offsets: Matrix,
	/// U32 source-token index for each expert-major packed route.
	pub packed_token: Matrix,
	/// U32 expert index for each expert-major packed route.
	pub packed_expert: Matrix,
	/// U32 original token-major flat route index for each packed route.
	pub packed_slot: Matrix,
	/// U32 packed position for each original token-major route.
	pub inverse: Matrix,
}

/// Compare every FP32 element with one scalar and return an FP32 zero/one mask.
///
/// Equality follows the device's ordinary IEEE comparison: NaN never compares
/// equal, while positive and negative zero compare equal. The result is
/// intentionally detached because a discrete mask has no admitted adjoint.
///
/// # Errors
///
/// Returns an error when `input` is not FP32, its element count exceeds the
/// shader ABI, allocation fails, or runtime recording fails.
pub fn equal(input: &Matrix, value: f32) -> Result<Matrix> {
	const CONTRACT: OperationContract = crate::core::operation::matrix::EQUAL;
	let operation = CONTRACT.name();
	if input.dtype() != DType::F32 {
		return Err(Error::invalid_argument(format!(
			"{operation} requires an F32 input"
		)));
	}
	let element_count = u32::try_from(input.element_count())
		.map_err(|_| Error::invalid_argument(format!("{operation} element count exceeds u32")))?;
	let output = Matrix::allocate(
		input.engine_handle(),
		input.shape().to_vec(),
		input.element_count(),
		DType::F32,
	)?;
	if element_count != 0 {
		let buffers = [
			BufferBinding::read(input.storage()),
			BufferBinding::write(output.storage()),
		];
		let push_constants = [PushConstant::U32(element_count), PushConstant::F32(value)];
		let inputs = [input];
		let outputs = [&output];
		let attributes = [OpAttribute::Float {
			name: "value".into(),
			value: f64::from(value),
		}];
		input.engine_handle().record_semantic(
			ComputeDispatch {
				kernel: KernelId::MatrixEqualF32,
				buffers: &buffers,
				push_constants: &push_constants,
				workgroups: KernelId::MatrixEqualF32.linear_workgroups(element_count),
			},
			SemanticDispatch {
				contract: CONTRACT,
				inputs: &inputs,
				outputs: &outputs,
				attributes: &attributes,
			},
		)?;
	}
	Ok(output)
}

/// Return the largest `k` FP32 values and their I32 indices on the last axis.
///
/// Rank-one and rank-two inputs are admitted. `dim` may be `-1` or the exact
/// last-axis index. Values are sorted descending; ties select lower indices.
/// `k` is clamped to the last-axis extent, matching OA C++.
///
/// # Errors
///
/// Returns an error for an unsupported rank, dtype, axis, negative `k`, a
/// request above the admitted 1024-entry GPU limit, an empty last axis, size
/// overflow, allocation failure, or runtime recording failure.
pub fn top_k(input: &Matrix, k: i32, dim: i32) -> Result<TopKResult> {
	const CONTRACT: OperationContract = crate::core::operation::matrix::TOP_K;
	let operation = CONTRACT.name();
	if input.dtype() != DType::F32 || !(1..=2).contains(&input.shape().len()) {
		return Err(Error::invalid_argument(format!(
			"{operation} requires a rank-one or rank-two F32 matrix"
		)));
	}
	let last_dim = i32::try_from(input.shape().len() - 1)
		.map_err(|_| Error::invalid_argument(format!("{operation} rank exceeds i32")))?;
	if dim != -1 && dim != last_dim {
		return Err(Error::invalid_argument(format!(
			"{operation} supports only the last axis; found dim {dim}"
		)));
	}
	let requested = usize::try_from(k)
		.map_err(|_| Error::invalid_argument(format!("{operation} k must be nonnegative")))?;
	let row_width = *input.shape().last().expect("validated nonempty rank");
	if row_width == 0 {
		return Err(Error::invalid_argument(format!(
			"{operation} requires a nonempty last axis"
		)));
	}
	let selected = requested.min(row_width);
	if selected > TOP_K_LIMIT {
		return Err(Error::invalid_argument(format!(
			"{operation} supports at most {TOP_K_LIMIT} selected values"
		)));
	}
	let rows = if input.shape().len() == 1 {
		1
	} else {
		input.shape()[0]
	};
	let output_count = rows.checked_mul(selected).ok_or_else(|| {
		Error::invalid_argument(format!("{operation} output size overflows usize"))
	})?;
	let mut output_shape = input.shape().to_vec();
	*output_shape.last_mut().expect("validated nonempty rank") = selected;
	let values = Matrix::allocate(
		input.engine_handle(),
		output_shape.clone(),
		output_count,
		DType::F32,
	)?;
	let indices = Matrix::allocate(
		input.engine_handle(),
		output_shape,
		output_count,
		DType::I32,
	)?;
	if selected != 0 && rows != 0 {
		let rows = as_u32(rows, operation, "row count")?;
		let row_width = as_u32(row_width, operation, "last-axis extent")?;
		let selected = as_u32(selected, operation, "k")?;
		let buffers = [
			BufferBinding::read(input.storage()),
			BufferBinding::write(values.storage()),
			BufferBinding::write(indices.storage()),
		];
		let push_constants = [
			PushConstant::U32(rows),
			PushConstant::U32(row_width),
			PushConstant::U32(selected),
		];
		let inputs = [input];
		let outputs = [&values, &indices];
		let attributes = [
			OpAttribute::SignedInteger {
				name: "k".into(),
				value: i64::from(k),
			},
			OpAttribute::SignedInteger {
				name: "dim".into(),
				value: i64::from(dim),
			},
		];
		input.engine_handle().record_semantic(
			ComputeDispatch {
				kernel: KernelId::MatrixTopKF32,
				buffers: &buffers,
				push_constants: &push_constants,
				workgroups: [rows, 1, 1],
			},
			SemanticDispatch {
				contract: CONTRACT,
				inputs: &inputs,
				outputs: &outputs,
				attributes: &attributes,
			},
		)?;
	}
	Ok(TopKResult { values, indices })
}

/// Expand I32 `[T, K]` expert indices into an exact FP32 `[T, E]` mask.
///
/// # Errors
///
/// Returns an error unless the input is nonempty rank-two I32, `num_experts`
/// is positive, all sizes fit the shader ABI, or runtime recording fails.
pub fn top_k_mask(indices: &Matrix, num_experts: usize) -> Result<Matrix> {
	const CONTRACT: OperationContract = crate::core::operation::matrix::TOP_K_MASK;
	let operation = CONTRACT.name();
	let [tokens, routes_per_token] = indices.shape() else {
		return Err(Error::invalid_argument(format!(
			"{operation} requires rank-two I32 indices"
		)));
	};
	if indices.dtype() != DType::I32 || *tokens == 0 || *routes_per_token == 0 {
		return Err(Error::invalid_argument(format!(
			"{operation} requires nonempty rank-two I32 indices"
		)));
	}
	if num_experts == 0 {
		return Err(Error::invalid_argument(format!(
			"{operation} requires at least one expert"
		)));
	}
	let output_count = tokens.checked_mul(num_experts).ok_or_else(|| {
		Error::invalid_argument(format!("{operation} output size overflows usize"))
	})?;
	let tokens_u32 = as_u32(*tokens, operation, "token count")?;
	let experts_u32 = as_u32(num_experts, operation, "expert count")?;
	let routes_u32 = as_u32(*routes_per_token, operation, "routes per token")?;
	let output_count_u32 = as_u32(output_count, operation, "output element count")?;
	let output = Matrix::allocate(
		indices.engine_handle(),
		vec![*tokens, num_experts],
		output_count,
		DType::F32,
	)?;
	let buffers = [
		BufferBinding::read(indices.storage()),
		BufferBinding::write(output.storage()),
	];
	let push_constants = [
		PushConstant::U32(tokens_u32),
		PushConstant::U32(experts_u32),
		PushConstant::U32(routes_u32),
	];
	let inputs = [indices];
	let outputs = [&output];
	let attributes = [OpAttribute::SignedInteger {
		name: "num_experts".into(),
		value: i64::try_from(num_experts).map_err(|_| {
			Error::invalid_argument(format!("{operation} expert count exceeds i64"))
		})?,
	}];
	indices.engine_handle().record_semantic(
		ComputeDispatch {
			kernel: KernelId::MatrixTopKMaskF32,
			buffers: &buffers,
			push_constants: &push_constants,
			workgroups: KernelId::MatrixTopKMaskF32.linear_workgroups(output_count_u32),
		},
		SemanticDispatch {
			contract: CONTRACT,
			inputs: &inputs,
			outputs: &outputs,
			attributes: &attributes,
		},
	)?;
	Ok(output)
}

/// Build a stable dropless expert-major plan from I32 `[T, K]` indices.
///
/// Within every expert, routes preserve their original token-major order. The
/// operation accepts at most 256 experts, matching the shader's shared state.
///
/// # Errors
///
/// Returns an error unless indices are nonempty rank-two I32, the expert count
/// is in `1..=256`, sizes fit the admitted ABI, allocation succeeds, or runtime
/// recording succeeds.
pub fn moe_expert_plan(indices: &Matrix, num_experts: usize) -> Result<MoeExpertPlan> {
	const CONTRACT: OperationContract = crate::core::operation::matrix::MOE_EXPERT_PLAN;
	let operation = CONTRACT.name();
	let [tokens, routes_per_token] = indices.shape() else {
		return Err(Error::invalid_argument(format!(
			"{operation} requires rank-two I32 indices"
		)));
	};
	if indices.dtype() != DType::I32 || *tokens == 0 || *routes_per_token == 0 {
		return Err(Error::invalid_argument(format!(
			"{operation} requires nonempty rank-two I32 indices"
		)));
	}
	if !(1..=EXPERT_LIMIT).contains(&num_experts) {
		return Err(Error::invalid_argument(format!(
			"{operation} expert count must be in 1..={EXPERT_LIMIT}"
		)));
	}
	let route_count = tokens.checked_mul(*routes_per_token).ok_or_else(|| {
		Error::invalid_argument(format!("{operation} route count overflows usize"))
	})?;
	let offsets_count = num_experts.checked_add(1).ok_or_else(|| {
		Error::invalid_argument(format!("{operation} offset count overflows usize"))
	})?;
	let tokens_u32 = as_u32(*tokens, operation, "token count")?;
	let routes_per_token_u32 = as_u32(*routes_per_token, operation, "routes per token")?;
	let experts_u32 = as_u32(num_experts, operation, "expert count")?;
	let route_count_u32 = as_u32(route_count, operation, "route count")?;
	let engine = indices.engine_handle();
	let counts = Matrix::allocate(engine, vec![num_experts], num_experts, DType::U32)?;
	let offsets = Matrix::allocate(engine, vec![offsets_count], offsets_count, DType::U32)?;
	let packed_token = Matrix::allocate(engine, vec![route_count], route_count, DType::U32)?;
	let packed_expert = Matrix::allocate(engine, vec![route_count], route_count, DType::U32)?;
	let packed_slot = Matrix::allocate(engine, vec![route_count], route_count, DType::U32)?;
	let inverse = Matrix::allocate(engine, vec![route_count], route_count, DType::U32)?;
	let buffers = [
		BufferBinding::read(indices.storage()),
		BufferBinding::write(counts.storage()),
		BufferBinding::write(offsets.storage()),
		BufferBinding::write(packed_token.storage()),
		BufferBinding::write(packed_expert.storage()),
		BufferBinding::write(packed_slot.storage()),
		BufferBinding::write(inverse.storage()),
	];
	let push_constants = [
		PushConstant::U32(tokens_u32),
		PushConstant::U32(routes_per_token_u32),
		PushConstant::U32(experts_u32),
		PushConstant::U32(route_count_u32),
	];
	let inputs = [indices];
	let outputs = [
		&counts,
		&offsets,
		&packed_token,
		&packed_expert,
		&packed_slot,
		&inverse,
	];
	let attributes = [OpAttribute::SignedInteger {
		name: "num_experts".into(),
		value: i64::try_from(num_experts).expect("expert limit fits i64"),
	}];
	engine.record_semantic(
		ComputeDispatch {
			kernel: KernelId::MatrixMoeExpertPlanU32,
			buffers: &buffers,
			push_constants: &push_constants,
			workgroups: [1, 1, 1],
		},
		SemanticDispatch {
			contract: CONTRACT,
			inputs: &inputs,
			outputs: &outputs,
			attributes: &attributes,
		},
	)?;
	Ok(MoeExpertPlan {
		counts,
		offsets,
		packed_token,
		packed_expert,
		packed_slot,
		inverse,
	})
}

/// Update an FP32 expert-routing bias from a dense FP32 selection mask.
///
/// `selection_mask` is `[T, E]`; `bias` may be `[E]` or `[1, E]`. Each expert
/// receives `-gamma`, `+gamma`, or no update when its observed route count is
/// above, below, or equal to the balanced target `T * K / E`. The update stays
/// device-resident and mutates `bias` through its existing value identity.
///
/// # Errors
///
/// Returns an error unless both matrices are FP32 on the same engine, the mask
/// and bias shapes agree, `experts_per_token` is in `1..=E`, `gamma` is finite
/// and positive, dimensions fit the shader ABI, or runtime recording fails.
pub fn moe_routing_bias_update(
	selection_mask: &Matrix,
	bias: &Matrix,
	experts_per_token: usize,
	gamma: f32,
) -> Result<()> {
	const CONTRACT: OperationContract = crate::core::operation::matrix::MOE_ROUTING_BIAS_UPDATE;
	let operation = CONTRACT.name();
	let [tokens, experts] = selection_mask.shape() else {
		return Err(Error::invalid_argument(format!(
			"{operation} requires a rank-two selection mask"
		)));
	};
	if selection_mask.dtype() != DType::F32 || bias.dtype() != DType::F32 {
		return Err(Error::invalid_argument(format!(
			"{operation} requires an F32 selection mask and bias"
		)));
	}
	if *tokens == 0 || *experts == 0 || bias.num_elements() != *experts {
		return Err(Error::invalid_argument(format!(
			"{operation} requires nonempty [T, E] mask and E-element bias"
		)));
	}
	if !matches!(bias.shape(), [bias_experts] if bias_experts == experts)
		&& !matches!(bias.shape(), [one, bias_experts] if *one == 1 && bias_experts == experts)
	{
		return Err(Error::invalid_argument(format!(
			"{operation} bias must have shape [E] or [1, E]"
		)));
	}
	if !(1..=*experts).contains(&experts_per_token) {
		return Err(Error::invalid_argument(format!(
			"{operation} experts_per_token must be in 1..=E"
		)));
	}
	if !gamma.is_finite() || gamma <= 0.0 {
		return Err(Error::invalid_argument(format!(
			"{operation} gamma must be finite and positive"
		)));
	}
	let engine = selection_mask.engine_handle();
	if !engine.same_as(bias.engine_handle()) {
		return Err(Error::invalid_argument(format!(
			"{operation} inputs must belong to the same engine"
		)));
	}
	let tokens_u32 = as_u32(*tokens, operation, "token count")?;
	let experts_u32 = as_u32(*experts, operation, "expert count")?;
	let experts_per_token_u32 = as_u32(experts_per_token, operation, "experts per token")?;
	let buffers = [
		BufferBinding::read(selection_mask.storage()),
		BufferBinding::read_write(bias.storage()),
	];
	let push_constants = [
		PushConstant::U32(tokens_u32),
		PushConstant::U32(experts_u32),
		PushConstant::U32(experts_per_token_u32),
		PushConstant::F32(gamma),
	];
	let inputs = [selection_mask, bias];
	let outputs = [bias];
	let attributes = [
		OpAttribute::SignedInteger {
			name: "experts_per_token".into(),
			value: i64::try_from(experts_per_token).map_err(|_| {
				Error::invalid_argument(format!("{operation} experts_per_token exceeds i64"))
			})?,
		},
		OpAttribute::Float {
			name: "gamma".into(),
			value: f64::from(gamma),
		},
	];
	engine.record_semantic(
		ComputeDispatch {
			kernel: KernelId::MatrixMoeRoutingBiasUpdateF32,
			buffers: &buffers,
			push_constants: &push_constants,
			workgroups: KernelId::MatrixMoeRoutingBiasUpdateF32.linear_workgroups(experts_u32),
		},
		SemanticDispatch {
			contract: CONTRACT,
			inputs: &inputs,
			outputs: &outputs,
			attributes: &attributes,
		},
	)
}

/// Repeat every FP32 element along one matrix axis.
///
/// This is the direct Rust spelling of OA C++ `FnMatrix::repeatInterleave`.
/// Rank-one through rank-four matrices are currently admitted.
///
/// # Errors
///
/// Returns an error for an unsupported rank or dtype, non-positive repeat
/// count, invalid axis, shape/ABI overflow, allocation failure, or recording
/// failure.
pub fn repeat_interleave(input: &Matrix, repeats: usize, dim: i32) -> Result<Matrix> {
	const CONTRACT: OperationContract = crate::core::operation::matrix::REPEAT_INTERLEAVE;
	let operation = CONTRACT.name();
	let rank = input.shape().len();
	if input.dtype() != DType::F32 || !(1..=4).contains(&rank) {
		return Err(Error::invalid_argument(format!(
			"{operation} requires a rank-one through rank-four F32 matrix"
		)));
	}
	if repeats == 0 {
		return Err(Error::invalid_argument(format!(
			"{operation} repeats must be positive"
		)));
	}
	let repeats_u32 = as_u32(repeats, operation, "repeat count")?;
	let dim = usize::try_from(dim)
		.map_err(|_| Error::invalid_argument(format!("{operation} dim must be nonnegative")))?;
	if dim >= rank {
		return Err(Error::invalid_argument(format!(
			"{operation} dim {dim} is outside rank {rank}"
		)));
	}
	let mut output_shape = input.shape().to_vec();
	output_shape[dim] = output_shape[dim]
		.checked_mul(repeats)
		.ok_or_else(|| Error::invalid_argument(format!("{operation} shape overflows usize")))?;
	let output_count = output_shape.iter().try_fold(1_usize, |count, extent| {
		count.checked_mul(*extent).ok_or_else(|| {
			Error::invalid_argument(format!("{operation} output size overflows usize"))
		})
	})?;
	let count = as_u32(output_count, operation, "output element count")?;
	let output = Matrix::allocate(
		input.engine_handle(),
		output_shape,
		output_count,
		DType::F32,
	)?;
	if count != 0 {
		let input_dims = shape_u32(input.shape(), operation)?;
		let input_strides = row_major_strides_u32(input.shape(), operation)?;
		let buffers = [
			BufferBinding::read(input.storage()),
			BufferBinding::write(output.storage()),
		];
		let mut push_constants = vec![
			PushConstant::U32(count),
			PushConstant::U32(repeats_u32),
			PushConstant::U32(dim as u32),
			PushConstant::U32(rank as u32),
		];
		push_constants.extend(input_dims.map(PushConstant::U32));
		push_constants.extend(input_strides.map(PushConstant::U32));
		let inputs = [input];
		let outputs = [&output];
		let attributes = [
			OpAttribute::SignedInteger {
				name: "repeats".into(),
				value: i64::from(repeats_u32),
			},
			OpAttribute::SignedInteger {
				name: "dim".into(),
				value: dim as i64,
			},
		];
		input.engine_handle().record_semantic(
			ComputeDispatch {
				kernel: KernelId::MatrixRepeatInterleaveF32,
				buffers: &buffers,
				push_constants: &push_constants,
				workgroups: KernelId::MatrixRepeatInterleaveF32.linear_workgroups(count),
			},
			SemanticDispatch {
				contract: CONTRACT,
				inputs: &inputs,
				outputs: &outputs,
				attributes: &attributes,
			},
		)?;
	}
	autograd::record(MatrixNode::RepeatInterleave {
		input: input.clone(),
		output_id: output.value_id(),
		repeats,
		dim,
	})?;
	Ok(output)
}

pub(crate) fn repeat_interleave_backward(
	input_shape: &[usize],
	repeats: usize,
	dim: usize,
	output_gradient: &Matrix,
) -> Result<Matrix> {
	const CONTRACT: OperationContract = crate::core::operation::matrix::REPEAT_INTERLEAVE_BACKWARD;
	let operation = CONTRACT.name();
	let rank = input_shape.len();
	if output_gradient.dtype() != DType::F32
		|| !(1..=4).contains(&rank)
		|| repeats == 0
		|| dim >= rank
	{
		return Err(Error::invalid_argument(format!(
			"{operation} received invalid saved repeat geometry"
		)));
	}
	let repeats_u32 = as_u32(repeats, operation, "repeat count")?;
	let mut expanded_shape = input_shape.to_vec();
	expanded_shape[dim] = expanded_shape[dim]
		.checked_mul(repeats)
		.ok_or_else(|| Error::invalid_argument(format!("{operation} shape overflows usize")))?;
	if output_gradient.shape() != expanded_shape {
		return Err(Error::invalid_argument(format!(
			"{operation} output gradient shape does not match the saved repeat"
		)));
	}
	let input_count = input_shape.iter().try_fold(1_usize, |count, extent| {
		count.checked_mul(*extent).ok_or_else(|| {
			Error::invalid_argument(format!("{operation} output size overflows usize"))
		})
	})?;
	let count = as_u32(input_count, operation, "input element count")?;
	let output = Matrix::allocate(
		output_gradient.engine_handle(),
		input_shape.to_vec(),
		input_count,
		DType::F32,
	)?;
	if count != 0 {
		let input_dims = shape_u32(input_shape, operation)?;
		let output_strides = row_major_strides_u32(&expanded_shape, operation)?;
		let buffers = [
			BufferBinding::read(output_gradient.storage()),
			BufferBinding::write(output.storage()),
		];
		let mut push_constants = vec![
			PushConstant::U32(count),
			PushConstant::U32(repeats_u32),
			PushConstant::U32(dim as u32),
			PushConstant::U32(rank as u32),
		];
		push_constants.extend(input_dims.map(PushConstant::U32));
		push_constants.extend(output_strides.map(PushConstant::U32));
		let inputs = [output_gradient];
		let outputs = [&output];
		let attributes = [
			OpAttribute::Shape {
				name: "input_shape".into(),
				value: input_shape.to_vec(),
			},
			OpAttribute::SignedInteger {
				name: "repeats".into(),
				value: i64::from(repeats_u32),
			},
			OpAttribute::SignedInteger {
				name: "dim".into(),
				value: dim as i64,
			},
		];
		output_gradient.engine_handle().record_semantic(
			ComputeDispatch {
				kernel: KernelId::MatrixRepeatInterleaveBackwardF32,
				buffers: &buffers,
				push_constants: &push_constants,
				workgroups: KernelId::MatrixRepeatInterleaveBackwardF32.linear_workgroups(count),
			},
			SemanticDispatch {
				contract: CONTRACT,
				inputs: &inputs,
				outputs: &outputs,
				attributes: &attributes,
			},
		)?;
	}
	Ok(output)
}

/// Gather I32-selected elements from every row's final axis.
///
/// `input` is FP32 `[R, C]` and `indices` is I32 `[R, K]`; the result is FP32
/// `[R, K]`. Out-of-range indices produce zero, matching OA C++. Repeated
/// indices accumulate deterministically in the reverse pass.
///
/// # Errors
///
/// Returns an error unless both inputs are rank two on the same engine, row
/// counts agree, dtypes match the contract, the input width is positive, sizes
/// fit the shader ABI, allocation fails, or runtime recording fails.
pub fn gather_last_dim(input: &Matrix, indices: &Matrix) -> Result<Matrix> {
	const CONTRACT: OperationContract = crate::core::operation::matrix::GATHER_LAST_DIM;
	let operation = CONTRACT.name();
	let [rows, input_width] = input.shape() else {
		return Err(Error::invalid_argument(format!(
			"{operation} requires rank-two F32 input"
		)));
	};
	let [index_rows, selected_width] = indices.shape() else {
		return Err(Error::invalid_argument(format!(
			"{operation} requires rank-two I32 indices"
		)));
	};
	if input.dtype() != DType::F32
		|| indices.dtype() != DType::I32
		|| rows != index_rows
		|| *input_width == 0
		|| !input.engine_handle().same_as(indices.engine_handle())
	{
		return Err(Error::invalid_argument(format!(
			"{operation} requires same-engine F32 [R,C] and I32 [R,K] with C > 0"
		)));
	}
	let output_count = rows.checked_mul(*selected_width).ok_or_else(|| {
		Error::invalid_argument(format!("{operation} output size overflows usize"))
	})?;
	let rows_u32 = as_u32(*rows, operation, "row count")?;
	let input_width_u32 = as_u32(*input_width, operation, "input width")?;
	let selected_width_u32 = as_u32(*selected_width, operation, "selected width")?;
	let output_count_u32 = as_u32(output_count, operation, "output element count")?;
	let output = Matrix::allocate(
		input.engine_handle(),
		vec![*rows, *selected_width],
		output_count,
		DType::F32,
	)?;
	if output_count_u32 != 0 {
		let buffers = [
			BufferBinding::read(input.storage()),
			BufferBinding::read(indices.storage()),
			BufferBinding::write(output.storage()),
		];
		let push_constants = [
			PushConstant::U32(rows_u32),
			PushConstant::U32(input_width_u32),
			PushConstant::U32(selected_width_u32),
		];
		let inputs = [input, indices];
		let outputs = [&output];
		input.engine_handle().record_semantic(
			ComputeDispatch {
				kernel: KernelId::MatrixGatherLastDimF32,
				buffers: &buffers,
				push_constants: &push_constants,
				workgroups: KernelId::MatrixGatherLastDimF32.linear_workgroups(output_count_u32),
			},
			SemanticDispatch {
				contract: CONTRACT,
				inputs: &inputs,
				outputs: &outputs,
				attributes: &[],
			},
		)?;
	}
	autograd::record(MatrixNode::GatherLastDim {
		input: input.clone(),
		indices: indices.clone(),
		output_id: output.value_id(),
		input_width: *input_width,
	})?;
	Ok(output)
}

pub(crate) fn gather_last_dim_backward(
	output_gradient: &Matrix,
	indices: &Matrix,
	input_width: usize,
) -> Result<Matrix> {
	const CONTRACT: OperationContract = crate::core::operation::matrix::GATHER_LAST_DIM_BACKWARD;
	let operation = CONTRACT.name();
	let [rows, selected_width] = output_gradient.shape() else {
		return Err(Error::invalid_argument(format!(
			"{operation} requires rank-two F32 output gradient"
		)));
	};
	if output_gradient.dtype() != DType::F32
		|| indices.dtype() != DType::I32
		|| indices.shape() != output_gradient.shape()
		|| input_width == 0
		|| !output_gradient
			.engine_handle()
			.same_as(indices.engine_handle())
	{
		return Err(Error::invalid_argument(format!(
			"{operation} received invalid saved gather geometry"
		)));
	}
	let output_count = rows.checked_mul(input_width).ok_or_else(|| {
		Error::invalid_argument(format!("{operation} output size overflows usize"))
	})?;
	let rows_u32 = as_u32(*rows, operation, "row count")?;
	let input_width_u32 = as_u32(input_width, operation, "input width")?;
	let selected_width_u32 = as_u32(*selected_width, operation, "selected width")?;
	let output_count_u32 = as_u32(output_count, operation, "output element count")?;
	let output = Matrix::allocate(
		output_gradient.engine_handle(),
		vec![*rows, input_width],
		output_count,
		DType::F32,
	)?;
	if output_count_u32 != 0 {
		let buffers = [
			BufferBinding::read(output_gradient.storage()),
			BufferBinding::read(indices.storage()),
			BufferBinding::write(output.storage()),
		];
		let push_constants = [
			PushConstant::U32(rows_u32),
			PushConstant::U32(input_width_u32),
			PushConstant::U32(selected_width_u32),
		];
		let inputs = [output_gradient, indices];
		let outputs = [&output];
		let attributes = [OpAttribute::SignedInteger {
			name: "input_width".into(),
			value: i64::from(input_width_u32),
		}];
		output_gradient.engine_handle().record_semantic(
			ComputeDispatch {
				kernel: KernelId::MatrixGatherLastDimBackwardF32,
				buffers: &buffers,
				push_constants: &push_constants,
				workgroups: KernelId::MatrixGatherLastDimBackwardF32
					.linear_workgroups(output_count_u32),
			},
			SemanticDispatch {
				contract: CONTRACT,
				inputs: &inputs,
				outputs: &outputs,
				attributes: &attributes,
			},
		)?;
	}
	Ok(output)
}

/// Concatenate one or more FP32 matrices along one axis.
///
/// This is the direct Rust spelling of OA C++ `FnMatrix::concat`. One semantic
/// operation lowers to one copy-region dispatch per nonempty input.
///
/// # Errors
///
/// Returns an error when there are no inputs, rank is outside one through four,
/// dtype, engine, rank, or non-concatenated extents differ, the axis is invalid,
/// shape or shader-ABI arithmetic overflows, allocation fails, or recording
/// fails.
pub fn concat(inputs: &[Matrix], dim: i32) -> Result<Matrix> {
	const CONTRACT: OperationContract = crate::core::operation::matrix::CONCAT;
	let operation = CONTRACT.name();
	let Some(first) = inputs.first() else {
		return Err(Error::invalid_argument(format!(
			"{operation} requires at least one input"
		)));
	};
	let rank = first.shape().len();
	if first.dtype() != DType::F32 || !(1..=4).contains(&rank) {
		return Err(Error::invalid_argument(format!(
			"{operation} requires rank-one through rank-four F32 inputs"
		)));
	}
	let dim = usize::try_from(dim)
		.map_err(|_| Error::invalid_argument(format!("{operation} dim must be nonnegative")))?;
	if dim >= rank {
		return Err(Error::invalid_argument(format!(
			"{operation} dim {dim} is outside rank {rank}"
		)));
	}
	let engine = first.engine_handle();
	let mut total_extent = 0_usize;
	let mut sizes = Vec::with_capacity(inputs.len());
	for input in inputs {
		if input.dtype() != DType::F32
			|| input.shape().len() != rank
			|| !engine.same_as(input.engine_handle())
			|| input
				.shape()
				.iter()
				.zip(first.shape())
				.enumerate()
				.any(|(axis, (actual, expected))| axis != dim && actual != expected)
		{
			return Err(Error::invalid_argument(format!(
				"{operation} requires same-engine F32 inputs with matching non-concatenated extents"
			)));
		}
		let extent = input.shape()[dim];
		total_extent = total_extent
			.checked_add(extent)
			.ok_or_else(|| Error::invalid_argument(format!("{operation} shape overflows usize")))?;
		sizes.push(extent);
	}
	let mut output_shape = first.shape().to_vec();
	output_shape[dim] = total_extent;
	let output_count = output_shape.iter().try_fold(1_usize, |count, extent| {
		count.checked_mul(*extent).ok_or_else(|| {
			Error::invalid_argument(format!("{operation} output size overflows usize"))
		})
	})?;
	as_u32(output_count, operation, "output element count")?;
	i64::try_from(total_extent)
		.map_err(|_| Error::invalid_argument(format!("{operation} axis extent exceeds i64")))?;
	let output = Matrix::allocate(engine, output_shape.clone(), output_count, DType::F32)?;

	let mut bindings = Vec::with_capacity(inputs.len());
	let mut payloads = Vec::with_capacity(inputs.len());
	let mut workgroups = Vec::with_capacity(inputs.len());
	let mut destination_start = 0_usize;
	for input in inputs {
		let count = as_u32(input.num_elements(), operation, "input element count")?;
		if count != 0 {
			bindings.push([
				BufferBinding::read(input.storage()),
				BufferBinding::write(output.storage()),
			]);
			payloads.push(region_push_constants(
				count,
				rank,
				dim,
				0,
				destination_start,
				input.shape(),
				&output_shape,
				input.shape(),
				operation,
			)?);
			workgroups.push(KernelId::MatrixConcatF32.linear_workgroups(count));
		}
		destination_start = destination_start
			.checked_add(input.shape()[dim])
			.ok_or_else(|| Error::internal("validated Concat extent overflowed"))?;
	}
	if !bindings.is_empty() {
		let dispatches = bindings
			.iter()
			.zip(&payloads)
			.zip(&workgroups)
			.map(|((buffers, push_constants), workgroups)| ComputeDispatch {
				kernel: KernelId::MatrixConcatF32,
				buffers,
				push_constants,
				workgroups: *workgroups,
			})
			.collect::<Vec<_>>();
		let semantic_inputs = inputs.iter().collect::<Vec<_>>();
		let semantic_outputs = [&output];
		let attributes = [OpAttribute::SignedInteger {
			name: "dim".into(),
			value: dim as i64,
		}];
		let semantic = SemanticDispatch {
			contract: CONTRACT,
			inputs: &semantic_inputs,
			outputs: &semantic_outputs,
			attributes: &attributes,
		};
		if dispatches.len() == 1 {
			engine.record_semantic(
				dispatches.into_iter().next().expect("one dispatch"),
				semantic,
			)?;
		} else {
			engine.record_split_semantic(&dispatches, semantic)?;
		}
	}
	autograd::record(MatrixNode::Concat {
		inputs: inputs.to_vec(),
		output_id: output.value_id(),
		dim,
		sizes,
	})?;
	Ok(output)
}

/// Materialize a swap of the last two axes of a rank-two or rank-three Matrix.
///
/// `dim0` and `dim1` accept positive or negative axes, but the current donor
/// contract only admits the final two axes in either order. The result owns new
/// storage and preserves the input dtype and batch order.
///
/// # Errors
///
/// Returns an error unless the input is rank-two or rank-three F32, the axes
/// select the distinct final two dimensions, all geometry fits the shader ABI,
/// allocation succeeds, and runtime recording succeeds.
pub fn transpose(input: &Matrix, dim0: i32, dim1: i32) -> Result<Matrix> {
	const CONTRACT: OperationContract = crate::core::operation::matrix::TRANSPOSE;
	let operation = CONTRACT.name();
	let rank = input.shape().len();
	if input.dtype() != DType::F32 || !(2..=3).contains(&rank) {
		return Err(Error::invalid_argument(format!(
			"{operation} requires a rank-two or rank-three F32 matrix"
		)));
	}
	let resolve = |dim: i32| -> Result<usize> {
		let rank_i32 = i32::try_from(rank)
			.map_err(|_| Error::invalid_argument(format!("{operation} rank exceeds i32")))?;
		let resolved = if dim < 0 {
			dim.checked_add(rank_i32).ok_or_else(|| {
				Error::invalid_argument(format!("{operation} axis normalization overflows i32"))
			})?
		} else {
			dim
		};
		usize::try_from(resolved)
			.ok()
			.filter(|axis| *axis < rank)
			.ok_or_else(|| {
				Error::invalid_argument(format!("{operation} axis {dim} is outside rank {rank}"))
			})
	};
	let dim0 = resolve(dim0)?;
	let dim1 = resolve(dim1)?;
	let final_axes = [rank - 2, rank - 1];
	if dim0 == dim1 || !final_axes.contains(&dim0) || !final_axes.contains(&dim1) {
		return Err(Error::invalid_argument(format!(
			"{operation} supports only a swap of the final two axes"
		)));
	}

	let rows = input.shape()[rank - 2];
	let columns = input.shape()[rank - 1];
	let batch_size = if rank == 3 { input.shape()[0] } else { 1 };
	let mut output_shape = input.shape().to_vec();
	output_shape.swap(rank - 2, rank - 1);
	let output = Matrix::allocate(
		input.engine_handle(),
		output_shape,
		input.element_count(),
		DType::F32,
	)?;
	if batch_size != 0 && rows != 0 && columns != 0 {
		let batch_size = as_u32(batch_size, operation, "batch size")?;
		let rows = as_u32(rows, operation, "row count")?;
		let columns = as_u32(columns, operation, "column count")?;
		let buffers = [
			BufferBinding::read(input.storage()),
			BufferBinding::write(output.storage()),
		];
		let push_constants = [
			PushConstant::U32(batch_size),
			PushConstant::U32(rows),
			PushConstant::U32(columns),
		];
		let inputs = [input];
		let outputs = [&output];
		let attributes = [
			OpAttribute::SignedInteger {
				name: "dim0".into(),
				value: i64::try_from(dim0)
					.map_err(|_| Error::invalid_argument("transpose dim0 exceeds i64"))?,
			},
			OpAttribute::SignedInteger {
				name: "dim1".into(),
				value: i64::try_from(dim1)
					.map_err(|_| Error::invalid_argument("transpose dim1 exceeds i64"))?,
			},
		];
		input.engine_handle().record_semantic(
			ComputeDispatch {
				kernel: KernelId::MatrixTransposeF32,
				buffers: &buffers,
				push_constants: &push_constants,
				workgroups: [columns.div_ceil(32), rows.div_ceil(32), batch_size],
			},
			SemanticDispatch {
				contract: CONTRACT,
				inputs: &inputs,
				outputs: &outputs,
				attributes: &attributes,
			},
		)?;
	}
	autograd::record(MatrixNode::Transpose {
		input: input.clone(),
		output_id: output.value_id(),
		dim0,
		dim1,
	})?;
	Ok(output)
}

/// Materialize a half-open interval along one matrix axis.
///
/// This is the direct Rust spelling of OA C++ `FnMatrix::slice`. The current
/// checkpoint admits rank-one through rank-four FP32 matrices.
///
/// # Errors
///
/// Returns an error when the rank, dtype, axis, range, shader ABI, allocation,
/// or runtime recording contract is invalid.
pub fn slice(input: &Matrix, dim: i32, start: i64, end: i64) -> Result<Matrix> {
	const CONTRACT: OperationContract = crate::core::operation::matrix::SLICE;
	let operation = CONTRACT.name();
	let rank = input.shape().len();
	if input.dtype() != DType::F32 || !(1..=4).contains(&rank) {
		return Err(Error::invalid_argument(format!(
			"{operation} requires a rank-one through rank-four F32 matrix"
		)));
	}
	let dim = usize::try_from(dim)
		.map_err(|_| Error::invalid_argument(format!("{operation} dim must be nonnegative")))?;
	if dim >= rank {
		return Err(Error::invalid_argument(format!(
			"{operation} dim {dim} is outside rank {rank}"
		)));
	}
	let start = usize::try_from(start)
		.map_err(|_| Error::invalid_argument(format!("{operation} start must be nonnegative")))?;
	let end = usize::try_from(end)
		.map_err(|_| Error::invalid_argument(format!("{operation} end must be nonnegative")))?;
	if start >= end || end > input.shape()[dim] {
		return Err(Error::invalid_argument(format!(
			"{operation} requires 0 <= start < end <= axis extent"
		)));
	}
	let mut output_shape = input.shape().to_vec();
	output_shape[dim] = end - start;
	let output_count = output_shape.iter().try_fold(1_usize, |count, extent| {
		count.checked_mul(*extent).ok_or_else(|| {
			Error::invalid_argument(format!("{operation} output size overflows usize"))
		})
	})?;
	let count = as_u32(output_count, operation, "output element count")?;
	let output = Matrix::allocate(
		input.engine_handle(),
		output_shape.clone(),
		output_count,
		DType::F32,
	)?;
	if count != 0 {
		let buffers = [
			BufferBinding::read(input.storage()),
			BufferBinding::write(output.storage()),
		];
		let push_constants = region_push_constants(
			count,
			rank,
			dim,
			start,
			0,
			input.shape(),
			&output_shape,
			&output_shape,
			operation,
		)?;
		let inputs = [input];
		let outputs = [&output];
		let attributes = [
			OpAttribute::SignedInteger {
				name: "dim".into(),
				value: dim as i64,
			},
			OpAttribute::SignedInteger {
				name: "start".into(),
				value: start as i64,
			},
			OpAttribute::SignedInteger {
				name: "end".into(),
				value: end as i64,
			},
		];
		input.engine_handle().record_semantic(
			ComputeDispatch {
				kernel: KernelId::MatrixSliceF32,
				buffers: &buffers,
				push_constants: &push_constants,
				workgroups: KernelId::MatrixSliceF32.linear_workgroups(count),
			},
			SemanticDispatch {
				contract: CONTRACT,
				inputs: &inputs,
				outputs: &outputs,
				attributes: &attributes,
			},
		)?;
	}
	autograd::record(MatrixNode::Slice {
		input: input.clone(),
		output_id: output.value_id(),
		dim,
		start,
		end,
	})?;
	Ok(output)
}

pub(crate) fn slice_backward(
	input_shape: &[usize],
	dim: usize,
	start: usize,
	end: usize,
	output_gradient: &Matrix,
) -> Result<Matrix> {
	const CONTRACT: OperationContract = crate::core::operation::matrix::SLICE_BACKWARD;
	let operation = CONTRACT.name();
	let rank = input_shape.len();
	if output_gradient.dtype() != DType::F32 || !(1..=4).contains(&rank) || dim >= rank {
		return Err(Error::invalid_argument(format!(
			"{operation} requires rank-one through rank-four F32 geometry"
		)));
	}
	if start >= end || end > input_shape[dim] {
		return Err(Error::invalid_argument(format!(
			"{operation} received an invalid saved slice range"
		)));
	}
	let mut slice_shape = input_shape.to_vec();
	slice_shape[dim] = end - start;
	if output_gradient.shape() != slice_shape {
		return Err(Error::invalid_argument(format!(
			"{operation} output gradient shape does not match the saved slice"
		)));
	}
	let output_count = input_shape.iter().try_fold(1_usize, |count, extent| {
		count.checked_mul(*extent).ok_or_else(|| {
			Error::invalid_argument(format!("{operation} output size overflows usize"))
		})
	})?;
	let copy_count = as_u32(
		output_gradient.num_elements(),
		operation,
		"copy element count",
	)?;
	let output = Matrix::allocate(
		output_gradient.engine_handle(),
		input_shape.to_vec(),
		output_count,
		DType::F32,
	)?;
	if copy_count != 0 {
		let buffers = [
			BufferBinding::read(output_gradient.storage()),
			BufferBinding::write(output.storage()),
		];
		let push_constants = region_push_constants(
			copy_count,
			rank,
			dim,
			0,
			start,
			output_gradient.shape(),
			input_shape,
			output_gradient.shape(),
			operation,
		)?;
		let inputs = [output_gradient];
		let outputs = [&output];
		let attributes = [
			OpAttribute::Shape {
				name: "input_shape".into(),
				value: input_shape.to_vec(),
			},
			OpAttribute::SignedInteger {
				name: "dim".into(),
				value: dim as i64,
			},
			OpAttribute::SignedInteger {
				name: "start".into(),
				value: start as i64,
			},
			OpAttribute::SignedInteger {
				name: "end".into(),
				value: end as i64,
			},
		];
		output_gradient.engine_handle().record_semantic(
			ComputeDispatch {
				kernel: KernelId::MatrixSliceBackwardF32,
				buffers: &buffers,
				push_constants: &push_constants,
				workgroups: KernelId::MatrixSliceBackwardF32.linear_workgroups(copy_count),
			},
			SemanticDispatch {
				contract: CONTRACT,
				inputs: &inputs,
				outputs: &outputs,
				attributes: &attributes,
			},
		)?;
	}
	Ok(output)
}

#[allow(
	clippy::too_many_arguments,
	reason = "the arguments preserve the donor MatrixCopyRegion ABI"
)]
fn region_push_constants(
	count: u32,
	rank: usize,
	dim: usize,
	source_start: usize,
	destination_start: usize,
	source_shape: &[usize],
	destination_shape: &[usize],
	copy_shape: &[usize],
	operation: &'static str,
) -> Result<Vec<PushConstant>> {
	let mut push_constants = Vec::with_capacity(17);
	for value in [rank, dim, source_start, destination_start] {
		as_u32(value, operation, "region field")?;
	}
	push_constants.push(PushConstant::U32(count));
	push_constants.push(PushConstant::U32(rank as u32));
	push_constants.push(PushConstant::U32(dim as u32));
	push_constants.push(PushConstant::U32(source_start as u32));
	push_constants.push(PushConstant::U32(destination_start as u32));
	for shape in [source_shape, destination_shape, copy_shape] {
		for axis in 0..4 {
			push_constants.push(PushConstant::U32(if axis < shape.len() {
				as_u32(shape[axis], operation, "region extent")?
			} else {
				0
			}));
		}
	}
	Ok(push_constants)
}

fn as_u32(value: usize, operation: &'static str, label: &str) -> Result<u32> {
	u32::try_from(value)
		.map_err(|_| Error::invalid_argument(format!("{operation} {label} exceeds u32")))
}

fn shape_u32(shape: &[usize], operation: &'static str) -> Result<[u32; 4]> {
	let mut dimensions = [0_u32; 4];
	for (axis, extent) in shape.iter().copied().enumerate() {
		dimensions[axis] = as_u32(extent, operation, "shape extent")?;
	}
	Ok(dimensions)
}

fn row_major_strides_u32(shape: &[usize], operation: &'static str) -> Result<[u32; 4]> {
	let mut strides = [0_u32; 4];
	let mut stride = 1_usize;
	for axis in (0..shape.len()).rev() {
		strides[axis] = as_u32(stride, operation, "row-major stride")?;
		stride = stride.checked_mul(shape[axis]).ok_or_else(|| {
			Error::invalid_argument(format!("{operation} row-major stride overflows usize"))
		})?;
	}
	Ok(strides)
}
