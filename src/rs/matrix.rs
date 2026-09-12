//! Stateless Matrix operations.

use crate::{
	DType, Engine, Error, Matrix, OpAttribute, OpShapeRule, OperationContract, Result,
	core::autograd::{self, MatrixNode},
	runtime::{BufferBinding, ComputeDispatch, KernelId, PushConstant, SemanticDispatch},
};

mod index;
mod rng;

pub use index::{
	MoeExpertPlan, TopKResult, concat, equal, gather_last_dim, moe_expert_plan,
	moe_routing_bias_update, repeat_interleave, slice, top_k, top_k_mask,
};
pub(crate) use index::{gather_last_dim_backward, repeat_interleave_backward, slice_backward};
pub(crate) use rng::dropout_backward;
pub use rng::{dropout, philox_normal, philox_uniform, sample_logits, set_rng_seed};

/// Create an FP32 matrix filled with ones.
///
/// # Errors
///
/// Returns an error when the shape or storage size overflows, or allocation and
/// upload fail.
pub fn ones(engine: &Engine, shape: impl Into<Vec<usize>>) -> Result<Matrix> {
	Matrix::filled_f32(engine, shape, 1.0)
}

/// Create an FP32 matrix filled with `value`.
///
/// # Errors
///
/// Returns an error when the shape or storage size overflows, or allocation and
/// upload fail.
pub fn full(engine: &Engine, shape: impl Into<Vec<usize>>, value: f32) -> Result<Matrix> {
	Matrix::filled_f32(engine, shape, value)
}

/// Return a zero-copy dense view with a different shape.
///
/// The result owns a new semantic value identity but shares storage, pending
/// execution state, dtype, and element order with `input`. When a gradient tape
/// is active, its adjoint reshapes the incoming gradient back to the input shape.
///
/// # Errors
///
/// Returns an error when the requested shape overflows or changes the number of
/// elements.
pub fn reshape(input: &Matrix, shape: impl Into<Vec<usize>>) -> Result<Matrix> {
	let output = input.reshape_view(shape.into())?;
	autograd::record(MatrixNode::Reshape {
		input: input.clone(),
		output_id: output.value_id(),
	})?;
	Ok(output)
}

pub(crate) fn reshape_semantic_output(input: &Matrix, shape: Vec<usize>) -> Result<Matrix> {
	let output = input.reshape_semantic_output(shape)?;
	autograd::record(MatrixNode::Reshape {
		input: input.clone(),
		output_id: output.value_id(),
	})?;
	Ok(output)
}

fn binary(
	left: &Matrix,
	right: &Matrix,
	routes: &[(DType, KernelId)],
	contract: OperationContract,
) -> Result<Matrix> {
	let operation = contract.name();
	if left.dtype() != right.dtype() {
		return Err(Error::invalid_argument(format!(
			"{operation} requires equal dtypes; left is {}, right is {}",
			left.dtype().token(),
			right.dtype().token()
		)));
	}
	let engine = left.engine_handle();
	if !engine.same_as(right.engine_handle()) {
		return Err(Error::invalid_argument(format!(
			"{operation} inputs must belong to the same engine"
		)));
	}
	let output = if left.shape() == right.shape() {
		let kernel = select_kernel(left.dtype(), routes, operation)?;
		let element_count = u32::try_from(left.element_count()).map_err(|_| {
			Error::invalid_argument(format!("{operation} element count exceeds u32"))
		})?;
		let output = Matrix::allocate(
			engine,
			left.shape().to_vec(),
			left.element_count(),
			left.dtype(),
		)?;
		if element_count != 0 {
			let buffers = [
				BufferBinding::read(left.storage()),
				BufferBinding::read(right.storage()),
				BufferBinding::write(output.storage()),
			];
			let push_constants = [PushConstant::U32(element_count)];
			let inputs = [left, right];
			let outputs = [&output];
			engine.record_semantic(
				ComputeDispatch {
					kernel,
					buffers: &buffers,
					push_constants: &push_constants,
					workgroups: kernel.linear_workgroups(element_count),
				},
				SemanticDispatch {
					contract,
					inputs: &inputs,
					outputs: &outputs,
					attributes: &[],
				},
			)?;
		}
		output
	} else if contract.shape_rule() == OpShapeRule::Broadcast {
		binary_broadcast(left, right, contract)?
	} else {
		return Err(Error::invalid_argument(format!(
			"{operation} requires equal shapes; left is {:?}, right is {:?}",
			left.shape(),
			right.shape()
		)));
	};
	if contract.hash() == crate::core::operation::matrix::ADD.hash() && left.dtype() == DType::F32 {
		autograd::record(MatrixNode::Add {
			left: left.clone(),
			right: right.clone(),
			output_id: output.value_id(),
		})?;
	} else if contract.hash() == crate::core::operation::matrix::SUB.hash()
		&& left.dtype() == DType::F32
	{
		autograd::record(MatrixNode::Sub {
			left: left.clone(),
			right: right.clone(),
			output_id: output.value_id(),
		})?;
	} else if contract.hash() == crate::core::operation::matrix::MUL.hash()
		&& left.dtype() == DType::F32
	{
		autograd::record(MatrixNode::Mul {
			left: left.clone(),
			right: right.clone(),
			output_id: output.value_id(),
		})?;
	} else if contract.hash() == crate::core::operation::matrix::DIV.hash()
		&& left.dtype() == DType::F32
	{
		autograd::record(MatrixNode::Div {
			left: left.clone(),
			right: right.clone(),
			output_id: output.value_id(),
		})?;
	}
	Ok(output)
}

const BROADCAST_RANK_LIMIT: usize = 8;

struct BroadcastLayout {
	output_shape: Vec<usize>,
	output_dims: [u32; BROADCAST_RANK_LIMIT],
	left_strides: [u32; BROADCAST_RANK_LIMIT],
	right_strides: [u32; BROADCAST_RANK_LIMIT],
	element_count: u32,
	rank: u32,
}

fn binary_broadcast(left: &Matrix, right: &Matrix, contract: OperationContract) -> Result<Matrix> {
	let operation = contract.name();
	let layout = resolve_broadcast(left.shape(), right.shape(), operation)?;
	let kernel = match (contract.hash(), left.dtype()) {
		(hash, DType::F32) if hash == crate::core::operation::matrix::ADD.hash() => {
			KernelId::MatrixAddBroadcastF32
		}
		(hash, DType::I32) if hash == crate::core::operation::matrix::ADD.hash() => {
			KernelId::MatrixAddBroadcastI32
		}
		(hash, DType::F32) if hash == crate::core::operation::matrix::MUL.hash() => {
			KernelId::MatrixMulBroadcastF32
		}
		(hash, DType::F32) if hash == crate::core::operation::matrix::SUB.hash() => {
			KernelId::MatrixSubBroadcastF32
		}
		(hash, DType::F32) if hash == crate::core::operation::matrix::DIV.hash() => {
			KernelId::MatrixDivBroadcastF32
		}
		_ => {
			return Err(Error::invalid_argument(format!(
				"{operation} does not support dtype {}",
				left.dtype().token()
			)));
		}
	};
	let output_count = usize::try_from(layout.element_count)
		.expect("u32 element count always fits the supported Rust targets");
	let output = Matrix::allocate(
		left.engine_handle(),
		layout.output_shape,
		output_count,
		left.dtype(),
	)?;
	if layout.element_count != 0 {
		let buffers = [
			BufferBinding::read(left.storage()),
			BufferBinding::read(right.storage()),
			BufferBinding::write(output.storage()),
		];
		let mut push_constants = Vec::with_capacity(26);
		push_constants.push(PushConstant::U32(layout.element_count));
		push_constants.push(PushConstant::U32(layout.rank));
		push_constants.extend(layout.output_dims.map(PushConstant::U32));
		push_constants.extend(layout.left_strides.map(PushConstant::U32));
		push_constants.extend(layout.right_strides.map(PushConstant::U32));
		let inputs = [left, right];
		let outputs = [&output];
		left.engine_handle().record_semantic(
			ComputeDispatch {
				kernel,
				buffers: &buffers,
				push_constants: &push_constants,
				workgroups: kernel.linear_workgroups(layout.element_count),
			},
			SemanticDispatch {
				contract,
				inputs: &inputs,
				outputs: &outputs,
				attributes: &[],
			},
		)?;
	}
	Ok(output)
}

fn resolve_broadcast(
	left: &[usize],
	right: &[usize],
	operation: &'static str,
) -> Result<BroadcastLayout> {
	let rank = left.len().max(right.len());
	if rank > BROADCAST_RANK_LIMIT {
		return Err(Error::invalid_argument(format!(
			"{operation} supports at most {BROADCAST_RANK_LIMIT} broadcast axes"
		)));
	}
	let mut output_shape = vec![1_usize; rank];
	for (axis, output_extent) in output_shape.iter_mut().enumerate() {
		let left_extent = aligned_extent(left, rank, axis);
		let right_extent = aligned_extent(right, rank, axis);
		*output_extent = if left_extent == right_extent {
			left_extent
		} else if left_extent == 1 {
			right_extent
		} else if right_extent == 1 {
			left_extent
		} else {
			return Err(Error::invalid_argument(format!(
				"{operation} cannot broadcast shapes {left:?} and {right:?}"
			)));
		};
	}
	let element_count = output_shape.iter().try_fold(1_usize, |product, extent| {
		product.checked_mul(*extent).ok_or_else(|| {
			Error::invalid_argument(format!("{operation} output size overflows usize"))
		})
	})?;
	let element_count = u32::try_from(element_count)
		.map_err(|_| Error::invalid_argument(format!("{operation} element count exceeds u32")))?;
	let mut output_dims = [1_u32; BROADCAST_RANK_LIMIT];
	for (destination, extent) in output_dims.iter_mut().zip(&output_shape) {
		*destination = u32::try_from(*extent).map_err(|_| {
			Error::invalid_argument(format!("{operation} output extent exceeds u32"))
		})?;
	}
	Ok(BroadcastLayout {
		left_strides: broadcast_strides(left, &output_shape, operation)?,
		right_strides: broadcast_strides(right, &output_shape, operation)?,
		output_shape,
		output_dims,
		element_count,
		rank: u32::try_from(rank).expect("broadcast rank limit fits u32"),
	})
}

fn broadcast_strides(
	shape: &[usize],
	output_shape: &[usize],
	operation: &'static str,
) -> Result<[u32; BROADCAST_RANK_LIMIT]> {
	let rank = output_shape.len();
	let offset = rank - shape.len();
	let mut strides = [0_u32; BROADCAST_RANK_LIMIT];
	let mut contiguous_stride = 1_usize;
	for source_axis in (0..shape.len()).rev() {
		let axis = offset + source_axis;
		if shape[source_axis] != 1 || output_shape[axis] == 1 {
			strides[axis] = u32::try_from(contiguous_stride).map_err(|_| {
				Error::invalid_argument(format!("{operation} input stride exceeds u32"))
			})?;
		}
		contiguous_stride = contiguous_stride
			.checked_mul(shape[source_axis])
			.ok_or_else(|| Error::invalid_argument(format!("{operation} input size overflows")))?;
	}
	Ok(strides)
}

fn aligned_extent(shape: &[usize], rank: usize, axis: usize) -> usize {
	axis.checked_sub(rank - shape.len())
		.map_or(1, |source_axis| shape[source_axis])
}

fn unary(
	input: &Matrix,
	routes: &[(DType, KernelId)],
	contract: OperationContract,
) -> Result<Matrix> {
	let operation = contract.name();
	let kernel = select_kernel(input.dtype(), routes, operation)?;
	let engine = input.engine_handle();
	let element_count = u32::try_from(input.element_count())
		.map_err(|_| Error::invalid_argument(format!("{operation} element count exceeds u32")))?;
	let output = Matrix::allocate(
		engine,
		input.shape().to_vec(),
		input.element_count(),
		input.dtype(),
	)?;
	if element_count != 0 {
		let buffers = [
			BufferBinding::read(input.storage()),
			BufferBinding::write(output.storage()),
		];
		let push_constants = [PushConstant::U32(element_count)];
		let inputs = [input];
		let outputs = [&output];
		engine.record_semantic(
			ComputeDispatch {
				kernel,
				buffers: &buffers,
				push_constants: &push_constants,
				workgroups: kernel.linear_workgroups(element_count),
			},
			SemanticDispatch {
				contract,
				inputs: &inputs,
				outputs: &outputs,
				attributes: &[],
			},
		)?;
	}
	if contract.hash() == crate::core::operation::matrix::RECIPROCAL.hash() {
		autograd::record(MatrixNode::Reciprocal {
			input: input.clone(),
			output: output.clone(),
			output_id: output.value_id(),
		})?;
	} else if contract.hash() == crate::core::operation::matrix::EXP.hash() {
		autograd::record(MatrixNode::Exp {
			input: input.clone(),
			output: output.clone(),
			output_id: output.value_id(),
		})?;
	} else if contract.hash() == crate::core::operation::matrix::LOG.hash() {
		autograd::record(MatrixNode::Log {
			input: input.clone(),
			output_id: output.value_id(),
		})?;
	} else if contract.hash() == crate::core::operation::matrix::ABS.hash() {
		autograd::record(MatrixNode::Abs {
			input: input.clone(),
			output_id: output.value_id(),
		})?;
	} else if contract.hash() == crate::core::operation::matrix::COPY.hash() {
		autograd::record(MatrixNode::Copy {
			input: input.clone(),
			output_id: output.value_id(),
		})?;
	} else if contract.hash() == crate::core::operation::matrix::SQRT.hash() {
		autograd::record(MatrixNode::Sqrt {
			input: input.clone(),
			output: output.clone(),
			output_id: output.value_id(),
		})?;
	} else if contract.hash() == crate::core::operation::matrix::NEG.hash() {
		autograd::record(MatrixNode::Scale {
			input: input.clone(),
			output_id: output.value_id(),
			scalar: -1.0,
		})?;
	}
	Ok(output)
}

fn unary_scalar(
	input: &Matrix,
	scalar: f32,
	routes: &[(DType, KernelId)],
	contract: OperationContract,
) -> Result<Matrix> {
	let operation = contract.name();
	let kernel = select_kernel(input.dtype(), routes, operation)?;
	let engine = input.engine_handle();
	let element_count = u32::try_from(input.element_count())
		.map_err(|_| Error::invalid_argument(format!("{operation} element count exceeds u32")))?;
	let output = Matrix::allocate(
		engine,
		input.shape().to_vec(),
		input.element_count(),
		input.dtype(),
	)?;
	if element_count != 0 {
		let buffers = [
			BufferBinding::read(input.storage()),
			BufferBinding::write(output.storage()),
		];
		let push_constants = [PushConstant::U32(element_count), PushConstant::F32(scalar)];
		let inputs = [input];
		let outputs = [&output];
		let attributes = [OpAttribute::Float {
			name: contract.attribute_specs()[0].name().into(),
			value: f64::from(scalar),
		}];
		engine.record_semantic(
			ComputeDispatch {
				kernel,
				buffers: &buffers,
				push_constants: &push_constants,
				workgroups: kernel.linear_workgroups(element_count),
			},
			SemanticDispatch {
				contract,
				inputs: &inputs,
				outputs: &outputs,
				attributes: &attributes,
			},
		)?;
	}
	if contract.hash() == crate::core::operation::matrix::SCALE.hash() {
		autograd::record(MatrixNode::Scale {
			input: input.clone(),
			output_id: output.value_id(),
			scalar,
		})?;
	} else if contract.hash() == crate::core::operation::matrix::ADD_SCALAR.hash()
		|| contract.hash() == crate::core::operation::matrix::SUB_SCALAR.hash()
	{
		autograd::record(MatrixNode::Scale {
			input: input.clone(),
			output_id: output.value_id(),
			scalar: 1.0,
		})?;
	} else if contract.hash() == crate::core::operation::matrix::DIV_SCALAR.hash() {
		autograd::record(MatrixNode::Scale {
			input: input.clone(),
			output_id: output.value_id(),
			scalar: 1.0 / scalar,
		})?;
	} else if contract.hash() == crate::core::operation::matrix::CLAMP_MAX.hash() {
		autograd::record(MatrixNode::ClampMax {
			input: input.clone(),
			output_id: output.value_id(),
			maximum: scalar,
		})?;
	} else if contract.hash() == crate::core::operation::matrix::CLAMP_MIN.hash() {
		autograd::record(MatrixNode::ClampMin {
			input: input.clone(),
			output_id: output.value_id(),
			minimum: scalar,
		})?;
	}
	Ok(output)
}

fn mat_mul_nt_impl(left: &Matrix, right: &Matrix, contract: OperationContract) -> Result<Matrix> {
	let operation = contract.name();
	let [m, k] = left.shape() else {
		return Err(Error::invalid_argument(format!(
			"{operation} requires rank-two inputs; left is {:?}",
			left.shape()
		)));
	};
	let [n, right_k] = right.shape() else {
		return Err(Error::invalid_argument(format!(
			"{operation} requires rank-two inputs; right is {:?}",
			right.shape()
		)));
	};
	if k != right_k {
		return Err(Error::invalid_argument(format!(
			"{operation} requires equal K extents; left is {:?}, right is {:?}",
			left.shape(),
			right.shape()
		)));
	}
	if left.dtype() != DType::F32 || right.dtype() != DType::F32 {
		return Err(Error::invalid_argument(format!(
			"{operation} requires two F32 matrices; left is {}, right is {}",
			left.dtype().token(),
			right.dtype().token()
		)));
	}
	let engine = left.engine_handle();
	if !engine.same_as(right.engine_handle()) {
		return Err(Error::invalid_argument(format!(
			"{operation} inputs must belong to the same engine"
		)));
	}

	let output_count = m.checked_mul(*n).ok_or_else(|| {
		Error::invalid_argument(format!("{operation} output size overflows usize"))
	})?;
	let m = u32::try_from(*m)
		.map_err(|_| Error::invalid_argument(format!("{operation} M extent exceeds u32")))?;
	let n = u32::try_from(*n)
		.map_err(|_| Error::invalid_argument(format!("{operation} N extent exceeds u32")))?;
	let k = u32::try_from(*k)
		.map_err(|_| Error::invalid_argument(format!("{operation} K extent exceeds u32")))?;
	for (label, count) in [
		("left", left.element_count()),
		("right", right.element_count()),
		("output", output_count),
	] {
		u32::try_from(count).map_err(|_| {
			Error::invalid_argument(format!("{operation} {label} element count exceeds u32"))
		})?;
	}

	let output = Matrix::allocate(
		engine,
		vec![m as usize, n as usize],
		output_count,
		DType::F32,
	)?;
	let kernel = KernelId::MatrixMatMulNtTiledF32;
	if m != 0 && n != 0 && k != 0 {
		let buffers = [
			BufferBinding::read(left.storage()),
			BufferBinding::read(right.storage()),
			BufferBinding::write(output.storage()),
		];
		let push_constants = [
			PushConstant::U32(m),
			PushConstant::U32(n),
			PushConstant::U32(k),
		];
		let inputs = [left, right];
		let outputs = [&output];
		engine.record_semantic(
			ComputeDispatch {
				kernel,
				buffers: &buffers,
				push_constants: &push_constants,
				workgroups: kernel.output_workgroups(m, n),
			},
			SemanticDispatch {
				contract,
				inputs: &inputs,
				outputs: &outputs,
				attributes: &[],
			},
		)?;
		autograd::record(MatrixNode::MatMulNt {
			left: left.clone(),
			right: right.clone(),
			output_id: output.value_id(),
		})?;
	}
	Ok(output)
}

struct AxisShape {
	outer_size: u32,
	dim_size: u32,
	inner_size: u32,
	group_count: u32,
}

fn resolve_axis(input: &Matrix, dim: i32, operation: &'static str) -> Result<AxisShape> {
	if input.dtype() != DType::F32 || input.shape().is_empty() || input.num_elements() == 0 {
		return Err(Error::invalid_argument(format!(
			"{operation} requires a nonempty FP32 matrix"
		)));
	}
	let byte_size = input
		.num_elements()
		.checked_mul(DType::F32.size_bytes())
		.ok_or_else(|| Error::invalid_argument(format!("{operation} byte size overflows usize")))?;
	u32::try_from(byte_size).map_err(|_| {
		Error::invalid_argument(format!(
			"{operation} byte-address range exceeds the admitted shader ABI"
		))
	})?;
	let rank = input.shape().len();
	let axis = if dim == -1 {
		rank - 1
	} else {
		usize::try_from(dim).map_err(|_| {
			Error::invalid_argument(format!(
				"{operation} dim must be -1 or a valid non-negative axis; found {dim}"
			))
		})?
	};
	if axis >= rank {
		return Err(Error::invalid_argument(format!(
			"{operation} dim {dim} is invalid for rank {rank}"
		)));
	}
	let outer_size = input.shape()[..axis]
		.iter()
		.try_fold(1_usize, |product, extent| product.checked_mul(*extent))
		.ok_or_else(|| {
			Error::invalid_argument(format!("{operation} outer size overflows usize"))
		})?;
	let dim_size = input.shape()[axis];
	let inner_size = input.shape()[axis + 1..]
		.iter()
		.try_fold(1_usize, |product, extent| product.checked_mul(*extent))
		.ok_or_else(|| {
			Error::invalid_argument(format!("{operation} inner size overflows usize"))
		})?;
	let group_count = outer_size.checked_mul(inner_size).ok_or_else(|| {
		Error::invalid_argument(format!("{operation} dispatch count overflows usize"))
	})?;
	Ok(AxisShape {
		outer_size: u32::try_from(outer_size)
			.map_err(|_| Error::invalid_argument(format!("{operation} outer size exceeds u32")))?,
		dim_size: u32::try_from(dim_size)
			.map_err(|_| Error::invalid_argument(format!("{operation} axis size exceeds u32")))?,
		inner_size: u32::try_from(inner_size)
			.map_err(|_| Error::invalid_argument(format!("{operation} inner size exceeds u32")))?,
		group_count: u32::try_from(group_count).map_err(|_| {
			Error::invalid_argument(format!("{operation} dispatch count exceeds u32"))
		})?,
	})
}

#[derive(Clone, Copy)]
enum AxisNormalization {
	Softmax,
	LogSoftmax,
}

fn axis_normalization_impl(
	input: &Matrix,
	dim: i32,
	contract: OperationContract,
	kernel: KernelId,
	kind: AxisNormalization,
) -> Result<Matrix> {
	let operation = contract.name();
	let axis = resolve_axis(input, dim, operation)?;
	let output = Matrix::allocate(
		input.engine_handle(),
		input.shape().to_vec(),
		input.num_elements(),
		DType::F32,
	)?;
	let buffers = [
		BufferBinding::read(input.storage()),
		BufferBinding::write(output.storage()),
	];
	let push_constants = [
		PushConstant::U32(axis.outer_size),
		PushConstant::U32(axis.dim_size),
		PushConstant::U32(axis.inner_size),
	];
	let attributes = [OpAttribute::SignedInteger {
		name: "dim".into(),
		value: i64::from(dim),
	}];
	let inputs = [input];
	let outputs = [&output];
	input.engine_handle().record_semantic(
		ComputeDispatch {
			kernel,
			buffers: &buffers,
			push_constants: &push_constants,
			workgroups: [axis.group_count, 1, 1],
		},
		SemanticDispatch {
			contract,
			inputs: &inputs,
			outputs: &outputs,
			attributes: &attributes,
		},
	)?;
	let node = match kind {
		AxisNormalization::Softmax => MatrixNode::Softmax {
			input: input.clone(),
			output: output.clone(),
			output_id: output.value_id(),
			dim,
		},
		AxisNormalization::LogSoftmax => MatrixNode::LogSoftmax {
			input: input.clone(),
			output: output.clone(),
			output_id: output.value_id(),
			dim,
		},
	};
	autograd::record(node)?;
	Ok(output)
}

fn softmax_impl(input: &Matrix, dim: i32, contract: OperationContract) -> Result<Matrix> {
	axis_normalization_impl(
		input,
		dim,
		contract,
		KernelId::MatrixSoftmaxF32,
		AxisNormalization::Softmax,
	)
}

fn log_softmax_impl(input: &Matrix, dim: i32, contract: OperationContract) -> Result<Matrix> {
	axis_normalization_impl(
		input,
		dim,
		contract,
		KernelId::MatrixLogSoftmaxF32,
		AxisNormalization::LogSoftmax,
	)
}

fn axis_normalization_backward_impl(
	forward_output: &Matrix,
	output_gradient: &Matrix,
	dim: i32,
	contract: OperationContract,
	kernel: KernelId,
) -> Result<Matrix> {
	let operation = contract.name();
	if forward_output.shape() != output_gradient.shape()
		|| forward_output.dtype() != output_gradient.dtype()
	{
		return Err(Error::invalid_argument(format!(
			"{operation} requires matching forward-output and output-gradient matrices"
		)));
	}
	if !forward_output
		.engine_handle()
		.same_as(output_gradient.engine_handle())
	{
		return Err(Error::invalid_argument(format!(
			"{operation} inputs must belong to the same engine"
		)));
	}
	let axis = resolve_axis(forward_output, dim, operation)?;
	let input_gradient = Matrix::allocate(
		forward_output.engine_handle(),
		forward_output.shape().to_vec(),
		forward_output.num_elements(),
		DType::F32,
	)?;
	let buffers = [
		BufferBinding::read(forward_output.storage()),
		BufferBinding::read(output_gradient.storage()),
		BufferBinding::write(input_gradient.storage()),
	];
	let push_constants = [
		PushConstant::U32(axis.outer_size),
		PushConstant::U32(axis.dim_size),
		PushConstant::U32(axis.inner_size),
	];
	let attributes = [OpAttribute::SignedInteger {
		name: "dim".into(),
		value: i64::from(dim),
	}];
	let inputs = [forward_output, output_gradient];
	let outputs = [&input_gradient];
	forward_output.engine_handle().record_semantic(
		ComputeDispatch {
			kernel,
			buffers: &buffers,
			push_constants: &push_constants,
			workgroups: [axis.group_count, 1, 1],
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

fn softmax_backward_impl(
	forward_output: &Matrix,
	output_gradient: &Matrix,
	dim: i32,
	contract: OperationContract,
) -> Result<Matrix> {
	axis_normalization_backward_impl(
		forward_output,
		output_gradient,
		dim,
		contract,
		KernelId::MatrixSoftmaxBackwardF32,
	)
}

fn log_softmax_backward_impl(
	forward_output: &Matrix,
	output_gradient: &Matrix,
	dim: i32,
	contract: OperationContract,
) -> Result<Matrix> {
	axis_normalization_backward_impl(
		forward_output,
		output_gradient,
		dim,
		contract,
		KernelId::MatrixLogSoftmaxBackwardF32,
	)
}

fn sum_shape(input: &Matrix, dim: i32, operation: &'static str) -> Result<(AxisShape, Vec<usize>)> {
	if dim < -1 {
		return Err(Error::invalid_argument(format!(
			"{operation} dim must be -1 or a valid non-negative axis; found {dim}"
		)));
	}
	if dim == -1 {
		resolve_axis(input, -1, operation)?;
		let element_count = u32::try_from(input.num_elements()).map_err(|_| {
			Error::invalid_argument(format!("{operation} element count exceeds u32"))
		})?;
		return Ok((
			AxisShape {
				outer_size: 1,
				dim_size: element_count,
				inner_size: 1,
				group_count: 1,
			},
			vec![1],
		));
	}
	let axis = resolve_axis(input, dim, operation)?;
	let mut output_shape = input.shape().to_vec();
	output_shape[usize::try_from(dim).map_err(|_| {
		Error::invalid_argument(format!("{operation} dim cannot be represented as usize"))
	})?] = 1;
	Ok((axis, output_shape))
}

fn sum_impl(input: &Matrix, dim: i32, contract: OperationContract) -> Result<Matrix> {
	let operation = contract.name();
	let (axis, output_shape) = sum_shape(input, dim, operation)?;
	let output_count = if dim == -1 {
		1
	} else {
		usize::try_from(axis.group_count).map_err(|_| {
			Error::invalid_argument(format!("{operation} output count exceeds usize"))
		})?
	};
	let output = Matrix::allocate(
		input.engine_handle(),
		output_shape,
		output_count,
		DType::F32,
	)?;
	let buffers = [
		BufferBinding::read(input.storage()),
		BufferBinding::write(output.storage()),
	];
	let full_push = [PushConstant::U32(axis.dim_size)];
	let axis_push = [
		PushConstant::U32(axis.outer_size),
		PushConstant::U32(axis.dim_size),
		PushConstant::U32(axis.inner_size),
	];
	let (kernel, push_constants, workgroups) = if dim == -1 {
		(KernelId::MatrixSumF32, full_push.as_slice(), [1, 1, 1])
	} else {
		(
			KernelId::MatrixSumAxisF32,
			axis_push.as_slice(),
			[axis.group_count.div_ceil(256), 1, 1],
		)
	};
	let attributes = [OpAttribute::SignedInteger {
		name: "dim".into(),
		value: i64::from(dim),
	}];
	let inputs = [input];
	let outputs = [&output];
	input.engine_handle().record_semantic(
		ComputeDispatch {
			kernel,
			buffers: &buffers,
			push_constants,
			workgroups,
		},
		SemanticDispatch {
			contract,
			inputs: &inputs,
			outputs: &outputs,
			attributes: &attributes,
		},
	)?;
	autograd::record(MatrixNode::Sum {
		input: input.clone(),
		output_id: output.value_id(),
		dim,
	})?;
	Ok(output)
}

fn sum_backward_impl(
	input: &Matrix,
	output_gradient: &Matrix,
	dim: i32,
	contract: OperationContract,
) -> Result<Matrix> {
	let operation = contract.name();
	let (axis, output_shape) = sum_shape(input, dim, operation)?;
	if output_gradient.dtype() != DType::F32 || output_gradient.shape() != output_shape {
		return Err(Error::invalid_argument(format!(
			"{operation} output gradient has {:?} {}; expected {:?} F32",
			output_gradient.shape(),
			output_gradient.dtype().token(),
			output_shape
		)));
	}
	if !input
		.engine_handle()
		.same_as(output_gradient.engine_handle())
	{
		return Err(Error::invalid_argument(format!(
			"{operation} inputs must belong to the same engine"
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
	let push_constants = [
		PushConstant::U32(axis.outer_size),
		PushConstant::U32(axis.dim_size),
		PushConstant::U32(axis.inner_size),
	];
	let attributes = [OpAttribute::SignedInteger {
		name: "dim".into(),
		value: i64::from(dim),
	}];
	let inputs = [input, output_gradient];
	let outputs = [&input_gradient];
	input.engine_handle().record_semantic(
		ComputeDispatch {
			kernel: KernelId::MatrixSumBackwardF32,
			buffers: &buffers,
			push_constants: &push_constants,
			workgroups: [
				u32::try_from(input.num_elements())
					.map_err(|_| {
						Error::invalid_argument(format!("{operation} element count exceeds u32"))
					})?
					.div_ceil(256),
				1,
				1,
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

fn categorical_accuracy_count_impl(
	logits: &Matrix,
	labels: &Matrix,
	mask: Option<&Matrix>,
	contract: OperationContract,
) -> Result<Matrix> {
	let operation = contract.name();
	if logits.dtype() != DType::F32 || logits.shape().len() < 2 || logits.num_elements() == 0 {
		return Err(Error::invalid_argument(format!(
			"{operation} requires nonempty rank-two-or-greater F32 logits"
		)));
	}
	let classes = *logits.shape().last().ok_or_else(|| {
		Error::invalid_argument(format!("{operation} logits must expose a class axis"))
	})?;
	let label_shape = &logits.shape()[..logits.shape().len() - 1];
	if labels.shape() != label_shape {
		return Err(Error::invalid_argument(format!(
			"{operation} labels have shape {:?}; expected {:?}",
			labels.shape(),
			label_shape
		)));
	}
	let label_dtype = match labels.dtype() {
		DType::U8 => 0,
		DType::U32 => 1,
		DType::I32 => 2,
		_ => {
			return Err(Error::invalid_argument(format!(
				"{operation} labels must use U8, U32, or I32; found {}",
				labels.dtype().token()
			)));
		}
	};
	let engine = logits.engine_handle();
	if !engine.same_as(labels.engine_handle()) {
		return Err(Error::invalid_argument(format!(
			"{operation} inputs must belong to the same engine"
		)));
	}
	if let Some(mask) = mask {
		if mask.dtype() != DType::F32 || mask.shape() != label_shape {
			return Err(Error::invalid_argument(format!(
				"{operation} mask has shape {:?} {}; expected {:?} F32",
				mask.shape(),
				mask.dtype().token(),
				label_shape
			)));
		}
		if !engine.same_as(mask.engine_handle()) {
			return Err(Error::invalid_argument(format!(
				"{operation} inputs must belong to the same engine"
			)));
		}
	}
	let validate_byte_range = |matrix: &Matrix, label: &str| -> Result<()> {
		let byte_size = matrix
			.num_elements()
			.checked_mul(matrix.dtype().size_bytes())
			.ok_or_else(|| {
				Error::invalid_argument(format!("{operation} {label} byte size overflows usize"))
			})?;
		u32::try_from(byte_size).map_err(|_| {
			Error::invalid_argument(format!(
				"{operation} {label} byte-address range exceeds the shader ABI"
			))
		})?;
		Ok(())
	};
	validate_byte_range(logits, "logits")?;
	validate_byte_range(labels, "labels")?;
	if let Some(mask) = mask {
		validate_byte_range(mask, "mask")?;
	}
	let rows = u32::try_from(labels.num_elements())
		.map_err(|_| Error::invalid_argument(format!("{operation} row count exceeds u32")))?;
	let classes = u32::try_from(classes)
		.map_err(|_| Error::invalid_argument(format!("{operation} class count exceeds u32")))?;
	let output = Matrix::allocate(engine, vec![1], 1, DType::U32)?;
	let push_constants = [
		PushConstant::U32(rows),
		PushConstant::U32(classes),
		PushConstant::U32(label_dtype),
	];
	let outputs = [&output];
	if let Some(mask) = mask {
		let buffers = [
			BufferBinding::read(logits.storage()),
			BufferBinding::read(labels.storage()),
			BufferBinding::read(mask.storage()),
			BufferBinding::write(output.storage()),
		];
		let inputs = [logits, labels, mask];
		engine.record_semantic(
			ComputeDispatch {
				kernel: KernelId::MatrixMaskedCategoricalAccuracyCountF32,
				buffers: &buffers,
				push_constants: &push_constants,
				workgroups: [1, 1, 1],
			},
			SemanticDispatch {
				contract,
				inputs: &inputs,
				outputs: &outputs,
				attributes: &[],
			},
		)?;
	} else {
		let buffers = [
			BufferBinding::read(logits.storage()),
			BufferBinding::read(labels.storage()),
			BufferBinding::write(output.storage()),
		];
		let inputs = [logits, labels];
		engine.record_semantic(
			ComputeDispatch {
				kernel: KernelId::MatrixCategoricalAccuracyCountF32,
				buffers: &buffers,
				push_constants: &push_constants,
				workgroups: [1, 1, 1],
			},
			SemanticDispatch {
				contract,
				inputs: &inputs,
				outputs: &outputs,
				attributes: &[],
			},
		)?;
	}
	Ok(output)
}

fn select_kernel(
	dtype: DType,
	routes: &[(DType, KernelId)],
	operation: &'static str,
) -> Result<KernelId> {
	routes
		.iter()
		.find_map(|(candidate, kernel)| (*candidate == dtype).then_some(*kernel))
		.ok_or_else(|| {
			Error::invalid_argument(format!(
				"{operation} does not support dtype {}",
				dtype.token()
			))
		})
}

include!("matrix/elemwise.gen.rs");
include!("matrix/blas.gen.rs");
include!("matrix/reduce.gen.rs");
