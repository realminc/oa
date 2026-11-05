//! Submit-time fusion of adjacent F32 elementwise chains.
//!
//! `MatrixFusedChainF32` interprets a host-written op tape: each four-word
//! step evaluates one semantic elementwise op over per-invocation registers,
//! leaf operand buffers, and `LOAD_IMM` immediates. Recording is unchanged;
//! this pass rewrites legal runs in the joined executable graph before
//! submission.
//!
//! A fused dispatch may emit several `STORE` steps: observed intermediates
//! (read outside the run or still owned outside the pending execution —
//! live `Matrix` values, autograd saves, observed capture outputs and stable
//! inputs, all detected through `Storage::owner_count`) materialize into
//! their real buffers while dead intermediates stay in registers. A write
//! stays a hard fusion boundary only when it is written by more than one
//! node, is read before its producer (aliasing), or carries a different
//! element count.
use std::sync::Arc;

use super::{
	Buffer, BufferAccess, Device, PushConstant, Storage,
	device::features::ExecutionProfile,
	dna::kernel::KernelId,
	executable_graph::{BufferUse, ComputeNode},
	semantic_graph::SemanticOpId,
};

// Tape ABI; must match `src/slang/matrix/elemwise/fused_chain.slang`.
const REGISTER_COUNT: u32 = 8;
const MAX_OPERAND_SLOTS: usize = 16;
const MAX_STORE_SLOTS: usize = 6;
const MAX_TAPE_STEPS: usize = 64;
const BINDING_SLOTS: usize = MAX_OPERAND_SLOTS + MAX_STORE_SLOTS + 1;

/// Above this element count the tape interpreter loses to the unfused
/// per-op kernels: per-element decode, branch, and shared-register traffic
/// outweighs the eliminated memory passes. Measured on HSW GT2 (chain-8
/// sweep in `core_elemwise_chain_bench`): fused wins through ~32K elements,
/// ties near 64K, and loses clearly at 1M. Routing policy, not a capability
/// bound.
const MAX_FUSION_ELEMENTS: u32 = 32 * 1024;

const OP_LOAD_IMM: u32 = 1;
const OP_STORE: u32 = 2;
const OP_ADD: u32 = 10;
const OP_SUB: u32 = 11;
const OP_MUL: u32 = 12;
const OP_DIV: u32 = 13;
const OP_POW: u32 = 14;
const OP_MIN: u32 = 15;
const OP_MAX: u32 = 16;
const OP_NEG: u32 = 20;
const OP_RECIP: u32 = 21;
const OP_ABS: u32 = 22;
const OP_EXP: u32 = 23;
const OP_LOG: u32 = 24;
const OP_SQRT: u32 = 25;
const OP_SIN: u32 = 26;
const OP_COS: u32 = 27;
const OP_TANH: u32 = 28;
const OP_SIGMOID: u32 = 30;
const OP_RELU: u32 = 31;
const OP_GELU: u32 = 32;
const OP_SILU: u32 = 33;

#[derive(Clone, Copy)]
enum ElementwiseShape {
	/// `[read, read, write]` buffers and one element-count push word.
	Binary,
	/// `[read, write]` buffers and one element-count push word.
	Unary,
	/// `[read, write]` buffers, element count plus an `f32` immediate.
	Scalar,
}

impl ElementwiseShape {
	const fn read_count(self) -> usize {
		match self {
			Self::Binary => 2,
			_ => 1,
		}
	}
}

/// One recorded node decoded into the fused kernel's tape vocabulary.
#[derive(Clone, Copy)]
struct NodeOp {
	opcode: u32,
	shape: ElementwiseShape,
	element_count: u32,
	scalar: f32,
}

fn elementwise_op(kernel: KernelId) -> Option<(u32, ElementwiseShape)> {
	Some(match kernel {
		KernelId::MatrixAddF32 => (OP_ADD, ElementwiseShape::Binary),
		KernelId::MatrixSubF32 => (OP_SUB, ElementwiseShape::Binary),
		KernelId::MatrixMulF32 => (OP_MUL, ElementwiseShape::Binary),
		KernelId::MatrixDivF32 => (OP_DIV, ElementwiseShape::Binary),
		KernelId::MatrixNegF32 => (OP_NEG, ElementwiseShape::Unary),
		KernelId::MatrixAbsF32 => (OP_ABS, ElementwiseShape::Unary),
		KernelId::MatrixLogF32 => (OP_LOG, ElementwiseShape::Unary),
		KernelId::MatrixSqrtF32 => (OP_SQRT, ElementwiseShape::Unary),
		KernelId::MatrixExpF32 => (OP_EXP, ElementwiseShape::Unary),
		KernelId::MatrixSinF32 => (OP_SIN, ElementwiseShape::Unary),
		KernelId::MatrixCosF32 => (OP_COS, ElementwiseShape::Unary),
		KernelId::MatrixReciprocalF32 => (OP_RECIP, ElementwiseShape::Unary),
		KernelId::MlTanhF32 => (OP_TANH, ElementwiseShape::Unary),
		KernelId::MlSigmoidF32 => (OP_SIGMOID, ElementwiseShape::Unary),
		KernelId::MlReluF32 => (OP_RELU, ElementwiseShape::Unary),
		KernelId::MlGeluF32 => (OP_GELU, ElementwiseShape::Unary),
		KernelId::MlSiluF32 => (OP_SILU, ElementwiseShape::Unary),
		KernelId::MatrixScaleF32 => (OP_MUL, ElementwiseShape::Scalar),
		KernelId::MatrixPowF32 => (OP_POW, ElementwiseShape::Scalar),
		KernelId::MatrixAddScalarF32 => (OP_ADD, ElementwiseShape::Scalar),
		KernelId::MatrixSubScalarF32 => (OP_SUB, ElementwiseShape::Scalar),
		KernelId::MatrixDivScalarF32 => (OP_DIV, ElementwiseShape::Scalar),
		KernelId::MatrixClampMinF32 => (OP_MAX, ElementwiseShape::Scalar),
		KernelId::MatrixClampMaxF32 => (OP_MIN, ElementwiseShape::Scalar),
		_ => return None,
	})
}

/// Whether a kernel is the lowering-only fused elementwise interpreter.
pub(in crate::runtime) fn is_fused_chain_kernel(kernel: KernelId) -> bool {
	kernel == KernelId::MatrixFusedChainF32
}

/// Whether `index` names the node's op-tape binding. The tape is always the
/// last binding of a fused dispatch and is internal state — it must not appear
/// as a user-rebindable plan input.
pub(in crate::runtime) fn is_tape_binding(node: &ComputeNode, index: usize) -> bool {
	is_fused_chain_kernel(node.kernel) && index + 1 == node.buffers.len()
}

/// Buffers bound to fused dispatches. Tape source words bake the resolved
/// storage-buffer index, so Strict-profile identity replacements (plan input
/// rebinding, alias arena materialization) must not swap them.
pub(in crate::runtime) fn buffer_is_fused_binding(nodes: &[ComputeNode], buffer: &Buffer) -> bool {
	nodes.iter().any(|node| {
		is_fused_chain_kernel(node.kernel)
			&& node
				.buffers
				.iter()
				.any(|usage| usage.buffer.same_as(buffer))
	})
}

/// Decode an admitted elementwise node, or `None` when its kernel or
/// physical shape lies outside the fused kernel's operand vocabulary.
fn decode_node(node: &ComputeNode) -> Option<NodeOp> {
	if node.indirect.is_some() {
		return None;
	}
	let (opcode, shape) = elementwise_op(node.kernel)?;
	let reads = shape.read_count();
	if node.buffers.len() != reads + 1
		|| node.buffers[..reads]
			.iter()
			.any(|usage| usage.access != BufferAccess::Read)
		|| node.buffers[reads].access != BufferAccess::Write
		|| node.buffers[..reads]
			.iter()
			.any(|usage| usage.buffer.same_as(&node.buffers[reads].buffer))
	{
		return None;
	}
	let scalar_pushes = usize::from(matches!(shape, ElementwiseShape::Scalar));
	if node.push_constants.len() != 1 + scalar_pushes {
		return None;
	}
	let element_count = match node.push_constants.first() {
		Some(PushConstant::U32(count)) if *count > 0 => *count,
		_ => return None,
	};
	let scalar = match (shape, node.push_constants.get(1)) {
		(ElementwiseShape::Scalar, Some(PushConstant::F32(value))) => *value,
		(ElementwiseShape::Scalar, _) => return None,
		(_, _) => 0.0,
	};
	Some(NodeOp {
		opcode,
		shape,
		element_count,
		scalar,
	})
}

fn write_buffer<'a>(node: &'a ComputeNode, op: &NodeOp) -> &'a Buffer {
	&node.buffers[op.shape.read_count()].buffer
}

/// Whether `buffer` is read by `node` (any non-write access).
fn node_reads(node: &ComputeNode, buffer: &Buffer) -> bool {
	node
		.buffers
		.iter()
		.any(|usage| usage.buffer.same_as(buffer) && usage.access != BufferAccess::Write)
}

/// Whether `node` writes `buffer` (write or read-write access).
fn node_writes(node: &ComputeNode, buffer: &Buffer) -> bool {
	node
		.buffers
		.iter()
		.any(|usage| usage.buffer.same_as(buffer) && usage.access != BufferAccess::Read)
}

/// Whether the storage owning `buffer` is referenced only by the pending
/// execution itself — no live `Matrix`, autograd save, stable input or other
/// external owner can observe its physical contents.
fn storage_unobservable(refs: &[&Storage], buffer: &Buffer) -> bool {
	let Some(storage) = refs.iter().find(|storage| {
		storage
			.buffer()
			.is_some_and(|candidate| candidate.same_as(buffer))
	}) else {
		return false;
	};
	let internal = refs
		.iter()
		.filter(|candidate| candidate.same_as(storage))
		.count();
	internal > 0 && storage.owner_count() == internal
}

struct FusedPlan {
	node: ComputeNode,
	tape: Storage,
}

/// Submit-time epilogue fusion of dedicated kernel pairs.
///
/// `MlLinearF32 → MatrixAddF32` collapses to `MlLinearResidualF32`,
/// `MlLinearF32 → elementwise activation` collapses to `MlLinearActF32`,
/// and `MatrixAddF32 → MlLayerNormF32` collapses to `MlResidualLayerNormF32`.
/// Each fused kernel materializes every buffer the unfused pair would have
/// written — the raw producer output plus the epilogue result — so autograd
/// saves and external readers observe identical storage. The fused node
/// takes the producer's slot, so adjacency preserves the recorded ordering
/// exactly; no operand becomes register-local and no observability proof is
/// needed. The win is one fewer dispatch and one fewer read pass over the
/// producer output. `refs` is accepted for symmetry with the tape pass but
/// unused.
pub(in crate::runtime) fn fuse_epilogue_pairs(
	device: &Arc<Device>,
	nodes: Vec<ComputeNode>,
	_refs: &[&Storage],
) -> Vec<ComputeNode> {
	if std::env::var_os("OA_DISABLE_EPILOGUE_FUSION").is_some() {
		return nodes;
	}
	let has_residual = device.require_kernel(KernelId::MlLinearResidualF32).is_ok();
	let has_act = device.require_kernel(KernelId::MlLinearActF32).is_ok();
	let has_residual_norm = device
		.require_kernel(KernelId::MlResidualLayerNormF32)
		.is_ok();
	let has_triple = device.require_kernel(KernelId::MlLinearTripleF32).is_ok();
	let has_embedding_pair = device.require_kernel(KernelId::MlEmbeddingPairF32).is_ok();
	let has_sum_axis_scale = device.require_kernel(KernelId::MlSumAxisScaleF32).is_ok();
	if !has_residual
		&& !has_act
		&& !has_residual_norm
		&& !has_triple
		&& !has_embedding_pair
		&& !has_sum_axis_scale
	{
		return nodes;
	}
	let mut result = Vec::with_capacity(nodes.len());
	let mut index = 0;
	while index < nodes.len() {
		if index + 2 < nodes.len()
			&& has_triple
			&& let Some(fused) = match_linear_triple(&nodes[index], &nodes[index + 1], &nodes[index + 2])
		{
			result.push(fused);
			index += 3;
			continue;
		}
		if index + 2 < nodes.len()
			&& has_embedding_pair
			&& let Some(fused) = match_embedding_pair(&nodes[index], &nodes[index + 1], &nodes[index + 2])
		{
			result.push(fused);
			index += 3;
			continue;
		}
		if index + 1 < nodes.len() {
			let fused = if has_residual {
				match_linear_residual(&nodes[index], &nodes[index + 1])
			} else {
				None
			};
			let fused =
				fused.or_else(|| has_act.then(|| match_linear_act(&nodes[index], &nodes[index + 1]))?);
			let fused = fused.or_else(|| {
				has_residual_norm.then(|| match_add_layer_norm(&nodes[index], &nodes[index + 1]))?
			});
			let fused = fused.or_else(|| {
				has_sum_axis_scale.then(|| match_sum_axis_scale(&nodes[index], &nodes[index + 1]))?
			});
			if let Some(fused) = fused {
				result.push(fused);
				index += 2;
				continue;
			}
		}
		result.push(nodes[index].clone());
		index += 1;
	}
	result
}

/// Decode the common `MlLinearF32` producer, or `None` when the node lies
/// outside the fused kernel's buffer and scalar vocabulary. Returns the
/// linear output buffer plus `(batch, output_features)` for coverage
/// checks against the consumer node.
fn match_linear_producer(linear: &ComputeNode) -> Option<(&Buffer, u32, u32)> {
	if linear.kernel != KernelId::MlLinearF32
		|| linear.indirect.is_some()
		|| linear.buffers.len() != 4
		|| linear.push_constants.len() != 3
		|| linear
			.buffers
			.iter()
			.take(3)
			.any(|usage| usage.access != BufferAccess::Read)
		|| linear.buffers[3].access != BufferAccess::Write
	{
		return None;
	}
	let (PushConstant::U32(batch), PushConstant::U32(output_features)) =
		(linear.push_constants[0], linear.push_constants[2])
	else {
		return None;
	};
	Some((&linear.buffers[3].buffer, batch, output_features))
}

/// Merge two fused nodes' semantic op owners so diagnostics keep reporting
/// the participating operations.
fn fused_node(
	linear: &ComputeNode,
	consumer: &ComputeNode,
	kernel: KernelId,
	buffers: Vec<BufferUse>,
	push_constants: Vec<PushConstant>,
	workgroups: [u32; 3],
) -> ComputeNode {
	let semantic_ops = merged_semantic_ops(&[&linear.semantic_ops, &consumer.semantic_ops]);
	let (operation, op_contract_hash) = fused_identity(&semantic_ops, linear, kernel);
	ComputeNode {
		operation,
		kernel,
		buffers,
		push_constants,
		workgroups,
		indirect: None,
		semantic_ops,
		op_contract_hash,
	}
}

/// Concatenate semantic operation owners preserving order while dropping
/// duplicates — split-semantic operations record several physical nodes under
/// one owner, and the executable graph rejects a node that repeats an owner.
fn merged_semantic_ops(parts: &[&[SemanticOpId]]) -> Vec<SemanticOpId> {
	let mut merged = Vec::new();
	for ops in parts {
		for op in ops.iter().copied() {
			if !merged.contains(&op) {
				merged.push(op);
			}
		}
	}
	merged
}

/// Node identity after a merge: a fused node owning several semantic
/// operations reports the fused kernel name and a distinct-implementation
/// hash of zero, while a node that still owns exactly one operation must keep
/// that operation's name and contract hash or the executable graph rejects
/// the identity.
fn fused_identity(
	merged: &[SemanticOpId],
	first: &ComputeNode,
	kernel: KernelId,
) -> (&'static str, u64) {
	if merged.len() == 1 {
		(first.operation, first.op_contract_hash)
	} else {
		(kernel.report_name(), 0)
	}
}

/// Decode the `linear → add` residual pair, or `None` when either node lies
/// outside the fused kernel's buffer and scalar vocabulary. The sum may not
/// alias any bound operand; residual may alias only read operands.
fn match_linear_residual(linear: &ComputeNode, add: &ComputeNode) -> Option<ComputeNode> {
	let (linear_output, batch, output_features) = match_linear_producer(linear)?;
	if add.kernel != KernelId::MatrixAddF32
		|| add.indirect.is_some()
		|| add.buffers.len() != 3
		|| add.push_constants.len() != 1
		|| add
			.buffers
			.iter()
			.take(2)
			.any(|usage| usage.access != BufferAccess::Read)
		|| add.buffers[2].access != BufferAccess::Write
	{
		return None;
	}
	let residual = match (
		add.buffers[0].buffer.same_as(linear_output),
		add.buffers[1].buffer.same_as(linear_output),
	) {
		(true, false) => &add.buffers[1].buffer,
		(false, true) => &add.buffers[0].buffer,
		_ => return None,
	};
	// The elementwise sum must cover the full [batch, output_features] tile.
	if add.push_constants[0] != PushConstant::U32(batch.checked_mul(output_features)?) {
		return None;
	}
	let sum = &add.buffers[2].buffer;
	let collides = |buffer: &Buffer| {
		linear
			.buffers
			.iter()
			.any(|usage| usage.buffer.same_as(buffer))
			|| residual.same_as(buffer)
	};
	if collides(sum) {
		return None;
	}
	let kernel = KernelId::MlLinearResidualF32;
	Some(fused_node(
		linear,
		add,
		kernel,
		vec![
			linear.buffers[0].clone(),
			linear.buffers[1].clone(),
			linear.buffers[2].clone(),
			BufferUse {
				buffer: residual.clone(),
				access: BufferAccess::Read,
			},
			BufferUse {
				buffer: linear_output.clone(),
				access: BufferAccess::Write,
			},
			BufferUse {
				buffer: sum.clone(),
				access: BufferAccess::Write,
			},
		],
		linear.push_constants.clone(),
		linear.workgroups,
	))
}

/// Decode three adjacent `MlLinearF32` nodes sharing one input and one
/// geometry — the Q/K/V projection group — into a single z-dispatched node.
/// Outputs must be three distinct buffers that alias none of the reads.
fn match_linear_triple(
	first: &ComputeNode,
	second: &ComputeNode,
	third: &ComputeNode,
) -> Option<ComputeNode> {
	let (first_output, _, _) = match_linear_producer(first)?;
	let (second_output, _, _) = match_linear_producer(second)?;
	let (third_output, _, _) = match_linear_producer(third)?;
	if first.push_constants != second.push_constants
		|| first.push_constants != third.push_constants
		|| first.workgroups != second.workgroups
		|| first.workgroups != third.workgroups
		|| !second.buffers[0].buffer.same_as(&first.buffers[0].buffer)
		|| !third.buffers[0].buffer.same_as(&first.buffers[0].buffer)
	{
		return None;
	}
	if first_output.same_as(second_output)
		|| second_output.same_as(third_output)
		|| first_output.same_as(third_output)
	{
		return None;
	}
	let collides = |buffer: &Buffer| {
		[first, second, third].iter().any(|node| {
			node
				.buffers
				.iter()
				.take(3)
				.any(|usage| usage.buffer.same_as(buffer))
		})
	};
	if [first_output, second_output, third_output]
		.iter()
		.any(|buffer| collides(buffer))
	{
		return None;
	}
	let kernel = KernelId::MlLinearTripleF32;
	let mut workgroups = first.workgroups;
	workgroups[2] = 3;
	let semantic_ops = merged_semantic_ops(&[
		&first.semantic_ops,
		&second.semantic_ops,
		&third.semantic_ops,
	]);
	let (operation, op_contract_hash) = fused_identity(&semantic_ops, first, kernel);
	Some(ComputeNode {
		operation,
		kernel,
		buffers: vec![
			first.buffers[0].clone(),
			first.buffers[1].clone(),
			second.buffers[1].clone(),
			third.buffers[1].clone(),
			first.buffers[2].clone(),
			second.buffers[2].clone(),
			third.buffers[2].clone(),
			BufferUse {
				buffer: first_output.clone(),
				access: BufferAccess::Write,
			},
			BufferUse {
				buffer: second_output.clone(),
				access: BufferAccess::Write,
			},
			BufferUse {
				buffer: third_output.clone(),
				access: BufferAccess::Write,
			},
		],
		push_constants: first.push_constants.clone(),
		workgroups,
		indirect: None,
		semantic_ops,
		op_contract_hash,
	})
}

/// Decode one `embedding`/`embedding_u8` gather node into its buffers and
/// scalar geometry, or `None` outside the fused kernel's vocabulary.
fn match_embedding(node: &ComputeNode) -> Option<(&Buffer, bool)> {
	if !matches!(
		node.kernel,
		KernelId::MlEmbeddingF32 | KernelId::MlEmbeddingU8F32
	) || node.indirect.is_some()
		|| node.buffers.len() != 3
		|| node.push_constants.len() != 3
		|| node.buffers[0].access != BufferAccess::Read
		|| node.buffers[1].access != BufferAccess::Read
		|| node.buffers[2].access != BufferAccess::Write
	{
		return None;
	}
	Some((
		&node.buffers[2].buffer,
		node.kernel == KernelId::MlEmbeddingU8F32,
	))
}

/// Decode the `embedding → embedding → add` token/position head into one
/// fused gather pair that writes both gathered tensors and their sum. Both
/// gathers must share index count and embedding width so the sum covers the
/// full element domain; every write lands on a distinct buffer aliasing none
/// of the reads.
fn match_embedding_pair(
	first: &ComputeNode,
	second: &ComputeNode,
	add: &ComputeNode,
) -> Option<ComputeNode> {
	let (first_output, first_u8) = match_embedding(first)?;
	let (second_output, second_u8) = match_embedding(second)?;
	if add.kernel != KernelId::MatrixAddF32
		|| add.indirect.is_some()
		|| add.buffers.len() != 3
		|| add.push_constants.len() != 1
		|| add
			.buffers
			.iter()
			.take(2)
			.any(|usage| usage.access != BufferAccess::Read)
		|| add.buffers[2].access != BufferAccess::Write
		|| first.push_constants[0] != second.push_constants[0]
		|| first.push_constants[2] != second.push_constants[2]
	{
		return None;
	}
	if add.push_constants[0]
		!= match (first.push_constants[0], first.push_constants[2]) {
			(PushConstant::U32(count), PushConstant::U32(dim)) => {
				PushConstant::U32(count.checked_mul(dim)?)
			}
			_ => return None,
		} {
		return None;
	}
	let consumes = |a: &BufferUse, b: &BufferUse| {
		(a.buffer.same_as(first_output) && b.buffer.same_as(second_output))
			|| (a.buffer.same_as(second_output) && b.buffer.same_as(first_output))
	};
	if !consumes(&add.buffers[0], &add.buffers[1]) {
		return None;
	}
	let sum = &add.buffers[2].buffer;
	let collides = |buffer: &Buffer| {
		[first, second]
			.iter()
			.flat_map(|node| node.buffers.iter())
			.any(|usage| usage.buffer.same_as(buffer))
	};
	if first_output.same_as(second_output) || collides(sum) {
		return None;
	}
	let kernel = KernelId::MlEmbeddingPairF32;
	let semantic_ops =
		merged_semantic_ops(&[&first.semantic_ops, &second.semantic_ops, &add.semantic_ops]);
	let (operation, op_contract_hash) = fused_identity(&semantic_ops, first, kernel);
	let u8_flag = |packed: bool| PushConstant::U32(packed as u32);
	Some(ComputeNode {
		operation,
		kernel,
		buffers: vec![
			first.buffers[0].clone(),
			first.buffers[1].clone(),
			second.buffers[0].clone(),
			second.buffers[1].clone(),
			BufferUse {
				buffer: first_output.clone(),
				access: BufferAccess::Write,
			},
			BufferUse {
				buffer: second_output.clone(),
				access: BufferAccess::Write,
			},
			BufferUse {
				buffer: sum.clone(),
				access: BufferAccess::Write,
			},
		],
		push_constants: vec![
			first.push_constants[0],
			first.push_constants[1],
			second.push_constants[1],
			first.push_constants[2],
			u8_flag(first_u8),
			u8_flag(second_u8),
		],
		workgroups: first.workgroups,
		indirect: None,
		semantic_ops,
		op_contract_hash,
	})
}

/// Decode the `sum_axis → scale` reduction tail — `matrix::mean`'s split
/// lowering and MoE auxiliary-loss reductions share it — into one fused node
/// writing both the raw sum and the scaled result. The scaled output may not
/// alias any other bound buffer.
fn match_sum_axis_scale(sum: &ComputeNode, scale: &ComputeNode) -> Option<ComputeNode> {
	if sum.kernel != KernelId::MatrixSumAxisF32
		|| sum.indirect.is_some()
		|| sum.buffers.len() != 2
		|| sum.push_constants.len() != 3
		|| sum.buffers[0].access != BufferAccess::Read
		|| sum.buffers[1].access != BufferAccess::Write
		|| scale.kernel != KernelId::MatrixScaleF32
		|| scale.indirect.is_some()
		|| scale.buffers.len() != 2
		|| scale.push_constants.len() != 2
		|| scale.buffers[0].access != BufferAccess::Read
		|| scale.buffers[1].access != BufferAccess::Write
		|| !scale.buffers[0].buffer.same_as(&sum.buffers[1].buffer)
	{
		return None;
	}
	let output_count = match (sum.push_constants[0], sum.push_constants[2]) {
		(PushConstant::U32(outer), PushConstant::U32(inner)) => outer.checked_mul(inner)?,
		_ => return None,
	};
	if scale.push_constants[0] != PushConstant::U32(output_count)
		|| !matches!(scale.push_constants[1], PushConstant::F32(_))
	{
		return None;
	}
	let scaled = &scale.buffers[1].buffer;
	if scaled.same_as(&sum.buffers[0].buffer) || scaled.same_as(&sum.buffers[1].buffer) {
		return None;
	}
	let kernel = KernelId::MlSumAxisScaleF32;
	let semantic_ops = merged_semantic_ops(&[&sum.semantic_ops, &scale.semantic_ops]);
	let (operation, op_contract_hash) = fused_identity(&semantic_ops, sum, kernel);
	Some(ComputeNode {
		operation,
		kernel,
		buffers: vec![
			sum.buffers[0].clone(),
			sum.buffers[1].clone(),
			BufferUse {
				buffer: scaled.clone(),
				access: BufferAccess::Write,
			},
		],
		push_constants: vec![
			sum.push_constants[0],
			sum.push_constants[1],
			sum.push_constants[2],
			scale.push_constants[1],
		],
		workgroups: sum.workgroups,
		indirect: None,
		semantic_ops,
		op_contract_hash,
	})
}

/// Map an elementwise activation kernel to its `linear_act` epilogue code,
/// matching the `act_kind` vocabulary in `linear_act.slang`.
fn act_kind(kernel: KernelId) -> Option<u32> {
	match kernel {
		KernelId::MlReluF32 => Some(0),
		KernelId::MlGeluF32 => Some(1),
		KernelId::MlSiluF32 => Some(2),
		KernelId::MlTanhF32 => Some(3),
		KernelId::MlSigmoidF32 => Some(4),
		KernelId::MlMishF32 => Some(5),
		KernelId::MlEluF32 => Some(6),
		KernelId::MlLeakyReluF32 => Some(7),
		KernelId::MlSoftplusF32 => Some(8),
		_ => None,
	}
}

/// Decode the `add → layer_norm` pair, or `None` when either node lies
/// outside the fused kernel's buffer and scalar vocabulary. The norm's input
/// must be exactly the recorded sum; every write buffer must be disjoint
/// from the read operands.
fn match_add_layer_norm(add: &ComputeNode, norm: &ComputeNode) -> Option<ComputeNode> {
	if add.kernel != KernelId::MatrixAddF32
		|| norm.kernel != KernelId::MlLayerNormF32
		|| add.indirect.is_some()
		|| norm.indirect.is_some()
		|| add.buffers.len() != 3
		|| norm.buffers.len() != 6
		|| add.push_constants.len() != 1
		|| norm.push_constants.len() != 3
		|| add
			.buffers
			.iter()
			.take(2)
			.any(|usage| usage.access != BufferAccess::Read)
		|| add.buffers[2].access != BufferAccess::Write
		|| norm
			.buffers
			.iter()
			.take(3)
			.any(|usage| usage.access != BufferAccess::Read)
		|| norm
			.buffers
			.iter()
			.skip(3)
			.any(|usage| usage.access != BufferAccess::Write)
	{
		return None;
	}
	let sum = &add.buffers[2].buffer;
	if !norm.buffers[0].buffer.same_as(sum) {
		return None;
	}
	let (PushConstant::U32(rows), PushConstant::U32(columns)) =
		(norm.push_constants[0], norm.push_constants[1])
	else {
		return None;
	};
	if add.push_constants[0] != PushConstant::U32(rows.checked_mul(columns)?) {
		return None;
	}
	let collides = |buffer: &Buffer| {
		add
			.buffers
			.iter()
			.take(2)
			.any(|usage| usage.buffer.same_as(buffer))
			|| norm
				.buffers
				.iter()
				.skip(1)
				.take(2)
				.any(|usage| usage.buffer.same_as(buffer))
	};
	if collides(sum)
		|| norm
			.buffers
			.iter()
			.skip(3)
			.any(|usage| collides(&usage.buffer))
	{
		return None;
	}
	let kernel = KernelId::MlResidualLayerNormF32;
	Some(fused_node(
		add,
		norm,
		kernel,
		vec![
			add.buffers[0].clone(),
			add.buffers[1].clone(),
			norm.buffers[1].clone(),
			norm.buffers[2].clone(),
			BufferUse {
				buffer: sum.clone(),
				access: BufferAccess::Write,
			},
			norm.buffers[3].clone(),
			norm.buffers[4].clone(),
			norm.buffers[5].clone(),
		],
		norm.push_constants.clone(),
		norm.workgroups,
	))
}

/// Decode the `linear → activation` pair, or `None` when either node lies
/// outside the fused kernel's buffer and scalar vocabulary. The activation
/// must consume the linear output elementwise over the full tile; `elu` and
/// `leaky_relu` carry their `alpha` push through.
fn match_linear_act(linear: &ComputeNode, act: &ComputeNode) -> Option<ComputeNode> {
	let (linear_output, batch, output_features) = match_linear_producer(linear)?;
	let kind = act_kind(act.kernel)?;
	if act.indirect.is_some()
		|| act.buffers.len() != 2
		|| act.buffers[0].access != BufferAccess::Read
		|| act.buffers[1].access != BufferAccess::Write
		|| !act.buffers[0].buffer.same_as(linear_output)
	{
		return None;
	}
	let (elements, alpha) = match act.push_constants.as_slice() {
		[PushConstant::U32(elements)] => (*elements, 0.0),
		[PushConstant::U32(elements), PushConstant::F32(alpha)] => (*elements, *alpha),
		_ => return None,
	};
	if elements != batch.checked_mul(output_features)? {
		return None;
	}
	let output = &act.buffers[1].buffer;
	let collides = |buffer: &Buffer| {
		linear
			.buffers
			.iter()
			.any(|usage| usage.buffer.same_as(buffer))
	};
	if collides(output) {
		return None;
	}
	let kernel = KernelId::MlLinearActF32;
	let mut push_constants = linear.push_constants.clone();
	push_constants.push(PushConstant::U32(kind));
	push_constants.push(PushConstant::F32(alpha));
	Some(fused_node(
		linear,
		act,
		kernel,
		vec![
			linear.buffers[0].clone(),
			linear.buffers[1].clone(),
			linear.buffers[2].clone(),
			BufferUse {
				buffer: linear_output.clone(),
				access: BufferAccess::Write,
			},
			BufferUse {
				buffer: output.clone(),
				access: BufferAccess::Write,
			},
		],
		push_constants,
		linear.workgroups,
	))
}

/// Rewrite the joined node list, splicing legal elementwise runs into fused
/// `MatrixFusedChainF32` dispatches. `refs` must contain every `Storage`
/// reference the pending execution holds; an intermediate whose storage has
/// no other owner proves nobody can observe its physical contents. The
/// returned storages retain the generated tapes for the pending lifetime.
pub(in crate::runtime) fn fuse_elementwise_nodes(
	device: &Arc<Device>,
	nodes: Vec<ComputeNode>,
	refs: &[&Storage],
) -> (Vec<ComputeNode>, Vec<Storage>) {
	// A missing fused pipeline falls back to the recorded nodes; the per-op
	// kernels remain the canonical path. The environment gate exists for
	// fused-vs-unfused measurement on one binary.
	if std::env::var_os("OA_DISABLE_ELEMENTWISE_FUSION").is_some()
		|| device
			.require_kernel(KernelId::MatrixFusedChainF32)
			.is_err()
	{
		return (nodes, Vec::new());
	}
	let decoded: Vec<Option<NodeOp>> = nodes.iter().map(decode_node).collect();
	let readers = |buffer: &Buffer| -> Vec<usize> {
		nodes
			.iter()
			.enumerate()
			.filter(|(_, node)| node_reads(node, buffer))
			.map(|(index, _)| index)
			.collect()
	};

	let mut result = Vec::with_capacity(nodes.len());
	let mut tapes = Vec::new();
	let mut start = 0;
	while start < nodes.len() {
		if decoded[start].is_none() {
			result.push(nodes[start].clone());
			start += 1;
			continue;
		}
		// Maximal contiguous run of admitted nodes.
		let mut run_end = start + 1;
		while run_end < nodes.len() && decoded[run_end].is_some() {
			run_end += 1;
		}
		let first_count = decoded[start].expect("run node decoded").element_count;
		let mut boundary = vec![false; run_end];
		for i in start..run_end {
			let op = decoded[i].expect("run node decoded");
			let write = write_buffer(&nodes[i], &op);
			let written_elsewhere = nodes
				.iter()
				.enumerate()
				.any(|(j, node)| j != i && node_writes(node, write));
			// Hard boundaries only: a write read before its producer
			// (aliasing), written by another node, or sized differently can
			// never live inside a fused dispatch. Escaping readers and
			// external owners materialize through STORE steps instead.
			boundary[i] = readers(write).iter().any(|reader| *reader < i)
				|| written_elsewhere
				|| (i > start && op.element_count != first_count);
		}
		// Emit segments ending at each hard boundary (inclusive). Capacity
		// overflow backs off to the nearest earlier cut; every cut is legal
		// because escaping writes materialize through STORE steps.
		let mut cursor = start;
		while cursor < run_end {
			let hard_end = boundary[cursor..run_end]
				.iter()
				.position(|flag| *flag)
				.map(|offset| cursor + offset)
				.unwrap_or(run_end - 1);
			let mut end = hard_end;
			let mut plan = if end > cursor {
				fit_segment(device, &nodes, &decoded, refs, cursor, end)
			} else {
				None
			};
			while plan.is_none() && end > cursor {
				end -= 1;
				plan = fit_segment(device, &nodes, &decoded, refs, cursor, end);
			}
			match plan {
				Some(plan) if end > cursor => {
					result.push(plan.node);
					tapes.push(plan.tape);
					cursor = end + 1;
				}
				_ => {
					result.push(nodes[cursor].clone());
					cursor += 1;
				}
			}
		}
		start = run_end;
	}
	(result, tapes)
}

/// Try to encode `nodes[start..=end]` into one fused dispatch; `None` when
/// operand, register, store, tape, or element-count routing policy rejects
/// the region.
fn fit_segment(
	device: &Arc<Device>,
	nodes: &[ComputeNode],
	decoded: &[Option<NodeOp>],
	refs: &[&Storage],
	start: usize,
	end: usize,
) -> Option<FusedPlan> {
	if end + 1 - start < 2 {
		return None;
	}
	let ops: Vec<NodeOp> = decoded[start..=end]
		.iter()
		.copied()
		.collect::<Option<_>>()?;
	let element_count = ops[0].element_count;
	if element_count > MAX_FUSION_ELEMENTS || ops.iter().any(|op| op.element_count != element_count) {
		return None;
	}
	// Observed writes materialize through STORE steps: escaping readers,
	// external owners, and the segment output itself.
	let needs_store: Vec<bool> = (start..=end)
		.map(|i| {
			if i == end {
				return true;
			}
			let op = decoded[i].expect("segment node decoded");
			let write = write_buffer(&nodes[i], &op);
			!storage_unobservable(refs, write)
				|| nodes
					.iter()
					.enumerate()
					.any(|(j, consumer)| !(start..=end).contains(&j) && node_reads(consumer, write))
		})
		.collect();
	if needs_store.iter().filter(|store| **store).count() > MAX_STORE_SLOTS {
		return None;
	}
	// Last in-segment consumer per produced register; a stored result is
	// consumed by its STORE step immediately after its compute step.
	let mut last_use: Vec<usize> = needs_store
		.iter()
		.enumerate()
		.map(|(i, store)| if *store { i } else { usize::MAX })
		.collect();
	for (i, op) in ops.iter().enumerate() {
		let write = write_buffer(&nodes[start + i], op);
		for (j, consumer) in nodes[start..=end].iter().enumerate().skip(i + 1) {
			if node_reads(consumer, write) {
				last_use[i] = j;
			}
		}
	}

	let profile = device.physical().profile;
	let mut operands: Vec<Buffer> = Vec::new();
	let mut stores: Vec<Buffer> = Vec::new();
	let mut registers = vec![0_u32; ops.len()];
	let mut free: Vec<u32> = (0..REGISTER_COUNT).rev().collect();
	let mut tape: Vec<u32> = Vec::new();

	for (i, op) in ops.iter().enumerate() {
		let node = &nodes[start + i];
		let mut resolve = |usage: &BufferUse| -> Option<u32> {
			// Interior: produced earlier in this segment -> register.
			for p in (0..i).rev() {
				if write_buffer(&nodes[start + p], &ops[p]).same_as(&usage.buffer) {
					return Some(registers[p]);
				}
			}
			// Leaf: resolved storage-buffer array index — the local slot on
			// Compatibility, the heap descriptor index on Strict.
			let slot = match operands
				.iter()
				.position(|candidate| candidate.same_as(&usage.buffer))
			{
				Some(slot) => slot as u32,
				None => {
					if operands.len() >= MAX_OPERAND_SLOTS {
						return None;
					}
					operands.push(usage.buffer.clone());
					(operands.len() - 1) as u32
				}
			};
			let resolved = match profile {
				ExecutionProfile::Compatibility => slot,
				ExecutionProfile::Strict => usage.buffer.descriptor_index(),
			};
			Some(REGISTER_COUNT + resolved)
		};

		let dst = free.pop()?;
		if matches!(op.shape, ElementwiseShape::Scalar) {
			let tmp = free.pop()?;
			tape.extend_from_slice(&[OP_LOAD_IMM, op.scalar.to_bits(), 0, tmp]);
			let a = resolve(&node.buffers[0])?;
			tape.extend_from_slice(&[op.opcode, a, tmp, dst]);
			free.push(tmp);
		} else {
			let a = resolve(&node.buffers[0])?;
			let b = if matches!(op.shape, ElementwiseShape::Binary) {
				resolve(&node.buffers[1])?
			} else {
				0
			};
			tape.extend_from_slice(&[op.opcode, a, b, dst]);
		}
		registers[i] = dst;
		if needs_store[i] {
			// The STORE target resolves like an operand source: bound store
			// slot on Compatibility, heap descriptor index on Strict.
			let store_slot = stores.len();
			let write = write_buffer(node, op);
			stores.push(write.clone());
			let resolved = match profile {
				ExecutionProfile::Compatibility => (MAX_OPERAND_SLOTS + store_slot) as u32,
				ExecutionProfile::Strict => write.descriptor_index(),
			};
			tape.extend_from_slice(&[OP_STORE, dst, REGISTER_COUNT + resolved, 0]);
		}
		// Release registers whose last in-segment consumer was this node.
		for (p, register) in registers.iter().enumerate().take(i + 1) {
			if last_use[p] == i {
				free.push(*register);
			}
		}
		if tape.len() / 4 > MAX_TAPE_STEPS {
			return None;
		}
	}

	let mut bytes = Vec::with_capacity(tape.len() * 4);
	for word in &tape {
		bytes.extend_from_slice(&word.to_le_bytes());
	}
	let tape_storage = Storage::new_from_bytes(device, &bytes).ok()?;
	let tape_buffer = tape_storage.buffer()?.clone();

	let first_operand = operands.first()?.clone();
	let mut buffers = Vec::with_capacity(BINDING_SLOTS);
	for slot in 0..MAX_OPERAND_SLOTS {
		buffers.push(BufferUse {
			buffer: operands
				.get(slot)
				.cloned()
				.unwrap_or_else(|| first_operand.clone()),
			access: BufferAccess::Read,
		});
	}
	for slot in 0..MAX_STORE_SLOTS {
		match stores.get(slot) {
			Some(buffer) => buffers.push(BufferUse {
				buffer: buffer.clone(),
				access: BufferAccess::Write,
			}),
			// Unused store slots stay read-only so they carry no write
			// contract and create no false hazards.
			None => buffers.push(BufferUse {
				buffer: first_operand.clone(),
				access: BufferAccess::Read,
			}),
		}
	}
	buffers.push(BufferUse {
		buffer: tape_buffer,
		access: BufferAccess::Read,
	});
	debug_assert_eq!(buffers.len(), BINDING_SLOTS);

	let kernel = KernelId::MatrixFusedChainF32;
	let semantic_ops = merged_semantic_ops(
		&nodes[start..=end]
			.iter()
			.map(|node| node.semantic_ops.as_slice())
			.collect::<Vec<_>>(),
	);
	let (operation, op_contract_hash) = fused_identity(&semantic_ops, &nodes[start], kernel);
	Some(FusedPlan {
		node: ComputeNode {
			operation,
			kernel,
			buffers,
			push_constants: vec![
				PushConstant::U32(element_count),
				PushConstant::U32((tape.len() / 4) as u32),
			],
			workgroups: kernel.linear_workgroups(element_count),
			indirect: None,
			semantic_ops,
			op_contract_hash,
		},
		tape: tape_storage,
	})
}
