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

pub(in crate::runtime) mod census;

use census::NodeOutcome;

use super::{
	Buffer, BufferAccess, Device, PushConstant, Storage,
	device::features::ExecutionProfile,
	dna::{
		elementwise::{Instruction, MAX_OPERAND_SLOTS, MAX_STORE_SLOTS, Program, Source},
		kernel::KernelId,
	},
	executable_graph::{BufferUse, ComputeNode, JitNode},
	semantic_graph::SemanticOpId,
};

const BINDING_SLOTS: usize = MAX_OPERAND_SLOTS + MAX_STORE_SLOTS + 1;

/// Above this element count the tape interpreter loses to the unfused
/// per-op kernels: per-element decode, branch, and shared-register traffic
/// outweighs the eliminated memory passes. Measured on HSW GT2 (chain-8
/// sweep in `core_elemwise_chain_bench`): fused wins through ~32K elements,
/// ties near 64K, and loses clearly at 1M. Routing policy, not a capability
/// bound.
const MAX_FUSION_ELEMENTS: u32 = 32 * 1024;

const OP_GATHER: u32 = 4;
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
	Broadcast,
	Gather,
	BinaryScalar,
}

impl ElementwiseShape {
	const fn read_count(self) -> usize {
		match self {
			Self::Binary | Self::Broadcast | Self::Gather | Self::BinaryScalar => 2,
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
	// Broadcast shape plus two input strides, or gather {rows,width,dtype}.
	metadata: [u32; 25],
}

fn elementwise_op(kernel: KernelId) -> Option<(u32, ElementwiseShape)> {
	Some(match kernel {
		KernelId::MatrixAddBroadcastF32 | KernelId::MatrixBiasAddBroadcastF32 => {
			(OP_ADD, ElementwiseShape::Broadcast)
		}
		KernelId::MatrixSubBroadcastF32 => (OP_SUB, ElementwiseShape::Broadcast),
		KernelId::MatrixMulBroadcastF32 => (OP_MUL, ElementwiseShape::Broadcast),
		KernelId::MatrixDivBroadcastF32 => (OP_DIV, ElementwiseShape::Broadcast),
		KernelId::MatrixGatherF32 => (OP_GATHER, ElementwiseShape::Gather),
		KernelId::MlEluBackwardF32 => (47, ElementwiseShape::BinaryScalar),
		KernelId::MlLeakyReluBackwardF32 => (48, ElementwiseShape::BinaryScalar),
		KernelId::MlReluBackwardF32 => (40, ElementwiseShape::Binary),
		KernelId::MlGeluBackwardF32 => (41, ElementwiseShape::Binary),
		KernelId::MlSiluBackwardF32 => (42, ElementwiseShape::Binary),
		KernelId::MlTanhBackwardF32 => (43, ElementwiseShape::Binary),
		KernelId::MlSigmoidBackwardF32 => (44, ElementwiseShape::Binary),
		KernelId::MlSoftplusBackwardF32 => (45, ElementwiseShape::Binary),
		KernelId::MlMishBackwardF32 => (46, ElementwiseShape::Binary),
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
	let scalar_pushes = usize::from(matches!(
		shape,
		ElementwiseShape::Scalar | ElementwiseShape::BinaryScalar
	));
	let push_count = match shape {
		ElementwiseShape::Broadcast => 26,
		ElementwiseShape::Gather => 4,
		_ => 1 + scalar_pushes,
	};
	if node.push_constants.len() != push_count {
		return None;
	}
	let element_count = match node.push_constants.first() {
		Some(PushConstant::U32(count)) if *count > 0 => *count,
		_ => return None,
	};
	let scalar = match (shape, node.push_constants.get(1)) {
		(ElementwiseShape::Scalar | ElementwiseShape::BinaryScalar, Some(PushConstant::F32(value))) => {
			*value
		}
		(ElementwiseShape::Scalar | ElementwiseShape::BinaryScalar, _) => return None,
		(_, _) => 0.0,
	};
	let mut metadata = [0u32; 25];
	if matches!(
		shape,
		ElementwiseShape::Broadcast | ElementwiseShape::Gather
	) {
		for (word, push) in metadata.iter_mut().zip(&node.push_constants[1..]) {
			let PushConstant::U32(value) = push else {
				return None;
			};
			*word = *value;
		}
	}
	let element_count = if matches!(shape, ElementwiseShape::Gather) {
		if metadata[1] == 0 || metadata[2] > 1 {
			return None;
		}
		element_count.checked_mul(metadata[1])?
	} else {
		element_count
	};
	if matches!(shape, ElementwiseShape::Broadcast) {
		let rank = usize::try_from(metadata[0]).ok()?;
		if !(1..=8).contains(&rank)
			|| metadata[1..1 + rank]
				.iter()
				.try_fold(1u32, |a, b| a.checked_mul(*b))?
				!= element_count
		{
			return None;
		}
	}
	if node.workgroups != node.kernel.linear_workgroups(element_count) {
		return None;
	}
	Some(NodeOp {
		opcode,
		shape,
		element_count,
		scalar,
		metadata,
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

/// Whether replacing `sources` with `fused` hides no observable write: every
/// buffer a source writes is either still written by a replacement node or
/// owned by nothing outside the pending execution. In-graph readers are the
/// matcher's responsibility; this closes the external-owner half of the proof.
fn elided_writes_unobservable<'a>(
	refs: &[&Storage],
	sources: impl IntoIterator<Item = &'a ComputeNode>,
	fused: &[&ComputeNode],
) -> bool {
	sources
		.into_iter()
		.flat_map(|node| node.buffers.iter())
		.filter(|usage| usage.access != BufferAccess::Read)
		.all(|usage| {
			fused.iter().any(|node| node_writes(node, &usage.buffer))
				|| storage_unobservable(refs, &usage.buffer)
		})
}

/// Whether none of `slots` was already absorbed by an earlier lookahead match.
/// A consumed slot either carries a deferred replacement or none at all, so
/// absorbing it again would fuse its work twice or skip that replacement.
fn slots_free(
	consumed: &std::collections::BTreeSet<usize>,
	mut slots: impl Iterator<Item = usize>,
) -> bool {
	!slots.any(|slot| consumed.contains(&slot))
}

struct FusedPlan {
	node: ComputeNode,
	tape: Storage,
}

/// Submit-time structural fusion through dedicated physical providers.
///
/// Covers arithmetic epilogues, router selection, packed causal attention,
/// concat projections, Conv lowering tails and backward prologues. Every
/// replacement retains all semantic owners and materializes each source write
/// that a later node reads or that any owner outside the pending execution
/// still holds: `refs` must contain every `Storage` reference the pending
/// execution holds, exactly as for [`fuse_elementwise_nodes`]. A lookahead
/// match never absorbs a slot an earlier match consumed. An independent
/// balancing Add may move before softmax only after checking their read/write
/// independence. Barriers are rebuilt from the resulting buffer accesses.
pub(in crate::runtime) fn fuse_epilogue_pairs(
	device: &Arc<Device>,
	nodes: Vec<ComputeNode>,
	refs: &[&Storage],
) -> Vec<ComputeNode> {
	if std::env::var_os("OA_DISABLE_EPILOGUE_FUSION").is_some() {
		return nodes;
	}
	let rl_fusion = std::env::var_os("OA_DISABLE_RL_FUSION").is_none();
	let has_residual = device.require_kernel(KernelId::MlLinearResidualF32).is_ok();
	let has_act = device.require_kernel(KernelId::MlLinearActF32).is_ok()
		&& device.require_kernel(KernelId::MlLinearGatedActF32).is_ok();
	let has_grouped_act = rl_fusion
		&& device
			.require_kernel(KernelId::MlGroupedLinearMActF32)
			.is_ok();
	let has_residual_norm = device
		.require_kernel(KernelId::MlResidualLayerNormF32)
		.is_ok();
	let has_triple = device.require_kernel(KernelId::MlLinearTripleF32).is_ok();
	let has_embedding_pair = device.require_kernel(KernelId::MlEmbeddingPairF32).is_ok();
	let has_sum_axis_scale = device.require_kernel(KernelId::MlSumAxisScaleF32).is_ok();
	let has_router = rl_fusion
		&& device
			.require_kernel(KernelId::MlMoeRouterSelectF32)
			.is_ok();
	let has_concat = rl_fusion && device.require_kernel(KernelId::MlLinearConcatF32).is_ok();
	let has_small_sdpa = rl_fusion
		&& device
			.require_kernel(KernelId::MlSmallCausalSdpaF32)
			.is_ok();
	let has_backward_prologue = device
		.require_kernel(KernelId::MlLinearBackwardPrologueF32)
		.is_ok();
	let has_backward_norm = device
		.require_kernel(KernelId::MlLinearBackwardNormF32)
		.is_ok();
	let has_backward_norm_add = device
		.require_kernel(KernelId::MlLinearBackwardNormAddF32)
		.is_ok();
	let has_backward_bmm = device
		.require_kernel(KernelId::MlLinearBackwardBmmChainF32)
		.is_ok();
	let has_grouped_backward = device
		.require_kernel(KernelId::MlGroupedBackwardPrologueF32)
		.is_ok();
	let has_linear_residual_norm = device
		.require_kernel(KernelId::MlLinearResidualNormF32)
		.is_ok();
	let has_embedding_pair_norm = device
		.require_kernel(KernelId::MlEmbeddingPairNormF32)
		.is_ok();
	let has_backward_norm_sum = device
		.require_kernel(KernelId::MlLayerNormBackwardSumF32)
		.is_ok();
	let has_embedding_backward_sum = device
		.require_kernel(KernelId::MlEmbeddingBackwardSumF32)
		.is_ok();
	let has_conv_1d_bias_tail = [
		KernelId::MlConv1dBiasTransposeF32,
		KernelId::MlConv1dBiasTransposeReluF32,
		KernelId::MlConv1dBiasTransposeResidualReluF32,
		KernelId::MlConv1dBiasTransposeNormF32,
		KernelId::MlConv1dBiasTransposeNormReluF32,
		KernelId::MlConv1dBiasCopyF32,
	]
	.iter()
	.all(|kernel| device.require_kernel(*kernel).is_ok());
	let has_conv_2d_relu = device.require_kernel(KernelId::MlConv2dReluF32).is_ok();
	let has_max_pool_relu_backward = device
		.require_kernel(KernelId::MlMaxPool2dReluBackwardF32)
		.is_ok();
	if !has_residual
		&& !has_act
		&& !has_grouped_act
		&& !has_residual_norm
		&& !has_triple
		&& !has_embedding_pair
		&& !has_sum_axis_scale
		&& !has_router
		&& !has_concat
		&& !has_small_sdpa
		&& !has_backward_prologue
		&& !has_backward_norm
		&& !has_backward_norm_add
		&& !has_backward_bmm
		&& !has_grouped_backward
		&& !has_linear_residual_norm
		&& !has_embedding_pair_norm
		&& !has_backward_norm_sum
		&& !has_embedding_backward_sum
		&& !has_conv_1d_bias_tail
		&& !has_conv_2d_relu
		&& !has_max_pool_relu_backward
	{
		return nodes;
	}
	let mut result = Vec::with_capacity(nodes.len());
	let mut regions = 0;
	let mut index = 0;
	let mut consumed = std::collections::BTreeSet::new();
	// Fused nodes deferred to a later slot: the combine feeding a norm
	// backward can only run once its last addend exists, which may be after
	// the earliest absorbed add the window was anchored at.
	let mut deferred = std::collections::BTreeMap::<usize, ComputeNode>::new();
	while index < nodes.len() {
		if let Some(fused) = deferred.remove(&index) {
			result.push(fused);
		}
		if consumed.contains(&index) {
			index += 1;
			continue;
		}
		// Admit a replacement of the adjacent `count`-node window at `index`.
		let window_ok = |count: usize, fused: &[&ComputeNode]| {
			index + count <= nodes.len()
				&& slots_free(&consumed, index + 1..index + count)
				&& elided_writes_unobservable(refs, &nodes[index..index + count], fused)
		};
		if has_backward_bmm
			&& index + 1 < nodes.len()
			&& let Some((fused, distance)) = match_linear_backward_bmm_chain(
				&nodes[index..(index + MAX_BMM_CHAIN_LOOKAHEAD + 1).min(nodes.len())],
			) && slots_free(&consumed, std::iter::once(index + distance))
			&& elided_writes_unobservable(refs, [&nodes[index], &nodes[index + distance]], &[&fused])
		{
			regions += 1;
			result.push(fused);
			consumed.insert(index + distance);
			index += 1;
			continue;
		}
		if has_small_sdpa {
			if index + 7 <= nodes.len()
				&& let Some(fused) = match_small_causal_sdpa(&nodes[index..index + 7])
				&& window_ok(7, &[&fused])
			{
				regions += 1;
				result.push(fused);
				index += 7;
				continue;
			}
			if index + 3 <= nodes.len()
				&& let Some(fused) = match_single_head_sdpa(&nodes[index..index + 3])
				&& window_ok(3, &[&fused])
			{
				regions += 1;
				result.push(fused);
				index += 3;
				continue;
			}
		}
		if has_concat {
			let mut matched = None;
			for count in [3, 2] {
				if index + count < nodes.len()
					&& let Some(fused) =
						match_concat_linear(&nodes[index..index + count], &nodes[index + count])
					&& window_ok(count + 1, &[&fused])
				{
					matched = Some((count, fused));
					break;
				}
			}
			if let Some((count, fused)) = matched {
				regions += 1;
				result.push(fused);
				index += count + 1;
				continue;
			}
		}
		if has_router {
			if index + 3 < nodes.len()
				&& let Some(fused) =
					match_router_select(&nodes[index], &nodes[index + 2], &nodes[index + 3])
				&& router_bias_can_precede_softmax(&nodes[index], &nodes[index + 1], &nodes[index + 2])
				&& window_ok(4, &[&nodes[index + 1], &fused])
			{
				result.push(nodes[index + 1].clone());
				regions += 1;
				result.push(fused);
				index += 4;
				continue;
			}
			if index + 2 < nodes.len()
				&& let Some(fused) =
					match_router_select(&nodes[index], &nodes[index + 1], &nodes[index + 2])
				&& window_ok(3, &[&fused])
			{
				regions += 1;
				result.push(fused);
				index += 3;
				continue;
			}
		}
		if index + 2 < nodes.len()
			&& has_triple
			&& let Some(fused) = match_linear_triple(&nodes[index], &nodes[index + 1], &nodes[index + 2])
			&& window_ok(3, &[&fused])
		{
			regions += 1;
			result.push(fused);
			index += 3;
			continue;
		}
		if index + 3 < nodes.len()
			&& has_embedding_pair_norm
			&& let Some(fused) = match_embedding_pair_norm(
				&nodes[index],
				&nodes[index + 1],
				&nodes[index + 2],
				&nodes[index + 3],
			) && window_ok(4, &[&fused])
		{
			regions += 1;
			result.push(fused);
			index += 4;
			continue;
		}
		if index + 2 < nodes.len()
			&& has_embedding_pair
			&& let Some(fused) = match_embedding_pair(&nodes[index], &nodes[index + 1], &nodes[index + 2])
			&& window_ok(3, &[&fused])
		{
			regions += 1;
			result.push(fused);
			index += 3;
			continue;
		}
		if index + 3 < nodes.len()
			&& has_backward_norm_add
			&& let Some((fused, reduction)) = match_linear_backward_norm_add(
				&nodes[index],
				&nodes[index + 1],
				&nodes[index + 2],
				&nodes[index + 3],
			) && window_ok(4, &[&fused, &reduction])
		{
			regions += 1;
			result.push(fused);
			result.push(reduction);
			index += 4;
			continue;
		}
		if index + 2 < nodes.len()
			&& has_backward_norm
			&& let Some((fused, reduction)) =
				match_linear_backward_norm(&nodes[index], &nodes[index + 1], &nodes[index + 2])
			&& window_ok(3, &[&fused, &reduction])
		{
			regions += 1;
			result.push(fused);
			result.push(reduction);
			index += 3;
			continue;
		}
		if has_backward_norm_sum
			&& let Some((fused, absorbed, emit)) = match_layer_norm_backward_sum(&nodes[index..])
			&& slots_free(&consumed, absorbed.iter().map(|distance| index + distance))
			&& elided_writes_unobservable(
				refs,
				std::iter::once(0)
					.chain(absorbed.iter().copied())
					.map(|distance| &nodes[index + distance]),
				&[&fused],
			) {
			regions += 1;
			for distance in absorbed {
				consumed.insert(index + distance);
			}
			if emit == 0 {
				result.push(fused);
			} else {
				deferred.insert(index + emit, fused);
			}
			index += 1;
			continue;
		}
		if has_embedding_backward_sum
			&& let Some(emissions) = match_embedding_backward_sum(&nodes[index..])
			&& slots_free(
				&consumed,
				emissions.iter().map(|(distance, _)| index + distance),
			) && elided_writes_unobservable(
			refs,
			std::iter::once(0)
				.chain(emissions.iter().map(|(distance, _)| *distance))
				.map(|distance| &nodes[index + distance]),
			&emissions.iter().map(|(_, fused)| fused).collect::<Vec<_>>(),
		) {
			regions += emissions.len();
			for (distance, fused) in emissions {
				consumed.insert(index + distance);
				deferred.insert(index + distance, fused);
			}
			index += 1;
			continue;
		}
		if has_conv_1d_bias_tail
			&& index + 1 < nodes.len()
			&& let Some((fused, count)) =
				match_conv_1d_bias_tail(&nodes[index..], |buffer| storage_unobservable(refs, buffer))
			&& window_ok(count, &[&fused])
		{
			regions += 1;
			result.push(fused);
			index += count;
			continue;
		}
		if has_conv_2d_relu
			&& index + 1 < nodes.len()
			&& let Some(fused) = match_conv_2d_relu(&nodes[index..])
			&& window_ok(2, &[&fused])
		{
			regions += 1;
			result.push(fused);
			index += 2;
			continue;
		}
		if has_max_pool_relu_backward
			&& index + 1 < nodes.len()
			&& let Some(fused) = match_max_pool_relu_backward(&nodes[index..])
			&& window_ok(2, &[&fused])
		{
			regions += 1;
			result.push(fused);
			index += 2;
			continue;
		}
		if index + 2 < nodes.len()
			&& has_linear_residual_norm
			&& let Some(fused) =
				match_linear_residual_norm(&nodes[index], &nodes[index + 1], &nodes[index + 2])
			&& window_ok(3, &[&fused])
		{
			regions += 1;
			result.push(fused);
			index += 3;
			continue;
		}
		if index + 1 < nodes.len()
			&& has_grouped_act
			&& let Some(fused) = match_grouped_linear_act(&nodes[index], &nodes[index + 1])
			&& window_ok(2, &[&fused])
		{
			regions += 1;
			result.push(fused);
			index += 2;
			continue;
		}
		if index + 1 < nodes.len() {
			type PairMatcher = fn(&ComputeNode, &ComputeNode) -> Option<ComputeNode>;
			let pairs: [(bool, PairMatcher); 6] = [
				(has_residual, match_linear_residual),
				(has_act, match_linear_act),
				(has_residual_norm, match_add_layer_norm),
				(has_sum_axis_scale, match_sum_axis_scale),
				(has_backward_prologue, match_linear_backward_prologue),
				(has_grouped_backward, match_grouped_backward_prologue),
			];
			let fused = pairs
				.iter()
				.filter(|(enabled, _)| *enabled)
				.find_map(|(_, matcher)| {
					matcher(&nodes[index], &nodes[index + 1]).filter(|fused| window_ok(2, &[fused]))
				});
			if let Some(fused) = fused {
				regions += 1;
				result.push(fused);
				index += 2;
				continue;
			}
		}
		result.push(nodes[index].clone());
		index += 1;
	}
	// Each replacement turns k source nodes into one, so the absorbed total
	// is the length reduction plus one per region.
	device
		.fusion_census()
		.structural(regions, nodes.len() - result.len() + regions);
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
		jit: None,
		fusion_signature: None,
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
		jit: None,
		fusion_signature: None,
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
		jit: None,
		fusion_signature: None,
	})
}

/// Decode the `embedding → embedding → add → layer_norm` token/position
/// head plus its normalization into one fused node. The row-workgroup
/// kernel gathers both tables, republishes both gathered outputs and the
/// sum, then runs the norm's statistics and affine passes — every write
/// the unfused four-node run performs stays materialized. The window is
/// tried before the pair matcher claims the add for `embedding_pair`.
fn match_embedding_pair_norm(
	first: &ComputeNode,
	second: &ComputeNode,
	add: &ComputeNode,
	norm: &ComputeNode,
) -> Option<ComputeNode> {
	let (first_output, first_u8) = match_embedding(first)?;
	let (second_output, second_u8) = match_embedding(second)?;
	if add.kernel != KernelId::MatrixAddF32
		|| norm.kernel != KernelId::MlLayerNormF32
		|| add.indirect.is_some()
		|| norm.indirect.is_some()
		|| add.buffers.len() != 3
		|| norm.buffers.len() != 6
		|| add.push_constants.len() != 1
		|| norm.push_constants.len() != 3
		|| first.push_constants[0] != second.push_constants[0]
		|| first.push_constants[2] != second.push_constants[2]
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
	let consumes = |a: &BufferUse, b: &BufferUse| {
		(a.buffer.same_as(first_output) && b.buffer.same_as(second_output))
			|| (a.buffer.same_as(second_output) && b.buffer.same_as(first_output))
	};
	if !consumes(&add.buffers[0], &add.buffers[1]) {
		return None;
	}
	let sum = &add.buffers[2].buffer;
	if !norm.buffers[0].buffer.same_as(sum) {
		return None;
	}
	let (PushConstant::U32(index_count), PushConstant::U32(embedding_dim)) =
		(first.push_constants[0], first.push_constants[2])
	else {
		return None;
	};
	let (PushConstant::U32(rows), PushConstant::U32(columns), PushConstant::F32(epsilon)) = (
		norm.push_constants[0],
		norm.push_constants[1],
		norm.push_constants[2],
	) else {
		return None;
	};
	if rows != index_count
		|| columns != embedding_dim
		|| embedding_dim > MAX_RESIDUAL_NORM_COLUMNS
		|| add.push_constants[0] != PushConstant::U32(index_count.checked_mul(embedding_dim)?)
	{
		return None;
	}
	let reads = [
		&first.buffers[0].buffer,
		&first.buffers[1].buffer,
		&second.buffers[0].buffer,
		&second.buffers[1].buffer,
		&norm.buffers[1].buffer,
		&norm.buffers[2].buffer,
	];
	let writes = [
		first_output,
		second_output,
		sum,
		&norm.buffers[3].buffer,
		&norm.buffers[4].buffer,
		&norm.buffers[5].buffer,
	];
	if writes
		.iter()
		.any(|write| reads.iter().any(|read| read.same_as(write)))
		|| (0..writes.len())
			.any(|left| (left + 1..writes.len()).any(|right| writes[left].same_as(writes[right])))
	{
		return None;
	}
	let kernel = KernelId::MlEmbeddingPairNormF32;
	let semantic_ops = merged_semantic_ops(&[
		&first.semantic_ops,
		&second.semantic_ops,
		&add.semantic_ops,
		&norm.semantic_ops,
	]);
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
			norm.buffers[1].clone(),
			norm.buffers[2].clone(),
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
			norm.buffers[3].clone(),
			norm.buffers[4].clone(),
			norm.buffers[5].clone(),
		],
		push_constants: vec![
			first.push_constants[0],
			first.push_constants[1],
			second.push_constants[1],
			first.push_constants[2],
			u8_flag(first_u8),
			u8_flag(second_u8),
			PushConstant::F32(epsilon),
		],
		workgroups: norm.workgroups,
		indirect: None,
		semantic_ops,
		op_contract_hash,
		jit: None,
		fusion_signature: None,
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
		jit: None,
		fusion_signature: None,
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

/// The widest row the `linear_residual_norm` in-tile norm admits: one
/// column tile must cover the whole output row. Wider outputs stay on the
/// unfused `linear_residual → layer_norm` route.
const MAX_RESIDUAL_NORM_COLUMNS: u32 = 64;

/// Decode the `linear → add → layer_norm` triple that forms the transformer
/// sublayer epilogue, or `None` when any node lies outside the fused
/// kernel's buffer and scalar vocabulary. The fused kernel republishes all
/// five writes the unfused run performs — the pre-residual linear output,
/// the residual sum, the norm output, the normalized row and the inverse
/// stddev — so downstream consumers and autograd saves observe identical
/// storage.
fn match_linear_residual_norm(
	linear: &ComputeNode,
	add: &ComputeNode,
	norm: &ComputeNode,
) -> Option<ComputeNode> {
	let (linear_output, batch, output_features) = match_linear_producer(linear)?;
	let PushConstant::U32(input_features) = linear.push_constants[1] else {
		return None;
	};
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
	let residual = match (
		add.buffers[0].buffer.same_as(linear_output),
		add.buffers[1].buffer.same_as(linear_output),
	) {
		(true, false) => &add.buffers[1].buffer,
		(false, true) => &add.buffers[0].buffer,
		_ => return None,
	};
	let sum = &add.buffers[2].buffer;
	if !norm.buffers[0].buffer.same_as(sum) {
		return None;
	}
	let (PushConstant::U32(rows), PushConstant::U32(columns), PushConstant::F32(epsilon)) = (
		norm.push_constants[0],
		norm.push_constants[1],
		norm.push_constants[2],
	) else {
		return None;
	};
	if rows != batch
		|| columns != output_features
		|| columns > MAX_RESIDUAL_NORM_COLUMNS
		|| add.push_constants[0] != PushConstant::U32(batch.checked_mul(output_features)?)
	{
		return None;
	}
	let reads = [
		&linear.buffers[0].buffer,
		&linear.buffers[1].buffer,
		&linear.buffers[2].buffer,
		residual,
		&norm.buffers[1].buffer,
		&norm.buffers[2].buffer,
	];
	let writes = [
		linear_output,
		sum,
		&norm.buffers[3].buffer,
		&norm.buffers[4].buffer,
		&norm.buffers[5].buffer,
	];
	if writes
		.iter()
		.any(|write| reads.iter().any(|read| read.same_as(write)))
		|| (0..writes.len())
			.any(|left| (left + 1..writes.len()).any(|right| writes[left].same_as(writes[right])))
	{
		return None;
	}
	let kernel = KernelId::MlLinearResidualNormF32;
	let semantic_ops =
		merged_semantic_ops(&[&linear.semantic_ops, &add.semantic_ops, &norm.semantic_ops]);
	let (operation, op_contract_hash) = fused_identity(&semantic_ops, linear, kernel);
	Some(ComputeNode {
		operation,
		kernel,
		buffers: vec![
			linear.buffers[0].clone(),
			linear.buffers[1].clone(),
			linear.buffers[2].clone(),
			BufferUse {
				buffer: residual.clone(),
				access: BufferAccess::Read,
			},
			norm.buffers[1].clone(),
			norm.buffers[2].clone(),
			BufferUse {
				buffer: linear_output.clone(),
				access: BufferAccess::Write,
			},
			BufferUse {
				buffer: sum.clone(),
				access: BufferAccess::Write,
			},
			norm.buffers[3].clone(),
			norm.buffers[4].clone(),
			norm.buffers[5].clone(),
		],
		push_constants: vec![
			PushConstant::U32(batch),
			PushConstant::U32(input_features),
			PushConstant::U32(output_features),
			PushConstant::F32(epsilon),
		],
		workgroups: [batch.div_ceil(64), output_features.div_ceil(64), 1],
		indirect: None,
		semantic_ops,
		op_contract_hash,
		jit: None,
		fusion_signature: None,
	})
}

/// Decode the `linear → activation` pair, or `None` when either node lies
/// outside the fused kernel's buffer and scalar vocabulary. The activation
/// must consume the linear output elementwise over the full tile; `elu` and
/// `leaky_relu` carry their `alpha` push through.
fn match_linear_act(linear: &ComputeNode, act: &ComputeNode) -> Option<ComputeNode> {
	let (linear_output, batch, output_features) = match_linear_producer(linear)?;
	let gated = act.kernel == KernelId::MlSiluMulF32;
	if gated && std::env::var_os("OA_DISABLE_RL_FUSION").is_some() {
		return None;
	}
	let kind = if gated { 9 } else { act_kind(act.kernel)? };
	if act.indirect.is_some()
		|| act.buffers.len() != 2
		|| act.buffers[0].access != BufferAccess::Read
		|| act.buffers[1].access != BufferAccess::Write
		|| !act.buffers[0].buffer.same_as(linear_output)
	{
		return None;
	}
	let (elements, alpha) = match act.push_constants.as_slice() {
		[PushConstant::U32(elements)] if !gated => (*elements, 0.0),
		[PushConstant::U32(elements), PushConstant::F32(alpha)] if !gated => (*elements, *alpha),
		[PushConstant::U32(elements), PushConstant::U32(width)]
			if gated && width.checked_mul(2)? == output_features =>
		{
			(*elements, 0.0)
		}
		_ => return None,
	};
	if elements
		!= batch.checked_mul(if gated {
			output_features / 2
		} else {
			output_features
		})? {
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
	let kernel = if gated {
		KernelId::MlLinearGatedActF32
	} else {
		KernelId::MlLinearActF32
	};
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
		if gated {
			[batch.div_ceil(64), (output_features / 2).div_ceil(64), 1]
		} else {
			linear.workgroups
		},
	))
}

/// Map an elementwise activation adjoint kernel to its
/// `linear_backward_prologue` kind — the first nine entries of the
/// `prologue_kind` vocabulary, matching `linear_act.slang`.
fn backward_act_kind(kernel: KernelId) -> Option<u32> {
	match kernel {
		KernelId::MlReluBackwardF32 => Some(0),
		KernelId::MlGeluBackwardF32 => Some(1),
		KernelId::MlSiluBackwardF32 => Some(2),
		KernelId::MlTanhBackwardF32 => Some(3),
		KernelId::MlSigmoidBackwardF32 => Some(4),
		KernelId::MlMishBackwardF32 => Some(5),
		KernelId::MlEluBackwardF32 => Some(6),
		KernelId::MlLeakyReluBackwardF32 => Some(7),
		KernelId::MlSoftplusBackwardF32 => Some(8),
		_ => None,
	}
}

/// The gated `silu_mul` adjoint's `prologue_kind`.
const PROLOGUE_SILU_MUL: u32 = 9;
/// Pointwise combine prologues folded from a producing Binary node.
const PROLOGUE_ADD: u32 = 10;
const PROLOGUE_SUB: u32 = 11;
const PROLOGUE_MUL: u32 = 12;
/// Broadcast `mul` where one operand is a one-element splat.
const PROLOGUE_SCALAR_MUL: u32 = 13;

/// Decode a pointwise combine producer — a Binary `add`/`sub`/`mul` or a
/// scalar broadcast `mul` — into `(kind, upstream_slot, saved_slot, alpha)`
/// for the fused kernel. The slots index `producer.buffers`: `upstream` is
/// the operand the prologue streams as its first operand (the subtraction
/// left side) and `saved` the second; a scalar broadcast operand binds as
/// `saved` and splats its single element.
fn backward_pointwise_prologue(
	producer: &ComputeNode,
	batch: u32,
	output_features: u32,
) -> Option<(u32, usize, usize, f32)> {
	let elements = batch.checked_mul(output_features)?;
	match producer.kernel {
		KernelId::MatrixAddF32 | KernelId::MatrixSubF32 | KernelId::MatrixMulF32 => {
			let [PushConstant::U32(count)] = producer.push_constants.as_slice() else {
				return None;
			};
			if *count != elements {
				return None;
			}
			let kind = match producer.kernel {
				KernelId::MatrixAddF32 => PROLOGUE_ADD,
				KernelId::MatrixSubF32 => PROLOGUE_SUB,
				_ => PROLOGUE_MUL,
			};
			Some((kind, 0, 1, 0.0))
		}
		KernelId::MatrixMulBroadcastF32 => {
			// Broadcast push layout: element count, rank, eight output
			// dimensions, eight left strides, eight right strides. Fuse
			// only a scalar splat: the streamed operand covers the whole
			// `[batch, output_features]` tile in row-major strides while
			// the other operand broadcasts from one element.
			let word = |offset: usize| match producer.push_constants.get(offset) {
				Some(PushConstant::U32(value)) => Some(*value),
				_ => None,
			};
			if producer.push_constants.len() != 26
				|| word(0)? != elements
				|| word(1)? != 2
				|| word(2)? != batch
				|| word(3)? != output_features
			{
				return None;
			}
			let strides = |offset: usize| -> Option<[u32; 8]> {
				let mut result = [0u32; 8];
				for (lane, value) in result.iter_mut().enumerate() {
					*value = word(offset + lane)?;
				}
				Some(result)
			};
			let row_major = [output_features, 1, 0, 0, 0, 0, 0, 0];
			let splat_strides = [0u32; 8];
			let (matrix, splat) = if strides(10)? == row_major && strides(18)? == splat_strides {
				(0usize, 1usize)
			} else if strides(18)? == row_major && strides(10)? == splat_strides {
				(1, 0)
			} else {
				return None;
			};
			if producer.buffers[splat].buffer.byte_len() != 4 {
				return None;
			}
			Some((PROLOGUE_SCALAR_MUL, matrix, splat, 0.0))
		}
		_ => None,
	}
}

/// Decode the `producer → linear_backward` pair, or `None` when either node
/// lies outside the fused kernel's buffer and scalar vocabulary. The
/// producer may be an activation adjoint, a pointwise `add`/`sub`/`mul`, or
/// a scalar broadcast `mul`; its output must feed the data adjoint's
/// upstream operand over the full `[batch, output_features]` tile, and both
/// writes must stay disjoint from every other bound operand. `elu` and
/// `leaky_relu` carry their `alpha` push through.
fn match_linear_backward_prologue(
	producer: &ComputeNode,
	backward: &ComputeNode,
) -> Option<ComputeNode> {
	if backward.kernel != KernelId::MlLinearBackwardF32
		|| producer.indirect.is_some()
		|| backward.indirect.is_some()
		|| producer.buffers.len() != 3
		|| backward.buffers.len() != 3
		|| backward.push_constants.len() != 3
		|| producer
			.buffers
			.iter()
			.take(2)
			.any(|usage| usage.access != BufferAccess::Read)
		|| producer.buffers[2].access != BufferAccess::Write
		|| backward
			.buffers
			.iter()
			.take(2)
			.any(|usage| usage.access != BufferAccess::Read)
		|| backward.buffers[2].access != BufferAccess::Write
		|| !backward.buffers[1]
			.buffer
			.same_as(&producer.buffers[2].buffer)
	{
		return None;
	}
	let (PushConstant::U32(batch), PushConstant::U32(output_features)) =
		(backward.push_constants[0], backward.push_constants[2])
	else {
		return None;
	};
	// Activation adjoint and silu_mul producers lay out `[saved, upstream]`;
	// the pointwise combines are `[left, right]` and map left to `upstream`
	// so `sub` keeps operand order.
	let (kind, upstream, saved, alpha) = if let Some(act_kind) = backward_act_kind(producer.kernel) {
		// Elementwise adjoints cover the upstream tile directly; the gated
		// silu_mul adjoint covers the packed [gate|up] projection whose
		// width is twice the silu_mul output width.
		match producer.push_constants.as_slice() {
			[PushConstant::U32(elements)] if *elements == batch.checked_mul(output_features)? => {
				(act_kind, 1usize, 0usize, 0.0)
			}
			[PushConstant::U32(elements), PushConstant::F32(alpha)]
				if *elements == batch.checked_mul(output_features)? =>
			{
				(act_kind, 1, 0, *alpha)
			}
			_ => return None,
		}
	} else if producer.kernel == KernelId::MlSiluMulBackwardF32 {
		if std::env::var_os("OA_DISABLE_RL_FUSION").is_some() {
			return None;
		}
		let [
			PushConstant::U32(output_count),
			PushConstant::U32(intermediate),
		] = producer.push_constants.as_slice()
		else {
			return None;
		};
		if intermediate.checked_mul(2)? != output_features
			|| *output_count != batch.checked_mul(*intermediate)?
		{
			return None;
		}
		(PROLOGUE_SILU_MUL, 1, 0, 0.0)
	} else {
		backward_pointwise_prologue(producer, batch, output_features)?
	};
	let dy = &producer.buffers[2].buffer;
	let dx = &backward.buffers[2].buffer;
	let collides = |buffer: &Buffer| {
		[
			&backward.buffers[0].buffer,
			&producer.buffers[upstream].buffer,
			&producer.buffers[saved].buffer,
		]
		.iter()
		.any(|operand| operand.same_as(buffer))
	};
	if collides(dy) || collides(dx) || dy.same_as(dx) {
		return None;
	}
	let kernel = KernelId::MlLinearBackwardPrologueF32;
	let mut push_constants = backward.push_constants.clone();
	push_constants.push(PushConstant::U32(kind));
	push_constants.push(PushConstant::F32(alpha));
	Some(fused_node(
		producer,
		backward,
		kernel,
		vec![
			backward.buffers[0].clone(),
			producer.buffers[upstream].clone(),
			producer.buffers[saved].clone(),
			BufferUse {
				buffer: dy.clone(),
				access: BufferAccess::Write,
			},
			backward.buffers[2].clone(),
		],
		push_constants,
		backward.workgroups,
	))
}

/// Decode the `producer → grouped data adjoint` pair, or `None` when either
/// node lies outside the fused kernel's buffer and scalar vocabulary. Both
/// grouped data-adjoint kernels share one donor body, so the same fused
/// kernel serves `MlGroupedLinearMDataBackwardF32` and
/// `MlGroupedGemmMDataBackwardF32`. The producer decode is the dense
/// prologue vocabulary with `rows` standing in for `batch`; the fused node
/// materializes `dY` so the grouped parameter adjoint keeps its unfused
/// operand.
fn match_grouped_backward_prologue(
	producer: &ComputeNode,
	backward: &ComputeNode,
) -> Option<ComputeNode> {
	if !matches!(
		backward.kernel,
		KernelId::MlGroupedLinearMDataBackwardF32 | KernelId::MlGroupedGemmMDataBackwardF32
	) || producer.indirect.is_some()
		|| backward.indirect.is_some()
		|| producer.buffers.len() != 3
		|| backward.buffers.len() != 4
		|| backward.push_constants.len() != 4
		|| producer
			.buffers
			.iter()
			.take(2)
			.any(|usage| usage.access != BufferAccess::Read)
		|| producer.buffers[2].access != BufferAccess::Write
		|| backward
			.buffers
			.iter()
			.take(3)
			.any(|usage| usage.access != BufferAccess::Read)
		|| backward.buffers[3].access != BufferAccess::Write
		|| !backward.buffers[0]
			.buffer
			.same_as(&producer.buffers[2].buffer)
	{
		return None;
	}
	let (
		PushConstant::U32(rows),
		PushConstant::U32(output_features),
		PushConstant::U32(input_features),
		PushConstant::U32(experts),
	) = (
		backward.push_constants[0],
		backward.push_constants[1],
		backward.push_constants[2],
		backward.push_constants[3],
	)
	else {
		return None;
	};
	let (kind, upstream, saved, alpha) = if let Some(act_kind) = backward_act_kind(producer.kernel) {
		match producer.push_constants.as_slice() {
			[PushConstant::U32(elements)] if *elements == rows.checked_mul(output_features)? => {
				(act_kind, 1usize, 0usize, 0.0)
			}
			[PushConstant::U32(elements), PushConstant::F32(alpha)]
				if *elements == rows.checked_mul(output_features)? =>
			{
				(act_kind, 1, 0, *alpha)
			}
			_ => return None,
		}
	} else if producer.kernel == KernelId::MlSiluMulBackwardF32 {
		if std::env::var_os("OA_DISABLE_RL_FUSION").is_some() {
			return None;
		}
		let [
			PushConstant::U32(output_count),
			PushConstant::U32(intermediate),
		] = producer.push_constants.as_slice()
		else {
			return None;
		};
		if intermediate.checked_mul(2)? != output_features
			|| *output_count != rows.checked_mul(*intermediate)?
		{
			return None;
		}
		(PROLOGUE_SILU_MUL, 1, 0, 0.0)
	} else {
		backward_pointwise_prologue(producer, rows, output_features)?
	};
	let dy = &producer.buffers[2].buffer;
	let dx = &backward.buffers[3].buffer;
	let collides = |buffer: &Buffer| {
		[
			&producer.buffers[upstream].buffer,
			&producer.buffers[saved].buffer,
			&backward.buffers[1].buffer,
			&backward.buffers[2].buffer,
		]
		.iter()
		.any(|operand| operand.same_as(buffer))
	};
	if collides(dy) || collides(dx) || dy.same_as(dx) {
		return None;
	}
	Some(fused_node(
		producer,
		backward,
		KernelId::MlGroupedBackwardPrologueF32,
		vec![
			producer.buffers[upstream].clone(),
			producer.buffers[saved].clone(),
			BufferUse {
				buffer: dy.clone(),
				access: BufferAccess::Write,
			},
			backward.buffers[1].clone(),
			backward.buffers[2].clone(),
			backward.buffers[3].clone(),
		],
		vec![
			PushConstant::U32(rows),
			PushConstant::U32(output_features),
			PushConstant::U32(input_features),
			PushConstant::U32(experts),
			PushConstant::U32(kind),
			PushConstant::F32(alpha),
		],
		backward.workgroups,
	))
}

/// Decode the `layer_norm_backward → sum_axis_pair → linear_backward`
/// window emitted by autograd, or `None` outside the fused kernel's buffer
/// and scalar vocabulary. The first node's `input_gradient` write must be
/// the data adjoint's upstream operand over the full
/// `[batch, output_features]` tile and the sibling reduction must consume
/// the same `weight_contribution` and upstream buffers — the replacement
/// keeps the reduction dispatch verbatim behind the fused node so `dγ`/`dβ`
/// keep their unfused operands. All three fused writes must stay disjoint
/// from every bound read.
fn match_linear_backward_norm(
	norm: &ComputeNode,
	reduction: &ComputeNode,
	backward: &ComputeNode,
) -> Option<(ComputeNode, ComputeNode)> {
	if norm.kernel != KernelId::MlLayerNormBackwardF32
		|| reduction.kernel != KernelId::MlSumAxisPairF32
		|| backward.kernel != KernelId::MlLinearBackwardF32
		|| norm.indirect.is_some()
		|| reduction.indirect.is_some()
		|| backward.indirect.is_some()
		|| norm.buffers.len() != 6
		|| reduction.buffers.len() != 4
		|| backward.buffers.len() != 3
		|| norm.push_constants.len() != 2
		|| reduction.push_constants.len() != 3
		|| backward.push_constants.len() != 3
		|| norm
			.buffers
			.iter()
			.take(4)
			.any(|usage| usage.access != BufferAccess::Read)
		|| norm.buffers[4].access != BufferAccess::Write
		|| norm.buffers[5].access != BufferAccess::Write
		|| reduction
			.buffers
			.iter()
			.take(2)
			.any(|usage| usage.access != BufferAccess::Read)
		|| reduction
			.buffers
			.iter()
			.skip(2)
			.any(|usage| usage.access != BufferAccess::Write)
		|| backward
			.buffers
			.iter()
			.take(2)
			.any(|usage| usage.access != BufferAccess::Read)
		|| backward.buffers[2].access != BufferAccess::Write
		|| !reduction.buffers[0].buffer.same_as(&norm.buffers[5].buffer)
		|| !reduction.buffers[1].buffer.same_as(&norm.buffers[3].buffer)
		|| !backward.buffers[1].buffer.same_as(&norm.buffers[4].buffer)
	{
		return None;
	}
	let (
		PushConstant::U32(batch),
		PushConstant::U32(_input_features),
		PushConstant::U32(output_features),
	) = (
		backward.push_constants[0],
		backward.push_constants[1],
		backward.push_constants[2],
	)
	else {
		return None;
	};
	let (PushConstant::U32(rows), PushConstant::U32(columns)) =
		(norm.push_constants[0], norm.push_constants[1])
	else {
		return None;
	};
	if rows != batch || columns != output_features {
		return None;
	}
	let reads = [
		&backward.buffers[0].buffer,
		&norm.buffers[0].buffer,
		&norm.buffers[1].buffer,
		&norm.buffers[2].buffer,
		&norm.buffers[3].buffer,
	];
	let writes = [
		&norm.buffers[4].buffer,
		&norm.buffers[5].buffer,
		&backward.buffers[2].buffer,
	];
	if writes
		.iter()
		.any(|write| reads.iter().any(|read| read.same_as(write)))
		|| writes[0].same_as(writes[1])
		|| writes[0].same_as(writes[2])
		|| writes[1].same_as(writes[2])
	{
		return None;
	}
	let fused = fused_node(
		norm,
		backward,
		KernelId::MlLinearBackwardNormF32,
		vec![
			backward.buffers[0].clone(),
			norm.buffers[0].clone(),
			norm.buffers[1].clone(),
			norm.buffers[2].clone(),
			norm.buffers[3].clone(),
			norm.buffers[4].clone(),
			norm.buffers[5].clone(),
			backward.buffers[2].clone(),
		],
		backward.push_constants.clone(),
		backward.workgroups,
	);
	Some((fused, reduction.clone()))
}

/// Decode the `layer_norm_backward → sum_axis_pair → matrix.add →
/// linear_backward` window emitted when the norm's input has a residual
/// consumer: the norm's `input_gradient` is one addend of the gradient
/// accumulation feeding the data adjoint. The replacement keeps the
/// reduction dispatch verbatim and republishes `dx_norm`,
/// `weight_contribution` and the combined `dY` — every downstream
/// consumer sees the buffers the unfused four-node run wrote. The fused
/// node runs at the norm's position, so it publishes the combined `dY`
/// and `dX` earlier than the dispatches they replace; the two
/// intervening nodes may only read the fused operands and must never
/// write them or touch the moved writes.
fn match_linear_backward_norm_add(
	norm: &ComputeNode,
	reduction: &ComputeNode,
	combine: &ComputeNode,
	backward: &ComputeNode,
) -> Option<(ComputeNode, ComputeNode)> {
	if norm.kernel != KernelId::MlLayerNormBackwardF32
		|| reduction.kernel != KernelId::MlSumAxisPairF32
		|| combine.kernel != KernelId::MatrixAddF32
		|| backward.kernel != KernelId::MlLinearBackwardF32
		|| norm.indirect.is_some()
		|| reduction.indirect.is_some()
		|| combine.indirect.is_some()
		|| backward.indirect.is_some()
		|| norm.buffers.len() != 6
		|| reduction.buffers.len() != 4
		|| combine.buffers.len() != 3
		|| backward.buffers.len() != 3
		|| norm.push_constants.len() != 2
		|| reduction.push_constants.len() != 3
		|| combine.push_constants.len() != 1
		|| backward.push_constants.len() != 3
		|| norm
			.buffers
			.iter()
			.take(4)
			.any(|usage| usage.access != BufferAccess::Read)
		|| norm.buffers[4].access != BufferAccess::Write
		|| norm.buffers[5].access != BufferAccess::Write
		|| reduction
			.buffers
			.iter()
			.take(2)
			.any(|usage| usage.access != BufferAccess::Read)
		|| reduction
			.buffers
			.iter()
			.skip(2)
			.any(|usage| usage.access != BufferAccess::Write)
		|| combine
			.buffers
			.iter()
			.take(2)
			.any(|usage| usage.access != BufferAccess::Read)
		|| combine.buffers[2].access != BufferAccess::Write
		|| backward
			.buffers
			.iter()
			.take(2)
			.any(|usage| usage.access != BufferAccess::Read)
		|| backward.buffers[2].access != BufferAccess::Write
		|| !reduction.buffers[0].buffer.same_as(&norm.buffers[5].buffer)
		|| !reduction.buffers[1].buffer.same_as(&norm.buffers[3].buffer)
		|| !backward.buffers[1]
			.buffer
			.same_as(&combine.buffers[2].buffer)
	{
		return None;
	}
	// One addend must be the norm's materialized input gradient.
	let addend = if combine.buffers[0].buffer.same_as(&norm.buffers[4].buffer) {
		&combine.buffers[1]
	} else if combine.buffers[1].buffer.same_as(&norm.buffers[4].buffer) {
		&combine.buffers[0]
	} else {
		return None;
	};
	let (
		PushConstant::U32(batch),
		PushConstant::U32(_input_features),
		PushConstant::U32(output_features),
	) = (
		backward.push_constants[0],
		backward.push_constants[1],
		backward.push_constants[2],
	)
	else {
		return None;
	};
	let (PushConstant::U32(rows), PushConstant::U32(columns)) =
		(norm.push_constants[0], norm.push_constants[1])
	else {
		return None;
	};
	let PushConstant::U32(combined) = combine.push_constants[0] else {
		return None;
	};
	if rows != batch
		|| columns != output_features
		|| u64::from(combined) != u64::from(batch) * u64::from(output_features)
	{
		return None;
	}
	let dy = &combine.buffers[2].buffer;
	let dx = &backward.buffers[2].buffer;
	let reads = [
		&backward.buffers[0].buffer,
		&norm.buffers[0].buffer,
		&norm.buffers[1].buffer,
		&norm.buffers[2].buffer,
		&norm.buffers[3].buffer,
		&addend.buffer,
	];
	let writes = [&norm.buffers[4].buffer, &norm.buffers[5].buffer, dy, dx];
	if writes
		.iter()
		.any(|write| reads.iter().any(|read| read.same_as(write)))
		|| writes[0].same_as(writes[1])
		|| writes[0].same_as(writes[2])
		|| writes[0].same_as(writes[3])
		|| writes[1].same_as(writes[2])
		|| writes[1].same_as(writes[3])
		|| writes[2].same_as(writes[3])
	{
		return None;
	}
	// The reduction may only read the republished operands; its dγ/dβ
	// writes must stay clear of every fused operand. The combine reads the
	// norm gradient and the addend and writes the combined gradient, so
	// the only early-publish hazard left is a buffer touching `dX`.
	if reduction.buffers.iter().skip(2).any(|usage| {
		reads.iter().any(|read| read.same_as(&usage.buffer))
			|| writes.iter().any(|write| write.same_as(&usage.buffer))
	}) || combine.buffers.iter().any(|usage| usage.buffer.same_as(dx))
	{
		return None;
	}
	let kernel = KernelId::MlLinearBackwardNormAddF32;
	let semantic_ops = merged_semantic_ops(&[
		&norm.semantic_ops,
		&combine.semantic_ops,
		&backward.semantic_ops,
	]);
	let (operation, op_contract_hash) = fused_identity(&semantic_ops, norm, kernel);
	Some((
		ComputeNode {
			operation,
			kernel,
			buffers: vec![
				backward.buffers[0].clone(),
				norm.buffers[0].clone(),
				norm.buffers[1].clone(),
				norm.buffers[2].clone(),
				norm.buffers[3].clone(),
				BufferUse {
					buffer: addend.buffer.clone(),
					access: BufferAccess::Read,
				},
				norm.buffers[4].clone(),
				norm.buffers[5].clone(),
				combine.buffers[2].clone(),
				backward.buffers[2].clone(),
			],
			push_constants: backward.push_constants.clone(),
			workgroups: backward.workgroups,
			indirect: None,
			semantic_ops,
			op_contract_hash,
			jit: None,
			fusion_signature: None,
		},
		reduction.clone(),
	))
}

/// The farthest the accumulation adds feeding a `layer_norm_backward` may
/// sit from the consumer: parameter adjoints interleave the QKV input
/// gradient accumulation, so the outer combine trails by a few dispatches.
const MAX_NORM_BACKWARD_SUM_LOOKAHEAD: usize = 8;

/// Decode a `matrix.add` chain feeding a `layer_norm_backward` — the
/// multi-branch gradient accumulation that sums the QKV data adjoints back
/// into the shared head input. `nodes[0]` anchors the earliest absorbed
/// add: when a later `matrix.add` combines its output with a third
/// gradient into the norm's `dY`, the fused kernel sums all three
/// addends in the unfused tree's left-associative order; when `nodes[0]`
/// is itself the combine, it sums two. The fused dispatch runs at the
/// combine's slot (returned as the third tuple element) because the third
/// addend's producer may sit between the two adds, and republishes the
/// combined `dY` for the surviving `sum_axis_pair`. Returns the fused
/// node, the distances of the consumed nodes, and the distance at which
/// the fused node must be emitted.
fn match_layer_norm_backward_sum(
	nodes: &[ComputeNode],
) -> Option<(ComputeNode, Vec<usize>, usize)> {
	let first = &nodes[0];
	if first.kernel != KernelId::MatrixAddF32
		|| first.indirect.is_some()
		|| first.buffers.len() != 3
		|| first.push_constants.len() != 1
		|| first
			.buffers
			.iter()
			.take(2)
			.any(|usage| usage.access != BufferAccess::Read)
		|| first.buffers[2].access != BufferAccess::Write
	{
		return None;
	}
	let first_output = &first.buffers[2].buffer;
	let norm_index = (1..=MAX_NORM_BACKWARD_SUM_LOOKAHEAD).find(|&distance| {
		distance < nodes.len() && nodes[distance].kernel == KernelId::MlLayerNormBackwardF32
	})?;
	let norm = &nodes[norm_index];
	if norm.indirect.is_some()
		|| norm.buffers.len() != 6
		|| norm.push_constants.len() != 2
		|| norm
			.buffers
			.iter()
			.take(4)
			.any(|usage| usage.access != BufferAccess::Read)
		|| norm
			.buffers
			.iter()
			.skip(4)
			.any(|usage| usage.access != BufferAccess::Write)
	{
		return None;
	}
	let dy = &norm.buffers[3].buffer;
	// Shape the addend list: `nodes[0]` is either the combine feeding the
	// norm directly (two addends) or the inner add of a two-level tree.
	let (outer_index, addend_a, addend_b, addend_c, addend_count) = if dy.same_as(first_output) {
		(
			0,
			&first.buffers[0].buffer,
			&first.buffers[1].buffer,
			&first.buffers[1].buffer,
			2u32,
		)
	} else {
		let outer_index = (1..norm_index).find(|&distance| {
			let node = &nodes[distance];
			node.kernel == KernelId::MatrixAddF32
				&& node.buffers.len() == 3
				&& node.buffers[2].access == BufferAccess::Write
				&& node.buffers[2].buffer.same_as(dy)
		})?;
		let outer = &nodes[outer_index];
		if outer.indirect.is_some()
			|| outer.push_constants.len() != 1
			|| outer
				.buffers
				.iter()
				.take(2)
				.any(|usage| usage.access != BufferAccess::Read)
			|| !outer
				.buffers
				.iter()
				.take(2)
				.any(|usage| usage.buffer.same_as(first_output))
		{
			return None;
		}
		// An absorbed inner add's output may have no other reader: it
		// is never written once the whole tree moves into the kernel.
		if nodes.iter().enumerate().skip(1).any(|(other, node)| {
			other != outer_index
				&& node
					.buffers
					.iter()
					.any(|usage| usage.buffer.same_as(first_output))
		}) {
			return None;
		}
		let third = if outer.buffers[0].buffer.same_as(first_output) {
			&outer.buffers[1].buffer
		} else {
			&outer.buffers[0].buffer
		};
		(
			outer_index,
			&first.buffers[0].buffer,
			&first.buffers[1].buffer,
			third,
			3u32,
		)
	};
	let outer = &nodes[outer_index];
	if outer.push_constants[0] != first.push_constants[0] {
		return None;
	}
	let (PushConstant::U32(rows), PushConstant::U32(columns)) =
		(norm.push_constants[0], norm.push_constants[1])
	else {
		return None;
	};
	if first.push_constants[0] != PushConstant::U32(rows.checked_mul(columns)?) {
		return None;
	}
	let reads = [
		&norm.buffers[0].buffer,
		&norm.buffers[1].buffer,
		&norm.buffers[2].buffer,
		addend_a,
		addend_b,
		addend_c,
	];
	let writes = [&norm.buffers[4].buffer, &norm.buffers[5].buffer, dy];
	if writes
		.iter()
		.any(|write| reads.iter().any(|read| read.same_as(write)))
		|| writes[0].same_as(writes[1])
		|| writes[0].same_as(writes[2])
		|| writes[1].same_as(writes[2])
	{
		return None;
	}
	// Nodes before the combine may produce the third addend but must not
	// rewrite the inner add's captured operands (the kernel reads them at
	// the combine's slot, not the inner's) nor touch the moved writes.
	if nodes
		.iter()
		.enumerate()
		.skip(1)
		.take(outer_index.saturating_sub(1))
		.any(|(_, node)| {
			node.buffers.iter().any(|usage| {
				writes.iter().any(|write| write.same_as(&usage.buffer))
					|| (usage.access != BufferAccess::Read
						&& [&first.buffers[0].buffer, &first.buffers[1].buffer]
							.iter()
							.any(|operand| operand.same_as(&usage.buffer)))
			})
		}) {
		return None;
	}
	// Nodes between the combine and the norm must leave every operand the
	// kernel still reads alone and never touch the moved dx/contribution/dy
	// writes.
	if nodes
		.iter()
		.enumerate()
		.skip(outer_index + 1)
		.take(norm_index - outer_index - 1)
		.any(|(_, node)| {
			node.buffers.iter().any(|usage| {
				reads
					.iter()
					.any(|read| read.same_as(&usage.buffer) && usage.access != BufferAccess::Read)
					|| writes.iter().any(|write| write.same_as(&usage.buffer))
			})
		}) {
		return None;
	}
	let kernel = KernelId::MlLayerNormBackwardSumF32;
	let semantic_ops = if outer_index != 0 {
		merged_semantic_ops(&[&first.semantic_ops, &outer.semantic_ops, &norm.semantic_ops])
	} else {
		merged_semantic_ops(&[&first.semantic_ops, &norm.semantic_ops])
	};
	let (operation, op_contract_hash) = fused_identity(&semantic_ops, first, kernel);
	let mut consumed = vec![norm_index];
	if outer_index != 0 {
		consumed.push(outer_index);
	}
	Some((
		ComputeNode {
			operation,
			kernel,
			buffers: vec![
				norm.buffers[0].clone(),
				norm.buffers[1].clone(),
				norm.buffers[2].clone(),
				BufferUse {
					buffer: addend_a.clone(),
					access: BufferAccess::Read,
				},
				BufferUse {
					buffer: addend_b.clone(),
					access: BufferAccess::Read,
				},
				BufferUse {
					buffer: addend_c.clone(),
					access: BufferAccess::Read,
				},
				norm.buffers[4].clone(),
				norm.buffers[5].clone(),
				BufferUse {
					buffer: dy.clone(),
					access: BufferAccess::Write,
				},
			],
			push_constants: vec![
				PushConstant::U32(rows),
				PushConstant::U32(columns),
				PushConstant::U32(addend_count),
			],
			workgroups: norm.workgroups,
			indirect: None,
			semantic_ops,
			op_contract_hash,
			jit: None,
			fusion_signature: None,
		},
		consumed,
		outer_index,
	))
}

/// Decode a `matrix.add` whose output feeds one or more
/// `embedding_backward` scatter adjoints — the head-tail sum that merges
/// the norm-combined and head-input gradients at the transformer head.
/// The combined buffer is never materialized: each fused consumer sums
/// the two addends inside its serial position scan, so every node that
/// reads the add's output must be an eligible embedding backward or the
/// match declines. Returns one `(distance, fused)` emission per consumer
/// plus the absorbed add — emissions run at the consumer slots where the
/// unfused dispatch would have read the sum.
fn match_embedding_backward_sum(nodes: &[ComputeNode]) -> Option<Vec<(usize, ComputeNode)>> {
	let add = &nodes[0];
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
	let sum = &add.buffers[2].buffer;
	let addend_a = &add.buffers[0].buffer;
	let addend_b = &add.buffers[1].buffer;
	let mut emissions = Vec::new();
	for (distance, consumer) in nodes.iter().enumerate().skip(1) {
		if !consumer
			.buffers
			.iter()
			.any(|usage| usage.buffer.same_as(sum))
		{
			continue;
		}
		if consumer.kernel != KernelId::MlEmbeddingBackwardF32
			|| consumer.indirect.is_some()
			|| consumer.buffers.len() != 3
			|| consumer.push_constants.len() != 3
			|| consumer
				.buffers
				.iter()
				.take(2)
				.any(|usage| usage.access != BufferAccess::Read)
			|| consumer.buffers[2].access != BufferAccess::Write
			|| !consumer.buffers[1].buffer.same_as(sum)
		{
			// Anything that is not an eligible embedding backward touching
			// the sum — including another writer — ends the match.
			return None;
		}
		let (
			PushConstant::U32(index_count),
			PushConstant::U32(_num_embeddings),
			PushConstant::U32(embedding_dim),
		) = (
			consumer.push_constants[0],
			consumer.push_constants[1],
			consumer.push_constants[2],
		)
		else {
			return None;
		};
		if add.push_constants[0] != PushConstant::U32(index_count.checked_mul(embedding_dim)?)
			|| consumer.buffers[2].buffer.same_as(addend_a)
			|| consumer.buffers[2].buffer.same_as(addend_b)
		{
			return None;
		}
		emissions.push((distance, consumer));
	}
	if emissions.is_empty() {
		return None;
	}
	let last = emissions.last().map(|(distance, _)| *distance)?;
	// The addends were captured at the add's slot; no node before the last
	// consumer may rewrite them or the per-consumer fused reads would see
	// a different value than the unfused sum did.
	if nodes
		.iter()
		.enumerate()
		.skip(1)
		.take(last)
		.any(|(_, node)| {
			node.buffers.iter().any(|usage| {
				usage.access != BufferAccess::Read
					&& (usage.buffer.same_as(addend_a) || usage.buffer.same_as(addend_b))
			})
		}) {
		return None;
	}
	let kernel = KernelId::MlEmbeddingBackwardSumF32;
	Some(
		emissions
			.iter()
			.enumerate()
			.map(|(ordinal, (distance, consumer))| {
				let semantic_ops = if ordinal == 0 {
					merged_semantic_ops(&[&add.semantic_ops, &consumer.semantic_ops])
				} else {
					consumer.semantic_ops.clone()
				};
				let (operation, op_contract_hash) = fused_identity(&semantic_ops, consumer, kernel);
				(
					*distance,
					ComputeNode {
						operation,
						kernel,
						buffers: vec![
							consumer.buffers[0].clone(),
							BufferUse {
								buffer: addend_a.clone(),
								access: BufferAccess::Read,
							},
							BufferUse {
								buffer: addend_b.clone(),
								access: BufferAccess::Read,
							},
							consumer.buffers[2].clone(),
						],
						push_constants: consumer.push_constants.clone(),
						workgroups: consumer.workgroups,
						indirect: None,
						semantic_ops,
						op_contract_hash,
						jit: None,
						fusion_signature: None,
					},
				)
			})
			.collect(),
	)
}

/// Decode the Conv1d tail `[MlConv1dBiasAddF32, MlConv1dTransposeF32]`
/// plus an eligible direct consumer of the transposed output into one
/// dispatch, or the pair alone when the tail ends there. The bias add
/// updates the projected GEMM buffer in place, so every fused shape must
/// leave it raw: no later node may touch that intermediate or it would
/// read a different value than the unfused chain produced. Variants that
/// drop the transposed output buffer additionally require it to have no
/// other readers; the channel-norm variants republish it because the
/// backward adjoint consumes it as saved state. `unobservable` proves that
/// no owner outside the pending execution holds a buffer whose final
/// contents the fused shape changes; without that proof the tail degrades
/// to the shape that still materializes it, or declines.
fn match_conv_1d_bias_tail(
	nodes: &[ComputeNode],
	unobservable: impl Fn(&Buffer) -> bool,
) -> Option<(ComputeNode, usize)> {
	let add = &nodes[0];
	if add.kernel != KernelId::MlConv1dBiasAddF32
		|| add.indirect.is_some()
		|| add.buffers.len() != 2
		|| add.push_constants.len() != 2
		|| add.buffers[0].access != BufferAccess::ReadWrite
		|| add.buffers[1].access != BufferAccess::Read
	{
		return None;
	}
	let transpose = &nodes[1];
	if transpose.kernel != KernelId::MlConv1dTransposeF32
		|| transpose.indirect.is_some()
		|| transpose.buffers.len() != 2
		|| transpose.push_constants.len() != 3
		|| transpose.buffers[0].access != BufferAccess::Read
		|| transpose.buffers[1].access != BufferAccess::Write
		|| !transpose.buffers[0].buffer.same_as(&add.buffers[0].buffer)
	{
		return None;
	}
	// The add covers the full [batch * rows, columns] tile the transpose
	// reshapes: `rows` is the output length T and `columns` is channels C.
	let (PushConstant::U32(batch), PushConstant::U32(rows), PushConstant::U32(columns)) = (
		transpose.push_constants[0],
		transpose.push_constants[1],
		transpose.push_constants[2],
	) else {
		return None;
	};
	if add.push_constants[0] != PushConstant::U32(batch.checked_mul(rows)?)
		|| add.push_constants[1] != PushConstant::U32(columns)
	{
		return None;
	}
	let projected = &add.buffers[0].buffer;
	let conv_out = &transpose.buffers[1].buffer;
	let buffer_free_after = |buffer: &Buffer, skip: usize| {
		unobservable(buffer)
			&& !nodes.iter().skip(skip).any(|node| {
				node
					.buffers
					.iter()
					.any(|usage| usage.buffer.same_as(buffer))
			})
	};
	let projected_free_after = |skip: usize| buffer_free_after(projected, skip);
	if nodes.len() > 2 {
		let consumer = &nodes[2];
		if consumer.kernel == KernelId::MlReluF32
			&& consumer.indirect.is_none()
			&& consumer.buffers.len() == 2
			&& consumer.push_constants.len() == 1
			&& consumer.buffers[0].access == BufferAccess::Read
			&& consumer.buffers[0].buffer.same_as(conv_out)
			&& consumer.buffers[1].access == BufferAccess::Write
			&& !consumer.buffers[1].buffer.same_as(projected)
			&& !consumer.buffers[1].buffer.same_as(&add.buffers[1].buffer)
			&& consumer.push_constants[0]
				== PushConstant::U32(batch.checked_mul(rows)?.checked_mul(columns)?)
		{
			let relu_out = &consumer.buffers[1].buffer;
			if nodes.len() > 3 {
				let sum_node = &nodes[3];
				// A residual add directly after the ReLU folds into the
				// same tile pass: the kernel republishes the activated
				// output the ReLU adjoint saves and writes
				// relu + residual for the sum's consumers.
				let residual = if sum_node.kernel == KernelId::MatrixAddF32
					&& sum_node.indirect.is_none()
					&& sum_node.buffers.len() == 3
					&& sum_node.push_constants.len() == 1
					&& sum_node
						.buffers
						.iter()
						.take(2)
						.all(|usage| usage.access == BufferAccess::Read)
					&& sum_node.buffers[2].access == BufferAccess::Write
					&& sum_node.push_constants[0]
						== PushConstant::U32(batch.checked_mul(rows)?.checked_mul(columns)?)
				{
					match (
						sum_node.buffers[0].buffer.same_as(relu_out),
						sum_node.buffers[1].buffer.same_as(relu_out),
					) {
						(true, false) => Some(&sum_node.buffers[1].buffer),
						(false, true) => Some(&sum_node.buffers[0].buffer),
						_ => None,
					}
				} else {
					None
				};
				if let Some(residual) = residual {
					let sum = &sum_node.buffers[2].buffer;
					// The fused pass leaves the projected buffer raw and
					// never materializes conv_out, so neither may feed or
					// be read past the fused window; the sum writes a
					// fresh buffer.
					if !residual.same_as(projected)
						&& !residual.same_as(conv_out)
						&& !sum.same_as(projected)
						&& !sum.same_as(&add.buffers[1].buffer)
						&& !sum.same_as(relu_out)
						&& !sum.same_as(residual)
						&& !sum.same_as(conv_out)
						&& projected_free_after(4)
						&& buffer_free_after(conv_out, 4)
					{
						let kernel = KernelId::MlConv1dBiasTransposeResidualReluF32;
						let semantic_ops = merged_semantic_ops(&[
							&add.semantic_ops,
							&transpose.semantic_ops,
							&consumer.semantic_ops,
							&sum_node.semantic_ops,
						]);
						let (operation, op_contract_hash) = fused_identity(&semantic_ops, add, kernel);
						let fused = ComputeNode {
							operation,
							kernel,
							buffers: vec![
								BufferUse {
									buffer: projected.clone(),
									access: BufferAccess::Read,
								},
								add.buffers[1].clone(),
								consumer.buffers[1].clone(),
								BufferUse {
									buffer: residual.clone(),
									access: BufferAccess::Read,
								},
								sum_node.buffers[2].clone(),
							],
							push_constants: transpose.push_constants.clone(),
							workgroups: transpose.workgroups,
							indirect: None,
							semantic_ops,
							op_contract_hash,
							jit: None,
							fusion_signature: None,
						};
						return Some((fused, 4));
					}
				}
			}
			if projected_free_after(3) && buffer_free_after(conv_out, 3) {
				let kernel = KernelId::MlConv1dBiasTransposeReluF32;
				let semantic_ops = merged_semantic_ops(&[
					&add.semantic_ops,
					&transpose.semantic_ops,
					&consumer.semantic_ops,
				]);
				let (operation, op_contract_hash) = fused_identity(&semantic_ops, add, kernel);
				let fused = ComputeNode {
					operation,
					kernel,
					buffers: vec![
						BufferUse {
							buffer: projected.clone(),
							access: BufferAccess::Read,
						},
						add.buffers[1].clone(),
						consumer.buffers[1].clone(),
					],
					push_constants: transpose.push_constants.clone(),
					workgroups: transpose.workgroups,
					indirect: None,
					semantic_ops,
					op_contract_hash,
					jit: None,
					fusion_signature: None,
				};
				return Some((fused, 3));
			}
		}
		if (consumer.kernel == KernelId::MlChannelNormF32
			|| consumer.kernel == KernelId::MlChannelNormReluF32)
			&& consumer.indirect.is_none()
			&& consumer.buffers.len() == 4
			&& consumer.push_constants.len() == 4
			&& consumer
				.buffers
				.iter()
				.take(3)
				.all(|usage| usage.access == BufferAccess::Read)
			&& consumer.buffers[0].buffer.same_as(conv_out)
			&& consumer.buffers[3].access == BufferAccess::Write
			&& !consumer.buffers[3].buffer.same_as(projected)
			&& !consumer.buffers[3].buffer.same_as(&add.buffers[1].buffer)
			&& consumer.push_constants[0] == PushConstant::U32(batch)
			&& consumer.push_constants[1] == PushConstant::U32(columns)
			&& consumer.push_constants[2] == PushConstant::U32(rows)
			&& matches!(consumer.push_constants[3], PushConstant::F32(_))
			&& projected_free_after(3)
		{
			let kernel = if consumer.kernel == KernelId::MlChannelNormReluF32 {
				KernelId::MlConv1dBiasTransposeNormReluF32
			} else {
				KernelId::MlConv1dBiasTransposeNormF32
			};
			let semantic_ops = merged_semantic_ops(&[
				&add.semantic_ops,
				&transpose.semantic_ops,
				&consumer.semantic_ops,
			]);
			let (operation, op_contract_hash) = fused_identity(&semantic_ops, add, kernel);
			let fused = ComputeNode {
				operation,
				kernel,
				buffers: vec![
					BufferUse {
						buffer: projected.clone(),
						access: BufferAccess::Read,
					},
					add.buffers[1].clone(),
					consumer.buffers[1].clone(),
					consumer.buffers[2].clone(),
					consumer.buffers[3].clone(),
					BufferUse {
						buffer: conv_out.clone(),
						access: BufferAccess::Write,
					},
				],
				push_constants: vec![
					consumer.push_constants[0],
					consumer.push_constants[1],
					consumer.push_constants[2],
					consumer.push_constants[3],
				],
				workgroups: consumer.workgroups,
				indirect: None,
				semantic_ops,
				op_contract_hash,
				jit: None,
				fusion_signature: None,
			};
			return Some((fused, 3));
		}
		if consumer.kernel == KernelId::MatrixTransposeF32
			&& consumer.indirect.is_none()
			&& consumer.buffers.len() == 2
			&& consumer.push_constants.len() == 3
			&& consumer.buffers[0].access == BufferAccess::Read
			&& consumer.buffers[0].buffer.same_as(conv_out)
			&& consumer.buffers[1].access == BufferAccess::Write
			&& !consumer.buffers[1].buffer.same_as(projected)
			&& !consumer.buffers[1].buffer.same_as(&add.buffers[1].buffer)
			// The generic transpose must be the exact inverse of the NCL
			// transpose: same batch, and its row/column extents swapped.
			&& consumer.push_constants[0] == PushConstant::U32(batch)
			&& consumer.push_constants[1] == PushConstant::U32(columns)
			&& consumer.push_constants[2] == PushConstant::U32(rows)
			&& projected_free_after(3)
			&& buffer_free_after(conv_out, 3)
		{
			let kernel = KernelId::MlConv1dBiasCopyF32;
			let semantic_ops = merged_semantic_ops(&[
				&add.semantic_ops,
				&transpose.semantic_ops,
				&consumer.semantic_ops,
			]);
			let (operation, op_contract_hash) = fused_identity(&semantic_ops, add, kernel);
			let fused = ComputeNode {
				operation,
				kernel,
				buffers: vec![
					BufferUse {
						buffer: projected.clone(),
						access: BufferAccess::Read,
					},
					add.buffers[1].clone(),
					consumer.buffers[1].clone(),
				],
				push_constants: vec![add.push_constants[0], add.push_constants[1]],
				workgroups: kernel.linear_workgroups(batch.checked_mul(rows)?.checked_mul(columns)?),
				indirect: None,
				semantic_ops,
				op_contract_hash,
				jit: None,
				fusion_signature: None,
			};
			return Some((fused, 3));
		}
	}
	if !projected_free_after(2) {
		return None;
	}
	let fused = fused_node(
		add,
		transpose,
		KernelId::MlConv1dBiasTransposeF32,
		vec![
			BufferUse {
				buffer: projected.clone(),
				access: BufferAccess::Read,
			},
			add.buffers[1].clone(),
			transpose.buffers[1].clone(),
		],
		transpose.push_constants.clone(),
		transpose.workgroups,
	);
	Some((fused, 2))
}

/// Decode a `[MlConv2dF32, MlReluF32]` pair into one direct-convolution
/// dispatch that applies the activation at the store. The raw conv output
/// is never materialized: Conv2d's backward reads input and `dY` only, so
/// the fused node's single write — the ReLU output the adjoint saves —
/// reproduces every value the unfused pair publishes, and the match
/// requires the conv output to have no other readers. The caller rejects
/// the replacement when an external owner still holds the conv output.
fn match_conv_2d_relu(nodes: &[ComputeNode]) -> Option<ComputeNode> {
	let conv = &nodes[0];
	if conv.kernel != KernelId::MlConv2dF32
		|| conv.indirect.is_some()
		|| conv.buffers.len() != 4
		|| conv.push_constants.len() != 11
		|| conv
			.buffers
			.iter()
			.take(3)
			.any(|usage| usage.access != BufferAccess::Read)
		|| conv.buffers[3].access != BufferAccess::Write
	{
		return None;
	}
	let relu = &nodes[1];
	if relu.kernel != KernelId::MlReluF32
		|| relu.indirect.is_some()
		|| relu.buffers.len() != 2
		|| relu.push_constants.len() != 1
		|| relu.buffers[0].access != BufferAccess::Read
		|| !relu.buffers[0].buffer.same_as(&conv.buffers[3].buffer)
		|| relu.buffers[1].access != BufferAccess::Write
	{
		return None;
	}
	// The ReLU element count must cover the conv's whole output tile:
	// batch × output_channels × output_height × output_width.
	let (PushConstant::U32(batch), PushConstant::U32(output_channels)) =
		(conv.push_constants[0], conv.push_constants[2])
	else {
		return None;
	};
	let (PushConstant::U32(output_height), PushConstant::U32(output_width)) =
		(conv.push_constants[8], conv.push_constants[9])
	else {
		return None;
	};
	let output_count = batch
		.checked_mul(output_channels)?
		.checked_mul(output_height)?
		.checked_mul(output_width)?;
	if relu.push_constants[0] != PushConstant::U32(output_count) {
		return None;
	}
	let conv_out = &conv.buffers[3].buffer;
	if relu.buffers[1].buffer.same_as(conv_out)
		|| conv
			.buffers
			.iter()
			.take(3)
			.any(|usage| relu.buffers[1].buffer.same_as(&usage.buffer))
		|| nodes.iter().skip(2).any(|node| {
			node
				.buffers
				.iter()
				.any(|usage| usage.buffer.same_as(conv_out))
		}) {
		return None;
	}
	Some(fused_node(
		conv,
		relu,
		KernelId::MlConv2dReluF32,
		vec![
			conv.buffers[0].clone(),
			conv.buffers[1].clone(),
			conv.buffers[2].clone(),
			relu.buffers[1].clone(),
		],
		conv.push_constants.clone(),
		conv.workgroups,
	))
}

/// Decode a `[MlMaxPool2dBackwardF32, MlReluBackwardF32]` pair — the
/// adjoint chain of `conv → relu → max_pool` — into one gather that
/// applies the ReLU mask at the store. The unmasked pool input gradient
/// is never materialized, so the match requires it to have no other
/// readers and the caller requires it to have no external owner; the fused
/// write is the ReLU input gradient, identical to the unfused pair's.
fn match_max_pool_relu_backward(nodes: &[ComputeNode]) -> Option<ComputeNode> {
	let pool = &nodes[0];
	if pool.kernel != KernelId::MlMaxPool2dBackwardF32
		|| pool.indirect.is_some()
		|| pool.buffers.len() != 3
		|| pool.push_constants.len() != 9
		|| pool
			.buffers
			.iter()
			.take(2)
			.any(|usage| usage.access != BufferAccess::Read)
		|| pool.buffers[2].access != BufferAccess::Write
	{
		return None;
	}
	let relu = &nodes[1];
	if relu.kernel != KernelId::MlReluBackwardF32
		|| relu.indirect.is_some()
		|| relu.buffers.len() != 3
		|| relu.push_constants.len() != 1
		|| relu.buffers[0].access != BufferAccess::Read
		|| relu.buffers[1].access != BufferAccess::Read
		|| !relu.buffers[1].buffer.same_as(&pool.buffers[2].buffer)
		|| relu.buffers[2].access != BufferAccess::Write
	{
		return None;
	}
	// The mask covers the pool's whole input tile:
	// batch × channels × input_height × input_width.
	let (PushConstant::U32(batch), PushConstant::U32(channels)) =
		(pool.push_constants[0], pool.push_constants[1])
	else {
		return None;
	};
	let (PushConstant::U32(input_height), PushConstant::U32(input_width)) =
		(pool.push_constants[2], pool.push_constants[3])
	else {
		return None;
	};
	let input_count = batch
		.checked_mul(channels)?
		.checked_mul(input_height)?
		.checked_mul(input_width)?;
	if relu.push_constants[0] != PushConstant::U32(input_count) {
		return None;
	}
	let pool_out = &pool.buffers[2].buffer;
	if relu.buffers[2].buffer.same_as(pool_out)
		|| relu.buffers[2].buffer.same_as(&pool.buffers[0].buffer)
		|| relu.buffers[2].buffer.same_as(&pool.buffers[1].buffer)
		|| relu.buffers[2].buffer.same_as(&relu.buffers[0].buffer)
		|| nodes.iter().skip(2).any(|node| {
			node
				.buffers
				.iter()
				.any(|usage| usage.buffer.same_as(pool_out))
		}) {
		return None;
	}
	Some(fused_node(
		pool,
		relu,
		KernelId::MlMaxPool2dReluBackwardF32,
		vec![
			pool.buffers[0].clone(),
			pool.buffers[1].clone(),
			relu.buffers[0].clone(),
			relu.buffers[2].clone(),
		],
		pool.push_constants.clone(),
		pool.workgroups,
	))
}
/// stream: the attention backward interleaves the three projection
/// adjoints and their parameter adjoints, so the consuming data adjoint
/// can trail its producer by a handful of dispatches.
const MAX_BMM_CHAIN_LOOKAHEAD: usize = 8;

/// Decode a `bmm_tiled_16`/`bmm_tn_tiled_16` producer whose output feeds a
/// later `MlLinearBackwardF32`. The fused kernel republishes the bmm
/// output (the parameter adjoint still reads it) and computes the input
/// gradient in the same dispatch. The consumer may sit behind unrelated
/// nodes, so the window scans forward; every intervening node must leave
/// the bmm operands, the weight, the republished `dY` and the consumer's
/// `dX` writes untouched, because the fused dispatch runs at the
/// producer's position with the consumer's original operands.
fn match_linear_backward_bmm_chain(nodes: &[ComputeNode]) -> Option<(ComputeNode, usize)> {
	let producer = &nodes[0];
	let kind = match producer.kernel {
		KernelId::MlBmmTiled16F32 => 0,
		KernelId::MlBmmTnTiled16F32 => 1,
		_ => return None,
	};
	if producer.indirect.is_some()
		|| producer.buffers.len() != 3
		|| producer.push_constants.len() != 4
		|| producer
			.buffers
			.iter()
			.take(2)
			.any(|usage| usage.access != BufferAccess::Read)
		|| producer.buffers[2].access != BufferAccess::Write
	{
		return None;
	}
	let (
		PushConstant::U32(batch),
		PushConstant::U32(rows),
		PushConstant::U32(inner),
		PushConstant::U32(columns),
	) = (
		producer.push_constants[0],
		producer.push_constants[1],
		producer.push_constants[2],
		producer.push_constants[3],
	)
	else {
		return None;
	};
	if columns > 32 {
		return None;
	}
	let dy = &producer.buffers[2].buffer;
	for distance in 1..nodes.len() {
		let consumer = &nodes[distance];
		if consumer
			.buffers
			.iter()
			.any(|usage| usage.access != BufferAccess::Read && usage.buffer.same_as(dy))
		{
			return None;
		}
		if consumer.kernel != KernelId::MlLinearBackwardF32
			|| consumer.buffers.len() != 3
			|| !consumer.buffers[1].buffer.same_as(dy)
		{
			continue;
		}
		if consumer.indirect.is_some()
			|| consumer.buffers.len() != 3
			|| consumer.push_constants.len() != 3
			|| consumer
				.buffers
				.iter()
				.take(2)
				.any(|usage| usage.access != BufferAccess::Read)
			|| consumer.buffers[2].access != BufferAccess::Write
		{
			return None;
		}
		let (
			PushConstant::U32(gradient_rows),
			PushConstant::U32(input_features),
			PushConstant::U32(output_features),
		) = (
			consumer.push_constants[0],
			consumer.push_constants[1],
			consumer.push_constants[2],
		)
		else {
			return None;
		};
		if output_features != columns
			|| input_features > 32
			|| u64::from(gradient_rows) != u64::from(batch) * u64::from(rows)
		{
			return None;
		}
		let weight = &consumer.buffers[0].buffer;
		let dx = &consumer.buffers[2].buffer;
		// Intervening nodes may freely read the operands and the bmm result,
		// but nothing may mutate them, and nothing may touch the gradient:
		// the fused node publishes dx at the producer's slot, earlier than
		// the consumer dispatch it replaces.
		if nodes[1..distance].iter().any(|node| {
			node.buffers.iter().any(|usage| {
				usage.buffer.same_as(dx)
					|| (usage.access != BufferAccess::Read
						&& (usage.buffer.same_as(&producer.buffers[0].buffer)
							|| usage.buffer.same_as(&producer.buffers[1].buffer)
							|| usage.buffer.same_as(weight)
							|| usage.buffer.same_as(dy)))
			})
		}) {
			return None;
		}
		let reads = [
			&producer.buffers[0].buffer,
			&producer.buffers[1].buffer,
			weight,
		];
		if reads
			.iter()
			.any(|read| read.same_as(dy) || read.same_as(dx))
			|| dy.same_as(dx)
		{
			return None;
		}
		return Some((
			fused_node(
				producer,
				consumer,
				KernelId::MlLinearBackwardBmmChainF32,
				vec![
					producer.buffers[0].clone(),
					producer.buffers[1].clone(),
					consumer.buffers[0].clone(),
					BufferUse {
						buffer: dy.clone(),
						access: BufferAccess::Write,
					},
					consumer.buffers[2].clone(),
				],
				vec![
					PushConstant::U32(batch),
					PushConstant::U32(rows),
					PushConstant::U32(inner),
					PushConstant::U32(columns),
					PushConstant::U32(input_features),
					PushConstant::U32(kind),
				],
				[1, rows.div_ceil(16), batch],
			),
			distance,
		));
	}
	None
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
	fuse_elementwise_with_limit(device, nodes, refs, MAX_FUSION_ELEMENTS)
}

#[cfg(test)]
#[path = "../../../test/rs/runtime/jit_fusion_unit.rs"]
pub(in crate::runtime) mod jit_lab;

fn fuse_elementwise_with_limit(
	device: &Arc<Device>,
	nodes: Vec<ComputeNode>,
	refs: &[&Storage],
	max_elements: u32,
) -> (Vec<ComputeNode>, Vec<Storage>) {
	let census = device.fusion_census();
	census.pass(nodes.len());
	// A missing fused pipeline falls back to the recorded nodes; the per-op
	// kernels remain the canonical path. The environment gate exists for
	// fused-vs-unfused measurement on one binary.
	if std::env::var_os("OA_DISABLE_ELEMENTWISE_FUSION").is_some()
		|| crate::runtime::device::jit::jit_route_override()
			== Some(crate::runtime::device::jit::JitRouteOverride::Unfused)
		|| device
			.require_kernel(KernelId::MatrixFusedChainF32)
			.is_err()
	{
		census.nodes(NodeOutcome::Disabled, nodes.len());
		return (nodes, Vec::new());
	}
	let extended = std::env::var_os("OA_DISABLE_RL_FUSION").is_none();
	let raw: Vec<Option<NodeOp>> = nodes.iter().map(decode_node).collect();
	let decoded: Vec<Option<NodeOp>> = raw
		.iter()
		.map(|op| {
			let op = (*op)?;
			if !extended
				&& ((40..=48).contains(&op.opcode)
					|| matches!(
						op.shape,
						ElementwiseShape::Broadcast | ElementwiseShape::Gather
					)) {
				return None;
			}
			Some(op)
		})
		.collect();
	let regions = discover_regions(&nodes, &decoded);
	let mut in_region = vec![false; nodes.len()];
	for &(start, end) in &regions {
		in_region[start..end].fill(true);
	}
	let outside = |index: usize| match (raw[index], decoded[index]) {
		(None, _) => NodeOutcome::NotElementwise,
		(Some(_), None) => NodeOutcome::PolicyExcluded,
		(Some(_), Some(_)) => NodeOutcome::Hazard,
	};
	let mut result = Vec::with_capacity(nodes.len());
	let mut tapes = Vec::new();
	let mut copied = 0;
	for (start, run_end) in regions {
		for index in copied..start {
			census.nodes(outside(index), 1);
		}
		result.extend_from_slice(&nodes[copied..start]);
		if run_end - start == 1 {
			// Adjacent admitted nodes with equal extents share one region, so
			// an admitted neighbour here always differs in element count.
			let boundary =
				(start > 0 && in_region[start - 1]) || (run_end < nodes.len() && in_region[run_end]);
			census.nodes(
				if boundary {
					NodeOutcome::ExtentBoundary
				} else {
					NodeOutcome::Isolated
				},
				1,
			);
		}
		// Why the previous attempt could not include the current node: the
		// last decline before a cursor advances is exactly the attempt that
		// ended at this node, and explains a trailing node that has no attempt
		// of its own.
		let mut split = None;
		let mut cursor = start;
		while cursor < run_end {
			let mut end = run_end - 1;
			let mut last_decline = None;
			let mut plan = None;
			while end > cursor {
				match fit_segment(device, &nodes, &decoded, refs, cursor, end, max_elements) {
					Ok(fused) => {
						plan = Some(fused);
						break;
					}
					Err(outcome) => {
						last_decline = Some(outcome);
						end -= 1;
					}
				}
			}
			match plan {
				Some(plan) => {
					census.fused(plan.node.jit.is_some(), end + 1 - cursor);
					result.push(plan.node);
					tapes.push(plan.tape);
					split = last_decline;
					cursor = end + 1;
				}
				None => {
					if run_end - start > 1 {
						census.nodes(last_decline.or(split).unwrap_or(NodeOutcome::Isolated), 1);
					}
					split = last_decline;
					result.push(nodes[cursor].clone());
					cursor += 1;
				}
			}
		}
		copied = run_end;
	}
	for index in copied..nodes.len() {
		census.nodes(outside(index), 1);
	}
	result.extend_from_slice(&nodes[copied..]);
	(result, tapes)
}

/// Whether `op`'s physical write is a legal SSA definition: no other node
/// writes it and no earlier node reads it.
fn write_is_exclusive(nodes: &[ComputeNode], index: usize, op: &NodeOp) -> bool {
	let write = write_buffer(&nodes[index], op);
	!nodes.iter().enumerate().any(|(i, node)| {
		(i != index && node_writes(node, write)) || (i < index && node_reads(node, write))
	})
}

/// Discover homogeneous, effect-free regions once for every lowering backend.
fn discover_regions(nodes: &[ComputeNode], decoded: &[Option<NodeOp>]) -> Vec<(usize, usize)> {
	let mut regions = Vec::new();
	let mut start = None;
	let mut count = 0;
	for (index, op) in decoded.iter().enumerate() {
		let admitted = op.filter(|op| write_is_exclusive(nodes, index, op));
		if admitted.is_none_or(|op| op.element_count != count)
			&& let Some(begin) = start.take()
		{
			regions.push((begin, index));
		}
		if let Some(op) = admitted
			&& start.is_none()
		{
			start = Some(index);
			count = op.element_count;
		}
	}
	if let Some(begin) = start {
		regions.push((begin, nodes.len()));
	}
	regions
}

/// A validated graph region, with physical bindings kept outside the portable IR.
struct FusionCandidate {
	program: Program,
	inputs: Vec<Buffer>,
	outputs: Vec<Buffer>,
}

fn build_program(
	nodes: &[ComputeNode],
	decoded: &[Option<NodeOp>],
	refs: &[&Storage],
	start: usize,
	end: usize,
) -> Result<FusionCandidate, NodeOutcome> {
	let rejected = NodeOutcome::ProgramRejected;
	if start > end || end >= nodes.len() {
		return Err(rejected);
	}
	let ops = decoded
		.get(start..=end)
		.ok_or(rejected)?
		.iter()
		.copied()
		.collect::<Option<Vec<_>>>()
		.ok_or(rejected)?;
	let count = ops.first().ok_or(rejected)?.element_count;
	if ops.iter().any(|op| op.element_count != count) {
		return Err(NodeOutcome::ExtentBoundary);
	}
	let mut inputs: Vec<Buffer> = Vec::new();
	let mut outputs = Vec::new();
	let mut instructions = Vec::new();
	let mut retained = Vec::new();
	for (index, op) in ops.iter().enumerate() {
		let node = &nodes[start + index];
		let write = write_buffer(node, op);
		// No region can turn a read-before-write or repeated physical writer
		// into an SSA register. These are source-preserving hard boundaries.
		if !write_is_exclusive(nodes, start + index, op) {
			return Err(NodeOutcome::Hazard);
		}
		let needs_store = index + 1 == ops.len()
			|| !storage_unobservable(refs, write)
			|| nodes
				.iter()
				.enumerate()
				.any(|(i, node)| !(start..=end).contains(&i) && node_reads(node, write));
		let mut sources = Vec::new();
		for operand in 0..op.shape.read_count() {
			let buffer = &node.buffers[operand].buffer;
			let mapping = if matches!(op.shape, ElementwiseShape::Broadcast) {
				let rank = op.metadata[0] as usize;
				let mut stride = 1u32;
				let mut linear = true;
				for axis in (0..rank).rev() {
					if op.metadata[1 + axis] > 1 && op.metadata[9 + operand * 8 + axis] != stride {
						linear = false;
					}
					stride = stride.checked_mul(op.metadata[1 + axis]).ok_or(rejected)?;
				}
				!linear
			} else {
				false
			};
			if let Some(producer) = (0..index)
				.rev()
				.find(|&i| write_buffer(&nodes[start + i], &ops[i]).same_as(buffer))
			{
				// Cross-invocation reads require a materialized source dispatch.
				if mapping || matches!(op.shape, ElementwiseShape::Gather) {
					return Err(NodeOutcome::CrossInvocation);
				}
				sources.push(Source::Value(producer));
			} else {
				let slot = match inputs.iter().position(|input| input.same_as(buffer)) {
					Some(slot) => slot,
					None => {
						inputs.push(buffer.clone());
						inputs.len() - 1
					}
				};
				sources.push(if mapping {
					Source::Broadcast {
						input: slot,
						rank: op.metadata[0],
						dimensions: op.metadata[1..9].try_into().map_err(|_| rejected)?,
						strides: op.metadata[9 + operand * 8..17 + operand * 8]
							.try_into()
							.map_err(|_| rejected)?,
					}
				} else {
					Source::Input(slot)
				});
			}
		}
		if matches!(op.shape, ElementwiseShape::Scalar) {
			sources.push(Source::Constant(op.scalar.to_bits()));
		}
		let parameters = match op.shape {
			ElementwiseShape::Gather => op.metadata[..3].to_vec(),
			ElementwiseShape::BinaryScalar => vec![op.scalar.to_bits()],
			_ => Vec::new(),
		};
		instructions.push(Instruction {
			opcode: op.opcode,
			sources,
			parameters,
		});
		if needs_store {
			retained.push(index);
			outputs.push(write.clone());
		}
	}
	let program = Program {
		element_count: count,
		input_bytes: inputs.iter().map(Buffer::size).collect(),
		instructions,
		outputs: retained,
	};
	if !program.validate() {
		return Err(rejected);
	}
	Ok(FusionCandidate {
		program,
		inputs,
		outputs,
	})
}

/// Try to encode `nodes[start..=end]` into one fused dispatch; the error is
/// the first legality, capacity, resource or routing reason that rejects the
/// region.
fn fit_segment(
	device: &Arc<Device>,
	nodes: &[ComputeNode],
	decoded: &[Option<NodeOp>],
	refs: &[&Storage],
	start: usize,
	end: usize,
	max_elements: u32,
) -> Result<FusedPlan, NodeOutcome> {
	if end + 1 - start < 2 {
		return Err(NodeOutcome::Isolated);
	}
	let ops: Vec<NodeOp> = decoded[start..=end]
		.iter()
		.copied()
		.collect::<Option<_>>()
		.ok_or(NodeOutcome::ProgramRejected)?;
	let element_count = ops[0].element_count;
	if ops.iter().any(|op| op.element_count != element_count) {
		return Err(NodeOutcome::ExtentBoundary);
	}
	let candidate = build_program(nodes, decoded, refs, start, end)?;
	if candidate
		.inputs
		.iter()
		.chain(&candidate.outputs)
		.any(|buffer| !buffer.belongs_to(device))
	{
		return Err(NodeOutcome::ForeignDevice);
	}

	let jit_enabled = device.jit_fusion_enabled();
	if element_count > max_elements && !jit_enabled {
		return Err(NodeOutcome::TapeBound);
	}
	let profile = device.physical().profile;
	let input_indices = candidate
		.inputs
		.iter()
		.enumerate()
		.map(|(slot, buffer)| {
			if profile == ExecutionProfile::Compatibility {
				slot as u32
			} else {
				buffer.descriptor_index()
			}
		})
		.collect::<Vec<_>>();
	let output_indices = candidate
		.outputs
		.iter()
		.enumerate()
		.map(|(slot, buffer)| {
			if profile == ExecutionProfile::Compatibility {
				(MAX_OPERAND_SLOTS + slot) as u32
			} else {
				buffer.descriptor_index()
			}
		})
		.collect::<Vec<_>>();
	let tape = candidate
		.program
		.tape(&input_indices, &output_indices)
		.ok_or(NodeOutcome::TapeCapacity)?;
	let mut bytes = Vec::with_capacity(tape.len() * 4);
	for word in &tape {
		bytes.extend_from_slice(&word.to_le_bytes());
	}
	let tape_storage = Storage::new_from_bytes(device, &bytes).map_err(|_| NodeOutcome::Resource)?;
	let tape_buffer = tape_storage.buffer().ok_or(NodeOutcome::Resource)?.clone();

	// Reject prefixes exceeding tape capacity before consuming compiler or
	// pipeline entries. The canonical fallback must be fully prepared first.
	// Opt-in specialization: admit the discovered program into the device's
	// jit table so recording dispatches the compiled pipeline. The tape
	// bindings stay canonical — a compile, admission or lookup failure keeps
	// this node's tape route unchanged. Above `max_elements` the tape
	// interpreter loses to the unfused kernels, so the fused node exists
	// only when the specialized route is admitted there. The observation
	// signature is attached whenever the route is enabled, including on tape
	// probes, so recording can measure whichever route runs.
	let tape_eligible = element_count <= max_elements;
	let jit = device
		.jit_admit(&candidate.program, tape_eligible)
		.map(|key| JitNode {
			key,
			buffers: candidate
				.inputs
				.iter()
				.map(|buffer| BufferUse {
					buffer: buffer.clone(),
					access: BufferAccess::Read,
				})
				.chain(candidate.outputs.iter().map(|buffer| BufferUse {
					buffer: buffer.clone(),
					access: BufferAccess::Write,
				}))
				.collect(),
			push_constants: vec![PushConstant::U32(element_count)],
		});
	if element_count > max_elements && jit.is_none() {
		return Err(NodeOutcome::SpecializationDeclined);
	}
	let fusion_signature = jit_enabled.then(|| {
		crate::runtime::device::vkcache::jit::signature_key(&candidate.program.canonical_key())
	});
	let operands = candidate.inputs;
	let stores = candidate.outputs;

	let first_operand = operands
		.first()
		.ok_or(NodeOutcome::ProgramRejected)?
		.clone();
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
	Ok(FusedPlan {
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
			jit,
			fusion_signature,
		},
		tape: tape_storage,
	})
}

// Donor composition: stable softmax, repeated argmax and membership-mask
// formulas retain their source order in the shader. Every source output is
// materialized, including top-k values and probabilities saved by autograd.
// Keep larger expert/route geometries on source until measured qualification.
fn match_router_select(
	softmax: &ComputeNode,
	top_k: &ComputeNode,
	mask: &ComputeNode,
) -> Option<ComputeNode> {
	let source = [
		(softmax, KernelId::MatrixSoftmaxF32, 1, 1),
		(top_k, KernelId::MatrixTopKF32, 1, 2),
		(mask, KernelId::MatrixTopKMaskF32, 1, 1),
	];
	for (node, kernel, reads, writes) in source {
		if node.kernel != kernel
			|| node.indirect.is_some()
			|| node.buffers.len() != reads + writes
			|| node.push_constants.len() != 3
			|| node.buffers[..reads]
				.iter()
				.any(|b| b.access != BufferAccess::Read)
			|| node.buffers[reads..]
				.iter()
				.any(|b| b.access != BufferAccess::Write)
		{
			return None;
		}
	}
	let [
		PushConstant::U32(rows),
		PushConstant::U32(experts),
		PushConstant::U32(k),
	] = top_k.push_constants[..]
	else {
		return None;
	};
	if rows == 0
		|| experts == 0
		|| experts > 256
		|| k == 0
		|| k > experts
		|| k > 32
		|| softmax.push_constants
			!= [
				PushConstant::U32(rows),
				PushConstant::U32(experts),
				PushConstant::U32(1),
			] || mask.push_constants != top_k.push_constants
		|| softmax.workgroups != [rows, 1, 1]
		|| top_k.workgroups != [rows, 1, 1]
		|| !mask.buffers[0].buffer.same_as(&top_k.buffers[2].buffer)
	{
		return None;
	}
	let count = rows.checked_mul(experts)?;
	if mask.workgroups != KernelId::MatrixTopKMaskF32.linear_workgroups(count) {
		return None;
	}
	let buffers = vec![
		softmax.buffers[0].clone(),
		top_k.buffers[0].clone(),
		softmax.buffers[1].clone(),
		top_k.buffers[1].clone(),
		top_k.buffers[2].clone(),
		mask.buffers[1].clone(),
	];
	let row_bytes = f32_bytes(count)?;
	let route_bytes = f32_bytes(rows.checked_mul(k)?)?;
	for (index, buffer) in buffers.iter().enumerate() {
		let required = if index == 3 || index == 4 {
			route_bytes
		} else {
			row_bytes
		};
		if buffer.buffer.byte_len() != required {
			return None;
		}
		if index >= 2
			&& buffers[..index]
				.iter()
				.any(|other| other.buffer.same_as(&buffer.buffer))
		{
			return None;
		}
	}
	let semantic_ops = merged_semantic_ops(&[
		&softmax.semantic_ops,
		&top_k.semantic_ops,
		&mask.semantic_ops,
	]);
	let kernel = KernelId::MlMoeRouterSelectF32;
	let (operation, op_contract_hash) = fused_identity(&semantic_ops, softmax, kernel);
	Some(ComputeNode {
		operation,
		kernel,
		buffers,
		push_constants: top_k.push_constants.clone(),
		workgroups: [rows, 1, 1],
		indirect: None,
		semantic_ops,
		op_contract_hash,
		jit: None,
		fusion_signature: None,
	})
}

// The balancing broadcast is independent of softmax. Move it before the
// combined router only when neither node can observe the other's writes.
// It keeps its original ABI, semantic owner and materialized output.
fn router_bias_can_precede_softmax(
	softmax: &ComputeNode,
	add: &ComputeNode,
	top_k: &ComputeNode,
) -> bool {
	if !matches!(
		add.kernel,
		KernelId::MatrixAddF32 | KernelId::MatrixAddBroadcastF32
	) || add.indirect.is_some()
		|| add.buffers.len() != 3
		|| add.buffers[..2]
			.iter()
			.any(|b| b.access != BufferAccess::Read)
		|| add.buffers[2].access != BufferAccess::Write
		|| !add.buffers[2].buffer.same_as(&top_k.buffers[0].buffer)
	{
		return false;
	}
	let probabilities = &softmax.buffers[1].buffer;
	!add.buffers.iter().any(|b| b.buffer.same_as(probabilities))
		&& !add.buffers[2].buffer.same_as(&softmax.buffers[0].buffer)
		&& !add.buffers[..2]
			.iter()
			.any(|b| b.buffer.same_as(&add.buffers[2].buffer))
}

// Rank-two contiguous concat into Linear. Source concat regions share one
// semantic owner; retain their materialized output and all Linear arithmetic.
fn match_concat_linear(parts: &[ComputeNode], linear: &ComputeNode) -> Option<ComputeNode> {
	if !(2..=3).contains(&parts.len()) {
		return None;
	}
	let (_, batch, outputs) = match_linear_producer(linear)?;
	let PushConstant::U32(inner) = linear.push_constants[1] else {
		return None;
	};
	let concat = &linear.buffers[0].buffer;
	let mut widths = [0u32; 3];
	let mut offset = 0u32;
	for (i, part) in parts.iter().enumerate() {
		if part.kernel != KernelId::MatrixConcatF32
			|| part.indirect.is_some()
			|| part.buffers.len() != 2
			|| part.push_constants.len() != 18
			|| part.buffers[0].access != BufferAccess::Read
			|| part.buffers[1].access != BufferAccess::Write
			|| !part.buffers[1].buffer.same_as(concat)
			|| part.semantic_ops != parts[0].semantic_ops
		{
			return None;
		}
		let words = part
			.push_constants
			.iter()
			.map(|p| {
				if let PushConstant::U32(v) = p {
					Some(*v)
				} else {
					None
				}
			})
			.collect::<Option<Vec<_>>>()?;
		let width = words[7];
		let expected = [
			batch.checked_mul(width)?,
			2,
			1,
			0,
			offset,
			4,
			batch,
			width,
			0,
			0,
			batch,
			inner,
			0,
			0,
			batch,
			width,
			0,
			0,
		];
		if words != expected
			|| width == 0
			|| part.buffers[0].buffer.byte_len() != f32_bytes(batch.checked_mul(width)?)?
		{
			return None;
		}
		widths[i] = width;
		offset = offset.checked_add(width)?;
	}
	if offset != inner || inner == 0 || batch == 0 || outputs == 0 {
		return None;
	}
	let third = parts.get(2).unwrap_or(&parts[0]);
	let buffers = vec![
		parts[0].buffers[0].clone(),
		parts[1].buffers[0].clone(),
		third.buffers[0].clone(),
		linear.buffers[1].clone(),
		linear.buffers[2].clone(),
		parts[0].buffers[1].clone(),
		linear.buffers[3].clone(),
	];
	if !distinct_outputs(&buffers, 5) {
		return None;
	}
	let mut owners = parts
		.iter()
		.map(|p| p.semantic_ops.as_slice())
		.collect::<Vec<_>>();
	owners.push(&linear.semantic_ops);
	let semantic_ops = merged_semantic_ops(&owners);
	let kernel = KernelId::MlLinearConcatF32;
	let (operation, op_contract_hash) = fused_identity(&semantic_ops, &parts[0], kernel);
	Some(ComputeNode {
		operation,
		kernel,
		buffers,
		push_constants: [batch, inner, outputs, widths[0], widths[1], widths[2]]
			.map(PushConstant::U32)
			.to_vec(),
		workgroups: linear.workgroups,
		indirect: None,
		semantic_ops,
		op_contract_hash,
		jit: None,
		fusion_signature: None,
	})
}

fn distinct_outputs(buffers: &[BufferUse], first_write: usize) -> bool {
	buffers.iter().enumerate().skip(first_write).all(|(i, b)| {
		!buffers[..i]
			.iter()
			.any(|other| other.buffer.same_as(&b.buffer))
	})
}

// Full packed causal Standard-attention region. Dropout, additive masks,
// unsupported sizes and aliases retain source work.
fn match_small_causal_sdpa(nodes: &[ComputeNode]) -> Option<ComputeNode> {
	let [q, k, v, score, probability, context, merge] = nodes else {
		return None;
	};
	for split in [q, k, v] {
		if split.kernel != KernelId::MlSplitHeadsF32
			|| split.indirect.is_some()
			|| split.buffers.len() != 2
			|| split.push_constants.len() != 4
			|| split.buffers[0].access != BufferAccess::Read
			|| split.buffers[1].access != BufferAccess::Write
			|| split.push_constants != q.push_constants
		{
			return None;
		}
	}
	let [
		PushConstant::U32(batch),
		PushConstant::U32(sequence),
		PushConstant::U32(heads),
		PushConstant::U32(dim),
	] = q.push_constants[..]
	else {
		return None;
	};
	if batch == 0
		|| heads == 0
		|| dim == 0
		|| dim > 128
		|| sequence == 0
		|| sequence > 128
		|| sequence.checked_mul(dim)? > 2048
	{
		return None;
	}
	let bh = batch.checked_mul(heads)?;
	for (node, kernel, reads) in [
		(score, score.kernel, 2),
		(context, context.kernel, 2),
		(probability, probability.kernel, 2),
		(merge, KernelId::MlMergeHeadsF32, 1),
	] {
		if node.kernel != kernel
			|| node.indirect.is_some()
			|| node.buffers.len() != reads + 1
			|| node.buffers[..reads]
				.iter()
				.any(|b| b.access != BufferAccess::Read)
			|| node.buffers[reads].access != BufferAccess::Write
		{
			return None;
		}
	}
	if !matches!(
		score.kernel,
		KernelId::MlBmmNtF32 | KernelId::MlBmmNtTiled16F32
	) || !matches!(
		context.kernel,
		KernelId::MlBmmF32 | KernelId::MlBmmTiled16F32
	) || !matches!(
		probability.kernel,
		KernelId::MlSdpaSoftmaxN32F32 | KernelId::MlSdpaSoftmaxF32
	) || score.push_constants != [bh, sequence, dim, sequence].map(PushConstant::U32)
		|| context.push_constants != [bh, sequence, sequence, dim].map(PushConstant::U32)
		|| merge.push_constants != q.push_constants
	{
		return None;
	}
	let [
		PushConstant::U32(rows),
		PushConstant::U32(columns),
		PushConstant::F32(scale),
		PushConstant::U32(0),
		PushConstant::U32(1),
	] = probability.push_constants[..]
	else {
		return None;
	};
	if rows != bh.checked_mul(sequence)?
		|| columns != sequence
		|| !scale.is_finite()
		|| !score.buffers[0].buffer.same_as(&q.buffers[1].buffer)
		|| !score.buffers[1].buffer.same_as(&k.buffers[1].buffer)
		|| !probability.buffers[0]
			.buffer
			.same_as(&score.buffers[2].buffer)
		|| !probability.buffers[1]
			.buffer
			.same_as(&score.buffers[2].buffer)
		|| !context.buffers[0]
			.buffer
			.same_as(&probability.buffers[2].buffer)
		|| !context.buffers[1].buffer.same_as(&v.buffers[1].buffer)
		|| !merge.buffers[0].buffer.same_as(&context.buffers[2].buffer)
	{
		return None;
	}
	let buffers = vec![
		q.buffers[0].clone(),
		k.buffers[0].clone(),
		v.buffers[0].clone(),
		q.buffers[1].clone(),
		k.buffers[1].clone(),
		v.buffers[1].clone(),
		score.buffers[2].clone(),
		probability.buffers[2].clone(),
		context.buffers[2].clone(),
		merge.buffers[1].clone(),
	];
	if !distinct_outputs(&buffers, 3) {
		return None;
	}
	let packed_bytes = f32_bytes(bh.checked_mul(sequence)?.checked_mul(dim)?)?;
	let score_bytes = f32_bytes(rows.checked_mul(sequence)?)?;
	for (i, b) in buffers.iter().enumerate() {
		if b.buffer.byte_len()
			!= if i == 6 || i == 7 {
				score_bytes
			} else {
				packed_bytes
			} {
			return None;
		}
	}
	let owners = nodes
		.iter()
		.map(|n| n.semantic_ops.as_slice())
		.collect::<Vec<_>>();
	let semantic_ops = merged_semantic_ops(&owners);
	let kernel = KernelId::MlSmallCausalSdpaF32;
	let (operation, op_contract_hash) = fused_identity(&semantic_ops, q, kernel);
	Some(ComputeNode {
		operation,
		kernel,
		buffers,
		push_constants: vec![
			PushConstant::U32(batch),
			PushConstant::U32(sequence),
			PushConstant::U32(heads),
			PushConstant::U32(dim),
			PushConstant::F32(scale),
		],
		workgroups: [bh, 1, 1],
		indirect: None,
		semantic_ops,
		op_contract_hash,
		jit: None,
		fusion_signature: None,
	})
}

// Single-head causal attention region. With one head, split/merge head
// transforms lower to reshapes and only the three SDPA dispatches remain:
// score BMM (NT), causal softmax, context BMM (NN). The packed kernel's
// per-head indexing collapses to the flat [B*S, D] layout, so the source
// Q/K/V buffers stand in for the split slots and the context buffer is
// also the merged output. The kernel writes at those aliased slots
// reproduce the loaded values at identical addresses.
fn match_single_head_sdpa(nodes: &[ComputeNode]) -> Option<ComputeNode> {
	let [score, probability, context] = nodes else {
		return None;
	};
	for node in [score, probability, context] {
		if node.indirect.is_some()
			|| node.buffers.len() != 3
			|| node.buffers[..2]
				.iter()
				.any(|b| b.access != BufferAccess::Read)
			|| node.buffers[2].access != BufferAccess::Write
		{
			return None;
		}
	}
	if !matches!(
		score.kernel,
		KernelId::MlBmmNtF32 | KernelId::MlBmmNtTiled16F32
	) || !matches!(
		context.kernel,
		KernelId::MlBmmF32 | KernelId::MlBmmTiled16F32
	) || !matches!(
		probability.kernel,
		KernelId::MlSdpaSoftmaxN32F32 | KernelId::MlSdpaSoftmaxF32
	) {
		return None;
	}
	let [
		PushConstant::U32(batch),
		PushConstant::U32(sequence),
		PushConstant::U32(dim),
		PushConstant::U32(score_columns),
	] = score.push_constants[..]
	else {
		return None;
	};
	if batch == 0
		|| dim == 0
		|| dim > 128
		|| sequence == 0
		|| sequence > 128
		|| score_columns != sequence
		|| sequence.checked_mul(dim)? > 2048
		|| context.push_constants != [batch, sequence, sequence, dim].map(PushConstant::U32)
	{
		return None;
	}
	let [
		PushConstant::U32(rows),
		PushConstant::U32(columns),
		PushConstant::F32(scale),
		PushConstant::U32(0),
		PushConstant::U32(1),
	] = probability.push_constants[..]
	else {
		return None;
	};
	if rows != batch.checked_mul(sequence)?
		|| columns != sequence
		|| !scale.is_finite()
		|| !probability.buffers[0]
			.buffer
			.same_as(&score.buffers[2].buffer)
		|| !probability.buffers[1]
			.buffer
			.same_as(&score.buffers[2].buffer)
		|| !context.buffers[0]
			.buffer
			.same_as(&probability.buffers[2].buffer)
	{
		return None;
	}
	let buffers = vec![
		score.buffers[0].clone(),
		score.buffers[1].clone(),
		context.buffers[1].clone(),
		BufferUse {
			buffer: score.buffers[0].buffer.clone(),
			access: BufferAccess::Write,
		},
		BufferUse {
			buffer: score.buffers[1].buffer.clone(),
			access: BufferAccess::Write,
		},
		BufferUse {
			buffer: context.buffers[1].buffer.clone(),
			access: BufferAccess::Write,
		},
		score.buffers[2].clone(),
		probability.buffers[2].clone(),
		context.buffers[2].clone(),
		context.buffers[2].clone(),
	];
	if buffers[6].buffer.same_as(&buffers[7].buffer)
		|| buffers[6].buffer.same_as(&buffers[8].buffer)
		|| buffers[7].buffer.same_as(&buffers[8].buffer)
		|| buffers[6..=8]
			.iter()
			.any(|b| buffers[..3].iter().any(|r| r.buffer.same_as(&b.buffer)))
	{
		return None;
	}
	let packed_bytes = f32_bytes(batch.checked_mul(sequence)?.checked_mul(dim)?)?;
	let score_bytes = f32_bytes(rows.checked_mul(sequence)?)?;
	for (i, b) in buffers.iter().enumerate() {
		if b.buffer.byte_len()
			!= if i == 6 || i == 7 {
				score_bytes
			} else {
				packed_bytes
			} {
			return None;
		}
	}
	let owners = nodes
		.iter()
		.map(|n| n.semantic_ops.as_slice())
		.collect::<Vec<_>>();
	let semantic_ops = merged_semantic_ops(&owners);
	let kernel = KernelId::MlSmallCausalSdpaF32;
	let (operation, op_contract_hash) = fused_identity(&semantic_ops, score, kernel);
	Some(ComputeNode {
		operation,
		kernel,
		buffers,
		push_constants: vec![
			PushConstant::U32(batch),
			PushConstant::U32(sequence),
			PushConstant::U32(1),
			PushConstant::U32(dim),
			PushConstant::F32(scale),
		],
		workgroups: [batch, 1, 1],
		indirect: None,
		semantic_ops,
		op_contract_hash,
		jit: None,
		fusion_signature: None,
	})
}

fn match_grouped_linear_act(linear: &ComputeNode, act: &ComputeNode) -> Option<ComputeNode> {
	if linear.kernel != KernelId::MlGroupedLinearMF32
		|| linear.indirect.is_some()
		|| linear.buffers.len() != 5
		|| linear.push_constants.len() != 4
		|| linear.buffers[..4]
			.iter()
			.any(|b| b.access != BufferAccess::Read)
		|| linear.buffers[4].access != BufferAccess::Write
		|| act.kernel != KernelId::MlSiluMulF32
		|| act.indirect.is_some()
		|| act.buffers.len() != 2
		|| act.buffers[0].access != BufferAccess::Read
		|| act.buffers[1].access != BufferAccess::Write
		|| !act.buffers[0].buffer.same_as(&linear.buffers[4].buffer)
	{
		return None;
	}
	let [
		PushConstant::U32(rows),
		PushConstant::U32(outputs),
		PushConstant::U32(inner),
		PushConstant::U32(experts),
	] = linear.push_constants[..]
	else {
		return None;
	};
	if rows == 0
		|| outputs == 0
		|| !outputs.is_multiple_of(2)
		|| inner == 0
		|| experts == 0
		|| act.push_constants
			!= [
				PushConstant::U32(rows.checked_mul(outputs / 2)?),
				PushConstant::U32(outputs / 2),
			]
	{
		return None;
	}
	let mut buffers = linear.buffers.clone();
	buffers.push(act.buffers[1].clone());
	if !distinct_outputs(&buffers, 4) {
		return None;
	}
	let mut push_constants = linear.push_constants.clone();
	push_constants.push(PushConstant::U32(9));
	Some(fused_node(
		linear,
		act,
		KernelId::MlGroupedLinearMActF32,
		buffers,
		push_constants,
		[rows.div_ceil(32), (outputs / 2).div_ceil(32), 1],
	))
}

fn f32_bytes(elements: u32) -> Option<usize> {
	usize::try_from(elements)
		.ok()?
		.checked_mul(std::mem::size_of::<f32>())
}
