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
mod region;
mod rules;

use census::NodeOutcome;

use crate::runtime::{
	Buffer, BufferAccess, Device, PushConstant, Storage,
	device::features::ExecutionProfile,
	dna::{
		ir::elementwise::{Instruction, MAX_OPERAND_SLOTS, MAX_STORE_SLOTS, Program, Source},
		kernel::{KernelId, KernelRegionRead, KernelScalarTail, KernelTapeShape},
	},
	graph::executable::{BufferUse, ComputeNode, JitNode},
	graph::semantic::SemanticOpId,
};

const BINDING_SLOTS: usize = MAX_OPERAND_SLOTS + MAX_STORE_SLOTS + 1;

thread_local! {
	static EPILOGUE_FUSION_DISABLE_DEPTH: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
	static SINGLE_HEAD_SDPA_FUSION_DEPTH: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
}

/// `#[doc(hidden)]` laboratory scope: while alive on the current thread,
/// epilogue pair fusion is declined as if `OA_DISABLE_EPILOGUE_FUSION`
/// were set — without mutating process-global environment state, so
/// parallel tests cannot leak the toggle into siblings.
pub struct EpilogueFusionDisabled(());

impl EpilogueFusionDisabled {
	#[must_use]
	pub fn enter() -> Self {
		EPILOGUE_FUSION_DISABLE_DEPTH.with(|depth| depth.set(depth.get() + 1));
		Self(())
	}
}

impl Drop for EpilogueFusionDisabled {
	fn drop(&mut self) {
		EPILOGUE_FUSION_DISABLE_DEPTH.with(|depth| depth.set(depth.get() - 1));
	}
}

/// `#[doc(hidden)]` laboratory scope: while alive on the current thread,
/// the single-head SDPA route is enabled as if
/// `OA_ENABLE_SINGLE_HEAD_SDPA_FUSION` were set — see above for why this
/// is thread-local.
pub struct SingleHeadSdpaFusionEnabled(());

impl SingleHeadSdpaFusionEnabled {
	#[must_use]
	pub fn enter() -> Self {
		SINGLE_HEAD_SDPA_FUSION_DEPTH.with(|depth| depth.set(depth.get() + 1));
		Self(())
	}
}

impl Drop for SingleHeadSdpaFusionEnabled {
	fn drop(&mut self) {
		SINGLE_HEAD_SDPA_FUSION_DEPTH.with(|depth| depth.set(depth.get() - 1));
	}
}

fn epilogue_fusion_disabled() -> bool {
	EPILOGUE_FUSION_DISABLE_DEPTH.with(|depth| depth.get() != 0)
		|| std::env::var_os("OA_DISABLE_EPILOGUE_FUSION").is_some()
}

/// Per-device route record for the single-head SDPA merge, seeded at device
/// creation. The route changed the canonical Byte-MoE trajectory on Xe and
/// was net-negative on Haswell, so every current class seeds `false`; the
/// env var and `SingleHeadSdpaFusionEnabled` stay as laboratory overrides.
pub(in crate::runtime) fn single_head_sdpa_seed(_info: &crate::DeviceInfo) -> bool {
	std::env::var_os("OA_ENABLE_SINGLE_HEAD_SDPA_FUSION").is_some()
}

fn single_head_sdpa_fusion_enabled(device: &Device) -> bool {
	SINGLE_HEAD_SDPA_FUSION_DEPTH.with(|depth| depth.get() != 0) || device.single_head_sdpa_route()
}

/// `OA_DISABLE_FUSION_RULES=name1,name2` suppresses the named pair/merge
/// rules for laboratory bisection — e.g. isolating which admitted rule
/// regresses a workload's device time. Names match `OA_JIT_TRACE` rule
/// diagnostics. Not a correctness gate: suppressed windows fall to the
/// remaining rules in priority order or the unfused decomposition, so
/// per-rule deltas are not additive across co-dependent rules.
fn fusion_rule_disabled(name: &str) -> bool {
	thread_local! {
		static DISABLED: std::cell::LazyCell<Vec<String>> = const {
			std::cell::LazyCell::new(|| {
				std::env::var("OA_DISABLE_FUSION_RULES")
					.map(|value| {
						value
							.split(',')
							.map(|name| name.trim().to_string())
							.filter(|name| !name.is_empty())
							.collect()
					})
					.unwrap_or_default()
			})
		};
	}
	DISABLED.with(|disabled| disabled.iter().any(|rule| rule == name))
}

/// Above this element count the tape interpreter loses to the unfused
/// per-op kernels: per-element decode, branch, and shared-register traffic
/// outweighs the eliminated memory passes. Measured on HSW GT2 (chain-8
/// sweep in `bench_core_elemwise_chain`): fused wins through ~32K elements,
/// ties near 64K, and loses clearly at 1M. Routing policy, not a capability
/// bound.
const MAX_FUSION_ELEMENTS: u32 = 32 * 1024;

/// Seed a device's tape-vs-per-op element bound. `OA_TAPE_BOUND_ELEMENTS`
/// overrides for laboratory measurement; otherwise the value is a function
/// of the queried device, not a global constant — each device carries its
/// own bound as a route record. The fallback is the HSW GT2 measurement;
/// device-class entries below are added only from sweep evidence on that
/// class.
pub(in crate::runtime) fn tape_bound_seed(_info: &crate::DeviceInfo) -> u32 {
	if let Ok(value) = std::env::var("OA_TAPE_BOUND_ELEMENTS")
		&& let Ok(bound) = value.parse::<u32>()
	{
		return bound;
	}
	MAX_FUSION_ELEMENTS
}

/// The fused-tape buffer/immediate shape a node presents — the
/// schema-declared `tape_shape` generated as `KernelTapeShape`
/// (OA-CC-34).
type ElementwiseShape = KernelTapeShape;

/// One recorded node decoded into the fused kernel's tape vocabulary.
#[derive(Clone, Copy)]
struct NodeOp {
	opcode: u32,
	shape: ElementwiseShape,
	element_count: u32,
	scalar: f32,
	extra: f32,
	// Broadcast shape plus two input strides, or gather {rows,width,dtype}.
	metadata: [u32; 25],
}

fn elementwise_op(kernel: KernelId) -> Option<(u32, ElementwiseShape)> {
	kernel.tape_op()
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
/// storage-buffer index, so Core-profile identity replacements (plan input
/// rebinding, alias arena materialization) must not swap them.
pub(in crate::runtime) fn buffer_is_fused_binding<'a>(
	mut nodes: impl Iterator<Item = &'a ComputeNode>,
	buffer: &Buffer,
) -> bool {
	nodes.any(|node| {
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
	let (count, operands) = node.push_constants.split_first()?;
	let operands = match (node.kernel, operands) {
		// The shared image pixel kernel has an operation selector and
		// four parameter slots. Check the selector before discarding it.
		(KernelId::ImageClampF32, [PushConstant::U32(6), _, _, _, _]) => &operands[1..3],
		(KernelId::ImageInvertF32, [PushConstant::U32(7), _, _, _, _]) => &operands[1..2],
		(KernelId::ImageClampF32 | KernelId::ImageInvertF32, _) => return None,
		(_, operands) => operands,
	};
	let operand_count = match shape {
		ElementwiseShape::Broadcast => 25,
		ElementwiseShape::Gather => 3,
		ElementwiseShape::ScalarPair => 2,
		ElementwiseShape::Scalar | ElementwiseShape::ScalarFirst | ElementwiseShape::BinaryScalar => 1,
		ElementwiseShape::Binary | ElementwiseShape::Unary => 0,
	};
	if operands.len() != operand_count {
		return None;
	}
	let element_count = match count {
		PushConstant::U32(count) if *count > 0 => *count,
		_ => return None,
	};
	let (scalar, extra) = match (shape, operands) {
		(ElementwiseShape::ScalarPair, [PushConstant::F32(min), PushConstant::F32(max)])
			if min.is_finite() && max.is_finite() && min <= max =>
		{
			(*min, *max)
		}
		(
			ElementwiseShape::Scalar | ElementwiseShape::ScalarFirst | ElementwiseShape::BinaryScalar,
			[PushConstant::F32(value)],
		) => (*value, 0.0),
		(
			ElementwiseShape::ScalarPair
			| ElementwiseShape::Scalar
			| ElementwiseShape::ScalarFirst
			| ElementwiseShape::BinaryScalar,
			_,
		) => return None,
		(_, _) => (0.0, 0.0),
	};
	let mut metadata = [0u32; 25];
	if matches!(
		shape,
		ElementwiseShape::Broadcast | ElementwiseShape::Gather
	) {
		for (word, push) in metadata.iter_mut().zip(operands) {
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
		extra,
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
	if epilogue_fusion_disabled() {
		return nodes;
	}
	rules::fuse(device, nodes, refs)
}

/// Decode the common `MlLinearF32` producer, or `None` when the node lies
/// outside the fused kernel's buffer and scalar vocabulary. Returns the
/// linear output buffer plus `(batch, output_features)` for coverage
/// checks against the consumer node.
fn linear_geometry(linear: &ComputeNode) -> Option<(&Buffer, u32, u32)> {
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

/// Pack the validated linear residual bindings and publish both writes.
fn lower_linear_residual(linear: &ComputeNode, add: &ComputeNode) -> ComputeNode {
	let linear_output = &linear.buffers[3].buffer;
	let residual = &add.buffers[usize::from(add.buffers[0].buffer.same_as(linear_output))].buffer;
	let sum = &add.buffers[2].buffer;
	let kernel = KernelId::MlLinearResidualF32;
	fused_node(
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
	)
}

/// Decode three adjacent `MlLinearF32` nodes sharing one input and one
/// geometry — the Q/K/V projection group — into a single z-dispatched node.
/// Outputs must be three distinct buffers that alias none of the reads.
fn lower_linear_triple(
	first: &ComputeNode,
	second: &ComputeNode,
	third: &ComputeNode,
) -> Option<ComputeNode> {
	let (first_output, _, _) = linear_geometry(first)?;
	let (second_output, _, _) = linear_geometry(second)?;
	let (third_output, _, _) = linear_geometry(third)?;
	if first_output.same_as(second_output)
		|| second_output.same_as(third_output)
		|| first_output.same_as(third_output)
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
fn embedding_output(node: &ComputeNode) -> (&Buffer, bool) {
	(
		&node.buffers[2].buffer,
		node.kernel == KernelId::MlEmbeddingU8F32,
	)
}

/// Decode the `embedding → embedding → add` token/position head into one
/// fused gather pair that writes both gathered tensors and their sum. Both
/// gathers must share index count and embedding width so the sum covers the
/// full element domain; every write lands on a distinct buffer aliasing none
/// of the reads.
fn lower_embedding_pair(
	first: &ComputeNode,
	second: &ComputeNode,
	add: &ComputeNode,
) -> Option<ComputeNode> {
	let (first_output, first_u8) = embedding_output(first);
	let (second_output, second_u8) = embedding_output(second);
	let sum = &add.buffers[2].buffer;
	if first_output.same_as(second_output) {
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
fn lower_embedding_pair_norm(
	first: &ComputeNode,
	second: &ComputeNode,
	add: &ComputeNode,
	norm: &ComputeNode,
) -> Option<ComputeNode> {
	let (first_output, first_u8) = embedding_output(first);
	let (second_output, second_u8) = embedding_output(second);
	let sum = &add.buffers[2].buffer;
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
fn lower_sum_axis_scale(sum: &ComputeNode, scale: &ComputeNode) -> Option<ComputeNode> {
	if !matches!(scale.push_constants[1], PushConstant::F32(_)) {
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
	kernel.act_kind()
}

/// Pack the validated residual norm bindings while retaining the sum and
/// every normalization output.
fn lower_add_layer_norm(add: &ComputeNode, norm: &ComputeNode) -> ComputeNode {
	let sum = &add.buffers[2].buffer;
	let kernel = KernelId::MlResidualLayerNormF32;
	fused_node(
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
	)
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
fn lower_linear_residual_norm(
	linear: &ComputeNode,
	add: &ComputeNode,
	norm: &ComputeNode,
) -> Option<ComputeNode> {
	let (linear_output, batch, output_features) = linear_geometry(linear)?;
	let PushConstant::U32(input_features) = linear.push_constants[1] else {
		return None;
	};
	let residual = &add.buffers[usize::from(add.buffers[0].buffer.same_as(linear_output))].buffer;
	let sum = &add.buffers[2].buffer;
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

/// Decode the `linear → add → activation` triple, or `None` when any node
/// lies outside the fused kernel's buffer and scalar vocabulary. The
/// activation must consume the residual sum elementwise over the full tile;
/// gated `silu_mul` stays with the pair route because its packed operands do
/// not follow a residual combine. All three source writes (linear output,
/// sum, activation output) stay materialized for autograd saves.
fn lower_linear_residual_act(
	linear: &ComputeNode,
	add: &ComputeNode,
	act: &ComputeNode,
) -> Option<ComputeNode> {
	let (linear_output, batch, output_features) = linear_geometry(linear)?;
	let PushConstant::U32(input_features) = linear.push_constants[1] else {
		return None;
	};
	let kind = act_kind(act.kernel)?;
	let residual = &add.buffers[usize::from(add.buffers[0].buffer.same_as(linear_output))].buffer;
	let sum = &add.buffers[2].buffer;
	let (elements, alpha) = match act.push_constants.as_slice() {
		[PushConstant::U32(elements)] => (*elements, 0.0),
		[PushConstant::U32(elements), PushConstant::F32(alpha)] => (*elements, *alpha),
		_ => return None,
	};
	if elements != batch.checked_mul(output_features)? {
		return None;
	}
	let act_output = &act.buffers[1].buffer;
	let reads = [
		&linear.buffers[0].buffer,
		&linear.buffers[1].buffer,
		&linear.buffers[2].buffer,
		residual,
	];
	let writes = [linear_output, sum, act_output];
	if writes
		.iter()
		.any(|write| reads.iter().any(|read| read.same_as(write)))
		|| (0..writes.len())
			.any(|left| (left + 1..writes.len()).any(|right| writes[left].same_as(writes[right])))
	{
		return None;
	}
	let kernel = KernelId::MlLinearResidualActF32;
	let semantic_ops =
		merged_semantic_ops(&[&linear.semantic_ops, &add.semantic_ops, &act.semantic_ops]);
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
			BufferUse {
				buffer: linear_output.clone(),
				access: BufferAccess::Write,
			},
			BufferUse {
				buffer: sum.clone(),
				access: BufferAccess::Write,
			},
			BufferUse {
				buffer: act_output.clone(),
				access: BufferAccess::Write,
			},
		],
		push_constants: vec![
			PushConstant::U32(batch),
			PushConstant::U32(input_features),
			PushConstant::U32(output_features),
			PushConstant::U32(kind),
			PushConstant::F32(alpha),
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
fn lower_linear_act(linear: &ComputeNode, act: &ComputeNode) -> Option<ComputeNode> {
	let (linear_output, batch, output_features) = linear_geometry(linear)?;
	let gated = act.kernel == KernelId::MlSiluMulF32;
	if gated && std::env::var_os("OA_DISABLE_RL_FUSION").is_some() {
		return None;
	}
	let kind = if gated { 9 } else { act_kind(act.kernel)? };
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
	kernel.act_kind()
}

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
			let kind = producer.kernel.prologue_kind()?;
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
			Some((producer.kernel.prologue_kind()?, matrix, splat, 0.0))
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
fn lower_linear_backward_prologue(
	producer: &ComputeNode,
	backward: &ComputeNode,
) -> Option<ComputeNode> {
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
		(producer.kernel.prologue_kind()?, 1, 0, 0.0)
	} else {
		backward_pointwise_prologue(producer, batch, output_features)?
	};
	let dy = &producer.buffers[2].buffer;
	let dx = &backward.buffers[2].buffer;
	if dy.same_as(dx) {
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
fn lower_grouped_backward_prologue(
	producer: &ComputeNode,
	backward: &ComputeNode,
) -> Option<ComputeNode> {
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
		(producer.kernel.prologue_kind()?, 1, 0, 0.0)
	} else {
		backward_pointwise_prologue(producer, rows, output_features)?
	};
	let dy = &producer.buffers[2].buffer;
	let dx = &backward.buffers[3].buffer;
	if dy.same_as(dx) {
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
		// The fused kernel walks column tiles and expert segments inside
		// one workgroup per row tile; a 2-D grid would recompute the
		// prologue per column tile.
		[rows.div_ceil(32), 1, 1],
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
fn lower_linear_backward_norm(
	norm: &ComputeNode,
	reduction: &ComputeNode,
	backward: &ComputeNode,
) -> Option<(ComputeNode, ComputeNode)> {
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
fn lower_linear_backward_norm_add(
	norm: &ComputeNode,
	reduction: &ComputeNode,
	combine: &ComputeNode,
	backward: &ComputeNode,
) -> Option<(ComputeNode, ComputeNode)> {
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
	// writes must stay clear of every fused operand on the one fused
	// dispatch. A combine buffer touching `dX` is the region proof's
	// ordering decline, not a matcher check.
	if reduction.buffers.iter().skip(2).any(|usage| {
		reads.iter().any(|read| read.same_as(&usage.buffer))
			|| writes.iter().any(|write| write.same_as(&usage.buffer))
	}) {
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
fn lower_layer_norm_backward_sum(
	nodes: &[ComputeNode],
) -> Option<(ComputeNode, Vec<usize>, usize)> {
	let first = &nodes[0];
	let first_output = &first.buffers[2].buffer;
	let norm_index = (1..=MAX_NORM_BACKWARD_SUM_LOOKAHEAD).find(|&distance| {
		distance < nodes.len() && nodes[distance].kernel == KernelId::MlLayerNormBackwardF32
	})?;
	let norm = &nodes[norm_index];
	if !rules::NORM_BACKWARD.matches(norm) {
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
		if !rules::ADD.matches(outer)
			|| !outer
				.buffers
				.iter()
				.take(2)
				.any(|usage| usage.buffer.same_as(first_output))
		{
			return None;
		}
		// A third-party reader of the elided inner output is the region
		// proof's materialization decline, not a matcher scan.
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
	// Nodes between the absorbed slots — producers of the third addend,
	// or anything else in the lookahead gap — keep running in place; the
	// region proof declines any that mutate a consumed operand or touch
	// the moved writes.
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
		// The scatter rule builds the proof's source list in absorbed
		// order; the emission claims owners in `[first, outer, norm]` order
		// and `outer_index < norm_index`, so keep the list ascending.
		consumed.sort_unstable();
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
fn lower_embedding_backward_sum(nodes: &[ComputeNode]) -> Option<Vec<(usize, ComputeNode)>> {
	let add = &nodes[0];
	let sum = &add.buffers[2].buffer;
	let addend_a = &add.buffers[0].buffer;
	let addend_b = &add.buffers[1].buffer;
	let mut emissions = Vec::new();
	for (distance, consumer) in nodes.iter().enumerate().skip(1) {
		if !rules::EMBEDDING_BACKWARD.matches(consumer) || !consumer.buffers[1].buffer.same_as(sum) {
			// Anything else touching the sum is not absorbed; the region
			// proof declines the window it poisons.
			continue;
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

/// Decode the Conv1d tail `[MlConv1dBiasAddF32, MlConv1dLayoutTransposeF32]`
/// plus an eligible direct consumer of the transposed output into one
/// dispatch, or the pair alone when the tail ends there. The bias add
/// updates the projected GEMM buffer in place, so every fused shape must
/// leave it raw: no later node may touch that intermediate or it would
/// read a different value than the unfused chain produced. Variants that
/// drop the transposed output buffer additionally require it to have no
/// other readers; the channel-norm variants republish it because the
/// backward adjoint consumes it as saved state. External ownership and
/// suffix-reader materialization are the region proof's, not the matcher's:
/// without that proof the tail degrades to the shape that still
/// materializes it, or declines.
fn lower_conv_1d_bias_tail(nodes: &[ComputeNode]) -> Option<(ComputeNode, usize)> {
	let add = &nodes[0];
	let transpose = &nodes[1];
	// The add covers the full [batch * rows, columns] tile the transpose
	// reshapes: `rows` is the output length T and `columns` is channels C.
	let (PushConstant::U32(batch), PushConstant::U32(rows), PushConstant::U32(columns)) = (
		transpose.push_constants[0],
		transpose.push_constants[1],
		transpose.push_constants[2],
	) else {
		return None;
	};
	let projected = &add.buffers[0].buffer;
	let conv_out = &transpose.buffers[1].buffer;
	if nodes.len() > 2 {
		let consumer = &nodes[2];
		if rules::RELU.matches(consumer)
			&& consumer.buffers[0].buffer.same_as(conv_out)
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
				let residual = if rules::ADD.matches(sum_node)
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
					{
						let kernel = KernelId::MlConv1dBiasLayoutTransposeResidualReluF32;
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
			{
				let kernel = KernelId::MlConv1dBiasLayoutTransposeReluF32;
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
		if rules::CHANNEL_NORM.matches(consumer)
			&& consumer.buffers[0].buffer.same_as(conv_out)
			&& !consumer.buffers[3].buffer.same_as(projected)
			&& !consumer.buffers[3].buffer.same_as(&add.buffers[1].buffer)
			&& consumer.push_constants[0] == PushConstant::U32(batch)
			&& consumer.push_constants[1] == PushConstant::U32(columns)
			&& consumer.push_constants[2] == PushConstant::U32(rows)
			&& matches!(consumer.push_constants[3], PushConstant::F32(_))
		{
			let kernel = if consumer.kernel == KernelId::MlChannelNormReluF32 {
				KernelId::MlConv1dBiasLayoutTransposeNormReluF32
			} else {
				KernelId::MlConv1dBiasLayoutTransposeNormF32
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
		if rules::TRANSPOSE.matches(consumer)
			&& consumer.buffers[0].buffer.same_as(conv_out)
			&& !consumer.buffers[1].buffer.same_as(projected)
			&& !consumer.buffers[1].buffer.same_as(&add.buffers[1].buffer)
			// The generic transpose must be the exact inverse of the NCL
			// transpose: same batch, and its row/column extents swapped.
			&& consumer.push_constants[0] == PushConstant::U32(batch)
			&& consumer.push_constants[1] == PushConstant::U32(columns)
			&& consumer.push_constants[2] == PushConstant::U32(rows)
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
	let fused = fused_node(
		add,
		transpose,
		KernelId::MlConv1dBiasLayoutTransposeF32,
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
fn lower_conv_2d_relu(nodes: &[ComputeNode]) -> Option<ComputeNode> {
	let conv = &nodes[0];
	let relu = &nodes[1];
	if conv
		.buffers
		.iter()
		.take(3)
		.any(|usage| relu.buffers[1].buffer.same_as(&usage.buffer))
	{
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
fn lower_max_pool_relu_backward(nodes: &[ComputeNode]) -> Option<ComputeNode> {
	let pool = &nodes[0];
	let relu = &nodes[1];
	if relu.buffers[2].buffer.same_as(&pool.buffers[0].buffer)
		|| relu.buffers[2].buffer.same_as(&pool.buffers[1].buffer)
		|| relu.buffers[2].buffer.same_as(&relu.buffers[0].buffer)
	{
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
fn lower_linear_backward_bmm_chain(nodes: &[ComputeNode]) -> Option<(ComputeNode, usize)> {
	let producer = &nodes[0];
	let kind = match producer.kernel {
		KernelId::MlBmmTiled16F32 => 0,
		KernelId::MlBmmTnTiled16F32 => 1,
		_ => return None,
	};
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
	for (distance, consumer) in nodes.iter().enumerate().skip(1) {
		if consumer.kernel != KernelId::MlLinearBackwardF32
			|| consumer.buffers.len() != 3
			|| !consumer.buffers[1].buffer.same_as(dy)
		{
			continue;
		}
		if !rules::LINEAR_BACKWARD.matches(consumer) {
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
		// Intervening nodes may freely read the operands and the bmm result;
		// a writer of any consumed buffer — and any earlier reader of `dx`,
		// which the fused node publishes at the producer's slot — is the
		// region proof's decline, not a matcher check.
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
	fuse_elementwise_with_limit(device, nodes, refs, device.tape_bound_elements())
}

#[cfg(test)]
#[path = "../../../../../test/rs/runtime/jit_fusion_unit.rs"]
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
		|| crate::runtime::dna::tuning::jit_route_override()
			== Some(crate::runtime::dna::tuning::JitRouteOverride::Unfused)
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
		match op.shape {
			ElementwiseShape::Scalar | ElementwiseShape::ScalarPair => {
				sources.push(Source::Constant(op.scalar.to_bits()));
			}
			ElementwiseShape::ScalarFirst => {
				sources.insert(0, Source::Constant(op.scalar.to_bits()));
			}
			_ => {}
		}
		let parameters = match op.shape {
			ElementwiseShape::Gather => op.metadata[..3].to_vec(),
			ElementwiseShape::BinaryScalar => vec![op.scalar.to_bits()],
			ElementwiseShape::ScalarPair => vec![op.extra.to_bits()],
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
		crate::runtime::device::logical::vkcache::jit::signature_key(&candidate.program.canonical_key())
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
fn lower_router_select(
	softmax: &ComputeNode,
	top_k: &ComputeNode,
	mask: &ComputeNode,
) -> Option<ComputeNode> {
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
			] || softmax.workgroups != [rows, 1, 1]
		|| top_k.workgroups != [rows, 1, 1]
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

// Rank-two contiguous concat into Linear. Source concat regions share one
// semantic owner; retain their materialized output and all Linear arithmetic.
fn lower_concat_linear(parts: &[ComputeNode], linear: &ComputeNode) -> Option<ComputeNode> {
	let (_, batch, outputs) = linear_geometry(linear)?;
	let PushConstant::U32(inner) = linear.push_constants[1] else {
		return None;
	};
	let mut widths = [0u32; 3];
	let mut offset = 0u32;
	for (i, part) in parts.iter().enumerate() {
		if part.semantic_ops != parts[0].semantic_ops {
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
fn lower_small_causal_sdpa(nodes: &[ComputeNode]) -> Option<ComputeNode> {
	let [q, k, v, score, probability, context, merge] = nodes else {
		return None;
	};
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
	if score.push_constants != [bh, sequence, dim, sequence].map(PushConstant::U32)
		|| context.push_constants != [bh, sequence, sequence, dim].map(PushConstant::U32)
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
	if rows != bh.checked_mul(sequence)? || columns != sequence || !scale.is_finite() {
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
fn lower_single_head_sdpa(nodes: &[ComputeNode]) -> Option<ComputeNode> {
	let [score, probability, context] = nodes else {
		return None;
	};
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
	if rows != batch.checked_mul(sequence)? || columns != sequence || !scale.is_finite() {
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

fn lower_grouped_linear_act(linear: &ComputeNode, act: &ComputeNode) -> Option<ComputeNode> {
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

/// The tile gather kernels whose `BlockProgram` factories accept a
/// scalar store tail — the image geometric family (OA-CC-33).
fn tile_scalar_tail_program(
	node: &ComputeNode,
) -> Option<crate::runtime::dna::ir::tile::BlockProgram> {
	node.jit.as_ref()?;
	if node.kernel == KernelId::AudioMelFilterbankF32 && node.semantic_ops.len() > 1 {
		return audio_mel_chain_program(node.push_constants.as_slice());
	}
	node.kernel.tile_program()
}

/// The tile consumers whose input binding is read only through
/// elementwise `LoadF32` — flip/crop/rotate permutations and the
/// channel-normalize reduction. A pointwise prologue commutes with a
/// permutation or a plain element read, so folding the producer into
/// the load is exact. Interpolating gathers (`resize`, `warp`,
/// `remap`) and bordered fetches (`pad`) read the input through typed
/// helpers where the fold would need to reach inside each tap —
/// deferred.
fn tile_scalar_prologue_elementwise(kernel: KernelId) -> bool {
	kernel.region_read() == KernelRegionRead::Elementwise
}

/// The admit route matching a tail-admissible tile kernel — the image
/// and audio gates stay separate per domain.
fn tile_scalar_tail_admit(
	device: &Arc<Device>,
	kernel: KernelId,
	program: &crate::runtime::dna::ir::tile::BlockProgram,
) -> Option<Vec<u8>> {
	match kernel {
		KernelId::AudioMelNormalizeF32 | KernelId::AudioMelFilterbankF32 => {
			device.jit_admit_tile_audio(program)
		}
		_ => device.jit_admit_tile_image(program),
	}
}

/// A `f32` scalar or unary pointwise consumer decodable into a
/// `ScalarTail`: scalar donors carry `[U32(elements), F32(scalar)]`,
/// unary donors carry `[U32(elements)]` — all one read and one write.
fn scalar_tail_op(node: &ComputeNode) -> Option<crate::runtime::dna::ir::tile::ScalarTail> {
	use crate::runtime::dna::ir::tile::ScalarTail;
	let tail = node.kernel.scalar_tail()?;
	match node.push_constants.as_slice() {
		[PushConstant::U32(_), PushConstant::F32(scalar)] => Some(match tail {
			KernelScalarTail::Add => ScalarTail::Add(*scalar),
			KernelScalarTail::Sub => ScalarTail::Sub(*scalar),
			KernelScalarTail::Mul => ScalarTail::Mul(*scalar),
			KernelScalarTail::Div => ScalarTail::Div(*scalar),
			KernelScalarTail::Max => ScalarTail::Max(*scalar),
			KernelScalarTail::Min => ScalarTail::Min(*scalar),
			_ => return None,
		}),
		[PushConstant::U32(_)] => Some(match tail {
			KernelScalarTail::Neg => ScalarTail::Neg,
			KernelScalarTail::Abs => ScalarTail::Abs,
			KernelScalarTail::Log => ScalarTail::Log,
			KernelScalarTail::Exp => ScalarTail::Exp,
			KernelScalarTail::Sqrt => ScalarTail::Sqrt,
			KernelScalarTail::Recip => ScalarTail::Recip,
			KernelScalarTail::Relu => ScalarTail::Relu,
			_ => return None,
		}),
		_ => None,
	}
}

/// The staged `mel_filterbank → mel_normalize` chain program's
/// staging capacity derives from the node's `frames × num_mels` run
/// length — the same value the merge proved fits the declared shared
/// row. `None` when the pushes do not describe an admitted chain.
fn audio_mel_chain_program(
	push_constants: &[PushConstant],
) -> Option<crate::runtime::dna::ir::tile::BlockProgram> {
	let [
		PushConstant::U32(_),
		PushConstant::U32(frames),
		PushConstant::U32(_),
		PushConstant::U32(num_mels),
		PushConstant::U32(_),
	] = push_constants
	else {
		return None;
	};
	let values = frames.checked_mul(*num_mels)?;
	if values == 0 || values > 4096 {
		return None;
	}
	let lanes = 256_u32;
	let capacity = values.next_power_of_two().max(lanes);
	crate::runtime::dna::ir::tile::BlockProgram::audio_mel_chain(lanes, capacity).ok()
}

/// Merge `AudioMelFilterbankF32 → AudioMelNormalizeF32` into the
/// staged `audio_mel_chain` region (OA-CC-32's post-FFT boundary):
/// the serial filterbank fold stages each element once in
/// `groupshared`, then the cooperative normalize reduces and rescales
/// the row — both donor passes in one dispatch. Admission mirrors the
/// scalar-epilogue resource proofs: the intermediate is written only
/// by the filterbank node, read only by the normalize node, invisible
/// to non-compute accesses, element-aligned with the recorded
/// `values_per_channel`, and within the staged-row capacity. The
/// merged node keeps the filterbank's buffers with the normalize
/// output substituted, and its `semantic_ops` records both donors so
/// `tile_scalar_tail_program` re-emits this program for a later
/// scalar tail.
fn try_fuse_audio_mel_chain(
	device: &Arc<Device>,
	nodes: &mut Vec<ComputeNode>,
	refs: &[&Storage],
	index: usize,
) -> bool {
	if !device.tile_audio_enabled() || index + 1 >= nodes.len() {
		return false;
	}
	let filterbank = &nodes[index];
	let normalize = &nodes[index + 1];
	if filterbank.kernel != KernelId::AudioMelFilterbankF32
		|| filterbank.semantic_ops.len() != 1
		|| normalize.kernel != KernelId::AudioMelNormalizeF32
		|| normalize.semantic_ops.len() != 1
	{
		return false;
	}
	let [spectrum, scratch, weights] = filterbank.buffers.as_slice() else {
		return false;
	};
	let [normalize_in, normalize_out] = normalize.buffers.as_slice() else {
		return false;
	};
	if spectrum.access != BufferAccess::Read
		|| scratch.access != BufferAccess::Write
		|| weights.access != BufferAccess::Read
		|| normalize_in.access != BufferAccess::Read
		|| normalize_out.access != BufferAccess::Write
		|| !scratch.buffer.same_as(&normalize_in.buffer)
		|| normalize_out.buffer.same_as(&scratch.buffer)
	{
		return false;
	}
	let [
		PushConstant::U32(channels),
		PushConstant::U32(frames),
		PushConstant::U32(_),
		PushConstant::U32(num_mels),
		PushConstant::U32(_),
	] = filterbank.push_constants.as_slice()
	else {
		return false;
	};
	let (channels, frames, num_mels) = (*channels, *frames, *num_mels);
	let [
		PushConstant::U32(normalize_channels),
		PushConstant::U32(values),
	] = normalize.push_constants.as_slice()
	else {
		return false;
	};
	let Some(expected_values) = frames.checked_mul(num_mels) else {
		return false;
	};
	if *normalize_channels != channels || *values != expected_values || expected_values == 0 {
		return false;
	}
	// Donor grids must still match the recorded nodes: the filterbank
	// tiles mels over lanes per (frame, channel); the normalize runs
	// one workgroup per channel.
	if filterbank.workgroups != [num_mels.div_ceil(32), frames, channels]
		|| normalize.workgroups != [channels, 1, 1]
		|| normalize_out.buffer.byte_len() != scratch.buffer.byte_len()
	{
		return false;
	}
	let normalize_output = normalize_out.buffer.clone();
	if !storage_unobservable(refs, &scratch.buffer)
		|| nodes
			.iter()
			.enumerate()
			.any(|(other, node)| other != index + 1 && node_reads(node, &scratch.buffer))
		|| nodes
			.iter()
			.enumerate()
			.any(|(other, node)| other != index && node_writes(node, &scratch.buffer))
	{
		return false;
	}
	let Some(program) = audio_mel_chain_program(filterbank.push_constants.as_slice()) else {
		return false;
	};
	let Some(key) = device.jit_admit_tile_audio(&program) else {
		return false;
	};
	let semantic_ops = merged_semantic_ops(&[
		nodes[index].semantic_ops.as_slice(),
		nodes[index + 1].semantic_ops.as_slice(),
	]);
	let node = &mut nodes[index];
	node.buffers[1] = BufferUse {
		buffer: normalize_output,
		access: BufferAccess::Write,
	};
	node.workgroups = [channels, 1, 1];
	node.fusion_signature = Some(device.tile_signature(&program));
	node.jit = Some(JitNode {
		key,
		buffers: node.buffers.clone(),
		push_constants: node.push_constants.clone(),
	});
	let (operation, contract_hash) = fused_identity(&semantic_ops, node, node.kernel);
	node.operation = operation;
	node.op_contract_hash = contract_hash;
	node.semantic_ops = semantic_ops;
	nodes.remove(index + 1);
	true
}

/// Merge scalar pointwise consumers into admitted tile gather
/// producers (OA-CC-33): `image.* → scalar chain` re-emits as one JIT
/// region whose shader folds the tail operands as constants, so the
/// merged dispatch keeps the producer's binding and push contract and
/// only the store target, signature, and key change. Runs before the
/// incumbent pair matchers and the elementwise tape so a tile program
/// claims its tail while the scalar ops are still discrete nodes.
///
/// Admission conditions mirror the tape fusion's resource proofs: the
/// intermediate must be produced by the tile node, consumed by exactly
/// the scalar node, invisible to every other node and to non-compute
/// accesses, and element-aligned with the scalar kernel's declared
/// count. Any failure leaves both nodes on their canonical routes.
pub(in crate::runtime) fn fuse_tile_scalar_epilogues(
	device: &Arc<Device>,
	mut nodes: Vec<ComputeNode>,
	refs: &[&Storage],
) -> Vec<ComputeNode> {
	if epilogue_fusion_disabled() || !(device.tile_image_enabled() || device.tile_audio_enabled()) {
		return nodes;
	}
	let mut index = 0;
	while index < nodes.len() {
		// A `mel_filterbank → mel_normalize` pair merges into the staged
		// chain region first; the merged node then flows through the
		// scalar-tail path below so a following pointwise consumer still
		// folds into the chain program.
		if try_fuse_audio_mel_chain(device, &mut nodes, refs, index) {
			continue;
		}
		let Some(program) = tile_scalar_tail_program(&nodes[index]) else {
			index += 1;
			continue;
		};
		let [_, producer_out] = nodes[index].buffers.as_slice() else {
			index += 1;
			continue;
		};
		if producer_out.access != BufferAccess::Write {
			index += 1;
			continue;
		}
		// Fold every consecutive scalar consumer into one tail; the
		// merged write target is the last consumer's output.
		let mut tail = Vec::new();
		let mut consumed = 0_usize;
		let mut current = producer_out.buffer.clone();
		while index + 1 + consumed < nodes.len() {
			let consumer = &nodes[index + 1 + consumed];
			let [input, output] = consumer.buffers.as_slice() else {
				break;
			};
			if input.access != BufferAccess::Read
				|| output.access != BufferAccess::Write
				|| !input.buffer.same_as(&current)
				|| output.buffer.same_as(&current)
			{
				break;
			}
			let Some(op) = scalar_tail_op(consumer) else {
				break;
			};
			let Some(PushConstant::U32(elements)) = consumer.push_constants.first() else {
				break;
			};
			if current.byte_len() != f32_bytes(*elements).unwrap_or(usize::MAX) {
				break;
			}
			// The folded value must be produced only by this node and
			// read only by this consumer — no other node or external
			// access may observe the intermediate.
			// `current`'s single writer is the previous chain node —
			// the tile producer for the first fold, the last absorbed
			// consumer afterwards.
			let writer = if consumed == 0 {
				index
			} else {
				index + consumed
			};
			if !storage_unobservable(refs, &current)
				|| nodes
					.iter()
					.enumerate()
					.any(|(other, node)| other != index + 1 + consumed && node_reads(node, &current))
				|| nodes
					.iter()
					.enumerate()
					.any(|(other, node)| other != writer && node_writes(node, &current))
			{
				break;
			}
			tail.push(op);
			current = output.buffer.clone();
			consumed += 1;
		}
		// Prologue direction: fold preceding scalar producers into this
		// node's input binding — the loaded value passes through each
		// op before the tile program's reads (OA-CC-33). Restricted to
		// consumers whose input read is elementwise.
		let mut prologue = Vec::new();
		let mut producers = 0_usize;
		let mut source = nodes[index].buffers[0].buffer.clone();
		while tile_scalar_prologue_elementwise(nodes[index].kernel) && index > producers {
			let producer = &nodes[index - 1 - producers];
			let [input, output] = producer.buffers.as_slice() else {
				break;
			};
			if input.access != BufferAccess::Read
				|| output.access != BufferAccess::Write
				|| !output.buffer.same_as(&source)
				|| input.buffer.same_as(&source)
			{
				break;
			}
			let Some(op) = scalar_tail_op(producer) else {
				break;
			};
			let Some(PushConstant::U32(elements)) = producer.push_constants.first() else {
				break;
			};
			if source.byte_len() != f32_bytes(*elements).unwrap_or(usize::MAX)
				|| input.buffer.byte_len() != source.byte_len()
			{
				break;
			}
			// `source` must be written only by this producer and read
			// only by this consumer — folding re-points the consumer's
			// input at the producer's input, so `source` never
			// materializes.
			if !storage_unobservable(refs, &source)
				|| nodes
					.iter()
					.enumerate()
					.any(|(other, node)| other != index && node_reads(node, &source))
				|| nodes
					.iter()
					.enumerate()
					.any(|(other, node)| other != index - 1 - producers && node_writes(node, &source))
			{
				break;
			}
			prologue.insert(0, op);
			source = input.buffer.clone();
			producers += 1;
		}
		if consumed == 0 && producers == 0 {
			index += 1;
			continue;
		}
		let kernel = nodes[index].kernel;
		let Some(key) = tile_scalar_tail_admit(
			device,
			kernel,
			&program
				.clone()
				.with_scalar_prologue(0, &prologue)
				.with_scalar_tail(&tail),
		) else {
			index += 1;
			continue;
		};
		let merged_program = program
			.with_scalar_prologue(0, &prologue)
			.with_scalar_tail(&tail);
		let parts = nodes[index + 1..=index + consumed]
			.iter()
			.map(|node| node.semantic_ops.as_slice())
			.chain(std::iter::once(nodes[index].semantic_ops.as_slice()))
			.chain(
				nodes[index - producers..index]
					.iter()
					.map(|node| node.semantic_ops.as_slice()),
			)
			.collect::<Vec<_>>();
		let merged = merged_semantic_ops(&parts);
		let producer = &mut nodes[index];
		let (operation, contract_hash) = fused_identity(&merged, producer, producer.kernel);
		producer.operation = operation;
		producer.op_contract_hash = contract_hash;
		if producers > 0 {
			producer.buffers[0] = BufferUse {
				buffer: source,
				access: BufferAccess::Read,
			};
		}
		if consumed > 0 {
			producer.buffers[1] = BufferUse {
				buffer: current,
				access: BufferAccess::Write,
			};
		}
		producer.fusion_signature = Some(device.tile_signature(&merged_program));
		producer.jit = Some(JitNode {
			key,
			buffers: producer.buffers.clone(),
			push_constants: producer.push_constants.clone(),
		});
		producer.semantic_ops = merged;
		nodes.drain(index + 1..=index + consumed);
		nodes.drain(index - producers..index);
		index -= producers;
		index += 1;
	}
	nodes
}
