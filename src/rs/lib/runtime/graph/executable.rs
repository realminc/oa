use std::sync::Arc;

use std::collections::BTreeMap;

use crate::{Error, Result};

use crate::runtime::{
	Buffer, BufferAccess, BufferBinding, ComputeDispatch, Device, PushConstant, SemanticGraph,
	SemanticLoweringAnalysis, SemanticOpId,
	device::features::ExecutionProfile,
	dna::{
		fusion,
		kernel::{KernelId, TrainingReplayRole},
	},
};

/// Owned snapshot of concrete work ready for Vulkan command recording.
#[derive(Clone)]
pub(in crate::runtime) struct ExecutableGraph {
	nodes: Vec<ComputeNode>,
	barriers: Vec<Vec<BufferHazard>>,
}

/// Conservative per-dispatch GPU floor measured on Xe GT2 (5–9µs per small
/// unfused dispatch in the `discovered_chains` tape sweep, rounded up).
/// Routing policy, not a capability bound; OA-CC-12 route records refine it.
const SUBMISSION_DISPATCH_FLOOR_NS: u64 = 8_000;
/// Memory-bandwidth proxy for node cost in bytes per nanosecond, from the
/// same Xe sweep (~6.6 f32 elements/ns ≈ 26B/ns). Memory-bound first-order
/// model — compute-bound kernels underestimate until OA-CC-12 supplies
/// measured per-kernel rates. This cannot bound watchdog exposure.
const SUBMISSION_BYTES_PER_NS: u64 = 26;
/// Default per-submission budget. Half a second sits below the ~0.67s
/// measured-safe 128-sequence PPO submission on Haswell (the 320-sequence
/// submission exceeded the i915 heartbeat, `VK_ERROR_DEVICE_LOST`
/// 2026-11-09) with margin, and far below typical OS GPU watchdog periods.
/// This is an estimate safety margin, not a guarantee against device loss.
const DEFAULT_SUBMISSION_BUDGET_NS: u64 = 500_000_000;

/// Seed a device's per-submission budget. `OA_SUBMISSION_BUDGET_MS`
/// overrides for laboratory runs; device-class entries arrive only with
/// measured evidence. The policy's source is recorded in the Xe
/// qualification ledger.
pub(in crate::runtime) fn submission_budget_seed(_info: &crate::DeviceInfo) -> u64 {
	if let Ok(value) = std::env::var("OA_SUBMISSION_BUDGET_MS")
		&& let Ok(ms) = value.parse::<u64>()
	{
		return ms.saturating_mul(1_000_000);
	}
	DEFAULT_SUBMISSION_BUDGET_NS
}

#[derive(Clone)]
pub(in crate::runtime) struct ComputeNode {
	pub(in crate::runtime) operation: &'static str,
	pub(in crate::runtime) kernel: KernelId,
	pub(in crate::runtime) buffers: Vec<BufferUse>,
	pub(in crate::runtime) push_constants: Vec<PushConstant>,
	pub(in crate::runtime) workgroups: [u32; 3],
	pub(in crate::runtime) indirect: Option<IndirectUse>,
	pub(in crate::runtime) semantic_ops: Vec<SemanticOpId>,
	pub(in crate::runtime) op_contract_hash: u64,
	/// An admitted runtime specialization of this node's fused elementwise
	/// chain. When the device table still holds the pipeline, recording
	/// dispatches it against `buffers` in program order (inputs then outputs)
	/// with `push_constants` as the trailing push words; otherwise the node's
	/// canonical tape dispatch executes unchanged.
	pub(in crate::runtime) jit: Option<JitNode>,
	/// Observation identity (`vkcache::jit::signature_key` of the canonical
	/// program) set on fused nodes while `OA_JIT_FUSION` is active. Recording
	/// timestamps the dispatch so the route store learns tape vs specialized
	/// medians; `None` on canonical nodes costs nothing.
	pub(in crate::runtime) fusion_signature: Option<u64>,
}

impl ComputeNode {
	/// First-order submission cost estimate: a fixed dispatch floor plus one
	/// memory pass over every bound buffer at the device bandwidth proxy.
	/// Compute-bound kernels (matmul, convolution) underestimate; the proxy
	/// is a splitting heuristic, not a safe upper bound. OA-CC-12's measured
	/// route records refine it. `None` for indirect dispatches whose element count is
	/// not knowable at record time.
	pub(in crate::runtime) fn estimated_cost_ns(&self) -> Option<u64> {
		if self.indirect.is_some() {
			return None;
		}
		let bytes = self.buffers.iter().fold(0u64, |sum, binding| {
			sum.saturating_add(u64::try_from(binding.buffer.byte_len()).unwrap_or(u64::MAX))
		});
		Some(SUBMISSION_DISPATCH_FLOOR_NS.saturating_add(bytes / SUBMISSION_BYTES_PER_NS))
	}
}

/// Record-time parameters for one admitted specialization. The pipeline is
/// owned by the device's jit table and resolved by `key` at recording, so a
/// missing or failed admission always falls back to the node's tape route.
#[derive(Clone)]
pub(in crate::runtime) struct JitNode {
	pub(in crate::runtime) key: Vec<u8>,
	pub(in crate::runtime) buffers: Vec<BufferUse>,
	pub(in crate::runtime) push_constants: Vec<PushConstant>,
}

#[derive(Clone)]
pub(in crate::runtime) struct BufferUse {
	pub(in crate::runtime) buffer: Buffer,
	pub(in crate::runtime) access: BufferAccess,
}

/// GPU-authored launch dimensions for one node: a retained buffer and the
/// four-byte-aligned byte offset of its `VkDispatchIndirectCommand` triple.
#[derive(Clone)]
pub(in crate::runtime) struct IndirectUse {
	pub(in crate::runtime) buffer: Buffer,
	pub(in crate::runtime) offset: u64,
}

#[derive(Clone)]
pub(in crate::runtime) struct BufferHazard {
	pub(in crate::runtime) buffer: Buffer,
	pub(in crate::runtime) source: AccessState,
	pub(in crate::runtime) destination: AccessState,
}

pub(in crate::runtime) struct ResourceLifetime {
	pub(in crate::runtime) buffer: Buffer,
	pub(in crate::runtime) first_access: usize,
	pub(in crate::runtime) last_access: usize,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(in crate::runtime) struct AccessState {
	read: bool,
	write: bool,
	indirect: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PlannedHazard<Resource> {
	resource: Resource,
	source: AccessState,
	destination: AccessState,
}

/// Resolve borrowed storage bindings into owned buffer uses for one node.
///
/// Zero-byte storage legitimately owns no buffer, but Vulkan descriptor
/// bindings still require a valid buffer handle. Kernels early-return on
/// zero extents, so a minimal fresh backing buffer satisfies the binding
/// without being touched.
fn resolve_buffers(
	device: &Arc<Device>,
	operation: &'static str,
	bindings: &[BufferBinding<'_>],
) -> Result<Vec<BufferUse>> {
	let mut buffers = Vec::with_capacity(bindings.len());
	for binding in bindings {
		if !binding.storage.belongs_to(device) {
			return Err(Error::invalid_argument(format!(
				"{operation} buffers must belong to the submitting engine"
			)));
		}
		let buffer = match binding.storage.buffer() {
			Some(buffer) => buffer.clone(),
			None => Buffer::host_visible_storage(device, 4)?,
		};
		buffers.push(BufferUse {
			buffer,
			access: binding.access,
		});
	}
	Ok(buffers)
}

impl ExecutableGraph {
	/// Resolve borrowed domain dispatches into one transactionally built graph.
	pub(in crate::runtime) fn from_dispatches(
		device: &Arc<Device>,
		dispatches: &[ComputeDispatch<'_>],
	) -> Result<Self> {
		if dispatches.is_empty() {
			return Err(Error::invalid_argument(
				"an executable graph requires at least one dispatch",
			));
		}

		let mut nodes = Vec::with_capacity(dispatches.len());
		for dispatch in dispatches {
			device.require_kernel(dispatch.kernel)?;
			let operation = dispatch.kernel.report_name();
			let buffers = resolve_buffers(device, operation, dispatch.buffers)?;
			let indirect = dispatch
				.indirect
				.map(|arguments| {
					if !arguments.storage.belongs_to(device) {
						return Err(Error::invalid_argument(format!(
							"{operation} indirect dispatch arguments must belong to the submitting engine"
						)));
					}
					arguments
						.storage
						.validate_recording_access(BufferAccess::Read)?;
					let buffer = arguments.storage.buffer().cloned().ok_or_else(|| {
						Error::backend_failure(
							"Vulkan",
							"compute-dispatch indirect argument validation",
							std::io::Error::other(format!(
								"{operation} references indirect storage without a Vulkan buffer"
							)),
						)
					})?;
					if arguments.offset % 4 != 0
						|| arguments
							.offset
							.checked_add(12)
							.is_none_or(|end| end > arguments.storage.byte_len() as u64)
					{
						return Err(Error::invalid_argument(format!(
							"{operation} indirect dispatch arguments require a four-byte-aligned VkDispatchIndirectCommand range inside the storage"
						)));
					}
					Ok(IndirectUse {
						buffer,
						offset: arguments.offset,
					})
				})
				.transpose()?;
			nodes.push(ComputeNode {
				operation,
				kernel: dispatch.kernel,
				buffers,
				push_constants: dispatch.push_constants.to_vec(),
				workgroups: dispatch.workgroups,
				indirect,
				semantic_ops: Vec::new(),
				op_contract_hash: 0,
				jit: None,
				fusion_signature: None,
			});
		}

		let barriers = plan_barriers(&nodes)?;
		Ok(Self { nodes, barriers })
	}

	/// Build a one-node graph for an extension kernel admitted through the
	/// public `oa::compute` surface.
	///
	/// The node reuses the jit pipeline-override path: the resolved extension
	/// pipeline substitutes the placeholder registry kernel at record time, so
	/// every kernel-derived contract (artifact write classification, bounded
	/// buffer count, secret gating) follows the jit route's admission-proved
	/// semantics. `jit.key` keeps plan-hash uniqueness across extension
	/// kernels; the placeholder identity only needs to exist for hashing and
	/// replay-role metadata, which are inert for extension dispatches.
	pub(in crate::runtime) fn from_extension(
		device: &Arc<Device>,
		operation: &'static str,
		extension_key: &[u8],
		bindings: &[BufferBinding<'_>],
		push_constants: &[PushConstant],
		workgroups: [u32; 3],
	) -> Result<Self> {
		let buffers = resolve_buffers(device, operation, bindings)?;
		let pushes = push_constants.to_vec();
		let nodes = vec![ComputeNode {
			operation,
			kernel: KernelId::MatrixAddF32,
			buffers: buffers.clone(),
			push_constants: pushes.clone(),
			workgroups,
			indirect: None,
			semantic_ops: Vec::new(),
			op_contract_hash: 0,
			jit: Some(JitNode {
				key: extension_key.to_vec(),
				buffers,
				push_constants: pushes,
			}),
			fusion_signature: None,
		}];
		let barriers = plan_barriers(&nodes)?;
		Ok(Self { nodes, barriers })
	}

	pub(in crate::runtime) fn attach_direct_semantic(
		&mut self,
		operation: SemanticOpId,
		name: &'static str,
		contract_hash: u64,
	) -> Result<()> {
		if self.nodes.len() != 1 || contract_hash == 0 {
			return Err(Error::failed_precondition(
				"direct semantic lowering requires one node and a nonzero contract hash",
			));
		}
		let node = &mut self.nodes[0];
		node.operation = name;
		node.semantic_ops.push(operation);
		node.op_contract_hash = contract_hash;
		Ok(())
	}

	pub(in crate::runtime) fn attach_split_semantic(
		&mut self,
		operation: SemanticOpId,
		name: &'static str,
		contract_hash: u64,
	) -> Result<()> {
		if self.nodes.len() < 2 || contract_hash == 0 {
			return Err(Error::failed_precondition(
				"split semantic lowering requires multiple nodes and a nonzero contract hash",
			));
		}
		for node in &mut self.nodes {
			node.operation = name;
			node.semantic_ops.push(operation);
			node.op_contract_hash = contract_hash;
		}
		Ok(())
	}

	pub(in crate::runtime) fn attach_composite_semantic(
		&mut self,
		operation: SemanticOpId,
		name: &'static str,
		contract_hash: u64,
	) -> Result<()> {
		if self.nodes.is_empty() || contract_hash == 0 {
			return Err(Error::failed_precondition(
				"composite semantic lowering requires executable nodes and a nonzero contract hash",
			));
		}
		for node in &mut self.nodes {
			node.operation = name;
			node.semantic_ops.clear();
			node.semantic_ops.push(operation);
			node.op_contract_hash = contract_hash;
		}
		Ok(())
	}

	pub(in crate::runtime) fn attach_fused_semantic(
		&mut self,
		operations: &[SemanticOpId],
	) -> Result<()> {
		if self.nodes.len() != 1 || operations.len() < 2 {
			return Err(Error::failed_precondition(
				"fused semantic lowering requires one node and at least two operations",
			));
		}
		let node = &mut self.nodes[0];
		if node.kernel.semantic_contract().is_some() {
			return Err(Error::failed_precondition(
				"fused semantic lowering requires a distinct lowering-only kernel",
			));
		}
		node.semantic_ops.extend_from_slice(operations);
		Ok(())
	}

	pub(in crate::runtime) fn join(graphs: Vec<Self>) -> Result<Self> {
		let nodes = graphs
			.into_iter()
			.flat_map(|graph| graph.nodes)
			.collect::<Vec<_>>();
		if nodes.is_empty() {
			return Err(Error::invalid_argument(
				"an executable graph requires at least one dispatch",
			));
		}
		let barriers = plan_barriers(&nodes)?;
		Ok(Self { nodes, barriers })
	}

	pub(in crate::runtime) fn from_nodes(nodes: Vec<ComputeNode>) -> Result<Self> {
		if nodes.is_empty() {
			return Err(Error::invalid_argument(
				"an executable graph requires at least one dispatch",
			));
		}
		let barriers = plan_barriers(&nodes)?;
		Ok(Self { nodes, barriers })
	}

	pub(in crate::runtime) fn nodes(&self) -> &[ComputeNode] {
		&self.nodes
	}

	/// Greedy node ranges whose summed cost estimates each fit `budget_ns`.
	/// Splits only happen between nodes: the timeline chain makes the
	/// previous chunk's writes available and visible before the next
	/// submission's first command, so every node boundary is a legal
	/// boundary. Indirect-dispatch nodes are unestimable — they admit at the
	/// dispatch floor explicitly rather than blocking neighbouring splits.
	/// `Err((index, estimate))` names a single node over the budget; the
	/// caller turns that into the typed pre-submission error.
	pub(in crate::runtime) fn budget_ranges(
		&self,
		budget_ns: u64,
	) -> std::result::Result<Vec<std::ops::Range<usize>>, (usize, u64)> {
		let mut ranges = Vec::new();
		let mut start = 0usize;
		let mut sum = 0u64;
		for (index, node) in self.nodes.iter().enumerate() {
			let cost = node
				.estimated_cost_ns()
				.unwrap_or(SUBMISSION_DISPATCH_FLOOR_NS);
			if cost > budget_ns {
				return Err((index, cost));
			}
			if index > start && sum.saturating_add(cost) > budget_ns {
				ranges.push(start..index);
				start = index;
				sum = 0;
			}
			sum = sum.saturating_add(cost);
		}
		ranges.push(start..self.nodes.len());
		Ok(ranges)
	}

	pub(in crate::runtime) fn validate_training_replay_safety(&self) -> Result<()> {
		validate_training_kernel_sequence(self.nodes.iter().map(|node| (node.operation, node.kernel)))?;
		for (index, replay) in self.nodes.iter().enumerate() {
			if replay.kernel.training_replay_role() != TrainingReplayRole::ReplayRng {
				continue;
			}
			let advance = &self.nodes[index + 1];
			let state = replay.buffers.last().ok_or_else(|| {
				Error::failed_precondition(format!(
					"training program replay RNG {} has no state binding",
					replay.operation
				))
			})?;
			let [advanced_state] = advance.buffers.as_slice() else {
				return Err(Error::failed_precondition(format!(
					"training program RNG state advance {} must bind exactly one state buffer",
					advance.operation
				)));
			};
			if state.access != BufferAccess::Read
				|| advanced_state.access != BufferAccess::ReadWrite
				|| !state.buffer.same_as(&advanced_state.buffer)
			{
				return Err(Error::failed_precondition(format!(
					"training program RNG state advance {} does not update the state read by {}",
					advance.operation, replay.operation
				)));
			}
		}
		Ok(())
	}

	pub(in crate::runtime) fn barriers_before(&self, node: usize) -> &[BufferHazard] {
		&self.barriers[node]
	}

	pub(in crate::runtime) fn barrier_count(&self) -> usize {
		self.barriers.iter().map(Vec::len).sum()
	}

	pub(in crate::runtime) fn identity(&self) -> u64 {
		let mut hash = StableHash::new();
		let mut resources = BTreeMap::<u32, u32>::new();
		let mut next_resource = 0_u32;
		hash.usize(self.nodes.len());
		for node in &self.nodes {
			hash.bytes(node.operation.as_bytes());
			hash.u64(node.op_contract_hash);
			hash.usize(node.semantic_ops.len());
			for operation in &node.semantic_ops {
				hash.u32(operation.index());
			}
			hash.u16(node.kernel as u16);
			hash.u8(u8::from(node.jit.is_some()));
			if let Some(jit) = &node.jit {
				hash.usize(jit.key.len());
				hash.bytes(&jit.key);
			}
			let artifact = node.kernel.artifact();
			hash.u64(artifact.content_id());
			match artifact.physical_write {
				None => hash.u8(0),
				Some(contract) => {
					hash.u8(1);
					hash.usize(contract.writes.len());
					for write in contract.writes {
						hash.u8(write.binding);
						hash.u8(write.domain as u8);
						hash.u8(write.partition as u8);
						hash.u8(write.extent as u8);
						hash.u8(write.collision as u8);
						hash.u8(write.tail as u8);
					}
					hash.u8(contract.workspace as u8);
				}
			}
			hash.usize(node.buffers.len());
			for buffer_use in &node.buffers {
				let descriptor = buffer_use.buffer.descriptor_index();
				let resource = *resources.entry(descriptor).or_insert_with(|| {
					let resource = next_resource;
					next_resource = next_resource.wrapping_add(1);
					resource
				});
				hash.u32(resource);
				hash.u8(match buffer_use.access {
					BufferAccess::Read => 0,
					BufferAccess::Write => 1,
					BufferAccess::ReadWrite => 2,
				});
			}
			hash.usize(node.push_constants.len());
			for constant in &node.push_constants {
				match *constant {
					PushConstant::U32(value) => {
						hash.u8(0);
						hash.u32(value);
					}
					PushConstant::F32(value) => {
						hash.u8(1);
						hash.u32(value.to_bits());
					}
				}
			}
			for dimension in node.workgroups {
				hash.u32(dimension);
			}
			match &node.indirect {
				None => hash.u8(0),
				Some(indirect) => {
					hash.u8(1);
					let resource = *resources
						.entry(indirect.buffer.descriptor_index())
						.or_insert_with(|| {
							let resource = next_resource;
							next_resource = next_resource.wrapping_add(1);
							resource
						});
					hash.u32(resource);
					hash.u64(indirect.offset);
				}
			}
		}
		hash.finish()
	}

	pub(in crate::runtime) fn read_only_buffers(&self) -> Vec<Buffer> {
		let mut resources = BTreeMap::<u32, (Buffer, AccessState)>::new();
		for node in &self.nodes {
			for (index, buffer_use) in node.buffers.iter().enumerate() {
				if fusion::is_tape_binding(node, index) {
					continue;
				}
				let state = AccessState::from_access(buffer_use.access);
				resources
					.entry(buffer_use.buffer.descriptor_index())
					.and_modify(|(_, access)| access.merge(state))
					.or_insert_with(|| (buffer_use.buffer.clone(), state));
			}
			if let Some(indirect) = &node.indirect {
				resources
					.entry(indirect.buffer.descriptor_index())
					.and_modify(|(_, access)| access.merge(AccessState::INDIRECT))
					.or_insert_with(|| (indirect.buffer.clone(), AccessState::INDIRECT));
			}
		}
		resources
			.into_values()
			.filter_map(|(buffer, access)| {
				(!access.write && (access.read || access.indirect)).then_some(buffer)
			})
			.collect()
	}

	pub(in crate::runtime) fn resource_lifetimes(&self) -> Vec<ResourceLifetime> {
		let mut resources = Vec::<ResourceLifetime>::new();
		for (node_index, node) in self.nodes.iter().enumerate() {
			for buffer in node
				.buffers
				.iter()
				.map(|buffer_use| &buffer_use.buffer)
				.chain(node.indirect.iter().map(|indirect| &indirect.buffer))
			{
				if let Some(lifetime) = resources
					.iter_mut()
					.find(|lifetime| lifetime.buffer.same_as(buffer))
				{
					lifetime.last_access = node_index;
				} else {
					resources.push(ResourceLifetime {
						buffer: buffer.clone(),
						first_access: node_index,
						last_access: node_index,
					});
				}
			}
		}
		resources
	}

	pub(in crate::runtime) fn rebind_read_only(
		&mut self,
		current: &Buffer,
		replacement: &Buffer,
		profile: ExecutionProfile,
	) -> Result<()> {
		let mut found = false;
		for node in &self.nodes {
			for buffer_use in &node.buffers {
				if buffer_use.buffer.same_as(current) {
					found = true;
					if buffer_use.access != BufferAccess::Read {
						return Err(Error::invalid_argument(
							"only read-only execution-plan Matrix inputs can be rebound",
						));
					}
					if fusion::is_fused_chain_kernel(node.kernel) && profile == ExecutionProfile::Core {
						return Err(Error::invalid_argument(
							"Core execution plans cannot rebind fused elementwise operands",
						));
					}
				} else if buffer_use.buffer.same_as(replacement) {
					return Err(Error::invalid_argument(
						"execution-plan input rebinding cannot introduce resource aliasing",
					));
				}
			}
			if let Some(indirect) = &node.indirect {
				if indirect.buffer.same_as(current) {
					found = true;
				} else if indirect.buffer.same_as(replacement) {
					return Err(Error::invalid_argument(
						"execution-plan input rebinding cannot introduce resource aliasing",
					));
				}
			}
		}
		if !found {
			return Err(Error::invalid_argument(
				"captured Matrix is not an input of this execution plan",
			));
		}
		let mut candidate = self.clone();
		for node in &mut candidate.nodes {
			for buffer_use in &mut node.buffers {
				if buffer_use.buffer.same_as(current) {
					buffer_use.buffer = replacement.clone();
				}
			}
			if let Some(jit) = &mut node.jit {
				for buffer_use in &mut jit.buffers {
					if buffer_use.buffer.same_as(current) {
						buffer_use.buffer = replacement.clone();
					}
				}
			}
			if let Some(indirect) = &mut node.indirect
				&& indirect.buffer.same_as(current)
			{
				indirect.buffer = replacement.clone();
			}
		}
		candidate.barriers = plan_barriers(&candidate.nodes)?;
		*self = candidate;
		Ok(())
	}

	pub(in crate::runtime) fn materialize_alias_replacements(
		&mut self,
		replacements: &[(Buffer, Buffer)],
	) -> Result<()> {
		let mut candidate = self.clone();
		for node in &mut candidate.nodes {
			for buffer_use in &mut node.buffers {
				if let Some((_, replacement)) = replacements
					.iter()
					.find(|(source, _)| source.same_as(&buffer_use.buffer))
				{
					buffer_use.buffer = replacement.clone();
				}
			}
			if let Some(jit) = &mut node.jit {
				for buffer_use in &mut jit.buffers {
					if let Some((_, replacement)) = replacements
						.iter()
						.find(|(source, _)| source.same_as(&buffer_use.buffer))
					{
						buffer_use.buffer = replacement.clone();
					}
				}
			}
			if let Some(indirect) = &mut node.indirect
				&& let Some((_, replacement)) = replacements
					.iter()
					.find(|(source, _)| source.same_as(&indirect.buffer))
			{
				indirect.buffer = replacement.clone();
			}
		}
		candidate.barriers = plan_barriers(&candidate.nodes)?;
		*self = candidate;
		Ok(())
	}

	pub(in crate::runtime) fn analyze_semantic_lowering(
		&self,
		semantic: &SemanticGraph,
	) -> Result<SemanticLoweringAnalysis> {
		semantic.validate()?;
		let mut analysis = SemanticLoweringAnalysis::empty(semantic.operations().len());
		let mut fusion_membership = vec![false; semantic.operations().len()];
		for node in &self.nodes {
			if node.semantic_ops.is_empty() {
				analysis.note_compatibility_node();
				continue;
			}
			analysis.note_schema_node(node.semantic_ops.len())?;
			if node.semantic_ops.len() > 1 && node.op_contract_hash != 0 {
				return Err(Error::failed_precondition(
					"fused executable node must have a distinct implementation identity",
				));
			}
			let mut owners = BTreeMap::new();
			for owner in &node.semantic_ops {
				let operation = semantic
					.operations()
					.get(owner.index() as usize)
					.ok_or_else(|| {
						Error::out_of_range("executable node references an unknown semantic operation")
					})?;
				if owners.insert(owner.index(), ()).is_some() {
					return Err(Error::already_exists(
						"executable node repeats a semantic operation owner",
					));
				}
				if node.semantic_ops.len() == 1
					&& (node.operation != operation.name()
						|| node.op_contract_hash != operation.contract_hash())
				{
					return Err(Error::failed_precondition(
						"executable node identity does not match its semantic operation",
					));
				}
				let count = &mut analysis.node_counts_mut()[owner.index() as usize];
				*count = count.saturating_add(1);
				if node.semantic_ops.len() > 1 {
					fusion_membership[owner.index() as usize] = true;
				}
			}
		}
		analysis.finish(&fusion_membership)?;
		Ok(analysis)
	}
}

fn validate_training_kernel_sequence<'a>(
	kernels: impl IntoIterator<Item = (&'a str, KernelId)>,
) -> Result<()> {
	let mut optimizer_state_advances = 0_usize;
	let mut optimizer_state_updates = 0_usize;
	let mut advance_seen = false;
	let mut awaiting_rng_advance = false;
	for (operation, kernel) in kernels {
		let role = kernel.training_replay_role();
		if awaiting_rng_advance && role != TrainingReplayRole::RngStateAdvance {
			return Err(Error::failed_precondition(format!(
				"training program replay RNG must be followed immediately by its state advance; found {operation}"
			)));
		}
		match role {
			TrainingReplayRole::Safe => {}
			TrainingReplayRole::FrozenRng => {
				return Err(Error::failed_precondition(format!(
					"training program operation {operation} embeds a host seed; use its replay-state kernel"
				)));
			}
			TrainingReplayRole::ReplayRng => {
				awaiting_rng_advance = true;
			}
			TrainingReplayRole::RngStateAdvance => {
				if !awaiting_rng_advance {
					return Err(Error::failed_precondition(format!(
						"training program operation {operation} advances RNG state without a preceding replay RNG operation"
					)));
				}
				awaiting_rng_advance = false;
			}
			TrainingReplayRole::HostSteppedOptimizer => {
				return Err(Error::failed_precondition(format!(
					"training program operation {operation} embeds host-stepped optimizer state; use a replay-state kernel"
				)));
			}
			TrainingReplayRole::OptimizerStateAdvance => {
				optimizer_state_advances = optimizer_state_advances.saturating_add(1);
				advance_seen = true;
			}
			TrainingReplayRole::OptimizerStateUpdate => {
				if !advance_seen {
					return Err(Error::failed_precondition(format!(
						"training program operation {operation} reads optimizer replay state before it is advanced"
					)));
				}
				optimizer_state_updates = optimizer_state_updates.saturating_add(1);
			}
		}
	}
	if awaiting_rng_advance {
		return Err(Error::failed_precondition(
			"training program contains a replay RNG operation without its state advance",
		));
	}
	if optimizer_state_updates != 0 && optimizer_state_advances != 1 {
		return Err(Error::failed_precondition(format!(
			"training program requires exactly one optimizer-state advance before replay updates; recorded {optimizer_state_advances}"
		)));
	}
	Ok(())
}

struct StableHash(u64);

impl StableHash {
	const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
	const PRIME: u64 = 0x0000_0100_0000_01b3;

	const fn new() -> Self {
		Self(Self::OFFSET)
	}

	fn bytes(&mut self, bytes: &[u8]) {
		self.usize(bytes.len());
		self.raw(bytes);
	}

	fn raw(&mut self, bytes: &[u8]) {
		for byte in bytes {
			self.0 ^= u64::from(*byte);
			self.0 = self.0.wrapping_mul(Self::PRIME);
		}
	}

	fn u8(&mut self, value: u8) {
		self.raw(&value.to_le_bytes());
	}

	fn u16(&mut self, value: u16) {
		self.raw(&value.to_le_bytes());
	}

	fn u32(&mut self, value: u32) {
		self.raw(&value.to_le_bytes());
	}

	fn u64(&mut self, value: u64) {
		self.raw(&value.to_le_bytes());
	}

	fn usize(&mut self, value: usize) {
		self.raw(value.to_string().as_bytes());
		self.u8(0xff);
	}

	const fn finish(self) -> u64 {
		self.0
	}
}

impl AccessState {
	const INDIRECT: Self = Self {
		read: false,
		write: false,
		indirect: true,
	};

	const fn from_access(access: BufferAccess) -> Self {
		match access {
			BufferAccess::Read => Self {
				read: true,
				write: false,
				indirect: false,
			},
			BufferAccess::Write => Self {
				read: false,
				write: true,
				indirect: false,
			},
			BufferAccess::ReadWrite => Self {
				read: true,
				write: true,
				indirect: false,
			},
		}
	}

	fn merge(&mut self, other: Self) {
		self.read |= other.read;
		self.write |= other.write;
		self.indirect |= other.indirect;
	}

	const fn conflicts_with(self, next: Self) -> bool {
		self.write || next.write
	}

	pub(in crate::runtime) const fn reads(self) -> bool {
		self.read
	}

	pub(in crate::runtime) const fn writes(self) -> bool {
		self.write
	}

	pub(in crate::runtime) const fn uses_indirect_commands(self) -> bool {
		self.indirect
	}

	pub(in crate::runtime) fn vk_access(self) -> vk::AccessFlags2 {
		let mut flags = vk::AccessFlags2::empty();
		if self.read {
			flags = flags | vk::AccessFlagBits2::SHADER_STORAGE_READ_BIT;
		}
		if self.write {
			flags = flags | vk::AccessFlagBits2::SHADER_STORAGE_WRITE_BIT;
		}
		if self.indirect {
			flags = flags | vk::AccessFlagBits2::INDIRECT_COMMAND_READ_BIT;
		}
		flags
	}
}

fn plan_barriers(nodes: &[ComputeNode]) -> Result<Vec<Vec<BufferHazard>>> {
	let mut resources = BTreeMap::<u32, Buffer>::new();
	let accesses = nodes
		.iter()
		.map(|node| {
			let mut bindings =
				Vec::with_capacity(node.buffers.len() + usize::from(node.indirect.is_some()));
			for buffer_use in &node.buffers {
				let identity = buffer_use.buffer.descriptor_index();
				resources
					.entry(identity)
					.or_insert_with(|| buffer_use.buffer.clone());
				bindings.push((identity, AccessState::from_access(buffer_use.access)));
			}
			if let Some(indirect) = &node.indirect {
				let identity = indirect.buffer.descriptor_index();
				resources
					.entry(identity)
					.or_insert_with(|| indirect.buffer.clone());
				bindings.push((identity, AccessState::INDIRECT));
			}
			aggregate_accesses(bindings)
		})
		.collect::<Vec<_>>();

	plan_access_states(&accesses)
		.into_iter()
		.map(|hazards| {
			hazards
				.into_iter()
				.map(|hazard| {
					let buffer = resources.get(&hazard.resource).cloned().ok_or_else(|| {
						Error::backend_failure(
							"Vulkan",
							"executable-graph hazard planning",
							std::io::Error::other("planned hazard lost its retained buffer identity"),
						)
					})?;
					Ok(BufferHazard {
						buffer,
						source: hazard.source,
						destination: hazard.destination,
					})
				})
				.collect::<Result<Vec<_>>>()
		})
		.collect()
}

fn aggregate_accesses(
	accesses: impl IntoIterator<Item = (u32, AccessState)>,
) -> BTreeMap<u32, AccessState> {
	let mut aggregate = BTreeMap::new();
	for (identity, access) in accesses {
		aggregate
			.entry(identity)
			.and_modify(|state: &mut AccessState| state.merge(access))
			.or_insert(access);
	}
	aggregate
}

fn plan_access_states<Resource: Copy + Ord>(
	nodes: &[BTreeMap<Resource, AccessState>],
) -> Vec<Vec<PlannedHazard<Resource>>> {
	let mut previous = BTreeMap::<Resource, AccessState>::new();
	let mut plan = Vec::with_capacity(nodes.len());

	for current in nodes {
		let mut barriers = Vec::new();
		for (resource, destination) in current {
			if let Some(source) = previous.get(resource)
				&& source.conflicts_with(*destination)
			{
				barriers.push(PlannedHazard {
					resource: *resource,
					source: *source,
					destination: *destination,
				});
			}
		}
		previous.extend(current);
		plan.push(barriers);
	}

	plan
}
