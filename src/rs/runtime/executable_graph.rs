use std::collections::BTreeMap;

use crate::{Error, Result};

use super::{
	BufferAccess, ComputeDispatch, PushConstant, SemanticGraph, SemanticLoweringAnalysis,
	SemanticOpId,
	shader::{KernelId, TrainingReplayRole},
	vk,
};

/// Owned snapshot of concrete work ready for Vulkan command recording.
#[derive(Clone)]
pub(in crate::runtime) struct ExecutableGraph {
	nodes: Vec<ComputeNode>,
	barriers: Vec<Vec<BufferHazard>>,
}

#[derive(Clone)]
pub(in crate::runtime) struct ComputeNode {
	pub(in crate::runtime) operation: &'static str,
	pub(in crate::runtime) kernel: KernelId,
	pub(in crate::runtime) buffers: Vec<BufferUse>,
	pub(in crate::runtime) push_constants: Vec<PushConstant>,
	pub(in crate::runtime) workgroups: [u32; 3],
	pub(in crate::runtime) semantic_ops: Vec<SemanticOpId>,
	pub(in crate::runtime) op_contract_hash: u64,
}

#[derive(Clone)]
pub(in crate::runtime) struct BufferUse {
	pub(in crate::runtime) buffer: vk::Buffer,
	pub(in crate::runtime) access: BufferAccess,
}

#[derive(Clone)]
pub(in crate::runtime) struct BufferHazard {
	pub(in crate::runtime) buffer: vk::Buffer,
	pub(in crate::runtime) source: AccessState,
	pub(in crate::runtime) destination: AccessState,
}

pub(in crate::runtime) struct ResourceLifetime {
	pub(in crate::runtime) buffer: vk::Buffer,
	pub(in crate::runtime) first_access: usize,
	pub(in crate::runtime) last_access: usize,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(in crate::runtime) struct AccessState {
	read: bool,
	write: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PlannedHazard<Resource> {
	resource: Resource,
	source: AccessState,
	destination: AccessState,
}

impl ExecutableGraph {
	/// Resolve borrowed domain dispatches into one transactionally built graph.
	pub(in crate::runtime) fn from_dispatches(
		device: &vk::Device,
		dispatches: &[ComputeDispatch<'_>],
	) -> Result<Self> {
		if dispatches.is_empty() {
			return Err(Error::invalid_argument(
				"an executable graph requires at least one dispatch",
			));
		}

		let mut nodes = Vec::with_capacity(dispatches.len());
		for dispatch in dispatches {
			let operation = dispatch.kernel.report_name();
			let mut buffers = Vec::with_capacity(dispatch.buffers.len());
			for binding in dispatch.buffers {
				if !binding.storage.belongs_to(device) {
					return Err(Error::invalid_argument(format!(
						"{} buffers must belong to the submitting engine",
						operation
					)));
				}
				let buffer = binding.storage.buffer().cloned().ok_or_else(|| {
					Error::backend_failure(
						"Vulkan",
						"compute-dispatch storage validation",
						std::io::Error::other(format!(
							"{} references storage without a Vulkan buffer",
							operation
						)),
					)
				})?;
				buffers.push(BufferUse {
					buffer,
					access: binding.access,
				});
			}
			nodes.push(ComputeNode {
				operation,
				kernel: dispatch.kernel,
				buffers,
				push_constants: dispatch.push_constants.to_vec(),
				workgroups: dispatch.workgroups,
				semantic_ops: Vec::new(),
				op_contract_hash: 0,
			});
		}

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

	pub(in crate::runtime) fn validate_training_replay_safety(&self) -> Result<()> {
		validate_training_kernel_sequence(
			self.nodes.iter().map(|node| (node.operation, node.kernel)),
		)?;
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
		}
		hash.finish()
	}

	pub(in crate::runtime) fn read_only_buffers(&self) -> Vec<vk::Buffer> {
		let mut resources = BTreeMap::<u32, (vk::Buffer, AccessState)>::new();
		for node in &self.nodes {
			for buffer_use in &node.buffers {
				let state = AccessState::from_access(buffer_use.access);
				resources
					.entry(buffer_use.buffer.descriptor_index())
					.and_modify(|(_, access)| access.merge(state))
					.or_insert_with(|| (buffer_use.buffer.clone(), state));
			}
		}
		resources
			.into_values()
			.filter_map(|(buffer, access)| (!access.write && access.read).then_some(buffer))
			.collect()
	}

	pub(in crate::runtime) fn resource_lifetimes(&self) -> Vec<ResourceLifetime> {
		let mut resources = Vec::<ResourceLifetime>::new();
		for (node_index, node) in self.nodes.iter().enumerate() {
			for buffer_use in &node.buffers {
				if let Some(lifetime) = resources
					.iter_mut()
					.find(|lifetime| lifetime.buffer.same_as(&buffer_use.buffer))
				{
					lifetime.last_access = node_index;
				} else {
					resources.push(ResourceLifetime {
						buffer: buffer_use.buffer.clone(),
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
		current: &vk::Buffer,
		replacement: &vk::Buffer,
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
				} else if buffer_use.buffer.same_as(replacement) {
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
		}
		candidate.barriers = plan_barriers(&candidate.nodes)?;
		*self = candidate;
		Ok(())
	}

	pub(in crate::runtime) fn materialize_alias_replacements(
		&mut self,
		replacements: &[(vk::Buffer, vk::Buffer)],
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
						Error::out_of_range(
							"executable node references an unknown semantic operation",
						)
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
	const fn from_access(access: BufferAccess) -> Self {
		match access {
			BufferAccess::Read => Self {
				read: true,
				write: false,
			},
			BufferAccess::Write => Self {
				read: false,
				write: true,
			},
			BufferAccess::ReadWrite => Self {
				read: true,
				write: true,
			},
		}
	}

	fn merge(&mut self, other: Self) {
		self.read |= other.read;
		self.write |= other.write;
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

	pub(in crate::runtime) fn vk_access(self) -> ash::vk::AccessFlags2 {
		let mut flags = ash::vk::AccessFlags2::empty();
		if self.read {
			flags |= ash::vk::AccessFlags2::SHADER_STORAGE_READ;
		}
		if self.write {
			flags |= ash::vk::AccessFlags2::SHADER_STORAGE_WRITE;
		}
		flags
	}
}

fn plan_barriers(nodes: &[ComputeNode]) -> Result<Vec<Vec<BufferHazard>>> {
	let mut resources = BTreeMap::<u32, vk::Buffer>::new();
	let accesses = nodes
		.iter()
		.map(|node| {
			let bindings = node.buffers.iter().map(|buffer_use| {
				let identity = buffer_use.buffer.descriptor_index();
				resources
					.entry(identity)
					.or_insert_with(|| buffer_use.buffer.clone());
				(identity, buffer_use.access)
			});
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
							std::io::Error::other(
								"planned hazard lost its retained buffer identity",
							),
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
	accesses: impl IntoIterator<Item = (u32, BufferAccess)>,
) -> BTreeMap<u32, AccessState> {
	let mut aggregate = BTreeMap::new();
	for (identity, access) in accesses {
		let access = AccessState::from_access(access);
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

#[cfg(test)]
mod tests {
	use super::{
		AccessState, aggregate_accesses, plan_access_states, validate_training_kernel_sequence,
	};
	use crate::runtime::{BufferAccess, shader::KernelId};

	fn state(access: BufferAccess) -> AccessState {
		AccessState::from_access(access)
	}

	fn plan(nodes: &[&[(u32, BufferAccess)]]) -> Vec<Vec<super::PlannedHazard<u32>>> {
		let accesses = nodes
			.iter()
			.map(|node| aggregate_accesses(node.iter().copied()))
			.collect::<Vec<_>>();
		plan_access_states(&accesses)
	}

	#[test]
	fn derives_raw_war_and_waw_without_read_after_read_barriers() {
		let raw = plan(&[&[(1, BufferAccess::Write)], &[(1, BufferAccess::Read)]]);
		assert_eq!(raw[1].len(), 1);
		assert_eq!(raw[1][0].source, state(BufferAccess::Write));
		assert_eq!(raw[1][0].destination, state(BufferAccess::Read));

		let war = plan(&[&[(1, BufferAccess::Read)], &[(1, BufferAccess::Write)]]);
		assert_eq!(war[1].len(), 1);
		assert_eq!(war[1][0].source, state(BufferAccess::Read));
		assert_eq!(war[1][0].destination, state(BufferAccess::Write));

		let waw = plan(&[&[(1, BufferAccess::Write)], &[(1, BufferAccess::Write)]]);
		assert_eq!(waw[1].len(), 1);

		let raw_in_place = plan(&[&[(1, BufferAccess::Write)], &[(1, BufferAccess::ReadWrite)]]);
		assert_eq!(raw_in_place[1].len(), 1);
		assert_eq!(
			raw_in_place[1][0].destination,
			AccessState {
				read: true,
				write: true,
			}
		);

		let rar = plan(&[&[(1, BufferAccess::Read)], &[(1, BufferAccess::Read)]]);
		assert!(rar[1].is_empty());
	}

	#[test]
	fn retains_state_across_unrelated_nodes_and_merges_aliases() {
		let planned = plan(&[
			&[(7, BufferAccess::Write)],
			&[(9, BufferAccess::Read)],
			&[(7, BufferAccess::Read), (7, BufferAccess::Write)],
		]);

		assert!(planned[0].is_empty());
		assert!(planned[1].is_empty());
		assert_eq!(planned[2].len(), 1);
		assert_eq!(planned[2][0].resource, 7);
		assert_eq!(
			planned[2][0].destination,
			AccessState {
				read: true,
				write: true,
			}
		);
	}

	#[test]
	fn bundles_conflicts_in_stable_resource_order() {
		let planned = plan(&[
			&[(8, BufferAccess::Write), (2, BufferAccess::Write)],
			&[(8, BufferAccess::Read), (2, BufferAccess::Read)],
		]);

		assert_eq!(
			planned[1]
				.iter()
				.map(|hazard| hazard.resource)
				.collect::<Vec<_>>(),
			[2, 8]
		);
	}

	#[test]
	fn maps_declared_access_to_exact_vulkan_storage_flags() {
		assert_eq!(
			state(BufferAccess::Read).vk_access(),
			ash::vk::AccessFlags2::SHADER_STORAGE_READ
		);
		assert_eq!(
			state(BufferAccess::Write).vk_access(),
			ash::vk::AccessFlags2::SHADER_STORAGE_WRITE
		);
		assert_eq!(
			state(BufferAccess::ReadWrite).vk_access(),
			ash::vk::AccessFlags2::SHADER_STORAGE_READ
				| ash::vk::AccessFlags2::SHADER_STORAGE_WRITE
		);
	}

	#[test]
	fn training_replay_requires_one_optimizer_advance_before_all_updates() {
		validate_training_kernel_sequence([
			("ml.adamw_graph_advance", KernelId::MlAdamWGraphAdvanceU32),
			("ml.adamw_graph", KernelId::MlAdamWGraphF32),
			("ml.adamw_graph", KernelId::MlAdamWGraphF32),
		])
		.unwrap();

		let missing =
			validate_training_kernel_sequence([("ml.adamw_graph", KernelId::MlAdamWGraphF32)])
				.unwrap_err();
		assert_eq!(
			missing.message(),
			"training program operation ml.adamw_graph reads optimizer replay state before it is advanced"
		);

		let duplicate = validate_training_kernel_sequence([
			("ml.adamw_graph_advance", KernelId::MlAdamWGraphAdvanceU32),
			("ml.adamw_graph_advance", KernelId::MlAdamWGraphAdvanceU32),
			("ml.adamw_graph", KernelId::MlAdamWGraphF32),
		])
		.unwrap_err();
		assert_eq!(
			duplicate.message(),
			"training program requires exactly one optimizer-state advance before replay updates; recorded 2"
		);
	}

	#[test]
	fn training_replay_rejects_host_stepped_optimizer_kernels() {
		let error =
			validate_training_kernel_sequence([("ml.adamw", KernelId::MlAdamWF32)]).unwrap_err();
		assert_eq!(
			error.message(),
			"training program operation ml.adamw embeds host-stepped optimizer state; use a replay-state kernel"
		);
	}

	#[test]
	fn training_replay_requires_one_advance_after_each_rng_operation() {
		validate_training_kernel_sequence([
			(
				"matrix.philox_uniform_replay",
				KernelId::MatrixPhiloxUniformReplayF32,
			),
			(
				"matrix.philox_replay_advance",
				KernelId::MatrixPhiloxReplayAdvanceU32,
			),
		])
		.expect("paired replay RNG was rejected");
		let missing = validate_training_kernel_sequence([(
			"matrix.philox_normal_replay",
			KernelId::MatrixPhiloxNormalReplayF32,
		)])
		.unwrap_err();
		assert!(missing.message().contains("without its state advance"));
		let interleaved = validate_training_kernel_sequence([
			(
				"matrix.philox_uniform_replay",
				KernelId::MatrixPhiloxUniformReplayF32,
			),
			("matrix.add", KernelId::MatrixAddF32),
			(
				"matrix.philox_replay_advance",
				KernelId::MatrixPhiloxReplayAdvanceU32,
			),
		])
		.unwrap_err();
		assert!(interleaved.message().contains("followed immediately"));
		let orphan = validate_training_kernel_sequence([(
			"matrix.philox_replay_advance",
			KernelId::MatrixPhiloxReplayAdvanceU32,
		)])
		.unwrap_err();
		assert!(orphan.message().contains("without a preceding"));
		let frozen = validate_training_kernel_sequence([(
			"matrix.philox_uniform",
			KernelId::MatrixPhiloxUniformF32,
		)])
		.unwrap_err();
		assert!(frozen.message().contains("embeds a host seed"));
	}
}
