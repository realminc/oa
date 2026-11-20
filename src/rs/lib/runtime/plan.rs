use std::sync::Arc;

use std::{
	cell::{Cell, RefCell},
	fmt,
};

use crate::{Error, Matrix, Result, core::Element};

use super::{
	Buffer, BufferAccess, Device, EngineHandle, Event, RecordedCommandBuffer, ReusableCommandBuffer,
	SemanticGraph, SemanticLoweringAnalysis, Storage,
	device::NativeFlagBits,
	dna::{DnnPlan, DnnPolicy, fusion},
	graph::executable::{AccessState, ExecutableGraph, GraphNode},
	session::{PendingExecution, SemanticStorageSnapshot, StableResourceSnapshot},
};

pub(super) mod serialization;

/// Immutable snapshot of execution-plan diagnostics.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExecutionPlanDiagnostics {
	graph_id: u64,
	device_capability_identity: u64,
	node_count: usize,
	barrier_count: usize,
	semantic_operation_count: usize,
	semantic_autograd_attachment_count: usize,
	semantic_autograd_expanded_count: usize,
	semantic_backward_operation_count: usize,
	schema_owned_node_count: usize,
	compatibility_node_count: usize,
	semantic_fused_operation_count: usize,
	semantic_fused_node_count: usize,
	maximum_semantic_operations_per_node: usize,
	dnn_graph_hash: u64,
	dnn_value_count: usize,
	dnn_external_value_count: usize,
	dnn_virtual_value_count: usize,
	dnn_partition_count: usize,
	dnn_recognized_partition_count: u32,
	dnn_portable_partition_count: usize,
	dnn_captured_operation_count: u32,
	dnn_applied_partition_count: u32,
	dnn_inherited_partition_count: u32,
	dnn_fallback_partition_count: u32,
	dnn_unexpected_fallback_count: u32,
	input_binding_count: usize,
	captured_resource_count: usize,
	physical_resource_count: usize,
	stable_replay_input_count: usize,
	stable_transient_count: usize,
	semantic_binding_count: usize,
	observed_output_count: usize,
	alias_candidate_count: usize,
	alias_materialized_count: usize,
	planned_alias_group_count: usize,
	potential_alias_savings: usize,
	materialized_alias_savings: usize,
	descriptor_write_count: u64,
	push_byte_count: usize,
	peak_memory_bytes: usize,
	cross_queue_edge_count: usize,
	ownership_transfer_count: usize,
	queue_run_count: usize,
	command_recording_count: u64,
	command_cache_hit_count: u64,
	submission_count: u64,
	input_rebinding_count: u64,
	input_upload_count: u64,
	fallback_count: u64,
}

impl ExecutionPlanDiagnostics {
	/// Return the deterministic identity of the captured executable structure.
	pub const fn graph_id(self) -> u64 {
		self.graph_id
	}

	/// Return the stable device/driver/capability identity used to prepare this plan.
	pub const fn device_capability_identity(self) -> u64 {
		self.device_capability_identity
	}

	/// Return the number of captured executable nodes.
	pub const fn node_count(self) -> usize {
		self.node_count
	}

	/// Return the number of intra-graph buffer hazards requiring barriers.
	pub const fn barrier_count(self) -> usize {
		self.barrier_count
	}

	/// Return the number of canonical semantic operations.
	pub const fn semantic_operation_count(self) -> usize {
		self.semantic_operation_count
	}

	/// Return forward outputs attached to a reverse-mode tape.
	pub const fn semantic_autograd_attachment_count(self) -> usize {
		self.semantic_autograd_attachment_count
	}

	/// Return tape attachments with a completed backward operation range.
	pub const fn semantic_autograd_expanded_count(self) -> usize {
		self.semantic_autograd_expanded_count
	}

	/// Return semantic operations owned by completed backward ranges.
	pub const fn semantic_backward_operation_count(self) -> usize {
		self.semantic_backward_operation_count
	}

	/// Return executable nodes with schema-owned semantic provenance.
	pub const fn schema_owned_node_count(self) -> usize {
		self.schema_owned_node_count
	}

	/// Return executable nodes still using the compatibility route.
	pub const fn compatibility_node_count(self) -> usize {
		self.compatibility_node_count
	}

	/// Return semantic operations represented by multi-operation executable nodes.
	pub const fn semantic_fused_operation_count(self) -> usize {
		self.semantic_fused_operation_count
	}

	/// Return executable nodes that own more than one semantic operation.
	pub const fn semantic_fused_node_count(self) -> usize {
		self.semantic_fused_node_count
	}

	/// Return the widest semantic fusion represented by one executable node.
	pub const fn maximum_semantic_operations_per_node(self) -> usize {
		self.maximum_semantic_operations_per_node
	}

	/// Return the deterministic identity of DNN planning inputs.
	pub const fn dnn_graph_hash(self) -> u64 {
		self.dnn_graph_hash
	}

	/// Return semantic values admitted into DNN analysis.
	pub const fn dnn_value_count(self) -> usize {
		self.dnn_value_count
	}

	/// Return DNN values entering from outside captured work.
	pub const fn dnn_external_value_count(self) -> usize {
		self.dnn_external_value_count
	}

	/// Return DNN values reserved for lowering-time materialization.
	pub const fn dnn_virtual_value_count(self) -> usize {
		self.dnn_virtual_value_count
	}

	/// Return the number of DNN candidate partitions.
	pub const fn dnn_partition_count(self) -> usize {
		self.dnn_partition_count
	}

	/// Return partitions recognized by a schema-owned DNN provider role.
	pub const fn dnn_recognized_partition_count(self) -> u32 {
		self.dnn_recognized_partition_count
	}

	/// Return partitions retained on the portable executable path.
	pub const fn dnn_portable_partition_count(self) -> usize {
		self.dnn_portable_partition_count
	}

	/// Return semantic operations admitted into DNN analysis.
	pub const fn dnn_captured_operation_count(self) -> u32 {
		self.dnn_captured_operation_count
	}

	/// Return provider-recognized partitions that replaced source executable nodes.
	pub const fn dnn_applied_partition_count(self) -> u32 {
		self.dnn_applied_partition_count
	}

	/// Return recognized single-node partitions already using an admitted provider.
	pub const fn dnn_inherited_partition_count(self) -> u32 {
		self.dnn_inherited_partition_count
	}

	/// Return recognized partitions deliberately retained on their source lowering.
	pub const fn dnn_fallback_partition_count(self) -> u32 {
		self.dnn_fallback_partition_count
	}

	/// Return fallback partitions caused by a violated internal lowering invariant.
	pub const fn dnn_unexpected_fallback_count(self) -> u32 {
		self.dnn_unexpected_fallback_count
	}

	/// Return the number of Matrix inputs admitted for stable rebinding.
	pub const fn input_binding_count(self) -> usize {
		self.input_binding_count
	}

	/// Return the number of physical resources present in the captured graph.
	pub const fn captured_resource_count(self) -> usize {
		self.captured_resource_count
	}

	/// Return the number of distinct physical buffers used after memory planning.
	pub const fn physical_resource_count(self) -> usize {
		self.physical_resource_count
	}

	/// Return captured resources allocated before the replay-input seal.
	pub const fn stable_replay_input_count(self) -> usize {
		self.stable_replay_input_count
	}

	/// Return captured resources allocated after the replay-input seal.
	///
	/// These resources have stable allocation ordinals and known graph lifetimes,
	/// but are not automatically eligible for physical aliasing.
	pub const fn stable_transient_count(self) -> usize {
		self.stable_transient_count
	}

	/// Return handle-free semantic value-to-resource bindings in the plan.
	pub const fn semantic_binding_count(self) -> usize {
		self.semantic_binding_count
	}

	/// Return captured resources explicitly retained as observable results.
	pub const fn observed_output_count(self) -> usize {
		self.observed_output_count
	}

	/// Return stable transients whose lifetimes and storage ownership are fully
	/// explained and which are not externally live.
	pub const fn alias_candidate_count(self) -> usize {
		self.alias_candidate_count
	}

	/// Return logical resources rebound into a graph-lifetime alias arena.
	pub const fn alias_materialized_count(self) -> usize {
		self.alias_materialized_count
	}

	/// Return the number of nontrivial alias groups found by memory planning.
	pub const fn planned_alias_group_count(self) -> usize {
		self.planned_alias_group_count
	}

	/// Return logical bytes removable if every planned alias group materializes.
	pub const fn potential_alias_savings(self) -> usize {
		self.potential_alias_savings
	}

	/// Return logical bytes removed by materialized alias arenas.
	pub const fn materialized_alias_savings(self) -> usize {
		self.materialized_alias_savings
	}

	/// Return how many Vulkan command buffers were recorded for this plan.
	pub const fn command_recording_count(self) -> u64 {
		self.command_recording_count
	}

	/// Return descriptor writes issued while this plan was captured.
	///
	/// On the Core profile binding amortizes at buffer/image creation, so a
	/// plan whose inputs were already bound reports zero — the bindless
	/// recording-cost fact OA-CC-21 measures.
	pub const fn descriptor_write_count(self) -> u64 {
		self.descriptor_write_count
	}

	/// Return total push-constant bytes recorded per submission of this plan.
	pub const fn push_byte_count(self) -> usize {
		self.push_byte_count
	}

	/// Return peak live physical-buffer bytes across the plan's node order.
	///
	/// Interval sweep over captured resource lifetimes: aliased transients
	/// share one physical buffer identity, so the peak counts each live
	/// allocation once — the memory-planning number OA-CC-23 minimizes.
	pub const fn peak_memory_bytes(self) -> usize {
		self.peak_memory_bytes
	}

	/// Derived cross-queue timeline edges: hazards whose source and
	/// destination nodes submit to different queue families (OA-CC-22).
	pub const fn cross_queue_edge_count(self) -> usize {
		self.cross_queue_edge_count
	}

	/// Derived queue-family ownership transfers required by cross-queue
	/// hazards on exclusive-owned buffers.
	pub const fn ownership_transfer_count(self) -> usize {
		self.ownership_transfer_count
	}

	/// Ordered queue runs the plan splits into by resolved queue family
	/// (OA-CC-22). One run submits on the compute family today; a mixed
	/// plan produces one timeline-chained submission per run.
	pub const fn queue_run_count(self) -> usize {
		self.queue_run_count
	}

	/// Return how many submissions reused an already recorded command buffer.
	pub const fn command_cache_hit_count(self) -> u64 {
		self.command_cache_hit_count
	}

	/// Return how many plan submissions completed queue admission.
	pub const fn submission_count(self) -> u64 {
		self.submission_count
	}

	/// Return how many stable Matrix input bindings changed.
	pub const fn input_rebinding_count(self) -> u64 {
		self.input_rebinding_count
	}

	/// Return how many stable Matrix input slots received new host values.
	pub const fn input_upload_count(self) -> u64 {
		self.input_upload_count
	}

	/// Return the number of runtime fallback decisions made by this plan.
	pub const fn fallback_count(self) -> u64 {
		self.fallback_count
	}
}

/// One buffer position in a stable input slot's triple-buffered staging ring.
///
/// `consumer` records the most recent submission event that reads this buffer;
/// an upload may write it only after that event completes.
struct InputStagingSlot {
	buffer: Buffer,
	consumer: RefCell<Option<Event>>,
}

struct MatrixInputBinding {
	original: Buffer,
	current: Buffer,
	/// Ring of staging buffers. Slot zero is the captured input itself; further
	/// slots are lazily allocated up to the staging depth.
	staging: Vec<InputStagingSlot>,
}

/// Handle-free lifetime and stable-frame evidence for one captured resource.
///
/// Resource IDs follow deterministic graph-resource order and never expose a
/// Vulkan handle or bindless descriptor. Alias-candidate state records completed
/// liveness, external-output, and storage-owner qualification; physical storage
/// is not shared until a later materialization stage succeeds transactionally.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CapturedResourceDesc {
	resource: u32,
	byte_len: usize,
	first_access: usize,
	last_access: usize,
	stable_replay_input: bool,
	stable_transient: bool,
	semantic_external: bool,
	observed_output: bool,
	capture_retained_owner_count: usize,
	unaccounted_owner_count: usize,
	alias_candidate: bool,
	alias_materialized: bool,
}

impl CapturedResourceDesc {
	pub const fn resource(self) -> u32 {
		self.resource
	}

	pub const fn byte_len(self) -> usize {
		self.byte_len
	}

	pub const fn first_access(self) -> usize {
		self.first_access
	}

	pub const fn last_access(self) -> usize {
		self.last_access
	}

	pub const fn stable_replay_input(self) -> bool {
		self.stable_replay_input
	}

	pub const fn stable_transient(self) -> bool {
		self.stable_transient
	}

	pub const fn semantic_external(self) -> bool {
		self.semantic_external
	}

	pub const fn observed_output(self) -> bool {
		self.observed_output
	}

	/// Return whether a semantic input, replay input, or observed output keeps
	/// this resource externally live across plan execution.
	pub const fn is_externally_live(self) -> bool {
		self.semantic_external || self.stable_replay_input || self.observed_output
	}

	/// Return storage owners retained by the capture transaction itself.
	pub const fn capture_retained_owner_count(self) -> usize {
		self.capture_retained_owner_count
	}

	/// Return storage owners not explained by capture-owned structures.
	pub const fn unaccounted_owner_count(self) -> usize {
		self.unaccounted_owner_count
	}

	/// Return whether this resource passed liveness and ownership qualification.
	///
	/// This is planning evidence only; it does not mean alias storage was
	/// materialized.
	pub const fn alias_candidate(self) -> bool {
		self.alias_candidate
	}

	pub const fn alias_materialized(self) -> bool {
		self.alias_materialized
	}
}

/// Position of one compiled specialization contract entry.
///
/// Donor: `oa::ExecutionBindingRole` in
/// `oacpp/source/cpp/include/oa/runtime/executionPlan.h`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExecutionBindingRole {
	/// Unproduced semantic value consumed by the captured graph.
	Input,
	/// Produced semantic value not consumed by any captured operation.
	Output,
}

/// Handle-free boundary contract for one exact compiled specialization.
///
/// A binding describes selection/validation only: plans retain their captured
/// resources and validation never rebinds them. Rust matrices are dense, so a
/// specialization candidate's implied strides are its dense strides.
///
/// Donor: `oa::ExecutionBindingDesc`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExecutionBindingDesc {
	slot: u32,
	semantic_value: super::SemanticValueId,
	kind: crate::OpValueKind,
	role: ExecutionBindingRole,
	shape: Vec<usize>,
	strides: Vec<isize>,
	dtype: crate::DType,
	byte_offset: u64,
}

impl ExecutionBindingDesc {
	/// Construct one contract entry. Field order mirrors the donor aggregate.
	#[allow(clippy::too_many_arguments)]
	pub(super) fn new(
		slot: u32,
		semantic_value: super::SemanticValueId,
		kind: crate::OpValueKind,
		role: ExecutionBindingRole,
		shape: Vec<usize>,
		strides: Vec<isize>,
		dtype: crate::DType,
		byte_offset: u64,
	) -> Self {
		Self {
			slot,
			semantic_value,
			kind,
			role,
			shape,
			strides,
			dtype,
			byte_offset,
		}
	}

	/// Return the position of this entry inside the plan contract.
	pub const fn slot(&self) -> u32 {
		self.slot
	}

	/// Return the semantic value described by this entry.
	pub const fn semantic_value(&self) -> super::SemanticValueId {
		self.semantic_value
	}

	/// Return the semantic value kind.
	pub const fn kind(&self) -> crate::OpValueKind {
		self.kind
	}

	/// Return whether this entry is an input or an output.
	pub const fn role(&self) -> ExecutionBindingRole {
		self.role
	}

	/// Return the required logical shape.
	pub fn shape(&self) -> &[usize] {
		&self.shape
	}

	/// Return the required element strides.
	pub fn strides(&self) -> &[isize] {
		&self.strides
	}

	/// Return the required scalar dtype.
	pub const fn dtype(&self) -> crate::DType {
		self.dtype
	}

	/// Return the required absolute byte offset inside its storage.
	pub const fn byte_offset(&self) -> u64 {
		self.byte_offset
	}
}

/// Deterministic bridge from one semantic value to a captured resource.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SemanticStorageBinding {
	value: super::SemanticValueId,
	resource: u32,
	semantic_external: bool,
	stable_replay_input: bool,
	observed_output: bool,
}

impl SemanticStorageBinding {
	pub const fn value(self) -> super::SemanticValueId {
		self.value
	}

	pub const fn resource(self) -> u32 {
		self.resource
	}

	pub const fn semantic_external(self) -> bool {
		self.semantic_external
	}

	pub const fn stable_replay_input(self) -> bool {
		self.stable_replay_input
	}

	pub const fn observed_output(self) -> bool {
		self.observed_output
	}
}

const MAX_CACHED_BINDING_COMMANDS: usize = 8;
const STABLE_INPUT_STAGING_DEPTH: usize = 3;

#[derive(Default)]
struct PlanCounters {
	command_recordings: Cell<u64>,
	command_cache_hits: Cell<u64>,
	submissions: Cell<u64>,
	input_rebindings: Cell<u64>,
	input_uploads: Cell<u64>,
	fallbacks: Cell<u64>,
}

/// Engine-associated executable work captured for repeated submission.
///
/// The captured executable structure is immutable. Read-only Matrix inputs may
/// be rebound to shape- and dtype-identical storage while preserving the graph
/// identity. Replays cache one recorded Vulkan command per binding set, so
/// rotating through a bounded set of input buffers reuses the recorded commands
/// after the first submission of each set.
#[must_use]
pub struct ExecutionPlan {
	engine: EngineHandle,
	graph_id: u64,
	device_capability_identity: u64,
	graph: ExecutableGraph,
	outputs: Vec<Storage>,
	semantic: SemanticGraph,
	semantic_lowering: SemanticLoweringAnalysis,
	dnn: DnnPlan,
	input_bindings: Vec<MatrixInputBinding>,
	bindings: Vec<ExecutionBindingDesc>,
	binding_contract_hash: u64,
	captured_resources: Vec<CapturedResourceDesc>,
	semantic_bindings: Vec<SemanticStorageBinding>,
	descriptor_write_count: u64,
	materialized_alias_savings: usize,
	alias_materialization_fallback_reason: Option<String>,
	cached_commands: RefCell<Vec<(Vec<u32>, Vec<ReusableCommandBuffer>)>>,
	last_submission: RefCell<Option<Event>>,
	counters: PlanCounters,
}

impl ExecutionPlan {
	pub(super) fn new(
		engine: EngineHandle,
		pending: PendingExecution,
		source_recording_retained: bool,
	) -> Result<Self> {
		let (mut graph, mut outputs, semantic, stable_resources, semantic_storage, observed_outputs) =
			pending.into_parts();
		let device_capability_identity = engine.device_info().capability_identity();
		let mut dnn = DnnPlan::plan(&semantic, DnnPolicy::default())?;
		if dnn.source_operation_count() as usize != semantic.operations().len() {
			return Err(Error::internal(
				"DNN plan source operation count does not match the semantic graph",
			));
		}
		let (bindings, binding_contract_hash) = dnn.build_binding_contract(&semantic)?;
		let input_bindings = graph
			.read_only_buffers()
			.into_iter()
			.map(|buffer| MatrixInputBinding {
				original: buffer.clone(),
				current: buffer.clone(),
				staging: vec![InputStagingSlot {
					buffer,
					consumer: RefCell::new(None),
				}],
			})
			.collect();
		let mut resource_lifetimes = graph.resource_lifetimes();
		// Composite semantic operations may expose a zero-copy input as an output
		// without any physical dispatch reading it. Retain that semantic-only
		// resource in the plan inventory even though it has no GPU access edge.
		// This keeps the semantic graph complete without manufacturing a shader
		// read or weakening storage identity checks.
		for binding in &semantic_storage {
			let Some(buffer) = binding.storage.buffer() else {
				continue;
			};
			if !resource_lifetimes
				.iter()
				.any(|lifetime| lifetime.buffer.same_as(buffer))
			{
				resource_lifetimes.push(super::graph::executable::ResourceLifetime {
					buffer: buffer.clone(),
					first_access: 0,
					last_access: 0,
				});
			}
		}
		let mut captured_resources = resource_lifetimes
			.iter()
			.enumerate()
			.map(|(resource, lifetime)| {
				let stable = stable_resources.iter().find(|candidate| {
					candidate
						.storage
						.buffer()
						.is_some_and(|buffer| buffer.same_as(&lifetime.buffer))
				});
				let owner = find_resource_storage(
					&lifetime.buffer,
					&stable_resources,
					&semantic_storage,
					&observed_outputs,
					&outputs,
				);
				let stable_owner_count = stable_resources
					.iter()
					.filter(|candidate| owner.is_some_and(|owner| candidate.storage.same_as(owner)))
					.count()
					.saturating_mul(2);
				let source_copy_count = usize::from(source_recording_retained);
				let capture_retained_owner_count = stable_owner_count
					.saturating_add(
						semantic_storage
							.iter()
							.filter(|candidate| owner.is_some_and(|owner| candidate.storage.same_as(owner)))
							.count()
							.saturating_mul(1 + source_copy_count),
					)
					.saturating_add(
						observed_outputs
							.iter()
							.filter(|candidate| owner.is_some_and(|owner| candidate.same_as(owner)))
							.count(),
					)
					.saturating_add(
						outputs
							.iter()
							.filter(|candidate| owner.is_some_and(|owner| candidate.same_as(owner)))
							.count()
							.saturating_mul(1 + source_copy_count),
					);
				let unaccounted_owner_count = owner.map_or(0, |owner| {
					owner
						.owner_count()
						.saturating_sub(capture_retained_owner_count)
				});
				Ok(CapturedResourceDesc {
					resource: u32::try_from(resource)
						.map_err(|_| Error::resource_exhausted("captured resource identity exceeds u32"))?,
					byte_len: lifetime.buffer.byte_len(),
					first_access: lifetime.first_access,
					last_access: lifetime.last_access,
					stable_replay_input: stable.is_some_and(|value| value.replay_input),
					stable_transient: stable.is_some_and(|value| value.transient),
					semantic_external: false,
					observed_output: false,
					capture_retained_owner_count,
					unaccounted_owner_count,
					alias_candidate: false,
					alias_materialized: false,
				})
			})
			.collect::<Result<Vec<_>>>()?;
		let mut semantic_bindings = Vec::with_capacity(semantic_storage.len());
		for binding in &semantic_storage {
			let value = semantic
				.values()
				.get(binding.value.index() as usize)
				.ok_or_else(|| Error::internal("semantic storage binding value is out of range"))?;
			let resource = find_resource(&resource_lifetimes, &binding.storage)?;
			let resource_id = u32::try_from(resource)
				.map_err(|_| Error::resource_exhausted("captured resource identity exceeds u32"))?;
			captured_resources[resource].semantic_external |= value.is_external();
			semantic_bindings.push(SemanticStorageBinding {
				value: binding.value,
				resource: resource_id,
				semantic_external: value.is_external(),
				stable_replay_input: captured_resources[resource].stable_replay_input,
				observed_output: false,
			});
		}
		for output in &observed_outputs {
			let resource = find_resource(&resource_lifetimes, output)?;
			captured_resources[resource].observed_output = true;
			for binding in &mut semantic_bindings {
				if binding.resource as usize == resource {
					binding.observed_output = true;
				}
			}
		}
		graph = dnn.lower_captured_plan(
			&engine,
			&semantic,
			&semantic_bindings,
			&resource_lifetimes,
			&captured_resources,
			!source_recording_retained,
			&graph,
		)?;
		engine.record_dnn_partitions(
			dnn.applied_partition_count(),
			dnn.fallback_partition_count(),
		);
		let graph_id = graph.identity();
		let semantic_lowering = graph.analyze_semantic_lowering(&semantic)?;
		// OA-CC-23: reusable storage must first be fully written. For buffers
		// outside the stable transient set, a final read also bounds the
		// lifetime to this replay.
		let mut first_access: std::collections::BTreeMap<u32, BufferAccess> =
			std::collections::BTreeMap::new();
		let mut last_access: std::collections::BTreeMap<u32, BufferAccess> =
			std::collections::BTreeMap::new();
		for node in graph.nodes() {
			if let Some(indirect) = node.indirect() {
				let identity = indirect.buffer.descriptor_index();
				first_access.entry(identity).or_insert(BufferAccess::Read);
				last_access.insert(identity, BufferAccess::Read);
			}
			for usage in node.buffers() {
				let identity = usage.buffer.descriptor_index();
				first_access.entry(identity).or_insert(usage.access);
				last_access.insert(identity, usage.access);
			}
		}
		for (index, resource) in captured_resources.iter_mut().enumerate() {
			let buffer = &resource_lifetimes[index].buffer;
			let identity = buffer.descriptor_index();
			// A first read or partial write depends on the original contents.
			resource.alias_candidate = matches!(first_access.get(&identity), Some(BufferAccess::Write))
				&& (resource.stable_transient
					|| matches!(last_access.get(&identity), Some(BufferAccess::Read)))
				&& !resource.is_externally_live()
				&& resource.unaccounted_owner_count == 0
				&& !fusion::buffer_is_fused_binding(graph.compute_nodes(), buffer);
		}
		let materialization = materialize_aliases(
			&engine,
			&graph,
			&outputs,
			&resource_lifetimes,
			&captured_resources,
		);
		let (retired, materialized_alias_savings, alias_materialization_fallback_reason) =
			match materialization {
				Ok(materialized) => {
					graph = materialized.graph;
					outputs = materialized.outputs;
					captured_resources = materialized.captured_resources;
					(materialized.retired, materialized.savings, None)
				}
				Err(error) => (
					Vec::new(),
					0,
					Some(format!("transient alias materialization failed: {error}")),
				),
			};
		engine.release_stable_transient_resources(&retired);
		let initial_fallback_count = u64::from(alias_materialization_fallback_reason.is_some())
			.saturating_add(u64::from(dnn.fallback_partition_count()));
		Ok(Self {
			engine,
			graph_id,
			device_capability_identity,
			graph,
			outputs,
			semantic,
			semantic_lowering,
			dnn,
			input_bindings,
			bindings,
			binding_contract_hash,
			captured_resources,
			semantic_bindings,
			descriptor_write_count: 0,
			materialized_alias_savings,
			alias_materialization_fallback_reason,
			cached_commands: RefCell::new(Vec::new()),
			last_submission: RefCell::new(None),
			counters: PlanCounters {
				fallbacks: Cell::new(initial_fallback_count),
				..PlanCounters::default()
			},
		})
	}

	/// Record the capture-time descriptor-write count the guard measured.
	pub(super) fn set_descriptor_write_count(&mut self, count: u64) {
		self.descriptor_write_count = count;
	}

	/// Return the canonical backend-independent graph captured by this plan.
	pub const fn semantic_graph(&self) -> &SemanticGraph {
		&self.semantic
	}

	/// Return the validated semantic-to-executable lowering provenance.
	pub const fn semantic_lowering(&self) -> &SemanticLoweringAnalysis {
		&self.semantic_lowering
	}

	/// Return deterministic physical-resource lifetime and stable-frame evidence.
	pub fn captured_resources(&self) -> &[CapturedResourceDesc] {
		&self.captured_resources
	}

	/// Return the handle-free semantic value-to-resource capture table.
	pub fn semantic_bindings(&self) -> &[SemanticStorageBinding] {
		&self.semantic_bindings
	}

	/// Return why transient alias materialization fell back to distinct storage.
	///
	/// The plan remains executable when this returns `Some`; diagnostics count the
	/// decision as one fallback and report zero materialized alias savings.
	pub fn alias_materialization_fallback_reason(&self) -> Option<&str> {
		self.alias_materialization_fallback_reason.as_deref()
	}

	/// Upload values into one stable captured Matrix input slot.
	///
	/// The captured shape, dtype, descriptor slot, and graph identity remain
	/// unchanged. The slot rotates through a bounded triple-buffered staging
	/// ring: each upload writes the next ring buffer and rebinds it, so a
	/// pending prior replay stalls only when the target ring buffer is still
	/// consumed by an in-flight submission. The captured Matrix's own buffer
	/// mirrors each upload, so the stable value remains coherent for host
	/// reads.
	///
	/// # Errors
	///
	/// Returns an error when `captured` is not a ring-bound read-only plan
	/// input, the element type or count differs, or prior completion/upload
	/// fails.
	pub fn upload_matrix_input<T: Element>(&mut self, captured: &Matrix, values: &[T]) -> Result<()> {
		if !self.engine.same_as(captured.engine_handle()) {
			return Err(Error::invalid_argument(
				"execution-plan Matrix input belongs to another engine",
			));
		}
		let captured_buffer = captured.storage().buffer().ok_or_else(|| {
			Error::invalid_argument("zero-sized Matrix storage cannot identify a captured plan input")
		})?;
		let binding_index = self
			.input_bindings
			.iter()
			.position(|binding| binding.original.same_as(captured_buffer))
			.ok_or_else(|| {
				Error::invalid_argument("captured Matrix is not a read-only input of this execution plan")
			})?;
		if captured.dtype() != T::DTYPE {
			return Err(Error::invalid_argument(format!(
				"matrix element type {} does not match dtype {}",
				T::DTYPE.token(),
				captured.dtype().token()
			)));
		}
		if values.len() != captured.element_count() {
			return Err(Error::invalid_argument(format!(
				"matrix upload requires {} elements; received {}",
				captured.element_count(),
				values.len()
			)));
		}
		let binding = &mut self.input_bindings[binding_index];
		let current_slot = binding
			.staging
			.iter()
			.position(|slot| slot.buffer.same_as(&binding.current))
			.ok_or_else(|| {
				Error::failed_precondition(
					"execution-plan host upload requires a ring-bound stable input slot",
				)
			})?;
		let next_slot = (current_slot + 1) % STABLE_INPUT_STAGING_DEPTH;
		if next_slot == binding.staging.len() {
			let buffer =
				Buffer::host_visible_storage(binding.original.device(), binding.original.byte_len())?;
			binding.staging.push(InputStagingSlot {
				buffer,
				consumer: RefCell::new(None),
			});
		}
		let slot = &binding.staging[next_slot];
		if let Some(event) = slot.consumer.borrow().as_ref() {
			event.wait()?;
		}
		// Both branches below write the captured buffer on the host: directly
		// when the ring lands on it, or through the mirror update. Other plans
		// sharing this Matrix may still have an in-flight submission bound to
		// it, which this plan's per-slot consumers cannot observe.
		binding.original.wait_bound_consumer()?;
		if slot.buffer.same_as(&binding.original) {
			captured.write_values(values)?;
		} else {
			let byte_len = values
				.len()
				.checked_mul(std::mem::size_of::<T>())
				.ok_or_else(|| Error::invalid_argument("matrix byte size overflows usize"))?;
			// SAFETY: `Element` is sealed to initialized, no-padding OA scalar
			// types and the checked length covers exactly the supplied slice.
			let bytes = unsafe { std::slice::from_raw_parts(values.as_ptr().cast(), byte_len) };
			if T::DTYPE == crate::DType::U8 {
				let mut padded = vec![0_u8; crate::matrix::byte_storage_len(byte_len, T::DTYPE)?];
				padded[..byte_len].copy_from_slice(bytes);
				slot.buffer.write(0, &padded)?;
			} else {
				slot.buffer.write(0, bytes)?;
			}
			// Mirror the newest upload into the captured Matrix's buffer so the
			// stable value stays coherent for host reads while the ring
			// pipelines device reads. The buffer-level wait above covered every
			// plan's last bound submission, so this wait is bounded by the ring
			// depth.
			captured.write_values(values)?;
		}
		if !binding.current.same_as(&slot.buffer) {
			self.graph.rebind_read_only(
				&binding.current,
				&slot.buffer,
				self.engine.execution_profile(),
			)?;
			binding.current = slot.buffer.clone();
		}
		increment(&self.counters.input_uploads);
		Ok(())
	}

	/// Rebind one read-only Matrix input while preserving its captured contract.
	///
	/// `captured` remains the stable slot identity even after repeated rebinding.
	/// The replacement must have the same shape and dtype, belong to the plan's
	/// engine, and not alias another resource already used by the plan. Rebinding
	/// does not submit or wait; the next submission records one command buffer per
	/// unseen binding set and later submissions of the same set reuse it.
	///
	/// # Errors
	///
	/// Returns an error when either Matrix belongs to another engine, their shape
	/// or dtype differs, `captured` is not a read-only plan input, the replacement
	/// aliases another plan resource, or its production state is invalid.
	pub fn bind_matrix_input(&mut self, captured: &Matrix, replacement: &Matrix) -> Result<()> {
		if !self.engine.same_as(captured.engine_handle())
			|| !self.engine.same_as(replacement.engine_handle())
		{
			return Err(Error::invalid_argument(
				"execution-plan Matrix bindings must belong to the plan's engine",
			));
		}
		if captured.shape() != replacement.shape() || captured.dtype() != replacement.dtype() {
			return Err(Error::invalid_argument(format!(
				"execution-plan Matrix input requires shape {:?} and dtype {}; replacement has shape {:?} and dtype {}",
				captured.shape(),
				captured.dtype().token(),
				replacement.shape(),
				replacement.dtype().token()
			)));
		}
		replacement
			.storage()
			.validate_recording_access(BufferAccess::Read)?;
		let captured_buffer = captured.storage().buffer().ok_or_else(|| {
			Error::invalid_argument("zero-sized Matrix storage cannot identify a captured plan input")
		})?;
		let replacement_buffer = replacement.storage().buffer().ok_or_else(|| {
			Error::invalid_argument("zero-sized Matrix storage cannot replace a captured plan input")
		})?;

		let existing = self
			.input_bindings
			.iter()
			.position(|binding| binding.original.same_as(captured_buffer))
			.ok_or_else(|| {
				Error::invalid_argument("captured Matrix is not a read-only input of this execution plan")
			})?;
		let current = self.input_bindings[existing].current.clone();
		if current.same_as(replacement_buffer) {
			return Ok(());
		}
		self.graph.rebind_read_only(
			&current,
			replacement_buffer,
			self.engine.execution_profile(),
		)?;
		self.input_bindings[existing].current = replacement_buffer.clone();
		increment(&self.counters.input_rebindings);
		Ok(())
	}

	/// Return the number of entries in the compiled specialization contract.
	///
	/// Donor: `oa::ExecutionPlan::bindingCount`.
	pub const fn binding_count(&self) -> usize {
		self.bindings.len()
	}

	/// Return the number of contract entries with an input role.
	///
	/// Donor: `oa::ExecutionPlan::inputBindingCount`.
	pub fn input_binding_count(&self) -> usize {
		self
			.bindings
			.iter()
			.filter(|binding| binding.role() == ExecutionBindingRole::Input)
			.count()
	}

	/// Return the number of contract entries with an output role.
	///
	/// Donor: `oa::ExecutionPlan::outputBindingCount`.
	pub fn output_binding_count(&self) -> usize {
		self.bindings.len() - self.input_binding_count()
	}

	/// Return the contract entry at `slot`.
	///
	/// Donor: `oa::ExecutionPlan::binding`.
	///
	/// # Errors
	///
	/// Returns `InvalidArgument` when `slot` is out of range.
	pub fn binding(&self, slot: u32) -> Result<ExecutionBindingDesc> {
		self
			.bindings
			.get(slot as usize)
			.cloned()
			.ok_or_else(|| Error::invalid_argument("execution plan binding slot is out of range"))
	}

	/// Return the deterministic hash of the compiled specialization contract.
	///
	/// Donor: `oa::ExecutionPlan::bindingContractHash`.
	pub const fn binding_contract_hash(&self) -> u64 {
		self.binding_contract_hash
	}

	/// Validate specialization input matrices against the compiled contract.
	///
	/// Validation checks the exact shape, dtype, strides, and byte offset of
	/// each contract input in order. It deliberately does not rebind resources:
	/// dynamic dimensions select or recapture another checked plan rather than
	/// mutating this plan's recorded command buffers.
	///
	/// Donor: `oa::ExecutionPlan::validateSpecializationInputs`.
	///
	/// # Errors
	///
	/// Returns `InvalidArgument` when the input count differs, and
	/// `FailedPrecondition` when any input's contract requires another plan or
	/// is not a matrix.
	pub fn validate_specialization_inputs(&self, inputs: &[crate::Matrix]) -> Result<()> {
		if inputs.len() != self.input_binding_count() {
			return Err(Error::invalid_argument(
				"specialization input count does not match the compiled plan",
			));
		}
		let mut input_index = 0_usize;
		for binding in &self.bindings {
			if binding.role() != ExecutionBindingRole::Input {
				continue;
			}
			if binding.kind() != crate::OpValueKind::Matrix {
				return Err(Error::failed_precondition(
					"matrix specialization validation cannot bind a non-Matrix input",
				));
			}
			let input = &inputs[input_index];
			input_index += 1;
			let absolute_offset = {
				let mut offset = 0_i64;
				let mut semantic = input.semantic().clone();
				loop {
					offset = offset
						.checked_add(semantic.view_byte_offset())
						.ok_or_else(|| Error::internal("matrix view byte offset overflows i64"))?;
					match semantic.view_source() {
						Some(source) => semantic = source.clone(),
						None => break,
					}
				}
				offset
			};
			if input.shape() != binding.shape()
				|| input.dtype() != binding.dtype()
				|| absolute_offset < 0
				|| absolute_offset as u64 != binding.byte_offset()
			{
				return Err(Error::failed_precondition(
					"specialization input shape, dtype or offset requires another plan",
				));
			}
			if super::graph::semantic::dense_strides(input.shape())? != binding.strides() {
				return Err(Error::failed_precondition(
					"specialization input strides require another plan",
				));
			}
		}
		Ok(())
	}

	/// Return current graph, cache, submission, rebinding, and fallback counters.
	pub fn diagnostics(&self) -> ExecutionPlanDiagnostics {
		let (planned_alias_group_count, potential_alias_savings) =
			alias_analysis(&self.captured_resources);
		ExecutionPlanDiagnostics {
			graph_id: self.graph_id,
			device_capability_identity: self.device_capability_identity,
			node_count: self.graph.nodes().len(),
			barrier_count: self.graph.barrier_count(),
			cross_queue_edge_count: self.graph.cross_queue_edge_count(),
			ownership_transfer_count: self.graph.ownership_transfer_count(),
			queue_run_count: self.graph.queue_runs().len(),
			semantic_operation_count: self.semantic.operations().len(),
			semantic_autograd_attachment_count: self.semantic.autograd().len(),
			semantic_autograd_expanded_count: self
				.semantic
				.autograd()
				.iter()
				.filter(|entry| entry.is_backward_expanded())
				.count(),
			semantic_backward_operation_count: self
				.semantic
				.autograd()
				.iter()
				.map(|entry| entry.backward_op_count())
				.sum(),
			schema_owned_node_count: self.semantic_lowering.schema_owned_node_count() as usize,
			compatibility_node_count: self.semantic_lowering.compatibility_node_count() as usize,
			semantic_fused_operation_count: self.semantic_lowering.fused_op_count() as usize,
			semantic_fused_node_count: self.semantic_lowering.fused_node_count() as usize,
			maximum_semantic_operations_per_node: self.semantic_lowering.maximum_ops_per_node() as usize,
			dnn_graph_hash: self.dnn.graph_hash(),
			dnn_value_count: self.dnn.value_count(),
			dnn_external_value_count: self.dnn.external_value_count(),
			dnn_virtual_value_count: self.dnn.virtual_value_count(),
			dnn_partition_count: self.dnn.partition_count(),
			dnn_recognized_partition_count: self.dnn.recognized_partition_count(),
			dnn_portable_partition_count: self.dnn.portable_partition_count(),
			dnn_captured_operation_count: self.dnn.captured_operation_count(),
			dnn_applied_partition_count: self.dnn.applied_partition_count(),
			dnn_inherited_partition_count: self.dnn.inherited_partition_count(),
			dnn_fallback_partition_count: self.dnn.fallback_partition_count(),
			dnn_unexpected_fallback_count: self.dnn.unexpected_fallback_count(),
			input_binding_count: self.input_bindings.len(),
			captured_resource_count: self.captured_resources.len(),
			physical_resource_count: self.graph.resource_lifetimes().len(),
			stable_replay_input_count: self
				.captured_resources
				.iter()
				.filter(|resource| resource.stable_replay_input)
				.count(),
			stable_transient_count: self
				.captured_resources
				.iter()
				.filter(|resource| resource.stable_transient)
				.count(),
			semantic_binding_count: self.semantic_bindings.len(),
			observed_output_count: self
				.captured_resources
				.iter()
				.filter(|resource| resource.observed_output)
				.count(),
			alias_candidate_count: self
				.captured_resources
				.iter()
				.filter(|resource| resource.alias_candidate)
				.count(),
			alias_materialized_count: self
				.captured_resources
				.iter()
				.filter(|resource| resource.alias_materialized)
				.count(),
			planned_alias_group_count,
			potential_alias_savings,
			materialized_alias_savings: self.materialized_alias_savings,
			descriptor_write_count: self.descriptor_write_count,
			peak_memory_bytes: {
				let mut lifetimes = self.graph.resource_lifetimes();
				lifetimes.sort_by_key(|lifetime| lifetime.first_access);
				let mut active: Vec<(usize, u32, usize)> = Vec::new();
				let mut peak = 0_usize;
				for lifetime in lifetimes {
					active.retain(|(last, _, _)| *last >= lifetime.first_access);
					let index = lifetime.buffer.descriptor_index();
					if !active.iter().any(|(_, live, _)| *live == index) {
						active.push((lifetime.last_access, index, lifetime.buffer.byte_len()));
					}
					peak = peak.max(active.iter().map(|(_, _, bytes)| *bytes).sum::<usize>());
				}
				peak
			},
			push_byte_count: self
				.graph
				.compute_nodes()
				.map(|node| node.push_constants.len() * 4)
				.sum(),
			command_recording_count: self.counters.command_recordings.get(),
			command_cache_hit_count: self.counters.command_cache_hits.get(),
			submission_count: self.counters.submissions.get(),
			input_rebinding_count: self.counters.input_rebindings.get(),
			input_upload_count: self.counters.input_uploads.get(),
			fallback_count: self.counters.fallbacks.get(),
		}
	}

	pub(crate) fn dnn_fallback_reasons(&self) -> impl Iterator<Item = &str> {
		self.dnn.fallback_reasons()
	}

	/// Return whether the most recently submitted replay has completed.
	///
	/// A plan that has not been submitted is complete.
	///
	/// # Errors
	///
	/// Returns an error when Vulkan cannot query the originating timeline.
	pub fn is_complete(&self) -> Result<bool> {
		let event = self.last_submission.borrow().clone();
		event.map_or(Ok(true), |event| event.is_complete())
	}

	/// Wait for the most recently submitted replay and its host retirement.
	///
	/// Same-queue timeline order means completion of the latest replay also proves
	/// completion of every earlier replay submitted through this plan.
	///
	/// # Errors
	///
	/// Returns an error when Vulkan completion or host retirement waiting fails.
	pub fn wait(&self) -> Result<()> {
		let event = self.last_submission.borrow().clone();
		if let Some(event) = event {
			event.wait()?;
		}
		Ok(())
	}

	/// Wait for pending replay and consume the plan, releasing retained state.
	///
	/// This is the Rust ownership equivalent of the donor's mutable `reset`: plan
	/// capture constructs a new value rather than repopulating an empty shell.
	/// Destruction remains non-blocking when this explicit boundary is not used.
	///
	/// # Errors
	///
	/// Returns an error from [`ExecutionPlan::wait`]. The consumed plan still drops
	/// safely on error; submitted resources remain owned by engine retirement.
	pub fn reset(self) -> Result<()> {
		self.wait()
	}

	/// Return the deterministic normalized executable graph report as JSON.
	///
	/// Resource and hazard-domain IDs follow first graph access. Vulkan handles,
	/// device addresses, descriptor indices, and push payloads are omitted.
	pub fn debug_report_json(&self, name: &str) -> String {
		let diagnostics = self.diagnostics();
		let lifetimes = self.graph.resource_lifetimes();
		let compiled = self
			.cached_commands
			.borrow()
			.iter()
			.any(|(cached_key, _)| *cached_key == self.binding_key());
		let last_event = self.last_submission.borrow().clone();
		let mut output = String::new();
		output.push_str("{\n  \"schema\": \"oa.execution_graph.v3\",\n  \"name\": ");
		crate::core::push_json_string(&mut output, name);
		crate::core::push_format(
			&mut output,
			format_args!(
				",\n  \"compiled\": {},\n  \"completion\": {{\"submitted\": {}, \"timeline_value\": {}}},\n",
				compiled,
				last_event.is_some(),
				last_event.as_ref().map_or(0, Event::epoch),
			),
		);
		let mut war_barriers = 0;
		let mut indirect_barriers = 0;
		for barrier in (0..self.graph.nodes().len()).flat_map(|node| self.graph.barriers_before(node)) {
			war_barriers += usize::from(!barrier.source.writes() && barrier.destination.writes());
			indirect_barriers += usize::from(barrier.destination.uses_indirect_commands());
		}
		let dispatches = self.graph.compute_nodes().count();
		let copies = self
			.graph
			.nodes()
			.iter()
			.filter(|node| matches!(node, GraphNode::BufferCopy(_)))
			.count();
		let video_commands = diagnostics.node_count() - dispatches - copies;
		let buffer_bytes = lifetimes.iter().fold(0_usize, |total, lifetime| {
			total.saturating_add(lifetime.buffer.byte_len())
		});
		crate::core::push_format(
			&mut output,
			format_args!(
				"  \"stats\": {{\"dispatches\": {dispatches}, \"barriers\": {}, \"war_barriers\": {}, \"indirect_barriers\": {}, \"alias_barriers\": 0, \"host_barriers\": 0, \"descriptor_sets\": 0, \"kernel_selections\": {dispatches}, \"kernel_fallbacks\": 0, \"precision_fallbacks\": 0, \"layout_fallbacks\": 0, \"naive_fallbacks\": 0, \"buffer_bytes\": {}, \"potential_alias_savings\": {}, \"materialized_alias_savings\": {}",
				diagnostics.barrier_count(),
				war_barriers,
				indirect_barriers,
				buffer_bytes,
				diagnostics.potential_alias_savings(),
				diagnostics.materialized_alias_savings(),
			),
		);
		if copies != 0 {
			crate::core::push_format(&mut output, format_args!(r#", "copies": {copies}"#));
		}
		if video_commands != 0 {
			let image_transfers = self.graph.image_transfer_count();
			crate::core::push_format(
				&mut output,
				format_args!(
					r#", "video_commands": {video_commands}, "image_transfers": {image_transfers}"#
				),
			);
		}
		output.push_str("},\n  \"resources\": [");
		for (resource, lifetime) in lifetimes.iter().enumerate() {
			output.push_str(if resource == 0 { "\n" } else { ",\n" });
			crate::core::push_format(
				&mut output,
				format_args!(
					"    {{\"id\": {resource}, \"hazard_domain\": {resource}, \"bytes\": {}, \"first_node\": {}, \"last_node\": {}}}",
					lifetime.buffer.byte_len(),
					lifetime.first_access,
					lifetime.last_access,
				),
			);
		}
		if !lifetimes.is_empty() {
			output.push('\n');
		}
		// OARS alias arenas deliberately collapse a group to one buffer identity.
		// Logical group membership remains available in the compilation report.
		output.push_str("  ],\n  \"alias_groups\": [],\n  \"barriers\": [");
		let mut barrier_index = 0_usize;
		for destination_node in 0..self.graph.nodes().len() {
			let barriers = self.graph.barriers_before(destination_node);
			for barrier in barriers {
				output.push_str(if barrier_index == 0 { "\n" } else { ",\n" });
				barrier_index = barrier_index.saturating_add(1);
				let resource = graph_resource_id(&lifetimes, &barrier.buffer)
					.expect("planned barrier resource must have a lifetime");
				let [first_source, last_source] = barrier.source_nodes;
				let source_stage = barrier.source.vk_stages().as_raw();
				let destination_stage = barrier.destination.vk_stages().as_raw();
				let source_access = barrier.source.vk_access().as_raw();
				let destination_access = barrier.destination.vk_access().as_raw();
				crate::core::push_format(
					&mut output,
					format_args!(
						"    {{\"reason\": \"{}\", \"scope\": \"buffer\", \"source_nodes\": [{first_source}, {last_source}], \"destination_node\": {destination_node}, \"source_resource\": {resource}, \"destination_resource\": {resource}, \"hazard_domain\": {resource}, \"range\": {{\"offset\": 0, \"bytes\": {}}}, \"source_queue\": \"{}\", \"destination_queue\": \"{}\", \"source_family\": {}, \"destination_family\": {}, \"cross_queue\": {}, \"ownership_transfer\": {}, \"source_stage_mask\": \"0x{source_stage:016x}\", \"source_stages\": [",
						hazard_reason(barrier.source, barrier.destination),
						barrier.buffer.byte_len(),
						barrier.source_queue.name(),
						barrier.destination_queue.name(),
						barrier.source_family,
						barrier.destination_family,
						barrier.is_cross_queue(),
						barrier.is_ownership_transfer(),
					),
				);
				push_stage_names(&mut output, barrier.source);
				crate::core::push_format(
					&mut output,
					format_args!(
						"], \"source_access_mask\": \"0x{source_access:016x}\", \"source_accesses\": ["
					),
				);
				push_access_names(&mut output, barrier.source);
				crate::core::push_format(
					&mut output,
					format_args!(
						"], \"destination_stage_mask\": \"0x{destination_stage:016x}\", \"destination_stages\": ["
					),
				);
				push_stage_names(&mut output, barrier.destination);
				crate::core::push_format(
					&mut output,
					format_args!(
						"], \"destination_access_mask\": \"0x{destination_access:016x}\", \"destination_accesses\": ["
					),
				);
				push_access_names(&mut output, barrier.destination);
				output.push_str("]}");
			}
		}
		if barrier_index != 0 {
			output.push('\n');
		}
		output.push_str("  ],\n  \"nodes\": [");
		for (index, graph_node) in self.graph.nodes().iter().enumerate() {
			let (operation, semantic_ops, contract) = graph_node.semantic();
			output.push_str(if index == 0 { "\n" } else { ",\n" });
			crate::core::push_format(
				&mut output,
				format_args!("    {{\"index\": {index}, \"operation\": "),
			);
			crate::core::push_json_string(&mut output, operation);
			output.push_str(", \"semantic_operations\": [");
			for (owner_index, owner) in semantic_ops.iter().enumerate() {
				if owner_index != 0 {
					output.push_str(", ");
				}
				crate::core::push_format(&mut output, format_args!("{}", owner.index()));
			}
			match graph_node {
				GraphNode::Compute(node) => {
					let implementation_id = node.kernel.artifact().content_id();
					crate::core::push_format(
						&mut output,
						format_args!(
							"], \"implementation_id\": \"0x{implementation_id:016x}\", \"operation_contract_hash\": "
						),
					);
					push_optional_hex(&mut output, contract);
					crate::core::push_format(
						&mut output,
						format_args!(
							", \"problem_contract_hash\": null, \"kernel_content_hash\": \"0x{implementation_id:016x}\", \"kernel_selection\": \"direct\", \"kernel\": "
						),
					);
					crate::core::push_json_string(&mut output, node.kernel.report_name());
					output.push_str(", \"dtype_class\": null, \"dtype\": ");
					crate::core::push_json_string(&mut output, node.kernel.dtype_report_token());
					output.push_str(", \"physical_write\": ");
					match node.kernel.artifact().physical_write {
						None => output.push_str("null"),
						Some(contract) => {
							output.push_str("{\"writes\": [");
							for (write_index, write) in contract.writes.iter().enumerate() {
								if write_index != 0 {
									output.push_str(", ");
								}
								crate::core::push_format(
									&mut output,
									format_args!(
										"{{\"binding\": {}, \"domain\": \"{}\", \"partition\": \"{}\", \"extent\": \"{}\", \"collision\": \"{}\", \"tail\": \"{}\"}}",
										write.binding,
										write.domain.token(),
										write.partition.token(),
										write.extent.token(),
										write.collision.token(),
										write.tail.token(),
									),
								);
							}
							output.push_str("], \"uniform_store\": ");
							match contract.uniform_store {
								None => output.push_str("null"),
								Some(proof) => crate::core::push_format(
									&mut output,
									format_args!(
										"{{\"domain\": \"{}\", \"partition\": \"{}\", \"extent\": \"{}\", \"collision\": \"{}\", \"tail\": \"{}\"}}",
										proof.domain.token(),
										proof.partition.token(),
										proof.extent.token(),
										proof.collision.token(),
										proof.tail.token(),
									),
								),
							}
							crate::core::push_format(
								&mut output,
								format_args!(", \"workspace\": \"{}\"}}", contract.workspace.token()),
							);
						}
					}
					crate::core::push_format(
						&mut output,
						format_args!(
							", \"groups\": [{}, {}, {}], \"queue\": \"{}\", \"family\": {}, \"indirect\": {}",
							node.workgroups[0],
							node.workgroups[1],
							node.workgroups[2],
							graph_node.queue().name(),
							self.graph.node_families()[index],
							node.indirect.is_some()
						),
					);
				}
				GraphNode::BufferCopy(copy) => {
					output.push_str(
						r#"], "kind": "buffer_copy", "implementation_id": "0x4255465f434f5059", "operation_contract_hash": "#,
					);
					push_optional_hex(&mut output, contract);
					crate::core::push_format(
						&mut output,
						format_args!(
							r#", "problem_contract_hash": null, "kernel_content_hash": "0x0000000000000000", "kernel_selection": "native", "kernel": "vk.copy_buffer", "dtype_class": null, "dtype": "float32", "physical_write": null, "groups": [0, 0, 0], "queue": "{}", "family": {}, "indirect": false, "copy_bytes": {}"#,
							graph_node.queue().name(),
							self.graph.node_families()[index],
							copy.bytes,
						),
					);
				}
				GraphNode::VideoDecode(video) => {
					crate::core::push_format(
						&mut output,
						format_args!(
							r#"], "kind": "video_decode", "implementation_id": "0x5649444445434f44", "operation_contract_hash": null, "problem_contract_hash": null, "kernel_content_hash": "0x0000000000000000", "kernel_selection": "native", "kernel": "{operation}", "dtype_class": null, "dtype": "uint8", "physical_write": null, "groups": [0, 0, 0], "queue": "{}", "family": {}, "indirect": false, "decode_picture": ["#,
							graph_node.queue().name(),
							self.graph.node_families()[index],
						),
					);
					for (index, word) in video.descriptor_words().into_iter().enumerate() {
						if index != 0 {
							output.push_str(", ");
						}
						crate::core::push_format(&mut output, format_args!("{word}"));
					}
					output.push(']');
				}

				GraphNode::VideoEncode(video) => {
					crate::core::push_format(
						&mut output,
						format_args!(
							r#"], "kind": "video_encode", "implementation_id": "0x564944454e434f44", "operation_contract_hash": null, "problem_contract_hash": null, "kernel_content_hash": "0x0000000000000000", "kernel_selection": "native", "kernel": "{operation}", "dtype_class": null, "dtype": "uint8", "physical_write": null, "groups": [0, 0, 0], "queue": "{}", "family": {}, "indirect": false, "encode_picture": ["#,
							graph_node.queue().name(),
							self.graph.node_families()[index],
						),
					);
					for (index, word) in video.descriptor_words().into_iter().enumerate() {
						if index != 0 {
							output.push_str(", ");
						}
						crate::core::push_format(&mut output, format_args!("{word}"));
					}
					output.push(']');
				}
				GraphNode::VideoEncodeInput(video) => {
					crate::core::push_format(
						&mut output,
						format_args!(
							r#"], "kind": "video_encode_input", "implementation_id": "0x564944454e43494e", "operation_contract_hash": null, "problem_contract_hash": null, "kernel_content_hash": "0x0000000000000000", "kernel_selection": "native", "kernel": "{}", "dtype_class": null, "dtype": "uint8", "physical_write": null, "groups": [0, 0, 0], "queue": "{}", "family": {}, "indirect": false, "image_geometry": ["#,
							video.copy.name(),
							graph_node.queue().name(),
							self.graph.node_families()[index],
						),
					);
					for (index, word) in video.copy.descriptor_words().enumerate() {
						if index != 0 {
							output.push_str(", ");
						}
						crate::core::push_format(&mut output, format_args!("{word}"));
					}
					output.push(']');
				}
				GraphNode::VideoPlanes(video) => {
					crate::core::push_format(
						&mut output,
						format_args!(
							r#"], "kind": "video_plane_command", "implementation_id": "0x564944504c414e45", "operation_contract_hash": null, "problem_contract_hash": null, "kernel_content_hash": "0x0000000000000000", "kernel_selection": "native", "kernel": "{}", "dtype_class": null, "dtype": "uint8", "physical_write": null, "groups": [0, 0, 0], "queue": "{}", "family": {}, "indirect": false, "image_geometry": ["#,
							video.phase.name(),
							graph_node.queue().name(),
							self.graph.node_families()[index],
						),
					);
					for (index, word) in video.copy.descriptor_words().enumerate() {
						if index != 0 {
							output.push_str(", ");
						}
						crate::core::push_format(&mut output, format_args!("{word}"));
					}
					output.push(']');
				}
			}
			output.push_str(r#", "effects": ["#);
			for (effect_index, effect) in graph_node.buffers().iter().enumerate() {
				if effect_index != 0 {
					output.push_str(", ");
				}
				let resource = graph_resource_id(&lifetimes, &effect.buffer)
					.expect("node buffer must have a resource lifetime");
				crate::core::push_format(
					&mut output,
					format_args!(
						"{{\"resource\": {resource}, \"access\": \"{}\"}}",
						buffer_access_token(effect.access)
					),
				);
			}
			output.push_str("]}");
		}
		if !self.graph.nodes().is_empty() {
			output.push('\n');
		}
		output.push_str("  ]\n}\n");
		output
	}

	pub(super) fn belongs_to(&self, engine: &EngineHandle) -> bool {
		self.engine.same_as(engine)
	}

	pub(crate) fn validate_training_replay_safety(&self) -> Result<()> {
		self.graph.validate_training_replay_safety()
	}

	fn binding_key(&self) -> Vec<u32> {
		self
			.input_bindings
			.iter()
			.map(|binding| binding.current.descriptor_index())
			.collect()
	}

	/// Node ranges of the executable graph that each fit the device's
	/// estimated submission budget. Splitting is per queue run — every
	/// emitted range resolves to one queue family. A single returned range
	/// keeps the one-buffer fast path; several ranges become sequential
	/// timeline submissions at node boundaries. A node whose own estimate
	/// exceeds the budget declines here, before anything records or
	/// submits.
	pub(super) fn submission_ranges(
		&self,
		device: &Arc<Device>,
	) -> Result<Vec<std::ops::Range<usize>>> {
		let budget = device.submission_budget_ns();
		let mut ranges = Vec::new();
		for run in self.graph.queue_runs() {
			for range in
				self
					.graph
					.budget_ranges_within(run.nodes, budget)
					.map_err(|(index, estimate)| {
						Error::resource_exhausted(format!(
							"node {} ({}) estimated {:.1}ms exceeds the {:.1}ms submission budget",
							index,
							self.graph.nodes()[index].semantic().0,
							estimate as f64 / 1e6,
							budget as f64 / 1e6,
						))
					})? {
				if !range.is_empty() {
					ranges.push(range);
				}
			}
		}
		Ok(ranges)
	}

	pub(super) fn graph(&self) -> &ExecutableGraph {
		&self.graph
	}

	/// One reusable command buffer per queue run, in submission order.
	/// A single-run graph returns one command; a mixed-family graph
	/// returns one command per run for the timeline-chained batch.
	pub(super) fn reusable_commands(
		&self,
		device: &Arc<Device>,
	) -> Result<Vec<RecordedCommandBuffer>> {
		let key = self.binding_key();
		if let Some((_, commands)) = self
			.cached_commands
			.borrow()
			.iter()
			.find(|(cached_key, _)| *cached_key == key)
		{
			increment(&self.counters.command_cache_hits);
			return Ok(
				commands
					.iter()
					.map(|command| command.submission())
					.collect(),
			);
		}
		let runs = self.graph.queue_runs();
		let commands = if runs.len() == 1 {
			vec![device.record_reusable_compute_graph(&self.graph)?]
		} else {
			runs
				.into_iter()
				.map(|run| device.record_reusable_graph_range(&self.graph, run.nodes))
				.collect::<Result<Vec<_>>>()?
		};
		increment(&self.counters.command_recordings);
		let submissions = commands
			.iter()
			.map(|command| command.submission())
			.collect();
		let mut cache = self.cached_commands.borrow_mut();
		if cache.len() == MAX_CACHED_BINDING_COMMANDS {
			cache.remove(0);
		}
		cache.push((key, commands));
		Ok(submissions)
	}

	pub(super) fn count_command_recording(&self) {
		increment(&self.counters.command_recordings);
	}

	pub(super) fn precompile(&self, device: &Arc<Device>) -> Result<()> {
		if self.graph.has_video() {
			let mut commands = super::device::command::RecordedBatch::new(device);
			for range in self.submission_ranges(device)? {
				commands.push(device.record_fresh_graph_range(&self.graph, range, false)?);
			}
		} else {
			let _ = self.reusable_commands(device)?;
		}
		Ok(())
	}

	pub(super) fn mark_submitted(&self, event: &Event) {
		for output in &self.outputs {
			output.mark_submitted(event.clone());
		}
		for binding in &self.input_bindings {
			if let Some(slot) = binding
				.staging
				.iter()
				.find(|slot| slot.buffer.same_as(&binding.current))
			{
				slot.consumer.replace(Some(event.clone()));
			}
			binding.current.record_bound_consumer(event);
		}
		self.last_submission.replace(Some(event.clone()));
		increment(&self.counters.submissions);
	}

	pub(super) fn mark_failed(&self) {
		for output in &self.outputs {
			output.mark_failed();
		}
	}
}

fn increment(counter: &Cell<u64>) {
	counter.set(counter.get().saturating_add(1));
}

fn graph_resource_id(
	lifetimes: &[super::graph::executable::ResourceLifetime],
	buffer: &Buffer,
) -> Option<usize> {
	lifetimes
		.iter()
		.position(|lifetime| lifetime.buffer.same_as(buffer))
}

const fn hazard_reason(source: AccessState, destination: AccessState) -> &'static str {
	match (source.writes(), destination.writes()) {
		(true, true) => "write_after_write",
		(true, false) => "read_after_write",
		(false, true) => "write_after_read",
		(false, false) => "queue_ownership",
	}
}

fn push_stage_names(output: &mut String, access: AccessState) {
	let mask = access.vk_stages();
	push_names(
		output,
		[
			(
				mask.contains(vk::PipelineStageFlagBits2::COMPUTE_SHADER_BIT),
				"compute_shader",
			),
			(
				mask.contains(vk::PipelineStageFlagBits2::DRAW_INDIRECT_BIT),
				"draw_indirect",
			),
			(mask.contains(vk::PipelineStageFlagBits2::COPY_BIT), "copy"),
		],
	);
}

fn push_access_names(output: &mut String, access: AccessState) {
	let mask = access.vk_access();
	push_names(
		output,
		[
			(
				mask.contains(vk::AccessFlagBits2::SHADER_STORAGE_READ_BIT),
				"shader_storage_read",
			),
			(
				mask.contains(vk::AccessFlagBits2::SHADER_STORAGE_WRITE_BIT),
				"shader_storage_write",
			),
			(
				mask.contains(vk::AccessFlagBits2::INDIRECT_COMMAND_READ_BIT),
				"indirect_command_read",
			),
			(
				mask.contains(vk::AccessFlagBits2::TRANSFER_READ_BIT),
				"transfer_read",
			),
			(
				mask.contains(vk::AccessFlagBits2::TRANSFER_WRITE_BIT),
				"transfer_write",
			),
		],
	);
}

fn push_names(output: &mut String, names: impl IntoIterator<Item = (bool, &'static str)>) {
	let mut separator = "";
	for (_, name) in names.into_iter().filter(|(present, _)| *present) {
		output.push_str(separator);
		crate::core::push_json_string(output, name);
		separator = ", ";
	}
}

const fn buffer_access_token(access: super::BufferAccess) -> &'static str {
	match access {
		super::BufferAccess::Read => "read",
		super::BufferAccess::Write => "write",
		super::BufferAccess::ReadWrite => "read_write",
	}
}

fn push_optional_hex(output: &mut String, value: u64) {
	if value == 0 {
		output.push_str("null");
	} else {
		crate::core::push_format(output, format_args!("\"0x{value:016x}\""));
	}
}

fn find_resource(
	lifetimes: &[super::graph::executable::ResourceLifetime],
	storage: &Storage,
) -> Result<usize> {
	let buffer = storage.buffer().ok_or_else(|| {
		Error::failed_precondition("captured semantic resource has no allocated storage")
	})?;
	lifetimes
		.iter()
		.position(|lifetime| lifetime.buffer.same_as(buffer))
		.ok_or_else(|| Error::failed_precondition("captured resource does not appear in the graph"))
}

fn find_resource_storage<'a>(
	buffer: &Buffer,
	stable_resources: &'a [StableResourceSnapshot],
	semantic_storage: &'a [SemanticStorageSnapshot],
	observed_outputs: &'a [Storage],
	outputs: &'a [Storage],
) -> Option<&'a Storage> {
	// A lowering may create an immutable physical parameter buffer that is not a
	// semantic value or public output. The executable graph's Buffer retains
	// that allocation; absence here only means there is no Matrix-level owner to
	// count or consider for transient aliasing.
	stable_resources
		.iter()
		.map(|candidate| &candidate.storage)
		.chain(semantic_storage.iter().map(|candidate| &candidate.storage))
		.chain(observed_outputs.iter())
		.chain(outputs.iter())
		.find(|storage| {
			storage
				.buffer()
				.is_some_and(|candidate| candidate.same_as(buffer))
		})
}

struct AliasGroup {
	resources: Vec<usize>,
	required_size: usize,
}

struct AliasMaterialization {
	graph: ExecutableGraph,
	outputs: Vec<Storage>,
	captured_resources: Vec<CapturedResourceDesc>,
	retired: Vec<Buffer>,
	savings: usize,
}

fn materialize_aliases(
	engine: &EngineHandle,
	graph: &ExecutableGraph,
	outputs: &[Storage],
	resource_lifetimes: &[super::graph::executable::ResourceLifetime],
	captured_resources: &[CapturedResourceDesc],
) -> Result<AliasMaterialization> {
	let mut graph = graph.clone();
	let mut outputs = outputs.to_vec();
	let mut captured_resources = captured_resources.to_vec();
	let groups = alias_groups(&captured_resources);
	let mut replacements = Vec::new();
	let mut retired = Vec::new();
	let mut savings = 0_usize;
	for group in groups.iter().filter(|group| group.resources.len() > 1) {
		let arena = engine.create_alias_arena(group.required_size)?;
		let original_bytes = group.resources.iter().try_fold(0_usize, |sum, resource| {
			sum
				.checked_add(captured_resources[*resource].byte_len)
				.ok_or_else(|| Error::resource_exhausted("materialized alias byte accounting overflowed"))
		})?;
		savings = savings
			.checked_add(original_bytes.saturating_sub(group.required_size))
			.ok_or_else(|| Error::resource_exhausted("alias savings accounting overflowed"))?;
		for resource in &group.resources {
			let source = resource_lifetimes[*resource].buffer.clone();
			retired.push(source.clone());
			replacements.push((source, arena.clone()));
			captured_resources[*resource].alias_materialized = true;
		}
	}
	if !replacements.is_empty() {
		graph.materialize_alias_replacements(&replacements)?;
		outputs.retain(|storage| {
			storage
				.buffer()
				.is_none_or(|buffer| !retired.iter().any(|candidate| candidate.same_as(buffer)))
		});
	}
	Ok(AliasMaterialization {
		graph,
		outputs,
		captured_resources,
		retired,
		savings,
	})
}

fn alias_groups(resources: &[CapturedResourceDesc]) -> Vec<AliasGroup> {
	let mut eligible = resources
		.iter()
		.enumerate()
		.filter_map(|(index, resource)| resource.alias_candidate.then_some(index))
		.collect::<Vec<_>>();
	eligible.sort_by_key(|index| (resources[*index].first_access, resources[*index].resource));
	let mut groups = Vec::<AliasGroup>::new();
	for resource in eligible {
		let desc = resources[resource];
		if let Some(group) = groups.iter_mut().find(|group| {
			group.resources.iter().all(|member| {
				let member = resources[*member];
				desc.first_access > member.last_access || desc.last_access < member.first_access
			})
		}) {
			group.required_size = group.required_size.max(desc.byte_len);
			group.resources.push(resource);
		} else {
			groups.push(AliasGroup {
				resources: vec![resource],
				required_size: desc.byte_len,
			});
		}
	}
	groups
}

fn alias_analysis(resources: &[CapturedResourceDesc]) -> (usize, usize) {
	let groups = alias_groups(resources);
	let mut count = 0_usize;
	let mut savings = 0_usize;
	for group in groups.iter().filter(|group| group.resources.len() > 1) {
		count = count.saturating_add(1);
		let bytes = group.resources.iter().fold(0_usize, |total, resource| {
			total.saturating_add(resources[*resource].byte_len)
		});
		savings = savings.saturating_add(bytes.saturating_sub(group.required_size));
	}
	(count, savings)
}

impl fmt::Debug for ExecutionPlan {
	fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
		formatter
			.debug_struct("ExecutionPlan")
			.field("graph_id", &format_args!("{:016x}", self.graph_id))
			.field("node_count", &self.graph.nodes().len())
			.finish_non_exhaustive()
	}
}

#[cfg(test)]
#[path = "../../../../test/rs/runtime/plan_binding_unit.rs"]
mod tests;
