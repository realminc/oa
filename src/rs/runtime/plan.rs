use std::{
	cell::{Cell, RefCell},
	fmt,
};

use crate::{Error, Matrix, Result, core::Element};

use super::{
	EngineHandle, Event, SemanticGraph, SemanticLoweringAnalysis, Storage,
	dnn::{DnnPlan, DnnPolicy},
	executable_graph::ExecutableGraph,
	session::{PendingExecution, SemanticStorageSnapshot, StableResourceSnapshot},
	vk::{self, RecordedCommandBuffer, ReusableCommandBuffer},
};

#[cfg(test)]
thread_local! {
	static FORCE_NEXT_COMPILATION_FAILURE: Cell<bool> = const { Cell::new(false) };
}

#[cfg(test)]
pub(crate) fn force_next_compilation_failure() {
	FORCE_NEXT_COMPILATION_FAILURE.set(true);
}

/// Immutable snapshot of execution-plan diagnostics.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExecutionPlanDiagnostics {
	graph_id: u64,
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

	/// Return the number of captured compute nodes.
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

struct MatrixInputBinding {
	original: vk::Buffer,
	current: vk::Buffer,
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
/// be rebound to shape- and dtype-identical storage; changing a binding
/// invalidates the recorded Vulkan command while preserving the graph identity.
/// Unchanged untimed replays share one simultaneously submittable command buffer.
#[must_use]
pub struct ExecutionPlan {
	engine: EngineHandle,
	graph_id: u64,
	graph: ExecutableGraph,
	outputs: Vec<Storage>,
	semantic: SemanticGraph,
	semantic_lowering: SemanticLoweringAnalysis,
	dnn: DnnPlan,
	input_bindings: Vec<MatrixInputBinding>,
	captured_resources: Vec<CapturedResourceDesc>,
	semantic_bindings: Vec<SemanticStorageBinding>,
	materialized_alias_savings: usize,
	alias_materialization_fallback_reason: Option<String>,
	cached_command: RefCell<Option<ReusableCommandBuffer>>,
	last_submission: RefCell<Option<Event>>,
	counters: PlanCounters,
}

impl ExecutionPlan {
	pub(super) fn new(
		engine: EngineHandle,
		pending: PendingExecution,
		source_recording_retained: bool,
	) -> Result<Self> {
		let (
			mut graph,
			mut outputs,
			semantic,
			stable_resources,
			semantic_storage,
			observed_outputs,
		) = pending.into_parts();
		#[cfg(test)]
		if FORCE_NEXT_COMPILATION_FAILURE.replace(false) {
			return Err(Error::internal("forced plan compilation failure"));
		}
		let mut dnn = DnnPlan::plan(&semantic, DnnPolicy::default())?;
		if dnn.source_operation_count() as usize != semantic.operations().len() {
			return Err(Error::internal(
				"DNN plan source operation count does not match the semantic graph",
			));
		}
		let input_bindings = graph
			.read_only_buffers()
			.into_iter()
			.map(|buffer| MatrixInputBinding {
				original: buffer.clone(),
				current: buffer,
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
				resource_lifetimes.push(super::executable_graph::ResourceLifetime {
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
							.filter(|candidate| {
								owner.is_some_and(|owner| candidate.storage.same_as(owner))
							})
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
					resource: u32::try_from(resource).map_err(|_| {
						Error::resource_exhausted("captured resource identity exceeds u32")
					})?,
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
			&semantic,
			&semantic_bindings,
			&resource_lifetimes,
			&captured_resources,
			!source_recording_retained,
			&graph,
		)?;
		let graph_id = graph.identity();
		let semantic_lowering = graph.analyze_semantic_lowering(&semantic)?;
		for resource in &mut captured_resources {
			resource.alias_candidate = resource.stable_transient
				&& !resource.is_externally_live()
				&& resource.unaccounted_owner_count == 0;
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
			graph,
			outputs,
			semantic,
			semantic_lowering,
			dnn,
			input_bindings,
			captured_resources,
			semantic_bindings,
			materialized_alias_savings,
			alias_materialization_fallback_reason,
			cached_command: RefCell::new(None),
			last_submission: RefCell::new(None),
			counters: PlanCounters {
				fallbacks: Cell::new(initial_fallback_count),
				..PlanCounters::default()
			},
		})
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
	/// The captured shape, dtype, Vulkan buffer, descriptor index, graph identity,
	/// and cached command buffer remain unchanged. If a prior replay is pending,
	/// this method waits for its exact event before writing host-visible storage.
	///
	/// # Errors
	///
	/// Returns an error when `captured` is not an unrebound read-only plan input,
	/// the element type or count differs, or prior completion/upload fails.
	pub fn upload_matrix_input<T: Element>(
		&mut self,
		captured: &Matrix,
		values: &[T],
	) -> Result<()> {
		if !self.engine.same_as(captured.engine_handle()) {
			return Err(Error::invalid_argument(
				"execution-plan Matrix input belongs to another engine",
			));
		}
		let captured_buffer = captured.storage().buffer().ok_or_else(|| {
			Error::invalid_argument(
				"zero-sized Matrix storage cannot identify a captured plan input",
			)
		})?;
		let binding = self
			.input_bindings
			.iter()
			.find(|binding| binding.original.same_as(captured_buffer))
			.ok_or_else(|| {
				Error::invalid_argument(
					"captured Matrix is not a read-only input of this execution plan",
				)
			})?;
		if !binding.current.same_as(&binding.original) {
			return Err(Error::failed_precondition(
				"execution-plan host upload requires an unrebound stable input slot",
			));
		}
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
		if let Some(event) = self.last_submission.borrow().as_ref() {
			event.wait()?;
		}
		captured.write_values(values)?;
		increment(&self.counters.input_uploads);
		Ok(())
	}

	/// Rebind one read-only Matrix input while preserving its captured contract.
	///
	/// `captured` remains the stable slot identity even after repeated rebinding.
	/// The replacement must have the same shape and dtype, belong to the plan's
	/// engine, and not alias another resource already used by the plan. Rebinding
	/// does not submit or wait; the next untimed submission records one new command
	/// buffer and later unchanged submissions reuse it.
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
		replacement.storage().validate_recording_access()?;
		let captured_buffer = captured.storage().buffer().ok_or_else(|| {
			Error::invalid_argument(
				"zero-sized Matrix storage cannot identify a captured plan input",
			)
		})?;
		let replacement_buffer = replacement.storage().buffer().ok_or_else(|| {
			Error::invalid_argument(
				"zero-sized Matrix storage cannot replace a captured plan input",
			)
		})?;

		let existing = self
			.input_bindings
			.iter()
			.position(|binding| binding.original.same_as(captured_buffer))
			.ok_or_else(|| {
				Error::invalid_argument(
					"captured Matrix is not a read-only input of this execution plan",
				)
			})?;
		let current = self.input_bindings[existing].current.clone();
		if current.same_as(replacement_buffer) {
			return Ok(());
		}
		self.graph.rebind_read_only(&current, replacement_buffer)?;
		self.input_bindings[existing].current = replacement_buffer.clone();
		self.cached_command.get_mut().take();
		increment(&self.counters.input_rebindings);
		Ok(())
	}

	/// Return current graph, cache, submission, rebinding, and fallback counters.
	pub fn diagnostics(&self) -> ExecutionPlanDiagnostics {
		let (planned_alias_group_count, potential_alias_savings) =
			alias_analysis(&self.captured_resources);
		ExecutionPlanDiagnostics {
			graph_id: self.graph_id,
			node_count: self.graph.nodes().len(),
			barrier_count: self.graph.barrier_count(),
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
			maximum_semantic_operations_per_node: self.semantic_lowering.maximum_ops_per_node()
				as usize,
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
		let compiled = self.cached_command.borrow().is_some();
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
		let war_barriers = (0..self.graph.nodes().len())
			.flat_map(|node| self.graph.barriers_before(node))
			.filter(|barrier| {
				barrier.source.reads() && !barrier.source.writes() && barrier.destination.writes()
			})
			.count();
		let buffer_bytes = lifetimes.iter().fold(0_usize, |total, lifetime| {
			total.saturating_add(lifetime.buffer.byte_len())
		});
		crate::core::push_format(
			&mut output,
			format_args!(
				"  \"stats\": {{\"dispatches\": {}, \"barriers\": {}, \"war_barriers\": {}, \"indirect_barriers\": 0, \"alias_barriers\": 0, \"host_barriers\": 0, \"descriptor_sets\": 0, \"kernel_selections\": {}, \"kernel_fallbacks\": 0, \"precision_fallbacks\": 0, \"layout_fallbacks\": 0, \"naive_fallbacks\": 0, \"buffer_bytes\": {}, \"potential_alias_savings\": {}, \"materialized_alias_savings\": {}}},\n  \"resources\": [",
				diagnostics.node_count(),
				diagnostics.barrier_count(),
				war_barriers,
				diagnostics.node_count(),
				buffer_bytes,
				diagnostics.potential_alias_savings(),
				diagnostics.materialized_alias_savings(),
			),
		);
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
				let source_node =
					previous_access_node(&self.graph, destination_node, &barrier.buffer)
						.expect("planned barrier must have a source access");
				let source_stage = ash::vk::PipelineStageFlags2::COMPUTE_SHADER.as_raw();
				let source_access = barrier.source.vk_access().as_raw();
				let destination_access = barrier.destination.vk_access().as_raw();
				crate::core::push_format(
					&mut output,
					format_args!(
						"    {{\"reason\": \"{}\", \"scope\": \"buffer\", \"source_nodes\": [{source_node}, {source_node}], \"destination_node\": {destination_node}, \"source_resource\": {resource}, \"destination_resource\": {resource}, \"hazard_domain\": {resource}, \"range\": {{\"offset\": 0, \"bytes\": {}}}, \"source_queue\": \"compute\", \"destination_queue\": \"compute\", \"cross_queue\": false, \"ownership_transfer\": false, \"source_stage_mask\": \"0x{source_stage:016x}\", \"source_stages\": [\"compute_shader\"], \"source_access_mask\": \"0x{source_access:016x}\", \"source_accesses\": [",
						hazard_reason(barrier.source, barrier.destination),
						barrier.buffer.byte_len(),
					),
				);
				push_access_names(&mut output, barrier.source);
				crate::core::push_format(
					&mut output,
					format_args!(
						"], \"destination_stage_mask\": \"0x{source_stage:016x}\", \"destination_stages\": [\"compute_shader\"], \"destination_access_mask\": \"0x{destination_access:016x}\", \"destination_accesses\": ["
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
		for (index, node) in self.graph.nodes().iter().enumerate() {
			output.push_str(if index == 0 { "\n" } else { ",\n" });
			crate::core::push_format(
				&mut output,
				format_args!("    {{\"index\": {index}, \"operation\": "),
			);
			crate::core::push_json_string(&mut output, node.operation);
			output.push_str(", \"semantic_operations\": [");
			for (owner_index, owner) in node.semantic_ops.iter().enumerate() {
				if owner_index != 0 {
					output.push_str(", ");
				}
				crate::core::push_format(&mut output, format_args!("{}", owner.index()));
			}
			let implementation_id = node.kernel.artifact().content_id();
			crate::core::push_format(
				&mut output,
				format_args!(
					"], \"implementation_id\": \"0x{implementation_id:016x}\", \"operation_contract_hash\": "
				),
			);
			push_optional_hex(&mut output, node.op_contract_hash);
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
					crate::core::push_format(
						&mut output,
						format_args!("], \"workspace\": \"{}\"}}", contract.workspace.token()),
					);
				}
			}
			crate::core::push_format(
				&mut output,
				format_args!(
					", \"groups\": [{}, {}, {}], \"queue\": \"compute\", \"indirect\": false, \"effects\": [",
					node.workgroups[0], node.workgroups[1], node.workgroups[2]
				),
			);
			for (effect_index, effect) in node.buffers.iter().enumerate() {
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

	pub(super) fn graph(&self) -> &ExecutableGraph {
		&self.graph
	}

	pub(super) fn reusable_command(&self, device: &vk::Device) -> Result<RecordedCommandBuffer> {
		if let Some(command) = self.cached_command.borrow().as_ref().cloned() {
			increment(&self.counters.command_cache_hits);
			return Ok(command.submission());
		}
		let command = device.record_reusable_compute_graph(&self.graph)?;
		increment(&self.counters.command_recordings);
		let submission = command.submission();
		self.cached_command.replace(Some(command));
		Ok(submission)
	}

	pub(super) fn precompile(&self, device: &vk::Device) -> Result<()> {
		let _ = self.reusable_command(device)?;
		Ok(())
	}

	pub(super) fn mark_timed_recorded(&self) {
		increment(&self.counters.command_recordings);
	}

	pub(super) fn mark_submitted(&self, event: &Event) {
		for output in &self.outputs {
			output.mark_submitted(event.clone());
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
	lifetimes: &[super::executable_graph::ResourceLifetime],
	buffer: &vk::Buffer,
) -> Option<usize> {
	lifetimes
		.iter()
		.position(|lifetime| lifetime.buffer.same_as(buffer))
}

fn previous_access_node(
	graph: &ExecutableGraph,
	destination_node: usize,
	buffer: &vk::Buffer,
) -> Option<usize> {
	graph.nodes()[..destination_node]
		.iter()
		.rposition(|node| node.buffers.iter().any(|use_| use_.buffer.same_as(buffer)))
}

const fn hazard_reason(
	source: super::executable_graph::AccessState,
	destination: super::executable_graph::AccessState,
) -> &'static str {
	if source.writes() && destination.writes() {
		"write_after_write"
	} else if source.writes() {
		"read_after_write"
	} else {
		"write_after_read"
	}
}

fn push_access_names(output: &mut String, access: super::executable_graph::AccessState) {
	let mut separator = "";
	if access.reads() {
		output.push_str("\"shader_storage_read\"");
		separator = ", ";
	}
	if access.writes() {
		output.push_str(separator);
		output.push_str("\"shader_storage_write\"");
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
	lifetimes: &[super::executable_graph::ResourceLifetime],
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
	buffer: &vk::Buffer,
	stable_resources: &'a [StableResourceSnapshot],
	semantic_storage: &'a [SemanticStorageSnapshot],
	observed_outputs: &'a [Storage],
	outputs: &'a [Storage],
) -> Option<&'a Storage> {
	// A lowering may create an immutable physical parameter buffer that is not a
	// semantic value or public output. The executable graph's vk::Buffer retains
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
	retired: Vec<vk::Buffer>,
	savings: usize,
}

fn materialize_aliases(
	engine: &EngineHandle,
	graph: &ExecutableGraph,
	outputs: &[Storage],
	resource_lifetimes: &[super::executable_graph::ResourceLifetime],
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
			sum.checked_add(captured_resources[*resource].byte_len)
				.ok_or_else(|| {
					Error::resource_exhausted("materialized alias byte accounting overflowed")
				})
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
mod tests {
	use super::{AliasGroup, CapturedResourceDesc, alias_analysis, alias_groups};

	fn resource(id: u32, first: usize, last: usize, candidate: bool) -> CapturedResourceDesc {
		CapturedResourceDesc {
			resource: id,
			byte_len: (id as usize + 1) * 16,
			first_access: first,
			last_access: last,
			stable_replay_input: false,
			stable_transient: candidate,
			semantic_external: false,
			observed_output: false,
			capture_retained_owner_count: 0,
			unaccounted_owner_count: 0,
			alias_candidate: candidate,
			alias_materialized: false,
		}
	}

	#[test]
	fn alias_groups_use_inclusive_lifetimes_and_stable_greedy_order() {
		let resources = [
			resource(0, 0, 1, true),
			resource(1, 1, 1, true),
			resource(2, 2, 3, true),
			resource(3, 4, 4, false),
		];
		let groups = alias_groups(&resources);
		let identities = groups
			.iter()
			.map(|AliasGroup { resources, .. }| resources.as_slice())
			.collect::<Vec<_>>();
		assert_eq!(identities, [&[0, 2][..], &[1][..]]);
		assert_eq!(groups[0].required_size, 48);
		assert_eq!(groups[1].required_size, 32);
		assert_eq!(alias_analysis(&resources), (1, 16));
	}
}
