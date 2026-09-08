use std::{
	cell::{Cell, RefCell},
	fmt,
};

use crate::{Error, Matrix, Result};

use super::{
	EngineHandle, Event, Storage,
	executable_graph::ExecutableGraph,
	vk::{self, RecordedCommandBuffer, ReusableCommandBuffer},
};

/// Immutable snapshot of execution-plan diagnostics.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExecutionPlanDiagnostics {
	graph_id: u64,
	node_count: usize,
	barrier_count: usize,
	input_binding_count: usize,
	command_recording_count: u64,
	command_cache_hit_count: u64,
	submission_count: u64,
	input_rebinding_count: u64,
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

	/// Return the number of Matrix inputs admitted for stable rebinding.
	pub const fn input_binding_count(self) -> usize {
		self.input_binding_count
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

	/// Return the number of runtime fallback decisions made by this plan.
	///
	/// The current runtime has no fallback path and therefore reports zero.
	pub const fn fallback_count(self) -> u64 {
		self.fallback_count
	}
}

struct MatrixInputBinding {
	original: vk::Buffer,
	current: vk::Buffer,
}

#[derive(Default)]
struct PlanCounters {
	command_recordings: Cell<u64>,
	command_cache_hits: Cell<u64>,
	submissions: Cell<u64>,
	input_rebindings: Cell<u64>,
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
	input_bindings: Vec<MatrixInputBinding>,
	cached_command: RefCell<Option<ReusableCommandBuffer>>,
	counters: PlanCounters,
}

impl ExecutionPlan {
	pub(super) fn new(engine: EngineHandle, graph: ExecutableGraph, outputs: Vec<Storage>) -> Self {
		let graph_id = graph.identity();
		let input_bindings = graph
			.read_only_buffers()
			.into_iter()
			.map(|buffer| MatrixInputBinding {
				original: buffer.clone(),
				current: buffer,
			})
			.collect();
		Self {
			engine,
			graph_id,
			graph,
			outputs,
			input_bindings,
			cached_command: RefCell::new(None),
			counters: PlanCounters::default(),
		}
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
		ExecutionPlanDiagnostics {
			graph_id: self.graph_id,
			node_count: self.graph.nodes().len(),
			barrier_count: self.graph.barrier_count(),
			input_binding_count: self.input_bindings.len(),
			command_recording_count: self.counters.command_recordings.get(),
			command_cache_hit_count: self.counters.command_cache_hits.get(),
			submission_count: self.counters.submissions.get(),
			input_rebinding_count: self.counters.input_rebindings.get(),
			fallback_count: self.counters.fallbacks.get(),
		}
	}

	pub(super) fn belongs_to(&self, engine: &EngineHandle) -> bool {
		self.engine.same_as(engine)
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

	pub(super) fn mark_timed_recorded(&self) {
		increment(&self.counters.command_recordings);
	}

	pub(super) fn mark_submitted(&self, event: &Event) {
		for output in &self.outputs {
			output.mark_submitted(event.clone());
		}
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

impl fmt::Debug for ExecutionPlan {
	fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
		formatter
			.debug_struct("ExecutionPlan")
			.field("graph_id", &format_args!("{:016x}", self.graph_id))
			.field("node_count", &self.graph.nodes().len())
			.finish_non_exhaustive()
	}
}
