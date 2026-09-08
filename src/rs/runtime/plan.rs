use std::fmt;

use super::{EngineHandle, Event, Storage, executable_graph::ExecutableGraph};

/// Immutable engine-associated executable work captured for repeated submission.
///
/// A plan retains the concrete buffers and kernel dispatches recorded during
/// capture. It is intentionally not a public graph editor.
#[must_use]
pub struct ExecutionPlan {
	engine: EngineHandle,
	graph: ExecutableGraph,
	outputs: Vec<Storage>,
}

impl ExecutionPlan {
	pub(super) fn new(engine: EngineHandle, graph: ExecutableGraph, outputs: Vec<Storage>) -> Self {
		Self {
			engine,
			graph,
			outputs,
		}
	}

	pub(super) fn belongs_to(&self, engine: &EngineHandle) -> bool {
		self.engine.same_as(engine)
	}

	pub(super) fn graph(&self) -> &ExecutableGraph {
		&self.graph
	}

	pub(super) fn mark_submitted(&self, event: &Event) {
		for output in &self.outputs {
			output.mark_submitted(event.clone());
		}
	}

	pub(super) fn mark_failed(&self) {
		for output in &self.outputs {
			output.mark_failed();
		}
	}
}

impl fmt::Debug for ExecutionPlan {
	fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
		formatter
			.debug_struct("ExecutionPlan")
			.field("node_count", &self.graph.nodes().len())
			.finish_non_exhaustive()
	}
}
