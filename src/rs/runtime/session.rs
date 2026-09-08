use crate::Result;

use super::{BufferAccess, ComputeDispatch, Storage, executable_graph::ExecutableGraph, vk};

/// Private mutable eager-recording owner for one engine.
pub(super) struct ExecutionSession {
	graphs: Vec<ExecutableGraph>,
	outputs: Vec<Storage>,
}

pub(super) struct PendingExecution {
	pub(super) graph: ExecutableGraph,
	outputs: Vec<Storage>,
}

impl ExecutionSession {
	pub(super) const fn new() -> Self {
		Self {
			graphs: Vec::new(),
			outputs: Vec::new(),
		}
	}

	pub(super) fn record(
		&mut self,
		device: &vk::Device,
		dispatch: ComputeDispatch<'_>,
	) -> Result<()> {
		for binding in dispatch.buffers {
			binding.storage.validate_recording_access()?;
		}
		let outputs = dispatch
			.buffers
			.iter()
			.filter(|binding| binding.access == BufferAccess::Write)
			.map(|binding| binding.storage.clone())
			.collect::<Vec<_>>();
		let graph = ExecutableGraph::from_dispatches(device, std::slice::from_ref(&dispatch))?;

		self.graphs.push(graph);
		for output in &outputs {
			output.mark_recorded();
		}
		self.outputs.extend(outputs);
		Ok(())
	}

	pub(super) fn is_empty(&self) -> bool {
		self.graphs.is_empty()
	}

	pub(super) fn abort(&mut self) {
		self.graphs.clear();
		for output in self.outputs.drain(..) {
			output.mark_failed();
		}
	}

	pub(super) fn take(&mut self) -> Result<Option<PendingExecution>> {
		if self.graphs.is_empty() {
			return Ok(None);
		}

		let graphs = std::mem::take(&mut self.graphs);
		let outputs = std::mem::take(&mut self.outputs);
		match ExecutableGraph::join(graphs) {
			Ok(graph) => Ok(Some(PendingExecution { graph, outputs })),
			Err(error) => {
				for output in outputs {
					output.mark_failed();
				}
				Err(error)
			}
		}
	}
}

impl PendingExecution {
	pub(super) fn mark_submitted(&self, event: &super::Event) {
		for output in &self.outputs {
			output.mark_submitted(event.clone());
		}
	}

	pub(super) fn mark_failed(&self) {
		for output in &self.outputs {
			output.mark_failed();
		}
	}

	pub(super) fn mark_captured(&self) {
		for output in &self.outputs {
			output.mark_captured();
		}
	}

	pub(super) fn into_parts(self) -> (ExecutableGraph, Vec<Storage>) {
		(self.graph, self.outputs)
	}
}
