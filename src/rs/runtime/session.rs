use std::collections::BTreeMap;

use crate::{Error, Matrix, OpValueKind, Result, core::MatrixSemantic};

use super::{
	BufferAccess, ComputeDispatch, SemanticDispatch, SemanticGraph, SemanticValueDesc,
	SemanticValueId, Storage, executable_graph::ExecutableGraph, vk,
};

/// Private mutable eager-recording owner for one engine.
pub(super) struct ExecutionSession {
	graphs: Vec<ExecutableGraph>,
	outputs: Vec<Storage>,
	semantic: SemanticGraph,
	semantic_values: BTreeMap<u64, SemanticValueBinding>,
	stable_resources: Vec<Storage>,
	stable_resource_cursor: usize,
	stable_resource_count: usize,
	stable_external_resource_count: usize,
	stable_resource_frame_active: bool,
	stable_resource_inputs_sealed: bool,
}

pub(super) struct PendingExecution {
	pub(super) graph: ExecutableGraph,
	outputs: Vec<Storage>,
	pub(super) semantic: SemanticGraph,
	stable_resources: Vec<StableResourceSnapshot>,
	semantic_bindings: Vec<SemanticStorageSnapshot>,
	observed_outputs: Vec<Storage>,
}

#[derive(Clone)]
pub(super) struct StableResourceSnapshot {
	pub(super) storage: Storage,
	pub(super) replay_input: bool,
	pub(super) transient: bool,
}

#[derive(Clone)]
struct SemanticValueBinding {
	value: SemanticValueId,
	storage: Storage,
}

#[derive(Clone)]
pub(super) struct SemanticStorageSnapshot {
	pub(super) value: SemanticValueId,
	pub(super) storage: Storage,
}

impl ExecutionSession {
	pub(super) fn new() -> Self {
		Self {
			graphs: Vec::new(),
			outputs: Vec::new(),
			semantic: SemanticGraph::new(),
			semantic_values: BTreeMap::new(),
			stable_resources: Vec::new(),
			stable_resource_cursor: 0,
			stable_resource_count: 0,
			stable_external_resource_count: 0,
			stable_resource_frame_active: false,
			stable_resource_inputs_sealed: false,
		}
	}

	pub(super) fn create_storage(&mut self, device: &vk::Device, bytes: &[u8]) -> Result<Storage> {
		if !self.stable_resource_frame_active || bytes.is_empty() {
			return Storage::from_bytes(device, bytes);
		}
		let slot = self.stable_resource_cursor;
		self.stable_resource_cursor = self
			.stable_resource_cursor
			.checked_add(1)
			.ok_or_else(|| Error::resource_exhausted("stable resource cursor exhausted"))?;
		if let Some(existing) = self.stable_resources.get(slot)
			&& existing.byte_len() == bytes.len()
		{
			existing.prepare_reuse(bytes)?;
			return Ok(existing.clone());
		}
		let storage = Storage::from_bytes(device, bytes)?;
		if slot < self.stable_resources.len() {
			self.stable_resources[slot] = storage.clone();
		} else {
			self.stable_resources.push(storage.clone());
		}
		Ok(storage)
	}

	pub(super) fn begin_stable_resource_frame(&mut self) -> Result<()> {
		if self.stable_resource_frame_active {
			return Err(Error::failed_precondition(
				"stable resource frames cannot be nested",
			));
		}
		self.stable_resource_cursor = 0;
		self.stable_resource_frame_active = true;
		self.stable_resource_inputs_sealed = false;
		Ok(())
	}

	pub(super) fn seal_all_stable_resources_external(&mut self) -> Result<()> {
		self.seal_stable_resource_inputs()
	}

	pub(super) fn seal_stable_resource_inputs(&mut self) -> Result<()> {
		if !self.stable_resource_frame_active || self.stable_resource_inputs_sealed {
			return Err(Error::failed_precondition(
				"stable resource inputs require one active unsealed frame",
			));
		}
		self.stable_external_resource_count = self.stable_resource_cursor;
		self.stable_resource_inputs_sealed = true;
		Ok(())
	}

	pub(super) fn end_stable_resource_frame(&mut self) {
		if !self.stable_resource_frame_active {
			return;
		}
		if self.stable_resource_inputs_sealed {
			self.stable_resource_count = self.stable_resource_cursor;
			self.stable_resources.truncate(self.stable_resource_count);
		}
		self.stable_resource_cursor = 0;
		self.stable_resource_frame_active = false;
		self.stable_resource_inputs_sealed = false;
	}

	pub(super) fn release_stable_transient_resources(&mut self, retired: &[vk::Buffer]) {
		let used = if self.stable_resource_frame_active {
			self.stable_resource_cursor
		} else {
			self.stable_resource_count
		};
		for storage in self
			.stable_resources
			.iter()
			.take(used)
			.skip(self.stable_external_resource_count)
		{
			if let Some(buffer) = storage.buffer()
				&& retired.iter().any(|candidate| candidate.same_as(buffer))
			{
				buffer.disable_recycling();
			}
		}
		self.stable_resources
			.truncate(self.stable_external_resource_count);
		self.stable_resource_count = self.stable_external_resource_count;
		if self.stable_resource_frame_active {
			self.stable_resource_cursor = self.stable_external_resource_count;
		}
	}

	pub(super) fn record(
		&mut self,
		device: &vk::Device,
		dispatch: ComputeDispatch<'_>,
	) -> Result<()> {
		self.record_impl(device, dispatch, None)
	}

	pub(super) fn record_semantic(
		&mut self,
		device: &vk::Device,
		dispatch: ComputeDispatch<'_>,
		semantic: SemanticDispatch<'_>,
	) -> Result<()> {
		self.record_impl(device, dispatch, Some(semantic))
	}

	fn record_impl(
		&mut self,
		device: &vk::Device,
		dispatch: ComputeDispatch<'_>,
		semantic: Option<SemanticDispatch<'_>>,
	) -> Result<()> {
		for binding in dispatch.buffers {
			binding.storage.validate_recording_access()?;
		}
		let outputs = dispatch
			.buffers
			.iter()
			.filter(|binding| binding.access != BufferAccess::Read)
			.map(|binding| binding.storage.clone())
			.collect::<Vec<_>>();
		let mut graph = ExecutableGraph::from_dispatches(device, std::slice::from_ref(&dispatch))?;
		if let Some(semantic) = semantic {
			let mut candidate = self.semantic.clone();
			let mut values = self.semantic_values.clone();
			let inputs = semantic
				.inputs
				.iter()
				.map(|matrix| admit_matrix(&mut candidate, &mut values, matrix, true).map(Some))
				.collect::<Result<Vec<_>>>()?;
			let outputs = semantic
				.outputs
				.iter()
				.map(|matrix| admit_matrix(&mut candidate, &mut values, matrix, false))
				.collect::<Result<Vec<_>>>()?;
			let operation = candidate.add_operation(
				semantic.contract,
				&inputs,
				&outputs,
				&[],
				semantic.attributes,
			)?;
			candidate.validate()?;
			graph.attach_direct_semantic(
				operation,
				semantic.contract.name(),
				semantic.contract.hash(),
			)?;
			self.semantic = candidate;
			self.semantic_values = values;
		}

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
		self.semantic.reset();
		self.semantic_values.clear();
		for output in self.outputs.drain(..) {
			output.mark_failed();
		}
	}

	pub(super) fn take(
		&mut self,
		observed_outputs: &[&Matrix],
	) -> Result<Option<PendingExecution>> {
		if self.graphs.is_empty() {
			return Ok(None);
		}
		if self.semantic_values.len() != self.semantic.values().len() {
			return Err(Error::failed_precondition(
				"semantic storage snapshot requires one binding per value",
			));
		}

		let stable_resources = self.snapshot_stable_resources()?;
		let graphs = std::mem::take(&mut self.graphs);
		let outputs = std::mem::take(&mut self.outputs);
		let semantic = self.semantic.take_recording()?;
		let semantic_values = std::mem::take(&mut self.semantic_values);
		let mut semantic_bindings = semantic_values
			.into_values()
			.map(|binding| SemanticStorageSnapshot {
				value: binding.value,
				storage: binding.storage,
			})
			.collect::<Vec<_>>();
		semantic_bindings.sort_by_key(|binding| binding.value);
		let observed_outputs = observed_outputs
			.iter()
			.map(|matrix| matrix.storage().clone())
			.collect();
		match ExecutableGraph::join(graphs) {
			Ok(graph) => Ok(Some(PendingExecution {
				graph,
				outputs,
				semantic,
				stable_resources,
				semantic_bindings,
				observed_outputs,
			})),
			Err(error) => {
				for output in outputs {
					output.mark_failed();
				}
				Err(error)
			}
		}
	}

	pub(super) fn snapshot(
		&self,
		observed_outputs: &[&Matrix],
	) -> Result<Option<PendingExecution>> {
		if self.graphs.is_empty() {
			return Ok(None);
		}
		if self.semantic_values.len() != self.semantic.values().len() {
			return Err(Error::failed_precondition(
				"semantic storage snapshot requires one binding per value",
			));
		}

		let stable_resources = self.snapshot_stable_resources()?;
		let graph = ExecutableGraph::join(self.graphs.clone())?;
		let outputs = self.outputs.clone();
		let semantic = self.semantic.clone();
		let mut semantic_bindings = self
			.semantic_values
			.values()
			.map(|binding| SemanticStorageSnapshot {
				value: binding.value,
				storage: binding.storage.clone(),
			})
			.collect::<Vec<_>>();
		semantic_bindings.sort_by_key(|binding| binding.value);
		let observed_outputs = observed_outputs
			.iter()
			.map(|matrix| matrix.storage().clone())
			.collect();
		Ok(Some(PendingExecution {
			graph,
			outputs,
			semantic,
			stable_resources,
			semantic_bindings,
			observed_outputs,
		}))
	}

	pub(super) fn commit_captured_snapshot(&mut self) -> Result<()> {
		if self.graphs.is_empty() {
			return Err(Error::failed_precondition(
				"execution capture recorded no executable work",
			));
		}
		let _ = self.semantic.take_recording()?;
		self.graphs.clear();
		self.semantic_values.clear();
		for output in self.outputs.drain(..) {
			output.mark_captured();
		}
		Ok(())
	}

	fn snapshot_stable_resources(&self) -> Result<Vec<StableResourceSnapshot>> {
		if self.stable_resource_frame_active && !self.stable_resource_inputs_sealed {
			return Err(Error::failed_precondition(
				"execution capture inside a stable resource frame requires sealed replay inputs",
			));
		}
		let used = if self.stable_resource_frame_active {
			self.stable_resource_cursor
		} else {
			self.stable_resource_count
		};
		if self.stable_external_resource_count > used || used > self.stable_resources.len() {
			return Err(Error::internal(
				"stable resource frame contains an invalid replay-input prefix",
			));
		}
		Ok(self
			.stable_resources
			.iter()
			.take(used)
			.enumerate()
			.map(|(index, storage)| StableResourceSnapshot {
				storage: storage.clone(),
				replay_input: index < self.stable_external_resource_count,
				transient: index >= self.stable_external_resource_count,
			})
			.collect())
	}

	pub(super) fn attach_autograd(
		&mut self,
		matrix_value: u64,
		sequence: u64,
	) -> Result<Option<(super::SemanticOpId, u64)>> {
		let Some(value) = self
			.semantic_values
			.get(&matrix_value)
			.map(|binding| binding.value)
		else {
			return Ok(None);
		};
		let Some(forward) = self.semantic.values()[value.index() as usize].producer() else {
			return Ok(None);
		};
		let output_index = self.semantic.operations()[forward.index() as usize]
			.outputs()
			.iter()
			.position(|output| *output == value)
			.ok_or_else(|| Error::internal("semantic producer omitted its output value"))?;
		self.semantic
			.attach_autograd(forward, output_index, sequence)?;
		Ok(Some((forward, self.semantic.generation())))
	}

	pub(super) fn semantic_operation_count(&self) -> usize {
		self.semantic.operations().len()
	}

	pub(super) fn complete_autograd(
		&mut self,
		forward: super::SemanticOpId,
		sequence: u64,
		generation: u64,
		backward_first: usize,
	) -> Result<()> {
		if self.semantic.generation() != generation {
			return Ok(());
		}
		let first = u32::try_from(backward_first)
			.map_err(|_| Error::resource_exhausted("semantic backward start exceeds u32"))?;
		let end = self.semantic.operations().len();
		let count = end
			.checked_sub(backward_first)
			.ok_or_else(|| Error::internal("semantic backward range moved backwards"))?;
		self.semantic.complete_autograd(
			forward,
			sequence,
			super::SemanticOpId::from_index(first),
			count,
		)
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

	pub(super) fn into_parts(
		self,
	) -> (
		ExecutableGraph,
		Vec<Storage>,
		SemanticGraph,
		Vec<StableResourceSnapshot>,
		Vec<SemanticStorageSnapshot>,
		Vec<Storage>,
	) {
		(
			self.graph,
			self.outputs,
			self.semantic,
			self.stable_resources,
			self.semantic_bindings,
			self.observed_outputs,
		)
	}
}

fn admit_matrix(
	graph: &mut SemanticGraph,
	values: &mut BTreeMap<u64, SemanticValueBinding>,
	matrix: &Matrix,
	external: bool,
) -> Result<SemanticValueId> {
	admit_matrix_semantic(graph, values, matrix.semantic(), matrix.storage(), external)
}

fn admit_matrix_semantic(
	graph: &mut SemanticGraph,
	values: &mut BTreeMap<u64, SemanticValueBinding>,
	matrix: &std::rc::Rc<MatrixSemantic>,
	storage: &Storage,
	external: bool,
) -> Result<SemanticValueId> {
	if let Some(id) = values.get(&matrix.id()) {
		if !id.storage.same_as(storage) {
			return Err(Error::internal(
				"one semantic matrix value was recorded with multiple storage identities",
			));
		}
		return Ok(id.value);
	}
	let source = matrix
		.view_source()
		.map(|source| admit_matrix_semantic(graph, values, source, storage, true))
		.transpose()?;
	let value = graph.add_value(
		SemanticValueDesc::new(
			format!("matrix.{}", matrix.id()),
			OpValueKind::Matrix,
			matrix.shape(),
			matrix.dtype(),
		)?
		.external(external && source.is_none()),
	)?;
	values.insert(
		matrix.id(),
		SemanticValueBinding {
			value,
			storage: storage.clone(),
		},
	);
	if let Some(source) = source {
		graph.add_view(source, value, matrix.view_byte_offset())?;
	}
	Ok(value)
}
