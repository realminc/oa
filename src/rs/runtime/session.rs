use std::collections::BTreeMap;

use crate::{Audio, Error, Image, Matrix, OpValueKind, Result, core::MatrixSemantic};

use super::{
	AudioSemanticDispatch, AudioSemanticOutput, BufferAccess, ComputeDispatch,
	ImageSemanticDispatch, ImageSemanticInput, OptionalSemanticDispatch, SemanticDispatch,
	SemanticGraph, SemanticValueDesc, SemanticValueId, Storage, executable_graph::ExecutableGraph,
	vk,
};

/// Private mutable eager-recording owner for one engine.
pub(super) struct ExecutionSession {
	graphs: Vec<ExecutableGraph>,
	outputs: Vec<Storage>,
	semantic: SemanticGraph,
	semantic_values: BTreeMap<SemanticValueKey, SemanticValueBinding>,
	semantic_storage: BTreeMap<SemanticValueId, Storage>,
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum SemanticValueKey {
	Matrix(u64),
	Audio(u64),
	Image(u64),
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
			semantic_storage: BTreeMap::new(),
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

	#[cfg(test)]
	pub(super) fn record(
		&mut self,
		device: &vk::Device,
		dispatch: ComputeDispatch<'_>,
	) -> Result<()> {
		self.record_impl(device, dispatch, &[])
	}

	pub(super) fn record_semantic(
		&mut self,
		device: &vk::Device,
		dispatch: ComputeDispatch<'_>,
		semantic: SemanticDispatch<'_>,
	) -> Result<()> {
		self.record_impl(device, dispatch, std::slice::from_ref(&semantic))
	}

	pub(super) fn record_optional_semantic(
		&mut self,
		device: &vk::Device,
		dispatch: ComputeDispatch<'_>,
		semantic: OptionalSemanticDispatch<'_>,
	) -> Result<()> {
		semantic.validate_kernel(dispatch.kernel)?;
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
		let mut candidate = self.semantic.clone();
		let mut values = self.semantic_values.clone();
		let mut storage = self.semantic_storage.clone();
		let operation =
			admit_optional_semantic_dispatch(&mut candidate, &mut values, &mut storage, &semantic)?;
		candidate.validate()?;
		graph.attach_direct_semantic(
			operation,
			semantic.contract.name(),
			semantic.contract.hash(),
		)?;
		self.semantic = candidate;
		self.semantic_values = values;
		self.semantic_storage = storage;
		self.graphs.push(graph);
		for output in &outputs {
			output.mark_recorded();
		}
		self.outputs.extend(outputs);
		Ok(())
	}

	pub(super) fn record_fused_semantic(
		&mut self,
		device: &vk::Device,
		dispatch: ComputeDispatch<'_>,
		semantics: &[SemanticDispatch<'_>],
	) -> Result<()> {
		if semantics.len() < 2 {
			return Err(Error::invalid_argument(
				"fused semantic recording requires at least two operations",
			));
		}
		self.record_impl(device, dispatch, semantics)
	}

	pub(super) fn record_split_semantic(
		&mut self,
		device: &vk::Device,
		dispatches: &[ComputeDispatch<'_>],
		semantic: SemanticDispatch<'_>,
	) -> Result<()> {
		if dispatches.len() < 2 {
			return Err(Error::invalid_argument(
				"split semantic recording requires multiple executable dispatches",
			));
		}
		if !dispatches.iter().any(|dispatch| {
			dispatch.kernel.semantic_contract().is_some_and(|contract| {
				contract.name() == semantic.contract.name()
					&& contract.hash() == semantic.contract.hash()
			})
		}) {
			return Err(Error::internal(format!(
				"split lowering has no physical kernel registered for {}",
				semantic.contract.name()
			)));
		}
		for dispatch in dispatches {
			for binding in dispatch.buffers {
				binding.storage.validate_recording_access()?;
			}
		}
		let outputs = dispatches
			.iter()
			.flat_map(|dispatch| dispatch.buffers)
			.filter(|binding| binding.access != BufferAccess::Read)
			.map(|binding| binding.storage.clone())
			.collect::<Vec<_>>();
		let mut graph = ExecutableGraph::from_dispatches(device, dispatches)?;
		let mut candidate = self.semantic.clone();
		let mut values = self.semantic_values.clone();
		let mut storage = self.semantic_storage.clone();
		let operation =
			admit_semantic_dispatch(&mut candidate, &mut values, &mut storage, &semantic)?;
		candidate.validate()?;
		graph.attach_split_semantic(
			operation,
			semantic.contract.name(),
			semantic.contract.hash(),
		)?;
		self.semantic = candidate;
		self.semantic_values = values;
		self.semantic_storage = storage;
		self.graphs.push(graph);
		for output in &outputs {
			output.mark_recorded();
		}
		self.outputs.extend(outputs);
		Ok(())
	}

	pub(super) fn record_split_optional_semantic(
		&mut self,
		device: &vk::Device,
		dispatches: &[ComputeDispatch<'_>],
		semantic: OptionalSemanticDispatch<'_>,
	) -> Result<()> {
		if dispatches.len() < 2 {
			return Err(Error::invalid_argument(
				"split optional semantic recording requires multiple executable dispatches",
			));
		}
		let owner = dispatches.iter().find(|dispatch| {
			dispatch.kernel.semantic_contract().is_some_and(|contract| {
				contract.name() == semantic.contract.name()
					&& contract.hash() == semantic.contract.hash()
			})
		});
		let Some(owner) = owner else {
			return Err(Error::internal(format!(
				"split lowering has no physical kernel registered for {}",
				semantic.contract.name()
			)));
		};
		semantic.validate_kernel(owner.kernel)?;
		for dispatch in dispatches {
			for binding in dispatch.buffers {
				binding.storage.validate_recording_access()?;
			}
		}
		let outputs = dispatches
			.iter()
			.flat_map(|dispatch| dispatch.buffers)
			.filter(|binding| binding.access != BufferAccess::Read)
			.map(|binding| binding.storage.clone())
			.collect::<Vec<_>>();
		let mut graph = ExecutableGraph::from_dispatches(device, dispatches)?;
		let mut candidate = self.semantic.clone();
		let mut values = self.semantic_values.clone();
		let mut storage = self.semantic_storage.clone();
		let operation =
			admit_optional_semantic_dispatch(&mut candidate, &mut values, &mut storage, &semantic)?;
		candidate.validate()?;
		graph.attach_split_semantic(
			operation,
			semantic.contract.name(),
			semantic.contract.hash(),
		)?;
		self.semantic = candidate;
		self.semantic_values = values;
		self.semantic_storage = storage;
		self.graphs.push(graph);
		for output in &outputs {
			output.mark_recorded();
		}
		self.outputs.extend(outputs);
		Ok(())
	}

	pub(super) fn record_audio_semantic(
		&mut self,
		device: &vk::Device,
		dispatch: ComputeDispatch<'_>,
		semantic: AudioSemanticDispatch<'_>,
	) -> Result<()> {
		semantic.validate_kernel(dispatch.kernel)?;
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
		let mut candidate = self.semantic.clone();
		let mut values = self.semantic_values.clone();
		let mut storage = self.semantic_storage.clone();
		let operation =
			admit_audio_semantic_dispatch(&mut candidate, &mut values, &mut storage, &semantic)?;
		candidate.validate()?;
		graph.attach_direct_semantic(
			operation,
			semantic.contract.name(),
			semantic.contract.hash(),
		)?;
		self.semantic = candidate;
		self.semantic_values = values;
		self.semantic_storage = storage;
		self.graphs.push(graph);
		for output in &outputs {
			output.mark_recorded();
		}
		self.outputs.extend(outputs);
		Ok(())
	}

	pub(super) fn record_image_semantic(
		&mut self,
		device: &vk::Device,
		dispatch: ComputeDispatch<'_>,
		semantic: ImageSemanticDispatch<'_>,
	) -> Result<()> {
		semantic.validate_kernel(dispatch.kernel)?;
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
		let mut candidate = self.semantic.clone();
		let mut values = self.semantic_values.clone();
		let mut storage = self.semantic_storage.clone();
		let operation =
			admit_image_semantic_dispatch(&mut candidate, &mut values, &mut storage, &semantic)?;
		candidate.validate()?;
		graph.attach_direct_semantic(
			operation,
			semantic.contract.name(),
			semantic.contract.hash(),
		)?;
		self.semantic = candidate;
		self.semantic_values = values;
		self.semantic_storage = storage;
		self.graphs.push(graph);
		for output in &outputs {
			output.mark_recorded();
		}
		self.outputs.extend(outputs);
		Ok(())
	}

	pub(super) fn record_audio_split_semantic(
		&mut self,
		device: &vk::Device,
		dispatches: &[ComputeDispatch<'_>],
		semantic: AudioSemanticDispatch<'_>,
	) -> Result<()> {
		if dispatches.len() < 2 {
			return Err(Error::invalid_argument(
				"split Audio semantic recording requires multiple executable dispatches",
			));
		}
		for dispatch in dispatches {
			for binding in dispatch.buffers {
				binding.storage.validate_recording_access()?;
			}
		}
		let outputs = dispatches
			.iter()
			.flat_map(|dispatch| dispatch.buffers)
			.filter(|binding| binding.access != BufferAccess::Read)
			.map(|binding| binding.storage.clone())
			.collect::<Vec<_>>();
		let mut graph = ExecutableGraph::from_dispatches(device, dispatches)?;
		let mut candidate = self.semantic.clone();
		let mut values = self.semantic_values.clone();
		let mut storage = self.semantic_storage.clone();
		let operation =
			admit_audio_semantic_dispatch(&mut candidate, &mut values, &mut storage, &semantic)?;
		candidate.validate()?;
		graph.attach_split_semantic(
			operation,
			semantic.contract.name(),
			semantic.contract.hash(),
		)?;
		self.semantic = candidate;
		self.semantic_values = values;
		self.semantic_storage = storage;
		self.graphs.push(graph);
		for output in &outputs {
			output.mark_recorded();
		}
		self.outputs.extend(outputs);
		Ok(())
	}

	fn record_impl(
		&mut self,
		device: &vk::Device,
		dispatch: ComputeDispatch<'_>,
		semantics: &[SemanticDispatch<'_>],
	) -> Result<()> {
		if semantics.len() == 1 {
			semantics[0].validate_kernel(dispatch.kernel)?;
		} else if !semantics.is_empty() && dispatch.kernel.semantic_contract().is_some() {
			return Err(Error::internal(format!(
				"direct kernel {} cannot own fused semantic operations",
				dispatch.kernel.report_name()
			)));
		}
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
		if !semantics.is_empty() {
			let mut candidate = self.semantic.clone();
			let mut values = self.semantic_values.clone();
			let mut storage = self.semantic_storage.clone();
			let mut operations = Vec::with_capacity(semantics.len());
			for semantic in semantics {
				operations.push(admit_semantic_dispatch(
					&mut candidate,
					&mut values,
					&mut storage,
					semantic,
				)?);
			}
			candidate.validate()?;
			if let [operation] = operations.as_slice() {
				graph.attach_direct_semantic(
					*operation,
					semantics[0].contract.name(),
					semantics[0].contract.hash(),
				)?;
			} else {
				graph.attach_fused_semantic(&operations)?;
			}
			self.semantic = candidate;
			self.semantic_values = values;
			self.semantic_storage = storage;
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
		self.semantic_storage.clear();
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
		if self.semantic_storage.len() != self.semantic.values().len() {
			return Err(Error::failed_precondition(
				"semantic storage snapshot requires one binding per value",
			));
		}

		let stable_resources = self.snapshot_stable_resources()?;
		let graphs = std::mem::take(&mut self.graphs);
		let outputs = std::mem::take(&mut self.outputs);
		let semantic = self.semantic.take_recording()?;
		self.semantic_values.clear();
		let semantic_storage = std::mem::take(&mut self.semantic_storage);
		let semantic_bindings = semantic_storage
			.into_iter()
			.map(|(value, storage)| SemanticStorageSnapshot { value, storage })
			.collect::<Vec<_>>();
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
		if self.semantic_storage.len() != self.semantic.values().len() {
			return Err(Error::failed_precondition(
				"semantic storage snapshot requires one binding per value",
			));
		}

		let stable_resources = self.snapshot_stable_resources()?;
		let graph = ExecutableGraph::join(self.graphs.clone())?;
		let outputs = self.outputs.clone();
		let semantic = self.semantic.clone();
		let semantic_bindings = self
			.semantic_storage
			.iter()
			.map(|(value, storage)| SemanticStorageSnapshot {
				value: *value,
				storage: storage.clone(),
			})
			.collect::<Vec<_>>();
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
		self.semantic_storage.clear();
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
			.get(&SemanticValueKey::Matrix(matrix_value))
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

fn admit_semantic_dispatch(
	graph: &mut SemanticGraph,
	values: &mut BTreeMap<SemanticValueKey, SemanticValueBinding>,
	storage: &mut BTreeMap<SemanticValueId, Storage>,
	semantic: &SemanticDispatch<'_>,
) -> Result<super::SemanticOpId> {
	let inputs = semantic
		.inputs
		.iter()
		.map(|matrix| admit_matrix(graph, values, storage, matrix, true).map(Some))
		.collect::<Result<Vec<_>>>()?;
	let outputs = semantic
		.outputs
		.iter()
		.enumerate()
		.map(|(output_index, matrix)| {
			let Some(input_index) = semantic.contract.alias_input(output_index) else {
				return admit_matrix(graph, values, storage, matrix, false);
			};
			let input = semantic
				.inputs
				.get(input_index)
				.ok_or_else(|| Error::internal("semantic alias input is outside the dispatch"))?;
			if !matrix.storage().same_as(input.storage()) {
				return Err(Error::invalid_argument(
					"semantic alias output must retain its input storage",
				));
			}
			if matrix.same_value_as(input) {
				if !semantic.contract.mutates_input(input_index) {
					return Err(Error::invalid_argument(
						"semantic in-place alias requires a declared input mutation",
					));
				}
				return admit_matrix_version(
					graph,
					values,
					storage,
					matrix,
					inputs[input_index].ok_or_else(|| {
						Error::internal("semantic alias references an absent input")
					})?,
				);
			}
			admit_matrix(graph, values, storage, matrix, false)
		})
		.collect::<Result<Vec<_>>>()?;
	graph.add_operation(
		semantic.contract,
		&inputs,
		&outputs,
		&[],
		semantic.attributes,
	)
}

fn admit_optional_semantic_dispatch(
	graph: &mut SemanticGraph,
	values: &mut BTreeMap<SemanticValueKey, SemanticValueBinding>,
	storage: &mut BTreeMap<SemanticValueId, Storage>,
	semantic: &OptionalSemanticDispatch<'_>,
) -> Result<super::SemanticOpId> {
	if !semantic.contract.accepts_input_count(semantic.inputs.len()) {
		return Err(Error::invalid_argument(
			"optional semantic input count does not match its contract",
		));
	}
	let inputs = semantic
		.inputs
		.iter()
		.enumerate()
		.map(|(index, matrix)| match matrix {
			Some(matrix) => admit_matrix(graph, values, storage, matrix, true).map(Some),
			None if semantic.contract.input_is_optional(index) => Ok(None),
			None => Err(Error::invalid_argument(
				"absent semantic input is not optional in its contract",
			)),
		})
		.collect::<Result<Vec<_>>>()?;
	let outputs = semantic
		.outputs
		.iter()
		.enumerate()
		.map(|(output_index, matrix)| {
			if semantic.contract.alias_input(output_index).is_some() {
				return Err(Error::failed_precondition(
					"optional semantic output aliasing is not admitted",
				));
			}
			admit_matrix(graph, values, storage, matrix, false)
		})
		.collect::<Result<Vec<_>>>()?;
	graph.add_operation(
		semantic.contract,
		&inputs,
		&outputs,
		&[],
		semantic.attributes,
	)
}

fn admit_audio_semantic_dispatch(
	graph: &mut SemanticGraph,
	values: &mut BTreeMap<SemanticValueKey, SemanticValueBinding>,
	storage: &mut BTreeMap<SemanticValueId, Storage>,
	semantic: &AudioSemanticDispatch<'_>,
) -> Result<super::SemanticOpId> {
	if semantic
		.outputs
		.iter()
		.enumerate()
		.any(|(index, _)| semantic.contract.alias_input(index).is_some())
	{
		return Err(Error::failed_precondition(
			"Audio semantic output aliasing is not admitted",
		));
	}
	let inputs = semantic
		.inputs
		.iter()
		.map(|audio| admit_audio(graph, values, storage, audio, true).map(Some))
		.collect::<Result<Vec<_>>>()?;
	let outputs = semantic
		.outputs
		.iter()
		.map(|output| match output {
			AudioSemanticOutput::Audio(audio) => admit_audio(graph, values, storage, audio, false),
			AudioSemanticOutput::Matrix(matrix) => {
				admit_matrix(graph, values, storage, matrix, false)
			}
		})
		.collect::<Result<Vec<_>>>()?;
	graph.add_operation(
		semantic.contract,
		&inputs,
		&outputs,
		&[],
		semantic.attributes,
	)
}

fn admit_image_semantic_dispatch(
	graph: &mut SemanticGraph,
	values: &mut BTreeMap<SemanticValueKey, SemanticValueBinding>,
	storage: &mut BTreeMap<SemanticValueId, Storage>,
	semantic: &ImageSemanticDispatch<'_>,
) -> Result<super::SemanticOpId> {
	if semantic
		.outputs
		.iter()
		.enumerate()
		.any(|(index, _)| semantic.contract.alias_input(index).is_some())
	{
		return Err(Error::failed_precondition(
			"Image semantic output aliasing is not admitted",
		));
	}
	let inputs = semantic
		.inputs
		.iter()
		.map(|input| match input {
			ImageSemanticInput::Image(image) => {
				admit_image(graph, values, storage, image, true).map(Some)
			}
			ImageSemanticInput::Matrix(matrix) => {
				admit_matrix(graph, values, storage, matrix, true).map(Some)
			}
		})
		.collect::<Result<Vec<_>>>()?;
	let outputs = semantic
		.outputs
		.iter()
		.map(|image| admit_image(graph, values, storage, image, false))
		.collect::<Result<Vec<_>>>()?;
	graph.add_operation(
		semantic.contract,
		&inputs,
		&outputs,
		&[],
		semantic.attributes,
	)
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
	values: &mut BTreeMap<SemanticValueKey, SemanticValueBinding>,
	storage_bindings: &mut BTreeMap<SemanticValueId, Storage>,
	matrix: &Matrix,
	external: bool,
) -> Result<SemanticValueId> {
	admit_matrix_semantic(
		graph,
		values,
		storage_bindings,
		matrix.semantic(),
		matrix.storage(),
		external,
	)
}

fn admit_matrix_semantic(
	graph: &mut SemanticGraph,
	values: &mut BTreeMap<SemanticValueKey, SemanticValueBinding>,
	storage_bindings: &mut BTreeMap<SemanticValueId, Storage>,
	matrix: &std::rc::Rc<MatrixSemantic>,
	storage: &Storage,
	external: bool,
) -> Result<SemanticValueId> {
	let key = SemanticValueKey::Matrix(matrix.id());
	if let Some(id) = values.get(&key) {
		if !id.storage.same_as(storage) {
			return Err(Error::internal(
				"one semantic matrix value was recorded with multiple storage identities",
			));
		}
		return Ok(id.value);
	}
	let source = matrix
		.view_source()
		.map(|source| admit_matrix_semantic(graph, values, storage_bindings, source, storage, true))
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
		key,
		SemanticValueBinding {
			value,
			storage: storage.clone(),
		},
	);
	storage_bindings.insert(value, storage.clone());
	if let Some(source) = source {
		graph.add_view(source, value, matrix.view_byte_offset())?;
	}
	Ok(value)
}

fn admit_matrix_version(
	graph: &mut SemanticGraph,
	values: &mut BTreeMap<SemanticValueKey, SemanticValueBinding>,
	storage_bindings: &mut BTreeMap<SemanticValueId, Storage>,
	matrix: &Matrix,
	expected_input: SemanticValueId,
) -> Result<SemanticValueId> {
	let key = SemanticValueKey::Matrix(matrix.value_id());
	let current = values.get(&key).ok_or_else(|| {
		Error::internal("semantic mutation input was not admitted before its output")
	})?;
	if current.value != expected_input || !current.storage.same_as(matrix.storage()) {
		return Err(Error::internal(
			"semantic mutation does not target the current value and storage version",
		));
	}
	let value = graph.add_value(SemanticValueDesc::new(
		format!("matrix.{}", matrix.value_id()),
		OpValueKind::Matrix,
		matrix.shape(),
		matrix.dtype(),
	)?)?;
	values.insert(
		key,
		SemanticValueBinding {
			value,
			storage: matrix.storage().clone(),
		},
	);
	storage_bindings.insert(value, matrix.storage().clone());
	Ok(value)
}

fn admit_audio(
	graph: &mut SemanticGraph,
	values: &mut BTreeMap<SemanticValueKey, SemanticValueBinding>,
	storage_bindings: &mut BTreeMap<SemanticValueId, Storage>,
	audio: &Audio,
	external: bool,
) -> Result<SemanticValueId> {
	let key = SemanticValueKey::Audio(audio.value_id());
	if let Some(binding) = values.get(&key) {
		if !binding.storage.same_as(audio.storage()) {
			return Err(Error::internal(
				"one semantic Audio value was recorded with multiple storage identities",
			));
		}
		return Ok(binding.value);
	}
	let value = graph.add_value(
		SemanticValueDesc::new(
			format!("audio.{}", audio.value_id()),
			OpValueKind::Audio,
			audio.as_matrix().shape(),
			audio.as_matrix().dtype(),
		)?
		.external(external),
	)?;
	values.insert(
		key,
		SemanticValueBinding {
			value,
			storage: audio.storage().clone(),
		},
	);
	storage_bindings.insert(value, audio.storage().clone());
	Ok(value)
}

fn admit_image(
	graph: &mut SemanticGraph,
	values: &mut BTreeMap<SemanticValueKey, SemanticValueBinding>,
	storage_bindings: &mut BTreeMap<SemanticValueId, Storage>,
	image: &Image,
	external: bool,
) -> Result<SemanticValueId> {
	let key = SemanticValueKey::Image(image.value_id());
	if let Some(binding) = values.get(&key) {
		if !binding.storage.same_as(image.storage()) {
			return Err(Error::internal(
				"one semantic Image value was recorded with multiple storage identities",
			));
		}
		return Ok(binding.value);
	}
	let value = graph.add_value(
		SemanticValueDesc::new(
			format!("image.{}", image.value_id()),
			OpValueKind::Image,
			image.as_matrix().shape(),
			image.as_matrix().dtype(),
		)?
		.external(external),
	)?;
	values.insert(
		key,
		SemanticValueBinding {
			value,
			storage: image.storage().clone(),
		},
	);
	storage_bindings.insert(value, image.storage().clone());
	Ok(value)
}
