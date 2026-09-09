//! Private DNN semantic-to-executable replacement.
//!
//! Port provenance: `oa/source/cpp/lib/oa/runtime/dnn/graphLowering.cpp`.

use crate::{DType, Result};

use super::{
	DnnEngineType, DnnOpDesc, DnnOpType, DnnPartition, DnnPartitionState, DnnPlan, DnnValueDesc,
};
use crate::runtime::{
	BufferAccess, CapturedResourceDesc, KernelId, PushConstant, SemanticGraph, SemanticOpId,
	SemanticStorageBinding,
	executable_graph::{BufferUse, ComputeNode, ExecutableGraph, ResourceLifetime},
};

const INVALID_NODE: usize = usize::MAX;

struct NodeRange {
	first: usize,
	count: usize,
	rejection: &'static str,
}

struct QkvProjectionCandidate {
	first_node: usize,
	node_count: usize,
	node: ComputeNode,
}

struct GateUpCandidate {
	first_node: usize,
	node_count: usize,
	node: ComputeNode,
}

pub(super) fn lower_captured_plan(
	plan: &mut DnnPlan,
	semantic: &SemanticGraph,
	bindings: &[SemanticStorageBinding],
	resources: &[ResourceLifetime],
	resource_descriptions: &[CapturedResourceDesc],
	allow_inference_replacement: bool,
	source: &ExecutableGraph,
) -> Result<ExecutableGraph> {
	plan.applied_partition_count = 0;
	plan.inherited_partition_count = 0;
	plan.fallback_partition_count = 0;
	plan.unexpected_fallback_count = 0;

	let mut lowered = Vec::with_capacity(source.nodes().len());
	let mut source_cursor = 0;
	for partition_index in 0..plan.partitions.len() {
		plan.partitions[partition_index].state = DnnPartitionState::Analyzed;
		plan.partitions[partition_index].fallback_reason = None;
		let engine = plan.partitions[partition_index].engine;
		let operation_count = plan.partitions[partition_index].operations.len();
		if engine == DnnEngineType::Portable {
			continue;
		}

		if engine == DnnEngineType::BlasLtEpilogue && operation_count == 1 {
			let range = partition_node_range(&plan.partitions[partition_index], source);
			if range.count == 1 {
				mark_inherited(plan, partition_index);
			} else {
				let reason = if range.rejection.is_empty() {
					"single-operation epilogue is not one executable dispatch"
				} else {
					range.rejection
				};
				mark_fallback(plan, partition_index, reason, false);
			}
			continue;
		}

		if engine == DnnEngineType::Attention && operation_count == 1 {
			let source_operation = plan.partitions[partition_index].operations[0];
			let operation = find_operation(plan, source_operation);
			let saved_state = operation
				.and_then(|operation| operation.outputs.get(1))
				.and_then(|value| find_value(plan, *value));
			if operation.is_some_and(|operation| operation.training)
				&& saved_state.is_some_and(|value| value.shape.len() == 2)
				&& !plan.policy.allow_recompute
			{
				mark_fallback(
					plan,
					partition_index,
					"attention source requires backward recomputation forbidden by policy",
					false,
				);
				continue;
			}
			let range = partition_node_range(&plan.partitions[partition_index], source);
			if range.count != 0 {
				mark_inherited(plan, partition_index);
			} else {
				let reason = if range.rejection.is_empty() {
					"attention source has no executable provider"
				} else {
					range.rejection
				};
				mark_fallback(plan, partition_index, reason, false);
			}
			continue;
		}

		if !allow_inference_replacement {
			mark_fallback(
				plan,
				partition_index,
				"training-program capture retains the proven source lowering",
				false,
			);
			continue;
		}

		if engine == DnnEngineType::GatedFfn {
			let candidate = match build_gate_up(
				plan,
				partition_index,
				semantic,
				bindings,
				resources,
				resource_descriptions,
				source,
			) {
				Ok(candidate) => candidate,
				Err(reason) => {
					mark_fallback(plan, partition_index, reason, false);
					continue;
				}
			};
			if candidate.first_node < source_cursor {
				mark_fallback(
					plan,
					partition_index,
					"gate/up partition overlaps an earlier applied source interval",
					true,
				);
				continue;
			}
			lowered.extend_from_slice(&source.nodes()[source_cursor..candidate.first_node]);
			lowered.push(candidate.node);
			source_cursor = candidate.first_node + candidate.node_count;
			plan.partitions[partition_index].state = DnnPartitionState::Applied;
			plan.applied_partition_count = plan.applied_partition_count.saturating_add(1);
			continue;
		}

		if engine != DnnEngineType::QkvProjectionGroup {
			mark_fallback(
				plan,
				partition_index,
				"recognized partition has no admitted executable provider",
				false,
			);
			continue;
		}

		let candidate =
			match build_qkv_projection(plan, partition_index, bindings, resources, source) {
				Ok(candidate) => candidate,
				Err(reason) => {
					mark_fallback(plan, partition_index, reason, false);
					continue;
				}
			};
		if candidate.first_node < source_cursor {
			mark_fallback(
				plan,
				partition_index,
				"QKV partition overlaps an earlier applied source interval",
				true,
			);
			continue;
		}
		lowered.extend_from_slice(&source.nodes()[source_cursor..candidate.first_node]);
		lowered.push(candidate.node);
		source_cursor = candidate.first_node + candidate.node_count;
		plan.partitions[partition_index].state = DnnPartitionState::Applied;
		plan.applied_partition_count = plan.applied_partition_count.saturating_add(1);
	}
	lowered.extend_from_slice(&source.nodes()[source_cursor..]);
	ExecutableGraph::from_nodes(lowered)
}

fn build_qkv_projection(
	plan: &DnnPlan,
	partition_index: usize,
	bindings: &[SemanticStorageBinding],
	resources: &[ResourceLifetime],
	source: &ExecutableGraph,
) -> std::result::Result<QkvProjectionCandidate, &'static str> {
	let partition = &plan.partitions[partition_index];
	if partition.operations.len() != 3 {
		return Err("QKV multi-output provider requires exactly three projections");
	}
	let operations = partition
		.operations
		.iter()
		.map(|operation| find_operation(plan, *operation))
		.collect::<Option<Vec<_>>>()
		.ok_or("QKV multi-output operation is unavailable")?;
	let shared = *operations[0]
		.inputs
		.first()
		.ok_or("QKV multi-output projection has no activation")?;
	if operations.iter().enumerate().any(|(index, operation)| {
		operation.op_type != DnnOpType::Matmul
			|| operation.epilogue != super::DnnEpilogue::Bias
			|| operation.training
			|| operation.inputs.len() != 3
			|| operation.outputs.len() != 1
			|| (index != 0 && operation.inputs.first() != Some(&shared))
	}) {
		return Err("QKV multi-output provider requires three inference linear+bias projections");
	}

	let value_ids = [
		operations[0].inputs[0],
		operations[0].inputs[1],
		operations[1].inputs[1],
		operations[2].inputs[1],
		operations[0].inputs[2],
		operations[1].inputs[2],
		operations[2].inputs[2],
		operations[0].outputs[0],
		operations[1].outputs[0],
		operations[2].outputs[0],
	];
	let values = value_ids
		.iter()
		.map(|id| find_value(plan, *id))
		.collect::<Option<Vec<_>>>()
		.ok_or("QKV multi-output value is missing layout metadata")?;
	let mut resource_ids = Vec::with_capacity(value_ids.len());
	for (id, value) in value_ids.iter().zip(&values) {
		let binding = bindings
			.iter()
			.find(|binding| binding.value() == *id)
			.ok_or("QKV multi-output value is missing a storage binding")?;
		let resource = binding.resource() as usize;
		if resource >= resources.len() {
			return Err("QKV multi-output resource binding is unavailable");
		}
		let byte_len = value
			.shape
			.iter()
			.try_fold(value.dtype.size_bytes(), |bytes, extent| {
				bytes.checked_mul(*extent)
			})
			.ok_or("QKV multi-output physical byte size overflows")?;
		if value.dtype != DType::F32
			|| value.storage_id != value.id
			|| value.byte_offset != 0
			|| !exact_row_major(value)
		{
			return Err(
				"QKV multi-output provider requires distinct zero-offset row-major F32 values",
			);
		}
		if resources[resource].buffer.byte_len() != byte_len {
			return Err("QKV multi-output semantic and physical layouts disagree");
		}
		if resource_ids.contains(&resource) {
			return Err("QKV multi-output provider rejects physical resource aliases");
		}
		resource_ids.push(resource);
	}
	if values[0].shape != [1024, 32] {
		return Err("QKV multi-output provider is qualified only for M=1024, K=32");
	}
	for projection in 0..3 {
		if values[1 + projection].shape != [32, 32]
			|| values[4 + projection].shape != [32]
			|| values[7 + projection].shape != [1024, 32]
		{
			return Err("QKV multi-output provider is qualified only for N=K=32");
		}
	}

	let range = partition_node_range(partition, source);
	if range.count == 0 {
		return Err(range.rejection);
	}
	if range.count == 1 {
		return Err("QKV multi-output source is already one executable dispatch");
	}
	let buffers = resource_ids
		.iter()
		.enumerate()
		.map(|(index, resource)| BufferUse {
			buffer: resources[*resource].buffer.clone(),
			access: if index < 7 {
				BufferAccess::Read
			} else {
				BufferAccess::Write
			},
		})
		.collect();
	Ok(QkvProjectionCandidate {
		first_node: range.first,
		node_count: range.count,
		node: ComputeNode {
			operation: "oa::Dnn::QkvProjectionBias",
			kernel: KernelId::MlQkvProjectionBiasF32,
			buffers,
			push_constants: vec![
				PushConstant::U32(1024),
				PushConstant::U32(32),
				PushConstant::U32(32),
			],
			workgroups: [32, 1, 1],
			semantic_ops: partition.operations.clone(),
			op_contract_hash: 0,
		},
	})
}

fn build_gate_up(
	plan: &DnnPlan,
	partition_index: usize,
	semantic: &SemanticGraph,
	bindings: &[SemanticStorageBinding],
	resources: &[ResourceLifetime],
	resource_descriptions: &[CapturedResourceDesc],
	source: &ExecutableGraph,
) -> std::result::Result<GateUpCandidate, &'static str> {
	let partition = &plan.partitions[partition_index];
	if partition.operations.len() != 3 {
		return Err("gate/up provider requires two projections and one gated multiply");
	}
	let operations = partition
		.operations
		.iter()
		.map(|operation| find_operation(plan, *operation))
		.collect::<Option<Vec<_>>>()
		.ok_or("gate/up partition is missing captured operations")?;
	let gate = operations[0];
	let up = operations[1];
	let gated = operations[2];
	if gate.op_type != DnnOpType::Matmul
		|| up.op_type != DnnOpType::Matmul
		|| gated.op_type != DnnOpType::GatedMultiply
		|| gate.epilogue != super::DnnEpilogue::Bias
		|| up.epilogue != super::DnnEpilogue::Bias
		|| operations.iter().any(|operation| operation.training)
		|| gate.inputs.len() != 3
		|| up.inputs.len() != 3
		|| gated.inputs.len() != 2
		|| gate.outputs.len() != 1
		|| up.outputs.len() != 1
		|| gated.outputs.len() != 1
	{
		return Err("gate/up provider requires inference linear+bias, linear+bias, SwiGLU");
	}
	if gate.inputs[0] != up.inputs[0]
		|| gated.inputs[0] != gate.outputs[0]
		|| gated.inputs[1] != up.outputs[0]
	{
		return Err("gate/up provider received an ambiguous projection topology");
	}

	let value_ids = [
		gate.inputs[0],
		gate.inputs[1],
		up.inputs[1],
		gate.inputs[2],
		up.inputs[2],
		gate.outputs[0],
		up.outputs[0],
		gated.outputs[0],
	];
	let values = value_ids
		.iter()
		.map(|id| find_value(plan, *id))
		.collect::<Option<Vec<_>>>()
		.ok_or("gate/up value is missing layout metadata")?;
	let mut resource_ids = Vec::with_capacity(value_ids.len());
	for (id, value) in value_ids.iter().zip(&values) {
		let binding = bindings
			.iter()
			.find(|binding| binding.value() == *id)
			.ok_or("gate/up value is missing a storage binding")?;
		let resource = binding.resource() as usize;
		if resource >= resources.len() || resource >= resource_descriptions.len() {
			return Err("gate/up resource binding is unavailable");
		}
		let byte_len = value
			.shape
			.iter()
			.try_fold(value.dtype.size_bytes(), |bytes, extent| {
				bytes.checked_mul(*extent)
			})
			.ok_or("gate/up physical byte size overflows")?;
		if value.dtype != DType::F32
			|| value.storage_id != value.id
			|| value.byte_offset != 0
			|| !exact_row_major(value)
		{
			return Err("gate/up provider requires distinct zero-offset row-major F32 values");
		}
		if resources[resource].buffer.byte_len() != byte_len
			|| resource_descriptions[resource].resource() as usize != resource
		{
			return Err("gate/up semantic and physical layouts disagree");
		}
		if resource_ids.contains(&resource) {
			return Err("gate/up provider rejects physical resource aliases");
		}
		resource_ids.push(resource);
	}

	if values[0].shape != [1024, 32]
		|| values[1].shape != [64, 32]
		|| values[2].shape != [64, 32]
		|| values[3].shape != [64]
		|| values[4].shape != [64]
		|| values[5].shape != [1024, 64]
		|| values[6].shape != [1024, 64]
		|| values[7].shape != [1024, 64]
	{
		return Err("gate/up provider is qualified only for M=1024, N=64, K=32");
	}

	for index in 5..=6 {
		let value_id = value_ids[index];
		let value = semantic
			.find_value(value_id)
			.ok_or("gate/up intermediate semantic value is unavailable")?;
		if value.is_external() {
			return Err("gate/up intermediate is externally visible");
		}
		if semantic.operations().iter().any(|operation| {
			!partition.operations.contains(&operation.id())
				&& operation
					.inputs()
					.iter()
					.flatten()
					.any(|input| *input == value_id)
		}) {
			return Err("gate/up intermediate has an external consumer");
		}
		let resource = resource_ids[index];
		let description = resource_descriptions[resource];
		if description.is_externally_live() || description.unaccounted_owner_count() != 0 {
			return Err("gate/up intermediate resource is externally live");
		}
	}

	let range = partition_node_range(partition, source);
	if range.count == 0 {
		return Err(range.rejection);
	}
	if range.count == 1 {
		return Err("gate/up source is already one executable dispatch");
	}
	let boundary = [0_usize, 1, 2, 3, 4, 7];
	let buffers = boundary
		.iter()
		.enumerate()
		.map(|(index, value)| BufferUse {
			buffer: resources[resource_ids[*value]].buffer.clone(),
			access: if index < 5 {
				BufferAccess::Read
			} else {
				BufferAccess::Write
			},
		})
		.collect();
	Ok(GateUpCandidate {
		first_node: range.first,
		node_count: range.count,
		node: ComputeNode {
			operation: "oa::Dnn::GateUpSwigluBias",
			kernel: KernelId::MlGateUpSwigluBiasF32,
			buffers,
			push_constants: vec![
				PushConstant::U32(1024),
				PushConstant::U32(64),
				PushConstant::U32(32),
			],
			workgroups: [32, 2, 1],
			semantic_ops: partition.operations.clone(),
			op_contract_hash: 0,
		},
	})
}

fn partition_node_range(partition: &DnnPartition, source: &ExecutableGraph) -> NodeRange {
	let mut seen = vec![false; partition.operations.len()];
	let mut first = INVALID_NODE;
	let mut last = INVALID_NODE;
	for (node_index, node) in source.nodes().iter().enumerate() {
		let mut owns_partition = false;
		let mut owns_outside = false;
		for semantic in &node.semantic_ops {
			if let Some(index) = partition
				.operations
				.iter()
				.position(|operation| operation == semantic)
			{
				seen[index] = true;
				owns_partition = true;
			} else {
				owns_outside = true;
			}
		}
		if !owns_partition {
			continue;
		}
		if owns_outside {
			return rejected("source node has mixed partition and external semantic provenance");
		}
		first = first.min(node_index);
		last = node_index;
	}
	if seen.contains(&false) {
		return rejected("partition operation has no executable source provenance");
	}
	if first == INVALID_NODE || last == INVALID_NODE {
		return rejected("partition has no executable source interval");
	}
	for node in &source.nodes()[first..=last] {
		if node.semantic_ops.is_empty() {
			return rejected("partition source interval contains compatibility-only work");
		}
		if node
			.semantic_ops
			.iter()
			.any(|operation| !partition.operations.contains(operation))
		{
			return rejected("partition source interval contains an external semantic operation");
		}
	}
	NodeRange {
		first,
		count: last - first + 1,
		rejection: "",
	}
}

const fn rejected(reason: &'static str) -> NodeRange {
	NodeRange {
		first: INVALID_NODE,
		count: 0,
		rejection: reason,
	}
}

fn exact_row_major(value: &DnnValueDesc) -> bool {
	let mut stride = 1_usize;
	for (extent, actual) in value.shape.iter().zip(&value.strides).rev() {
		if usize::try_from(*actual) != Ok(stride) {
			return false;
		}
		let Some(next) = stride.checked_mul(*extent) else {
			return false;
		};
		stride = next;
	}
	true
}

fn find_operation(plan: &DnnPlan, id: SemanticOpId) -> Option<&DnnOpDesc> {
	plan.operations
		.iter()
		.find(|operation| operation.source_op == id)
}

fn find_value(plan: &DnnPlan, id: crate::runtime::SemanticValueId) -> Option<&DnnValueDesc> {
	plan.values.iter().find(|value| value.id == id)
}

fn mark_inherited(plan: &mut DnnPlan, partition: usize) {
	plan.partitions[partition].state = DnnPartitionState::Inherited;
	plan.inherited_partition_count = plan.inherited_partition_count.saturating_add(1);
}

fn mark_fallback(plan: &mut DnnPlan, partition: usize, reason: &'static str, unexpected: bool) {
	let partition = &mut plan.partitions[partition];
	partition.state = DnnPartitionState::ExplicitFallback;
	partition.fallback_reason = Some(reason.to_owned());
	plan.fallback_partition_count = plan.fallback_partition_count.saturating_add(1);
	if unexpected {
		plan.unexpected_fallback_count = plan.unexpected_fallback_count.saturating_add(1);
	}
}
