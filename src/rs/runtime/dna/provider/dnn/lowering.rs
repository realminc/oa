//! Private DNN semantic-to-executable replacement.
//!
//! Port provenance: `oa/source/cpp/lib/oa/runtime/dnn/graphLowering.cpp`.

use crate::{DType, Result};

use super::{
	DnnEngineType, DnnEpilogue, DnnOpDesc, DnnOpType, DnnPartition, DnnPartitionState, DnnPlan,
	DnnValueDesc,
};
use crate::runtime::{
	BufferAccess, CapturedResourceDesc, EngineHandle, GemmEpilogue, KernelId, MatmulPreference,
	MatmulProblem, PushConstant, SemanticGraph, SemanticOpId, SemanticStorageBinding,
	executable_graph::{BufferUse, ComputeNode, ExecutableGraph, ResourceLifetime},
	lower_mat_mul_nt_plan, plan_mat_mul_nt_problem,
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

struct VisionPreprocessCandidate {
	first_node: usize,
	node_count: usize,
	node: ComputeNode,
}

struct LinearEpilogueCandidate {
	first_node: usize,
	node_count: usize,
	node: ComputeNode,
}

struct LinearEpilogueRejection {
	reason: &'static str,
	unexpected: bool,
}

#[allow(clippy::too_many_arguments)]
pub(super) fn lower_captured_plan(
	engine_handle: &EngineHandle,
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

		if engine == DnnEngineType::BlasLtEpilogue {
			if operation_count == 1 {
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
			if !allow_inference_replacement {
				mark_fallback(
					plan,
					partition_index,
					"training-program capture retains the proven source lowering",
					false,
				);
				continue;
			}
			let candidate = match build_linear_epilogue(
				engine_handle,
				plan,
				partition_index,
				semantic,
				bindings,
				resources,
				resource_descriptions,
				source,
			) {
				Ok(candidate) => candidate,
				Err(rejection) => {
					mark_fallback(
						plan,
						partition_index,
						rejection.reason,
						rejection.unexpected,
					);
					continue;
				}
			};
			if candidate.first_node < source_cursor {
				mark_fallback(
					plan,
					partition_index,
					"linear epilogue partition overlaps an earlier applied source interval",
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
				engine_handle,
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

		if engine == DnnEngineType::VisionPreprocess {
			let candidate = match build_vision_preprocess(
				engine_handle,
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
					"vision preprocess partition overlaps an earlier applied source interval",
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

		let candidate = match build_qkv_projection(
			engine_handle,
			plan,
			partition_index,
			bindings,
			resources,
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
	engine_handle: &EngineHandle,
	plan: &DnnPlan,
	partition_index: usize,
	bindings: &[SemanticStorageBinding],
	resources: &[ResourceLifetime],
	source: &ExecutableGraph,
) -> std::result::Result<QkvProjectionCandidate, &'static str> {
	static DISABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
	if *DISABLED.get_or_init(|| crate::EnvFlag::is_set("OA_DISABLE_DNN_QKV")) {
		return Err("QKV multi-output provider is disabled by environment");
	}
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
			return Err("QKV multi-output provider requires distinct zero-offset row-major F32 values");
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
	if engine_handle
		.require_kernel(KernelId::MlQkvProjectionBiasF32)
		.is_err()
	{
		return Err("QKV projection provider found no admitted device pipeline");
	}
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
			indirect: None,
			semantic_ops: partition.operations.clone(),
			op_contract_hash: 0,
			jit: None,
			fusion_signature: None,
		},
	})
}

#[allow(clippy::too_many_arguments)]
fn build_gate_up(
	engine_handle: &EngineHandle,
	plan: &DnnPlan,
	partition_index: usize,
	semantic: &SemanticGraph,
	bindings: &[SemanticStorageBinding],
	resources: &[ResourceLifetime],
	resource_descriptions: &[CapturedResourceDesc],
	source: &ExecutableGraph,
) -> std::result::Result<GateUpCandidate, &'static str> {
	static DISABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
	if *DISABLED.get_or_init(|| crate::EnvFlag::is_set("OA_DISABLE_DNN_GATE_UP")) {
		return Err("gate/up direct-output provider is disabled by environment");
	}
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
	if engine_handle
		.require_kernel(KernelId::MlGateUpSwigluBiasF32)
		.is_err()
	{
		return Err("gate/up provider found no admitted device pipeline");
	}
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
			indirect: None,
			semantic_ops: partition.operations.clone(),
			op_contract_hash: 0,
			jit: None,
			fusion_signature: None,
		},
	})
}

fn vision_preprocess_disabled() -> bool {
	static DISABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
	*DISABLED.get_or_init(|| crate::EnvFlag::is_set("OA_DISABLE_DNN_VISION_PREPROCESS"))
}

fn enum_attribute<'a>(
	operation: &'a crate::runtime::semantic_graph::SemanticOpDesc,
	name: &str,
) -> Option<&'a str> {
	match operation
		.attributes()
		.iter()
		.find(|attribute| attribute.name() == name)?
	{
		crate::OpAttribute::Enum { value, .. } => Some(value),
		_ => None,
	}
}

fn float_attribute(
	operation: &crate::runtime::semantic_graph::SemanticOpDesc,
	name: &str,
) -> Option<f64> {
	match operation
		.attributes()
		.iter()
		.find(|attribute| attribute.name() == name)?
	{
		crate::OpAttribute::Float { value, .. } => Some(*value),
		_ => None,
	}
}

fn unsigned_attribute(
	operation: &crate::runtime::semantic_graph::SemanticOpDesc,
	name: &str,
) -> Option<u64> {
	match operation
		.attributes()
		.iter()
		.find(|attribute| attribute.name() == name)?
	{
		crate::OpAttribute::UnsignedInteger { value, .. } => Some(*value),
		_ => None,
	}
}

#[allow(clippy::too_many_arguments)]
fn build_vision_preprocess(
	engine_handle: &EngineHandle,
	plan: &DnnPlan,
	partition_index: usize,
	semantic: &SemanticGraph,
	bindings: &[SemanticStorageBinding],
	resources: &[ResourceLifetime],
	resource_descriptions: &[CapturedResourceDesc],
	source: &ExecutableGraph,
) -> std::result::Result<VisionPreprocessCandidate, &'static str> {
	if vision_preprocess_disabled() {
		return Err("vision preprocess provider is disabled by environment");
	}
	let partition = &plan.partitions[partition_index];
	if partition.operations.len() != 2 {
		return Err("vision preprocess provider requires color-convert then resize-normalize");
	}
	let convert = find_operation(plan, partition.operations[0])
		.ok_or("vision preprocess operation is unavailable")?;
	let resize_normalize = find_operation(plan, partition.operations[1])
		.ok_or("vision preprocess operation is unavailable")?;
	if convert.op_type != DnnOpType::ColorConvert
		|| resize_normalize.op_type != DnnOpType::ResizeNormalize
		|| convert.training
		|| resize_normalize.training
		|| convert.inputs.len() != 1
		|| convert.outputs.len() != 1
		|| resize_normalize.inputs.len() != 1
		|| resize_normalize.outputs.len() != 1
		|| convert.outputs[0] != resize_normalize.inputs[0]
	{
		return Err("vision preprocess provider received an incompatible semantic topology");
	}

	let value_ids = [
		convert.inputs[0],
		convert.outputs[0],
		resize_normalize.outputs[0],
	];
	let mut resource_ids = Vec::with_capacity(3);
	for id in value_ids {
		let value = find_value(plan, id).ok_or("vision preprocess value is missing layout metadata")?;
		let binding = bindings
			.iter()
			.find(|binding| binding.value() == id)
			.ok_or("vision preprocess value is missing a storage binding")?;
		let resource = binding.resource() as usize;
		if resource >= resources.len() || resource >= resource_descriptions.len() {
			return Err("vision preprocess resource binding is unavailable");
		}
		let byte_len = value
			.shape
			.iter()
			.try_fold(value.dtype.size_bytes(), |bytes, extent| {
				bytes.checked_mul(*extent)
			})
			.ok_or("vision preprocess physical byte size overflows")?;
		if value.kind != crate::OpValueKind::Image
			|| value.dtype != DType::F32
			|| value.storage_id != value.id
			|| value.byte_offset != 0
			|| !exact_row_major(value)
		{
			return Err("vision preprocess provider requires distinct zero-offset row-major F32 Images");
		}
		if resources[resource].buffer.byte_len() != byte_len
			|| resource_descriptions[resource].resource() as usize != resource
		{
			return Err("vision preprocess semantic and physical layouts disagree");
		}
		if resource_ids.contains(&resource) {
			return Err("vision preprocess provider rejects resource aliases");
		}
		resource_ids.push(resource);
	}

	let source_value =
		find_value(plan, value_ids[0]).ok_or("vision preprocess value is missing layout metadata")?;
	let converted_value =
		find_value(plan, value_ids[1]).ok_or("vision preprocess value is missing layout metadata")?;
	let output_value =
		find_value(plan, value_ids[2]).ok_or("vision preprocess value is missing layout metadata")?;
	let rank = source_value.shape.len();
	if source_value.shape != converted_value.shape
		|| !(3..=4).contains(&rank)
		|| output_value.shape.len() != rank
	{
		return Err("vision preprocess provider requires CHW or NCHW Images");
	}
	let channel_dimension = if rank == 4 { 1 } else { 0 };
	let height_dimension = if rank == 4 { 2 } else { 1 };
	let width_dimension = if rank == 4 { 3 } else { 2 };
	if source_value.shape[channel_dimension] != 3
		|| output_value.shape[channel_dimension] != 3
		|| source_value.shape[height_dimension] == 0
		|| source_value.shape[width_dimension] == 0
		|| output_value.shape[height_dimension] == 0
		|| output_value.shape[width_dimension] == 0
		|| (rank == 4 && source_value.shape[0] != output_value.shape[0])
	{
		return Err("vision preprocess provider requires shape-compatible three-channel Images");
	}

	let semantic_convert = semantic
		.operations()
		.iter()
		.find(|operation| operation.id() == convert.source_op)
		.ok_or("vision preprocess semantic attributes are unavailable")?;
	let semantic_resize = semantic
		.operations()
		.iter()
		.find(|operation| operation.id() == resize_normalize.source_op)
		.ok_or("vision preprocess semantic attributes are unavailable")?;
	let source_format = enum_attribute(semantic_convert, "source_format")
		.ok_or("vision preprocess format/layout contract is unavailable")?;
	let destination_format = enum_attribute(semantic_convert, "destination_format")
		.ok_or("vision preprocess format/layout contract is unavailable")?;
	let layout = enum_attribute(semantic_convert, "layout")
		.ok_or("vision preprocess format/layout contract is unavailable")?;
	if layout != if rank == 4 { "nchw" } else { "chw" } {
		return Err("vision preprocess layout metadata disagrees with shape");
	}
	let identity = source_format == destination_format;
	let swap = (source_format == "rgb" && destination_format == "bgr")
		|| (source_format == "bgr" && destination_format == "rgb");
	if (!identity && !swap) || (identity && source_format != "rgb" && source_format != "bgr") {
		return Err("vision preprocess provider supports RGB/BGR only");
	}

	let mut normalization = [0.0_f32; 6];
	for (index, name) in ["mean0", "mean1", "mean2", "std0", "std1", "std2"]
		.iter()
		.enumerate()
	{
		normalization[index] = float_attribute(semantic_resize, name)
			.ok_or("vision preprocess normalization contract is unavailable")?
			as f32;
	}
	let width = unsigned_attribute(semantic_resize, "width")
		.ok_or("vision preprocess output shape contract is unavailable")?;
	let height = unsigned_attribute(semantic_resize, "height")
		.ok_or("vision preprocess output shape contract is unavailable")?;
	if width != output_value.shape[width_dimension] as u64
		|| height != output_value.shape[height_dimension] as u64
	{
		return Err("vision preprocess output shape contract is unavailable");
	}
	if normalization[3..].contains(&0.0) {
		return Err("vision preprocess normalization standard deviation is zero");
	}

	let intermediate = semantic
		.find_value(value_ids[1])
		.ok_or("vision preprocess intermediate semantic value is unavailable")?;
	if intermediate.is_external() {
		return Err("vision preprocess intermediate is externally visible");
	}
	if semantic.operations().iter().any(|operation| {
		!partition.operations.contains(&operation.id())
			&& operation
				.inputs()
				.iter()
				.flatten()
				.any(|input| *input == value_ids[1])
	}) {
		return Err("vision preprocess intermediate has an external consumer");
	}
	let intermediate_description = resource_descriptions[resource_ids[1]];
	if intermediate_description.is_externally_live()
		|| intermediate_description.unaccounted_owner_count() != 0
	{
		return Err("vision preprocess intermediate resource is externally live");
	}

	let range = partition_node_range(partition, source);
	if range.count == 0 {
		return Err(range.rejection);
	}
	if range.count == 1 {
		return Err("vision preprocess source is already one executable dispatch");
	}
	let batch = u32::try_from(if rank == 4 { source_value.shape[0] } else { 1 })
		.map_err(|_| "vision preprocess batch extent exceeds the push contract")?;
	let input_height = u32::try_from(source_value.shape[height_dimension])
		.map_err(|_| "vision preprocess extent exceeds the push contract")?;
	let input_width = u32::try_from(source_value.shape[width_dimension])
		.map_err(|_| "vision preprocess extent exceeds the push contract")?;
	let output_height = u32::try_from(output_value.shape[height_dimension])
		.map_err(|_| "vision preprocess extent exceeds the push contract")?;
	let output_width = u32::try_from(output_value.shape[width_dimension])
		.map_err(|_| "vision preprocess extent exceeds the push contract")?;
	let buffers = [0_usize, 2]
		.iter()
		.enumerate()
		.map(|(index, value)| BufferUse {
			buffer: resources[resource_ids[*value]].buffer.clone(),
			access: if index == 0 {
				BufferAccess::Read
			} else {
				BufferAccess::Write
			},
		})
		.collect();
	if engine_handle
		.require_kernel(KernelId::ImageResizeNormalizeF32)
		.is_err()
	{
		return Err("vision preprocess provider found no admitted device pipeline");
	}
	Ok(VisionPreprocessCandidate {
		first_node: range.first,
		node_count: range.count,
		node: ComputeNode {
			operation: "oa::Dnn::VisionPreprocess",
			kernel: KernelId::ImageResizeNormalizeF32,
			buffers,
			push_constants: vec![
				PushConstant::U32(batch),
				PushConstant::U32(3),
				PushConstant::U32(input_height),
				PushConstant::U32(input_width),
				PushConstant::U32(output_height),
				PushConstant::U32(output_width),
				PushConstant::F32(normalization[0]),
				PushConstant::F32(normalization[1]),
				PushConstant::F32(normalization[2]),
				PushConstant::F32(normalization[3]),
				PushConstant::F32(normalization[4]),
				PushConstant::F32(normalization[5]),
				PushConstant::U32(u32::from(swap)),
			],
			workgroups: [
				output_width.div_ceil(16),
				output_height.div_ceil(16),
				batch.saturating_mul(3),
			],
			indirect: None,
			semantic_ops: partition.operations.clone(),
			op_contract_hash: 0,
			jit: None,
			fusion_signature: None,
		},
	})
}

fn single_edge(producer: &DnnOpDesc, consumer: &DnnOpDesc) -> bool {
	consumer
		.inputs
		.iter()
		.filter(|input| producer.outputs.contains(input))
		.count()
		== 1
}

#[allow(clippy::too_many_arguments)]
fn build_linear_epilogue(
	engine_handle: &EngineHandle,
	plan: &DnnPlan,
	partition_index: usize,
	semantic: &SemanticGraph,
	bindings: &[SemanticStorageBinding],
	resources: &[ResourceLifetime],
	resource_descriptions: &[CapturedResourceDesc],
	source: &ExecutableGraph,
) -> std::result::Result<LinearEpilogueCandidate, LinearEpilogueRejection> {
	let reject = |reason: &'static str| {
		Err(LinearEpilogueRejection {
			reason,
			unexpected: false,
		})
	};
	let unexpected = |reason: &'static str| {
		Err(LinearEpilogueRejection {
			reason,
			unexpected: true,
		})
	};
	let partition = &plan.partitions[partition_index];
	if partition.operations.len() != 2 && partition.operations.len() != 3 {
		return reject("linear epilogue provider requires matmul+bias with optional activation");
	}
	let missing = || LinearEpilogueRejection {
		reason: "linear epilogue partition is missing captured operations",
		unexpected: false,
	};
	let matmul = find_operation(plan, partition.operations[0]).ok_or_else(missing)?;
	let bias_add = find_operation(plan, partition.operations[1]).ok_or_else(missing)?;
	let activation = if partition.operations.len() == 3 {
		Some(find_operation(plan, partition.operations[2]).ok_or_else(missing)?)
	} else {
		None
	};
	if matmul.op_type != DnnOpType::Matmul
		|| matmul.epilogue != DnnEpilogue::None
		|| bias_add.op_type != DnnOpType::BiasAdd
		|| !single_edge(matmul, bias_add)
	{
		return reject("linear epilogue partition topology is not compositional matmul+bias");
	}
	if matmul.training || bias_add.training || activation.is_some_and(|op| op.training) {
		return reject("training partition requires saved-state and backward planning");
	}
	let mut epilogue = GemmEpilogue::Bias;
	if let Some(activation) = activation {
		if !single_edge(bias_add, activation) {
			return reject("linear epilogue activation does not consume the bias result");
		}
		epilogue = match activation.op_type {
			DnnOpType::Relu => GemmEpilogue::BiasRelu,
			DnnOpType::Gelu => GemmEpilogue::BiasGelu,
			DnnOpType::Silu => GemmEpilogue::BiasSilu,
			_ => {
				return reject("linear epilogue activation has no admitted provider contract");
			}
		};
	}
	if matmul.inputs.len() != 2
		|| matmul.outputs.len() != 1
		|| bias_add.inputs.len() != 2
		|| bias_add.outputs.len() != 1
		|| activation.is_some_and(|op| op.inputs.len() != 1 || op.outputs.len() != 1)
	{
		return reject("linear epilogue operations have unsupported arity");
	}
	let a_id = matmul.inputs[0];
	let b_id = matmul.inputs[1];
	let matmul_out_id = matmul.outputs[0];
	let bias_id = if bias_add.inputs[0] == matmul_out_id {
		bias_add.inputs[1]
	} else {
		bias_add.inputs[0]
	};
	let bias_out_id = bias_add.outputs[0];
	let output_id = activation.map_or(bias_out_id, |op| op.outputs[0]);
	if bias_id == matmul_out_id || activation.is_some_and(|op| op.inputs[0] != bias_out_id) {
		return reject("linear epilogue bias/activation edges are ambiguous");
	}

	let value_ids = [a_id, b_id, matmul_out_id, bias_id, bias_out_id, output_id];
	let mut resource_ids = Vec::with_capacity(6);
	for &id in &value_ids {
		let value = find_value(plan, id).ok_or(LinearEpilogueRejection {
			reason: "linear epilogue value is missing layout metadata",
			unexpected: false,
		})?;
		let binding = bindings
			.iter()
			.find(|binding| binding.value() == id)
			.ok_or(LinearEpilogueRejection {
				reason: "linear epilogue value is missing a storage binding",
				unexpected: false,
			})?;
		let resource = binding.resource() as usize;
		if resource >= resources.len() || resource >= resource_descriptions.len() {
			return reject("linear epilogue resource binding is unavailable");
		}
		let byte_len = value
			.shape
			.iter()
			.try_fold(value.dtype.size_bytes(), |bytes, extent| {
				bytes.checked_mul(*extent)
			})
			.ok_or(LinearEpilogueRejection {
				reason: "linear epilogue physical byte size overflows",
				unexpected: false,
			})?;
		if value.kind != crate::OpValueKind::Matrix
			|| value.dtype != DType::F32
			|| value.storage_id != value.id
			|| value.byte_offset != 0
			|| !exact_row_major(value)
		{
			return reject(
				"linear epilogue provider requires zero-offset non-aliased row-major F32 values",
			);
		}
		if resources[resource].buffer.byte_len() != byte_len
			|| resource_descriptions[resource].resource() as usize != resource
		{
			return reject("linear epilogue semantic and physical layouts disagree");
		}
		resource_ids.push(resource);
	}
	for (index, &resource) in resource_ids.iter().enumerate() {
		if resource_ids[..index]
			.iter()
			.zip(&value_ids[..index])
			.any(|(previous, &id)| *previous == resource && id != value_ids[index])
		{
			return reject("linear epilogue provider rejects physical resource aliases");
		}
	}
	let a = find_value(plan, a_id).expect("validated above");
	let b = find_value(plan, b_id).expect("validated above");
	let bias = find_value(plan, bias_id).expect("validated above");
	let matmul_out = find_value(plan, matmul_out_id).expect("validated above");
	let bias_out = find_value(plan, bias_out_id).expect("validated above");
	let output = find_value(plan, output_id).expect("validated above");
	if a.shape.len() != 2
		|| b.shape.len() != 2
		|| bias.shape.len() != 1
		|| matmul_out.shape.len() != 2
		|| bias_out.shape != matmul_out.shape
		|| output.shape != matmul_out.shape
		|| a.shape[0] == 0
		|| a.shape[1] == 0
		|| b.shape[0] == 0
		|| b.shape[1] != a.shape[1]
		|| bias.shape[0] != b.shape[0]
		|| matmul_out.shape[0] != a.shape[0]
		|| matmul_out.shape[1] != b.shape[0]
	{
		return reject("linear epilogue provider received incompatible rank-2 shapes");
	}

	let both_intermediates = [matmul_out_id, bias_out_id];
	let single_intermediate = [matmul_out_id];
	let intermediate_ids = if activation.is_some() {
		&both_intermediates[..]
	} else {
		&single_intermediate[..]
	};
	for &id in intermediate_ids {
		let value = semantic.find_value(id).ok_or(LinearEpilogueRejection {
			reason: "linear epilogue intermediate semantic value is unavailable",
			unexpected: false,
		})?;
		if value.is_external() {
			return reject("linear epilogue intermediate is externally visible");
		}
		if semantic.operations().iter().any(|operation| {
			!partition.operations.contains(&operation.id())
				&& operation
					.inputs()
					.iter()
					.flatten()
					.any(|input| *input == id)
		}) {
			return reject("linear epilogue intermediate has an external consumer");
		}
		let binding = bindings
			.iter()
			.find(|binding| binding.value() == id)
			.expect("validated above");
		let description = &resource_descriptions[binding.resource() as usize];
		if description.is_externally_live() || description.unaccounted_owner_count() != 0 {
			return reject("linear epilogue intermediate resource is externally live");
		}
	}

	let range = partition_node_range(partition, source);
	if range.count == 0 {
		return reject(range.rejection);
	}
	if range.count == 1 {
		return reject("linear epilogue source is already one executable dispatch");
	}

	let dims = u32::try_from(a.shape[0])
		.and_then(|m| u32::try_from(b.shape[0]).map(|n| (m, n)))
		.and_then(|(m, n)| u32::try_from(a.shape[1]).map(|k| (m, n, k)))
		.map_err(|_| LinearEpilogueRejection {
			reason: "linear epilogue extents exceed the push contract",
			unexpected: false,
		})?;
	let mut problem = MatmulProblem::contiguous(dims.0, dims.1, dims.2);
	problem.epilogue = epilogue;
	problem.training = false;
	let preference = MatmulPreference {
		max_workspace_bytes: plan.policy.max_workspace_bytes,
		..MatmulPreference::default()
	};
	let matmul_plan =
		plan_mat_mul_nt_problem(engine_handle, &problem, &preference).map_err(|_| {
			LinearEpilogueRejection {
				reason: "linear epilogue provider found no legal device plan",
				unexpected: false,
			}
		})?;
	let launch =
		lower_mat_mul_nt_plan(&matmul_plan, &problem).map_err(|_| LinearEpilogueRejection {
			reason: "linear epilogue provider produced an invalid device plan",
			unexpected: true,
		})?;
	if launch.buffer_count != 4 || launch.buffer_order[..4] != [0, 1, 4, 2] {
		return unexpected("linear epilogue provider returned an unsupported buffer permutation");
	}
	if engine_handle.require_kernel(launch.kernel_id).is_err() {
		return reject("linear epilogue provider found no admitted device pipeline");
	}

	let problem_buffers = [
		&resources[resource_ids[0]].buffer,
		&resources[resource_ids[1]].buffer,
		&resources[resource_ids[5]].buffer,
		&resources[resource_ids[5]].buffer,
		&resources[resource_ids[3]].buffer,
	];
	let mut buffers = Vec::with_capacity(launch.buffer_count);
	for slot in 0..launch.buffer_count {
		let index = usize::from(launch.buffer_order[slot]);
		buffers.push(BufferUse {
			buffer: problem_buffers[index].clone(),
			access: launch.buffer_accesses[slot],
		});
	}
	Ok(LinearEpilogueCandidate {
		first_node: range.first,
		node_count: range.count,
		node: ComputeNode {
			operation: "oa::Dnn::LinearEpilogue",
			kernel: launch.kernel_id,
			buffers,
			push_constants: launch.push_constants[..launch.push_count].to_vec(),
			workgroups: launch.grid,
			indirect: None,
			semantic_ops: partition.operations.clone(),
			op_contract_hash: 0,
			jit: None,
			fusion_signature: None,
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
	plan
		.operations
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
