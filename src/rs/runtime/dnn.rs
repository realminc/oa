//! Private semantic DNN compatibility planner.
//!
//! Port provenance: `oa/source/cpp/lib/oa/runtime/dnn.{h,cpp}` and generated
//! `oa/source/cpp/lib/oa/runtime/gen/dnnOpRoles.inc`. This layer recognizes
//! candidate semantic regions and delegates admitted physical replacements to
//! the private donor-backed graph lowerer.

use std::collections::BTreeSet;

use crate::{DType, Error, OpAttribute, OperationContract, Result};

use super::{SemanticGraph, SemanticOpId, SemanticValueId};

mod lowering;

#[allow(dead_code)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum DnnOpType {
	Matmul,
	BiasAdd,
	Relu,
	Gelu,
	Silu,
	Multiply,
	Add,
	RmsNorm,
	ScaledDotProductAttention,
	GroupedGemm,
	GatedMultiply,
	ColorConvert,
	ResizeNormalize,
	ResidualRmsNorm,
	Portable,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DnnEngineType {
	Portable,
	BlasLtEpilogue,
	QkvProjectionGroup,
	GatedFfn,
	ResidualNorm,
	Attention,
	GroupedMoe,
	VisionPreprocess,
}

#[allow(dead_code)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum DnnEpilogue {
	None,
	Bias,
	BiasRelu,
	BiasGelu,
	BiasSilu,
}

#[allow(dead_code)]
#[derive(Clone, Copy)]
pub(super) enum DnnProvider {
	BlasLtEpilogue,
	QkvProjectionGroup,
	GatedFfn,
	ResidualNorm,
	Attention,
	GroupedMoe,
	VisionPreprocess,
}

impl DnnProvider {
	pub(super) const fn bit(self) -> u32 {
		1 << self as u32
	}
}

pub(super) struct DnnOpRole {
	pub(super) contract: OperationContract,
	pub(super) op_type: DnnOpType,
	pub(super) epilogue: DnnEpilogue,
	pub(super) epilogue_required_input: Option<usize>,
	pub(super) provider_mask: u32,
}

include!("dnn/generated.rs");

#[derive(Clone, Copy)]
pub(super) struct DnnPolicy {
	max_workspace_bytes: u64,
	require_deterministic: bool,
	allow_recompute: bool,
}

impl Default for DnnPolicy {
	fn default() -> Self {
		Self {
			max_workspace_bytes: 0,
			require_deterministic: true,
			allow_recompute: true,
		}
	}
}

#[derive(Clone)]
struct DnnValueDesc {
	id: SemanticValueId,
	shape: Vec<usize>,
	strides: Vec<isize>,
	dtype: DType,
	storage_id: SemanticValueId,
	byte_offset: u64,
	external: bool,
	virtual_value: bool,
}

#[derive(Clone)]
struct DnnOpDesc {
	source_op: SemanticOpId,
	op_type: DnnOpType,
	epilogue: DnnEpilogue,
	inputs: Vec<SemanticValueId>,
	outputs: Vec<SemanticValueId>,
	provider_mask: u32,
	training: bool,
}

struct DnnPartition {
	engine: DnnEngineType,
	operations: Vec<SemanticOpId>,
	saved_for_backward: Vec<SemanticValueId>,
	workspace_bytes: u64,
	deterministic: bool,
	state: DnnPartitionState,
	fallback_reason: Option<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DnnPartitionState {
	Analyzed,
	Applied,
	Inherited,
	ExplicitFallback,
}

pub(super) struct DnnPlan {
	values: Vec<DnnValueDesc>,
	operations: Vec<DnnOpDesc>,
	partitions: Vec<DnnPartition>,
	policy: DnnPolicy,
	graph_hash: u64,
	source_operation_count: u32,
	captured_operation_count: u32,
	recognized_partition_count: u32,
	applied_partition_count: u32,
	inherited_partition_count: u32,
	fallback_partition_count: u32,
	unexpected_fallback_count: u32,
}

impl DnnPlan {
	pub(super) fn plan(graph: &SemanticGraph, policy: DnnPolicy) -> Result<Self> {
		graph.validate()?;
		let storage_roots = storage_roots(graph)?;
		let training_operations = graph
			.autograd()
			.iter()
			.map(|entry| entry.forward_op())
			.collect::<BTreeSet<_>>();
		let mut values = Vec::new();
		for value in graph.values() {
			let Some(id) = value.id() else {
				return Err(Error::internal(
					"DNN capture found an unassigned semantic value",
				));
			};
			if value.shape().is_empty() || value.shape().contains(&0) || value.shape().len() > 4 {
				continue;
			}
			values.push(DnnValueDesc {
				id,
				shape: value.shape().to_vec(),
				strides: value.strides_ref().to_vec(),
				dtype: value.dtype(),
				storage_id: storage_roots[id.index() as usize],
				byte_offset: value.byte_offset_value(),
				external: value.is_external() || value.producer().is_none(),
				virtual_value: value.is_virtual(),
			});
		}
		let captured_values = values.iter().map(|value| value.id).collect::<BTreeSet<_>>();
		let mut operations = Vec::new();
		for operation in graph.operations() {
			let inputs = operation
				.inputs()
				.iter()
				.flatten()
				.copied()
				.collect::<Vec<_>>();
			if inputs.is_empty()
				|| operation.outputs().is_empty()
				|| inputs.iter().any(|value| !captured_values.contains(value))
				|| operation
					.outputs()
					.iter()
					.any(|value| !captured_values.contains(value))
			{
				continue;
			}
			let role = find_role(operation.name(), operation.contract_hash());
			let epilogue = role.map_or(DnnEpilogue::None, |role| {
				if role
					.epilogue_required_input
					.is_some_and(|index| operation.inputs().get(index).is_none_or(Option::is_none))
				{
					DnnEpilogue::None
				} else {
					role.epilogue
				}
			});
			operations.push(DnnOpDesc {
				source_op: operation.id(),
				op_type: role.map_or(DnnOpType::Portable, |role| role.op_type),
				epilogue,
				inputs,
				outputs: operation.outputs().to_vec(),
				provider_mask: role.map_or(0, |role| role.provider_mask),
				training: training_operations.contains(&operation.id()),
			});
		}
		let produced_values = operations
			.iter()
			.flat_map(|operation| operation.outputs.iter().copied())
			.collect::<BTreeSet<_>>();
		for value in &mut values {
			// Shape/rank filtering may omit a semantic producer while retaining a
			// downstream DNN-compatible consumer. That value is an input boundary of
			// the captured DNN subgraph even though it is internal to the full graph.
			value.external |= !produced_values.contains(&value.id);
		}
		validate_capture(&values, &operations)?;
		let partitions = partition(&operations, &values, policy);
		let source_operation_count = checked_u32(graph.operations().len(), "DNN source operation")?;
		let captured_operation_count = checked_u32(operations.len(), "DNN captured operation")?;
		let recognized_partition_count = checked_u32(
			partitions
				.iter()
				.filter(|partition| partition.engine != DnnEngineType::Portable)
				.count(),
			"DNN recognized partition",
		)?;
		let graph_hash = semantic_graph_hash(graph, policy);
		let plan = Self {
			values,
			operations,
			partitions,
			policy,
			graph_hash,
			source_operation_count,
			captured_operation_count,
			recognized_partition_count,
			applied_partition_count: 0,
			inherited_partition_count: 0,
			fallback_partition_count: 0,
			unexpected_fallback_count: 0,
		};
		plan.validate_retained()?;
		Ok(plan)
	}

	pub(super) fn lower_captured_plan(
		&mut self,
		semantic: &SemanticGraph,
		bindings: &[super::SemanticStorageBinding],
		resources: &[super::executable_graph::ResourceLifetime],
		resource_descriptions: &[super::CapturedResourceDesc],
		allow_inference_replacement: bool,
		source: &super::executable_graph::ExecutableGraph,
	) -> Result<super::executable_graph::ExecutableGraph> {
		let graph = lowering::lower_captured_plan(
			self,
			semantic,
			bindings,
			resources,
			resource_descriptions,
			allow_inference_replacement,
			source,
		)?;
		self.validate_lowered()?;
		Ok(graph)
	}

	pub(super) const fn graph_hash(&self) -> u64 {
		self.graph_hash
	}

	pub(super) fn partition_count(&self) -> usize {
		self.partitions.len()
	}

	pub(super) const fn recognized_partition_count(&self) -> u32 {
		self.recognized_partition_count
	}

	pub(super) const fn applied_partition_count(&self) -> u32 {
		self.applied_partition_count
	}

	pub(super) const fn inherited_partition_count(&self) -> u32 {
		self.inherited_partition_count
	}

	pub(super) const fn fallback_partition_count(&self) -> u32 {
		self.fallback_partition_count
	}

	pub(super) const fn unexpected_fallback_count(&self) -> u32 {
		self.unexpected_fallback_count
	}

	pub(super) fn fallback_reasons(&self) -> impl Iterator<Item = &str> {
		self.partitions
			.iter()
			.filter_map(|partition| partition.fallback_reason.as_deref())
	}

	pub(super) fn portable_partition_count(&self) -> usize {
		self.partitions
			.iter()
			.filter(|partition| partition.engine == DnnEngineType::Portable)
			.count()
	}

	pub(super) const fn source_operation_count(&self) -> u32 {
		self.source_operation_count
	}

	pub(super) const fn captured_operation_count(&self) -> u32 {
		self.captured_operation_count
	}

	fn validate_retained(&self) -> Result<()> {
		if self.graph_hash == 0
			|| self.source_operation_count < self.captured_operation_count
			|| self.captured_operation_count as usize != self.operations.len()
		{
			return Err(Error::internal("DNN plan counters are inconsistent"));
		}
		let mut value_ids = BTreeSet::new();
		for value in &self.values {
			if !value_ids.insert(value.id)
				|| value.shape.is_empty()
				|| value.shape.len() > 4
				|| value.shape.contains(&0)
				|| value.strides.len() != value.shape.len()
				|| value.strides.iter().any(|stride| *stride <= 0)
				|| value.byte_offset % value.dtype.size_bytes() as u64 != 0
			{
				return Err(Error::internal("DNN plan retained an invalid value"));
			}
			let root = self
				.values
				.iter()
				.find(|candidate| candidate.id == value.storage_id)
				.ok_or_else(|| Error::internal("DNN plan value has no storage root"))?;
			if root.id > value.id || root.storage_id != root.id || root.dtype != value.dtype {
				return Err(Error::internal(
					"DNN plan value has an invalid storage root",
				));
			}
		}
		let mut source_operations = BTreeSet::new();
		for operation in &self.operations {
			if !source_operations.insert(operation.source_op)
				|| operation
					.inputs
					.iter()
					.any(|value| !value_ids.contains(value))
				|| operation
					.outputs
					.iter()
					.any(|value| !value_ids.contains(value))
			{
				return Err(Error::internal("DNN plan retained an invalid operation"));
			}
		}
		let mut partition_operations = Vec::new();
		for partition in &self.partitions {
			if partition.operations.is_empty()
				|| partition
					.saved_for_backward
					.iter()
					.any(|value| !value_ids.contains(value))
				|| (self.policy.require_deterministic && !partition.deterministic)
				|| (self.policy.max_workspace_bytes != 0
					&& partition.workspace_bytes > self.policy.max_workspace_bytes)
			{
				return Err(Error::internal("DNN plan retained an invalid partition"));
			}
			partition_operations.extend(partition.operations.iter().copied());
		}
		if partition_operations
			!= self
				.operations
				.iter()
				.map(|operation| operation.source_op)
				.collect::<Vec<_>>()
		{
			return Err(Error::internal(
				"DNN partitions do not cover captured operations in order",
			));
		}
		Ok(())
	}

	fn validate_lowered(&self) -> Result<()> {
		let mut applied = 0_u32;
		let mut inherited = 0_u32;
		let mut fallback = 0_u32;
		for partition in &self.partitions {
			if partition.engine == DnnEngineType::Portable {
				if partition.state != DnnPartitionState::Analyzed
					|| partition.fallback_reason.is_some()
				{
					return Err(Error::internal(
						"portable DNN partition acquired a provider state",
					));
				}
				continue;
			}
			match partition.state {
				DnnPartitionState::Applied => applied = applied.saturating_add(1),
				DnnPartitionState::Inherited => inherited = inherited.saturating_add(1),
				DnnPartitionState::ExplicitFallback => fallback = fallback.saturating_add(1),
				DnnPartitionState::Analyzed => {
					return Err(Error::internal(
						"recognized DNN partition has no lowering state",
					));
				}
			}
			if (partition.state == DnnPartitionState::ExplicitFallback)
				!= partition.fallback_reason.is_some()
			{
				return Err(Error::internal(
					"DNN fallback state and reason are inconsistent",
				));
			}
		}
		if applied != self.applied_partition_count
			|| inherited != self.inherited_partition_count
			|| fallback != self.fallback_partition_count
			|| applied.saturating_add(inherited).saturating_add(fallback)
				!= self.recognized_partition_count
			|| self.unexpected_fallback_count > fallback
		{
			return Err(Error::internal("DNN lowering counters are inconsistent"));
		}
		Ok(())
	}

	pub(super) fn value_count(&self) -> usize {
		self.values.len()
	}

	pub(super) fn external_value_count(&self) -> usize {
		self.values.iter().filter(|value| value.external).count()
	}

	pub(super) fn virtual_value_count(&self) -> usize {
		self.values
			.iter()
			.filter(|value| value.virtual_value)
			.count()
	}
}

fn find_role(name: &str, contract_hash: u64) -> Option<&'static DnnOpRole> {
	DNN_OP_ROLES
		.iter()
		.find(|role| role.contract.name() == name && role.contract.hash() == contract_hash)
}

fn storage_roots(graph: &SemanticGraph) -> Result<Vec<SemanticValueId>> {
	let mut roots = Vec::with_capacity(graph.values().len());
	for value in graph.values() {
		let id = value
			.id()
			.ok_or_else(|| Error::internal("semantic value has no identity"))?;
		let mut root = id;
		if let Some(source) = value.view_source() {
			root = *roots
				.get(source.index() as usize)
				.ok_or_else(|| Error::internal("DNN view storage root is unavailable"))?;
		} else if let Some(producer) = value.producer() {
			let operation = &graph.operations()[producer.index() as usize];
			if let Some(alias) = operation
				.aliases()
				.iter()
				.find(|alias| alias.output() == id)
			{
				root = *roots
					.get(alias.input().index() as usize)
					.ok_or_else(|| Error::internal("DNN alias storage root is unavailable"))?;
			}
		}
		roots.push(root);
	}
	Ok(roots)
}

fn validate_capture(values: &[DnnValueDesc], operations: &[DnnOpDesc]) -> Result<()> {
	let value_ids = values.iter().map(|value| value.id).collect::<BTreeSet<_>>();
	let mut produced = BTreeSet::new();
	for operation in operations {
		for input in &operation.inputs {
			let value = values
				.iter()
				.find(|value| value.id == *input)
				.ok_or_else(|| Error::invalid_argument("DNN operation has a dangling input"))?;
			if !value.external && !produced.contains(input) {
				return Err(Error::invalid_argument(
					"DNN operation consumes a value before its producer",
				));
			}
		}
		for output in &operation.outputs {
			if !value_ids.contains(output) {
				return Err(Error::invalid_argument(
					"DNN operation has a dangling output",
				));
			}
			if !produced.insert(*output) {
				return Err(Error::invalid_argument(
					"DNN graph violates single-assignment output semantics",
				));
			}
		}
	}
	Ok(())
}

fn partition(
	operations: &[DnnOpDesc],
	values: &[DnnValueDesc],
	policy: DnnPolicy,
) -> Vec<DnnPartition> {
	let mut partitions = Vec::new();
	let mut index = 0;
	while index < operations.len() {
		let operation = &operations[index];
		if let Some(partition) = match_vision_preprocess(operations, index) {
			partitions.push(partition);
			index += 2;
			continue;
		}
		if let Some(partition) = match_qkv(operations, index) {
			partitions.push(partition);
			index += 3;
			continue;
		}
		if let Some(partition) = match_gated_ffn(operations, index) {
			partitions.push(partition);
			index += 4;
			continue;
		}
		if let Some((partition, consumed)) = match_blas_epilogue(operations, index, policy) {
			partitions.push(partition);
			index += consumed;
			continue;
		}
		if operation.op_type == DnnOpType::Matmul
			&& operation.epilogue != DnnEpilogue::None
			&& supports(operation, DnnProvider::BlasLtEpilogue)
		{
			let mut saved = Vec::new();
			if operation.training && !policy.allow_recompute {
				saved.extend(operation.outputs.first().copied());
			}
			partitions.push(DnnPartition {
				engine: DnnEngineType::BlasLtEpilogue,
				operations: vec![operation.source_op],
				saved_for_backward: saved,
				workspace_bytes: 0,
				deterministic: true,
				state: DnnPartitionState::Analyzed,
				fallback_reason: None,
			});
			index += 1;
			continue;
		}
		if let Some(partition) = match_residual_norm(operations, index) {
			partitions.push(partition);
			index += 2;
			continue;
		}
		let mut engine = DnnEngineType::Portable;
		let mut saved = Vec::new();
		match operation.op_type {
			DnnOpType::ScaledDotProductAttention if supports(operation, DnnProvider::Attention) => {
				engine = DnnEngineType::Attention;
				if operation.training {
					saved.extend(operation.inputs.iter().take(3).copied());
					if let Some(saved_state) = operation.outputs.get(1) {
						if values
							.iter()
							.find(|value| value.id == *saved_state)
							.is_some_and(|value| value.shape.len() == 2)
						{
							saved.extend(operation.outputs.first().copied());
						}
						saved.push(*saved_state);
					}
				}
			}
			DnnOpType::GroupedGemm if supports(operation, DnnProvider::GroupedMoe) => {
				engine = DnnEngineType::GroupedMoe;
				if operation.training {
					saved.extend(operation.inputs.first().copied());
				}
			}
			DnnOpType::ResidualRmsNorm if supports(operation, DnnProvider::ResidualNorm) => {
				engine = DnnEngineType::ResidualNorm;
			}
			_ => {}
		}
		partitions.push(DnnPartition {
			engine,
			operations: vec![operation.source_op],
			saved_for_backward: saved,
			workspace_bytes: 0,
			deterministic: true,
			state: DnnPartitionState::Analyzed,
			fallback_reason: None,
		});
		index += 1;
	}
	partitions
}

fn match_vision_preprocess(operations: &[DnnOpDesc], index: usize) -> Option<DnnPartition> {
	let group = operations.get(index..index + 2)?;
	if group[0].op_type != DnnOpType::ColorConvert
		|| group[1].op_type != DnnOpType::ResizeNormalize
		|| !supports(&group[0], DnnProvider::VisionPreprocess)
		|| !supports(&group[1], DnnProvider::VisionPreprocess)
		|| !single_edge(&group[0], &group[1])
	{
		return None;
	}
	Some(DnnPartition {
		engine: DnnEngineType::VisionPreprocess,
		operations: group.iter().map(|operation| operation.source_op).collect(),
		saved_for_backward: Vec::new(),
		workspace_bytes: 0,
		deterministic: true,
		state: DnnPartitionState::Analyzed,
		fallback_reason: None,
	})
}

fn match_qkv(operations: &[DnnOpDesc], index: usize) -> Option<DnnPartition> {
	let group = operations.get(index..index + 3)?;
	let shared = *group[0].inputs.first()?;
	if group.iter().all(|operation| {
		operation.op_type == DnnOpType::Matmul
			&& supports(operation, DnnProvider::QkvProjectionGroup)
			&& operation.inputs.first() == Some(&shared)
	}) {
		let mut saved = Vec::new();
		if group[0].training {
			saved.push(shared);
		}
		return Some(DnnPartition {
			engine: DnnEngineType::QkvProjectionGroup,
			operations: group.iter().map(|operation| operation.source_op).collect(),
			saved_for_backward: saved,
			workspace_bytes: 0,
			deterministic: true,
			state: DnnPartitionState::Analyzed,
			fallback_reason: None,
		});
	}
	None
}

fn match_gated_ffn(operations: &[DnnOpDesc], index: usize) -> Option<DnnPartition> {
	if let Some(group) = operations.get(index..index + 3) {
		let shared = group[0].inputs.first().copied();
		let matches = shared.is_some()
			&& group[0].op_type == DnnOpType::Matmul
			&& group[1].op_type == DnnOpType::Matmul
			&& group[2].op_type == DnnOpType::GatedMultiply
			&& group
				.iter()
				.all(|operation| supports(operation, DnnProvider::GatedFfn))
			&& group[1].inputs.first().copied() == shared
			&& single_edge(&group[0], &group[2])
			&& single_edge(&group[1], &group[2]);
		if matches {
			let mut saved = Vec::new();
			if group[0].training {
				saved.extend(shared);
			}
			return Some(DnnPartition {
				engine: DnnEngineType::GatedFfn,
				operations: group.iter().map(|operation| operation.source_op).collect(),
				saved_for_backward: saved,
				workspace_bytes: 0,
				deterministic: true,
				state: DnnPartitionState::Analyzed,
				fallback_reason: None,
			});
		}
	}
	let group = operations.get(index..index + 4)?;
	let shared = *group[0].inputs.first()?;
	let matches = group[0].op_type == DnnOpType::Matmul
		&& group[1].op_type == DnnOpType::Matmul
		&& group[2].op_type == DnnOpType::Silu
		&& group[3].op_type == DnnOpType::Multiply
		&& group
			.iter()
			.all(|operation| supports(operation, DnnProvider::GatedFfn))
		&& group[1].inputs.first() == Some(&shared)
		&& single_edge(&group[0], &group[2])
		&& single_edge(&group[2], &group[3])
		&& single_edge(&group[1], &group[3]);
	if !matches {
		return None;
	}
	let mut saved = Vec::new();
	if group[0].training {
		saved.push(shared);
	}
	Some(DnnPartition {
		engine: DnnEngineType::GatedFfn,
		operations: group.iter().map(|operation| operation.source_op).collect(),
		saved_for_backward: saved,
		workspace_bytes: 0,
		deterministic: true,
		state: DnnPartitionState::Analyzed,
		fallback_reason: None,
	})
}

fn match_blas_epilogue(
	operations: &[DnnOpDesc],
	index: usize,
	policy: DnnPolicy,
) -> Option<(DnnPartition, usize)> {
	let matmul = operations.get(index)?;
	let bias = operations.get(index + 1)?;
	if matmul.op_type != DnnOpType::Matmul
		|| bias.op_type != DnnOpType::BiasAdd
		|| !supports(matmul, DnnProvider::BlasLtEpilogue)
		|| !supports(bias, DnnProvider::BlasLtEpilogue)
		|| !single_edge(matmul, bias)
	{
		return None;
	}
	let mut group = vec![matmul.source_op, bias.source_op];
	let mut consumed = 2;
	let mut saved = Vec::new();
	if let Some(activation) = operations.get(index + 2)
		&& matches!(
			activation.op_type,
			DnnOpType::Relu | DnnOpType::Gelu | DnnOpType::Silu
		) && supports(activation, DnnProvider::BlasLtEpilogue)
		&& single_edge(bias, activation)
	{
		group.push(activation.source_op);
		consumed = 3;
		if matmul.training && !policy.allow_recompute {
			saved.extend(bias.outputs.first().copied());
		}
	}
	Some((
		DnnPartition {
			engine: DnnEngineType::BlasLtEpilogue,
			operations: group,
			saved_for_backward: saved,
			workspace_bytes: 0,
			deterministic: true,
			state: DnnPartitionState::Analyzed,
			fallback_reason: None,
		},
		consumed,
	))
}

fn match_residual_norm(operations: &[DnnOpDesc], index: usize) -> Option<DnnPartition> {
	let group = operations.get(index..index + 2)?;
	if group[0].op_type != DnnOpType::Add
		|| group[1].op_type != DnnOpType::RmsNorm
		|| !supports(&group[0], DnnProvider::ResidualNorm)
		|| !supports(&group[1], DnnProvider::ResidualNorm)
		|| !single_edge(&group[0], &group[1])
	{
		return None;
	}
	Some(DnnPartition {
		engine: DnnEngineType::ResidualNorm,
		operations: group.iter().map(|operation| operation.source_op).collect(),
		saved_for_backward: Vec::new(),
		workspace_bytes: 0,
		deterministic: true,
		state: DnnPartitionState::Analyzed,
		fallback_reason: None,
	})
}

fn supports(operation: &DnnOpDesc, provider: DnnProvider) -> bool {
	operation.provider_mask & provider.bit() != 0
}

fn single_edge(left: &DnnOpDesc, right: &DnnOpDesc) -> bool {
	left.outputs.len() == 1 && right.inputs.contains(&left.outputs[0])
}

fn semantic_graph_hash(graph: &SemanticGraph, policy: DnnPolicy) -> u64 {
	let mut hash = StableHash::new();
	hash.u64(6);
	hash.u64(policy.max_workspace_bytes);
	hash.bool(policy.require_deterministic);
	hash.bool(policy.allow_recompute);
	hash.usize(graph.values().len());
	hash.usize(graph.operations().len());
	hash.usize(graph.autograd().len());
	for value in graph.values() {
		hash.u32(value.id().map_or(u32::MAX, SemanticValueId::index));
		hash.bytes(value.name().as_bytes());
		hash.u8(value_kind_code(value.kind()));
		hash.shape(value.shape());
		for stride in value.strides_ref() {
			hash.i64(*stride as i64);
		}
		hash.bytes(value.dtype().token().as_bytes());
		hash.u64(value.byte_offset_value());
		hash.bool(value.is_external());
		hash.bool(value.is_virtual());
		hash.u32(value.producer().map_or(u32::MAX, SemanticOpId::index));
		hash.u32(value.view_source().map_or(u32::MAX, SemanticValueId::index));
		hash.i64(value.view_byte_offset());
	}
	for operation in graph.operations() {
		let role = find_role(operation.name(), operation.contract_hash());
		hash.u32(operation.id().index());
		hash.bytes(operation.name().as_bytes());
		hash.u64(operation.contract_hash());
		hash.u32(role.map_or(0, |role| role.provider_mask));
		hash.u8(role.map_or(DnnEpilogue::None, |role| role.epilogue) as u8);
		hash.u8(differentiation_code(operation.differentiation()));
		hash.u8(lowering_code(operation.lowering()));
		hash.u8(control_flow_code(operation.control_flow()));
		hash.u8(operation.optional_input_mask());
		hash.usize(operation.inputs().len());
		for input in operation.inputs() {
			hash.u32(input.map_or(u32::MAX, SemanticValueId::index));
		}
		hash.usize(operation.outputs().len());
		for output in operation.outputs() {
			hash.u32(output.index());
		}
		hash.usize(operation.attributes().len());
		for attribute in operation.attributes() {
			hash_attribute(&mut hash, attribute);
		}
		hash.usize(operation.accesses().len());
		for access in operation.accesses() {
			hash.u32(access.value().index());
			hash.u8(match access.mode() {
				super::SemanticAccessMode::Read => 0,
				super::SemanticAccessMode::Write => 1,
				super::SemanticAccessMode::ReadWrite => 2,
			});
		}
		hash.usize(operation.mutated_inputs().len());
		for input in operation.mutated_inputs() {
			hash.u32(input.index());
		}
		hash.usize(operation.aliases().len());
		for alias in operation.aliases() {
			hash.u32(alias.output().index());
			hash.u32(alias.input().index());
		}
		hash.usize(operation.control_dependencies().len());
		for dependency in operation.control_dependencies() {
			hash.u32(dependency.index());
		}
		hash.u32(
			operation
				.backward_of()
				.map_or(u32::MAX, SemanticOpId::index),
		);
		hash.u64(operation.backward_sequence());
	}
	for entry in graph.autograd() {
		hash.u32(entry.forward_op().index());
		hash.u32(entry.output().index());
		hash.usize(entry.output_index());
		hash.u64(entry.sequence());
		hash.u32(
			entry
				.backward_first_op()
				.map_or(u32::MAX, SemanticOpId::index),
		);
		hash.usize(entry.backward_op_count());
		hash.bool(entry.is_backward_expanded());
	}
	hash.finish()
}

const fn value_kind_code(kind: crate::OpValueKind) -> u8 {
	match kind {
		crate::OpValueKind::Matrix => 1,
		crate::OpValueKind::Image => 2,
		crate::OpValueKind::Audio => 3,
		crate::OpValueKind::VideoFrame => 4,
		crate::OpValueKind::QuantMatrix => 5,
	}
}

const fn differentiation_code(value: crate::OpDifferentiation) -> u8 {
	match value {
		crate::OpDifferentiation::None => 0,
		crate::OpDifferentiation::Reverse => 1,
	}
}

const fn lowering_code(value: crate::OpLowering) -> u8 {
	match value {
		crate::OpLowering::Dispatch => 0,
		crate::OpLowering::Gemm => 1,
	}
}

const fn control_flow_code(value: crate::OpControlFlow) -> u8 {
	match value {
		crate::OpControlFlow::StraightLine => 0,
		crate::OpControlFlow::Conditional => 1,
		crate::OpControlFlow::Loop => 2,
	}
}

fn hash_attribute(hash: &mut StableHash, attribute: &OpAttribute) {
	hash.bytes(attribute.name().as_bytes());
	hash.u8(match attribute.kind() {
		crate::OpAttributeKind::Boolean => 1,
		crate::OpAttributeKind::SignedInteger => 2,
		crate::OpAttributeKind::UnsignedInteger => 3,
		crate::OpAttributeKind::Float => 4,
		crate::OpAttributeKind::String => 5,
		crate::OpAttributeKind::Shape => 6,
		crate::OpAttributeKind::Enum => 7,
	});
	match attribute {
		OpAttribute::Boolean { value, .. } => hash.bool(*value),
		OpAttribute::SignedInteger { value, .. } => hash.i64(*value),
		OpAttribute::UnsignedInteger { value, .. } => hash.u64(*value),
		OpAttribute::Float { value, .. } => hash.u64(value.to_bits()),
		OpAttribute::String { value, .. } | OpAttribute::Enum { value, .. } => {
			hash.bytes(value.as_bytes());
		}
		OpAttribute::Shape { value, .. } => hash.shape(value),
	}
}

fn checked_u32(value: usize, label: &str) -> Result<u32> {
	u32::try_from(value)
		.map_err(|_| Error::resource_exhausted(format!("{label} count exceeds u32")))
}

struct StableHash(u64);

impl StableHash {
	const fn new() -> Self {
		Self(0xcbf2_9ce4_8422_2325)
	}

	fn bytes(&mut self, bytes: &[u8]) {
		self.usize(bytes.len());
		for byte in bytes {
			self.u8(*byte);
		}
	}

	fn shape(&mut self, shape: &[usize]) {
		self.usize(shape.len());
		for extent in shape {
			self.usize(*extent);
		}
	}

	fn bool(&mut self, value: bool) {
		self.u8(value as u8);
	}

	fn usize(&mut self, value: usize) {
		self.u64(value as u64);
	}

	fn i64(&mut self, value: i64) {
		self.u64(value as u64);
	}

	fn u8(&mut self, value: u8) {
		self.0 ^= u64::from(value);
		self.0 = self.0.wrapping_mul(0x100_0000_01b3);
	}

	fn u32(&mut self, value: u32) {
		self.u64(u64::from(value));
	}

	fn u64(&mut self, value: u64) {
		for byte in value.to_le_bytes() {
			self.u8(byte);
		}
	}

	const fn finish(self) -> u64 {
		self.0
	}
}
