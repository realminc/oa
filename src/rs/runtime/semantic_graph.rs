//! Canonical backend-independent semantic operation graph.
//!
//! Port provenance: `oa/source/cpp/include/oa/runtime/semanticGraph.h` and
//! `oa/source/cpp/lib/oa/runtime/semanticGraph.cpp`. Rust `Option` and typed IDs
//! replace the C++ invalid-ID sentinels; graph meaning and validation are kept.

use std::collections::BTreeSet;

use crate::{
	DType, Error, OpAttribute, OpDifferentiation, OpEffect, OpLowering, OpValueKind,
	OperationContract, Result,
};

/// Canonical graph-local identity of one semantic value.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SemanticValueId(u32);

impl SemanticValueId {
	/// Return the canonical zero-based graph index.
	pub const fn index(self) -> u32 {
		self.0
	}

	fn usize(self) -> usize {
		self.0 as usize
	}
}

/// Canonical graph-local identity of one semantic operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SemanticOpId(u32);

impl SemanticOpId {
	pub(crate) const fn from_index(index: u32) -> Self {
		Self(index)
	}

	/// Return the canonical zero-based graph index.
	pub const fn index(self) -> u32 {
		self.0
	}

	fn usize(self) -> usize {
		self.0 as usize
	}
}

/// Deterministic many-to-many provenance summary for semantic lowering.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SemanticLoweringAnalysis {
	executable_node_counts: Vec<u32>,
	schema_owned_node_count: u32,
	compatibility_node_count: u32,
	direct_op_count: u32,
	decomposed_op_count: u32,
	fused_op_count: u32,
	fused_node_count: u32,
	maximum_nodes_per_op: u32,
	maximum_ops_per_node: u32,
}

impl SemanticLoweringAnalysis {
	pub(crate) fn empty(operation_count: usize) -> Self {
		Self {
			executable_node_counts: vec![0; operation_count],
			schema_owned_node_count: 0,
			compatibility_node_count: 0,
			direct_op_count: 0,
			decomposed_op_count: 0,
			fused_op_count: 0,
			fused_node_count: 0,
			maximum_nodes_per_op: 0,
			maximum_ops_per_node: 0,
		}
	}

	/// Return how many executable nodes lower one semantic operation.
	pub fn executable_node_count(&self, operation: SemanticOpId) -> u32 {
		self
			.executable_node_counts
			.get(operation.usize())
			.copied()
			.unwrap_or(0)
	}

	/// Return the number of executable nodes with semantic ownership.
	pub const fn schema_owned_node_count(&self) -> u32 {
		self.schema_owned_node_count
	}

	/// Return executable nodes still using the compatibility route.
	pub const fn compatibility_node_count(&self) -> u32 {
		self.compatibility_node_count
	}

	/// Return semantic operations lowered directly to one unfused node.
	pub const fn direct_op_count(&self) -> u32 {
		self.direct_op_count
	}

	/// Return semantic operations decomposed into multiple nodes.
	pub const fn decomposed_op_count(&self) -> u32 {
		self.decomposed_op_count
	}

	/// Return semantic operations participating in fused nodes.
	pub const fn fused_op_count(&self) -> u32 {
		self.fused_op_count
	}

	/// Return executable nodes owning multiple semantic operations.
	pub const fn fused_node_count(&self) -> u32 {
		self.fused_node_count
	}

	/// Return the widest decomposition of one semantic operation.
	pub const fn maximum_nodes_per_op(&self) -> u32 {
		self.maximum_nodes_per_op
	}

	/// Return the widest semantic fusion represented by one executable node.
	pub const fn maximum_ops_per_node(&self) -> u32 {
		self.maximum_ops_per_node
	}

	pub(crate) fn node_counts_mut(&mut self) -> &mut [u32] {
		&mut self.executable_node_counts
	}

	pub(crate) fn note_compatibility_node(&mut self) {
		self.compatibility_node_count = self.compatibility_node_count.saturating_add(1);
	}

	pub(crate) fn note_schema_node(&mut self, owner_count: usize) -> Result<()> {
		let owner_count = u32::try_from(owner_count)
			.map_err(|_| Error::resource_exhausted("semantic node owner count exceeds u32"))?;
		self.schema_owned_node_count = self.schema_owned_node_count.saturating_add(1);
		self.maximum_ops_per_node = self.maximum_ops_per_node.max(owner_count);
		if owner_count > 1 {
			self.fused_node_count = self.fused_node_count.saturating_add(1);
		}
		Ok(())
	}

	pub(crate) fn finish(&mut self, fusion_membership: &[bool]) -> Result<()> {
		for (index, count) in self.executable_node_counts.iter().copied().enumerate() {
			if count == 0 {
				return Err(Error::failed_precondition(format!(
					"semantic operation {index} has no executable lowering"
				)));
			}
			if count == 1 && !fusion_membership[index] {
				self.direct_op_count = self.direct_op_count.saturating_add(1);
			}
			if count > 1 {
				self.decomposed_op_count = self.decomposed_op_count.saturating_add(1);
			}
			if fusion_membership[index] {
				self.fused_op_count = self.fused_op_count.saturating_add(1);
			}
			self.maximum_nodes_per_op = self.maximum_nodes_per_op.max(count);
		}
		Ok(())
	}
}

/// Logical access mode derived from an operation contract.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SemanticAccessMode {
	/// Read without semantic mutation.
	Read,
	/// Write without reading the prior value.
	Write,
	/// Read and semantically mutate the same value.
	ReadWrite,
}

impl SemanticAccessMode {
	/// Return the stable semantic-report token.
	pub const fn token(self) -> &'static str {
		match self {
			Self::Read => "read",
			Self::Write => "write",
			Self::ReadWrite => "read_write",
		}
	}
}

/// Handle-free description of one logical value.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SemanticValueDesc {
	id: Option<SemanticValueId>,
	name: String,
	kind: OpValueKind,
	shape: Vec<usize>,
	strides: Vec<isize>,
	dtype: DType,
	byte_offset: u64,
	external: bool,
	virtual_value: bool,
	producer: Option<SemanticOpId>,
	view_source: Option<SemanticValueId>,
	view_byte_offset: i64,
}

impl SemanticValueDesc {
	/// Construct a dense handle-free value description.
	pub fn new(
		name: impl Into<String>,
		kind: OpValueKind,
		shape: impl Into<Vec<usize>>,
		dtype: DType,
	) -> Result<Self> {
		let shape = shape.into();
		let strides = dense_strides(&shape)?;
		Ok(Self {
			id: None,
			name: name.into(),
			kind,
			shape,
			strides,
			dtype,
			byte_offset: 0,
			external: false,
			virtual_value: false,
			producer: None,
			view_source: None,
			view_byte_offset: 0,
		})
	}

	/// Mark whether storage is supplied from outside the captured graph.
	pub const fn external(mut self, external: bool) -> Self {
		self.external = external;
		self
	}

	/// Mark whether the value should be materialized only during lowering.
	pub const fn virtual_value(mut self, virtual_value: bool) -> Self {
		self.virtual_value = virtual_value;
		self
	}

	/// Set the captured absolute byte offset within a storage identity.
	pub const fn byte_offset(mut self, byte_offset: u64) -> Self {
		self.byte_offset = byte_offset;
		self
	}

	/// Replace dense strides with explicit element strides.
	pub fn strides(mut self, strides: impl Into<Vec<isize>>) -> Result<Self> {
		let strides = strides.into();
		if strides.len() != self.shape.len() || strides.iter().any(|stride| *stride < 0) {
			return Err(Error::invalid_argument(
				"semantic value strides must be non-negative and match shape rank",
			));
		}
		self.strides = strides;
		Ok(self)
	}

	/// Return the graph-assigned identity, if this value has been admitted.
	pub const fn id(&self) -> Option<SemanticValueId> {
		self.id
	}

	/// Return the diagnostic name.
	pub fn name(&self) -> &str {
		&self.name
	}

	/// Return the semantic value kind.
	pub const fn kind(&self) -> OpValueKind {
		self.kind
	}

	/// Return the logical shape.
	pub fn shape(&self) -> &[usize] {
		&self.shape
	}

	/// Return element strides.
	pub fn strides_ref(&self) -> &[isize] {
		&self.strides
	}

	/// Return the scalar dtype.
	pub const fn dtype(&self) -> DType {
		self.dtype
	}

	/// Return the absolute byte offset within captured storage.
	pub const fn byte_offset_value(&self) -> u64 {
		self.byte_offset
	}

	/// Return whether storage enters from outside this graph.
	pub const fn is_external(&self) -> bool {
		self.external
	}

	/// Return whether lowering owns materialization.
	pub const fn is_virtual(&self) -> bool {
		self.virtual_value
	}

	/// Return the producing semantic operation, if any.
	pub const fn producer(&self) -> Option<SemanticOpId> {
		self.producer
	}

	/// Return the earlier value from which this metadata-only view derives.
	pub const fn view_source(&self) -> Option<SemanticValueId> {
		self.view_source
	}

	/// Return the signed byte delta from the view source.
	pub const fn view_byte_offset(&self) -> i64 {
		self.view_byte_offset
	}
}

/// One merged semantic value access made by an operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SemanticValueAccess {
	value: SemanticValueId,
	mode: SemanticAccessMode,
}

impl SemanticValueAccess {
	/// Return the accessed value.
	pub const fn value(self) -> SemanticValueId {
		self.value
	}

	/// Return the merged access mode.
	pub const fn mode(self) -> SemanticAccessMode {
		self.mode
	}
}

/// Proven storage alias from one fresh semantic output version to its input.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SemanticAliasDesc {
	output: SemanticValueId,
	input: SemanticValueId,
}

impl SemanticAliasDesc {
	/// Return the aliasing output.
	pub const fn output(self) -> SemanticValueId {
		self.output
	}

	/// Return the aliased input.
	pub const fn input(self) -> SemanticValueId {
		self.input
	}
}

/// Provenance between one forward output and its autograd tape expansion.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SemanticAutogradDesc {
	forward_op: SemanticOpId,
	output: SemanticValueId,
	output_index: usize,
	sequence: u64,
	backward_first_op: Option<SemanticOpId>,
	backward_op_count: usize,
	backward_expanded: bool,
}

impl SemanticAutogradDesc {
	/// Return the forward operation.
	pub const fn forward_op(self) -> SemanticOpId {
		self.forward_op
	}

	/// Return the attached forward output.
	pub const fn output(self) -> SemanticValueId {
		self.output
	}

	/// Return the output's index within the forward contract.
	pub const fn output_index(self) -> usize {
		self.output_index
	}

	/// Return the nonzero tape sequence.
	pub const fn sequence(self) -> u64 {
		self.sequence
	}

	/// Return the first operation in the expanded backward range.
	pub const fn backward_first_op(self) -> Option<SemanticOpId> {
		self.backward_first_op
	}

	/// Return the number of operations in the expanded backward range.
	pub const fn backward_op_count(self) -> usize {
		self.backward_op_count
	}

	/// Return whether the backward range has been attached.
	pub const fn is_backward_expanded(self) -> bool {
		self.backward_expanded
	}
}

/// One mathematical or domain operation before executable lowering.
#[derive(Clone, Debug, PartialEq)]
pub struct SemanticOpDesc {
	id: SemanticOpId,
	name: &'static str,
	contract_hash: u64,
	differentiation: OpDifferentiation,
	lowering: OpLowering,
	control_flow: crate::OpControlFlow,
	optional_input_mask: u8,
	inputs: Vec<Option<SemanticValueId>>,
	outputs: Vec<SemanticValueId>,
	attributes: Vec<OpAttribute>,
	accesses: Vec<SemanticValueAccess>,
	mutated_inputs: Vec<SemanticValueId>,
	aliases: Vec<SemanticAliasDesc>,
	control_dependencies: Vec<SemanticOpId>,
	backward_of: Option<SemanticOpId>,
	backward_sequence: u64,
}

impl SemanticOpDesc {
	/// Return the graph-local operation identity.
	pub const fn id(&self) -> SemanticOpId {
		self.id
	}

	/// Return the stable qualified operation name.
	pub const fn name(&self) -> &'static str {
		self.name
	}

	/// Return the schema-derived operation contract hash.
	pub const fn contract_hash(&self) -> u64 {
		self.contract_hash
	}

	/// Return the differentiation behavior.
	pub const fn differentiation(&self) -> OpDifferentiation {
		self.differentiation
	}

	/// Return the requested lowering family.
	pub const fn lowering(&self) -> OpLowering {
		self.lowering
	}

	/// Return the semantic control-flow class.
	pub const fn control_flow(&self) -> crate::OpControlFlow {
		self.control_flow
	}

	/// Return logical input identities; `None` preserves an absent optional slot.
	pub fn inputs(&self) -> &[Option<SemanticValueId>] {
		&self.inputs
	}

	/// Return logical output identities.
	pub fn outputs(&self) -> &[SemanticValueId] {
		&self.outputs
	}

	/// Return ordered non-value semantic inputs.
	pub fn attributes(&self) -> &[OpAttribute] {
		&self.attributes
	}

	/// Return merged logical accesses.
	pub fn accesses(&self) -> &[SemanticValueAccess] {
		&self.accesses
	}

	/// Return semantically mutated inputs.
	pub fn mutated_inputs(&self) -> &[SemanticValueId] {
		&self.mutated_inputs
	}

	/// Return output-to-input aliases.
	pub fn aliases(&self) -> &[SemanticAliasDesc] {
		&self.aliases
	}

	/// Return explicit control dependencies.
	pub fn control_dependencies(&self) -> &[SemanticOpId] {
		&self.control_dependencies
	}

	/// Return the fixed-input optionality mask preserved from the schema.
	pub const fn optional_input_mask(&self) -> u8 {
		self.optional_input_mask
	}

	/// Return the forward operation owning this backward operation.
	pub const fn backward_of(&self) -> Option<SemanticOpId> {
		self.backward_of
	}

	/// Return the owning autograd tape sequence.
	pub const fn backward_sequence(&self) -> u64 {
		self.backward_sequence
	}
}

/// Canonical semantic graph, deliberately independent of Vulkan objects.
#[derive(Clone, Debug)]
pub struct SemanticGraph {
	values: Vec<SemanticValueDesc>,
	operations: Vec<SemanticOpDesc>,
	autograd: Vec<SemanticAutogradDesc>,
	generation: u64,
}

impl Default for SemanticGraph {
	fn default() -> Self {
		Self::new()
	}
}

impl SemanticGraph {
	/// Construct an empty first-generation graph.
	pub const fn new() -> Self {
		Self {
			values: Vec::new(),
			operations: Vec::new(),
			autograd: Vec::new(),
			generation: 1,
		}
	}

	pub(crate) fn take_recording(&mut self) -> Result<Self> {
		let generation = self
			.generation
			.checked_add(1)
			.ok_or_else(|| Error::resource_exhausted("semantic graph generation exhausted"))?;
		Ok(std::mem::replace(
			self,
			Self {
				values: Vec::new(),
				operations: Vec::new(),
				autograd: Vec::new(),
				generation,
			},
		))
	}

	/// Add a handle-free value and assign its canonical identity.
	pub fn add_value(&mut self, mut value: SemanticValueDesc) -> Result<SemanticValueId> {
		if value.id.is_some()
			|| value.producer.is_some()
			|| value.view_source.is_some()
			|| value.view_byte_offset != 0
		{
			return Err(Error::invalid_argument(
				"semantic value identity and provenance are assigned by the graph",
			));
		}
		let id = SemanticValueId(next_u32(self.values.len(), "semantic value")?);
		value.id = Some(id);
		self.values.push(value);
		Ok(id)
	}

	/// Record a metadata-only value derived from an earlier value.
	pub fn add_view(
		&mut self,
		source: SemanticValueId,
		view: SemanticValueId,
		byte_offset: i64,
	) -> Result<()> {
		if source.usize() >= self.values.len() || view.usize() >= self.values.len() {
			return Err(Error::out_of_range(
				"semantic view references an unknown value",
			));
		}
		if source >= view {
			return Err(Error::invalid_argument(
				"semantic view source must precede its derived value",
			));
		}
		let source_value = &self.values[source.usize()];
		let source_kind = source_value.kind;
		let source_dtype = source_value.dtype;
		let source_offset = source_value.byte_offset;
		let view_value = &self.values[view.usize()];
		if view_value.producer.is_some() {
			return Err(Error::invalid_argument(
				"semantic view cannot also have an operation producer",
			));
		}
		if view_value.view_source.is_some() {
			return Err(Error::already_exists(
				"semantic value already has a view source",
			));
		}
		if view_value.kind != source_kind || view_value.dtype != source_dtype {
			return Err(Error::invalid_argument(
				"semantic view must preserve value kind and dtype",
			));
		}
		if view_value.external {
			return Err(Error::invalid_argument(
				"semantic view cannot be an external value",
			));
		}
		let source_offset = i64::try_from(source_offset)
			.map_err(|_| Error::out_of_range("semantic view source byte offset exceeds i64"))?;
		let absolute_offset = source_offset
			.checked_add(byte_offset)
			.filter(|offset| *offset >= 0)
			.ok_or_else(|| Error::out_of_range("semantic view byte offset is invalid"))?;
		let view_value = &mut self.values[view.usize()];
		view_value.view_source = Some(source);
		view_value.view_byte_offset = byte_offset;
		view_value.byte_offset = absolute_offset as u64;
		Ok(())
	}

	/// Add one schema-owned semantic operation transactionally.
	pub fn add_operation(
		&mut self,
		contract: OperationContract,
		inputs: &[Option<SemanticValueId>],
		outputs: &[SemanticValueId],
		control_dependencies: &[SemanticOpId],
		attributes: &[OpAttribute],
	) -> Result<SemanticOpId> {
		if !contract.validate(attributes) {
			return Err(Error::invalid_argument(
				"semantic operation has an invalid schema contract or attribute signature",
			));
		}
		if !contract.accepts_input_count(inputs.len())
			|| !contract.accepts_output_count(outputs.len())
			|| !contract.variadic_alias_counts_match(inputs.len(), outputs.len())
		{
			return Err(Error::invalid_argument(
				"semantic operation arity does not match its contract",
			));
		}
		for (index, input) in inputs.iter().enumerate() {
			let Some(input) = input else {
				if contract.input_is_optional(index) {
					continue;
				}
				return Err(Error::invalid_argument(
					"semantic operation omits a required input",
				));
			};
			let value = self
				.values
				.get(input.usize())
				.ok_or_else(|| Error::out_of_range("semantic operation has an unknown input"))?;
			if Some(value.kind) != contract.input_kind(index) {
				return Err(Error::invalid_argument(
					"semantic operation input kind does not match its contract",
				));
			}
		}
		for (index, output) in outputs.iter().copied().enumerate() {
			let value = self
				.values
				.get(output.usize())
				.ok_or_else(|| Error::out_of_range("semantic operation has an unknown output"))?;
			if Some(value.kind) != contract.output_kind(index) {
				return Err(Error::invalid_argument(
					"semantic operation output kind does not match its contract",
				));
			}
			if value.producer.is_some() {
				return Err(Error::already_exists(
					"semantic operation output already has a producer",
				));
			}
			if inputs.contains(&Some(output)) {
				return Err(Error::invalid_argument(
					"semantic operation outputs must be fresh SSA values",
				));
			}
			if let Some(alias_input) = contract.alias_input(index) {
				if alias_input >= inputs.len() {
					return Err(Error::out_of_range(
						"semantic output alias references an unknown input",
					));
				}
				if contract.input_is_optional(alias_input) {
					return Err(Error::invalid_argument(
						"semantic output cannot alias an optional input",
					));
				}
			}
		}
		let mut seen_dependencies = BTreeSet::new();
		for dependency in control_dependencies {
			if dependency.usize() >= self.operations.len() {
				return Err(Error::out_of_range(
					"semantic control dependency must reference an earlier operation",
				));
			}
			if !seen_dependencies.insert(*dependency) {
				return Err(Error::already_exists(
					"semantic control dependency is duplicated",
				));
			}
		}

		let id = SemanticOpId(next_u32(self.operations.len(), "semantic operation")?);
		let mut operation = SemanticOpDesc {
			id,
			name: contract.name(),
			contract_hash: contract.hash(),
			differentiation: contract.differentiation(),
			lowering: contract.lowering(),
			control_flow: contract.control_flow(),
			optional_input_mask: contract.optional_input_mask(),
			inputs: inputs.to_vec(),
			outputs: outputs.to_vec(),
			attributes: attributes.to_vec(),
			accesses: Vec::new(),
			mutated_inputs: Vec::new(),
			aliases: Vec::new(),
			control_dependencies: control_dependencies.to_vec(),
			backward_of: None,
			backward_sequence: 0,
		};
		if contract.effects_value().contains(OpEffect::READ_INPUTS) {
			for (index, input) in inputs.iter().copied().enumerate() {
				if let Some(input) = input {
					merge_access(
						&mut operation.accesses,
						input,
						if contract.mutates_input(index) {
							SemanticAccessMode::ReadWrite
						} else {
							SemanticAccessMode::Read
						},
					);
				}
			}
		}
		if contract.effects_value().contains(OpEffect::WRITE_OUTPUTS) {
			for output in outputs {
				merge_access(&mut operation.accesses, *output, SemanticAccessMode::Write);
			}
		}
		for (index, input) in inputs.iter().copied().enumerate() {
			if !contract.mutates_input(index) {
				continue;
			}
			let input = input.ok_or_else(|| {
				Error::invalid_argument("semantic operation cannot mutate an absent input")
			})?;
			operation.mutated_inputs.push(input);
			if !contract.effects_value().contains(OpEffect::READ_INPUTS) {
				merge_access(&mut operation.accesses, input, SemanticAccessMode::Write);
			}
		}
		for (index, output) in outputs.iter().copied().enumerate() {
			if let Some(alias_input) = contract.alias_input(index) {
				let input = inputs[alias_input]
					.ok_or_else(|| Error::invalid_argument("semantic output cannot alias an absent input"))?;
				operation.aliases.push(SemanticAliasDesc { output, input });
			}
		}

		for output in outputs {
			self.values[output.usize()].producer = Some(id);
		}
		self.operations.push(operation);
		Ok(id)
	}

	/// Attach a reverse-differentiable forward output to a tape sequence.
	pub fn attach_autograd(
		&mut self,
		forward_op: SemanticOpId,
		output_index: usize,
		sequence: u64,
	) -> Result<()> {
		let operation = self
			.operations
			.get(forward_op.usize())
			.ok_or_else(|| Error::out_of_range("semantic autograd references an unknown operation"))?;
		if operation.differentiation != OpDifferentiation::Reverse {
			return Err(Error::invalid_argument(
				"semantic autograd requires a reverse-mode operation contract",
			));
		}
		let output = *operation
			.outputs
			.get(output_index)
			.ok_or_else(|| Error::out_of_range("semantic autograd references an unknown output"))?;
		if sequence == 0 {
			return Err(Error::invalid_argument(
				"semantic autograd requires a nonzero tape sequence",
			));
		}
		if self
			.autograd
			.iter()
			.any(|entry| entry.output == output || entry.sequence == sequence)
		{
			return Err(Error::already_exists(
				"semantic output or tape sequence is already attached",
			));
		}
		self.autograd.push(SemanticAutogradDesc {
			forward_op,
			output,
			output_index,
			sequence,
			backward_first_op: None,
			backward_op_count: 0,
			backward_expanded: false,
		});
		Ok(())
	}

	/// Complete one tape attachment with its contiguous backward operation range.
	pub fn complete_autograd(
		&mut self,
		forward_op: SemanticOpId,
		sequence: u64,
		backward_first_op: SemanticOpId,
		backward_op_count: usize,
	) -> Result<()> {
		let attachment_index = self
			.autograd
			.iter()
			.position(|entry| entry.forward_op == forward_op && entry.sequence == sequence)
			.ok_or_else(|| Error::not_found("semantic backward expansion has no tape attachment"))?;
		if self.autograd[attachment_index].backward_expanded {
			return Err(Error::already_exists(
				"semantic backward expansion is already complete",
			));
		}
		let first = backward_first_op.usize();
		let end = first
			.checked_add(backward_op_count)
			.ok_or_else(|| Error::out_of_range("semantic backward operation range overflows"))?;
		if end > self.operations.len() {
			return Err(Error::out_of_range(
				"semantic backward operation range is outside the graph",
			));
		}
		for operation in &self.operations[first..end] {
			if operation.id <= forward_op {
				return Err(Error::invalid_argument(
					"semantic backward expansion must follow its forward operation",
				));
			}
			if operation.backward_of.is_some() {
				return Err(Error::already_exists(
					"semantic operation already belongs to a backward expansion",
				));
			}
		}
		for operation in &mut self.operations[first..end] {
			operation.backward_of = Some(forward_op);
			operation.backward_sequence = sequence;
		}
		let attachment = &mut self.autograd[attachment_index];
		attachment.backward_first_op = Some(backward_first_op);
		attachment.backward_op_count = backward_op_count;
		attachment.backward_expanded = true;
		Ok(())
	}

	/// Validate canonical identities and every graph-local provenance edge.
	pub fn validate(&self) -> Result<()> {
		for (index, value) in self.values.iter().enumerate() {
			if value.id.map(SemanticValueId::usize) != Some(index) {
				return Err(Error::internal("semantic value ids are not canonical"));
			}
			if value
				.producer
				.is_some_and(|producer| producer.usize() >= self.operations.len())
			{
				return Err(Error::internal("semantic value has an invalid producer"));
			}
			if let Some(source) = value.view_source {
				if source.usize() >= index
					|| value.producer.is_some()
					|| value.external
					|| self.values[source.usize()].kind != value.kind
					|| self.values[source.usize()].dtype != value.dtype
				{
					return Err(Error::internal("semantic value has an invalid view source"));
				}
				let source_offset = i64::try_from(self.values[source.usize()].byte_offset)
					.map_err(|_| Error::internal("semantic view source offset exceeds i64"))?;
				let absolute = source_offset
					.checked_add(value.view_byte_offset)
					.filter(|offset| *offset >= 0)
					.ok_or_else(|| Error::internal("semantic view offset is inconsistent"))?;
				if value.byte_offset != absolute as u64 {
					return Err(Error::internal("semantic view offset is inconsistent"));
				}
			}
		}
		for (index, operation) in self.operations.iter().enumerate() {
			if operation.id.usize() != index || operation.name.is_empty() || operation.contract_hash == 0
			{
				return Err(Error::internal("semantic operation identity is invalid"));
			}
			for (input_index, input) in operation.inputs.iter().enumerate() {
				if input.is_none()
					&& input_index < u8::BITS as usize
					&& operation.optional_input_mask & (1 << input_index) != 0
				{
					continue;
				}
				if input.is_none_or(|value| value.usize() >= self.values.len()) {
					return Err(Error::internal("semantic operation has an invalid input"));
				}
			}
			for output in &operation.outputs {
				if output.usize() >= self.values.len()
					|| self.values[output.usize()].producer != Some(operation.id)
				{
					return Err(Error::internal(
						"semantic producer/output relationship is invalid",
					));
				}
			}
			let mut names = BTreeSet::new();
			for attribute in &operation.attributes {
				if !attribute.validate() || !names.insert(attribute.name()) {
					return Err(Error::internal("semantic operation has invalid attributes"));
				}
			}
			if operation
				.accesses
				.iter()
				.any(|access| access.value.usize() >= self.values.len())
				|| operation
					.mutated_inputs
					.iter()
					.any(|value| value.usize() >= self.values.len())
				|| operation.aliases.iter().any(|alias| {
					alias.output.usize() >= self.values.len()
						|| alias.input.usize() >= self.values.len()
						|| alias.output == alias.input
						|| !operation.outputs.contains(&alias.output)
						|| !operation.inputs.contains(&Some(alias.input))
				}) {
				return Err(Error::internal(
					"semantic operation has invalid value provenance",
				));
			}
			if operation.mutated_inputs.iter().any(|mutated| {
				operation
					.aliases
					.iter()
					.filter(|alias| alias.input == *mutated)
					.count()
					!= 1
			}) {
				return Err(Error::internal(
					"semantic mutation does not produce one fresh alias version",
				));
			}
			if operation
				.control_dependencies
				.iter()
				.any(|dependency| dependency.usize() >= index)
			{
				return Err(Error::internal(
					"semantic operation has an invalid control dependency",
				));
			}
			if operation
				.backward_of
				.is_some_and(|forward| forward.usize() >= index || operation.backward_sequence == 0)
			{
				return Err(Error::internal(
					"semantic backward operation has invalid provenance",
				));
			}
		}
		let mut outputs = BTreeSet::new();
		let mut sequences = BTreeSet::new();
		for attachment in &self.autograd {
			let operation = self
				.operations
				.get(attachment.forward_op.usize())
				.ok_or_else(|| Error::internal("semantic autograd has an invalid forward operation"))?;
			if operation.differentiation != OpDifferentiation::Reverse
				|| operation.outputs.get(attachment.output_index) != Some(&attachment.output)
				|| attachment.sequence == 0
				|| !outputs.insert(attachment.output)
				|| !sequences.insert(attachment.sequence)
			{
				return Err(Error::internal("semantic autograd attachment is invalid"));
			}
			if attachment.backward_expanded {
				let first = attachment.backward_first_op.ok_or_else(|| {
					Error::internal("expanded semantic backward range has no first operation")
				})?;
				let end = first
					.usize()
					.checked_add(attachment.backward_op_count)
					.filter(|end| *end <= self.operations.len())
					.ok_or_else(|| Error::internal("semantic backward range is invalid"))?;
				for backward in &self.operations[first.usize()..end] {
					if backward.backward_of != Some(attachment.forward_op)
						|| backward.backward_sequence != attachment.sequence
					{
						return Err(Error::internal(
							"semantic backward range disagrees with its operations",
						));
					}
				}
			} else if attachment.backward_first_op.is_some() || attachment.backward_op_count != 0 {
				return Err(Error::internal(
					"incomplete semantic backward attachment owns a range",
				));
			}
		}
		Ok(())
	}

	/// Replace this empty graph with a validated clone of `source`.
	pub fn copy_from(&mut self, source: &Self) -> Result<()> {
		if std::ptr::eq(self, source) {
			return Err(Error::invalid_argument(
				"semantic graph cannot copy onto itself",
			));
		}
		if !self.values.is_empty() || !self.operations.is_empty() || !self.autograd.is_empty() {
			return Err(Error::failed_precondition(
				"semantic graph copy requires an empty destination",
			));
		}
		source.validate()?;
		self.values.clone_from(&source.values);
		self.operations.clone_from(&source.operations);
		self.autograd.clone_from(&source.autograd);
		Ok(())
	}

	/// Return a value by graph-local identity.
	pub fn find_value(&self, id: SemanticValueId) -> Option<&SemanticValueDesc> {
		self.values.get(id.usize())
	}

	/// Return all semantic values in canonical order.
	pub fn values(&self) -> &[SemanticValueDesc] {
		&self.values
	}

	/// Return all semantic operations in canonical order.
	pub fn operations(&self) -> &[SemanticOpDesc] {
		&self.operations
	}

	/// Return all autograd attachments in capture order.
	pub fn autograd(&self) -> &[SemanticAutogradDesc] {
		&self.autograd
	}

	/// Return the deterministic donor-compatible semantic graph report as JSON.
	///
	/// The report is backend-independent and contains no storage handles,
	/// descriptors, addresses, kernels, or device policy.
	pub fn debug_report_json(&self, name: &str) -> String {
		let mut output = String::new();
		output.push_str("{\n  \"schema\": \"oa.semantic_graph.v2\",\n  \"name\": ");
		crate::core::push_json_string(&mut output, name);
		output.push_str(",\n  \"values\": [");
		for (index, value) in self.values.iter().enumerate() {
			output.push_str(if index == 0 { "\n" } else { ",\n" });
			crate::core::push_format(
				&mut output,
				format_args!(
					"    {{\"id\": {}, \"name\": ",
					value
						.id
						.expect("validated graph value must have an identity")
						.index()
				),
			);
			crate::core::push_json_string(&mut output, &value.name);
			output.push_str(", \"kind\": ");
			crate::core::push_json_string(&mut output, value.kind.token());
			output.push_str(", \"dtype\": ");
			crate::core::push_json_string(&mut output, dtype_report_token(value.dtype));
			output.push_str(", \"shape\": [");
			push_usize_list(&mut output, &value.shape);
			crate::core::push_format(
				&mut output,
				format_args!(
					"], \"byte_offset\": {}, \"external\": {}, \"virtual\": {}, \"strides\": [",
					value.byte_offset, value.external, value.virtual_value
				),
			);
			push_isize_list(&mut output, &value.strides);
			output.push_str("], \"view_source\": ");
			push_optional_value_id(&mut output, value.view_source);
			crate::core::push_format(
				&mut output,
				format_args!(
					", \"view_byte_offset\": {}, \"producer\": ",
					value.view_byte_offset
				),
			);
			push_optional_op_id(&mut output, value.producer);
			output.push('}');
		}
		if !self.values.is_empty() {
			output.push('\n');
		}
		output.push_str("  ],\n  \"operations\": [");
		for (index, operation) in self.operations.iter().enumerate() {
			output.push_str(if index == 0 { "\n" } else { ",\n" });
			crate::core::push_format(
				&mut output,
				format_args!("    {{\"id\": {}, \"name\": ", operation.id.index()),
			);
			crate::core::push_json_string(&mut output, operation.name);
			crate::core::push_format(
				&mut output,
				format_args!(
					", \"contract_hash\": \"0x{:016x}\", \"lowering\": ",
					operation.contract_hash
				),
			);
			crate::core::push_json_string(&mut output, operation.lowering.token());
			output.push_str(", \"differentiation\": ");
			crate::core::push_json_string(&mut output, operation.differentiation.token());
			output.push_str(", \"control_flow\": ");
			crate::core::push_json_string(&mut output, operation.control_flow.token());
			output.push_str(", \"inputs\": [");
			for (input_index, input) in operation.inputs.iter().copied().enumerate() {
				if input_index != 0 {
					output.push_str(", ");
				}
				push_optional_value_id(&mut output, input);
			}
			output.push_str("], \"outputs\": [");
			push_value_ids(&mut output, &operation.outputs);
			output.push_str("], \"attributes\": [");
			for (attribute_index, attribute) in operation.attributes.iter().enumerate() {
				if attribute_index != 0 {
					output.push_str(", ");
				}
				output.push_str("{\"name\": ");
				crate::core::push_json_string(&mut output, attribute.name());
				output.push_str(", \"kind\": ");
				crate::core::push_json_string(&mut output, attribute.kind().token());
				output.push_str(", \"value\": ");
				push_attribute_value(&mut output, attribute);
				output.push('}');
			}
			output.push_str("], \"accesses\": [");
			for (access_index, access) in operation.accesses.iter().enumerate() {
				if access_index != 0 {
					output.push_str(", ");
				}
				crate::core::push_format(
					&mut output,
					format_args!("{{\"value\": {}, \"mode\": ", access.value.index()),
				);
				crate::core::push_json_string(&mut output, access.mode.token());
				output.push('}');
			}
			output.push_str("], \"mutated_inputs\": [");
			push_value_ids(&mut output, &operation.mutated_inputs);
			output.push_str("], \"aliases\": [");
			for (alias_index, alias) in operation.aliases.iter().enumerate() {
				if alias_index != 0 {
					output.push_str(", ");
				}
				crate::core::push_format(
					&mut output,
					format_args!(
						"{{\"output\": {}, \"input\": {}}}",
						alias.output.index(),
						alias.input.index()
					),
				);
			}
			output.push_str("], \"control_dependencies\": [");
			push_op_ids(&mut output, &operation.control_dependencies);
			output.push_str("], \"backward_of\": ");
			push_optional_op_id(&mut output, operation.backward_of);
			crate::core::push_format(
				&mut output,
				format_args!(", \"backward_sequence\": {}}}", operation.backward_sequence),
			);
		}
		if !self.operations.is_empty() {
			output.push('\n');
		}
		output.push_str("  ],\n  \"autograd\": [");
		for (index, attachment) in self.autograd.iter().enumerate() {
			output.push_str(if index == 0 { "\n" } else { ",\n" });
			crate::core::push_format(
				&mut output,
				format_args!(
					"    {{\"forward_operation\": {}, \"output\": {}, \"output_index\": {}, \"sequence\": {}, \"backward_expanded\": {}, \"backward_first_operation\": ",
					attachment.forward_op.index(),
					attachment.output.index(),
					attachment.output_index,
					attachment.sequence,
					attachment.backward_expanded,
				),
			);
			push_optional_op_id(&mut output, attachment.backward_first_op);
			crate::core::push_format(
				&mut output,
				format_args!(
					", \"backward_operation_count\": {}}}",
					attachment.backward_op_count
				),
			);
		}
		if !self.autograd.is_empty() {
			output.push('\n');
		}
		output.push_str("  ]\n}\n");
		output
	}

	/// Return the number of metadata-only views.
	pub fn view_count(&self) -> usize {
		self
			.values
			.iter()
			.filter(|value| value.view_source.is_some())
			.count()
	}

	/// Return the generation incremented by each reset.
	pub const fn generation(&self) -> u64 {
		self.generation
	}

	/// Clear all graph content and advance its finite generation.
	pub fn reset(&mut self) {
		self.values.clear();
		self.operations.clear();
		self.autograd.clear();
		self.generation = self.generation.checked_add(1).unwrap_or(1);
	}
}

const fn dtype_report_token(dtype: DType) -> &'static str {
	match dtype {
		DType::U8 => "uint8",
		DType::F32 => "float32",
		DType::I32 => "int32",
		DType::U32 => "uint32",
	}
}

fn push_usize_list(output: &mut String, values: &[usize]) {
	for (index, value) in values.iter().enumerate() {
		if index != 0 {
			output.push_str(", ");
		}
		crate::core::push_format(output, format_args!("{value}"));
	}
}

fn push_isize_list(output: &mut String, values: &[isize]) {
	for (index, value) in values.iter().enumerate() {
		if index != 0 {
			output.push_str(", ");
		}
		crate::core::push_format(output, format_args!("{value}"));
	}
}

fn push_value_ids(output: &mut String, values: &[SemanticValueId]) {
	for (index, value) in values.iter().enumerate() {
		if index != 0 {
			output.push_str(", ");
		}
		crate::core::push_format(output, format_args!("{}", value.index()));
	}
}

fn push_op_ids(output: &mut String, values: &[SemanticOpId]) {
	for (index, value) in values.iter().enumerate() {
		if index != 0 {
			output.push_str(", ");
		}
		crate::core::push_format(output, format_args!("{}", value.index()));
	}
}

fn push_optional_value_id(output: &mut String, value: Option<SemanticValueId>) {
	if let Some(value) = value {
		crate::core::push_format(output, format_args!("{}", value.index()));
	} else {
		output.push_str("null");
	}
}

fn push_optional_op_id(output: &mut String, value: Option<SemanticOpId>) {
	if let Some(value) = value {
		crate::core::push_format(output, format_args!("{}", value.index()));
	} else {
		output.push_str("null");
	}
}

fn push_attribute_value(output: &mut String, attribute: &OpAttribute) {
	match attribute {
		OpAttribute::Boolean { value, .. } => {
			crate::core::push_format(output, format_args!("{value}"));
		}
		OpAttribute::SignedInteger { value, .. } => {
			crate::core::push_format(output, format_args!("{value}"));
		}
		OpAttribute::UnsignedInteger { value, .. } => {
			crate::core::push_format(output, format_args!("{value}"));
		}
		OpAttribute::Float { value, .. } if value.is_finite() => {
			crate::core::push_format(output, format_args!("{value}"));
		}
		OpAttribute::Float { .. } => output.push_str("null"),
		OpAttribute::String { value, .. } | OpAttribute::Enum { value, .. } => {
			crate::core::push_json_string(output, value);
		}
		OpAttribute::Shape { value, .. } => {
			output.push('[');
			push_usize_list(output, value);
			output.push(']');
		}
	}
}

fn merge_access(
	accesses: &mut Vec<SemanticValueAccess>,
	value: SemanticValueId,
	mode: SemanticAccessMode,
) {
	if let Some(access) = accesses.iter_mut().find(|access| access.value == value) {
		if access.mode != mode {
			access.mode = SemanticAccessMode::ReadWrite;
		}
	} else {
		accesses.push(SemanticValueAccess { value, mode });
	}
}

fn dense_strides(shape: &[usize]) -> Result<Vec<isize>> {
	let mut stride = 1_usize;
	let mut strides = vec![0_isize; shape.len()];
	for (index, dimension) in shape.iter().copied().enumerate().rev() {
		strides[index] = isize::try_from(stride)
			.map_err(|_| Error::invalid_argument("semantic value stride exceeds isize"))?;
		stride = stride
			.checked_mul(dimension)
			.ok_or_else(|| Error::invalid_argument("semantic value shape overflows usize"))?;
	}
	Ok(strides)
}

fn next_u32(length: usize, noun: &str) -> Result<u32> {
	u32::try_from(length)
		.map_err(|_| Error::resource_exhausted(format!("{noun} identity space is exhausted")))
}
