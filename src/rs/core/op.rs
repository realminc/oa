//! Backend-independent operation contracts.

use bitflags::bitflags;

/// Semantic kind of one operation value.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OpValueKind {
	/// Dense numeric matrix.
	Matrix,
	/// Image value.
	Image,
	/// Audio value.
	Audio,
	/// Video-frame value.
	VideoFrame,
	/// Quantized matrix value.
	QuantMatrix,
}

impl OpValueKind {
	/// Return the stable semantic-report token.
	pub const fn token(self) -> &'static str {
		match self {
			Self::Matrix => "matrix",
			Self::Image => "image",
			Self::Audio => "audio",
			Self::VideoFrame => "video_frame",
			Self::QuantMatrix => "quant_matrix",
		}
	}
}

/// Uniform shape relationship declared by an operation schema.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum OpShapeRule {
	/// Output shape matches the admitted input shape.
	#[default]
	MatchInput,
	/// Inputs use multidirectional broadcasting.
	Broadcast,
	/// Right input uses OA's `[N, K]` transposed-weight convention.
	MatMulNt,
	/// Operation-specific validation and inference is required.
	Explicit,
}

/// Uniform dtype relationship declared by an operation schema.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum OpDTypeRule {
	/// Inputs and outputs retain one dtype.
	#[default]
	MatchInput,
	/// Inputs are promoted to a floating-point result.
	PromoteFloat,
	/// Operation-specific validation and inference is required.
	Explicit,
}

/// Reverse-differentiation behavior declared by an operation schema.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum OpDifferentiation {
	/// The operation has no reverse derivative.
	#[default]
	None,
	/// The operation participates in reverse-mode differentiation.
	Reverse,
}

impl OpDifferentiation {
	/// Return the stable semantic-report token.
	pub const fn token(self) -> &'static str {
		match self {
			Self::None => "none",
			Self::Reverse => "reverse",
		}
	}
}

/// Backend-neutral lowering family requested by an operation schema.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum OpLowering {
	/// Lower through the generic compute-dispatch path.
	#[default]
	Dispatch,
	/// Lower through the engine's GEMM service.
	Gemm,
}

impl OpLowering {
	/// Return the stable semantic-report token.
	pub const fn token(self) -> &'static str {
		match self {
			Self::Dispatch => "dispatch",
			Self::Gemm => "gemm",
		}
	}
}

/// Semantic control-flow class of an operation.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum OpControlFlow {
	/// Ordinary acyclic dataflow.
	#[default]
	StraightLine,
	/// Conditional execution.
	Conditional,
	/// Loop or recurrence.
	Loop,
}

impl OpControlFlow {
	/// Return the stable semantic-report token.
	pub const fn token(self) -> &'static str {
		match self {
			Self::StraightLine => "straight_line",
			Self::Conditional => "conditional",
			Self::Loop => "loop",
		}
	}
}

bitflags! {
	/// Logical value effects of an operation.
	#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
	pub struct OpEffect: u8 {
		/// Read logical inputs.
		const READ_INPUTS = 1 << 0;
		/// Write logical outputs.
		const WRITE_OUTPUTS = 1 << 1;
	}
}

/// Stable type tag for a non-value operation attribute.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OpAttributeKind {
	/// Boolean value.
	Boolean,
	/// Signed integer value.
	SignedInteger,
	/// Unsigned integer value.
	UnsignedInteger,
	/// Floating-point value.
	Float,
	/// Free-form UTF-8 string.
	String,
	/// Matrix shape.
	Shape,
	/// Symbolic enum value.
	Enum,
}

impl OpAttributeKind {
	/// Return the stable semantic-report token.
	pub const fn token(self) -> &'static str {
		match self {
			Self::Boolean => "boolean",
			Self::SignedInteger => "signed_integer",
			Self::UnsignedInteger => "unsigned_integer",
			Self::Float => "float",
			Self::String => "string",
			Self::Shape => "shape",
			Self::Enum => "enum",
		}
	}
}

/// One ordered non-value input that affects operation meaning.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq)]
pub enum OpAttribute {
	/// Boolean attribute.
	Boolean { name: String, value: bool },
	/// Signed integer attribute.
	SignedInteger { name: String, value: i64 },
	/// Unsigned integer attribute.
	UnsignedInteger { name: String, value: u64 },
	/// Floating-point attribute.
	Float { name: String, value: f64 },
	/// String attribute.
	String { name: String, value: String },
	/// Shape attribute.
	Shape { name: String, value: Vec<usize> },
	/// Symbolic enum attribute.
	Enum { name: String, value: String },
}

impl OpAttribute {
	/// Return this attribute's schema name.
	pub fn name(&self) -> &str {
		match self {
			Self::Boolean { name, .. }
			| Self::SignedInteger { name, .. }
			| Self::UnsignedInteger { name, .. }
			| Self::Float { name, .. }
			| Self::String { name, .. }
			| Self::Shape { name, .. }
			| Self::Enum { name, .. } => name,
		}
	}

	/// Return this attribute's stable kind.
	pub const fn kind(&self) -> OpAttributeKind {
		match self {
			Self::Boolean { .. } => OpAttributeKind::Boolean,
			Self::SignedInteger { .. } => OpAttributeKind::SignedInteger,
			Self::UnsignedInteger { .. } => OpAttributeKind::UnsignedInteger,
			Self::Float { .. } => OpAttributeKind::Float,
			Self::String { .. } => OpAttributeKind::String,
			Self::Shape { .. } => OpAttributeKind::Shape,
			Self::Enum { .. } => OpAttributeKind::Enum,
		}
	}

	pub(crate) fn validate(&self) -> bool {
		!self.name().is_empty() && !matches!(self, Self::Enum { value, .. } if value.is_empty())
	}
}

/// Ordered schema declaration for one operation attribute.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OpAttributeSpec {
	name: &'static str,
	kind: OpAttributeKind,
}

impl OpAttributeSpec {
	/// Construct one generated attribute declaration.
	pub const fn new(name: &'static str, kind: OpAttributeKind) -> Self {
		Self { name, kind }
	}

	/// Return the declared attribute name.
	pub const fn name(self) -> &'static str {
		self.name
	}

	/// Return the declared attribute kind.
	pub const fn kind(self) -> OpAttributeKind {
		self.kind
	}
}

/// Stable semantic contract for one operation before backend lowering.
///
/// Generated operation registries own these descriptors. The builder-style
/// `const` methods are public so generated code remains ordinary inspectable
/// Rust rather than a private binary table.
#[derive(Clone, Copy, Debug)]
pub struct OperationContract {
	name: &'static str,
	hash: u64,
	input_kinds: &'static [OpValueKind],
	output_kinds: &'static [OpValueKind],
	variadic_input: Option<(OpValueKind, usize)>,
	variadic_output: Option<(OpValueKind, usize)>,
	aligned_variadic_aliases: bool,
	attributes: &'static [OpAttributeSpec],
	shape_rule: OpShapeRule,
	dtype_rule: OpDTypeRule,
	differentiation: OpDifferentiation,
	lowering: OpLowering,
	effects: OpEffect,
	mutated_input_mask: u8,
	optional_input_mask: u8,
	output_alias_inputs: [Option<u8>; Self::MAX_VALUES],
	control_flow: OpControlFlow,
}

impl OperationContract {
	/// Maximum number of fixed logical inputs or outputs.
	pub const MAX_VALUES: usize = 8;
	/// Maximum number of semantic attributes.
	pub const MAX_ATTRIBUTES: usize = 8;

	/// Construct the fixed-arity portion of an operation contract.
	pub const fn new(
		name: &'static str,
		hash: u64,
		input_kinds: &'static [OpValueKind],
		output_kinds: &'static [OpValueKind],
	) -> Self {
		Self {
			name,
			hash,
			input_kinds,
			output_kinds,
			variadic_input: None,
			variadic_output: None,
			aligned_variadic_aliases: false,
			attributes: &[],
			shape_rule: OpShapeRule::MatchInput,
			dtype_rule: OpDTypeRule::MatchInput,
			differentiation: OpDifferentiation::None,
			lowering: OpLowering::Dispatch,
			effects: OpEffect::empty(),
			mutated_input_mask: 0,
			optional_input_mask: 0,
			output_alias_inputs: [None; Self::MAX_VALUES],
			control_flow: OpControlFlow::StraightLine,
		}
	}

	/// Declare a homogeneous variadic input tail and its minimum length.
	pub const fn variadic_inputs(mut self, kind: OpValueKind, minimum: usize) -> Self {
		self.variadic_input = Some((kind, minimum));
		self
	}

	/// Declare a homogeneous variadic output tail and its minimum length.
	pub const fn variadic_outputs(mut self, kind: OpValueKind, minimum: usize) -> Self {
		self.variadic_output = Some((kind, minimum));
		self
	}

	/// Declare that each variadic output aliases and mutates the variadic input
	/// at the same tail position.
	pub const fn aligned_variadic_aliases(mut self) -> Self {
		self.aligned_variadic_aliases = true;
		self
	}

	/// Declare the ordered non-value input signature.
	pub const fn attributes(mut self, attributes: &'static [OpAttributeSpec]) -> Self {
		self.attributes = attributes;
		self
	}

	/// Declare reverse-differentiation behavior.
	pub const fn with_differentiation(mut self, value: OpDifferentiation) -> Self {
		self.differentiation = value;
		self
	}

	/// Declare the schema's uniform shape rule.
	pub const fn with_shape_rule(mut self, value: OpShapeRule) -> Self {
		self.shape_rule = value;
		self
	}

	/// Declare the schema's uniform dtype rule.
	pub const fn with_dtype_rule(mut self, value: OpDTypeRule) -> Self {
		self.dtype_rule = value;
		self
	}

	/// Declare the backend-neutral lowering family.
	pub const fn with_lowering(mut self, value: OpLowering) -> Self {
		self.lowering = value;
		self
	}

	/// Declare logical read and write effects.
	pub const fn effects(mut self, value: OpEffect) -> Self {
		self.effects = value;
		self
	}

	/// Declare the fixed-input mutation bit mask.
	pub const fn mutated_inputs(mut self, mask: u8) -> Self {
		self.mutated_input_mask = mask;
		self
	}

	/// Declare the fixed-input optionality bit mask.
	pub const fn optional_inputs(mut self, mask: u8) -> Self {
		self.optional_input_mask = mask;
		self
	}

	/// Declare that one output aliases a fixed input's storage.
	///
	/// Aliasing does not imply mutation. A pass-through output is a fresh SSA
	/// value over the same bytes; an in-place write additionally declares the
	/// input in [`Self::mutated_inputs`].
	pub const fn alias(mut self, output: usize, input: u8) -> Self {
		assert!(
			output < Self::MAX_VALUES,
			"operation alias output exceeds fixed capacity"
		);
		self.output_alias_inputs[output] = Some(input);
		self
	}

	/// Declare semantic control flow.
	pub const fn with_control_flow(mut self, value: OpControlFlow) -> Self {
		self.control_flow = value;
		self
	}

	/// Return the stable qualified operation name.
	pub const fn name(self) -> &'static str {
		self.name
	}

	/// Return the schema-derived semantic contract hash.
	pub const fn hash(self) -> u64 {
		self.hash
	}

	/// Return the declared differentiation behavior.
	pub const fn differentiation(self) -> OpDifferentiation {
		self.differentiation
	}

	/// Return the uniform shape rule.
	pub const fn shape_rule(self) -> OpShapeRule {
		self.shape_rule
	}

	/// Return the uniform dtype rule.
	pub const fn dtype_rule(self) -> OpDTypeRule {
		self.dtype_rule
	}

	/// Return the ordered non-value attribute signature.
	pub const fn attribute_specs(self) -> &'static [OpAttributeSpec] {
		self.attributes
	}

	/// Return the declared lowering family.
	pub const fn lowering(self) -> OpLowering {
		self.lowering
	}

	/// Return the declared control-flow class.
	pub const fn control_flow(self) -> OpControlFlow {
		self.control_flow
	}

	pub(crate) fn accepts_input_count(self, count: usize) -> bool {
		self.variadic_input
			.map_or(count == self.input_kinds.len(), |(_, minimum)| {
				count >= self.input_kinds.len().saturating_add(minimum)
			})
	}

	pub(crate) fn accepts_output_count(self, count: usize) -> bool {
		self.variadic_output
			.map_or(count == self.output_kinds.len(), |(_, minimum)| {
				count >= self.output_kinds.len().saturating_add(minimum)
			})
	}

	pub(crate) fn input_kind(self, index: usize) -> Option<OpValueKind> {
		self.input_kinds
			.get(index)
			.copied()
			.or_else(|| self.variadic_input.map(|(kind, _)| kind))
	}

	pub(crate) fn output_kind(self, index: usize) -> Option<OpValueKind> {
		self.output_kinds
			.get(index)
			.copied()
			.or_else(|| self.variadic_output.map(|(kind, _)| kind))
	}

	pub(crate) const fn mutates_input(self, index: usize) -> bool {
		(index < Self::MAX_VALUES && self.mutated_input_mask & (1 << index) != 0)
			|| (self.aligned_variadic_aliases && index >= self.input_kinds.len())
	}

	pub(crate) const fn input_is_optional(self, index: usize) -> bool {
		index < Self::MAX_VALUES && self.optional_input_mask & (1 << index) != 0
	}

	pub(crate) fn alias_input(self, output: usize) -> Option<usize> {
		let fixed = self
			.output_alias_inputs
			.get(output)
			.copied()
			.flatten()
			.map(usize::from);
		fixed.or_else(|| {
			(self.aligned_variadic_aliases && output >= self.output_kinds.len())
				.then(|| self.input_kinds.len() + (output - self.output_kinds.len()))
		})
	}

	pub(crate) fn variadic_alias_counts_match(
		self,
		input_count: usize,
		output_count: usize,
	) -> bool {
		!self.aligned_variadic_aliases
			|| input_count.saturating_sub(self.input_kinds.len())
				== output_count.saturating_sub(self.output_kinds.len())
	}

	pub(crate) fn validate(self, attributes: &[OpAttribute]) -> bool {
		if self.name.is_empty()
			|| self.hash == 0
			|| self.input_kinds.len() > Self::MAX_VALUES
			|| self.output_kinds.len() > Self::MAX_VALUES
			|| self.attributes.len() > Self::MAX_ATTRIBUTES
			|| attributes.len() != self.attributes.len()
		{
			return false;
		}
		let input_mask = if self.input_kinds.is_empty() {
			0
		} else {
			((1_u16 << self.input_kinds.len()) - 1) as u8
		};
		if self.variadic_input.is_some_and(|(_, minimum)| minimum == 0)
			|| self
				.variadic_output
				.is_some_and(|(_, minimum)| minimum == 0)
			|| self.optional_input_mask & !input_mask != 0
			|| self.mutated_input_mask & !input_mask != 0
			|| self.optional_input_mask & self.mutated_input_mask != 0
		{
			return false;
		}
		if self.aligned_variadic_aliases
			&& (!matches!(
				(self.variadic_input, self.variadic_output),
				(Some((input, input_minimum)), Some((output, output_minimum)))
					if input == output && input_minimum == output_minimum
			) || !self.effects.contains(OpEffect::WRITE_OUTPUTS))
		{
			return false;
		}
		for (output, alias) in self.output_alias_inputs.iter().copied().enumerate() {
			let Some(input) = alias else { continue };
			let input = usize::from(input);
			if output >= self.output_kinds.len() || input >= self.input_kinds.len() {
				return false;
			}
		}
		for input in 0..self.input_kinds.len() {
			if self.mutates_input(input)
				&& self
					.output_alias_inputs
					.iter()
					.take(self.output_kinds.len())
					.filter(|alias| alias.is_some_and(|alias| usize::from(alias) == input))
					.count() != 1
			{
				return false;
			}
		}
		attributes.iter().zip(self.attributes).all(|(value, spec)| {
			value.validate() && value.name() == spec.name && value.kind() == spec.kind
		})
	}

	pub(crate) const fn effects_value(self) -> OpEffect {
		self.effects
	}

	pub(crate) const fn optional_input_mask(self) -> u8 {
		self.optional_input_mask
	}
}
