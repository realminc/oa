//! Private saved-value nodes consumed by reverse traversal.

use crate::Matrix;
use crate::ml::Parameter;
use crate::ml::matrix::UpsampleMode;

pub(super) enum Node {
	Add {
		left: Matrix,
		right: Matrix,
		output_id: u64,
	},
	Mul {
		left: Matrix,
		right: Matrix,
		output_id: u64,
	},
	Div {
		left: Matrix,
		right: Matrix,
		output_id: u64,
	},
	Scale {
		input: Matrix,
		output_id: u64,
		scalar: f32,
	},
	Reciprocal {
		input: Matrix,
		output: Matrix,
		output_id: u64,
	},
	Exp {
		input: Matrix,
		output: Matrix,
		output_id: u64,
	},
	Log {
		input: Matrix,
		output_id: u64,
	},
	Abs {
		input: Matrix,
		output_id: u64,
	},
	Copy {
		input: Matrix,
		output_id: u64,
	},
	Sqrt {
		input: Matrix,
		output: Matrix,
		output_id: u64,
	},
	ClampMax {
		input: Matrix,
		output_id: u64,
		maximum: f32,
	},
	ClampMin {
		input: Matrix,
		output_id: u64,
		minimum: f32,
	},
	Sub {
		left: Matrix,
		right: Matrix,
		output_id: u64,
	},
	Slice {
		input: Matrix,
		output_id: u64,
		dim: usize,
		start: usize,
		end: usize,
	},
	RepeatInterleave {
		input: Matrix,
		output_id: u64,
		repeats: usize,
		dim: usize,
	},
	Concat {
		inputs: Vec<Matrix>,
		output_id: u64,
		dim: usize,
		sizes: Vec<usize>,
	},
	GatherLastDim {
		input: Matrix,
		indices: Matrix,
		output_id: u64,
		input_width: usize,
	},
	FlowLinearState {
		clean: Matrix,
		noise: Matrix,
		time: Matrix,
		output_id: u64,
	},
	FlowLinearVelocity {
		clean: Matrix,
		noise: Matrix,
		output_id: u64,
	},
	FlowEulerStep {
		state: Matrix,
		velocity: Matrix,
		delta_time: f32,
		output_id: u64,
	},
	FlowMaskedMse {
		prediction: Matrix,
		target: Matrix,
		mask: Matrix,
		denominator: Matrix,
		output_id: u64,
	},
	Dropout {
		input: Matrix,
		output_id: u64,
		probability: f32,
		seed: u64,
	},
	Reshape {
		input: Matrix,
		output_id: u64,
	},
	Linear {
		input: Matrix,
		output_id: u64,
		weight: Parameter,
		weight_value: Matrix,
		weight_version: u64,
		bias: Option<Parameter>,
		bias_version: Option<u64>,
	},
	Conv1d {
		input: Matrix,
		output_id: u64,
		weight: Option<Parameter>,
		weight_value: Matrix,
		weight_version: Option<u64>,
		bias: Option<Parameter>,
		bias_value: Matrix,
		bias_version: Option<u64>,
		stride: usize,
		padding: usize,
		dilation: usize,
	},
	ConvTranspose1d {
		input: Matrix,
		output_id: u64,
		weight: Option<Parameter>,
		weight_value: Matrix,
		weight_version: Option<u64>,
		stride: usize,
		padding: usize,
		dilation: usize,
	},
	ConvTranspose2d {
		input: Matrix,
		output_id: u64,
		weight: Option<Parameter>,
		weight_value: Matrix,
		weight_version: Option<u64>,
		bias: Option<Parameter>,
		bias_value: Matrix,
		bias_version: Option<u64>,
		stride: usize,
		padding: usize,
	},
	Conv2d {
		input: Matrix,
		output_id: u64,
		weight: Option<Parameter>,
		weight_value: Matrix,
		weight_version: Option<u64>,
		bias: Option<Parameter>,
		bias_value: Matrix,
		bias_version: Option<u64>,
		stride: usize,
		padding: usize,
		groups: usize,
	},
	LayerNorm {
		input: Matrix,
		normalized: Matrix,
		inverse_stddev: Matrix,
		output_id: u64,
		weight: Parameter,
		weight_value: Matrix,
		weight_version: u64,
		bias: Parameter,
		bias_version: u64,
	},
	BatchNorm2d {
		input: Matrix,
		mean: Matrix,
		variance: Matrix,
		output_id: u64,
		weight: Option<Parameter>,
		weight_value: Matrix,
		weight_version: Option<u64>,
		bias: Option<Parameter>,
		bias_value: Matrix,
		bias_version: Option<u64>,
		epsilon: f32,
		training: bool,
	},
	RmsNorm {
		input: Matrix,
		output_id: u64,
		weight: Option<Parameter>,
		weight_value: Matrix,
		weight_version: Option<u64>,
		epsilon: f32,
	},
	Rope {
		input: Matrix,
		output_id: u64,
		num_heads: usize,
		head_dim: usize,
		theta_base: f32,
		position_offset: u32,
	},
	Gelu {
		input: Matrix,
		output_id: u64,
	},
	Silu {
		input: Matrix,
		output_id: u64,
	},
	Relu {
		input: Matrix,
		output: Matrix,
		output_id: u64,
	},
	Tanh {
		input: Matrix,
		output: Matrix,
		output_id: u64,
	},
	Sigmoid {
		input: Matrix,
		output: Matrix,
		output_id: u64,
	},
	LeakyRelu {
		input: Matrix,
		alpha: f32,
		output_id: u64,
	},
	Elu {
		input: Matrix,
		output: Matrix,
		alpha: f32,
		output_id: u64,
	},
	Mish {
		input: Matrix,
		output_id: u64,
	},
	Softplus {
		input: Matrix,
		output: Matrix,
		output_id: u64,
	},
	Softmax {
		input: Matrix,
		output: Matrix,
		output_id: u64,
		dim: i32,
	},
	LogSoftmax {
		input: Matrix,
		output: Matrix,
		output_id: u64,
		dim: i32,
	},
	AvgPool2d {
		input: Matrix,
		output_id: u64,
		kernel_size: usize,
		stride: usize,
		padding: usize,
	},
	MaxPool2d {
		input: Matrix,
		indices: Matrix,
		output_id: u64,
		kernel_size: usize,
		stride: usize,
		padding: usize,
	},
	AdaptiveAvgPool2d {
		input: Matrix,
		output_id: u64,
		output_height: usize,
		output_width: usize,
	},
	Upsample2d {
		input: Matrix,
		output_id: u64,
		scale_factor: usize,
		mode: UpsampleMode,
	},
	Sum {
		input: Matrix,
		output_id: u64,
		dim: i32,
	},
	MatMulNt {
		left: Matrix,
		right: Matrix,
		output_id: u64,
	},
	Swiglu {
		gate: Matrix,
		up: Matrix,
		output_id: u64,
	},
	SiluMul {
		input: Matrix,
		intermediate_size: usize,
		output_id: u64,
	},
	Bmm {
		left: Matrix,
		right: Matrix,
		output_id: u64,
	},
	BmmNt {
		left: Matrix,
		right: Matrix,
		output_id: u64,
	},
	BmmTn {
		left: Matrix,
		right: Matrix,
		output_id: u64,
	},
	SplitHeads {
		input: Matrix,
		output_id: u64,
		batch: usize,
		sequence_length: usize,
		num_heads: usize,
	},
	MergeHeads {
		input: Matrix,
		output_id: u64,
		batch: usize,
		sequence_length: usize,
		num_heads: usize,
	},
	SoftmaxScaledMasked {
		input: Matrix,
		output: Matrix,
		output_id: u64,
		scale: f32,
	},
	ScaledDotProductAttention {
		query: Matrix,
		key: Matrix,
		value: Matrix,
		probabilities: Matrix,
		output_id: u64,
		scale: f32,
	},
	FlashAttention {
		query: Matrix,
		key: Matrix,
		value: Matrix,
		output: Matrix,
		log_sum_exp: Matrix,
		output_id: u64,
		scale: f32,
	},
	MoeRouteWeights {
		probabilities: Matrix,
		expert_indices: Matrix,
		output: Matrix,
		output_id: u64,
	},
	MoeGather {
		input: Matrix,
		inverse: Matrix,
		output_id: u64,
	},
	MoeCombine {
		packed: Matrix,
		route_gate: Matrix,
		inverse: Matrix,
		packed_slot: Matrix,
		output_id: u64,
	},
	GroupedGemmM {
		input: Matrix,
		weight: Option<Parameter>,
		weight_value: Matrix,
		weight_version: Option<u64>,
		offsets: Matrix,
		output_id: u64,
	},
	GroupedLinearM {
		input: Matrix,
		weight: Option<Parameter>,
		weight_value: Matrix,
		weight_version: Option<u64>,
		bias: Option<Parameter>,
		bias_value: Matrix,
		bias_version: Option<u64>,
		offsets: Matrix,
		output_id: u64,
	},
	Embedding {
		indices: Matrix,
		output_id: u64,
		weight: Parameter,
		weight_value: Matrix,
		weight_version: u64,
	},
	GruCell(Box<GruCellNode>),
	GruScan(Box<GruScanNode>),
	RnnCell(Box<RnnCellNode>),
	RnnScan(Box<RnnScanNode>),
	Mamba3Siso(Box<Mamba3SisoNode>),
	CrossEntropy {
		logits: Matrix,
		targets: Matrix,
		output_id: u64,
	},
	MaskedCrossEntropy {
		logits: Matrix,
		targets: Matrix,
		mask: Matrix,
		valid_count: usize,
		output_id: u64,
	},
	SmoothL1 {
		prediction: Matrix,
		target: Matrix,
		output_id: u64,
	},
	Mse {
		prediction: Matrix,
		target: Matrix,
		output_id: u64,
	},
	L1 {
		prediction: Matrix,
		target: Matrix,
		output_id: u64,
	},
	Bce {
		prediction: Matrix,
		target: Matrix,
		output_id: u64,
	},
	PpoClippedPolicy {
		new_log_probability: Matrix,
		old_log_probability: Matrix,
		advantage: Matrix,
		clip_epsilon: f32,
		output_id: u64,
	},
}

impl Node {
	pub(super) const fn output_id(&self) -> u64 {
		match self {
			Self::Add { output_id, .. }
			| Self::Mul { output_id, .. }
			| Self::Div { output_id, .. }
			| Self::Scale { output_id, .. }
			| Self::Reciprocal { output_id, .. }
			| Self::Exp { output_id, .. }
			| Self::Log { output_id, .. }
			| Self::Abs { output_id, .. }
			| Self::Copy { output_id, .. }
			| Self::Sqrt { output_id, .. }
			| Self::ClampMax { output_id, .. }
			| Self::ClampMin { output_id, .. }
			| Self::Sub { output_id, .. }
			| Self::Slice { output_id, .. }
			| Self::RepeatInterleave { output_id, .. }
			| Self::Concat { output_id, .. }
			| Self::GatherLastDim { output_id, .. }
			| Self::FlowLinearState { output_id, .. }
			| Self::FlowLinearVelocity { output_id, .. }
			| Self::FlowEulerStep { output_id, .. }
			| Self::FlowMaskedMse { output_id, .. }
			| Self::Reshape { output_id, .. }
			| Self::Dropout { output_id, .. }
			| Self::Linear { output_id, .. }
			| Self::Conv1d { output_id, .. }
			| Self::ConvTranspose1d { output_id, .. }
			| Self::ConvTranspose2d { output_id, .. }
			| Self::Conv2d { output_id, .. }
			| Self::LayerNorm { output_id, .. }
			| Self::BatchNorm2d { output_id, .. }
			| Self::RmsNorm { output_id, .. }
			| Self::Rope { output_id, .. }
			| Self::Gelu { output_id, .. }
			| Self::Silu { output_id, .. }
			| Self::Relu { output_id, .. }
			| Self::Tanh { output_id, .. }
			| Self::Sigmoid { output_id, .. }
			| Self::LeakyRelu { output_id, .. }
			| Self::Elu { output_id, .. }
			| Self::Mish { output_id, .. }
			| Self::Softplus { output_id, .. }
			| Self::Softmax { output_id, .. }
			| Self::LogSoftmax { output_id, .. }
			| Self::AvgPool2d { output_id, .. }
			| Self::MaxPool2d { output_id, .. }
			| Self::AdaptiveAvgPool2d { output_id, .. }
			| Self::Upsample2d { output_id, .. }
			| Self::Sum { output_id, .. }
			| Self::MatMulNt { output_id, .. }
			| Self::Swiglu { output_id, .. }
			| Self::SiluMul { output_id, .. }
			| Self::Bmm { output_id, .. }
			| Self::BmmNt { output_id, .. }
			| Self::BmmTn { output_id, .. }
			| Self::SplitHeads { output_id, .. }
			| Self::MergeHeads { output_id, .. }
			| Self::SoftmaxScaledMasked { output_id, .. }
			| Self::ScaledDotProductAttention { output_id, .. }
			| Self::FlashAttention { output_id, .. }
			| Self::MoeRouteWeights { output_id, .. }
			| Self::MoeGather { output_id, .. }
			| Self::MoeCombine { output_id, .. }
			| Self::GroupedGemmM { output_id, .. }
			| Self::GroupedLinearM { output_id, .. }
			| Self::Embedding { output_id, .. }
			| Self::CrossEntropy { output_id, .. }
			| Self::MaskedCrossEntropy { output_id, .. }
			| Self::SmoothL1 { output_id, .. }
			| Self::Mse { output_id, .. }
			| Self::L1 { output_id, .. }
			| Self::Bce { output_id, .. }
			| Self::PpoClippedPolicy { output_id, .. } => *output_id,
			Self::GruCell(node) => node.output_id,
			Self::GruScan(node) => node.output_id,
			Self::RnnCell(node) => node.output_id,
			Self::RnnScan(node) => node.output_id,
			Self::Mamba3Siso(node) => node.output_id,
		}
	}
}

pub(super) struct Mamba3SisoNode {
	pub(super) c: Matrix,
	pub(super) b: Matrix,
	pub(super) x: Matrix,
	pub(super) z: Matrix,
	pub(super) adt: Matrix,
	pub(super) dt: Matrix,
	pub(super) trap: Matrix,
	pub(super) angle: Matrix,
	pub(super) c_bias: Matrix,
	pub(super) b_bias: Matrix,
	pub(super) d: Matrix,
	pub(super) config: crate::ml::matrix::SsmConfig,
	pub(super) output_id: u64,
}

pub(super) struct RnnScanNode {
	pub(super) gates_i: Matrix,
	pub(super) hidden_previous: Matrix,
	pub(super) output_id: u64,
	pub(super) weight: Option<Parameter>,
	pub(super) weight_value: Matrix,
	pub(super) weight_version: Option<u64>,
	pub(super) bias: Option<Parameter>,
	pub(super) bias_value: Matrix,
	pub(super) bias_version: Option<u64>,
	pub(super) has_bias: bool,
}

pub(super) struct RnnCellNode {
	pub(super) gates_i: Matrix,
	pub(super) gates_h: Matrix,
	pub(super) hidden: Matrix,
	pub(super) output_id: u64,
	pub(super) weight: Option<Parameter>,
	pub(super) weight_value: Matrix,
	pub(super) weight_version: Option<u64>,
	pub(super) bias: Option<Parameter>,
	pub(super) bias_value: Matrix,
	pub(super) bias_version: Option<u64>,
	pub(super) has_bias: bool,
}

pub(super) struct GruScanNode {
	pub(super) gates_i: Matrix,
	pub(super) hidden_previous: Matrix,
	pub(super) output_id: u64,
	pub(super) weight: Option<Parameter>,
	pub(super) weight_value: Matrix,
	pub(super) weight_version: Option<u64>,
	pub(super) bias: Option<Parameter>,
	pub(super) bias_value: Matrix,
	pub(super) bias_version: Option<u64>,
	pub(super) has_bias: bool,
}

pub(super) struct GruCellNode {
	pub(super) gates_i: Matrix,
	pub(super) gates_h: Matrix,
	pub(super) hidden: Matrix,
	pub(super) output_id: u64,
	pub(super) weight: Option<Parameter>,
	pub(super) weight_value: Matrix,
	pub(super) weight_version: Option<u64>,
	pub(super) bias: Option<Parameter>,
	pub(super) bias_value: Matrix,
	pub(super) bias_version: Option<u64>,
	pub(super) has_bias: bool,
}
