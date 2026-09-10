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
	Swiglu {
		gate: Matrix,
		up: Matrix,
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
}

impl Node {
	pub(super) const fn output_id(&self) -> u64 {
		match self {
			Self::Add { output_id, .. }
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
			| Self::Swiglu { output_id, .. }
			| Self::Bmm { output_id, .. }
			| Self::BmmNt { output_id, .. }
			| Self::BmmTn { output_id, .. }
			| Self::SplitHeads { output_id, .. }
			| Self::MergeHeads { output_id, .. }
			| Self::SoftmaxScaledMasked { output_id, .. }
			| Self::ScaledDotProductAttention { output_id, .. }
			| Self::FlashAttention { output_id, .. }
			| Self::MoeRouteWeights { output_id, .. }
			| Self::Embedding { output_id, .. }
			| Self::CrossEntropy { output_id, .. }
			| Self::MaskedCrossEntropy { output_id, .. }
			| Self::SmoothL1 { output_id, .. }
			| Self::Mse { output_id, .. }
			| Self::L1 { output_id, .. }
			| Self::Bce { output_id, .. } => *output_id,
			Self::GruCell(node) => node.output_id,
			Self::GruScan(node) => node.output_id,
			Self::RnnCell(node) => node.output_id,
			Self::RnnScan(node) => node.output_id,
		}
	}

	pub(super) const fn is_loss_root(&self, value_id: u64) -> bool {
		matches!(
			self,
			Self::CrossEntropy { output_id, .. }
				| Self::MaskedCrossEntropy { output_id, .. }
				| Self::SmoothL1 { output_id, .. }
				| Self::Mse { output_id, .. }
				| Self::L1 { output_id, .. }
				| Self::Bce { output_id, .. }
				if *output_id == value_id
		)
	}
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
