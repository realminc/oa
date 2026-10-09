//! Private saved-value nodes consumed by reverse traversal.

use crate::Matrix;
use crate::ml::Parameter;
use operation::GradNodeOperation;

#[path = "node/operation.gen.rs"]
pub(super) mod operation;

mod validation;

pub(super) enum GradNode {
	Matrix(crate::matrix::autograd::GradNodeMatrix),
	Operation(GradNodeOperation),
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
	RmsNormGated(Box<GradNodeRmsNormGated>),
	ChannelNorm(Box<GradNodeChannelNorm>),
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
	GruCell(Box<GradNodeGruCell>),
	GruScan(Box<GradNodeGruScan>),
	RnnCell(Box<GradNodeRnnCell>),
	RnnScan(Box<GradNodeRnnScan>),
	Mamba3Preprocess(Box<GradNodeMamba3Preprocess>),
	Mamba3Siso(Box<GradNodeMamba3Siso>),
	Mamba3Mimo(Box<GradNodeMamba3Mimo>),
}

pub(super) struct GradNodeChannelNorm {
	pub(super) input: Matrix,
	pub(super) output: Option<Matrix>,
	pub(super) output_id: u64,
	pub(super) weight: Option<Parameter>,
	pub(super) weight_value: Matrix,
	pub(super) weight_version: Option<u64>,
	pub(super) bias: Option<Parameter>,
	pub(super) bias_value: Matrix,
	pub(super) bias_version: Option<u64>,
	pub(super) epsilon: f32,
}

impl GradNode {
	pub(super) fn output_id(&self) -> u64 {
		match self {
			Self::Operation(node) => node.output_id(),
			Self::Matrix(node) => node.output_id(),
			Self::FlowLinearState { output_id, .. }
			| Self::FlowLinearVelocity { output_id, .. }
			| Self::FlowEulerStep { output_id, .. }
			| Self::FlowMaskedMse { output_id, .. }
			| Self::Linear { output_id, .. }
			| Self::Conv1d { output_id, .. }
			| Self::ConvTranspose1d { output_id, .. }
			| Self::ConvTranspose2d { output_id, .. }
			| Self::Conv2d { output_id, .. }
			| Self::LayerNorm { output_id, .. }
			| Self::BatchNorm2d { output_id, .. }
			| Self::RmsNorm { output_id, .. }
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
			| Self::Embedding { output_id, .. } => *output_id,
			Self::GruCell(node) => node.output_id,
			Self::GruScan(node) => node.output_id,
			Self::RnnCell(node) => node.output_id,
			Self::RnnScan(node) => node.output_id,
			Self::RmsNormGated(node) => node.output_id,
			Self::ChannelNorm(node) => node.output_id,
			Self::Mamba3Preprocess(node) => node.outputs[0].value_id(),
			Self::Mamba3Siso(node) => node.output_id,
			Self::Mamba3Mimo(node) => node.output_id,
		}
	}

	pub(super) fn produces(&self, value_id: u64) -> bool {
		match self {
			Self::Mamba3Preprocess(node) => node
				.outputs
				.iter()
				.any(|output| output.value_id() == value_id),
			_ => self.output_id() == value_id,
		}
	}
}

pub(super) struct GradNodeRmsNormGated {
	pub(super) input: Matrix,
	pub(super) weight: Option<Parameter>,
	pub(super) weight_value: Matrix,
	pub(super) weight_version: Option<u64>,
	pub(super) bias: Option<Parameter>,
	pub(super) bias_value: Option<Matrix>,
	pub(super) bias_version: Option<u64>,
	pub(super) gate: Matrix,
	pub(super) epsilon: f32,
	pub(super) output_id: u64,
}

pub(super) struct GradNodeMamba3Preprocess {
	pub(super) projected: Matrix,
	pub(super) dt_bias: Matrix,
	pub(super) outputs: [Matrix; 8],
	pub(super) config: crate::ml::matrix::Mamba3PreprocessConfig,
}

pub(super) struct GradNodeMamba3Siso {
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

pub(super) struct GradNodeMamba3Mimo {
	pub(super) inputs: [Matrix; 15],
	pub(super) config: crate::ml::matrix::SsmConfig,
	pub(super) output_id: u64,
}

pub(super) struct GradNodeRnnScan {
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

pub(super) struct GradNodeRnnCell {
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

pub(super) struct GradNodeGruScan {
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

pub(super) struct GradNodeGruCell {
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

impl From<crate::matrix::autograd::GradNodeMatrix> for GradNode {
	fn from(node: crate::matrix::autograd::GradNodeMatrix) -> Self {
		Self::Matrix(node)
	}
}
