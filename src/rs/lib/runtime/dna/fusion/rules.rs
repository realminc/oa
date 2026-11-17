//! Stable identities for the first incumbent rule family.
//!
//! Donor/adaptation: the six ordered pair matchers in `fusion.rs`, unchanged
//! numerical lowering. This is an identity/dispatch table, not the future
//! declarative pattern, admission-domain or tile-template registry.

use super::ComputeNode;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(usize)]
pub(super) enum PairRule {
	LinearResidual,
	LinearAct,
	ResidualNorm,
	SumAxisScale,
	LinearBackwardPrologue,
	GroupedBackwardPrologue,
}

impl PairRule {
	/// Original first-match priority, shared by dispatch and diagnostics.
	pub(super) const ALL: [Self; 6] = [
		Self::LinearResidual,
		Self::LinearAct,
		Self::ResidualNorm,
		Self::SumAxisScale,
		Self::LinearBackwardPrologue,
		Self::GroupedBackwardPrologue,
	];

	pub(super) const fn name(self) -> &'static str {
		match self {
			Self::LinearResidual => "linear_residual",
			Self::LinearAct => "linear_act",
			Self::ResidualNorm => "residual_norm",
			Self::SumAxisScale => "sum_axis_scale",
			Self::LinearBackwardPrologue => "linear_backward_prologue",
			Self::GroupedBackwardPrologue => "grouped_backward_prologue",
		}
	}

	pub(super) fn lower(self, first: &ComputeNode, second: &ComputeNode) -> Option<ComputeNode> {
		match self {
			Self::LinearResidual => super::match_linear_residual(first, second),
			Self::LinearAct => super::match_linear_act(first, second),
			Self::ResidualNorm => super::match_add_layer_norm(first, second),
			Self::SumAxisScale => super::match_sum_axis_scale(first, second),
			Self::LinearBackwardPrologue => super::match_linear_backward_prologue(first, second),
			Self::GroupedBackwardPrologue => super::match_grouped_backward_prologue(first, second),
		}
	}
}

/// Stable identities for the second incumbent rule family: contiguous
/// merges over windows of two or more nodes. Single-emit rules replace the
/// window with one fused node; `accepts_many` rules emit a fused node plus a
/// sibling (e.g. a separate bias-grad reduction) against the same window.
/// `window` rules absorb non-adjacent positions under lookahead, including
/// the router reorder where a surviving passthrough node moves before the
/// fused emission. `Conv1dBiasTail` is the parameterized observability
/// matcher; its window count is resolved by the matcher before the proof.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(usize)]
pub(super) enum MergeRule {
	SmallCausalSdpa,
	SingleHeadSdpa,
	ConcatLinear,
	LinearTriple,
	EmbeddingPairNorm,
	EmbeddingPair,
	ConvRelu,
	MaxPoolReluBackward,
	LinearResidualNorm,
	LinearResidualAct,
	GroupedLinearAct,
	LinearBackwardNormAdd,
	LinearBackwardNorm,
	LinearBackwardBmmChain,
	LayerNormBackwardSum,
	EmbeddingBackwardSum,
	RouterSelect,
	Conv1dBiasTail,
}

impl MergeRule {
	pub(super) const COUNT: usize = 18;

	pub(super) const ALL: [Self; Self::COUNT] = [
		Self::SmallCausalSdpa,
		Self::SingleHeadSdpa,
		Self::ConcatLinear,
		Self::LinearTriple,
		Self::EmbeddingPairNorm,
		Self::EmbeddingPair,
		Self::ConvRelu,
		Self::MaxPoolReluBackward,
		Self::LinearResidualNorm,
		Self::LinearResidualAct,
		Self::GroupedLinearAct,
		Self::LinearBackwardNormAdd,
		Self::LinearBackwardNorm,
		Self::LinearBackwardBmmChain,
		Self::LayerNormBackwardSum,
		Self::EmbeddingBackwardSum,
		Self::RouterSelect,
		Self::Conv1dBiasTail,
	];

	pub(super) const fn name(self) -> &'static str {
		match self {
			Self::SmallCausalSdpa => "small_causal_sdpa",
			Self::SingleHeadSdpa => "single_head_sdpa",
			Self::ConcatLinear => "concat_linear",
			Self::LinearTriple => "linear_triple",
			Self::EmbeddingPairNorm => "embedding_pair_norm",
			Self::EmbeddingPair => "embedding_pair",
			Self::ConvRelu => "conv_relu",
			Self::MaxPoolReluBackward => "max_pool_relu_backward",
			Self::LinearResidualNorm => "linear_residual_norm",
			Self::LinearResidualAct => "linear_residual_act",
			Self::GroupedLinearAct => "grouped_linear_act",
			Self::LinearBackwardNormAdd => "linear_backward_norm_add",
			Self::LinearBackwardNorm => "linear_backward_norm",
			Self::LinearBackwardBmmChain => "linear_backward_bmm_chain",
			Self::LayerNormBackwardSum => "layer_norm_backward_sum",
			Self::EmbeddingBackwardSum => "embedding_backward_sum",
			Self::RouterSelect => "router_select",
			Self::Conv1dBiasTail => "conv_1d_bias_tail",
		}
	}
}
