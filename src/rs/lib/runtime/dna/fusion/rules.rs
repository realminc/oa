//! Ordered structural fusion rules and their executable input patterns.

use std::collections::{BTreeMap, BTreeSet};

use super::{
	census::PairOutcome,
	region::{RegionDecline, RegionGraph, WriteAdmission},
	*,
};
use crate::runtime::dna::tile;

type Specialize = fn(&Arc<Device>, &mut ComputeNode);

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

#[derive(Clone, Copy)]
enum RuleId {
	Pair(PairRule),
	Merge(MergeRule),
}

impl RuleId {
	fn name(self) -> &'static str {
		match self {
			Self::Pair(rule) => rule.name(),
			Self::Merge(rule) => rule.name(),
		}
	}

	fn record(self, device: &Device, index: usize, outcome: PairOutcome) {
		if crate::EnvFlag::is_set("OA_JIT_TRACE") {
			let (kind, name) = match self {
				Self::Pair(rule) => ("pair", format!("{rule:?}")),
				Self::Merge(rule) => ("merge", format!("{rule:?}")),
			};
			match outcome {
				PairOutcome::Applied => crate::log_info!(
					crate::LogComponent::COMPUTE,
					"fusion {kind} applied {name} at {index}"
				),
				PairOutcome::Declined(reason) => crate::log_info!(
					crate::LogComponent::COMPUTE,
					"fusion {kind} decline {name} at {index}: {reason:?}"
				),
				_ => {}
			}
		}
		match self {
			Self::Pair(rule) => device.fusion_census().pair(rule, outcome),
			Self::Merge(rule) => device.fusion_census().merge(rule, outcome),
		}
	}
}

#[derive(Clone, Copy)]
enum Gate {
	Always,
	Rl,
	SingleHead,
}

#[derive(Clone, Copy)]
enum Kernel {
	Exact(KernelId),
	OneOf(&'static [KernelId]),
	Activation,
	LinearActivation,
	Prologue,
}

impl Kernel {
	fn matches(self, kernel: KernelId) -> bool {
		match self {
			Self::Exact(expected) => kernel == expected,
			Self::OneOf(choices) => choices.contains(&kernel),
			Self::Activation => kernel.act_kind().is_some(),
			Self::LinearActivation => kernel.act_kind().is_some() || kernel == KernelId::MlSiluMulF32,
			Self::Prologue => kernel.act_kind().is_some() || kernel.prologue_kind().is_some(),
		}
	}
}

#[derive(Clone, Copy)]
pub(super) struct NodePattern {
	kernel: Kernel,
	bindings: &'static [BufferAccess],
	push_count: Option<usize>,
}

impl NodePattern {
	pub(super) fn matches(self, node: &ComputeNode) -> bool {
		self.kernel.matches(node.kernel)
			&& node.indirect.is_none()
			&& self
				.push_count
				.is_none_or(|count| node.push_constants.len() == count)
			&& node.buffers.len() == self.bindings.len()
			&& node
				.buffers
				.iter()
				.zip(self.bindings)
				.all(|(binding, &access)| binding.access == access)
	}
}

const fn node(
	kernel: Kernel,
	bindings: &'static [BufferAccess],
	push_count: Option<usize>,
) -> NodePattern {
	NodePattern {
		kernel,
		bindings,
		push_count,
	}
}

const fn exact(
	kernel: KernelId,
	bindings: &'static [BufferAccess],
	push_count: usize,
) -> NodePattern {
	node(Kernel::Exact(kernel), bindings, Some(push_count))
}

use BufferAccess::{Read as R, ReadWrite as RW, Write as W};
use KernelId::*;

pub(super) const ADD: NodePattern = exact(MatrixAddF32, &[R, R, W], 1);
const LINEAR: NodePattern = exact(MlLinearF32, &[R, R, R, W], 3);
pub(super) const LINEAR_BACKWARD: NodePattern = exact(MlLinearBackwardF32, &[R, R, W], 3);
const NORM: NodePattern = exact(MlLayerNormF32, &[R, R, R, W, W, W], 3);
pub(super) const NORM_BACKWARD: NodePattern = exact(MlLayerNormBackwardF32, &[R, R, R, R, W, W], 2);
const SUM_PAIR: NodePattern = exact(MlSumAxisPairF32, &[R, R, W, W], 3);
const EMBEDDING: NodePattern = node(
	Kernel::OneOf(&[MlEmbeddingF32, MlEmbeddingU8F32]),
	&[R, R, W],
	Some(3),
);
const CONCAT: NodePattern = exact(MatrixConcatF32, &[R, W], 18);
const SPLIT_HEADS: NodePattern = exact(MlSplitHeadsF32, &[R, W], 4);
const SCORE: NodePattern = node(
	Kernel::OneOf(&[MlBmmNtF32, MlBmmNtTiled16F32]),
	&[R, R, W],
	Some(4),
);
const PROBABILITY: NodePattern = node(
	Kernel::OneOf(&[MlSdpaSoftmaxN32F32, MlSdpaSoftmaxF32]),
	&[R, R, W],
	Some(5),
);
const CONTEXT: NodePattern = node(
	Kernel::OneOf(&[MlBmmF32, MlBmmTiled16F32]),
	&[R, R, W],
	Some(4),
);
const SOFTMAX: NodePattern = exact(MatrixSoftmaxF32, &[R, W], 3);
const TOP_K: NodePattern = exact(MatrixTopKF32, &[R, W, W], 3);
const TOP_K_MASK: NodePattern = exact(MatrixTopKMaskF32, &[R, W], 3);
pub(super) const RELU: NodePattern = exact(MlReluF32, &[R, W], 1);
pub(super) const EMBEDDING_BACKWARD: NodePattern = exact(MlEmbeddingBackwardF32, &[R, R, W], 3);
pub(super) const CHANNEL_NORM: NodePattern = node(
	Kernel::OneOf(&[MlChannelNormF32, MlChannelNormReluF32]),
	&[R, R, R, W],
	Some(4),
);
pub(super) const TRANSPOSE: NodePattern = exact(MatrixTransposeF32, &[R, W], 3);
const ACT: NodePattern = node(Kernel::Activation, &[R, W], None);
const PROLOGUE: NodePattern = node(Kernel::Prologue, &[R, R, W], None);

/// Relative node and binding positions in the rule's source window.
type Slot = (usize, usize);

#[derive(Clone, Copy)]
enum Constraint {
	Wire(Slot, Slot),
	Disjoint(Slot, Slot),
	OneInput(Slot, usize),
	InputPair(Slot, Slot, usize),
	EqualWord(Slot, Slot),
	EqualPush(usize, usize),
	EqualGrid(usize, usize),
	CountProduct(Slot, usize, &'static [usize]),
}

impl Constraint {
	fn matches(self, nodes: &[ComputeNode]) -> bool {
		let buffer = |(node, slot): Slot| &nodes[node].buffers[slot].buffer;
		match self {
			Self::Wire(from, to) => buffer(from).same_as(buffer(to)),
			Self::Disjoint(a, b) => !buffer(a).same_as(buffer(b)),
			Self::OneInput(from, consumer) => {
				buffer(from).same_as(buffer((consumer, 0))) != buffer(from).same_as(buffer((consumer, 1)))
			}
			Self::InputPair(a, b, consumer) => {
				let (left, right) = (buffer((consumer, 0)), buffer((consumer, 1)));
				(buffer(a).same_as(left) && buffer(b).same_as(right))
					|| (buffer(b).same_as(left) && buffer(a).same_as(right))
			}
			Self::EqualWord((a, word_a), (b, word_b)) => {
				nodes[a].push_constants[word_a] == nodes[b].push_constants[word_b]
			}
			Self::EqualPush(a, b) => nodes[a].push_constants == nodes[b].push_constants,
			Self::EqualGrid(a, b) => nodes[a].workgroups == nodes[b].workgroups,
			Self::CountProduct((target, word), source, factors) => factors
				.iter()
				.try_fold(1u32, |product, &index| {
					match nodes[source].push_constants[index] {
						PushConstant::U32(value) => product.checked_mul(value),
						_ => None,
					}
				})
				.is_some_and(|product| nodes[target].push_constants[word] == PushConstant::U32(product)),
		}
	}
}

#[derive(Clone, Copy)]
enum Admission {
	Default,
	// The fused single-head provider writes identical Q/K/V bytes at these bindings.
	Identity(&'static [usize]),
	// Rank-two concat parts cover disjoint extents of the linear input.
	Partitioned(Slot),
}

impl Admission {
	fn resolve(self, nodes: &[ComputeNode], replacement: &ComputeNode) -> WriteAdmission {
		match self {
			Self::Default => WriteAdmission::default(),
			Self::Identity(bindings) => WriteAdmission::identity(
				bindings
					.iter()
					.map(|&slot| replacement.buffers[slot].buffer.descriptor_index()),
			),
			Self::Partitioned((node, slot)) => WriteAdmission::partitioned(std::iter::once(
				nodes[node].buffers[slot].buffer.descriptor_index(),
			)),
		}
	}
}

enum Rewrite {
	Contiguous {
		count: usize,
		primary: ComputeNode,
		sibling: Option<Box<ComputeNode>>,
	},
	Scattered {
		/// Source offsets in their semantic ownership order, including the anchor.
		sources: Vec<usize>,
		/// Replacement offset and node, in emission order.
		replacements: Vec<(usize, ComputeNode)>,
	},
	Reordered {
		primary: ComputeNode,
		count: usize,
		moved: usize,
	},
}

impl Rewrite {
	fn single(count: usize, primary: ComputeNode) -> Self {
		Self::Contiguous {
			count,
			primary,
			sibling: None,
		}
	}

	fn pair(count: usize, (primary, sibling): (ComputeNode, ComputeNode)) -> Self {
		Self::Contiguous {
			count,
			primary,
			sibling: Some(Box::new(sibling)),
		}
	}
}

struct Rule {
	id: RuleId,
	pattern: &'static [NodePattern],
	constraints: &'static [Constraint],
	providers: &'static [KernelId],
	gate: Gate,
	admission: Admission,
	specialize: &'static [Specialize],
	lower: fn(&[ComputeNode]) -> Option<Rewrite>,
}

impl Rule {
	fn enabled(&self, device: &Device, rl: bool, single_head: bool) -> bool {
		let route = match self.gate {
			Gate::Always => true,
			Gate::Rl => rl,
			Gate::SingleHead => rl && single_head,
		};
		route
			&& !fusion_rule_disabled(self.id.name())
			&& self
				.providers
				.iter()
				.all(|&kernel| device.require_kernel(kernel).is_ok())
	}

	fn propose(&self, nodes: &[ComputeNode]) -> Option<Rewrite> {
		let prefix = nodes.get(..self.pattern.len())?;
		if !self
			.pattern
			.iter()
			.zip(prefix)
			.all(|(&pattern, node)| pattern.matches(node))
			|| !self
				.constraints
				.iter()
				.all(|constraint| constraint.matches(prefix))
		{
			return None;
		}
		(self.lower)(nodes)
	}

	fn specialize(&self, device: &Arc<Device>, node: &mut ComputeNode) {
		for specialize in self.specialize {
			specialize(device, node);
		}
	}
}

use Constraint::{
	CountProduct, Disjoint, EqualGrid, EqualPush, EqualWord, InputPair, OneInput, Wire,
};
use Gate::{Always, Rl, SingleHead};
use Rewrite::{Reordered, Scattered};

// Table order preserves first-match priority, including the longer concat and
// reordered router alternatives. The lowerers pack the existing shader ABI.
const RULES: &[Rule] = &[
	Rule {
		id: RuleId::Merge(MergeRule::LinearBackwardBmmChain),
		pattern: &[node(
			Kernel::OneOf(&[MlBmmTiled16F32, MlBmmTnTiled16F32]),
			&[R, R, W],
			Some(4),
		)],
		constraints: &[],
		providers: &[MlLinearBackwardBmmChainF32],
		gate: Always,
		admission: Admission::Default,
		specialize: &[tile::specialize_linear_backward_node],
		lower: |nodes| {
			let limit = nodes.len().min(MAX_BMM_CHAIN_LOOKAHEAD + 1);
			let (primary, distance) = lower_linear_backward_bmm_chain(&nodes[..limit])?;
			Some(Scattered {
				sources: vec![0, distance],
				replacements: vec![(0, primary)],
			})
		},
	},
	Rule {
		id: RuleId::Merge(MergeRule::SmallCausalSdpa),
		pattern: &[
			SPLIT_HEADS,
			SPLIT_HEADS,
			SPLIT_HEADS,
			SCORE,
			PROBABILITY,
			CONTEXT,
			exact(MlMergeHeadsF32, &[R, W], 4),
		],
		constraints: &[
			EqualPush(0, 1),
			EqualPush(0, 2),
			EqualPush(0, 6),
			Wire((0, 1), (3, 0)),
			Wire((1, 1), (3, 1)),
			Wire((3, 2), (4, 0)),
			Wire((3, 2), (4, 1)),
			Wire((4, 2), (5, 0)),
			Wire((2, 1), (5, 1)),
			Wire((5, 2), (6, 0)),
		],
		providers: &[MlSmallCausalSdpaF32],
		gate: Rl,
		admission: Admission::Default,
		specialize: &[tile::specialize_attention_node],
		lower: |nodes| Some(Rewrite::single(7, lower_small_causal_sdpa(&nodes[..7])?)),
	},
	Rule {
		id: RuleId::Merge(MergeRule::SingleHeadSdpa),
		pattern: &[SCORE, PROBABILITY, CONTEXT],
		constraints: &[
			Wire((0, 2), (1, 0)),
			Wire((0, 2), (1, 1)),
			Wire((1, 2), (2, 0)),
		],
		providers: &[MlSmallCausalSdpaF32],
		gate: SingleHead,
		admission: Admission::Identity(&[3, 4, 5]),
		specialize: &[tile::specialize_attention_node],
		lower: |nodes| Some(Rewrite::single(3, lower_single_head_sdpa(&nodes[..3])?)),
	},
	Rule {
		id: RuleId::Merge(MergeRule::ConcatLinear),
		pattern: &[CONCAT, CONCAT, CONCAT, LINEAR],
		constraints: &[
			Wire((0, 1), (3, 0)),
			Wire((1, 1), (3, 0)),
			Wire((2, 1), (3, 0)),
		],
		providers: &[MlLinearConcatF32],
		gate: Rl,
		admission: Admission::Partitioned((3, 0)),
		specialize: &[tile::specialize_linear_concat_node],
		lower: |nodes| {
			Some(Rewrite::single(
				4,
				lower_concat_linear(&nodes[..3], &nodes[3])?,
			))
		},
	},
	Rule {
		id: RuleId::Merge(MergeRule::ConcatLinear),
		pattern: &[CONCAT, CONCAT, LINEAR],
		constraints: &[Wire((0, 1), (2, 0)), Wire((1, 1), (2, 0))],
		providers: &[MlLinearConcatF32],
		gate: Rl,
		admission: Admission::Partitioned((2, 0)),
		specialize: &[tile::specialize_linear_concat_node],
		lower: |nodes| {
			Some(Rewrite::single(
				3,
				lower_concat_linear(&nodes[..2], &nodes[2])?,
			))
		},
	},
	Rule {
		id: RuleId::Merge(MergeRule::RouterSelect),
		pattern: &[
			SOFTMAX,
			node(
				Kernel::OneOf(&[MatrixAddF32, MatrixAddBroadcastF32]),
				&[R, R, W],
				None,
			),
			TOP_K,
			TOP_K_MASK,
		],
		constraints: &[
			Wire((1, 2), (2, 0)),
			Wire((2, 2), (3, 0)),
			EqualPush(2, 3),
			Disjoint((1, 2), (1, 0)),
			Disjoint((1, 2), (1, 1)),
		],
		providers: &[MlMoeRouterSelectF32],
		gate: Rl,
		admission: Admission::Default,
		specialize: &[tile::specialize_router_select_node],
		lower: |nodes| {
			Some(Reordered {
				primary: lower_router_select(&nodes[0], &nodes[2], &nodes[3])?,
				count: 4,
				moved: 1,
			})
		},
	},
	Rule {
		id: RuleId::Merge(MergeRule::RouterSelect),
		pattern: &[SOFTMAX, TOP_K, TOP_K_MASK],
		constraints: &[Wire((1, 2), (2, 0)), EqualPush(1, 2)],
		providers: &[MlMoeRouterSelectF32],
		gate: Rl,
		admission: Admission::Default,
		specialize: &[tile::specialize_router_select_node],
		lower: |nodes| {
			Some(Rewrite::single(
				3,
				lower_router_select(&nodes[0], &nodes[1], &nodes[2])?,
			))
		},
	},
	Rule {
		id: RuleId::Merge(MergeRule::LinearTriple),
		pattern: &[LINEAR; 3],
		constraints: &[
			Wire((0, 0), (1, 0)),
			Wire((0, 0), (2, 0)),
			EqualPush(0, 1),
			EqualPush(0, 2),
			EqualGrid(0, 1),
			EqualGrid(0, 2),
		],
		providers: &[MlLinearTripleF32],
		gate: Always,
		admission: Admission::Default,
		specialize: &[tile::specialize_linear_triple_node],
		lower: |nodes| {
			Some(Rewrite::single(
				3,
				lower_linear_triple(&nodes[0], &nodes[1], &nodes[2])?,
			))
		},
	},
	Rule {
		id: RuleId::Merge(MergeRule::EmbeddingPairNorm),
		pattern: &[EMBEDDING, EMBEDDING, ADD, NORM],
		constraints: &[
			InputPair((0, 2), (1, 2), 2),
			EqualWord((0, 0), (1, 0)),
			EqualWord((0, 2), (1, 2)),
			Wire((2, 2), (3, 0)),
		],
		providers: &[MlEmbeddingPairNormF32],
		gate: Always,
		admission: Admission::Default,
		specialize: &[],
		lower: |nodes| {
			Some(Rewrite::single(
				4,
				lower_embedding_pair_norm(&nodes[0], &nodes[1], &nodes[2], &nodes[3])?,
			))
		},
	},
	Rule {
		id: RuleId::Merge(MergeRule::EmbeddingPair),
		pattern: &[EMBEDDING, EMBEDDING, ADD],
		constraints: &[
			InputPair((0, 2), (1, 2), 2),
			EqualWord((0, 0), (1, 0)),
			EqualWord((0, 2), (1, 2)),
			CountProduct((2, 0), 0, &[0, 2]),
		],
		providers: &[MlEmbeddingPairF32],
		gate: Always,
		admission: Admission::Default,
		specialize: &[],
		lower: |nodes| {
			Some(Rewrite::single(
				3,
				lower_embedding_pair(&nodes[0], &nodes[1], &nodes[2])?,
			))
		},
	},
	Rule {
		id: RuleId::Merge(MergeRule::LinearBackwardNormAdd),
		pattern: &[NORM_BACKWARD, SUM_PAIR, ADD, LINEAR_BACKWARD],
		constraints: &[
			Wire((0, 5), (1, 0)),
			Wire((0, 3), (1, 1)),
			Wire((2, 2), (3, 1)),
			OneInput((0, 4), 2),
		],
		providers: &[MlLinearBackwardNormAddF32],
		gate: Always,
		admission: Admission::Default,
		specialize: &[tile::specialize_linear_backward_node],
		lower: |nodes| {
			Some(Rewrite::pair(
				4,
				lower_linear_backward_norm_add(&nodes[0], &nodes[1], &nodes[2], &nodes[3])?,
			))
		},
	},
	Rule {
		id: RuleId::Merge(MergeRule::LinearBackwardNorm),
		pattern: &[NORM_BACKWARD, SUM_PAIR, LINEAR_BACKWARD],
		constraints: &[
			Wire((0, 5), (1, 0)),
			Wire((0, 3), (1, 1)),
			Wire((0, 4), (2, 1)),
		],
		providers: &[MlLinearBackwardNormF32],
		gate: Always,
		admission: Admission::Default,
		specialize: &[tile::specialize_linear_backward_node],
		lower: |nodes| {
			Some(Rewrite::pair(
				3,
				lower_linear_backward_norm(&nodes[0], &nodes[1], &nodes[2])?,
			))
		},
	},
	Rule {
		id: RuleId::Merge(MergeRule::LayerNormBackwardSum),
		pattern: &[ADD],
		constraints: &[],
		providers: &[MlLayerNormBackwardSumF32],
		gate: Always,
		admission: Admission::Default,
		specialize: &[tile::specialize_layer_norm_backward_sum_node],
		lower: |nodes| {
			let (primary, absorbed, emit) = lower_layer_norm_backward_sum(nodes)?;
			Some(Scattered {
				sources: std::iter::once(0).chain(absorbed).collect(),
				replacements: vec![(emit, primary)],
			})
		},
	},
	Rule {
		id: RuleId::Merge(MergeRule::EmbeddingBackwardSum),
		pattern: &[ADD],
		constraints: &[],
		providers: &[MlEmbeddingBackwardSumF32],
		gate: Always,
		admission: Admission::Default,
		specialize: &[tile::specialize_embedding_backward_sum_node],
		lower: |nodes| {
			let replacements = lower_embedding_backward_sum(nodes)?;
			Some(Scattered {
				sources: std::iter::once(0)
					.chain(replacements.iter().map(|&(slot, _)| slot))
					.collect(),
				replacements,
			})
		},
	},
	Rule {
		id: RuleId::Merge(MergeRule::Conv1dBiasTail),
		pattern: &[
			exact(MlConv1dBiasAddF32, &[RW, R], 2),
			exact(MlConv1dLayoutTransposeF32, &[R, W], 3),
		],
		constraints: &[
			Wire((0, 0), (1, 0)),
			CountProduct((0, 0), 1, &[0, 1]),
			EqualWord((0, 1), (1, 2)),
		],
		providers: &[
			MlConv1dBiasLayoutTransposeF32,
			MlConv1dBiasLayoutTransposeReluF32,
			MlConv1dBiasLayoutTransposeResidualReluF32,
			MlConv1dBiasLayoutTransposeNormF32,
			MlConv1dBiasLayoutTransposeNormReluF32,
			MlConv1dBiasCopyF32,
		],
		gate: Always,
		admission: Admission::Default,
		specialize: &[tile::specialize_conv_node],
		lower: |nodes| {
			let (primary, count) = lower_conv_1d_bias_tail(nodes)?;
			Some(Rewrite::single(count, primary))
		},
	},
	Rule {
		id: RuleId::Merge(MergeRule::ConvRelu),
		pattern: &[exact(MlConv2dF32, &[R, R, R, W], 11), RELU],
		constraints: &[Wire((0, 3), (1, 0)), CountProduct((1, 0), 0, &[0, 2, 8, 9])],
		providers: &[MlConv2dReluF32],
		gate: Always,
		admission: Admission::Default,
		specialize: &[tile::specialize_conv_node],
		lower: |nodes| Some(Rewrite::single(2, lower_conv_2d_relu(nodes)?)),
	},
	Rule {
		id: RuleId::Merge(MergeRule::MaxPoolReluBackward),
		pattern: &[
			exact(MlMaxPool2dBackwardF32, &[R, R, W], 9),
			exact(MlReluBackwardF32, &[R, R, W], 1),
		],
		constraints: &[Wire((0, 2), (1, 1)), CountProduct((1, 0), 0, &[0, 1, 2, 3])],
		providers: &[MlMaxPool2dReluBackwardF32],
		gate: Always,
		admission: Admission::Default,
		specialize: &[tile::specialize_pool_node],
		lower: |nodes| Some(Rewrite::single(2, lower_max_pool_relu_backward(nodes)?)),
	},
	Rule {
		id: RuleId::Merge(MergeRule::LinearResidualNorm),
		pattern: &[LINEAR, ADD, NORM],
		constraints: &[OneInput((0, 3), 1), Wire((1, 2), (2, 0))],
		providers: &[MlLinearResidualNormF32],
		gate: Always,
		admission: Admission::Default,
		specialize: &[tile::specialize_linear_residual_norm_node],
		lower: |nodes| {
			Some(Rewrite::single(
				3,
				lower_linear_residual_norm(&nodes[0], &nodes[1], &nodes[2])?,
			))
		},
	},
	Rule {
		id: RuleId::Merge(MergeRule::LinearResidualAct),
		pattern: &[LINEAR, ADD, ACT],
		constraints: &[
			OneInput((0, 3), 1),
			Wire((1, 2), (2, 0)),
			CountProduct((1, 0), 0, &[0, 2]),
		],
		providers: &[MlLinearResidualActF32],
		gate: Always,
		admission: Admission::Default,
		specialize: &[tile::specialize_fused_linear_node],
		lower: |nodes| {
			Some(Rewrite::single(
				3,
				lower_linear_residual_act(&nodes[0], &nodes[1], &nodes[2])?,
			))
		},
	},
	Rule {
		id: RuleId::Merge(MergeRule::GroupedLinearAct),
		pattern: &[
			exact(MlGroupedLinearMF32, &[R, R, R, R, W], 4),
			exact(MlSiluMulF32, &[R, W], 2),
		],
		constraints: &[Wire((0, 4), (1, 0))],
		providers: &[MlGroupedLinearMActF32],
		gate: Rl,
		admission: Admission::Default,
		specialize: &[tile::specialize_grouped_linear_act_node],
		lower: |nodes| {
			Some(Rewrite::single(
				2,
				lower_grouped_linear_act(&nodes[0], &nodes[1])?,
			))
		},
	},
	Rule {
		id: RuleId::Pair(PairRule::LinearResidual),
		pattern: &[LINEAR, ADD],
		constraints: &[OneInput((0, 3), 1), CountProduct((1, 0), 0, &[0, 2])],
		providers: &[MlLinearResidualF32],
		gate: Always,
		admission: Admission::Default,
		specialize: &[tile::specialize_fused_linear_node],
		lower: |nodes| {
			Some(Rewrite::single(
				2,
				lower_linear_residual(&nodes[0], &nodes[1]),
			))
		},
	},
	Rule {
		id: RuleId::Pair(PairRule::LinearAct),
		pattern: &[LINEAR, node(Kernel::LinearActivation, &[R, W], None)],
		constraints: &[Wire((0, 3), (1, 0))],
		providers: &[MlLinearActF32, MlLinearGatedActF32],
		gate: Always,
		admission: Admission::Default,
		specialize: &[tile::specialize_fused_linear_node],
		lower: |nodes| Some(Rewrite::single(2, lower_linear_act(&nodes[0], &nodes[1])?)),
	},
	Rule {
		id: RuleId::Pair(PairRule::ResidualNorm),
		pattern: &[ADD, NORM],
		constraints: &[Wire((0, 2), (1, 0)), CountProduct((0, 0), 1, &[0, 1])],
		providers: &[MlResidualLayerNormF32],
		gate: Always,
		admission: Admission::Default,
		specialize: &[],
		lower: |nodes| {
			Some(Rewrite::single(
				2,
				lower_add_layer_norm(&nodes[0], &nodes[1]),
			))
		},
	},
	Rule {
		id: RuleId::Pair(PairRule::SumAxisScale),
		pattern: &[
			exact(MatrixSumAxisF32, &[R, W], 3),
			exact(MatrixScaleF32, &[R, W], 2),
		],
		constraints: &[Wire((0, 1), (1, 0)), CountProduct((1, 0), 0, &[0, 2])],
		providers: &[MlSumAxisScaleF32],
		gate: Always,
		admission: Admission::Default,
		specialize: &[],
		lower: |nodes| {
			Some(Rewrite::single(
				2,
				lower_sum_axis_scale(&nodes[0], &nodes[1])?,
			))
		},
	},
	Rule {
		id: RuleId::Pair(PairRule::LinearBackwardPrologue),
		pattern: &[PROLOGUE, LINEAR_BACKWARD],
		constraints: &[Wire((0, 2), (1, 1))],
		providers: &[MlLinearBackwardPrologueF32],
		gate: Always,
		admission: Admission::Default,
		specialize: &[tile::specialize_linear_backward_node],
		lower: |nodes| {
			Some(Rewrite::single(
				2,
				lower_linear_backward_prologue(&nodes[0], &nodes[1])?,
			))
		},
	},
	Rule {
		id: RuleId::Pair(PairRule::GroupedBackwardPrologue),
		pattern: &[
			PROLOGUE,
			node(
				Kernel::OneOf(&[
					MlGroupedLinearMDataBackwardF32,
					MlGroupedGemmMDataBackwardF32,
				]),
				&[R, R, R, W],
				Some(4),
			),
		],
		constraints: &[Wire((0, 2), (1, 0))],
		providers: &[MlGroupedBackwardPrologueF32],
		gate: Rl,
		admission: Admission::Default,
		specialize: &[tile::specialize_grouped_backward_prologue_node],
		lower: |nodes| {
			Some(Rewrite::single(
				2,
				lower_grouped_backward_prologue(&nodes[0], &nodes[1])?,
			))
		},
	},
];

pub(super) fn fuse(
	device: &Arc<Device>,
	nodes: Vec<ComputeNode>,
	refs: &[&Storage],
) -> Vec<ComputeNode> {
	let rl = std::env::var_os("OA_DISABLE_RL_FUSION").is_none();
	let single_head = single_head_sdpa_fusion_enabled(device);
	let enabled: Vec<_> = RULES
		.iter()
		.filter(|rule| rule.enabled(device, rl, single_head))
		.collect();
	if enabled.is_empty() {
		return nodes;
	}

	let graph = std::cell::OnceCell::new();
	let mut consumed = BTreeSet::new();
	let mut deferred = BTreeMap::new();
	let mut result = Vec::with_capacity(nodes.len());
	let mut regions = 0;
	let mut index = 0;
	while index < nodes.len() {
		if let Some(node) = deferred.remove(&index) {
			result.push(node);
		}
		if consumed.contains(&index) {
			index += 1;
			continue;
		}
		let candidate = enabled.iter().find_map(|rule| {
			let rewrite = rule.propose(&nodes[index..])?;
			let graph = graph.get_or_init(|| RegionGraph::from_compute_executed(device, &nodes, refs));
			let outcome = prove(graph, &nodes, &consumed, index, &rewrite, rule.admission);
			rule.id.record(device, index, outcome);
			matches!(outcome, PairOutcome::Applied).then_some((*rule, rewrite))
		});
		let Some((rule, rewrite)) = candidate else {
			result.push(nodes[index].clone());
			index += 1;
			continue;
		};
		match rewrite {
			Rewrite::Contiguous {
				count,
				mut primary,
				sibling,
			} => {
				rule.specialize(device, &mut primary);
				result.push(primary);
				result.extend(sibling.map(|node| *node));
				regions += 1;
				index += count;
			}
			Scattered {
				sources,
				replacements,
			} => {
				consumed.extend(sources.into_iter().skip(1).map(|offset| index + offset));
				regions += replacements.len();
				for (offset, mut replacement) in replacements {
					rule.specialize(device, &mut replacement);
					if offset == 0 {
						result.push(replacement);
					} else {
						deferred.insert(index + offset, replacement);
					}
				}
				index += 1;
			}
			Reordered {
				mut primary,
				count,
				moved,
			} => {
				rule.specialize(device, &mut primary);
				result.push(nodes[index + moved].clone());
				result.push(primary);
				regions += 1;
				index += count;
			}
		}
	}
	device
		.fusion_census()
		.structural(regions, nodes.len() - result.len() + regions);
	result
}

fn prove(
	graph: &RegionGraph,
	nodes: &[ComputeNode],
	consumed: &BTreeSet<usize>,
	index: usize,
	rewrite: &Rewrite,
	admission: Admission,
) -> PairOutcome {
	let proof: Result<(), RegionDecline> = match rewrite {
		Rewrite::Contiguous {
			count,
			primary,
			sibling,
		} => {
			let admission = admission.resolve(&nodes[index..], primary);
			let range = index..index + count;
			graph
				.contiguous(range.clone(), consumed, &admission)
				.and_then(|candidate| match sibling {
					Some(sibling) => candidate.accepts_many(&nodes[range], &[primary, sibling], &admission),
					None => candidate.accepts_compute(&nodes[range], primary, &admission),
				})
		}
		Scattered {
			sources,
			replacements,
		} => {
			let positions: BTreeSet<_> = sources.iter().map(|offset| index + offset).collect();
			let sources = sources.iter().map(|offset| &nodes[index + offset]);
			let replacements: Vec<_> = replacements.iter().map(|(_, node)| node).collect();
			if !slots_free(consumed, positions.iter().copied().skip(1)) {
				return PairOutcome::LegacyGuard;
			}
			let admission = WriteAdmission::default();
			graph
				.window(&positions, consumed, &admission, &BTreeSet::new())
				.and_then(|candidate| match replacements.as_slice() {
					[primary] => candidate.accepts_compute(sources, primary, &admission),
					_ => candidate.accepts_many(sources, &replacements, &admission),
				})
		}
		Reordered {
			primary,
			count,
			moved,
		} => {
			if !slots_free(consumed, index + 1..index + count) {
				return PairOutcome::LegacyGuard;
			}
			let admission = WriteAdmission::default();
			let moved = index + moved;
			let positions = (index..index + count)
				.filter(|&slot| slot != moved)
				.collect();
			let moved = BTreeSet::from([moved]);
			graph
				.window(&positions, consumed, &admission, &moved)
				.and_then(|candidate| {
					candidate.accepts_compute(
						positions.iter().map(|&slot| &nodes[slot]),
						primary,
						&admission,
					)
				})
		}
	};
	match proof {
		Ok(()) => PairOutcome::Applied,
		Err(reason) => PairOutcome::Declined(reason),
	}
}
