use crate::{Matrix, Result, ml::Parameter};

use super::super::{node::Node, tape::record_node};

pub(in crate::ml) fn record_grouped_gemm_m(
	input: &Matrix,
	output: &Matrix,
	weight: Option<(Parameter, u64)>,
	weight_value: Matrix,
	offsets: &Matrix,
) -> Result<()> {
	let (weight, weight_version) = weight
		.map(|(parameter, version)| (Some(parameter), Some(version)))
		.unwrap_or((None, None));
	record_node(Node::GroupedGemmM {
		input: input.clone(),
		weight,
		weight_value,
		weight_version,
		offsets: offsets.clone(),
		output_id: output.value_id(),
	})
}

#[allow(
	clippy::too_many_arguments,
	reason = "the node retains exact grouped-Linear values and optional module parameters"
)]
pub(in crate::ml) fn record_grouped_linear_m(
	input: &Matrix,
	output: &Matrix,
	weight: Option<(Parameter, u64)>,
	weight_value: Matrix,
	bias: Option<(Parameter, u64)>,
	bias_value: Matrix,
	offsets: &Matrix,
) -> Result<()> {
	let (weight, weight_version) = weight
		.map(|(parameter, version)| (Some(parameter), Some(version)))
		.unwrap_or((None, None));
	let (bias, bias_version) = bias
		.map(|(parameter, version)| (Some(parameter), Some(version)))
		.unwrap_or((None, None));
	record_node(Node::GroupedLinearM {
		input: input.clone(),
		weight,
		weight_value,
		weight_version,
		bias,
		bias_value,
		bias_version,
		offsets: offsets.clone(),
		output_id: output.value_id(),
	})
}

pub(in crate::ml) fn record_silu_mul(
	input: &Matrix,
	intermediate_size: usize,
	output: &Matrix,
) -> Result<()> {
	record_node(Node::SiluMul {
		input: input.clone(),
		intermediate_size,
		output_id: output.value_id(),
	})
}

pub(in crate::ml) fn record_moe_route_weights(
	probabilities: &Matrix,
	expert_indices: &Matrix,
	output: &Matrix,
) -> Result<()> {
	record_node(Node::MoeRouteWeights {
		probabilities: probabilities.clone(),
		expert_indices: expert_indices.clone(),
		output: output.clone(),
		output_id: output.value_id(),
	})
}

pub(in crate::ml) fn record_moe_gather(
	input: &Matrix,
	inverse: &Matrix,
	output: &Matrix,
) -> Result<()> {
	record_node(Node::MoeGather {
		input: input.clone(),
		inverse: inverse.clone(),
		output_id: output.value_id(),
	})
}

pub(in crate::ml) fn record_moe_combine(
	packed: &Matrix,
	route_gate: &Matrix,
	inverse: &Matrix,
	packed_slot: &Matrix,
	output: &Matrix,
) -> Result<()> {
	record_node(Node::MoeCombine {
		packed: packed.clone(),
		route_gate: route_gate.clone(),
		inverse: inverse.clone(),
		packed_slot: packed_slot.clone(),
		output_id: output.value_id(),
	})
}
