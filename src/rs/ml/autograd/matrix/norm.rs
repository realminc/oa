use crate::ml::Parameter;
use crate::{Matrix, Result};

use super::super::{node::Node, tape::record_node};

#[allow(
	clippy::too_many_arguments,
	reason = "the node retains affine values, optional parameters, and fused activation state"
)]
pub(in crate::ml) fn record_channel_norm(
	input: &Matrix,
	output: &Matrix,
	weight: Option<(Parameter, u64)>,
	weight_value: Matrix,
	bias: Option<(Parameter, u64)>,
	bias_value: Matrix,
	epsilon: f32,
	relu: bool,
) -> Result<()> {
	let (weight, weight_version) = weight
		.map(|(parameter, version)| (Some(parameter), Some(version)))
		.unwrap_or((None, None));
	let (bias, bias_version) = bias
		.map(|(parameter, version)| (Some(parameter), Some(version)))
		.unwrap_or((None, None));
	record_node(Node::ChannelNorm(Box::new(
		super::super::node::ChannelNormNode {
			input: input.clone(),
			output: relu.then(|| output.clone()),
			output_id: output.value_id(),
			weight,
			weight_value,
			weight_version,
			bias,
			bias_value,
			bias_version,
			epsilon,
		},
	)))
}

#[allow(
	clippy::too_many_arguments,
	reason = "the node retains exact BatchNorm state and optional module parameters"
)]
pub(in crate::ml) fn record_batch_norm_2d(
	input: &Matrix,
	output: &Matrix,
	mean: Matrix,
	variance: Matrix,
	weight: Option<(Parameter, u64)>,
	weight_value: Matrix,
	bias: Option<(Parameter, u64)>,
	bias_value: Matrix,
	epsilon: f32,
	training: bool,
) -> Result<()> {
	let (weight, weight_version) = weight
		.map(|(parameter, version)| (Some(parameter), Some(version)))
		.unwrap_or((None, None));
	let (bias, bias_version) = bias
		.map(|(parameter, version)| (Some(parameter), Some(version)))
		.unwrap_or((None, None));
	record_node(Node::BatchNorm2d {
		input: input.clone(),
		mean,
		variance,
		output_id: output.value_id(),
		weight,
		weight_value,
		weight_version,
		bias,
		bias_value,
		bias_version,
		epsilon,
		training,
	})
}

#[allow(clippy::too_many_arguments)]
pub(in crate::ml) fn record_layer_norm(
	input: &Matrix,
	output: &Matrix,
	normalized: Matrix,
	inverse_stddev: Matrix,
	weight: Parameter,
	weight_value: Matrix,
	weight_version: u64,
	bias: Parameter,
	bias_version: u64,
) -> Result<()> {
	record_node(Node::LayerNorm {
		input: input.clone(),
		normalized,
		inverse_stddev,
		output_id: output.value_id(),
		weight,
		weight_value,
		weight_version,
		bias,
		bias_version,
	})
}

pub(in crate::ml) fn record_rms_norm(
	input: &Matrix,
	output: &Matrix,
	weight: Option<(Parameter, u64)>,
	weight_value: Matrix,
	epsilon: f32,
) -> Result<()> {
	let (weight, weight_version) = weight
		.map(|(weight, version)| (Some(weight), Some(version)))
		.unwrap_or((None, None));
	record_node(Node::RmsNorm {
		input: input.clone(),
		output_id: output.value_id(),
		weight,
		weight_value,
		weight_version,
		epsilon,
	})
}

#[allow(
	clippy::too_many_arguments,
	reason = "the node retains the complete gated normalization state and optional parameters"
)]
pub(in crate::ml) fn record_rms_norm_gated(
	input: &Matrix,
	output: &Matrix,
	weight: Option<(Parameter, u64)>,
	weight_value: Matrix,
	bias: Option<(Parameter, u64)>,
	bias_value: Option<Matrix>,
	gate: &Matrix,
	epsilon: f32,
) -> Result<()> {
	let (weight, weight_version) = weight
		.map(|(parameter, version)| (Some(parameter), Some(version)))
		.unwrap_or((None, None));
	let (bias, bias_version) = bias
		.map(|(parameter, version)| (Some(parameter), Some(version)))
		.unwrap_or((None, None));
	record_node(Node::RmsNormGated(Box::new(
		super::super::node::RmsNormGatedNode {
			input: input.clone(),
			weight,
			weight_value,
			weight_version,
			bias,
			bias_value,
			bias_version,
			gate: gate.clone(),
			epsilon,
			output_id: output.value_id(),
		},
	)))
}
