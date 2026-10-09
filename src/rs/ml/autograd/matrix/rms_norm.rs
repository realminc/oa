//! Saved state for rms norm operations.

use super::super::{node::GradNode, tape::record_node};
use crate::ml::Parameter;
use crate::{Matrix, Result};

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
	record_node(GradNode::RmsNorm {
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
	record_node(GradNode::RmsNormGated(Box::new(
		super::super::node::GradNodeRmsNormGated {
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
