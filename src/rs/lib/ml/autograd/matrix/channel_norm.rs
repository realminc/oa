//! Saved state for channel norm operations.

use super::super::{node::GradNode, tape::record_node};
use crate::ml::Parameter;
use crate::{Matrix, Result};

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
	record_node(GradNode::ChannelNorm(Box::new(
		super::super::node::GradNodeChannelNorm {
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
