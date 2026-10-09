//! Saved state for batch norm operations.

use super::super::{node::GradNode, tape::record_node};
use crate::ml::Parameter;
use crate::{Matrix, Result};

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
	record_node(GradNode::BatchNorm2d {
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
