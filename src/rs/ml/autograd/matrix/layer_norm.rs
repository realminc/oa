//! Saved state for layer norm operations.

use super::super::{node::GradNode, tape::record_node};
use crate::ml::Parameter;
use crate::{Matrix, Result};

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
	record_node(GradNode::LayerNorm {
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
