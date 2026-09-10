use crate::ml::Parameter;
use crate::{Matrix, Result};

use super::super::{node::Node, tape::record_node};

pub(in crate::ml) fn record_linear(
	input: &Matrix,
	output: &Matrix,
	weight: Parameter,
	weight_value: Matrix,
	weight_version: u64,
	bias: Option<(Parameter, u64)>,
) -> Result<()> {
	let (bias, bias_version) = bias
		.map(|(bias, version)| (Some(bias), Some(version)))
		.unwrap_or((None, None));
	record_node(Node::Linear {
		input: input.clone(),
		output_id: output.value_id(),
		weight,
		weight_value,
		weight_version,
		bias,
		bias_version,
	})
}
