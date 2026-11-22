//! Saved-state attachments for the GRU operation family.

use crate::ml::Parameter;
use crate::{Matrix, Result};

use super::super::{node::GradNode, tape::record_node};

#[allow(
	clippy::too_many_arguments,
	reason = "saved recurrent values and optional parameter identities form one autograd node"
)]
pub(in crate::ml) fn record_gru_scan(
	gates_i: &Matrix,
	output: &Matrix,
	hidden_previous: Matrix,
	weight: Option<(Parameter, u64)>,
	weight_value: Matrix,
	bias: Option<(Parameter, u64)>,
	bias_value: Matrix,
	has_bias: bool,
) -> Result<()> {
	let (weight, weight_version) = weight
		.map(|(parameter, version)| (Some(parameter), Some(version)))
		.unwrap_or((None, None));
	let (bias, bias_version) = bias
		.map(|(parameter, version)| (Some(parameter), Some(version)))
		.unwrap_or((None, None));
	record_node(GradNode::GruScan(Box::new(
		super::super::node::GradNodeGruScan {
			gates_i: gates_i.clone(),
			hidden_previous,
			output_id: output.value_id(),
			weight,
			weight_value,
			weight_version,
			bias,
			bias_value,
			bias_version,
			has_bias,
		},
	)))
}

#[allow(
	clippy::too_many_arguments,
	reason = "saved recurrent values and optional parameter identities form one autograd node"
)]
pub(in crate::ml) fn record_gru_cell(
	gates_i: &Matrix,
	hidden: &Matrix,
	output: &Matrix,
	gates_h: Matrix,
	weight: Option<(Parameter, u64)>,
	weight_value: Matrix,
	bias: Option<(Parameter, u64)>,
	bias_value: Matrix,
	has_bias: bool,
) -> Result<()> {
	let (weight, weight_version) = weight
		.map(|(parameter, version)| (Some(parameter), Some(version)))
		.unwrap_or((None, None));
	let (bias, bias_version) = bias
		.map(|(parameter, version)| (Some(parameter), Some(version)))
		.unwrap_or((None, None));
	record_node(GradNode::GruCell(Box::new(
		super::super::node::GradNodeGruCell {
			gates_i: gates_i.clone(),
			gates_h,
			hidden: hidden.clone(),
			output_id: output.value_id(),
			weight,
			weight_value,
			weight_version,
			bias,
			bias_value,
			bias_version,
			has_bias,
		},
	)))
}
