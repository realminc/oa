use crate::ml::Parameter;
use crate::{Matrix, Result};

use super::super::{node::Node, tape::record_node};

#[allow(
	clippy::too_many_arguments,
	reason = "the node retains exact Conv1d state and optional module parameters"
)]
pub(in crate::ml) fn record_conv_1d(
	input: &Matrix,
	output: &Matrix,
	weight: Option<(Parameter, u64)>,
	weight_value: Matrix,
	bias: Option<(Parameter, u64)>,
	bias_value: Matrix,
	stride: usize,
	padding: usize,
	dilation: usize,
) -> Result<()> {
	let (weight, weight_version) = weight
		.map(|(parameter, version)| (Some(parameter), Some(version)))
		.unwrap_or((None, None));
	let (bias, bias_version) = bias
		.map(|(parameter, version)| (Some(parameter), Some(version)))
		.unwrap_or((None, None));
	record_node(Node::Conv1d {
		input: input.clone(),
		output_id: output.value_id(),
		weight,
		weight_value,
		weight_version,
		bias,
		bias_value,
		bias_version,
		stride,
		padding,
		dilation,
	})
}

#[allow(
	clippy::too_many_arguments,
	reason = "the node retains exact ConvTranspose1d state and optional module parameter"
)]
pub(in crate::ml) fn record_conv_transpose_1d(
	input: &Matrix,
	output: &Matrix,
	weight: Option<(Parameter, u64)>,
	weight_value: Matrix,
	stride: usize,
	padding: usize,
	dilation: usize,
) -> Result<()> {
	let (weight, weight_version) = weight
		.map(|(parameter, version)| (Some(parameter), Some(version)))
		.unwrap_or((None, None));
	record_node(Node::ConvTranspose1d {
		input: input.clone(),
		output_id: output.value_id(),
		weight,
		weight_value,
		weight_version,
		stride,
		padding,
		dilation,
	})
}

#[allow(
	clippy::too_many_arguments,
	reason = "the node retains exact ConvTranspose2d state and optional module parameters"
)]
pub(in crate::ml) fn record_conv_transpose_2d(
	input: &Matrix,
	output: &Matrix,
	weight: Option<(Parameter, u64)>,
	weight_value: Matrix,
	bias: Option<(Parameter, u64)>,
	bias_value: Matrix,
	stride: usize,
	padding: usize,
) -> Result<()> {
	let (weight, weight_version) = weight
		.map(|(parameter, version)| (Some(parameter), Some(version)))
		.unwrap_or((None, None));
	let (bias, bias_version) = bias
		.map(|(parameter, version)| (Some(parameter), Some(version)))
		.unwrap_or((None, None));
	record_node(Node::ConvTranspose2d {
		input: input.clone(),
		output_id: output.value_id(),
		weight,
		weight_value,
		weight_version,
		bias,
		bias_value,
		bias_version,
		stride,
		padding,
	})
}

#[allow(
	clippy::too_many_arguments,
	reason = "the node retains exact grouped-convolution state and optional module parameters"
)]
pub(in crate::ml) fn record_conv_2d(
	input: &Matrix,
	output: &Matrix,
	weight: Option<(Parameter, u64)>,
	weight_value: Matrix,
	bias: Option<(Parameter, u64)>,
	bias_value: Matrix,
	stride: usize,
	padding: usize,
	groups: usize,
) -> Result<()> {
	let (weight, weight_version) = weight
		.map(|(parameter, version)| (Some(parameter), Some(version)))
		.unwrap_or((None, None));
	let (bias, bias_version) = bias
		.map(|(parameter, version)| (Some(parameter), Some(version)))
		.unwrap_or((None, None));
	record_node(Node::Conv2d {
		input: input.clone(),
		output_id: output.value_id(),
		weight,
		weight_value,
		weight_version,
		bias,
		bias_value,
		bias_version,
		stride,
		padding,
		groups,
	})
}
