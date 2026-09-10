use crate::{Matrix, Result};

use super::super::{node::Node, tape::record_node};

pub(in crate::ml) fn record_avg_pool_2d(
	input: &Matrix,
	output: &Matrix,
	kernel_size: usize,
	stride: usize,
	padding: usize,
) -> Result<()> {
	record_node(Node::AvgPool2d {
		input: input.clone(),
		output_id: output.value_id(),
		kernel_size,
		stride,
		padding,
	})
}

pub(in crate::ml) fn record_max_pool_2d(
	input: &Matrix,
	output: &Matrix,
	indices: Matrix,
	kernel_size: usize,
	stride: usize,
	padding: usize,
) -> Result<()> {
	record_node(Node::MaxPool2d {
		input: input.clone(),
		indices,
		output_id: output.value_id(),
		kernel_size,
		stride,
		padding,
	})
}

pub(in crate::ml) fn record_adaptive_avg_pool_2d(
	input: &Matrix,
	output: &Matrix,
	output_height: usize,
	output_width: usize,
) -> Result<()> {
	record_node(Node::AdaptiveAvgPool2d {
		input: input.clone(),
		output_id: output.value_id(),
		output_height,
		output_width,
	})
}
