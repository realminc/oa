use crate::{Matrix, Result};

use super::super::{node::Node, tape::record_node};

pub(in crate::ml) fn record_rope(
	input: &Matrix,
	output: &Matrix,
	num_heads: usize,
	head_dim: usize,
	theta_base: f32,
	position_offset: u32,
) -> Result<()> {
	record_node(Node::Rope {
		input: input.clone(),
		output_id: output.value_id(),
		num_heads,
		head_dim,
		theta_base,
		position_offset,
	})
}
