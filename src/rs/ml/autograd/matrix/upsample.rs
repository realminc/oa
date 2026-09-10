use crate::ml::matrix::UpsampleMode;
use crate::{Matrix, Result};

use super::super::{node::Node, tape::record_node};

pub(in crate::ml) fn record_upsample_2d(
	input: &Matrix,
	output: &Matrix,
	scale_factor: usize,
	mode: UpsampleMode,
) -> Result<()> {
	record_node(Node::Upsample2d {
		input: input.clone(),
		output_id: output.value_id(),
		scale_factor,
		mode,
	})
}
