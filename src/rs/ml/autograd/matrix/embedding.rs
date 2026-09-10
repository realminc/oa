use crate::ml::Parameter;
use crate::{Matrix, Result};

use super::super::{node::Node, tape::record_node};

pub(in crate::ml) fn record_embedding(
	indices: &Matrix,
	output: &Matrix,
	weight: Parameter,
	weight_value: Matrix,
	weight_version: u64,
) -> Result<()> {
	record_node(Node::Embedding {
		indices: indices.clone(),
		output_id: output.value_id(),
		weight,
		weight_value,
		weight_version,
	})
}
