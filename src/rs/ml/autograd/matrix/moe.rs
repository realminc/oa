use crate::{Matrix, Result};

use super::super::{node::Node, tape::record_node};

pub(in crate::ml) fn record_moe_route_weights(
	probabilities: &Matrix,
	expert_indices: &Matrix,
	output: &Matrix,
) -> Result<()> {
	record_node(Node::MoeRouteWeights {
		probabilities: probabilities.clone(),
		expert_indices: expert_indices.clone(),
		output: output.clone(),
		output_id: output.value_id(),
	})
}
