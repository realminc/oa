use crate::{Matrix, Result};

use super::{node::GradNode, tape::record_node};

pub(in crate::ml) fn record_flow_linear_match(
	clean: &Matrix,
	noise: &Matrix,
	time: &Matrix,
	state: &Matrix,
	velocity: &Matrix,
) -> Result<()> {
	record_node(GradNode::FlowLinearState {
		clean: clean.clone(),
		noise: noise.clone(),
		time: time.clone(),
		output_id: state.value_id(),
	})?;
	record_node(GradNode::FlowLinearVelocity {
		clean: clean.clone(),
		noise: noise.clone(),
		output_id: velocity.value_id(),
	})
}

pub(in crate::ml) fn record_flow_euler_step(
	state: &Matrix,
	velocity: &Matrix,
	delta_time: f32,
	output: &Matrix,
) -> Result<()> {
	record_node(GradNode::FlowEulerStep {
		state: state.clone(),
		velocity: velocity.clone(),
		delta_time,
		output_id: output.value_id(),
	})
}

pub(in crate::ml) fn record_flow_masked_mse(
	prediction: &Matrix,
	target: &Matrix,
	mask: &Matrix,
	denominator: &Matrix,
	output: &Matrix,
) -> Result<()> {
	record_node(GradNode::FlowMaskedMse {
		prediction: prediction.clone(),
		target: target.clone(),
		mask: mask.clone(),
		denominator: denominator.clone(),
		output_id: output.value_id(),
	})
}
