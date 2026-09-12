//! Private autograd attachments for loss operation families.

include!("loss/core.gen.rs");

pub(in crate::ml) fn record_ppo_clipped_policy(
	new_log_probability: &crate::Matrix,
	old_log_probability: &crate::Matrix,
	advantage: &crate::Matrix,
	clip_epsilon: f32,
	output: &crate::Matrix,
) -> crate::Result<()> {
	super::tape::record_node(super::node::Node::PpoClippedPolicy {
		new_log_probability: new_log_probability.clone(),
		old_log_probability: old_log_probability.clone(),
		advantage: advantage.clone(),
		clip_epsilon,
		output_id: output.value_id(),
	})
}
