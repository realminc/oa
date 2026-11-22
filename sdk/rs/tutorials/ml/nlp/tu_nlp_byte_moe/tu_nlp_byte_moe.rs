#[path = "../_byte_common.rs"]
mod _byte_common;

fn main() -> oa::Result<()> {
	_byte_common::run(
		_byte_common::Tutorial {
			title: "OA Tutorial — Byte MoE Transformer · all-position LM (autograd)",
			description: "Byte + position Embedding → Attention + MoE(E=4,K=2,DFF=16) → LN → Linear",
			timer_name: "byte_moe_training_step",
			checkpoint_stem: "oars_byte_moe",
			learning_rate: 0.01,
			expected_loss: sdk::ml::nlp::BYTE_MOE_FINAL_LOSS,
			loss_tolerance: sdk::ml::nlp::BYTE_MOE_FINAL_LOSS_TOLERANCE,
			expected_accuracy: sdk::ml::nlp::BYTE_MOE_ACCURACY,
			expected_generation: sdk::ml::nlp::BYTE_MOE_REFERENCE_GENERATION,
		},
		sdk::ml::nlp::ByteMoeTransformer::new,
	)
}
