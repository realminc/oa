mod byte_common;

fn main() -> oa::Result<()> {
	byte_common::run(
		byte_common::Tutorial {
			title: "OA Tutorial — Byte MoE Transformer · all-position LM (autograd)",
			description: "Byte + position Embedding → Attention + MoE(E=4,K=2,DFF=16) → LN → Linear",
			timer_name: "byte_moe_training_step",
			checkpoint_stem: "oars_byte_moe",
			learning_rate: 0.01,
			expected_loss: oa::ml::nlp::BYTE_MOE_FINAL_LOSS,
			expected_accuracy: oa::ml::nlp::BYTE_MOE_ACCURACY,
			expected_generation: oa::ml::nlp::BYTE_MOE_REFERENCE_GENERATION,
		},
		oa::ml::nlp::ByteMoeTransformer::new,
	)
}
