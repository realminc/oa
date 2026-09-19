mod _byte_common;

fn main() -> oa::Result<()> {
	_byte_common::run(
		_byte_common::Tutorial {
			title: "OA Tutorial — Byte Transformer · all-position LM (autograd)",
			description: "Byte + position Embedding → Transformer(32,64) → LN → Linear(32→256)",
			timer_name: "byte_transformer_training_step",
			checkpoint_stem: "oars_byte_transformer",
			learning_rate: 0.01,
			expected_loss: oa::sdk::ml::nlp::BYTE_TRANSFORMER_FINAL_LOSS,
			expected_accuracy: oa::sdk::ml::nlp::BYTE_TRANSFORMER_ACCURACY,
			expected_generation: oa::sdk::ml::nlp::BYTE_TRANSFORMER_REFERENCE_GENERATION,
		},
		oa::sdk::ml::nlp::ByteTransformer::new,
	)
}
