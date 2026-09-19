mod _byte_common;

fn main() -> oa::Result<()> {
	_byte_common::run(
		_byte_common::Tutorial {
			title: "OA Tutorial — Byte GRU · all-position LM (autograd)",
			description: "ByteEmbedding(32) → GRU(64) → ByteHead(256)",
			timer_name: "byte_gru_training_step",
			checkpoint_stem: "oars_byte_gru",
			learning_rate: 0.01,
			expected_loss: oa::sdk::ml::nlp::BYTE_GRU_FINAL_LOSS,
			expected_accuracy: oa::sdk::ml::nlp::BYTE_GRU_ACCURACY,
			expected_generation: oa::sdk::ml::nlp::BYTE_GRU_REFERENCE_GENERATION,
		},
		oa::sdk::ml::nlp::ByteGru::new,
	)
}
