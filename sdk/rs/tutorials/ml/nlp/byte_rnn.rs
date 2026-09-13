mod byte_common;

fn main() -> oa::Result<()> {
	byte_common::run(
		byte_common::Tutorial {
			title: "OA Tutorial — Byte RNN · all-position LM (autograd)",
			description: "ByteEmbedding(32) → RNN(64) → ByteHead(256)",
			timer_name: "byte_rnn_training_step",
			checkpoint_stem: "oars_byte_rnn",
			learning_rate: 0.01,
			expected_loss: oa::ml::nlp::BYTE_RNN_FINAL_LOSS,
			expected_accuracy: oa::ml::nlp::BYTE_RNN_ACCURACY,
			expected_generation: oa::ml::nlp::BYTE_RNN_REFERENCE_GENERATION,
		},
		oa::ml::nlp::ByteRnn::new,
	)
}
