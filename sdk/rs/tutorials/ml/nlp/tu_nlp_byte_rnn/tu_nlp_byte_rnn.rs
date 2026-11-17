#[path = "../_byte_common.rs"]
mod _byte_common;

fn main() -> oa::Result<()> {
	_byte_common::run(
		_byte_common::Tutorial {
			title: "OA Tutorial — Byte RNN · all-position LM (autograd)",
			description: "ByteEmbedding(32) → RNN(64) → ByteHead(256)",
			timer_name: "byte_rnn_training_step",
			checkpoint_stem: "oars_byte_rnn",
			learning_rate: 0.01,
			expected_loss: sdk::ml::nlp::BYTE_RNN_FINAL_LOSS,
			loss_tolerance: 0.001,
			expected_accuracy: sdk::ml::nlp::BYTE_RNN_ACCURACY,
			expected_generation: sdk::ml::nlp::BYTE_RNN_REFERENCE_GENERATION,
		},
		sdk::ml::nlp::ByteRnn::new,
	)
}
