mod bpe_common;

fn main() -> oa::Result<()> {
	bpe_common::run(
		bpe_common::Tutorial {
			title: "OA Tutorial — BPE RNN · all-position LM (autograd)",
			description: "BPE Embedding → RNN(32→64) → Linear(64→320)",
			timer_name: "bpe_rnn_training_step",
			checkpoint_stem: "oars_bpe_rnn",
			learning_rate: 0.01,
			expected_loss: oa::ml::nlp::BPE_RNN_FINAL_LOSS,
			expected_accuracy: oa::ml::nlp::BPE_RNN_ACCURACY,
			expected_generation: oa::ml::nlp::BPE_RNN_REFERENCE_GENERATION,
		},
		oa::ml::nlp::BpeRnn::new,
	)
}
