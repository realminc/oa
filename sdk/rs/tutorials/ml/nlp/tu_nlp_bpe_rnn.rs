mod _bpe_common;

fn main() -> oa::Result<()> {
	_bpe_common::run(
		_bpe_common::Tutorial {
			title: "OA Tutorial — BPE RNN · all-position LM (autograd)",
			description: "BPE Embedding → RNN(32→64) → Linear(64→320)",
			timer_name: "bpe_rnn_training_step",
			checkpoint_stem: "oars_bpe_rnn",
			learning_rate: 0.01,
			expected_loss: oa::sdk::ml::nlp::BPE_RNN_FINAL_LOSS,
			expected_accuracy: oa::sdk::ml::nlp::BPE_RNN_ACCURACY,
			expected_generation: oa::sdk::ml::nlp::BPE_RNN_REFERENCE_GENERATION,
		},
		oa::sdk::ml::nlp::BpeRnn::new,
	)
}
