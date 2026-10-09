mod _bpe_common;

fn main() -> oa::Result<()> {
	_bpe_common::run(
		_bpe_common::Tutorial {
			title: "OA Tutorial — BPE GRU · all-position LM (autograd)",
			description: "BPE Embedding → GRU(32→64) → Linear(64→320)",
			timer_name: "bpe_gru_training_step",
			checkpoint_stem: "oars_bpe_gru",
			learning_rate: 0.01,
			expected_loss: oa::sdk::ml::nlp::BPE_GRU_FINAL_LOSS,
			expected_accuracy: oa::sdk::ml::nlp::BPE_GRU_ACCURACY,
			expected_generation: oa::sdk::ml::nlp::BPE_GRU_REFERENCE_GENERATION,
		},
		oa::sdk::ml::nlp::BpeGru::new,
	)
}
