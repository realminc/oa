mod bpe_common;

fn main() -> oa::Result<()> {
	bpe_common::run(
		bpe_common::Tutorial {
			title: "OA Tutorial — BPE GRU · all-position LM (autograd)",
			description: "BPE Embedding → GRU(32→64) → Linear(64→320)",
			timer_name: "bpe_gru_training_step",
			checkpoint_stem: "oars_bpe_gru",
			learning_rate: 0.01,
			expected_loss: oa::ml::nlp::BPE_GRU_FINAL_LOSS,
			expected_accuracy: oa::ml::nlp::BPE_GRU_ACCURACY,
			expected_generation: oa::ml::nlp::BPE_GRU_REFERENCE_GENERATION,
		},
		oa::ml::nlp::BpeGru::new,
	)
}
