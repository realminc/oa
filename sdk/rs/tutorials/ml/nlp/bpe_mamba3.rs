mod bpe_common;

fn main() -> oa::Result<()> {
	bpe_common::run(
		bpe_common::Tutorial {
			title: "OA Tutorial — BPE Mamba-3 · all-position LM (autograd)",
			description: "BPE Embedding → Mamba-3(32,state=32,expand=2) + residual → Linear",
			timer_name: "bpe_mamba3_training_step",
			checkpoint_stem: "oars_bpe_mamba3",
			learning_rate: 0.003,
			expected_loss: oa::ml::nlp::BPE_MAMBA3_FINAL_LOSS,
			expected_accuracy: oa::ml::nlp::BPE_MAMBA3_ACCURACY,
			expected_generation: oa::ml::nlp::BPE_MAMBA3_REFERENCE_GENERATION,
		},
		oa::ml::nlp::BpeMamba3::new,
	)
}
