mod _bpe_common;

fn main() -> oa::Result<()> {
	_bpe_common::run(
		_bpe_common::Tutorial {
			title: "OA Tutorial — BPE Transformer · all-position LM (autograd)",
			description: "BPE + position Embedding → Transformer(32,64) → LN → Linear(32→320)",
			timer_name: "bpe_transformer_training_step",
			checkpoint_stem: "oars_bpe_transformer",
			learning_rate: 0.01,
			expected_loss: oa::sdk::ml::nlp::BPE_TRANSFORMER_FINAL_LOSS,
			expected_accuracy: oa::sdk::ml::nlp::BPE_TRANSFORMER_ACCURACY,
			expected_generation: oa::sdk::ml::nlp::BPE_TRANSFORMER_REFERENCE_GENERATION,
		},
		oa::sdk::ml::nlp::BpeTransformer::new,
	)
}
