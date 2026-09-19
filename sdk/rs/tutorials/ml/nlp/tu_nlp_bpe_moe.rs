mod _bpe_common;

fn main() -> oa::Result<()> {
	_bpe_common::run(
		_bpe_common::Tutorial {
			title: "OA Tutorial — BPE MoE Transformer · all-position LM (autograd)",
			description: "BPE + position Embedding → Attention + MoE(E=4,K=2,DFF=16) → LN → Linear",
			timer_name: "bpe_moe_training_step",
			checkpoint_stem: "oars_bpe_moe",
			learning_rate: 0.01,
			expected_loss: oa::sdk::ml::nlp::BPE_MOE_FINAL_LOSS,
			expected_accuracy: oa::sdk::ml::nlp::BPE_MOE_ACCURACY,
			expected_generation: oa::sdk::ml::nlp::BPE_MOE_REFERENCE_GENERATION,
		},
		oa::sdk::ml::nlp::BpeMoeTransformer::new,
	)
}
