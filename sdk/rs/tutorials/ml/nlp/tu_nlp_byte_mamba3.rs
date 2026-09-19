mod _byte_common;

fn main() -> oa::Result<()> {
	_byte_common::run(
		_byte_common::Tutorial {
			title: "OA Tutorial — Byte Mamba-3 · all-position LM (autograd)",
			description: "Byte Embedding → Mamba-3(32,state=32,expand=2) + residual → Linear(32→256)",
			timer_name: "byte_mamba3_training_step",
			checkpoint_stem: "oars_byte_mamba3",
			learning_rate: 0.003,
			expected_loss: oa::sdk::ml::nlp::BYTE_MAMBA3_FINAL_LOSS,
			expected_accuracy: oa::sdk::ml::nlp::BYTE_MAMBA3_ACCURACY,
			expected_generation: oa::sdk::ml::nlp::BYTE_MAMBA3_REFERENCE_GENERATION,
		},
		oa::sdk::ml::nlp::ByteMamba3::new,
	)
}
