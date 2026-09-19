mod _byte_common;

fn main() -> oa::Result<()> {
	_byte_common::run(
		_byte_common::Tutorial {
			title: "OA Tutorial — Empyrealm Core · autograd fidelity",
			description: "EmpyrealmCore(Byte Embedding + Mamba-3 mixer + residual) → Linear(32→256)",
			timer_name: "byte_empyrealm_training_step",
			checkpoint_stem: "oars_byte_empyrealm",
			learning_rate: 0.003,
			expected_loss: oa::sdk::ml::nlp::BYTE_EMPYREALM_FINAL_LOSS,
			expected_accuracy: oa::sdk::ml::nlp::BYTE_EMPYREALM_ACCURACY,
			expected_generation: oa::sdk::ml::nlp::BYTE_EMPYREALM_REFERENCE_GENERATION,
		},
		oa::sdk::ml::nlp::ByteEmpyrealm::new,
	)
}
