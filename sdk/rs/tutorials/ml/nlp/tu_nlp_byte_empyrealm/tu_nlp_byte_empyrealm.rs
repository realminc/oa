#[path = "../_byte_common.rs"]
mod _byte_common;

fn main() -> oa::Result<()> {
	_byte_common::run(
		_byte_common::Tutorial {
			title: "OA Tutorial — Empyrealm Core · autograd fidelity",
			description: "Empyrealm(Byte Embedding + Mamba-3/attention hybrid stack) → Linear(32→256)",
			timer_name: "byte_empyrealm_training_step",
			checkpoint_stem: "oars_byte_empyrealm",
			learning_rate: 0.003,
			expected_loss: sdk::ml::nlp::BYTE_EMPYREALM_FINAL_LOSS,
			loss_tolerance: 0.001,
			expected_accuracy: sdk::ml::nlp::BYTE_EMPYREALM_ACCURACY,
			expected_generation: sdk::ml::nlp::BYTE_EMPYREALM_REFERENCE_GENERATION,
		},
		sdk::ml::nlp::ByteEmpyrealm::new,
	)
}
