#!/usr/bin/env python3
"""OA Tutorial — Empyrealm Core · autograd fidelity.

Mirrors sdk/rs/tutorials/ml/nlp/tu_nlp_byte_empyrealm.rs.
"""

import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import _nlp_common
import oa.ml.nlp as nlp


def main() -> None:
	_nlp_common.run_byte(
		_nlp_common.Tutorial(
			title="OA Tutorial — Empyrealm Core · autograd fidelity",
			description="Empyrealm(Byte Embedding + Mamba-3 mixer + residual) → Linear(32→256)",
			timer_name="byte_empyrealm_training_step",
			checkpoint_stem="oars_byte_empyrealm",
			learning_rate=0.003,
			expected_loss=nlp.BYTE_EMPYREALM_FINAL_LOSS,
			loss_tolerance=0.001,
			expected_accuracy=nlp.BYTE_EMPYREALM_ACCURACY,
			expected_generation=nlp.BYTE_EMPYREALM_REFERENCE_GENERATION,
		),
		nlp.ByteEmpyrealm,
	)


if __name__ == "__main__":
	main()
