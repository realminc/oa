#!/usr/bin/env python3
"""OA Tutorial — Byte Mamba-3 · all-position LM (autograd).

Mirrors sdk/rs/tutorials/ml/nlp/tu_nlp_byte_mamba3.rs.
"""

import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import _nlp_common
import oa.ml.nlp as nlp


def main() -> None:
	_nlp_common.run_byte(
		_nlp_common.Tutorial(
			title="OA Tutorial — Byte Mamba-3 · all-position LM (autograd)",
			description="Byte Embedding → Mamba-3(32,state=32,expand=2) + residual → Linear(32→256)",
			timer_name="byte_mamba3_training_step",
			checkpoint_stem="oars_byte_mamba3",
			learning_rate=0.003,
			expected_loss=nlp.BYTE_MAMBA3_FINAL_LOSS,
			loss_tolerance=0.001,
			expected_accuracy=nlp.BYTE_MAMBA3_ACCURACY,
			expected_generation=nlp.BYTE_MAMBA3_REFERENCE_GENERATION,
		),
		nlp.ByteMamba3,
	)


if __name__ == "__main__":
	main()
