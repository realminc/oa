#!/usr/bin/env python3
"""OA Tutorial — Char Mamba-3 · all-position LM (autograd).

Mirrors sdk/rs/tutorials/ml/nlp/tu_nlp_char_mamba3.rs.
"""

import os
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import _nlp_common
import oa.ml.nlp as nlp


def main() -> None:
	_nlp_common.run_char(
		_nlp_common.CharTutorial(
			prog="tu_nlp_char_mamba3",
			cli_title="OA Tutorial — Char Mamba-3",
			header_line="║  OA Tutorial — Char Mamba-3 · all-position LM (autograd)         ║",
			model_line="Model: Embedding(32) → Mamba-3 SISO → residual → Linear(27)",
			sequence_unit="token",
			timer_name="char_mamba3_training_step",
			log_dir_leaf="log/nlp/char_mamba3",
			checkpoint_name="oars_char_mamba3.oam",
			default_lr=0.003,
			show_training_line=True,
			eval_loss_line=True,
			accuracy_note=True,
			loss_tolerance=0.001,
			accuracy_tolerance=0.000001,
			expected_loss=nlp.CHAR_MAMBA3_FINAL_LOSS,
			expected_accuracy=nlp.CHAR_MAMBA3_ACCURACY,
			expected_generation=nlp.CHAR_MAMBA3_REFERENCE_GENERATION,
		),
		nlp.CharMamba3,
	)


if __name__ == "__main__":
	main()
