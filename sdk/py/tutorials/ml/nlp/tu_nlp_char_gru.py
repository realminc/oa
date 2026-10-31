#!/usr/bin/env python3
"""OA Tutorial — Char GRU · all-position LM (autograd).

Mirrors sdk/rs/tutorials/ml/nlp/tu_nlp_char_gru.rs.
"""

import os
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import _nlp_common
import oa.ml.nlp as nlp


def main() -> None:
	_nlp_common.run_char(
		_nlp_common.CharTutorial(
			prog="tu_nlp_char_gru",
			cli_title="OA Tutorial — Char GRU",
			header_line="║  OA Tutorial — Char GRU · all-position LM (autograd)             ║",
			model_line="Model: Embedding(27,32) → GRU(64) → Linear(64,27)",
			sequence_unit="character token",
			timer_name="char_gru_training_step",
			log_dir_leaf="log/nlp/char_gru",
			checkpoint_name=f"oars-char-gru-{os.getpid()}.oam",
			default_lr=0.01,
			show_training_line=False,
			eval_loss_line=True,
			accuracy_note=False,
			loss_tolerance=0.001,
			accuracy_tolerance=0.001,
			expected_loss=nlp.CHAR_GRU_FINAL_LOSS,
			expected_accuracy=nlp.CHAR_GRU_ACCURACY,
			expected_generation=nlp.CHAR_GRU_REFERENCE_GENERATION,
		),
		nlp.CharGru,
	)


if __name__ == "__main__":
	main()
