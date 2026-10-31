#!/usr/bin/env python3
"""OA Tutorial — Char RNN · all-position LM (autograd).

Mirrors sdk/rs/tutorials/ml/nlp/tu_nlp_char_rnn.rs.
"""

import os
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import _nlp_common
import oa.ml.nlp as nlp


def main() -> None:
	_nlp_common.run_char(
		_nlp_common.CharTutorial(
			prog="tu_nlp_char_rnn",
			cli_title="OA Tutorial — Char RNN",
			header_line="║  OA Tutorial — Char RNN · all-position LM (autograd)             ║",
			model_line=(
				f"Model: Embedding({nlp.MODEL_WIDTH}) → RNN({nlp.HIDDEN_WIDTH}) "
				f"→ Linear({nlp.CHAR_VOCAB_SIZE})"
			),
			sequence_unit="token",
			timer_name="char_rnn_training_step",
			log_dir_leaf="log/nlp/char_rnn",
			checkpoint_name="oars_char_rnn.oam",
			default_lr=0.01,
			show_training_line=True,
			eval_loss_line=False,
			accuracy_note=True,
			final_loss_max=0.25,
			accuracy_min=0.9,
			expected_generation=nlp.CHAR_RNN_REFERENCE_GENERATION,
		),
		nlp.CharRnn,
	)


if __name__ == "__main__":
	main()
