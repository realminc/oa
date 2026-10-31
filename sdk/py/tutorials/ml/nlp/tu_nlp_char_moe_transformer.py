#!/usr/bin/env python3
"""OA Tutorial — Char MoE Transformer · all-position LM.

Mirrors sdk/rs/tutorials/ml/nlp/tu_nlp_char_moe_transformer.rs.
"""

import os
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import _nlp_common
import oa.ml.nlp as nlp


def main() -> None:
	_nlp_common.run_char(
		_nlp_common.CharTutorial(
			prog="tu_nlp_char_moe_transformer",
			cli_title="OA Tutorial — Char MoE Transformer",
			header_line="║  OA Tutorial — Char MoE Transformer · all-position LM           ║",
			model_line=(
				"Model: token+position embedding → attention → "
				f"sparse MoE(E={nlp.MOE_NUM_EXPERTS}, K={nlp.MOE_EXPERTS_PER_TOKEN}, "
				f"DFF={nlp.MOE_EXPERT_HIDDEN_WIDTH}) → LayerNorm → Linear"
			),
			sequence_unit="token",
			timer_name="char_moe_transformer_training_step",
			log_dir_leaf="log/nlp/char_moe_transformer",
			checkpoint_name="oars_char_moe_transformer.oam",
			default_lr=0.01,
			show_training_line=True,
			eval_loss_line=True,
			accuracy_note=False,
			loss_tolerance=0.02,
			accuracy_tolerance=0.01,
			expected_loss=nlp.CHAR_MOE_TRANSFORMER_CPP_FINAL_LOSS,
			expected_accuracy=nlp.CHAR_MOE_TRANSFORMER_CPP_ACCURACY,
			expected_generation=nlp.CHAR_MOE_TRANSFORMER_REFERENCE_GENERATION,
		),
		nlp.CharMoeTransformer,
	)


if __name__ == "__main__":
	main()
