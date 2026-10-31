#!/usr/bin/env python3
"""OA Tutorial — Byte MoE Transformer · all-position LM (autograd).

Mirrors sdk/rs/tutorials/ml/nlp/tu_nlp_byte_moe.rs.
"""

import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import _nlp_common
import oa.ml.nlp as nlp


def main() -> None:
	_nlp_common.run_byte(
		_nlp_common.Tutorial(
			title="OA Tutorial — Byte MoE Transformer · all-position LM (autograd)",
			description="Byte + position Embedding → Attention + MoE(E=4,K=2,DFF=16) → LN → Linear",
			timer_name="byte_moe_training_step",
			checkpoint_stem="oars_byte_moe",
			learning_rate=0.01,
			expected_loss=nlp.BYTE_MOE_FINAL_LOSS,
			loss_tolerance=nlp.BYTE_MOE_FINAL_LOSS_TOLERANCE,
			expected_accuracy=nlp.BYTE_MOE_ACCURACY,
			expected_generation=nlp.BYTE_MOE_REFERENCE_GENERATION,
		),
		nlp.ByteMoeTransformer,
	)


if __name__ == "__main__":
	main()
