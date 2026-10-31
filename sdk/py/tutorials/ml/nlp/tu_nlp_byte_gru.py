#!/usr/bin/env python3
"""OA Tutorial — Byte GRU · all-position LM (autograd).

Mirrors sdk/rs/tutorials/ml/nlp/tu_nlp_byte_gru.rs.
"""

import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import _nlp_common
import oa.ml.nlp as nlp


def main() -> None:
	_nlp_common.run_byte(
		_nlp_common.Tutorial(
			title="OA Tutorial — Byte GRU · all-position LM (autograd)",
			description="ByteEmbedding(32) → GRU(64) → ByteHead(256)",
			timer_name="byte_gru_training_step",
			checkpoint_stem="oars_byte_gru",
			learning_rate=0.01,
			expected_loss=nlp.BYTE_GRU_FINAL_LOSS,
			loss_tolerance=0.001,
			expected_accuracy=nlp.BYTE_GRU_ACCURACY,
			expected_generation=nlp.BYTE_GRU_REFERENCE_GENERATION,
		),
		nlp.ByteGru,
	)


if __name__ == "__main__":
	main()
