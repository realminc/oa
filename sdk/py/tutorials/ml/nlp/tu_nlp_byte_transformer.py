#!/usr/bin/env python3
"""OA Tutorial — Byte Transformer · all-position LM (autograd).

Mirrors sdk/rs/tutorials/ml/nlp/tu_nlp_byte_transformer.rs.
"""

import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import _nlp_common
import oa.ml.nlp as nlp


def main() -> None:
	_nlp_common.run_byte(
		_nlp_common.Tutorial(
			title="OA Tutorial — Byte Transformer · all-position LM (autograd)",
			description="Byte + position Embedding → Transformer(32,64) → LN → Linear(32→256)",
			timer_name="byte_transformer_training_step",
			checkpoint_stem="oars_byte_transformer",
			learning_rate=0.01,
			expected_loss=nlp.BYTE_TRANSFORMER_FINAL_LOSS,
			loss_tolerance=0.001,
			expected_accuracy=nlp.BYTE_TRANSFORMER_ACCURACY,
			expected_generation=nlp.BYTE_TRANSFORMER_REFERENCE_GENERATION,
		),
		nlp.ByteTransformer,
	)


if __name__ == "__main__":
	main()
