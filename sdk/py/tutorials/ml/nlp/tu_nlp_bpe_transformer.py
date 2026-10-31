#!/usr/bin/env python3
"""OA Tutorial — BPE Transformer · all-position LM (autograd).

Mirrors sdk/rs/tutorials/ml/nlp/tu_nlp_bpe_transformer.rs.
"""

import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import _nlp_common
import oa.ml.nlp as nlp


def main() -> None:
	_nlp_common.run_bpe(
		_nlp_common.Tutorial(
			title="OA Tutorial — BPE Transformer · all-position LM (autograd)",
			description="BPE + position Embedding → Transformer(32,64) → LN → Linear(32→320)",
			timer_name="bpe_transformer_training_step",
			checkpoint_stem="oars_bpe_transformer",
			learning_rate=0.01,
			expected_loss=nlp.BPE_TRANSFORMER_FINAL_LOSS,
			expected_accuracy=nlp.BPE_TRANSFORMER_ACCURACY,
			expected_generation=nlp.BPE_TRANSFORMER_REFERENCE_GENERATION,
		),
		nlp.BpeTransformer,
	)


if __name__ == "__main__":
	main()
