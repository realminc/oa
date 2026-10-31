#!/usr/bin/env python3
"""OA Tutorial — BPE GRU · all-position LM (autograd).

Mirrors sdk/rs/tutorials/ml/nlp/tu_nlp_bpe_gru.rs.
"""

import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import _nlp_common
import oa.ml.nlp as nlp


def main() -> None:
	_nlp_common.run_bpe(
		_nlp_common.Tutorial(
			title="OA Tutorial — BPE GRU · all-position LM (autograd)",
			description="BPE Embedding → GRU(32→64) → Linear(64→320)",
			timer_name="bpe_gru_training_step",
			checkpoint_stem="oars_bpe_gru",
			learning_rate=0.01,
			expected_loss=nlp.BPE_GRU_FINAL_LOSS,
			expected_accuracy=nlp.BPE_GRU_ACCURACY,
			expected_generation=nlp.BPE_GRU_REFERENCE_GENERATION,
		),
		nlp.BpeGru,
	)


if __name__ == "__main__":
	main()
