#!/usr/bin/env python3
"""OA Tutorial — BPE Mamba-3 · all-position LM (autograd).

Mirrors sdk/rs/tutorials/ml/nlp/tu_nlp_bpe_mamba3.rs.
"""

import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import _nlp_common
import oa.ml.nlp as nlp


def main() -> None:
	_nlp_common.run_bpe(
		_nlp_common.Tutorial(
			title="OA Tutorial — BPE Mamba-3 · all-position LM (autograd)",
			description="BPE Embedding → Mamba-3(32,state=32,expand=2) + residual → Linear",
			timer_name="bpe_mamba3_training_step",
			checkpoint_stem="oars_bpe_mamba3",
			learning_rate=0.003,
			expected_loss=nlp.BPE_MAMBA3_FINAL_LOSS,
			expected_accuracy=nlp.BPE_MAMBA3_ACCURACY,
			expected_generation=nlp.BPE_MAMBA3_REFERENCE_GENERATION,
		),
		nlp.BpeMamba3,
	)


if __name__ == "__main__":
	main()
