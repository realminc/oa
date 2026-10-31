#!/usr/bin/env python3
"""OA Tutorial — BPE MoE Transformer · all-position LM (autograd).

Mirrors sdk/rs/tutorials/ml/nlp/tu_nlp_bpe_moe.rs.
"""

import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import _nlp_common
import oa.ml.nlp as nlp


def main() -> None:
	_nlp_common.run_bpe(
		_nlp_common.Tutorial(
			title="OA Tutorial — BPE MoE Transformer · all-position LM (autograd)",
			description="BPE + position Embedding → Attention + MoE(E=4,K=2,DFF=16) → LN → Linear",
			timer_name="bpe_moe_training_step",
			checkpoint_stem="oars_bpe_moe",
			learning_rate=0.01,
			expected_loss=nlp.BPE_MOE_FINAL_LOSS,
			expected_accuracy=nlp.BPE_MOE_ACCURACY,
			expected_generation=nlp.BPE_MOE_REFERENCE_GENERATION,
		),
		nlp.BpeMoeTransformer,
	)


if __name__ == "__main__":
	main()
