#!/usr/bin/env python3
"""OA Tutorial — Char MoE Transformer · all-position LM (autograd).

Mirrors sdk/rs/tutorials/ml/nlp/tu_nlp_char_moe_transformer.rs.
"""

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import _nlp_common as nlp
import oa


def main() -> None:
	engine = oa.Engine()
	model = oa.ml.CharMoeTransformer(engine)
	optimizer = oa.ml.AdamW(model.all_parameters(), learning_rate=0.01)
	nlp.run_char("Char MoE Transformer · all-position LM", model, optimizer, engine)


if __name__ == "__main__":
	main()
