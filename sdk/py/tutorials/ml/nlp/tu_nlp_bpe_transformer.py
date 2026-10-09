#!/usr/bin/env python3
"""OA Tutorial — BPE Transformer · all-position LM (autograd).

Mirrors sdk/rs/tutorials/ml/nlp/tu_nlp_bpe_transformer.rs.
"""

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import _nlp_common as nlp
import oa


def main() -> None:
	engine = oa.Engine()
	tokenizer = nlp.build_bpe_tokenizer()
	model = oa.ml.BpeTransformer(engine)
	optimizer = oa.ml.AdamW(model.all_parameters(), learning_rate=0.01)
	nlp.run_bpe("BPE Transformer · all-position LM", model, optimizer, tokenizer, engine)


if __name__ == "__main__":
	main()
