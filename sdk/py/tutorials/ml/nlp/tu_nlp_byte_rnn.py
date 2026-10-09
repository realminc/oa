#!/usr/bin/env python3
"""OA Tutorial — Byte Rnn · all-position LM (autograd).

Mirrors sdk/rs/tutorials/ml/nlp/tu_nlp_byte_rnn.rs.
"""

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import _nlp_common as nlp
import oa


def main() -> None:
	engine = oa.Engine()
	model = oa.ml.ByteRnn(engine)
	optimizer = oa.ml.AdamW(model.all_parameters(), learning_rate=0.01)
	nlp.run_byte("Byte Rnn · all-position LM", model, optimizer, engine)


if __name__ == "__main__":
	main()
