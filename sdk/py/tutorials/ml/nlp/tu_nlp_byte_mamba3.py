#!/usr/bin/env python3
"""OA Tutorial — Byte Mamba-3 · all-position LM (autograd).

Mirrors sdk/rs/tutorials/ml/nlp/tu_nlp_byte_mamba3.rs.
"""

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import _nlp_common as nlp
import oa


def main() -> None:
	engine = oa.Engine()
	model = oa.ml.ByteMamba3(engine)
	optimizer = oa.ml.AdamW(model.all_parameters(), learning_rate=0.003)
	nlp.run_byte("Byte Mamba-3 · all-position LM", model, optimizer, engine)


if __name__ == "__main__":
	main()
