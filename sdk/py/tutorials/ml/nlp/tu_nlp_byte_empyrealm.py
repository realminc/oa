#!/usr/bin/env python3
"""OA Tutorial — Empyrealm Core · autograd fidelity (byte-level).

Mirrors sdk/rs/tutorials/ml/nlp/tu_nlp_byte_empyrealm.rs.

The Empyrealm Core model reuses the canonical Mamba-3 operation providers to
demonstrate fidelity parity without duplicating donor shader bodies under new
names.  Its topology and parameter count differ from the Char/Byte Mamba-3
entry; the lower learning rate (0.003) matches the Rust and C++ reference.
"""

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import _nlp_common as nlp
import oa


def main() -> None:
	engine = oa.Engine()
	model = oa.ml.ByteEmpyrealm(engine)
	# lr=0.003 matches the Rust donor reference for the Empyrealm Core model.
	optimizer = oa.ml.AdamW(model.all_parameters(), learning_rate=0.003)
	nlp.run_byte(
		"Empyrealm Core · autograd fidelity",
		model,
		optimizer,
		engine,
	)


if __name__ == "__main__":
	main()
