#!/usr/bin/env python3
"""1:1 Python entry point for TutorialNlpBpeTransformerAg.cpp."""

# pyright: reportWildcardImportFromLibrary=false
from oa import *
import _nlp_common as nlp


def main() -> None:
	nlp.run_suite_member(
		OaNlpArchitecture.Transformer,
		OaNlpTokenizerKind.Bpe,
	)


if __name__ == "__main__":
	main()
