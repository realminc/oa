#!/usr/bin/env python3
"""1:1 Python entry point for TutorialNlpByteTransformerAg.cpp."""

# pyright: reportWildcardImportFromLibrary=false
from oa import *
import _nlpCommon as nlp


def main() -> None:
	nlp.runSuiteMember(
		NlpArchitecture.Transformer,
		NlpTokenizerKind.Byte,
	)


if __name__ == "__main__":
	main()
