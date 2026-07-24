#!/usr/bin/env python3
"""1:1 Python entry point for TutorialNlpCharRnnAg.cpp."""

# pyright: reportWildcardImportFromLibrary=false
from oa import *
import _nlp_common as nlp


def main() -> None:
	nlp.run_suite_member(OaNlpArchitecture.Rnn, OaNlpTokenizerKind.Char)


if __name__ == "__main__":
	main()
