#!/usr/bin/env python3
"""1:1 Python entry point for TutorialNlpCharGruAg.cpp."""

# pyright: reportWildcardImportFromLibrary=false
from oa import *
import _nlpCommon as nlp


def main() -> None:
	nlp.runSuiteMember(NlpArchitecture.Gru, NlpTokenizerKind.Char)


if __name__ == "__main__":
	main()
