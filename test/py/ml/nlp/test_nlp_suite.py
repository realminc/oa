#!/usr/bin/env python3
"""Tests for the canonical Python NLP suite bridge."""

from __future__ import annotations

import pytest

import oa_python_test  # noqa: F401 - bootstraps source builds

import oa


@pytest.mark.parametrize(
	("tokenizer", "vocab"),
	(
		(oa.NlpTokenizerKind.Byte, 256),
		(oa.NlpTokenizerKind.Bpe, 320),
		(oa.NlpTokenizerKind.Char, 27),
	),
)
def test_recipe_and_sampler_contract(tokenizer, vocab):
	recipe = oa.NlpSuiteRecipe(oa.NlpArchitecture.Gru, tokenizer)
	sampler = oa.NlpSuiteSampler(recipe, 2)

	assert recipe.vocabSize() == vocab
	assert recipe.contextLength() == oa.NlpSuiteContextLength
	assert recipe.modelWidth() == oa.NlpSuiteModelWidth
	assert recipe.hiddenWidth() == oa.NlpSuiteHiddenWidth
	assert sampler.decode(sampler.encode(sampler.corpus())) == sampler.corpus()


def test_bpe_reduces_canonical_corpus_positions():
	recipe = oa.NlpSuiteRecipe(
		oa.NlpArchitecture.Transformer,
		oa.NlpTokenizerKind.Bpe,
	)
	sampler = oa.NlpSuiteSampler(recipe, 1)
	corpus = sampler.corpus()

	assert len(sampler.encode(corpus)) < len(corpus.encode("utf-8"))


def test_byte_decode_preserves_non_utf8_tokens():
	recipe = oa.NlpSuiteRecipe(
		oa.NlpArchitecture.Rnn,
		oa.NlpTokenizerKind.Byte,
	)
	sampler = oa.NlpSuiteSampler(recipe, 1)

	assert sampler.decodeBytes([0xA5]) == b"\xA5"
	assert sampler.decode([0xA5]) == "\N{REPLACEMENT CHARACTER}"


def test_public_metric_namespace_is_installed():
	assert oa.FnMetric.accuracy.__name__ == "accuracy"
	assert oa.FnMetric.scalarLoss.__name__ == "scalarLoss"
