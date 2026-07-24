#!/usr/bin/env python3
"""Focused binding contracts for the canonical Python NLP suite bridge."""

from __future__ import annotations

import pytest


oa = pytest.importorskip("oa", reason="oa Python package is not importable")


@pytest.mark.parametrize(
	("tokenizer", "vocab"),
	(
		(oa.OaNlpTokenizerKind.Byte, 256),
		(oa.OaNlpTokenizerKind.Bpe, 320),
		(oa.OaNlpTokenizerKind.Char, 27),
	),
)
def test_recipe_and_sampler_contract(tokenizer, vocab):
	recipe = oa.OaNlpSuiteRecipe(oa.OaNlpArchitecture.Gru, tokenizer)
	sampler = oa.OaNlpSuiteSampler(recipe, 2)

	assert recipe.VocabSize() == vocab
	assert recipe.ContextLength() == oa.OaNlpSuiteContextLength
	assert recipe.ModelWidth() == oa.OaNlpSuiteModelWidth
	assert recipe.HiddenWidth() == oa.OaNlpSuiteHiddenWidth
	assert sampler.Decode(sampler.Encode(sampler.Corpus())) == sampler.Corpus()


def test_bpe_reduces_canonical_corpus_positions():
	recipe = oa.OaNlpSuiteRecipe(
		oa.OaNlpArchitecture.Transformer,
		oa.OaNlpTokenizerKind.Bpe,
	)
	sampler = oa.OaNlpSuiteSampler(recipe, 1)
	corpus = sampler.Corpus()

	assert len(sampler.Encode(corpus)) < len(corpus.encode("utf-8"))


def test_byte_decode_preserves_non_utf8_tokens():
	recipe = oa.OaNlpSuiteRecipe(
		oa.OaNlpArchitecture.Rnn,
		oa.OaNlpTokenizerKind.Byte,
	)
	sampler = oa.OaNlpSuiteSampler(recipe, 1)

	assert sampler.DecodeBytes([0xA5]) == b"\xA5"
	assert sampler.Decode([0xA5]) == "\N{REPLACEMENT CHARACTER}"


def test_public_metric_namespace_is_installed():
	assert oa.OaFnMetric.Accuracy is oa.ml.MetricAccuracy
	assert oa.OaFnMetric.ScalarLoss is oa.ml.MetricScalarLoss
