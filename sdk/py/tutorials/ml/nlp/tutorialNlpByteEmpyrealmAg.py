#!/usr/bin/env python3
"""1:1 Python entry point for TutorialNlpByteEmpyrealmAg.cpp.

Empyrealm remains the sixteenth experimental regression member outside the
canonical 3-tokenizer × 5-architecture matrix.
"""

from __future__ import annotations

import math

# pyright: reportWildcardImportFromLibrary=false
from oa import *
import _nlpCommon as nlp


class ByteEmpyrealmLM:
	def __init__(self) -> None:
		self.core = EmpyrealmCore(
			256,
			NlpSuiteModelWidth,
			dState=32,
			expand=2,
			headDim=16,
			nGroups=1,
			ropeFraction=0.5,
			mimo=False,
			mimoRank=1,
			dtMin=0.001,
			dtMax=0.1,
			dtInitFloor=1e-4,
			aFloor=1e-4,
			outprojNorm=True,
		)
		self.head = Linear(NlpSuiteModelWidth, 256)
		for parameter in self.parameters():
			parameter.data.setRequiresGrad(True)

	def parameters(self):
		return (
			list(self.core.allParameterPtrs())
			+ list(self.head.allParameterPtrs())
		)

	def forward(self, tokens):
		return self.head.forward(self.core.forward(tokens))

	def save(self, prefix) -> None:
		self.core.save(prefix.string() + ".core.oam")
		self.head.save(prefix.string() + ".head.oam")

	def load(self, prefix) -> None:
		self.core.load(prefix.string() + ".core.oam")
		self.head.load(prefix.string() + ".head.oam")


def main() -> None:
	if not initComputeEngine():
		raise RuntimeError("OA compute engine initialization failed")

	steps = nlp.configuredSteps()
	batch = nlp.configuredBatch()
	generationUnits = nlp.configuredGenerationUnits()
	fullGate = (
		steps == NlpSuiteTrainingSteps
		and batch == NlpSuiteBatchSize
		and generationUnits == NlpSuiteGenerationSourceUnits
	)
	# The recipe supplies the canonical Byte sampler. Empyrealm owns a distinct
	# model but otherwise uses the same workload as the 15 controlled members.
	recipe = NlpSuiteRecipe(
		NlpArchitecture.Gru,
		NlpTokenizerKind.Byte,
	)
	FnMatrix.setRngSeed(NlpSuiteRngSeed)
	model = ByteEmpyrealmLM()
	parameters = model.parameters()
	optimizer = AdamW(parameters, 0.003)
	sampler = NlpSuiteSampler(recipe, batch)

	print("\n" + "=" * 72)
	print("  OA Python Tutorial — Byte Empyrealm · all-position LM")
	print("=" * 72)
	print(
		"Model: Byte embedding + Empyrealm SSM(state=32) + residual "
		"-> Linear(256)"
	)
	print(
		f"Params: {sum(p.data.numElements() for p in parameters)} · "
		"AdamW(lr=0.003)"
	)
	print(
		f"Training: {steps} steps · batch={batch} · "
		f"sequence={recipe.contextLength()} byte tokens"
	)

	config = ItTrainingConfig()
	config.totalSteps = steps
	config.batchSize = batch
	config.sequenceLength = recipe.contextLength()
	config.sequenceUnit = "token"
	config.sourceUnit = "byte"
	config.timerName = "empyrealm_ag_step"
	training = ItTraining(optimizer, config)

	initialLoss = 0.0
	x = y = None
	while not training.isDone():
		x, y = sampler.next()
		training.recordSourceUnits(sampler.lastSourceUnits())
		optimizer.zeroGrad()
		with GradientTape() as tape:
			logits = model.forward(x)
			loss = FnLoss.crossEntropy(
				logits,
				y.reshape([y.numElements()]),
			)
			tape.backward(loss)
		training.next(loss)
		if training.index() == 1:
			initialLoss = training.lastLoss()
	training.finish()

	if x is None or y is None:
		raise RuntimeError("Empyrealm training produced no batch")
	finalLoss = training.lastLoss()
	accuracy = 100.0 * FnMetric.accuracy(model.forward(x), y)
	generated = nlp._generateSliding(
		model,
		NlpSuiteSampler(recipe, 1),
		recipe,
		generationUnits,
	)

	checkpoint = Paths.temp() / "oa_py_nlp_byte_empyrealm"
	model.save(checkpoint)
	reloaded = ByteEmpyrealmLM()
	reloaded.load(checkpoint)
	reloadedAccuracy = 100.0 * FnMetric.accuracy(
		reloaded.forward(x), y
	)
	reloadedGeneration = nlp._generateSliding(
		reloaded,
		NlpSuiteSampler(recipe, 1),
		recipe,
		generationUnits,
	)

	print(
		f"\nEvaluation: baseline={math.log(256):.4f} · "
		f"initial CE={initialLoss:.4f} · final CE={finalLoss:.4f} · "
		f"accuracy={accuracy:.1f}%"
	)
	print(f"Generation: {generated!r}")
	assert initialLoss > 0.0
	assert math.isfinite(finalLoss)
	assert abs(reloadedAccuracy - accuracy) < 0.5
	assert reloadedGeneration == generated
	if fullGate:
		assert finalLoss < initialLoss
		assert accuracy > 30.0
	print("✓ All checks passed")


if __name__ == "__main__":
	main()
