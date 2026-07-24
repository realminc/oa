#!/usr/bin/env python3
"""1:1 Python entry point for TutorialNlpByteEmpyrealmAg.cpp.

Empyrealm remains the sixteenth experimental regression member outside the
canonical 3-tokenizer × 5-architecture matrix.
"""

from __future__ import annotations

import math

# pyright: reportWildcardImportFromLibrary=false
from oa import *
import _nlp_common as nlp


class ByteEmpyrealmLM:
	def __init__(self) -> None:
		self.core = OaEmpyrealmCore(
			256,
			OaNlpSuiteModelWidth,
			DState=32,
			Expand=2,
			HeadDim=16,
			NGroups=1,
			RopeFraction=0.5,
			Mimo=False,
			MimoRank=1,
			DtMin=0.001,
			DtMax=0.1,
			DtInitFloor=1e-4,
			AFloor=1e-4,
			OutprojNorm=True,
		)
		self.head = OaLinear(OaNlpSuiteModelWidth, 256)
		for parameter in self.parameters():
			parameter.Data.SetRequiresGrad(True)

	def parameters(self):
		return (
			list(self.core.AllParameterPtrs())
			+ list(self.head.AllParameterPtrs())
		)

	def Forward(self, tokens):
		return self.head.Forward(self.core.Forward(tokens))

	def Save(self, prefix) -> None:
		self.core.Save(prefix.String() + ".core.oam")
		self.head.Save(prefix.String() + ".head.oam")

	def Load(self, prefix) -> None:
		self.core.Load(prefix.String() + ".core.oam")
		self.head.Load(prefix.String() + ".head.oam")


def main() -> None:
	if not OaInitComputeEngine():
		raise RuntimeError("OA compute engine initialization failed")

	steps = nlp.configured_steps()
	batch = nlp.configured_batch()
	generation_units = nlp.configured_generation_units()
	full_gate = (
		steps == OaNlpSuiteTrainingSteps
		and batch == OaNlpSuiteBatchSize
		and generation_units == OaNlpSuiteGenerationSourceUnits
	)
	# The recipe supplies the canonical Byte sampler. Empyrealm owns a distinct
	# model but otherwise uses the same workload as the 15 controlled members.
	recipe = OaNlpSuiteRecipe(
		OaNlpArchitecture.Gru,
		OaNlpTokenizerKind.Byte,
	)
	OaFnMatrix.SetRngSeed(OaNlpSuiteRngSeed)
	model = ByteEmpyrealmLM()
	parameters = model.parameters()
	optimizer = OaAdamW(parameters, 0.003)
	sampler = OaNlpSuiteSampler(recipe, batch)

	print("\n" + "=" * 72)
	print("  OA Python Tutorial — Byte Empyrealm · all-position LM")
	print("=" * 72)
	print(
		"Model: Byte embedding + Empyrealm SSM(state=32) + residual "
		"-> Linear(256)"
	)
	print(
		f"Params: {sum(p.Data.NumElements() for p in parameters)} · "
		"AdamW(lr=0.003)"
	)
	print(
		f"Training: {steps} steps · batch={batch} · "
		f"sequence={recipe.ContextLength()} byte tokens"
	)

	config = OaItTrainingConfig()
	config.TotalSteps = steps
	config.BatchSize = batch
	config.SequenceLength = recipe.ContextLength()
	config.SequenceUnit = "token"
	config.SourceUnit = "byte"
	config.TimerName = "empyrealm_ag_step"
	training = OaItTraining(optimizer, config)

	initial_loss = 0.0
	x = y = None
	while not training.IsDone():
		x, y = sampler.Next()
		training.RecordSourceUnits(sampler.LastSourceUnits())
		optimizer.ZeroGrad()
		with OaGradientTape() as tape:
			logits = model.Forward(x)
			loss = OaFnLoss.CrossEntropy(
				logits,
				y.Reshape([y.NumElements()]),
			)
			tape.Backward(loss)
		training.Next(loss)
		if training.Index() == 1:
			initial_loss = training.LastLoss()
	training.Finish()

	if x is None or y is None:
		raise RuntimeError("Empyrealm training produced no batch")
	final_loss = training.LastLoss()
	accuracy = 100.0 * OaFnMetric.Accuracy(model.Forward(x), y)
	generated = nlp._generate_sliding(
		model,
		OaNlpSuiteSampler(recipe, 1),
		recipe,
		generation_units,
	)

	checkpoint = OaPaths.Temp() / "oa_py_nlp_byte_empyrealm"
	model.Save(checkpoint)
	reloaded = ByteEmpyrealmLM()
	reloaded.Load(checkpoint)
	reloaded_accuracy = 100.0 * OaFnMetric.Accuracy(
		reloaded.Forward(x), y
	)
	reloaded_generation = nlp._generate_sliding(
		reloaded,
		OaNlpSuiteSampler(recipe, 1),
		recipe,
		generation_units,
	)

	print(
		f"\nEvaluation: baseline={math.log(256):.4f} · "
		f"initial CE={initial_loss:.4f} · final CE={final_loss:.4f} · "
		f"accuracy={accuracy:.1f}%"
	)
	print(f"Generation: {generated!r}")
	assert initial_loss > 0.0
	assert math.isfinite(final_loss)
	assert abs(reloaded_accuracy - accuracy) < 0.5
	assert reloaded_generation == generated
	if full_gate:
		assert final_loss < initial_loss
		assert accuracy > 30.0
	print("✓ All checks passed")


if __name__ == "__main__":
	main()
