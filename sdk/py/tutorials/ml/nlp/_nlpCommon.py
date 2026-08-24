"""Shared runner for the 1:1 OA Python/C++ NLP comparison suite.

The model, tokenizer, sampler, corpus, and recipe live in the OA SDK parity
bridge. Python only owns this visible training/evaluation loop, so the two
languages cannot silently drift into separate suite implementations.
"""

from __future__ import annotations

import math
import os

# pyright: reportWildcardImportFromLibrary=false
from oa import *


SUITE_MEMBERS = (
	"TutorialNlpByteRnnAg",
	"TutorialNlpByteGruAg",
	"TutorialNlpByteTransformerAg",
	"TutorialNlpByteMoeAg",
	"TutorialNlpByteMamba3Ag",
	"TutorialNlpByteEmpyrealmAg",
	"TutorialNlpBpeRnnAg",
	"TutorialNlpBpeGruAg",
	"TutorialNlpBpeTransformerAg",
	"TutorialNlpBpeMoeAg",
	"TutorialNlpBpeMamba3Ag",
	"TutorialNlpCharRnnAg",
	"TutorialNlpCharGruAg",
	"TutorialNlpCharTransformerAg",
	"TutorialNlpCharMoeAg",
	"TutorialNlpCharMamba3Ag",
)


def configuredSteps() -> int:
	return max(1, int(os.getenv("OA_TUTORIAL_STEPS", NlpSuiteTrainingSteps)))


def configuredBatch() -> int:
	return max(1, int(os.getenv("OA_TUTORIAL_BATCH", NlpSuiteBatchSize)))


def configuredGenerationUnits() -> int:
	return max(
		1,
		int(
			os.getenv(
				"OA_TUTORIAL_GENERATION_UNITS",
				NlpSuiteGenerationSourceUnits,
			)
		),
	)


def _qualityThreshold(recipe) -> float:
	if recipe.tokenizer() == NlpTokenizerKind.Char:
		return 50.0
	if recipe.tokenizer() == NlpTokenizerKind.Bpe:
		return 30.0
	if recipe.architecture() in (
		NlpArchitecture.Transformer,
		NlpArchitecture.MoeTransformer,
		NlpArchitecture.Mamba3,
	):
		return 30.0
	return 50.0


def _displayText(value: bytes) -> str:
	return value.decode("utf-8", errors="replace")


def _generateSliding(model, sampler, recipe, count: int) -> str:
	contextLen = recipe.contextLength()
	pad = 26 if recipe.tokenizer() == NlpTokenizerKind.Char else 0
	context = [pad] * contextLen
	promptTokens = sampler.encode(NlpSuiteGenerationPrompt)
	copyCount = min(len(promptTokens), contextLen)
	context[:copyCount] = promptTokens[:copyCount]
	filled = max(copyCount, 1)
	logitRow = filled - 1
	output = NlpSuiteGenerationPrompt.encode("utf-8")
	generatedUnits = 0

	for _ in range(count):
		if generatedUnits >= count:
			break
		logits = model.forward(sampler.inputMatrix(context))
		row = FnMatrix.slice(logits, 0, logitRow, logitRow + 1)
		token = int(FnMatrix.argmax(row.reshape([recipe.vocabSize()])))
		decoded = sampler.decodeBytes([token])
		output += decoded
		generatedUnits += len(decoded)

		if filled < contextLen:
			context[filled] = token
			filled += 1
			logitRow = filled - 1
		else:
			context = context[1:] + [token]
			logitRow = contextLen - 1

	target = len(NlpSuiteGenerationPrompt.encode("utf-8")) + count
	return _displayText(output[:target])


def _generateMamba3(model, sampler, recipe, count: int) -> str:
	model.resetGenerationState(1)

	logits = None
	for token in sampler.encode(NlpSuiteGenerationPrompt):
		logits = model.forwardGenerationStep(sampler.inputStepMatrix(token))
	if logits is None:
		raise RuntimeError("NLP generation prompt encoded to no tokens")

	output = NlpSuiteGenerationPrompt.encode("utf-8")
	generatedUnits = 0
	for index in range(count):
		if generatedUnits >= count:
			break
		token = int(
			FnMatrix.argmax(logits.reshape([recipe.vocabSize()]))
		)
		decoded = sampler.decodeBytes([token])
		output += decoded
		generatedUnits += len(decoded)
		if index + 1 < count and generatedUnits < count:
			logits = model.forwardGenerationStep(
				sampler.inputStepMatrix(token)
			)

	target = len(NlpSuiteGenerationPrompt.encode("utf-8")) + count
	return _displayText(output[:target])


def generate(model, sampler, recipe, count: int) -> str:
	# Match the desktop/mobile oracle: Mamba-3 uses its persistent step state;
	# the other four architectures use the causal full-window path.
	if (
		recipe.architecture() == NlpArchitecture.Mamba3
		and model.supportsStatefulGeneration()
	):
		return _generateMamba3(model, sampler, recipe, count)
	return _generateSliding(model, sampler, recipe, count)


def runSuiteMember(architecture, tokenizer) -> None:
	if not initComputeEngine():
		raise RuntimeError("OA compute engine initialization failed")

	recipe = NlpSuiteRecipe(architecture, tokenizer)
	steps = configuredSteps()
	batch = configuredBatch()
	generationUnits = configuredGenerationUnits()
	fullGate = (
		steps == NlpSuiteTrainingSteps
		and batch == NlpSuiteBatchSize
		and generationUnits == NlpSuiteGenerationSourceUnits
	)

	FnMatrix.setRngSeed(NlpSuiteRngSeed)
	model = NlpSuiteModel(recipe)
	parameters = model.allParameterPtrs()
	optimizer = AdamW(parameters, recipe.learningRate())
	sampler = NlpSuiteSampler(recipe, batch)

	print("\n" + "=" * 72)
	print(
		f"  OA Python Tutorial — {recipe.tokenizerName()} "
		f"{recipe.architectureName()} · all-position LM"
	)
	print("=" * 72)
	print(
		f"Tokenizer: {recipe.tokenizerName()} · vocab={recipe.vocabSize()} · "
		f"context={recipe.contextLength()}"
	)
	print(f"Model: {recipe.modelDescription()}")
	print(
		f"Params: {model.numParameters()} · AdamW(lr={recipe.learningRate():.3g})"
	)
	print(
		f"Training: {steps} steps · batch={batch} · "
		f"sequence={recipe.contextLength()} tokens"
	)

	config = ItTrainingConfig()
	config.totalSteps = steps
	config.batchSize = batch
	config.sequenceLength = recipe.contextLength()
	config.sequenceUnit = "token"
	config.sourceUnit = "byte"
	config.timerName = recipe.timerName()
	training = ItTraining(optimizer, config)

	initialLoss = 0.0
	x = y = None
	while not training.isDone():
		x, y = sampler.next()
		training.recordSourceUnits(sampler.lastSourceUnits())
		optimizer.zeroGrad()
		with GradientTape() as tape:
			logits = model.forward(x)
			targets = y.reshape([y.numElements()])
			loss = FnLoss.crossEntropy(logits, targets)
			tape.backward(loss)
		training.next(loss)
		if training.index() == 1:
			initialLoss = training.lastLoss()
	training.finish()

	if x is None or y is None:
		raise RuntimeError("NLP training produced no batch")
	finalLoss = training.lastLoss()
	accuracy = 100.0 * FnMetric.accuracy(model.forward(x), y)
	bytesPerToken = sampler.lastSourceUnits() / max(1, y.numElements())
	bitsPerByte = finalLoss / (
		math.log(2.0) * max(bytesPerToken, 1e-12)
	)
	generated = generate(
		model,
		NlpSuiteSampler(recipe, 1),
		recipe,
		generationUnits,
	)

	print("\nEvaluation:")
	print(
		f"  Random baseline ln({recipe.vocabSize()}) = "
		f"{math.log(recipe.vocabSize()):.4f}"
	)
	print(
		f"  Initial CE={initialLoss:.4f} · Final CE={finalLoss:.4f} · "
		f"Accuracy={accuracy:.1f}%"
	)
	print(
		f"  Source coverage={bytesPerToken:.3f} byte/token · "
		f"{bitsPerByte:.4f} bits/byte"
	)
	print(
		f"  Wall={training.wallMsPerStep():.3f} ms/step · "
		f"{training.wallSourceUnitsPerSecond() / 1000.0:.2f}K byte/s"
	)
	print(
		f"\nGeneration:\n  Prompt:    {NlpSuiteGenerationPrompt!r}\n"
		f"  Generated: {generated!r}"
	)

	checkpoint = Paths.temp() / (
		f"oa_py_nlp_{recipe.tokenizerId()}_{recipe.architectureId()}.oam"
	)
	model.save(checkpoint, optimizer)
	reloaded = NlpSuiteModel(recipe)
	reloadOptimizer = AdamW(
		reloaded.allParameterPtrs(), recipe.learningRate()
	)
	reloaded.load(checkpoint, reloadOptimizer)
	reloadedAccuracy = 100.0 * FnMetric.accuracy(
		reloaded.forward(x), y
	)
	reloadedGeneration = generate(
		reloaded,
		NlpSuiteSampler(recipe, 1),
		recipe,
		generationUnits,
	)

	assert initialLoss > 0.0
	assert math.isfinite(finalLoss)
	assert abs(reloadedAccuracy - accuracy) < 0.5
	assert reloadedGeneration == generated
	assert reloadOptimizer.getStep() == optimizer.getStep()
	if fullGate:
		assert finalLoss < initialLoss
		assert accuracy > _qualityThreshold(recipe)

	print(
		f"\nCheckpoint: accuracy {reloadedAccuracy:.1f}% · "
		f"optimizer step {reloadOptimizer.getStep()} · deterministic generation"
	)
	print("✓ All checks passed")
