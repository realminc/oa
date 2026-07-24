"""Shared runner for the 1:1 OA Python/C++ NLP comparison suite.

The model, tokenizer, sampler, corpus, and recipe live in the OA library. Python
only owns this visible training/evaluation loop, so the language binding cannot
silently drift into a second implementation of the suite.
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


def configured_steps() -> int:
	return max(1, int(os.getenv("OA_TUTORIAL_STEPS", OaNlpSuiteTrainingSteps)))


def configured_batch() -> int:
	return max(1, int(os.getenv("OA_TUTORIAL_BATCH", OaNlpSuiteBatchSize)))


def configured_generation_units() -> int:
	return max(
		1,
		int(
			os.getenv(
				"OA_TUTORIAL_GENERATION_UNITS",
				OaNlpSuiteGenerationSourceUnits,
			)
		),
	)


def _quality_threshold(recipe) -> float:
	if recipe.Tokenizer() == OaNlpTokenizerKind.Char:
		return 50.0
	if recipe.Tokenizer() == OaNlpTokenizerKind.Bpe:
		return 30.0
	if recipe.Architecture() in (
		OaNlpArchitecture.Transformer,
		OaNlpArchitecture.MoeTransformer,
		OaNlpArchitecture.Mamba3,
	):
		return 30.0
	return 50.0


def _display_text(value: bytes) -> str:
	return value.decode("utf-8", errors="replace")


def _generate_sliding(model, sampler, recipe, count: int) -> str:
	context_len = recipe.ContextLength()
	pad = 26 if recipe.Tokenizer() == OaNlpTokenizerKind.Char else 0
	context = [pad] * context_len
	prompt_tokens = sampler.Encode(OaNlpSuiteGenerationPrompt)
	copy_count = min(len(prompt_tokens), context_len)
	context[:copy_count] = prompt_tokens[:copy_count]
	filled = max(copy_count, 1)
	logit_row = filled - 1
	output = OaNlpSuiteGenerationPrompt.encode("utf-8")
	generated_units = 0

	for _ in range(count):
		if generated_units >= count:
			break
		logits = model.Forward(sampler.InputMatrix(context))
		row = OaFnMatrix.Slice(logits, 0, logit_row, logit_row + 1)
		token = int(OaFnMatrix.Argmax(row.Reshape([recipe.VocabSize()])))
		decoded = sampler.DecodeBytes([token])
		output += decoded
		generated_units += len(decoded)

		if filled < context_len:
			context[filled] = token
			filled += 1
			logit_row = filled - 1
		else:
			context = context[1:] + [token]
			logit_row = context_len - 1

	target = len(OaNlpSuiteGenerationPrompt.encode("utf-8")) + count
	return _display_text(output[:target])


def _generate_mamba3(model, sampler, recipe, count: int) -> str:
	context = OaContextGetDefault()
	model.ResetGenerationState(1)
	context.Execute()
	context.Sync()

	logits = None
	for token in sampler.Encode(OaNlpSuiteGenerationPrompt):
		logits = model.ForwardGenerationStep(sampler.InputStepMatrix(token))
		context.Execute()
		context.Sync()
	if logits is None:
		raise RuntimeError("NLP generation prompt encoded to no tokens")

	output = OaNlpSuiteGenerationPrompt.encode("utf-8")
	generated_units = 0
	for index in range(count):
		if generated_units >= count:
			break
		token = int(
			OaFnMatrix.Argmax(logits.Reshape([recipe.VocabSize()]))
		)
		decoded = sampler.DecodeBytes([token])
		output += decoded
		generated_units += len(decoded)
		if index + 1 < count and generated_units < count:
			logits = model.ForwardGenerationStep(
				sampler.InputStepMatrix(token)
			)
			context.Execute()
			context.Sync()

	target = len(OaNlpSuiteGenerationPrompt.encode("utf-8")) + count
	return _display_text(output[:target])


def generate(model, sampler, recipe, count: int) -> str:
	# Match the desktop/mobile oracle: Mamba-3 uses its persistent step state;
	# the other four architectures use the causal full-window path.
	if (
		recipe.Architecture() == OaNlpArchitecture.Mamba3
		and model.SupportsStatefulGeneration()
	):
		return _generate_mamba3(model, sampler, recipe, count)
	return _generate_sliding(model, sampler, recipe, count)


def run_suite_member(architecture, tokenizer) -> None:
	if not OaInitComputeEngine():
		raise RuntimeError("OA compute engine initialization failed")

	recipe = OaNlpSuiteRecipe(architecture, tokenizer)
	steps = configured_steps()
	batch = configured_batch()
	generation_units = configured_generation_units()
	full_gate = (
		steps == OaNlpSuiteTrainingSteps
		and batch == OaNlpSuiteBatchSize
		and generation_units == OaNlpSuiteGenerationSourceUnits
	)

	OaFnMatrix.SetRngSeed(OaNlpSuiteRngSeed)
	model = OaNlpSuiteModel(recipe)
	parameters = model.AllParameterPtrs()
	optimizer = OaAdamW(parameters, recipe.LearningRate())
	sampler = OaNlpSuiteSampler(recipe, batch)

	print("\n" + "=" * 72)
	print(
		f"  OA Python Tutorial — {recipe.TokenizerName()} "
		f"{recipe.ArchitectureName()} · all-position LM"
	)
	print("=" * 72)
	print(
		f"Tokenizer: {recipe.TokenizerName()} · vocab={recipe.VocabSize()} · "
		f"context={recipe.ContextLength()}"
	)
	print(f"Model: {recipe.ModelDescription()}")
	print(
		f"Params: {model.NumParameters()} · AdamW(lr={recipe.LearningRate():.3g})"
	)
	print(
		f"Training: {steps} steps · batch={batch} · "
		f"sequence={recipe.ContextLength()} tokens"
	)

	config = OaItTrainingConfig()
	config.TotalSteps = steps
	config.BatchSize = batch
	config.SequenceLength = recipe.ContextLength()
	config.SequenceUnit = "token"
	config.SourceUnit = "byte"
	config.TimerName = recipe.TimerName()
	training = OaItTraining(optimizer, config)

	initial_loss = 0.0
	x = y = None
	while not training.IsDone():
		x, y = sampler.Next()
		training.RecordSourceUnits(sampler.LastSourceUnits())
		optimizer.ZeroGrad()
		with OaGradientTape() as tape:
			logits = model.Forward(x)
			targets = y.Reshape([y.NumElements()])
			loss = OaFnLoss.CrossEntropy(logits, targets)
			tape.Backward(loss)
		training.Next(loss)
		if training.Index() == 1:
			initial_loss = training.LastLoss()
	training.Finish()

	if x is None or y is None:
		raise RuntimeError("NLP training produced no batch")
	final_loss = training.LastLoss()
	accuracy = 100.0 * OaFnMetric.Accuracy(model.Forward(x), y)
	bytes_per_token = sampler.LastSourceUnits() / max(1, y.NumElements())
	bits_per_byte = final_loss / (
		math.log(2.0) * max(bytes_per_token, 1e-12)
	)
	generated = generate(
		model,
		OaNlpSuiteSampler(recipe, 1),
		recipe,
		generation_units,
	)

	print("\nEvaluation:")
	print(
		f"  Random baseline ln({recipe.VocabSize()}) = "
		f"{math.log(recipe.VocabSize()):.4f}"
	)
	print(
		f"  Initial CE={initial_loss:.4f} · Final CE={final_loss:.4f} · "
		f"Accuracy={accuracy:.1f}%"
	)
	print(
		f"  Source coverage={bytes_per_token:.3f} byte/token · "
		f"{bits_per_byte:.4f} bits/byte"
	)
	print(
		f"  Wall={training.WallMsPerStep():.3f} ms/step · "
		f"{training.WallSourceUnitsPerSecond() / 1000.0:.2f}K byte/s"
	)
	print(
		f"\nGeneration:\n  Prompt:    {OaNlpSuiteGenerationPrompt!r}\n"
		f"  Generated: {generated!r}"
	)

	checkpoint = OaPaths.Temp() / (
		f"oa_py_nlp_{recipe.TokenizerId()}_{recipe.ArchitectureId()}.oam"
	)
	model.Save(checkpoint, optimizer)
	reloaded = OaNlpSuiteModel(recipe)
	reload_optimizer = OaAdamW(
		reloaded.AllParameterPtrs(), recipe.LearningRate()
	)
	reloaded.Load(checkpoint, reload_optimizer)
	reloaded_accuracy = 100.0 * OaFnMetric.Accuracy(
		reloaded.Forward(x), y
	)
	reloaded_generation = generate(
		reloaded,
		OaNlpSuiteSampler(recipe, 1),
		recipe,
		generation_units,
	)

	assert initial_loss > 0.0
	assert math.isfinite(final_loss)
	assert abs(reloaded_accuracy - accuracy) < 0.5
	assert reloaded_generation == generated
	assert reload_optimizer.GetStep() == optimizer.GetStep()
	if full_gate:
		assert final_loss < initial_loss
		assert accuracy > _quality_threshold(recipe)

	print(
		f"\nCheckpoint: accuracy {reloaded_accuracy:.1f}% · "
		f"optimizer step {reload_optimizer.GetStep()} · deterministic generation"
	)
	print("✓ All checks passed")
