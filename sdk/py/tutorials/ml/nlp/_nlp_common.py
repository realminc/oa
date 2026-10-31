"""Shared training loops for the OA Rust-parity NLP suite.

Mirrors sdk/rs/tutorials/ml/nlp/_byte_common.rs, _bpe_common.rs, and the
standalone Char tutorials: every loop is driven by `oa.ml.ItTraining` so the
same lifecycle, progress bar, training summary, metrics log, whole-program
capture, and program diagnostics apply in both languages.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass

import oa
import oa.ml.nlp as nlp

# Match the native tutorials' unbuffered interleaving with ProgressBar output.
sys.stdout.reconfigure(line_buffering=True)

# ── Workload overrides (OA_TUTORIAL_* env + --steps/--batch/--lr) ─────────────


def _configured_positive(name: str, default: int) -> int:
	"""Read a positive workload override; reject invalid recipes before training."""
	value = int(os.environ.get(name, default))
	if value <= 0:
		raise ValueError(f"{name} must be a positive integer")
	return value


def _steps() -> int:
	return _configured_positive("OA_TUTORIAL_STEPS", nlp.TRAINING_STEPS)


def _batch() -> int:
	return _configured_positive("OA_TUTORIAL_BATCH", nlp.BATCH_SIZE)


def _generation_units() -> int:
	return _configured_positive("OA_TUTORIAL_GENERATION_UNITS", nlp.GENERATION_LENGTH)


def _default_recipe(steps: int, batch: int, lr: float, default_lr: float) -> bool:
	return steps == nlp.TRAINING_STEPS and batch == nlp.BATCH_SIZE and lr == default_lr


def _num_parameters(model) -> int:
	return sum(p.data().num_elements for p in model.all_parameters())


def _print_header(title: str) -> None:
	print()
	print("╔══════════════════════════════════════════════════════════════════╗")
	print(f"║  {title:<62}║")
	print("╚══════════════════════════════════════════════════════════════════╝")
	print()


def _train(
	training: oa.ml.ItTraining,
	model,
	sampler,
	engine: oa.Engine,
	batch: int,
	steps: int,
	input_dtype: str,
	metrics: oa.LogMetrics | None = None,
	source_units=None,
) -> tuple[float, list, list]:
	"""Drive the ItTraining step loop; return (initial_loss, last_input, last_target)."""
	new_input = oa.Matrix.new_from_u8 if input_dtype == "u8" else oa.Matrix.new_from_u32
	initial_loss = 0.0
	last_input: list = []
	last_target: list = []
	for _ in range(steps):
		if source_units is not None:
			training.record_source_units(source_units())

		def prepare():
			nonlocal last_input, last_target
			last_input, last_target = sampler.next_values()
			return (
				new_input(engine, [batch, nlp.CONTEXT_LENGTH], last_input),
				oa.Matrix.new_from_u32(engine, [batch * nlp.CONTEXT_LENGTH], last_target),
			)

		def record(prepared):
			input_matrix, target = prepared
			with oa.ml.GradientTape() as tape:
				logits = model.forward(input_matrix)
				loss = oa.ml.loss.cross_entropy(logits, target)
				tape.backward(loss)
			return loss

		assert training.step(prepare, record)
		snap = training.snapshot()
		if snap.step_count == 1:
			initial_loss = snap.last_loss or 0.0
		if metrics is not None and snap.last_loss is not None:
			metrics.log_scalar("loss", snap.step_count, snap.last_loss)
	return initial_loss, last_input, last_target


def _open_metrics(log_dir_leaf: str) -> oa.LogMetrics:
	metrics = oa.LogMetrics()
	try:
		metrics.open(oa.Path.var_rel(log_dir_leaf))
	except Exception as e:
		print(f"[OA] metrics unavailable: {e}", file=sys.stderr)
	return metrics


def _eval_input(engine: oa.Engine, dtype: str, batch: int, values: list) -> oa.Matrix:
	new_input = oa.Matrix.new_from_u8 if dtype == "u8" else oa.Matrix.new_from_u32
	return new_input(engine, [batch, nlp.CONTEXT_LENGTH], values)


# ── Byte workload (mirrors _byte_common.rs) ───────────────────────────────────


@dataclass
class Tutorial:
	"""Mirrors the shared `Tutorial` struct of _byte_common.rs/_bpe_common.rs."""

	title: str
	description: str
	timer_name: str
	checkpoint_stem: str
	learning_rate: float
	expected_loss: float
	expected_accuracy: float
	expected_generation: bytes | str
	loss_tolerance: float = 0.001


def run_byte(tutorial: Tutorial, construct) -> None:
	steps = _steps()
	batch = _batch()
	engine = oa.Engine()
	model = construct(engine)
	# Complete asynchronous parameter initialization before isolated capture.
	engine.checkpoint().wait()
	optimizer = oa.ml.AdamW(model.all_parameters(), learning_rate=tutorial.learning_rate)
	sampler = nlp.ByteSampler(batch)

	_print_header(tutorial.title)
	print(f"tokenizer: raw bytes · vocab={oa.ml.byte.VOCAB_SIZE}")
	print("Task: dense next-byte prediction at every position\n")
	print(f"Model: {tutorial.description}")
	print(
		f"params: {_num_parameters(model)}    "
		f"Optimizer: AdamW(lr={tutorial.learning_rate})\n"
	)

	loss_metric = oa.ml.LossMetric()
	progress = oa.ml.ProgressBar()
	summary_callback = oa.ml.TrainingSummary()
	training = oa.ml.ItTraining(
		engine,
		optimizer,
		oa.ml.ItTrainingConfig(
			total_steps=steps,
			batch_size=batch,
			sequence_length=nlp.CONTEXT_LENGTH,
			sequence_unit="byte token",
			source_unit="byte",
			source_units_per_sample=nlp.CONTEXT_LENGTH,
			timer_name=tutorial.timer_name,
			enable_gpu_timing=True,
		),
	)
	training.add_metric(loss_metric)
	training.add_callback(progress)
	training.add_callback(summary_callback)

	metrics = _open_metrics(f"log/nlp/{tutorial.timer_name}")
	initial_loss, last_input, last_target = _train(
		training, model, sampler, engine, batch, steps, "u8", metrics
	)
	program_diagnostics = training.program_diagnostics()
	result = training.finish()
	final_training_loss = result.last_loss or 0.0

	input_matrix = _eval_input(engine, "u8", batch, last_input)
	target = oa.Matrix.new_from_u32(engine, [batch * nlp.CONTEXT_LENGTH], last_target)
	logits = model.forward(input_matrix)
	accuracy = nlp.accuracy(logits, target)
	evaluation_loss = oa.ml.loss.cross_entropy(logits, target).read_f32()[0]
	generated = nlp.generate_bytes_greedy(
		engine, model, nlp.GENERATION_PROMPT.encode(), _generation_units()
	).decode("utf-8", errors="replace")

	print("\nEvaluation:")
	print(
		f"  Random-loss baseline ln({oa.ml.byte.VOCAB_SIZE}) = "
		f"{math.log(oa.ml.byte.VOCAB_SIZE):.4f}"
	)
	print(f"  byte-token accuracy: {accuracy * 100.0:.1f}%")
	print(f"  post-update evaluation loss: {evaluation_loss:.6f}")
	print(
		f'\nGeneration:\n  prompt: "{nlp.GENERATION_PROMPT}"\n'
		f'  generated: "{generated}"'
	)
	if program_diagnostics is not None:
		print(
			f"program: {program_diagnostics.node_count} nodes · "
			f"{program_diagnostics.command_recording_count} recording · "
			f"{program_diagnostics.command_cache_hit_count} cache hits · "
			f"{program_diagnostics.input_upload_count} uploads"
		)

	checkpoint = oa.Path.temp().join(f"{tutorial.checkpoint_stem}-{os.getpid()}.oam")
	params = model.all_parameters()
	oa.ml.save_checkpoint(str(checkpoint), params, optimizer)
	reloaded = construct(engine)
	reloaded_optimizer = oa.ml.AdamW(
		reloaded.all_parameters(), learning_rate=tutorial.learning_rate
	)
	oa.ml.load_checkpoint(
		engine, str(checkpoint), reloaded.all_parameters(), reloaded_optimizer
	)
	assert reloaded_optimizer.step_count == optimizer.step_count
	assert nlp.accuracy(reloaded.forward(input_matrix), target) == accuracy
	assert (
		nlp.generate_bytes_greedy(
			engine, reloaded, nlp.GENERATION_PROMPT.encode(), _generation_units()
		).decode("utf-8", errors="replace")
		== generated
	)
	oa.Filesystem.remove_file(checkpoint)
	print("checkpoint: optimizer step, accuracy, and exact generation restored")

	if _default_recipe(steps, batch, tutorial.learning_rate, tutorial.learning_rate):
		assert initial_loss > 5.0
		assert final_training_loss < initial_loss
		assert abs(evaluation_loss - tutorial.expected_loss) < tutorial.loss_tolerance
		assert abs(accuracy - tutorial.expected_accuracy) < 0.001
		expected = tutorial.expected_generation
		assert generated == (
			expected.decode("utf-8") if isinstance(expected, bytes) else expected
		)


# ── BPE workload (mirrors _bpe_common.rs) ─────────────────────────────────────


def run_bpe(tutorial: Tutorial, construct) -> None:
	steps = _steps()
	batch = _batch()
	engine = oa.Engine()
	tokenizer = oa.ml.BpeTokenizer(nlp.BPE_VOCAB_SIZE)
	tokenizer.train_text(nlp.CORPUS, nlp.BPE_MERGES)
	corpus_tokens = tokenizer.encode_text(nlp.CORPUS)
	assert tokenizer.decode_text(corpus_tokens) == nlp.CORPUS
	bytes_per_token = len(nlp.CORPUS) / len(corpus_tokens)
	model = construct(engine)
	optimizer = oa.ml.AdamW(model.all_parameters(), learning_rate=tutorial.learning_rate)
	sampler = nlp.BpeSampler(batch, tokenizer)

	_print_header(tutorial.title)
	print(
		f"tokenizer: byte BPE · vocab={tokenizer.vocab_size()} "
		f"(256 bytes + {tokenizer.num_merges()} merges)"
	)
	print(
		f"compression: {len(nlp.CORPUS)} bytes → {len(corpus_tokens)} tokens "
		f"· {bytes_per_token:.3f} byte/token "
		f"· {100.0 * (1.0 - len(corpus_tokens) / len(nlp.CORPUS)):.1f}% fewer positions"
	)
	print(
		f"context coverage: {nlp.CONTEXT_LENGTH} BPE tokens ≈ "
		f"{nlp.CONTEXT_LENGTH * bytes_per_token:.1f} source bytes at corpus average"
	)
	print("Task: dense next-token prediction at every position\n")
	print(f"Model: {tutorial.description}")
	print(
		f"params: {_num_parameters(model)}    "
		f"Optimizer: AdamW(lr={tutorial.learning_rate})\n"
	)

	loss_metric = oa.ml.LossMetric()
	progress = oa.ml.ProgressBar()
	summary_callback = oa.ml.TrainingSummary()
	training = oa.ml.ItTraining(
		engine,
		optimizer,
		oa.ml.ItTrainingConfig(
			total_steps=steps,
			batch_size=batch,
			sequence_length=nlp.CONTEXT_LENGTH,
			sequence_unit="BPE token",
			source_unit="byte",
			timer_name=tutorial.timer_name,
			enable_gpu_timing=True,
		),
	)
	training.add_metric(loss_metric)
	training.add_callback(progress)
	training.add_callback(summary_callback)

	initial_loss, last_input, last_target = _train(
		training,
		model,
		sampler,
		engine,
		batch,
		steps,
		"u32",
		source_units=lambda: sampler.next_batch_bytes(),
	)
	program_diagnostics = training.program_diagnostics()
	result = training.finish()
	final_training_loss = result.last_loss or 0.0

	input_matrix = _eval_input(engine, "u32", batch, last_input)
	target = oa.Matrix.new_from_u32(engine, [batch * nlp.CONTEXT_LENGTH], last_target)
	logits = model.forward(input_matrix)
	accuracy = nlp.accuracy(logits, target)
	evaluation_loss = oa.ml.loss.cross_entropy(logits, target).read_f32()[0]
	bytes_per_token_last = sampler.last_batch_bytes_per_token()
	generated = nlp.generate_bpe_greedy(
		engine, model, tokenizer, nlp.GENERATION_PROMPT.encode(), _generation_units()
	).decode("utf-8", errors="replace")

	print("\nEvaluation:")
	print(
		f"  Random-loss baseline ln({tokenizer.vocab_size()}) = "
		f"{math.log(tokenizer.vocab_size()):.4f}"
	)
	print(
		f"  final batch: {bytes_per_token_last:.3f} byte/token · "
		f"{final_training_loss / math.log(2) / bytes_per_token_last:.4f} bits/byte"
	)
	print(f"  BPE-token accuracy: {accuracy * 100.0:.1f}%")
	print(f"  post-update evaluation loss: {evaluation_loss:.6f}")
	print(
		f'\nGeneration:\n  prompt: "{nlp.GENERATION_PROMPT}"\n'
		f'  generated: "{generated}"'
	)
	if program_diagnostics is not None:
		print(
			f"program: {program_diagnostics.node_count} nodes · "
			f"{program_diagnostics.command_recording_count} recording · "
			f"{program_diagnostics.command_cache_hit_count} cache hits · "
			f"{program_diagnostics.input_upload_count} uploads"
		)

	if _default_recipe(steps, batch, tutorial.learning_rate, tutorial.learning_rate):
		assert initial_loss > 5.0
		assert final_training_loss < initial_loss
		assert abs(evaluation_loss - tutorial.expected_loss) < 0.001
		assert abs(accuracy - tutorial.expected_accuracy) < 0.001
		expected = tutorial.expected_generation
		assert generated == (
			expected.decode("utf-8") if isinstance(expected, bytes) else expected
		)

	base = oa.Path.temp().join(tutorial.checkpoint_stem)
	checkpoint = base.with_extension("oam")
	vocabulary = base.with_extension("bpe")
	params = model.all_parameters()
	oa.ml.save_checkpoint(str(checkpoint), params, optimizer)
	tokenizer.save(str(vocabulary))
	reloaded_tokenizer = oa.ml.BpeTokenizer(256)
	reloaded_tokenizer.load(str(vocabulary))
	reloaded = construct(engine)
	reloaded_optimizer = oa.ml.AdamW(
		reloaded.all_parameters(), learning_rate=tutorial.learning_rate
	)
	oa.ml.load_checkpoint(
		engine, str(checkpoint), reloaded.all_parameters(), reloaded_optimizer
	)
	assert reloaded_optimizer.step_count == optimizer.step_count
	assert nlp.accuracy(reloaded.forward(input_matrix), target) == accuracy
	assert (
		nlp.generate_bpe_greedy(
			engine,
			reloaded,
			reloaded_tokenizer,
			nlp.GENERATION_PROMPT.encode(),
			_generation_units(),
		).decode("utf-8", errors="replace")
		== generated
	)
	oa.Filesystem.remove_file(checkpoint)
	oa.Filesystem.remove_file(vocabulary)


# ── Char workload (mirrors the standalone tu_nlp_char_* tutorials) ────────────


@dataclass
class CharTutorial:
	"""Per-file literals of the standalone Rust Char tutorials."""

	prog: str
	cli_title: str
	header_line: str
	model_line: str
	sequence_unit: str
	timer_name: str
	log_dir_leaf: str
	checkpoint_name: str
	default_lr: float
	show_training_line: bool
	eval_loss_line: bool
	accuracy_note: bool
	expected_generation: str
	loss_tolerance: float | None = None
	accuracy_tolerance: float | None = None
	expected_loss: float = 0.0
	expected_accuracy: float = 0.0
	final_loss_max: float | None = None
	accuracy_min: float | None = None


def run_char(tutorial: CharTutorial, construct) -> None:
	parser = argparse.ArgumentParser(prog=tutorial.prog, description=tutorial.cli_title)
	parser.add_argument("--steps", type=int, default=0, help="training steps")
	parser.add_argument("--batch", type=int, default=0, help="batch size")
	parser.add_argument("--lr", type=float, default=0.0, help="learning rate")
	args = parser.parse_args()
	steps = args.steps if args.steps > 0 else _steps()
	batch = args.batch if args.batch > 0 else _batch()
	lr = args.lr if args.lr > 0.0 else tutorial.default_lr

	engine = oa.Engine()
	model = construct(engine)
	optimizer = oa.ml.AdamW(model.all_parameters(), learning_rate=lr)
	sampler = nlp.CharSampler(batch)

	print()
	print("╔══════════════════════════════════════════════════════════════════╗")
	print(tutorial.header_line)
	print("╚══════════════════════════════════════════════════════════════════╝")
	print()
	print(f"tokenizer: character · vocab={nlp.CHAR_VOCAB_SIZE} (a-z + space)")
	print("Task: dense next-character prediction at every position\n")
	print(tutorial.model_line)
	print(f"params: {_num_parameters(model)}    Optimizer: AdamW(lr={lr})\n")
	if tutorial.show_training_line:
		print(
			f"training: {steps} steps · batch={batch} · "
			f"sequence={nlp.CONTEXT_LENGTH} character tokens"
		)

	loss_metric = oa.ml.LossMetric()
	progress = oa.ml.ProgressBar()
	summary_callback = oa.ml.TrainingSummary()
	training = oa.ml.ItTraining(
		engine,
		optimizer,
		oa.ml.ItTrainingConfig(
			total_steps=steps,
			batch_size=batch,
			sequence_length=nlp.CONTEXT_LENGTH,
			sequence_unit=tutorial.sequence_unit,
			timer_name=tutorial.timer_name,
			enable_gpu_timing=True,
		),
	)
	training.add_metric(loss_metric)
	training.add_callback(progress)
	training.add_callback(summary_callback)

	metrics = _open_metrics(tutorial.log_dir_leaf)
	initial_loss, last_input, last_target = _train(
		training, model, sampler, engine, batch, steps, "u32", metrics
	)
	program_diagnostics = training.program_diagnostics()
	result = training.finish()
	final_loss = result.last_loss or 0.0

	input_matrix = _eval_input(engine, "u32", batch, last_input)
	target = oa.Matrix.new_from_u32(engine, [batch * nlp.CONTEXT_LENGTH], last_target)
	logits = model.forward(input_matrix)
	accuracy = nlp.accuracy(logits, target)
	evaluation_loss = (
		oa.ml.loss.cross_entropy(logits, target).read_f32()[0]
		if tutorial.eval_loss_line
		else 0.0
	)
	generated = nlp.generate_greedy(
		engine, model, nlp.GENERATION_PROMPT, _generation_units()
	)

	print("\nEvaluation:")
	print(
		f"  Random-loss baseline ln({nlp.CHAR_VOCAB_SIZE}) = "
		f"{math.log(nlp.CHAR_VOCAB_SIZE):.4f}"
	)
	if tutorial.eval_loss_line:
		print(
			f"  loss: evaluation {evaluation_loss:.6f} · "
			f"last training {final_loss:.6f}"
		)
	note = " (compare within Char only)" if tutorial.accuracy_note else ""
	print(f"  character-token accuracy: {accuracy * 100.0:.1f}%{note}")
	print(
		f'\nGeneration:\n  prompt: "{nlp.GENERATION_PROMPT}"\n'
		f'  generated: "{generated}"'
	)
	if program_diagnostics is not None:
		print(
			f"program: {program_diagnostics.node_count} nodes · "
			f"{program_diagnostics.command_recording_count} recording · "
			f"{program_diagnostics.command_cache_hit_count} cache hits · "
			f"{program_diagnostics.input_upload_count} uploads"
		)

	if _default_recipe(steps, batch, lr, tutorial.default_lr):
		assert initial_loss > 3.0
		if tutorial.loss_tolerance is not None:
			assert abs(evaluation_loss - tutorial.expected_loss) < tutorial.loss_tolerance
		if tutorial.accuracy_tolerance is not None:
			assert abs(accuracy - tutorial.expected_accuracy) < tutorial.accuracy_tolerance
		if tutorial.final_loss_max is not None:
			assert final_loss < tutorial.final_loss_max
		if tutorial.accuracy_min is not None:
			assert accuracy > tutorial.accuracy_min
		assert generated == tutorial.expected_generation

	checkpoint = oa.Path.temp().join(tutorial.checkpoint_name)
	params = model.all_parameters()
	oa.ml.save_checkpoint(str(checkpoint), params, optimizer)
	reloaded = construct(engine)
	reloaded_optimizer = oa.ml.AdamW(reloaded.all_parameters(), learning_rate=lr)
	oa.ml.load_checkpoint(
		engine, str(checkpoint), reloaded.all_parameters(), reloaded_optimizer
	)
	assert reloaded_optimizer.step_count == optimizer.step_count
	assert nlp.accuracy(reloaded.forward(input_matrix), target) == accuracy
	assert (
		nlp.generate_greedy(engine, reloaded, nlp.GENERATION_PROMPT, _generation_units())
		== generated
	)
	oa.Filesystem.remove_file(checkpoint)
