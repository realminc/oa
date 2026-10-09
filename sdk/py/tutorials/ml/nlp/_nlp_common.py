"""Shared training loop for the OA Rust-parity NLP suite.

The models, samplers, and tokenizer are owned by the native extension; this
module owns the visible Python training loop so Rust and Python cannot
silently drift into separate implementations.
"""

from __future__ import annotations

import math
import os
import tempfile

import oa

# ── Canonical constants (mirrors oa::ml::nlp) ─────────────────────────────────
TRAINING_STEPS: int = 300
BATCH_SIZE: int = 64
CONTEXT_LENGTH: int = 16
CHAR_VOCAB_SIZE: int = 27
BPE_VOCAB_SIZE: int = 320
BPE_MERGES: int = BPE_VOCAB_SIZE - 256   # byte.VOCAB_SIZE = 256
GENERATION_PROMPT: str = "to be"
GENERATION_LENGTH: int = 80
RNG_SEED: int = 20_260_714

# Canonical 576-character teaching corpus (mirrors oa::ml::nlp::CORPUS)
CORPUS: str = (
	"to be or not to be that is the question whether tis nobler in the mind "
	"to suffer the slings and arrows of outrageous fortune or to take arms "
	"against a sea of troubles and by opposing end them "
	"to be or not to be that is the question whether tis nobler in the mind "
	"to suffer the slings and arrows of outrageous fortune or to take arms "
	"against a sea of troubles and by opposing end them "
	"to be or not to be that is the question whether tis nobler in the mind "
	"to suffer the slings and arrows of outrageous fortune or to take arms "
	"against a sea of troubles and by opposing end them "
)


def configured_positive(name: str, default: int) -> int:
	"""Read a positive workload override; reject invalid recipes before training."""
	value = int(os.environ.get(name, default))
	if value <= 0:
		raise ValueError(f"{name} must be a positive integer")
	return value


def configured_batch() -> int:
	return configured_positive("OA_TUTORIAL_BATCH", BATCH_SIZE)


def configured_generation_units() -> int:
	return configured_positive("OA_TUTORIAL_GENERATION_UNITS", GENERATION_LENGTH)


def build_bpe_tokenizer() -> oa.ml.BpeTokenizer:
	"""Train and return the canonical BPE tokenizer over the teaching corpus."""
	tokenizer = oa.ml.BpeTokenizer(BPE_VOCAB_SIZE)
	tokenizer.train_text(CORPUS, BPE_MERGES)
	return tokenizer


def _count_parameters(params: list) -> int:
	return sum(p.data().num_elements for p in params)


def run_char(
	label: str,
	model: object,
	optimizer: object,
	engine: oa.Engine,
	steps: int | None = None,
) -> None:
	"""Train a character-level model and print evaluation + generation."""
	sampler = oa.ml.CharSampler(configured_batch())
	_run(label, model, optimizer, sampler, engine,
		 vocab_size=CHAR_VOCAB_SIZE,
		 steps=steps,
		 generate_fn=lambda m, e, t: oa.ml.nlp_generate_greedy(
			 e, m, GENERATION_PROMPT, configured_generation_units()
		 ))


def run_byte(
	label: str,
	model: object,
	optimizer: object,
	engine: oa.Engine,
	steps: int | None = None,
) -> None:
	"""Train a byte-level model and print evaluation + generation."""
	sampler = oa.ml.ByteSampler(configured_batch())
	_run(label, model, optimizer, sampler, engine,
		 vocab_size=256,
		 steps=steps,
		 generate_fn=lambda m, e, t: oa.ml.nlp_generate_bytes_greedy(
			 e, m, GENERATION_PROMPT.encode(), configured_generation_units()
		 ).decode("utf-8", errors="replace"))


def run_bpe(
	label: str,
	model: object,
	optimizer: object,
	tokenizer: oa.ml.BpeTokenizer,
	engine: oa.Engine,
	steps: int | None = None,
) -> None:
	"""Train a BPE-level model and print evaluation + generation."""
	corpus_tokens = tokenizer.encode_text(CORPUS)
	bytes_per_token = len(CORPUS) / max(1, len(corpus_tokens))
	vocab_size = tokenizer.vocab_size()
	num_merges = tokenizer.num_merges()
	sampler = oa.ml.BpeSampler(configured_batch(), tokenizer)
	_run(label, model, optimizer, sampler, engine,
		 vocab_size=vocab_size,
		 steps=steps,
		 tokenizer=tokenizer,
		 generate_fn=lambda m, e, t: oa.ml.nlp_generate_bpe_greedy(
			 e, m, t, GENERATION_PROMPT.encode(), configured_generation_units()
		 ).decode("utf-8", errors="replace"),
		 extra_header=(
			 f"tokenizer: byte BPE · vocab={vocab_size} "
			 f"(256 bytes + {num_merges} merges)\n"
			 f"compression: {len(CORPUS)} bytes → {len(corpus_tokens)} tokens "
			 f"· {bytes_per_token:.3f} byte/token"
		 ))


def _run(
	label: str,
	model: object,
	optimizer: object,
	sampler: object,
	engine: oa.Engine,
	vocab_size: int,
	steps: int | None,
	generate_fn,
	extra_header: str = "",
	tokenizer: oa.ml.BpeTokenizer | None = None,
) -> None:
	total = steps if steps is not None else configured_positive("OA_TUTORIAL_STEPS", TRAINING_STEPS)
	if total <= 0:
		raise ValueError("steps must be positive")
	batch = configured_batch()
	n_params = _count_parameters(model.all_parameters())
	lr_str = f"{optimizer.learning_rate:.4g}"

	print()
	print("╔══════════════════════════════════════════════════════════════════╗")
	title = f"OA Tutorial — {label}"
	print(f"║  {title:<66}  ║")
	print("╚══════════════════════════════════════════════════════════════════╝")
	print()
	if extra_header:
		print(extra_header)
	print(f"params: {n_params}    optimizer: {type(optimizer).__name__}(lr={lr_str})")
	print(f"training: {total} steps · batch={batch} · "
		  f"sequence={CONTEXT_LENGTH} tokens")
	print()

	initial_loss = 0.0
	last_x = last_y = None
	loss = None

	for step in range(total):
		last_x, last_y = sampler.next(engine)
		optimizer.zero_grad()
		with oa.ml.GradientTape() as tape:
			logits = model.forward(last_x)
			flat_y = last_y.reshape([batch * CONTEXT_LENGTH])
			loss = oa.ml.loss.cross_entropy(logits, flat_y)
			tape.backward(loss)
		optimizer.step()

		if step == 0:
			initial_loss = oa.ml.metric.scalar_loss(loss)

		if (step + 1) % 50 == 0 or step == total - 1:
			current = oa.ml.metric.scalar_loss(loss)
			print(f"  step {step + 1:>4}/{total}  loss {current:.6f}")

	final_loss = oa.ml.metric.scalar_loss(loss)
	logits = model.forward(last_x)
	flat_y = last_y.reshape([batch * CONTEXT_LENGTH])
	evaluation_loss = oa.ml.metric.scalar_loss(oa.ml.loss.cross_entropy(logits, flat_y))
	accuracy = oa.ml.metric.accuracy(logits, flat_y)
	generated = generate_fn(model, engine, tokenizer)

	print()
	print("Evaluation:")
	print(f"  Random-loss baseline ln({vocab_size}) = {math.log(vocab_size):.4f}")
	print(f"  loss: initial {initial_loss:.6f} → final {final_loss:.6f}")
	print(f"  post-update loss: {evaluation_loss:.6f}")
	print(f"  token accuracy: {accuracy * 100.0:.1f}%")
	print(f"\nGeneration:")
	print(f"  prompt:    {GENERATION_PROMPT!r}")
	print(f"  generated: {generated!r}")

	# Checkpoint round-trip
	tag = label.lower().replace(" ", "_").replace("-", "_").replace("·", "").strip("_")
	checkpoint_directory = tempfile.TemporaryDirectory(prefix="oa-py-nlp-")
	ckpt = os.path.join(checkpoint_directory.name, f"{tag}.oam")
	params = model.all_parameters()
	try:
		oa.ml.save_checkpoint(ckpt, params, optimizer)
		reloaded_tokenizer = None
		if tokenizer is not None:
			vocabulary = os.path.join(checkpoint_directory.name, "tokenizer.bpe")
			tokenizer.save(vocabulary)
			reloaded_tokenizer = oa.ml.BpeTokenizer(tokenizer.vocab_size())
			reloaded_tokenizer.load(vocabulary)
			assert reloaded_tokenizer.encode_text(CORPUS) == tokenizer.encode_text(CORPUS)
			assert reloaded_tokenizer.decode_text(reloaded_tokenizer.encode_text(CORPUS)) == CORPUS
		reloaded_model = type(model)(engine)
		reloaded_params = reloaded_model.all_parameters()
		reload_opt = type(optimizer)(reloaded_params, optimizer.learning_rate)
		oa.ml.load_checkpoint(engine, ckpt, reloaded_params, reload_opt)
		assert reload_opt.step_count == optimizer.step_count, "optimizer step mismatch after reload"
		reload_logits = reloaded_model.forward(last_x)
		reload_acc = oa.ml.metric.accuracy(reload_logits, flat_y)
		reload_gen = generate_fn(reloaded_model, engine, reloaded_tokenizer)
		assert abs(reload_acc - accuracy) < 0.001, "accuracy mismatch after reload"
		assert reload_gen == generated, "generation mismatch after reload"
		print(f"\nCheckpoint: accuracy {reload_acc * 100.0:.1f}%  "
			  f"optimizer step {reload_opt.step_count}  deterministic generation ✓")
	finally:
		checkpoint_directory.cleanup()

	assert initial_loss > 0.0
	assert math.isfinite(final_loss)
	assert math.isfinite(evaluation_loss)
	assert 0.0 <= accuracy <= 1.0
	assert generated.startswith(GENERATION_PROMPT)
	print("\n✓ All checks passed")
