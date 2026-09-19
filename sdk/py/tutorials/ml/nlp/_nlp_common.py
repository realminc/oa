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
	sampler = oa.ml.CharSampler(BATCH_SIZE)
	_run(label, model, optimizer, sampler, engine,
		 vocab_size=CHAR_VOCAB_SIZE,
		 steps=steps,
		 generate_fn=lambda m, e: oa.ml.nlp_generate_greedy(
			 e, m, GENERATION_PROMPT, GENERATION_LENGTH
		 ))


def run_byte(
	label: str,
	model: object,
	optimizer: object,
	engine: oa.Engine,
	steps: int | None = None,
) -> None:
	"""Train a byte-level model and print evaluation + generation."""
	sampler = oa.ml.ByteSampler(BATCH_SIZE)
	_run(label, model, optimizer, sampler, engine,
		 vocab_size=256,
		 steps=steps,
		 generate_fn=lambda m, e: oa.ml.nlp_generate_bytes_greedy(
			 e, m, GENERATION_PROMPT.encode(), GENERATION_LENGTH
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
	sampler = oa.ml.BpeSampler(BATCH_SIZE, tokenizer)
	_run(label, model, optimizer, sampler, engine,
		 vocab_size=vocab_size,
		 steps=steps,
		 generate_fn=lambda m, e: oa.ml.nlp_generate_bpe_greedy(
			 e, m, tokenizer, GENERATION_PROMPT.encode(), GENERATION_LENGTH
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
) -> None:
	total = steps if steps is not None else TRAINING_STEPS
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
	print(f"training: {total} steps · batch={BATCH_SIZE} · "
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
			flat_y = last_y.reshape([BATCH_SIZE * CONTEXT_LENGTH])
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
	flat_y = last_y.reshape([BATCH_SIZE * CONTEXT_LENGTH])
	accuracy = oa.ml.metric.accuracy(logits, flat_y)
	generated = generate_fn(model, engine)

	print()
	print("Evaluation:")
	print(f"  Random-loss baseline ln({vocab_size}) = {math.log(vocab_size):.4f}")
	print(f"  loss: initial {initial_loss:.6f} → final {final_loss:.6f}")
	print(f"  token accuracy: {accuracy * 100.0:.1f}%")
	print(f"\nGeneration:")
	print(f"  prompt:    {GENERATION_PROMPT!r}")
	print(f"  generated: {generated!r}")

	# Checkpoint round-trip
	tag = label.lower().replace(" ", "_").replace("-", "_").replace("·", "").strip("_")
	ckpt = os.path.join(tempfile.gettempdir(), f"oars_py_nlp_{tag}.oam")
	params = model.all_parameters()
	oa.ml.save_checkpoint(ckpt, params, optimizer)
	try:
		reload_opt = type(optimizer)(params, optimizer.learning_rate)
		oa.ml.load_checkpoint(engine, ckpt, params, reload_opt)
		reload_logits = model.forward(last_x)
		reload_acc = oa.ml.metric.accuracy(reload_logits, flat_y)
		reload_gen = generate_fn(model, engine)
		assert abs(reload_acc - accuracy) < 0.001, "accuracy mismatch after reload"
		assert reload_gen == generated, "generation mismatch after reload"
		print(f"\nCheckpoint: accuracy {reload_acc * 100.0:.1f}%  "
			  f"optimizer step {reload_opt.step_count}  deterministic generation ✓")
	finally:
		if os.path.isfile(ckpt):
			os.remove(ckpt)

	assert initial_loss > 0.0
	assert math.isfinite(final_loss)
	print("\n✓ All checks passed")
