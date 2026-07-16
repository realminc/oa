"""Shared scaffolding for the OA Python NLP fair-comparison suite.

Python port of Tutorial/Ml/TutorialNlpCommon.h. Every NLP tutorial trains the SAME
all-position next-token task on the SAME corpus with the SAME dims, so their
loss/accuracy curves are directly comparable and the set doubles as a regression
test for the recurrent (and, later, attention/SSM) autograd paths through Python.

The one axis that legitimately differs is the *vocabulary*:
  - Byte tutorials  -> byte vocab (256), OaByteEmbedding. Universal, no tokenizer.
  - Char tutorials  -> tiny tokenizer vocab (27 = a-z + space), OaEmbedding.
Loss scales with ln(vocab), so compare within a vocab family; the corpus text is
identical (lowercase + spaces) so it is valid for both.
"""

from __future__ import annotations

import array

from _oa_import import core, ml, oa, runtime


# ─── Shared hyperparameters (identical across the whole suite) ───────────────

CONTEXT_LEN = 16   # sequence window S
D_MODEL = 32       # embedding / model width
HIDDEN_DIM = 64    # recurrent hidden / FFN inner width
STEPS = 300        # training steps
BATCH = 64         # batch size

BYTE_VOCAB_SIZE = 256
CHAR_VOCAB_SIZE = 27


# ─── Shared corpus ───────────────────────────────────────────────────────────
# Lowercase letters + spaces only, so the exact same text is a valid stream for
# both the byte vocab (raw bytes) and the 27-symbol char vocab.

CORPUS = (
    "hello world hello oa hello vulkan hello model "
    "byte level text generation tutorial trains next token prediction "
    "no tokenizer no vocabulary just bytes universal fast "
    "hello world hello oa hello vulkan hello model "
)


def char_encode(ch: str) -> int:
    """a-z -> 0..25, everything else (space/unknown) -> 26."""
    o = ord(ch)
    if ord("a") <= o <= ord("z"):
        return o - ord("a")
    if ord("A") <= o <= ord("Z"):
        return o - ord("A")
    return 26


def char_decode(token: int) -> str:
    return chr(ord("a") + token) if 0 <= token < 26 else " "


# ─── All-position batch sampler ──────────────────────────────────────────────
# Produces X=[batch, S] and Y=[batch, S] where Y is X shifted by one (the dense
# next-token targets). Pass encode=None for raw byte tokens, or char_encode for
# the 27-symbol char vocab.

class NlpAllPositionSampler:
    def __init__(self, text: str, batch_size: int, encode=None):
        self.batch_size = batch_size
        self.encode_fn = encode
        self.tokens = [self._encode(c) for c in text]
        self.cursor = 0

    def _encode(self, ch: str) -> int:
        return self.encode_fn(ch) if self.encode_fn else (ord(ch) & 0xFF)

    def next_batch(self):
        """Returns (X, Y) as [batch, S] UInt8 matrices; Y is X shifted by one."""
        b, s = self.batch_size, CONTEXT_LEN
        limit = len(self.tokens) - s - 1
        x = array.array("B", bytes(b * s))
        y = array.array("B", bytes(b * s))
        for bi in range(b):
            start = (self.cursor + bi * 7) % limit
            for t in range(s):
                x[bi * s + t] = self.tokens[start + t]
                y[bi * s + t] = self.tokens[start + t + 1]
        self.cursor = (self.cursor + b) % limit
        xm = core.FromBytes(x, b, s, core.OaScalarType.UInt8)
        ym = core.FromBytes(y, b, s, core.OaScalarType.UInt8)
        return xm, ym

    def encode_prompt_left(self, prompt: str) -> list[int]:
        """Left-aligned prompt ids at positions 0.., padded with 0 (byte) / 26 (char)."""
        pad = 26 if self.encode_fn else 0
        out = [pad] * CONTEXT_LEN
        for i, ch in enumerate(prompt[:CONTEXT_LEN]):
            out[i] = self._encode(ch)
        return out


# ─── Shared evaluation: all-position argmax accuracy ─────────────────────────
# model_forward(X) -> logits [B*S, V]; targets Y are [B, S]. Argmax at every position.

def accuracy_all_positions(model_forward, x, y, vocab: int) -> float:
    # Cast to fp32 for the host-side argmax (a no-op in fp32 mode; in bf16 the model
    # emits 2-byte logits that a float readback would misread as garbage).
    logits = core.Cast(model_forward(x), core.OaScalarType.Float32)
    ctx = runtime.OaContext_GetDefault()
    ctx.Execute()
    ctx.Sync()

    labels = core.CopyToHost(y)          # B*S ints
    flat = core.CopyToHost(logits)       # B*S*V floats
    positions = len(labels)
    correct = 0
    for i in range(positions):
        base = i * vocab
        best = 0
        best_val = flat[base]
        for j in range(1, vocab):
            v = flat[base + j]
            if v > best_val:
                best_val = v
                best = j
        if best == labels[i]:
            correct += 1
    return 100.0 * correct / positions if positions else 0.0


# ─── Shared generation: greedy, all-position ─────────────────────────────────
# Feeds a [1, S] window, reads the logit row at the last filled position, takes the
# argmax, appends it, and slides the window. Greedy (deterministic).

def generate_greedy(model_forward, prompt: str, count: int, vocab: int, encode=None) -> str:
    sampler = NlpAllPositionSampler("hello world", 1, encode)
    context = sampler.encode_prompt_left(prompt)
    filled = min(len(prompt), CONTEXT_LEN)
    logit_row = max(0, filled - 1)
    out = prompt

    ctx = runtime.OaContext_GetDefault()
    for _ in range(count):
        xm = core.FromBytes(array.array("B", context), 1, CONTEXT_LEN, core.OaScalarType.UInt8)
        logits = core.Cast(model_forward(xm), core.OaScalarType.Float32)
        ctx.Execute()
        ctx.Sync()

        flat = core.CopyToHost(logits)
        base = logit_row * vocab
        nxt = 0
        best_val = flat[base]
        for j in range(1, vocab):
            v = flat[base + j]
            if v > best_val:
                best_val = v
                nxt = j

        out += char_decode(nxt) if encode else chr(nxt & 0xFF)

        if filled < CONTEXT_LEN:
            context[filled] = nxt
            filled += 1
            logit_row = filled - 1
        else:
            context = context[1:] + [nxt]
            logit_row = CONTEXT_LEN - 1
    return out


# ─── Shared training loop (OaItTraining + implicit autograd) ─────────────────

def train_all_position(model_forward, params, sampler, *, steps=STEPS, lr=0.01,
                       batch=BATCH, timer_name="nlp_step"):
    """Runs the shared all-position training loop and returns (opt, initial, final)."""
    opt = ml.OaAdamW(params, lr)

    loss_metric = ml.OaMetricLoss()
    progress = ml.OaCbProgressBar()
    progress.AddMetric(loss_metric)

    config = ml.OaItTrainingConfig()
    config.TotalSteps = steps
    config.BatchSize = batch

    loop = ml.OaItTraining(opt, config)
    loop.AddMetric(loss_metric)
    loop.AddCallback(progress)

    initial_loss = 0.0
    last_x = last_y = None
    while not loop.IsDone():
        x, y = sampler.next_batch()
        last_x, last_y = x, y
        opt.ZeroGrad()                       # implicit autograd accumulates; clear each step
        tape = ml.GradientTape()
        logits = model_forward(x)
        loss = ml.CrossEntropy(logits, y.Reshape([y.Size(0) * y.Size(1)]))
        tape.Backward(loss)
        loop.Next(loss)
        if loop.Index() == 1:
            initial_loss = loop.LiveLoss()
    loop.Finish()

    # Wall + GPU timing from OaItTraining's own clocks (the same measures the C++ suite
    # reports), so Python-vs-C++ wall/GPU timing is directly comparable on the same
    # hardware. The wall-GPU gap is not a direct CPU measurement; asynchronous execution
    # and timer boundaries can make it negative.
    elapsed = loop.ElapsedSeconds()
    if elapsed > 0:
        wall_ms = 1000.0 * elapsed / steps
        print(f"Wall: {wall_ms:.2f} ms/step · {batch * steps / elapsed / 1000.0:.2f}K sps · "
              f"Steps: {steps}")
        gpu = loop.GpuTimingStats()
        if gpu.Count > 0 and gpu.MeanMs > 0.0:
            gpu_sps = batch / (gpu.MeanMs / 1000.0)   # same basis as the C++ suite
            wall_gpu_gap = 100.0 * (wall_ms - gpu.MeanMs) / wall_ms
            print(f"GPU:  {gpu.MeanMs:.3f} ms/step · {gpu_sps / 1000.0:.2f}K sps · "
                  f"wall-GPU gap: {wall_gpu_gap:.0f}%  (n={gpu.Count})")

    return opt, initial_loss, loop.LastLoss(), last_x, last_y
