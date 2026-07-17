#!/usr/bin/env python3
"""
OA Python Tutorial — Character Text Generation (Implicit Autograd)

Python port of Tutorial/Ml/Nlp/TutorialNlpRnnAg.cpp.

Model: Embedding → Reshape → Linear+Relu → Linear (head)
Loss: CrossEntropy
Optimizer: AdamW
Backward: Implicit autograd via OaGradientTape

Architecture:
  Vocab: 27 chars (a-z + space)
  Context: 8 chars
  Embed: 16
  Hidden: 64 (Linear + Relu)
  Head: 27 (Linear, no activation)
  Params: ~10,443
  Batch: 64
  Steps: 300
  Optimizer: AdamW(lr=0.01, weight_decay=0.01)

Key difference from manual backward version:
  - No hand-wired backward() method
  - Uses OaGradientTape to automatically compute gradients
  - Must call opt.ZeroGrad() each step (autograd accumulates)
  - SetRequiresGrad(True) on all parameters

Autograd parity:
  OaGradientTape in Python trains to parity with the manual-backward version
  (~98% batch accuracy, loss 3.3 → ~0.007), matching the C++ tutorial. An earlier
  binding bug — the tape's RAII destructor ran on the temporary, disabling gradient
  tracking before the forward pass recorded anything — has been fixed via the
  __enter__/__exit__ context-manager binding (see OaPython.md §4).
"""

import sys
import time
import array
import math

from _oa_import import core, ml, oa, runtime, set_grad_enabled

# ─── Hyperparameters ─────────────────────────────────────────────────────────

VOCAB_SIZE = 27
CONTEXT_LEN = 8
EMBED_DIM = 16
HIDDEN_DIM = 64
FLAT_DIM = CONTEXT_LEN * EMBED_DIM
BATCH_SIZE = 64
STEPS = 300
LR = 0.01

CORPUS = (
    "hello world hello oa hello vulkan hello model "
    "tiny text generation tutorial trains next token prediction "
    "hello world hello oa hello vulkan hello model "
)


# ─── Char vocabulary (a–z + space) ─────────────────────────────────────────

class TinyCharVocab:
    def encode(self, ch: str) -> int:
        if ch == " ":
            return 26
        if "a" <= ch <= "z":
            return ord(ch) - ord("a")
        if "A" <= ch <= "Z":
            return ord(ch) - ord("A")
        return 26

    def decode(self, token: int) -> str:
        if token == 26:
            return " "
        if 0 <= token < 26:
            return chr(ord("a") + token)
        return "?"


# ─── Batch sampler ─────────────────────────────────────────────────────────

class TextBatchSampler:
    def __init__(self, text: str, batch_size: int):
        self.vocab = TinyCharVocab()
        self.tokens = [self.vocab.encode(c) for c in text]
        self.batch_size = batch_size
        self.cursor = 0
        self.limit = len(self.tokens) - CONTEXT_LEN - 1

    def next_batch(self):
        x = array.array("B")
        y = array.array("B")
        for b in range(self.batch_size):
            start = (self.cursor + b * 7) % self.limit
            for t in range(CONTEXT_LEN):
                x.append(self.tokens[start + t])
            y.append(self.tokens[start + CONTEXT_LEN])
        self.cursor = (self.cursor + self.batch_size) % self.limit

        x_mat = core.FromBytes(x, self.batch_size, CONTEXT_LEN, core.OaScalarType.UInt8)
        y_mat = core.FromBytes(y, self.batch_size, core.OaScalarType.UInt8)
        return x_mat, y_mat

    def encode_prompt(self, prompt: str) -> array.array:
        out = array.array("B", [26] * CONTEXT_LEN)
        plen = len(prompt)
        for i in range(min(CONTEXT_LEN, plen)):
            out[CONTEXT_LEN - 1 - i] = self.vocab.encode(prompt[plen - 1 - i])
        return out


# ─── Model: Embedding → Reshape → Linear+Relu → Linear (head) ────────────────
# No manual backward — OaGradientTape handles gradient computation

class OaTextGenerationRnnAutograd:
    def __init__(self):
        wd = core.GetWeightDtype()
        
        # Embedding layer
        self.embed = ml.OaEmbedding(VOCAB_SIZE, EMBED_DIM)
        self._embed_p = self.embed.Parameters()
        self._embed_p[0].Data = core.RandN(VOCAB_SIZE, EMBED_DIM, wd)
        self._embed_p[0].Data.SetRequiresGrad(True)
        
        # Hidden layer (Linear + Relu)
        self.hidden = ml.OaLinear(FLAT_DIM, HIDDEN_DIM)
        self.hidden.SetActivation(ml.OaActivation.Relu)
        self._hidden_p = self.hidden.Parameters()
        self._hidden_p[0].Data = core.RandXavier(HIDDEN_DIM, FLAT_DIM, wd)
        self._hidden_p[0].Data.SetRequiresGrad(True)
        self._hidden_p[1].Data.SetRequiresGrad(True)
        
        # Head layer (output)
        self.head = ml.OaLinear(HIDDEN_DIM, VOCAB_SIZE)
        self._head_p = self.head.Parameters()
        self._head_p[0].Data = core.RandXavier(VOCAB_SIZE, HIDDEN_DIM, wd)
        self._head_p[0].Data.SetRequiresGrad(True)
        self._head_p[1].Data.SetRequiresGrad(True)

    def parameters(self):
        return self._embed_p + self._hidden_p + self._head_p

    def forward(self, tokens):
        """Forward pass — returns logits [B, V]."""
        batch = int(tokens.Size(0))
        
        emb = self.embed.Forward(tokens)                    # [B, S, D]
        flat = core.Reshape(emb, batch, FLAT_DIM)           # [B, S*D]
        hpost = self.hidden.Forward(flat)                   # [B, H] (LinearRelu fused)
        return self.head.Forward(hpost)                     # [B, V]


# ─── Inference helpers ───────────────────────────────────────────────────────

def argmax_row(probs, row):
    """Host-side argmax for a single row of a flat probability matrix."""
    best = 0
    best_val = probs[row * VOCAB_SIZE]
    for i in range(1, VOCAB_SIZE):
        v = probs[row * VOCAB_SIZE + i]
        if v > best_val:
            best_val = v
            best = i
    return best


def batch_accuracy(model, sampler):
    x, y = sampler.next_batch()
    logits = model.forward(x)
    probs = core.Softmax(logits, -1)
    ctx = runtime.OaContext_GetDefault()
    ctx.Execute()
    ctx.Sync()
    flat = core.CopyToHost(probs)
    labels = core.CopyToHost(y)
    correct = sum(
        1 for i in range(BATCH_SIZE)
        if argmax_row(flat, i) == labels[i]
    )
    return 100.0 * correct / BATCH_SIZE


def generate_greedy(model, prompt, n_chars):
    vocab = TinyCharVocab()
    sampler = TextBatchSampler("hello world", 1)
    context = sampler.encode_prompt(prompt)
    out = list(prompt)
    ctx = runtime.OaContext_GetDefault()

    for _ in range(n_chars):
        buf = array.array("B", context)
        x = core.FromBytes(buf, 1, CONTEXT_LEN, core.OaScalarType.UInt8)
        logits = model.forward(x)
        probs = core.Softmax(logits, -1)
        ctx.Execute()
        ctx.Sync()
        flat = core.CopyToHost(probs)
        nxt = argmax_row(flat, 0)
        out.append(vocab.decode(nxt))
        for t in range(1, CONTEXT_LEN):
            context[t - 1] = context[t]
        context[CONTEXT_LEN - 1] = nxt

    return "".join(out)


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    print()
    print("╔════════════════════════════════════════════════════════════════════╗")
    print("║  OA Python Tutorial — Character Text Generation (Autograd)        ║")
    print("╚════════════════════════════════════════════════════════════════════╝")
    print()
    print(f"Vocab: {VOCAB_SIZE} chars (a-z + space)")
    print(f"Context: {CONTEXT_LEN}  Embed: {EMBED_DIM}  Hidden: {HIDDEN_DIM}")
    print("Task: predict next char from previous 8 chars\n")

    # Enable autograd globally BEFORE initializing engine
    set_grad_enabled(True)

    if not runtime.OaInitComputeEngine():
        print("Failed to initialize OA compute engine")
        sys.exit(1)

    # Model + optimizer
    model = OaTextGenerationRnnAutograd()
    params = model.parameters()
    adam = ml.OaAdamW(params, LR)
    sampler = TextBatchSampler(CORPUS, BATCH_SIZE)

    n_params = sum(p.Data.NumElements() for p in params)
    print(
        f"Model: Embed({VOCAB_SIZE}→{EMBED_DIM}) → flatten → "
        f"Linear({FLAT_DIM}→{HIDDEN_DIM}) + Relu → Linear({HIDDEN_DIM}→{VOCAB_SIZE})"
    )
    print(f"Params: {n_params}    Optimizer: AdamW(lr={LR})\n")

    # Training loop with OaItTraining
    progress_bar = ml.OaCbProgressBar()
    summary = ml.OaCbSummary()
    loss_metric = ml.OaMetricLoss()

    progress_bar.AddMetric(loss_metric)
    
    config = ml.OaItTrainingConfig()
    config.TotalSteps = STEPS
    config.BatchSize = BATCH_SIZE
    
    loop = ml.OaItTraining(adam, config)
    loop.AddMetric(loss_metric)
    loop.AddCallback(progress_bar)
    loop.AddCallback(summary)

    print(f"Training: {STEPS} steps · batch={BATCH_SIZE}")
    
    initial_loss = 0.0
    
    while not loop.IsDone():
        x, y = sampler.next_batch()
        adam.ZeroGrad()  # implicit-autograd accumulates, so clear each step
        tape = ml.GradientTape()
        logits = model.forward(x)
        loss = ml.CrossEntropy(logits, y)
        tape.Backward(loss)
        loop.Next(loss)
        
        if loop.Index() == 1:
            initial_loss = loop.LiveLoss()
    
    loop.Finish()
    last_loss = loop.LastLoss()

    # Evaluation
    acc = batch_accuracy(model, sampler)
    print(f"Batch accuracy: {acc:.1f}%")

    # Generation
    generated = generate_greedy(model, "hello", 32)
    print(f"\nGeneration:\n  Prompt:    'hello'")
    print(f"  Generated: '{generated}'")

    # Baseline sanity check
    baseline = math.log(VOCAB_SIZE)
    print(f"\nBaseline: ln({VOCAB_SIZE}) = {baseline:.4f}")

    # Assert convergence — autograd trains to parity with the manual-backward version.
    assert initial_loss > 0.0, "Initial loss must be positive"
    assert last_loss < initial_loss, f"Loss must decrease: {last_loss} >= {initial_loss}"
    assert acc > 45.0, f"Final batch accuracy should exceed 45%, got {acc:.1f}%"

    print(f"\n✓ Training converged (accuracy: {acc:.1f}%)")


if __name__ == "__main__":
    main()
