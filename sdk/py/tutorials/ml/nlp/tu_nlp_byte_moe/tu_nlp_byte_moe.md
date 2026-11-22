# tu_nlp_byte_moe — Byte × sparse MoE Transformer

The byte-tokenizer × sparse MoE Transformer cell of the
[NLP tutorial matrix](../README.md) — same 576-character corpus, same
300-step recipe and captured training lifecycle. This entry selects
the tokenizer and model shown below. See
[`tu_nlp_char_rnn`](../tu_nlp_char_rnn/tu_nlp_char_rnn.md) for the full
walkthrough.

## This cell

| | |
| --- | --- |
| Tokenizer | the 256-token packed-U8 byte vocabulary (`ByteSampler`) — one token per source byte |
| Model | `Byte + position Embedding → Attention + MoE(E=4,K=2,DFF=16) → LN → Linear` |
| Parameters | 28,068 |
| Optimizer | AdamW(lr=0.01) — the canonical recipe |

The loop is a `prepare`/`record` pair per step — `GradientTape` → `cross_entropy` → `backward` → AdamW —
driven by `oa.ml.ItTraining`, so the whole run is one captured GPU program replayed
299 times (64 nodes for this model).

## Measured — Surface Laptop 5 (ADL GT2)

![Byte × sparse MoE Transformer training loss](asset/training-curve.png)

Canonical 300-step training loss, recorded by this tutorial.

| Loss init→final | Accuracy | Program nodes | Wall ms/step | Token/s |
| --- | --- | --- | --- | --- |
| 5.687→0.199 | 92.6% | 64 | 9.86 | 103.9K |

Reference generation on the default recipe (asserted by the tutorial):

```text
prompt:    "to be"
generated: "to be that is the question whether tis nobler in the mind to suffer the slings and ar"
```

## Run

```bash
python3 sdk/py/tutorials/ml/nlp/tu_nlp_byte_moe/tu_nlp_byte_moe.py
```

Regenerate the training-loss image from the repository root:

```bash
python3 sdk/py/tutorials/ml/nlp/tu_nlp_byte_moe/tu_nlp_byte_moe.py --plot sdk/py/tutorials/ml/nlp/tu_nlp_byte_moe/asset/training-curve.png
```

On the default recipe the tutorial self-tests: loss regime, accuracy
floor, exact reference generation, and `.oam` checkpoint round-trip.

## Terminal output

Recorded canonical 300-step run on Intel Iris Xe (ADL GT2), Mesa 26.2.7.
Startup logs, intermediate progress updates and the plot-save path are
omitted. Timings vary by run; these are not new benchmark baselines.

```console
╔══════════════════════════════════════════════════════════════════╗
║  OA Tutorial — Byte MoE Transformer · all-position LM (autograd)║
╚══════════════════════════════════════════════════════════════════╝

tokenizer: raw bytes · vocab=256
Task: dense next-byte prediction at every position

Model: Byte + position Embedding → Attention + MoE(E=4,K=2,DFF=16) → LN → Linear
params: 28068    Optimizer: AdamW(lr=0.01)


  300/300 |██████████| 3.02s · 10.05 ms/step · 6.37K sample/s · 101.86K byte token/s · 101.86K byte/s · loss: 0.1991

Summary:
  loss: initial 5.686717 · final 0.199083 · mean 0.805950
  Wall: 10.06 ms/step · 6.36K sample/s · 101.81K byte token/s · 101.81K byte/s
  GPU (byte_moe_training_step): mean 7.716 ms/step · p50 6.163 · p95 13.887 · min 5.841 · max 19.702 · 8.29K sample/s · 132.71K byte token/s · 132.71K byte/s · wall-GPU gap 23%
  run: 3.02s · 300 steps · batch 64 · sequence 16 byte token/sample

Evaluation:
  Random-loss baseline ln(256) = 5.5452
  byte-token accuracy: 92.6%
  post-update evaluation loss: 0.193275

Generation:
  prompt: "to be"
  generated: "to be that is the question whether tis nobler in the mind to suffer the slings and ar"
program: 64 nodes · 1 recording · 299 cache hits · 0 uploads
checkpoint: optimizer step, accuracy, and exact generation restored
```

## Source

- [`tu_nlp_byte_moe.py`](./tu_nlp_byte_moe.py)
- Shared training loop: [`../_nlp_common.py`](../_nlp_common.py)
- [Suite README](../README.md)
- Rust peer: [`sdk/rs/tutorials/ml/nlp/tu_nlp_byte_moe`](../../../../../rs/tutorials/ml/nlp/tu_nlp_byte_moe/tu_nlp_byte_moe.md)
