# tu_nlp_byte_gru — Byte × GRU

The byte-tokenizer × GRU cell of the
[NLP tutorial matrix](../README.md) — same 576-character corpus, same
300-step recipe and captured training lifecycle. This entry selects
the tokenizer and model shown below. See
[`tu_nlp_char_rnn`](../tu_nlp_char_rnn/tu_nlp_char_rnn.md) for the full
walkthrough.

## This cell

| | |
| --- | --- |
| Tokenizer | the 256-token packed-U8 byte vocabulary (`ByteSampler`) — one token per source byte |
| Model | `ByteEmbedding(32) → GRU(64) → ByteHead(256)` |
| Parameters | 43,648 |
| Optimizer | AdamW(lr=0.01) — the canonical recipe |

The loop is a `prepare`/`record` pair per step — `GradientTape` → `cross_entropy` → `backward` → AdamW —
driven by `oa.ml.ItTraining`, so the whole run is one captured GPU program replayed
299 times (19 nodes for this model).

## Measured — Surface Laptop 5 (ADL GT2)

![Byte × GRU training loss](asset/training-curve.png)

Canonical 300-step training loss, recorded by this tutorial.

| Loss init→final | Accuracy | Program nodes | Wall ms/step | Token/s |
| --- | --- | --- | --- | --- |
| 5.618→0.178 | 92.7% | 19 | 15.08 | 67.9K |

Reference generation on the default recipe (asserted by the tutorial):

```text
prompt:    "to be"
generated: "to be that is the question whether tis nobler in the mind to suffer the slings and ar"
```

## Run

```bash
python3 sdk/py/tutorials/ml/nlp/tu_nlp_byte_gru/tu_nlp_byte_gru.py
```

Regenerate the training-loss image from the repository root:

```bash
python3 sdk/py/tutorials/ml/nlp/tu_nlp_byte_gru/tu_nlp_byte_gru.py --plot sdk/py/tutorials/ml/nlp/tu_nlp_byte_gru/asset/training-curve.png
```

On the default recipe the tutorial self-tests: loss regime, accuracy
floor, exact reference generation, and `.oam` checkpoint round-trip.

## Terminal output

Recorded canonical 300-step run on Intel Iris Xe (ADL GT2), Mesa 26.2.7.
Startup logs, intermediate progress updates and the plot-save path are
omitted. Timings vary by run; these are not new benchmark baselines.

```console
╔══════════════════════════════════════════════════════════════════╗
║  OA Tutorial — Byte GRU · all-position LM (autograd)           ║
╚══════════════════════════════════════════════════════════════════╝

tokenizer: raw bytes · vocab=256
Task: dense next-byte prediction at every position

Model: ByteEmbedding(32) → GRU(64) → ByteHead(256)
params: 43648    Optimizer: AdamW(lr=0.01)


  300/300 |██████████| 4.53s · 15.11 ms/step · 4.24K sample/s · 67.78K byte token/s · 67.78K byte/s · loss: 0.1783

Summary:
  loss: initial 5.618298 · final 0.178275 · mean 0.503147
  Wall: 15.11 ms/step · 4.24K sample/s · 67.78K byte token/s · 67.78K byte/s
  GPU (byte_gru_training_step): mean 12.600 ms/step · p50 10.509 · p95 19.751 · min 8.532 · max 21.995 · 5.08K sample/s · 81.27K byte token/s · 81.27K byte/s · wall-GPU gap 17%
  run: 4.53s · 300 steps · batch 64 · sequence 16 byte token/sample

Evaluation:
  Random-loss baseline ln(256) = 5.5452
  byte-token accuracy: 92.7%
  post-update evaluation loss: 0.176507

Generation:
  prompt: "to be"
  generated: "to be that is the question whether tis nobler in the mind to suffer the slings and ar"
program: 19 nodes · 1 recording · 299 cache hits · 0 uploads
checkpoint: optimizer step, accuracy, and exact generation restored
```

## Source

- [`tu_nlp_byte_gru.py`](./tu_nlp_byte_gru.py)
- Shared training loop: [`../_nlp_common.py`](../_nlp_common.py)
- [Suite README](../README.md)
- Rust peer: [`sdk/rs/tutorials/ml/nlp/tu_nlp_byte_gru`](../../../../../rs/tutorials/ml/nlp/tu_nlp_byte_gru/tu_nlp_byte_gru.md)
