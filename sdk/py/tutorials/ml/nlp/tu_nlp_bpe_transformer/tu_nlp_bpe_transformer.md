# tu_nlp_bpe_transformer — BPE × Transformer

The BPE-tokenizer × Transformer cell of the
[NLP tutorial matrix](../README.md) — same 576-character corpus, same
300-step recipe and captured training lifecycle. This entry selects
the tokenizer and model shown below. See
[`tu_nlp_char_rnn`](../tu_nlp_char_rnn/tu_nlp_char_rnn.md) for the full
walkthrough.

## This cell

| | |
| --- | --- |
| Tokenizer | the 320-token learned BPE vocabulary (`BpeTokenizer`/`BpeSampler`: 256 bytes + 64 merges) — merges collapse the corpus into short predictable runs, which is why BPE rows reach ~0.02 loss |
| Model | `BPE + position Embedding → Transformer(32,64) → LN → Linear(32→320)` |
| Parameters | 29,920 |
| Optimizer | AdamW(lr=0.01) — the canonical recipe |

The loop is a `prepare`/`record` pair per step — `GradientTape` → `cross_entropy` → `backward` → AdamW —
driven by `oa.ml.ItTraining`, so the whole run is one captured GPU program replayed
299 times (44 nodes for this model).

## Measured — Surface Laptop 5 (ADL GT2)

![BPE × Transformer training loss](asset/training-curve.png)

Canonical 300-step training loss, recorded by this tutorial.

| Loss init→final | Accuracy | Program nodes | Wall ms/step | Token/s |
| --- | --- | --- | --- | --- |
| 5.871→0.021 | 98.9% | 44 | 7.51 | 136.3K |

Reference generation on the default recipe (asserted by the tutorial):

```text
prompt:    "to be"
generated: "to be end them to be or not to be that is the question whether tis nobler in the mind"
```

## Run

```bash
python3 sdk/py/tutorials/ml/nlp/tu_nlp_bpe_transformer/tu_nlp_bpe_transformer.py
```

Regenerate the training-loss image from the repository root:

```bash
python3 sdk/py/tutorials/ml/nlp/tu_nlp_bpe_transformer/tu_nlp_bpe_transformer.py --plot sdk/py/tutorials/ml/nlp/tu_nlp_bpe_transformer/asset/training-curve.png
```

On the default recipe the tutorial self-tests: loss regime, accuracy
floor, exact reference generation, and `.oam` checkpoint round-trip.

## Terminal output

Recorded canonical 300-step run on Intel Iris Xe (ADL GT2), Mesa 26.2.7.
Startup logs, intermediate progress updates and the plot-save path are
omitted. Timings vary by run; these are not new benchmark baselines.

```console
╔══════════════════════════════════════════════════════════════════╗
║  OA Tutorial — BPE Transformer · all-position LM (autograd)    ║
╚══════════════════════════════════════════════════════════════════╝

tokenizer: byte BPE · vocab=320 (256 bytes + 64 merges)
compression: 576 bytes → 210 tokens · 2.743 byte/token · 63.5% fewer positions
context coverage: 16 BPE tokens ≈ 43.9 source bytes at corpus average
Task: dense next-token prediction at every position

Model: BPE + position Embedding → Transformer(32,64) → LN → Linear(32→320)
params: 29920    Optimizer: AdamW(lr=0.01)


  300/300 |██████████| 2.55s · 8.50 ms/step · 7.53K sample/s · 120.40K BPE token/s · 334.18K byte/s · loss: 0.0206

Summary:
  loss: initial 5.871270 · final 0.020626 · mean 0.243361
  Wall: 8.51 ms/step · 7.52K sample/s · 120.38K BPE token/s · 334.11K byte/s
  GPU (bpe_transformer_training_step): mean 5.807 ms/step · p50 4.281 · p95 12.590 · min 3.944 · max 14.999 · 11.02K sample/s · 176.34K BPE token/s · 489.44K byte/s · wall-GPU gap 32%
  run: 2.55s · 300 steps · batch 64 · sequence 16 BPE token/sample

Evaluation:
  Random-loss baseline ln(320) = 5.7683
  final batch: 2.771 byte/token · 0.0107 bits/byte
  BPE-token accuracy: 98.9%
  post-update evaluation loss: 0.020028

Generation:
  prompt: "to be"
  generated: "to be end them to be or not to be that is the question whether tis nobler in the mind"
program: 44 nodes · 1 recording · 299 cache hits · 0 uploads
```

## Source

- [`tu_nlp_bpe_transformer.py`](./tu_nlp_bpe_transformer.py)
- Shared training loop: [`../_nlp_common.py`](../_nlp_common.py)
- [Suite README](../README.md)
- Rust peer: [`sdk/rs/tutorials/ml/nlp/tu_nlp_bpe_transformer`](../../../../../rs/tutorials/ml/nlp/tu_nlp_bpe_transformer/tu_nlp_bpe_transformer.md)
