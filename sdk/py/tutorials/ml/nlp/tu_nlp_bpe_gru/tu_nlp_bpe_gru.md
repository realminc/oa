# tu_nlp_bpe_gru — BPE × GRU

The BPE-tokenizer × GRU cell of the
[NLP tutorial matrix](../README.md) — same 576-character corpus, same
300-step recipe and captured training lifecycle. This entry selects
the tokenizer and model shown below. See
[`tu_nlp_char_rnn`](../tu_nlp_char_rnn/tu_nlp_char_rnn.md) for the full
walkthrough.

## This cell

| | |
| --- | --- |
| Tokenizer | the 320-token learned BPE vocabulary (`BpeTokenizer`/`BpeSampler`: 256 bytes + 64 merges) — merges collapse the corpus into short predictable runs, which is why BPE rows reach ~0.02 loss |
| Model | `BPE Embedding → GRU(32→64) → Linear(64→320)` |
| Parameters | 49,856 |
| Optimizer | AdamW(lr=0.01) — the canonical recipe |

The loop is a `prepare`/`record` pair per step — `GradientTape` → `cross_entropy` → `backward` → AdamW —
driven by `oa.ml.ItTraining`, so the whole run is one captured GPU program replayed
299 times (19 nodes for this model).

## Measured — Surface Laptop 5 (ADL GT2)

![BPE × GRU training loss](asset/training-curve.png)

Canonical 300-step training loss, recorded by this tutorial.

| Loss init→final | Accuracy | Program nodes | Wall ms/step | Token/s |
| --- | --- | --- | --- | --- |
| 5.766→0.021 | 98.8% | 19 | 14.79 | 69.2K |

Reference generation on the default recipe (asserted by the tutorial):

```text
prompt:    "to be"
generated: "to bee arms against a sea of troubles and by opposing end them to be or not to be tha"
```

## Run

```bash
python3 sdk/py/tutorials/ml/nlp/tu_nlp_bpe_gru/tu_nlp_bpe_gru.py
```

Regenerate the training-loss image from the repository root:

```bash
python3 sdk/py/tutorials/ml/nlp/tu_nlp_bpe_gru/tu_nlp_bpe_gru.py --plot sdk/py/tutorials/ml/nlp/tu_nlp_bpe_gru/asset/training-curve.png
```

On the default recipe the tutorial self-tests: loss regime, accuracy
floor, exact reference generation, and `.oam` checkpoint round-trip.

## Terminal output

Recorded canonical 300-step run on Intel Iris Xe (ADL GT2), Mesa 26.2.7.
Startup logs, intermediate progress updates and the plot-save path are
omitted. Timings vary by run; these are not new benchmark baselines.

```console
╔══════════════════════════════════════════════════════════════════╗
║  OA Tutorial — BPE GRU · all-position LM (autograd)            ║
╚══════════════════════════════════════════════════════════════════╝

tokenizer: byte BPE · vocab=320 (256 bytes + 64 merges)
compression: 576 bytes → 210 tokens · 2.743 byte/token · 63.5% fewer positions
context coverage: 16 BPE tokens ≈ 43.9 source bytes at corpus average
Task: dense next-token prediction at every position

Model: BPE Embedding → GRU(32→64) → Linear(64→320)
params: 49856    Optimizer: AdamW(lr=0.01)


  300/300 |██████████| 4.85s · 16.16 ms/step · 3.96K sample/s · 63.36K BPE token/s · 175.87K byte/s · loss: 0.0215

Summary:
  loss: initial 5.766479 · final 0.021488 · mean 0.505326
  Wall: 16.16 ms/step · 3.96K sample/s · 63.36K BPE token/s · 175.85K byte/s
  GPU (bpe_gru_training_step): mean 12.757 ms/step · p50 10.203 · p95 19.846 · min 8.764 · max 23.776 · 5.02K sample/s · 80.27K BPE token/s · 222.78K byte/s · wall-GPU gap 21%
  run: 4.85s · 300 steps · batch 64 · sequence 16 BPE token/sample

Evaluation:
  Random-loss baseline ln(320) = 5.7683
  final batch: 2.771 byte/token · 0.0112 bits/byte
  BPE-token accuracy: 98.8%
  post-update evaluation loss: 0.021224

Generation:
  prompt: "to be"
  generated: "to bee arms against a sea of troubles and by opposing end them to be or not to be tha"
program: 19 nodes · 1 recording · 299 cache hits · 0 uploads
```

## Source

- [`tu_nlp_bpe_gru.py`](./tu_nlp_bpe_gru.py)
- Shared training loop: [`../_nlp_common.py`](../_nlp_common.py)
- [Suite README](../README.md)
- Rust peer: [`sdk/rs/tutorials/ml/nlp/tu_nlp_bpe_gru`](../../../../../rs/tutorials/ml/nlp/tu_nlp_bpe_gru/tu_nlp_bpe_gru.md)
