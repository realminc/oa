# tu_nlp_bpe_rnn — BPE × RNN

The BPE-tokenizer × RNN cell of the
[NLP tutorial matrix](../README.md) — same 576-character corpus, same
300-step recipe and captured training lifecycle. This entry selects
the tokenizer and model shown below. See
[`tu_nlp_char_rnn`](../tu_nlp_char_rnn/tu_nlp_char_rnn.md) for the full
walkthrough.

## This cell

| | |
| --- | --- |
| Tokenizer | the 320-token learned BPE vocabulary (`BpeTokenizer`/`BpeSampler`: 256 bytes + 64 merges) — merges collapse the corpus into short predictable runs, which is why BPE rows reach ~0.02 loss |
| Model | `BPE Embedding → RNN(32→64) → Linear(64→320)` |
| Parameters | 37,312 |
| Optimizer | AdamW(lr=0.01) — the canonical recipe |

The loop is a `prepare`/`record` pair per step — `GradientTape` → `cross_entropy` → `backward` → AdamW —
driven by `oa.ml.ItTraining`, so the whole run is one captured GPU program replayed
299 times (19 nodes for this model).

## Measured — Surface Laptop 5 (ADL GT2)

![BPE × RNN training loss](asset/training-curve.png)

Canonical 300-step training loss, recorded by this tutorial.

| Loss init→final | Accuracy | Program nodes | Wall ms/step | Token/s |
| --- | --- | --- | --- | --- |
| 5.774→0.021 | 98.8% | 19 | 8.40 | 121.9K |

Reference generation on the default recipe (asserted by the tutorial):

```text
prompt:    "to be"
generated: "to be not to be that is the question whether tis nobler in the mind to suffer the sli"
```

## Run

```bash
python3 sdk/py/tutorials/ml/nlp/tu_nlp_bpe_rnn/tu_nlp_bpe_rnn.py
```

Regenerate the training-loss image from the repository root:

```bash
python3 sdk/py/tutorials/ml/nlp/tu_nlp_bpe_rnn/tu_nlp_bpe_rnn.py --plot sdk/py/tutorials/ml/nlp/tu_nlp_bpe_rnn/asset/training-curve.png
```

On the default recipe the tutorial self-tests: loss regime, accuracy
floor, exact reference generation, and `.oam` checkpoint round-trip.

## Terminal output

Recorded canonical 300-step run on Intel Iris Xe (ADL GT2), Mesa 26.2.7.
Startup logs, intermediate progress updates and the plot-save path are
omitted. Timings vary by run; these are not new benchmark baselines.

```console
╔══════════════════════════════════════════════════════════════════╗
║  OA Tutorial — BPE RNN · all-position LM (autograd)            ║
╚══════════════════════════════════════════════════════════════════╝

tokenizer: byte BPE · vocab=320 (256 bytes + 64 merges)
compression: 576 bytes → 210 tokens · 2.743 byte/token · 63.5% fewer positions
context coverage: 16 BPE tokens ≈ 43.9 source bytes at corpus average
Task: dense next-token prediction at every position

Model: BPE Embedding → RNN(32→64) → Linear(64→320)
params: 37312    Optimizer: AdamW(lr=0.01)


  300/300 |██████████| 2.82s · 9.41 ms/step · 6.80K sample/s · 108.84K BPE token/s · 302.10K byte/s · loss: 0.0211

Summary:
  loss: initial 5.774095 · final 0.021129 · mean 0.230191
  Wall: 9.41 ms/step · 6.80K sample/s · 108.83K BPE token/s · 302.05K byte/s
  GPU (bpe_rnn_training_step): mean 6.586 ms/step · p50 4.786 · p95 14.230 · min 4.508 · max 16.938 · 9.72K sample/s · 155.48K BPE token/s · 431.54K byte/s · wall-GPU gap 30%
  run: 2.82s · 300 steps · batch 64 · sequence 16 BPE token/sample

Evaluation:
  Random-loss baseline ln(320) = 5.7683
  final batch: 2.771 byte/token · 0.0110 bits/byte
  BPE-token accuracy: 98.8%
  post-update evaluation loss: 0.020861

Generation:
  prompt: "to be"
  generated: "to be not to be that is the question whether tis nobler in the mind to suffer the sli"
program: 19 nodes · 1 recording · 299 cache hits · 0 uploads
```

## Source

- [`tu_nlp_bpe_rnn.py`](./tu_nlp_bpe_rnn.py)
- Shared training loop: [`../_nlp_common.py`](../_nlp_common.py)
- [Suite README](../README.md)
- Rust peer: [`sdk/rs/tutorials/ml/nlp/tu_nlp_bpe_rnn`](../../../../../rs/tutorials/ml/nlp/tu_nlp_bpe_rnn/tu_nlp_bpe_rnn.md)
