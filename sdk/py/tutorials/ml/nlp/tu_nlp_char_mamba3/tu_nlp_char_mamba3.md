# tu_nlp_char_mamba3 — Character × Mamba-3

The character-tokenizer × Mamba-3 cell of the
[NLP tutorial matrix](../README.md) — same 576-character corpus, same
300-step recipe and captured training lifecycle. This entry selects
the tokenizer and model shown below. See
[`tu_nlp_char_rnn`](../tu_nlp_char_rnn/tu_nlp_char_rnn.md) for the full
walkthrough.

## This cell

| | |
| --- | --- |
| Tokenizer | the 27-token character vocabulary (`CharSampler`) — accuracy is comparable within Char rows only |
| Model | `Embedding(32) → Mamba-3 SISO → residual → Linear(27)` |
| Parameters | 10,915 |
| Optimizer | AdamW(lr=0.003) — lower rate for the Mamba-3/Empyrealm family |

The loop is a `prepare`/`record` pair per step — `GradientTape` → `cross_entropy` → `backward` → AdamW —
driven by `oa.ml.ItTraining`, so the whole run is one captured GPU program replayed
299 times (34 nodes for this model).

## Measured — Surface Laptop 5 (ADL GT2)

![Character × Mamba-3 training loss](asset/training-curve.png)

Canonical 300-step training loss, recorded by this tutorial.

| Loss init→final | Accuracy | Program nodes | Wall ms/step | Token/s |
| --- | --- | --- | --- | --- |
| 3.303→0.195 | 93.0% | 34 | 10.69 | 95.8K |

Reference generation on the default recipe (asserted by the tutorial):

```text
prompt:    "to be"
generated: "to be that is the question whether tis nobler in the mind to suffer the slings and ar"
```

## Run

```bash
python3 sdk/py/tutorials/ml/nlp/tu_nlp_char_mamba3/tu_nlp_char_mamba3.py
# render the loss curve into asset/
python3 sdk/py/tutorials/ml/nlp/tu_nlp_char_mamba3/tu_nlp_char_mamba3.py --plot sdk/py/tutorials/ml/nlp/tu_nlp_char_mamba3/asset/training-curve.png
```

On the default recipe the tutorial self-tests: loss regime, accuracy
floor, exact reference generation, and `.oam` checkpoint round-trip.

## Terminal output

Recorded canonical 300-step run on Intel Iris Xe (ADL GT2), Mesa 26.2.7.
Startup logs, intermediate progress updates and the plot-save path are
omitted. Timings vary by run; these are not new benchmark baselines.

```console
╔══════════════════════════════════════════════════════════════════╗
║  OA Tutorial — Char Mamba-3 · all-position LM (autograd)         ║
╚══════════════════════════════════════════════════════════════════╝

tokenizer: character · vocab=27 (a-z + space)
Task: dense next-character prediction at every position

Model: Embedding(32) → Mamba-3 SISO → residual → Linear(27)
params: 10915    Optimizer: AdamW(lr=0.003)

training: 300 steps · batch=64 · sequence=16 character tokens

  300/300 |██████████| 7.02s · 23.38 ms/step · 2.74K sample/s · 43.79K token/s · loss: 0.1945

Summary:
  loss: initial 3.303064 · final 0.194512 · mean 0.714765
  Wall: 23.39 ms/step · 2.74K sample/s · 43.79K token/s
  GPU (char_mamba3_training_step): mean 18.877 ms/step · p50 19.136 · p95 25.516 · min 6.425 · max 27.824 · 3.39K sample/s · 54.25K token/s · wall-GPU gap 19%
  run: 7.02s · 300 steps · batch 64 · sequence 16 token/sample

Evaluation:
  Random-loss baseline ln(27) = 3.2958
  loss: evaluation 0.190578 · last training 0.194512
  character-token accuracy: 93.0% (compare within Char only)

Generation:
  prompt: "to be"
  generated: "to be that is the question whether tis nobler in the mind to suffer the slings and ar"
program: 34 nodes · 1 recording · 299 cache hits · 0 uploads
```

## Source

- [`tu_nlp_char_mamba3.py`](./tu_nlp_char_mamba3.py)
- Shared training loop: [`../_nlp_common.py`](../_nlp_common.py)
- [Suite README](../README.md)
- Rust peer: [`sdk/rs/tutorials/ml/nlp/tu_nlp_char_mamba3`](../../../../../rs/tutorials/ml/nlp/tu_nlp_char_mamba3/tu_nlp_char_mamba3.md)
