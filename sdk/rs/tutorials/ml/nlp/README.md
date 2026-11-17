# The NLP tutorial suite

Status: Experimental. The recorded GPU results qualify the named device
and driver; they do not establish cross-device numerical equivalence.

Sixteen tutorials that are really one advanced example: the *same*
corpus, recipe, and training loop run across a matrix of three
tokenizers and five model architectures. Start with
[`tu_nlp_char_rnn`](tu_nlp_char_rnn/tu_nlp_char_rnn.md) — it is the
introductory walkthrough; the other entries explore tokenizers and architectures; the Empyrealm
row also changes the model topology.

## The shared base

Every entry runs the identical workload, defined once in
`sdk/rs/lib/ml/nlp.rs`:

| Knob | Value |
| --- | --- |
| Corpus | 192-char Hamlet excerpt ×3 = 576 characters |
| Context | 16 tokens per sequence |
| Recipe | 300 steps · batch 64 · AdamW(lr=0.01; Mamba-3/Empyrealm: 0.003) |
| Task | dense next-token prediction at every position |
| Self-test | loss regime, accuracy floor, exact reference generation |

The loop is one prepare/record pair per step driven by
`oa::ml::ItTraining`, with `GradientTape` → `backward` → `AdamW` inside
the record closure. Because the step shape never changes, each run is a
single captured GPU program replayed 299 times — the run's own
diagnostics line reports it (`19 nodes · 1 recording · 299 cache hits`
for the RNN row). Char and Byte metrics stream to `var/log/nlp/<timer_name>/events.jsonl`,
and every tutorial ends by proving `.oam` checkpoint round-trip:
identical optimizer step count, accuracy, and greedy generation.

Shared helpers: `_byte_common.rs` and `_bpe_common.rs` drive the 11
Byte/BPE entries through a `Tutorial` descriptor; the Char entries are
standalone with the same structure. The Python suite mirrors this
through its own `_nlp_common.py` — see its suite root.

## The three tokenizers — the first axis

| Tokenizer | Vocab | Random baseline | Notes |
| --- | --- | --- | --- |
| `char_*` | 27 (a-z + space) | ln(27) = 3.296 | simplest; `accuracy` comparable within Char only |
| `byte_*` | 256 (packed U8) | ln(256) = 5.545 | raw bytes, one token per byte |
| `bpe_*` | 320 (256 + 64 merges) | ln(320) = 5.768 | learned merges, ~2.4–2.8× fewer positions |

Loss and accuracy are **not comparable across tokenizers** — different
vocab sizes shift both scales. BPE rows correctly reach ~0.02 loss /
98.8% accuracy because merges collapse the corpus into short,
predictable token runs. The comparable signal is inside each row
group.

## The five architectures — the second axis

| Arch | Skeleton | Character |
| --- | --- | --- |
| `*_rnn` | Embedding → Elman RNN(64) → Linear | cheapest step; sequential scan |
| `*_gru` | Embedding → GRU(64) → Linear | gated recurrence; slowest per step |
| `*_transformer` | +pos Embedding → TransformerBlock → LN → Linear | causal attention; fastest non-recurrent |
| `*_moe`/`char_moe_transformer` | attention → sparse MoE (E=4, K=2, DFF=16) | routed experts; widest program (62 nodes) |
| `*_mamba3` | Embedding → Mamba-3(32, state=32, expand=2) + residual → Linear | selective state-space recurrence |
| `byte_empyrealm` | Empyrealm hybrid stack (Mamba-3/attention) | fidelity topology — same Mamba-3 providers, different module shape; 184-node program |

## The matrix

15 public cells + the Empyrealm fidelity row = 16 executable entries:

| Tokenizer | RNN | GRU | Transformer | MoE | Mamba-3 |
| --- | --- | --- | --- | --- | --- |
| char | [rnn](tu_nlp_char_rnn/tu_nlp_char_rnn.md) | [gru](tu_nlp_char_gru/tu_nlp_char_gru.md) | [transformer](tu_nlp_char_transformer/tu_nlp_char_transformer.md) | [moe_transformer](tu_nlp_char_moe_transformer/tu_nlp_char_moe_transformer.md) | [mamba3](tu_nlp_char_mamba3/tu_nlp_char_mamba3.md) |
| byte | [rnn](tu_nlp_byte_rnn/tu_nlp_byte_rnn.md) | [gru](tu_nlp_byte_gru/tu_nlp_byte_gru.md) | [transformer](tu_nlp_byte_transformer/tu_nlp_byte_transformer.md) | [moe](tu_nlp_byte_moe/tu_nlp_byte_moe.md) | [mamba3](tu_nlp_byte_mamba3/tu_nlp_byte_mamba3.md) · plus [empyrealm](tu_nlp_byte_empyrealm/tu_nlp_byte_empyrealm.md) |
| bpe | [rnn](tu_nlp_bpe_rnn/tu_nlp_bpe_rnn.md) | [gru](tu_nlp_bpe_gru/tu_nlp_bpe_gru.md) | [transformer](tu_nlp_bpe_transformer/tu_nlp_bpe_transformer.md) | [moe](tu_nlp_bpe_moe/tu_nlp_bpe_moe.md) | [mamba3](tu_nlp_bpe_mamba3/tu_nlp_bpe_mamba3.md) |

## Measured: 16/16, Surface Laptop 5 (ADL GT2)

Condensed from the `surface-laptop-5-2026-11-15` benchmark report — serial pass on an idle GPU, canonical recipe per entry:

| Entry | Loss init→final | Acc | Nodes | Wall ms/step | Token/s | GPU ms | Gap |
| --- | --- | --- | --- | --- | --- | --- | --- |
| char_rnn | 3.311→0.186 | 92.5% | 19 | 4.75 | 215.5K | 4.07 | 14% |
| char_gru | 3.293→0.183 | 92.8% | 19 | 13.22 | 77.5K | 12.24 | 7% |
| char_transformer | 4.041→0.200 | 92.4% | 44 | 5.95 | 172.2K | 4.94 | 17% |
| char_moe_transformer | 3.514→0.190 | 92.6% | 62 | 8.41 | 121.8K | 7.18 | 15% |
| char_mamba3 | 3.303→0.195 | 93.0% | 34 | 10.69 | 95.8K | 9.33 | 13% |
| byte_rnn | 5.569→0.191 | 92.2% | 19 | 7.98 | 128.3K | 6.98 | 13% |
| byte_gru | 5.618→0.178 | 92.7% | 19 | 15.08 | 67.9K | 14.01 | 7% |
| byte_transformer | 5.657→0.201 | 92.9% | 46 | 7.85 | 130.5K | 6.81 | 13% |
| byte_moe | 5.687→0.199 | 92.6% | 64 | 9.86 | 103.9K | 8.90 | 10% |
| byte_mamba3 | 5.539→0.214 | 93.0% | 34 | 11.68 | 87.7K | 10.74 | 8% |
| byte_empyrealm | 5.667→0.182 | 92.9% | 184 | 40.20 | 25.5K | 38.68 | 4% |
| bpe_rnn | 5.774→0.021 | 98.8% | 19 | 8.40 | 121.9K | 6.92 | 18% |
| bpe_gru | 5.766→0.021 | 98.8% | 19 | 14.79 | 69.2K | 13.80 | 7% |
| bpe_transformer | 5.871→0.021 | 98.9% | 44 | 7.51 | 136.3K | 6.40 | 15% |
| bpe_moe | 5.844→0.020 | 98.8% | 62 | 10.43 | 98.2K | 9.29 | 11% |
| bpe_mamba3 | 5.775→0.020 | 98.9% | 34 | 12.23 | 83.7K | 11.10 | 9% |

Reading: GRU is the slowest family (recurrent serialization, 13–15
ms/step); RNN and Transformer are fastest (~5–8 ms); `byte_empyrealm`
is the structural outlier at 184 nodes — the fidelity topology, not the
shared Mamba-3 providers, sets its cost. The wall−GPU gap column is the
per-step host/submission boundary, tracked as OA-CC-02. Full table with
p50/p95 and byte-rate columns lives in the benchmark report.

## Run

Every tutorial page displays its own `asset/training-curve.png`, generated
by that Rust entry with the canonical 300-step recipe. Each page includes
the command to regenerate its image from the repository root.

```bash
# one entry
cargo run --release -p sdk --example tu_nlp_char_rnn
# all sixteen
for e in char_rnn char_gru char_transformer char_moe_transformer char_mamba3 \
         byte_rnn byte_gru byte_transformer byte_moe byte_mamba3 byte_empyrealm \
         bpe_rnn bpe_gru bpe_transformer bpe_moe bpe_mamba3; do
	cargo run --release -p sdk --example tu_nlp_$e || break
done
```

Every entry accepts `--steps` and `--plot`. Char entries additionally
accept `--batch` and `--lr`. Use `--plot out.png` to render the loss curve
through `oa::plot`. The Python
suite runs the same matrix via `run_nlp_suite.py` — see
[`sdk/py/tutorials/ml/nlp`](../../../../py/tutorials/ml/nlp/README.md).
