# OA SDK

Runnable source-companion validation for the OA Rust crate.

## Structure

```
sdk/
├── rs/
│   ├── benchmarks/   # Correctness-gated measurement workloads
│   │   └── core/
│   └── tutorials/    # Runnable Rust workflows
│       ├── core/
│       └── ml/nlp/
└── py/               # Future generated-binding consumers
```

## Core matrix multiplication

`core_mat_mul_intro` validates the admitted FP32 matrix multiplication path
against an independent CPU implementation over the same shape archetypes as
the C++ tutorial. OA stores its right-hand matrix as `[N, K]`, so
`mat_mul_nt([M, K], [N, K])` returns `[M, N]`.

```sh
cargo build --release --example core_mat_mul_intro
python3 tools/build/stage.py --profile release --target core_mat_mul_intro
./bin/release/sdk/tutorials/core/core_mat_mul_intro
```

By default it runs correctness and an explicitly exploratory single-process
performance table. Use `--correctness-only`, `--benchmark-only`, `--case NAME`,
`--warmup N`, and `--samples N` to narrow it.
Performance timing spans the eager operation through synchronized full output
readback; it is not directly equivalent to the C++ reusable-plan timing.

Capture and immutable reusable-plan replay are now Experimental. BF16 routing,
autotuning, and `linear` remain Planned until their runtime contracts exist.

## Character RNN training

`ml_nlp_char_rnn` is the first complete Rust member of OA's controlled NLP
suite. It uses the exact C++ Char-RNN workload: the 576-character corpus,
batch 64, context 16, widths 32/64, 8,891 parameters, AdamW at `0.01`, 300
optimizer steps, all-position accuracy, and 80-character greedy generation
from `to be`.

```sh
cargo build --release --example ml_nlp_char_rnn
python3 tools/build/stage.py --profile release --target ml_nlp_char_rnn
./bin/release/sdk/tutorials/ml/nlp/ml_nlp_char_rnn
```

The tutorial fails unless loss reaches the accepted small-corpus regime,
accuracy exceeds 90%, and greedy generation matches the reference continuation.
Checkpoint roundtrip and qualified performance parity remain separate gates.

## Character Transformer training

`ml_nlp_char_transformer` matches the C++ Char-Transformer tutorial topology:
token and position Embeddings, one width-32 pre-norm Transformer block with one
causal attention head and width-64 GELU FFN, final LayerNorm, and vocabulary
projection. Its 22 registered parameter tensors contain exactly 10,875 scalars.

```sh
cargo build --release --example ml_nlp_char_transformer
python3 tools/build/stage.py --profile release --target ml_nlp_char_transformer
./bin/release/sdk/tutorials/ml/nlp/ml_nlp_char_transformer
```

The 300-step executable requires final loss below `0.3`, accuracy above 90%,
the exact C++ reference continuation from prompt `to be`, and a fresh-model
parameter/AdamW checkpoint roundtrip preserving step, accuracy, and generation.

## Core matrix multiplication benchmark

`core_mat_mul_bench` first checks every FP32 output against the closed-form
constant-input oracle, then measures repeated captured-plan submissions with a
fresh Vulkan timestamp pair per submission. It reports device time separately
from synchronized wall time.

```sh
cargo build --release --example core_mat_mul_bench
python3 tools/build/stage.py --profile release --target core_mat_mul_bench
python3 tools/profiling/suite.py --workload core.matmul_nt.square_1024
```

The checked-in suite contains three square and three ML-shaped workloads. Each
result preserves raw output, seven fresh-process medians, median absolute
deviation, spread, executable hash, source identity, device/driver identity,
tool versions, and thermal observations under `var/report/benchmark-suite/`.
Use `--allow-dirty` only for explicitly noncanonical development recordings.
