# OA SDK

Runnable source-companion validation for the OA Rust crate.

## Structure

```
sdk/
├── rs/
│   ├── ml/           # SDK-owned native workloads and environments
│   ├── slang/        # SDK kernel packs compiled into the OA crate
│   ├── benchmarks/   # Correctness-gated measurement workloads
│   │   └── core/
│   └── tutorials/    # Runnable Rust workflows
│       ├── core/
│       └── ml/nlp/
└── py/               # Native PyO3 preview and Python tests
```

The Python preview mirrors Rust ownership: root `Engine`/`Matrix` identities,
`oa.core` and `oa.runtime` discovery modules, and stateless functions under
`oa.matrix`. It currently binds the introductory dense FP32 Matrix slice; see
[`py/README.md`](py/README.md).

SDK-owned workloads are available below `oa::sdk` without moving concrete
tasks into the reusable ML library. The native vectorized CartPole environment
is `oa::sdk::ml::rl::CartPole`; it borrows the crate's sole Engine and uses
schema-owned SDK shaders through the ordinary semantic and executable graphs.
The Lunar Lander port exposes its versioned deterministic manifest, checked
terrain, scalar double-precision dynamics/observation, complete episode reward
and termination behavior, and scripted controller through `oa::sdk::ml::rl`.
Its full seeded scalar trace matches the C++ donor digest. The native flat-terrain
`LunarLander3dVector` records mechanically adapted donor reset/step shaders
through the same Engine, semantic graph, executable graph, and Event lifecycle.
Hardware tests compare reset, a five-lane 24-step trace, and complete contact /
terminal episodes against independent scalar environments. They also cover
transactional reseeding, FP32 admission edges, compact telemetry, completed-lane
reset, and repeated poisoned-action reuse across 257 lanes.

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

## Character MoE Transformer training

`ml_nlp_char_moe_transformer` ports the donor sparse-MoE row: the same character
workload and attention topology with four width-16 experts and deterministic
top-two routing. Its 23 registered parameter tensors contain exactly 13,183
scalars. The default execution path is the grouped sparse route; the dense
all-expert path remains an opt-in correctness oracle.

```sh
cargo build --release --example ml_nlp_char_moe_transformer
python3 tools/build/stage.py --profile release --target ml_nlp_char_moe_transformer
./bin/release/sdk/tutorials/ml/nlp/ml_nlp_char_moe_transformer
```

The tutorial uses the shared training iterator, loss metric, progress bar,
wall/GPU timing summary, captured training program, evaluation, greedy
generation, and complete parameter/buffer/AdamW checkpoint round trip.

## CartPole PPO training

`ml_rl_cart_pole_ppo` ports the donor's complete native acceptance workload:
64 vectorized GPU environments, 128-step rollouts, four full-batch PPO epochs,
40 rollouts, independent `4 -> 64 -> 64` actor/critic towers, and AdamW at
`2.5e-4` with zero weight decay.

```sh
cargo run --release --example ml_rl_cart_pole_ppo
```

The executable compares a fixed-seed greedy evaluation before and after
training, requires at least +25 mean return and an absolute return of 75, and
round-trips the complete model plus AdamW state through native `.oam`. The
Intel Iris Xe validation run improved mean completed return from `33.36` to
`453.30`; restore reproduced `453.30` exactly at optimizer step 160. The
timestamp-enabled run took 188.09 seconds end to end, while the 160 optimizer
updates averaged 14.48 ms of Vulkan device time each; collection graph
construction, submission preparation, and evaluation dominate the wall gap.

The smaller donor rollout tutorial is also available independently:

```sh
cargo run --release --example ml_rl_cart_pole_rollout
```

Its fixed policy collected all 2,048 transitions in one GPU transaction,
produced 2,048 reward and one completed episode, finalized GAE, and performed no
host tensor reads during collection.

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
