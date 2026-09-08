# OA SDK

Runnable source-companion validation for the OA Rust crate.

## Structure

```
sdk/
├── rs/
│   ├── benchmarks/   # Correctness-gated measurement workloads
│   │   └── core/
│   └── tutorials/    # Runnable Rust workflows
│       └── core/
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
