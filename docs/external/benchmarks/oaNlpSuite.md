# OA NLP Training Benchmark

**Status:** Current public benchmark with dated historical baselines

OA's canonical NLP matrix trains five neural-network architectures with three
tokenizers through the same Vulkan runtime, autograd engine, AdamW optimizer,
metrics, generation and checkpoint path.

This document is updated at preview releases so performance work remains
measurable rather than anecdotal.

## Workload

- 300 optimizer steps;
- batch 64, sequence length 16;
- 1,024 predicted positions per step;
- model width 32 and recurrent/FFN width 64;
- sparse MoE uses four `DFF=16` experts and top-2 routing;
- identical 576-byte teaching corpus and deterministic sampling;
- complete wall time includes forward, loss, backward, AdamW, submission,
  synchronization, scalar metrics and callbacks;
- evaluation, generation and checkpoint reload are mandatory pass conditions.

These are deliberately small educational models. They measure implementation
correctness and runtime overhead; they are not claims about production language
quality.

## Reference system

| Component | Configuration |
|---|---|
| System | Lenovo ThinkPad X1 Carbon Gen 9 |
| CPU | Intel Core i5-1145G7, 4 cores / 8 threads |
| GPU | Intel Iris Xe TGL GT2 integrated GPU |
| Driver | Mesa Intel 26.1.7, Xe KMD, Vulkan 1.4.354 |
| Precision | FP32 |
| Build | CMake Release, Clang 22.1.8 |

## Current-tree Mamba-3 grouped-state checkpoint — 2026-08-14

This private engineering checkpoint measures the dirty dev tree based on
`00203ac1`; it is not a public tag or package comparison. The candidate carries
grouped B/C state directly through preprocessing and every selective-scan route,
uses deterministic short-backward reductions, and capability-gates its P=16
subgroup implementation. The common NLP graph falls from 44 to 31 physical
dispatches.

Each tokenizer ran one excluded warm-up and seven measured fresh Release
processes with FP32 on Intel Iris Xe TGL GT2, Mesa 26.1.7, and Vulkan 1.4.354.
Every process completed 300 optimizer steps, evaluation, generation,
checkpoint reload, and its GoogleTest assertion: 24/24 processes passed the
end-to-end correctness contract. Performance is accepted only when the seven
measured processes have at most 15% full spread.

| Tokenizer | Median ms/step | MAD | Range ms/step | Full spread | Gate |
|---|---:|---:|---:|---:|---|
| Byte | 17.42 | 0.70 | 16.39-18.12 | 9.93% | **PASS** |
| BPE | 17.25 | 0.07 | 16.46-18.09 | 9.45% | **PASS** |
| Char | 16.18 | 0.04 | 15.30-19.34 | 24.97% | **REJECTED — spread** |

Byte and BPE satisfy the fixed spread ceiling. Char's outlying process
coincides with a package-temperature rise to 75 C, so that row remains
correctness evidence but cannot support a performance claim. Earlier
cooling-controlled, borderline-BPE, and firmware-degraded `performance`
confirmations are retained in `var/report` as rejected evidence; they are not
substituted for these exact-tree rows.

An exact clean pre-rewrite Byte build produced a 22.93 ms/step median versus
17.42 for the candidate, a 24.03% diagnostic reduction. Its 57.87% full spread
invalidates the baseline, so this document does not promote that movement to a
formal before/after speedup. The accepted claim is narrower: the final Byte and
BPE candidates are stable at 17.42 and 17.25 ms/step on this host, and all three
candidate rows are functionally complete. Transformer/MoE remain faster at
sequence length 16; Mamba's linear sequence scaling and recurrent-state value
require a separate long-context quality-at-equal-wall-time benchmark.

The dedicated shared-state MIMO kernels landed after this measurement. The
canonical NLP suite continues to instantiate SISO Mamba-3, so these numbers are
not relabeled as MIMO performance evidence. MIMO has independent numerical and
streaming correctness gates; its workload-level performance remains unmeasured.

## Current-tree correctness and stability audit — 2026-08-13

The complete fifteen-workload matrix was rebuilt from private dev commit
`c372a1e9` (the working tree had only an unrelated pending
`AudioCapture.h` formatting edit) with Clang Release, FP32, Intel Iris Xe TGL
GT2, Mesa 26.1.5, and Vulkan 1.4.354. Each executable ran one excluded warm-up
and seven measured fresh processes through `tools/diagnostics/oabench.py`.
All 120 processes completed training, evaluation, generation, checkpoint
round-trip, and GoogleTest assertions. This is a correctness pass for all 15
rows, including Mamba-3.

The host reported the `performance` profile as degraded (`lap-detected`), and
the package sensor reached 74 C during the run. The strict 70 C pre-run gate
could not be maintained after the preceding full Release rebuild, so this
attempt used an explicit 80 C package ceiling and is not a publishable
before/after performance comparison. A row is marked **stable** only when its
seven-process full spread is at most 10%; all other rows are retained as
diagnostic evidence and rejected for speed claims.

| Architecture | Tokenizer | Current median ms/step | MAD | Full spread | Status |
|---|---|---:|---:|---:|---|
| RNN | Byte | 11.09 | 0.74 | 29.67% | **REJECTED — spread** |
| RNN | BPE | 9.83 | 0.62 | 31.13% | **REJECTED — spread** |
| RNN | Char | 9.70 | 0.52 | 71.65% | **REJECTED — spread** |
| GRU | Byte | 19.40 | 1.06 | 25.93% | **REJECTED — spread** |
| GRU | BPE | 22.00 | 0.35 | 12.50% | **REJECTED — spread** |
| GRU | Char | 10.38 | 0.32 | 117.53% | **REJECTED — spread** |
| Transformer | Byte | 8.67 | 0.58 | 22.26% | **REJECTED — spread** |
| Transformer | BPE | 9.92 | 0.71 | 77.82% | **REJECTED — spread** |
| Transformer | Char | 4.45 | 0.10 | 19.33% | **REJECTED — spread** |
| Sparse MoE Transformer | Byte | 11.03 | 0.43 | 32.28% | **REJECTED — spread** |
| Sparse MoE Transformer | BPE | 21.14 | 5.08 | 58.61% | **REJECTED — spread** |
| Sparse MoE Transformer | Char | 6.09 | 0.06 | 9.69% | **STABLE diagnostic** |
| Mamba-3 | Byte | 16.65 | 0.99 | 25.47% | **REJECTED — spread** |
| Mamba-3 | BPE | 24.41 | 1.71 | 42.48% | **REJECTED — spread** |
| Mamba-3 | Char | 16.48 | 0.31 | 9.04% | **STABLE diagnostic** |

The two stable rows are useful current-tree observations, not revision deltas:
the paired `v0.6.105` baseline was not run under the same thermal state and the
host was explicitly degraded. In particular, the current Mamba-3 code is
functionally complete for this matrix and produces stable Char timing in this
attempt, but its Byte and BPE timing remains too noisy for an end-to-end speed
claim. A rested, clean-tree paired rerun is still required before replacing the
accepted twelve-row `v0.6.105` → `v0.6.106` comparison below.

## Private `v0.6.105` to `v0.6.106` checkpoint — 2026-08-12

This private engineering checkpoint compares exact `v0.6.105` commit
`47ce5f9c` with clean pre-squash candidate `5879d7ad`. Private tag `v0.6.106`
at `f640e131` records the same tested code; its tree differs from the candidate
only in CI and documentation. This is not a public package comparison. Both
revisions were rebuilt in detached worktrees with CMake Release, Clang 22.1.8,
FP32, and separate vcpkg installation roots. The test device was Intel Iris Xe
TGL GT2 with Mesa 26.1.5, Xe KMD, and Vulkan 1.4.354.

The complete five-architecture, three-tokenizer matrix ran one excluded warm-up
and seven measured fresh processes per executable on each revision. All 240
processes completed training, evaluation, generation, checkpoint save/load,
and their GoogleTest assertions. This is the checkpoint's complete NLP
correctness result.

The first performance matrix was rejected because the laptop entered persistent
bimodal performance states while the interactive desktop shared the integrated
GPU. Mamba-3 processes on the same binary alternated between approximately
30 and 74 ms/step, and revision ordering reversed across confirmations. Those
measurements cannot support a revision-performance verdict. All 48 Mamba-3
processes in the complete correctness matrix still passed; Mamba-3 performance
is excluded until its planned fused, reduced-submission rewrite is implemented
and remeasured.

The accepted performance confirmation covers RNN, GRU, dense Transformer, and
sparse-MoE Transformer. It ran baseline then candidate per workload, serially,
with the `performance` power profile and package temperature at or below 70 C
before every eight-process batch. The 12 workload pairs contain 192 additional
processes, all of which passed the same end-to-end correctness contract. The
table retains all 15 canonical rows so the three excluded Mamba-3 results cannot
be mistaken for missing tests. Delta is `(v0.6.106 code / v0.6.105 - 1) * 100`;
negative is faster. The gate fails only when a regression exceeds both 3% and
the larger baseline/candidate relative MAD noise band.

| Architecture | Tokenizer | `v0.6.105` median ms/step | `v0.6.106` code median ms/step | Delta | Noise band | Gate |
|---|---|---:|---:|---:|---:|---|
| RNN | Byte | 5.00 | 4.88 | -2.40% | 2.20% | PASS |
| RNN | BPE | 12.36 | 10.07 | -18.53% | 2.10% | PASS |
| RNN | Char | 7.22 | 6.66 | -7.76% | 0.97% | PASS |
| GRU | Byte | 10.84 | 10.78 | -0.55% | 1.67% | PASS |
| GRU | BPE | 21.81 | 22.02 | +0.96% | 2.68% | PASS |
| GRU | Char | 15.62 | 15.29 | -2.11% | 0.77% | PASS |
| Transformer | Byte | 7.79 | 5.21 | -33.12% | 2.69% | PASS |
| Transformer | BPE | 15.46 | 7.98 | -48.38% | 0.78% | PASS |
| Transformer | Char | 11.22 | 6.51 | -41.98% | 1.84% | PASS |
| Sparse MoE Transformer | Byte | 9.40 | 6.89 | -26.70% | 1.74% | PASS |
| Sparse MoE Transformer | BPE | 13.66 | 10.74 | -21.38% | 1.68% | PASS |
| Sparse MoE Transformer | Char | 16.30 | 12.33 | -24.36% | 0.80% | PASS |
| Mamba-3 | Byte | — | — | — | — | EXCLUDED |
| Mamba-3 | BPE | — | — | — | — | EXCLUDED |
| Mamba-3 | Char | — | — | — | — | EXCLUDED |

All 12 accepted comparisons pass. At the 3% materiality boundary, RNN Byte and
all three GRU rows are flat; RNN BPE, RNN Char, all three dense-Transformer rows,
and all three sparse-MoE rows improved. No accepted row regressed. Mamba-3 has a
complete correctness result but no defensible before/after performance number.
These are thermally controlled measurements of small educational workloads on
this specific integrated-GPU system, not an aggregate framework-speedup or
cross-device claim. In particular, absolute medians must not be mixed with the
historical OA 0.7.4 means below or with the rejected full-matrix timing pass.

## Private architecture-rewrite comparison — 2026-07-20

This engineering comparison accompanies the July 21 curated source refresh; it is not a
public-version or package comparison. It compares exact private tag `v0.6.100`
(`48a9a205`) with the clean pre-squash rewrite candidate `cd8c41bf`. Four later
source cleanups are present in private `v0.6.101` but were not part of the
candidate binaries, so these measurements must not be relabeled as an exact
`v0.6.101` or `v0.7.6` result.

Both sides were rebuilt in clean detached worktrees. Each side and executable
used one excluded warm-up followed by seven measured fresh processes, run
serially with `OA_LOG_TRAINING_PHASES=1`, FP32, and the system `performance`
power profile. Every process had to complete training, evaluation, generation,
checkpoint round-trip, and its GoogleTest assertions. The complete controlled
matrix therefore contains 240 processes.

The ThinkPad X1 Carbon is a thermally constrained integrated-GPU laptop, not a
fixed-clock benchmark host. The first full matrix overlapped interactive desktop
activity during Transformer rows and is excluded from the table. The confirmation
matrix waited for package temperature at or below 70 C before every baseline and
candidate batch, but the machine can still change clocks while a batch runs.
Rows whose verdict reversed between the two full passes remain inconclusive; no
aggregate framework speedup or regression is claimed.

Delta is `(rewrite / v0.6.100 - 1) * 100`; negative is faster. `FAIL` is the
benchmark runner's actionable-regression result at the 3% threshold after its
recorded noise band, not a correctness failure.

| Architecture | Tokenizer | Before median ms/step | Rewrite median ms/step | Delta | Gate | Cross-pass interpretation |
|---|---|---:|---:|---:|---|---|
| RNN | Byte | 4.72 | 5.18 | +9.75% | **FAIL** | Regression reproduced; first pass was +7.84% |
| RNN | BPE | 6.48 | 5.92 | -8.64% | PASS | Candidate favored twice; controlled baseline remained noisy |
| RNN | Char | 5.03 | 4.64 | -7.75% | PASS | Inconclusive; first pass was +5.53% |
| GRU | Byte | 10.45 | 10.56 | +1.05% | PASS | Flat within the gate in both passes |
| GRU | BPE | 10.91 | 12.32 | +12.92% | **FAIL** | Inconclusive; first pass was +1.00% |
| GRU | Char | 11.12 | 11.11 | -0.09% | PASS | Flat controlled result; first baseline was noisy |
| Transformer | Byte | 7.70 | 8.12 | +5.45% | **FAIL** | Regression direction reproduced; first pass was +5.74% |
| Transformer | BPE | 7.88 | 9.51 | +20.69% | **FAIL** | Inconclusive; first pass was -1.50% and desktop-contaminated |
| Transformer | Char | 6.92 | 6.96 | +0.58% | PASS | Flat controlled result; first baseline was contaminated |
| Sparse MoE Transformer | Byte | 8.81 | 8.73 | -0.91% | PASS | Flat across the two passes |
| Sparse MoE Transformer | BPE | 9.82 | 9.05 | -7.84% | PASS | Improvement reproduced; first pass was -4.87% |
| Sparse MoE Transformer | Char | 8.19 | 8.00 | -2.32% | PASS | Improvement direction reproduced; first pass was -6.18% |
| Mamba-3 | Byte | 35.39 | 33.49 | -5.37% | PASS | Controlled gain; first pass was flat at +1.02% |
| Mamba-3 | BPE | 33.40 | 33.89 | +1.47% | PASS | Flat within the gate in both passes |
| Mamba-3 | Char | 32.62 | 35.66 | +9.32% | **FAIL** | Inconclusive; first pass was -1.04% |

Ten rows pass the runner gate and five are flagged. Only Byte RNN and Byte
Transformer repeat the same greater-than-3% regression direction across both
full passes. BPE GRU, BPE Transformer, and Char Mamba-3 reverse their first-pass
verdict and remain explicitly inconclusive. BPE and Char sparse MoE show a
repeatable improvement direction.

Learning quality remained intact across all 240 controlled processes. Median
Byte final loss and accuracy were bit-for-bit unchanged between baseline and
candidate for all five architectures. BPE median accuracy was unchanged; Char
median accuracy moved by at most 0.1 percentage point. Fresh-process BPE/Char
initialization can change exact loss and generated text, so this comparison does
not claim bitwise determinism for those rows.

The remaining tables are the historical OA 0.7.4 release reference: one excluded
warm-up followed by three sequential measured processes per executable. All 60
processes passed training, evaluation, generation, and checkpoint assertions.
Those values are arithmetic mean ± population standard deviation and should not
be mixed with the seven-process medians above.

## OA 0.7.4 results

| Architecture | Tokenizer | Parameters | Wall ms/step | token/s | source byte/s |
|---|---|---:|---:|---:|---:|
| RNN | Byte | 31,104 | 5.00 ± 0.08 | 204.69 ± 3.09K | 204.68 ± 3.09K |
| RNN | BPE | 37,312 | 5.39 ± 0.17 | 189.99 ± 5.75K | 527.31 ± 15.97K |
| RNN | Char | 8,891 | 4.75 ± 0.09 | 215.66 ± 4.04K | — |
| GRU | Byte | 43,648 | 11.17 ± 1.26 | 92.82 ± 9.74K | 92.81 ± 9.74K |
| GRU | BPE | 49,856 | 10.85 ± 0.40 | 94.55 ± 3.39K | 262.43 ± 9.42K |
| GRU | Char | 21,435 | 10.92 ± 0.45 | 93.97 ± 3.84K | — |
| Transformer | Byte | 25,760 | 7.99 ± 0.63 | 128.94 ± 9.54K | 128.93 ± 9.53K |
| Transformer | BPE | 29,920 | 7.93 ± 0.45 | 129.50 ± 6.97K | 359.41 ± 19.36K |
| Transformer | Char | 10,875 | 6.89 ± 0.17 | 148.63 ± 3.63K | — |
| Sparse MoE Transformer | Byte | 28,068 | 8.62 ± 0.10 | 118.80 ± 1.39K | 118.79 ± 1.38K |
| Sparse MoE Transformer | BPE | 32,228 | 8.84 ± 0.14 | 115.78 ± 1.80K | 321.36 ± 4.99K |
| Sparse MoE Transformer | Char | 13,183 | 8.36 ± 0.34 | 122.77 ± 4.88K | — |
| Mamba-3 | Byte | 25,800 | 33.34 ± 0.46 | 30.72 ± 0.42K | 30.72 ± 0.42K |
| Mamba-3 | BPE | 29,960 | 33.39 ± 0.19 | 30.67 ± 0.17K | 85.12 ± 0.49K |
| Mamba-3 | Char | 10,915 | 32.45 ± 0.04 | 31.55 ± 0.03K | — |

## GPU execution

| Architecture | Tokenizer | GPU mean ms/step | p50 | p95 | Wall-GPU gap |
|---|---|---:|---:|---:|---:|
| RNN | Byte | 4.482 ± 0.063 | 4.224 ± 0.014 | 5.819 ± 0.391 | 10.7 ± 0.5% |
| RNN | BPE | 4.843 ± 0.167 | 4.652 ± 0.026 | 5.659 ± 0.729 | 10.3 ± 0.5% |
| RNN | Char | 4.213 ± 0.061 | 3.838 ± 0.007 | 5.996 ± 0.287 | 11.3 ± 0.5% |
| GRU | Byte | 10.498 ± 1.190 | 10.243 ± 1.008 | 12.398 ± 2.502 | 6.0 ± 0.0% |
| GRU | BPE | 10.225 ± 0.387 | 9.928 ± 0.067 | 11.704 ± 1.357 | 6.0 ± 0.0% |
| GRU | Char | 10.282 ± 0.433 | 9.772 ± 0.204 | 12.683 ± 1.461 | 6.0 ± 0.0% |
| Transformer | Byte | 7.033 ± 0.644 | 6.640 ± 0.367 | 9.060 ± 2.087 | 12.0 ± 1.4% |
| Transformer | BPE | 6.952 ± 0.454 | 6.559 ± 0.053 | 8.307 ± 1.714 | 12.3 ± 0.9% |
| Transformer | Char | 5.945 ± 0.150 | 5.706 ± 0.013 | 7.133 ± 0.965 | 13.7 ± 0.5% |
| Sparse MoE Transformer | Byte | 7.514 ± 0.071 | 7.405 ± 0.033 | 8.163 ± 0.304 | 13.0 ± 0.0% |
| Sparse MoE Transformer | BPE | 7.784 ± 0.146 | 7.616 ± 0.058 | 8.618 ± 0.814 | 12.3 ± 0.5% |
| Sparse MoE Transformer | Char | 7.228 ± 0.345 | 7.006 ± 0.174 | 8.165 ± 1.092 | 13.7 ± 0.5% |
| Mamba-3 | Byte | 32.319 ± 0.468 | 31.863 ± 0.110 | 37.012 ± 2.570 | 3.0 ± 0.0% |
| Mamba-3 | BPE | 32.386 ± 0.185 | 32.122 ± 0.002 | 34.737 ± 1.189 | 3.0 ± 0.0% |
| Mamba-3 | Char | 31.445 ± 0.040 | 31.358 ± 0.012 | 33.176 ± 0.405 | 3.0 ± 0.0% |

## Learning gate

| Architecture | Byte final CE / accuracy | BPE final CE / accuracy | Char final CE / accuracy |
|---|---:|---:|---:|
| RNN | 0.1877 / 92.30% | 0.0208 / 98.80% | 0.1833 / 92.83% |
| GRU | 0.1758 / 92.30% | 0.0208 / 98.80% | 0.1778 / 92.63% |
| Transformer | 0.1997 / 92.50% | 0.0205 / 98.87% | 0.1940 / 92.87% |
| Sparse MoE Transformer | 0.1947 / 92.20% | 0.0203 / 98.90% | 0.1955 / 92.87% |
| Mamba-3 | 0.2166 / 93.00% | 0.0221 / 98.87% | 0.1971 / 92.83% |

BPE accuracy is not directly comparable with Byte or Char. Its learned tokens
cover more source bytes and are trained on this same small teaching corpus.
Held-out bits per source byte is the appropriate production tokenizer metric.

## Preview-to-preview progress

The most important 0.7.4 change is sparse execution efficiency. GPU-written
dispatch arguments, deterministic route compaction/scatter, grouped projections,
packed Transformer projections and stable compiled replay close most of the tiny
MoE execution gap without changing the public model API.

| Tokenizer | 0.7.3 MoE/dense wall ratio | 0.7.4 ratio | Relative MoE overhead |
|---|---:|---:|---:|
| Byte | 1.46× | **1.08×** | 46% → 8% |
| BPE | 1.46× | **1.12×** | 46% → 12% |
| Char | 1.57× | **1.21×** | 57% → 21% |

Dense Transformer remains the default because this teaching corpus cannot prove
an expert-capacity quality advantage. MoE is now a credible optional capacity
path: the next gate is held-out quality at equal wall time, not basic executor
correctness.

## Reproduce

Build the 15 Release tutorials named `TutorialNlp{Byte,Bpe,Char}` ×
`{Rnn,Gru,Transformer,Moe,Mamba3}Ag`. For a release comparison, run one excluded
warm-up and at least seven sequential measured fresh processes per executable
through `tools/diagnostics/oabench.py`, with the power profile and thermal
conditions recorded. Do not compare these results with the historical
three-process arithmetic means as if they used the same estimator.

Every process must pass:

- 300 optimizer updates;
- finite and converged training loss;
- evaluation accuracy;
- deterministic generation checks;
- `.oam` save/load and optimizer-step round trip;
- the executable's GoogleTest assertions.
