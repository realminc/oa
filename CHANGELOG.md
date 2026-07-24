# Changelog

All notable changes to OA are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/); versioning is the single
`VERSION` file at the repo root (read by CMake, `OaVersion()`, and the Python package).

## [0.7.7] — 2026-07-24 (public API convergence preview)

### Added

- **C++-parity Python root** — `from oa import *` exposes the verified
  PascalCase `Oa*` values and real `OaFn*` namespace modules. Generated
  runtime/native stubs keep Pylance and Pyright aligned with the live binding.
- **Paired Python tutorials** — Core, ML, Fashion-MNIST, Audio, Vision, Viewer,
  and all 16 Byte/BPE/Char NLP entries use the same public vocabulary and native
  Vulkan implementation as their C++ counterparts.
- **Format-neutral still-image I/O** — `OaImageDecoder` and `OaImageEncoder`
  provide semantic JPEG, PNG, BMP, TGA, and capability-gated WebP load/save
  without exposing backend-specific public classes.
- **Python matrix language parity** — matrix/matrix and matrix/scalar
  arithmetic, unary negation, and compound assignment now mirror existing C++
  `OaMatrix` operators.

### Changed

- `OaAudio` is the sole semantic whole-audio value across decode, operations,
  encode, viewer, C++, and Python. Redundant audio buffer/meta aliases are
  removed.
- Public NN generated headers use normal API names and installed include paths;
  generator ownership no longer leaks `.gen.h` spelling to consumers.
- `OaFilesystem`, `OaPath`, and `OaPaths` replace ad-hoc source-tree path
  construction in public Python examples.
- `OaViewer` is the compact direct-resource display path over the one
  `OaEngine` plus composed presenter; obsolete device-UI and split-engine
  concepts remain removed.
- Vulkan Video decode negotiates the exact codec profile derived from the
  stream rather than silently substituting a nearby profile.

### Verification

- The Python extension and complete binding suite passed 66 tests with two
  optional capability skips; fresh-process smoke and canonical 300-step
  quality runs passed all 16 NLP entries on Intel Iris Xe.
- Generic image codecs passed 11 native codec tests and 22 focused Python
  Vision/tutorial tests. The Core operator/tutorial checkpoint passed 23
  targeted Python tests and direct Iris Xe execution.
- The preceding private architecture checkpoint passed its focused
  Lunar/render, Audio, Video, NN/Fashion, camera, generated-source,
  architecture, and documentation gates. Exact `0.7.7` package and wheel
  results are recorded by the release workflow.

### Preview boundaries

- The public API and Python ABI remain pre-1.0. Lowercase Python domain modules
  are migration aliases, not a second canonical API.
- WebP depends on the build including libwebp. Vulkan Video remains
  codec/profile/device dependent and fails closed when the exact profile is
  unavailable.
- Lunar Lander 3D is an Experimental tutorial, not evidence for a complete
  generic renderer, physics engine, or learned PPO policy.

## [0.7.6] — 2026-07-21 (architecture-convergence preview)

### Changed
- Public `main` now carries the verified private `v0.6.102` architecture-convergence
  checkpoint: explicit execution/session ownership, immutable executable plans,
  deterministic semantic-to-executable provenance, broader schema-owned operations,
  engine-composed presentation, and removal of obsolete compatibility surfaces.
- Release-facing Build Week documents no longer link to private engineering paths, and
  the controlled architecture-rewrite NLP comparison is published with its thermal,
  estimator, correctness, and provenance limits intact.
- Generated-source drift remains strict for source and test artifacts while recognizing
  the one internal operations reference intentionally omitted from sanitized public trees.

### Verification
- The exact release-source code completed the full Release target matrix in an isolated
  worktree after a static-library reconfiguration, including all 15 NLP targets, and
  passed the Core MatMul oracle, focused allocator/kernel/engine/stream/graph runtime
  tests, architecture and diagnostic checks, and generated-source drift checks.
- The preceding controlled rewrite comparison completed 240/240 NLP training, evaluation,
  generation, checkpoint-round-trip, and GoogleTest processes with quality intact.
- The comparison retains five performance flags and names the two repeatable regressions;
  it does not claim an aggregate framework speedup.

### Preview boundaries
- This release advances the source, C++ packages, and Python wheel together to `0.7.6`.
  The controlled performance comparison predates the final release-source cleanup and is
  not presented as an exact `v0.7.6` benchmark or an aggregate speedup.
- The unfinished Lunar Lander 3D experiment is not part of this public checkpoint.

## [0.7.5] — 2026-07-18 (OpenAI Build Week preview)

### Added
- **Deterministic execution evidence** — completed training replays can emit
  `oa.execution_graph.v1` reports containing semantic-operation provenance, selected
  implementation and kernel-content hashes, resource lifetimes, alias groups, barriers,
  and the observed completion timeline.
- **Canonical architecture and migration contract** — one current architecture replaces
  conflicting historical rewrite plans, with explicit ownership, value/operation/session
  boundaries, compatibility seams, and acceptance gates.
- **Enforced module dependencies** — `oacheck` validates source-layer boundaries in CI
  and includes focused regression coverage for its parser and policy rules.
- **Build Week evidence package** — public Mobile Lab screenshots, a compressed OaAlm
  prompt-to-USD capture, reproducible desktop/mobile commands, and an honest before/after
  development record.

### Changed
- OaMobileLab release builds support explicit environment-based signing and expose the
  same bounded native training report through app-private files in debug packages or
  filtered logcat records in non-debuggable packages.
- The foreground Mobile Lab activity keeps its live training metrics visible; background
  execution remains owned independently by the service's bounded partial wake lock.
- The Android suite now treats generation quality as a release gate at the canonical
  300-step workload while retaining a bounded one-step packaging smoke test.
- Training graph reports are published only after completed replay, so planned work is
  not misrepresented as executed work.

### Fixed
- Packed Transformer projection no longer selects Q/K/V descriptors through a
  thread-varying bindless index. Explicit uniform branches preserve the single dispatch
  while producing correct results on both Intel ANV and Qualcomm Turnip.
- Raw-buffer `AddMatMul` planning again preserves its inference-only routing contract;
  a regression test protects the legacy selection behavior.
- Interrupted Android training now still emits a bounded diagnostic report instead of
  leaving automation waiting for a file that can never appear.

### Verification
- The signed release APK completed the five canonical 300-step Byte routes on a physical
  Adreno 610 phone, including forward, loss, backward, AdamW, evaluation, generation,
  `.oam` save/load, and exact checkpoint generation parity.
- Desktop Transformer and sparse-MoE routes completed the same 300-step acceptance gate;
  compute-graph regression coverage passes 20/20 on the Iris Xe reference system.
- The public snapshot is generated from the verified private tree and fails closed on
  internal documentation, credentials, workstation paths, and private-remote references.

### Preview boundaries
- The Build Week phone gate covers one Android/Adreno/Turnip configuration; it is not a
  universal mobile compatibility claim.
- Execution reports expose deterministic graph identity and observed completion, but do
  not yet replace a full trace profiler.
- The public API, Python ABI, and `.oam` format remain pre-1.0 contracts.

## [0.7.4] — 2026-07-16 (development preview)

### Added
- **OaBlasLt v1 planning path** — exact immutable plans now cover contiguous,
  arbitrary-stride and strided-batched FP32 matmul, ranked legal candidates, measured
  cache replay, and persisted median/p95/sample statistics.
- **OaDnn semantic planner** — validated operation graphs partition packed QKV, gated
  FFN, linear epilogues, residual normalization, Flash Attention and grouped MoE while
  retaining a portable fallback.
- **Generated BF16/CoopMat registry families** — subgroup/workgroup capability contracts,
  current epilogues, stable IDs and cache identity now come from checked OaTile schemas.
- **Public NLP benchmark** — the complete 15-model Byte/BPE/Char training matrix records
  wall/GPU distributions, learning gates and preview-to-preview MoE progress.

### Changed
- Multi-head attention preserves checkpoint-compatible Q/K/V modules while executing
  their projections through one reusable packed operation. FFN gate/up projections use
  the same API without concatenating or renaming model weights.
- Row compaction and scatter now write and consume exact GPU-side indirect dispatch
  arguments, including zero-work dispatches for empty selections.
- Compute graphs can materialize explicitly eligible, non-overlapping transient buffers
  as distinct bindless views over allocator-backed alias storage.
- The GEMM route cache keys exact offsets, row/column/batch strides and batch count;
  malformed or incompatible legacy entries remain fail-closed.

### Performance
- On the Iris Xe reference NLP matrix, sparse-MoE/dense Transformer wall ratios improved
  from `1.46x` to `1.08x` for Byte, `1.46x` to `1.12x` for BPE, and `1.57x` to `1.21x`
  for Char while preserving convergence, generation and checkpoint gates.
- Packed projections reduce launch plumbing without changing the public module or
  checkpoint contract. Dense Transformer remains broadly flat within laptop clock
  variance; the repeatable gain is the reduced sparse execution gap.

### Verification
- All 60 canonical NLP processes passed: 15 excluded warm-ups and 45 measured runs.
- MoE gradchecks pass 22/22 and systems tests pass 9/9, including shuffled repetition.
- GEMM routing passes 18/18; compute-graph/DNN planning passes 17/17; attention passes
  11/11; OaTile generation passes 6/6; generated-source drift is clean.

### Preview boundaries
- Split-K, persistent GEMM and portable serialized weight prepacking require a future
  multi-stage plan/workspace ABI and are not represented by placeholder routes.
- OaDnn planning is not yet automatically captured from arbitrary model/autograd graphs.
- BF16/CoopMat execution still requires fresh validation on capable dGPU hardware.
- This release adds no new Python API surface; existing bindings inherit the runtime and
  training improvements.

## [0.7.3] — 2026-07-14 (development preview)

### Added
- **OaMobileLab** — an Android Vulkan application that trains, evaluates, generates,
  saves, and reloads the five canonical Byte NLP architectures on a physical Adreno
  device. The controlled desktop/mobile reference covers RNN, GRU, Transformer,
  sparse-MoE Transformer, and Mamba-3.
- **OaAlm end-to-end product path** — native frozen CLIP text conditioning, temporal
  Conv1d VQ motion tokenization, a dense Transformer prior, held-out validation,
  one-file `.oam` deployment, and prompt-to-USD generation.
- **GPU-native dropless sparse MoE** — expert planning, packed selected routes, grouped
  projections, fused route normalization/combine, load balancing, telemetry,
  checkpoints, and dense-oracle forward/gradient parity.
- **Curated Vision surface** — 50 schema-backed graph-native NCHW operations with CPU
  oracles and Python coverage; image codecs and preprocessing compose through the same
  context graph.
- **Native media pipeline** — H.264/H.265/AV1/VP9 Vulkan Video decode paths, H.264/H.265
  encode surfaces, camera/screen capture, native MP4/Matroska/WebM/MPEG-TS containers,
  recording, transcoding, completion tokens, and audio/video synchronization plumbing.
- **Audio module** — WAV/FLAC/MP3 decode, WAV-F32 output, PCM16 streaming, capture and
  playback surfaces, plus 15 curated GPU DSP/feature operations.
- **Crypto module** — strict host hashing/signature APIs, optional liboqs integration,
  and GPU batch hashing/public-data acceleration with differential tests.
- **Module-shaped Python API** — `oa.core`, `oa.runtime`, `oa.ml`, `oa.vision`,
  `oa.audio`, and `oa.crypto` are assembled from responsibility-scoped nanobind units
  while retaining one native extension and one shared type registry.
- **Model weight-transfer framework** — reusable inspect/map/convert/verify contracts
  replace architecture-specific SafeTensors loaders.

### Changed
- Training metrics now distinguish samples, tokens/bytes/frames, wall time, and GPU
  time; validation loss, quality metrics, and best-checkpoint restore are first-class.
- Attention and Transformer ownership were consolidated, and ALM can select dense, MoE,
  or hybrid FFNs without changing the product wrapper.
- Runtime barriers cover indirect-command consumption correctly; normal training,
  routing, compaction, dropout, accuracy, and mask paths no longer perform tensor-sized
  host loops.
- Cold native-pipeline preload uses a dynamic multi-threaded queue and per-worker Vulkan
  pipeline caches, then merges them safely. On the 4C/8T X1 Gen 9 reference system it
  reduced mean cold preload from 9.592 s to 3.504 s (2.74x); warm startup remains serial.
- Empty generated artifacts were removed across Core, ML, Audio, and Vision; schemas now
  emit only declarations and translation units that contain real generated code.

### Fixed
- `SiluMul` and `Geglu` shape-changing dispatch bounds no longer write one output tensor
  beyond their allocation, eliminating allocator-order-dependent MoE failures.
- Mamba-3 backward short-sequence routing and Android bounded backward execution were
  corrected and covered by shuffled/order-independent tests.
- Native H.264/H.265/VP9/AV1 parsing, DPB retention, display ordering, queue-family
  synchronization, and YCbCr conversion were hardened against the real Shibuya fixtures.
- ALM now consumes the full variable-length corpus with masked loss instead of silently
  filtering almost all short clips; CLIP conditioning reaches both training and native
  prompt inference.
- Metric formatting no longer hides small loss changes or reports invalid accumulated
  GPU timing after interrupted multi-epoch training.

### Preview boundaries
- The API, Python ABI, and `.oam` format remain pre-1.0 and may change.
- BF16/cooperative-matrix performance and the full codec-by-vendor matrix still require
  additional hardware validation.
- OaAlm is a working small-model demonstration, not a production-quality universal
  motion generator.
- Crypto has not received an independent security audit.

## [0.7.2] — 2026-07-10 (development preview)

### Added
- **Intel Xe / iGPU support** — the X1 Carbon Gen9 (Tiger Lake Iris Xe) now runs the
  NLP byte-GRU tutorial end-to-end (fp32, ~93.8% acc) with no segfault.
- `OaVersion()` runtime accessor + single-source `VERSION` file (`0.7.0-dev`) driving
  CMake, C++, and the Python package.
- **BF16 vendor-trust gate** (`OaBf16Trust`) — mirrors the CoopMat gate; distrusts Intel
  pre-Xe2 / AMD pre-RDNA3 drivers that advertise bf16 but miscompile it. `OA_FORCE_BF16`
  overrides, `OA_DISABLE_BF16` forces off.
- `OA_BINDLESS_BUFFER_CAP=N` runtime override for the bindless buffer capacity.
- **CI GPU smoke** — `gpu_smoke` ctest label + a self-hosted `gpu-smoke` CI job so the
  bindless compute path is validated on real hardware (gated by `HAVE_GPU_RUNNER`).
- README **supported-hardware matrix** + **Status & limitations** section.
- Intel **VTune** GPU-profiling workflow in the profiling docs/rules (Nsight analog).
- Capability-gating **design** for cross-vendor Vulkan Video (queue-codec cross-check +
  topology-gated conversion path) in `VideoDecoderDeviceCompatibility.md`.

### Changed
- **`OaShape` → `OaMatrixShape`** (rank-≤8 N-D shape). The `OaShape1D..5D` helper
  functions are removed — use brace-init `OaMatrixShape{d0, d1, ...}` for any rank.
  Autogen generators updated so codegen stays consistent. `Shape.h` is a forwarding shim
  (remove with `git rm` once nothing includes it).
- Bindless buffer cap is now **integrated-GPU-aware** (1M discrete / 256K integrated+CPU).
- GEMM benchmark (`TutorialCoreMatMulIntro`) reports **min-of-N** (reproducible peak)
  instead of the mean.
- Vision docs consolidated **6 → 4** (`OaVideo.md` merges the two architecture docs).

### Fixed
- **CoopMat preload crash on Tiger Lake Xe** — the preload gate matched a stale
  `"CoopMat"` substring that the renamed `GemmCmSg/CmWgBf16` kernels don't contain, so a
  bf16 workgroup-coopmat pipeline reached `vkCreateComputePipelines` and the Mesa/ANV
  driver died. Now gated by real registry caps + vendor trust in `ComputeCapsMask`.
- **Release-build asserts that vanished under `NDEBUG`** — load-bearing guards
  (`FnContext` null runtime/graph + buffer-count mismatch, `GpuTimer`) now log and
  fail-safe instead of segfaulting.
- **Silent result corruption guard** (`Stream.cpp`, the GRU fused-kernel bug class) is
  now always-compiled — it refuses the dispatch instead of only warning in debug.
- **`OaGru`/`OaRnn::Forward`** wrong-rank input now returns empty (loud, non-corrupting)
  instead of falling through to an O(seq²) mis-slice in Release.
- Version-string triple-mismatch (`0.5.84` / `0.6.38` / git) collapsed to one source.
- README profane sign-off removed; dead doc links repointed to real files.
- Defensive `VideoDecodeQueue == nullptr` guards at the decoder submit sites.

### Performance
- `GemmTiled` (the fp32/iGPU fallback GEMM): 128-bit `float4` vectorized loads +
  software-pipelined register-prefetch double-buffering. ~+25–60% at small/medium
  shapes on the Xe; large shapes are occupancy-bound (tuning pending on-device profiling).

### Notes
- Video decode remains **experimental** in this historical release (H.264 solid on
  NVIDIA; iGPU conversion and broader codec coverage were the next milestone).
- Distributed/multi-node, MoE training, and video encode are not part of the preview
  (they return explicit errors, not silent failure).

## [0.6.77] — 2026-07-08

### Fixed
- Intel Xe iGPU brought up: integrated-aware bindless cap kills the 100M-slot OOM /
  minute-long init; per-variant CoopMat preload gating; `GemmTiled` vectorized fp32 loads.

### Changed
- MotionGPT/Gen3dAnim training refinements (Api2 path, autograd variant, fused Conv1d
  backward, sliding-window LM epochs, multiphase LR). Docs cleanup (removed internal
  business plans from the tree).
