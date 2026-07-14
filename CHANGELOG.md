# Changelog

All notable changes to OA are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/); versioning is the single
`VERSION` file at the repo root (read by CMake, `OaVersion()`, and the Python package).

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
