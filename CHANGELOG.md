# Changelog

## 0.17.0

- Rename `oa::Engine` to `oa::Runtime` across the public API (Python
  `oa.Engine` -> `oa.Runtime`).
- Rename the `mat_mul` operation family to `matmul`; the NT-layout GEMM
  is now `matrix::matmul` / `Matrix::matmul` / `oa.matrix.matmul`.
- Remove the `oa.ml.nlp`/`oa.ml.audio` bindings and the `sdk/` source
  tree; SDK workloads move to the standalone oasdk repository.
- Move the satellite NLP recipe out of the shipped library into the SDK.
- Publish the v0.16.1 vision milestones (never released publicly):
  Canny/CCL/distance-transform/Hough/contours (CV-1), ORB + Hamming kNN
  (CV-3), `pyr_down`/`pyr_up`/auto-threshold/`integral` (CV-6), and
  CLAHE/`calc_back_project`/HSV-Lab color/statistics breadth (CV-7).

## 0.16.1

- OpenCV-parity vision milestones CV-1, CV-3, CV-6, CV-7 (private tag;
  first published as part of 0.17.0).

## 0.16.0

- Complete the `nn` module surface: 3-D and transposed convolutions, the
  full pooling/unpooling family, padding modes, dropout variants, and the
  remaining misc modules; `oa::ml::nn` now exports 95 modules.
- Add `oa::ml::loss::ctc` with forward-backward DP kernels and generated
  autograd, plus nine composed losses and `nn::AdaptiveLogSoftmaxWithLoss`.
- Add `matrix::cast_f32_i32`/`cast_i32_f32`, the first dtype-cast ops.
- Add fused LSTM kernels, KV-cache Transformer decode, encoder-decoder
  cross-attention, GQA/MQA attention, gradient clipping, device-resident
  AMP scaling, ONNX graph import, QuantizedLinear, async-prefetch data
  loading, and image histogram statistics.
- Rename transposed-convolution functions to `conv_<dim>_transpose*`.
- Gate PipeWire audio capture behind an optional feature.

## 0.15.3

- Add TIFF to the image codec family: enable the `image` crate's `tiff`
  feature and route it through detection, extension inference, decode,
  and encode on the Rust and Python surfaces. Unlocks TIFF dataset packs
  (for example UC Merced land use).

## 0.15.2

- Fix shared-memory bank conflicts in the wide tiled F32 and F16 GEMM kernels and vectorize shared-memory tile loads across the tiled matrix family; drop a dead left-tile padding allocation. Measured `mat_mul_nt` throughput on Iris Xe improves on NLP/ALM representative shapes at the compiled-replay boundary.
- Update the `vkrust` dependency to 0.13.4 and serialize precision mode across kernel-census tests.
- Add a compiled-replay (`replay_submit_timed`) boundary to the matmul tutorial alongside `eager_submit_readback`.
- Record the VTune characterization (EU stall-bound) and rejected schedule experiments in the benchmark ledger.

## 0.14.1

- Fix a throughput regression: `Engine::current()` clones ran `.vkcache`
  persistence on every drop, rewriting the cache file per defaulted
  operation. Persistence now runs once when the last `Engine` clone
  drops. `tu_nlp_bpe_transformer` wall time on Iris Xe recovers from
  32.24 ms/step to 7.7 ms/step.
- Fix engine clones restoring the previous logger on drop; only the
  engine that installed the log selection restores it.

## 0.14.0

- Add `Engine::set_default`, `Engine::current`, and defaulted constructor twins for every engine-first value constructor across matrix, nn layers, ml models and checkpoints, audio, image, video frames, render textures, and the SDK library. Explicit `..._on(&engine, ...)` forms remain for engine selection; sessions, capture, presentation, and device queries keep an explicit `&Engine`. Breaking: engine-first value constructors move to the `_on` spelling.
- Consolidate the internal Device command surface under `runtime/device/logical`; `runtime/device.rs` is the single composed device API.
- Add occupancy-gated 128x64 wide-tile and FP16-storage wide GEMM variants with the admission gate derived from the device contract.
- Raise the trading observation element budget to 128M and allow packed market histories.
- Simplify SDK tutorials, examples, benchmarks, and tests to defaulted constructors; simple programs no longer thread an engine.

## 0.13.0

- Add Iris Xe replay qualification for the Vulkan Video and multimodal pipelines, including repeated decode, inference, overlay and encode.
- Preserve intervening buffer reads in fusion legality, allow valid fusion across later overwrites, and reject invalid region ranges before allocating slot metadata.
- Record the remaining device qualification and provider enablement decisions for the Experimental compute compiler.

## 0.12.1

- Harden Vulkan Video packet and image bounds, reduce host readback work, and tighten encode submission and playback synchronization. Split the Vulkan encode and decode implementations by codec.
- Refine plot backgrounds and axis labels, and open figures on the viewer’s GPU image canvas.
- Extend the Experimental compute compiler with opt-in tile providers, tuned BLAS routes, stricter fusion admission, and per-plan recording, memory, and queue-dependency evidence.
- Add Experimental compiled CartPole and LunarLander collection and expose per-plan diagnostics for PPO.

## 0.12.0

- Add `oa record` and portal/PipeWire `oa capture`, sharing the image/video GPU canvas, navigation camera, recording controls, elapsed timing and safe save-copy flow.
- Repair native video colors, hardware YCbCr routing, reverse scrubbing and AV1 backward-seek robustness; complete the schema-owned Python video encode/recorder surface.
- Document every Rust/Python NLP tutorial with its local loss curve and recorded terminal output.
- Fix SDL camera byte order, aligned-U8 staging and unnecessary texture-upload waits; preserve exact GPU producer completion.
- Repair release workspace version translation, native build/runtime dependencies and SDK application packaging; select the admitted software-Vulkan cryptography pack explicitly.

## 0.11.3

- Rename Strict to Core and select one Vulkan execution profile per build with `OA_BUILD_PROFILE=core|compatibility`; Core is the default and requires Vulkan 1.3 plus the admitted bindless features. Older hardware needs an explicit Compatibility build.
- Compile and embed one shader catalog and binding backend, removing the duplicate bounded artifact table and deriving Compatibility binding capacities directly from schema metadata. Preserve profile separation in JIT and `.vkcache` admission.
- Fix raster feature negotiation by querying/enabling `shaderDrawParameters` and declaring read-only storage buffers correctly, with shared read-only FP32 load helpers.
- Add profile-selection, artifact-inventory and hardware-admission proofs; qualify Compatibility replay/JIT/cache and raster pixel cases under core and synchronization validation.

## 0.11.2

- Replace the UI compute compositor with a Vulkan raster pipeline: `record_render` records one dynamic-rendering pass per frame, elements draw in recorded order with fixed-function alpha blending, Vulkan scissor provides clipping, and RGBA media samples through per-command image samplers; the 14 UI compute shaders are retired.
- Grow the retained `plot` model: log/symlog scales, error-band fill, grouped and stacked bars, box summaries, explicit tick control, bounded streaming series, and qualified figure file sinks (PNG/JPEG/WebP) with byte-identical cross-process headless gallery output on the Compatibility path.
- Add `plot::FigureView` interactivity: pointer-anchored wheel zoom, drag pan, shift-drag box zoom, reset, legend toggles, crosshair tooltip, serializable `FigureViewState`, and view-transition animation.
- Complete the `Ui` interaction surface: data table, filterable combo, color picker, multi-line text input, menus over the popup route, generic drag-drop payloads, keyboard and gamepad navigation, IME candidate-rect routing, and generated `UiStyle` token presets (`realm_dark` base; `realm_light` and `viewer_dark` derived).
- Add `ViewerMode::Workspace`: a serializable dock tree over `split_pane`/`tab_bar` with tab transfer, edge-drop split zones, empty-leaf collapse, the `oa workspace` subcommand, and `FigureView` as a rect-aware `WorkspaceItem`.
- Extend Python parity: the full recorded `Axes` surface, `FigureView` and `Workspace` bindings, and the gallery tutorial covering the artist batch.

## 0.11.1

- Add device-planar YUV video storage, host-planar upload, tier-1 native operations (resize, letterbox, plane stack, normalize), overlay/composite, range convert, and threshold/luma analysis behind schema-owned contracts, with exactly one fused `video::to_matrix` YUV→RGB/Matrix conversion boundary (VNV-01/02/03/04/06).
- Re-compose `nn::Empyrealm` from a Mamba-3 alias into a Nemotron-style hybrid: `EmpyrealmConfig` schedules interleaved Mamba-3/causal-attention mixers with dense or routed-expert feed-forward tails, and inference-only single-token decode advances caller-owned `EmpyrealmState` with per-block `Mamba3State`/`AttentionCache` KV state.
- Add `MultiHeadAttention::new_cache`/`step` plus `AttentionCache`; update `ByteEmpyrealm`, parity pins, Python bindings, and generated stubs for the hybrid contract.
- Fix DNN partition operation-consumption coverage exposed by SwiGLU-tail training captures.

## 0.11.0

- Add an engine-level fusion-decline census (OA-JIT-03a) and record real-workload census plus PPO phase-split evidence (OA-JIT-03b/c); reorder NLP kernel work by measured census data.
- Compile JIT specializations on a bounded background worker instead of blocking execution (OA-JIT-04); record clean regression checks on pinned Haswell clocks.
- Fuse activation adjoints into linear data-backward chains, decide L1 ownership for backward-anchored fusion, generalize the linear-backward producer prologue, and fuse the norm adjoint (OA-JIT-05/08); cover gated silu_mul adjoints and record char-transformer coverage.
- Admit DNN provider kernels before inserting lowered nodes, force the elementwise JIT route per run, and name the measured workload set.

## 0.10.3

- Add Experimental opt-in elementwise Slang specialization through a shared constrained fusion IR, with retained-output correctness and bounded artifact admission. Existing tape/source routes remain the fallback.
- Consolidate driver data, GEMM routes, JIT artifacts, and route observations in scoped `.vkcache` files with legacy import, bounded parsing, checksums, atomic writes, and OS advisory locking.
- Fix JIT device/query lifetimes, probe-versus-winner selection, cache restoration without a compiler, and artifact admission during long-chain partitioning.
- Add paired Haswell evidence and PPO benchmark coverage; correct historical NLP workload and JIT attribution. JIT remains disabled by default because no end-to-end training improvement is established.
- Archive the original port/JIT plans, preserve their links, and record the proposed next compiler work.

## 0.10.2

- Update to vkrust 0.12 and use `vk::mem` constructors and fluent setters for allocation records throughout the Vulkan backend and its consumer proof.

## 0.10.1

- Add Experimental lowering-only RL fusion for unbiased router probabilities with deterministic biased selection, bounded packed causal attention (seven-node split/score/softmax/context/merge chains and the three-node single-head region), dense/grouped gate-up SiLU epilogues, and two/three-input Concat-to-Linear projection. Preserve all source buffers needed by observers and backward passes.
- Extend the bounded elementwise tape with broadcast indexing, typed row-gather leaves and nine activation adjoints. Unsupported geometry and cross-invocation dependencies retain source execution.
- Evaluate PPO successor values once per rollout in compiled trading collection: pre-reset `next_observation` rows are retained by the `next_observation_append` kernel and evaluated in one batched GAE-plan pass, replacing a second per-step model evaluation; measured -6.7% mean collection GPU submission intervals over three rollouts on the MoE trader recipe on Intel HSW GT2 with identical loss trajectories.
- Complete compiled PPO update batching with one final untimed completion wait; retain bounded cursor-ring reuse waits and independent timed results.
- Fix cross-plan captured-buffer reuse: bound input buffers track their last bound consumer at the `Buffer` level, so host uploads wait for in-flight submissions that still read a shared original buffer.
- Build both CLI and SDK example targets before default release staging.
- Repair rustdoc references that resolved in the private tree but failed the public-crate `cargo doc -D warnings` gate.

## 0.10.0

- Add submit-time graph fusion over the joined executable graph, with no
  public recording API change:
  - An elementwise tape pass fuses adjacent legal F32 chains into one
    `MatrixFusedChainF32` dispatch, gated at 32768 elements on measured
    routing evidence; multi-STORE tapes materialize intermediates that
    autograd or external readers observe.
  - An epilogue pass fuses structural pairs into lowering-only kernels:
    `linear → add` residual, `linear →` one of nine unary activations,
    `add → layer_norm`, a z=3 shared-input `linear` triple for QKV
    projections, `embedding + embedding + add` (mixed U8/U32), and
    `sum_axis → scale`.
  - Record-site collapses remove purely internal dispatches: the
    `sum → scale` loss tail (`sum_scale`) across cross-entropy, masked
    cross-entropy, pointwise losses, and PPO, and the dual `sum_axis`
    pair (`sum_axis_pair`) in layer/RMS/channel norm backward.
  - Every fused kernel materializes all buffers its unfused sequence
    wrote; `fused_op_count`/`fused_node_count` diagnostics report the
    rewritten graph.
- Qualify Flash causal attention on Intel HSW GT2: roughly 4.8× slower
  forward at the tutorial shape, so `AttentionBackend::Auto` remains on
  the standard provider for this device class.
- Measured on the 16-model NLP tutorial suite: transformer graphs drop
  67 → 56 nodes and MoE graphs 82 → 73 with unchanged convergence.

## 0.9.1

- Consolidate kernel identity, artifact reflection, and BLAS routing under
  `runtime/dna`: `kernel.rs` plus the generated `kernel.gen.rs` manifest, and
  `provider/{blas,dnn}` with content IDs baked into build artifacts.
- Add the measured `mat_mul_nt_gemv` candidate (`GemmSmallM`, stable kernel ID
  636): a 256-lane workgroup reduction routed for `m == 1` decode shapes with
  n,k ≥ 512, measured 1.7–2.1× over the tiled route in oracle-gated tuner
  sweeps and roughly 3.5–4.9× faster end-to-end on Intel HSW GT2 in the
  exploratory baseline.
- Unify provider dtype vocabulary on canonical `DType` spelling (F32/F16)
  without changing serialized discriminants or external literal names.
- Update shared rule/template paths for the DNA layout and add the Haswell
  exploratory performance baseline.

## 0.9.0

- Migrate the Vulkan backend from Ash and vk-mem to native `vkrust` bindings,
  preserving device ownership, allocator synchronization, capability admission,
  and the public `oa` namespace.
- Persist the Vulkan pipeline cache under `var/vk` and preload admitted shader
  pipelines in parallel (`OA_SHADER_PRELOAD`, `OA_SHADER_LOAD_THREADS`) across
  compute, graphics, and UI pipeline creation.
- Generate methods, Python bindings, and constructor vocabulary from signature
  records, covering handwritten, template-embedded, session, codec, and export
  surfaces.
- Restore native `ItTraining` capture/replay for Python NLP tutorials with
  progress, GPU timing, metrics logging, and checkpoint roundtrips matching the
  Rust tutorials.
- Add a parameterized GPU trading research environment with PPO collection and
  minibatch replay, bounded linear attention, and execution and observation
  corrections.
- Gate heavyweight cryptography and device tests behind `OA_TEST_SLOW=1`.
- Extend cryptography and columnar dataset support; repair consumer builds,
  first-use pipelines, and viewer composition.
- Strip private changelog sections from public snapshots and normalize
  workspace formatting.

See [release notes](RELEASE_NOTES.md) for compatibility and remaining limitations.

## 0.8.6

- Publish the real `oarust` crate with an audited archive and unchanged `oa` namespace.
- Bundle SDL3 and gate registry publication on native and portable-wheel validation.
- Pin and checksum Slang 2026.14 for native and Python CI builds; repair the
  compiler mismatch that prevented v0.8.4 package publication.
- Consolidate the public Rust implementation into its first release checkpoint.
- Provide schema-owned Matrix, ML, autograd, Image, Audio and Vision APIs.
- Include runtime compatibility profiles, explicit device selection and
  software-Vulkan validation paths, media and cryptography/PQC integration.
- Correct Python NLP smoke controls and fresh-model checkpoint restoration.
- Exclude internal documents and private engineering tooling from publication;
  enforce public-source inventory checks in CI.
- Remove obsolete generator previews and the local crates.io reservation scaffold.

See [release notes](RELEASE_NOTES.md) for compatibility and remaining limitations.
