# Changelog

All notable changes to OA are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/); versioning is the single
`VERSION` file at the repo root (read by CMake, `oa::version()`, and the Python package).

## [0.7.17] — 2026-08-25 (source-owned SDK example proof)

This patch closes the beginner SDK example set as executable C++ and Python
product evidence while preserving explicit engine ownership and hardware-scoped
claims.

### Added

- **Minimal native application boundary** — `OA_MAIN`, `OA_MAIN_MODE`, and
  `OA_MAIN_PREVIEW` create exactly one engine for a lexical application body
  without introducing a singleton, implicit submission, or a second runtime.
- **Complete beginner example set** — generated Core, Audio, Crypto, Vision, and
  Ui programs use the umbrella `oa/oa.h` header and matching `import oa` surface.
- **Semantic grayscale** — `oa::FnImage::grayscale` and
  `oa.FnImage.grayscale` preserve `oa::Image` extent/layout while producing a
  Rec.709 Gray image through the existing GPU kernel.
- **Rendering mode seam** — the public renderer configuration distinguishes
  rasterization from a reserved ray-tracing request; the latter fails closed
  until a qualified implementation exists.

### Changed

- Shader pipelines loaded on demand report one compact batch summary instead of
  one log pair per pipeline.
- Vision and Ui examples save checked image outputs and publish exact 1280×720
  Viewer captures. The retained figure example is now correctly owned by
  `sdk/{cpp,py}/examples/ui` rather than a nonexistent Plot module.
- Python native-extension discovery accepts the canonical lowercase `build/`
  tree used by local VS Code example launches.

### Verification

- Generated C++ and Python Core, Audio, Crypto, Vision, and Ui examples execute
  on the recorded Intel Iris Xe Vulkan device. Semantic grayscale matches an
  independent Rec.709 CPU oracle; generated-source, asset, documentation, and
  developer-site drift gates pass.
- Hosted build, sanitizer, package, wheel, release, and PyPI results remain
  owned by the tagged public workflow.

## [0.7.16] — 2026-08-24 (host sanitizer gate repair)

This patch preserves the `0.7.15` SDK, Audio, and Viewer surface while repairing
the tagged host-sanitizer workflow. The `0.7.15` public tag remains immutable.

### Fixed

- The ASAN/UBSAN build target list now includes `TestVlm`, which is selected by
  the `core` CTest label. The `0.7.15` workflow built and passed every requested
  sanitizer executable but failed when CTest could not find this omitted target.

## [0.7.15] — 2026-08-24 (native audio reverb and executable SDK evidence)

This preview turns the paired SDK examples into small, executable product proofs
while keeping C++ and Python generated from one checked inventory. It adds no new
cross-vendor qualification or performance claim.

### Added

- **GPU-native room reverb** — `oa::FnAudio::reverb` and `oa.FnAudio.reverb`
  apply the same Freeverb-derived comb/allpass topology through four schema-owned
  Vulkan kernels, with an explicit output tail and a CPU differential oracle.
- **Unified preview entry point** — `oa::Viewer::preview` and
  `oa.Viewer.preview` open checked image, audio, or video paths; direct Matrix and
  Image values retain the same presentation owner.
- **Executable example evidence** — the Audio example decodes speech, normalizes,
  fades, reverberates, saves WAV output, and optionally previews it. The checked
  example manifest publishes playable before/after assets and a cropped,
  GTK-framed JPEG captured from the exact generated C++ program.

### Changed

- Beginner examples use lazy Python runtime ownership and paired C++/Python
  generation instead of maintaining independent snippets or manual DSP glue.
- SDK module cover, audio, image, video, and documentation assets use one checked
  inventory with camelCase filenames, exact hashes, dimensions, and provenance.
- The public README now leads with the end-to-end One API/Open Architecture pitch
  and presents Python as a front end over the same native Vulkan runtime.

### Verification

- The generated C++ and Python Audio programs both execute native reverb and the
  960×360 Viewer path on the recorded Intel Iris Xe; generator, stub, asset,
  documentation, drift, and focused Python example gates pass.
- Broader build, sanitizer, packaging, and hosted release results remain owned by
  the tagged public workflow.

## [0.7.14] — 2026-08-24 (unified ML and Vulkan-native spatial math)

This source-breaking preview removes the remaining parallel RL, training,
model-artifact, and spatial-math vocabularies. It promotes no new hardware or
performance claim.

### Added

- **Native ML reinforcement surface** — environment, rollout, replay, policy,
  advantage, PPO, DQN, and SAC contracts now live directly under `oa/ml` and
  reuse the ordinary engine, matrices, modules, losses, optimizers, and
  training lifecycle. Stateless Python operations are split into
  `oa.FnAdvantage`, `oa.FnEnvironment`, and `oa.FnPolicy` instead of one broad
  `FnRl` namespace.
- **Vulkan Linear Math** — `oa::vlm` provides the bounded vector, quaternion,
  matrix, transform, projection, interpolation, and coordinate operations used
  by current OA and SDK workloads with one explicit right-handed Vulkan
  convention.
- **Model-file contracts** — model metadata, tensors, read/write, conversion,
  and checkpoint paths share the named model-file and weight-transfer
  boundaries; SDK dataset archives remain outside the production ML API.

### Changed

- Concrete CartPole and Lunar Lander environments, their task kernels, and
  application-specific helpers are SDK material. The framework retains the
  reusable environment execution contract and GPU operations.
- Ordinary training uses one documented `oa::ItTraining` lifecycle. Algorithm
  trainers compose it at their update boundary instead of inheriting a generic
  trainer body.
- `oa::QuantMatrix` names the semantic Q4/Q8 inference value directly; legacy
  `QuantizedMatrix`, `oam.h`, `training.h`, `nnType.h`, and GLM bridge surfaces
  are removed rather than retained as compatibility aliases.
- Vendored GLM and per-call GLM/Vulkan axis conversion are replaced by the
  bounded native `oa::vlm` implementation. External coordinate conversion now
  belongs once at each format boundary.

### Verification

- The release candidate is required to pass generated-source drift, internal
  and external documentation checks, architecture checks, the complete
  applicable Release suite, sanitizer profiles, package installation, Python
  surface tests, and the hosted public workflow before publication is called
  complete.

## [0.7.13] — 2026-08-23 (SDK, API documentation, and hosted Vulkan proof)

This preview preserves the canonical 0.7.12 API while tightening the SDK,
generated-reference, artifact-layout, and public CI boundaries. It promotes no
new hardware-performance claim.

### Added

- **Hosted CPU-Vulkan smoke** — public CI selects Mesa Lavapipe explicitly and
  runs filtered, no-skip engine, graph, and GRU correctness cases. The profile
  records its ICD and capability evidence and does not replace physical GPU,
  validation, media, or performance gates.
- **Source-owned API documentation contract** — public C++ declarations use
  clang-doc-compatible `///` comments with parameter directions, return
  behavior, ownership, synchronization, and failure contracts. The generated
  developer reference now publishes a complete detailed `oa::Matrix` entry and
  all extracted Matrix members from the checked header.
- **SDK roadmap authority** — one SDK roadmap owns the example, tutorial,
  application, Android laboratory, asset, and qualification backlog, including
  the former standalone Lunar Lander plan.
- **GitHub Sponsors metadata** — the correctly cased funding manifest points to
  Realm's approved organization profile.

### Changed

- Native executable outputs omit the redundant language segment and use
  `bin/<preset>/{app,example,tutorial,test}/<domain>`; source and SDK roots retain
  their explicit `cpp` and `py` ownership axes.
- Nested repository and extension dependency boundaries are explicit in the SDK
  contract and agent rules instead of being inferred from the parent checkout.
- Documentation authoring provides paired header/implementation templates:
  generated public behavior lives once in headers, while source files contain
  only non-duplicated implementation rationale.

### Verification

- Local Iris Xe execution passes the three filtered CPU-Vulkan profile tests
  with runtime skips forbidden. The public Lavapipe result is owned by the
  tagged GitHub Actions run and is not claimed before that run completes.
- C++/Python API generation, documentation boundary checks, rule synchronization,
  architecture/diagnostic checks, and generated-source drift checks pass on the
  release candidate.

## [0.7.12] — 2026-08-22 (canonical API and SDK layout)

This is an intentionally source-breaking pre-1.0 checkpoint. It publishes the
canonical naming, source, SDK, documentation, and package layout after the full
private rewrite. It changes no accepted performance baseline and promotes no
additional hardware pack.

### Added

- **Canonical SDK tree** — maintained C++ and Python examples, tutorials,
  applications, tutorial datasets, environments, and model suites now live under
  `sdk/{cpp,py,android}` instead of framework headers or parallel top-level trees.
- **Native Python NLP suite parity** — Python exposes the SDK-backed recipe,
  sampler, model, and runner used by the five-architecture by three-tokenizer
  canonical NLP matrix. The separate Byte Empyrealm executable remains a
  regression workload rather than a sixteenth canonical architecture.
- **Installed-package consumer gate** — tag CI merges the staged runtime and SDK
  components, resolves exact `0.7.12` through `find_package`, and compiles and
  links a clean external CMake consumer before creating packages.
- **Executable paired examples** — one checked manifest owns matched C++ and
  Python examples for Core, Audio, Vision, Crypto, and Plot. Generated developer
  documentation consumes the exact checked source.
- **GPU Vision augmentation walkthrough** — matched C++ and Python tutorials
  decode one source image, apply five deterministic `oa::FnImage` operations,
  compose the views through retained Plot/Ui, and encode one terminal PNG.

### Changed

- C++ uses lowercase `oa` and internal `oavk` namespaces, PascalCase types,
  camelCase methods/functions, camelCase parameters and locals, and lowercase
  include paths. Python mirrors the public type and operation names while root
  and lowercase-domain imports resolve to the same objects.
- Native implementation and headers live under
  `source/cpp/{include,lib,thirdparty}`; Python bindings and package code live
  under `source/py`. Extensions use the same `include`/`lib` convention.
- Tutorial-only datasets, Lunar Lander, ALM, GPT-OSS, and the NLP suite no longer
  expand the production framework surface; the wheel links the required SDK
  support library only for explicitly bound tutorial-suite parity.
- Public tutorials, examples, benchmarks, assets, and authoring guidance live
  under `docs/external`. Private engineering evidence is excluded from public
  snapshots.
- Viewer defaults and documentation use the checked Space Cathedral JPEG. UI
  text-atlas and detection-overlay owners use typed opaque storage rather than
  owning raw pointers or `void*` state.

### Fixed

- Generated headers install to their lowercase canonical paths (`fnmatrix`,
  `fnloss`, `fnaudio`, and `fnimage`) on case-sensitive systems.
- CI sanitizer, validation, and wheel jobs use the canonical lowercase build and
  executable paths and assert the current Python surface.
- The scikit-build wheel directory follows the canonical lowercase build layout.
- MNIST tutorials include their SDK dataset owner directly instead of relying on
  accidental transitive framework includes.

### Verification

- Applicable native host, Iris Xe GPU, and Iris Xe media profiles pass 90/90
  tests with runtime skips forbidden. ASAN/LSAN and UBSAN each pass 16/16
  applicable core tests.
- Separate Vulkan core, synchronization, and GPU-assisted validation profiles
  report zero errors on the recorded Intel Iris Xe device.
- A clean raw CPython 3.12 wheel installs outside the source tree, exposes the
  exact root/module identities and typing payload, and passes 33 host, 88 GPU,
  8 crypto-host, and 10 crypto-GPU tests with skips forbidden.
- The staged native package contains 31 runtime and 710 SDK files, including 259
  headers and 369 SPIR-V modules; its exact-version external consumer compiles
  and links. All 162 generated outputs and 133 checked Markdown documents are
  clean.

## [0.7.11] — 2026-08-17 (GPU Audio and multi-engine runtime)

This preview promotes the verified post-`0.7.9` source line. Existing `0.7.9`
packages remain immutable; all new source and binary artifacts use `0.7.11`.
The `v0.7.10` tag remains the immutable failed prerelease marker: its UBSAN
suite passed every built test but rejected the unbuilt, newly core-labelled
`TestVkDispatch` executable before release creation or PyPI publication.

### Added

- **GPU-native Audio effects foundation** — schema-owned saturation,
  block-parallel zero-state Biquad, and one-to-64-section zero-state SOS
  filtering share the semantic Audio value, C++/Python operation surface,
  explicit graph ownership, and independent host oracles. SOS reuses the three
  Biquad passes with two waveform buffers and bounded shared workspaces rather
  than creating another shader family.
- **Python Audio parity** — `OaBiquadCoefficients` plus
  `OaFnAudio.Saturate`, `Biquad`, and `SosFilter` ship in the native wheel,
  root/module exports, generated stubs, and clean-install surface tests.

### Changed

- Vulkan instance/device dispatch is immutable per `OaVkDevice`. Constructing a
  second engine no longer replaces entry points used by the first; only loader
  discovery remains process-global. Physical cross-vendor placement and
  transport remain hardware-unverified.

### Fixed

- Logger selection remains safe during static-engine shutdown after
  thread-local logger state has already been destroyed.
- Release checksum finalization now verifies the complete public asset set
  after the independently built wheel is attached.
- Host sanitizer jobs explicitly build `TestVkDispatch` before running the
  complete `core`-label suite.

### Verification

- The dispatch-isolation checkpoint passed the complete 148-test Release suite
  with zero skips, focused ASAN/UBSAN, and separate core/synchronization
  validation on Intel Iris Xe.
- The final Audio surface passes 45/45 native cases, 12/12 Python tests, and
  26/26 DSP tests under ASAN/LSAN and UBSAN. Separate core, synchronization,
  and focused GPU-assisted validation report zero errors on the recorded Iris
  Xe capability pack.
- FnAutogen passes 31/31, KernelAutogen passes 4/4, all 162 generated outputs
  are clean, schema depth passes 315/315 contracts, public operation coverage
  passes 431/431, and the strict 369-kernel manifest audit is clean.

## [0.7.9] — 2026-08-15 (release-gate repair)

### Fixed

- The public sanitizer workflow now builds `TestMcp` before running the
  `core`-label suite. The failed `v0.7.8` workflow registered that test but did
  not request its executable, so CTest correctly rejected the incomplete gate;
  this was a CI target-closure failure, not a sanitizer finding.
- Five Fashion-MNIST and Byte NLP tutorials no longer include the removed
  `Oa/Runtime/GpuTimer.h` compatibility header. A stale workstation SDK had
  masked those dead includes; the clean public runner exposed them.
- Wheel builds and their clean-install stub gate now use the same pinned
  nanobind `2.14.0`. The failed candidate installed 2.14 for compilation and
  then resolved 2.15 in its later smoke environment, allowing stubgen output to
  change while one tagged workflow was running.

`0.7.9` otherwise carries the same engine, inference, Mamba-3, MCP, Ui/Plot,
benchmark, and hardware-scoped feature set documented for `0.7.8` below.

## [0.7.8] — 2026-08-15 (engine, inference, and presentation convergence preview)

### Added

- **Semantic Q4/Q8 inference** — `oa::QuantMatrix` and
  `oa::Quantization::{Q4,Q8}` provide one explicit pack/dequantize contract, a
  fused quantized-weight `MatMulNt` route, and Dense/Q4/Q8 `.oam` v3 storage.
  Quantization is an inference representation rather than a fake training
  dtype; FP64 vocabulary fails closed until its numerical pack exists.
- **Local MCP control plane** — `OaMcpServer` implements bounded local stdio
  JSON-RPC/MCP requests, while `OaMcpTraining` exposes guarded inspection and
  training-control commands. It does not claim remote transport or tensor
  streaming.
- **Grouped Mamba-3** — grouped B/C SISO state, deterministic short backward,
  capability-gated long/subgroup routes, and shared-state grouped MIMO
  forward/backward/recurrent-step kernels now have numerical and streaming
  correctness coverage.
- **GPU Plot gallery** — the retained `OaPlot` surface now renders Line,
  Scatter, Bar, Histogram, Heatmap, and Image artists with explicit X/Y data,
  adaptive square grids, rounded legends, dark/light themes, deterministic
  headless PNG output, and paired C++/Python walkthroughs.

### Changed

- `OaEngine` is the sole concrete local execution owner. It is factory-created,
  pinned, opaque, and composed by optional services. Public execution uses
  explicit `Submit`/`OaEvent` completion and reusable `OaExecutionPlan`
  capture; Vulkan graphs, contexts, allocators, routers, and pipelines remain
  private lowering details.
- Kernel compilation, embedding, stable identities, capabilities, route rows,
  hashes, and validation fixtures now derive from one manifest. Dynamic SPIR-V
  provider/search-path state and the unused extension registry were removed.
- Public APIs consistently separate semantic values, stateless `OaFn*`
  operations, and stateful sessions. Codec helpers and obsolete `Destroy()`
  compatibility wrappers no longer form parallel public routes.
- Logging is engine-owned and event-capable, with extensible component names
  and PascalCase `OaLog*` entry points. The private repository no longer spends
  hosted CI time; release automation is repository-gated to `realminc/oa`.
- Rendering, Ui, Viewer, and Plot use the single `OaRenderer` ownership path.
  Bounded capability-gated MSAA, explicit readback, temporal Viewer controls,
  and GPU-side plot composition replace duplicate or host-raster paths without
  claiming a general scene renderer.
- Python tests live under the mirrored test hierarchy, run through explicit
  capability profiles, and now include one-to-one Plot bindings, stubs,
  tutorials, and output assets.

### Performance and verification

- The controlled private `v0.6.105` → `v0.6.106` Iris Xe comparison passed all
  12 accepted non-Mamba rows. Dense Transformer improved 33–48% and sparse MoE
  21–27%; no accepted row regressed. The benchmark document retains the exact
  commits, seven-process medians, spread, and rejected measurements.
- Rewritten SISO Mamba-3 completes the canonical 300-step workloads at stable
  17.42 ms/step Byte and 17.25 ms/step BPE medians on the recorded Iris Xe pack.
  The invalid high-spread pre-rewrite baseline is not promoted as a speedup;
  long-context quality and MIMO workload performance remain unmeasured.
- The final Plot/Ui candidate passed 54/54 tests under Release, ASAN/LSAN, and
  UBSAN, the selected Python smoke/tutorial profile passed 34/34 with skips
  forbidden, all six C++ and Python gallery outputs were deterministic, and
  separate core, synchronization, and GPU-assisted validation profiles were
  clean on Intel Iris Xe. Complete release-source and package results are
  recorded by the release workflow.

### Preview boundaries

- The exact current capability pack is Intel Iris Xe FP32. The older physical
  Adreno NLP proof remains historical evidence; the complete rewrite still
  requires new Adreno, NVIDIA, AMD/RADV, Strix Halo, and datacenter runs.
- BF16/cooperative-matrix routes remain capability- and trust-gated. Physical
  heterogeneous distribution and device collectives remain unverified.
- `OaRenderer` is deliberately bounded, Plot is not yet interactive, MCP is
  local-only, and Crypto has no independent security audit.

## [0.7.7] — 2026-07-24 (public API convergence preview)

### Added

- **C++-parity Python root** — `from oa import *` exposes the verified
  PascalCase `Oa*` values and real `OaFn*` namespace modules. Generated
  runtime/native stubs keep Pylance and Pyright aligned with the live binding.
- **Paired Python tutorials** — Core, ML, Fashion-MNIST, Audio, Vision, Viewer,
  and all 16 Byte/BPE/Char NLP entries use the same public vocabulary and native
  Vulkan implementation as their C++ counterparts.
- **Format-neutral still-image I/O** — `OaFnImage` provides semantic JPEG,
  PNG, BMP, TGA, and capability-gated WebP decode/encode operations without
  exposing backend-specific public classes.
- **Python matrix language parity** — matrix/matrix and matrix/scalar
  arithmetic, unary negation, and compound assignment now mirror existing C++
  `OaMatrix` operators.

### Changed

- Stateless audio decode and one-shot WAV-F32 encode now live on `OaFnAudio`,
  matching the image operation family. `OaAudioStreamEncoder` remains the
  stateful packetization session; the former static decoder/encoder classes are
  removed.
- Lunar Lander scalar reference physics is exposed as schema-owned
  `OaFnRl::LunarLander3d*` operations; terrain, state, and environment lifecycle
  remain explicit value/session classes.
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
- Release-facing historical documents no longer link to private engineering paths, and
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

## [0.7.5] — 2026-07-18 (mobile training preview)

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
- **Mobile Lab evidence package** — public screenshots, a compressed OaAlm
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
- The Mobile Lab phone gate covers one Android/Adreno/Turnip configuration; it is not a
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
- Intel **VTune** GPU-counter workflow, paired with RenderDoc state inspection and
  OA's cross-vendor timestamp/benchmark evidence contract.
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
