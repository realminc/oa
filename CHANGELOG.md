# Changelog

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
