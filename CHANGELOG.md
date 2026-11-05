# Changelog

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
