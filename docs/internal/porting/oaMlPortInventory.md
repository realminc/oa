# OA ML to OARS port inventory

**Status:** Canonical active migration ledger

**Updated:** 2026-09-08

This document prevents the Rust port from independently rebuilding facilities
and kernels already implemented in OA C++. The donor repository is `oa`; paths
below are relative to its root. Rust source layout and public syntax may differ,
but numerical algorithms, graph semantics, lifecycle behavior, schemas, shader
bodies, and acceptance oracles are port inputs rather than design prompts.

## Port rule

Before implementing an ML or execution feature:

1. locate its OA operation or NN schema;
2. inspect the public contract, private implementation, Slang source, tests,
   and owning architecture document;
3. classify each imported component as `verbatim`, `mechanical adaptation`,
   `Rust redesign`, or `replacement`;
4. preserve shader math and dispatch geometry unless a replacement passes the
   donor oracle, differential comparison, and canonical benchmark gate;
5. generate all mechanically derivable Rust, Python, registry, autograd,
   documentation, and test surfaces from the ported schema;
6. remove the prototype route only after the donor-equivalent route is proven.

Mechanical adaptation includes Slang module/import spelling, OA attributes,
bindless push-field names, and Rust ABI metadata. It does not include changing
the algorithm, reduction order, tiling, saved-state contract, dtype behavior,
or numerical edge cases. A `replacement` records why reuse is impossible and
keeps the donor route as the correctness baseline until qualification.

Every checked-in ported shader will carry its donor path in the source banner
or owning generated metadata. A new shader with no donor states that explicitly.

## Current scale and conflict

The current repositories contain:

| Surface | OA C++ | OARS | Meaning |
| --- | ---: | ---: | --- |
| ML Slang modules | 231 | 25 | OARS has only a narrow tutorial subset. |
| ML operation-schema rows | 224 | 24 | Twenty-two rows are semantic operations; two are private lowering-only kernels. The Rust schema is not yet the OA schema port. |
| ML public headers / Rust implementation files | 74 | 19 | Counts are directional, not API equivalence. |
| ML C++ / Rust test files | 54 | 5 | Most donor oracle packs are not ported. |
| ML tutorial translation units | 31 | 2 | Char RNN and Transformer are only the first rows. |

All 24 current OARS ML schema entries overlap an existing OA operation or
lowering family. Twenty-two are semantic operations; QKV projection+bias and
gate/up SwiGLU+bias are explicitly lowering-only donor shaders. Most earlier
semantic rows were independently authored for the first Rust tutorial
instead of ported from the donor. They remain Experimental compatibility
baselines, not the canonical final implementations.

## Execution and training substrate

| OA contract and donor | OARS state | Port decision |
| --- | --- | --- |
| Canonical semantic SSA graph: `source/cpp/include/oa/core/op.h`, `source/cpp/include/oa/runtime/semanticGraph.h`, `source/cpp/lib/oa/runtime/semanticGraph.cpp` | Partial connected port: typed operation contracts and a Vulkan-free graph preserve values, views, attributes, effects, aliases, mutation, control, and autograd ranges. Matrix and current non-mutating ML schemas generate compatibility identities; capture records their values, semantic attributes, metadata views, many-to-many executable provenance, and reached `GradientTape` forward/backward ranges. Recording generations prevent a submitted or aborted graph from accepting a stale tape completion. The deterministic `oa.semantic_graph.v2` diagnostic report is connected. AdamW remains a compatibility node because versioned SSA mutation is not yet ported. Persisted descriptor serialization and exact OA schema identities remain missing. | Port OA schema rows and hashes progressively. Keep the current narrower OARS hashes explicitly distinct until behavior matches the donor contract. Port mutation versions before attaching in-place optimizer semantics. |
| Private mutable execution owner: `source/cpp/lib/oa/runtime/executionSession*.cpp` | Partial semantic recording now extends the existing eager owner. Matrix identities, producer reuse, metadata-view lineage, transactional recording, and compatibility-node distinction exist. | Continue the existing Rust owner; do not add another context or public runtime facade. Port stable-resource frames, complete domain bindings, stats, and failure recovery. |
| Immutable compiled plan: `source/cpp/include/oa/runtime/executionPlan.h`, `source/cpp/lib/oa/runtime/executionPlan*.cpp` | Partial compute replay plus a retained semantic graph, validated many-to-many lowering provenance, private DNN analysis/replacement, exact events, barriers, command caching, input rebinding/upload, timestamps, logical/physical resource counts, and ordered training compilation-stage evidence exist. Schema-generated replay roles drive pre-commit optimizer replay validation. The normalized handle-free `oa.execution_graph.v3` report is connected with generated kernel/dtype names and exact current hazard scopes. | Preserve and extend. Add persisted descriptor serialization, RNG state, richer allocator evidence, multi-queue evidence, and more qualified DNN providers. |
| OaDna compatibility planner: `source/cpp/lib/oa/runtime/dnn.{h,cpp}`, `source/cpp/lib/oa/runtime/dnn/graphLowering.{h,cpp}`, generated `dnnOpRoles.inc` | Partial private port: schema-generated OARS compatibility roles feed a deterministic semantic-graph analyzer with portable, QKV, gated-FFN, residual-norm, attention, grouped-MoE, BLAS epilogue, and vision-preprocess partition vocabulary. Physical providers mechanically port the exact `[1024,32]` FP32 inference QKV projection+bias replacement and the exact `M=1024, N=64, K=32` Linear+Linear+SwiGLU gate/up replacement. Both preserve many-to-one semantic ownership and report applied/inherited/fallback state; gate/up additionally proves its eliminated intermediates have no external consumer or owner. Training and every unqualified candidate retain source execution with an explicit reason. | Port exact OA identities and remaining providers as their complete behavior lands, attach mutation provenance, and keep provider policy private. |
| Stable-resource frames and transient alias materialization in `ExecutionSession` and `TrainingProgram` | Connected Experimental slice: the engine-owned session reuses exact-size Matrix storage by allocation ordinal; `TrainingLoop::seal_replay_inputs` separates the stable external prefix from capture-local temporaries after one eager warm-up, and automatic training capture invokes that boundary after every preparation. Captured plans retain deterministic resource IDs, byte sizes, first/last executable-node access, semantic storage bindings, external-value state, explicit observed training loss, and fail-closed `Rc` retained-owner accounting. Qualified disjoint intervals are greedily grouped and rebound transactionally to one private Vulkan buffer arena; replaced allocations bypass the general pool. Allocation or replacement-planning failure retains the original graph and records an observable fallback reason. OARS deliberately uses one buffer identity per group rather than OA C++'s distinct aliasing buffer views so the current hazard planner sees alias transitions and bindless pressure drops. | Add allocator statistics that distinguish logical savings from VMA allocation/block bytes, and run GPU-assisted plus broader odd-size/guard qualification before promotion. |
| Replay RNG: `philoxGraphAdvance.slang`, `philoxUniformGraph.slang`, `philoxNormalGraph.slang`, and `TrainingProgram::prepareReplayRng_` | Missing. | Port the Philox state transformation and rejection rules. Do not freeze host seeds in a reusable command. |
| `TrainingProgram`: `source/cpp/include/oa/ml/trainingProgram.h`, `source/cpp/lib/oa/ml/trainingProgram.cpp` | Narrow fixed-shape forward/backward/AdamW capture exists. Automatic training capture snapshots rather than consumes the source recording, validates schema-classified replay roles, pre-records the reusable untimed Vulkan command, commits only after those stages succeed, and leaves exact eager work available on rejection. Its first replay is therefore a command-cache hit. Successful programs expose the donor-ordered eleven compilation stages and stage-specific counts; explicit capture reports lazy command recording as not run. Latest-replay completion query/wait and consuming wait-and-release reset are connected. Deterministic `oa.training_compilation.v2`, `oa.semantic_graph.v2`, and `oa.execution_graph.v3` JSON reports cover compiler, semantic, lowering, memory, executable, hazard, and replay evidence. Training capture deliberately preserves source lowering and reports inference-only multi-operation candidates as fallbacks. | Treat current code as a compatibility slice. Port Philox replay state, failed-stage reports, additional OaDna providers, and persisted descriptor serialization. |
| `ItTraining`: `source/cpp/include/oa/ml/itTraining.h`, `source/cpp/lib/oa/ml/itTraining.cpp` | Partial connected `TrainingLoop` port: fixed/variable epoch lifecycle, eager and existing-program completion, checked workload/loss accounting, opt-in exact GPU timestamps, callback stop/error propagation, explicit `finish(self)`, and conservative post-warm-up stable frames exist. The automatic `step(prepare, record)` path performs one eager warm-up, captures the next fixed-shape step, skips recording during replay, rejects executable preparation work, and supports explicit safe recapture. Plan rejection submits the preserved source step eagerly, disables repeated capture attempts, and exposes cumulative fallback count/latest reason; recapture re-enables compilation without erasing that evidence. Untimed replay retains one-command caching, and the retained program exposes all three donor diagnostic reports. | Port reusable training-timestamp recording, phase distributions, and general optimizer composition. |
| Metrics: `metric.h/.cpp` and `mlFnMetric.toml` | Partial: object-safe completed-step metrics and mean/last `LossMetric` exist and are sampled once after exact completion. | Port GPU categorical accuracy and schema-owned stateless metric operations. Avoid per-callback GPU reads. |
| Callbacks and schedules: `callbacks.h/.cpp`, `callback/lrScheduler.cpp` | Partial foundation: borrowed object-safe callbacks receive immutable snapshots in registration order and return explicit continue/stop or errors. | Port early stop, checkpoint, evaluation, LR scheduling, CSV/progress reporting, and summaries on this one lifecycle. |
| Live control: `trainingSession.h/.cpp`, `oaTrainingControl.md` | Missing. | Port after `ItTraining`: bounded commands/results/snapshots, revisions, safe points, pause/resume/stop, typed parameter classes, recapture/rebuild, and observer independence. |
| Registration-addressed model and optimizer checkpoint | Narrow OARS-specific binary roundtrip exists. | Preserve the working checkpoint as Experimental, then port OA format/provenance and failure semantics instead of creating another incompatible format. |

## Current shader overlap

| OARS family | OA donor authority | Required action |
| --- | --- | --- |
| `linear`, input backward, parameter backward | routed Core GEMM/BlasLt plus `biasAdd.slang`, `linearDataBwd.slang`, and `linearWeightBiasBwd*.slang` | Port the routed operation family and its specialized backward kernels. The tutorial-specific direct Linear shaders are temporary baselines. |
| `cross_entropy`, backward | `crossEntropy.slang`, `crossEntropyBwd.slang`, `crossEntropyLossGradBwd.slang` and `mlFnLoss.toml` | Port shader math, target dtype behavior, reduction contract, schema, and tests. |
| eager and graph AdamW | `adamw.slang`, `adamwGraph.slang`, `adamwGraphAdvance.slang`, `adamwMany4.slang`, `adamwMany4Graph.slang`, and `mlFnMatrixOptim.toml` | Replace the independent Rust bodies with mechanically adapted donor shaders; port the proven four-parameter batching route instead of designing a new fusion. |
| embedding and scatter adjoint | `ops/gather.slang`, `nn/gatherBwd.slang`, generated embed operation schema and tests | Port UInt8/UInt32 behavior and retain the Rust shape-preserving public improvement as an explicit API adaptation. |
| GELU and backward | activation `gelu.slang`/`geluBwd.slang`, generated variants, and `mlFnMatrixActivation.toml` | Port exact tanh approximation and derivative; retain OA numerical order. |
| two-input SwiGLU and gate/up lowering | `swiglu.slang`, `swigluBwd.slang`, `gateUpSwigluBias.slang`, activation schema, and `graphLowering.cpp` | Mechanically ported for F32 with a finite-difference adjoint oracle and the exact donor-qualified inference replacement. Broader shapes retain the source nodes. |
| LayerNorm and backward | `layerNorm.slang`, `layerNormN32.slang`, `layerNormBwd.slang`, normalization schemas and tests | Port both generic and narrow-row routes plus the structured adjoint. Do not keep one tutorial-only schedule as universal. |
| Elman RNN and BPTT | `rnnCellLinear.slang`, `rnnCellPointwise*.slang`, `rnnScan.slang`, `rnnScanBwd.slang`, recurrent schema/tests | Port scan/cell decomposition and routing. The current combined Rust scan remains a temporary correctness baseline. |
| causal attention and backward | schema-owned scaled-dot-product attention, BMM NN/NT/TN routes, scaled-masked softmax variants, FlashAttention implementation and tests | Port the semantic operation and provider routing. Do not make the current monolithic causal tutorial kernels the only attention engine. |

OA shader paths above are rooted under
`source/cpp/lib/oa/ml/shader/compute`; Core GEMM and common storage helpers are
under `source/cpp/lib/oa/core/shader/compute`.

## Dependency order

The active port proceeds in this order:

1. introduce provenance fields and donor-drift checks in the OARS operation
   generator;
2. connect the structural semantic-graph port to engine capture and port the
   generated operation-role vocabulary;
3. extend the existing execution session with semantic bindings and stable
   resource frames;
4. port OaDna/DNN planning and lowering into the existing execution plan;
5. port replay RNG, transient alias planning, compilation reports, and plan
   reset/wait behavior;
6. replace the remaining semantic prototype shader entries family-by-family with donor-backed
   schema rows, sources, and oracle packs;
7. port `ItTraining`, metrics, callbacks, and checkpoint/evaluation lifecycle;
8. port `TrainingSession`, then the remaining NN modules and NLP/RL/tutorial
   matrix in dependency order.

Until steps 1–5 land, new model families and tutorial-specific kernels are
paused. The existing Char RNN and Transformer gates remain regression oracles
for progressive migration; they are not justification for a parallel Rust ML
framework.
