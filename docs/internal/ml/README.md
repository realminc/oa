# OARS ML document map

**Status:** Canonical migration ledger

**Updated:** 2026-09-11

The C++ OA repository contains both current subsystem contracts and large
model/research records. OARS migrates a document only when its dependency is
active, and rewrites language, ownership, execution, and evidence claims for
Rust. A copied filename is never evidence that its C++ implementation shipped
in OARS.

## Active Rust contracts

- [ML foundation](oaMl.md): implemented vertical slices and current limits.
- [Autograd](oaAutograd.md): tape ownership, semantic identity, saved versions,
  traversal, and gradient evidence.
- [Neural-network modules](oaNnModule.md): current `Parameter`, `Linear`,
  `Embedding`, `LayerNorm`, and `Rnn` behavior plus recursive module ownership
  and traversal.
- [Native model files](oaModelFile.md): `.oam` v3 layout, integrity,
  optimizer checkpoint mapping, manager/callback policy, legacy reads, and
  cross-language evidence.
- [Elman RNN](oaRnn.md): whole-sequence scan, complete BPTT, current limits,
  and numerical evidence.
- [GRU](oaGru.md): hoisted input projection, whole-sequence recurrent scan,
  BPTT composition, optional bias ownership, and qualification status.
- [Mamba-3](oaMamba3.md): complete FP32 grouped SISO and shared-state MIMO
  forward/backward/step, parameter module, Char training gate, provider limits,
  and evidence.
- [Byte and BPE NLP](oaByteNlp.md): deterministic `oa_bpe_v1` tokenization,
  raw-byte upload/decoding, packed-U8 embedding and adjoint, and the complete
  canonical Byte/BPE acceptance matrix.
- [Mixture of Experts](oaMoe.md): sparse top-k routing, native stacked expert
  parameters, grouped projections, an opt-in dense differential oracle, full
  reverse mode, current evidence, and remaining integration work.
- [Vector quantization](oaVq.md): deterministic nearest-code assignment,
  straight-through gradients, persistent EMA codebooks, dead-code revival, and
  residual quantization.
- [Animation Language Model](oaAlm.md): SDK-owned temporal Conv1d VQ-VAE,
  dense/hybrid-MoE motion-token prior, conditioning, generation, composition,
  persistence ownership, hardware evidence, and remaining product gaps.
- [Batched matrix multiplication](oaBmm.md): NN/NT/TN shape contracts,
  private generic/tiled routing, reverse composition, and Vulkan evidence.
- [Scaled dot-product attention](oaAttention.md): head transforms, optional
  masks, standard and causal Flash lowering, reverse mode, and provider boundary.
- [NLP tutorial suite](oaNlpSuite.md): complete canonical 5×3 Char/Byte/BPE
  matrix, exact 300-step workloads, quality evidence, and remaining
  qualification gates.
- [Reinforcement-learning foundation](oaRl.md): checked environment spaces and
  transition values, plus the execution and algorithm admission order.
- [Precision and dtype](oaPrecisionDtype.md): exact current `DType` vocabulary
  and fail-closed admission rules.
- [Training lifecycle](oaItTraining.md): Experimental connected
	`ItTraining`, metrics, progress/summary/early-stop/scheduler/CSV/validation/
	checkpoint/phase callbacks, the donor learning-rate schedule family, eager completion,
	captured-program boundary, and bounded `TrainingSession` live control.
- [ML tutorial template](mlTutorialTemplate.md): acceptance shape shared by
  current and future Rust training tutorials.
- [OA ML port inventory](../porting/oaMlPortInventory.md): donor-source map,
  current Rust coverage, reuse policy, and dependency order for the remaining
  graph, training, operation, module, shader, and tutorial port.

## Deferred OA records

The following OA documents remain useful source evidence but are not copied
into OARS yet:

- Remaining MoE model integration work, gpt-oss, and weight transfer require
  their remaining operation and module packs first. Mamba-3's remaining work
  is dtype, mobile-provider, and performance/validation qualification rather
  than missing FP32 topology.
- Remaining Empyrealm, cognitive-architecture, and sequence-model documents are
  research or application programs rather than Rust foundation contracts.

When one of those dependencies becomes active, audit the live OA source and
tests again. Rewrite the smallest stable contract; do not import old measured
status, platform workarounds, C++ file paths, or public spellings as Rust facts.
