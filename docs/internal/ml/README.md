# OARS ML document map

**Status:** Canonical migration ledger

**Updated:** 2026-09-08

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
- [Elman RNN](oaRnn.md): whole-sequence scan, complete BPTT, current limits,
  and numerical evidence.
- [NLP tutorial suite](oaNlpSuite.md): the implemented canonical 300-step
  Char-RNN and Char-Transformer rows, exact workloads, quality evidence, and
  remaining matrix.
- [Precision and dtype](oaPrecisionDtype.md): exact current `DType` vocabulary
  and fail-closed admission rules.
- [Training lifecycle](oaItTraining.md): Experimental connected
  `TrainingLoop`, metric/callback, eager completion, and captured-program boundary.
- [ML tutorial template](mlTutorialTemplate.md): acceptance shape shared by
  current and future Rust training tutorials.
- [OA ML port inventory](../porting/oaMlPortInventory.md): donor-source map,
  current Rust coverage, reuse policy, and dependency order for the remaining
  graph, training, operation, module, shader, and tutorial port.

## Deferred OA records

The following OA documents remain useful source evidence but are not copied
into OARS yet:

- FlashAttention, MoE, Mamba-3, gpt-oss, weight transfer, and reinforcement
  learning require their primitive operation and module packs first.
- live training control requires `Training`, callbacks, safe points, and
  snapshot transport.
- ALM, Empyrealm, cognitive-architecture, and sequence-model documents are
  research or application programs rather than Rust foundation contracts.

When one of those dependencies becomes active, audit the live OA source and
tests again. Rewrite the smallest stable contract; do not import old measured
status, platform workarounds, C++ file paths, or public spellings as Rust facts.
