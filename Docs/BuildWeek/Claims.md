# OA Build Week claims — Evidence and publication boundaries

**Status:** ✅ ACTIVE
**Date:** 2026-07-20
**Module:** Documentation
**Sister docs:** [Submission index](README.md), [Mobile benchmark](../Benchmarks/OaMobileLab.md), [NLP benchmark](../Benchmarks/OaNlpSuite.md), [Public releases](https://github.com/realminc/oa/releases)

---

## TL;DR

This ledger defines what the submission may say and what evidence must appear on
screen or remain one link away. Broad architectural language is allowed only when
the current capability boundary follows immediately. Unmeasured Python overhead,
universal zero-copy, universal hardware support, and unmatched competitor
comparisons are excluded.

---

## Claim ledger

| Topic | State | Safe public wording | Evidence | Do not say |
|---|---|---|---|---|
| Product | Shipped preview | “OA is a GPU-first C++ and Python architecture framework for HPC, ML, Vision, Audio, media, plotting, and compact rendering on Vulkan.” | Public headers; [README architecture](../../README.md#architecture); [framework guide](Framework.md); [v0.7.5 release](https://github.com/realminc/oa/releases/tag/v0.7.5) | “OA replaces PyTorch, OpenCV, FFmpeg, and Unreal.” |
| Architecture convergence | `0.7.6-dev` source refresh | “Public main includes the verified private `v0.6.101` architecture-convergence source checkpoint.” | [Changelog](../../CHANGELOG.md); controlled results in [NLP suite](../Benchmarks/OaNlpSuite.md) | “A new `v0.7.6` release,” “all migration work is complete,” or an aggregate speedup. |
| Cross-vendor execution | Validated subset | “The Build Week training contract passes on Intel Iris Xe and Qualcomm Adreno through Vulkan.” | [Mobile Lab](../Benchmarks/OaMobileLab.md); [NLP suite](../Benchmarks/OaNlpSuite.md) | “Runs identically on every GPU.” |
| Mobile training | Validated | “Five small neural-network architectures complete local forward, backward, AdamW, evaluation, generation, and checkpoint recreation on the reference phone.” | Signed-release five-route report in [Mobile Lab](../Benchmarks/OaMobileLab.md) | “Production model training on all Android phones.” |
| Sparse MoE | Experimental, measured | “At the controlled Byte shape, sparse MoE measured 1.08× the dense Transformer wall time on the desktop reference.” | Protocol and device in [NLP suite](../Benchmarks/OaNlpSuite.md) | “Sparse MoE is faster than dense” or a device-independent ratio. |
| Autograd | Shipped preview | “OA records and executes reverse-mode gradients for the demonstrated training paths.” | Backward, gradcheck, and training gates in release/status docs | “Complete PyTorch-compatible autograd.” |
| Vision | Shipped curated surface | “OA exposes 50 graph-native Float32 NCHW Vision operations with focused C++ or Python coverage.” | [Public Vision API](../../Source/Public/Oa/Vision/FnImage.h); `Source/Python/test_vision.py` | “OpenCV replacement” or “all image formats and operations.” |
| Audio | Shipped curated surface | “OA decodes WAV, FLAC, and MP3 and exposes 14 unique public Audio function names: 13 stateless operation candidates plus one checked zero-copy matrix view.” | [Public Audio operations](../../Source/Public/Oa/Audio/FnAudio.h); `Test/Audio`; `Source/Python/test_audio.py` | “Studio-complete audio framework” or “15 stateless Audio operations.” |
| Operation inventory | Planning snapshot | “The roadmap baseline inventories 167 schema entries for reconciliation; it does not claim 167 shipped public operations.” | [Framework guide](Framework.md); `Tools/FnAutogen/Schema` | A progress percentage derived from 167, or comparison with another framework's raw API count. |
| Video and capture | Experimental by capability | “Native video decode, encode, capture, recording, and container paths exist; availability is capability- and profile-gated.” | [Public Video API](../../Source/Public/Oa/Vision/Video.h); [README limits](../../README.md#honest-preview-boundaries) | “Every codec on every vendor.” |
| OaAlm | Demonstrated pipeline | “The text-to-motion proof runs text conditioning, motion-token generation, decode, and USD export end to end.” | [OaAlm example](../../Examples/Ml/Alm); [capture](../Media/BuildWeek/OaAlmTextToMotion720p.mp4) | “Production text-to-motion quality.” |
| RL | Shipped tutorial foundation | “The repository includes a vectorized CartPole PPO tutorial with evaluation and checkpoint restoration gates.” | `Tutorial/Ml/Rl`; `Test/Ml/Rl` | “Complete reinforcement-learning platform.” |
| Crypto | Shipped minimal, non-certified | “OA contains a minimal hybrid Crypto surface with differential tests and GPU batching for public data.” | `Source/Public/Oa/Crypto`; `Test/Crypto`; [README limits](../../README.md#honest-preview-boundaries) | “Secure,” “audited,” “certified,” or suitable for custody. |
| Python | Shipped preview binding | “Python calls the same native C++/Vulkan operation library through one extension.” | `Source/Python`; public wheel smoke | “Almost zero Python overhead” until a matched Python/C++ benchmark exists. |
| Zero-copy | Conditional architecture | “Typed values can share device storage without copying where representation and lifetime permit it.” | [Public memory contract](../../Source/Public/Oa/Core/Memory.h); [matrix value](../../Source/Public/Oa/Core/Matrix.h) | “Zero-copy everywhere.” Codec, file, and host metric boundaries can copy. |
| Packaging | Shipped preview | “Judges can install the signed APK or Linux/Python packages without rebuilding OA.” | Public `v0.7.5` assets; the automated C++ packages have a checksum manifest, while APK/wheel/demo digest coverage remains an explicit release checklist item | “Fifteen-minute build on every free runner.” |
| Operating systems | Linux shipped; Android validated; Windows and macOS planned | “OA 0.7.5 ships Linux packages and a validated Android reference app. Windows through Vulkan and macOS through MoltenVK are planned, capability-gated compute targets.” | Public packages; [Mobile Lab](../Benchmarks/OaMobileLab.md); [README limits](../../README.md#honest-preview-boundaries) | “Cross-platform support,” current Windows/macOS support, or “Vulkan Video on macOS.” |
| UI / plotting | In migration | “OA has viewer and diagnostic UI foundations; plotting and editor coverage remain incomplete.” | [Public UI headers](../../Source/Public/Oa/Ui); [framework guide](Framework.md) | “Complete plotting and editor suite.” |

## Hardware statement

Use this exact order when a hardware claim appears:

```text
device -> driver -> precision -> workload -> result -> date
```

Reference examples:

- Intel Iris Xe TGL GT2, Mesa Intel 26.1.4, FP32, canonical 300-step NLP suite,
  all 15 architecture/tokenizer processes passed in the July 18 reference sweep.
- Qualcomm Adreno 610, Android 16, app-local Turnip 26.1.4, FP32, canonical
  five-route Byte suite, all signed-release routes passed in the July 18 sweep.

Do not collapse those into “cross-platform deterministic results.” Exact
determinism is asserted only within the documented seeded contract and checkpoint
round trip.

## Build Week delta statistics

The following is an internal engineering snapshot, not a product-quality metric.
It compares private commit `8e5a32b5` (July 10, version `0.7.2`) with private
checkpoint `92fa8776` (July 17, version `0.7.4`), the architecture baseline used
for the final Build Week convergence work.

| Measure | Value |
|---|---:|
| Checkpoint commits | 14 |
| Files changed | 1,059 |
| Text additions | 79,579 |
| Text deletions | 50,802 |
| Added / modified / deleted paths | 391 / 471 / 177 |
| Changed test files | 86 |
| Changed tutorial files | 69 |
| Changed tool files | 44 |
| Changed shader-source files | 144 |

These counts include generated files, moves, removals, and broad source
reorganization. They may be used in a written engineering appendix with the
comparison refs. They should not appear in the main narration, where they would
reward churn rather than correctness.

## Claims that need new evidence

| Desired claim | Required proof |
|---|---|
| “Almost zero Python tax” | Same installed build, same device and graph, Python versus C++ authoring/replay benchmark with at least seven fresh processes, median and spread. |
| “Zero-copy saves CPU/GPU cycles” | Instrumented copy/allocation/host-wait counts for a matched OA pipeline and defined alternative, with identical output and workload. |
| “CUDA-class performance” | Same operation, shape, dtype, numerical tolerance, device, warmup, synchronization, and library versions; median and spread; zero unexpected fallback. |
| “Builds in 15 minutes on a free runner” | Clean hosted-runner job from the public tag with exact image, cache state, wall time, and successful install/test. |
| “Works on AMD/NVIDIA/Windows/macOS” | Fresh device- and operating-system-specific capability and conformance runs for the exact release. Historical ancestry or compilation is insufficient. |

## Quote and tone policy

Do not use “Miles ahead of the competition” as a mock testimonial. It resembles
an invented endorsement and weakens the evidence-first voice. Do not use Arthur
C. Clarke's “advanced technology” line in the main video; OA should make the
mechanism visible, not call it magic.

The informal integrated-GPU idea becomes:

> Integrated GPUs are programmable accelerators already inside the machines
> people own.

The central thesis line is:

> The accelerator you already own should be able to train.

The final coda directs viewers to the evidence rather than making another claim:

> Explore the architecture, APIs, evidence, and roadmap at dev.realm.software.

## Release provenance

- Private engineering repository: `empyrealm/oa`; incremental checkpoints use
  internal `v0.6.x` tags while the root `VERSION` records the product version.
- Pre-event engineering baseline: `empyrealm/oa` `8e5a32b5`, July 10, root
  version `0.7.2`.
- Public sanitized release repository: `realminc/oa`; product releases use
  `v0.7.x` tags.
- First public sanitized Build Week tag: `realminc/oa` `v0.7.3`, July 14.
- Final judgeable Build Week package: `realminc/oa` prerelease `v0.7.5`, released
  July 18.
- Current private migration work after that release is not part of the public
  Build Week claim unless a new sanitized release is cut and rerun.
