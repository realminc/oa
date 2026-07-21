# OA Devpost submission — Paste-ready project copy

**Status:** ✅ ACTIVE
**Date:** 2026-07-20
**Module:** Documentation
**Sister docs:** [Submission index](README.md), [Presentation](Presentation.md), [Claims](Claims.md), [Submission checklist](SubmissionChecklist.md)

---

## TL;DR

This is the paste-ready Devpost description. It presents OA as the developer tool
and OA Mobile Lab as the physical proof, distinguishes the pre-existing framework
from the Build Week delta, and points judges to a signed APK and packaged release
that do not require a source build.

---

## Title

OA — GPU-First Architecture Framework

## Tagline

A GPU-first C++ and Python architecture framework for HPC, ML, Vision, Audio,
media, plotting, and compact rendering through one capability-driven Vulkan
runtime.

## Category

Developer Tools

## Elevator pitch

OA lets developers build GPU applications without assembling a different
high-level stack for numerical compute, training, vision, audio, media, plotting,
rendering, and each hardware vendor. OA is the product; the Android application
is one physical proof. During Build Week, OA Mobile Lab trained five
neural-network architectures locally on a phone through the same C++ operation
library, autograd engine, AdamW optimizer, checkpoint format, and Vulkan kernels
used on desktop.

## Inspiration

A useful local-AI application rarely ends at model inference. It captures or
decodes data, transforms it, runs or trains a model, evaluates the result,
checkpoints state, and then displays, plays, encodes, or streams output.

Those stages are usually assembled from several libraries. Each boundary can add
another data model, allocator, device policy, copy, synchronization primitive,
and vendor backend. The application developer owns the seams as well as the
product.

OA asks whether a curated library can preserve the semantics of matrices, images,
audio, video, and model state while sharing one explicit Vulkan execution
substrate. Integrated and mobile GPUs are central to that idea: they are
programmable accelerators already inside machines people own.

## What it does

OA is a pre-1.0 GPU-first C++ framework with module-shaped Python bindings. One
`OaEngine` owns device, memory, queues, kernels, scheduling, and profiling.
Stateless operations record semantic work; stateful decoders, encoders, streams,
training, and presentation remain explicit sessions. Semantic operations lower
into executable dispatch, transfer, barrier, alias, and completion records.

The current preview includes:

- numerical Core and Runtime operations with recorded and compiled execution;
- reverse-mode autograd, modules, optimizers, checkpoints, ML training, and RL;
- a curated 50-operation Vision surface plus native video and capture paths;
- audio decode, capture, streaming, and encoding around 13 stateless operation
  candidates plus one checked zero-copy matrix view;
- minimal Crypto, focused plotting, and compact 2D/3D viewer and rendering
  foundations;
- an early OaAlm text-conditioning, motion-token, decode, and USD-export pipeline;
- C++ and Python APIs over the same native Vulkan operation library.

OA Mobile Lab is the Build Week proof. On the reference Adreno 610 phone it trains
RNN, GRU, Transformer, sparse-MoE Transformer, and Mamba-3 routes. A route passes
only after Vulkan forward, loss, autograd backward, AdamW, evaluation,
autoregressive generation, checkpoint save, fresh-model reload, and exact
post-reload generation parity.

## How we built it

- C++20 values, operations, autograd, modules, optimizer, checkpoint, and runtime.
- Vulkan compute with SPIR-V kernels generated from Slang.
- Capability-driven kernel selection rather than a public backend per vendor.
- Explicit resource effects, graph lowering, barriers, and completion evidence.
- Nanobind Python modules over one native extension.
- Android foreground execution, live metrics, cancellation, signed packaging, and
  an app-local Mesa Turnip path on the reference phone.
- A deterministic desktop/mobile protocol with the same corpus, dimensions,
  prompt, update count, learning gates, generation checks, and checkpoint round
  trip.
- Sanitized public snapshots and release automation for APK, Linux runtime/SDK,
  distro packages, Python wheel, and checksums.

## What changed during Build Week

OA existed before the challenge. Development happens in the private
`empyrealm/oa` engineering repository, whose incremental checkpoint tags remain
on the `v0.6.x` line. Sanitized product releases are published from
`realminc/oa` with `v0.7.x` tags. Root version 0.7.2 on July 10 is the pre-event
engineering baseline; it was not published as a public Git tag. Public `v0.7.3`
is the first sanitized event snapshot, and public prerelease `v0.7.5` was the
judgeable package submitted on July 18. The maintained installation path is the
July 21 `v0.7.6` prerelease.

During Build Week we:

- built OA Mobile Lab as a coherent Android product rather than a command-line
  probe;
- unified physical-phone and desktop training around the same end-to-end gate;
- added packed Transformer work, GPU-authored sparse execution, and bounded mobile
  backward routes;
- reduced the controlled desktop Byte/BPE sparse-MoE wall-time ratio to 1.08× and
  1.12× the dense Transformer while retaining oracle and learning gates;
- found and fixed a packed-QKV descriptor defect that appeared on Turnip but not
  the desktop driver;
- added deterministic completed-execution reports with operation provenance,
  selected implementation IDs, kernel hashes, lifetimes, alias groups, barriers,
  and completion timelines;
- added module-boundary checks, real APK signing, release-safe report automation,
  and fail-closed public snapshot generation.

## How Codex and GPT-5.6 were used

OA is developed by one engineer working with Codex. During Build Week, GPT-5.6 in
Codex acted as an engineering collaborator across repository audit, design review,
implementation, differential diagnosis, test construction, documentation, and
release automation.

The collaboration followed a repeated evidence loop: inspect the live call path,
state an ownership or correctness invariant, implement the smallest vertical
change, run the narrow oracle, then run the physical-device and cross-module
gates. Hardware output came from executing OA on the referenced devices; it was
not generated as a benchmark claim.

The clearest example was a release-phone failure that desktop testing did not
reproduce. The Transformer reached corrupt learning behavior on Turnip. Codex
helped trace the failure through the packed projection path to a bindless
descriptor selection that varied across shader threads without the required
uniform contract. We rewrote the branch, rebuilt and installed the signed app,
and reran the complete phone suite plus Intel regressions.

Codex also helped convert implicit architectural intent into enforceable evidence:
generated-source drift checks, dependency ratchets, deterministic graph reports,
fallback visibility, validation profiles, reproducible package paths, and the
public release boundary.

## Challenges

### Training is a larger portability contract than inference

On-device inference can freeze parameters and pre-plan a forward graph. Training
adds saved activations, gradients, mutable parameters, optimizer and random state,
checkpoint recovery, and many more synchronization edges. We therefore treated
generation and fresh checkpoint recreation as required gates, not optional demos.

### Cross-vendor shader correctness

One driver can tolerate an invalid assumption another driver exposes. The
packed-QKV defect reinforced why a “works on my GPU” forward test is insufficient.

### Small graphs are dominated by orchestration

On teaching-size models, dispatch, barriers, host synchronization, and temporary
memory can dominate arithmetic. Recorded work, compiled replay, packed kernels,
sparse route compaction, and memory reuse must all improve without changing the
numerical or checkpoint contract.

### Breadth can obscure the product

OA spans several domains, but the submission needed one judgeable result. Mobile
Lab became the physical proof; the other modules show the value of sharing its
runtime rather than competing for screen time.

## Accomplishments

- Five signed-release Android routes passed 300-step training, evaluation,
  generation, checkpoint save, fresh reload, and exact generation parity on the
  reference Adreno 610 phone.
- The equivalent controlled Transformer and sparse-MoE workloads passed on Intel
  Iris Xe through ANV.
- The public release includes a signed APK, CPython 3.12 wheel, Linux runtime and
  SDK archives, distro packages, and checksums.
- The execution report makes the completed semantic and executable work inspectable
  instead of reducing success to a timing number.
- The same repository contains verified Audio and Vision foundations, CartPole PPO,
  native media paths, and an early text-to-motion deployment proof.
- Unsupported capability paths remain explicit instead of silently selecting an
  unrelated CPU algorithm.

## What we learned

Cross-vendor correctness cannot be inferred from one desktop driver. Low
teacher-forced loss also does not prove a useful or restorable model. Physical
generation and checkpoint tests found defects that narrower tensor comparisons
would have missed.

We also learned that a serious existing codebase can be a strong Build Week project
when the baseline, eligible delta, collaboration history, and final proof are made
explicit. Codex was most valuable when it operated inside that discipline: source
and tests over memory, hardware evidence over confident prose, and a completed
vertical slice over a broad unfinished rewrite.

## Potential impact

OA targets developers building local, private, edge, media, scientific, and ML
applications that cross domain boundaries. One capability-driven runtime can
reduce integration work, keep more intermediate data device-resident, and let the
same high-level application structure reach integrated, discrete, and mobile
Vulkan devices.

Linux is the current packaged developer platform, and Android is the validated
reference deployment. Windows through Vulkan and macOS through MoltenVK are
planned compute targets, gated by the capabilities each platform actually
exposes; this preview does not claim current Windows/macOS support or Vulkan
Video on macOS.

This preview does not prove universal support or vendor-performance parity. It
does prove that complete local training and multi-domain composition are viable
through one Vulkan-first architecture on two materially different GPU/driver
stacks.

## Honest boundaries

- OA is a development preview; its C++, Python, and `.oam` compatibility contracts
  are not frozen.
- The Android gate covers one Adreno 610 phone with app-local Turnip 26.1.4, not
  every Android GPU or system driver.
- FP32 is the current fully exercised path on the reference Intel and Qualcomm
  hardware. Reduced-precision paths require fresh capable-device evidence.
- Codec support is capability-, profile-, and device-gated.
- The teaching NLP corpus proves executor correctness and learning behavior, not
  production language quality.
- The OaAlm checkpoint proves the pipeline and deployment format, not production
  motion quality.
- Crypto is not independently audited or certified.
- OA does not claim to replace full PyTorch, OpenCV, FFmpeg, or 3D-editor products.

## What's next

The next milestone is architectural convergence rather than another breadth wave:
one declarative operation source, explicit completion, separate semantic and
executable graphs, smaller stateful sessions, a persistent real-GPU CI gate, and a
stable pre-1.0 public surface. The current 167-entry catalog is a migration
inventory, not a shipped-operation claim. After reconciliation, accepted
Core+ML workloads are expected to fit within about 100 semantic operations,
Vision within about 80, while later breadth remains capped and evidence-driven.
Audio and Plot/UI have named ceilings rather than parity goals. The detailed
current-base and roadmap matrix is in [Framework.md](Framework.md).

## Test it

The fastest judge path is the signed `OaMobileLab-release.apk` attached to public
prerelease [`v0.7.6`](https://github.com/realminc/oa/releases/tag/v0.7.6).
Linux users can install the packaged runtime/SDK or CPython 3.12 wheel from the
same release without rebuilding the framework.

Repository: <https://github.com/realminc/oa>

Project: <https://realm.software/>

Reproduction guide: <https://github.com/realminc/oa/blob/v0.7.6/BUILD_WEEK.md>

Mobile evidence: <https://github.com/realminc/oa/blob/v0.7.6/Docs/Benchmarks/OaMobileLab.md>

Desktop evidence: <https://github.com/realminc/oa/blob/v0.7.6/Docs/Benchmarks/OaNlpSuite.md>
