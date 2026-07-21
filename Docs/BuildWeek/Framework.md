# OA framework guide — Current base, evidence, and bounded roadmap

**Status:** Active Build Week product description
**Date:** 2026-07-20
**Module:** Documentation
**Sister docs:** [Submission index](README.md), [Claims](Claims.md), [Narrative](Narrative.md)

---

## Product statement

OA is a GPU-first C++ and Python architecture framework for HPC, machine
learning, computer vision, audio, media, compact rendering, plotting, and
supporting data services over one capability-driven Vulkan runtime.

OA Mobile Lab is not the product boundary. It is one demanding reference
application proving that the same operation, autograd, optimizer, checkpoint,
and SPIR-V path can execute complete training on desktop and mobile GPUs.

The framework is a development preview. Current, experimental, and planned
surfaces are deliberately separated below.

## Platform boundary

| Operating system | Current state | Exact boundary |
|---|---|---|
| Linux | Shipped preview | The public release provides x86-64 runtime, SDK, distro, and CPython packages. Linux is the current developer platform. |
| Android | Validated reference | The signed Mobile Lab app proves complete training on the named Qualcomm device; a reusable Android SDK/AAR is still planned. |
| Windows | Planned | Vulkan compute is an architectural target, but the release has no Windows package or published conformance run. |
| macOS | Planned | MoltenVK compute is plausible only when it exposes OA's required capabilities. Vulkan Video is not assumed, and no macOS support claim is made. |

The goal is one capability-driven operation contract across vendors and operating
systems, not the fiction that every platform exposes identical features or speed.

## Current module map

| Module | Current credible base | Working proof | Next bounded target |
|---|---|---|---|
| Core and Runtime | Matrix values, memory, kernels, semantic/executable graphs, profiling, and explicit completion. The planning inventory contains 49 Core catalog entries. | The same engine executes the desktop and phone training contracts; validation and completed-execution reports expose implementations, resources, barriers, and events. | Reconcile every retained operation under schema-v2, keep `OaEngine` as the sole execution owner, and remove compatibility recording paths. |
| ML and RL | The planning inventory contains 43 ML entries at mixed maturity. Autograd, modules, AdamW, checkpoints, recurrent models, Transformers, sparse MoE, Mamba-3, and vectorized CartPole PPO are present. | Five 300-step mobile routes pass forward, loss, backward, AdamW, evaluation, generation, save, fresh reload, and exact generation parity. CartPole demonstrates the native environment, renderer, policy, and live metrics path. | Fit accepted Core+ML workloads within approximately 100 semantic operations; admit later workload-driven breadth only below a 150-operation ceiling. |
| Vision | 50 distinct public Float32 NCHW image operations plus six adjacent detection/evaluation helpers. Video/session inventory is counted separately. | CPU-oracle image tests, image/video viewers, transforms, capture, and model-preprocessing paths exercise the current surface. | Reconcile the current 50 under semantic `OaImage` and schema-v2; cover baseline workloads within about 80 operations and cap curated breadth at 100. |
| Audio | 14 unique public C++ names: 13 stateless operation candidates plus the checked `ToMatrix` value view. WAV, FLAC, and MP3 decode sit at an explicit CPU boundary. | The named Intel Iris Xe profile records 37/37 native Audio tests, including DSP oracles, malformed input, codecs, streaming, and the GPU waveform viewer. | Grow only after semantic `OaAudioBuffer` convergence, toward at most 50 offline operations and 18 stateful real-time processor families. |
| Video and capture | Four fixed 8-bit 4:2:0 decode profiles and narrow H.264/H.265 encode implementations; availability is device, profile, format, and feature gated. | Decode, encode, capture, recording, container, conversion, and viewer paths exist on named capability rows. | Cover the seven codec-direction families represented by the pinned Vulkan registry through explicit profile/format/tool classes; never report an unsupported combination as a fallback success. |
| Plot, UI, and render | Focused image, video, audio, training, evaluation, and RL viewers exist. Plot currently supports one mutually exclusive line, heatmap, or image per axes; compact 2D/3D rendering drives CartPole, while generated USD is inspected in USDView. | The montage shows the waveform canvas, video viewer, CartPole renderer and metrics, and USDView output as working tools rather than static mockups. | One backend-neutral figure/display-list path with up to 12 ordered artist families, plus one renderer shared by interactive and truthful headless sinks. |
| OaAlm and animation | A frozen-CLIP-conditioned motion-token pipeline packages text encoder, tokenizer, prior, weights, and BPE state into one `.oam` artifact and exports canonical motion to USD. | The time-boxed Intel Iris Xe checkpoint produced prompt-dependent, recognizably attack-like motion end to end. It proves the pipeline, not production motion quality. | Improve tokenizer boundaries, contact and reconstruction quality, EOS behavior, diversity, rig solving, and retargeting while keeping authoring tools downstream of OA. |
| Supporting modules | Data, Crypto, Network, and animation utilities exist at different maturity levels. Crypto remains minimal and explicitly non-certified. | Consumer paths cover datasets and upload, public-data hashing, TCP/HTTP foundations, packaging, and diagnostic tools. | Add consumer-driven vertical slices only; these modules have no operation quota and may not introduce another engine or hidden backend. |

## Operation inventory: what the numbers mean

The architecture-roadmap snapshot at committed revision `f007eae9` contains 167
TOML `[[ops]]` entries:

| Domain | Inventory entries | Schema-v2 at the snapshot | Interpretation |
|---|---:|---:|---|
| Core | 49 | 26 | Mixed public, internal, and legacy contracts |
| ML | 43 | 7 | Mixed public, backward, fused, and legacy routes |
| Vision | 63 | 0 | 50 image entries plus 13 video/session-related entries |
| Audio | 5 | 0 | Legacy schema coverage does not match the 14-name public surface |
| UI | 3 | 0 | Blits only |
| Crypto | 4 | 0 | Deferred public-data hash operations |
| **Total** | **167** | **33** | Migration inventory, not 167 shipped operations |

The rewrite first classifies each entry as a public semantic operation, internal
lowering, value view, session action, alias, or rejected declaration. Only the
retained public operations count toward later module budgets. Kernels, overloads,
vendor routes, backward helpers, and session verbs do not inflate that number.

## Measured base and performance comparison

The most useful comparison is the same controlled 300-step training contract on
the two documented devices. Desktop values are three-process means; mobile values
are one uninterrupted signed-release run and are functional references rather
than a statistically averaged silicon benchmark.

| Workload | Intel Iris Xe wall ms/step | Adreno 610 wall ms/step | Result boundary |
|---|---:|---:|---|
| Byte Transformer, 25,760 parameters | 7.99 | 263.11 | Both pass learning, generation, and checkpoint recreation |
| Byte sparse-MoE Transformer, 28,068 parameters | 8.62 | 221.31 | Both pass the same complete contract |

On the Intel reference, the Byte sparse-MoE/dense wall-time ratio moved from
1.46× in 0.7.3 to 1.08× in 0.7.4, reducing relative overhead from 46% to 8%.
On the single mobile reference run it measured 0.84× the dense Transformer wall
time. Neither result establishes a general MoE advantage, cross-vendor parity,
or comparison with CUDA, PyTorch, or another unmatched stack.

Full protocols and results are in [OaNlpSuite.md](../Benchmarks/OaNlpSuite.md)
and [OaMobileLab.md](../Benchmarks/OaMobileLab.md).

## How much remains

Architecture convergence is the next milestone, before another breadth wave:

1. Finish one schema-v2 source for C++, Python, validation, autograd, registry,
   documentation, and tests.
2. Complete semantic Image, AudioBuffer, VideoFrame, Texture, and Mesh values.
3. Finish explicit events and engine-owned retirement for stateful sessions.
4. Delete duplicate, no-op, handwritten, and compatibility public routes.
5. Make real-GPU validation and capability manifests persistent release gates.
6. Expand only through accepted workloads and independent oracles within the
   bounded Core+ML, Vision, Audio, Video, and Plot/UI destinations above.

These are roadmap ceilings, not promises to fill a catalog. OA may stop below a
ceiling when the accepted applications are complete.

## Editorial module frames

The presentation uses established Realm artwork as module framing, never as
runtime evidence:

| Module frame | Landing asset |
|---|---|
| Foundation, architecture, and plotting roadmap | `library-office.webp` |
| ML and RL | `ml-space.webp` |
| Vision and video | `vision-interview.webp` |
| Audio | `audio-piano.webp` |

Live captures remain the evidence: terminal output, viewers, renderers, device
reports, generated APIs, and the linked benchmark records.
