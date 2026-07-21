# OA Build Week narrative — Product thesis and technical case

**Status:** ✅ ACTIVE
**Date:** 2026-07-20
**Module:** Documentation
**Sister docs:** [Submission index](README.md), [Presentation](Presentation.md), [Claims](Claims.md), [README architecture](../../README.md#architecture)

---

## TL;DR

GPU applications are commonly assembled from separate compute, training, media,
vision, audio, and deployment stacks. OA tests a narrower but consequential idea:
one GPU-first library can preserve domain semantics while sharing one Vulkan
execution, memory, scheduling, kernel, and profiling substrate. Build Week proves
that claim through complete local training on desktop and mobile hardware.

---

## 1. The problem

### 1.1 Useful applications cross library boundaries

A real local-AI product rarely ends at `model.forward()`.

```text
capture or decode
  -> normalize and transform
  -> infer or train
  -> evaluate and checkpoint
  -> visualize, play, encode, or stream
```

Each boundary can introduce another data model, allocator, device policy, copy,
synchronization primitive, build system, and vendor backend. The developer then
owns the seams between libraries as well as the product.

OA does not claim to replace complete products such as PyTorch, OpenCV, FFmpeg,
or a 3D authoring suite. It provides a curated set of operations behind one small
execution substrate so applications can cross these boundaries without changing
their fundamental ownership and completion model.

### 1.2 Portable inference is not portable training

Inference can freeze parameters, remove optimizer state, pre-plan memory, and
specialize a forward graph. Training adds saved activations, gradient propagation,
mutable parameters, optimizer and random-number state, numerical stability,
checkpoint recovery, and a much larger synchronization surface.

That difference explains OA's proof choice. A successful forward tensor would be
easy to overstate. The Mobile Lab gate requires forward, loss, autograd backward,
AdamW, evaluation, autoregressive generation, checkpoint save, fresh-model reload,
and exact post-reload generation parity.

## 2. The product thesis

OA's thesis is:

> Device-resident values can retain matrix, image, audio, video, and mesh semantics
> while sharing one explicit Vulkan execution substrate.

The target architecture has three public contract kinds:

| Contract | Responsibility | Examples |
|---|---|---|
| Value | Data plus semantics; no active process | `OaMatrix`, `OaImage`, `OaAudioBuffer`, `OaVideoFrame` |
| Operation | Stateless transformation or query | `OaFnMatrix::MatMul`, `OaFnImage::Resize`, `OaFnAudio::Stft` |
| Session | Stateful process with lifecycle and external effects | decoder, encoder, stream, training, and presentation sessions |

`OaEngine` is the sole local owner of device, memory, queue, kernel, scheduling,
and profiling services. Optional systems borrow it. Semantic operations remain
separate from executable Vulkan nodes, and completion is explicit through
`OaEvent`.

This is the unifying mechanism. The module count is not.

## 3. Why Vulkan

Vulkan provides a standardized, explicit compute and memory model across multiple
GPU vendors and operating environments. OA uses device capability queries to
select internal kernel implementations rather than exposing vendor policy in the
high-level API.

The direction of the standard is also relevant. Vulkan 1.4 includes a
vendor-neutral cooperative-matrix abstraction for hardware matrix units. The
[Khronos proposal](https://docs.vulkan.org/features/latest/features/proposals/VK_KHR_cooperative_matrix.html)
also states the practical difficulty: subgroup implementations require substantial
device-specific knowledge and do not always reach optimal hardware use.

Vulkan is therefore an enabling substrate, not an automatic performance result.
Kernel design, graph lowering, synchronization, memory reuse, precision policy,
and hardware-specific measurement remain OA's responsibility.

## 4. The difficult part

### 4.1 One operation contract

The mechanically derivable C++, Python, validation, autograd, registry,
documentation, and test surfaces must come from one schema. Otherwise each new
operation multiplies signature drift and backend ambiguity.

### 4.2 Two graph levels

The semantic graph records what the user requested. The executable graph records
dispatches, transfers, barriers, alias groups, and completion. A semantic
operation may decompose into several executable nodes; several operations may
fuse into one node. Treating those as the same graph makes both optimization and
debugging unreliable.

### 4.3 Cross-vendor correctness

Shader code accepted by one driver can expose invalid assumptions on another.
The Build Week packed-QKV defect is the compact example: a thread-varying bindless
descriptor choice appeared to work on desktop and corrupted learning on Turnip.
The phone gate found it because it tested generation and checkpoint recreation,
not only submission success or a low loss.

### 4.4 Performance without a second backend

OA amortizes overhead through recorded work, compiled replay, internal kernel
planning, packed kernels, sparse execution, memory reuse, and explicit host
boundaries. The strategy is to improve the one Vulkan path, not to maintain a
parallel public backend for each vendor.

This is a design strategy. Performance claims remain workload- and device-scoped.

## 5. Adjacent systems

These systems validate parts of the same problem space. They did not “try and
fail”; they chose different product boundaries.

| System | Documented center of gravity | OA distinction |
|---|---|---|
| [ExecuTorch](https://docs.pytorch.org/executorch/stable/intro-overview.html) | Export and efficient on-device inference through a lightweight runtime and hardware delegates. | OA's demonstrated mobile contract includes training, optimizer state, generation, and checkpoint recreation in the device runtime. |
| [Ollama](https://github.com/ollama/ollama) | Running and serving local open models, with chat and model-management APIs. | OA is a general compute and domain library; local language-model inference is one possible application rather than the product boundary. |
| [Kompute](https://github.com/KomputeProject/kompute) | A portable Vulkan compute framework with tensors, algorithms, sequences, Android support, and C++/Python APIs. | OA adds a curated semantic operation library, autograd, training, media types, codecs, and end-to-end conformance gates. |
| Vendor-native ML stacks | Deep optimization and mature tooling for a particular hardware ecosystem. | OA accepts a smaller optimized surface in exchange for one cross-vendor Vulkan contract. |

The comparison belongs in the long-form description or an appendix. The video
has no time for a competitor history lesson.

## 6. What Build Week proves

The public `v0.7.5` package demonstrates:

- five small neural-network architectures training, evaluating, generating,
  saving, and reloading on a physical Adreno 610 phone;
- the equivalent Transformer and sparse-MoE workload contract on Intel Iris Xe;
- one Android product using the same C++ operation, autograd, optimizer,
  checkpoint, and SPIR-V training implementation as desktop;
- deterministic completed-execution reports that expose selected implementations,
  resource lifetimes, aliasing, barriers, and completion;
- a signed APK, Linux packages, a Python wheel, checksums, and reproduction
  instructions.

The framework breadth then shows why the proof matters:

- Core and Runtime provide matrices, memory, execution, kernel planning, and
  profiling;
- ML and RL provide autograd, modules, optimizers, checkpoints, language-model
  examples, and CartPole PPO;
- Vision provides a curated 50-operation image surface plus native video and
  capture paths;
- Audio provides decode, capture, stream, and encode around 13 stateless
  operation candidates plus one checked zero-copy matrix view;
- Crypto provides a minimal, explicitly non-certified hybrid surface;
- UI and compact rendering provide viewers and diagnostic presentation rather
  than a general editor or game engine;
- OaAlm demonstrates text conditioning through motion tokens to USD output, with
  an explicitly early checkpoint.

The counts, proof levels, measured comparisons, and bounded module destinations
are detailed in [Framework.md](Framework.md). The 167-entry architecture snapshot
is a migration inventory containing internal helpers, session actions, and legacy
contracts; it is not a shipped-operation total or a breadth score.

## 7. Why it matters

Hardware choice is a product capability. Developers should be able to use an
integrated, discrete, or mobile GPU without rewriting the application around a
new high-level framework.

The practical impact is not an abstract claim of freedom. It is:

- local training and adaptation on hardware already deployed;
- fewer domain and device boundaries in media-plus-ML applications;
- one failure and profiling model across C++ and Python entry points;
- unsupported capabilities that fail explicitly instead of silently changing
  the algorithm or device;
- a smaller stack that an individual developer or small team can inspect.

## 8. Current boundaries

### Shipped

- FP32 desktop and reference-phone training gates described above.
- C++ framework and module-shaped Python bindings.
- Curated Core, ML, Vision, Audio, media, Crypto, and viewer surfaces.
- Public `v0.7.5` preview packages and signed reference APK.

### Experimental

- Sparse MoE as an optional capacity path.
- Vulkan Video coverage outside the verified profile/device subsets.
- OaAlm prompt-to-motion quality beyond the pipeline proof.
- Pre-1.0 Python and C++ APIs.

### Planned

- A reusable Android SDK/AAR instead of an application integration.
- Windows compute packaging and conformance on the capability-driven Vulkan path.
- Broader real-GPU CI and vendor capability matrices.
- Stable API/ABI policy and broader binary portability.
- A small pinned-engine presenter and unified plotting primitives.
- macOS exploration where MoltenVK exposes the required compute capabilities;
  Vulkan Video is not assumed.

### Rejected claims

- Universal hardware support.
- “Almost zero Python tax” without a Python-versus-C++ benchmark.
- Zero-copy everywhere.
- Production-quality text or motion models.
- A complete 3D engine, editor, or distributed runtime.
- Performance parity with CUDA without matched, reproducible workloads.

## 9. Message hierarchy

Use these lines in order:

1. **Result:** This phone trains a Transformer locally through Vulkan.
2. **Mechanism:** The same OA engine, autograd, optimizer, and kernels run on desktop.
3. **Product:** OA composes compute, ML, Vision, Audio, and media behind that engine.
4. **Differentiator:** One capability-driven path replaces per-domain, per-device seams.
5. **Evidence:** Training, generation, checkpoint recreation, validation, and public packages.
6. **Boundary:** A pre-1.0 development preview with a deliberately curated surface.

The framework is broad. The story stays singular.

## References

- [OpenAI Build Week challenge and judging criteria](https://openai.devpost.com/)
- [ExecuTorch overview](https://docs.pytorch.org/executorch/stable/intro-overview.html)
- [Kompute project](https://github.com/KomputeProject/kompute)
- [Ollama project](https://github.com/ollama/ollama)
- [Vulkan cooperative-matrix proposal](https://docs.vulkan.org/features/latest/features/proposals/VK_KHR_cooperative_matrix.html)
