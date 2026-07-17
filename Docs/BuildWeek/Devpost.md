# Devpost submission copy — OA Mobile Lab

## Title

OA Mobile Lab — Local AI Training Across Vulkan GPUs

## Tagline

Train, generate, and checkpoint real neural networks on a phone through the same
cross-vendor Vulkan engine used on desktop.

## Category

Developer Tools

## What it does

OA Mobile Lab is a physical Android front end for OA, a GPU-first C++ framework built on
Vulkan. It trains five real neural-network architectures locally on an Adreno phone:
RNN, GRU, Transformer, sparse-MoE Transformer, and Mamba-3. Each route runs forward,
loss, autograd backward, AdamW, evaluation, autoregressive generation, `.oam` checkpoint
save, fresh-model reload, and exact generation-parity validation.

The phone implementation is not a Java demonstration model or a remote inference client.
It embeds the same OA operation library, optimizer, model modules, checkpoint format, and
SPIR-V training kernels used by the desktop tutorials. A reproducible runner applies the
same seeded corpus, dimensions, prompt, update count, and quality gates on both systems.

## Why it matters

GPU software usually fragments at the device boundary: one stack for desktop accelerators,
another for mobile inference, and often no practical path for local mobile training. OA
tests a different contract—one explicit execution substrate whose capabilities are
selected and validated at runtime rather than encoded as a vendor-specific product fork.

The immediate audience is developers building private, local, or edge ML tools. The
larger OA framework also applies the substrate to Vision, Audio, native media, and
high-performance numerical operations.

## How it was built

- C++20 model, autograd, optimizer, checkpoint, and runtime code.
- Vulkan 1.4 compute with embedded SPIR-V kernels generated from Slang.
- App-local Mesa Turnip 26.1.4 through `libadrenotools` on the reference phone.
- Android foreground service, partial wake lock, live metrics, cancellation, and
  release-safe bounded reports.
- One deterministic desktop/mobile test protocol covering training, generation, and
  checkpoint recreation rather than only comparing one forward tensor.
- Sanitized public snapshots generated from the verified private engineering tree with
  fail-closed checks for credentials, workstation paths, and internal documents.

## How Codex and GPT-5.6 were used

OA existed before Build Week; public tag `v0.7.2` from July 10 is the baseline. During
Build Week, Codex with GPT-5.6 acted as an engineering collaborator across implementation,
architecture review, differential diagnosis, test construction, documentation, and
release automation.

The collaboration produced OA Mobile Lab, unified desktop/mobile gates, optimized sparse
execution, packed training kernels, deterministic execution-graph reports, enforceable
module dependencies, real APK signing, and the public release package. Codex repeatedly
ran the code on the attached Intel and Qualcomm devices and treated hardware output as
evidence rather than generating benchmark claims.

The strongest example was a release-phone failure that desktop drivers did not reproduce.
Codex traced corrupt Transformer learning to a packed Q/K/V shader whose bindless
descriptor selection varied across threads without the correct descriptor contract. The
kernel was rewritten with explicit uniform branches, rebuilt, installed, and then passed
the complete signed-release suite on Turnip and the desktop regressions on ANV.

## Challenges

- Making small training graphs efficient when dispatch and synchronization dominate.
- Preserving one checkpoint/module API while packing Transformer and sparse-MoE work.
- Handling materially different mobile shader-compiler limits without substituting a CPU
  training implementation.
- Proving correctness across generation and fresh checkpoint recreation, not just loss.
- Turning a large pre-existing framework into a coherent, judgeable one-week delta.

## Accomplishments

- Five signed-release Android routes pass 300-step training, evaluation, generation, and
  checkpoint parity on a physical Adreno 610 phone.
- The same Transformer and sparse-MoE workloads pass on Intel Iris Xe through ANV.
- Sparse MoE reached 0.84x the dense Transformer's mobile wall time in the controlled
  phone run, demonstrating that GPU-authored sparse execution is viable on this device.
- Execution reports now identify the completed semantic graph, selected implementations,
  kernel contents, lifetimes, aliasing, barriers, and completion timeline deterministically.
- The public release contains screenshots, a working text-to-motion proof, installable
  packages, exact reproduction commands, and explicit limitations.

## What was learned

Cross-vendor correctness cannot be inferred from one desktop driver, and a low training
loss is not a sufficient product gate. Physical-device generation and checkpoint tests
found defects that narrower tensor comparisons would have missed. The project also
confirmed that a serious existing codebase can be a good Build Week target when the
eligible delta, baseline, and collaboration evidence are stated precisely.

## Next

The next milestone is architectural convergence: one declarative operation source,
explicit completion, separate semantic and execution graphs, and smaller stateful session
interfaces. Mobile coverage will expand by device and driver only when each combination
passes the same end-to-end gate.

## Test it

Use the signed APK attached to the GitHub `v0.7.5` prerelease for the reference Android
experience. Desktop build and Android source-build commands are in
[BUILD_WEEK.md](../../BUILD_WEEK.md), with measured results in
[OaMobileLab.md](../Benchmarks/OaMobileLab.md).

Repository: <https://github.com/realminc/oa>

## Demo narration outline (under three minutes)

**0:00–0:20 — Problem.** GPU developer stacks fragment across desktop and mobile. OA
Mobile Lab asks whether one real training implementation can cross that boundary.

**0:20–0:55 — Product.** Show model/tokenizer selection and the physical Adreno/Turnip
device. Start Transformer training and point out that the metrics come from native Vulkan
forward, autograd backward, and AdamW—not a remote service.

**0:55–1:25 — Gate.** Show completion, generated continuation, checkpoint path, and
round-trip pass. Explain the identical 300-step desktop/mobile contract and five supported
architectures.

**1:25–1:55 — Build Week/Codex.** Show the deterministic execution report and a compact
diff of the packed projection fix. Explain how Codex with GPT-5.6 diagnosed the Turnip-only
descriptor failure, implemented the portable correction, and reran both devices.

**1:55–2:25 — Breadth without losing the story.** Briefly show OaAlm prompt-to-USD motion
generation and the public repository's Vision/Audio surfaces. State that Mobile Lab is the
judgeable product while OA is its shared substrate.

**2:25–2:45 — Result.** Show the signed five-model results table and close with the exact
claim: one framework, two GPU vendors, complete local training/generation/checkpoint gates.
