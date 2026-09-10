# OA Lunar — product and port-integration roadmap

**Status:** Planned

**Updated:** 2026-09-10

**Authority:** [OA Rust architecture](../architecture/oaArchitecture.md)

**Port order:** [OA Rust port roadmap](../architecture/roadmap/portRoadmap.md)

**Subsystems:** [Render](../render/oaRender.md),
[Reinforcement Learning](../ml/oaRl.md), [Image](../image/oaImage.md),
[Vision](../vision/oaVision.md), and [Video](../video/oaVideo.md)

**Donor product evidence:** OA C++ `docs/internal/tutorial/oaRlLunarLander3d.md`,
`sdk/cpp/lib/ml/rl/lunarlander3d/`, and
`sdk/cpp/tutorials/ml/rl/lunarLander3d/`

## 1. Decision

OA Lunar is the first planned game product and the first strict external
consumer of the Rust OA library. It combines a short-session interactive Lunar
Lander game with an on-device reinforcement-learning laboratory. Its purpose is
threefold:

1. ship a focused mobile game capable of honest advertising and purchase
   monetization;
2. make human control, policy inference, training, replay, and visualization
   parts of one understandable experience;
3. prove that OA can power a complete product without private source access,
   Vulkan escape hatches, or game-specific behavior in the library.

The product is not implemented in this repository. This document records the
requirements and extraction boundary while OARS is ported. A separate consumer
repository is created only after the game-readiness gate in section 8 is met.

The provisional product and repository name is `oa-lunar`. Naming, package ID,
trademark review, store title, and final visual identity remain open product
decisions.

## 2. Product concept

One deterministic simulation supports every mode. A state produced by human
control, a policy, replay, or evaluation must have the same physics, collision,
terrain, observation, terminal, and scoring meaning.

| Mode | Player experience | OA proof |
|---|---|---|
| Pilot | Land manually in a short portrait session | real-time rendering, touch input, deterministic simulation |
| Copilot | Player chooses intent while a policy stabilizes attitude | low-latency policy inference composed with human input |
| Beat the AI | Human and policy attempt the same seeded mission | exact episode manifest, replay, comparable telemetry |
| AI Lab | Train or select a small pilot and inspect its attempts | RL lifecycle, training, checkpoints, plots, and visualization |
| Daily Mission | One bounded seed defines terrain, start state, fuel, and conditions | versioned reproducibility without a hidden online simulation |

The first control scheme is one-thumb and auto-stabilized. Exact touch mapping
must be tested on device before it becomes a contract. Accelerometer-only
control, a large virtual gamepad, and the donor's complete eight-action research
surface are not assumed to be suitable player controls.

The first session target is one landing attempt measured in seconds rather than
an open world. Progression may unlock terrain classes, lander mass/payload
profiles, policy personalities, and cosmetic presentation. Policy progression
must not turn the AI into an unavoidable pay-to-win control path.

## 3. Product loop and differentiation

```text
manual or assisted landing
          |
          v
score + fuel + landing-quality telemetry
          |
          +----> replay human and AI trajectories
          |
          +----> earn cosmetic/research progression
          |
          v
harder deterministic mission or AI Lab experiment
```

OA is visible through the experience, not through a benchmark pasted onto the
menu. Useful product-facing evidence includes:

- a tasteful `Powered by OA` capability page with the selected device and
  honestly supported on-device features;
- optional trajectory, action, value, reward-term, fuel, and contact overlays;
- human-versus-policy replays over the same episode manifest;
- local training progress and before/after policy evaluation;
- shareable daily-mission identity and result cards that contain no device or
  account secrets.

The game does not claim that a policy learns successfully merely because a
training command ran. Each published training configuration needs a fixed
quality gate and a fresh-process evaluation population.

## 4. Repository boundary

The initial program has two repositories:

```text
oars/
  reusable OA values, operations, sessions, rendering, ML/RL, and runtime

oa-lunar/
  game simulation, product state, rendering adapter, AI scenarios,
  platform shell, assets, monetization, analytics, and store delivery
```

A future editor or studio is a third downstream consumer:

```text
oa-studio/                 Planned only after reuse is proved
  model and scene inspection
  animation and material inspection
  RL rollout and training visualization
  asset diagnostics
```

OA never depends on either consumer. `oa-lunar` must build against installed,
path-pinned, Git-pinned, or released OA public surfaces. It must not include
`src/rs/runtime/vk`, reach through crate-private APIs, or receive Lunar-specific
friend access. During local development it may use a sibling path dependency;
recorded and release builds pin an exact OA commit or version.

### 4.1 OA ownership

OA owns reusable contracts:

- `Engine`, storage, graphs, queues, synchronization, events, diagnostics, and
  profiling;
- VLM vectors, matrices, quaternions, transforms, projection, and viewport
  formulas;
- Image, VideoFrame, Texture, Mesh, Material, Scene, Camera, Renderer, and
  Presenter values or sessions after their individual admission gates;
- generic environment metadata, policies, rollout/replay storage, advantage
  operations, algorithms, training lifecycle, metrics, plots, and model files;
- generic glTF/GLB import after the bounded asset profile is specified and
  conformance-gated.

### 4.2 OA Lunar ownership

The game owns product and domain policy:

- Lunar configuration, episode manifest, state, action, transition, terrain,
  collision, reward terms, scoring, and termination;
- manual-control mapping, copilot arbitration, missions, difficulty,
  progression, saves, achievements, cosmetics, and game UI;
- Lunar-specific observation encoding, policies, curricula, evaluation rows,
  shaders, and render-scene adapter;
- Android Activity, lifecycle restoration, consent, advertising, billing,
  analytics, crash reporting, signing, and store metadata;
- product assets and their complete license/provenance manifest.

The current OA C++ Lunar implementation is donor evidence. Port its algorithms,
version fields, fingerprints, fixtures, and negative tests into the consumer;
do not move the concrete task into OARS Core or `oa::ml`.

## 5. OARS program and game dependency graph

```text
Core/VLM/Runtime/Graph
   |       |       |
   |       |       +------> ML + generic RL -----------+
   |       |                                            |
   |       +--------------> Image -> Texture            |
   |                                  |                 |
   +------------------------------> Renderer            |
                                      |                 |
                           glTF -> Viewer/UI/Plot        |
                                      |                 |
                                      +----> Animation  |
                                      |                 |
                                      v                 v
                              OA Lunar desktop consumer
                                      |
                            Presenter + Android shell
                                      |
                                      v
                              monetized mobile product
```

Port completeness remains the program objective, but it is an evidence ledger,
not a single all-or-nothing event. OA Lunar begins when the bounded
game-readiness gate passes. It must not wait for unrelated distributed,
cryptography, every-codec, or broad editor work. Conversely, a scaffolded module
or compiling shader does not satisfy a dependency.

## 6. Parallel workstream protocol

The active program has independent ML/RL and Image/Vision/Video workstreams.
Render, Viewer, asset, animation, platform, and integration work must be added
without allowing those lanes to overwrite shared authority.

| Lane | Primary ownership | Required handoff | Files it does not silently own |
|---|---|---|---|
| ML/RL | `src/rs/ml`, ML shaders, ML/RL schemas and tests, ML docs | admitted operations, exact donor mapping, algorithm/oracle evidence | crate facade, global registry mechanics, root roadmap |
| Image/Vision/Video | owning modules, domain shaders/schemas/tests/docs | typed values, operations, codec/session evidence, Texture input requirements | Render implementation, Presenter, root roadmap |
| Render/Viewer/Asset/Animation | `src/rs/render`, future UI/Plot owners, render shaders/tests/docs | Texture-to-present path, scene/model contracts, model-viewer evidence | Image/Video semantics, game rules, Android product shell |
| Android/Product | future consumer repository and narrow OA platform adapters | lifecycle, input, WSI, frame pacing, package and physical-device evidence | Engine duplication, renderer duplication, domain algorithms |
| Integration | shared architecture, facade, build, generator plumbing, final gates | reconciled vertical checkpoint | no independent feature implementation hidden in glue |

The following are serialized integration hotspots even when a domain change
requires them:

- `AGENTS.md`, `Cargo.toml`, `build.rs`, and `src/rs/lib.rs`;
- canonical architecture, source structure, compatibility ledger, and root port
  roadmap;
- shared operation generator, generated operation catalog, shader registry,
  runtime dispatch, Engine, execution session, and plan;
- any repository-wide formatter, generator, rename, or source reorganization.

Before editing a hotspot, a workstream records a file lease with the integrator.
Each handoff reports:

1. base commit and exact owned files;
2. donor schema/source/test/document references;
3. public contract or private seam changed;
4. narrow oracle and edge cases run;
5. generation, formatting, lint, and applicable Vulkan-validation result;
6. missing capability, known fallback, and documents still requiring
   reconciliation.

Integration occurs one verified vertical checkpoint at a time. Generated output
lands with its schema/generator owner. Umbrella status documents change only
after source, tests, and evidence agree. If agents share one physical worktree,
they must inspect the live diff immediately before editing a hotspot and must
not use broad cleanup or formatting to absorb another lane's changes.

## 7. Viewer-first 3D integration plan

The improved Viewer is the proving ground between porting OARS and starting the
game. It remains a focused inspection application, not a DCC editor, scene
authoring format, or second runtime.

The visual target is a restrained NVIDIA-style diagnostic/PBR presentation:
clear materials, stable camera behavior, useful overlays, a dark studio/grid
environment, and trustworthy render statistics. This is a quality reference,
not a vendor dependency or parity claim.

glTF 2.0/GLB is the runtime asset interchange. The normative source is the
[Khronos glTF registry](https://registry.khronos.org/glTF/); the
[Khronos Sample Viewer](https://github.com/KhronosGroup/glTF-Sample-Viewer)
is a differential visual reference, not OA source code.

### V0 — native Texture and completion

- Native Vulkan image backing with explicit usage, layout, queue-family,
  readiness, and retirement semantics.
- Image/VideoFrame-to-Texture adapters without erasing source metadata.
- Headless clear/blit/readback oracles and exact producer/consumer completion.

### V1 — bounded raster renderer

- Port the donor indexed vertex-color raster path and shader bodies.
- Camera and Scene consume the one VLM convention.
- Color/depth targets, resize, target-ring reuse, cancellation, collection, and
  explicit close.
- Procedural triangle, cube, hierarchy, odd-size, zero/invalid, depth, alias,
  reuse, and poison fixtures.

### V2 — strict static GLB

Freeze a self-contained GLB profile before choosing a parser. The first profile
admits only the fields actually consumed by one vertical render path. At
minimum, decide and test:

- scene/node hierarchy and transforms;
- indexed triangle primitives;
- `POSITION`, `NORMAL`, `TEXCOORD_0`, and bounded vertex-color handling;
- base-color and metallic-roughness material semantics with exact color-space
  rules;
- PNG/JPEG images, samplers, and Texture ownership;
- buffer/accessor bounds, alignment, sparse-accessor policy, URI policy,
  extension policy, malformed input, and resource limits.

Unsupported required extensions fail. External network fetch, arbitrary file
escape, silent material fallback, and partial scene success are rejected for
the first profile.

### V3 — model inspection

- Open or drag one model, orbit/pan/zoom, frame selection, reset camera, and
  resize without device recreation.
- Scene tree, node visibility, material summary, bounds, wireframe, normals,
  UV, depth, and render-statistic overlays as their renderer data exists.
- Deterministic screenshots against procedural fixtures plus differential glTF
  reference renders with an explicitly defined tolerance.
- No transforms, undo stack, asset browser, project format, or save-back yet.

### V4 — animation and character inspection

- Reconcile OA animation values with glTF sampler/channel, node hierarchy,
  units, coordinate conversion, and quaternion-order semantics.
- Add clip selection, play/pause, scrub, loop, and fixed-time capture.
- Add skinning and morph targets only after bind-pose, joint-palette, bounds,
  synchronization, and malformed-asset tests pass.
- Preserve a headless fixed-time render oracle; an interactive timeline alone
  is not proof.

### V5 — reusable presentation quality

- Bounded PBR, image-based lighting, tone mapping, transparency, shadow, and
  post-processing enter in measured dependency order.
- The same semantic scene and output contract serve Viewer and OA Lunar.
- Mobile quality tiers change admitted features or resolution explicitly; they
  do not silently select a different semantic algorithm.

## 8. OA Lunar game-readiness gate

The separate consumer repository may start when all of the following are true:

1. OARS has one stable public Engine, VLM, Image/Texture, Scene, Camera,
   Renderer, and headless frame-completion route.
2. Viewer loads and displays at least one strict self-contained static GLB and
   the procedural donor Lunar scene through the same renderer.
3. Generic RL environment metadata, policy inference, training lifecycle,
   checkpoint, metric, and Plot dependencies needed by the first AI Lab row are
   implemented or explicitly reduced to a smaller accepted V0.
4. The C++ scalar Lunar donor fixtures can be consumed without importing OA C++
   runtime ownership into Rust.
5. The installed/path-consumer test proves that a separate crate can construct
   the required values and sessions without private access.
6. Release, sanitizer where applicable, core validation, synchronization
   validation, and focused GPU-assisted validation evidence exists for the
   exact renderer path on the recorded desktop device.

Animation breadth, complete PBR, Vulkan ray tracing, native video decode,
multi-device execution, and an editor are not prerequisites for starting the
game. They remain parallel OA capabilities and enter the product only when a
game feature consumes them.

## 9. Lunar extraction stages

### G0 — donor lock and scalar parity

- Inventory the donor configuration, version constants, contract fingerprint,
  episode manifest, random purposes, terrain, physics, observation, reward, and
  terminal rules.
- Port the scalar simulation into the consumer repository with no OA engine
  dependency.
- Match donor golden transitions, boundary cases, seeded episodes, and complete
  replay digests in fresh processes.

### G1 — desktop interactive game

- Build procedural terrain and lander scenes through public OA Render.
- Fixed-step simulation remains independent of render cadence.
- Add manual control, pause/resume, restart, score, fuel, landing quality,
  deterministic replay, local save, and explicit shutdown.
- Prove repeated resize, minimize/restore, device loss/failure reporting, and
  resource retirement.

### G2 — OA policy and comparison

- Load one fixed evaluated `.oam` policy.
- Run Pilot, Copilot, and Beat-the-AI through the same simulation transition.
- Record action/value/reward telemetry without changing simulation results.
- Reject incompatible model, observation, action, environment, and random
  versions before an episode begins.

### G3 — interactive AI Lab

- Start with one bounded, correctness-gated training recipe.
- Training occurs outside the render/input thread and publishes immutable
  completed snapshots.
- Pause, resume, stop, checkpoint, restore-best, evaluation, and application
  lifecycle transitions remain explicit.
- Plot loss, return, success rate, fuel, episode length, and evaluation
  confidence without reading incomplete device state.

The first release may use scalar CPU physics for interactive play and OA for
rendering plus policy inference/training. Batched Lunar Reset/Step shaders stay
in the consumer. A future OA consumer-operation-pack contract is justified only
when those shaders need to participate in OA semantic/executable graphs; it
must validate reflected ABI and declared effects and must not expose arbitrary
raw Vulkan submission.

### G4 — Android product slice

- Rust game/library code builds as an ARM64 Android native library.
- A thin GameActivity/Kotlin boundary owns lifecycle, touch delivery, Android
  UI composition, consent, ads, billing, and platform services.
- OA Presenter borrows the existing Engine and owns surface/swapchain state;
  surface loss and Activity recreation do not create a second engine owner.
- Add frame pacing, pre-rotation/orientation, app pause/resume, save restoration,
  audio interruption, memory pressure, thermal behavior, and background rules.
- Add a capability-gated dense F16 vertical slice to OARS with explicit storage,
  conversion, selected operation coverage, FP32 accumulation where numerically
  required, model persistence, and independent numeric evidence. F16 is an
  optimization candidate, not a prerequisite for the first playable APK or an
  assumed speedup.
- Default to the system Vulkan driver. Optional app-local drivers are selected
  only for an explicitly supported GPU family and execute in an isolated
  process with a system-driver fallback. Mesa Turnip is an Adreno-only route and
  is never selected for a Mali device.
- Qualify stock drivers from more than one Android GPU family before making a
  broad support claim. The existing C++ Adreno/Turnip compute evidence is donor
  evidence, not Rust graphics qualification.

### G5 — monetization and soft launch

- Free Pilot, one useful AI demonstration, and fair progression remain usable
  without payment.
- Rewarded ads may offer one continue or an optional post-result reward.
- A one-time purchase may remove forced ads; bounded cosmetic or laboratory
  content packs may be sold without altering the deterministic competitive
  contract.
- Interstitials appear only at an expected completed-session boundary and are
  frequency-capped. An unavailable ad never blocks play or loses earned state.
- Purchase entitlement is idempotent, locally cached, restorable, and reconciled
  with the store. Billing or analytics failure never corrupts a simulation or
  checkpoint.

Retention, stability, acquisition cost, lifetime value, ad completion, payer
conversion, and player feedback decide expansion. Revenue is not inferred from
a successful store upload or ad impression.

## 10. Proof matrix

| Contract | Required proof |
|---|---|
| Scalar simulation | donor differential vectors, invariants, malformed configuration, fixed-seed fresh-process replay |
| Renderer | independent image/depth oracle, lifecycle/reuse tests, core/sync/GPU-assisted validation |
| glTF | Khronos validator/assets, malformed and adversarial bounds suite, differential reference images |
| Animation | fixed-time pose oracle, hierarchy/skin/morph bounds, loop and interpolation edge cases |
| Policy inference | fixed checkpoint, exact preprocessing, action/value comparison, incompatible-contract rejection |
| RL training | independent evaluation population, before/after quality gate, checkpoint/recreate/replay |
| Viewer | input/state-machine tests plus real windowed render qualification |
| Android | Activity/surface lifecycle instrumentation, touch-to-frame behavior, stock-driver device matrix, thermal/memory runs |
| Monetization | test products/ad units, offline/error/idempotency paths, policy review, no simulation-state loss |

Each checkpoint records its exact OA commit, consumer commit, build profile,
device, driver, runtime capabilities, shader/compiler provenance, tests, and
known unsupported paths. A passing desktop renderer is not Android evidence; a
passing headless frame is not presentation evidence; an evaluated policy is not
a general RL-training claim.

## 11. Rejected directions

- Building a general game engine or editor before OA Lunar proves a second
  consumer need.
- Keeping Lunar-specific simulation, progression, Android, advertising, or
  billing policy inside OARS.
- Waiting for every OA domain to reach parity before beginning the bounded
  consumer.
- Forking the renderer for Viewer, game, headless tests, or Android.
- Treating glTF as an authoring/project format or accepting every extension in
  the first importer.
- Porting disconnected public declarations, TODO-backed methods, or C++ source
  hierarchy merely to increase a parity count.
- Giving the consumer raw queues, command buffers, descriptors, or Vulkan
  handles to bypass a missing OA contract.
- Claiming on-device learning from inference alone, or profitable monetization
  from implementation completion alone.

## 12. Promotion criteria

This document remains **Planned** until a separate consumer repository exists.
Individual sections become **Experimental** only when their complete vertical
path exists and passes the named proof. OA Lunar becomes a shipped product only
through its own release evidence; its release does not automatically promote
all underlying OARS domains to Shipped.

An editor/studio repository is admitted after Viewer and OA Lunar expose at
least two concrete consumers for an authoring or inspection capability. Until
then, improvements remain in the focused Viewer and the game owns its product
workflow.
