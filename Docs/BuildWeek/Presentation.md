# OA Build Week presentation — Demo script and production specification

**Status:** 🟡 LOCAL MASTER READY FOR REVIEW
**Date:** 2026-07-20
**Module:** Documentation
**Sister docs:** [Submission index](README.md), [Narrative](Narrative.md), [Claims](Claims.md), [Devpost](Devpost.md)

---

## TL;DR

The video opens on a phone training a Transformer, establishes OA's single-runtime
thesis, proves the full training/checkpoint contract, and only then shows framework
breadth. The master is 1920×1080, 30 fps, no longer than 2:55, with a human voice
preferred and a restrained Realm visual system derived from the existing landing
site.

## Rendered draft

The reproducible renderer and scene-split narration live in
[`Video/`](Video/README.md). The current local review artifacts are:

- `Docs/Media/BuildWeek/OaBuildWeek2026Master1080p.mp4`
- `Docs/Media/BuildWeek/OaBuildWeek2026Captions.srt`
- `Docs/Media/BuildWeek/OaBuildWeek2026Poster.webp`

Measured master properties: **174.667 seconds**, 1920×1080, 30 fps, H.264,
48 kHz AAC, −19.0 LUFS integrated, and −3.7 dBFS true peak in the local FFmpeg
analysis pass. Narration is generated as scene-sized neural clips and then
processed through a separate 1.5× tempo pass. The clean MP4 has no embedded or
burned-in subtitle stream; the adjacent SRT is the selectable YouTube caption
track. Both still require a listening pass, and the YouTube encode must be checked
after upload.

---

## 1. Editorial decision

**Question:** Should the video present Mobile Lab or the whole framework?

**Answer:** Present OA as the product and Mobile Lab as the decisive proof.

**Rationale:**

- Mobile Lab is physical, visible, and easy to judge.
- The framework supplies the novelty and potential impact.
- A complete local training gate proves more than a module montage.
- The remaining domains explain why one shared runtime is useful.

**Rejected alternatives:**

- Mobile-only story — understates the project and makes the shared runtime look
  incidental.
- Equal-time module tour — produces breadth without a memorable result.
- Architecture lecture — hides the working product behind abstractions.

## 2. Storyboard

| Time | Purpose | Picture | Evidence on screen |
|---|---|---|---|
| 0:00–0:18 | Cold open | Branded framework identification and physical-phone training proof. | `Adreno 610 · Turnip 26.1.4 · FP32 · local` |
| 0:18–0:34 | Problem | Six separate blocks—compute, ML, Vision, Audio, media, UI—resolve into one runtime thesis. | `One runtime · one capability model · explicit completion` |
| 0:34–0:55 | Product | Clean architecture diagram centered on `OaEngine` and the four vendor families. | `NVIDIA · AMD · Intel · Qualcomm · mobile` |
| 0:55–1:15 | Proof | Phone training and the final five-route contract. | `5/5 routes passed`; forward/backward/AdamW/generation/reload |
| 1:15–1:31 | Why it is hard | Semantic operation lowers into dispatches, resource visibility, barriers, and an event. | Explicit operation-to-completion contract |
| 1:31–1:50 | Toolkit montage | Beat-timed full-screen cuts: waveform viewer, video viewer, CartPole renderer and metrics, NLP training terminal, then the final USDView skeleton animation. | `Audio · media · RL · ML · generative 3D` |
| 1:50–2:01 | Public surface | Side-by-side C++ and Python operations in the OA code widget. | Same semantic operations and explicit ownership |
| 2:01–2:23 | Codex + GPT-5.6 | Audit, correction, and cross-device proof loop. | `Observed -> traced -> fixed -> verified` |
| 2:23–2:41 | Release and thesis | Public `v0.7.5` release assets, cross-OS direction, and website; return to the training thesis. | `Linux + Android now · Windows + macOS planned` |
| 2:41–2:55 | Framework roadmap | Fast developer-site cuts alternate with ML/RL, Vision, Audio, and architecture hero cards, then finish on generated APIs. | Current base, bounded targets, and `dev.realm.software` |

Keep proof shots between two and eight seconds. The toolkit montage may use
one-to-two-second inserts when a wide/close pair communicates one continuous
action. Hold tables long enough to read one number, not every row.

## 3. Final voiceover script

Target delivery: 132–138 words per minute. Read mechanisms calmly; pause after the
first and final lines.

> This phone is training a Transformer locally. No cloud service, and no mobile-only
> model runtime. It is using the same OA operation library, autograd engine, AdamW
> optimizer, checkpoint format, and Vulkan kernels that run on our Intel desktop.
> The capability-driven runtime spans NVIDIA, AMD, Intel, and Qualcomm paths across
> discrete, integrated, and mobile GPUs.
>
> GPU applications are usually assembled from separate stacks for numerical compute,
> training, vision, audio, codecs, and presentation. Every boundary can add another
> data model, allocator, copy, synchronization rule, and hardware backend.
>
> OA tests a different architecture: one GPU-first C++ and Python framework over
> Vulkan. Matrices, images, audio buffers, and video frames retain their semantics,
> but share one engine for device memory, kernel selection, graph execution, and
> profiling. Capabilities are selected at runtime. Unsupported paths fail explicitly.
>
> For Build Week, we made that claim physical. OA Mobile Lab trains five small neural
> network architectures on an Adreno phone: RNN, GRU, Transformer, sparse-MoE
> Transformer, and Mamba-3. A route passes only after forward, loss, backward, AdamW,
> evaluation, autoregressive generation, checkpoint save, fresh-model reload, and
> exact generation parity. The same controlled Transformer and MoE contracts pass on
> Intel Iris Xe.
>
> Under the high-level API, semantic operations lower into concrete dispatches,
> transfers, barriers, aliases, and explicit completion. Recorded work and compiled
> replay amortize host overhead without introducing a second vendor backend.
>
> The same substrate also drives a curated Vision library, audio decode and feature
> extraction, native media paths, CartPole reinforcement learning, and an early
> text-to-motion pipeline that exports USD.
>
> We built this week with GPT-5.6 in Codex as an engineering collaborator. Codex
> audited ownership and synchronization, implemented vertical changes, constructed
> differential tests, and repeatedly ran the real hardware gates. When the phone
> exposed corrupt Transformer learning that desktop did not, the investigation traced
> it to a thread-varying bindless descriptor choice in a packed QKV shader. We fixed
> the contract, rebuilt the signed app, and reran both Qualcomm and Intel paths.
>
> OA zero point seven point five is a development preview, available as a signed APK,
> Python wheel, Linux packages, and source. It targets NVIDIA, AMD, Intel, and Qualcomm
> through Vulkan, with verified scope declared per device. Linux and Android ship now;
> Windows and macOS compute are planned. The accelerator you already own should be able
> to train.
>
> This was a walkthrough, not the finish line. The developer pages publish the
> working base, benchmarks, gaps, and bounded roadmap: fifty Vision operations,
> thirteen Audio operation candidates, five complete mobile training routes,
> CartPole PPO, and early text-to-USD. Explore the architecture, APIs, evidence,
> and roadmap at dev dot realm dot software.

Do not speak punctuation-heavy identifiers literally. Record `OA 0.7.5` as “OA zero
point seven point five,” `QKV` as “Q K V,” and `USD` as “U S D.”

## 4. Shot specification

### Live proof shots

1. Clean-install the public signed APK on the reference phone.
2. Show the device/driver page before training.
3. Start Transformer training from a known clean state.
4. Record at least one continuous sequence containing live updates and completion.
5. Show generated output and checkpoint parity.
6. Record the desktop command and final gate from the public tag, not the private
   migration worktree.
7. Capture the public release assets page and checksum file.

The finished edit may shorten waits, but a visible `TIME COMPRESSED` label must
appear whenever training footage is accelerated.

### Existing OA media

| Asset | Use |
|---|---|
| `Docs/Media/BuildWeek/OaMobileLab1.webp` | Mobile Lab introduction or fallback still. |
| `Docs/Media/BuildWeek/OaMobileLab2.webp` | Device and Vulkan capability proof. |
| `Docs/Media/BuildWeek/OaMobileLab3.webp` | Training proof. |
| `Docs/Media/BuildWeek/OaAlmTextToMotion720p.mp4` | Two-to-four-second motion-pipeline shot with `EARLY CHECKPOINT` label. |
| `$OA_SCREENCAST_DIR/LandingDevDocsPageWalkthrough.mp4` | Final module, evidence, and generated-API montage; use the developer-site section beginning around 26 seconds. |
| `realm.software` Audio, ML, Vision, and HPC pages | Brief product and code shots after the landing-site capture blockers in [SubmissionChecklist.md](SubmissionChecklist.md) are closed. Do not scroll continuously. |

The sibling landing repository contains the source editorial media under
`public/images/realm/`: `core-engine.webp`, `library-office.webp`, `ml-space.webp`,
`vision-interview.webp`, `audio-piano.webp`, and the four Audio pipeline images.
The roadmap coda uses the middle four as module framing. Use them as editorial
art, never as product-output evidence.

### Codex evidence shot

Show one compact sequence:

```text
phone-only failure
  -> hypothesis and shader inspection
  -> packed-QKV descriptor correction
  -> focused oracle
  -> signed phone rerun
  -> Intel regression rerun
```

Redact account details, local absolute paths, tokens, and unrelated conversation.
The point is the closed engineering loop, not the length of the chat.

## 5. Visual system

### Canvas and grid

- 1920×1080, 16:9, 30 fps.
- 120 px safe margin; 12-column grid; 24 px gutters.
- One focal point per shot.
- Use split screen only for a meaningful comparison: phone/desktop or
  semantic/executable graph.

### Typography

| Role | Typeface | Size at 1080p | Weight |
|---|---|---:|---:|
| Hero | IBM Plex Sans | 76–88 px | 500 |
| Section statement | IBM Plex Sans | 48–60 px | 500 |
| Body / annotation | IBM Plex Sans | 28–34 px | 400 |
| Metrics / code | Intel One Mono | 25–30 px | 400–500 |
| Small label | IBM Plex Sans uppercase | 22–24 px | 500 |

No more than two lines of prose on a frame. Captions remain at least 42 px from
the bottom safe edge.

### Color

- Black `#0a0a0a`
- White `#ffffff`
- Light surface `#fafafa`
- Secondary text `#525252` on light, `#a3a3a3` on dark
- Borders at 8–12% opacity
- Functional blue only for the measured fast path or current active node
- Red only for the observed failure; green only for a passed gate

Do not use decorative gradients, glass panels, neon Vulkan effects, or animated
particle fields. Let real product footage and clean diagrams carry the presentation.

### Motion

- 180–300 ms for labels and state changes.
- 500–700 ms for scene entrances.
- One-axis movement; no elastic easing.
- Use graph lines to reveal execution order.
- Use hard cuts for proof footage and short dissolves between editorial sections.

This follows IBM's guidance that data graphics should be understandable,
essential, consistent, and contextual, and that technical diagrams prioritize
clarity and legibility:
[data visualization](https://www.ibm.com/design/language/data-visualization/overview/),
[technical diagrams](https://www.ibm.com/design/language/infographics/technical-diagrams/usage/),
and [animation](https://www.ibm.com/design/language/animation/tips-and-techniques/).

## 6. Production tools

### Recommended stack

| Stage | Tool | Reason |
|---|---|---|
| Programmatic titles and diagrams | [Remotion](https://www.remotion.dev/docs/) in a dedicated presentation project | React composition can reuse the landing site's typography, tokens, and editorial assets while keeping timing deterministic. Do not add it to OA's C++ build. |
| Screen and device capture | [OBS Studio](https://obsproject.com/) | Captures window, browser, terminal, camera, and audio sources into controlled scenes on Linux. |
| Edit, subtitles, and final conform | [Kdenlive](https://docs.kdenlive.org/en/) | Linux-native timeline, title, audio, and subtitle workflow; subtitle import/export supports a separate transcript pass. |
| Voice cleanup | [Audacity](https://www.audacityteam.org/) | Record and clean a human narration in uncompressed WAV. |
| TTS fallback | [ElevenLabs Multilingual v2](https://elevenlabs.io/docs/overview/capabilities/text-to-speech) | Its documentation positions this model as the stable long-form option. Generate sentence-sized clips and review every technical pronunciation. |

A human voice is preferred. It makes the technical judgment and personal Build Week
story credible. If TTS is required, use a licensed stock voice or a consented clone;
never imitate a public person.

### Audio contract

- Record mono 48 kHz, 24-bit WAV.
- Record in sentence-sized takes; keep room tone for repair.
- Apply high-pass filtering only if needed, light noise reduction, gentle
  compression, and a limiter below −1 dBTP.
- Target consistent speech around −16 LUFS integrated; measure after the final mix.
- Music is optional, instrumental, and at least 16 dB below narration.
- Export a corrected English SRT and upload it as YouTube's selectable caption
  track; do not burn captions into the clean master.

### Export contract

- H.264, 1920×1080, 30 fps, progressive.
- AAC, 48 kHz, 320 kbps.
- Duration no longer than 2:55 to retain upload/player margin under the strict
  three-minute rule.
- Public or unlisted YouTube visibility only if the challenge accepts unlisted;
  the official requirement says public, so **public is the safe choice**.
- Watch the uploaded YouTube encode once at 1080p with captions enabled before
  submitting.

## 7. Edit QA

- The project is visible working in the first five seconds.
- “OA,” “Codex,” and “GPT-5.6” are all spoken.
- The Build Week delta is distinguished from the pre-existing framework.
- The phone device and driver are readable at least once.
- No performance number appears without a device/workload qualifier.
- No accelerated footage lacks a time-compression label.
- No generated editorial image is presented as framework output.
- All small text is legible on a 13-inch 1080p display.
- Captions match identifiers and do not rewrite `MoE`, `Mamba-3`, `QKV`, or `Vulkan`.
- The final uploaded runtime remains below three minutes.
