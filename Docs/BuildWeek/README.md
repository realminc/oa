# OA Build Week submission — Narrative and production index

**Status:** ✅ SUBMITTED; public source refresh in progress
**Date:** 2026-07-21
**Module:** Documentation
**Sister docs:** [Devpost](Devpost.md), [Presentation](Presentation.md), [Claims](Claims.md), [Tutorials](Tutorials.md), [Submission checklist](SubmissionChecklist.md)

---

## TL;DR

OA is the project. OA Mobile Lab is the physical proof that makes the larger
framework judgeable: the same C++ operation library, autograd engine, optimizer,
checkpoint format, and Vulkan kernels train models on Intel and Qualcomm GPUs.
The capability-driven Vulkan runtime has NVIDIA, AMD, Intel, and Qualcomm paths
across discrete, integrated, and mobile devices. The submission leads with the
fresh Intel/Qualcomm training proof, then shows how that runtime composes compute,
ML, Vision, Audio, media, and compact presentation surfaces. Linux packages and
the Android proof ship now; Windows through Vulkan and macOS through MoltenVK
remain planned, capability-gated compute targets.

The Build Week entry, public repository, `v0.7.5` packages, private `/feedback`
session reference, and [public demo](https://www.youtube.com/watch?v=SEw20xx0SJY)
were submitted on July 18. The July 21 repository update is a source/documentation
refresh; it does not rewrite the submitted release tag or PyPI package.

---

## Submission spine

| Field | Decision |
|---|---|
| Title | **OA — GPU-First Architecture Framework** |
| Category | **Developer Tools** |
| One-line pitch | A GPU-first C++ and Python architecture framework for HPC, ML, Vision, Audio, media, plotting, and compact rendering through one capability-driven Vulkan runtime. |
| Demonstrated result | A physical Android phone trains five neural-network architectures locally; the same end-to-end contract passes on Intel desktop. |
| Build Week contribution | Mobile Lab, cross-device gates, packed training and sparse execution, deterministic execution reports, release automation, and public packaging. |
| Audience | Developers building local, edge, media, scientific, or ML software who do not want one framework and backend per vendor or domain. |
| Thesis line | **The accelerator you already own should be able to train.** |
| Final CTA | Explore the architecture, APIs, evidence, and roadmap at `dev.realm.software`. |

The narrative is deliberately asymmetric:

```text
OA framework                         the product
  ├─ Core / Runtime                  the common substrate
  ├─ ML / RL / Vision / Audio        the useful library
  ├─ Media / UI / compact rendering  composition surfaces
  └─ OA Mobile Lab                   the Build Week proof
```

The video does not give every module equal time. It proves one hard claim on real
hardware, uses short examples to establish breadth, and closes with a compact
module-and-roadmap coda.

## Documentation set

| Document | Purpose |
|---|---|
| [Devpost.md](Devpost.md) | Paste-ready submission copy centered on OA, with Mobile Lab as the proof. |
| [Presentation.md](Presentation.md) | Sub-three-minute script, storyboard, shot list, visual system, audio plan, and production workflow. |
| [Video/README.md](Video/README.md) | Reproducible branded renderer, 1.5× narration pipeline, and local master outputs. |
| [Narrative.md](Narrative.md) | Long-form problem, architecture, adjacent-system comparison, impact, and roadmap language. |
| [Framework.md](Framework.md) | Module-by-module current base, operation inventory, measured results, performance boundaries, and bounded roadmap. |
| [Claims.md](Claims.md) | Claim-to-evidence ledger, exact safe phrasing, prohibited overclaims, and Build Week delta statistics. |
| [Tutorials.md](Tutorials.md) | OpenCV-style showcase plan and verified C++/Python starter examples. |
| [SubmissionChecklist.md](SubmissionChecklist.md) | Official requirements, render QA, repository QA, and unresolved submission fields. |

## Official requirements snapshot

The OpenAI Build Week deadline is **July 21, 2026 at 5:00 PM PDT**. The official
challenge requires:

- a working project built with Codex using GPT-5.6;
- one category and a project description;
- a public YouTube demo shorter than three minutes;
- audio explaining how both Codex and GPT-5.6 were used;
- a repository URL, setup instructions, sample data where required, and clear
  instructions for running the project;
- a `/feedback` Codex Session ID for the session in which most core functionality
  was built;
- for developer tools, installation instructions, supported platforms, and a way
  to test without rebuilding from source.

Judging covers technological implementation, design, potential impact, and idea
quality. The exact wording remains on the
[official Devpost challenge page](https://openai.devpost.com/).

## Provenance boundary

OA uses two repositories with different tag roles:

| Repository | Role | Versioning visible here |
|---|---|---|
| Private `empyrealm/oa` engineering repository | Full development history, internal evidence, and incremental architecture checkpoints | Internal Git tags remain on the `v0.6.x` line; the root `VERSION` file records the public product version. |
| Public `realminc/oa` release repository | Sanitized judge- and user-facing snapshots, packages, and release notes | Product releases use `v0.7.x` tags. |

OA predates Build Week. The baseline used for the private engineering comparison
is `empyrealm/oa` commit `8e5a32b5` from July 10, whose root `VERSION` is `0.7.2`.
It was not published as a public Git tag. Public sanitized releases begin with
`realminc/oa` `v0.7.3`; the judgeable Build Week package is public prerelease
[`v0.7.5`](https://github.com/realminc/oa/releases/tag/v0.7.5).

This distinction matters. Public copy must say **pre-event version 0.7.2 baseline**,
not **public tag v0.7.2**.

## Canonical public links

- Project: <https://realm.software/>
- Documentation: <https://dev.realm.software/>
- Repository: <https://github.com/realminc/oa>
- Release: <https://github.com/realminc/oa/releases/tag/v0.7.5>
- Video: <https://www.youtube.com/watch?v=SEw20xx0SJY>
- Challenge: <https://openai.devpost.com/>

## Editorial rule

The strongest version of this submission is not the loudest one. Every number in
the video must name its workload or point to the evidence record. Every capability
must be labeled Shipped, Experimental, Planned, or Research. Competitors are
described by their documented scope, never as failures.
