# OA Build Week checklist — Submission and publication gate

**Status:** ✅ SUBMITTED; `v0.7.6` maintenance publication gate
**Date:** 2026-07-21
**Module:** Documentation
**Sister docs:** [Submission index](README.md), [Devpost](Devpost.md), [Presentation](Presentation.md), [Claims](Claims.md)

---

## TL;DR

The Build Week entry was submitted on July 18 with the public `v0.7.5` release,
repository, install paths, YouTube demo, and private `/feedback` session reference.
A July 21 `v0.7.6` maintenance release supersedes that GitHub release while preserving
the immutable PyPI `0.7.5` files as historical package evidence.
A revised 174.667-second local master and corrected caption text exist, but replacing
the submitted video is optional and requires a new YouTube URL plus a Devpost edit.

---

## Official submission requirements

Source: [OpenAI Build Week on Devpost](https://openai.devpost.com/), checked
2026-07-19.

| Requirement | State | Artifact / action |
|---|---|---|
| Working project using Codex with GPT-5.6 | Ready | Public OA `v0.7.6` prerelease and signed Mobile Lab APK. |
| Category | Ready | Developer Tools. |
| Project description | Submitted | [Devpost.md](Devpost.md) retains the source copy. |
| Public YouTube demo under three minutes | Submitted | [Public 172-second demo](https://www.youtube.com/watch?v=SEw20xx0SJY). A revised 174.667-second master remains local. |
| Demo audio covers Codex and GPT-5.6 | Submitted | Both are spoken in the submitted demo. |
| Repository URL | Ready | <https://github.com/realminc/oa> |
| README setup and sample-data guidance | Present; final audit required | Public `BUILD_WEEK.md`, README, Mobile Lab README, and release assets. |
| Developer-tool installation | Ready | Signed APK, `oapython==0.7.6`, Linux runtime/SDK packages. |
| Test path without rebuilding | Ready; verify fresh | Install APK or wheel from the release; record one clean smoke. |
| `/feedback` Codex Session ID | Submitted privately | The identifier is intentionally omitted from public documentation. |
| Licensing / repository access | Submitted | Public repository includes its current license. |

## Post-submission maintenance

1. Keep the published PyPI `0.7.5` artifact immutable; carry the signed APK and demo
   forward to `v0.7.6` before removing the superseded GitHub release and tag.
2. Publish the verified architecture-convergence checkpoint as the matching `v0.7.6`
   source, C++ package, and Python-wheel release.
3. If the revised master replaces the submitted demo, upload it as a new YouTube
   video, review the public encode and captions, then update the Devpost URL.

## Judge quick path

### Android

- Download `OaMobileLab-release.apk` from public `v0.7.6`.
- Verify its published checksum.
- Install on a supported Android device.
- Open the device page before selecting a training route.
- State clearly that the accepted reference is Adreno 610 with app-local Turnip
  26.1.4; another phone is not guaranteed to expose the same path.

### Python on Linux

```bash
python -m pip install oapython==0.7.6
python -c 'import oa; print(oa)'
```

The final judge command must execute a real Vulkan operation, not only import the
extension. Add the exact verified one-liner or small script to the public README
after testing it from a clean environment.

### Source

Use only commands confirmed against the public sanitized tag. Private checkpoint
hashes and internal documentation do not belong in judge instructions.

## Repository QA

- [ ] Public `v0.7.6` is downloadable with the carried-forward APK and demo assets.
- [ ] All release assets have checksums.
      The automated manifest initially covers only the C++ packages; keep this unchecked
      until one final manifest covers those packages plus the wheel, APK, and demo video.
- [ ] APK installs without local signing material.
- [ ] Python wheel installs into a clean CPython 3.12 environment on its declared
      glibc baseline.
- [ ] README quick path uses public paths and current API names.
- [ ] Sample data is included or downloaded from a stable, licensed location.
- [ ] No internal docs, workstation paths, credentials, or private agent rules are
      present in the public snapshot.
- [ ] Every Devpost link works in a logged-out browser.
- [ ] Limitations are visible without opening internal documentation.

## Landing-site capture blockers

The site is useful presentation material only after these live-copy conflicts are
resolved or excluded from the recording. The sibling landing worktree already has
unrelated active edits, so this documentation pass does not change it.

- [ ] Replace the Audio sample's nonexistent `.Unwrap()` call with the checked
      `OaResult` form validated in [Tutorials.md](Tutorials.md).
- [ ] Replace or qualify “Every Vendor” and “one binary, every accelerator.” Current
      release evidence covers named Intel and Qualcomm training paths, with other
      capability matrices remaining device-specific.
- [ ] Replace “deterministic results across vendors” with the exact seeded scope
      that was tested.
- [ ] Remove “Built with passion.” It conflicts with the requested engineering
      voice and adds no product information.
- [ ] Update the homepage matrix example if the ongoing explicit-execution rewrite
      removes or supersedes `OaContext::Execute()` / `Sync()` compatibility syntax.
- [ ] Record the deployed site only after a logged-out browser confirms the new
      copy is live.

## Video QA

- [x] Final local duration is 2:55 or shorter (174.667 seconds).
- [x] Public YouTube URL is present and public.
- [ ] 1080p processing is complete.
- [x] The project works on screen within five seconds.
- [x] OA, Codex, and GPT-5.6 are present in the rendered narration source.
- [ ] Build Week work is separated from the pre-existing OA baseline.
- [x] Adreno 610, Turnip 26.1.4, and FP32 are visible in the proof scene.
- [ ] Training, generation, and checkpoint reload are visible.
- [ ] Any time-compressed segment is labeled.
- [ ] Captions are corrected manually.
- [ ] Narration remains intelligible on laptop speakers and a phone.
- [x] No private terminal paths, tokens, notifications, or account data appear.
- [x] The closing frame contains project, repository, and release URLs long enough
      to read.

## Devpost form values

| Field | Value |
|---|---|
| Project title | OA — GPU-First Architecture Framework |
| Tagline | A GPU-first C++ and Python architecture framework for HPC, ML, Vision, Audio, media, plotting, and compact rendering through one capability-driven Vulkan runtime. |
| Category | Developer Tools |
| Repository | <https://github.com/realminc/oa> |
| Website | <https://realm.software/> |
| Release | <https://github.com/realminc/oa/releases/tag/v0.7.6> |
| Video | <https://www.youtube.com/watch?v=SEw20xx0SJY> |
| Codex Session ID | Submitted privately; intentionally not reproduced here. |

## Final publication gate

Do not submit if any of these are true:

- the video is three minutes or longer;
- the video does not explicitly describe both Codex and GPT-5.6 use;
- the judge path requires private files, a local build cache, or unpublished data;
- a performance number differs from the linked evidence;
- the public release cannot reproduce the on-screen result;
- the Session ID refers mainly to documentation rather than the core Build Week
  implementation session.
