# Test video assets

Small, self-contained clips used by `Test/Vision/Codec/TestVideoDecodeReference`
to validate end-to-end `.mp4` decode (container demux → hardware decode →
RGBA readback) against an independent ffmpeg decode, without visual inspection.

The canonical profile corpus lives in the sibling dataset repository under
`video/profiles/` and is tracked through Git LFS. These byte-identical snapshots
remain in OA so standalone builds and CI do not silently skip codec correctness
tests when the dataset checkout is absent.

`manifest.tsv` is the byte-identical standalone snapshot of the canonical
dataset manifest and drives OA's fixture inventory test. It pins the declared
profile, chroma, depth, picture layout or grain requirement, level/tier, color
metadata, provenance, and digest. When both repositories are present, their
manifests must compare byte-for-byte.

Each clip is the first 60 frames of its corresponding 1080p30 benchmark source,
scaled to 1280×720 and re-encoded with real inter-frame prediction.

| File | Standard profile | Format | Level / tier | SHA-256 |
|---|---|---|---|---|
| `shibuya_720p_h264_high_8bit_420.mp4` | H.264 High, progressive | `yuv420p` | 3.1 / n/a | `4f9d04b3e67540c42d1a8b82b650ee26e5447126c0f1b78e58f38c9f250e9bcc` |
| `shibuya_720p_h265_main_8bit_420.mp4` | H.265 Main | `yuv420p` | 3.1 / Main | `8bb97852b606cbce71b3c4608b8dc36424e17c63b4466dfcf19746099f1523c2` |
| `shibuya_720p_av1_main_8bit_420.mp4` | AV1 Main, no film grain required | `yuv420p` | 3.1 / Main | `dd0097378581c9e2d3cb39efdd88e659eb53bb8bcd05c0237919fcf4b09c2441` |
| `shibuya_720p_vp9_profile0_8bit_420.mp4` | VP9 Profile 0 | `yuv420p` | 3.1 / n/a | `79eb63aa80e3e6481dcad07d1ece79c6be4597be0584018b989a5daa2e4a2bce` |

Capability tests use the complete stream-derived profile. A device whose
reported maximum level is below a fixture's level skips that exact fixture;
the test never retries with a weaker profile.

## Visual profile tutorials

The generic `TutorialViewerVideo{Codec}` targets retain the original 1080p
dataset sources. These profile-explicit targets play the converted OA snapshots
above, so their executable and window names identify the exact stream:

| Target | Default fixture |
|---|---|
| `TutorialViewerVideoH264High8bit420` | `shibuya_720p_h264_high_8bit_420.mp4` |
| `TutorialViewerVideoH265Main8bit420` | `shibuya_720p_h265_main_8bit_420.mp4` |
| `TutorialViewerVideoAv1Main8bit420` | `shibuya_720p_av1_main_8bit_420.mp4` |
| `TutorialViewerVideoVp9Profile0_8bit420` | `shibuya_720p_vp9_profile0_8bit_420.mp4` |

Each accepts an optional video path and `--device-index N`. Successful visual
playback remains a manual presentation check; the independent FFmpeg
differential test is the correctness gate. The profile fixtures contain only
60 frames, so these four viewers intentionally stop on the final frame instead
of recreating the decoder every two seconds and disguising the loop boundary as
continuous playback.

After a Release build, launch one directly, for example:

```bash
Bin/Release/Tutorial/Vision/TutorialViewerVideoAv1Main8bit420
```

The TGL Iris Xe/Mesa 26.1.4 evidence accepts and opens the H.264, H.265, and AV1
rows. It rejects the VP9 fixture because that stream requires level 3.1 while
the driver reports level 3.0; the VP9 tutorial surfaces `Unavailable` instead
of weakening the request.

## Provenance and regeneration

The clips derive from Basile Morin's
[Shibuya Crossing video](https://commons.wikimedia.org/wiki/File:Shibuya_Crossing,_Tokyo,_Japan_(video).webm),
licensed [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
The dataset's `video/profiles/README.md`, `manifest.tsv`, and `generate.sh` own
the profile naming, exact metadata, digests, and regeneration commands.

To reproduce a decoder bug that only manifests at full resolution, re-cut with
`-c copy` (no scale/re-encode) from the 1080p sources instead.
