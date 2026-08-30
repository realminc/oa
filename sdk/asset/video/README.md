# Test video assets

This directory separates playable clips from minimal conformance streams:

```text
video/
  clip/          # short containerized visual and differential fixtures
  conformance/   # tiny elementary-stream and IVF parser/session fixtures
```

The clips validate end-to-end `.mp4` decode (container demux → hardware decode
→ RGBA readback) against an independent FFmpeg decode, without visual
inspection. The conformance streams isolate parameter parsing, session updates,
single-picture recording, truncation, and malformed-input behavior without
making a two-second demo clip the smallest debugging unit.

These bounded snapshots remain in OA so standalone builds, CI, and performance
evidence do not silently depend on a sibling checkout or mutable download.

`clip/manifest.tsv` is the detailed codec-profile record for the standalone
clips. Together with `sdk/asset/manifest.toml`, it pins the declared
profile, chroma, depth, picture layout or grain requirement, level/tier, color
metadata, provenance, and digest.

The Shibuya clips are 60-frame visual and differential fixtures. The Ready Set
Go clips are 300-frame low-latency performance fixtures derived from a genuine
120 fps master by selecting every even frame; none duplicates or interpolates
a 30 fps source.

| File | Standard profile | Format | Level / tier | SHA-256 |
|---|---|---|---|---|
| `clip/ready_set_go_1080p_60fps_av1_main_8bit_420.mp4` | AV1 Main, low delay, no film grain | `yuv420p` | 4.1 / Main | `de3b3d89c04997d502b9e7c211529a0893f32af72d843bb552ca03267fda280e` |
| `clip/ready_set_go_1080p_60fps_h264_high_8bit_420.mp4` | H.264 High, progressive | `yuv420p` | 4.2 / n/a | `725351e7586c03b5f371c6fb69140a7622aea1dd1ab9d5a7d7bf9212565aae22` |
| `clip/ready_set_go_1080p_60fps_h265_main_8bit_420.mp4` | H.265 Main, low delay | `yuv420p` | 4.1 / Main | `6dc3c57694fbf4e344accb4e5931111b4e9379c95070d4ca5e5f55099ddd4b75` |
| `clip/ready_set_go_1080p_60fps_vp9_profile0_8bit_420.mp4` | VP9 Profile 0, low delay | `yuv420p` | 4.0 / n/a | `1c7d5f8affe7bbf9fa017f466437219f447f9bfbf3fa3c3918a8f4d078384dfe` |
| `clip/ready_set_go_2160p_60fps_av1_main_8bit_420.mp4` | AV1 Main, low delay, no film grain | `yuv420p` | 5.1 / Main | `200432fd39c78cd25a76ddff2bde081de15575e02d9f5d8a65139355efd30a4b` |
| `clip/ready_set_go_2160p_60fps_h264_high_8bit_420.mp4` | H.264 High, progressive | `yuv420p` | 5.2 / n/a | `3b9302f7eeddb04dab046c051b30b9cb3ff3d5aaff90f8cf4cdf72b1f6472682` |
| `clip/ready_set_go_2160p_60fps_h265_main_8bit_420.mp4` | H.265 Main, low delay | `yuv420p` | 5.1 / Main | `04c1793ddf5393efcb002b8b73658e8d662063cbd027a69468dd6da16b93a97f` |
| `clip/ready_set_go_2160p_60fps_vp9_profile0_8bit_420.mp4` | VP9 Profile 0, low delay | `yuv420p` | 5.0 / n/a | `2c2b0e34ae208c8fb3dcd88bc3cb6299f465f8a7419ff32c85a3a49dadc08912` |
| `clip/shibuya_720p_30fps_h264_baseline_8bit_420.mp4` | H.264 Constrained Baseline, progressive | `yuv420p` | 3.1 / n/a | `d388fe195add3b79d2ae57115dc6d63d482257ce5b9229e32b83d9b0a451492b` |
| `clip/shibuya_720p_30fps_h264_high_8bit_420.mp4` | H.264 High, progressive | `yuv420p` | 3.1 / n/a | `4f9d04b3e67540c42d1a8b82b650ee26e5447126c0f1b78e58f38c9f250e9bcc` |
| `clip/shibuya_720p_30fps_h264_main_8bit_420.mp4` | H.264 Main, progressive | `yuv420p` | 3.1 / n/a | `9ec623563cc8f29f005e911f8e54e94cee3216fbe7824188d4f939606436a865` |
| `clip/shibuya_720p_30fps_h265_main_8bit_420.mp4` | H.265 Main | `yuv420p` | 3.1 / Main | `8bb97852b606cbce71b3c4608b8dc36424e17c63b4466dfcf19746099f1523c2` |
| `clip/shibuya_720p_30fps_av1_main_8bit_420.mp4` | AV1 Main, no film grain required | `yuv420p` | 3.1 / Main | `dd0097378581c9e2d3cb39efdd88e659eb53bb8bcd05c0237919fcf4b09c2441` |
| `clip/shibuya_720p_30fps_vp9_profile0_8bit_420.mp4` | VP9 Profile 0 | `yuv420p` | 3.1 / n/a | `79eb63aa80e3e6481dcad07d1ece79c6be4597be0584018b989a5daa2e4a2bce` |

Clip filenames follow
`<sample>_<resolution>p_<fps>fps_<codec>_<profile>_<depth>bit_<chroma>.<container>`.
The manifest owns details that would make filenames unbounded: coded width,
color metadata, level/tier, frame count, source and digest.

Capability tests use the complete stream-derived profile. A device whose
reported maximum level is below a fixture's level skips that exact fixture;
the test never retries with a weaker profile.

The checked corpus is broader than one device's qualified matrix. In
particular, the H.264 UHD60 fixture is correctly signaled as High 5.2 and is
expected to fail the exact capability gate on the Intel TGL/Mesa evidence
device, which reports H.264 maximum level 5.1.

The `conformance/` files retain the smaller `test_pattern_72p_*` names. Here
`72p` means a 128×72 progressive coded extent. These are single-picture
parser/session fixtures, so their filenames intentionally omit playback FPS.
The H.264 IDR is exact Constrained Baseline and the base HEVC IDR is exact Main.
The separate HEVC Main-Intra and Main-10-Intra fixtures signal Format Range
Extensions profile IDC 4 and are decoded under that exact session profile; they
are never relabeled as Main/Main 10. The admitted subset is 8/10-bit 4:2:0
without unparsed extension tools. The 10-bit fixture is reproducible with
FFmpeg n9.0.1 and x265 4.3 using `-profile:v main10-intra` and the same bounded
single-picture test source.

The native 10-bit conformance set adds HEVC Main 10, HEVC Range Extensions,
VP9 Profile 2, and AV1 Main streams. Each successful path is checked as P010
code values against an independent FFmpeg decode and then through OA's compute
P010-to-RGBA shader against a CPU color-conversion oracle. The samples are
synthetic 8-to-10-bit transport coverage, not HDR or native 10-bit
source-quality evidence.

The checkpoint was generated and inspected with FFmpeg n9.0.1, x265 4.3
(build 217), libvpx-vp9 and libaom-av1. For example, the admitted AV1 sample is
reproducible byte-for-byte on that toolchain with:

```bash
ffmpeg -hide_banner -loglevel error -y \
  -f lavfi -i 'testsrc2=size=128x72:rate=1' -frames:v 1 \
  -pix_fmt yuv420p10le -c:v libaom-av1 -cpu-used 8 -crf 30 \
  -still-picture 0 -enable-intrabc 0 -enable-restoration 1 -enable-cdef 1 \
  -f obu test_pattern_72p_av1_main_key_10bit_420.obu
```

The manifest digest, byte count and `ffprobe` profile/pixel-format checks are
the authority for every checked snapshot; encoder output is not assumed stable
across codec-library versions.

AV1 Main 10 loop restoration remains explicitly unqualified. The retained
`*_loop_restoration.obu` stream is rejected before GPU submission: on Intel
TGL GT2 with Mesa 26.1.7, submitting that picture stalls the video queue while
Khronos core validation reports no VUID. The simpler Main 10 fixture proves
P010 decode and color conversion without converting that hardware/metadata
gap into a universal profile claim.

## Visual profile tutorials

The generic `TutorialViewerVideo{Codec}` targets retain the original 1080p
dataset sources. These profile-explicit targets play the converted OA snapshots
above, so their executable and window names identify the exact stream:

| Target | Default fixture |
|---|---|
| `TutorialViewerVideoH264High8bit420` | `clip/shibuya_720p_30fps_h264_high_8bit_420.mp4` |
| `TutorialViewerVideoH265Main8bit420` | `clip/shibuya_720p_30fps_h265_main_8bit_420.mp4` |
| `TutorialViewerVideoAv1Main8bit420` | `clip/shibuya_720p_30fps_av1_main_8bit_420.mp4` |
| `TutorialViewerVideoVp9Profile0_8bit420` | `clip/shibuya_720p_30fps_vp9_profile0_8bit_420.mp4` |

Each accepts an optional video path and `--device-index N`. Successful visual
playback remains a manual presentation check; the independent FFmpeg
differential test is the correctness gate. The profile fixtures contain only
60 frames, so these four viewers intentionally stop on the final frame instead
of recreating the decoder every two seconds and disguising the loop boundary as
continuous playback.

After a Release build, launch one directly, for example:

```bash
bin/release/sdk/tutorials/vision/tutorialViewerVideoAv1Main8bit420
```

The TGL Iris Xe/Mesa evidence accepts all six 720p rows. H.264 Baseline, Main,
and High, H.265 Main, AV1 Main, and VP9 Profile 0 each pass the independent
FFmpeg RGBA differential at approximately 43.5–43.6 dB. VP9 uses the separately
documented bounded TGL/Mesa level-report qualification; the exact stream profile
is never weakened.

## Provenance and regeneration

The clips derive from Basile Morin's
[Shibuya Crossing video](https://commons.wikimedia.org/wiki/File:Shibuya_Crossing,_Tokyo,_Japan_(video).webm),
licensed [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
The checked manifest owns profile naming, exact metadata, and digests. The
byte-identical source fixtures and regeneration script live in the independently
versioned dataset repository under `video/profiles`; OA never discovers that
sibling checkout implicitly. When `OA_DATA_DIR` points at that dataset, tests
may use its snake-case filenames explicitly. Publishing a new fixture requires
recording its command and provenance before updating the content-addressed SDK
manifest.

To reproduce a decoder bug that only manifests at full resolution, re-cut with
`-c copy` (no scale/re-encode) from the 1080p sources instead.

The Ready Set Go derivatives come from the Ultra Video Group sequence and are
distributed under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).
They are non-commercial evidence assets, not code-license-covered SDK content.
`tools/data/deriveVideoRealtime.py` pins the upstream master at 30,724,839 bytes
and SHA-256
`d2bc5ffd0f9b967239d3082e6dd9d9f6e32ae979f2b284631f56def0c7d2a6c3`,
records every encoder argument, and verifies exact output hashes, 300/300
unique frames, 60/1 cadence, and a 40 dB independent PSNR floor.
