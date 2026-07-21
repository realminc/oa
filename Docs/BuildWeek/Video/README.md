# OA Build Week video source

This directory is the reproducible source for the sub-three-minute Build Week
master. It deliberately remains outside OA's C++ build and reuses the established
Realm typography and editorial assets from the sibling landing repository.

## Build

Requirements: Node.js, Python 3, FFmpeg, ImageMagick, curl, network access, the
sibling `landing` and `ui` repositories at `../landing` and `../ui` relative to OA,
and the source tool captures. Set `OA_SCREENCAST_DIR` to their directory; when
unset, the renderer uses the ignored local path `var/build-week/screencasts`.

```bash
Docs/BuildWeek/Video/build.sh
```

The script creates an isolated `.build` directory, installs `edge-tts` there,
downloads the carried-forward public `v0.7.6` Mobile Lab demo, generates sentence-sized neural
narration, applies the requested independent `1.5x` tempo pass, and renders:

- `Docs/Media/BuildWeek/OaBuildWeek2026Master1080p.mp4`
- `Docs/Media/BuildWeek/OaBuildWeek2026Captions.srt`
- `Docs/Media/BuildWeek/OaBuildWeek2026Poster.webp`

The MP4 is a clean master without burned-in subtitles. The SRT is the selectable
English caption track for YouTube. The renderer refuses to export if its
calculated duration reaches 2:55. Source narration is split by scene under
`narration/` so technical pronunciation and timing can be revised without
touching the visual program.

## Evidence boundaries

- Phone footage comes from the signed Build Week asset carried forward to `v0.7.6`.
- Audio, video, RL, NLP, and text-to-motion shots come from live OA tool captures;
  presentation chrome uses the canonical Realm wordmark from the UI repository.
- The final framework coda uses the developer-site walkthrough plus the existing
  `library-office`, `ml-space`, `vision-interview`, and `audio-piano` module art.
- The release scene distinguishes shipped Linux and validated Android from planned
  Windows and macOS/MoltenVK compute targets.
- Editorial imagery is presentation framing, not benchmark or runtime output.
- The master makes no universal hardware, zero-copy, or unqualified performance
  claim.
