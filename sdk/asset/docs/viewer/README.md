# Viewer documentation assets

**Status:** Current presentation assets

The `oa-viewer-{image,video,audio}-*.jpg` set records the expanded Viewer shell
on 2026-08-31: image and video canvas/fullspace states plus waveform,
spectrogram, and mel-spectrogram audio modes. These are application-state
captures for documentation, not independent correctness evidence. The JPEGs
are cropped to the exact native window bounds before compression: 2560×1440 at
`+6+10` for image/video and 1920×720 at `+12+14` for audio. They are stripped
and compressed at quality 88 with 4:2:0 sampling.

`oa-viewer-canvas-demo-720p.mp4` is the corresponding 21.1-second canvas
interaction recording. Its 2560×1440 window is cropped at `+6+10` before being
normalized from the variable-rate source to 1280×720, square pixels, 30 fps
H.264 High, yuv420p, CRF 23, with fast-start metadata and no audio. It is a
presentation derivative rather than a codec or cadence qualification artifact.

The Space Cathedral source artwork was supplied for the Realm/OA presentation
set and is copied into OA as `sdk/asset/image/coverMl.jpg`. Before
redistributing either image outside Realm/OA documentation, preserve the
project's source-artwork rights record; this file does not invent a third-party
license. The video captures use Basile Morin's Shibuya Crossing source under CC
BY-SA 4.0, matching `sdk/asset/video/README.md` and the checked clip records.

Exact sizes and hashes for the current presentation set are pinned in
`sdk/asset/manifest.toml`.

The captures used the X11 SDL backend only to obtain deterministic window-only
media under GNOME/Wayland. They do not claim an X11-only Viewer path.
