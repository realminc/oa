# OA v0.12.0 — Experimental Rust and Python preview

## Summary

This milestone completes the local camera/screen recording applications,
repairs video presentation and reverse scrubbing, expands Python video-encode
parity, and brings the NLP tutorials into a consistent documented workflow.
The library remains Experimental; hardware-specific qualification is separate
from source connection and software Vulkan correctness.

## Added and improved

- `oa record` captures a selected camera. `oa capture` uses the desktop's
  monitor/window picker, with window-only and hidden-cursor options. Both share
  the image/video GPU 2D canvas, pan/zoom camera, fit, actual-size and channel
  controls, with a record/stop button, elapsed time and Ctrl+S save-copy flow.
- Linux screen capture ports the donor portal/PipeWire mapped-memory route.
  Frames own their copied pixels, static screens preserve recording time, and
  a source-size change finishes the existing movie before recording another.
- Native video presentation selects hardware YCbCr conversion when available.
  Playback color handling, image/video navigation, AV1 reverse-scrub crashes,
  and backward frame-cache behavior receive correctness and robustness fixes.
- Python exposes the video encoder/recorder and exact profile, capability and
  format queries through the shared schema-owned Rust implementation.
- Rust and Python NLP tutorials include local loss curves, reproducible run
  commands, recorded terminal output, evaluation and checkpoint evidence.
- SDK release archives and distro packages include the `oa` application.
  Public snapshot translation covers every workspace package and lockfile
  identity, including the SDK used by the Python extension.

## Fixed

- Request SDL's byte-order-aware RGBA32 camera format, preventing reversed
  channels on little-endian machines.
- Remove redundant aligned-U8 frame staging and unrelated engine-wide waits
  before native texture upload while preserving exact producer completion.
- Preserve the original recording on save cancellation or copy failure, and
  stage destination replacement so saving onto a link cannot truncate it.
- Align retained-secret ownership assertions and reject invalid ML-KEM cache
  operands. Software CI selects its admitted ML-KEM pack before execution;
  unadmitted ML-DSA operations are not silently converted into test successes.

## Compatibility and migration

Core remains the default build profile and requires its Vulkan 1.3 feature
contract. Older hardware must explicitly build with `OA_BUILD_PROFILE=compatibility`.
The public semantic Rust library namespace is `oa`; the registry packages remain
`oarust` and `oapython`. Python screen/camera capture bindings are not added by
these Rust application commands.

Linux builds now need PipeWire/SPA development headers, pkg-config and libclang.
Screen capture additionally requires libportal and a desktop ScreenCast portal
at runtime. The desktop always owns the capture permission decision. Optional
capture services are not initialized by headless Engine construction.

## Verification

Local Intel Iris Xe qualification includes native-resolution 2496×1664 screen
recording, 61 submitted and independently decoded frames in a two-second run,
MP4 finalization, camera capture, and GPU canvas transform/channel oracles.
Aligned/padded byte uploads, texture ownership and GPU-produced texture uploads
pass independent readback checks; the latter also passes synchronization
validation. These are local correctness/cadence samples, not cross-driver
performance claims. Rust/Python regression, formatting, generation and Clippy
checks are recorded with the release candidate. Published package and CI
verification are separate gates and are not implied by those local results.

## Known limitations

DMA-BUF zero-copy capture, capture audio and the remaining manual desktop
lifecycle checks remain open. Vulkan Video codec/profile support depends on
the selected device and driver. Sustained HEVC encoding on the recorded Intel
platform needs the qualified Mesa development-driver fix; older Mesa 26.2.7
can lose the device. Software Vulkan does not qualify physical media, WSI,
performance, or secret ML-DSA providers outside its admitted resource limits.
Ignored or skipped tests are not capability claims.

## Upgrade

Use `cargo add oarust` or `python -m pip install --upgrade oapython` after the
public release completes. Use the matching SDK archive/package for the `oa`
applications, and verify its complete SHA-256 manifest before installation.
