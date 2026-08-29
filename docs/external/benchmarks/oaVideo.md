# OA local-video benchmark

**Status:** device-scoped decode and color-conversion checkpoint

**Measured:** 2026-08-29

This report characterizes OA's native Vulkan Video decode-to-RGBA path on one
Intel Iris Xe configuration. It compares Vulkan sampler YCbCr reconstruction
with OA's manual plane-sampling compute conversion for identical compressed
streams and identical RGBA output. It is not a remote-desktop, compositor
scanout, cross-vendor, encode, or universal performance claim.

## Current result

All admitted rows decoded, converted, and made 288 frames application-ready
without CPU pixel readback. Every measured process proved that the hardware
case dispatched Vulkan sampler YCbCr conversion and the compute case dispatched
none. Across both execution orders, even the slowest observed process exceeded
the 60 fps source cadence:

| Stream | Hardware observed range | Compute observed range | Slowest / 60 fps |
|---|---:|---:|---:|
| 1080p60 H.264 High 8-bit 4:2:0 | 408.316-579.323 fps | 347.963-470.715 fps | 5.80x |
| 1080p60 H.265 Main 8-bit 4:2:0 | 119.318-701.152 fps | 109.063-634.946 fps | 1.82x |
| 1080p60 AV1 Main 8-bit 4:2:0 | 456.477-547.273 fps | 356.606-579.525 fps | 5.94x |
| 1080p60 VP9 Profile 0 8-bit 4:2:0 | 457.563-703.728 fps | 387.460-721.995 fps | 6.46x |
| 2160p60 H.265 Main 8-bit 4:2:0 | 219.652-233.436 fps | 198.771-247.528 fps | 3.31x |
| 2160p60 AV1 Main 8-bit 4:2:0 | 214.926-228.316 fps | 210.596-226.976 fps | 3.51x |

These are unpaced saturation results. They establish application-ready
capacity, not compositor presentation deadlines or single-frame latency.

## Hardware YCbCr versus manual conversion

Each fresh process ran both paths. Before measurement, each path decoded and
converted 120 untimed frames, rewound without recreating the decoder, then
processed 288 timed frames. Two seven-process campaigns reversed path order to
expose shared-GPU frequency and queue-order effects. The ratio is hardware
throughput divided by manual-compute throughput; values above 1.0 favor
hardware.

| Stream | Hardware-first median; range; MAD | Compute-first median; range; MAD | Strict interpretation |
|---|---:|---:|---|
| 1080p60 H.264 | 1.288x; 1.207-1.331x; 0.026 | 0.979x; 0.885-1.509x; 0.094 | characterization only; reversed order fails the 15% spread gate |
| 1080p60 H.265 | 1.024x; 0.772-1.673x; 0.064 | 1.105x; 1.010-2.522x; 0.095 | characterization only; both orders fail the spread gate |
| 1080p60 AV1 | 1.043x; 0.944-1.124x; 0.070 | 1.205x; 0.792-1.468x; 0.222 | characterization only; both orders fail the spread gate |
| 1080p60 VP9 | 0.978x; 0.950-1.254x; 0.027 | 1.244x; 0.918-1.590x; 0.109 | characterization only; both orders fail the spread gate |
| 2160p60 H.265 | 1.099x; 1.046-1.168x; 0.014 | 1.046x; 0.925-1.112x; 0.052 | characterization only; reversed order narrowly fails the spread gate |
| 2160p60 AV1 | 1.006x; 0.972-1.080x; 0.032 | 1.000x; 0.970-1.023x; 0.011 | qualified parity; both orders pass the spread gate and differ from 1.0 by less than 1% |

No ratio is promoted as a speedup or regression on this laptop. Ten of twelve
order-specific campaigns exceed the predeclared 15% full-spread gate. UHD AV1
is the exception: both execution orders pass, and its 1.006x and 1.000x
medians establish parity for this exact workload and device. The other values
remain characterization rather than a selection policy.

AV1 uses a different ownership lowering. OA copies it into an ordinary sampled
staging image before hardware YCbCr conversion, releasing the DPB before the
following decode needs it. A preliminary direct-DPB experiment suggested a
regression, but it ran during the rejected contended session and is not
performance evidence. The admitted uncontended staging path measured 1.006x
and 1.000x order medians, with both passing the spread gate. H.264, H.265, and
VP9 retain direct coincident-DPB hardware sampling on this device.

“Hardware YCbCr” is not a compute-free scanout path. A compute dispatch samples
through a Vulkan YCbCr conversion sampler and writes the presentation RGBA
image. The manual fallback dispatch samples Y and UV planes and performs the
matrix/range reconstruction in shader code. AV1's admitted hardware path also
contains a device-local DPB-to-staging copy; every row performs zero CPU pixel
readback.

## Decode profile coverage

The exact live device query and OA admission matrix cover:

- H.264 Constrained Baseline, Main, and High, 8-bit 4:2:0;
- H.265 Main 8-bit and Main 10 P010, plus Range Extensions profile-IDC-4
  8/10-bit 4:2:0 single-picture subsets;
- AV1 Main 8-bit and a Main 10 P010 subset with loop restoration rejected
  before submission;
- VP9 Profile 0 8-bit and Profile 2 P010 single-picture coverage.

H.265 Main Still Picture remains fixture-gated because the installed x265 maps
its advertised selection to profile IDC 4 and the current QSV/`xe` stack cannot
initialize an honest profile-IDC-3 encode. Unsupported 4:2:2, 4:4:4, 12-bit,
SCC, H.264 High 4:4:4, AV1 High/Professional, and VP9 Profiles 1/3 remain
hardware-gated on this device. The TGL driver also rejects the UHD60 H.264
fixture's level 5.2 against its level-5.1 ceiling, and rejects the checked UHD
VP9 fixture. OA does not relabel unsupported streams into admitted evidence.

The maintained video implementation, SDK tutorials, conformance tests,
FFmpeg differential harness, and realtime benchmark use OA-owned containers,
paths, timing, synchronization, scalar math, and formatted output. Raw process
and `FILE*` calls remain only at the explicit FFmpeg test-oracle boundary.

The sustained performance matrix is intentionally 8-bit. Native 10-bit decode
is covered by P010 code-value differentials against FFmpeg and CPU color
oracles, but this checkpoint does not claim sustained HDR or 10-bit playback
performance.

## Measurement provenance

| Field | Recorded value |
|---|---|
| Runner checkpoint | clean `1826ea5d188133e6e9a0840f143c3d8eda19426e` for every row; the final release amend changes only these evidence documents |
| Final executable | `bin/release/test/vision/codec/benchVideoRealtime`, SHA-256 `afbf186e8c6f9f4e6f365821dff24c694b12a7e63ff4836b7f66b907ec9a7845` |
| Build | Release, Clang 22.1.8, static OA, embedded shaders, Vulkan validation off |
| Device | Intel Iris Xe Graphics TGL GT2, PCI device `0x9a49`, integrated GPU |
| Driver | Mesa 26.1.7-arch1.1, Intel open-source Mesa driver |
| Vulkan | API 1.4.354, conformance 1.4.0.0 |
| Host | Linux 7.1.8-arch1-3, x86_64 |
| Power | AC power, `intel_pstate` powersave governor, balanced platform profile |
| Protocol | one excluded fresh-process warmup, seven measured fresh processes, 5 s cooldown, 65 C package-temperature start gate, median + MAD + full range; measured starts 40-65 C |

All earlier YCbCr timing campaigns from this session were rejected after
concurrent VLC UHD 10-bit playback was disclosed. An earlier unprimed campaign
also left a large within-process DVFS order effect. Neither contributes values
to the tables above. The replacement campaign verified that VLC had exited,
used the clean release-candidate tree, and retained both execution orders. The
remaining spread is reported rather than filtered away.

## Correctness and safety gates

The final checkpoint passes:

- all 41 `TestVideoDecoder` Release tests;
- all 41 tests under ASAN with leak detection and strict string checks;
- all 41 tests under UBSAN with halt-on-error;
- focused UHD AV1 staged hardware/manual conversion under core,
  synchronization, and GPU-assisted Vulkan validation, with zero reported
  errors;
- hardware-dispatch counters for every hardware sample and zero hardware
  dispatches for every forced-compute sample;
- exact fixture hash, extent, cadence, codec/profile, frame completion, and
  zero-CPU-readback benchmark gates.

## Reproduce

Build and smoke the matrix first:

```sh
cmake --build build/release --target BenchVideoRealtime TestVideoDecoder -j
./bin/release/test/vision/codec/testVideoDecoder

OA_VIDEO_BENCH_VARIANT=2160p60-av1 \
OA_VIDEO_BENCH_PAIR_ORDER=hardware-first \
OA_VIDEO_BENCH_PRIME_FRAMES=120 \
./bin/release/test/vision/codec/benchVideoRealtime \
  --gtest_filter=BenchVideoRealtime.DecodePresentYcbcrPair
```

For publication evidence, wrap that command with
`tools/diagnostics/oaBench.py`: use one excluded warmup, seven measured runs,
5 s cooldown, a 65 C package start gate, and the metric
`hardware_over_compute`. Repeat with `OA_VIDEO_BENCH_PAIR_ORDER=compute-first`
and for each admitted variant:

```text
1080p60-h264  1080p60-h265  1080p60-av1  1080p60-vp9
2160p60-h265  2160p60-av1
```

Require `hardware_ycbcr_conversions=[1-9][0-9]*`,
`compute_ycbcr_conversions=0`, `pixel_readback_bytes=0`, and one passing test.
Keep a 15% maximum-spread gate for promoted performance claims. A failed
spread gate remains characterization evidence; it is not a speedup.
