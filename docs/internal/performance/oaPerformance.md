# OA Rust Performance Evidence

**Status:** Canonical protocol; benchmark infrastructure is Planned

**Updated:** 2026-09-07

This document consolidates benchmarking, profiling, and GPU inspection policy
for the Rust implementation. It preserves the useful C++ OA evidence model
without claiming that its runners, timestamps, graph counters, or tuned routes
already exist here.

## Current status

OARS has no canonical fresh-process benchmark runner, calibrated Vulkan
timestamp service, route/fallback telemetry, checked-in benchmark suite, or
device roofline artifact yet. Consequently it has no admitted performance,
regression, utilization, occupancy, or bandwidth claims.

Ordinary `cargo test` timings and one interactive hardware run are correctness
or development evidence only. They are not a performance baseline.

## Evidence stack

| Question | Required evidence |
|---|---|
| Is the result correct? | independent oracle, property, conformance stream, or differential reference |
| Is Vulkan usage valid? | separate core, synchronization, and applicable GPU-assisted validation |
| Which work executed? | semantic/executable identity, kernel ID/hash, dispatch/submission counts, fallback counters |
| What state reached a dispatch? | RenderDoc with named objects and command-buffer labels |
| What happened across submissions/lifetimes? | graph/event logs plus GFXReconstruct or equivalent API trace |
| Did end-to-end performance change? | canonical fresh-process runner, raw samples, GPU timestamps, synchronized wall time |
| Why is a kernel slow? | matching vendor hardware profiler after the controlled benchmark identifies the region |

No tool substitutes for another row. A clean RenderDoc frame is not a
synchronization proof; a vendor trace is not an independent oracle; a kernel
timestamp is not end-to-end latency.

## Required benchmark protocol

Before comparing baseline and candidate:

1. Pass the same correctness oracle and workload-equivalence gate.
2. Fix inputs, shapes, layout, dtype, accumulator/numeric mode, batching,
   synchronization, route, fallback policy, build flags, device, driver, and
   power/thermal mode.
3. Build both artifacts from exact immutable commits and record dirty state.
4. Define whether the measurement includes engine startup, shader/pipeline
   creation, cache warmup, upload, execution, readback, and teardown.
5. Use a fixed warmup/cooldown policy and at least seven fresh-process measured
   samples per candidate.
6. Balance or alternate baseline/candidate order when cache, thermal, or
   background drift could bias sequential groups.
7. Report median and spread with every raw sample; never decide from one run or
   an average alone.
8. Require unexpected fallback counters to be zero.
9. Re-measure a squash/tag at its final commit; a similar tree is not provenance.

Changed precision, workload, batching, synchronization, fallback, oracle, or
measurement boundary invalidates a direct speedup comparison.

## Timing asynchronous execution

An eager operation currently returns before its GPU work necessarily completes.
Timing only the Rust function call measures allocation/record/submit overhead,
not completed computation. Every wall-time interval names its terminal host
boundary, such as `read_f32`, explicit event wait, or future plan completion.

Keep these measurements separate:

- cold engine and pipeline initialization;
- warm steady-state operation construction/lowering;
- command recording and queue submission;
- calibrated GPU execution time for the submitted region;
- transfer, host observation, and readback;
- end-to-end synchronized wall time;
- resource retirement and teardown when intentionally included.

GPU timestamps require the queue's timestamp-valid-bit handling and calibrated
host/device clock provenance when correlated with CPU phases. Unsupported
calibration is reported as unavailable; no offset is invented.

Validation and GPU-assisted instrumentation are disabled for release timing
only after the exact artifact passes those profiles separately. The benchmark
records both configurations.

## Counters and interpretation

Launch coverage, theoretical occupancy, achieved occupancy, useful-lane work,
compute utilization, and bandwidth utilization are different quantities.

- Predication or divergent lanes can reduce useful work without reducing
  theoretical occupancy.
- High occupancy can still underfill the device or introduce spills.
- Low reported DRAM throughput can mean latency, instruction, synchronization,
  or launch underfill; increasing traffic is not automatically an improvement.
- A lower barrier or dispatch count is not a win if packing, fusion, cache
  behavior, readback, or end-to-end latency regresses.

Use a measured current-device compute and effective-bandwidth roofline to form a
bottleneck hypothesis. Theoretical vendor peak is context, not acceptance
evidence. Algorithmic bytes/GFLOPs must be labeled separately from physical
traffic or hardware-unit saturation.

## Tool responsibilities

Use RenderDoc for single-dispatch pipeline state, descriptors, resources,
labels, and exploratory event location. Use GFXReconstruct or an equivalent
trace for multi-submit ordering and lifetime analysis.

Hardware-counter claims require the profiler matching the device that ran the
workload:

- Intel: VTune GPU Compute/Media Hotspots;
- NVIDIA: Nsight Graphics GPU Trace, plus Nsight Systems for system timelines;
- AMD: Radeon GPU Profiler;
- Arm Mali: Performance Studio Streamline;
- supported Android GPUs: Android GPU Inspector.

Record tool/version, permissions, device, driver, counter set, capture range,
operation/kernel identity, and raw artifact. An unavailable counter remains
unmeasured. Do not infer it from elapsed time, source code, another vendor, or a
failed capture.

## Runner acceptance contract

The future OARS benchmark runner must emit machine-readable raw artifacts with:

- executable and commit identity, dirty state, build profile and flags;
- Rust, Slang, SPIR-V tools, Vulkan loader/registry, device, and driver;
- schema version, semantic operation, physical kernel ID/hash, and route;
- dimensions, dtype, accumulator policy, layout, batch, and seed;
- correctness/oracle result and fallback counters;
- named timing boundaries, warmups, measured samples, cooldown/thermal data;
- median, spread, and units without discarding raw samples.

Until that exists, performance work may be labeled **Exploratory** with exact
commands and limitations. It cannot be labeled a regression gate, speedup,
cross-vendor result, or Shipped performance capability.

## References

- [Khronos Vulkan profiling guide](https://docs.vulkan.org/guide/latest/profiling.html)
- [Khronos development tools guide](https://github.khronos.org/Vulkan-Site/guide/latest/development_tools.html)
- [RenderDoc](https://github.com/baldurk/renderdoc)
- [GFXReconstruct](https://github.com/LunarG/gfxreconstruct)
