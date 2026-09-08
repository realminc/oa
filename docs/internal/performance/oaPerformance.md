# OA Rust Performance Evidence

**Status:** Canonical protocol; recording infrastructure is Experimental

**Updated:** 2026-09-08

This document consolidates benchmarking, profiling, and GPU inspection policy
for the Rust implementation. It preserves the useful C++ OA evidence model
while distinguishing the implemented recording foundation from missing
baseline, calibrated-clock, graph-counter, and tuned-route systems.

## Current status

OARS has Experimental whole-plan Vulkan device duration for explicitly timed
`ExecutionPlan` replay, a generic fresh-process recording runner, and a
checked-in six-shape FP32 MatMul suite. The workload checks every output before
timing, identifies its one fixed route and structurally unavailable fallback,
records synchronized wall and device duration separately, and preserves raw
logs plus source, executable, toolchain, Vulkan, device, driver, power, and
thermal provenance.

The runner does not yet accept hardware-scoped baselines or compare OA and
OARS. The runtime also lacks calibrated host/device clocks, physical shader
artifact hashes in execution telemetry, general route/fallback counters,
per-node timing, and a device roofline artifact. Consequently OARS still has no
admitted performance, regression, utilization, occupancy, or bandwidth claims.

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

The current device-only path is:

```rust
let event = engine.submit_timed(&plan)?;
event.wait()?;
let gpu = event.device_duration()?;
```

The recorded timestamps bracket the complete executable graph. One fresh query
pair belongs to each timed submission, and the event retains it through exact
timeline completion. `try_device_duration` is non-blocking; ordinary untimed
events reject duration readback. Timestamp values use modular subtraction at
the selected compute queue's valid-bit width and scale by the queried timestamp
period. This is device elapsed time only—not host submission, synchronized wall
time, or a calibrated host/device clock mapping.

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

## Recording runner

Build and stage the dedicated release workload, then select one or more suite
entries:

```bash
cargo build --release --example core_mat_mul_bench
python3 tools/build/stage.py --profile release --target core_mat_mul_bench
python3 tools/profiling/suite.py --workload core.matmul_nt.square_1024
```

`tools/profiling/bench.py` runs every warmup and measured sample in a fresh
process. `tools/profiling/suite.py` owns the checked-in workload set and writes
one JSON document plus raw stdout/stderr per workload. It refuses fewer than
seven measured processes and Vulkan validation-layer timing. A dirty tree
requires the explicit `--allow-dirty` escape hatch and is always recorded as
noncanonical. Debug profiles, unresolved selected-device identity, and a
missing local Vulkan registry are also noncanonical.

The current machine-readable artifacts include:

- executable and commit identity, dirty state, build profile and flags;
- Rust, Slang, SPIR-V tools, Vulkan loader/registry, device, and driver;
- schema version, semantic operation, stable command identity, fixed route,
  and complete executable hash;
- dimensions, dtype, accumulator policy, layout, batch, and seed;
- correctness/oracle result and fallback counters;
- named timing boundaries, warmups, measured samples, cooldown/thermal data;
- median, median absolute deviation, percentiles, spread, and units without
  discarding raw samples.

Canonical recording is necessary but is not baseline acceptance. Until
hardware-scoped baseline comparison and physical artifact telemetry exist,
numbers may be labeled **Experimental evidence** with exact commands and
limitations. They cannot be labeled a regression gate, OA-versus-OARS speedup,
cross-vendor result, or Shipped performance capability.

## References

- [Khronos Vulkan profiling guide](https://docs.vulkan.org/guide/latest/profiling.html)
- [Khronos development tools guide](https://github.khronos.org/Vulkan-Site/guide/latest/development_tools.html)
- [RenderDoc](https://github.com/baldurk/renderdoc)
- [GFXReconstruct](https://github.com/LunarG/gfxreconstruct)
