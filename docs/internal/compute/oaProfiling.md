# OA Rust Vulkan Profiling and GPU Inspection

**Status:** Experimental whole-plan timing and fresh-process runner; broader profiling is Planned

**Updated:** 2026-09-09

OARS performance work begins with correctness and route identity. Vulkan
timestamps and the canonical fresh-process runner are measurement authority;
vendor tools explain utilization and stalls but do not replace controlled
benchmarks.

## Current tools

- `Engine::submit_timed` records one timestamp pair around an exact executable
  graph and attaches the duration to its event.
- `tools/profiling/bench.py` runs a named command in fresh processes and writes
  raw JSON with warmup, runs, cooldown, metric extraction and provenance.
- `tools/profiling/suite.py` runs checked-in workload configurations, including
  the Experimental six-shape MatMul recording suite.
- `tools/profiling/stable_clocks.sh` is the platform wrapper for qualified clock
  control; a degraded or partially applied profile is not canonical evidence.

Current timestamps exclude host planning, recording, queue submission, waiting
and readback. Calibrated host/device clocks, per-node phases, accepted baselines
and cross-implementation comparison remain Planned.

## Required measurement boundaries

Report separately:

- cold process/device/pipeline/cache startup;
- semantic graph construction and compilation;
- candidate query/autotuning;
- command recording or cache hit;
- host queue submission;
- GPU graph duration;
- wait/synchronization;
- upload, download and host observation;
- teardown.

A comparison must use equivalent work, precision, inputs, outputs, fallback,
synchronization and boundary. Comparing one path's kernel timestamp with another
path's end-to-end wall time is invalid.

## Protocol

1. Pass the independent oracle and required Vulkan validation first.
2. Fix executable/commit/dirty state, compiler flags, device/driver, power,
   dimensions, precision, route and fallback policy.
3. Use fixed warmup/cooldown and at least seven fresh-process measurements.
4. Report median and spread, retain every raw sample and balance baseline/
   candidate order against thermal drift.
5. Require zero unexpected fallbacks and record exact candidate/artifact/cache
   identity.
6. Repeat the final immutable tree; an earlier working-tree number is not
   release provenance.

The anecdotal 0.03-0.04 ms Vulkan versus 0.01 ms CUDA submit estimate is an
investigation lead only. A valid experiment must compare equivalent empty or
one-dispatch pre-recorded plans/graphs, matching wait/signaling behavior,
validation disabled only after validation passes, and the same host/power
conditions. Report API enqueue time, actual submit boundary and GPU completion
separately.

## Investigation ladder

1. Inspect semantic/executable reports, selected route, cache state and fallback
   counters.
2. Separate host, transfer and device time.
3. Determine whether the kernel is bandwidth, compute, occupancy, dependency or
   launch limited against a measured device roofline.
4. Use RenderDoc for a minimal single-frame resource/pipeline/barrier inspection.
5. Use GFXReconstruct or equivalent for multi-submit lifetime/order failures.
6. Use the profiler for the GPU that ran the workload for hardware counters.

Suggested tools are Radeon GPU Profiler on AMD, Nsight Graphics/System/Compute
as appropriate on NVIDIA, VTune GPU Compute/Media Hotspots on Intel, Arm
Performance Studio on Mali, and Android GPU Inspector on supported Android
devices. Record exact tool/version, counter set, permissions, capture range,
device, driver and raw artifact.

## Questions for kernel work

- Are all compute units occupied, and is useful occupancy limited by registers,
  shared memory, workgroup shape or insufficient waves?
- Is memory traffic expected after fusion and transient elimination?
- Are cache misses, bank conflicts, divergence or tail work dominant?
- Does the candidate pay workspace/reduction cost that its heuristic omitted?
- Is the “win” actually a cache hit, different precision, missing output,
  implicit fallback or changed synchronization?
- On mobile, do temperature and sustained clocks remain inside the admitted
  envelope?

## Graph and microfusion evidence

A fusion report includes source and fused semantic owners, eliminated bytes and
launches, compile/tune overhead, steady replay count, workspace, device time and
end-to-end time. Validate that no observable intermediate or domain metadata was
lost. A microsecond kernel win that increases plan compilation or prevents
reuse may lose for the real workload.

## Primary references

- [Vulkan timestamp queries](https://registry.khronos.org/vulkan/specs/latest/html/vkspec.html#queries-timestamps)
- [RenderDoc](https://renderdoc.org/)
- [Radeon GPU Profiler](https://gpuopen.com/rgp/)
- [NVIDIA Nsight](https://developer.nvidia.com/tools-overview)
- [Intel VTune Profiler](https://www.intel.com/content/www/us/en/developer/tools/oneapi/vtune-profiler.html)
