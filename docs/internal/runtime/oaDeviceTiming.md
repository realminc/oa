# OA Rust Device Timing

**Status:** Experimental

**Updated:** 2026-09-08

OARS exposes opt-in whole-plan device timing through the exact `Event` returned
by `Engine::submit_timed`. Normal eager execution and `Engine::submit` remain
uninstrumented.

```rust
let event = engine.submit_timed(&plan)?;
event.wait()?;
let duration = event.device_duration()?;
```

`Event::device_duration` waits for that event and returns `std::time::Duration`.
`Event::try_device_duration` never waits: it returns `Ok(None)` while the event
is incomplete and the completed duration afterward. Both return
`FailedPrecondition` for an event created by an untimed submission.

## Vulkan region and ownership

Each timed replay creates one timestamp query pool containing exactly two
queries. Its primary command buffer records this sequence:

```text
reset query pair
write TOP_OF_PIPE start timestamp
record complete executable graph
write BOTTOM_OF_PIPE end timestamp
submit and signal exact timeline epoch
```

The query pair is never reset or reused. This permits multiple timed replays to
remain in flight without an active-query collision or stale result. The
recorded command retains the pair until timeline retirement, while the event
retains it for later readback. Dropping either object never submits or waits;
the last owner destroys the pool only after no submitted command can reference
it.

Creation fails with `MissingCapability` when the selected compute queue reports
zero timestamp-valid bits or an invalid timestamp period. Timestamp readback is
64-bit. Elapsed ticks use wrapping subtraction followed by the compute queue's
valid-bit mask, then scale by `VkPhysicalDeviceLimits::timestampPeriod` into a
Rust duration.

The stage choices and query lifecycle follow the Vulkan timestamp-query model:
[`vkCmdWriteTimestamp2`](https://docs.vulkan.org/refpages/latest/refpages/source/vkCmdWriteTimestamp2.html),
[`vkCmdResetQueryPool`](https://docs.vulkan.org/refpages/latest/refpages/source/vkCmdResetQueryPool.html),
and
[`vkGetQueryPoolResults`](https://docs.vulkan.org/refpages/latest/refpages/source/vkGetQueryPoolResults.html).

## Measurement boundary

The returned duration covers the complete recorded executable graph, including
its internal compute barriers. It excludes host lowering, command allocation
and recording, queue submission, timeline waiting, mapped readback, and
retirement. It is not synchronized wall time and does not map a device
timestamp onto a host clock.

Host intervals use an explicitly scoped monotonic `std::time::Instant` at the
caller or benchmark-runner boundary. The Experimental fresh-process MatMul
runner consumes whole-plan duration while recording synchronized wall time as
a separate metric. A calibrated host/device clock mapping, per-node regions,
query pooling, and structured runtime metrics remain Planned.

## Acceptance evidence

Pure tests cover normal tick conversion and sub-64-bit wrap. Hardware tests
cover untimed-event rejection, two queued timed replays with independent query
pairs, completed non-blocking readback, output correctness, and event/query
lifetime after public engine and plan drop. Core, synchronization, and
GPU-assisted validation run as separate profiles.
