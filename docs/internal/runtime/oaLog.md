# OA Rust logging

**Status:** Experimental

**Updated:** 2026-09-08

The Rust runtime has one structured host-logging session per `Engine`. The
engine owns the sink; a weak thread-local selection routes the namespaced
`oa::log_*` macros without creating a process-global owner. Calls made before
an engine exists, or on an unassociated thread, fall back synchronously to
standard error.

## Public contract

`LogOptions` configures the engine during construction:

```rust
let engine = oa::Engine::builder()
    .logging(
        oa::LogOptions::new()
            .minimum_level(oa::LogLevel::Info)
            .directory("var/log")
            .file_output(true),
    )
    .build()?;

oa::log_info!(oa::LogComponent::COMPUTE, "compiled {} kernels", count);
engine.flush_log()?;
engine.close()?;
```

Records contain a local timestamp, severity, fixed-width component, and text.
Console output is synchronous and optionally colored. File output appends to a
dated file using a validated filename prefix. Sink writes are serialized and
retain the first I/O failure for `flush_log` or `close`; macro calls do not turn
logging severity into control flow.

Trace and debug macro expansions do not evaluate their arguments in release
builds. Every other level checks the selected logger threshold before formatting.
Verbose per-dispatch, graph, barrier, and route diagnostics remain opt-in rather
than becoming default hot-path output.

After logical-device construction succeeds, the engine emits one version/count
line followed by one stable indexed compute-device record. The current runtime
owns one device and therefore reports `[0]`; the index and count syntax extends
without renaming when multi-device ownership is implemented. The compact
startup record includes the queried device name/type, Vulkan API, device-local
heap capacity, driver provider/info/raw version, Vulkan conformance version,
PCI vendor/device IDs, and selected compute queue family. It exposes no Vulkan
handles and makes no unqueried vendor inference.

```text
oa engine v0.1.4 · Vulkan · 1 compute device
[0] ComputeDevice · Intel(R) Iris(R) Xe Graphics (TGL GT2) · integrated GPU · Vulkan 1.4.354 · 11.49 GiB local memory
    Driver · Intel open-source Mesa driver · Mesa 26.2.2-arch1.1 · id INTEL_OPEN_SOURCE_MESA · version 0x06802002 · conformance 1.4.0.0
    Hardware · PCI 8086:9a49 · compute queue family 0
```

Those values are hardware/driver observations, not stable cross-machine output.
The preceding Mesa platform warning, when present, is emitted by the driver and
is intentionally not suppressed or rewritten by OA.

## Ownership and lifetime

- `Engine` owns the logger; `core` owns no stateful sink.
- Thread-local selection contains only `Weak` ownership and is restored when
  the engine leaves scope.
- Nested engines restore the previous live selection.
- Logging never submits work, waits for an event, or establishes synchronization.
- `Drop` only releases owned sink state. `Engine::flush_log` and
  `Engine::close` are the explicit failure-bearing flush and close boundaries.
- Internal worker threads do not implicitly inherit a caller's logger. A later
  scoped internal selection seam may associate them without changing ownership.

Structured performance metrics and GPU timestamps are separate runtime
facilities. A log timestamp is diagnostic context, never performance evidence.

## Current evidence

Unit tests cover component validation, severity filtering without argument
evaluation, file flushing and closure, and nested thread-local restoration. A
hardware integration test covers engine configuration, queried startup identity,
macro routing, explicit flush, path exposure, and close.
