# OA Rust logging

**Status:** Experimental

**Updated:** 2026-09-07

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
hardware integration test covers engine configuration, macro routing, explicit
flush, path exposure, and close.
