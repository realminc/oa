# OARS training lifecycle

**Status:** Experimental connected lifecycle

**Updated:** 2026-09-09

`oa::ml::TrainingLoop` is the Rust port of OA's ordinary `ItTraining`
lifecycle. It borrows one `Engine` and `AdamW`, owns step/epoch/timing state,
and borrows registered metrics and callbacks. It does not hide the model- or
dataset-specific body and is not an inheritance root for algorithm trainers.

The automatic fixed-shape pattern mirrors OA's prepare/record split:

```rust,ignore
let mut training = oa::ml::TrainingLoop::new(&engine, &mut optimizer, config)?;
while training.step(
	|| load_batch(),
	|(batch, targets)| {
		let tape = oa::ml::GradientTape::new();
		let logits = model.forward(&batch)?;
		let loss = oa::ml::loss::cross_entropy(&logits, &targets)?;
		tape.backward(&loss)?;
		Ok(loss)
	},
)? {}
let summary = training.finish()?;
```

Preparation runs every step. The record closure runs for the first eager
warm-up and the second-step capture, after which the loop owns and replays the
`TrainingProgram`. `request_program_recapture` discards a completed program so
the next step records a replacement. Preparation may allocate or refresh stable
inputs but may not record executable work; a violation is rejected and its
newly recorded nodes are cleared before the loop stops.

Automatic capture records the reusable untimed Vulkan command before committing
the source session. Command-recording failure therefore follows the same eager
fallback contract, while a successful program reaches its first replay through
the command cache instead of paying a lazy recording cost.

Before that commit, replay-safety validation walks schema-classified kernel
roles. A captured optimizer-state advance must occur exactly once and before
every graph-state AdamW update; a host-stepped AdamW kernel is rejected. The
successful program retains an ordered eleven-stage compilation record covering
semantic validation through command recording. Stage records are immutable
capture evidence: automatic capture reports command recording as `Applied`,
while explicit `TrainingProgram::capture` truthfully reports it as `NotRun`
because that API still records lazily on first untimed replay.

Whole-program compilation is an optimization boundary. If compilation rejects
the recorded step, the source recording remains intact and that exact
forward/backward/AdamW work is submitted eagerly without calling `record`
again. Later steps remain eager until `request_program_recapture` explicitly
re-enables capture. `program_capture_fallback_count` and
`last_program_capture_fallback` make the decision observable. A record-closure,
optimizer, submission, wait, or metric failure remains a hard lifecycle error.

The explicit eager authoring pattern remains available:

```rust,ignore
let mut training = oa::ml::TrainingLoop::new(&engine, &mut optimizer, config)?;
training.add_metric(&mut loss_metric);
training.add_callback(&mut callback);

while training.begin_step()? {
	let batch = load_batch()?;
	training.seal_replay_inputs()?;
	training.zero_grad();
	let tape = oa::ml::GradientTape::new();
	let logits = model.forward(&batch)?;
	let loss = oa::ml::loss::cross_entropy(&logits, &targets)?;
	tape.backward(&loss)?;
	training.complete_step(&loss)?;
}
let summary = training.finish()?;
```

`complete_step` records the optimizer update, submits the complete eager batch,
waits for its exact event, reads the scalar loss once, updates metrics, and
fires callbacks. With an existing fixed-shape `TrainingProgram`,
`complete_program_step` replays that program without reauthoring its graph.
GPU timing uses a Vulkan timestamp pair around the exact eager batch or captured
program when enabled. It is currently opt-in because timed captured submission
records an independently owned query pair per replay; ordinary untimed replay
retains the one-command cache and remains the performance default.

## Rust ownership adaptation

The C++ callback base receives a mutable iterator and reports a retained status.
Rust uses `TrainingCallback`, an object-safe trait whose ordered hooks receive a
restricted context containing a copy-only `TrainingSnapshot` and the borrowed
optimizer, then return `Result<TrainingControl>`. The context admits scheduling,
checkpoint, and excluded validation-time policy without exposing re-entrant
loop mutation. `Stop` is an explicit cooperative decision; hook failure stops
the lifecycle and propagates immediately. Metrics implement `TrainingMetric` and are
updated exactly once after completed device work. `LossMetric` currently
provides mean and last-value policies.

The loop borrows metrics and callbacks rather than owning trait objects. They
remain application state and become accessible again when `finish(self)`
consumes the loop. `Drop` performs no submission, wait, callback, or implicit
finish.

## Current accounting

- fixed or variable-length epochs, including partial final epochs;
- exact completed steps, samples, sequence units, and source units;
- current-epoch and whole-training mean loss;
- wall throughput with explicitly excluded application time;
- exact last GPU duration when timing is enabled;
- lazy train/epoch begin and ordered step/epoch/train end hooks;
- eager execution and existing captured-program replay;
- automatic warm-up, fixed-shape capture, replay, and requested recapture;
- allocation-ordinal stable storage reuse after the first eager warm-up step;
- explicit or automatic preparation/replay-input sealing with capture-local
  resource ranges.

`seal_replay_inputs` is the Rust prepare/record boundary. Allocations before it
form an externally retained stable prefix; allocations after it are classified
as capture-local transients. Omitting it conservatively seals every frame
allocation as external. Equal-size slots reuse the exact Vulkan buffer only
after the prior step's event is complete. Size changes replace that ordinal; no
hidden fallback suppresses a reuse failure. Captured plans retain deterministic
handle-free resource IDs plus first/last executable-node access ranges. This is
liveness evidence, not alias authorization. Semantic external values and the
captured training loss are now marked explicitly; retained-owner accounting
also fails closed when a Matrix, view, parameter, or other owner escapes the
capture transaction. Non-overlapping qualified groups are materialized into one
private Vulkan buffer arena per group during capture. Alias-arena allocation or
replacement-planning failure preserves distinct storage and is observable in
the plan's fallback counter and reason.

Variable `epoch_steps` entries are normalized to at least one and their checked
sum owns the total-step budget, matching the donor behavior. A zero total is
open-ended; zero fixed epoch length disables epoch boundaries.

## Not yet ported

Built-in progress, validation, checkpoint, early-stop, CSV, summary, phase, and
learning-rate callbacks remain Planned. So do general optimizer abstraction,
graph reports, live `TrainingSession`, GPU timing distributions, and
phase-breakdown statistics. Replay-safe Philox kernels and state transformation
also remain Planned; the current generated kernel set has no device-state RNG
variant to admit. The current loop is a connected lifecycle slice, not complete
`ItTraining` parity.

## Acceptance

The external-style Vulkan tests run real eager forward/backward/AdamW work and
captured replay. They assert exact callback order, epoch positions, one metric
sample per completed step, optimizer/program step counts, automatic
prepare/record behavior, safe recapture, preparation-node rejection, workload
accounting, and nonzero device timestamps:

```bash
cargo test --all-features --test ml training_loop -- --ignored --test-threads=1
```
