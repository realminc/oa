# OARS training lifecycle

**Status:** Experimental connected lifecycle

**Updated:** 2026-09-10

`oa::ml::ItTraining` and `oa::ml::training::ItTraining` are identity exports of
the Rust port of OA's ordinary `ItTraining` lifecycle. It currently borrows one
`Engine` and optimizer, owns step/epoch/timing state, and borrows registered
metrics and callbacks. It does not hide the model- or dataset-specific body and is not an
inheritance root for algorithm trainers.
`TrainingLoop` and `TrainingLoopConfig` remain temporary identity aliases for
the earlier Rust prototype spelling.

The subsystem follows the same facade/implementation split as `ml::nn`:

```text
src/rs/ml/training.rs             public umbrella and curated re-exports
src/rs/ml/training/iterator.rs    ItTraining lifecycle, metrics, snapshots
src/rs/ml/training/program.rs     immutable captured TrainingProgram
src/rs/ml/training/callbacks.rs   built-in presentation callbacks
src/rs/ml/training/callbacks/     validation, checkpoint, phase, CSV, and policy families
src/rs/ml/training/schedule.rs    pure learning-rate schedule policies
src/rs/ml/training/session.rs     bounded live commands, results, parameters, and snapshots
```

`ItTraining` and `TrainingProgram` are intentionally not merged. The former is
a mutable lifecycle/session owner; the latter is an immutable captured
executable value that the iterator may own and replay.

## Live control session

`TrainingSession` is the thread-safe control and observation handle for one
attached iterator. Rust reverses the donor C++ constructor dependency:

```rust,ignore
let session = oa::ml::TrainingSession::default();
let mut training = oa::ml::ItTraining::new(&engine, &mut optimizer, config)?;
training.attach_session(&session, oa::ml::TrainingSessionHandlers::default())?;
```

The handle is cloneable and contains only bounded commands and immutable
observation data. `ItTraining` retains every engine-thread handler and is the
only owner that applies commands. This avoids a mutable raw back-pointer and
does not falsely make the Vulkan-backed iterator `Send`.

`begin_step` drains commands before admitting a new forward/backward body;
`wait_begin_step` additionally blocks while paused and wakes when another
thread enqueues resume or stop. Pause, resume, stop, checkpoint, evaluation,
typed parameter mutation, program recapture, and rebuild all produce retained
revisioned results. Non-zero expected revisions implement optimistic
concurrency and stale commands are rejected without mutation. Result-history
queries use observer-owned cursors; `take_results` advances only its convenience
cursor and does not delete evidence needed by other observers.

The built-in `learning_rate` parameter is hot. Application parameters declare
their value kind, hot/recapture/rebuild/immutable class, optional numeric range,
and engine-thread getter/setter. Completed steps publish bounded snapshots with
loss, learning rate, last GPU time, mean wall time, and named finite scalar
metrics. Explicit `finish` publishes `Completed`; an iterator failure publishes
`Failed`. UI, Python, networking, and MCP remain consumers of this same handle,
not alternate training owners.

`Optimizer` is the object-safe behavior contract used by training policy:
zeroing gradients, stepping, learning-rate access, logical step observation,
and engine compatibility. `ItTraining::new_eager` admits any implementation of
that contract. `NoOpOptimizer` is the donor-backed route for a body that owns
its parameter updates but still wants iterator cadence, metrics, and callbacks;
the same route admits the current FP32 `Sgd`, `Adam`, and `Muon` owners.
`ItTraining::new` retains the concise AdamW route and is currently the only
constructor that admits whole-program capture and replay.

The automatic fixed-shape pattern mirrors OA's prepare/record split:

```rust,ignore
let mut training = oa::ml::ItTraining::new(&engine, &mut optimizer, config)?;
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
let mut training = oa::ml::ItTraining::new(&engine, &mut optimizer, config)?;
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

`ProgressBar` (`CbProgressBar` identity alias) implements the donor's throttled
block progress display with wall latency, sample rate, sequence/source rate,
and loss. `TrainingSummary` (`CbSummary` alias) reports initial/final/mean loss,
wall latency and throughput, GPU mean/p50/p95/min/max latency and throughput,
wall-to-GPU gap, duration, steps, batch, and sequence contract. Reporting uses
already completed scalar/timestamp evidence and does not add a second submit.

`EarlyStopping` (`CbEarlyStop`) implements the donor epoch-boundary patience,
minimum-delta, and minimize/maximize contract. Its default monitor is completed
epoch mean loss; an owned monitor closure can read application-shared
validation state without borrowing or re-entering the iterator. Non-finite
monitor output is a callback failure rather than an accidental stop decision.

`LearningRateScheduler` (`CbLrScheduler`) applies the donor one-based next-step
rule after each completed optimizer step. Cosine, warmup, one-cycle, cyclic,
cosine warm restarts, reduce-on-plateau, sequential, and linear-warmup-cosine
schedules are ported under `training/schedule.rs`. AdamW updates both its host
rate and graph-resident replay state after exact completion, so the same
callback works for eager and captured training without recompiling the plan.

`CsvLogger` (`CbCsvLogger`) writes the donor's explicit-unit schema with exactly
one row per completed step. It uses a bounded host buffer, flushes at epoch and
train-end boundaries, escapes textual CSV fields, and propagates creation,
write, and flush errors. Abandoning the callback does not turn `Drop` into a
hidden fallible finish operation.

`Validation` (`CbValidation`) runs an application-owned evaluator after every
completed epoch. Step-only lifecycles may select a non-zero interval and still
receive one final evaluation when the last step did not land on that interval.
It publishes the latest finite sample-weighted loss through a cloneable
thread-affine `ValidationMetric`; summary and early-stop callbacks observe that
same result without rerunning evaluation. Evaluation duration is measured and
excluded from both whole-run and epoch training-throughput denominators. The
evaluator owns its inference batching and returns explicit errors; the callback
does not submit hidden validation work.

`Checkpoint` (`CbCheckpoint`) connects completed step/epoch boundaries to the
one native `.oam` codec through `CheckpointManager`. It can save bounded
incremental checkpoints, update one best/master artifact from loss or a
preceding `ValidationMetric`, and restore the complete best model plus optimizer
state at train end. Checkpoint serialization, readback, and restoration time is
excluded from training-throughput accounting. The callback obtains only the
sealed `CheckpointOptimizer` capability from its restricted context; an eager
custom optimizer that does not implement exact persistence fails closed.

`PhaseSchedule` (`CbPhase`) maps consecutive named phase epoch ranges onto the
one iterator. It validates each phase and proves that every phase's epoch and
step totals exactly match the iterator configuration before the first step.
Its transition hook runs once per entered phase and receives restricted access
to the existing AdamW owner, allowing learning-rate or related optimizer policy
changes without a second loop. Phase descriptions never mutate the configured
work budget after construction.

## Rust ownership adaptation

The C++ callback base receives a mutable iterator and reports a retained status.
Rust uses `TrainingCallback`, an object-safe trait whose ordered hooks receive a
restricted context containing a copy-only `TrainingSnapshot` and the borrowed
optimizer through the object-safe `Optimizer` behavior contract, then return
`Result<TrainingControl>`. The context admits scheduling,
checkpoint, and excluded validation-time policy without exposing re-entrant
loop mutation. `Stop` is an explicit cooperative decision. Every callback at
that already-reached boundary still runs in registration order, matching the
donor's request-stop behavior; a hook failure stops dispatch immediately and
propagates. Metrics implement `TrainingMetric` and are updated exactly once
after completed device work. `LossMetric` currently provides mean and
last-value policies.

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
- GPU timing count, mean, median, p95, minimum, maximum, and workload rates;
- lazy train/epoch begin and ordered step/epoch/train end hooks;
- eager execution and existing captured-program replay;
- donor learning-rate schedules with eager/captured-safe AdamW updates;
- epoch-boundary early stopping with loss or application-owned monitoring;
- epoch and step-only validation with shared completed-result observation;
- best/latest native checkpoint policy, bounded rotation, and restore-best;
- validated consecutive phase ranges and one transition hook per phase;
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

Captured device-state Muon and phase-breakdown statistics remain unported.
Schema-owned replay-safe
Philox and Dropout now use graph-resident
per-operation counters with fail-closed training validation; shared streams and
serialized RNG state remain Planned. The current loop is a connected lifecycle
slice, not complete `ItTraining` parity.

## Acceptance

The external-style Vulkan tests run real eager forward/backward/AdamW work,
validation/checkpoint/phase policy, and captured replay. They assert exact callback order,
including callbacks after a cooperative stop at the same boundary; epoch
positions; one metric sample per completed step; non-duplicated step-only final
validation; phase transitions; optimizer/program step counts; automatic
prepare/record behavior, safe recapture, preparation-node rejection, workload
accounting, nonzero device timestamps, bounded checkpoint rotation, validation
best selection, exact model/optimizer restoration, revisioned live commands,
cross-thread pause wakeup, bounded snapshots, and observer-independent results:

```bash
cargo test --all-features --test ml it_training -- --ignored --test-threads=1
cargo test --all-features --test ml callbacks -- --ignored --test-threads=1
cargo test --all-features --test ml \
  it_training::training_session_applies_revisioned_commands_at_safe_points \
  -- --ignored --exact
```
