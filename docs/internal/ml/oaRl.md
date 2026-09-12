# OARS reinforcement-learning foundation

**Status:** Experimental native CartPole and flat Lunar Lander workflows, environment values, policy/loss primitives, rollout/replay storage, Actor-Critic, and PPO/DQN/SAC training; remaining environment integrations Planned

**Updated:** 2026-09-12

**Donor authority:** OA C++ `docs/internal/ml/oaRl.md`, `environment.h`,
`environment.cpp`, `rollout.h`, `rollout.cpp`, `replay.h`, `replay.cpp`,
`itRolloutTraining.{h,cpp}`, `dqnTrainer.{h,cpp}`, and `sacTrainer.{h,cpp}`

## Current implementation

`oa::ml::EnvironmentSpace`, `EnvironmentSpec`, and `EnvironmentTransition`
preserve the donor's backend-neutral environment metadata and five-field step
boundary. Shapes exclude the leading environment dimension; an empty field
shape means one scalar per environment. Box, Discrete, and Binary values have
checked dtype, shape, range, and cardinality contracts.

The currently admitted Rust dtypes narrow the donor surface deliberately:
Box accepts F32, Discrete accepts U8/I32/U32, and Binary uses U8. Broader scalar
types become legal only when Core Matrix storage admits them. Constructors
reject invalid definitions immediately instead of publishing mutable metadata
that can become inconsistent later.

`EnvironmentSpec` requires scalar floating reward plus separate scalar Binary
`terminated` and `truncated` fields. `terminated` denotes an underlying task
terminal and disables value bootstrap; `truncated` denotes an external/time
limit boundary and retains one-step bootstrap. They are never collapsed into a
single `done` flag.

The Matrix validation boundary is metadata-only: it checks exact batched shapes
and dtypes without readback, submission, or synchronization. A hardware-backed
test validates a complete three-lane CartPole-shaped transition and rejects
wrong action dtype and observation batch size.

`oa::ml::environment::{normalize_observation,scale_action,clip_reward}` ports
the complete donor `FnEnvironment` transform family. Each public operation is a
schema-owned differentiable semantic composition over existing Core broadcast,
scalar, and clamp kernels; no environment-only shader or execution owner was
added. Intel Vulkan reproduces the donor normalization, clamped affine mapping,
and reward-clipping vectors exactly and rejects invalid numerical policies.

`oa::ml::advantage::gae` is the first connected RL algorithm primitive. It is
a mechanical port of OA's one-invocation-per-environment reverse-time shader
over nonempty `[time, environments]` rollouts. Reward, value, and next value
are F32; termination and truncation are distinct U8 masks. Termination disables
bootstrap, while either boundary stops trace propagation, so truncation retains
the donor's one-step bootstrap without leaking an autoreset episode backward.
The operation records one `oa::ml::advantage::gae` semantic identity and returns
device-resident advantage and return matrices without a host boundary.
`gae_into` records that same operation into checked, non-aliasing caller-owned
F32 matrices so a fixed-capacity rollout can reuse storage without introducing
a duplicate semantic contract.

Independent Intel Vulkan evidence reproduces the donor eight-element oracle,
including the deliberately extreme next value behind a termination, and checks
invalid shapes, dtypes, discount parameters, exact output reuse, and alias
rejection.

`oa::ml::advantage::normalize` ports the donor differentiable whole-rollout
standardization without adding an ML shader. One schema-owned semantic
operation composes Core Sum, Scale, broadcast Sub/Mul/Div, AddScalar, and Sqrt;
constant inputs remain finite zeros because epsilon is strictly positive.
Independent Intel Vulkan evidence covers exact population standard scores,
the constant-input case, invalid epsilon/dtype contracts, one-operation graph
provenance, and the complete analytic input adjoint.

`oa::ml::loss::ppo_clipped_policy` and its explicit backward mechanically port
the donor ratio, sign-dependent clipping, negative-surrogate, and mean-scaling
rules. `oa::ml::loss::ppo` composes that policy term with MSE critic loss and
mean entropy using the donor `0.2`, `0.5`, and `0.01` defaults, while retaining
all four scalar results for metrics. The composite owns one semantic operation;
its physical children and reverse work remain schema-owned. Intel Vulkan
evidence reproduces the donor four-ratio loss/gradient vector and independently
checks policy, value, and entropy adjoints through the total loss.

`oa::ml::loss::dqn` mechanically ports the donor selected-action gather,
maximum-next-Q Bellman target, and unit-beta Smooth L1 mean. Termination alone
suppresses bootstrap; truncation retains the discounted next-state value. The
target dispatch is private physical lowering beneath one DQN semantic operation,
and the target path is detached, so reverse mode reaches only the selected
entries of Q. Intel Vulkan evidence reproduces the donor `[11, 2, 33]` target
and `[1, 4, 5]` selected values, independently checks the exact scalar loss and
sparse Q adjoint, proves next-Q receives no gradient, and covers invalid dtype
and discount contracts.

`oa::ml::loss::{sac_critic,sac_actor}` completes the donor's foundational SAC
loss pair. The critic target takes the smaller detached next-Q value, subtracts
the entropy-weighted next log probability, suppresses bootstrap only for
termination, and feeds the existing twin MSE route. The actor computes
`mean(alpha * log_probability - min(q1, q2))` entirely from differentiable Core
operations. Both compositions retain one semantic identity and the donor
`0.99`/`0.2` defaults. Intel Vulkan evidence reproduces targets
`[6.1, 2, 28.3]`, total critic loss `426.25`, actor loss `-1.65`, complete
twin-Q/actor/log-probability adjoints, and target detachment.

`oa::ml::RolloutBuffer` is the connected fixed-capacity rollout session. Its
validated `RolloutConfig` allocates one set of time-major Matrix values and its
`append` method records the donor's fused device-side copy for observation,
action, reward, value, next value, old log probability, termination,
truncation, and validity. The first eight transition values are fixed semantic
inputs; the nine retained destinations use the runtime's aligned variadic
mutation contract, so one semantic append owns all nine fresh SSA aliases
without widening the global fixed-arity contract.

`finalize` requires a full buffer and delegates to the allocation-free GAE
route. `reset` clears only validity on the device and resets host cursor state;
the other allocations remain available for overwrite. OARS uses an atomic
packed-byte store for U8 masks because its portable Vulkan baseline does not
require `StorageBuffer8BitAccess`. This is a mechanical ABI adaptation of the
donor byte stores, not a changed rollout algorithm.

Intel Vulkan evidence captures three appends plus GAE as four semantic
operations, checks all eleven retained fields against an independent
time-major/GAE oracle, proves nine mutation aliases per append, exercises
packed-byte writes across a partial word, and verifies reset/reuse, lifecycle,
capacity, invalid configuration, invalid transition, and storage-alias
rejection. A rejected surrounding capture after a successful append restores
the exact pre-capture readiness of retained rollout storage. Storage created as
a failed capture output remains failed. The rollout coordinator then rewinds
the host cursor only after that command transaction has been rejected.

`oa::ml::training::ItRolloutTraining` ports the donor's alternating Collect and
Update lifecycle over one `RolloutBuffer` and one ordinary `ItTraining` update
unit. Its checked schedule fixes rollout count, horizon, environment count, and
full-batch update epochs. `begin_rollout` resets and binds one matching buffer;
`finalize_rollout` requires full collection and records GAE; each paired
`begin_update`/`complete_update` advances the existing optimizer, metric,
callback, timing, and completion lifecycle. `finish(self)` remains the sole
terminal callback boundary.

`abort_rollout` is deliberately narrow: it is valid from collection through
finalization only before the first update, and only after the enclosing Engine
capture has rejected its unsubmitted commands. It restores host collection
state without inventing a second command owner. Intel Vulkan tests prove two
update epochs and rejected-capture reuse of the same retained buffer.

`oa::ml::ReplayBuffer` is the matching off-policy storage session. It owns one
preallocated circular observation/action/next-observation/reward/boundary set;
batched append wraps by physical slot without host copies, and host-only reset
makes old bytes inaccessible without recording pointless device work. Action
storage accepts the donor's I32 categorical or F32 continuous representation.
Deterministic sampling with replacement preserves the donor `hash32` sequence,
returns sampled physical U32 indices, and gathers every field on the GPU.
Append is one schema-owned mutation with six aligned variadic aliases; sampling
is one read-only semantic operation with seven fresh outputs.

The replay oracle fills past capacity with two differently sized batches,
checks exact circular slot order, derives all seeded indices independently,
and checks every gathered field including packed termination and truncation
bytes. Lifecycle tests cover empty/zero sampling, host reset, invalid configs,
and invalid transition metadata. Replay randomness is deliberately local to
the donor operation today; graph-resident shared RNG streams remain Planned.

`oa::ml::training::DqnTrainer` is the first connected algorithm coordinator.
It borrows the one Engine, online and target Modules, Optimizer, ReplayBuffer,
metrics, and callbacks; `ItTraining` remains the sole optimizer/callback
lifecycle. Construction validates scalar I32 replay actions and matching
observation metadata, then matches registration-derived parameter paths and
deep-copies every online parameter into independent target storage. Public
`matrix::copy` is schema-owned for U8/F32/I32/U32, records one asynchronous
device dispatch, and has the identity FP32 adjoint.

Each update preserves the donor `seed + zero_based_update` replay sequence,
flattens observations to `[batch, observation_elements]`, evaluates the target
before opening the tape, and differentiates only the online Q path through the
connected DQN loss. `ItTraining` performs the optimizer step, exact wait,
scalar loss read, metrics, and callbacks. The configured cadence performs an
exact named-parameter target copy after completed updates. Drop performs no
terminal work; `finish(self)` is the explicit callback boundary required by
the Rust runtime contract. Intel Vulkan evidence covers the donor two-update,
batch-four, interval-one recipe, finite loss, exact target synchronization,
disabled target gradients, and optimizer step count. Native trainer checkpoint
and restore remain pending because iterator progress must be restored together
with model and optimizer state.

`oa::ml::training::SacTrainer` ports the donor's fixed-temperature coordinator
without collapsing its two optimizer units. The primary critic and actor each
retain an independently observable `ItTraining` lifecycle. One update samples
continuous replay, evaluates the next-state tanh-normal policy and both target
critics outside any tape, trains both online critics through `sac_critic`, then
trains the actor through both online critics and `sac_actor`. Actor and critic
policy seeds preserve the donor offsets; critic inputs concatenate observation
and action on-device, and `[B,1]` critic outputs become differentiable `[B]`
views.

Both target critics use the same preflighted exact synchronization authority as
DQN. Every path, dtype, shape, owner, version budget, and source/target identity
is validated before any copy is recorded; every copy allocation succeeds before
the live target handles are replaced, and both critic pairs share one exact
completion boundary. Intel Vulkan evidence covers the donor batch-four,
one-action, one-update recipe, finite actor/critic losses, disabled target
gradients, exact synchronized targets, and one step in each optimizer. Learned
temperature, Polyak targets, and native trainer checkpoint/resume remain
Planned because the donor slice itself is fixed-alpha/exact-copy.

The Core prerequisites for categorical policy are connected. Public
`matrix::gather_last_dim` and its deterministic adjoint preserve repeated and
out-of-range I32 index behavior. Public `matrix::sample_logits` preserves the
donor greedy, dense-temperature, TopK, and nucleus routes, including
first-maximum ties and exact Philox seeds. Captured stochastic routes use a
private graph-resident RNG counter followed immediately by its advance; replay
therefore changes samples without freezing a host seed.

`ml::policy::{evaluate_categorical,sample_categorical}` now mechanically port
the donor compositions. Both return action, selected log probability, entropy,
and critic value through one `PolicyResult`; action and value are read-only
zero-copy output aliases. Evaluation lowers LogSoftmax, last-axis gather, Exp,
Mul, Sum, Neg, and metadata reshapes while capture records one parent policy
operation over all executable nodes. Sampling prepends the existing
replay-safe `sample_logits` path and preserves the complete caller seed as a
semantic attribute. No policy-only shader or second graph exists.

Independent Intel Vulkan evidence checks the donor probability/entropy
formula, seeded sampling and reevaluation, the single-operation/multi-node
provenance, read-only SSA aliases, and the complete entropy adjoint. Composite
lowering rollback is transactionally covered by a separate runtime test.

`ml::policy::{evaluate_tanh_normal,sample_tanh_normal}` ports the continuous
donor composition with matching FP32 `[E,A]` mean/log-standard-deviation/raw
action values, FP32 `[E]` critic values, log-standard-deviation clamp
`[-20,2]`, configurable action range, tanh Jacobian correction, base-normal
entropy, and reparameterized Philox sampling. Raw action and critic value retain
zero-copy aliases. Capture again owns one policy operation across every Matrix,
activation, and RNG node, including nested evaluation inside sampling.

The Vulkan oracle covers exact CPU action/probability/entropy values, seeded
sample reevaluation, plan provenance, reparameterization gradients, and clamp
boundary gradients. OARS deliberately corrects OA C++ donor defects in
`GradClampMax`/`GradClampMin` and `GradAbs`: donor clamp masks propagate through
the clipped side and the donor absolute-value masks invert the mathematical
sign. Rust propagates through the retained clamp side with inclusive boundary
derivatives, and uses `sign(x)` with a zero subgradient for absolute value. The
SAC actor oracle exercises both signs through its differentiable minimum.

`oa::ml::ActorCritic` is the object-safe discrete policy/value behavior that
PPO consumes through the existing `Module` registry. The default
`CategoricalActorCritic` mechanically preserves the donor's independent policy
and value `Linear -> ReLU -> Linear -> ReLU -> Linear` towers. Rust adds an
explicit seed and uses six adjacent deterministic Xavier streams. The value
head remains a zero-copy `[batch,1] -> [batch]` semantic reshape. An independent
CPU oracle checks both complete towers and all twelve registration-derived
parameter paths against Vulkan output.

`oa::ml::training::PpoTrainer` now composes that behavior, one retained
`RolloutBuffer`, one `ItRolloutTraining`, categorical policy evaluation, GAE,
advantage normalization, the connected PPO loss, reverse mode, and any
object-safe Optimizer. Environment stepping remains caller-owned. Collection
uses the donor `seed + one_based_action_index` sequence and stores action,
log-probability, entropy-derived policy state, current value, and separately
evaluated next value without host copies. Each update re-evaluates the entire
finalized batch, completes the ordinary optimizer/callback/timing boundary, and
publishes synchronized total, policy, value, and entropy scalars.

The donor environment-neutral one-rollout/two-environment gate passes on Intel
Vulkan with a real Adam update, finite metrics, parameter gradients, and exact
step count. A second gate rejects the complete collection capture, restores
retained storage plus the action-seed cursor, recollects, and completes training.
Native atomic trainer checkpoint/resume remains Planned with DQN and SAC; OARS
does not expose the donor's partial model-only `save`/`load` as trainer resume.

Native environments implement `oa::ml::environment::Environment` and embed one
`EnvironmentExecution`. That lifecycle borrows the canonical Engine recorder as
a movable transaction: begin is idempotent while recording, submit returns one
exact Event, another recording is rejected until wait, cancel discards only
unsubmitted work, and close explicitly waits accepted work. Transactional host
metadata commits only after accepted submission and rolls back on recording
failure/cancel. Drop never waits. This replaces the donor's private secondary
`ExecutionSession` instance while preserving its observable lifecycle.

`oa::ml::RolloutCollector` borrows an Environment and ActorCritic, records reset,
categorical sampling, stepping, completed-lane reset, retained rollout append,
and GAE as one transaction, and returns its Event without waiting. Accepted
metrics count horizons and individual lanes. Recording failure rewinds the
rollout cursor and deterministic action-seed index together with environment
host state.

The donor zero-sized `PolicyEvaluator` becomes the idiomatic one-shot
`oa::ml::evaluation::evaluate_categorical`. It records a greedy categorical
horizon, submits and waits exactly once, then reads only reward, terminated, and
truncated arrays. Its summary keeps per-lane episode return and treats either
boundary as completion. Hardware gates cover the complete transaction,
collector, and evaluator; a pure oracle separately covers multiple terminations
and truncations across lanes.

The donor's concrete CartPole workload remains SDK-owned in Rust at
`oa::sdk::ml::rl::CartPole`; reusable ML contracts do not depend on one task.
Its vectorized reset and step operations are ordinary generated semantic
contracts backed by an SDK Slang kernel pack and lowered through the sole
Engine. The environment commits deterministic seed and episode metadata only
after a recording is accepted, so cancellation cannot advance host-visible
state. Hardware differential tests compare every state component and
reward/boundary field against an independent scalar CPU oracle and cover
deterministic reset, time-limit truncation, invalid actions, completed-lane
masking, and selective reset.

The Rust `ml_rl_cart_pole_ppo` executable now matches the donor training recipe:
64 environments, horizon 128, 40 rollouts, four full-batch update epochs,
separate `4 -> 64 -> 64` actor and critic towers, and AdamW at `2.5e-4` with
zero weight decay. On Intel Iris Xe the fixed-seed acceptance improved mean
completed return from 33.36 to 453.30, exceeding both the +25 improvement and
75 absolute gates. Native `.oam` restoration reproduced 453.30 exactly and
restored optimizer step 160. PPO update timing can collect Vulkan timestamps
independently from end-to-end wall time. The timestamp-enabled validation took
188.09 seconds end to end while its optimizer updates averaged 14.48 ms of
Vulkan device time, identifying repeated collection/evaluation graph authoring
and preparation as the next performance seam rather than the update kernels.
The separate donor-sized `ml_rl_cart_pole_rollout` executable records its fixed
policy, 2,048 transitions, completed-lane resets, and GAE as one transaction;
its Intel run produced 2,048 reward with no collection-time host tensor reads.

## Planned dependency order

The first native flat-terrain Lunar Lander vector environment and its native
policy workflow are implemented as Experimental checkpoints; seeded terrain
batches and Python adapters remain incomplete.
Remaining work must reuse the one Engine, semantic graph, executable graph, optimizer,
`ItTraining`, completion Event, and `.oam` persistence contracts. No nested RL
tensor type, trainer framework, queue owner, or implicit wait path is admitted.

Port order is:

1. reusable compiled CartPole collection/evaluation graphs without changing
   seeded results;
2. native PPO/DQN/SAC checkpoint/resume with iterator progress;
3. configurable PPO minibatching only after the donor-compatible full-batch
   baseline remains covered;
4. extend Lunar Lander vector execution with seeded/caller-provided terrain and
   long-horizon differential qualification;
5. qualify the native Lunar teacher and raw-PPO long learning gates, then add
   optional Python/Gymnasium adapters after that native evidence is accepted.

The Lunar Lander scalar foundation is complete below
`oa::sdk::ml::rl::lunar_lander`: manifest, terrain, configuration, state,
double-precision integration, fixed-order contacts, frozen 33-value
observation, reward decomposition, deterministic spawn, all terminal/truncation
reasons, and the scripted landing controller are connected. The full seeded
96-step trace matches the C++ digest `a85ea7bfd77f9ab5`, in addition to focused
dynamics and lifecycle oracles. `LunarLander3dVector` adds schema-owned donor
reset/step kernels over the canonical `EnvironmentExecution`. The SDK-owned
workflow now composes the existing categorical Actor-Critic, PPO, `ItTraining`,
autograd, Event, and `.oam` owners; it adds no Lunar trainer framework. It ports
the donor's policy-only scripted-teacher curriculum, chunked greedy first-episode
evaluation, terminal-reason and flight telemetry, one-rollout PPO collection,
and exact checkpoint-restoration comparison. Intel Vulkan
qualification covers seeded reset, a 24-step five-lane FP32-vs-FP64 trace,
scripted landing through contact and dwell, all ordinary terminal causes,
transactional cancellation/reseeding, FP32 admission edges, and repeated odd
257-lane reuse. The bounded donor-shaped teacher/PPO/checkpoint smoke passes on
the same device. The expensive default 512-episode teacher and raw-PPO learning
gates remain opt-in and are not yet accepted convergence evidence. Procedural
terrain batches, rendering, and Python adapters remain incomplete.
