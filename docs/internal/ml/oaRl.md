# OARS reinforcement-learning foundation

**Status:** Experimental environment value contracts; execution and algorithms Planned

**Updated:** 2026-09-10

**Donor authority:** OA C++ `docs/internal/ml/oaRl.md`, `environment.h`, and
`environment.cpp`

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

## Planned dependency order

The donor `Environment` execution session, rollout/replay storage, categorical
and continuous policies, GAE, PPO, DQN, SAC, collectors, evaluators, transforms,
Python adapters, and CartPole acceptance run are not yet shipped in OARS. They
must reuse the one Engine, semantic graph, executable graph, optimizer,
`ItTraining`, completion Event, and `.oam` persistence contracts. No nested RL
tensor type, trainer framework, queue owner, or implicit wait path is admitted.

Port order is:

1. environment execution lifecycle with explicit exact completion;
2. schema-owned policy sampling/evaluation and GAE operations;
3. fixed-capacity rollout and replay storage;
4. PPO, DQN, and SAC loss/optimizer integration;
5. collection/evaluation sessions and native CartPole learning proof;
6. optional Python/Gymnasium adapters after the native contract is complete.
