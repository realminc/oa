# OARS ML tutorial template

**Status:** Active acceptance contract; first Char-RNN tutorial Experimental

**Updated:** 2026-09-08

Rust ML tutorials will live under `sdk/rs/tutorials/ml/<domain>/` and be
registered as named Cargo examples. Their correctness contracts belong in the
`ml` integration-test target; a runnable tutorial is not a substitute for an
oracle test.

Every training tutorial must keep this phase order:

1. fixed model and dataset dimensions;
2. deterministic sampler with an explicit seed;
3. model and optimizer construction;
4. initial held-out evaluation;
5. exact-step forward, loss, backward, optimizer loop;
6. final evaluation and a material improvement assertion;
7. inference or generation when applicable;
8. save, fresh-model reload, and metric roundtrip once persistence exists.

The eventual training lifecycle and standard callbacks own cadence, timing,
progress, and summaries. Tutorial bodies own batch creation and
forward/loss/backward composition. They do not manually select kernels, submit
each operation, wait after each layer, print an ad-hoc line per step, or omit a
phase silently.

Sequence tutorials additionally require fixed-seed token generation and a
visible prompt/output boundary. Classification tutorials require held-out
accuracy. Multi-stage tutorials repeat model/optimizer/training setup per stage
and persist every shape/configuration value needed for a fresh-process reload.

The first admitted tutorial is the character RNN vertical slice. Its exact
300-step corpus, model, loss, accuracy, prompt, and greedy-generation contract
is implemented and hardware-verified. It retains an explicit provisional loop;
the shared training lifecycle, metrics/callbacks, and checkpoint persistence
remain open before the complete suite contract is reached.
