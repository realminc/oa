# Foundational Matrix autograd recipes

Existing Matrix operation rows own attachment policy, named saved state and
exhaustive routing. The metadata-only `matrix_view.json` row owns reshape;
it creates no kernel ID or new runtime operation contract. Scalar compositions
reuse the canonical Scale record instead of duplicating an adjoint.

The numerical formulas behind these adapters are mechanically extracted from the admitted Rust
reverse handlers (the formulas at `c45a529`). Donors are
`source/cpp/lib/oa/core/autograd/matrix/autogradElemwise{,.gen}.h`,
`autogradReduce.h`, `autogradBlas.h`, `autogradIndex.h` and `autogradShape.h`.
Existing broadcast, clamp/Abs, deterministic gather and Dropout replay
adaptations are preserved. `@fields` records each recipe's required saved-state
interface; generation rejects any mismatch with the owning schema row.

The generator emits complete attachment functions and tape traversal adapters.
These adapters only take an output adjoint, call the named numerical backward
provider and accumulate its results. Elementwise/BLAS formulas are generated
into the regular operation files from `../backward/`; index and view formulas
are handwritten alongside their forward functions. Observer selection, Drop
and the borrowed tape context remain handwritten. Shared broadcast reduction
belongs to `src/rs/matrix/broadcast.rs`. There are no Rust body strings in JSON.
Numerical forward/backward providers remain in their existing Matrix files.
Source generation is not numerical or physical-device qualification.
