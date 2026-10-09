# Private autograd recipe templates

The existing `ml_training.json` forward rows own attachment arguments, saved
Matrix destinations/values, typed scalar state, output identity, family and node
names. These templates retain the admitted reverse algorithms. They do not own
another operation or saved-field catalog. Generation checks exact operation
coverage, rejects duplicate recipes, and checks each recipe's `Self` bindings
against the row's fields.

The generator emits complete family records and handlers under
`src/rs/ml/autograd/node/`, plus exhaustive `GradNodeOperation` routing. Attachments
construct those records directly. Records retain the same Matrix handles and
are stored by value; no new boxes, device copies, submissions or waits are added.
`BackwardStatus::Skipped` preserves the original reverse arm's `continue`.
Semantic completion, traversal and shared accumulation remain tape-owned.

## Donor and adaptation evidence

These recipe bodies were mechanically extracted from OARS commit `ce8e825`'s
`src/rs/ml/autograd/backward.rs`. Only the surrounding match pattern became a
`Self` binding, the borrowed gradient-map argument changed, and loop skipping
became an explicit status return. Provider calls, incoming-gradient scaling,
detached targets, derivative intermediates and accumulation order are unchanged.

OA C++ behavior authorities:

- Activation/SwiGLU: `source/cpp/lib/oa/ml/autograd/matrix/autogradActivation.gen.h`,
  `autogradActivation.h`, and `tools/gen/fn/schema/ml/mlFnMatrixActivation.toml`.
- Loss: `source/cpp/lib/oa/ml/autograd/loss/autogradLoss.h` and the matching
  `tools/gen/fn/schema/ml/` loss rows.
- Pooling: `source/cpp/lib/oa/ml/autograd/matrix/autogradPool.h` and
  `tools/gen/fn/schema/ml/mlFnMatrixPool.toml`.
- RoPE: `source/cpp/lib/oa/ml/autograd/matrix/autogradRope.h`.
- Upsample/adaptive pooling: the donor numerical forward/backward providers,
  especially `source/cpp/lib/oa/ml/fnmatrix/backward/fnMatrixBackward.cpp`,
  and the existing Rust adaptations recorded in `ml_training.json` provenance.

Classification: ownership, metadata and mechanical module adaptation of the
existing Rust providers. This pass neither replaces donor shader algorithms nor
upgrades prior provider provenance. In particular, existing deterministic
Upsample gathering and rectangular adaptive pooling corrections remain intact.

Complex parameter/version and multi-output records remain handwritten. Extending
those requires corresponding schema state and audited family templates. Source
preservation and software checks do not imply new GPU or performance qualification.
