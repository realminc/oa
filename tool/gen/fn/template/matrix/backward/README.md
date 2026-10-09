# Matrix numerical backward providers

The same canonical operation rows that own forward generation and tape records
select these complete numerical formulas. `@fields` checks their required saved
state against the row; coverage, provider names and tape independence are checked
before publication. No new kernel or semantic operation ID is introduced:
compositions record their admitted component operations.

Donors: `source/cpp/lib/oa/core/autograd/matrix/autogradElemwise{,.gen}.h`
and `autogradBlas.h`. Adaptation: mechanical extraction of the admitted Rust
formulas at `c45a529`, preserving existing broadcast, clamp/Abs, and BMM/BMM-TN
routes. The latter reuse `ml::matrix` providers without transpose materialization;
this is existing numerical-provider reuse, not a new backend or algorithm.

The formulas and saved-value choices are unchanged. Binary adjoints are returned
before tape accumulation, preserving input accumulation order but permitting
accumulation dispatches to follow both calculations. GPU qualification remains
separate from schema and source-generation checks.
