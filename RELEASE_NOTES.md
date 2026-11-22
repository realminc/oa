# OA v0.15.1 — Experimental Rust and Python preview

## Summary

Patch release for v0.15.0. Added the `oa.data.MatrixDataset` Python
binding and its top-level `oa.MatrixDataset` export. The class wraps the
native labeled feature/label pair, exposing `len`, `get_item`,
`get_sample` and the `x`/`y` matrices so Python consumers share the same
`oa::data` dataset contract as Rust callers.

## Verification

`cargo clippy -p oa-python --all-targets --all-features -- -D warnings`
and `cargo check -p oa-python --all-features` pass on the release commit;
the regenerated `_native.pyi` surface is produced by
`python3 tool/gen/fn/generate.py` with `--check` reporting no drift.
`oa.data.MatrixDataset` construction and top-level import are verified
against the built editable extension.

## Known limitations

Compiler tile providers and hardware video are Experimental. Multi-engine
workloads are usable through `set_default`/`_on` but distributed execution
is not shipped. Additional device qualification and precision-specific
training gates remain open.

## Upgrade

Use matching Rust, Python and SDK versions. Verify the published artifact
checksums before installing from GitHub Releases.
