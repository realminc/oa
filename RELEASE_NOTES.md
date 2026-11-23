# OA v0.15.3 — Experimental Rust and Python preview

## Summary

Patch release for v0.15.2. Adds TIFF to the image codec family: the
`image` crate's `tiff` feature is enabled and routed through format
detection, extension inference, decode, and encode on the Rust and
Python surfaces. TIFF dataset packs such as UC Merced land use can now
load through the standard image path. No API changes beyond the codec
coverage itself.

## Verification

Release-profile build and the `cargo test --release` suite pass on the
release commit (`0` failures; hardware-gated entries remain ignored).
TIFF detection, decode, and encode are exercised through the existing
codec coverage in that suite.

## Known limitations

Compiler tile providers and hardware video are Experimental. Multi-engine
workloads are usable through `set_default`/`_on` but distributed execution
is not shipped. Additional device qualification and precision-specific
training gates remain open.

## Upgrade

Use matching Rust, Python and SDK versions. Verify the published artifact
checksums before installing from GitHub Releases.
