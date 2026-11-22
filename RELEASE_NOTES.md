# OA v0.15.0 — Experimental Rust and Python preview

## Summary

Feature release for v0.14.1.

Added complex STFT and inverse STFT audio transforms with autograd
adjoints, deterministic audio and image augmentation operations, and
variable-length audio batching helpers. Handwritten Python owners now
take a trailing optional `engine`, matching the defaulted-engine
convention.

`oa::data` gains `MatrixDataset`, a dense labeled `[N, ...]`/`[N]`
feature–label pair implementing `Dataset` with the slice, bounds, and
sample-pairing contract `DataLoader`/`collate` consume. The previous
`DsOhlcv` adapter is renamed `OhlcvDataset` to match the `Dataset`
family spelling.

Removed the bundled `sdk/asset/audio/commands` subset. The
`tu_audio_commands` tutorials now resolve the mini_speech_commands tree
under the shared data root (`OA_DATA_DIR`, else `~/.oads`) or an explicit
`--dataset` path; the subset-calibrated `COMMANDS_FINAL_LOSS` and
`COMMANDS_TEST_ACCURACY` oracles are removed with the asset.

## Verification

`cargo fmt`, `cargo clippy --workspace --all-targets --all-features
-- -D warnings` and `cargo test --workspace --all-features` pass on the
release commit. `python3 tool/gen/fn/generate.py --check` reports no
drift. Device-admission tests remain environment-gated through
`OA_TEST_DEVICE_INDEX`.

## Known limitations

Compiler tile providers and hardware video are Experimental. Multi-engine
workloads are usable through `set_default`/`_on` but distributed execution
is not shipped. Additional device qualification and precision-specific
training gates remain open.

## Upgrade

Use matching Rust, Python and SDK versions. `DsOhlcv` is renamed
`OhlcvDataset` on both language surfaces. Speech-commands data is no
longer bundled; fetch the `mini_speech_commands` pack via `oads` or the
upstream archive and extract it under the OA data root, or pass
`--dataset`. Verify the published artifact checksums before installing
from GitHub Releases.
