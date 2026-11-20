# OA v0.13.0 — Experimental Rust and Python preview

## Summary

OA strengthens compute fusion legality and records Iris Xe qualification for
captured media pipelines. This preview includes the compute compiler and
media improvements developed since the public 0.12.0 preview.

## Added and improved

- Retained decode, inference, YUV overlay and encode share a captured plan.
  On Iris Xe, eight 720p replay rows reached 91.3–112.7 fps and the
  multimodal workload reached 245.0 ten-frame clips per second.
- The compiler uses an ordered structural fusion rule table with
  schema-owned effect metadata, workload-specific tile providers and
  per-plan diagnostics.
- The viewer supports synchronized audio playback and indexed video seeks.
  Dolby AC-3 and E-AC-3 decoding runs inside OA.

## Fixed

- Preserve intervening reads when checking partitioned-write fusion.
- Keep later overwrites outside the fusion window and validate range
  bounds before allocating slot metadata.

## Compatibility and migration

The Rust crate is `oarust`, the Python distribution is `oapython` and both
use the `oa` namespace. Core remains the default Rust build profile.
GPU, queue, codec and driver capability determine route availability.

## Verification

The region legality suite includes 23 host checks. Twelve focused
fusion tests passed on Iris Xe with synchronization validation.
The media qualification included 256 sequential validated replays
for each of the nine video and multimodal workloads.

## Known limitations

Compiler tile providers and hardware video are Experimental. Additional
device qualification, provider default decisions and precision-specific
training gates remain open.

## Upgrade

Use matching Rust, Python and SDK versions. Verify the published artifact
checksums before installing from GitHub Releases.
