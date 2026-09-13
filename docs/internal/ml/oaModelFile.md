# OARS native model files

**Status:** Experimental checkpoint integration; general artifact API Planned

**Updated:** 2026-09-11

**Donor authority:** OA C++ `source/cpp/include/oa/ml/modelFile.h` and
`source/cpp/lib/oa/ml/modelFile.cpp`

## Identity and ownership

`.oam` is OA's native model-file wire identity. It is a model file, not an
archive, Rust serializer dump, or backend cache. The private `ml/model_file.rs`
codec owns byte layout and integrity. `ml/checkpoint.rs` translates between
that host representation and live engine-owned Module plus sealed
`CheckpointOptimizer` values.

The current public surface includes `oa::ml::save_checkpoint`,
`oa::ml::load_checkpoint`, and `CheckpointManager` policy used by
`CbCheckpoint`. A future general artifact surface is named
`oa::ml::ModelFile`. External formats belong to explicit translator/import
operations; an `io` module must not become a second owner of `.oam`.

## Implemented wire contract

OARS writes version 3 and reads versions 1 through 3. It preserves the donor's:

- `OAM\0` magic, 64-byte file header, and 64-byte section records;
- Config, Weights, State, Optimizer, and Progress section identities;
- 4096-byte first-payload alignment and donor tensor-section padding;
- 216-byte tensor records with dense, Q4, and Q8 metadata;
- complete scalar-type numbering, rank-eight bound, and fixed name widths;
- FNV-1a payload hashes, v2/v3 manifest hash, and v1 XOR checksum;
- exact file/range/overlap/duplicate/shape/encoding validation;
- Adam/AdamW/SGD two-moment and Muon one-moment payload parsing;
- sibling temporary-file write, file synchronization, atomic rename on Unix,
  and parent-directory synchronization.

Malformed or integrity-invalid files return
`ErrorKind::CheckpointCorrupt`. A structurally valid artifact that cannot bind
to the requested Rust owner—for example, a Q4 inference weight passed to dense
training restore—returns a normal contract error instead.

## Checkpoint mapping

```text
Module registration path                 .oam section
trainable Parameter                      Weights
persistent registered buffer             State
registered `u32` scalar module state      State as a rank-zero dense tensor
non-persistent registered buffer         excluded
Adam/AdamW first/second moments          Optimizer, flattened in parameter order
Muon momentum                            Optimizer first-state array
no-momentum SGD                          Optimizer, empty state arrays
optimizer step and learning rate         Optimizer + Progress
```

Loading validates the entire file, destination paths, shapes, dtypes, engine
ownership, optimizer kind, step identity, and all allocations before replacing live values.
Parameters retain their stable handles and advance mutation versions. Named
persistent buffers and registered scalar state retain stable registry handles.
Non-persistent buffers are not read or overwritten. Scalar state uses the same
dotted module paths and wire tensor validation as numerical state; it does not
create a Rust-only sidecar format.

## Evidence

`test/rs/ml/test_model_file.rs` proves:

- exact v3 header, section count, size, and alignment/padding behavior;
- replacement of an existing artifact through the atomic save path;
- parameter and persistent-state restoration with non-persistent exclusion;
- exact `VectorQuantizer` EMA-counter restoration and equality of the next
  dead-code revival transition against uninterrupted execution;
- payload bit-flip rejection as `CheckpointCorrupt`;
- v1 legacy and v2 manifest checksum compatibility;
- OA C++ `modelctl verify` accepting a Rust-written artifact; and
- Rust structurally accepting a C++-rewritten Q4 inference artifact before the
  dense-training binding rejects its encoding;
- Adam, AdamW, and no-momentum SGD fresh-owner restoration, with live momentum
  SGD rejected instead of silently omitting state; and
- best/latest selection, bounded incremental rotation, filename/progress step
  agreement, and callback-driven restore-best behavior.

Run the cross-language proof with:

```bash
OA_CPP_MODELCTL=/path/to/oa/bin/release/sdk/apps/ml/modelctl \
  cargo test --all-features --test ml \
  model_file::native_oam_roundtrip_integrity_and_legacy_checksums \
  -- --ignored --exact --test-threads=1
```

## Remaining work

The public general `ModelFile` object, architecture-specific config access,
quantization creation/loading, model translators, Windows
durable replacement evidence, and serialized
execution-plan/RNG state remain Planned. These extend this codec; they must not
introduce another `.oam` implementation.
