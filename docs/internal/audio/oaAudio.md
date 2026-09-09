# OA Rust Audio

**Status:** Experimental host codec/value checkpoint

**OA donor:** `source/cpp/include/oa/audio/type.h`,
`source/cpp/lib/oa/audio/audioDecoder.cpp`, and
`source/cpp/lib/oa/audio/audioEncoder.cpp`

## Implemented contract

`oa::Audio` is the sole whole-audio value. It composes an existing Matrix with:

```text
dtype = f32
shape = [channels, samples]
layout = planar
sample rate > 0
known speaker layout agrees with channel count
```

`Audio::as_matrix` is a zero-copy view. Cloning an Audio retains the same
Matrix storage and metadata. Three raw channels infer `Unknown`, because a
count alone cannot distinguish 2.1 L/R/LFE from 3.0 L/C/R.

`audio::decode_file` and `audio::decode_memory` synchronously decode WAV/PCM,
FLAC, or MP3 on the CPU, convert to interleaved FP32 through Symphonia,
deinterleave, and upload through the caller's `Engine`. The third-party codec
error remains private and is translated into `core::Error`.

The OA donor uses miniaudio. Rust deliberately uses the feature-bounded
Symphonia Cargo dependency instead of copying a single-header C implementation
or adding a new unsafe FFI owner. This is a Rust adapter replacement at the
third-party codec seam, not a replacement for an OA numerical algorithm.

`audio::encode_interleaved_wav_f32` directly ports OA's checked 46-byte
IEEE-float WAV writer. `audio::encode_wav_f32` and `audio::save_wav_f32` are
explicit blocking host-observation boundaries: they complete/read the planar
Matrix, interleave it, and encode or write the result.

## Evidence

The external Audio suite covers channel-layout inference, WAV header and
payload bits, malformed inputs, Matrix dtype/rank/layout validation, planar
deinterleave, metadata, and a bit-exact synthetic WAV round trip.

The optional real-file gate resolves `audio/oaNarration.{wav,flac,mp3}` beneath
`OA_DATA_DIR`. It checks WAV/FLAC sample parity and finite, non-silent MP3
decode. The current named-device run used Intel Iris Xe (TGL GT2), Mesa
26.2.2-arch1.1, Vulkan 1.4.354.

```bash
cargo test --test audio
cargo test --test audio -- --ignored --test-threads=1
OA_DATA_DIR=/path/to/assets cargo test --test audio \
  real_wav_flac_and_mp3_assets_decode -- --ignored --test-threads=1
```

## Planned

The OA DSP operation schema, fourteen Slang kernels, and operation oracles are
not part of this checkpoint. They must enter through the shared generator and
semantic/executable graph rather than handwritten Matrix forwarding functions.
Capture, playback, streaming encode, and low-latency effect processing remain
stateful session work.

The GPL-2.0 AudioNoise experiment was audited as research, not imported as a
code donor. Its sample-at-a-time compressor, EQ, delay, modulation, parameter
smoothing, and wet/dry transition ideas may inform an independently specified
real-time CPU effect session. Its global mono/48 kHz state and FFmpeg shell
decode path are not OA's value or codec architecture.
