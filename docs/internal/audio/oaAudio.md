# OA Rust Audio

**Status:** Experimental complete OA Audio vertical slice

**Updated:** 2026-09-10

**OA donors:**

- `source/cpp/include/oa/audio/type.h`
- `source/cpp/include/oa/audio/fnAudio.h`
- `source/cpp/include/oa/audio/audioCapture.h`
- `source/cpp/include/oa/audio/audioEncoder.h`
- `source/cpp/include/oa/audio/audioPlayer.h`
- `source/cpp/lib/oa/audio/fnaudio/`
- `source/cpp/lib/oa/audio/shader/compute/`
- `source/cpp/lib/oa/audio/audioCapture.cpp`
- `source/cpp/lib/oa/audio/audioDecoder.cpp`
- `source/cpp/lib/oa/audio/audioEncoder.cpp`
- `source/cpp/lib/oa/audio/audioPlayer.cpp`

## Experimental implementation

`oa::Audio` is the sole whole-audio value. It composes checked Matrix storage
with semantic metadata:

```text
dtype = f32
shape = [channels, samples]
layout = planar
sample rate > 0
known speaker layout agrees with channel count
```

`Audio::as_matrix` is a zero-copy view. An Audio clone retains the same Matrix
storage, semantic value identity, and metadata. Audio inputs and outputs are
admitted to the semantic graph as `OpValueKind::Audio`; feature transforms
produce `Matrix` values rather than erasing the distinction at the API edge.

The shared operation schema owns all 17 donor contracts:

```text
normalize, resample, gain, clip, saturate, biquad, reverb, sos_filter,
pre_emphasis, to_mono, fade, mix, amplitude_to_db, waveform_envelope,
stft, mel_spectrogram, mfcc
```

Those operations lower to 24 generated physical kernel identities. Direct
signal operations use one dispatch; normalize, biquad, reverb, Mel, and MFCC
retain their multi-dispatch donor algorithms. The 14 donor Slang bodies are
ported under `src/slang/audio`; small direct kernels replace C++ compositions
through Matrix operations that OARS does not yet expose. This is an adaptation
at the lowering seam, not a CPU fallback.

## Codec and session boundaries

One-shot `audio::decode_file` and `audio::decode_memory` synchronously decode
WAV/PCM, FLAC, or MP3 on the CPU, convert to interleaved FP32 through
Symphonia, deinterleave, and upload through the caller's `Engine`.
`encode_interleaved_wav_f32`, `encode_wav_f32`, and `save_wav_f32` directly
port OA's checked IEEE-float WAV writer. Matrix observation and filesystem I/O
remain explicit blocking host boundaries.

There is deliberately no public `AudioDecoder` session. The donor exposes
one-shot decode functions and keeps its incremental decoder inside
`AudioPlayer`; Rust preserves that distinction. The private `codec/` split
separates one-shot decode and encode implementations without creating a public
`oa::audio::codec` namespace.

`AudioEncoder` is the stateful PCM-S16 packet session. It preserves donor
packet timing, explicit partial `flush`, and non-flushing `close`/Drop
behavior. Quantization silences NaN, saturates infinities and out-of-range
samples, and preserves asymmetric signed endpoints.

`AudioPlayer` incrementally decodes on a worker into a fixed-capacity,
lock-free SPSC frame ring. The device callback does not allocate or lock.
Generation-tagged frames make seek discard stale PCM without racing the
decoder. `close` is the result-bearing stop/join boundary; Drop schedules
retirement and never performs that work inline.

`AudioCapture` writes converted FP32 frames from the device callback into a
fixed-capacity, lock-free SPSC ring. Frame indices advance across dropped
input, and `poll` ends a chunk at each physical gap so a recorder can insert
the exact missing duration. Timestamps use a process-monotonic microsecond
epoch. `start`, `stop`, and `close` preserve the donor state transitions.

The donor uses miniaudio for codecs and device I/O. Rust uses two bounded safe
adapters:

- Symphonia owns WAV/FLAC/MP3 container and codec work.
- CPAL owns cross-platform input/output streams and native sample conversion;
  `rtrb` owns the real-time SPSC queues.

This replacement avoids adding an unsafe single-header C owner. The available
Rust miniaudio crate was rejected because its old bindgen dependency fails on
the current glibc headers. CPAL currently requires the default device to expose
the requested channel count and sample rate; unsupported configurations return
a backend-neutral OA capability error.

## Source and public paths

```text
src/rs/audio.rs              curated oa::audio facade
src/rs/audio/clip.rs         planar Audio value and channel metadata
src/rs/audio/codec.rs        private one-shot codec family facade
src/rs/audio/codec/decode.rs synchronous WAV/FLAC/MP3 decode
src/rs/audio/codec/encode.rs synchronous WAV-F32 encode/save
src/rs/audio/signal.rs       public signal operations and validation
src/rs/audio/transform.rs    public feature transforms and validation
src/rs/audio/lowering.rs     private semantic-to-physical dispatch assembly
src/rs/audio/encoder.rs      stateful PCM-S16 packet encoder
src/rs/audio/player.rs       incremental decode and output session
src/rs/audio/capture.rs      real-time input session
src/slang/audio/signal/      signal kernels
src/slang/audio/transform/   feature-transform kernels
```

`Audio`, `AudioCapture`, `AudioEncoder`, and `AudioPlayer` have one definition
in `oa::audio` and admitted root identity re-exports. Configuration, packet,
layout, and transform-support types remain module-owned. Stateless functions
remain only at `oa::audio::*`. Callers may locally write
`use oa::audio as oaa`; OA publishes no second alias module and no repeated
`audio/audio` layer.

## Evidence

The host Audio suite covers channel-layout inference, malformed input,
bit-exact WAV construction and round trip, Matrix metadata, PCM-S16 endpoints,
packet timing, partial flush, session defaults, and root type identity.

The Vulkan oracle suite covers every signal operation plus STFT, Mel, and MFCC
shape/finiteness contracts. The named run used Intel Iris Xe (TGL GT2), Mesa
26.2.2-arch1.1, and Vulkan 1.4.354. Device-session lifecycle tests passed on
the named workstation and remain ignored by default because they additionally
require matching default input/output hardware; compilation does not claim
that a CI host has those devices.

```bash
cargo test --test audio
cargo test --test audio audio::audio_signal_operations_match_host_oracles \
  -- --ignored --exact --test-threads=1
cargo test --test audio audio::audio_feature_transforms_have_checked_layouts \
  -- --ignored --exact --test-threads=1
cargo test --test audio audio::default_audio_player_runs_incremental_session_lifecycle \
  -- --ignored --exact --test-threads=1
cargo test --test audio audio::default_audio_capture_runs_explicit_session_lifecycle \
  -- --ignored --exact --test-threads=1
```

The optional real-file gate resolves `audio/oaNarration.{wav,flac,mp3}` below
`OA_DATA_DIR`; it checks WAV/FLAC parity and finite, non-silent MP3 decode.

## Planned

Hardware qualification across CPAL backends, named-device selection, device
hot-plug recovery, sample-rate/channel conversion when no exact device mode is
available, compressed streaming encode, and low-latency effect processing
remain Planned. The GPL-2.0 AudioNoise experiment remains research-only: its
effect-transition ideas may inform an independently specified CPU effect
session, while its global mono/48 kHz state and shell-based FFmpeg decode do
not fit this architecture.
