# Audio example assets

`oaNarration.wav` is the canonical beginner audio input: 12.520 seconds of
24 kHz mono speech with the following OA-authored narration:

> Oa is a cross-vendor compute library built with C++ and Vulkan. One binary
> runs machine learning, vision, audio, and general compute across NVIDIA, AMD,
> and Intel GPUs.

The WAV retains a short natural pause after the narration while omitting the
original recording's long trailing silence, so delay-based examples produce an
audible effect tail. It is the source fixture. `oaNarration.flac` is its lossless FLAC
derivative, and `oaNarration.mp3` is a 64 kbit/s MP3 derivative. The three files
let examples use a meaningful input while the decoder gate still checks WAV,
FLAC, and MP3 against one canonical recording.

The complete byte counts and SHA-256 pins live in `sdk/asset/manifest.toml`.
