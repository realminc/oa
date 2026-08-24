// oa::Audio — GPU-accelerated audio processing. Public umbrella.
//
// oa::Audio            — semantic matrix value with rate and channel layout.
// oa::FnAudio          — stateless decode, encode, DSP, and feature operations.
// oa::AudioEncoder       — stateful native PCM packetization session.
// oa::AudioCapture       — timestamped real-time F32 device input session.
// oa::AudioPlayer        — incremental decode and realtime playback session.
//
// Typical ML pipeline:
//   auto dec = oa::FnAudio::decodeFile("speech.wav").unwrap();
//   auto mel = oa::FnAudio::melSpectrogram(dec);  // → oa::Matrix
//   model.forward(mel);

#pragma once

#include <oa/audio/type.h>
#include <oa/audio/audioCapture.h>
#include <oa/audio/audioEncoder.h>
#include <oa/audio/audioPlayer.h>
#include <oa/audio/fnAudio.h>
