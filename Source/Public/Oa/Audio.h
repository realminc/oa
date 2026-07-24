// OaAudio — GPU-accelerated audio processing. Public umbrella.
//
// OaAudio        — semantic OaMatrix composition with rate and channel layout.
// OaAudioDecoder — CPU decode (WAV/FLAC/MP3 via miniaudio) → GPU upload.
// OaAudioEncoder — WAV-F32 and streaming PCM-S16 codec boundaries.
// OaAudioCapture — timestamped real-time F32 device input.
// OaFnAudio      — stateless DSP ops on the OaFnMatrix auto-context pattern.
//
// Typical ML pipeline:
//   auto dec = OaAudioDecoder::LoadFile("speech.wav").Unwrap();
//   auto mel = OaFnAudio::MelSpectrogram(dec);  // → OaMatrix
//   model.Forward(mel);

#pragma once

#include <Oa/Audio/Type.h>
#include <Oa/Audio/AudioDecoder.h>
#include <Oa/Audio/AudioEncoder.h>
#include <Oa/Audio/AudioCapture.h>
#include <Oa/Audio/AudioStream.h>
#include <Oa/Audio/FnAudio.h>
