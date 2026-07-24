// FnAudio.cpp — category ownership anchor.
// Category implementations live next to their schema-owned units, per the
// FnMatrix convention:
//   Signal/FnAudioSignal.cpp       (waveform-preserving and measurement ops)
//   Transform/FnAudioTransform.cpp (Stft, MelSpectrogram, Mfcc)
// There is no CPU fallback path — audio DSP is GPU-only like the rest of OA.

#include <Oa/Audio/FnAudio.h>
