// FnAudio.cpp — generic OaFnAudio utilities (ToMatrix).
// Category manual implementations live next to their generated units, per the
// FnMatrix convention:
//   Signal/FnAudioSignal.cpp       (Normalize, Gain, Clip, AmplitudeToDb, ToMono, Fade)
//   Transform/FnAudioTransform.cpp (Stft, MelSpectrogram, Mfcc)
// There is no CPU fallback path — audio DSP is GPU-only like the rest of OA.

#include <Oa/Audio/FnAudio.h>
#include <Oa/Core/FnMatrix.h>

namespace OaFnAudio {

OaResult<OaMatrix> ToMatrix(const OaAudioBuffer& InBuf) {
	// Audio buffer is already an OaMatrix [Channels, Samples]
	// Reshape to [Channels, 1, Samples] for models expecting 3D input
	const auto& shape = InBuf.GetShape();
	if (shape.Rank != 2) {
		return OaStatus::InvalidArgument("OaFnAudio::ToMatrix: expected 2D buffer");
	}

	return OaFnMatrix::Reshape(InBuf, OaMatrixShape{shape[0], 1, shape[1]});
}

} // namespace OaFnAudio
