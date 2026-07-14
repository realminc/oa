// OaAudioDecoder — miniaudio-backed decode implementation.
// miniaudio is a single-header C library (MIT).  We define the implementation
// in this one TU only — all other TUs just include the header (declarations).

#define MINIAUDIO_IMPLEMENTATION
#include <miniaudio.h>

#include <Oa/Audio/AudioDecoder.h>
#include <Oa/Core/FnMatrix.h>
#include <Oa/Runtime/Engine.h>

#include <limits>

// ─── helpers ──────────────────────────────────────────────────────────────────

static OaResult<OaAudioDecodeResult> UploadToGpu(
	const OaF32* InInterleaved,
	OaU32        InChannelCount,
	OaU64        InSampleCount,
	OaU32        InSampleRate)
{
	OaAudioDecodeResult res;
	res.SampleRate   = InSampleRate;
	res.ChannelCount = InChannelCount;
	res.SampleCount  = InSampleCount;

	if (InInterleaved == nullptr || InChannelCount == 0 || InSampleCount == 0 || InSampleRate == 0) {
		return OaStatus::InvalidArgument("Empty audio data");
	}
	if (InSampleCount > std::numeric_limits<OaUsize>::max() / InChannelCount ||
		InSampleCount > static_cast<OaU64>(std::numeric_limits<OaI64>::max())) {
		return OaStatus::InvalidArgument("Audio data exceeds OA matrix limits");
	}

	// Deinterleave: LRLRLR... → [C, N] planar
	OaVec<OaF32> planar;
	planar.Resize(InChannelCount * InSampleCount);
	for (OaU64 s = 0; s < InSampleCount; ++s) {
		for (OaU32 c = 0; c < InChannelCount; ++c) {
			planar[c * InSampleCount + s] = InInterleaved[s * InChannelCount + c];
		}
	}

	res.Buffer = OaFnMatrix::Empty(
		OaMatrixShape{static_cast<OaI64>(InChannelCount), static_cast<OaI64>(InSampleCount)},
		OaScalarType::Float32);
	OaMemcpy(res.Buffer.DataAs<OaF32>(), planar.Data(), planar.Size() * sizeof(OaF32));
	return res;
}

// ─── OaAudioDecoder::LoadFile ─────────────────────────────────────────────────

OaResult<OaAudioDecodeResult> OaAudioDecoder::LoadFile(const char* InPath)
{
	if (InPath == nullptr || InPath[0] == '\0') {
		return OaStatus::InvalidArgument("OaAudioDecoder: input path is empty");
	}
	ma_decoder_config cfg = ma_decoder_config_init(ma_format_f32, 0, 0);
	ma_uint64 frameCount = 0;
	void* pcm = nullptr;
	const ma_result r = ma_decode_file(InPath, &cfg, &frameCount, &pcm);
	if (r != MA_SUCCESS || pcm == nullptr || frameCount == 0) {
		if (pcm != nullptr) ma_free(pcm, nullptr);
		return OaStatus::Error(OaString("OaAudioDecoder: failed to open '") + InPath + "'");
	}
	auto result = UploadToGpu(
		static_cast<const OaF32*>(pcm), cfg.channels, frameCount, cfg.sampleRate);
	ma_free(pcm, nullptr);
	return result;
}

// ─── OaAudioDecoder::LoadMemory ───────────────────────────────────────────────

OaResult<OaAudioDecodeResult> OaAudioDecoder::LoadMemory(OaSpan<const OaU8> InData)
{
	if (InData.Empty()) {
		return OaStatus::InvalidArgument("OaAudioDecoder: input buffer is empty");
	}
	ma_decoder_config cfg = ma_decoder_config_init(ma_format_f32, 0, 0);
	ma_uint64 frameCount = 0;
	void* pcm = nullptr;
	const ma_result r = ma_decode_memory(
		InData.Data(), InData.Size(), &cfg, &frameCount, &pcm);
	if (r != MA_SUCCESS || pcm == nullptr || frameCount == 0) {
		if (pcm != nullptr) ma_free(pcm, nullptr);
		return OaStatus::Error("OaAudioDecoder: failed to decode in-memory audio");
	}
	auto result = UploadToGpu(
		static_cast<const OaF32*>(pcm), cfg.channels, frameCount, cfg.sampleRate);
	ma_free(pcm, nullptr);
	return result;
}

// WAV encode moved to the header-only OaAudioEncoder (<Oa/Audio/AudioEncoder.h>).
