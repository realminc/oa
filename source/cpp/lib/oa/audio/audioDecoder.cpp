// oa::FnAudio decode — miniaudio-backed implementation.
// miniaudio is a single-header C library (MIT).  We define the implementation
// in this one TU only — all other TUs just include the header (declarations).

#define MINIAUDIO_IMPLEMENTATION
#include <miniaudio.h>

#include <oa/audio/fnAudio.h>
#include <oa/core/fnMatrix.h>
#include <oa/runtime/engine.h>

#include <limits>

// ─── helpers ──────────────────────────────────────────────────────────────────

static oa::Result<oa::Audio> uploadToGpu(
	const oa::F32* inInterleaved,
	oa::U32        inChannelCount,
	oa::U64        inSampleCount,
	oa::U32        inSampleRate)
{
	if (inInterleaved == nullptr || inChannelCount == 0 || inSampleCount == 0 || inSampleRate == 0) {
		return oa::Status::invalidArgument("Empty audio data");
	}
	if (inSampleCount > std::numeric_limits<oa::Usize>::max() / inChannelCount ||
		inSampleCount > static_cast<oa::U64>(std::numeric_limits<oa::I64>::max())) {
		return oa::Status::invalidArgument("Audio data exceeds OA matrix limits");
	}

	// Deinterleave: LRLRLR... → [C, N] planar
	oa::Vec<oa::F32> planar;
	planar.resize(inChannelCount * inSampleCount);
	for (oa::U64 s = 0; s < inSampleCount; ++s) {
		for (oa::U32 c = 0; c < inChannelCount; ++c) {
			planar[c * inSampleCount + s] = inInterleaved[s * inChannelCount + c];
		}
	}

	oa::Matrix buffer = oa::FnMatrix::empty(
		oa::MatrixShape{static_cast<oa::I64>(inChannelCount), static_cast<oa::I64>(inSampleCount)},
		oa::ScalarType::Float32);
	oa::memcpy(buffer.dataAs<oa::F32>(), planar.data(), planar.size() * sizeof(oa::F32));
	return oa::Audio(
		oa::move(buffer), inSampleRate, oa::layoutForChannels(inChannelCount));
}

// ─── oa::FnAudio::DecodeFile ───────────────────────────────────────────────────

oa::Result<oa::Audio> oa::FnAudio::decodeFile(const oa::Path& inPath)
{
	if (inPath.empty()) {
		return oa::Status::invalidArgument("oa::FnAudio::decodeFile: input path is empty");
	}
	ma_decoder_config cfg = ma_decoder_config_init(ma_format_f32, 0, 0);
	ma_uint64 frameCount = 0;
	void* pcm = nullptr;
	const ma_result r = ma_decode_file(inPath.cStr(), &cfg, &frameCount, &pcm);
	if (r != MA_SUCCESS || pcm == nullptr || frameCount == 0) {
		if (pcm != nullptr) ma_free(pcm, nullptr);
		return oa::Status::error(oa::String("oa::FnAudio::decodeFile: failed to open '") + inPath.string() + "'");
	}
	auto result = uploadToGpu(
		static_cast<const oa::F32*>(pcm), cfg.channels, frameCount, cfg.sampleRate);
	ma_free(pcm, nullptr);
	return result;
}

// ─── oa::FnAudio::DecodeMemory ─────────────────────────────────────────────────

oa::Result<oa::Audio> oa::FnAudio::decodeMemory(oa::Span<const oa::U8> inData)
{
	if (inData.empty()) {
		return oa::Status::invalidArgument("oa::FnAudio::decodeMemory: input buffer is empty");
	}
	ma_decoder_config cfg = ma_decoder_config_init(ma_format_f32, 0, 0);
	ma_uint64 frameCount = 0;
	void* pcm = nullptr;
	const ma_result r = ma_decode_memory(
		inData.data(), inData.size(), &cfg, &frameCount, &pcm);
	if (r != MA_SUCCESS || pcm == nullptr || frameCount == 0) {
		if (pcm != nullptr) ma_free(pcm, nullptr);
		return oa::Status::error("oa::FnAudio::decodeMemory: failed to decode in-memory audio");
	}
	auto result = uploadToGpu(
		static_cast<const oa::F32*>(pcm), cfg.channels, frameCount, cfg.sampleRate);
	ma_free(pcm, nullptr);
	return result;
}
