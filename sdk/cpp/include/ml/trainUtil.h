// ML training Utilities — shared helpers for all architectures
//
// LR schedule: see Ml/Train/lmLr.h (oa::WarmupScheduler + oa::CosineScheduler via OaLmLinearWarmupCosineScheduler).
// SampleBatch:   random batch sampling from raw byte data
// DeriveContext:  extract context name from data path

#pragma once

#include <oa/core/types.h>
#include <oa/core/filesystem.h>

#include <random>

static inline void sampleBatch(
	oa::Span<const oa::U8> inData, oa::I32 inBatchSize, oa::I32 inSeqLen,
	std::mt19937& inRng, oa::Vec<oa::U32>& outIndices, oa::Vec<oa::U32>& outTargets
) {
	const oa::I32 T = inBatchSize * inSeqLen;
	outIndices.resize(T);
	outTargets.resize(T);
	oa::I64 maxStart = static_cast<oa::I64>(inData.size()) - inSeqLen - 1;
	std::uniform_int_distribution<oa::I64> dist(0, maxStart);
	for (oa::I32 b = 0; b < inBatchSize; ++b) {
		oa::I64 start = dist(inRng);
		for (oa::I32 s = 0; s < inSeqLen; ++s) {
			oa::I32 idx = b * inSeqLen + s;
			outIndices[idx] = inData[start + s];
			outTargets[idx] = inData[start + s + 1];
		}
	}
}

static inline oa::String deriveContext(const oa::String& inContext, const oa::String& inDataPath) {
	if (!inContext.empty()) return inContext;
	return oa::Path(inDataPath).stem().string();
}
