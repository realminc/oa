// ML Training Utilities — shared helpers for all architectures
//
// LR schedule: see Ml/Train/LmLr.h (OaWarmupScheduler + OaCosineScheduler via OaLmLinearWarmupCosineScheduler).
// SampleBatch:   random batch sampling from raw byte data
// DeriveContext:  extract context name from data path

#pragma once

#include <Oa/Core/Types.h>
#include <Oa/Core/FileIo.h>

#include <random>

static inline void SampleBatch(
	OaSpan<const OaU8> InData, OaI32 InBatchSize, OaI32 InSeqLen,
	std::mt19937& InRng, OaVec<OaU32>& OutIndices, OaVec<OaU32>& OutTargets
) {
	const OaI32 T = InBatchSize * InSeqLen;
	OutIndices.Resize(T);
	OutTargets.Resize(T);
	OaI64 maxStart = static_cast<OaI64>(InData.size()) - InSeqLen - 1;
	std::uniform_int_distribution<OaI64> dist(0, maxStart);
	for (OaI32 b = 0; b < InBatchSize; ++b) {
		OaI64 start = dist(InRng);
		for (OaI32 s = 0; s < InSeqLen; ++s) {
			OaI32 idx = b * InSeqLen + s;
			OutIndices[idx] = InData[start + s];
			OutTargets[idx] = InData[start + s + 1];
		}
	}
}

static inline OaString DeriveContext(const OaString& InContext, const OaString& InDataPath) {
	if (!InContext.empty()) return InContext;
	return OaFileIo::GetStem(OaPath(InDataPath));
}
