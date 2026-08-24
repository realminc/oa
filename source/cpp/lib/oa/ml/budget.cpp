// OA ML — VRAM budget & auto-tuning

#include <oa/ml/config.h>
#include <oa/runtime/engine.h>
#include <algorithm>
#include <cstdio>
#include <cmath>

oa::VramBudgetResult oa::computeVramBudget(
	const oa::Engine& inEngine,
	oa::Usize inModelParams,
	oa::Usize inActivationBytesPerToken,
	oa::VramBudgetConfig inConfig
) {
	oa::VramBudgetResult result;

	// Budgeting borrows the exact runtime owner; a logical device value cannot
	// identify allocator usage or distinguish two engines of the same type.
	oa::MemoryUsage mem = inEngine.getMemoryUsage();
	if (mem.totalBytes == 0) {
		// No GPU or query failed — return minimum viable config
		result.batchSize = inConfig.minBatchSize;
		result.seqLen = inConfig.minSeqLen;
		result.fitsInVRAM = false;
		return result;
	}

	// apply safety margin
	oa::Usize safetyBytes = static_cast<oa::Usize>(static_cast<oa::F64>(mem.freeBytes) * inConfig.safetyMarginPercent);
	result.availableBytes = mem.freeBytes - safetyBytes;

	// Model memory: params * bytes_per_param * (weights + grads + optimizer_states)
	// For FP32 AdamW: 4 bytes * 4 copies (w, g, m, v) = 16 bytes per param
	result.modelBytes = inModelParams * static_cast<oa::Usize>(inConfig.bytesPerParam) * static_cast<oa::Usize>(inConfig.optimizerStatesPerParam);

	// Check if model itself fits
	if (result.modelBytes >= result.availableBytes) {
		result.batchSize = inConfig.minBatchSize;
		result.seqLen = inConfig.minSeqLen;
		result.fitsInVRAM = false;
		result.totalBytes = result.modelBytes;
		return result;
	}

	// remaining VRAM for activations
	oa::Usize activationBudget = result.availableBytes - result.modelBytes;

	// Max total tokens (B * S) that fit
	oa::Usize maxTokens = (inActivationBytesPerToken > 0)
		? activationBudget / inActivationBytesPerToken
		: 0;

	if (maxTokens == 0) {
		result.batchSize = inConfig.minBatchSize;
		result.seqLen = inConfig.minSeqLen;
		result.fitsInVRAM = false;
		result.totalBytes = result.modelBytes;
		return result;
	}

	oa::I32 B = 0;
	oa::I32 S = 0;

	if (inConfig.preferredBatchSize > 0 && inConfig.preferredSeqLen > 0) {
		// Both specified — just validate
		B = inConfig.preferredBatchSize;
		S = inConfig.preferredSeqLen;
	} else if (inConfig.preferredBatchSize > 0) {
		// Batch fixed, solve for seq len
		B = inConfig.preferredBatchSize;
		S = static_cast<oa::I32>(maxTokens / static_cast<oa::Usize>(B));
		S = std::clamp(S, inConfig.minSeqLen, inConfig.maxSeqLen);
	} else if (inConfig.preferredSeqLen > 0) {
		// seq len fixed, solve for batch
		S = inConfig.preferredSeqLen;
		B = static_cast<oa::I32>(maxTokens / static_cast<oa::Usize>(S));
		B = std::clamp(B, inConfig.minBatchSize, inConfig.maxBatchSize);
	} else {
		// Full auto: maximize S first (longer context = better quality), then B
		// Start with a reasonable S, then maximize B
		// Strategy: try S candidates from max down, find largest B >= minBatchSize
		S = std::min(inConfig.maxSeqLen, static_cast<oa::I32>(maxTokens));
		S = std::max(S, inConfig.minSeqLen);

		// find best S where we can fit at least minBatchSize
		while (S >= inConfig.minSeqLen) {
			B = static_cast<oa::I32>(maxTokens / static_cast<oa::Usize>(S));
			if (B >= inConfig.minBatchSize) break;
			S /= 2; // Halve S and try again
		}

		if (S < inConfig.minSeqLen) {
			S = inConfig.minSeqLen;
			B = static_cast<oa::I32>(maxTokens / static_cast<oa::Usize>(S));
		}

		B = std::clamp(B, inConfig.minBatchSize, inConfig.maxBatchSize);
		S = std::clamp(S, inConfig.minSeqLen, inConfig.maxSeqLen);
	}

	result.batchSize = B;
	result.seqLen = S;
	result.activationBytes = static_cast<oa::Usize>(B) * static_cast<oa::Usize>(S) * inActivationBytesPerToken;
	result.totalBytes = result.modelBytes + result.activationBytes;
	result.fitsInVRAM = (result.totalBytes <= result.availableBytes);
	result.utilizationPercent = (result.availableBytes > 0)
		? static_cast<oa::F32>(100.0 * static_cast<oa::F64>(result.totalBytes) / static_cast<oa::F64>(result.availableBytes))
		: 0.0f;

	return result;
}

void oa::printVramBudget(const oa::VramBudgetResult& inResult) {
	oa::Usize modelMB = inResult.modelBytes / (1024 * 1024);
	oa::Usize activMB = inResult.activationBytes / (1024 * 1024);
	oa::Usize totalMB = inResult.totalBytes / (1024 * 1024);
	oa::Usize availMB = inResult.availableBytes / (1024 * 1024);

	fprintf(stderr, "VRAM budget\n");
	fprintf(stderr, "  Batch: %d  seqLen: %d  tokens/step: %d\n",
		inResult.batchSize, inResult.seqLen, inResult.batchSize * inResult.seqLen);
	fprintf(stderr, "  Model:       %zu MB  (weights + grads + optimizer)\n", modelMB);
	fprintf(stderr, "  Activations: %zu MB  (per-step buffers)\n", activMB);
	fprintf(stderr, "  total:       %zu MB / %zu MB available (%.1f%%)\n",
		totalMB, availMB, static_cast<double>(inResult.utilizationPercent));
	fprintf(stderr, "  status:      %s\n",
		inResult.fitsInVRAM ? "FITS" : "DOES NOT FIT");
}
