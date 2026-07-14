// OA ML — VRAM budget & auto-tuning

#include <Oa/Ml/Config.h>
#include <algorithm>
#include <cstdio>
#include <cmath>

OaVRAMBudgetResult OaComputeVRAMBudget(
	OaUsize InModelParams,
	OaUsize InActivationBytesPerToken,
	OaVRAMBudgetConfig InConfig,
	OaDevice InDevice
) {
	OaVRAMBudgetResult result;

	// Query device memory budget (VRAM on GPU, RAM for VkCpu, none for VkOther / NPU-class).
	OaMemoryUsage mem = OaGetMemoryUsage(InDevice);
	if (mem.TotalBytes == 0) {
		// No GPU or query failed — return minimum viable config
		result.BatchSize = InConfig.MinBatchSize;
		result.SeqLen = InConfig.MinSeqLen;
		result.FitsInVRAM = false;
		return result;
	}

	// Apply safety margin
	OaUsize safetyBytes = static_cast<OaUsize>(static_cast<OaF64>(mem.FreeBytes) * InConfig.SafetyMarginPercent);
	result.AvailableBytes = mem.FreeBytes - safetyBytes;

	// Model memory: params * bytes_per_param * (weights + grads + optimizer_states)
	// For FP32 AdamW: 4 bytes * 4 copies (w, g, m, v) = 16 bytes per param
	result.ModelBytes = InModelParams * static_cast<OaUsize>(InConfig.BytesPerParam) * static_cast<OaUsize>(InConfig.OptimizerStatesPerParam);

	// Check if model itself fits
	if (result.ModelBytes >= result.AvailableBytes) {
		result.BatchSize = InConfig.MinBatchSize;
		result.SeqLen = InConfig.MinSeqLen;
		result.FitsInVRAM = false;
		result.TotalBytes = result.ModelBytes;
		return result;
	}

	// Remaining VRAM for activations
	OaUsize activationBudget = result.AvailableBytes - result.ModelBytes;

	// Max total tokens (B * S) that fit
	OaUsize maxTokens = (InActivationBytesPerToken > 0)
		? activationBudget / InActivationBytesPerToken
		: 0;

	if (maxTokens == 0) {
		result.BatchSize = InConfig.MinBatchSize;
		result.SeqLen = InConfig.MinSeqLen;
		result.FitsInVRAM = false;
		result.TotalBytes = result.ModelBytes;
		return result;
	}

	OaI32 B = 0;
	OaI32 S = 0;

	if (InConfig.PreferredBatchSize > 0 && InConfig.PreferredSeqLen > 0) {
		// Both specified — just validate
		B = InConfig.PreferredBatchSize;
		S = InConfig.PreferredSeqLen;
	} else if (InConfig.PreferredBatchSize > 0) {
		// Batch fixed, solve for seq len
		B = InConfig.PreferredBatchSize;
		S = static_cast<OaI32>(maxTokens / static_cast<OaUsize>(B));
		S = std::clamp(S, InConfig.MinSeqLen, InConfig.MaxSeqLen);
	} else if (InConfig.PreferredSeqLen > 0) {
		// Seq len fixed, solve for batch
		S = InConfig.PreferredSeqLen;
		B = static_cast<OaI32>(maxTokens / static_cast<OaUsize>(S));
		B = std::clamp(B, InConfig.MinBatchSize, InConfig.MaxBatchSize);
	} else {
		// Full auto: maximize S first (longer context = better quality), then B
		// Start with a reasonable S, then maximize B
		// Strategy: try S candidates from max down, find largest B >= MinBatchSize
		S = std::min(InConfig.MaxSeqLen, static_cast<OaI32>(maxTokens));
		S = std::max(S, InConfig.MinSeqLen);

		// Find best S where we can fit at least MinBatchSize
		while (S >= InConfig.MinSeqLen) {
			B = static_cast<OaI32>(maxTokens / static_cast<OaUsize>(S));
			if (B >= InConfig.MinBatchSize) break;
			S /= 2; // Halve S and try again
		}

		if (S < InConfig.MinSeqLen) {
			S = InConfig.MinSeqLen;
			B = static_cast<OaI32>(maxTokens / static_cast<OaUsize>(S));
		}

		B = std::clamp(B, InConfig.MinBatchSize, InConfig.MaxBatchSize);
		S = std::clamp(S, InConfig.MinSeqLen, InConfig.MaxSeqLen);
	}

	result.BatchSize = B;
	result.SeqLen = S;
	result.ActivationBytes = static_cast<OaUsize>(B) * static_cast<OaUsize>(S) * InActivationBytesPerToken;
	result.TotalBytes = result.ModelBytes + result.ActivationBytes;
	result.FitsInVRAM = (result.TotalBytes <= result.AvailableBytes);
	result.UtilizationPercent = (result.AvailableBytes > 0)
		? static_cast<OaF32>(100.0 * static_cast<OaF64>(result.TotalBytes) / static_cast<OaF64>(result.AvailableBytes))
		: 0.0f;

	return result;
}

void OaPrintVRAMBudget(const OaVRAMBudgetResult& InResult) {
	OaUsize modelMB = InResult.ModelBytes / (1024 * 1024);
	OaUsize activMB = InResult.ActivationBytes / (1024 * 1024);
	OaUsize totalMB = InResult.TotalBytes / (1024 * 1024);
	OaUsize availMB = InResult.AvailableBytes / (1024 * 1024);

	fprintf(stderr, "VRAM Budget\n");
	fprintf(stderr, "  Batch: %d  SeqLen: %d  Tokens/step: %d\n",
		InResult.BatchSize, InResult.SeqLen, InResult.BatchSize * InResult.SeqLen);
	fprintf(stderr, "  Model:       %zu MB  (weights + grads + optimizer)\n", modelMB);
	fprintf(stderr, "  Activations: %zu MB  (per-step buffers)\n", activMB);
	fprintf(stderr, "  Total:       %zu MB / %zu MB available (%.1f%%)\n",
		totalMB, availMB, static_cast<double>(InResult.UtilizationPercent));
	fprintf(stderr, "  Status:      %s\n",
		InResult.FitsInVRAM ? "FITS" : "DOES NOT FIT");
}
