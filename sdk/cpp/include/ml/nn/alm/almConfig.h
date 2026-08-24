#pragma once

// oa::Alm configuration — two-stage pipeline config structs (MotionGPT lineage).
// stage 1: VQ-VAE tokenizer (motion to discrete tokens).
// stage 2: Autoregressive LM (token generation).

#include <oa/core.h>

namespace oa {

// Dataset configuration
struct AlmDatasetConfig {
	oa::String dataDir;
	oa::String split   = "train";
	oa::String corpus  = "cmp";  // cmp | humanml3d | kit
	oa::I32    maxClips = 0;     // 0 = all clips

	static AlmDatasetConfig fromEnv();
};

// stage 1: VQ-VAE tokenizer config.
// Faithful temporal-Conv1d VQ-VAE (T2M-GPT / MotionGPT). See AlmTokenizerAg.h.
struct AlmTokenizerConfig {
	// architecture (conv backbone)
	oa::I32 inputDim    = 263;    // HumanML3D feature dim (root 4 + joints 21*3*3 + foot 21*4 = 263)
	oa::I32 width       = 512;    // Conv channel width (reference 512)
	oa::I32 codeDim     = 512;    // codebook code dimension (reference 512)
	oa::I32 numCodes    = 512;    // K codes per level
	oa::I32 downT       = 2;      // Temporal downsample stages → factor 2^downT (=4×, reference)
	oa::I32 depth       = 3;      // residual conv blocks per stage (reference)

	// Quantization (reuses oa::ResidualVectorQuantizer, single level)
	// VQ-collapse-safe defaults — 0.02/0.999/0.0 collapsed the codebook to 1 code
	// (perplexity 1.0); these give perplexity ~130+ in real training.
	oa::F32 commitBeta  = 0.25F;  // Commitment loss weight (0.02 was too weak → collapse)
	oa::F32 emaDecay    = 0.99F;  // EMA decay for codebook (0.999 adapts too slowly)
	oa::F32 emaEps      = 1e-5F;
	oa::F32 deadThresh  = 2.0F;   // Dead-code revival threshold (0 disables revival entirely)

	// training
	oa::I32 batchSize    = 32;
	oa::I32 seqLen       = 64;    // training window in frames (must be a multiple of 2^downT)
	oa::F32 learningRate = 2e-4F;  // Reference lr
	oa::I32 numEpochs    = 2000;
};

enum class AlmFfnType : oa::U8 {
	Dense,
	Moe,
	Hybrid,
};

// stage 2: Autoregressive LM config
struct AlmPriorConfig {
	// vocabulary (computed from tokenizer)
	oa::I32 vocabSize   = 515;    // numCodes + 3 special tokens
	oa::I32 numCodes    = 512;    // motion codes (must match tokenizer)
	oa::I32 somToken    = 512;    // Start-of-motion [SOM]
	oa::I32 eomToken    = 513;    // End-of-motion [EOM]
	oa::I32 padToken    = 514;    // Padding [PAD]

	// architecture (decoder-only)
	oa::I32 dModel      = 384;    // Model dimension (reference: 384)
	oa::I32 numHeads    = 1;      // Attention heads; product config uses 6 (64/32 head dim)
	oa::I32 numLayers   = 6;      // Decoder layers (reference: 6)
	oa::I32 dFfn        = 1536;   // FFN hidden dimension (reference: 4*dModel)

	// Frozen text encoder contract. A precomputed caption feature [B,textFeatureDim]
	// is projected into one learned prefix token. Zero keeps the motion-only oracle.
	oa::I32 textFeatureDim = 0;

	// The permanent backbone is a causal Transformer. MoE is an FFN policy, not
	// another backbone: changing this keeps embeddings, attention, generation,
	// tokenization, and the output head identical. Hybrid replaces every
	// moeEvery-th block's dense FFN with oa::Moe (1-based layer numbering).
	AlmFfnType ffnType = AlmFfnType::Dense;
	oa::I32 moeNumExperts = 4;
	oa::I32 moeExpertsPerToken = 2;
	oa::I32 moeEvery = 2;
	oa::F32 moeBalanceRate = 0.0F;         // training policy; 0 disables
	oa::F32 moeAuxLossAlpha = 0.0F;        // training policy; 0 disables
	oa::F32 moeRouterZLossBeta = 0.0F;     // training policy; 0 disables

	// training
	oa::I32 batchSize   = 32;
	oa::I32 seqLen      = 128;    // Runtime training sequence length
	oa::I32 maxSeqLen   = 260;    // architecture: learned-position table size
	oa::F32 learningRate = 1e-4F;
	oa::I32 numEpochs   = 100;

	// generation
	oa::F32 temperature = 1.0F;   // Sampling temperature
	oa::I32 topK        = 0;      // top-k sampling (0 = disabled)
	oa::F32 topP        = 0.9F;   // Nucleus sampling
	oa::I32 maxGenLen   = 256;    // Max generated sequence length

	[[nodiscard]] bool usesMoe(oa::I32 inLayer) const {
		if (ffnType == AlmFfnType::Moe) return true;
		return ffnType == AlmFfnType::Hybrid and moeEvery > 0 and ((inLayer + 1) % moeEvery == 0);
	}

	// Sync vocab with tokenizer config
	void syncVocab(oa::I32 inNumCodes) {
		numCodes  = inNumCodes;
		vocabSize = numCodes + 3;
		somToken  = numCodes;
		eomToken  = numCodes + 1;
		padToken  = numCodes + 2;
	}
};

} // namespace oa
