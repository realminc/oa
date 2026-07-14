#pragma once

// OaAlm configuration — two-stage pipeline config structs (MotionGPT lineage).
// Stage 1: VQ-VAE tokenizer (motion to discrete tokens).
// Stage 2: Autoregressive LM (token generation).

#include <Oa/Core.h>

// Dataset configuration
struct OaAlmDatasetConfig {
	OaString DataDir;
	OaString Split   = "train";
	OaString Corpus  = "cmp";  // cmp | humanml3d | kit
	OaI32    MaxClips = 0;     // 0 = all clips

	static OaAlmDatasetConfig FromEnv();
};

// Stage 1: VQ-VAE tokenizer config.
// Faithful temporal-Conv1d VQ-VAE (T2M-GPT / MotionGPT). See AlmTokenizerAg.h.
struct OaAlmTokenizerConfig {
	// Architecture (conv backbone)
	OaI32 InputDim    = 263;    // HumanML3D feature dim (root 4 + joints 21*3*3 + foot 21*4 = 263)
	OaI32 Width       = 512;    // Conv channel width (reference 512)
	OaI32 CodeDim     = 512;    // Codebook code dimension (reference 512)
	OaI32 NumCodes    = 512;    // K codes per level
	OaI32 DownT       = 2;      // Temporal downsample stages → factor 2^DownT (=4×, reference)
	OaI32 Depth       = 3;      // Residual conv blocks per stage (reference)

	// Quantization (reuses OaResidualVectorQuantizer, single level)
	// VQ-collapse-safe defaults — 0.02/0.999/0.0 collapsed the codebook to 1 code
	// (perplexity 1.0); these give perplexity ~130+ in real training.
	OaF32 CommitBeta  = 0.25F;  // Commitment loss weight (0.02 was too weak → collapse)
	OaF32 EmaDecay    = 0.99F;  // EMA decay for codebook (0.999 adapts too slowly)
	OaF32 EmaEps      = 1e-5F;
	OaF32 DeadThresh  = 2.0F;   // Dead-code revival threshold (0 disables revival entirely)

	// Training
	OaI32 BatchSize    = 32;
	OaI32 SeqLen       = 64;    // Training window in frames (must be a multiple of 2^DownT)
	OaF32 LearningRate = 2e-4F;  // Reference lr
	OaI32 NumEpochs    = 2000;
};

enum class OaAlmFfnType : OaU8 {
	Dense,
	Moe,
	Hybrid,
};

// Stage 2: Autoregressive LM config
struct OaAlmPriorConfig {
	// Vocabulary (computed from tokenizer)
	OaI32 VocabSize   = 515;    // NumCodes + 3 special tokens
	OaI32 NumCodes    = 512;    // Motion codes (must match tokenizer)
	OaI32 SomToken    = 512;    // Start-of-motion [SOM]
	OaI32 EomToken    = 513;    // End-of-motion [EOM]
	OaI32 PadToken    = 514;    // Padding [PAD]

	// Architecture (decoder-only)
	OaI32 DModel      = 384;    // Model dimension (reference: 384)
	OaI32 NumHeads    = 1;      // Attention heads; product config uses 6 (64/32 head dim)
	OaI32 NumLayers   = 6;      // Decoder layers (reference: 6)
	OaI32 DFfn        = 1536;   // FFN hidden dimension (reference: 4*DModel)

	// Frozen text encoder contract. A precomputed caption feature [B,TextFeatureDim]
	// is projected into one learned prefix token. Zero keeps the motion-only oracle.
	OaI32 TextFeatureDim = 0;

	// The permanent backbone is a causal Transformer. MoE is an FFN policy, not
	// another backbone: changing this keeps embeddings, attention, generation,
	// tokenization, and the output head identical. Hybrid replaces every
	// MoeEvery-th block's dense FFN with OaMoE (1-based layer numbering).
	OaAlmFfnType FfnType = OaAlmFfnType::Dense;
	OaI32 MoeNumExperts = 4;
	OaI32 MoeExpertsPerToken = 2;
	OaI32 MoeEvery = 2;
	OaF32 MoeBalanceRate = 0.0F;         // training policy; 0 disables
	OaF32 MoeAuxLossAlpha = 0.0F;        // training policy; 0 disables
	OaF32 MoeRouterZLossBeta = 0.0F;     // training policy; 0 disables

	// Training
	OaI32 BatchSize   = 32;
	OaI32 SeqLen      = 128;    // Runtime training sequence length
	OaI32 MaxSeqLen   = 260;    // Architecture: learned-position table size
	OaF32 LearningRate = 1e-4F;
	OaI32 NumEpochs   = 100;

	// Generation
	OaF32 Temperature = 1.0F;   // Sampling temperature
	OaI32 TopK        = 0;      // Top-k sampling (0 = disabled)
	OaF32 TopP        = 0.9F;   // Nucleus sampling
	OaI32 MaxGenLen   = 256;    // Max generated sequence length

	[[nodiscard]] bool UsesMoe(OaI32 InLayer) const {
		if (FfnType == OaAlmFfnType::Moe) return true;
		return FfnType == OaAlmFfnType::Hybrid and MoeEvery > 0 and ((InLayer + 1) % MoeEvery == 0);
	}

	// Sync vocab with tokenizer config
	void SyncVocab(OaI32 InNumCodes) {
		NumCodes  = InNumCodes;
		VocabSize = NumCodes + 3;
		SomToken  = NumCodes;
		EomToken  = NumCodes + 1;
		PadToken  = NumCodes + 2;
	}
};
