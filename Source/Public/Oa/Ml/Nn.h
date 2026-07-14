// OA ML - Neural Network Layers
//
// Standard building blocks, all inheriting from OaModule.
// PyTorch torch.nn equivalent with OA naming.
//
// Layers dispatch to Vulkan compute. No CPU fallbacks.

#pragma once

#include <Oa/Ml/FnMatrix.h>
#include <Oa/Ml/Module.h>

// ENUMS (needed by generated layers)

enum class OaActivation : OaU8 {
	None,
	Relu,
	Gelu,
};

enum class OaUpsampleMode : OaU8 { Nearest, Bilinear };

// GENERATED LAYERS
// Regenerate via: python3 Tools/NnAutogen/oannautogen.py --live
#include "../../../Private/Oa/Ml/Nn/Nn.gen.h"

// MANUAL LAYERS (not yet generated)

// MOE (Mixture of Experts)
#include "../../../Private/Oa/Ml/Nn/Moe/Moe.h"

// FFN (Feed-Forward Network with SwiGLU)
#include "../../../Private/Oa/Ml/Nn/Ffn/Ffn.h"

// GRU (gated recurrent unit) cell + stacked sequence module
#include "../../../Private/Oa/Ml/Nn/Gru/Gru.h"

// RNN (vanilla Elman) cell + stacked sequence module
#include "../../../Private/Oa/Ml/Nn/Rnn/Rnn.h"

// MAMBA-3 (Selective State Space Model)
#include "../../../Private/Oa/Ml/Nn/Mamba3/Mamba3.h"

// TRANSFORMER — pre-norm transformer block with causal self-attention
#include "../../../Private/Oa/Ml/Nn/Transformer/Transformer.h"

// EMPYREALM CORE — high-utilization sequential backbone
// (byte + mixer + flat residual pattern, designed for subclassing/specialization
//  across text, motion sequences, audio, etc.)
#include "../../../Private/Oa/Ml/Nn/Empyrealm/EmpyrealmCore.h"

// VECTOR QUANTIZATION (VQ-VAE / RVQ discrete bottleneck) — modality-agnostic
// tokenizer front-end for discrete-token generation (MoMask / T2M-GPT). EMA
// codebook, checkpointed via Buffers().
#include "../../../Private/Oa/Ml/Nn/Vq/Vq.h"

// Legacy sequence/conv VQ tokenizer experiments are surfaced via <Ml/Nn.h>.

// OaAlm (the skeletal-motion pipeline: Conv1d VQ-VAE tokenizer + AR token LM)
// and its PoseClip codec live in Extensions/Public/Ml/Nn/Alm/ and are surfaced
// via <Ml/Nn.h> — not from core, since they're a concrete model line.

// DROPOUT

class OaDropout : public OaModule {
public:
	explicit OaDropout(OaF32 P = 0.1f) : P_(P) {}
	OaMatrix Forward(const OaMatrix& InInput) override;
private:
	OaF32 P_;
};

// ROTARY POSITION EMBEDDING (RoPE)

class OaRoPE : public OaModule {
public:
	OaRoPE(OaI32 InNumHeads, OaI32 InHeadDim, OaF32 InThetaBase = 10000.0f);
	OaMatrix Forward(const OaMatrix& InInput) override;
private:
	OaI32 NumHeads_;
	OaI32 HeadDim_;
	OaF32 ThetaBase_;
};

// STATE SPACE MODEL

class OaSsmModule : public OaModule {
public:
	OaSsmModule(OaI32 InDModel, OaI32 InNumHeads, OaI32 InStateSize, OaI32 InSeqLen, OaI32 InBatch = 1);

	OaMatrix Forward(const OaMatrix& InInput) override;

private:
	OaI32 DModel_;
	OaI32 NumHeads_;
	OaI32 StateSize_;
	OaI32 SeqLen_;
	OaI32 Batch_;
	OaI32 ChannelsPerHead_;
	OaI32 InnerDim_;

	OaSharedPtr<OaLinear> InProj_;
	OaSharedPtr<OaLinear> BProj_;
	OaSharedPtr<OaLinear> CProj_;
	OaSharedPtr<OaLinear> DtProj_;
	OaSharedPtr<OaLinear> GateProj_;
	OaSharedPtr<OaLinear> OutProj_;
};
