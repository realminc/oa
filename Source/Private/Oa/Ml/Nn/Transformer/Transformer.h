// OaTransformerBlock — pre-norm transformer block (Level 1 API)
//
// Single-block architecture:
//   x → [residual] CausalSelfAttention(LN(x)) → dense FFN or MoE
//
// Attention: the shared OaMultiHeadAttention module. FFN: Linear→GELU→Linear.
// The legacy constructors retain one-head behavior; explicit-head overloads are
// the product path used by ALM and future Transformer models.

#pragma once

#include <Oa/Ml/Module.h>
#include <Oa/Ml/Nn.h>

class OaTransformerBlock : public OaModule {
public:
	OaTransformerBlock() = default;
	OaTransformerBlock(OaI32 InDModel, OaI32 InDFF, OaI32 InSeqLen, OaF32 InEps = 1e-5f);
	OaTransformerBlock(OaI32 InDModel, OaI32 InDFF, OaI32 InSeqLen,
		OaI32 InNumHeads, OaF32 InEps);
	OaTransformerBlock(OaI32 InDModel, OaI32 InExpertDFF, OaI32 InSeqLen,
		OaI32 InNumExperts, OaI32 InExpertsPerToken, OaF32 InEps = 1e-5f);
	OaTransformerBlock(OaI32 InDModel, OaI32 InExpertDFF, OaI32 InSeqLen,
		OaI32 InNumHeads, OaI32 InNumExperts, OaI32 InExpertsPerToken, OaF32 InEps);

	void Init(OaI32 InDModel, OaI32 InDFF, OaI32 InSeqLen, OaF32 InEps = 1e-5f);
	void Init(OaI32 InDModel, OaI32 InDFF, OaI32 InSeqLen,
		OaI32 InNumHeads, OaF32 InEps);
	void InitMoe(OaI32 InDModel, OaI32 InExpertDFF, OaI32 InSeqLen,
		OaI32 InNumExperts, OaI32 InExpertsPerToken, OaF32 InEps = 1e-5f);
	void InitMoe(OaI32 InDModel, OaI32 InExpertDFF, OaI32 InSeqLen,
		OaI32 InNumHeads, OaI32 InNumExperts, OaI32 InExpertsPerToken, OaF32 InEps);

	// Forward: x [B*S, D] → [B*S, D]. Block-diagonal causal mask uses InSeqLen
	// to separate sequences within the flattened batch.
	OaMatrix Forward(const OaMatrix& InX) override;

	[[nodiscard]] OaI32 DModel() const { return DModel_; }
	[[nodiscard]] OaI32 DFF() const { return DFF_; }
	[[nodiscard]] OaI32 SeqLen() const { return SeqLen_; }
	[[nodiscard]] OaI32 NumHeads() const { return NumHeads_; }
	[[nodiscard]] bool IsMoe() const { return static_cast<bool>(Moe_); }
	[[nodiscard]] OaMoE* Moe() { return Moe_.Get(); }
	[[nodiscard]] const OaMoE* Moe() const { return Moe_.Get(); }

	// The weights are sequence-length independent. Updating the runtime length
	// only changes the B/S view and its causal mask, which lets one block serve
	// fixed-length training and growing-prefix autoregressive generation.
	void SetSeqLen(OaI32 InSeqLen);

private:
	OaI32 DModel_ = 0;
	OaI32 DFF_ = 0;
	OaI32 SeqLen_ = 0;
	OaI32 NumHeads_ = 1;

	OaSharedPtr<OaLayerNorm> LnAttn_;
	OaSharedPtr<OaMultiHeadAttention> Attention_;
	OaSharedPtr<OaLayerNorm> LnFfn_;
	OaSharedPtr<OaLinear> Ffn1_;
	OaSharedPtr<OaLinear> Ffn2_;
	OaSharedPtr<OaMoE> Moe_;

	void InitAttention(OaI32 InNumHeads, OaF32 InEps);
};
