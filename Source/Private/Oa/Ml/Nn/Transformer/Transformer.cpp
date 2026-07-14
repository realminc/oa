// OaTransformerBlock — pre-norm transformer block implementation

#include "Transformer.h"
#include <Oa/Core/FnMatrix.h>
#include <stdexcept>

OaTransformerBlock::OaTransformerBlock(OaI32 InDModel, OaI32 InDFF, OaI32 InSeqLen, OaF32 InEps)
	: DModel_(InDModel), DFF_(InDFF), SeqLen_(InSeqLen)
{
	Init(InDModel, InDFF, InSeqLen, 1, InEps);
}

OaTransformerBlock::OaTransformerBlock(OaI32 InDModel, OaI32 InDFF, OaI32 InSeqLen,
	OaI32 InNumHeads, OaF32 InEps)
	: DModel_(InDModel), DFF_(InDFF), SeqLen_(InSeqLen), NumHeads_(InNumHeads)
{
	Init(InDModel, InDFF, InSeqLen, InNumHeads, InEps);
}

OaTransformerBlock::OaTransformerBlock(OaI32 InDModel, OaI32 InExpertDFF, OaI32 InSeqLen,
	OaI32 InNumExperts, OaI32 InExpertsPerToken, OaF32 InEps)
	: DModel_(InDModel), DFF_(InExpertDFF), SeqLen_(InSeqLen)
{
	InitMoe(InDModel, InExpertDFF, InSeqLen, 1, InNumExperts, InExpertsPerToken, InEps);
}

OaTransformerBlock::OaTransformerBlock(OaI32 InDModel, OaI32 InExpertDFF, OaI32 InSeqLen,
	OaI32 InNumHeads, OaI32 InNumExperts, OaI32 InExpertsPerToken, OaF32 InEps)
	: DModel_(InDModel), DFF_(InExpertDFF), SeqLen_(InSeqLen), NumHeads_(InNumHeads)
{
	InitMoe(InDModel, InExpertDFF, InSeqLen, InNumHeads, InNumExperts, InExpertsPerToken, InEps);
}

void OaTransformerBlock::Init(OaI32 InDModel, OaI32 InDFF, OaI32 InSeqLen, OaF32 InEps) {
	Init(InDModel, InDFF, InSeqLen, 1, InEps);
}

void OaTransformerBlock::Init(OaI32 InDModel, OaI32 InDFF, OaI32 InSeqLen,
	OaI32 InNumHeads, OaF32 InEps) {
	DModel_ = InDModel;
	DFF_ = InDFF;
	SeqLen_ = InSeqLen;
	NumHeads_ = InNumHeads;

	InitAttention(InNumHeads, InEps);
	auto wd = OaFnMatrix::GetWeightDtype();

	LnFfn_ = OaMakeSharedPtr<OaLayerNorm>(DModel_, InEps);
	RegisterModule("ln_ffn", LnFfn_);

	Ffn1_ = OaMakeSharedPtr<OaLinear>(DModel_, DFF_);
	Ffn1_->Parameters()[0].Data = OaFnMatrix::RandXavier(OaMatrixShape{DFF_, DModel_}, wd);
	Ffn1_->SetActivation(OaActivation::Gelu);  // fused LinearGelu epilogue: Gelu(Linear(x)) in one dispatch
	RegisterModule("ffn1", Ffn1_);

	Ffn2_ = OaMakeSharedPtr<OaLinear>(DFF_, DModel_);
	Ffn2_->Parameters()[0].Data = OaFnMatrix::RandXavier(OaMatrixShape{DModel_, DFF_}, wd);
	RegisterModule("ffn2", Ffn2_);
}

void OaTransformerBlock::InitMoe(OaI32 InDModel, OaI32 InExpertDFF, OaI32 InSeqLen,
	OaI32 InNumExperts, OaI32 InExpertsPerToken, OaF32 InEps) {
	InitMoe(InDModel, InExpertDFF, InSeqLen, 1, InNumExperts, InExpertsPerToken, InEps);
}

void OaTransformerBlock::InitMoe(OaI32 InDModel, OaI32 InExpertDFF, OaI32 InSeqLen,
	OaI32 InNumHeads, OaI32 InNumExperts, OaI32 InExpertsPerToken, OaF32 InEps) {
	DModel_ = InDModel;
	DFF_ = InExpertDFF;
	SeqLen_ = InSeqLen;
	NumHeads_ = InNumHeads;
	InitAttention(InNumHeads, InEps);
	Moe_ = OaMakeSharedPtr<OaMoE>(DModel_, DFF_, InNumExperts, InExpertsPerToken, InEps);
	RegisterModule("moe", Moe_);
}

void OaTransformerBlock::InitAttention(OaI32 InNumHeads, OaF32 InEps) {
	if (DModel_ <= 0 or SeqLen_ <= 0 or InNumHeads <= 0 or DModel_ % InNumHeads != 0) {
		throw std::invalid_argument("OaTransformerBlock requires positive dimensions and DModel divisible by NumHeads");
	}
	LnAttn_ = OaMakeSharedPtr<OaLayerNorm>(DModel_, InEps);
	RegisterModule("ln_attn", LnAttn_);
	Attention_ = OaMakeSharedPtr<OaMultiHeadAttention>(DModel_, InNumHeads, 0.0F, true);
	Attention_->SetSeqLen(SeqLen_);
	RegisterModule("attention", Attention_);
}

OaMatrix OaTransformerBlock::Forward(const OaMatrix& InX) {
	const OaI32 n = static_cast<OaI32>(InX.Size(0));
	if (SeqLen_ <= 0 or n % SeqLen_ != 0) {
		throw std::invalid_argument("OaTransformerBlock input rows must be divisible by a positive sequence length");
	}
	// Pre-norm attention + residual.
	auto xn = LnAttn_->Forward(InX);
	auto x = OaFnMatrix::Add(InX, Attention_->Forward(xn));
	if (Moe_) return Moe_->Forward(x);

	// Pre-norm FFN + residual. Ffn1 carries the GELU activation (fused LinearGelu
	// epilogue), so Forward already returns gelu(Ffn1(x)) — no separate Gelu dispatch.
	auto fn = LnFfn_->Forward(x);
	auto h = Ffn1_->Forward(fn);
	x = OaFnMatrix::Add(x, Ffn2_->Forward(h));

	return x;
}

void OaTransformerBlock::SetSeqLen(OaI32 InSeqLen) {
	if (InSeqLen <= 0) {
		throw std::invalid_argument("OaTransformerBlock sequence length must be positive");
	}
	if (SeqLen_ == InSeqLen) return;
	SeqLen_ = InSeqLen;
	Attention_->SetSeqLen(InSeqLen);
}
