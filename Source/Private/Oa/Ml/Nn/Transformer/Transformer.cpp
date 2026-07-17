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
	Attention_->SetMode(AttentionMode_);
	RegisterModule("attention", Attention_);
}

OaMatrix OaTransformerBlock::Forward(const OaMatrix& InX) {
	return ForwardImpl(InX, nullptr);
}

OaMatrix OaTransformerBlock::ForwardMasked(
	const OaMatrix& InX, const OaMatrix& InAdditiveMask) {
	return ForwardImpl(InX, &InAdditiveMask);
}

void OaTransformerBlock::EnableAdaptiveConditioning(OaI32 InConditionDim) {
	if (InConditionDim <= 0) {
		throw std::invalid_argument(
			"OaTransformerBlock adaptive condition dimension must be positive");
	}
	if (AdaptiveModulation_) {
		if (ConditionDim_ != InConditionDim) {
			throw std::invalid_argument(
				"OaTransformerBlock adaptive conditioning is already configured with a different dimension");
		}
		return;
	}
	ConditionDim_ = InConditionDim;
	AdaptiveModulation_ = OaMakeSharedPtr<OaLinear>(ConditionDim_, 6 * DModel_);
	// AdaLN-Zero starts as the identity block. Gates and scale/shift learn from
	// the time/class/text context without perturbing an untrained residual path.
	auto& parameters = AdaptiveModulation_->Parameters();
	parameters[0].Data = OaFnMatrix::Zeros(
		parameters[0].Data.GetShape(), parameters[0].Data.GetDtype());
	parameters[1].Data = OaFnMatrix::Zeros(
		parameters[1].Data.GetShape(), parameters[1].Data.GetDtype());
	parameters[0].Data.SetRequiresGrad(true);
	parameters[1].Data.SetRequiresGrad(true);
	RegisterModule("adaptive_modulation", AdaptiveModulation_);
}

OaMatrix OaTransformerBlock::ForwardConditioned(
	const OaMatrix& InX,
	const OaMatrix& InCondition,
	const OaMatrix& InAdditiveMask) {
	if (!AdaptiveModulation_) {
		throw std::invalid_argument(
			"OaTransformerBlock adaptive conditioning must be enabled before ForwardConditioned");
	}
	const OaI64 rows = InX.Size(0);
	if (InX.Rank() != 2 || InX.Size(1) != DModel_ || SeqLen_ <= 0
		|| rows % SeqLen_ != 0 || InCondition.Rank() != 2
		|| InCondition.Size(0) != rows / SeqLen_
		|| InCondition.Size(1) != ConditionDim_) {
		throw std::invalid_argument(
			"OaTransformerBlock conditioned input must be [B*S,D] with condition [B,C]");
	}
	auto modulation = AdaptiveModulation_->Forward(InCondition);
	modulation = OaFnMatrix::RepeatInterleave(modulation, SeqLen_, 0);
	OaI64 sizes[] = {DModel_, DModel_, DModel_, DModel_, DModel_, DModel_};
	auto parts = OaFnMatrix::Split(modulation, OaSpan<OaI64>(sizes, 6), 1);

	auto attnNorm = LnAttn_->Forward(InX);
	attnNorm = attnNorm * (parts[1] + 1.0F) + parts[0];
	auto attnDelta = InAdditiveMask.IsEmpty()
		? Attention_->Forward(attnNorm)
		: Attention_->ForwardMasked(attnNorm, InAdditiveMask);
	auto x = InX + attnDelta * parts[2];

	auto ffnNorm = LnFfn_ ? LnFfn_->Forward(x) : x;
	ffnNorm = ffnNorm * (parts[4] + 1.0F) + parts[3];
	OaMatrix ffnDelta;
	if (Moe_) {
		ffnDelta = Moe_->Forward(ffnNorm);
	} else {
		ffnDelta = Ffn2_->Forward(Ffn1_->Forward(ffnNorm));
	}
	return x + ffnDelta * parts[5];
}

OaMatrix OaTransformerBlock::ForwardImpl(
	const OaMatrix& InX, const OaMatrix* InAdditiveMask) {
	const OaI32 n = static_cast<OaI32>(InX.Size(0));
	if (SeqLen_ <= 0 or n % SeqLen_ != 0) {
		throw std::invalid_argument("OaTransformerBlock input rows must be divisible by a positive sequence length");
	}
	// Pre-norm attention + residual.
	auto xn = LnAttn_->Forward(InX);
	auto attention = InAdditiveMask
		? Attention_->ForwardMasked(xn, *InAdditiveMask)
		: Attention_->Forward(xn);
	auto x = OaFnMatrix::Add(InX, attention);
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

void OaTransformerBlock::SetAttentionMode(OaAttentionMode InMode) {
	if (AttentionMode_ == InMode) return;
	AttentionMode_ = InMode;
	if (Attention_) Attention_->SetMode(InMode);
}
