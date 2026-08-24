// oa::TransformerBlock — pre-norm transformer block implementation

#include <oa/ml/nn/transformer/transformer.h>
#include <oa/core/fnMatrix.h>
#include <stdexcept>

oa::TransformerBlock::TransformerBlock(oa::I32 inDModel, oa::I32 inDFF, oa::I32 inSeqLen, oa::F32 inEps)
	: dModel_(inDModel), dFF_(inDFF), seqLen_(inSeqLen)
{
	init(inDModel, inDFF, inSeqLen, 1, inEps);
}

oa::TransformerBlock::TransformerBlock(oa::I32 inDModel, oa::I32 inDFF, oa::I32 inSeqLen,
	oa::I32 inNumHeads, oa::F32 inEps)
	: dModel_(inDModel), dFF_(inDFF), seqLen_(inSeqLen), numHeads_(inNumHeads) {
	init(inDModel, inDFF, inSeqLen, inNumHeads, inEps);
}

oa::TransformerBlock::TransformerBlock(oa::I32 inDModel, oa::I32 inExpertDFF, oa::I32 inSeqLen,
	oa::I32 inNumExperts, oa::I32 inExpertsPerToken, oa::F32 inEps)
	: dModel_(inDModel), dFF_(inExpertDFF), seqLen_(inSeqLen)
{
	initMoe(inDModel, inExpertDFF, inSeqLen, 1, inNumExperts, inExpertsPerToken, inEps);
}

oa::TransformerBlock::TransformerBlock(oa::I32 inDModel, oa::I32 inExpertDFF, oa::I32 inSeqLen,
	oa::I32 inNumHeads, oa::I32 inNumExperts, oa::I32 inExpertsPerToken, oa::F32 inEps)
	: dModel_(inDModel), dFF_(inExpertDFF), seqLen_(inSeqLen), numHeads_(inNumHeads)
{
	initMoe(inDModel, inExpertDFF, inSeqLen, inNumHeads, inNumExperts, inExpertsPerToken, inEps);
}

void oa::TransformerBlock::init(oa::I32 inDModel, oa::I32 inDFF, oa::I32 inSeqLen, oa::F32 inEps) {
	init(inDModel, inDFF, inSeqLen, 1, inEps);
}

void oa::TransformerBlock::init(oa::I32 inDModel, oa::I32 inDFF, oa::I32 inSeqLen, oa::I32 inNumHeads, oa::F32 inEps) {
	dModel_ = inDModel;
	dFF_ = inDFF;
	seqLen_ = inSeqLen;
	numHeads_ = inNumHeads;

	initAttention(inNumHeads, inEps);
	auto wd = oa::FnMatrix::weightDtype();

	lnFfn_ = oa::makeShared<oa::LayerNorm>(dModel_, inEps);
	registerModule("ln_ffn", lnFfn_);

	ffn1_ = oa::makeShared<oa::Linear>(dModel_, dFF_);
	ffn1_->parameters()[0].data = oa::FnMatrix::randXavier(oa::MatrixShape{dFF_, dModel_}, wd);
	ffn1_->setActivation(oa::Activation::Gelu);  // fused LinearGelu epilogue: Gelu(Linear(x)) in one dispatch
	registerModule("ffn1", ffn1_);

	ffn2_ = oa::makeShared<oa::Linear>(dFF_, dModel_);
	ffn2_->parameters()[0].data = oa::FnMatrix::randXavier(oa::MatrixShape{dModel_, dFF_}, wd);
	registerModule("ffn2", ffn2_);
}

void oa::TransformerBlock::initMoe(oa::I32 inDModel, oa::I32 inExpertDFF, oa::I32 inSeqLen,	oa::I32 inNumExperts, oa::I32 inExpertsPerToken, oa::F32 inEps) {
	initMoe(inDModel, inExpertDFF, inSeqLen, 1, inNumExperts, inExpertsPerToken, inEps);
}

void oa::TransformerBlock::initMoe(oa::I32 inDModel, oa::I32 inExpertDFF, oa::I32 inSeqLen,	oa::I32 inNumHeads, oa::I32 inNumExperts, oa::I32 inExpertsPerToken, oa::F32 inEps) {
	dModel_ = inDModel;
	dFF_ = inExpertDFF;
	seqLen_ = inSeqLen;
	numHeads_ = inNumHeads;
	initAttention(inNumHeads, inEps);
	moe_ = oa::makeShared<oa::Moe>(dModel_, dFF_, inNumExperts, inExpertsPerToken, inEps);
	registerModule("moe", moe_);
}

void oa::TransformerBlock::initAttention(oa::I32 inNumHeads, oa::F32 inEps) {
	if (dModel_ <= 0 or seqLen_ <= 0 or inNumHeads <= 0 or dModel_ % inNumHeads != 0) {
		throw std::invalid_argument("oa::TransformerBlock requires positive dimensions and dModel divisible by numHeads");
	}
	lnAttn_ = oa::makeShared<oa::LayerNorm>(dModel_, inEps);
	registerModule("ln_attn", lnAttn_);
	attention_ = oa::makeShared<oa::MultiHeadAttention>(dModel_, inNumHeads, 0.0F, true);
	attention_->setSeqLen(seqLen_);
	attention_->setMode(attentionMode_);
	registerModule("attention", attention_);
}

oa::Matrix oa::TransformerBlock::forward(const oa::Matrix& inX) {
	return forwardImpl(inX, nullptr);
}

oa::Matrix oa::TransformerBlock::forwardMasked(const oa::Matrix& inX, const oa::Matrix& inAdditiveMask) {
	return forwardImpl(inX, &inAdditiveMask);
}

void oa::TransformerBlock::enableAdaptiveConditioning(oa::I32 inConditionDim) {
	if (inConditionDim <= 0) {
		throw std::invalid_argument("oa::TransformerBlock adaptive condition dimension must be positive");
	}
	if (adaptiveModulation_) {
		if (conditionDim_ != inConditionDim) {
			throw std::invalid_argument("oa::TransformerBlock adaptive conditioning is already configured with a different dimension");
		}
		return;
	}
	conditionDim_ = inConditionDim;
	adaptiveModulation_ = oa::makeShared<oa::Linear>(conditionDim_, 6 * dModel_);
	// AdaLN-Zero starts as the identity block. Gates and scale/shift learn from
	// the time/class/text context without perturbing an untrained residual path.
	auto& parameters = adaptiveModulation_->parameters();
	parameters[0].data = oa::FnMatrix::zeros(parameters[0].data.getShape(), parameters[0].data.getDtype());
	parameters[1].data = oa::FnMatrix::zeros(parameters[1].data.getShape(), parameters[1].data.getDtype());
	parameters[0].data.setRequiresGrad(true);
	parameters[1].data.setRequiresGrad(true);
	registerModule("adaptive_modulation", adaptiveModulation_);
}

oa::Matrix oa::TransformerBlock::forwardConditioned(const oa::Matrix& inX, const oa::Matrix& inCondition,	const oa::Matrix& inAdditiveMask) {
	if (!adaptiveModulation_) {
		throw std::invalid_argument("oa::TransformerBlock adaptive conditioning must be enabled before forwardConditioned");
	}
	const oa::I64 rows = inX.size(0);
	if (inX.rank() != 2 || inX.size(1) != dModel_ || seqLen_ <= 0
		|| rows % seqLen_ != 0 || inCondition.rank() != 2
		|| inCondition.size(0) != rows / seqLen_
		|| inCondition.size(1) != conditionDim_) {
		throw std::invalid_argument("oa::TransformerBlock conditioned input must be [B*S,D] with condition [B,C]");
	}
	auto modulation = adaptiveModulation_->forward(inCondition);
	modulation = oa::FnMatrix::repeatInterleave(modulation, seqLen_, 0);
	oa::I64 sizes[] = {dModel_, dModel_, dModel_, dModel_, dModel_, dModel_};
	auto parts = oa::FnMatrix::split(modulation, oa::Span<oa::I64>(sizes, 6), 1);

	auto attnNorm = lnAttn_->forward(inX);
	attnNorm = attnNorm * (parts[1] + 1.0F) + parts[0];
	auto attnDelta = inAdditiveMask.isEmpty()
		? attention_->forward(attnNorm)
		: attention_->forwardMasked(attnNorm, inAdditiveMask);
	auto x = inX + attnDelta * parts[2];

	auto ffnNorm = lnFfn_ ? lnFfn_->forward(x) : x;
	ffnNorm = ffnNorm * (parts[4] + 1.0F) + parts[3];
	oa::Matrix ffnDelta;
	if (moe_) {
		ffnDelta = moe_->forward(ffnNorm);
	} else {
		ffnDelta = ffn2_->forward(ffn1_->forward(ffnNorm));
	}
	return x + ffnDelta * parts[5];
}

oa::Matrix oa::TransformerBlock::forwardImpl(const oa::Matrix& inX, const oa::Matrix* inAdditiveMask) {
	const oa::I32 n = static_cast<oa::I32>(inX.size(0));
	if (seqLen_ <= 0 or n % seqLen_ != 0) {
		throw std::invalid_argument("oa::TransformerBlock input rows must be divisible by a positive sequence length");
	}
	// Pre-norm attention + residual.
	auto xn = lnAttn_->forward(inX);
	auto attention = inAdditiveMask
		? attention_->forwardMasked(xn, *inAdditiveMask)
		: attention_->forward(xn);
	auto x = oa::FnMatrix::add(inX, attention);
	if (moe_) return moe_->forward(x);

	// Pre-norm FFN + residual. Ffn1 carries the GELU activation (fused LinearGelu
	// epilogue), so forward already returns gelu(ffn1(x)) — no separate Gelu dispatch.
	auto fn = lnFfn_->forward(x);
	auto h = ffn1_->forward(fn);
	x = oa::FnMatrix::add(x, ffn2_->forward(h));

	return x;
}

void oa::TransformerBlock::setSeqLen(oa::I32 inSeqLen) {
	if (inSeqLen <= 0) {
		throw std::invalid_argument("oa::TransformerBlock sequence length must be positive");
	}

	if (seqLen_ == inSeqLen) { return; }
	seqLen_ = inSeqLen;
	attention_->setSeqLen(inSeqLen);
}

void oa::TransformerBlock::setAttentionMode(oa::AttentionMode inMode) {
	if (attentionMode_ == inMode) { return; }
	attentionMode_ = inMode;
	if (attention_) attention_->setMode(inMode);
}
