// oa::TransformerBlock — pre-norm transformer block implementation

#include <oa/ml/nn/transformer/transformer.h>
#include <oa/core/assert.h>
#include <oa/core/fnMatrix.h>
#include <oa/core/std/format.h>

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
	OA_REQUIRE_MSG(dModel_ > 0 && seqLen_ > 0 && inNumHeads > 0
		&& dModel_ % inNumHeads == 0,
		"oa::TransformerBlock requires positive dimensions and dModel divisible by numHeads");
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
	OA_REQUIRE_MSG(inConditionDim > 0,
		"oa::TransformerBlock adaptive condition dimension must be positive");
	if (adaptiveModulation_) {
		OA_REQUIRE_MSG(conditionDim_ == inConditionDim,
			"oa::TransformerBlock adaptive conditioning is already configured with a different dimension");
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
	OA_REQUIRE_MSG(adaptiveModulation_,
		"oa::TransformerBlock adaptive conditioning must be enabled before forwardConditioned");
	const oa::I64 rows = inX.size(0);
	OA_REQUIRE_MSG(inX.rank() == 2 && inX.size(1) == dModel_ && seqLen_ > 0
		&& rows % seqLen_ == 0 && inCondition.rank() == 2
		&& inCondition.size(0) == rows / seqLen_
		&& inCondition.size(1) == conditionDim_,
		"oa::TransformerBlock conditioned input must be [B*S,D] with condition [B,C]");
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
	OA_REQUIRE_MSG(seqLen_ > 0 && n % seqLen_ == 0,
		"oa::TransformerBlock input rows must be divisible by a positive sequence length");
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
	OA_REQUIRE_MSG(inSeqLen > 0,
		"oa::TransformerBlock sequence length must be positive");

	if (seqLen_ == inSeqLen) { return; }
	seqLen_ = inSeqLen;
	attention_->setSeqLen(inSeqLen);
}

void oa::TransformerBlock::setAttentionMode(oa::AttentionMode inMode) {
	if (attentionMode_ == inMode) { return; }
	attentionMode_ = inMode;
	if (attention_) attention_->setMode(inMode);
}

oa::NnTransformer::NnTransformer(
	oa::I32 inVocabSize,
	oa::I32 inContextLength,
	oa::I32 inModelWidth,
	oa::I32 inHiddenWidth,
	oa::I32 inNumLayers,
	oa::I32 inNumHeads,
	oa::F32 inEps)
	: vocabSize_(inVocabSize)
	, contextLength_(inContextLength)
	, modelWidth_(inModelWidth)
	, hiddenWidth_(inHiddenWidth)
	, numHeads_(inNumHeads) {
	OA_REQUIRE_MSG(vocabSize_ > 0 && contextLength_ > 0 && modelWidth_ > 0
		&& hiddenWidth_ > 0 && inNumLayers > 0 && numHeads_ > 0
		&& modelWidth_ % numHeads_ == 0,
		"oa::NnTransformer requires positive dimensions and modelWidth divisible by numHeads");

	tokenEmbedding_ = oa::makeShared<oa::Embedding>(vocabSize_, modelWidth_);
	positionEmbedding_ = oa::makeShared<oa::Embedding>(contextLength_, modelWidth_);
	registerModule("token_embedding", tokenEmbedding_);
	registerModule("position_embedding", positionEmbedding_);

	blocks_.reserve(static_cast<oa::Usize>(inNumLayers));
	for (oa::I32 index = 0; index < inNumLayers; ++index) {
		auto block = oa::makeShared<oa::TransformerBlock>(
			modelWidth_, hiddenWidth_, contextLength_, numHeads_, inEps);
		const oa::String name = oa::format("block_%d", index);
		registerModule(name.cStr(), block);
		blocks_.pushBack(oa::move(block));
	}

	finalNorm_ = oa::makeShared<oa::LayerNorm>(modelWidth_, inEps);
	head_ = oa::makeShared<oa::Linear>(modelWidth_, vocabSize_);
	registerModule("final_norm", finalNorm_);
	registerModule("head", head_);

	for (auto* parameter : allParameterPtrs()) {
		parameter->data.setRequiresGrad(true);
	}
}

oa::Matrix oa::NnTransformer::forward(const oa::Matrix& inTokens) {
	OA_REQUIRE_MSG(inTokens.rank() == 2 && inTokens.size(1) == contextLength_,
		"oa::NnTransformer expects token ids shaped [batch, contextLength]");
	const auto batch = static_cast<oa::I32>(inTokens.size(0));
	const auto rows = static_cast<oa::I64>(batch) * contextLength_;
	auto value = tokenEmbedding_->forward(inTokens).reshape({rows, modelWidth_})
		+ positionEmbedding_->forward(positionIds(batch));
	for (auto& block : blocks_) value = block->forward(value);
	return head_->forward(finalNorm_->forward(value));
}

oa::Matrix oa::NnTransformer::positionIds(oa::I32 inBatch) const {
	oa::Vec<oa::I32> ids(static_cast<oa::Usize>(inBatch * contextLength_));
	for (oa::Usize index = 0; index < ids.size(); ++index) {
		ids[index] = static_cast<oa::I32>(index % static_cast<oa::Usize>(contextLength_));
	}
	return oa::FnMatrix::fromInt32(
		oa::Span<const oa::I32>(ids.data(), ids.size()),
		{inBatch * contextLength_},
		oa::ScalarType::UInt32);
}
