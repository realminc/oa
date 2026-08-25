// TransformerBlock — pre-norm transformer block (level 1 API)
//
// Single-block architecture:
//   x → [residual] selfAttention(LN(x)) → dense FFN or MoE
//
// Attention: the shared oa::MultiHeadAttention module. FFN: Linear→GELU→Linear.
// The legacy constructors retain one-head behavior; explicit-head overloads are
// the product path used by ALM and future Transformer models.

#pragma once

#include <oa/ml/module.h>
#include <oa/ml/nn/attention.h>

#include <oa/ml/nn/attention/multiheadattention/multiHeadAttention.h>
#include <oa/ml/nn/embedding/embedding.h>
#include <oa/ml/nn/layernorm/layerNorm.h>
#include <oa/ml/nn/linear/linear.h>
#include <oa/ml/nn/moe/moe.h>

namespace oa {

class TransformerBlock : public oa::Module {
public:
	TransformerBlock() = default;
	TransformerBlock(oa::I32 inDModel, oa::I32 inDFF, oa::I32 inSeqLen, oa::F32 inEps = 1e-5f);
	TransformerBlock(oa::I32 inDModel, oa::I32 inDFF, oa::I32 inSeqLen,
		oa::I32 inNumHeads, oa::F32 inEps);
	TransformerBlock(oa::I32 inDModel, oa::I32 inExpertDFF, oa::I32 inSeqLen,
		oa::I32 inNumExperts, oa::I32 inExpertsPerToken, oa::F32 inEps = 1e-5f);
	TransformerBlock(oa::I32 inDModel, oa::I32 inExpertDFF, oa::I32 inSeqLen,
		oa::I32 inNumHeads, oa::I32 inNumExperts, oa::I32 inExpertsPerToken, oa::F32 inEps);

	void init(oa::I32 inDModel, oa::I32 inDFF, oa::I32 inSeqLen, oa::F32 inEps = 1e-5f);
	void init(oa::I32 inDModel, oa::I32 inDFF, oa::I32 inSeqLen,
		oa::I32 inNumHeads, oa::F32 inEps);
	void initMoe(oa::I32 inDModel, oa::I32 inExpertDFF, oa::I32 inSeqLen,
		oa::I32 inNumExperts, oa::I32 inExpertsPerToken, oa::F32 inEps = 1e-5f);
	void initMoe(oa::I32 inDModel, oa::I32 inExpertDFF, oa::I32 inSeqLen,
		oa::I32 inNumHeads, oa::I32 inNumExperts, oa::I32 inExpertsPerToken, oa::F32 inEps);

	// forward: x [B*S, D] → [B*S, D]. inSeqLen separates sequences within the
	// flattened batch. Causal visibility remains the language-model default;
	// flow/denoising encoders explicitly select Bidirectional.
	oa::Matrix forward(const oa::Matrix& inX) override;
	// Same block with an explicit additive attention mask [B*H*S,S]. The mask
	// is shared by language padding and bidirectional denoising; Flash attention
	// is intentionally bypassed because its current kernel is causal-only.
	oa::Matrix forwardMasked(const oa::Matrix& inX, const oa::Matrix& inAdditiveMask);
	// Enables DiT-style AdaLN-Zero modulation. The ordinary language-model path
	// remains unchanged until this is explicitly enabled by a conditioned model.
	void enableAdaptiveConditioning(oa::I32 inConditionDim);
	oa::Matrix forwardConditioned(
		const oa::Matrix& inX,
		const oa::Matrix& inCondition,
		const oa::Matrix& inAdditiveMask = {});

	[[nodiscard]] oa::I32 dModel() const { return dModel_; }
	[[nodiscard]] oa::I32 dFF() const { return dFF_; }
	[[nodiscard]] oa::I32 seqLen() const { return seqLen_; }
	[[nodiscard]] oa::I32 numHeads() const { return numHeads_; }
	[[nodiscard]] oa::AttentionMode attentionMode() const { return attentionMode_; }
	[[nodiscard]] bool isMoe() const { return static_cast<bool>(moe_); }
	[[nodiscard]] Moe* moe() { return moe_.get(); }
	[[nodiscard]] const Moe* moe() const { return moe_.get(); }

	// The weights are sequence-length independent. Updating the runtime length
	// only changes the B/S view and its causal mask, which lets one block serve
	// fixed-length training and growing-prefix autoregressive generation.
	void setSeqLen(oa::I32 inSeqLen);
	void setAttentionMode(oa::AttentionMode inMode);

private:
	oa::I32 dModel_ = 0;
	oa::I32 dFF_ = 0;
	oa::I32 seqLen_ = 0;
	oa::I32 numHeads_ = 1;
	oa::AttentionMode attentionMode_ = oa::AttentionMode::Causal;

	oa::SharedPtr<oa::LayerNorm> lnAttn_;
	oa::SharedPtr<oa::MultiHeadAttention> attention_;
	oa::SharedPtr<oa::LayerNorm> lnFfn_;
	oa::SharedPtr<oa::Linear> ffn1_;
	oa::SharedPtr<oa::Linear> ffn2_;
	oa::SharedPtr<Moe> moe_;
	oa::SharedPtr<oa::Linear> adaptiveModulation_;
	oa::I32 conditionDim_ = 0;

	void initAttention(oa::I32 inNumHeads, oa::F32 inEps);
	oa::Matrix forwardImpl(const oa::Matrix& inX, const oa::Matrix* inAdditiveMask);
};

/// Ready-to-train causal language model: token + position embeddings, a stack
/// of Transformer blocks, final normalization, and vocabulary projection.
/// Input token ids are [batch, contextLength]; logits are [batch*contextLength,
/// vocabSize] for all-position next-token training.
class NnTransformer final : public oa::Module {
public:
	NnTransformer(
		oa::I32 inVocabSize,
		oa::I32 inContextLength,
		oa::I32 inModelWidth = 32,
		oa::I32 inHiddenWidth = 64,
		oa::I32 inNumLayers = 1,
		oa::I32 inNumHeads = 1,
		oa::F32 inEps = 1e-5F);

	oa::Matrix forward(const oa::Matrix& inTokens) override;

	[[nodiscard]] oa::I32 vocabSize() const { return vocabSize_; }
	[[nodiscard]] oa::I32 contextLength() const { return contextLength_; }
	[[nodiscard]] oa::I32 modelWidth() const { return modelWidth_; }
	[[nodiscard]] oa::I32 hiddenWidth() const { return hiddenWidth_; }
	[[nodiscard]] oa::I32 numLayers() const {
		return static_cast<oa::I32>(blocks_.size());
	}
	[[nodiscard]] oa::I32 numHeads() const { return numHeads_; }

private:
	[[nodiscard]] oa::Matrix positionIds(oa::I32 inBatch) const;

	oa::I32 vocabSize_ = 0;
	oa::I32 contextLength_ = 0;
	oa::I32 modelWidth_ = 0;
	oa::I32 hiddenWidth_ = 0;
	oa::I32 numHeads_ = 0;
	oa::SharedPtr<oa::Embedding> tokenEmbedding_;
	oa::SharedPtr<oa::Embedding> positionEmbedding_;
	oa::Vec<oa::SharedPtr<oa::TransformerBlock>> blocks_;
	oa::SharedPtr<oa::LayerNorm> finalNorm_;
	oa::SharedPtr<oa::Linear> head_;
};

} // namespace oa
