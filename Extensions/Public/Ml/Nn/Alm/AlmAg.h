#pragma once

// OaAlmAg — complete autograd Animation Language Model.
//
// This is the product-level module and persistence boundary. The tokenizer and
// motion-token prior remain independently trainable children, but generation,
// architecture identity, and .oam persistence are owned here so applications do
// not reconstruct the pipeline from unrelated checkpoint paths and CLI flags.

#include <Ml/Nn/Alm/AlmConfig.h>
#include <Ml/Nn/Alm/AlmPriorAg.h>
#include <Ml/Nn/Alm/AlmTokenizerAg.h>
#include <Ml/Nn/Alm/ClipTextAg.h>
#include <Ml/Nn/Alm/ClipTokenizer.h>

struct OaAlmAgConfig {
	OaAlmTokenizerConfig Tokenizer;
	OaAlmPriorConfig Prior;
	// Exact frozen text encoder used to produce Prior.TextFeatureDim inputs.
	// Empty only for an unconditional prior.
	OaString TextEncoder;
	OaU32 ClipMergesBytes = 0; // non-zero means native OaClipTextAg is bundled
};

class OaAlmAg : public OaModule {
public:
	explicit OaAlmAg(const OaAlmAgConfig& InConfig);

	OaAlmAg(OaSharedPtr<OaAlmTokenizerAg> InTokenizer,
		OaSharedPtr<OaAlmPriorAg> InPrior, OaStringView InTextEncoder = {}
	);
	OaAlmAg(OaSharedPtr<OaAlmTokenizerAg> InTokenizer,
		OaSharedPtr<OaAlmPriorAg> InPrior, OaSharedPtr<OaClipTextAg> InTextEncoder,
		OaSpan<const OaU8> InClipMerges, OaStringView InTextEncoderIdentity);

	// OaModule compatibility: token ids [B,T] -> prior logits [B,T,V].
	OaMatrix Forward(const OaMatrix& InTokenIds) override;
	[[nodiscard]] OaMatrix ForwardConditioned(
		const OaMatrix& InTokenIds, const OaMatrix& InTextFeatures
	);

	[[nodiscard]] OaVec<OaMatrix> Tokenize(
		const OaMatrix& InMotion, OaI32 InBatch, OaI32 InFrames
	);
	[[nodiscard]] OaMatrix Detokenize(
		const OaVec<OaMatrix>& InTokenIds, OaI32 InBatch, OaI32 InTokenLength
	);

	// End-to-end token sampling + VQ decode. Output is normalized HumanML3D
	// features [B*Frames, InputDim]; denormalization belongs to the dataset.
	[[nodiscard]] OaMatrix GenerateMotion(
		OaI32 InBatchSize, OaF32 InTemperature = 1.0F, OaI32 InTopK = 0,
		OaF32 InTopP = 0.9F, OaI32 InMaxTokens = 256
	);
	[[nodiscard]] OaMatrix GenerateMotionConditioned(
		const OaMatrix& InTextFeatures, OaF32 InTemperature = 1.0F,
		OaI32 InTopK = 0, OaF32 InTopP = 0.9F, OaI32 InMaxTokens = 256
	);
	[[nodiscard]] OaResult<OaMatrix> EncodePrompt(OaStringView InPrompt);
	[[nodiscard]] OaResult<OaMatrix> GenerateMotionPrompt(
		OaStringView InPrompt, OaF32 InTemperature = 1.0F,
		OaI32 InTopK = 0, OaF32 InTopP = 0.9F, OaI32 InMaxTokens = 256);
	[[nodiscard]] bool HasNativeTextEncoder() const noexcept { return static_cast<bool>(TextEncoder_); }

	[[nodiscard]] OaAlmTokenizerAg& Tokenizer() noexcept { return *Tokenizer_; }
	[[nodiscard]] const OaAlmTokenizerAg& Tokenizer() const noexcept { return *Tokenizer_; }
	[[nodiscard]] OaAlmPriorAg& Prior() noexcept { return *Prior_; }
	[[nodiscard]] const OaAlmPriorAg& Prior() const noexcept { return *Prior_; }
	[[nodiscard]] const OaAlmAgConfig& Config() const noexcept { return Config_; }

	// One exact, versioned product artifact containing both children and persistent
	// tokenizer state. Stage-specific optimizer checkpoints remain trainer internals.
	[[nodiscard]] OaStatus SaveBundle(const OaString& InPath) const;
	[[nodiscard]] static OaResult<OaSharedPtr<OaAlmAg>> LoadBundle(const OaString& InPath);

private:
	void RegisterChildren();

	OaAlmAgConfig Config_;
	OaSharedPtr<OaAlmTokenizerAg> Tokenizer_;
	OaSharedPtr<OaAlmPriorAg> Prior_;
	OaSharedPtr<OaClipTextAg> TextEncoder_;
	OaUniquePtr<OaClipTokenizer> ClipTokenizer_;
};
