#pragma once

// AlmAg — complete autograd Animation Language Model.
//
// This is the product-level module and persistence boundary. The tokenizer and
// motion-token prior remain independently trainable children, but generation,
// architecture identity, and .oam persistence are owned here so applications do
// not reconstruct the pipeline from unrelated checkpoint paths and CLI flags.

#include <ml/nn/alm/almConfig.h>
#include <ml/nn/alm/almPriorAg.h>
#include <ml/nn/alm/almTokenizerAg.h>
#include <ml/nn/alm/clipTextAg.h>
#include <ml/nn/alm/clipTokenizer.h>

namespace oa {

struct AlmAgConfig {
	AlmTokenizerConfig tokenizer;
	AlmPriorConfig prior;
	// exact frozen text encoder used to produce prior.textFeatureDim inputs.
	// Empty only for an unconditional prior.
	oa::String textEncoder;
	oa::U32 clipMergesBytes = 0; // non-zero means native ClipTextAg is bundled
};

class AlmAg : public oa::Module {
public:
	explicit AlmAg(const AlmAgConfig& inConfig);

	AlmAg(oa::SharedPtr<AlmTokenizerAg> inTokenizer,
		oa::SharedPtr<AlmPriorAg> inPrior, oa::StringView inTextEncoder = {}
	);
	AlmAg(oa::SharedPtr<AlmTokenizerAg> inTokenizer,
		oa::SharedPtr<AlmPriorAg> inPrior, oa::SharedPtr<ClipTextAg> inTextEncoder,
		oa::Span<const oa::U8> inClipMerges, oa::StringView inTextEncoderIdentity);

	// oa::Module compatibility: token ids [B,T] -> prior logits [B,T,V].
	oa::Matrix forward(const oa::Matrix& inTokenIds) override;
	[[nodiscard]] oa::Matrix forwardConditioned(
		const oa::Matrix& inTokenIds, const oa::Matrix& inTextFeatures
	);

	[[nodiscard]] oa::Vec<oa::Matrix> tokenize(
		const oa::Matrix& inMotion, oa::I32 inBatch, oa::I32 inFrames
	);
	[[nodiscard]] oa::Matrix detokenize(
		const oa::Vec<oa::Matrix>& inTokenIds, oa::I32 inBatch, oa::I32 inTokenLength
	);

	// End-to-end token sampling + VQ decode. output is normalized HumanML3D
	// features [B*frames, inputDim]; denormalization belongs to the dataset.
	[[nodiscard]] oa::Matrix generateMotion(
		oa::I32 inBatchSize, oa::F32 inTemperature = 1.0F, oa::I32 inTopK = 0,
		oa::F32 inTopP = 0.9F, oa::I32 inMaxTokens = 256
	);
	[[nodiscard]] oa::Matrix generateMotionConditioned(
		const oa::Matrix& inTextFeatures, oa::F32 inTemperature = 1.0F,
		oa::I32 inTopK = 0, oa::F32 inTopP = 0.9F, oa::I32 inMaxTokens = 256
	);
	[[nodiscard]] oa::Result<oa::Matrix> encodePrompt(oa::StringView inPrompt);
	[[nodiscard]] oa::Result<oa::Matrix> generateMotionPrompt(
		oa::StringView inPrompt, oa::F32 inTemperature = 1.0F,
		oa::I32 inTopK = 0, oa::F32 inTopP = 0.9F, oa::I32 inMaxTokens = 256);
	[[nodiscard]] bool hasNativeTextEncoder() const noexcept { return static_cast<bool>(textEncoder_); }

	[[nodiscard]] AlmTokenizerAg& tokenizer() noexcept { return *tokenizer_; }
	[[nodiscard]] const AlmTokenizerAg& tokenizer() const noexcept { return *tokenizer_; }
	[[nodiscard]] AlmPriorAg& prior() noexcept { return *prior_; }
	[[nodiscard]] const AlmPriorAg& prior() const noexcept { return *prior_; }
	[[nodiscard]] const AlmAgConfig& config() const noexcept { return config_; }

	// One exact, versioned product artifact containing both children and persistent
	// tokenizer state. stage-specific optimizer checkpoints remain trainer internals.
	[[nodiscard]] oa::Status saveBundle(
		oa::Engine& inEngine, const oa::String& inPath) const;
	[[nodiscard]] static oa::Result<oa::SharedPtr<AlmAg>> loadBundle(
		oa::Engine& inEngine, const oa::String& inPath);

private:
	void registerChildren();

	AlmAgConfig config_;
	oa::SharedPtr<AlmTokenizerAg> tokenizer_;
	oa::SharedPtr<AlmPriorAg> prior_;
	oa::SharedPtr<ClipTextAg> textEncoder_;
	oa::UniquePtr<ClipTokenizer> clipTokenizer_;
};

} // namespace oa
