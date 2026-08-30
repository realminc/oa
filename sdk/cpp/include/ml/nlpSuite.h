// OA NLP Suite — controlled language-model recipes shared by tutorials and apps.

#pragma once

#include <oa/core/matrix.h>
#include <oa/core/types.h>
#include <oa/ml/module.h>
#include <oa/ml/nn.h>
#include <oa/ml/tokenizer.h>

namespace oa {

enum class NlpArchitecture : oa::U8 {
	Rnn,
	Gru,
	Transformer,
	MoeTransformer,
	Mamba3,
};

enum class NlpTokenizerKind : oa::U8 {
	Byte,
	Bpe,
	Char,
};

// Canonical controlled-workload contract shared by desktop tutorials and the
// Android Mobile Lab. Keep these in SDK support rather than duplicating magic
// numbers in each frontend: the suite is only a fair end-to-end comparison when
// corpus, dimensions, optimizer workload, prompt, and generated source length
// are identical.
inline constexpr oa::I32 NlpSuiteContextLength = 16;
inline constexpr oa::I32 NlpSuiteModelWidth = 32;
inline constexpr oa::I32 NlpSuiteHiddenWidth = 64;
inline constexpr oa::I32 NlpSuiteTrainingSteps = 300;
inline constexpr oa::I32 NlpSuiteBatchSize = 64;
inline constexpr oa::U64 NlpSuiteRngSeed = 20260714ULL;
inline constexpr const char* NlpSuiteGenerationPrompt = "to be";
inline constexpr oa::I32 NlpSuiteGenerationSourceUnits = 80;

class NlpSuiteRecipe {
public:
	NlpSuiteRecipe(
		NlpArchitecture inArchitecture = NlpArchitecture::Gru,
		NlpTokenizerKind inTokenizer = NlpTokenizerKind::Byte,
		oa::I32 inContextLength = NlpSuiteContextLength,
		oa::I32 inModelWidth = NlpSuiteModelWidth,
		oa::I32 inHiddenWidth = NlpSuiteHiddenWidth);

	[[nodiscard]] NlpArchitecture architecture() const { return architecture_; }
	[[nodiscard]] NlpTokenizerKind tokenizer() const { return tokenizer_; }
	[[nodiscard]] oa::I32 vocabSize() const;
	[[nodiscard]] oa::I32 contextLength() const { return contextLength_; }
	[[nodiscard]] oa::I32 modelWidth() const { return modelWidth_; }
	[[nodiscard]] oa::I32 hiddenWidth() const { return hiddenWidth_; }
	[[nodiscard]] oa::F32 learningRate() const;
	[[nodiscard]] const char* architectureId() const;
	[[nodiscard]] const char* architectureName() const;
	[[nodiscard]] const char* tokenizerId() const;
	[[nodiscard]] const char* tokenizerName() const;
	[[nodiscard]] const char* modelDescription() const;
	[[nodiscard]] const char* timerName() const;

private:
	NlpArchitecture architecture_;
	NlpTokenizerKind tokenizer_;
	oa::I32 contextLength_ = NlpSuiteContextLength;
	oa::I32 modelWidth_ = NlpSuiteModelWidth;
	oa::I32 hiddenWidth_ = NlpSuiteHiddenWidth;
};

class NlpSuiteModel final : public oa::Module {
public:
	explicit NlpSuiteModel(const NlpSuiteRecipe& inRecipe);

	oa::Matrix forward(const oa::Matrix& inTokens) override;
	// Stateful single-token decoding is the canonical autoregressive path for
	// RNN, GRU, and Mamba-3. Full-window forward remains training/evaluation;
	// Transformer families use their causal sliding window for generation.
	[[nodiscard]] bool supportsStatefulGeneration() const;
	void resetGenerationState(oa::I32 inBatch = 1);
	oa::Matrix forwardGenerationStep(const oa::Matrix& inToken);
	[[nodiscard]] const NlpSuiteRecipe& recipe() const { return recipe_; }

private:
	[[nodiscard]] oa::Matrix positionIds(
		oa::I32 inBatch,
		oa::I32 inSequence) const;

	NlpSuiteRecipe recipe_;
	oa::SharedPtr<oa::Module> tokenEmbedding_;
	oa::SharedPtr<oa::Embedding> positionEmbedding_;
	oa::SharedPtr<oa::Rnn> rnn_;
	oa::SharedPtr<oa::Gru> gru_;
	oa::SharedPtr<oa::TransformerBlock> transformer_;
	oa::SharedPtr<oa::LayerNorm> finalNorm_;
	oa::SharedPtr<oa::Mamba3Module> mamba3_;
	oa::SharedPtr<oa::Linear> head_;
	oa::Matrix rnnGenerationHidden_;
	oa::Matrix gruGenerationHidden_;
};

class NlpSuiteSampler {
public:
	explicit NlpSuiteSampler(const NlpSuiteRecipe& inRecipe, oa::I32 inBatchSize);

	void next(oa::Matrix& outInput, oa::Matrix& outTarget);
	[[nodiscard]] oa::I64 lastSourceUnits() const { return lastSourceUnits_; }
	[[nodiscard]] oa::Vector<oa::I32> encode(const char* inText) const;
	[[nodiscard]] oa::String decode(const oa::Vector<oa::I32>& inTokens) const;
	[[nodiscard]] oa::Matrix inputMatrix(const oa::Vector<oa::I32>& inTokens) const;
	[[nodiscard]] oa::Matrix inputStepMatrix(oa::I32 inToken) const;

	[[nodiscard]] static const char* corpus();

private:
	[[nodiscard]] oa::I32 encodeChar(char inCharacter) const;
	[[nodiscard]] oa::String decodeChar(oa::I32 inToken) const;
	[[nodiscard]] oa::I32 tokenSourceUnits(oa::I32 inToken) const;
	[[nodiscard]] oa::Matrix toMatrix(
		const oa::Vector<oa::I32>& inTokens,
		oa::I32 inBatchSize) const;

	NlpSuiteRecipe recipe_;
	oa::I32 batchSize_ = 1;
	oa::I64 cursor_ = 0;
	oa::I64 lastSourceUnits_ = 0;
	oa::BpeTokenizer bpeTokenizer_{320};
	oa::Vector<oa::I32> tokens_;
	oa::Vector<oa::I32> tokenSourceUnits_;
};

} // namespace oa
