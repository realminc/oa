// OA NLP Suite - controlled language-model recipes shared by tutorials and apps.

#pragma once

#include <Oa/Core/Matrix.h>
#include <Oa/Core/Types.h>
#include <Oa/Ml/Module.h>
#include <Oa/Ml/Nn.h>
#include <Oa/Ml/Tokenizer.h>

enum class OaNlpArchitecture : OaU8 {
	Rnn,
	Gru,
	Transformer,
	MoeTransformer,
	Mamba3,
};

enum class OaNlpTokenizerKind : OaU8 {
	Byte,
	Bpe,
	Char,
};

// Canonical controlled-workload contract shared by desktop tutorials and the
// Android Mobile Lab. Keep these in the library rather than duplicating magic
// numbers in each frontend: the suite is only a fair end-to-end comparison when
// corpus, dimensions, optimizer workload, prompt, and generated source length
// are identical.
inline constexpr OaI32 OaNlpSuiteContextLength = 16;
inline constexpr OaI32 OaNlpSuiteModelWidth = 32;
inline constexpr OaI32 OaNlpSuiteHiddenWidth = 64;
inline constexpr OaI32 OaNlpSuiteTrainingSteps = 300;
inline constexpr OaI32 OaNlpSuiteBatchSize = 64;
inline constexpr OaU64 OaNlpSuiteRngSeed = 20260714ULL;
inline constexpr const char* OaNlpSuiteGenerationPrompt = "to be";
inline constexpr OaI32 OaNlpSuiteGenerationSourceUnits = 80;

class OaNlpSuiteRecipe {
public:
	OaNlpSuiteRecipe(
		OaNlpArchitecture InArchitecture = OaNlpArchitecture::Gru,
		OaNlpTokenizerKind InTokenizer = OaNlpTokenizerKind::Byte);

	[[nodiscard]] OaNlpArchitecture Architecture() const { return Architecture_; }
	[[nodiscard]] OaNlpTokenizerKind Tokenizer() const { return Tokenizer_; }
	[[nodiscard]] OaI32 VocabSize() const;
	[[nodiscard]] OaI32 ContextLength() const { return OaNlpSuiteContextLength; }
	[[nodiscard]] OaI32 ModelWidth() const { return OaNlpSuiteModelWidth; }
	[[nodiscard]] OaI32 HiddenWidth() const { return OaNlpSuiteHiddenWidth; }
	[[nodiscard]] OaF32 LearningRate() const;
	[[nodiscard]] const char* ArchitectureId() const;
	[[nodiscard]] const char* ArchitectureName() const;
	[[nodiscard]] const char* TokenizerId() const;
	[[nodiscard]] const char* TokenizerName() const;
	[[nodiscard]] const char* ModelDescription() const;
	[[nodiscard]] const char* TimerName() const;

private:
	OaNlpArchitecture Architecture_;
	OaNlpTokenizerKind Tokenizer_;
};

class OaNlpSuiteModel final : public OaModule {
public:
	explicit OaNlpSuiteModel(const OaNlpSuiteRecipe& InRecipe);

	OaMatrix Forward(const OaMatrix& InTokens) override;
	// Stateful single-token decoding is the canonical autoregressive path for
	// RNN, GRU, and Mamba-3. Full-window Forward remains training/evaluation;
	// Transformer families use their causal sliding window for generation.
	[[nodiscard]] bool SupportsStatefulGeneration() const;
	void ResetGenerationState(OaI32 InBatch = 1);
	OaMatrix ForwardGenerationStep(const OaMatrix& InToken);
	[[nodiscard]] const OaNlpSuiteRecipe& Recipe() const { return Recipe_; }

private:
	[[nodiscard]] OaMatrix PositionIds(OaI32 InBatch, OaI32 InSequence) const;

	OaNlpSuiteRecipe Recipe_;
	OaSharedPtr<OaModule> TokenEmbedding_;
	OaSharedPtr<OaEmbedding> PositionEmbedding_;
	OaSharedPtr<OaRnn> Rnn_;
	OaSharedPtr<OaGru> Gru_;
	OaSharedPtr<OaTransformerBlock> Transformer_;
	OaSharedPtr<OaLayerNorm> FinalNorm_;
	OaSharedPtr<OaMamba3Module> Mamba3_;
	OaSharedPtr<OaLinear> Head_;
	OaMatrix RnnGenerationHidden_;
	OaMatrix GruGenerationHidden_;
};

class OaNlpSuiteSampler {
public:
	explicit OaNlpSuiteSampler(const OaNlpSuiteRecipe& InRecipe, OaI32 InBatchSize);

	void Next(OaMatrix& OutInput, OaMatrix& OutTarget);
	[[nodiscard]] OaI64 LastSourceUnits() const { return LastSourceUnits_; }
	[[nodiscard]] OaVec<OaI32> Encode(const char* InText) const;
	[[nodiscard]] OaString Decode(const OaVec<OaI32>& InTokens) const;
	[[nodiscard]] OaMatrix InputMatrix(const OaVec<OaI32>& InTokens) const;
	[[nodiscard]] OaMatrix InputStepMatrix(OaI32 InToken) const;

	[[nodiscard]] static const char* Corpus();

private:
	[[nodiscard]] OaI32 EncodeChar(char InCharacter) const;
	[[nodiscard]] OaString DecodeChar(OaI32 InToken) const;
	[[nodiscard]] OaI32 TokenSourceUnits(OaI32 InToken) const;
	[[nodiscard]] OaMatrix ToMatrix(const OaVec<OaI32>& InTokens, OaI32 InBatchSize) const;

	OaNlpSuiteRecipe Recipe_;
	OaI32 BatchSize_ = 1;
	OaI64 Cursor_ = 0;
	OaI64 LastSourceUnits_ = 0;
	OaBpeTokenizer BpeTokenizer_{320};
	OaVec<OaI32> Tokens_;
	OaVec<OaI32> TokenSourceUnits_;
};
