#include <Oa/Ml/NlpSuite.h>

#include <Oa/Core/FnMatrix.h>
#include <Oa/Ml/Byte.h>

#include <algorithm>
#include <cstring>
#include <stdexcept>

OaNlpSuiteRecipe::OaNlpSuiteRecipe(
	OaNlpArchitecture InArchitecture,
	OaNlpTokenizerKind InTokenizer)
	: Architecture_(InArchitecture)
	, Tokenizer_(InTokenizer) {
}

OaI32 OaNlpSuiteRecipe::VocabSize() const {
	switch (Tokenizer_) {
	case OaNlpTokenizerKind::Byte:
		return 256;
	case OaNlpTokenizerKind::Bpe:
		return 320;
	case OaNlpTokenizerKind::Char:
		return 27;
	}
	return 256;
}

OaF32 OaNlpSuiteRecipe::LearningRate() const {
	return Architecture_ == OaNlpArchitecture::Mamba3 ? 0.003F : 0.01F;
}

const char* OaNlpSuiteRecipe::ArchitectureId() const {
	switch (Architecture_) {
	case OaNlpArchitecture::Rnn:
		return "rnn";
	case OaNlpArchitecture::Gru:
		return "gru";
	case OaNlpArchitecture::Transformer:
		return "transformer";
	case OaNlpArchitecture::MoeTransformer:
		return "moe";
	case OaNlpArchitecture::Mamba3:
		return "mamba3";
	}
	return "gru";
}

const char* OaNlpSuiteRecipe::ArchitectureName() const {
	switch (Architecture_) {
	case OaNlpArchitecture::Rnn:
		return "RNN";
	case OaNlpArchitecture::Gru:
		return "GRU";
	case OaNlpArchitecture::Transformer:
		return "Transformer";
	case OaNlpArchitecture::MoeTransformer:
		return "MoE Transformer";
	case OaNlpArchitecture::Mamba3:
		return "Mamba-3";
	}
	return "GRU";
}

const char* OaNlpSuiteRecipe::TokenizerId() const {
	switch (Tokenizer_) {
	case OaNlpTokenizerKind::Byte:
		return "byte";
	case OaNlpTokenizerKind::Bpe:
		return "bpe";
	case OaNlpTokenizerKind::Char:
		return "char";
	}
	return "byte";
}

const char* OaNlpSuiteRecipe::TokenizerName() const {
	switch (Tokenizer_) {
	case OaNlpTokenizerKind::Byte:
		return "Byte";
	case OaNlpTokenizerKind::Bpe:
		return "BPE";
	case OaNlpTokenizerKind::Char:
		return "Char";
	}
	return "Byte";
}

const char* OaNlpSuiteRecipe::ModelDescription() const {
	switch (Architecture_) {
	case OaNlpArchitecture::Rnn:
		return "Embedding(32) -> RNN(32x64) -> Linear(vocab)";
	case OaNlpArchitecture::Gru:
		return "Embedding(32) -> GRU(32x64) -> Linear(vocab)";
	case OaNlpArchitecture::Transformer:
		return "Embedding(32) -> Transformer(FFN=64) -> Linear(vocab)";
	case OaNlpArchitecture::MoeTransformer:
		return "Embedding(32) -> MoE(E=4,K=2,DFF=16) -> Linear(vocab)";
	case OaNlpArchitecture::Mamba3:
		return "Embedding(32) -> Mamba-3(state=32) -> Linear(vocab)";
	}
	return "Embedding(32) -> GRU(32x64) -> Linear(vocab)";
}

const char* OaNlpSuiteRecipe::TimerName() const {
	switch (Architecture_) {
	case OaNlpArchitecture::Rnn:
		return "nlp_rnn_step";
	case OaNlpArchitecture::Gru:
		return "nlp_gru_step";
	case OaNlpArchitecture::Transformer:
		return "nlp_transformer_step";
	case OaNlpArchitecture::MoeTransformer:
		return "nlp_moe_step";
	case OaNlpArchitecture::Mamba3:
		return "nlp_mamba3_step";
	}
	return "nlp_gru_step";
}

OaNlpSuiteModel::OaNlpSuiteModel(const OaNlpSuiteRecipe& InRecipe)
	: Recipe_(InRecipe) {
	const OaI32 vocabSize = Recipe_.VocabSize();
	const OaI32 modelWidth = Recipe_.ModelWidth();
	const OaI32 hiddenWidth = Recipe_.HiddenWidth();
	const auto weightDtype = OaFnMatrix::GetWeightDtype();

	if (Recipe_.Tokenizer() == OaNlpTokenizerKind::Byte and
		(Recipe_.Architecture() == OaNlpArchitecture::Rnn or
		 Recipe_.Architecture() == OaNlpArchitecture::Gru)) {
		auto embedding = OaMakeSharedPtr<OaByteEmbedding>(modelWidth);
		embedding->Parameters()[0].Data =
			OaFnMatrix::RandN(OaMatrixShape{vocabSize, modelWidth}, weightDtype);
		TokenEmbedding_ = embedding;
	} else {
		auto embedding = OaMakeSharedPtr<OaEmbedding>(vocabSize, modelWidth);
		embedding->Parameters()[0].Data =
			OaFnMatrix::RandN(OaMatrixShape{vocabSize, modelWidth}, weightDtype);
		TokenEmbedding_ = embedding;
	}
	RegisterModule("embed", TokenEmbedding_);

	switch (Recipe_.Architecture()) {
	case OaNlpArchitecture::Rnn:
		Rnn_ = OaMakeSharedPtr<OaRnn>(modelWidth, hiddenWidth, 1);
		Head_ = OaMakeSharedPtr<OaLinear>(hiddenWidth, vocabSize);
		Head_->Parameters()[0].Data =
			OaFnMatrix::Rand(OaMatrixShape{vocabSize, hiddenWidth}, weightDtype);
		RegisterModule("rnn", Rnn_);
		break;
	case OaNlpArchitecture::Gru:
		Gru_ = OaMakeSharedPtr<OaGru>(modelWidth, hiddenWidth, 1);
		Head_ = OaMakeSharedPtr<OaLinear>(hiddenWidth, vocabSize);
		// Preserve the desktop tutorial recipe exactly. The compact character
		// model uses Xavier here; byte and BPE use the historical uniform head.
		Head_->Parameters()[0].Data = Recipe_.Tokenizer() == OaNlpTokenizerKind::Char
			? OaFnMatrix::RandXavier(
				OaMatrixShape{vocabSize, hiddenWidth}, weightDtype)
			: OaFnMatrix::Rand(
				OaMatrixShape{vocabSize, hiddenWidth}, weightDtype);
		RegisterModule("gru", Gru_);
		break;
	case OaNlpArchitecture::Transformer:
	case OaNlpArchitecture::MoeTransformer:
		PositionEmbedding_ = OaMakeSharedPtr<OaEmbedding>(Recipe_.ContextLength(), modelWidth);
		Transformer_ = Recipe_.Architecture() == OaNlpArchitecture::MoeTransformer
			? OaMakeSharedPtr<OaTransformerBlock>(modelWidth, 16, Recipe_.ContextLength(), 4, 2)
			: OaMakeSharedPtr<OaTransformerBlock>(modelWidth, hiddenWidth, Recipe_.ContextLength());
		FinalNorm_ = OaMakeSharedPtr<OaLayerNorm>(modelWidth, 1e-5F);
		Head_ = OaMakeSharedPtr<OaLinear>(modelWidth, vocabSize);
		Head_->Parameters()[0].Data =
			OaFnMatrix::Rand(OaMatrixShape{vocabSize, modelWidth}, weightDtype);
		RegisterModule("pos_embed", PositionEmbedding_);
		RegisterModule("block", Transformer_);
		RegisterModule("ln_final", FinalNorm_);
		break;
	case OaNlpArchitecture::Mamba3:
		Mamba3_ = OaMakeSharedPtr<OaMamba3Module>(
			modelWidth, 32, 2, 16, 1, 0.5F, false, 4,
			0.001F, 0.1F, 1e-4F, 1e-4F, true);
		Head_ = OaMakeSharedPtr<OaLinear>(modelWidth, vocabSize);
		RegisterModule("mamba3", Mamba3_);
		break;
	}
	RegisterModule("head", Head_);

	for (auto* parameter : AllParameterPtrs()) {
		parameter->Data.SetRequiresGrad(true);
	}
}

OaMatrix OaNlpSuiteModel::Forward(const OaMatrix& InTokens) {
	const OaI32 batch = static_cast<OaI32>(InTokens.Size(0));
	const OaI32 sequence = static_cast<OaI32>(InTokens.Size(1));
	const OaI32 modelWidth = Recipe_.ModelWidth();
	const OaI32 hiddenWidth = Recipe_.HiddenWidth();
	const OaI64 rows = static_cast<OaI64>(batch) * sequence;
	auto embedded = TokenEmbedding_->Forward(InTokens);

	switch (Recipe_.Architecture()) {
	case OaNlpArchitecture::Rnn:
		return Head_->Forward(Rnn_->Forward(
			embedded.Reshape(OaMatrixShape{batch, sequence, modelWidth}))
			.Reshape(OaMatrixShape{rows, hiddenWidth}));
	case OaNlpArchitecture::Gru:
		return Head_->Forward(Gru_->Forward(
			embedded.Reshape(OaMatrixShape{batch, sequence, modelWidth}))
			.Reshape(OaMatrixShape{rows, hiddenWidth}));
	case OaNlpArchitecture::Transformer:
	case OaNlpArchitecture::MoeTransformer: {
		auto positioned = embedded.Reshape(OaMatrixShape{rows, modelWidth}) +
			PositionEmbedding_->Forward(PositionIds(batch, sequence));
		return Head_->Forward(FinalNorm_->Forward(Transformer_->Forward(positioned)));
	}
	case OaNlpArchitecture::Mamba3: {
		auto sequenceOutput = Mamba3_->Forward(
			embedded.Reshape(OaMatrixShape{batch, sequence, modelWidth}));
		return Head_->Forward(
			sequenceOutput.Reshape(OaMatrixShape{rows, modelWidth}) +
			embedded.Reshape(OaMatrixShape{rows, modelWidth}));
	}
	}
	return {};
}

bool OaNlpSuiteModel::SupportsStatefulGeneration() const {
	return Recipe_.Architecture() == OaNlpArchitecture::Rnn
		or Recipe_.Architecture() == OaNlpArchitecture::Gru
		or Recipe_.Architecture() == OaNlpArchitecture::Mamba3;
}

void OaNlpSuiteModel::ResetGenerationState(OaI32 InBatch) {
	switch (Recipe_.Architecture()) {
	case OaNlpArchitecture::Rnn:
		RnnGenerationHidden_ = Rnn_->ZeroState(InBatch);
		return;
	case OaNlpArchitecture::Gru:
		GruGenerationHidden_ = Gru_->ZeroState(InBatch);
		return;
	case OaNlpArchitecture::Mamba3:
		Mamba3_->ResetState(InBatch);
		return;
	default:
		throw std::logic_error(
			"ResetGenerationState requires an RNN, GRU, or Mamba-3 NLP recipe");
	}
}

OaMatrix OaNlpSuiteModel::ForwardGenerationStep(const OaMatrix& InToken) {
	if (not SupportsStatefulGeneration()) {
		throw std::logic_error(
			"ForwardGenerationStep requires an RNN, GRU, or Mamba-3 NLP recipe");
	}
	if (InToken.Rank() != 2 or InToken.Size(1) != 1) {
		throw std::invalid_argument(
			"ForwardGenerationStep expects token ids with shape [batch, 1]");
	}
	const OaI64 batch = InToken.Size(0);
	auto embedded = TokenEmbedding_->Forward(InToken)
		.Reshape(OaMatrixShape{batch, Recipe_.ModelWidth()});
	switch (Recipe_.Architecture()) {
	case OaNlpArchitecture::Rnn:
		return Head_->Forward(Rnn_->Step(embedded, RnnGenerationHidden_));
	case OaNlpArchitecture::Gru:
		return Head_->Forward(Gru_->Step(embedded, GruGenerationHidden_));
	case OaNlpArchitecture::Mamba3: {
		auto sequenceOutput = Mamba3_->Step(
			embedded.Reshape(OaMatrixShape{batch, 1, Recipe_.ModelWidth()}));
		return Head_->Forward(
			sequenceOutput.Reshape(OaMatrixShape{batch, Recipe_.ModelWidth()}) +
			embedded);
	}
	default:
		break;
	}
	return {};
}

OaMatrix OaNlpSuiteModel::PositionIds(OaI32 InBatch, OaI32 InSequence) const {
	const OaUsize count = static_cast<OaUsize>(InBatch) *
		static_cast<OaUsize>(InSequence);
	OaVec<OaI32> ids(count);
	for (OaUsize index = 0; index < ids.Size(); ++index) {
		ids[index] = static_cast<OaI32>(index % static_cast<OaUsize>(InSequence));
	}
	return OaFnMatrix::FromInt32(
		OaSpan<const OaI32>(ids.Data(), ids.Size()),
		OaMatrixShape{static_cast<OaI64>(ids.Size())},
		OaScalarType::UInt32);
}

OaNlpSuiteSampler::OaNlpSuiteSampler(
	const OaNlpSuiteRecipe& InRecipe,
	OaI32 InBatchSize)
	: Recipe_(InRecipe)
	, BatchSize_(std::max(1, InBatchSize)) {
	if (Recipe_.Tokenizer() == OaNlpTokenizerKind::Bpe) {
		BpeTokenizer_.Train(Corpus(), 64);
	}
	Tokens_ = Encode(Corpus());
	TokenSourceUnits_.Resize(static_cast<OaUsize>(Recipe_.VocabSize()));
	for (OaI32 token = 0; token < Recipe_.VocabSize(); ++token) {
		TokenSourceUnits_[static_cast<OaUsize>(token)] = TokenSourceUnits(token);
	}
}

void OaNlpSuiteSampler::Next(OaMatrix& OutInput, OaMatrix& OutTarget) {
	const OaI32 contextLength = Recipe_.ContextLength();
	const OaUsize count = static_cast<OaUsize>(BatchSize_) *
		static_cast<OaUsize>(contextLength);
	OaVec<OaI32> input(count);
	OaVec<OaI32> target(count);
	const OaI64 limit = static_cast<OaI64>(Tokens_.Size()) - contextLength - 1;
	LastSourceUnits_ = 0;
	for (OaI32 batch = 0; batch < BatchSize_; ++batch) {
		const OaI64 start = (Cursor_ + static_cast<OaI64>(batch) * 7) % limit;
		for (OaI32 position = 0; position < contextLength; ++position) {
			const OaUsize outputIndex = static_cast<OaUsize>(batch) *
				static_cast<OaUsize>(contextLength) + static_cast<OaUsize>(position);
			const OaUsize inputIndex = static_cast<OaUsize>(start + position);
			const OaUsize targetIndex = static_cast<OaUsize>(start + position + 1);
			input[outputIndex] = Tokens_[inputIndex];
			target[outputIndex] = Tokens_[targetIndex];
			LastSourceUnits_ += TokenSourceUnits_[
				static_cast<OaUsize>(target[outputIndex])];
		}
	}
	Cursor_ = (Cursor_ + BatchSize_) % limit;
	OutInput = ToMatrix(input, BatchSize_);
	OutTarget = ToMatrix(target, BatchSize_);
}

OaVec<OaI32> OaNlpSuiteSampler::Encode(const char* InText) const {
	if (Recipe_.Tokenizer() == OaNlpTokenizerKind::Bpe) {
		return BpeTokenizer_.Encode(InText);
	}
	const OaUsize length = std::strlen(InText);
	OaVec<OaI32> tokens(length);
	for (OaUsize index = 0; index < length; ++index) {
		tokens[index] = Recipe_.Tokenizer() == OaNlpTokenizerKind::Char
			? EncodeChar(InText[index])
			: static_cast<OaU8>(InText[index]);
	}
	return tokens;
}

OaString OaNlpSuiteSampler::Decode(const OaVec<OaI32>& InTokens) const {
	if (Recipe_.Tokenizer() == OaNlpTokenizerKind::Bpe) {
		return BpeTokenizer_.Decode(InTokens);
	}
	OaString text;
	for (const OaI32 token : InTokens) {
		if (Recipe_.Tokenizer() == OaNlpTokenizerKind::Char) {
			text += DecodeChar(token);
		} else {
			text += static_cast<char>(static_cast<OaU8>(token));
		}
	}
	return text;
}

OaMatrix OaNlpSuiteSampler::InputMatrix(const OaVec<OaI32>& InTokens) const {
	const OaI64 contextLength = Recipe_.ContextLength();
	const OaI32 batchSize = static_cast<OaI32>(
		static_cast<OaI64>(InTokens.Size()) / contextLength);
	return ToMatrix(InTokens, batchSize);
}

OaMatrix OaNlpSuiteSampler::InputStepMatrix(OaI32 InToken) const {
	return OaFnMatrix::FromInt32(
		OaSpan<const OaI32>(&InToken, 1),
		OaMatrixShape{1, 1},
		OaScalarType::UInt32);
}

const char* OaNlpSuiteSampler::Corpus() {
	return
		"to be or not to be that is the question whether tis nobler in the mind "
		"to suffer the slings and arrows of outrageous fortune or to take arms "
		"against a sea of troubles and by opposing end them "
		"to be or not to be that is the question whether tis nobler in the mind "
		"to suffer the slings and arrows of outrageous fortune or to take arms "
		"against a sea of troubles and by opposing end them "
		"to be or not to be that is the question whether tis nobler in the mind "
		"to suffer the slings and arrows of outrageous fortune or to take arms "
		"against a sea of troubles and by opposing end them ";
}

OaI32 OaNlpSuiteSampler::EncodeChar(char InCharacter) const {
	return InCharacter >= 'a' and InCharacter <= 'z'
		? static_cast<OaI32>(InCharacter - 'a')
		: 26;
}

OaString OaNlpSuiteSampler::DecodeChar(OaI32 InToken) const {
	if (InToken < 0 or InToken >= 26) {
		return OaString(" ");
	}
	const char character[2]{static_cast<char>('a' + InToken), '\0'};
	return OaString(character);
}

OaI32 OaNlpSuiteSampler::TokenSourceUnits(OaI32 InToken) const {
	if (Recipe_.Tokenizer() == OaNlpTokenizerKind::Bpe) {
		OaVec<OaI32> token{InToken};
		return static_cast<OaI32>(BpeTokenizer_.Decode(token).size());
	}
	return 1;
}

OaMatrix OaNlpSuiteSampler::ToMatrix(
	const OaVec<OaI32>& InTokens,
	OaI32 InBatchSize) const {
	const OaMatrixShape shape{InBatchSize, Recipe_.ContextLength()};
	return OaFnMatrix::FromInt32(
		OaSpan<const OaI32>(InTokens.Data(), InTokens.Size()),
		shape, OaScalarType::UInt32);
}
