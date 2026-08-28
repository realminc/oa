#include <ml/nlpSuite.h>

#include <oa/core/fnMatrix.h>
#include <oa/ml/byte.h>

#include <algorithm>
#include <cstring>
#include <stdexcept>

oa::NlpSuiteRecipe::NlpSuiteRecipe(
	oa::NlpArchitecture inArchitecture,
	oa::NlpTokenizerKind inTokenizer)
	: architecture_(inArchitecture)
	, tokenizer_(inTokenizer) {
}

oa::I32 oa::NlpSuiteRecipe::vocabSize() const {
	switch (tokenizer_) {
	case oa::NlpTokenizerKind::Byte:
		return 256;
	case oa::NlpTokenizerKind::Bpe:
		return 320;
	case oa::NlpTokenizerKind::Char:
		return 27;
	}
	return 256;
}

oa::F32 oa::NlpSuiteRecipe::learningRate() const {
	return architecture_ == oa::NlpArchitecture::Mamba3 ? 0.003F : 0.01F;
}

const char* oa::NlpSuiteRecipe::architectureId() const {
	switch (architecture_) {
	case oa::NlpArchitecture::Rnn:
		return "rnn";
	case oa::NlpArchitecture::Gru:
		return "gru";
	case oa::NlpArchitecture::Transformer:
		return "transformer";
	case oa::NlpArchitecture::MoeTransformer:
		return "moe";
	case oa::NlpArchitecture::Mamba3:
		return "mamba3";
	}
	return "gru";
}

const char* oa::NlpSuiteRecipe::architectureName() const {
	switch (architecture_) {
	case oa::NlpArchitecture::Rnn:
		return "RNN";
	case oa::NlpArchitecture::Gru:
		return "GRU";
	case oa::NlpArchitecture::Transformer:
		return "Transformer";
	case oa::NlpArchitecture::MoeTransformer:
		return "MoE Transformer";
	case oa::NlpArchitecture::Mamba3:
		return "Mamba-3";
	}
	return "GRU";
}

const char* oa::NlpSuiteRecipe::tokenizerId() const {
	switch (tokenizer_) {
	case oa::NlpTokenizerKind::Byte:
		return "byte";
	case oa::NlpTokenizerKind::Bpe:
		return "bpe";
	case oa::NlpTokenizerKind::Char:
		return "char";
	}
	return "byte";
}

const char* oa::NlpSuiteRecipe::tokenizerName() const {
	switch (tokenizer_) {
	case oa::NlpTokenizerKind::Byte:
		return "Byte";
	case oa::NlpTokenizerKind::Bpe:
		return "BPE";
	case oa::NlpTokenizerKind::Char:
		return "Char";
	}
	return "Byte";
}

const char* oa::NlpSuiteRecipe::modelDescription() const {
	switch (architecture_) {
	case oa::NlpArchitecture::Rnn:
		return "Embedding(32) -> RNN(32x64) -> Linear(vocab)";
	case oa::NlpArchitecture::Gru:
		return "Embedding(32) -> GRU(32x64) -> Linear(vocab)";
	case oa::NlpArchitecture::Transformer:
		return "Embedding(32) -> transformer(FFN=64) -> Linear(vocab)";
	case oa::NlpArchitecture::MoeTransformer:
		return "Embedding(32) -> moE(E=4,K=2,DFF=16) -> Linear(vocab)";
	case oa::NlpArchitecture::Mamba3:
		return "Embedding(32) -> Mamba-3(state=32) -> Linear(vocab)";
	}
	return "Embedding(32) -> GRU(32x64) -> Linear(vocab)";
}

const char* oa::NlpSuiteRecipe::timerName() const {
	switch (architecture_) {
	case oa::NlpArchitecture::Rnn:
		return "nlp_rnn_step";
	case oa::NlpArchitecture::Gru:
		return "nlp_gru_step";
	case oa::NlpArchitecture::Transformer:
		return "nlp_transformer_step";
	case oa::NlpArchitecture::MoeTransformer:
		return "nlp_moe_step";
	case oa::NlpArchitecture::Mamba3:
		return "nlp_mamba3_step";
	}
	return "nlp_gru_step";
}

oa::NlpSuiteModel::NlpSuiteModel(const oa::NlpSuiteRecipe& inRecipe)
	: recipe_(inRecipe) {
	const oa::I32 vocabSize = recipe_.vocabSize();
	const oa::I32 modelWidth = recipe_.modelWidth();
	const oa::I32 hiddenWidth = recipe_.hiddenWidth();
	const auto weightDtype = oa::FnMatrix::weightDtype();

	if (recipe_.tokenizer() == oa::NlpTokenizerKind::Byte and
		(recipe_.architecture() == oa::NlpArchitecture::Rnn or
		 recipe_.architecture() == oa::NlpArchitecture::Gru)) {
		auto embedding = oa::makeShared<oa::ByteEmbedding>(modelWidth);
		embedding->parameters()[0].data =
			oa::FnMatrix::randN(oa::MatrixShape{vocabSize, modelWidth}, weightDtype);
		tokenEmbedding_ = embedding;
	} else {
		auto embedding = oa::makeShared<oa::Embedding>(vocabSize, modelWidth);
		embedding->parameters()[0].data =
			oa::FnMatrix::randN(oa::MatrixShape{vocabSize, modelWidth}, weightDtype);
		tokenEmbedding_ = embedding;
	}
	registerModule("embed", tokenEmbedding_);

	switch (recipe_.architecture()) {
	case oa::NlpArchitecture::Rnn:
		rnn_ = oa::makeShared<oa::Rnn>(modelWidth, hiddenWidth, 1);
		head_ = oa::makeShared<oa::Linear>(hiddenWidth, vocabSize);
		head_->parameters()[0].data =
			oa::FnMatrix::rand(oa::MatrixShape{vocabSize, hiddenWidth}, weightDtype);
		registerModule("rnn", rnn_);
		break;
	case oa::NlpArchitecture::Gru:
		gru_ = oa::makeShared<oa::Gru>(modelWidth, hiddenWidth, 1);
		head_ = oa::makeShared<oa::Linear>(hiddenWidth, vocabSize);
		// Preserve the desktop tutorial recipe exactly. The compact character
		// model uses Xavier here; byte and BPE use the historical uniform head.
		head_->parameters()[0].data = recipe_.tokenizer() == oa::NlpTokenizerKind::Char
			? oa::FnMatrix::randXavier(
				oa::MatrixShape{vocabSize, hiddenWidth}, weightDtype)
			: oa::FnMatrix::rand(
				oa::MatrixShape{vocabSize, hiddenWidth}, weightDtype);
		registerModule("gru", gru_);
		break;
	case oa::NlpArchitecture::Transformer:
	case oa::NlpArchitecture::MoeTransformer:
		positionEmbedding_ = oa::makeShared<oa::Embedding>(recipe_.contextLength(), modelWidth);
		transformer_ = recipe_.architecture() == oa::NlpArchitecture::MoeTransformer
			? oa::makeShared<oa::TransformerBlock>(modelWidth, 16, recipe_.contextLength(), 4, 2)
			: oa::makeShared<oa::TransformerBlock>(modelWidth, hiddenWidth, recipe_.contextLength());
		finalNorm_ = oa::makeShared<oa::LayerNorm>(modelWidth, 1e-5F);
		head_ = oa::makeShared<oa::Linear>(modelWidth, vocabSize);
		head_->parameters()[0].data =
			oa::FnMatrix::rand(oa::MatrixShape{vocabSize, modelWidth}, weightDtype);
		registerModule("pos_embed", positionEmbedding_);
		registerModule("block", transformer_);
		registerModule("ln_final", finalNorm_);
		break;
	case oa::NlpArchitecture::Mamba3:
		mamba3_ = oa::makeShared<oa::Mamba3Module>(
			modelWidth, 32, 2, 16, 1, 0.5F, false, 4,
			0.001F, 0.1F, 1e-4F, 1e-4F, true);
		head_ = oa::makeShared<oa::Linear>(modelWidth, vocabSize);
		registerModule("mamba3", mamba3_);
		break;
	}
	registerModule("head", head_);

	for (auto* parameter : allParameterPtrs()) {
		parameter->data.setRequiresGrad(true);
	}
}

oa::Matrix oa::NlpSuiteModel::forward(const oa::Matrix& inTokens) {
	const oa::I32 batch = static_cast<oa::I32>(inTokens.size(0));
	const oa::I32 sequence = static_cast<oa::I32>(inTokens.size(1));
	const oa::I32 modelWidth = recipe_.modelWidth();
	const oa::I32 hiddenWidth = recipe_.hiddenWidth();
	const oa::I64 rows = static_cast<oa::I64>(batch) * sequence;
	auto embedded = tokenEmbedding_->forward(inTokens);

	switch (recipe_.architecture()) {
	case oa::NlpArchitecture::Rnn:
		return head_->forward(rnn_->forward(
			embedded.reshape(oa::MatrixShape{batch, sequence, modelWidth}))
			.reshape(oa::MatrixShape{rows, hiddenWidth}));
	case oa::NlpArchitecture::Gru:
		return head_->forward(gru_->forward(
			embedded.reshape(oa::MatrixShape{batch, sequence, modelWidth}))
			.reshape(oa::MatrixShape{rows, hiddenWidth}));
	case oa::NlpArchitecture::Transformer:
	case oa::NlpArchitecture::MoeTransformer: {
		auto positioned = embedded.reshape(oa::MatrixShape{rows, modelWidth}) +
			positionEmbedding_->forward(positionIds(batch, sequence));
		return head_->forward(finalNorm_->forward(transformer_->forward(positioned)));
	}
	case oa::NlpArchitecture::Mamba3: {
		auto sequenceOutput = mamba3_->forward(
			embedded.reshape(oa::MatrixShape{batch, sequence, modelWidth}));
		return head_->forward(
			sequenceOutput.reshape(oa::MatrixShape{rows, modelWidth}) +
			embedded.reshape(oa::MatrixShape{rows, modelWidth}));
	}
	}
	return {};
}

bool oa::NlpSuiteModel::supportsStatefulGeneration() const {
	return recipe_.architecture() == oa::NlpArchitecture::Rnn
		or recipe_.architecture() == oa::NlpArchitecture::Gru
		or recipe_.architecture() == oa::NlpArchitecture::Mamba3;
}

void oa::NlpSuiteModel::resetGenerationState(oa::I32 inBatch) {
	switch (recipe_.architecture()) {
	case oa::NlpArchitecture::Rnn:
		rnnGenerationHidden_ = rnn_->zeroState(inBatch);
		return;
	case oa::NlpArchitecture::Gru:
		gruGenerationHidden_ = gru_->zeroState(inBatch);
		return;
	case oa::NlpArchitecture::Mamba3:
		mamba3_->resetState(inBatch);
		return;
	default:
		throw std::logic_error(
			"resetGenerationState requires an RNN, GRU, or Mamba-3 NLP recipe");
	}
}

oa::Matrix oa::NlpSuiteModel::forwardGenerationStep(const oa::Matrix& inToken) {
	if (not supportsStatefulGeneration()) {
		throw std::logic_error(
			"forwardGenerationStep requires an RNN, GRU, or Mamba-3 NLP recipe");
	}
	if (inToken.rank() != 2 or inToken.size(1) != 1) {
		throw std::invalid_argument(
			"forwardGenerationStep expects token ids with shape [batch, 1]");
	}
	const oa::I64 batch = inToken.size(0);
	auto embedded = tokenEmbedding_->forward(inToken)
		.reshape(oa::MatrixShape{batch, recipe_.modelWidth()});
	switch (recipe_.architecture()) {
	case oa::NlpArchitecture::Rnn:
		return head_->forward(rnn_->step(embedded, rnnGenerationHidden_));
	case oa::NlpArchitecture::Gru:
		return head_->forward(gru_->step(embedded, gruGenerationHidden_));
	case oa::NlpArchitecture::Mamba3: {
		auto sequenceOutput = mamba3_->step(
			embedded.reshape(oa::MatrixShape{batch, 1, recipe_.modelWidth()}));
		return head_->forward(
			sequenceOutput.reshape(oa::MatrixShape{batch, recipe_.modelWidth()}) +
			embedded);
	}
	default:
		break;
	}
	return {};
}

oa::Matrix oa::NlpSuiteModel::positionIds(oa::I32 inBatch, oa::I32 inSequence) const {
	const oa::Usize count = static_cast<oa::Usize>(inBatch) *
		static_cast<oa::Usize>(inSequence);
	oa::Vector<oa::I32> ids(count);
	for (oa::Usize index = 0; index < ids.size(); ++index) {
		ids[index] = static_cast<oa::I32>(index % static_cast<oa::Usize>(inSequence));
	}
	return oa::FnMatrix::fromInt32(
		oa::Span<const oa::I32>(ids.data(), ids.size()),
		oa::MatrixShape{static_cast<oa::I64>(ids.size())},
		oa::ScalarType::UInt32);
}

oa::NlpSuiteSampler::NlpSuiteSampler(
	const oa::NlpSuiteRecipe& inRecipe,
	oa::I32 inBatchSize)
	: recipe_(inRecipe)
	, batchSize_(std::max(1, inBatchSize)) {
	if (recipe_.tokenizer() == oa::NlpTokenizerKind::Bpe) {
		bpeTokenizer_.train(corpus(), 64);
	}
	tokens_ = encode(corpus());
	tokenSourceUnits_.resize(static_cast<oa::Usize>(recipe_.vocabSize()));
	for (oa::I32 token = 0; token < recipe_.vocabSize(); ++token) {
		tokenSourceUnits_[static_cast<oa::Usize>(token)] = tokenSourceUnits(token);
	}
}

void oa::NlpSuiteSampler::next(oa::Matrix& outInput, oa::Matrix& outTarget) {
	const oa::I32 contextLength = recipe_.contextLength();
	const oa::Usize count = static_cast<oa::Usize>(batchSize_) *
		static_cast<oa::Usize>(contextLength);
	oa::Vector<oa::I32> input(count);
	oa::Vector<oa::I32> target(count);
	const oa::I64 limit = static_cast<oa::I64>(tokens_.size()) - contextLength - 1;
	lastSourceUnits_ = 0;
	for (oa::I32 batch = 0; batch < batchSize_; ++batch) {
		const oa::I64 start = (cursor_ + static_cast<oa::I64>(batch) * 7) % limit;
		for (oa::I32 position = 0; position < contextLength; ++position) {
			const oa::Usize outputIndex = static_cast<oa::Usize>(batch) *
				static_cast<oa::Usize>(contextLength) + static_cast<oa::Usize>(position);
			const oa::Usize inputIndex = static_cast<oa::Usize>(start + position);
			const oa::Usize targetIndex = static_cast<oa::Usize>(start + position + 1);
			input[outputIndex] = tokens_[inputIndex];
			target[outputIndex] = tokens_[targetIndex];
			lastSourceUnits_ += tokenSourceUnits_[
				static_cast<oa::Usize>(target[outputIndex])];
		}
	}
	cursor_ = (cursor_ + batchSize_) % limit;
	outInput = toMatrix(input, batchSize_);
	outTarget = toMatrix(target, batchSize_);
}

oa::Vector<oa::I32> oa::NlpSuiteSampler::encode(const char* inText) const {
	if (recipe_.tokenizer() == oa::NlpTokenizerKind::Bpe) {
		return bpeTokenizer_.encode(inText);
	}
	const oa::Usize length = std::strlen(inText);
	oa::Vector<oa::I32> tokens(length);
	for (oa::Usize index = 0; index < length; ++index) {
		tokens[index] = recipe_.tokenizer() == oa::NlpTokenizerKind::Char
			? encodeChar(inText[index])
			: static_cast<oa::U8>(inText[index]);
	}
	return tokens;
}

oa::String oa::NlpSuiteSampler::decode(const oa::Vector<oa::I32>& inTokens) const {
	if (recipe_.tokenizer() == oa::NlpTokenizerKind::Bpe) {
		return bpeTokenizer_.decode(inTokens);
	}
	oa::String text;
	for (const oa::I32 token : inTokens) {
		if (recipe_.tokenizer() == oa::NlpTokenizerKind::Char) {
			text += decodeChar(token);
		} else {
			text += static_cast<char>(static_cast<oa::U8>(token));
		}
	}
	return text;
}

oa::Matrix oa::NlpSuiteSampler::inputMatrix(const oa::Vector<oa::I32>& inTokens) const {
	const oa::I64 contextLength = recipe_.contextLength();
	const oa::I32 batchSize = static_cast<oa::I32>(
		static_cast<oa::I64>(inTokens.size()) / contextLength);
	return toMatrix(inTokens, batchSize);
}

oa::Matrix oa::NlpSuiteSampler::inputStepMatrix(oa::I32 inToken) const {
	return oa::FnMatrix::fromInt32(
		oa::Span<const oa::I32>(&inToken, 1),
		oa::MatrixShape{1, 1},
		oa::ScalarType::UInt32);
}

const char* oa::NlpSuiteSampler::corpus() {
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

oa::I32 oa::NlpSuiteSampler::encodeChar(char inCharacter) const {
	return inCharacter >= 'a' and inCharacter <= 'z'
		? static_cast<oa::I32>(inCharacter - 'a')
		: 26;
}

oa::String oa::NlpSuiteSampler::decodeChar(oa::I32 inToken) const {
	if (inToken < 0 or inToken >= 26) {
		return oa::String(" ");
	}
	const char character[2]{static_cast<char>('a' + inToken), '\0'};
	return oa::String(character);
}

oa::I32 oa::NlpSuiteSampler::tokenSourceUnits(oa::I32 inToken) const {
	if (recipe_.tokenizer() == oa::NlpTokenizerKind::Bpe) {
		oa::Vector<oa::I32> token{inToken};
		return static_cast<oa::I32>(bpeTokenizer_.decode(token).size());
	}
	return 1;
}

oa::Matrix oa::NlpSuiteSampler::toMatrix(
	const oa::Vector<oa::I32>& inTokens,
	oa::I32 inBatchSize) const {
	const oa::MatrixShape shape{inBatchSize, recipe_.contextLength()};
	return oa::FnMatrix::fromInt32(
		oa::Span<const oa::I32>(inTokens.data(), inTokens.size()),
		shape, oa::ScalarType::UInt32);
}
