// Models and shared runner for the byte-pair NLP tutorial family.

#pragma once

#include "tutorialMl.h"
#include "tutorialNlpBpeCommon.h"

#include <oa/ml.h>
#include <oa/ml/autograd.h>

#include <stdlib.h>
inline bool nlpBpeUseMuon() {
	const char* value = ::getenv("OA_NLP_USE_MUON");
	return value != nullptr and oa::strcmp(value, "1") == 0;
}

inline oa::UniquePtr<oa::Optimizer> makeNlpBpeOptimizer(
	oa::Vector<oa::Parameter*>& inParams,
	oa::F32 inAdamWLearningRate
) {
	if (nlpBpeUseMuon()) {
		// This deliberately measures the pure oa::Muon contract. Canonical Muon
		// language-model recipes keep embeddings, heads, biases, and gains on an
		// auxiliary AdamW optimizer; OA does not hide that split inside Muon.
		return oa::makeUnique<oa::Muon>(inParams, 0.02F);
	}
	return oa::makeUnique<oa::AdamW>(inParams, inAdamWLearningRate);
}

class BpeRnnLM : public oa::Module {
public:
	BpeRnnLM() {
		auto wd = oa::FnMatrix::weightDtype();
		embed_ = oa::makeShared<oa::Embedding>(kBpeVocabSize, kDModel);
		embed_->parameters()[0].data = oa::FnMatrix::randN(oa::MatrixShape{kBpeVocabSize, kDModel}, wd);
		rnn_ = oa::makeShared<oa::Rnn>(kDModel, kHiddenDim, 1);
		head_ = oa::makeShared<oa::Linear>(kHiddenDim, kBpeVocabSize);
		head_->parameters()[0].data = oa::FnMatrix::rand(oa::MatrixShape{kBpeVocabSize, kHiddenDim}, wd);
		registerModule("embed", embed_); registerModule("rnn", rnn_); registerModule("head", head_);
		for (auto* p : allParameterPtrs()) p->data.setRequiresGrad(true);
	}
	oa::Matrix forward(const oa::Matrix& inTokens) override {
		const oa::I32 b = static_cast<oa::I32>(inTokens.size(0));
		const oa::I32 s = static_cast<oa::I32>(inTokens.size(1));
		auto e = embed_->forward(inTokens).reshape(oa::MatrixShape{b, s, kDModel});
		return head_->forward(rnn_->forward(e).reshape(oa::MatrixShape{b * s, kHiddenDim}));
	}
private:
	oa::SharedPtr<oa::Embedding> embed_; oa::SharedPtr<oa::Rnn> rnn_; oa::SharedPtr<oa::Linear> head_;
};

class BpeGruLM : public oa::Module {
public:
	BpeGruLM() {
		auto wd = oa::FnMatrix::weightDtype();
		embed_ = oa::makeShared<oa::Embedding>(kBpeVocabSize, kDModel);
		embed_->parameters()[0].data = oa::FnMatrix::randN(oa::MatrixShape{kBpeVocabSize, kDModel}, wd);
		gru_ = oa::makeShared<oa::Gru>(kDModel, kHiddenDim, 1);
		head_ = oa::makeShared<oa::Linear>(kHiddenDim, kBpeVocabSize);
		head_->parameters()[0].data = oa::FnMatrix::rand(oa::MatrixShape{kBpeVocabSize, kHiddenDim}, wd);
		registerModule("embed", embed_); registerModule("gru", gru_); registerModule("head", head_);
		for (auto* p : allParameterPtrs()) p->data.setRequiresGrad(true);
	}
	oa::Matrix forward(const oa::Matrix& inTokens) override {
		const oa::I32 b = static_cast<oa::I32>(inTokens.size(0));
		const oa::I32 s = static_cast<oa::I32>(inTokens.size(1));
		auto e = embed_->forward(inTokens).reshape(oa::MatrixShape{b, s, kDModel});
		return head_->forward(gru_->forward(e).reshape(oa::MatrixShape{b * s, kHiddenDim}));
	}
private:
	oa::SharedPtr<oa::Embedding> embed_; oa::SharedPtr<oa::Gru> gru_; oa::SharedPtr<oa::Linear> head_;
};

class BpeTransformerLM : public oa::Module {
public:
	BpeTransformerLM() {
		auto wd = oa::FnMatrix::weightDtype();
		tokEmbed_ = oa::makeShared<oa::Embedding>(kBpeVocabSize, kDModel);
		posEmbed_ = oa::makeShared<oa::Embedding>(kContextLen, kDModel);
		block_ = oa::makeShared<oa::TransformerBlock>(kDModel, kHiddenDim, kContextLen);
		lnFinal_ = oa::makeShared<oa::LayerNorm>(kDModel, 1e-5F);
		head_ = oa::makeShared<oa::Linear>(kDModel, kBpeVocabSize);
		head_->parameters()[0].data = oa::FnMatrix::rand(oa::MatrixShape{kBpeVocabSize, kDModel}, wd);
		registerModule("tok_embed", tokEmbed_); registerModule("pos_embed", posEmbed_);
		registerModule("block", block_); registerModule("ln_final", lnFinal_); registerModule("head", head_);
		for (auto* p : allParameterPtrs()) p->data.setRequiresGrad(true);
	}
	oa::Matrix forward(const oa::Matrix& inTokens) override {
		const oa::I32 b = static_cast<oa::I32>(inTokens.size(0));
		const oa::I32 n = b * kContextLen;
		auto x = tokEmbed_->forward(inTokens).reshape(oa::MatrixShape{n, kDModel}) + posEmbed_->forward(positionIds(b));
		return head_->forward(lnFinal_->forward(block_->forward(x)));
	}
private:
	oa::Matrix positionIds(oa::I32 inBatch) const {
		oa::Vector<oa::U8> ids(static_cast<oa::I64>(inBatch) * kContextLen);
		for (oa::I64 i = 0; i < static_cast<oa::I64>(inBatch) * kContextLen; ++i) ids[i] = static_cast<oa::U8>(i % kContextLen);
		return oa::FnMatrix::fromBytes(oa::Span<const oa::U8>(ids.data(), ids.size()),
			oa::MatrixShape{static_cast<oa::I64>(inBatch) * kContextLen}, oa::ScalarType::UInt8);
	}
	oa::SharedPtr<oa::Embedding> tokEmbed_, posEmbed_; oa::SharedPtr<oa::TransformerBlock> block_;
	oa::SharedPtr<oa::LayerNorm> lnFinal_; oa::SharedPtr<oa::Linear> head_;
};

class BpeMamba3LM : public oa::Module {
public:
	BpeMamba3LM() {
		embed_ = oa::makeShared<oa::Embedding>(kBpeVocabSize, kDModel);
		mamba_ = oa::makeShared<oa::Mamba3Module>(kDModel, 32, 2, 16, 1, 0.5F, false, 4,
			0.001F, 0.1F, 1e-4F, 1e-4F, true);
		head_ = oa::makeShared<oa::Linear>(kDModel, kBpeVocabSize);
		registerModule("embed", embed_); registerModule("mamba3", mamba_); registerModule("head", head_);
	}
	oa::Matrix forward(const oa::Matrix& inTokens) override {
		const oa::I64 b = inTokens.size(0), s = inTokens.size(1);
		auto flat = embed_->forward(inTokens);
		auto y = mamba_->forward(flat.reshape(oa::MatrixShape{b, s, kDModel}));
		return head_->forward(y.reshape(oa::MatrixShape{b * s, kDModel}) + flat.reshape(oa::MatrixShape{b * s, kDModel}));
	}
private:
	oa::SharedPtr<oa::Embedding> embed_; oa::SharedPtr<oa::Mamba3Module> mamba_; oa::SharedPtr<oa::Linear> head_;
};

class BpeMoeLM : public oa::Module {
public:
	BpeMoeLM() {
		auto wd = oa::FnMatrix::weightDtype();
		tokEmbed_ = oa::makeShared<oa::Embedding>(kBpeVocabSize, kDModel);
		posEmbed_ = oa::makeShared<oa::Embedding>(kContextLen, kDModel);
		block_ = oa::makeShared<oa::TransformerBlock>(kDModel, 16, kContextLen, 4, 2);
		lnFinal_ = oa::makeShared<oa::LayerNorm>(kDModel, 1e-5F);
		head_ = oa::makeShared<oa::Linear>(kDModel, kBpeVocabSize);
		head_->parameters()[0].data = oa::FnMatrix::rand(oa::MatrixShape{kBpeVocabSize, kDModel}, wd);
		registerModule("tok_embed", tokEmbed_); registerModule("pos_embed", posEmbed_);
		registerModule("block", block_); registerModule("ln_final", lnFinal_); registerModule("head", head_);
		for (auto* p : allParameterPtrs()) p->data.setRequiresGrad(true);
	}
	oa::Matrix forward(const oa::Matrix& inTokens) override {
		const oa::I32 b = static_cast<oa::I32>(inTokens.size(0));
		const oa::I32 n = b * kContextLen;
		auto x = tokEmbed_->forward(inTokens).reshape(oa::MatrixShape{n, kDModel}) + posEmbed_->forward(positionIds(b));
		return head_->forward(lnFinal_->forward(block_->forward(x)));
	}
private:
	oa::Matrix positionIds(oa::I32 inBatch) const {
		oa::Vector<oa::U8> ids(static_cast<oa::I64>(inBatch) * kContextLen);
		for (oa::I64 i = 0; i < static_cast<oa::I64>(inBatch) * kContextLen; ++i) ids[i] = static_cast<oa::U8>(i % kContextLen);
		return oa::FnMatrix::fromBytes(oa::Span<const oa::U8>(ids.data(), ids.size()),
			oa::MatrixShape{static_cast<oa::I64>(inBatch) * kContextLen}, oa::ScalarType::UInt8);
	}
	oa::SharedPtr<oa::Embedding> tokEmbed_, posEmbed_; oa::SharedPtr<oa::TransformerBlock> block_;
	oa::SharedPtr<oa::LayerNorm> lnFinal_; oa::SharedPtr<oa::Linear> head_;
};

template <class Model>
void runNlpBpeTutorial(const char* inTitle, const char* inModelDescription,
	const char* inTimerName, const char* inCheckpointPath, oa::F32 inLearningRate) {
	auto& rt = testEngine();
	NlpBpeTokenizer tokenizer(nlpCorpus(), kBpeVocabSize);
	const auto corpusTokens = tokenizer.encode(nlpCorpus());
	ASSERT_EQ(tokenizer.decode(corpusTokens), oa::String(nlpCorpus()));
	const oa::I64 corpusBytes = static_cast<oa::I64>(oa::strlen(nlpCorpus()));
	const oa::F64 corpusBytesPerToken = static_cast<oa::F64>(corpusBytes) / static_cast<oa::F64>(corpusTokens.size());

	oa::print("\n╔══════════════════════════════════════════════════════════════════╗");
	oa::print("║  {:<62}║", inTitle);
	oa::print("╚══════════════════════════════════════════════════════════════════╝\n");
	oa::print("tokenizer: byte BPE · vocab={} (256 bytes + {} merges)", tokenizer.vocabSize(), tokenizer.mergeCount());
	oa::print("compression: {} bytes → {} tokens · {:.3f} byte/token · {:.1f}% fewer positions",
		static_cast<long long>(corpusBytes), corpusTokens.size(), corpusBytesPerToken,
		100.0 * (1.0 - static_cast<oa::F64>(corpusTokens.size()) / static_cast<oa::F64>(corpusBytes)));
	oa::print("context coverage: {} BPE tokens ≈ {:.1f} source bytes at corpus average",
		kContextLen, static_cast<oa::F64>(kContextLen) * corpusBytesPerToken);
	oa::print("Task: dense next-token at every position · source throughput measured as exact byte/s\n");

	auto model = oa::makeShared<Model>();
	auto params = model->allParameterPtrs();
	auto opt = makeNlpBpeOptimizer(params, inLearningRate);
	oa::print("Model: {}", inModelDescription);
	if (nlpBpeUseMuon()) {
		oa::print("params: {}    Optimizer: Muon(lr=0.02, all-parameter experiment)\n",
			static_cast<long long>(model->numParameters()));
	} else {
		oa::print("params: {}    Optimizer: AdamW(lr={:.3g})\n",
			static_cast<long long>(model->numParameters()), static_cast<double>(inLearningRate));
	}

	NlpBpeAllPositionSampler sampler(nlpCorpus(), kBatch, tokenizer);
	TutorialTrainingLoop training(rt, *opt, oa::ItTrainingConfig{
		.totalSteps = kSteps,
		.batchSize = kBatch,
		.sequenceLength = kContextLen,
		.sequenceUnit = "token",
		.sourceUnit = "byte",
		.timerName = inTimerName,
	});
	oa::print("training: {} steps · batch={} · sequence={} BPE tokens", kSteps, kBatch, kContextLen);

	oa::Matrix x, y;
	oa::F32 initialLoss = 0.0F;
	while (not training.loop.isDone()) {
		sampler.nextBatch(x, y);
		training.loop.recordSourceUnits(sampler.lastBatchBytes());
		opt->zeroGrad();
		oa::GradientTape tape;
		auto logits = model->forward(x);
		auto loss = oa::FnLoss::crossEntropy(logits, y.reshape(oa::MatrixShape{y.numElements()}));
		tape.backward(loss);
		training.loop.next(loss);
		if (training.loop.index() == 1) initialLoss = training.loop.lastLoss();
	}
	ASSERT_TRUE(training.loop.finish().isOk());
	const oa::F32 finalLoss = training.loop.lastLoss();
	const oa::F32 accuracy = nlpAccuracyAllPositions(*model, x, y, tokenizer.vocabSize());
	const oa::F64 finalBytesPerToken = sampler.lastBatchBytesPerToken();

	oa::print("\nEvaluation:");
	oa::print("  Random-loss baseline ln({}) = {:.4f}", tokenizer.vocabSize(), oa::log(static_cast<double>(tokenizer.vocabSize())));
	oa::print("  Final batch: {:.3f} byte/token · {:.4f} bits/byte",
		finalBytesPerToken, nlpBitsPerByte(finalLoss, finalBytesPerToken));
	oa::print("  Token accuracy: {:.1f}% (compare within BPE only)", accuracy);
	oa::print("\nGeneration:\n  prompt: '{}'\n  generated: '{}'\n",
		kNlpGenerationPrompt,
		nlpGenerateBpeGreedy(*model, tokenizer, kNlpGenerationPrompt,
			kNlpGenerationBytes).cStr());

	ASSERT_GT(initialLoss, 0.0F);
	EXPECT_LT(finalLoss, initialLoss);
	EXPECT_GT(accuracy, 30.0F);
	ASSERT_TRUE(model->save(rt, inCheckpointPath, *opt).isOk());
	auto reloaded = oa::makeShared<Model>();
	auto reloadParams = reloaded->allParameterPtrs();
	auto reloadOpt = makeNlpBpeOptimizer(reloadParams, inLearningRate);
	ASSERT_TRUE(reloaded->load(rt, inCheckpointPath, *reloadOpt).isOk());
	EXPECT_NEAR(nlpAccuracyAllPositions(*reloaded, x, y, tokenizer.vocabSize()), accuracy, 0.5F);
	EXPECT_EQ(reloadOpt->getStep(), opt->getStep());
}
