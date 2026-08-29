// Models and shared runner for the 27-symbol character NLP tutorial family.

#pragma once

#include "tutorialMl.h"
#include "tutorialNlpCommon.h"

#include <oa/ml.h>
#include <oa/ml/autograd.h>


class CharRnnLM : public oa::Module {
public:
	CharRnnLM() {
		auto wd = oa::FnMatrix::weightDtype();
		embed_ = oa::makeShared<oa::Embedding>(kCharVocabSize, kDModel);
		embed_->parameters()[0].data = oa::FnMatrix::randN(oa::MatrixShape{kCharVocabSize, kDModel}, wd);
		rnn_ = oa::makeShared<oa::Rnn>(kDModel, kHiddenDim, 1);
		head_ = oa::makeShared<oa::Linear>(kHiddenDim, kCharVocabSize);
		head_->parameters()[0].data = oa::FnMatrix::rand(oa::MatrixShape{kCharVocabSize, kHiddenDim}, wd);
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

class CharGruLM : public oa::Module {
public:
	CharGruLM() {
		auto wd = oa::FnMatrix::weightDtype();
		embed_ = oa::makeShared<oa::Embedding>(kCharVocabSize, kDModel);
		embed_->parameters()[0].data = oa::FnMatrix::randN(oa::MatrixShape{kCharVocabSize, kDModel}, wd);
		gru_ = oa::makeShared<oa::Gru>(kDModel, kHiddenDim, 1);
		head_ = oa::makeShared<oa::Linear>(kHiddenDim, kCharVocabSize);
		head_->parameters()[0].data = oa::FnMatrix::randXavier(oa::MatrixShape{kCharVocabSize, kHiddenDim}, wd);
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

class CharTransformerLM : public oa::Module {
public:
	CharTransformerLM() {
		auto wd = oa::FnMatrix::weightDtype();
		tokEmbed_ = oa::makeShared<oa::Embedding>(kCharVocabSize, kDModel);
		posEmbed_ = oa::makeShared<oa::Embedding>(kContextLen, kDModel);
		block_ = oa::makeShared<oa::TransformerBlock>(kDModel, kHiddenDim, kContextLen);
		lnFinal_ = oa::makeShared<oa::LayerNorm>(kDModel, 1e-5F);
		head_ = oa::makeShared<oa::Linear>(kDModel, kCharVocabSize);
		head_->parameters()[0].data = oa::FnMatrix::rand(oa::MatrixShape{kCharVocabSize, kDModel}, wd);
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

class CharMamba3LM : public oa::Module {
public:
	CharMamba3LM() {
		embed_ = oa::makeShared<oa::Embedding>(kCharVocabSize, kDModel);
		mamba_ = oa::makeShared<oa::Mamba3Module>(kDModel, 32, 2, 16, 1, 0.5F, false, 4,
			0.001F, 0.1F, 1e-4F, 1e-4F, true);
		head_ = oa::makeShared<oa::Linear>(kDModel, kCharVocabSize);
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

class CharMoeLM : public oa::Module {
public:
	CharMoeLM() {
		auto wd = oa::FnMatrix::weightDtype();
		tokEmbed_ = oa::makeShared<oa::Embedding>(kCharVocabSize, kDModel);
		posEmbed_ = oa::makeShared<oa::Embedding>(kContextLen, kDModel);
		// Four experts with top-2 routing. DFF=16 gives 1.5x the dense FFN's
		// parameter capacity but only 0.75x its active projection work once the
		// sparse grouped-GEMM path replaces the current dense oracle.
		block_ = oa::makeShared<oa::TransformerBlock>(kDModel, 16, kContextLen, 4, 2);
		lnFinal_ = oa::makeShared<oa::LayerNorm>(kDModel, 1e-5F);
		head_ = oa::makeShared<oa::Linear>(kDModel, kCharVocabSize);
		head_->parameters()[0].data = oa::FnMatrix::rand(oa::MatrixShape{kCharVocabSize, kDModel}, wd);
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
void runNlpCharTutorial(const char* inTitle, const char* inModelDescription,
	const char* inTimerName, const char* inCheckpointPath, oa::F32 inLearningRate) {
	auto& rt = testEngine();
	oa::print("\n╔══════════════════════════════════════════════════════════════════╗");
	oa::print("║  {:<62}║", inTitle);
	oa::print("╚══════════════════════════════════════════════════════════════════╝\n");
	oa::print("tokenizer: character · vocab={} (a-z + space)", kCharVocabSize);
	oa::print("Task: dense next-character prediction at every position\n");

	auto model = oa::makeShared<Model>();
	auto params = model->allParameterPtrs();
	auto opt = oa::makeUnique<oa::AdamW>(params, inLearningRate);
	oa::print("Model: {}", inModelDescription);
	oa::print("params: {}    Optimizer: AdamW(lr={:.3g})\n",
		static_cast<long long>(model->numParameters()), static_cast<double>(inLearningRate));

	NlpAllPositionSampler sampler(nlpCorpus(), kBatch, nlpCharEncode);
	TutorialTrainingLoop training(rt, *opt, oa::ItTrainingConfig{
		.totalSteps = kSteps,
		.epochSteps = {},
		.batchSize = kBatch,
		.sequenceLength = kContextLen,
		.sequenceUnit = "token",
		.timerName = inTimerName,
	});
	oa::print("training: {} steps · batch={} · sequence={} character tokens", kSteps, kBatch, kContextLen);

	oa::Matrix x, y;
	oa::F32 initialLoss = 0.0F;
	while (not training.loop.isDone()) {
		sampler.nextBatch(x, y);
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
	const oa::F32 accuracy = nlpAccuracyAllPositions(*model, x, y, kCharVocabSize);

	oa::print("\nEvaluation:");
	oa::print("  Random-loss baseline ln({}) = {:.4f}", kCharVocabSize, oa::log(static_cast<double>(kCharVocabSize)));
	oa::print("  character-token accuracy: {:.1f}% (compare within Char only)", accuracy);
	oa::print("\nGeneration:\n  prompt: '{}'\n  generated: '{}'\n",
		kNlpGenerationPrompt,
		nlpGenerateGreedy(*model, kNlpGenerationPrompt, kNlpGenerationBytes,
			kCharVocabSize, nlpCharEncode).cStr());

	ASSERT_GT(initialLoss, 0.0F);
	EXPECT_LT(finalLoss, initialLoss);
	EXPECT_GT(accuracy, 50.0F);
	ASSERT_TRUE(model->save(rt, inCheckpointPath, *opt).isOk());
	auto reloaded = oa::makeShared<Model>();
	auto reloadParams = reloaded->allParameterPtrs();
	auto reloadOpt = oa::makeUnique<oa::AdamW>(reloadParams, inLearningRate);
	ASSERT_TRUE(reloaded->load(rt, inCheckpointPath, *reloadOpt).isOk());
	EXPECT_NEAR(nlpAccuracyAllPositions(*reloaded, x, y, kCharVocabSize), accuracy, 0.5F);
	EXPECT_EQ(reloadOpt->getStep(), opt->getStep());
}
