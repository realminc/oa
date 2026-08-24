#include "oaTest.h"
#include "tutorialMl.h"
#include "tutorialNlpCommon.h"

#include <oa/ml.h>
#include <oa/ml/autograd.h>
#include <oa/core/envFlag.h>

#include <algorithm>
#include <cmath>

namespace {

constexpr oa::I32 kByteVocab = 256;

class ByteMoeLM : public oa::Module {
public:
	ByteMoeLM() {
		auto wd = oa::FnMatrix::weightDtype();
		tokEmbed_ = oa::makeShared<oa::Embedding>(kByteVocab, kDModel);
		posEmbed_ = oa::makeShared<oa::Embedding>(kContextLen, kDModel);
		block_ = oa::makeShared<oa::TransformerBlock>(kDModel, 16, kContextLen, 4, 2);
		lnFinal_ = oa::makeShared<oa::LayerNorm>(kDModel, 1e-5F);
		head_ = oa::makeShared<oa::Linear>(kDModel, kByteVocab);
		head_->parameters()[0].data = oa::FnMatrix::rand(oa::MatrixShape{kByteVocab, kDModel}, wd);
		registerModule("tok_embed", tokEmbed_); registerModule("pos_embed", posEmbed_);
		registerModule("block", block_); registerModule("ln_final", lnFinal_); registerModule("head", head_);
		for (auto* parameter : allParameterPtrs()) parameter->data.setRequiresGrad(true);
	}

	oa::Matrix forward(const oa::Matrix& inTokens) override {
		const oa::I32 batch = static_cast<oa::I32>(inTokens.size(0));
		const oa::I32 rows = batch * kContextLen;
		auto x = tokEmbed_->forward(inTokens).reshape(oa::MatrixShape{rows, kDModel});
		x = oa::FnMatrix::add(x, posEmbed_->forward(positionIds(batch)));
		return head_->forward(lnFinal_->forward(block_->forward(x)));
	}

private:
	oa::Matrix positionIds(oa::I32 inBatch) const {
		oa::Vec<oa::U8> ids(static_cast<oa::I64>(inBatch) * kContextLen);
		for (oa::I64 i = 0; i < ids.size(); ++i) ids[i] = static_cast<oa::U8>(i % kContextLen);
		return oa::FnMatrix::fromBytes(oa::Span<const oa::U8>(ids.data(), ids.size()),
			oa::MatrixShape{static_cast<oa::I64>(ids.size())}, oa::ScalarType::UInt8);
	}

	oa::SharedPtr<oa::Embedding> tokEmbed_, posEmbed_;
	oa::SharedPtr<oa::TransformerBlock> block_;
	oa::SharedPtr<oa::LayerNorm> lnFinal_;
	oa::SharedPtr<oa::Linear> head_;
};

} // namespace

TEST(TutorialNlpByteMoeAg, MoeAllPositionLM) {
	std::printf("\n╔══════════════════════════════════════════════════════════════════╗\n");
	std::printf("║  OA Tutorial — Byte MoE Transformer · all-position LM           ║\n");
	std::printf("╚══════════════════════════════════════════════════════════════════╝\n\n");
	std::printf("tokenizer: raw byte · vocab=%d\n", kByteVocab);
	std::printf("Task: dense next-byte prediction at every position\n\n");

	oa::FnMatrix::setRngSeed(oa::NlpSuiteRngSeed);
	auto model = oa::makeShared<ByteMoeLM>();
	auto parameters = model->allParameterPtrs();
	auto optimizer = oa::makeUnique<oa::AdamW>(parameters, 0.01F);
	auto& rt = testEngine();
	std::printf("Model: Byte + position Embed → Attention + moE(E=4,K=2,DFF=16) → LN → Linear(32→256)\n");
	std::printf("params: %lld    Optimizer: AdamW(lr=0.01)\n\n", static_cast<long long>(model->numParameters()));

	NlpAllPositionSampler sampler(nlpCorpus(), kBatch);
	const oa::I32 steps = static_cast<oa::I32>(
		std::max<oa::I64>(oa::EnvFlag::getInt("OA_TUTORIAL_STEPS", kSteps), 1));
	const oa::Bool useTrainingProgram = oa::EnvFlag::isSet("OA_TRAINING_PROGRAM");
	oa::TrainingProgram program;
	TutorialTrainingLoop training(rt, *optimizer, oa::ItTrainingConfig{
		.totalSteps = steps,
		.batchSize = kBatch,
		.sequenceLength = kContextLen,
		.sequenceUnit = "token",
		.sourceUnitsPerSample = kContextLen,
		.sourceUnit = "byte",
		.timerName = "byte_moe_step",
		.program = useTrainingProgram ? &program : nullptr,
	});
	std::printf("training: %d steps · batch=%d · sequence=%d byte tokens · execution=%s\n",
		steps, kBatch, kContextLen, useTrainingProgram ? "captured" : "eager");

	oa::Matrix x, y;
	oa::F32 initialLoss = 0.0F;
	while (not training.loop.isDone()) {
		training.loop.step(
			[&] { sampler.nextBatch(x, y); },
			[&] {
				optimizer->zeroGrad();
				oa::GradientTape tape;
				auto loss = oa::FnLoss::crossEntropy(
					model->forward(x), y.reshape(oa::MatrixShape{y.numElements()}));
				tape.backward(loss);
				training.loop.recordLoss(loss);
			});
		if (training.loop.index() == 1) initialLoss = training.loop.lastLoss();
	}
	ASSERT_TRUE(training.loop.finish().isOk());
	const oa::F32 finalLoss = training.loop.lastLoss();
	const oa::F32 accuracy = nlpAccuracyAllPositions(*model, x, y, kByteVocab);

	std::printf("\nEvaluation:\n");
	std::printf("  Random-loss baseline ln(%d) = %.4f\n", kByteVocab, std::log(static_cast<double>(kByteVocab)));
	std::printf("  bits/byte: %.4f\n", nlpBitsPerByte(finalLoss));
	std::printf("  Accuracy: %.1f%%\n", accuracy);
	std::printf("\nGeneration:\n  prompt: '%s'\n  generated: '%s'\n\n", kNlpGenerationPrompt,
		nlpGenerateGreedy(*model, kNlpGenerationPrompt, kNlpGenerationBytes, kByteVocab).cStr());

	ASSERT_GT(initialLoss, 0.0F);
	EXPECT_LT(finalLoss, initialLoss);
	EXPECT_GT(accuracy, 30.0F);
	ASSERT_TRUE(model->save(rt, "/tmp/byte_moe.oam", *optimizer).isOk());
	auto reloaded = oa::makeShared<ByteMoeLM>();
	auto reloadParameters = reloaded->allParameterPtrs();
	auto reloadOptimizer = oa::makeUnique<oa::AdamW>(reloadParameters, 0.01F);
	ASSERT_TRUE(reloaded->load(rt, "/tmp/byte_moe.oam", *reloadOptimizer).isOk());
	EXPECT_NEAR(nlpAccuracyAllPositions(*reloaded, x, y, kByteVocab), accuracy, 0.5F);
	EXPECT_EQ(reloadOptimizer->getStep(), optimizer->getStep());
}
