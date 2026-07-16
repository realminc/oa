#include "../../Test/OaTest.h"
#include "TutorialMl.h"
#include "TutorialNlpCommon.h"

#include <Oa/Ml.h>
#include <Oa/Ml/Autograd.h>
#include <Oa/Core/EnvFlag.h>

#include <algorithm>
#include <cmath>

namespace {

constexpr OaI32 kByteVocab = 256;

class OaByteMoeLM : public OaModule {
public:
	OaByteMoeLM() {
		auto wd = OaFnMatrix::GetWeightDtype();
		TokEmbed_ = OaMakeSharedPtr<OaEmbedding>(kByteVocab, kDModel);
		PosEmbed_ = OaMakeSharedPtr<OaEmbedding>(kContextLen, kDModel);
		Block_ = OaMakeSharedPtr<OaTransformerBlock>(kDModel, 16, kContextLen, 4, 2);
		LnFinal_ = OaMakeSharedPtr<OaLayerNorm>(kDModel, 1e-5F);
		Head_ = OaMakeSharedPtr<OaLinear>(kDModel, kByteVocab);
		Head_->Parameters()[0].Data = OaFnMatrix::Rand(OaMatrixShape{kByteVocab, kDModel}, wd);
		RegisterModule("tok_embed", TokEmbed_); RegisterModule("pos_embed", PosEmbed_);
		RegisterModule("block", Block_); RegisterModule("ln_final", LnFinal_); RegisterModule("head", Head_);
		for (auto* parameter : AllParameterPtrs()) parameter->Data.SetRequiresGrad(true);
	}

	OaMatrix Forward(const OaMatrix& InTokens) override {
		const OaI32 batch = static_cast<OaI32>(InTokens.Size(0));
		const OaI32 rows = batch * kContextLen;
		auto x = TokEmbed_->Forward(InTokens).Reshape(OaMatrixShape{rows, kDModel});
		x = OaFnMatrix::Add(x, PosEmbed_->Forward(PositionIds(batch)));
		return Head_->Forward(LnFinal_->Forward(Block_->Forward(x)));
	}

private:
	OaMatrix PositionIds(OaI32 InBatch) const {
		OaVec<OaU8> ids(static_cast<OaI64>(InBatch) * kContextLen);
		for (OaI64 i = 0; i < ids.Size(); ++i) ids[i] = static_cast<OaU8>(i % kContextLen);
		return OaFnMatrix::FromBytes(OaSpan<const OaU8>(ids.Data(), ids.Size()),
			OaMatrixShape{static_cast<OaI64>(ids.Size())}, OaScalarType::UInt8);
	}

	OaSharedPtr<OaEmbedding> TokEmbed_, PosEmbed_;
	OaSharedPtr<OaTransformerBlock> Block_;
	OaSharedPtr<OaLayerNorm> LnFinal_;
	OaSharedPtr<OaLinear> Head_;
};

} // namespace

TEST(TutorialNlpByteMoeAg, MoeAllPositionLM) {
	std::printf("\n╔══════════════════════════════════════════════════════════════════╗\n");
	std::printf("║  OA Tutorial — Byte MoE Transformer · all-position LM           ║\n");
	std::printf("╚══════════════════════════════════════════════════════════════════╝\n\n");
	std::printf("Tokenizer: raw byte · vocab=%d\n", kByteVocab);
	std::printf("Task: dense next-byte prediction at every position\n\n");

	OaFnMatrix::SetRngSeed(OaNlpSuiteRngSeed);
	auto model = OaMakeSharedPtr<OaByteMoeLM>();
	auto parameters = model->AllParameterPtrs();
	auto optimizer = OaMakeUniquePtr<OaAdamW>(parameters, 0.01F);
	std::printf("Model: Byte + position Embed → Attention + MoE(E=4,K=2,DFF=16) → LN → Linear(32→256)\n");
	std::printf("Params: %lld    Optimizer: AdamW(lr=0.01)\n\n", static_cast<long long>(model->NumParameters()));

	NlpAllPositionSampler sampler(NlpCorpus(), kBatch);
	const OaI32 steps = static_cast<OaI32>(
		std::max<OaI64>(OaEnvFlag::GetInt("OA_TUTORIAL_STEPS", kSteps), 1));
	const OaBool useTrainingProgram = OaEnvFlag::IsSet("OA_TRAINING_PROGRAM");
	OaTrainingProgram program;
	TutorialTrainingLoop training(*optimizer, OaItTrainingConfig{
		.TotalSteps = steps,
		.BatchSize = kBatch,
		.SequenceLength = kContextLen,
		.SequenceUnit = "token",
		.SourceUnitsPerSample = kContextLen,
		.SourceUnit = "byte",
		.TimerName = "byte_moe_step",
		.Program = useTrainingProgram ? &program : nullptr,
	});
	std::printf("Training: %d steps · batch=%d · sequence=%d byte tokens · execution=%s\n",
		steps, kBatch, kContextLen, useTrainingProgram ? "captured" : "eager");

	OaMatrix x, y;
	OaF32 initialLoss = 0.0F;
	while (not training.Loop.IsDone()) {
		training.Loop.Step(
			[&] { sampler.NextBatch(x, y); },
			[&] {
				optimizer->ZeroGrad();
				OaGradientTape tape;
				auto loss = OaFnLoss::CrossEntropy(
					model->Forward(x), y.Reshape(OaMatrixShape{y.NumElements()}));
				tape.Backward(loss);
				training.Loop.RecordLoss(loss);
			});
		if (training.Loop.Index() == 1) initialLoss = training.Loop.LastLoss();
	}
	ASSERT_TRUE(training.Loop.Finish().IsOk());
	const OaF32 finalLoss = training.Loop.LastLoss();
	const OaF32 accuracy = NlpAccuracyAllPositions(*model, x, y, kByteVocab);

	std::printf("\nEvaluation:\n");
	std::printf("  Random-loss baseline ln(%d) = %.4f\n", kByteVocab, std::log(static_cast<double>(kByteVocab)));
	std::printf("  Bits/byte: %.4f\n", NlpBitsPerByte(finalLoss));
	std::printf("  Accuracy: %.1f%%\n", accuracy);
	std::printf("\nGeneration:\n  Prompt: '%s'\n  Generated: '%s'\n\n", kNlpGenerationPrompt,
		NlpGenerateGreedy(*model, kNlpGenerationPrompt, kNlpGenerationBytes, kByteVocab).c_str());

	ASSERT_GT(initialLoss, 0.0F);
	EXPECT_LT(finalLoss, initialLoss);
	EXPECT_GT(accuracy, 30.0F);
	ASSERT_TRUE(model->Save("/tmp/byte_moe.oam", *optimizer).IsOk());
	auto reloaded = OaMakeSharedPtr<OaByteMoeLM>();
	auto reloadParameters = reloaded->AllParameterPtrs();
	auto reloadOptimizer = OaMakeUniquePtr<OaAdamW>(reloadParameters, 0.01F);
	ASSERT_TRUE(reloaded->Load("/tmp/byte_moe.oam", *reloadOptimizer).IsOk());
	EXPECT_NEAR(NlpAccuracyAllPositions(*reloaded, x, y, kByteVocab), accuracy, 0.5F);
	EXPECT_EQ(reloadOptimizer->GetStep(), optimizer->GetStep());
}
