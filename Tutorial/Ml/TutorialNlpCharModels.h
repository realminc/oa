// Models and shared runner for the 27-symbol character NLP tutorial family.

#pragma once

#include "TutorialMl.h"
#include "TutorialNlpCommon.h"

#include <Oa/Ml.h>
#include <Oa/Ml/Autograd.h>

#include <cmath>

class OaCharRnnLM : public OaModule {
public:
	OaCharRnnLM() {
		auto wd = OaFnMatrix::GetWeightDtype();
		Embed_ = OaMakeSharedPtr<OaEmbedding>(kCharVocabSize, kDModel);
		Embed_->Parameters()[0].Data = OaFnMatrix::RandN(OaMatrixShape{kCharVocabSize, kDModel}, wd);
		Rnn_ = OaMakeSharedPtr<OaRnn>(kDModel, kHiddenDim, 1);
		Head_ = OaMakeSharedPtr<OaLinear>(kHiddenDim, kCharVocabSize);
		Head_->Parameters()[0].Data = OaFnMatrix::Rand(OaMatrixShape{kCharVocabSize, kHiddenDim}, wd);
		RegisterModule("embed", Embed_); RegisterModule("rnn", Rnn_); RegisterModule("head", Head_);
		for (auto* p : AllParameterPtrs()) p->Data.SetRequiresGrad(true);
	}
	OaMatrix Forward(const OaMatrix& InTokens) override {
		const OaI32 b = static_cast<OaI32>(InTokens.Size(0));
		const OaI32 s = static_cast<OaI32>(InTokens.Size(1));
		auto e = Embed_->Forward(InTokens).Reshape(OaMatrixShape{b, s, kDModel});
		return Head_->Forward(Rnn_->Forward(e).Reshape(OaMatrixShape{b * s, kHiddenDim}));
	}
private:
	OaSharedPtr<OaEmbedding> Embed_; OaSharedPtr<OaRnn> Rnn_; OaSharedPtr<OaLinear> Head_;
};

class OaCharGruLM : public OaModule {
public:
	OaCharGruLM() {
		auto wd = OaFnMatrix::GetWeightDtype();
		Embed_ = OaMakeSharedPtr<OaEmbedding>(kCharVocabSize, kDModel);
		Embed_->Parameters()[0].Data = OaFnMatrix::RandN(OaMatrixShape{kCharVocabSize, kDModel}, wd);
		Gru_ = OaMakeSharedPtr<OaGru>(kDModel, kHiddenDim, 1);
		Head_ = OaMakeSharedPtr<OaLinear>(kHiddenDim, kCharVocabSize);
		Head_->Parameters()[0].Data = OaFnMatrix::RandXavier(OaMatrixShape{kCharVocabSize, kHiddenDim}, wd);
		RegisterModule("embed", Embed_); RegisterModule("gru", Gru_); RegisterModule("head", Head_);
		for (auto* p : AllParameterPtrs()) p->Data.SetRequiresGrad(true);
	}
	OaMatrix Forward(const OaMatrix& InTokens) override {
		const OaI32 b = static_cast<OaI32>(InTokens.Size(0));
		const OaI32 s = static_cast<OaI32>(InTokens.Size(1));
		auto e = Embed_->Forward(InTokens).Reshape(OaMatrixShape{b, s, kDModel});
		return Head_->Forward(Gru_->Forward(e).Reshape(OaMatrixShape{b * s, kHiddenDim}));
	}
private:
	OaSharedPtr<OaEmbedding> Embed_; OaSharedPtr<OaGru> Gru_; OaSharedPtr<OaLinear> Head_;
};

class OaCharTransformerLM : public OaModule {
public:
	OaCharTransformerLM() {
		auto wd = OaFnMatrix::GetWeightDtype();
		TokEmbed_ = OaMakeSharedPtr<OaEmbedding>(kCharVocabSize, kDModel);
		PosEmbed_ = OaMakeSharedPtr<OaEmbedding>(kContextLen, kDModel);
		Block_ = OaMakeSharedPtr<OaTransformerBlock>(kDModel, kHiddenDim, kContextLen);
		LnFinal_ = OaMakeSharedPtr<OaLayerNorm>(kDModel, 1e-5F);
		Head_ = OaMakeSharedPtr<OaLinear>(kDModel, kCharVocabSize);
		Head_->Parameters()[0].Data = OaFnMatrix::Rand(OaMatrixShape{kCharVocabSize, kDModel}, wd);
		RegisterModule("tok_embed", TokEmbed_); RegisterModule("pos_embed", PosEmbed_);
		RegisterModule("block", Block_); RegisterModule("ln_final", LnFinal_); RegisterModule("head", Head_);
		for (auto* p : AllParameterPtrs()) p->Data.SetRequiresGrad(true);
	}
	OaMatrix Forward(const OaMatrix& InTokens) override {
		const OaI32 b = static_cast<OaI32>(InTokens.Size(0));
		const OaI32 n = b * kContextLen;
		auto x = TokEmbed_->Forward(InTokens).Reshape(OaMatrixShape{n, kDModel}) + PosEmbed_->Forward(PositionIds(b));
		return Head_->Forward(LnFinal_->Forward(Block_->Forward(x)));
	}
private:
	OaMatrix PositionIds(OaI32 InBatch) const {
		OaVec<OaU8> ids(static_cast<OaI64>(InBatch) * kContextLen);
		for (OaI64 i = 0; i < static_cast<OaI64>(InBatch) * kContextLen; ++i) ids[i] = static_cast<OaU8>(i % kContextLen);
		return OaFnMatrix::FromBytes(OaSpan<const OaU8>(ids.Data(), ids.Size()),
			OaMatrixShape{static_cast<OaI64>(InBatch) * kContextLen}, OaScalarType::UInt8);
	}
	OaSharedPtr<OaEmbedding> TokEmbed_, PosEmbed_; OaSharedPtr<OaTransformerBlock> Block_;
	OaSharedPtr<OaLayerNorm> LnFinal_; OaSharedPtr<OaLinear> Head_;
};

class OaCharMamba3LM : public OaModule {
public:
	OaCharMamba3LM() {
		Embed_ = OaMakeSharedPtr<OaEmbedding>(kCharVocabSize, kDModel);
		Mamba_ = OaMakeSharedPtr<OaMamba3Module>(kDModel, 32, 2, 16, 1, 0.5F, false, 4,
			0.001F, 0.1F, 1e-4F, 1e-4F, true);
		Head_ = OaMakeSharedPtr<OaLinear>(kDModel, kCharVocabSize);
		RegisterModule("embed", Embed_); RegisterModule("mamba3", Mamba_); RegisterModule("head", Head_);
	}
	OaMatrix Forward(const OaMatrix& InTokens) override {
		const OaI64 b = InTokens.Size(0), s = InTokens.Size(1);
		auto flat = Embed_->Forward(InTokens);
		auto y = Mamba_->Forward(flat.Reshape(OaMatrixShape{b, s, kDModel}));
		return Head_->Forward(y.Reshape(OaMatrixShape{b * s, kDModel}) + flat.Reshape(OaMatrixShape{b * s, kDModel}));
	}
private:
	OaSharedPtr<OaEmbedding> Embed_; OaSharedPtr<OaMamba3Module> Mamba_; OaSharedPtr<OaLinear> Head_;
};

class OaCharMoeLM : public OaModule {
public:
	OaCharMoeLM() {
		auto wd = OaFnMatrix::GetWeightDtype();
		TokEmbed_ = OaMakeSharedPtr<OaEmbedding>(kCharVocabSize, kDModel);
		PosEmbed_ = OaMakeSharedPtr<OaEmbedding>(kContextLen, kDModel);
		// Four experts with top-2 routing. DFF=16 gives 1.5x the dense FFN's
		// parameter capacity but only 0.75x its active projection work once the
		// sparse grouped-GEMM path replaces the current dense oracle.
		Block_ = OaMakeSharedPtr<OaTransformerBlock>(kDModel, 16, kContextLen, 4, 2);
		LnFinal_ = OaMakeSharedPtr<OaLayerNorm>(kDModel, 1e-5F);
		Head_ = OaMakeSharedPtr<OaLinear>(kDModel, kCharVocabSize);
		Head_->Parameters()[0].Data = OaFnMatrix::Rand(OaMatrixShape{kCharVocabSize, kDModel}, wd);
		RegisterModule("tok_embed", TokEmbed_); RegisterModule("pos_embed", PosEmbed_);
		RegisterModule("block", Block_); RegisterModule("ln_final", LnFinal_); RegisterModule("head", Head_);
		for (auto* p : AllParameterPtrs()) p->Data.SetRequiresGrad(true);
	}
	OaMatrix Forward(const OaMatrix& InTokens) override {
		const OaI32 b = static_cast<OaI32>(InTokens.Size(0));
		const OaI32 n = b * kContextLen;
		auto x = TokEmbed_->Forward(InTokens).Reshape(OaMatrixShape{n, kDModel}) + PosEmbed_->Forward(PositionIds(b));
		return Head_->Forward(LnFinal_->Forward(Block_->Forward(x)));
	}
private:
	OaMatrix PositionIds(OaI32 InBatch) const {
		OaVec<OaU8> ids(static_cast<OaI64>(InBatch) * kContextLen);
		for (OaI64 i = 0; i < static_cast<OaI64>(InBatch) * kContextLen; ++i) ids[i] = static_cast<OaU8>(i % kContextLen);
		return OaFnMatrix::FromBytes(OaSpan<const OaU8>(ids.Data(), ids.Size()),
			OaMatrixShape{static_cast<OaI64>(InBatch) * kContextLen}, OaScalarType::UInt8);
	}
	OaSharedPtr<OaEmbedding> TokEmbed_, PosEmbed_; OaSharedPtr<OaTransformerBlock> Block_;
	OaSharedPtr<OaLayerNorm> LnFinal_; OaSharedPtr<OaLinear> Head_;
};

template <class Model>
void RunNlpCharTutorial(const char* InTitle, const char* InModelDescription,
	const char* InTimerName, const char* InCheckpointPath, OaF32 InLearningRate) {
	std::printf("\n╔══════════════════════════════════════════════════════════════════╗\n");
	std::printf("║  %-62s║\n", InTitle);
	std::printf("╚══════════════════════════════════════════════════════════════════╝\n\n");
	std::printf("Tokenizer: character · vocab=%d (a-z + space)\n", kCharVocabSize);
	std::printf("Task: dense next-character prediction at every position\n\n");

	auto model = OaMakeSharedPtr<Model>();
	auto params = model->AllParameterPtrs();
	auto opt = OaMakeUniquePtr<OaAdamW>(params, InLearningRate);
	std::printf("Model: %s\n", InModelDescription);
	std::printf("Params: %lld    Optimizer: AdamW(lr=%.3g)\n\n",
		static_cast<long long>(model->NumParameters()), static_cast<double>(InLearningRate));

	NlpAllPositionSampler sampler(NlpCorpus(), kBatch, NlpCharEncode);
	TutorialTrainingLoop training(*opt, OaItTrainingConfig{
		.TotalSteps = kSteps,
		.EpochSteps = {},
		.BatchSize = kBatch,
		.SequenceLength = kContextLen,
		.SequenceUnit = "token",
		.TimerName = InTimerName,
	});
	std::printf("Training: %d steps · batch=%d · sequence=%d character tokens\n", kSteps, kBatch, kContextLen);

	OaMatrix x, y;
	OaF32 initialLoss = 0.0F;
	while (not training.Loop.IsDone()) {
		sampler.NextBatch(x, y);
		opt->ZeroGrad();
		OaGradientTape tape;
		auto logits = model->Forward(x);
		auto loss = OaFnLoss::CrossEntropy(logits, y.Reshape(OaMatrixShape{y.NumElements()}));
		tape.Backward(loss);
		training.Loop.Next(loss);
		if (training.Loop.Index() == 1) initialLoss = training.Loop.LastLoss();
	}
	ASSERT_TRUE(training.Loop.Finish().IsOk());
	const OaF32 finalLoss = training.Loop.LastLoss();
	const OaF32 accuracy = NlpAccuracyAllPositions(*model, x, y, kCharVocabSize);

	std::printf("\nEvaluation:\n");
	std::printf("  Random-loss baseline ln(%d) = %.4f\n", kCharVocabSize, std::log(static_cast<double>(kCharVocabSize)));
	std::printf("  Character-token accuracy: %.1f%% (compare within Char only)\n", accuracy);
	std::printf("\nGeneration:\n  Prompt: '%s'\n  Generated: '%s'\n\n",
		kNlpGenerationPrompt,
		NlpGenerateGreedy(*model, kNlpGenerationPrompt, kNlpGenerationBytes,
			kCharVocabSize, NlpCharEncode).c_str());

	ASSERT_GT(initialLoss, 0.0F);
	EXPECT_LT(finalLoss, initialLoss);
	EXPECT_GT(accuracy, 50.0F);
	ASSERT_TRUE(model->Save(InCheckpointPath, *opt).IsOk());
	auto reloaded = OaMakeSharedPtr<Model>();
	auto reloadParams = reloaded->AllParameterPtrs();
	auto reloadOpt = OaMakeUniquePtr<OaAdamW>(reloadParams, InLearningRate);
	ASSERT_TRUE(reloaded->Load(InCheckpointPath, *reloadOpt).IsOk());
	EXPECT_NEAR(NlpAccuracyAllPositions(*reloaded, x, y, kCharVocabSize), accuracy, 0.5F);
	EXPECT_EQ(reloadOpt->GetStep(), opt->GetStep());
}
