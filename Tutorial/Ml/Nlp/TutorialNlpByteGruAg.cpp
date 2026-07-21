// ═══════════════════════════════════════════════════════════════════════════
// OA Tutorial — Byte-Level GRU, all-position LM (Implicit Autograd)
// Module API: OaByteEmbedding + OaGru + OaLinear + OaAdamW + OaGradientTape
// ═══════════════════════════════════════════════════════════════════════════
//
// The GRU member of the NLP fair-comparison suite (TutorialNlpCommon.h). Same
// all-position next-token task, corpus and dims as the RNN/Transformer/Mamba-3/
// Empyrealm siblings — the gated recurrent unit adds reset/update gates over the
// plain RNN. We project every timestep to the vocab (a real LM), not just the last.
// ═══════════════════════════════════════════════════════════════════════════

#include "OaTest.h"
#include "TutorialMl.h"
#include "TutorialNlpCommon.h"
#include <Oa/Ml.h>
#include <Oa/Ml/Byte.h>
#include <Oa/Ml/Autograd.h>
#include <Oa/Runtime/Context.h>
#include <Oa/Runtime/Engine.h>
#include <cmath>

static constexpr OaI32 kVocabSize = OA_BYTE_VOCAB_SIZE;  // 256 — byte vocab family

// ─── Model: ByteEmbedding → GRU → Linear, projected at every step ────────────

class OaByteGruLM : public OaModule {
public:
	OaByteGruLM() {
		auto wd = OaFnMatrix::GetWeightDtype();

		Embed_ = OaMakeSharedPtr<OaByteEmbedding>(kDModel);
		Embed_->Parameters()[0].Data = OaFnMatrix::RandN(OaMatrixShape{kVocabSize, kDModel}, wd);
		Embed_->Parameters()[0].Data.SetRequiresGrad(true);
	
		Gru_ = OaMakeSharedPtr<OaGru>(kDModel, kHiddenDim, 1);
		for (auto& param : Gru_->Parameters()) {
			param.Data.SetRequiresGrad(true);
		}
	
		Head_ = OaMakeSharedPtr<OaLinear>(kHiddenDim, kVocabSize);
		Head_->Parameters()[0].Data = OaFnMatrix::Rand(OaMatrixShape{kVocabSize, kHiddenDim}, wd);
		for (auto& param : Head_->Parameters()) {
			param.Data.SetRequiresGrad(true);
		}

		RegisterModule("embed", Embed_);
		RegisterModule("gru",   Gru_);
		RegisterModule("head",  Head_);
	}

	OaMatrix Forward(const OaMatrix& InTokens) override {
		const OaI32 batch  = static_cast<OaI32>(InTokens.Size(0));
		const OaI32 seqLen = static_cast<OaI32>(InTokens.Size(1));

		// Gather/ByteEmbedding flattens to [batch*seq, d_model]; restore the rank-3
		// shape OaGru::Forward expects (else it slices the wrong axis → O(seq^2)).
		auto embedded = Embed_->Forward(InTokens).Reshape(OaMatrixShape{batch, seqLen, kDModel});
		auto gruOut   = Gru_->Forward(embedded);                          // [B, S, H]
		auto flat     = gruOut.Reshape(OaMatrixShape{static_cast<OaI64>(batch) * seqLen, kHiddenDim});  // [B*S, H]
		return Head_->Forward(flat);                                      // [B*S, V]
	}

private:
	OaSharedPtr<OaByteEmbedding> Embed_;
	OaSharedPtr<OaGru>           Gru_;
	OaSharedPtr<OaLinear>        Head_;
};

// ─── Tutorial ──────────────────────────────────────────────────────────────

TEST(TutorialNlpByteGruAg, GruAllPositionLM) {
	printf("\n╔══════════════════════════════════════════════════════════════════╗\n");
	printf("║  OA Tutorial — Byte GRU · all-position LM (Autograd)            ║\n");
	printf("╚══════════════════════════════════════════════════════════════════╝\n\n");
	printf("Vocab: %d bytes · Context: %d · DModel: %d · Hidden: %d\n",
		kVocabSize, kContextLen, kDModel, kHiddenDim);
	printf("Task: dense next-byte at every position via a gated recurrent unit\n\n");

	OaFnMatrix::SetRngSeed(OaNlpSuiteRngSeed);
	auto  model  = OaMakeSharedPtr<OaByteGruLM>();
	auto  params = model->AllParameterPtrs();
	auto  opt    = OaMakeUniquePtr<OaAdamW>(params, 0.01F);
	auto& rt     = *OaEngine::GetGlobal();

	printf("Model: ByteEmbed(%d→%d) → GRU(%d→%d, layers=1) → Linear(%d→%d)\n",
		kVocabSize, kDModel, kDModel, kHiddenDim, kHiddenDim, kVocabSize);
	printf("Params: %lld    Optimizer: AdamW(lr=0.01)\n\n",
		static_cast<long long>(model->NumParameters()));

	NlpAllPositionSampler sampler(NlpCorpus(), kBatch);
	TutorialTrainingLoop training(*opt, OaItTrainingConfig{
		.TotalSteps     = kSteps,
		.EpochSteps     = {},
		.BatchSize      = kBatch,
		.SequenceLength = kContextLen,
		.SequenceUnit   = "token",
		.SourceUnitsPerSample = kContextLen,
		.SourceUnit     = "byte",
		.TimerName      = "byte_gru_allpos_step",
		.Callbacks      = {},
	});
	printf("Training: %d steps · batch=%d · sequence=%d tokens\n", kSteps, kBatch, kContextLen);

	OaMatrix batchX;
	OaMatrix batchY;
	OaF32 initialLoss = 0.0F;

	while (not training.Loop.IsDone()) {
		sampler.NextBatch(batchX, batchY);
		opt->ZeroGrad();
		OaGradientTape tape;
		auto logits = model->Forward(batchX);
		auto loss   = OaFnLoss::CrossEntropy(logits,
			batchY.Reshape(OaMatrixShape{batchY.Size(0) * batchY.Size(1)}));
		tape.Backward(loss);
		training.Loop.Next(loss);
		if (training.Loop.Index() == 1) { initialLoss = training.Loop.LiveLoss(); }
	}
	ASSERT_TRUE(training.Loop.Finish().IsOk()) << "Finish failed";
	const OaF32 lastLoss = training.Loop.LastLoss();

	const OaF32 finalAcc = NlpAccuracyAllPositions(*model, batchX, batchY, kVocabSize);
	(void)rt;
	printf("\nEvaluation:\n");
	printf("  Random-loss baseline ln(%d) = %.4f\n",
		kVocabSize, std::log(static_cast<double>(kVocabSize)));
	printf("  Bits/byte: %.4f\n", NlpBitsPerByte(lastLoss));
	printf("  Accuracy: %.1f%%\n", finalAcc);
	printf("\nGeneration:\n  Prompt: '%s'\n  Generated: '%s'\n\n", kNlpGenerationPrompt,
		NlpGenerateGreedy(*model, kNlpGenerationPrompt, kNlpGenerationBytes, kVocabSize).c_str());

	ASSERT_GT(initialLoss, 0.0F);
	EXPECT_LT(lastLoss, initialLoss) << "Loss must decrease during training";
	EXPECT_GT(finalAcc, 50.0F)       << "All-position accuracy should exceed 50%";

	// ── Save / Load round-trip (model + optimizer) ──
	const OaString ckptPath = "/tmp/byte_gru_allpos.oam";
	ASSERT_TRUE(model->Save(ckptPath, *opt).IsOk());
	auto reloaded    = OaMakeSharedPtr<OaByteGruLM>();
	auto reloadParam = reloaded->AllParameterPtrs();
	auto reloadedOpt = OaMakeUniquePtr<OaAdamW>(reloadParam, 0.01F);
	ASSERT_TRUE(reloaded->Load(ckptPath, *reloadedOpt).IsOk());
	const OaF32 reloadedAcc = NlpAccuracyAllPositions(*reloaded, batchX, batchY, kVocabSize);
	EXPECT_NEAR(reloadedAcc, finalAcc, 0.5F)          << "Reload accuracy must match";
	EXPECT_EQ(reloadedOpt->GetStep(), opt->GetStep()) << "Optimizer step must round-trip";
}
