// ═══════════════════════════════════════════════════════════════════════════
// OA Tutorial — Mamba-3 Byte LM (Autograd Reference)
//
// Untouched Mamba-3 reference path: OaEmbedding + OaMamba3Module::Forward()
// (Ssm/Mamba3/Mamba3Siso* kernels) + flat token residual + Linear head.
//
// Does NOT go through OaEmpyrealmCore or Empyrealm-branded shaders.
// Siblings:
//   TutorialNlpByteEmpyrealmAg — Empyrealm fused/traceable SSM (OaEmpyrealmCore)
//   TutorialNlpByteRnnAg / TutorialNlpByteGruAg — recurrent baselines
//   TutorialNlpByteTransformerAg — causal self-attention baseline
// ═══════════════════════════════════════════════════════════════════════════

#include "OaTest.h"
#include "TutorialMl.h"
#include "TutorialNlpCommon.h"
#include <Oa/Ml.h>
#include <Oa/Ml/Byte.h>
#include <Oa/Ml/Autograd.h>
#include <Oa/Runtime/Context.h>
#include <Oa/Runtime/Engine.h>
#include <Oa/Runtime/GpuTimer.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <vector>

static constexpr OaI32 kVocabSize  = OA_BYTE_VOCAB_SIZE;  // 256 — byte vocab family
// kContextLen / kDModel come from TutorialNlpCommon.h (shared suite dims).
static constexpr OaI32 kDState     = 32;
static constexpr OaI32 kExpand     = 2;
static constexpr OaI32 kHeadDim    = 16;

// Direct Mamba-3 reference wiring (no OaEmpyrealmCore).
class OaMamba3ByteLM : public OaModule {
public:
	OaMamba3ByteLM() {
		Embed_ = OaMakeSharedPtr<OaEmbedding>(kVocabSize, kDModel);
		Mamba_ = OaMakeSharedPtr<OaMamba3Module>(
			kDModel, kDState, kExpand, kHeadDim,
			/*NGroups*/ 1, /*RopeFraction*/ 0.5f, /*Mimo*/ false, /*MimoRank*/ 4,
			/*DtMin*/ 0.001f, /*DtMax*/ 0.1f, /*DtInitFloor*/ 1e-4f, /*AFloor*/ 1e-4f,
			/*OutprojNorm*/ true);
		Head_ = OaMakeSharedPtr<OaLinear>(kDModel, kVocabSize);
		RegisterModule("embed", Embed_);
		RegisterModule("mamba3", Mamba_);
		RegisterModule("head", Head_);
	}

	OaMatrix Forward(const OaMatrix& InTokens) override {
		auto bs    = static_cast<OaI64>(InTokens.Size(0));
		auto sl    = static_cast<OaI64>(InTokens.Size(1));
		auto embFlat = Embed_->Forward(InTokens);             // flat [B*S, D]
		auto emb3d   = embFlat.Reshape(OaMatrixShape{bs, sl, kDModel});  // Mamba3 needs [B, S, D]
		auto y3d   = Mamba_->Forward(emb3d);     // [B, S, D] via Mamba3Siso*
		auto mixed = y3d.Reshape(OaMatrixShape{bs * sl, kDModel})
		           + embFlat.Reshape(OaMatrixShape{bs * sl, kDModel});
		return Head_->Forward(mixed);
	}

	void ResetGenerationState(OaI32 InBatch) {
		Mamba_->ResetState(InBatch);
	}

	OaMatrix ForwardGenerationStep(const OaMatrix& InToken) {
		auto batch = static_cast<OaI64>(InToken.Size(0));
		auto embedded = Embed_->Forward(InToken)
			.Reshape(OaMatrixShape{batch, 1, kDModel});
		auto sequenceOutput = Mamba_->Step(embedded);
		return Head_->Forward(
			sequenceOutput.Reshape(OaMatrixShape{batch, kDModel}) +
			embedded.Reshape(OaMatrixShape{batch, kDModel}));
	}

	[[nodiscard]] OaSharedPtr<OaMamba3Module> Mamba() const { return Mamba_; }

private:
	OaSharedPtr<OaEmbedding>     Embed_;
	OaSharedPtr<OaMamba3Module>  Mamba_;
	OaSharedPtr<OaLinear>        Head_;
};

TEST(TutorialNlpByteMamba3Ag, Mamba3Reference) {
	printf("\n╔══════════════════════════════════════════════════════════════════╗\n");
	printf("║  OA Tutorial — Mamba-3 Reference (Autograd)                      ║\n");
	printf("║  OaMamba3Module::Forward · Ssm/Mamba3/ kernels (untouched)       ║\n");
	printf("╚══════════════════════════════════════════════════════════════════╝\n\n");

	OaFnMatrix::SetRngSeed(OaNlpSuiteRngSeed);
	auto  model  = OaMakeSharedPtr<OaMamba3ByteLM>();
	auto  params = model->AllParameterPtrs();
	// lr 0.003: with the corrected [B,S,D] batching the model reaches ~0.3 CE by
	// ~step 120; 0.01 was hot enough to diverge late once batching was fixed.
	auto  opt    = OaMakeUniquePtr<OaAdamW>(params, 0.003F);

	printf("Model: OaEmbedding + OaMamba3Module + flat residual + Linear head\n");
	printf("Params: %lld\n\n", static_cast<long long>(model->NumParameters()));

	constexpr OaI32 kSteps = 300;
	constexpr OaI32 kBatch = 64;
	NlpAllPositionSampler sampler(NlpCorpus(), kBatch);

	TutorialTrainingLoop training(*opt, OaItTrainingConfig{
		.TotalSteps     = kSteps,
		.EpochSteps     = {},
#if defined(_WIN32)
		// The Windows NVIDIA driver loses the device when this long SSM graph
		// has more than one training submission in flight.
#endif
		.BatchSize      = kBatch,
		.SequenceLength = kContextLen,
		.SequenceUnit   = "token",
		.SourceUnitsPerSample = kContextLen,
		.SourceUnit     = "byte",
		.TimerName      = "mamba3_ref_step",
		.Callbacks      = {}
	});

	// Ring-buffered batches: OaItTraining may submit up to MaxAsyncSubmissions()
	// steps before syncing; reusing one batchX/batchY would let the GPU read
	// overwritten token buffers while earlier steps are still in flight.
	auto& ctx = OaContext::GetDefault();
	OaVec<OaMatrix> xRing(static_cast<OaI64>(ctx.MaxAsyncSubmissions()));
	OaVec<OaMatrix> yRing(static_cast<OaI64>(ctx.MaxAsyncSubmissions()));
	OaF32 initialLoss = 0.0F;
	OaF32 lastLoss    = 0.0F;
	float inProjGradL1 = 0.0F;
	float outProjGradL1 = 0.0F;

	while (not training.Loop.IsDone()) {
		const OaI64 step = training.Loop.Index();
		OaMatrix& batchX = xRing[(step - 1) % xRing.Size()];
		OaMatrix& batchY = yRing[(step - 1) % yRing.Size()];
		sampler.NextBatch(batchX, batchY);
		opt->ZeroGrad();
		OaGradientTape tape;
		auto logits = model->Forward(batchX);
		auto loss   = OaFnLoss::CrossEntropy(logits,
			batchY.Reshape(OaMatrixShape{batchY.Size(0) * batchY.Size(1)}));
		tape.Backward(loss);
		training.Loop.Next(loss);

		if (training.Loop.Index() == 1) {
			initialLoss = training.Loop.LiveLoss();
			printf("\n─── Step-1 gradient magnitudes (Mamba3 reference, L1, fp32-read) ───\n");
			auto& ctx2 = OaContext::GetDefault();
			struct MagEntry { const char* name; OaMatrix result; OaI64 numel; };
			std::vector<MagEntry> entries;
			for (auto* p : params) {
				auto g = p->Data.GradMatrix();
				OaMatrix s;
				OaI64 numel = 0;
				if (!g.IsEmpty() && g.NumElements() > 0) {
					auto flat = OaFnMatrix::Cast(g.Reshape(OaMatrixShape{g.NumElements()}), OaScalarType::Float32);
					auto absg = OaFnMatrix::Abs(flat);
					s = OaFnMatrix::Sum(absg, 0);
					numel = p->Data.NumElements();
				}
				entries.push_back({p->Name.CStr(), std::move(s), numel});
			}
			(void)ctx2.Execute();
			(void)ctx2.Sync();
			for (const auto& e : entries) {
				float mag = 0.0F;
				if (e.result.NumElements() > 0) mag = e.result.At(0);
				printf("  %-32s  L1=%.6g  (numel=%lld)\n",
					e.name, mag, static_cast<long long>(e.numel));
				if (std::strcmp(e.name, "in_proj") == 0) inProjGradL1 = mag;
				if (std::strcmp(e.name, "out_proj") == 0) outProjGradL1 = mag;
			}
			printf("\n");
			fflush(stdout);
		}
	}
	ASSERT_TRUE(training.Loop.Finish().IsOk());
	lastLoss = training.Loop.LastLoss();

	OaMatrix& evalX = xRing[(kSteps - 1) % xRing.Size()];
	OaMatrix& evalY = yRing[(kSteps - 1) % yRing.Size()];
	const OaF32 finalBatchAcc = NlpAccuracyAllPositions(*model, evalX, evalY, kVocabSize);
	printf("\nEvaluation:\n");
	printf("  Random-loss baseline ln(%d) = %.4f\n",
		kVocabSize, std::log(static_cast<double>(kVocabSize)));
	printf("  Bits/byte: %.4f\n", NlpBitsPerByte(lastLoss));
	printf("  Accuracy: %.1f%%\n", finalBatchAcc);

	const OaString generated = NlpGenerateStatefulGreedy(
		*model, kNlpGenerationPrompt, kNlpGenerationBytes, kVocabSize);
	printf("\nGeneration:\n");
	printf("  Prompt: '%s'\n", kNlpGenerationPrompt);
	printf("  Generated: '%s'\n\n", generated.c_str());

	EXPECT_LT(lastLoss, initialLoss);
	EXPECT_GT(finalBatchAcc, 30.0F);
	EXPECT_GT(outProjGradL1, 0.0F) << "out_proj must receive gradient on Mamba3 reference path";

	const OaString ckptPath = "/tmp/mamba3_ref_autograd.oam";
	ASSERT_TRUE(model->Save(ckptPath, *opt).IsOk());
	auto reloaded = OaMakeSharedPtr<OaMamba3ByteLM>();
	auto reloadParams = reloaded->AllParameterPtrs();
	auto reloadedOpt = OaMakeUniquePtr<OaAdamW>(reloadParams, 0.003F);
	ASSERT_TRUE(reloaded->Load(ckptPath, *reloadedOpt).IsOk());
	EXPECT_NEAR(NlpAccuracyAllPositions(*reloaded, evalX, evalY, kVocabSize), finalBatchAcc, 1.0F);
	EXPECT_EQ(NlpGenerateStatefulGreedy(
		*reloaded, kNlpGenerationPrompt, kNlpGenerationBytes, kVocabSize), generated);
}
