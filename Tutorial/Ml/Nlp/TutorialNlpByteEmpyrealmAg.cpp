// ═══════════════════════════════════════════════════════════════════════════
// OA Tutorial — Empyrealm Core Ag (Autograd Fidelity)
//
// OaEmpyrealmCore with UseFused=false: traceable Preprocess + EmpyrealmSiso
// + out_proj MatMul. Validates full Mamba-3 mixer autograd (in_proj, dt/B/C
// bias, D, norm_weight, out_proj) through the flat residual + CE head.
//
// Siblings:
//   TutorialNlpByteMamba3Ag    — untouched OaMamba3Module + Ssm/Mamba3/ reference
//   TutorialNlpByteRnnAg / TutorialNlpByteGruAg — recurrent baselines
// ═══════════════════════════════════════════════════════════════════════════

#include "OaTest.h"
#include "TutorialMl.h"
#include "TutorialNlpCommon.h"
#include <Oa/Ml.h>
#include <Oa/Ml/Byte.h>
#include <Oa/Ml/Autograd.h>
#include <Oa/Ml/OptimUtil.h>
#include <cstdlib>
#include <Oa/Runtime/Context.h>
#include <Oa/Runtime/Engine.h>
#include <Oa/Runtime/GpuTimer.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <vector>

static bool TutorialUseMuonOptimizer() {
	const char* env = std::getenv("OA_USE_MUON");
	return env && env[0] == '1';
}

static OaUniquePtr<OaOptimizer> TutorialMakeOptimizer(OaModule& InModel, OaF32 InLr) {
	if (TutorialUseMuonOptimizer()) {
		OaMuonAdamWConfig cfg{};
		cfg.AdamWLr = InLr;
		return MakeMuonAdamWOptimizer(InModel, cfg);
	}
	auto params = InModel.AllParameterPtrs();
	return OaMakeUniquePtr<OaAdamW>(params, InLr);
}

static constexpr OaI32 kVocabSize  = OA_BYTE_VOCAB_SIZE;  // 256 — byte vocab family
// kContextLen / kDModel come from TutorialNlpCommon.h (shared suite dims).
static constexpr OaI32 kDState     = 32;
static constexpr OaI32 kExpand     = 2;
static constexpr OaI32 kHeadDim    = 16;

class OaEmpyrealmByteLMAg : public OaModule {
public:
	OaEmpyrealmByteLMAg() {
		Core_ = OaMakeSharedPtr<OaEmpyrealmCore>(kVocabSize, kDModel,
			kDState, kExpand, kHeadDim,
			/*NGroups*/ 1, /*RopeFraction*/ 0.5f, /*Mimo*/ false, /*MimoRank*/ 1,
			/*DtMin*/ 0.001f, /*DtMax*/ 0.1f, /*DtInitFloor*/ 1e-4f, /*AFloor*/ 1e-4f,
			/*OutprojNorm*/ true);
		Head_ = OaMakeSharedPtr<OaLinear>(kDModel, kVocabSize);
		RegisterModule("core", Core_);
		RegisterModule("head", Head_);
	}

	OaMatrix Forward(const OaMatrix& InTokens) override {
		auto mixed = Core_->Forward(InTokens);
		return Head_->Forward(mixed);
	}

	[[nodiscard]] OaSharedPtr<OaEmpyrealmCore> Core() const { return Core_; }

private:
	OaSharedPtr<OaEmpyrealmCore> Core_;
	OaSharedPtr<OaLinear>        Head_;
};

static float ParamGradL1(const OaMatrix& g) {
	if (g.IsEmpty() || g.NumElements() == 0) return 0.0F;
	auto s = OaFnMatrix::Sum(OaFnMatrix::Abs(g.Reshape(OaMatrixShape{g.NumElements()})), 0);
	return s.At(0);
}

TEST(TutorialNlpByteEmpyrealmAg, TraceableAutograd) {
	printf("\n╔══════════════════════════════════════════════════════════════════╗\n");
	printf("║  OA Tutorial — Empyrealm Core Ag (Autograd Fidelity)             ║\n");
	printf("║  EmpyrealmPreprocess + EmpyrealmSiso traced path                 ║\n");
	printf("╚══════════════════════════════════════════════════════════════════╝\n\n");

	auto  model  = OaMakeSharedPtr<OaEmpyrealmByteLMAg>();
	auto  params = model->AllParameterPtrs();
	// lr 0.003: with correct [B,S,D] batching (EmpyrealmCore::Forward) the model
	// converges fast; 0.01 is hot enough to diverge late, same as the Mamba3 ref.
	auto  opt    = TutorialMakeOptimizer(*model, 0.003F);

	printf("Path: traceable split (Preprocess MatMul + EmpyrealmSiso + out_proj)\n");
	printf("Params: %lld    Optimizer: %s(lr=0.003)\n\n",
		static_cast<long long>(model->NumParameters()),
		TutorialUseMuonOptimizer() ? "Muon+AdamW" : "AdamW");

	constexpr OaI32 kSteps = 300;
	constexpr OaI32 kBatch = 64;
	NlpAllPositionSampler sampler(NlpCorpus(), kBatch);

	TutorialTrainingLoop training(*opt, OaItTrainingConfig{
		.TotalSteps     = kSteps,
		.EpochSteps     = {},
#if defined(_WIN32)
		// Same long SSM graph as the Mamba3 reference (UseFused=false delegates to
		// the identical OaMamba3Module::Forward): the Windows NVIDIA driver loses
		// the device with more than one training submission in flight, so sync
		// every step.
#endif
		.BatchSize      = kBatch,
		.SequenceLength = kContextLen,
		.SequenceUnit   = "token",
		.SourceUnitsPerSample = kContextLen,
		.SourceUnit     = "byte",
		.TimerName      = "empyrealm_ag_step",
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
			printf("\n─── Step-1 gradient magnitudes (traceable path, L1) ───\n");
			auto& ctx2 = OaContext::GetDefault();
			struct MagEntry { const char* name; OaMatrix result; OaI64 numel; };
			std::vector<MagEntry> entries;
			for (auto* p : params) {
				auto g = p->Data.GradMatrix();
				OaMatrix s;
				OaI64 numel = 0;
				if (!g.IsEmpty() && g.NumElements() > 0) {
					auto flat = g.Reshape(OaMatrixShape{g.NumElements()});
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
				printf("  %-32s  L1=%.6f  (numel=%lld)\n",
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

	printf("\nGeneration:\n");
	printf("  Prompt: '%s'\n", kNlpGenerationPrompt);
	printf("  Generated: '%s'\n\n", NlpGenerateGreedy(*model, kNlpGenerationPrompt,
		kNlpGenerationBytes, kVocabSize).c_str());

	EXPECT_LT(lastLoss, initialLoss);
	EXPECT_GT(finalBatchAcc, 30.0F);
	EXPECT_GT(inProjGradL1, 0.0F)  << "mixer.in_proj must receive gradient on traceable path";
	EXPECT_GT(outProjGradL1, 0.0F) << "mixer.out_proj must receive gradient on traceable path";

	const OaString ckptPath = "/tmp/empyrealm_ag.oam";
	ASSERT_TRUE(model->Save(ckptPath, *opt).IsOk());
	auto reloaded = OaMakeSharedPtr<OaEmpyrealmByteLMAg>();
	auto reloadedOpt = TutorialMakeOptimizer(*reloaded, 0.003F);
	ASSERT_TRUE(reloaded->Load(ckptPath, *reloadedOpt).IsOk());
	EXPECT_NEAR(NlpAccuracyAllPositions(*reloaded, evalX, evalY, kVocabSize), finalBatchAcc, 1.0F);
}
