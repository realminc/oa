// ═══════════════════════════════════════════════════════════════════════════
// OA Tutorial — empyrealm Core ag (autograd Fidelity)
//
// oa::EmpyrealmCore with UseFused=false: traceable Preprocess + EmpyrealmSiso
// + out_proj MatMul. Validates full Mamba-3 mixer autograd (in_proj, dt/B/C
// bias, D, norm_weight, out_proj) through the flat residual + CE head.
//
// siblings:
//   TutorialNlpByteMamba3Ag    — untouched oa::Mamba3Module + Ssm/Mamba3/ reference
//   TutorialNlpByteRnnAg / TutorialNlpByteGruAg — recurrent baselines
// ═══════════════════════════════════════════════════════════════════════════

#include "oaTest.h"
#include "tutorialMl.h"
#include "tutorialNlpCommon.h"
#include <oa/ml.h>
#include <oa/ml/byte.h>
#include <oa/ml/autograd.h>
#include <stdlib.h>
#include <oa/runtime/engine.h>

static bool tutorialUseMuonOptimizer() {
	const char* env = ::getenv("OA_USE_MUON");
	return env && env[0] == '1';
}

static oa::UniquePtr<oa::Optimizer> tutorialMakeOptimizer(oa::Module& inModel, oa::F32 inLr) {
	auto params = inModel.allParameterPtrs();
	if (tutorialUseMuonOptimizer()) {
		return oa::makeUnique<oa::Muon>(params, inLr);
	}
	return oa::makeUnique<oa::AdamW>(params, inLr);
}

static constexpr oa::I32 kVocabSize  = oa::ByteVocabSize;  // 256 — byte vocab family
// kContextLen / kDModel come from TutorialNlpCommon.h (shared suite dims).
static constexpr oa::I32 kDState     = 32;
static constexpr oa::I32 kExpand     = 2;
static constexpr oa::I32 kHeadDim    = 16;

class EmpyrealmByteLMAg : public oa::Module {
public:
	EmpyrealmByteLMAg() {
		core_ = oa::makeShared<oa::EmpyrealmCore>(kVocabSize, kDModel,
			kDState, kExpand, kHeadDim,
			/*nGroups*/ 1, /*RopeFraction*/ 0.5f, /*Mimo*/ false, /*mimoRank*/ 1,
			/*dtMin*/ 0.001f, /*dtMax*/ 0.1f, /*DtInitFloor*/ 1e-4f, /*aFloor*/ 1e-4f,
			/*OutprojNorm*/ true);
		head_ = oa::makeShared<oa::Linear>(kDModel, kVocabSize);
		registerModule("core", core_);
		registerModule("head", head_);
	}

	oa::Matrix forward(const oa::Matrix& inTokens) override {
		auto mixed = core_->forward(inTokens);
		return head_->forward(mixed);
	}

	[[nodiscard]] oa::SharedPtr<oa::EmpyrealmCore> core() const { return core_; }

private:
	oa::SharedPtr<oa::EmpyrealmCore> core_;
	oa::SharedPtr<oa::Linear>        head_;
};

static float paramGradL1(const oa::Matrix& g) {
	if (g.isEmpty() || g.numElements() == 0) return 0.0F;
	auto s = oa::FnMatrix::sum(oa::FnMatrix::abs(g.reshape(oa::MatrixShape{g.numElements()})), 0);
	return s.at(0);
}

TEST(TutorialNlpByteEmpyrealmAg, TraceableAutograd) {
	oa::print("\n╔══════════════════════════════════════════════════════════════════╗");
	oa::print("║  OA Tutorial — empyrealm Core ag (autograd Fidelity)             ║");
	oa::print("║  EmpyrealmPreprocess + EmpyrealmSiso traced path                 ║");
	oa::print("╚══════════════════════════════════════════════════════════════════╝\n");

	auto  model  = oa::makeShared<EmpyrealmByteLMAg>();
	auto  params = model->allParameterPtrs();
	// lr 0.003: with correct [B,S,D] batching (EmpyrealmCore::forward) the model
	// converges fast; 0.01 is hot enough to diverge late, same as the Mamba3 ref.
	auto  opt    = tutorialMakeOptimizer(*model, 0.003F);
	auto& rt     = testEngine();

	oa::print("Path: traceable split (Preprocess MatMul + EmpyrealmSiso + out_proj)");
	oa::print("params: {}    Optimizer: {}(lr=0.003)\n",
		static_cast<long long>(model->numParameters()),
		tutorialUseMuonOptimizer() ? "Muon" : "AdamW");

	constexpr oa::I32 kSteps = 300;
	constexpr oa::I32 kBatch = 64;
	NlpAllPositionSampler sampler(nlpCorpus(), kBatch);

	TutorialTrainingLoop training(rt, *opt, oa::ItTrainingConfig{
		.totalSteps     = kSteps,
		.epochSteps     = {},
		.batchSize      = kBatch,
		.sequenceLength = kContextLen,
		.sequenceUnit   = "token",
		.sourceUnitsPerSample = kContextLen,
		.sourceUnit     = "byte",
		.timerName      = "empyrealm_ag_step",
		.callbacks      = {}
	});

	// next() waits for the exact submitted step, so the sampler may safely
	// refill one reusable input pair on the following iteration.
	oa::Matrix batchX;
	oa::Matrix batchY;
	oa::F32 initialLoss = 0.0F;
	oa::F32 lastLoss    = 0.0F;
	float inProjGradL1 = 0.0F;
	float outProjGradL1 = 0.0F;

	while (not training.loop.isDone()) {
		sampler.nextBatch(batchX, batchY);
		opt->zeroGrad();
		oa::GradientTape tape;
		auto logits = model->forward(batchX);
		auto loss   = oa::FnLoss::crossEntropy(logits,
			batchY.reshape(oa::MatrixShape{batchY.size(0) * batchY.size(1)}));
		tape.backward(loss);
		training.loop.next(loss);

		if (training.loop.index() == 1) {
			initialLoss = training.loop.lastLoss();
			oa::print("\n─── step-1 gradient magnitudes (traceable path, L1) ───");
			struct MagEntry { const char* name; oa::Matrix result; oa::I64 numel; };
			oa::Vector<MagEntry> entries;
			for (auto* p : params) {
				auto g = p->data.gradMatrix();
				oa::Matrix s;
				oa::I64 numel = 0;
				if (!g.isEmpty() && g.numElements() > 0) {
					auto flat = g.reshape(oa::MatrixShape{g.numElements()});
					auto absg = oa::FnMatrix::abs(flat);
					s = oa::FnMatrix::sum(absg, 0);
					numel = p->data.numElements();
				}
				entries.pushBack({p->name.cStr(), oa::move(s), numel});
			}
			ASSERT_TRUE(tutorialSubmitAndWait(rt).isOk());
			for (const auto& e : entries) {
				float mag = 0.0F;
				if (e.result.numElements() > 0) mag = e.result.at(0);
				oa::print("  {:<32}  L1={:.6f}  (numel={})",
					e.name, mag, static_cast<long long>(e.numel));
				if (oa::strcmp(e.name, "in_proj") == 0) inProjGradL1 = mag;
				if (oa::strcmp(e.name, "out_proj") == 0) outProjGradL1 = mag;
			}
			oa::print("");
			fflush(stdout);
		}
	}

	ASSERT_TRUE(training.loop.finish().isOk());
	lastLoss = training.loop.lastLoss();

	const oa::F32 finalBatchAcc =
		nlpAccuracyAllPositions(*model, batchX, batchY, kVocabSize);
	oa::print("\nEvaluation:");
	oa::print("  Random-loss baseline ln({}) = {:.4f}",
		kVocabSize, oa::log(static_cast<double>(kVocabSize)));
	oa::print("  bits/byte: {:.4f}", nlpBitsPerByte(lastLoss));
	oa::print("  Accuracy: {:.1f}%", finalBatchAcc);

	oa::print("\nGeneration:");
	oa::print("  prompt: '{}'", kNlpGenerationPrompt);
	oa::print("  generated: '{}'\n", nlpGenerateGreedy(*model, kNlpGenerationPrompt,
		kNlpGenerationBytes, kVocabSize).cStr());

	EXPECT_LT(lastLoss, initialLoss);
	EXPECT_GT(finalBatchAcc, 30.0F);
	EXPECT_GT(inProjGradL1, 0.0F)  << "mixer.in_proj must receive gradient on traceable path";
	EXPECT_GT(outProjGradL1, 0.0F) << "mixer.out_proj must receive gradient on traceable path";

	const oa::String ckptPath = "/tmp/empyrealm_ag.oam";
	ASSERT_TRUE(model->save(rt, ckptPath, *opt).isOk());
	auto reloaded = oa::makeShared<EmpyrealmByteLMAg>();
	auto reloadedOpt = tutorialMakeOptimizer(*reloaded, 0.003F);
	ASSERT_TRUE(reloaded->load(rt, ckptPath, *reloadedOpt).isOk());
	EXPECT_NEAR(
		nlpAccuracyAllPositions(*reloaded, batchX, batchY, kVocabSize),
		finalBatchAcc, 1.0F);
}
