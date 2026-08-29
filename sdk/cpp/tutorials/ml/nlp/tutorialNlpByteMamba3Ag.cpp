// ═══════════════════════════════════════════════════════════════════════════
// OA Tutorial — Mamba-3 Byte LM (autograd Reference)
//
// Untouched Mamba-3 reference path: oa::Embedding + oa::Mamba3Module::forward()
// (Ssm/Mamba3/Mamba3Siso* kernels) + flat token residual + Linear head.
//
// Does NOT go through oa::EmpyrealmCore or empyrealm-branded shaders.
// siblings:
//   TutorialNlpByteEmpyrealmAg — empyrealm fused/traceable SSM (oa::EmpyrealmCore)
//   TutorialNlpByteRnnAg / TutorialNlpByteGruAg — recurrent baselines
//   TutorialNlpByteTransformerAg — causal self-attention baseline
// ═══════════════════════════════════════════════════════════════════════════

#include "oaTest.h"
#include "tutorialMl.h"
#include "tutorialNlpCommon.h"
#include <oa/ml.h>
#include <oa/ml/byte.h>
#include <oa/ml/autograd.h>
#include <oa/runtime/engine.h>

static constexpr oa::I32 kVocabSize  = oa::ByteVocabSize;  // 256 — byte vocab family
// kContextLen / kDModel come from TutorialNlpCommon.h (shared suite dims).
static constexpr oa::I32 kDState     = 32;
static constexpr oa::I32 kExpand     = 2;
static constexpr oa::I32 kHeadDim    = 16;

// Direct Mamba-3 reference wiring (no oa::EmpyrealmCore).
class Mamba3ByteLM : public oa::Module {
public:
	Mamba3ByteLM() {
		embed_ = oa::makeShared<oa::Embedding>(kVocabSize, kDModel);
		mamba_ = oa::makeShared<oa::Mamba3Module>(
			kDModel, kDState, kExpand, kHeadDim,
			/*nGroups*/ 1, /*RopeFraction*/ 0.5f, /*Mimo*/ false, /*mimoRank*/ 4,
			/*dtMin*/ 0.001f, /*dtMax*/ 0.1f, /*DtInitFloor*/ 1e-4f, /*aFloor*/ 1e-4f,
			/*OutprojNorm*/ true);
		head_ = oa::makeShared<oa::Linear>(kDModel, kVocabSize);
		registerModule("embed", embed_);
		registerModule("mamba3", mamba_);
		registerModule("head", head_);
	}

	oa::Matrix forward(const oa::Matrix& inTokens) override {
		auto bs    = static_cast<oa::I64>(inTokens.size(0));
		auto sl    = static_cast<oa::I64>(inTokens.size(1));
		auto embFlat = embed_->forward(inTokens);             // flat [B*S, D]
		auto emb3d   = embFlat.reshape(oa::MatrixShape{bs, sl, kDModel});  // Mamba3 needs [B, S, D]
		auto y3d   = mamba_->forward(emb3d);     // [B, S, D] via Mamba3Siso*
		auto mixed = y3d.reshape(oa::MatrixShape{bs * sl, kDModel})
		           + embFlat.reshape(oa::MatrixShape{bs * sl, kDModel});
		return head_->forward(mixed);
	}

	void resetGenerationState(oa::I32 inBatch) {
		mamba_->resetState(inBatch);
	}

	oa::Matrix forwardGenerationStep(const oa::Matrix& inToken) {
		auto batch = static_cast<oa::I64>(inToken.size(0));
		auto embedded = embed_->forward(inToken)
			.reshape(oa::MatrixShape{batch, 1, kDModel});
		auto sequenceOutput = mamba_->step(embedded);
		return head_->forward(
			sequenceOutput.reshape(oa::MatrixShape{batch, kDModel}) +
			embedded.reshape(oa::MatrixShape{batch, kDModel}));
	}

	[[nodiscard]] oa::SharedPtr<oa::Mamba3Module> mamba() const { return mamba_; }

private:
	oa::SharedPtr<oa::Embedding>     embed_;
	oa::SharedPtr<oa::Mamba3Module>  mamba_;
	oa::SharedPtr<oa::Linear>        head_;
};

TEST(TutorialNlpByteMamba3Ag, Mamba3Reference) {
	oa::print("\n╔══════════════════════════════════════════════════════════════════╗");
	oa::print("║  OA Tutorial — Mamba-3 reference (autograd)                      ║");
	oa::print("║  oa::Mamba3Module::forward · Ssm/Mamba3/ kernels (untouched)       ║");
	oa::print("╚══════════════════════════════════════════════════════════════════╝\n");

	oa::FnMatrix::setRngSeed(oa::NlpSuiteRngSeed);
	auto  model  = oa::makeShared<Mamba3ByteLM>();
	auto  params = model->allParameterPtrs();
	// lr 0.003: with the corrected [B,S,D] batching the model reaches ~0.3 CE by
	// ~step 120; 0.01 was hot enough to diverge late once batching was fixed.
	auto  opt    = oa::makeUnique<oa::AdamW>(params, 0.003F);
	auto& rt     = testEngine();

	oa::print("Model: oa::Embedding + oa::Mamba3Module + flat residual + Linear head");
	oa::print("params: {}\n", static_cast<long long>(model->numParameters()));

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
		.timerName      = "mamba3_ref_step",
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
			oa::print("\n─── step-1 gradient magnitudes (Mamba3 reference, L1, fp32-read) ───");
			struct MagEntry { const char* name; oa::Matrix result; oa::I64 numel; };
			oa::Vector<MagEntry> entries;
			for (auto* p : params) {
				auto g = p->data.gradMatrix();
				oa::Matrix s;
				oa::I64 numel = 0;
				if (!g.isEmpty() && g.numElements() > 0) {
					auto flat = oa::FnMatrix::cast(g.reshape(oa::MatrixShape{g.numElements()}), oa::ScalarType::Float32);
					auto absg = oa::FnMatrix::abs(flat);
					s = oa::FnMatrix::sum(absg, 0);
					numel = p->data.numElements();
				}
				entries.pushBack({p->name.cStr(), oa::move(s), numel});
			}
			ASSERT_TRUE(tutorialSubmitAndWait(testEngine()).isOk());
			for (const auto& e : entries) {
				float mag = 0.0F;
				if (e.result.numElements() > 0) mag = e.result.at(0);
				oa::print("  {:<32}  L1={:.6g}  (numel={})",
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

	const oa::String generated = nlpGenerateStatefulGreedy(
		*model, kNlpGenerationPrompt, kNlpGenerationBytes, kVocabSize);
	oa::print("\nGeneration:");
	oa::print("  prompt: '{}'", kNlpGenerationPrompt);
	oa::print("  generated: '{}'\n", generated.cStr());

	EXPECT_LT(lastLoss, initialLoss);
	EXPECT_GT(finalBatchAcc, 30.0F);
	EXPECT_GT(outProjGradL1, 0.0F) << "out_proj must receive gradient on Mamba3 reference path";

	const oa::String ckptPath = "/tmp/mamba3_ref_autograd.oam";
	ASSERT_TRUE(model->save(rt, ckptPath, *opt).isOk());
	auto reloaded = oa::makeShared<Mamba3ByteLM>();
	auto reloadParams = reloaded->allParameterPtrs();
	auto reloadedOpt = oa::makeUnique<oa::AdamW>(reloadParams, 0.003F);
	ASSERT_TRUE(reloaded->load(rt, ckptPath, *reloadedOpt).isOk());
	EXPECT_NEAR(
		nlpAccuracyAllPositions(*reloaded, batchX, batchY, kVocabSize),
		finalBatchAcc, 1.0F);
	EXPECT_EQ(nlpGenerateStatefulGreedy(
		*reloaded, kNlpGenerationPrompt, kNlpGenerationBytes, kVocabSize), generated);
}
