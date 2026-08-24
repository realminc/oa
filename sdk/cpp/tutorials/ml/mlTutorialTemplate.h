// ═══════════════════════════════════════════════════════════════════════════
// MlTutorialTemplate.h — THE canonical shape of an OA ML tutorial.
//
// This file is a GUIDELINE, not code to include. It fixes "what goes first" so tutorials stop
// freestyling. Every Tutorial/Ml/**/*.cpp should be readable as this skeleton with
// the toy model swapped out — same phase order, same helpers, same training
// loop, same save/load proof. If your tutorial needs to deviate, deviate
// *visibly* (a comment saying why), never silently.
//
// Shared helpers it leans on:  Tutorial/Ml/TutorialMl.h.
// Live exemplars to copy:
//   - Simplest oa::ItTraining : sdk/cpp/examples/ml/itTraining.cpp
//   - Classification : TutorialMnistClassifierAg.cpp  (accuracy, epochs, ckpt mgr)
//   - sequence/LM    : Nlp/TutorialNlpByteMamba3Ag.cpp (generation, batch sampler)
//   - Multi-stage    : TutorialMotionGen.cpp           (staged loops, sidecar cfg)
//
// ─── NON-NEGOTIABLES ───────────────────────────────────────────────────────
//   1. Drive the loop with oa::ItTraining or its in-test sibling
//      TutorialTrainingLoop. The iterator owns optimizer-step completion,
//      GPU timing, exact metrics, and epochs; you own the body. No bare for-loops, no
//      hand-rolled sync/submit. (This skeleton shows oa::ItTraining; the mapping to
//      TutorialTrainingLoop is documented beside its test helper.)
//   2. metrics + progress + summary come from the standard callbacks
//      (iterator metrics / oa::CbProgressBar / oa::CbSummary), composed explicitly
//      at setup. Do not printf a per-step loss line yourself — that is the
//      progress bar's job.
//   3. oa::ItTraining::next() completes the exact submitted step before returning,
//      so one reusable input pair is sufficient. A future asynchronous training
//      session must own and expose its own input-slot policy.
//   4. End with a save → reload-into-fresh-model → re-eval round-trip that
//      ASSERTs the reloaded metric matches. A tutorial that can't reload is a
//      demo, not a tutorial.
//   5. It is a GTest TEST(...) (run via `ctest -L tutorial`). main() lives in
//      Test/Ml/MlTestMain.cpp — do not write your own. Inside a TEST the global
//      engine is borrowed explicitly; oa::ItTraining never owns runtime state.
//
// ─── MANDATORY PHASE ORDER ─────────────────────────────────────────────────
//   phase 0  Constants & toy model      — file-scope constexpr dims; oa::Module.
//   phase 1  Data sampler / loader       — nextBatch(outX, outY); seeded; reset.
//   phase 2  Banner                      — tutorialPrintBanner(); dataset line.
//   phase 3  Model + optimizer           — oa::makeShared; allParameterPtrs; oa::AdamW.
//   phase 4  training loop config        — oa::ItTraining + oa::ItTrainingConfig.
//   phase 5  exact-step loop              — while(!train.isDone()){ … train.next(loss); }.
//   phase 6  finish()                     — ASSERT .isOk(); validates lifecycle completion.
//   phase 7  Evaluate                     — accuracy / held-out loss / recon error.
//   phase 8  Inference / generation       — only if generative; else skip visibly.
//   phase 9  Assertions                   — loss fell; metric beats a floor.
//   phase 10 save / reload round-trip     — fresh model + opt; re-eval; ASSERT_NEAR.
//
// Classification skips phase 8 and adds stepsPerEpoch + accuracy metric.
// Generative skips epoch boundaries and adds phase 8 + (if cross-process) a
// sidecar .cfg next to the .oam. Multi-stage repeats phases 3–6 per stage and
// shares one save in phase 10. Nothing else moves.
// ═══════════════════════════════════════════════════════════════════════════

#pragma once

// This header intentionally declares nothing. The reference implementation
// below is a block comment so it never compiles or drifts behind a real build —
// it documents the shape. Copy it into the appropriate
// Tutorial/Ml/<Domain>/ directory and fill in the toy model + data.

/*  ── REFERENCE SKELETON — copy, rename, fill in ───────────────────────────

#include "oaTest.h"   // TEST(...) + aSSERT_* / eXPECT_*
#include "tutorialMl.h"          // TutorialPrintBanner, helpers (TutorialTrainingLoop too)
#include <oa/ml.h>               // oa::ItTraining, modules, oa::FnLoss, oa::AdamW
#include <oa/ml/autograd.h>      // oa::GradientTape
#include <cstdio>
#include <vector>

// ── phase 0: constants + toy model ─────────────────────────────────────────
// file-scope dims (constexpr, kPascalCase). Keep the model tiny — a tutorial
// proves the *pipeline*, not SOTA. One oa::Module subclass; no manual backward
// unless the whole point is to contrast with autograd (see *Api1 siblings).
static constexpr oa::I32 kInDim   = 16;
static constexpr oa::I32 kHidden  = 32;
static constexpr oa::I32 kOutDim  = 4;

class TemplateModel : public oa::Module {
public:
	TemplateModel() {
		fc1_ = oa::makeShared<oa::Linear>(kInDim, kHidden);
		fc1_->setActivation(oa::Activation::Relu);
		fc2_ = oa::makeShared<oa::Linear>(kHidden, kOutDim);
		registerModule("fc1", fc1_);   // names are the checkpoint keys — stable, descriptive
		registerModule("fc2", fc2_);
	}
	oa::Matrix forward(const oa::Matrix& inX) override {
		return fc2_->forward(fc1_->forward(inX));
	}
private:
	oa::SharedPtr<oa::Linear> fc1_, fc2_;
};

// ── phase 1: data sampler ──────────────────────────────────────────────────
// One class, one nextBatch(outX, outY). deterministic (seedable) so the test is
// reproducible. Real tutorials use oa::DsMnist / oa::DsGen3dAnim / a byte sampler;
// here a synthetic linear-separable toy stands in.
class TemplateSampler {
public:
	explicit TemplateSampler(oa::I32 inBatch, oa::U64 inSeed = 1234) : batch_(inBatch), rng_(inSeed) {}
	void nextBatch(oa::Matrix& outX, oa::Matrix& outY) {
		// fill host buffers deterministically from rng_, upload via oa::FnMatrix::fromBytes …
	}
private:
	oa::I32 batch_;
	oa::U64 rng_;
};

static oa::F32 evalAccuracy(TemplateModel& inModel, TemplateSampler& inEval) {
	// forward held-out batches, argmax, then wait on the exact submission.
	return 0.0F;
}

// ── phase 2..10: the tutorial ──────────────────────────────────────────────
TEST(TutorialTemplate, EndToEnd) {
	const oa::I32 kBatch = 64;
	const oa::I32 kSteps = 300;

	// phase 2: banner (the ONLY hand-printed header; everything per-step is callbacks).
	tutorialPrintBanner("OA Tutorial — Template", "oa::Module + oa::AdamW + oa::GradientTape");

	// phase 3: model + optimizer.
	auto model  = oa::makeShared<TemplateModel>();
	auto params = model->allParameterPtrs();
	auto opt    = oa::makeUnique<oa::AdamW>(params, 0.001F);
	std::printf("params: %lld\n\n", static_cast<long long>(model->numParameters()));

	// phase 4: training setup. Compose the standard metric/progress/summary
	// callbacks explicitly — never re-printf a loss line.
	// The test engine is borrowed explicitly; training never discovers ambient
	// runtime ownership.
	TemplateSampler sampler(kBatch);
	oa::MetricLoss lossMetric;
	oa::CbProgressBar progress;
	progress.addMetric(&lossMetric);
	oa::CbSummary summary;
	oa::ItTraining training(testEngine(), *opt, oa::ItTrainingConfig{
		.totalSteps     = kSteps,
		// .stepsPerEpoch = kStepsPerEpoch,   // classification only → epoch metrics
		//                                    // driver (0 = auto, which already does this)
		.batchSize      = kBatch,
		.timerName      = "template_step",
		.metrics        = {&lossMetric},
		.callbacks      = {&progress, &summary},
	});

	// phase 5: exact-step loop. Body = zeroGrad → tape → forward → loss →
	// backward → next(loss). Next completes the submitted step, so the next
	// sampler call can safely reuse these matrices.
	oa::Matrix batchX;
	oa::Matrix batchY;
	oa::F32 initialLoss = 0.0F, lastLoss = 0.0F;

	while (not training.isDone()) {
		const oa::I64 step = training.stepCount();
		sampler.nextBatch(batchX, batchY);

		opt->zeroGrad();              // implicit autograd accumulates — clear each step
		oa::GradientTape tape;
		auto logits = model->forward(batchX);
		auto loss   = oa::FnLoss::crossEntropy(logits, batchY);
		tape.backward(loss);
		training.next(loss);          // records one optimizer step and fires callbacks

		if (step == 0) initialLoss = training.lastLoss();
	}

	// phase 6: drain.
	ASSERT_TRUE(training.finish().isOk());
	lastLoss = training.lastLoss();

	// phase 7: evaluate on held-out data.
	const oa::F32 acc = evalAccuracy(*model, sampler);
	std::printf("eval accuracy: %.2f%%\n\n", acc);

	// phase 8: inference / generation — generative tutorials only. A classifier
	// prints a few predictions here instead; skip with a comment, never silently.

	// phase 9: assertions — the test's reason to exist.
	ASSERT_GT(initialLoss, 0.0F);
	EXPECT_LT(lastLoss, initialLoss) << "loss must fall";
	EXPECT_GT(acc, 50.0F)            << "must beat chance after training";

	// phase 10: save → reload into a FRESH model+opt → re-eval → assert match.
	// Small tutorials may use model->save()/load() directly (see TutorialNlpByteMamba3Ag);
	// epoch/checkpoint-managed ones use oa::CheckpointManager (see TutorialMnistClassifierAg).
	const oa::String ckpt = "/tmp/tutorial_template.oam";
	ASSERT_TRUE(model->save(rt, ckpt, *opt).isOk());
	auto reloaded    = oa::makeShared<TemplateModel>();
	auto reloadOpt   = oa::makeUnique<oa::AdamW>(reloaded->allParameterPtrs(), 0.001F);
	ASSERT_TRUE(reloaded->load(rt, ckpt, *reloadOpt).isOk());
	EXPECT_NEAR(evalAccuracy(*reloaded, sampler), acc, 1.0F) << "reload must reproduce eval";
}

── END REFERENCE SKELETON ──────────────────────────────────────────────────── */
