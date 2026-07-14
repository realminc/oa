// ═══════════════════════════════════════════════════════════════════════════
// MlTutorialTemplate.h — THE canonical shape of an OA ML tutorial.
//
// This file is a GUIDELINE, not code to include. It fixes "what goes first" so tutorials stop
// freestyling. Every Tutorial/Ml/*.cpp should be readable as this skeleton with
// the toy model swapped out — same phase order, same helpers, same training
// loop, same save/load proof. If your tutorial needs to deviate, deviate
// *visibly* (a comment saying why), never silently.
//
// Shared helpers it leans on:  Tutorial/Ml/TutorialMl.h.
// Live exemplars to copy:
//   - Simplest OaMlTraining : Examples/Ml/MlTrainingSimple.cpp (wrapper + macro + ckpt)
//   - Classification : TutorialMnistClassifierAg.cpp  (accuracy, epochs, ckpt mgr)
//   - Sequence/LM    : TutorialNlpByteMamba3Ag.cpp     (generation, batch sampler)
//   - Multi-stage    : TutorialMotionGen.cpp           (staged loops, sidecar cfg)
//
// ─── NON-NEGOTIABLES ───────────────────────────────────────────────────────
//   1. Drive the loop with OaMlTraining (the OaRuntime wrapper, the default for
//      new tutorials) — or its in-test sibling TutorialTrainingLoop. Both wrap
//      the same OaItTraining; the iterator owns optimizer-step completion,
//      GPU timing, exact metrics, and epochs; you own the body. No bare for-loops, no
//      hand-rolled sync/submit. (This skeleton shows OaMlTraining; the mapping to
//      TutorialTrainingLoop is in OaMlTraining.md §5.)
//   2. Metrics + progress + summary come from the standard callbacks
//      (iterator metrics / OaCbProgressBar / OaCbSummary), wired automatically by
//      OaMlTraining. Do not printf a per-step loss line yourself — that is the
//      progress bar's job.
//   3. Batches are ring-buffered to ctx.MaxAsyncSubmissions(). The wrapper owns
//      the runtime, NOT your data: one reused OaMatrix would be overwritten while
//      the GPU still reads it. (A single static batch is the only exception.)
//   4. End with a Save → reload-into-fresh-model → re-eval round-trip that
//      ASSERTs the reloaded metric matches. A tutorial that can't reload is a
//      demo, not a tutorial.
//   5. It is a GTest TEST(...) (run via `ctest -L tutorial`). main() lives in
//      Test/Ml/MlTestMain.cpp — do not write your own. Inside a TEST the global
//      engine already exists, so OaMlTraining reuses it (does not own it).
//
// ─── MANDATORY PHASE ORDER ─────────────────────────────────────────────────
//   Phase 0  Constants & toy model      — file-scope constexpr dims; OaModule.
//   Phase 1  Data sampler / loader       — NextBatch(OutX, OutY); seeded; reset.
//   Phase 2  Banner                      — TutorialPrintBanner(); dataset line.
//   Phase 3  Model + optimizer           — OaMakeSharedPtr; AllParameterPtrs; OaAdamW.
//   Phase 4  Training loop config        — OaMlTraining + OaMlTrainingConfig.
//   Phase 5  Ring-buffered step loop     — while(train.Step()){ … train.Next(loss); }.
//   Phase 6  Finish()                     — ASSERT .IsOk(); drains in-flight batches.
//   Phase 7  Evaluate                     — accuracy / held-out loss / recon error.
//   Phase 8  Inference / generation       — only if generative; else skip visibly.
//   Phase 9  Assertions                   — loss fell; metric beats a floor.
//   Phase 10 Save / reload round-trip     — fresh model + opt; re-eval; ASSERT_NEAR.
//
// Classification skips Phase 8 and adds StepsPerEpoch + accuracy metric.
// Generative skips epoch boundaries and adds Phase 8 + (if cross-process) a
// sidecar .cfg next to the .oam. Multi-stage repeats Phases 3–6 per stage and
// shares one Save in Phase 10. Nothing else moves.
// ═══════════════════════════════════════════════════════════════════════════

#pragma once

// This header intentionally declares nothing. The reference implementation
// below is a block comment so it never compiles or drifts behind a real build —
// it documents the shape. Copy it into a new Tutorial/Ml/Tutorial<Name>.cpp and
// fill in the toy model + data.

/*  ── REFERENCE SKELETON — copy, rename, fill in ───────────────────────────

#include "../../Test/OaTest.h"   // TEST(...) + ASSERT_* / EXPECT_*
#include "TutorialMl.h"          // TutorialPrintBanner, helpers (TutorialTrainingLoop too)
#include <Oa/Ml.h>               // OaMlTraining (Training.h), modules, OaFnLoss, OaAdamW
#include <Oa/Ml/Autograd.h>      // OaGradientTape
#include <cstdio>
#include <vector>

// ── Phase 0: constants + toy model ─────────────────────────────────────────
// File-scope dims (constexpr, kPascalCase). Keep the model tiny — a tutorial
// proves the *pipeline*, not SOTA. One OaModule subclass; no manual Backward
// unless the whole point is to contrast with autograd (see *Api1 siblings).
static constexpr OaI32 kInDim   = 16;
static constexpr OaI32 kHidden  = 32;
static constexpr OaI32 kOutDim  = 4;

class OaTemplateModel : public OaModule {
public:
	OaTemplateModel() {
		Fc1_ = OaMakeSharedPtr<OaLinear>(kInDim, kHidden);
		Fc1_->SetActivation(OaActivation::Relu);
		Fc2_ = OaMakeSharedPtr<OaLinear>(kHidden, kOutDim);
		RegisterModule("fc1", Fc1_);   // names are the checkpoint keys — stable, descriptive
		RegisterModule("fc2", Fc2_);
	}
	OaMatrix Forward(const OaMatrix& InX) override {
		return Fc2_->Forward(Fc1_->Forward(InX));
	}
private:
	OaSharedPtr<OaLinear> Fc1_, Fc2_;
};

// ── Phase 1: data sampler ──────────────────────────────────────────────────
// One class, one NextBatch(OutX, OutY). Deterministic (seedable) so the test is
// reproducible. Real tutorials use OaDsMnist / OaDsGen3dAnim / a byte sampler;
// here a synthetic linear-separable toy stands in.
class TemplateSampler {
public:
	explicit TemplateSampler(OaI32 InBatch, OaU64 InSeed = 1234) : Batch_(InBatch), Rng_(InSeed) {}
	void NextBatch(OaMatrix& OutX, OaMatrix& OutY) {
		// fill host buffers deterministically from Rng_, upload via OaFnMatrix::FromBytes …
	}
private:
	OaI32 Batch_;
	OaU64 Rng_;
};

static OaF32 EvalAccuracy(OaTemplateModel& InModel, TemplateSampler& InEval) {
	// Forward held-out batches, argmax, Execute()+Sync() once, count correct.
	return 0.0F;
}

// ── Phase 2..10: the tutorial ──────────────────────────────────────────────
TEST(TutorialTemplate, EndToEnd) {
	const OaI32 kBatch = 64;
	const OaI32 kSteps = 300;

	// Phase 2: banner (the ONLY hand-printed header; everything per-step is callbacks).
	TutorialPrintBanner("OA Tutorial — Template", "OaModule + OaAdamW + OaGradientTape");

	// Phase 3: model + optimizer.
	auto model  = OaMakeSharedPtr<OaTemplateModel>();
	auto params = model->AllParameterPtrs();
	auto opt    = OaMakeUniquePtr<OaAdamW>(params, 0.001F);
	std::printf("Params: %lld\n\n", static_cast<long long>(model->NumParameters()));

	// Phase 4: training setup. OaMlTraining wires the runtime + the standard
	// metric/progress/summary callbacks internally — never re-printf a loss line.
	// (Inside a TEST it reuses the global engine; it does not own it.)
	TemplateSampler sampler(kBatch);
	OaMlTraining training(model, *opt, OaMlTrainingConfig{
		.TotalSteps     = kSteps,
		// .StepsPerEpoch = kStepsPerEpoch,   // classification only → epoch metrics
		//                                    // driver (0 = auto, which already does this)
		.BatchSize      = kBatch,
		.TimerName      = "template_step",
		// .EnableCheckpoints = true, .CheckpointDir = "var/model/template",
	});

	// Phase 5: ring-buffered step loop. Body = ZeroGrad → tape → Forward → loss
	// → Backward → Next(loss). OaMlTraining owns the runtime, NOT the data, so the
	// batches still ring-buffer to MaxAsyncSubmissions(). Read CurrentStep() at the
	// TOP of the body for the 1-based current step.
	auto& ctx = training.GetContext();
	OaVec<OaMatrix> xRing(static_cast<OaI64>(ctx.MaxAsyncSubmissions()));
	OaVec<OaMatrix> yRing(static_cast<OaI64>(ctx.MaxAsyncSubmissions()));
	OaF32 initialLoss = 0.0F, lastLoss = 0.0F;

	while (training.Step()) {
		const OaI64 step = training.CurrentStep();
		OaMatrix& batchX = xRing[(step - 1) % xRing.Size()];
		OaMatrix& batchY = yRing[(step - 1) % yRing.Size()];
		sampler.NextBatch(batchX, batchY);

		opt->ZeroGrad();              // implicit autograd accumulates — clear each step
		OaGradientTape tape;
		auto logits = model->Forward(batchX);
		auto loss   = OaFnLoss::CrossEntropy(logits, batchY);
		tape.Backward(loss);
		training.Next(loss);          // records one optimizer step and fires callbacks

		if (step == 1) initialLoss = training.CurrentLoss();
	}

	// Phase 6: drain.
	ASSERT_TRUE(training.Finish().IsOk());
	lastLoss = training.CurrentLoss();

	// Phase 7: evaluate on held-out data.
	const OaF32 acc = EvalAccuracy(*model, sampler);
	std::printf("Eval accuracy: %.2f%%\n\n", acc);

	// Phase 8: inference / generation — generative tutorials only. A classifier
	// prints a few predictions here instead; skip with a comment, never silently.

	// Phase 9: assertions — the test's reason to exist.
	ASSERT_GT(initialLoss, 0.0F);
	EXPECT_LT(lastLoss, initialLoss) << "loss must fall";
	EXPECT_GT(acc, 50.0F)            << "must beat chance after training";

	// Phase 10: save → reload into a FRESH model+opt → re-eval → assert match.
	// Small tutorials may use model->Save()/Load() directly (see TutorialNlpByteMamba3Ag);
	// epoch/checkpoint-managed ones use OaCheckpointManager (see TutorialMnistClassifierAg).
	const OaString ckpt = "/tmp/tutorial_template.oam";
	ASSERT_TRUE(model->Save(ckpt, *opt).IsOk());
	auto reloaded    = OaMakeSharedPtr<OaTemplateModel>();
	auto reloadOpt   = OaMakeUniquePtr<OaAdamW>(reloaded->AllParameterPtrs(), 0.001F);
	ASSERT_TRUE(reloaded->Load(ckpt, *reloadOpt).IsOk());
	EXPECT_NEAR(EvalAccuracy(*reloaded, sampler), acc, 1.0F) << "reload must reproduce eval";
}

── END REFERENCE SKELETON ──────────────────────────────────────────────────── */
