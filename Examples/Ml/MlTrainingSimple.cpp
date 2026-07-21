// ═══════════════════════════════════════════════════════════════════════════
// OA Tutorial — OaMlTraining Simple Example
// Demonstrates the zero-boilerplate ML training wrapper
// ═══════════════════════════════════════════════════════════════════════════

#include "../../Test/OaTest.h"
#include <Oa/Ml.h>         // Includes Training.h (OaMlTraining)
#include <Oa/Ml/Autograd.h>

// ═══════════════════════════════════════════════════════════════════════════
// Simple model for demonstration
// ═══════════════════════════════════════════════════════════════════════════

class SimpleModel : public OaModule {
public:
	SimpleModel() {
		Fc1_ = OaMakeSharedPtr<OaLinear>(10, 32);
		Fc1_->SetActivation(OaActivation::Relu);
		Fc2_ = OaMakeSharedPtr<OaLinear>(32, 2);
		RegisterModule("fc1", Fc1_);
		RegisterModule("fc2", Fc2_);
	}

	OaMatrix Forward(const OaMatrix& InX) override {
		return Fc2_->Forward(Fc1_->Forward(InX));
	}

private:
	OaSharedPtr<OaLinear> Fc1_, Fc2_;
};

// ═══════════════════════════════════════════════════════════════════════════
// Example 1: Simplest possible training loop
// ═══════════════════════════════════════════════════════════════════════════

TEST(TutorialMlTraining, SimplestUsage) {
	printf("\n╔══════════════════════════════════════════════════════════════════╗\n");
	printf("║  OA Tutorial — OaMlTraining Simplest Usage                      ║\n");
	printf("╚══════════════════════════════════════════════════════════════════╝\n\n");

	auto model = OaMakeSharedPtr<SimpleModel>();
	auto paramPtrs = model->AllParameterPtrs();
	auto opt = OaMakeUniquePtr<OaAdamW>(paramPtrs, 0.001F);

	// Create synthetic data
	auto x = OaFnMatrix::RandN(OaMatrixShape{64, 10});
	auto y = OaFnMatrix::Zeros(OaMatrixShape{64}, OaScalarType::Int32);

	printf("Training simple model: 10 -> 32 -> 2\n");
	printf("Steps: 100, Batch: 64\n\n");

	// THE ENTIRE TRAINING LOOP - just 6 lines!
	OaMlTraining train(model, *opt, OaMlTrainingConfig{
		.TotalSteps = 100,
		.BatchSize = 64,
	});

	while (train.Step()) {
		opt->ZeroGrad();
		OaGradientTape tape;
		auto logits = model->Forward(x);
		auto loss = OaFnLoss::CrossEntropy(logits, y);
		tape.Backward(loss);
		train.Next(loss);
	}

	ASSERT_TRUE(train.Finish().IsOk());
	printf("\nTraining complete!\n\n");
}

// ═══════════════════════════════════════════════════════════════════════════
// Example 2: Using the macro for even simpler code
// ═══════════════════════════════════════════════════════════════════════════

TEST(TutorialMlTraining, MacroUsage) {
	printf("\n╔══════════════════════════════════════════════════════════════════╗\n");
	printf("║  OA Tutorial — OaMlTraining Macro Usage                         ║\n");
	printf("╚══════════════════════════════════════════════════════════════════╝\n\n");

	auto model = OaMakeSharedPtr<SimpleModel>();
	auto paramPtrs = model->AllParameterPtrs();
	auto opt = OaMakeUniquePtr<OaAdamW>(paramPtrs, 0.001F);

	auto x = OaFnMatrix::RandN(OaMatrixShape{64, 10});
	auto y = OaFnMatrix::Zeros(OaMatrixShape{64}, OaScalarType::Int32);

	printf("Training with macro - even simpler!\n\n");

	// ULTRA-SIMPLE: Just use the macro
	OA_ML_TRAIN(model, *opt, 100, 64) {
		opt->ZeroGrad();
		OaGradientTape tape;
		auto logits = model->Forward(x);
		auto loss = OaFnLoss::CrossEntropy(logits, y);
		tape.Backward(loss);
		OA_ML_NEXT(loss);
	}
	OA_ML_FINISH();

	printf("\nTraining complete!\n\n");
}

// ═══════════════════════════════════════════════════════════════════════════
// Example 3: With checkpointing
// ═══════════════════════════════════════════════════════════════════════════

TEST(TutorialMlTraining, WithCheckpoints) {
	printf("\n╔══════════════════════════════════════════════════════════════════╗\n");
	printf("║  OA Tutorial — OaMlTraining With Checkpoints                    ║\n");
	printf("╚══════════════════════════════════════════════════════════════════╝\n\n");

	auto model = OaMakeSharedPtr<SimpleModel>();
	auto paramPtrs = model->AllParameterPtrs();
	auto opt = OaMakeUniquePtr<OaAdamW>(paramPtrs, 0.001F);

	auto x = OaFnMatrix::RandN(OaMatrixShape{64, 10});
	auto y = OaFnMatrix::Zeros(OaMatrixShape{64}, OaScalarType::Int32);

	printf("Training with automatic checkpointing\n");
	printf("Checkpoints every 50 steps to /tmp/ml_training_test\n\n");

	OaMlTraining train(model, *opt, OaMlTrainingConfig{
		.TotalSteps = 150,
		.BatchSize = 64,
		.EnableCheckpoints = true,
		.CheckpointDir = "/tmp/ml_training_test",
		.ModelName = "simple_model",
		.CheckpointSaveEvery = 50
	});

	while (train.Step()) {
		opt->ZeroGrad();
		OaGradientTape tape;
		auto logits = model->Forward(x);
		auto loss = OaFnLoss::CrossEntropy(logits, y);
		tape.Backward(loss);
		train.Next(loss);
	}

	ASSERT_TRUE(train.Finish().IsOk());
	printf("\nTraining complete with checkpoints saved!\n\n");
}

// ═══════════════════════════════════════════════════════════════════════════
// Example 4: Comparison with old style
// ═══════════════════════════════════════════════════════════════════════════

TEST(TutorialMlTraining, ComparisonWithOldStyle) {
	printf("\n╔══════════════════════════════════════════════════════════════════╗\n");
	printf("║  OA Tutorial — Old vs New Style Comparison                      ║\n");
	printf("╚══════════════════════════════════════════════════════════════════╝\n\n");

	printf("OLD STYLE (verbose):\n");
	printf("  auto& rt = *OaEngine::GetGlobal();\n");
	printf("  auto& ctx = OaContext::GetDefault();\n");
	printf("  OaMetricLoss lossMetric;\n");
	printf("  iterator.AddMetric(&lossMetric);\n");
	printf("  OaCbProgressBar progressBar;\n");
	printf("  progressBar.AddMetric(&lossMetric);\n");
	printf("  OaCbSummary summary;\n");
	printf("  OaItTrainingConfig itCfg;\n");
	printf("  itCfg.TotalSteps = steps;\n");
	printf("  itCfg.BatchSize = batch;\n");
	printf("  itCfg.SequenceLength = sequenceLength;\n");
	printf("  itCfg.SequenceUnit = \"token\";\n");
	printf("  OaItTraining loop(*opt, itCfg);\n");
	printf("  loop.AddCallback(&metrics);\n");
	printf("  loop.AddCallback(&progressBar);\n");
	printf("  loop.AddCallback(&summary);\n");
	printf("  while (!loop.IsDone()) {\n");
	printf("    // training code\n");
	printf("    loop.Next(loss);\n");
	printf("  }\n");
	printf("  loop.Finish();\n\n");

	printf("NEW STYLE (simple):\n");
	printf("  OaMlTraining train(model, *opt, OaMlTrainingConfig{\n");
	printf("    .TotalSteps = steps,\n");
	printf("    .BatchSize = batch\n");
	printf("  });\n");
	printf("  while (train.Step()) {\n");
	printf("    // training code\n");
	printf("    train.Next(loss);\n");
	printf("  }\n");
	printf("  train.Finish();\n\n");

	printf("MACRO STYLE (ultra-simple):\n");
	printf("  OA_ML_TRAIN(model, *opt, steps, batch) {\n");
	printf("    // training code\n");
	printf("    OA_ML_NEXT(loss);\n");
	printf("  }\n");
	printf("  OA_ML_FINISH();\n\n");

	printf("Lines of boilerplate eliminated: ~20 -> 0\n\n");
}
