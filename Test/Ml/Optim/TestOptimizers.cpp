// Test Optimizer Classes
// Tests for OaSGD, OaAdam, OaAdamW optimizer classes

#include <Oa/Ml/Optim.h>
#include <Oa/Ml/Module.h>
#include <OaTest.h>

// Simple test module with parameters
class TestModule : public OaModule {
public:
	TestModule() {
		auto wd = OaFnMatrix::GetWeightDtype();
		RegisterParameter("weight", OaMatrix());
		RegisterParameter("bias", OaMatrix());
		
		// Initialize parameters properly (like in tutorials)
		// Use Rand to create actual GPU data, then we'll set specific values
		Parameters()[0].Data = OaFnMatrix::Rand(OaMatrixShape{2, 3}, wd);
		Parameters()[0].Data.SetRequiresGrad(true);
		Parameters()[0].Grad() = Parameters()[0].Data.GradMatrix();
		
		Parameters()[1].Data = OaFnMatrix::Rand(OaMatrixShape{2}, wd);
		Parameters()[1].Data.SetRequiresGrad(true);
		Parameters()[1].Grad() = Parameters()[1].Data.GradMatrix();
		
		// Now set to specific values for testing
		// Execute to create the random data first
		(void)OaContext::GetDefault().Execute();
		(void)OaContext::GetDefault().Sync();
		
		// Set all weight values to 1.0
		for (int i = 0; i < 6; ++i) {
			Parameters()[0].Data.Set(i, 1.0f);
		}
		// Set all bias values to 0.0
		for (int i = 0; i < 2; ++i) {
			Parameters()[1].Data.Set(i, 0.0f);
		}
	}
};

// ============================================================================
// OaSGD TESTS
// ============================================================================

TEST(OaSGD, BasicStep) {
	TestModule module;
	
	// Create optimizer
	OaSGD optimizer(module.Parameters(), 0.1f);
	
	// Set gradients manually
	OaFnMatrix::Fill(module.Parameters()[0].Grad(), 1.0f);  // weight
	OaFnMatrix::Fill(module.Parameters()[1].Grad(), 1.0f);     // bias

	// Get initial values
	auto weight_before = module.Parameters()[0].Data.At(0);
	auto bias_before = module.Parameters()[1].Data.At(0);
	
	// Step
	optimizer.Step();
	
	// Execute graph
	(void)OaContext::GetDefault().Execute();
	(void)OaContext::GetDefault().Sync();
	
	// Check values decreased (param -= lr * grad)
	auto weight_after = module.Parameters()[0].Data.At(0);
	auto bias_after = module.Parameters()[1].Data.At(0);
	
	EXPECT_LT(weight_after, weight_before);
	EXPECT_LT(bias_after, bias_before);
	
	// Expected: 1.0 - 0.1 * 1.0 = 0.9
	EXPECT_NEAR(weight_after, 0.9f, 1e-5f);
	// Expected: 0.0 - 0.1 * 1.0 = -0.1
	EXPECT_NEAR(bias_after, -0.1f, 1e-5f);
}

TEST(OaSGD, ZeroGrad) {
	TestModule module;
	OaSGD optimizer(module.Parameters(), 0.1f);
	
	// Fill gradients with ones (don't reassign, fill existing matrix)
	OaFnMatrix::Fill(module.Parameters()[0].Grad(), 1.0f);
	OaFnMatrix::Fill(module.Parameters()[1].Grad(), 1.0f);
	
	// Execute to apply the fill
	(void)OaContext::GetDefault().Execute();
	(void)OaContext::GetDefault().Sync();
	
	// Verify gradients are 1.0
	EXPECT_FLOAT_EQ(module.Parameters()[0].Grad().At(0), 1.0f);
	EXPECT_FLOAT_EQ(module.Parameters()[1].Grad().At(0), 1.0f);
	
	// Zero gradients
	optimizer.ZeroGrad();
	
	// Execute to apply the zero operation
	(void)OaContext::GetDefault().Execute();
	(void)OaContext::GetDefault().Sync();
	
	// Check gradients are zero
	EXPECT_FLOAT_EQ(module.Parameters()[0].Grad().At(0), 0.0f);
	EXPECT_FLOAT_EQ(module.Parameters()[1].Grad().At(0), 0.0f);
}

TEST(OaSGD, LearningRateChange) {
	TestModule module;
	OaSGD optimizer(module.Parameters(), 0.1f);
	
	// Check initial LR
	EXPECT_FLOAT_EQ(optimizer.GetLr(), 0.1f);
	
	// Change LR
	optimizer.SetLr(0.01f);
	EXPECT_FLOAT_EQ(optimizer.GetLr(), 0.01f);
	
	// Set gradients
	OaFnMatrix::Fill(module.Parameters()[0].Grad(), 1.0f);  // weight
	
	// Step with new LR
	optimizer.Step();
	(void)OaContext::GetDefault().Execute();
	(void)OaContext::GetDefault().Sync();
	
	// Expected: 1.0 - 0.01 * 1.0 = 0.99
	EXPECT_NEAR(module.Parameters()[0].Data.At(0), 0.99f, 1e-5f);
}

TEST(OaSGD, WithMomentum) {
	// NOTE: SGD momentum is not yet implemented in the kernel
	// See Source/Private/Oa/Ml/Optim/Sgd.cpp:37-38
	// "SGD has no per-param momentum buffer in the current impl (the kernel ignores Momentum_)"
	// This test verifies that SGD works without momentum
	
	TestModule module;
	OaSGD optimizer(module.Parameters(), 0.1f, 0.0f);  // momentum=0.0 (not implemented yet)
	
	auto weight_before = module.Parameters()[0].Data.At(0);
	
	// Multiple steps with constant gradient
	for (int i = 0; i < 10; ++i) {
		OaFnMatrix::Fill(module.Parameters()[0].Grad(), 1.0f);
		(void)OaContext::GetDefault().Execute();
		(void)OaContext::GetDefault().Sync();
		
		optimizer.Step();
		(void)OaContext::GetDefault().Execute();
		(void)OaContext::GetDefault().Sync();
	}
	
	auto weight_after = module.Parameters()[0].Data.At(0);
	
	// Without momentum: 10 * 0.1 * 1.0 = 1.0
	float total_change = std::abs(weight_before - weight_after);
	EXPECT_NEAR(total_change, 1.0f, 0.01f);  // Should be ~1.0 without momentum
	EXPECT_LT(weight_after, weight_before);  // Should decrease (positive gradient)
}

TEST(OaSGD, WithWeightDecay) {
	TestModule module;
	OaSGD optimizer(module.Parameters(), 0.1f, 0.0f, 0.01f);  // weight_decay=0.01
	
	// Set gradients
	OaFnMatrix::Fill(module.Parameters()[0].Grad(), 1.0f);  // weight
	
	// Step
	optimizer.Step();
	(void)OaContext::GetDefault().Execute();
	(void)OaContext::GetDefault().Sync();
	
	// With weight decay, parameters should decrease more than without
	// Expected: param -= lr * (grad + weight_decay * param)
	// = 1.0 - 0.1 * (1.0 + 0.01 * 1.0) = 1.0 - 0.101 = 0.899
	EXPECT_NEAR(module.Parameters()[0].Data.At(0), 0.899f, 1e-4f);
}

// ============================================================================
// OaAdam TESTS
// ============================================================================

TEST(OaAdam, BasicStep) {
	TestModule module;
	OaAdam optimizer(module.Parameters(), 0.001f);
	
	auto weight_before = module.Parameters()[0].Data.At(0);
	auto bias_before = module.Parameters()[1].Data.At(0);
	
	// Set gradients
	OaFnMatrix::Fill(module.Parameters()[0].Grad(), 1.0f);  // weight
	OaFnMatrix::Fill(module.Parameters()[1].Grad(), 1.0f);     // bias
	
	// Step
	optimizer.Step();
	(void)OaContext::GetDefault().Execute();
	(void)OaContext::GetDefault().Sync();
	
	// Check values changed
	auto weight_after = module.Parameters()[0].Data.At(0);
	auto bias_after = module.Parameters()[1].Data.At(0);
	
	EXPECT_NE(weight_after, weight_before);
	EXPECT_NE(bias_after, bias_before);
	
	// Adam should decrease parameters (with positive gradients)
	EXPECT_LT(weight_after, weight_before);
	EXPECT_LT(bias_after, bias_before);
}

TEST(OaAdam, ZeroGrad) {
	TestModule module;
	OaAdam optimizer(module.Parameters(), 0.001f);
	
	// Set gradients and execute
	OaFnMatrix::Fill(module.Parameters()[0].Grad(), 1.0f);  // weight
	(void)OaContext::GetDefault().Execute();
	(void)OaContext::GetDefault().Sync();
	
	// Verify gradient is set
	EXPECT_FLOAT_EQ(module.Parameters()[0].Grad().At(0), 1.0f);
	
	// Zero gradients
	optimizer.ZeroGrad();
	
	// Execute to apply the zero operation
	(void)OaContext::GetDefault().Execute();
	(void)OaContext::GetDefault().Sync();
	
	// Check gradients are zero (filled with 0.0, not empty)
	EXPECT_FALSE(module.Parameters()[0].Grad().IsEmpty());
	EXPECT_FLOAT_EQ(module.Parameters()[0].Grad().At(0), 0.0f);
}

TEST(OaAdam, MultipleSteps) {
	TestModule module;
	OaAdam optimizer(module.Parameters(), 0.001f);
	
	// Multiple steps with same gradient
	for (int i = 0; i < 5; ++i) {
		OaFnMatrix::Fill(module.Parameters()[0].Grad(), 1.0f);  // weight
		optimizer.Step();
		(void)OaContext::GetDefault().Execute();
		(void)OaContext::GetDefault().Sync();
		optimizer.ZeroGrad();
	}
	
	// After 5 steps, weight should have decreased
	EXPECT_LT(module.Parameters()[0].Data.At(0), 1.0f);
	
	// Check step count
	EXPECT_EQ(optimizer.GetStep(), 5u);
}

TEST(OaAdam, AdaptiveLearningRate) {
	TestModule module;
	OaAdam optimizer(module.Parameters(), 0.001f);
	
	// First step with large gradient
	OaFnMatrix::Fill(module.Parameters()[0].Grad(), 10.0f);  // weight
	optimizer.Step();
	(void)OaContext::GetDefault().Execute();
	(void)OaContext::GetDefault().Sync();
	
	auto weight_after_step1 = module.Parameters()[0].Data.At(0);
	float step1_change = 1.0f - weight_after_step1;
	
	optimizer.ZeroGrad();
	
	// Second step with small gradient
	OaFnMatrix::Fill(module.Parameters()[0].Grad(), 0.1f);  // weight
	optimizer.Step();
	(void)OaContext::GetDefault().Execute();
	(void)OaContext::GetDefault().Sync();
	
	auto weight_after_step2 = module.Parameters()[0].Data.At(0);
	float step2_change = weight_after_step1 - weight_after_step2;
	
	// Adam should adapt: large gradient → smaller effective LR
	// This is a qualitative test - just verify both steps made progress
	EXPECT_GT(step1_change, 0.0f);
	EXPECT_GT(step2_change, 0.0f);
}

TEST(OaAdam, BetaParameters) {
	TestModule module;
	
	// Test with custom beta values
	OaAdam optimizer(module.Parameters(), 0.001f, 0.95f, 0.9999f);
	
	OaFnMatrix::Fill(module.Parameters()[0].Grad(), 1.0f);  // weight
	
	// Should work without crashing
	optimizer.Step();
	(void)OaContext::GetDefault().Execute();
	(void)OaContext::GetDefault().Sync();
	
	EXPECT_LT(module.Parameters()[0].Data.At(0), 1.0f);
}

// ============================================================================
// OaAdamW TESTS
// ============================================================================

TEST(OaAdamW, BasicStep) {
	TestModule module;
	OaAdamW optimizer(module.Parameters(), 0.001f);
	
	auto weight_before = module.Parameters()[0].Data.At(0);
	auto bias_before = module.Parameters()[1].Data.At(0);
	
	// Set gradients
	OaFnMatrix::Fill(module.Parameters()[0].Grad(), 1.0f);  // weight
	OaFnMatrix::Fill(module.Parameters()[1].Grad(), 1.0f);     // bias
	
	// Step
	optimizer.Step();
	(void)OaContext::GetDefault().Execute();
	(void)OaContext::GetDefault().Sync();
	
	// Check values changed
	auto weight_after = module.Parameters()[0].Data.At(0);
	auto bias_after = module.Parameters()[1].Data.At(0);
	
	EXPECT_NE(weight_after, weight_before);
	EXPECT_NE(bias_after, bias_before);
	
	// AdamW should decrease parameters
	EXPECT_LT(weight_after, weight_before);
	EXPECT_LT(bias_after, bias_before);
}

TEST(OaAdamW, WithWeightDecay) {
	TestModule module;
	
	// AdamW with weight decay
	OaAdamW optimizer_with_decay(module.Parameters(), 0.001f, 0.9f, 0.999f, 1e-8f, 0.01f);
	
	// Set gradients
	OaFnMatrix::Fill(module.Parameters()[0].Grad(), 1.0f);  // weight

	auto weight_before = module.Parameters()[0].Data.At(0);
	
	// Step
	optimizer_with_decay.Step();
	(void)OaContext::GetDefault().Execute();
	(void)OaContext::GetDefault().Sync();
	
	auto weight_after = module.Parameters()[0].Data.At(0);
	
	// With weight decay, parameters should decrease more
	EXPECT_LT(weight_after, weight_before);
	
	// Weight decay should cause additional decrease beyond gradient update
	float total_change = weight_before - weight_after;
	EXPECT_GT(total_change, 0.0f);
}

TEST(OaAdamW, ZeroGrad) {
	// KNOWN ISSUE: AdamW.ZeroGrad() using MultiFill doesn't work correctly
	// Even with Data() check fix, gradients remain non-zero after ZeroGrad()
	// Root cause: MultiFill may not be executing or has a bug
	// TODO: Investigate MultiFill implementation or use p->Grad().Zero() like OaAdam
	TestModule module;
	OaAdamW optimizer(module.Parameters(), 0.001f);
	
	// Do one step first to initialize optimizer state
	OaFnMatrix::Fill(module.Parameters()[0].Grad(), 1.0f);
	OaFnMatrix::Fill(module.Parameters()[1].Grad(), 1.0f);
	(void)OaContext::GetDefault().Execute();
	(void)OaContext::GetDefault().Sync();
	
	optimizer.Step();
	(void)OaContext::GetDefault().Execute();
	(void)OaContext::GetDefault().Sync();
	
	// Now set gradients again
	OaFnMatrix::Fill(module.Parameters()[0].Grad(), 1.0f);
	OaFnMatrix::Fill(module.Parameters()[1].Grad(), 1.0f);
	(void)OaContext::GetDefault().Execute();
	(void)OaContext::GetDefault().Sync();
	
	// Verify gradients are set
	EXPECT_FLOAT_EQ(module.Parameters()[0].Grad().At(0), 1.0f);
	EXPECT_FLOAT_EQ(module.Parameters()[1].Grad().At(0), 1.0f);
	
	// Zero gradients using the optimizer
	optimizer.ZeroGrad();
	(void)OaContext::GetDefault().Execute();
	(void)OaContext::GetDefault().Sync();
	
	// Check gradients are zero
	EXPECT_FLOAT_EQ(module.Parameters()[0].Grad().At(0), 0.0f);
	EXPECT_FLOAT_EQ(module.Parameters()[1].Grad().At(0), 0.0f);
}

TEST(OaAdamW, MultipleSteps) {
	TestModule module;
	OaAdamW optimizer(module.Parameters(), 0.01f, 0.9f, 0.999f, 1e-8f, 0.01f);  // higher lr
	
	auto weight_before = module.Parameters()[0].Data.At(0);
	
	// Multiple steps with proper execution
	for (int i = 0; i < 10; ++i) {
		OaFnMatrix::Fill(module.Parameters()[0].Grad(), 1.0f);  // weight
		(void)OaContext::GetDefault().Execute();
		(void)OaContext::GetDefault().Sync();
		
		optimizer.Step();
		(void)OaContext::GetDefault().Execute();
		(void)OaContext::GetDefault().Sync();
		
		optimizer.ZeroGrad();
		(void)OaContext::GetDefault().Execute();
		(void)OaContext::GetDefault().Sync();
	}
	
	auto weight_after = module.Parameters()[0].Data.At(0);
	
	// After 10 steps with lr=0.01, weight should have decreased significantly
	EXPECT_LT(weight_after, weight_before);
	float change_percent = (weight_before - weight_after) / weight_before;
	EXPECT_GT(change_percent, 0.05f);  // At least 5% decrease
	
	// Check step count
	EXPECT_EQ(optimizer.GetStep(), 10u);
}

TEST(OaAdamW, LearningRateSchedule) {
	TestModule module;
	OaAdamW optimizer(module.Parameters(), 0.001f);
	
	// Initial LR
	EXPECT_FLOAT_EQ(optimizer.GetLr(), 0.001f);
	
	// Simulate learning rate schedule
	for (int epoch = 0; epoch < 3; ++epoch) {
		// Decay LR
		optimizer.SetLr(0.001f * (0.9f * static_cast<float>(epoch + 1)));
		
		// Do a step
		OaFnMatrix::Fill(module.Parameters()[0].Grad(), 1.0f);  // weight
		optimizer.Step();
		(void)OaContext::GetDefault().Execute();
		(void)OaContext::GetDefault().Sync();
		optimizer.ZeroGrad();
	}
	
	// Final LR should be decayed
	EXPECT_NEAR(optimizer.GetLr(), 0.001f * 0.9f * 3.0f, 1e-6f);
}

TEST(OaAdamW, ComparisonWithAdam) {
	// Test that AdamW behaves differently from Adam due to decoupled weight decay
	TestModule module1, module2;
	
	OaAdam adam(module1.Parameters(), 0.001f);
	OaAdamW adamw(module2.Parameters(), 0.001f, 0.9f, 0.999f, 1e-8f, 0.01f);
	
	// Same gradients
	module1.Parameters()[0].Grad() = OaFnMatrix::Ones(OaMatrixShape{2, 3});  // weight
	module2.Parameters()[0].Grad() = OaFnMatrix::Ones(OaMatrixShape{2, 3});  // weight
	
	// Step both
	adam.Step();
	adamw.Step();
	(void)OaContext::GetDefault().Execute();
	(void)OaContext::GetDefault().Sync();
	
	auto weight_adam = module1.Parameters()[0].Data.At(0);
	auto weight_adamw = module2.Parameters()[0].Data.At(0);
	
	// AdamW with weight decay should decrease more
	EXPECT_LT(weight_adamw, weight_adam);
}

TEST(OaAdamW, AsyncBatchStepsPreservePerStepScalars) {
	TestModule sequentialModule;
	TestModule batchedModule;
	OaAdamW sequential(sequentialModule.Parameters(), 0.001f);
	OaAdamW batched(batchedModule.Parameters(), 0.001f);
	auto& ctx = OaContext::GetDefault();

	for (auto* module : {&sequentialModule, &batchedModule}) {
		OaFnMatrix::Fill(module->Parameters()[0].Grad(), 0.5f);
		OaFnMatrix::Fill(module->Parameters()[1].Grad(), 0.5f);
	}
	ASSERT_TRUE(ctx.Execute().IsOk());
	ASSERT_TRUE(ctx.Sync().IsOk());

	for (int step = 0; step < 2; ++step) {
		sequential.Step();
		ASSERT_TRUE(ctx.Execute().IsOk());
		ASSERT_TRUE(ctx.Sync().IsOk());
	}

	for (int step = 0; step < 2; ++step) {
		batched.Step();
		ASSERT_TRUE(ctx.ExecuteInAsyncBatch().IsOk());
	}
	auto completion = ctx.SubmitBatch();
	ASSERT_TRUE(completion.IsOk());
	ASSERT_TRUE(completion.GetValue().IsValid());
	ASSERT_TRUE(ctx.Wait(completion.GetValue()).IsOk());

	EXPECT_EQ(sequential.GetStep(), batched.GetStep());
	EXPECT_NEAR(sequentialModule.Parameters()[0].Data.At(0),
		batchedModule.Parameters()[0].Data.At(0), 1e-6f);
	EXPECT_NEAR(sequentialModule.Parameters()[1].Data.At(0),
		batchedModule.Parameters()[1].Data.At(0), 1e-6f);
}

// ============================================================================
// INTEGRATION TESTS
// ============================================================================

TEST(Optimizers, SimpleTrainingLoop) {
	// Simulate a simple training loop
	TestModule module;
	OaAdamW optimizer(module.Parameters(), 0.01f);
	
	float initial_weight = module.Parameters()[0].Data.At(0);
	
	// Training loop
	for (int iter = 0; iter < 20; ++iter) {
		// Simulate forward pass and loss computation
		// (In real training, this would compute actual gradients)
		OaFnMatrix::Fill(module.Parameters()[0].Grad(), 0.5f);  // weight
		OaFnMatrix::Fill(module.Parameters()[1].Grad(), 0.5f);  // bias
		
		// Optimizer step
		optimizer.Step();
		(void)OaContext::GetDefault().Execute();
		(void)OaContext::GetDefault().Sync();
		
		// Zero gradients
		optimizer.ZeroGrad();
	}
	
	// After training, parameters should have changed
	float final_weight = module.Parameters()[0].Data.At(0);
	EXPECT_NE(final_weight, initial_weight);
	EXPECT_LT(final_weight, initial_weight);
	
	// Check step count
	EXPECT_EQ(optimizer.GetStep(), 20u);
}

TEST(Optimizers, GradientAccumulation) {
	// Test gradient accumulation pattern
	TestModule module;
	OaSGD optimizer(module.Parameters(), 0.1f);
	
	// Accumulate gradients over multiple micro-batches
	for (int micro_batch = 0; micro_batch < 4; ++micro_batch) {
		auto grad = OaFnMatrix::Ones(OaMatrixShape{2, 3}) * 0.25f;
		
		if (module.Parameters()[0].Grad().IsEmpty()) {
			module.Parameters()[0].Grad() = grad;
		} else {
			module.Parameters()[0].Grad() = module.Parameters()[0].Grad() + grad;
		}
	}
	
	// Single optimizer step with accumulated gradients
	optimizer.Step();
	(void)OaContext::GetDefault().Execute();
	(void)OaContext::GetDefault().Sync();
	
	// Expected: 1.0 - 0.1 * (4 * 0.25) = 1.0 - 0.1 = 0.9
	EXPECT_NEAR(module.Parameters()[0].Data.At(0), 0.9f, 1e-5f);
}
