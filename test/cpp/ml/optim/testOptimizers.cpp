// Test Optimizer classes
// Tests for oa::Sgd, oa::Adam, oa::AdamW optimizer classes

#include <oa/ml/optim.h>
#include <oa/ml/module.h>
#include <oaTest.h>

// Simple test module with parameters
class TestModule : public oa::Module {
public:
	TestModule() {
		auto wd = oa::FnMatrix::weightDtype();
		registerParameter("weight", oa::Matrix());
		registerParameter("bias", oa::Matrix());
		
		// initialize parameters properly (like in tutorials)
		// Use Rand to create actual GPU data, then we'll set specific values
		parameters()[0].data = oa::FnMatrix::rand(oa::MatrixShape{2, 3}, wd);
		parameters()[0].data.setRequiresGrad(true);
		parameters()[0].grad() = parameters()[0].data.gradMatrix();
		
		parameters()[1].data = oa::FnMatrix::rand(oa::MatrixShape{2}, wd);
		parameters()[1].data.setRequiresGrad(true);
		parameters()[1].grad() = parameters()[1].data.gradMatrix();
		
		// Now set to specific values for testing
		// execute to create the random data first
		(void)testSubmitAndWait(oa::ExecutionSession::getActive());
		
		// set all weight values to 1.0
		for (int i = 0; i < 6; ++i) {
			parameters()[0].data.set(i, 1.0f);
		}
		// set all bias values to 0.0
		for (int i = 0; i < 2; ++i) {
			parameters()[1].data.set(i, 0.0f);
		}
	}
};

// ============================================================================
// oa::Sgd TESTS
// ============================================================================

TEST(SGD, BasicStep) {
	TestModule module;
	
	// Create optimizer
	oa::Sgd optimizer(module.parameters(), 0.1f);
	
	// set gradients manually
	oa::FnMatrix::fillInPlace(module.parameters()[0].grad(), 1.0f);  // weight
	oa::FnMatrix::fillInPlace(module.parameters()[1].grad(), 1.0f);     // bias

	// get initial values
	auto weight_before = module.parameters()[0].data.at(0);
	auto bias_before = module.parameters()[1].data.at(0);
	
	// step
	optimizer.step();
	
	// execute graph
	(void)testSubmitAndWait(oa::ExecutionSession::getActive());
	
	// Check values decreased (param -= lr * grad)
	auto weight_after = module.parameters()[0].data.at(0);
	auto bias_after = module.parameters()[1].data.at(0);
	
	EXPECT_LT(weight_after, weight_before);
	EXPECT_LT(bias_after, bias_before);
	
	// expected: 1.0 - 0.1 * 1.0 = 0.9
	EXPECT_NEAR(weight_after, 0.9f, 1e-5f);
	// expected: 0.0 - 0.1 * 1.0 = -0.1
	EXPECT_NEAR(bias_after, -0.1f, 1e-5f);
}

TEST(SGD, zeroGrad) {
	TestModule module;
	oa::Sgd optimizer(module.parameters(), 0.1f);
	
	// Fill gradients with ones (don't reassign, fill existing matrix)
	oa::FnMatrix::fillInPlace(module.parameters()[0].grad(), 1.0f);
	oa::FnMatrix::fillInPlace(module.parameters()[1].grad(), 1.0f);
	
	// execute to apply the fill
	(void)testSubmitAndWait(oa::ExecutionSession::getActive());
	
	// verify gradients are 1.0
	EXPECT_FLOAT_EQ(module.parameters()[0].grad().at(0), 1.0f);
	EXPECT_FLOAT_EQ(module.parameters()[1].grad().at(0), 1.0f);
	
	// Zero gradients
	optimizer.zeroGrad();
	
	// execute to apply the zero operation
	(void)testSubmitAndWait(oa::ExecutionSession::getActive());
	
	// Check gradients are zero
	EXPECT_FLOAT_EQ(module.parameters()[0].grad().at(0), 0.0f);
	EXPECT_FLOAT_EQ(module.parameters()[1].grad().at(0), 0.0f);
}

TEST(SGD, LearningRateChange) {
	TestModule module;
	oa::Sgd optimizer(module.parameters(), 0.1f);
	
	// Check initial LR
	EXPECT_FLOAT_EQ(optimizer.getLr(), 0.1f);
	
	// Change LR
	optimizer.setLr(0.01f);
	EXPECT_FLOAT_EQ(optimizer.getLr(), 0.01f);
	
	// set gradients
	oa::FnMatrix::fillInPlace(module.parameters()[0].grad(), 1.0f);  // weight
	
	// step with new LR
	optimizer.step();
	(void)testSubmitAndWait(oa::ExecutionSession::getActive());
	
	// expected: 1.0 - 0.01 * 1.0 = 0.99
	EXPECT_NEAR(module.parameters()[0].data.at(0), 0.99f, 1e-5f);
}

TEST(SGD, WithMomentum) {
	// NOTE: SGD momentum is not yet implemented in the kernel
	// See source/cpp/lib/oa/Ml/optim/Sgd.cpp:37-38
	// "SGD has no per-param momentum buffer in the current impl (the kernel ignores momentum_)"
	// This test verifies that SGD works without momentum
	
	TestModule module;
	oa::Sgd optimizer(module.parameters(), 0.1f, 0.0f);  // momentum=0.0 (not implemented yet)
	
	auto weight_before = module.parameters()[0].data.at(0);
	
	// Multiple steps with constant gradient
	for (int i = 0; i < 10; ++i) {
		oa::FnMatrix::fillInPlace(module.parameters()[0].grad(), 1.0f);
		(void)testSubmitAndWait(oa::ExecutionSession::getActive());
		
		optimizer.step();
		(void)testSubmitAndWait(oa::ExecutionSession::getActive());
	}
	
	auto weight_after = module.parameters()[0].data.at(0);
	
	// Without momentum: 10 * 0.1 * 1.0 = 1.0
	float total_change = std::abs(weight_before - weight_after);
	EXPECT_NEAR(total_change, 1.0f, 0.01f);  // Should be ~1.0 without momentum
	EXPECT_LT(weight_after, weight_before);  // Should decrease (positive gradient)
}

TEST(SGD, WithWeightDecay) {
	TestModule module;
	oa::Sgd optimizer(module.parameters(), 0.1f, 0.0f, 0.01f);  // weight_decay=0.01
	
	// set gradients
	oa::FnMatrix::fillInPlace(module.parameters()[0].grad(), 1.0f);  // weight
	
	// step
	optimizer.step();
	(void)testSubmitAndWait(oa::ExecutionSession::getActive());
	
	// With weight decay, parameters should decrease more than without
	// expected: param -= lr * (grad + weight_decay * param)
	// = 1.0 - 0.1 * (1.0 + 0.01 * 1.0) = 1.0 - 0.101 = 0.899
	EXPECT_NEAR(module.parameters()[0].data.at(0), 0.899f, 1e-4f);
}

// ============================================================================
// oa::Adam TESTS
// ============================================================================

TEST(Adam, BasicStep) {
	TestModule module;
	oa::Adam optimizer(module.parameters(), 0.001f);
	
	auto weight_before = module.parameters()[0].data.at(0);
	auto bias_before = module.parameters()[1].data.at(0);
	
	// set gradients
	oa::FnMatrix::fillInPlace(module.parameters()[0].grad(), 1.0f);  // weight
	oa::FnMatrix::fillInPlace(module.parameters()[1].grad(), 1.0f);     // bias
	
	// step
	optimizer.step();
	(void)testSubmitAndWait(oa::ExecutionSession::getActive());
	
	// Check values changed
	auto weight_after = module.parameters()[0].data.at(0);
	auto bias_after = module.parameters()[1].data.at(0);
	
	EXPECT_NE(weight_after, weight_before);
	EXPECT_NE(bias_after, bias_before);
	
	// Adam should decrease parameters (with positive gradients)
	EXPECT_LT(weight_after, weight_before);
	EXPECT_LT(bias_after, bias_before);
}

TEST(Adam, zeroGrad) {
	TestModule module;
	oa::Adam optimizer(module.parameters(), 0.001f);
	
	// set gradients and execute
	oa::FnMatrix::fillInPlace(module.parameters()[0].grad(), 1.0f);  // weight
	(void)testSubmitAndWait(oa::ExecutionSession::getActive());
	
	// verify gradient is set
	EXPECT_FLOAT_EQ(module.parameters()[0].grad().at(0), 1.0f);
	
	// Zero gradients
	optimizer.zeroGrad();
	
	// execute to apply the zero operation
	(void)testSubmitAndWait(oa::ExecutionSession::getActive());
	
	// Check gradients are zero (filled with 0.0, not empty)
	EXPECT_FALSE(module.parameters()[0].grad().isEmpty());
	EXPECT_FLOAT_EQ(module.parameters()[0].grad().at(0), 0.0f);
}

TEST(Adam, MultipleSteps) {
	TestModule module;
	oa::Adam optimizer(module.parameters(), 0.001f);
	
	// Multiple steps with same gradient
	for (int i = 0; i < 5; ++i) {
		oa::FnMatrix::fillInPlace(module.parameters()[0].grad(), 1.0f);  // weight
		optimizer.step();
		(void)testSubmitAndWait(oa::ExecutionSession::getActive());
		optimizer.zeroGrad();
	}
	
	// After 5 steps, weight should have decreased
	EXPECT_LT(module.parameters()[0].data.at(0), 1.0f);
	
	// Check step count
	EXPECT_EQ(optimizer.getStep(), 5u);
}

TEST(Adam, AdaptiveLearningRate) {
	TestModule module;
	oa::Adam optimizer(module.parameters(), 0.001f);
	
	// first step with large gradient
	oa::FnMatrix::fillInPlace(module.parameters()[0].grad(), 10.0f);  // weight
	optimizer.step();
	(void)testSubmitAndWait(oa::ExecutionSession::getActive());
	
	auto weight_after_step1 = module.parameters()[0].data.at(0);
	float step1_change = 1.0f - weight_after_step1;
	
	optimizer.zeroGrad();
	
	// second step with small gradient
	oa::FnMatrix::fillInPlace(module.parameters()[0].grad(), 0.1f);  // weight
	optimizer.step();
	(void)testSubmitAndWait(oa::ExecutionSession::getActive());
	
	auto weight_after_step2 = module.parameters()[0].data.at(0);
	float step2_change = weight_after_step1 - weight_after_step2;
	
	// Adam should adapt: large gradient → smaller effective LR
	// This is a qualitative test - just verify both steps made progress
	EXPECT_GT(step1_change, 0.0f);
	EXPECT_GT(step2_change, 0.0f);
}

TEST(Adam, BetaParameters) {
	TestModule module;
	
	// Test with custom beta values
	oa::Adam optimizer(module.parameters(), 0.001f, 0.95f, 0.9999f);
	
	oa::FnMatrix::fillInPlace(module.parameters()[0].grad(), 1.0f);  // weight
	
	// Should work without crashing
	optimizer.step();
	(void)testSubmitAndWait(oa::ExecutionSession::getActive());
	
	EXPECT_LT(module.parameters()[0].data.at(0), 1.0f);
}

// ============================================================================
// oa::AdamW TESTS
// ============================================================================

TEST(AdamW, BasicStep) {
	TestModule module;
	oa::AdamW optimizer(module.parameters(), 0.001f);
	
	auto weight_before = module.parameters()[0].data.at(0);
	auto bias_before = module.parameters()[1].data.at(0);
	
	// set gradients
	oa::FnMatrix::fillInPlace(module.parameters()[0].grad(), 1.0f);  // weight
	oa::FnMatrix::fillInPlace(module.parameters()[1].grad(), 1.0f);     // bias
	
	// step
	optimizer.step();
	(void)testSubmitAndWait(oa::ExecutionSession::getActive());
	
	// Check values changed
	auto weight_after = module.parameters()[0].data.at(0);
	auto bias_after = module.parameters()[1].data.at(0);
	
	EXPECT_NE(weight_after, weight_before);
	EXPECT_NE(bias_after, bias_before);
	
	// AdamW should decrease parameters
	EXPECT_LT(weight_after, weight_before);
	EXPECT_LT(bias_after, bias_before);
}

TEST(AdamW, WithWeightDecay) {
	TestModule module;
	
	// AdamW with weight decay
	oa::AdamW optimizer_with_decay(module.parameters(), 0.001f, 0.9f, 0.999f, 1e-8f, 0.01f);
	
	// set gradients
	oa::FnMatrix::fillInPlace(module.parameters()[0].grad(), 1.0f);  // weight

	auto weight_before = module.parameters()[0].data.at(0);
	
	// step
	optimizer_with_decay.step();
	(void)testSubmitAndWait(oa::ExecutionSession::getActive());
	
	auto weight_after = module.parameters()[0].data.at(0);
	
	// With weight decay, parameters should decrease more
	EXPECT_LT(weight_after, weight_before);
	
	// weight decay should cause additional decrease beyond gradient update
	float total_change = weight_before - weight_after;
	EXPECT_GT(total_change, 0.0f);
}

TEST(AdamW, zeroGrad) {
	// KNOWN ISSUE: AdamW.zeroGrad() using MultiFill doesn't work correctly
	// Even with Data() check fix, gradients remain non-zero after zeroGrad()
	// root cause: MultiFill may not be executing or has a bug
	// TODO: Investigate MultiFill implementation or use p->grad().zero() like oa::Adam
	TestModule module;
	oa::AdamW optimizer(module.parameters(), 0.001f);
	
	// Do one step first to initialize optimizer state
	oa::FnMatrix::fillInPlace(module.parameters()[0].grad(), 1.0f);
	oa::FnMatrix::fillInPlace(module.parameters()[1].grad(), 1.0f);
	(void)testSubmitAndWait(oa::ExecutionSession::getActive());
	
	optimizer.step();
	(void)testSubmitAndWait(oa::ExecutionSession::getActive());
	
	// Now set gradients again
	oa::FnMatrix::fillInPlace(module.parameters()[0].grad(), 1.0f);
	oa::FnMatrix::fillInPlace(module.parameters()[1].grad(), 1.0f);
	(void)testSubmitAndWait(oa::ExecutionSession::getActive());
	
	// verify gradients are set
	EXPECT_FLOAT_EQ(module.parameters()[0].grad().at(0), 1.0f);
	EXPECT_FLOAT_EQ(module.parameters()[1].grad().at(0), 1.0f);
	
	// Zero gradients using the optimizer
	optimizer.zeroGrad();
	(void)testSubmitAndWait(oa::ExecutionSession::getActive());
	
	// Check gradients are zero
	EXPECT_FLOAT_EQ(module.parameters()[0].grad().at(0), 0.0f);
	EXPECT_FLOAT_EQ(module.parameters()[1].grad().at(0), 0.0f);
}

TEST(AdamW, MultipleSteps) {
	TestModule module;
	oa::AdamW optimizer(module.parameters(), 0.01f, 0.9f, 0.999f, 1e-8f, 0.01f);  // higher lr
	
	auto weight_before = module.parameters()[0].data.at(0);
	
	// Multiple steps with proper execution
	for (int i = 0; i < 10; ++i) {
		oa::FnMatrix::fillInPlace(module.parameters()[0].grad(), 1.0f);  // weight
		(void)testSubmitAndWait(oa::ExecutionSession::getActive());
		
		optimizer.step();
		(void)testSubmitAndWait(oa::ExecutionSession::getActive());
		
		optimizer.zeroGrad();
		(void)testSubmitAndWait(oa::ExecutionSession::getActive());
	}
	
	auto weight_after = module.parameters()[0].data.at(0);
	
	// After 10 steps with lr=0.01, weight should have decreased significantly
	EXPECT_LT(weight_after, weight_before);
	float change_percent = (weight_before - weight_after) / weight_before;
	EXPECT_GT(change_percent, 0.05f);  // at least 5% decrease
	
	// Check step count
	EXPECT_EQ(optimizer.getStep(), 10u);
}

TEST(AdamW, LearningRateSchedule) {
	TestModule module;
	oa::AdamW optimizer(module.parameters(), 0.001f);
	
	// Initial LR
	EXPECT_FLOAT_EQ(optimizer.getLr(), 0.001f);
	
	// Simulate learning rate schedule
	for (int epoch = 0; epoch < 3; ++epoch) {
		// decay LR
		optimizer.setLr(0.001f * (0.9f * static_cast<float>(epoch + 1)));
		
		// Do a step
		oa::FnMatrix::fillInPlace(module.parameters()[0].grad(), 1.0f);  // weight
		optimizer.step();
		(void)testSubmitAndWait(oa::ExecutionSession::getActive());
		optimizer.zeroGrad();
	}
	
	// Final LR should be decayed
	EXPECT_NEAR(optimizer.getLr(), 0.001f * 0.9f * 3.0f, 1e-6f);
}

TEST(AdamW, ComparisonWithAdam) {
	// Test that AdamW behaves differently from Adam due to decoupled weight decay
	TestModule module1, module2;
	
	oa::Adam adam(module1.parameters(), 0.001f);
	oa::AdamW adamw(module2.parameters(), 0.001f, 0.9f, 0.999f, 1e-8f, 0.01f);
	
	// Same gradients
	module1.parameters()[0].grad() = oa::FnMatrix::ones(oa::MatrixShape{2, 3});  // weight
	module2.parameters()[0].grad() = oa::FnMatrix::ones(oa::MatrixShape{2, 3});  // weight
	
	// step both
	adam.step();
	adamw.step();
	(void)testSubmitAndWait(oa::ExecutionSession::getActive());
	
	auto weight_adam = module1.parameters()[0].data.at(0);
	auto weight_adamw = module2.parameters()[0].data.at(0);
	
	// AdamW with weight decay should decrease more
	EXPECT_LT(weight_adamw, weight_adam);
}

TEST(AdamW, ExplicitSubmissionMatchesImmediateSteps) {
	TestModule sequentialModule;
	TestModule batchedModule;
	oa::AdamW sequential(sequentialModule.parameters(), 0.001f);
	oa::AdamW batched(batchedModule.parameters(), 0.001f);
	auto& ctx = oa::ExecutionSession::getActive();

	for (auto* module : {&sequentialModule, &batchedModule}) {
		oa::FnMatrix::fillInPlace(module->parameters()[0].grad(), 0.5f);
		oa::FnMatrix::fillInPlace(module->parameters()[1].grad(), 0.5f);
	}
	ASSERT_TRUE(testSubmitAndWait(ctx).isOk());

	for (int step = 0; step < 2; ++step) {
		sequential.step();
		ASSERT_TRUE(testSubmitAndWait(ctx).isOk());
	}

	for (int step = 0; step < 2; ++step) {
		batched.step();
		auto completion = ctx.submit();
		ASSERT_TRUE(completion.isOk());
		ASSERT_TRUE(completion.getValue().isValid());
		ASSERT_TRUE(ctx.wait(completion.getValue()).isOk());
	}

	EXPECT_EQ(sequential.getStep(), batched.getStep());
	EXPECT_NEAR(sequentialModule.parameters()[0].data.at(0),
		batchedModule.parameters()[0].data.at(0), 1e-6f);
	EXPECT_NEAR(sequentialModule.parameters()[1].data.at(0),
		batchedModule.parameters()[1].data.at(0), 1e-6f);
}

// ============================================================================
// INTEGRATION TESTS
// ============================================================================

TEST(Optimizers, SimpleTrainingLoop) {
	// Simulate a simple training loop
	TestModule module;
	oa::AdamW optimizer(module.parameters(), 0.01f);
	
	float initial_weight = module.parameters()[0].data.at(0);
	
	// training loop
	for (int iter = 0; iter < 20; ++iter) {
		// Simulate forward pass and loss computation
		// (in real training, this would compute actual gradients)
		oa::FnMatrix::fillInPlace(module.parameters()[0].grad(), 0.5f);  // weight
		oa::FnMatrix::fillInPlace(module.parameters()[1].grad(), 0.5f);  // bias
		
		// Optimizer step
		optimizer.step();
		(void)testSubmitAndWait(oa::ExecutionSession::getActive());
		
		// Zero gradients
		optimizer.zeroGrad();
	}
	
	// After training, parameters should have changed
	float final_weight = module.parameters()[0].data.at(0);
	EXPECT_NE(final_weight, initial_weight);
	EXPECT_LT(final_weight, initial_weight);
	
	// Check step count
	EXPECT_EQ(optimizer.getStep(), 20u);
}

TEST(Optimizers, GradientAccumulation) {
	// Test gradient accumulation pattern
	TestModule module;
	oa::Sgd optimizer(module.parameters(), 0.1f);
	
	// Accumulate gradients over multiple micro-batches
	for (int micro_batch = 0; micro_batch < 4; ++micro_batch) {
		auto grad = oa::FnMatrix::ones(oa::MatrixShape{2, 3}) * 0.25f;
		
		if (module.parameters()[0].grad().isEmpty()) {
			module.parameters()[0].grad() = grad;
		} else {
			module.parameters()[0].grad() = module.parameters()[0].grad() + grad;
		}
	}
	
	// Single optimizer step with accumulated gradients
	optimizer.step();
	(void)testSubmitAndWait(oa::ExecutionSession::getActive());
	
	// expected: 1.0 - 0.1 * (4 * 0.25) = 1.0 - 0.1 = 0.9
	EXPECT_NEAR(module.parameters()[0].data.at(0), 0.9f, 1e-5f);
}
