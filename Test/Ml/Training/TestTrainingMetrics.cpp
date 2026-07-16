#include "../../OaTest.h"

#include <Oa/Ml/Callbacks.h>
#include <Oa/Ml/FnOptim.h>
#include <Oa/Ml/Optim.h>
#include <Oa/Ml/TrainingProgram.h>

#include <cmath>
#include <cstring>

namespace {

class SampleRecorder final : public OaCbTraining {
public:
	void OnStepEnd(OaItTraining& InIter) override {
		if (InIter.HasLossSample()) {
			LossSteps.PushBack(InIter.LastLossStep());
			LossValues.PushBack(InIter.LastLoss());
		}
	}

	OaVec<OaI64> LossSteps;
	OaVec<OaF32> LossValues;
};

} // namespace

TEST(TrainingProgram, CapturedMutableStepProgressesAcrossReplays) {
	if (not OaVkTestEngineOk()) GTEST_SKIP();

	auto& ctx = OaContext::GetDefault();
	OaMatrix param = OaFnMatrix::Full(OaMatrixShape{4}, 1.0F);
	OaMatrix grad = OaFnMatrix::Full(OaMatrixShape{4}, 0.25F);
	OaMatrix momentum;
	ASSERT_TRUE(ctx.Execute().IsOk());
	ASSERT_TRUE(ctx.Sync().IsOk());

	OaFnOptim::SgdStep(param, momentum, grad, 0.1F, 0.0F, 0.0F);
	ASSERT_EQ(ctx.NodeCount(), 1U);

	OaTrainingProgram program;
	ASSERT_TRUE(program.Capture(ctx).IsOk());
	EXPECT_TRUE(program.IsCaptured());
	EXPECT_EQ(program.NodeCount(), 1U);
	EXPECT_EQ(ctx.NodeCount(), 0U);

	for (OaU32 i = 0; i < 3; ++i) ASSERT_TRUE(program.Replay().IsOk());
	ASSERT_TRUE(program.Wait().IsOk());
	for (OaI64 i = 0; i < param.NumElements(); ++i) {
		EXPECT_NEAR(param.At(i), 0.925F, 1e-6F);
	}
	ASSERT_TRUE(program.Reset().IsOk());
}

TEST(TrainingProgram, CapturedPhiloxAdvancesWithoutFreezingRandomValues) {
	if (not OaVkTestEngineOk()) GTEST_SKIP();

	auto& ctx = OaContext::GetDefault();
	OaMatrix shape = OaFnMatrix::Zeros(OaMatrixShape{8});
	ASSERT_TRUE(ctx.Execute().IsOk());
	ASSERT_TRUE(ctx.Sync().IsOk());
	[[maybe_unused]] OaMatrix random = OaFnMatrix::PhiloxUniform(shape, 0.0F, 1.0F, 42);
	const OaU32 eagerNodes = ctx.NodeCount();
	ASSERT_GT(eagerNodes, 0U);

	OaTrainingProgram program;
	ASSERT_TRUE(program.Capture(ctx).IsOk());
	EXPECT_EQ(ctx.NodeCount(), 0U);
	EXPECT_EQ(program.NodeCount(), eagerNodes + 1U); // RNG + counter advance

	ASSERT_TRUE(program.Replay().IsOk());
	ASSERT_TRUE(program.Wait().IsOk());
	OaF32 first[8]{};
	std::memcpy(first, random.Data(), sizeof(first));
	ASSERT_TRUE(program.Replay().IsOk());
	ASSERT_TRUE(program.Wait().IsOk());
	OaF32 second[8]{};
	std::memcpy(second, random.Data(), sizeof(second));

	OaBool changed = false;
	for (OaU32 i = 0; i < 8; ++i) {
		EXPECT_GE(first[i], 0.0F);
		EXPECT_LT(first[i], 1.0F);
		EXPECT_GE(second[i], 0.0F);
		EXPECT_LT(second[i], 1.0F);
		changed = changed or first[i] != second[i];
	}
	EXPECT_TRUE(changed);
	ASSERT_TRUE(program.Reset().IsOk());
}

TEST(TrainingProgram, IteratorRecordsTwiceThenReplaysFixedShapeStep) {
	if (not OaVkTestEngineOk()) GTEST_SKIP();

	auto& ctx = OaContext::GetDefault();
	OaMatrix param = OaFnMatrix::Full(OaMatrixShape{4}, 1.0F);
	OaMatrix grad = OaFnMatrix::Full(OaMatrixShape{4}, 0.25F);
	OaMatrix momentum;
	ASSERT_TRUE(ctx.Execute().IsOk());
	ASSERT_TRUE(ctx.Sync().IsOk());

	OaOptimizerNoOp optimizer;
	OaTrainingProgram program;
	OaItTraining iter(optimizer, OaItTrainingConfig{
		.TotalSteps = 4,
		.Program = &program,
	});
	OaI32 prepareCalls = 0;
	OaI32 recordCalls = 0;
	while (not iter.IsDone()) {
		iter.Step(
			[&] { ++prepareCalls; },
			[&] {
				++recordCalls;
				OaFnOptim::SgdStep(param, momentum, grad, 0.1F, 0.0F, 0.0F);
				iter.RecordLoss(param);
			});
	}
	ASSERT_TRUE(iter.Finish().IsOk());

	EXPECT_TRUE(program.IsCaptured());
	EXPECT_EQ(prepareCalls, 4);
	EXPECT_EQ(recordCalls, 2); // eager warm-up + capture, never rebuilt afterward
	EXPECT_NEAR(param.At(0), 0.9F, 1e-6F);
	EXPECT_NEAR(iter.LastLoss(), 0.9F, 1e-6F);
	EXPECT_EQ(iter.GpuTimingStats().Count, 3);
	EXPECT_GT(iter.LastGpuMs(), 0.0);
}

TEST(TrainingProgram, ExplicitRecaptureRecordsNewProgramOnce) {
	if (not OaVkTestEngineOk()) GTEST_SKIP();

	auto& ctx = OaContext::GetDefault();
	OaMatrix param = OaFnMatrix::Full(OaMatrixShape{4}, 1.0F);
	OaMatrix grad = OaFnMatrix::Full(OaMatrixShape{4}, 0.25F);
	OaMatrix momentum;
	ASSERT_TRUE(ctx.Execute().IsOk());
	ASSERT_TRUE(ctx.Sync().IsOk());

	OaOptimizerNoOp optimizer;
	OaTrainingProgram program;
	OaItTraining iter(optimizer, OaItTrainingConfig{
		.TotalSteps = 4,
		.Program = &program,
	});
	OaI32 recordCalls = 0;
	while (not iter.IsDone()) {
		if (iter.Index() == 3) ASSERT_TRUE(iter.RequestProgramRecapture().IsOk());
		iter.Step(
			[] {},
			[&] {
				++recordCalls;
				OaFnOptim::SgdStep(param, momentum, grad, 0.1F, 0.0F, 0.0F);
				iter.RecordLoss(param);
			});
	}
	ASSERT_TRUE(iter.Finish().IsOk());

	EXPECT_TRUE(program.IsCaptured());
	EXPECT_EQ(recordCalls, 3); // warm-up, initial capture, explicit recapture
	EXPECT_NEAR(param.At(0), 0.9F, 1e-6F);
}

TEST(TrainingProgram, AdamWMatchesReferenceAndAcceptsReplayLrUpdate) {
	if (not OaVkTestEngineOk()) GTEST_SKIP();

	auto& ctx = OaContext::GetDefault();
	OaParameter param;
	param.Name = "weight";
	param.Data = OaFnMatrix::Full(OaMatrixShape{4}, 1.0F);
	param.Data.SetRequiresGrad(true);
	OaFnMatrix::Fill(param.Grad(), 0.25F);
	ASSERT_TRUE(ctx.Execute().IsOk());
	ASSERT_TRUE(ctx.Sync().IsOk());

	OaVec<OaParameter*> params;
	params.PushBack(&param);
	constexpr OaF32 beta1 = 0.9F;
	constexpr OaF32 beta2 = 0.999F;
	constexpr OaF32 eps = 1e-8F;
	constexpr OaF32 decay = 0.01F;
	OaAdamW optimizer(params, 0.1F, beta1, beta2, eps, decay);
	OaTrainingProgram program;
	OaItTraining iter(optimizer, OaItTrainingConfig{
		.TotalSteps = 4,
		.Program = &program,
	});
	while (not iter.IsDone()) {
		iter.Step(
			[] {},
			[&] { iter.RecordLoss(param.Data); });
		if (iter.Index() == 2) optimizer.SetLr(0.05F);
	}
	ASSERT_TRUE(iter.Finish().IsOk());

	OaF32 expected = 1.0F;
	OaF32 m = 0.0F;
	OaF32 v = 0.0F;
	for (OaI32 step = 1; step <= 4; ++step) {
		const OaF32 lr = step <= 2 ? 0.1F : 0.05F;
		expected -= lr * decay * expected;
		m = beta1 * m + (1.0F - beta1) * 0.25F;
		v = beta2 * v + (1.0F - beta2) * 0.25F * 0.25F;
		const OaF32 mHat = m / (1.0F - std::pow(beta1, static_cast<OaF32>(step)));
		const OaF32 vHat = v / (1.0F - std::pow(beta2, static_cast<OaF32>(step)));
		expected -= lr * mHat / (std::sqrt(vHat) + eps);
	}

	EXPECT_TRUE(program.IsCaptured());
	EXPECT_EQ(optimizer.GetStep(), 4U);
	EXPECT_NEAR(param.Data.At(0), expected, 2e-5F);
	EXPECT_NEAR(iter.LastLoss(), expected, 2e-5F);
}

TEST(TrainingMetrics, LossIsCountedExactlyOncePerStep) {
	if (not OaVkTestEngineOk()) GTEST_SKIP();

	OaOptimizerNoOp opt;
	OaMetricLoss lossMetric;
	OaMetricLoss lastLossMetric("loss", OaMetricLoss::Mode::Last);
	SampleRecorder recorder;

	OaItTraining iter(opt, OaItTrainingConfig{
		.TotalSteps = 5,
		.BatchSize = 4,
		.Metrics = {&lossMetric, &lastLossMetric},
		.Callbacks = {&recorder},
	});

	while (not iter.IsDone()) {
		const OaF32 value = static_cast<OaF32>(iter.StepCount());
		iter.Next(OaFnMatrix::Full(OaMatrixShape{1}, value));
	}
	ASSERT_TRUE(iter.Finish().IsOk());

	ASSERT_EQ(recorder.LossSteps.Size(), 5U);
	EXPECT_EQ(recorder.LossSteps[0], 1);
	EXPECT_EQ(recorder.LossSteps[1], 2);
	EXPECT_EQ(recorder.LossSteps[2], 3);
	EXPECT_EQ(recorder.LossSteps[3], 4);
	EXPECT_EQ(recorder.LossSteps[4], 5);
	EXPECT_FLOAT_EQ(recorder.LossValues[0], 1.0F);
	EXPECT_FLOAT_EQ(recorder.LossValues[1], 2.0F);
	EXPECT_FLOAT_EQ(recorder.LossValues[2], 3.0F);
	EXPECT_FLOAT_EQ(recorder.LossValues[3], 4.0F);
	EXPECT_FLOAT_EQ(recorder.LossValues[4], 5.0F);
	EXPECT_DOUBLE_EQ(lossMetric.Result(), 3.0);
	EXPECT_DOUBLE_EQ(lastLossMetric.Result(), 5.0);
	EXPECT_EQ(iter.EpochLossCount(), 5);
	EXPECT_DOUBLE_EQ(iter.EpochMeanLoss(), 3.0);
}

TEST(TrainingMetrics, WorkloadRatesHaveUnambiguousUnits) {
	if (not OaVkTestEngineOk()) GTEST_SKIP();

	OaOptimizerNoOp opt;
	OaItTraining iter(opt, OaItTrainingConfig{
		.TotalSteps = 4,
		.BatchSize = 8,
		.SequenceLength = 4,
		.SequenceUnit = "token",
		.SourceUnitsPerSample = 4.0,
		.SourceUnit = "byte",
	});

	while (not iter.IsDone()) {
		iter.Next(OaFnMatrix::Full(OaMatrixShape{1}, 1.0F));
	}
	ASSERT_TRUE(iter.Finish().IsOk());

	EXPECT_GT(iter.WallMsPerStep(), 0.0);
	EXPECT_GT(iter.WallSamplesPerSecond(), 0.0);
	EXPECT_GT(iter.WallUnitsPerSecond(), 0.0);
	EXPECT_NEAR(iter.WallUnitsPerSecond(), iter.WallSamplesPerSecond() * 4.0,
		iter.WallSamplesPerSecond() * 0.01);
	EXPECT_EQ(iter.TotalSamples(), 32);
	EXPECT_EQ(iter.TotalUnits(), 128);
	EXPECT_EQ(iter.TotalSourceUnits(), 128);
	EXPECT_NEAR(iter.WallSourceUnitsPerSecond(), iter.WallSamplesPerSecond() * 4.0,
		iter.WallSamplesPerSecond() * 0.01);
}

TEST(TrainingMetrics, VariableSourceWorkOverridesFixedRate) {
	if (not OaVkTestEngineOk()) GTEST_SKIP();
	OaOptimizerNoOp opt;
	OaItTraining iter(opt, OaItTrainingConfig{
		.TotalSteps = 3,
		.BatchSize = 2,
		.SequenceLength = 4,
		.SequenceUnit = "token",
		.SourceUnit = "byte",
	});
	const OaI64 bytes[] = {11, 13, 17};
	while (not iter.IsDone()) {
		iter.RecordSourceUnits(bytes[iter.Index() - 1]);
		iter.Next(OaFnMatrix::Full(OaMatrixShape{1}, 1.0F));
	}
	ASSERT_TRUE(iter.Finish().IsOk());
	EXPECT_EQ(iter.TotalSourceUnits(), 41);
	EXPECT_GT(iter.WallSourceUnitsPerSecond(), 0.0);
}

TEST(TrainingMetrics, BatchDefaultsToOneSample) {
	if (not OaVkTestEngineOk()) GTEST_SKIP();
	OaOptimizerNoOp opt;
	OaItTraining iter(opt, OaItTrainingConfig{.TotalSteps = 3});
	while (not iter.IsDone()) iter.Next(OaFnMatrix::Full(OaMatrixShape{1}, 1.0F));
	ASSERT_TRUE(iter.Finish().IsOk());
	EXPECT_EQ(iter.Cfg().BatchSize, 1);
	EXPECT_EQ(iter.TotalSamples(), 3);
}

TEST(TrainingMetrics, LiveLossRenderOmitsAggregationSuffix) {
	OaMetricLoss meanLoss("recon");
	meanLoss.Update(0.25F);
	meanLoss.Update(0.75F);
	char buffer[64]{};
	ASSERT_GT(meanLoss.Render(buffer, sizeof(buffer), false), 0);
	EXPECT_STREQ(buffer, "recon: 0.5");
	EXPECT_EQ(std::strchr(buffer, '('), nullptr);
}

TEST(TrainingMetrics, LiveValueFormattingExpandsOnlyWhenRoundedValueStalls) {
	OaMetricLoss lastLoss("recon", OaMetricLoss::Mode::Last);
	char buffer[64]{};
	lastLoss.Update(0.23444F);
	ASSERT_GT(lastLoss.Render(buffer, sizeof(buffer), false), 0);
	EXPECT_STREQ(buffer, "recon: 0.2344");
	lastLoss.Update(0.234449F);
	ASSERT_GT(lastLoss.Render(buffer, sizeof(buffer), false), 0);
	EXPECT_STREQ(buffer, "recon: 0.23445");

	OaMetricValueFormatter smallValue;
	ASSERT_GT(smallValue.Format(buffer, sizeof(buffer), 4.95e-5), 0);
	EXPECT_STREQ(buffer, "0.0000495");
}

TEST(TrainingMetrics, AccuracyReducesLogitsOnGpu) {
	if (not OaVkTestEngineOk()) GTEST_SKIP();
	const std::vector<OaF32> logitsHost = {
		4, 1, 0,  0, 5, 1,  0, 2, 3,  3, 2, 1,
	};
	const std::vector<OaU8> labelsHost = {0, 1, 1, 0}; // row 2 intentionally wrong
	auto logits = OaFnMatrix::FromBytes(
		OaSpan<const OaU8>(reinterpret_cast<const OaU8*>(logitsHost.data()), logitsHost.size() * sizeof(OaF32)),
		OaMatrixShape{4, 3}, OaScalarType::Float32);
	auto labels = OaFnMatrix::FromBytes(
		OaSpan<const OaU8>(labelsHost.data(), labelsHost.size()),
		OaMatrixShape{4}, OaScalarType::UInt8);
	OaMetricAccuracy accuracy;
	accuracy.Update(logits, labels);
	EXPECT_DOUBLE_EQ(accuracy.Result(), 0.75);
	EXPECT_EQ(accuracy.Count(), 4);
}

// ─── Callback Lifecycle ───────────────────────────────────────────────────

class LifecycleRecorder final : public OaCbTraining {
public:
	void OnTrainBegin(OaItTraining&) override { ++TrainBeginCount; }
	void OnEpochBegin(OaItTraining&) override { ++EpochBeginCount; }
	void OnStepEnd(OaItTraining&) override { ++StepEndCount; }
	void OnEpochEnd(OaItTraining&) override { ++EpochEndCount; }
	void OnTrainEnd(OaItTraining&) override { ++TrainEndCount; }

	int TrainBeginCount = 0;
	int EpochBeginCount = 0;
	int StepEndCount = 0;
	int EpochEndCount = 0;
	int TrainEndCount = 0;
};

TEST(TrainingMetrics, CallbackLifecycleOrder) {
	if (not OaVkTestEngineOk()) GTEST_SKIP();

	OaOptimizerNoOp opt;
	LifecycleRecorder rec;
	OaItTraining iter(opt, OaItTrainingConfig{
		.TotalSteps = 6,
		.StepsPerEpoch = 3,
		.BatchSize = 2,
		.Callbacks = {&rec},
	});

	while (not iter.IsDone()) {
		iter.Next(OaFnMatrix::Full(OaMatrixShape{1}, 1.0F));
	}
	ASSERT_TRUE(iter.Finish().IsOk());

	EXPECT_EQ(rec.TrainBeginCount, 1);
	EXPECT_EQ(rec.EpochBeginCount, 2);  // epoch 1 and epoch 2
	EXPECT_EQ(rec.StepEndCount, 6);
	EXPECT_EQ(rec.EpochEndCount, 2);    // after step 3 and step 6
	EXPECT_EQ(rec.TrainEndCount, 1);
}

TEST(TrainingMetrics, ValidationRunsOncePerEpochAndPublishesLastLoss) {
	if (not OaVkTestEngineOk()) GTEST_SKIP();

	OaOptimizerNoOp opt;
	OaI32 evalCount = 0;
	OaCbValidation validation([&evalCount](OaItTraining& InIter) {
		++evalCount;
		return OaValidationResult{
			.Loss = 1.0 / static_cast<OaF64>(InIter.Epoch()),
			.Batches = 2,
			.Samples = 8,
		};
	});
	OaItTraining iter(opt, OaItTrainingConfig{
		.TotalSteps = 6,
		.StepsPerEpoch = 3,
		.BatchSize = 4,
		.Callbacks = {&validation},
	});

	while (not iter.IsDone()) {
		iter.Next(OaFnMatrix::Full(OaMatrixShape{1}, 1.0F));
	}
	ASSERT_TRUE(iter.Finish().IsOk());

	EXPECT_EQ(evalCount, 2);
	EXPECT_DOUBLE_EQ(validation.Metric().Result(), 0.5);
	EXPECT_EQ(validation.LastResult().Batches, 2);
	EXPECT_EQ(validation.LastResult().Samples, 8);
	EXPECT_GE(validation.LastSeconds(), 0.0);
}

TEST(TrainingMetrics, PartialFinalEpochStillCloses) {
	if (not OaVkTestEngineOk()) GTEST_SKIP();
	OaOptimizerNoOp opt;
	LifecycleRecorder rec;
	OaItTraining iter(opt, OaItTrainingConfig{
		.TotalSteps = 5,
		.StepsPerEpoch = 3,
		.Callbacks = {&rec},
	});
	while (not iter.IsDone()) iter.Next(OaFnMatrix::Full(OaMatrixShape{1}, 1.0F));
	ASSERT_TRUE(iter.Finish().IsOk());
	EXPECT_EQ(iter.TotalEpochs(), 2);
	EXPECT_EQ(rec.EpochBeginCount, 2);
	EXPECT_EQ(rec.EpochEndCount, 2);
	EXPECT_EQ(iter.StepsInCurrentEpoch(), 2);
}

// ─── Epoch Boundary Sampling ──────────────────────────────────────────────

class EpochBoundaryRecorder final : public OaCbTraining {
public:
	void OnStepEnd(OaItTraining& InIter) override {
		if (InIter.HasLossSample()) {
			EpochAtSample.PushBack(InIter.Epoch());
			StepInEpochAtSample.PushBack(InIter.StepInEpoch());
			LossAtSample.PushBack(InIter.LastLoss());
		}
	}
	void OnEpochEnd(OaItTraining& InIter) override {
		EpochEndCount.PushBack(static_cast<OaF32>(InIter.Epoch()));
		MeanLossAtEpochEnd.PushBack(static_cast<OaF32>(InIter.EpochMeanLoss()));
	}

	OaVec<OaI64> EpochAtSample;
	OaVec<OaI64> StepInEpochAtSample;
	OaVec<OaF32> LossAtSample;
	OaVec<OaF32> EpochEndCount;
	OaVec<OaF32> MeanLossAtEpochEnd;
};

TEST(TrainingMetrics, FirstFinalAndEpochBoundaryAreSampled) {
	if (not OaVkTestEngineOk()) GTEST_SKIP();

	OaOptimizerNoOp opt;
	EpochBoundaryRecorder rec;
	OaItTraining iter(opt, OaItTrainingConfig{
		.TotalSteps = 9,
		.StepsPerEpoch = 3,
		.BatchSize = 2,
		.Callbacks = {&rec},
	});

	while (not iter.IsDone()) {
		const OaF32 value = static_cast<OaF32>(iter.StepCount());
		iter.Next(OaFnMatrix::Full(OaMatrixShape{1}, value));
	}
	ASSERT_TRUE(iter.Finish().IsOk());

	ASSERT_EQ(rec.LossAtSample.Size(), 9U);
	EXPECT_EQ(rec.StepInEpochAtSample[0], 1);  // step 1
	EXPECT_EQ(rec.StepInEpochAtSample[1], 2);  // step 2
	EXPECT_EQ(rec.StepInEpochAtSample[2], 3);  // step 3 (boundary)
	EXPECT_EQ(rec.StepInEpochAtSample[3], 1);  // step 4
	EXPECT_EQ(rec.StepInEpochAtSample[4], 2);  // step 5
	EXPECT_EQ(rec.StepInEpochAtSample[5], 3);  // step 6 (boundary)
	EXPECT_EQ(rec.StepInEpochAtSample[6], 1);  // step 7
	EXPECT_EQ(rec.StepInEpochAtSample[7], 2);  // step 8
	EXPECT_EQ(rec.StepInEpochAtSample[8], 3);  // step 9 (final+boundary)

	// Verify final step was actually captured
	EXPECT_FLOAT_EQ(rec.LossAtSample.Back(), 9.0F);
}

TEST(TrainingMetrics, EpochResetDoesNotDiscardOrDuplicateBoundarySample) {
	if (not OaVkTestEngineOk()) GTEST_SKIP();

	OaOptimizerNoOp opt;
	EpochBoundaryRecorder rec;
	OaItTraining iter(opt, OaItTrainingConfig{
		.TotalSteps = 8,
		.StepsPerEpoch = 4,
		.BatchSize = 2,
		.Callbacks = {&rec},
	});

	while (not iter.IsDone()) {
		const OaF32 value = static_cast<OaF32>(iter.StepCount());
		iter.Next(OaFnMatrix::Full(OaMatrixShape{1}, value));
	}
	ASSERT_TRUE(iter.Finish().IsOk());

	// Every completed step is sampled.
	// Epoch 1: steps 1,2,3,4  -> mean = (1+2+3+4)/4 = 2.5
	// Epoch 2: steps 5,6,7,8  -> mean = (5+6+7+8)/4 = 6.5
	ASSERT_EQ(rec.LossAtSample.Size(), 8U);
	ASSERT_EQ(rec.EpochEndCount.Size(), 2U);

	EXPECT_FLOAT_EQ(rec.MeanLossAtEpochEnd[0], 2.5F);
	EXPECT_FLOAT_EQ(rec.MeanLossAtEpochEnd[1], 6.5F);

	// Verify no boundary sample leaked into wrong epoch
	for (size_t i = 0; i < 4; ++i) EXPECT_EQ(rec.EpochAtSample[i], 1);
	for (size_t i = 4; i < 8; ++i) EXPECT_EQ(rec.EpochAtSample[i], 2);
}

TEST(TrainingMetrics, EarlyExitAlreadyHasExactFinalSample) {
	if (not OaVkTestEngineOk()) GTEST_SKIP();

	OaOptimizerNoOp opt;
	OaItTraining iter(opt, OaItTrainingConfig{
		.TotalSteps = 10,  // declare 10 but only run 2
		.BatchSize = 2,
	});

	// Run only 2 steps.
	// Must call IsDone() before each Next() so Index_ advances properly.
	int step = 0;
	while (not iter.IsDone() and step < 2) {
		iter.Next(OaFnMatrix::Full(OaMatrixShape{1}, static_cast<OaF32>(iter.StepCount())));
		++step;
	}

	// The completed step is already exact before Finish.
	EXPECT_EQ(iter.LastLossStep(), 2);
	EXPECT_FLOAT_EQ(iter.LastLoss(), 2.0F);

	ASSERT_TRUE(iter.Finish().IsOk());

	// Finish does not manufacture another sample.
	EXPECT_EQ(iter.LastLossStep(), 2);
	EXPECT_FLOAT_EQ(iter.LastLoss(), 2.0F);
}
