#include "../../OaTest.h"

#include <Oa/Ml/Callbacks.h>
#include <Oa/Ml/Optim.h>

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
