#include "../../OaTest.h"

#include <Oa/Ml/Callbacks.h>
#include <Oa/Ml/FnOptim.h>
#include <Oa/Ml/Optim.h>
#include <Oa/Ml/TrainingProgram.h>
#include <Oa/Ml/TrainingSession.h>
#include <Oa/Ui/TrainingViewer.h>

#include <Oa/Runtime/ExecutionMemory.h>

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

class FailingTrainingCallback final : public OaCbTraining {
public:
	void OnStepEnd(OaItTraining&) override {
		Status_ = OaStatus::Error(OaStatusCode::DataLoss,
			"injected checkpoint callback failure");
	}

	[[nodiscard]] OaStatus GetStatus() const override { return Status_; }

private:
	OaStatus Status_ = OaStatus::Ok();
};

} // namespace

TEST(TrainingCallbacks, FailureStopsAndPropagatesThroughFinish) {
	if (not OaVkTestEngineOk()) GTEST_SKIP();
	OaOptimizerNoOp optimizer;
	FailingTrainingCallback callback;
	OaItTraining training(optimizer, OaItTrainingConfig{
		.TotalSteps = 2,
		.EnableGpuTiming = false,
		.Callbacks = {&callback},
	});
	ASSERT_FALSE(training.IsDone());
	training.Next();
	EXPECT_TRUE(training.StopRequested());
	EXPECT_EQ(training.LastStatus().GetCode(), OaStatusCode::DataLoss);
	const auto status = training.Finish();
	EXPECT_EQ(status.GetCode(), OaStatusCode::DataLoss);
}

TEST(TrainingSession, CommandsAreTypedRevisionedAndSafePointApplied) {
	OaOptimizerNoOp optimizer;
	optimizer.SetLr(1.0e-3F);
	OaItTraining training(optimizer, OaItTrainingConfig{.TotalSteps = 2});
	OaI32 checkpointCount = 0;
	OaI32 evaluationCount = 0;
	OaTrainingSession session(training, OaTrainingSessionConfig{
		.Handlers = {
			.Checkpoint = [&] {
				++checkpointCount;
				return OaStatus::Ok();
			},
			.Evaluate = [&] {
				++evaluationCount;
				return OaStatus::Ok();
			},
		},
	});

	EXPECT_EQ(session.Revision(), 1U);
	ASSERT_TRUE(session.Pause().IsOk());
	EXPECT_FALSE(session.TryBeginStep());
	EXPECT_EQ(session.State(), OaTrainingState::Paused);
	const OaU64 pausedRevision = session.Revision();

	ASSERT_TRUE(session.SetParameter(
		"learning_rate", OaTrainingValue::FromFloat(2.5e-4),
		pausedRevision).IsOk());
	ASSERT_TRUE(session.Checkpoint().IsOk());
	ASSERT_TRUE(session.Evaluate().IsOk());
	ASSERT_TRUE(session.Resume().IsOk());
	ASSERT_TRUE(session.Poll().IsOk());
	EXPECT_EQ(session.State(), OaTrainingState::Running);
	EXPECT_FLOAT_EQ(optimizer.GetLr(), 2.5e-4F);
	EXPECT_EQ(checkpointCount, 1);
	EXPECT_EQ(evaluationCount, 1);

	// The old revision was accepted by Enqueue but rejected atomically at the
	// next safe point after preceding commands advanced the session revision.
	ASSERT_TRUE(session.SetParameter(
		"learning_rate", OaTrainingValue::FromFloat(9.0e-4),
		pausedRevision).IsOk());
	ASSERT_TRUE(session.Poll().IsOk());
	EXPECT_FLOAT_EQ(optimizer.GetLr(), 2.5e-4F);

	const auto observed = session.ResultsAfter(0);
	ASSERT_EQ(observed.Size(), 6U);
	EXPECT_EQ(session.ResultsAfter(0).Size(), 6U);
	EXPECT_TRUE(session.ResultsAfter(observed.Back().Sequence).Empty());

	auto results = session.TakeResults();
	ASSERT_EQ(results.Size(), 6U);
	EXPECT_TRUE(session.TakeResults().Empty());
	EXPECT_EQ(session.ResultsAfter(0).Size(), 6U);
	EXPECT_EQ(results[0].Disposition, OaTrainingCommandDisposition::Applied);
	EXPECT_EQ(results[5].Disposition, OaTrainingCommandDisposition::Rejected);
	EXPECT_EQ(results[5].Status.GetCode(), OaStatusCode::Aborted);
}

TEST(TrainingSession, IteratorPublishesBoundedSnapshotsAndStop) {
	if (not OaVkTestEngineOk()) GTEST_SKIP();
	OaOptimizerNoOp optimizer;
	OaItTraining training(optimizer, OaItTrainingConfig{.TotalSteps = 3});
	OaTrainingSession session(training, OaTrainingSessionConfig{
		.SnapshotCapacity = 2,
	});
	OaTrainingViewerSource viewer(session, {
		.HistoryCapacity = 2,
		.MaxMetricPlots = 4,
	});
	session.PublishMetric("accuracy", 0.75);

	ASSERT_TRUE(session.TryBeginStep());
	auto loss = OaFnMatrix::Full(OaMatrixShape{1}, 0.5F);
	training.Next(loss);
	auto snapshot = session.LatestSnapshot();
	ASSERT_TRUE(snapshot.HasValue());
	EXPECT_EQ(snapshot->Step, 1);
	EXPECT_FLOAT_EQ(snapshot->Loss, 0.5F);
	ASSERT_EQ(snapshot->Metrics.Size(), 1U);
	EXPECT_EQ(snapshot->Metrics[0].Name, "accuracy");
	EXPECT_DOUBLE_EQ(snapshot->Metrics[0].Value, 0.75);
	viewer.Update(16.0F);
	const auto viewedSnapshot = viewer.LatestSnapshot();
	ASSERT_TRUE(viewedSnapshot.HasValue());
	EXPECT_EQ(viewedSnapshot->Step, 1);
	EXPECT_EQ(viewer.MetricSeriesCount(), 4U);
	EXPECT_EQ(viewer.MetricSampleCount("loss"), 1U);
	EXPECT_EQ(viewer.MetricSampleCount("accuracy"), 1U);

	ASSERT_TRUE(session.Stop().IsOk());
	EXPECT_FALSE(session.TryBeginStep());
	EXPECT_EQ(session.State(), OaTrainingState::Stopping);
	ASSERT_TRUE(training.Finish().IsOk());
	EXPECT_EQ(session.State(), OaTrainingState::Completed);
	ASSERT_TRUE(session.LatestSnapshot().HasValue());
	EXPECT_EQ(session.LatestSnapshot()->State, OaTrainingState::Completed);
}

TEST(TrainingSession, ViewerPromotesOnlyLatestCompletedPreview) {
	OaOptimizerNoOp optimizer;
	OaItTraining training(optimizer, OaItTrainingConfig{.TotalSteps = 1});
	OaTrainingSession session(training);
	OaTrainingViewerSource viewer(session);
	auto first = OaMakeSharedPtr<OaTexture>();
	first->DeviceBuf.Buffer = reinterpret_cast<void*>(1);
	first->Width = 8;
	first->Height = 8;
	auto latest = OaMakeSharedPtr<OaTexture>();
	latest->DeviceBuf.Buffer = reinterpret_cast<void*>(2);
	latest->Width = 16;
	latest->Height = 16;

	ASSERT_TRUE(viewer.PublishPreview({
		.Texture = first,
		.Label = "first",
		.Step = 1,
	}).IsOk());
	ASSERT_TRUE(viewer.PublishPreview({
		.Texture = latest,
		.Label = "latest",
		.Step = 2,
	}).IsOk());
	EXPECT_FALSE(viewer.LatestPreview().HasValue());
	viewer.Update(16.0F);
	const auto preview = viewer.LatestPreview();
	ASSERT_TRUE(preview.HasValue());
	EXPECT_EQ(preview->Label, "latest");
	EXPECT_EQ(preview->Step, 2);
	EXPECT_EQ(preview->Texture->Width, 16);
}

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

TEST(TrainingProgram, DroppedReplayRetiresWithoutDestructorWait) {
	if (not OaVkTestEngineOk()) GTEST_SKIP();

	auto& ctx = OaContext::GetDefault();
	OaWeakPtr<OaVkBuffer> inputStorage;
	{
		OaMatrix input = OaFnMatrix::Full(OaMatrixShape{4096}, 1.0F);
		OaMatrix increment = OaFnMatrix::Full(OaMatrixShape{4096}, 0.25F);
		ASSERT_TRUE(ctx.Execute().IsOk());
		ASSERT_TRUE(ctx.Sync().IsOk());

		OaFnMatrix::AddInPlace(input, increment);
		auto program = OaMakeUniquePtr<OaTrainingProgram>();
		ASSERT_TRUE(program->Capture(ctx).IsOk());
		inputStorage = OaWeakPtr<OaVkBuffer>(input.VkBuf_);
		ASSERT_TRUE(program->Replay().IsOk());
		// Program destruction must not host-wait. Its compiled graph and input
		// owners transfer to OaEngine retirement until exact replay completion.
	}
	EXPECT_FALSE(inputStorage.Expired());

	// The first submission is ordered after the retired replay and its explicit
	// wait proves completion. The second submission gives the engine a collection
	// boundary at which releasing the retired graph is safe.
	{
		auto pulse = OaFnMatrix::Full(OaMatrixShape{1}, 1.0F);
		(void)pulse;
		ASSERT_TRUE(ctx.Execute().IsOk());
		ASSERT_TRUE(ctx.Sync().IsOk());
	}
	{
		auto pulse = OaFnMatrix::Full(OaMatrixShape{1}, 2.0F);
		(void)pulse;
		ASSERT_TRUE(ctx.Execute().IsOk());
		ASSERT_TRUE(ctx.Sync().IsOk());
	}
	EXPECT_TRUE(inputStorage.Expired());
}

TEST(TrainingProgram, CaptureOwnsSemanticAndExecutableRecordingTogether) {
	if (not OaVkTestEngineOk()) GTEST_SKIP();

	auto& ctx = OaContext::GetDefault();
	ctx.Clear();
	const auto a = OaFnMatrix::Empty({2, 3});
	const auto b = OaFnMatrix::Empty({2, 3});
	ctx.Clear();
	const auto sum = OaFnMatrix::Add(a, b);
	ASSERT_EQ(ctx.NodeCount(), 1U);
	ASSERT_EQ(ctx.SemanticGraph()->OperationCount(), 1U);

	OaTrainingProgram program;
	const auto unrelated = OaFnMatrix::Empty({2, 3});
	const OaMatrix* invalidOutputs[] = {&unrelated};
	OaTrainingProgramOptions invalidOptions;
	invalidOptions.ObservedOutputs = {invalidOutputs, 1U};
	const auto invalidStatus = program.Capture(ctx, invalidOptions);
	EXPECT_FALSE(invalidStatus.IsOk());
	EXPECT_EQ(invalidStatus.GetCode(), OaStatusCode::InvalidArgument);
	EXPECT_EQ(ctx.NodeCount(), 1U);
	EXPECT_EQ(ctx.SemanticGraph()->OperationCount(), 1U);

	const OaMatrix* observedOutputs[] = {&sum};
	OaTrainingProgramOptions options;
	options.ObservedOutputs = {observedOutputs, 1U};
	ASSERT_TRUE(program.Capture(ctx, options).IsOk());
	EXPECT_EQ(program.NodeCount(), 1U);
	EXPECT_EQ(program.SemanticOperationCount(), 1U);
	EXPECT_EQ(program.CapturedResourceCount(), 3U);
	ASSERT_TRUE(program.SemanticGraph().Validate().IsOk());
	EXPECT_EQ(program.SemanticGraph().Operations()[0].Name, "Add");
	const auto bindings = program.SemanticStorageBindings();
	ASSERT_EQ(bindings.Size(), 3U);
	for (OaU32 index = 0; index < bindings.Size(); ++index) {
		EXPECT_EQ(bindings[index].Value, index);
		EXPECT_EQ(bindings[index].Resource, index);
		EXPECT_FALSE(bindings[index].StableReplayInput);
	}
	EXPECT_TRUE(bindings[0].SemanticExternal);
	EXPECT_TRUE(bindings[1].SemanticExternal);
	EXPECT_FALSE(bindings[2].SemanticExternal);
	EXPECT_FALSE(bindings[0].ObservedOutput);
	EXPECT_FALSE(bindings[1].ObservedOutput);
	EXPECT_TRUE(bindings[2].ObservedOutput);
	const auto resources = program.CapturedResources();
	ASSERT_EQ(resources.Size(), 3U);
	EXPECT_TRUE(resources[0].SemanticExternal);
	EXPECT_TRUE(resources[1].SemanticExternal);
	EXPECT_TRUE(resources[2].ObservedOutput);
	EXPECT_TRUE(resources[2].IsExternallyLive());
	EXPECT_EQ(program.AliasCandidateCount(), 0U);
	EXPECT_EQ(program.PlannedAliasGroupCount(), 0U);
	EXPECT_EQ(program.PotentialAliasSavings(), 0U);
	EXPECT_EQ(ctx.NodeCount(), 0U);
	EXPECT_EQ(ctx.SemanticGraph()->OperationCount(), 0U);
	EXPECT_EQ(ctx.SemanticGraph()->ValueCount(), 0U);

	const auto semanticReport =
		program.SemanticDebugReportJson("captured-training-step").StdStr();
	const auto executionReport =
		program.DebugReportJson("captured-training-step").StdStr();
	const auto compilationReport =
		program.CompilationDebugReportJson("captured-training-step").StdStr();
	const auto compilationReportAgain =
		program.CompilationDebugReportJson("captured-training-step").StdStr();
	EXPECT_NE(semanticReport.find("\"schema\": \"oa.semantic_graph.v2\""),
		std::string::npos);
	EXPECT_NE(semanticReport.find("\"name\": \"Add\""), std::string::npos);
	EXPECT_NE(executionReport.find("\"semantic_operations\": [0]"),
		std::string::npos);
	EXPECT_EQ(compilationReport, compilationReportAgain);
	EXPECT_NE(compilationReport.find(
		"\"schema\": \"oa.training_compilation.v2\""), std::string::npos);
	EXPECT_NE(compilationReport.find(
		"\"stage\": \"decomposition\", \"state\": \"analyzed\""),
		std::string::npos);
	EXPECT_NE(compilationReport.find(
		"\"stage\": \"fusion\", \"state\": \"analyzed\""),
		std::string::npos);
	EXPECT_NE(compilationReport.find(
		"\"lowering_analysis\": {"), std::string::npos);
	EXPECT_NE(compilationReport.find(
		"\"direct_operation_count\": 1"), std::string::npos);
	EXPECT_NE(compilationReport.find(
		"\"decomposed_operation_count\": 0"), std::string::npos);
	EXPECT_NE(compilationReport.find(
		"\"fused_operation_count\": 0"), std::string::npos);
	EXPECT_NE(compilationReport.find(
		"\"fused_node_count\": 0"), std::string::npos);
	EXPECT_NE(compilationReport.find(
		"\"stage\": \"kernel_selection\", \"state\": \"inherited\""),
		std::string::npos);
	EXPECT_NE(compilationReport.find(
		"\"stage\": \"memory_planning\", \"state\": \"analyzed\""),
		std::string::npos);
	EXPECT_NE(compilationReport.find(
		"\"materialized\": false"), std::string::npos);
	EXPECT_NE(compilationReport.find(
		"\"stage\": \"command_recording\", \"state\": \"applied\""),
		std::string::npos);
	const auto stages = program.CompilationStages();
	ASSERT_EQ(stages.Size(), 11U);
	EXPECT_EQ(stages[0].Stage,
		OaTrainingCompilationStage::SemanticValidation);
	EXPECT_EQ(stages[1].Stage, OaTrainingCompilationStage::ReplaySafety);
	EXPECT_EQ(stages[2].Stage, OaTrainingCompilationStage::Decomposition);
	EXPECT_EQ(stages[2].State, OaTrainingCompilationState::Analyzed);
	EXPECT_EQ(stages[3].State, OaTrainingCompilationState::Analyzed);
	EXPECT_EQ(program.LoweringAnalysis().OperationCount(), 1U);
	EXPECT_EQ(program.LoweringAnalysis().DirectOperationCount(), 1U);
	EXPECT_EQ(program.LoweringAnalysis().DecomposedOperationCount(), 0U);
	EXPECT_EQ(program.LoweringAnalysis().FusedOperationCount(), 0U);
	EXPECT_EQ(program.LoweringAnalysis().FusedNodeCount(), 0U);
	EXPECT_EQ(stages[8].Stage, OaTrainingCompilationStage::MemoryPlanning);
	EXPECT_EQ(stages[8].State, OaTrainingCompilationState::Analyzed);
	EXPECT_EQ(stages[9].Stage,
		OaTrainingCompilationStage::SynchronizationPlanning);
	EXPECT_EQ(stages[10].Stage,
		OaTrainingCompilationStage::CommandRecording);

	// A new recording starts a new semantic SSA namespace instead of appending
	// to the graph consumed by the captured program.
	const auto next = OaFnMatrix::Add(a, b);
	(void)next;
	ASSERT_EQ(ctx.SemanticGraph()->OperationCount(), 1U);
	EXPECT_EQ(ctx.SemanticGraph()->Operations()[0].Id, 0U);
	ctx.Clear();
	ASSERT_TRUE(program.Reset().IsOk());
}

TEST(TrainingProgram, RejectedCapturePreservesBothSourceGraphs) {
	if (not OaVkTestEngineOk()) GTEST_SKIP();

	auto& ctx = OaContext::GetDefault();
	ctx.Clear();
	const auto a = OaFnMatrix::Empty({4});
	const auto b = OaFnMatrix::Empty({4});
	const auto m = OaFnMatrix::Empty({4});
	const auto v = OaFnMatrix::Empty({4});
	ctx.Clear();
	const auto sum = OaFnMatrix::Add(a, b);

	struct AdamPush {
		OaU32 Count;
		OaF32 Lr;
		OaF32 Beta1;
		OaF32 Beta2;
		OaF32 Eps;
		OaU32 Step;
	} push{4, 1e-3F, 0.9F, 0.999F, 1e-8F, 1};
	OaBufferAccess access[] = {
		OaBufferAccess::ReadWrite,
		OaBufferAccess::Read,
		OaBufferAccess::ReadWrite,
		OaBufferAccess::ReadWrite,
	};
	ctx.Add("Adam", {&sum, &b, &m, &v},
		OaSpan<OaBufferAccess>(access, 4), &push, sizeof(push), 1);
	ASSERT_EQ(ctx.NodeCount(), 2U);
	ASSERT_EQ(ctx.SemanticGraph()->OperationCount(), 1U);

	OaTrainingProgram program;
	const auto status = program.Capture(ctx);
	ASSERT_FALSE(status.IsOk());
	EXPECT_EQ(status.GetCode(), OaStatusCode::FailedPrecondition);
	EXPECT_FALSE(program.IsCaptured());
	EXPECT_EQ(program.NodeCount(), 0U);
	EXPECT_EQ(program.SemanticOperationCount(), 0U);
	const auto failedStages = program.CompilationStages();
	ASSERT_EQ(failedStages.Size(), 2U);
	EXPECT_EQ(failedStages[0].Stage,
		OaTrainingCompilationStage::SemanticValidation);
	EXPECT_EQ(failedStages[0].State, OaTrainingCompilationState::Applied);
	EXPECT_EQ(failedStages[1].Stage,
		OaTrainingCompilationStage::ReplaySafety);
	EXPECT_EQ(failedStages[1].State, OaTrainingCompilationState::Failed);
	const auto failedReport =
		program.CompilationDebugReportJson("rejected-training-step").StdStr();
	EXPECT_NE(failedReport.find("\"captured\": false"), std::string::npos);
	EXPECT_NE(failedReport.find(
		"\"stage\": \"replay_safety\", \"state\": \"failed\""),
		std::string::npos);
	EXPECT_EQ(ctx.NodeCount(), 2U);
	EXPECT_EQ(ctx.SemanticGraph()->OperationCount(), 1U);
	EXPECT_EQ(ctx.SemanticGraph()->Operations()[0].Name, "Add");
	ctx.Clear();
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

TEST(TrainingProgram, TwoPhaseStepClassifiesStableReplayInputs) {
	if (not OaVkTestEngineOk()) GTEST_SKIP();

	auto& ctx = OaContext::GetDefault();
	OaOptimizerNoOp optimizer;
	OaTrainingProgram program;
	OaItTraining iter(optimizer, OaItTrainingConfig{
		.TotalSteps = 3,
		.Program = &program,
	});
	OaMatrix input;
	OaMatrix loss;
	while (not iter.IsDone()) {
		iter.Step(
			[&] {
				input = OaFnMatrix::Empty(
					OaMatrixShape{4}, OaScalarType::Float32,
					OaMemoryPlacement::HostUpload);
				auto* values = input.DataAs<OaF32>();
				ASSERT_NE(values, nullptr);
				for (OaU32 index = 0; index < 4U; ++index) values[index] = 1.0F;
			},
			[&] {
				OaMatrix temporary1 = OaFnMatrix::Add(input, input);
				OaMatrix temporary2 = OaFnMatrix::Add(temporary1, input);
				OaMatrix temporary3 = OaFnMatrix::Add(temporary2, input);
				// The schema-owned reduction contributes the final semantic value;
				// observing it keeps the same physical output externally live.
				loss = OaFnMatrix::Sum(temporary3);
				iter.RecordLoss(loss);
			});
	}
	ASSERT_TRUE(iter.Finish().IsOk());

	ASSERT_TRUE(program.IsCaptured());
	const auto bindings = program.SemanticStorageBindings();
	ASSERT_EQ(bindings.Size(), 5U);
	EXPECT_TRUE(bindings[0].SemanticExternal);
	EXPECT_TRUE(bindings[0].StableReplayInput);
	for (OaU32 index = 1; index + 1U < bindings.Size(); ++index) {
		EXPECT_FALSE(bindings[index].SemanticExternal);
		EXPECT_FALSE(bindings[index].StableReplayInput);
		EXPECT_FALSE(bindings[index].ObservedOutput);
	}
	EXPECT_FALSE(bindings[4].SemanticExternal);
	EXPECT_FALSE(bindings[4].StableReplayInput);
	EXPECT_TRUE(bindings[4].ObservedOutput);
	const auto resources = program.CapturedResources();
	ASSERT_EQ(resources.Size(), 5U);
	EXPECT_TRUE(resources[0].StableReplayInput);
	EXPECT_TRUE(resources[1].AliasCandidate);
	EXPECT_TRUE(resources[2].AliasCandidate);
	EXPECT_TRUE(resources[3].AliasCandidate);
	EXPECT_TRUE(resources[1].AliasMaterialized);
	EXPECT_FALSE(resources[2].AliasMaterialized);
	EXPECT_TRUE(resources[3].AliasMaterialized);
	EXPECT_TRUE(resources[4].ObservedOutput);
	EXPECT_TRUE(resources[0].IsExternallyLive());
	EXPECT_FALSE(resources[1].IsExternallyLive());
	EXPECT_TRUE(resources[4].IsExternallyLive());
	EXPECT_EQ(program.AliasCandidateCount(), 3U);
	EXPECT_EQ(program.PlannedAliasGroupCount(), 1U);
	EXPECT_EQ(program.PotentialAliasSavings(), 4U * sizeof(OaF32));
	EXPECT_EQ(program.MaterializedAliasSavings(), 4U * sizeof(OaF32));
	EXPECT_EQ(program.Stats().AliasBarrierCount, 1U);
	EXPECT_FLOAT_EQ(loss.Item(), 16.0F);
	const auto report = program.CompilationDebugReportJson(
		"captured-memory-analysis").StdStr();
	EXPECT_NE(report.find("\"candidate_count\": 3"), std::string::npos);
	EXPECT_NE(report.find("\"materialization_eligible_count\": 3"),
		std::string::npos);
	EXPECT_NE(report.find("\"alias_group_count\": 1"), std::string::npos);
	EXPECT_NE(report.find("\"potential_savings_bytes\": 16"),
		std::string::npos);
	EXPECT_NE(report.find("\"materialized_savings_bytes\": 16"),
		std::string::npos);
	EXPECT_NE(report.find("\"materialized\": true"), std::string::npos);
	const auto stages = program.CompilationStages();
	ASSERT_EQ(stages.Size(), 11U);
	EXPECT_EQ(stages[8].Stage, OaTrainingCompilationStage::MemoryPlanning);
	EXPECT_EQ(stages[8].State, OaTrainingCompilationState::Applied);
	EXPECT_EQ(OaExecutionMemory::StableExternalResourceCount(ctx), 1U);
	EXPECT_EQ(OaExecutionMemory::StableTransientResourceCount(ctx), 0U);
}

TEST(TrainingProgram, RetainedTransientMatrixPreventsAliasMaterialization) {
	if (not OaVkTestEngineOk()) GTEST_SKIP();

	OaOptimizerNoOp optimizer;
	OaTrainingProgram program;
	OaItTraining iter(optimizer, OaItTrainingConfig{
		.TotalSteps = 3,
		.Program = &program,
	});
	OaMatrix input;
	OaMatrix retained1;
	OaMatrix retained2;
	OaMatrix retained3;
	OaMatrix loss;
	while (not iter.IsDone()) {
		iter.Step(
			[&] {
				input = OaFnMatrix::Empty(
					OaMatrixShape{4}, OaScalarType::Float32,
					OaMemoryPlacement::HostUpload);
				auto* values = input.DataAs<OaF32>();
				ASSERT_NE(values, nullptr);
				for (OaU32 index = 0; index < 4U; ++index) values[index] = 1.0F;
			},
			[&] {
				retained1 = OaFnMatrix::Add(input, input);
				retained2 = OaFnMatrix::Add(retained1, input);
				retained3 = OaFnMatrix::Add(retained2, input);
				loss = OaFnMatrix::Sum(retained3);
				iter.RecordLoss(loss);
			});
	}
	ASSERT_TRUE(iter.Finish().IsOk());

	EXPECT_TRUE(program.IsCaptured());
	EXPECT_EQ(program.AliasCandidateCount(), 3U);
	EXPECT_EQ(program.PotentialAliasSavings(), 4U * sizeof(OaF32));
	EXPECT_EQ(program.MaterializedAliasSavings(), 0U);
	for (OaU32 index = 1; index <= 3U; ++index) {
		EXPECT_FALSE(program.CapturedResources()[index].AliasMaterialized);
	}
	const auto report = program.CompilationDebugReportJson(
		"retained-transient").StdStr();
	EXPECT_NE(report.find("\"materialization_eligible_count\": 0"),
		std::string::npos);
	EXPECT_NE(report.find("\"fallback_reason\": \"\""), std::string::npos);
	const auto stages = program.CompilationStages();
	ASSERT_EQ(stages.Size(), 11U);
	EXPECT_EQ(stages[8].Stage, OaTrainingCompilationStage::MemoryPlanning);
	EXPECT_EQ(stages[8].State, OaTrainingCompilationState::Analyzed);
	EXPECT_FLOAT_EQ(loss.Item(), 16.0F);
}

TEST(TrainingProgram, SinglePhaseStepConservativelyRetainsStableResources) {
	if (not OaVkTestEngineOk()) GTEST_SKIP();

	auto& ctx = OaContext::GetDefault();
	OaOptimizerNoOp optimizer;
	OaItTraining iter(optimizer, OaItTrainingConfig{.TotalSteps = 2});
	OaMatrix first;
	OaMatrix second;
	while (not iter.IsDone()) {
		iter.Step([&] {
			first = OaFnMatrix::Empty(OaMatrixShape{4});
			second = OaFnMatrix::Full(OaMatrixShape{4}, 1.0F);
			iter.RecordLoss(second);
		});
	}
	ASSERT_TRUE(iter.Finish().IsOk());

	EXPECT_EQ(OaExecutionMemory::StableExternalResourceCount(ctx), 2U);
	EXPECT_EQ(OaExecutionMemory::StableTransientResourceCount(ctx), 0U);
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
