#include "../../oaTest.h"

#include <oa/ml/callbacks.h>
#include <oa/ml/fnOptim.h>
#include <oa/ml/mcpTraining.h>
#include <oa/ml/optim.h>
#include <oa/ml/trainingProgram.h>
#include <oa/ml/trainingSession.h>
#include <oa/ui/image.h>
#include <oa/ui/trainingViewer.h>

#include <oa/core/matrixAccess.h>
#include <oa/core/op.h>
#include <oa/core/thread.h>
#include <oa/runtime/executionSession.h>

#include <cmath>
#include <cstring>
#include <type_traits>

static_assert(std::is_constructible_v<oa::ItTraining, oa::Engine &, oa::Optimizer &,
                                      oa::ItTrainingConfig>);
static_assert(
    !std::is_constructible_v<oa::ItTraining, oa::Optimizer &, oa::ItTrainingConfig>);

namespace {

class SampleRecorder final : public oa::CbTraining {
public:
  void onStepEnd(oa::ItTraining &inIter) override {
    if (inIter.hasLossSample()) {
      lossSteps.pushBack(inIter.lastLossStep());
      lossValues.pushBack(inIter.lastLoss());
    }
  }

  oa::Vec<oa::I64> lossSteps;
  oa::Vec<oa::F32> lossValues;
};

class FailingTrainingCallback final : public oa::CbTraining {
public:
  void onStepEnd(oa::ItTraining &) override {
    status_ = oa::Status::error(oa::StatusCode::DataLoss,
                              "injected checkpoint callback failure");
  }

  [[nodiscard]] oa::Status getStatus() const override { return status_; }

private:
  oa::Status status_ = oa::Status::ok();
};

oa::String mcpToolRequest(oa::I64 inId, oa::StringView inName,
                        oa::StringView inArguments) {
  oa::String request = R"({"jsonrpc":"2.0","id":)";
  const std::string idText = std::to_string(inId);
  request += oa::StringView(idText.data(), idText.size());
  request +=
      R"(,"method":"tools/call","params":{"_meta":{"io.modelcontextprotocol/protocolVersion":"2026-07-28","io.modelcontextprotocol/clientCapabilities":{}},"name":")";
  request += inName;
  request += R"(","arguments":)";
  request += inArguments;
  request += "}}";
  return request;
}

std::string handleMcp(oa::McpServer &inServer, oa::StringView inRequest) {
  auto result = inServer.handleMessage(inRequest);
  EXPECT_TRUE(result.isOk()) << result.getStatus().toString();
  return result.isOk()
      ? std::string(result->data(), result->size())
      : std::string{};
}

} // namespace

TEST(TrainingCallbacks, FailureStopsAndPropagatesThroughFinish) {
  if (not vkTestEngineOk())
    GTEST_SKIP();
  oa::OptimizerNoOp optimizer;
  FailingTrainingCallback callback;
  oa::ItTraining training(testEngine(), optimizer,
                        oa::ItTrainingConfig{
                            .totalSteps = 2,
                            .enableGpuTiming = false,
                            .callbacks = {&callback},
                        });
  ASSERT_FALSE(training.isDone());
  training.next();
  EXPECT_TRUE(training.stopRequested());
  EXPECT_EQ(training.lastStatus().getCode(), oa::StatusCode::DataLoss);
  const auto status = training.finish();
  EXPECT_EQ(status.getCode(), oa::StatusCode::DataLoss);
}

TEST(TrainingSession, CommandsAreTypedRevisionedAndSafePointApplied) {
  if (not vkTestEngineOk())
    GTEST_SKIP();
  oa::OptimizerNoOp optimizer;
  optimizer.setLr(1.0e-3F);
  oa::ItTraining training(testEngine(), optimizer,
                        oa::ItTrainingConfig{.totalSteps = 2});
  oa::I32 checkpointCount = 0;
  oa::I32 evaluationCount = 0;
  oa::TrainingSession session(training, oa::TrainingSessionConfig{
                                          .handlers = {
                                              .checkpoint =
                                                  [&] {
                                                    ++checkpointCount;
                                                    return oa::Status::ok();
                                                  },
                                              .evaluate =
                                                  [&] {
                                                    ++evaluationCount;
                                                    return oa::Status::ok();
                                                  },
                                          },
                                      });

  EXPECT_EQ(session.revision(), 1U);
  ASSERT_TRUE(session.pause().isOk());
  EXPECT_FALSE(session.tryBeginStep());
  EXPECT_EQ(session.state(), oa::TrainingState::Paused);
  const oa::U64 pausedRevision = session.revision();

  ASSERT_TRUE(session
                  .setParameter("learning_rate",
                                oa::TrainingValue::fromFloat(2.5e-4),
                                pausedRevision)
                  .isOk());
  ASSERT_TRUE(session.checkpoint().isOk());
  ASSERT_TRUE(session.evaluate().isOk());
  ASSERT_TRUE(session.resume().isOk());
  ASSERT_TRUE(session.poll().isOk());
  EXPECT_EQ(session.state(), oa::TrainingState::Running);
  EXPECT_FLOAT_EQ(optimizer.getLr(), 2.5e-4F);
  EXPECT_EQ(checkpointCount, 1);
  EXPECT_EQ(evaluationCount, 1);

  // The old revision was accepted by Enqueue but rejected atomically at the
  // next safe point after preceding commands advanced the session revision.
  ASSERT_TRUE(session
                  .setParameter("learning_rate",
                                oa::TrainingValue::fromFloat(9.0e-4),
                                pausedRevision)
                  .isOk());
  ASSERT_TRUE(session.poll().isOk());
  EXPECT_FLOAT_EQ(optimizer.getLr(), 2.5e-4F);

  const auto observed = session.resultsAfter(0);
  ASSERT_EQ(observed.size(), 6U);
  EXPECT_EQ(session.resultsAfter(0).size(), 6U);
  EXPECT_TRUE(session.resultsAfter(observed.back().sequence).empty());

  auto results = session.takeResults();
  ASSERT_EQ(results.size(), 6U);
  EXPECT_TRUE(session.takeResults().empty());
  EXPECT_EQ(session.resultsAfter(0).size(), 6U);
  EXPECT_EQ(results[0].disposition, oa::TrainingCommandDisposition::Applied);
  EXPECT_EQ(results[5].disposition, oa::TrainingCommandDisposition::Rejected);
  EXPECT_EQ(results[5].status.getCode(), oa::StatusCode::Aborted);
}

TEST(TrainingSession, BlockingSafePointWakesForResume) {
  if (not vkTestEngineOk())
    GTEST_SKIP();
  oa::OptimizerNoOp optimizer;
  oa::ItTraining training(testEngine(), optimizer,
                        oa::ItTrainingConfig{.totalSteps = 1});
  oa::TrainingSession session(training);

  ASSERT_TRUE(session.pause().isOk());
  ASSERT_TRUE(session.poll().isOk());
  ASSERT_EQ(session.state(), oa::TrainingState::Paused);

  oa::Atomic<oa::Bool> entered{false};
  oa::Atomic<oa::Bool> returned{false};
  oa::Atomic<oa::Bool> mayBegin{false};
  auto waiterResult = oa::Thread::create([&] {
    entered.store(true, oa::MemoryOrder::Release);
    mayBegin.store(session.waitBeginStep(), oa::MemoryOrder::Release);
    returned.store(true, oa::MemoryOrder::Release);
  });
  ASSERT_TRUE(waiterResult.isOk())
      << waiterResult.getStatus().toString();
  oa::Thread waiter = oa::move(*waiterResult);

  while (not entered.load(oa::MemoryOrder::Acquire))
    oa::Thread::yield();
  oa::Thread::sleepFor(oa::Duration::fromMilliseconds(5));
  EXPECT_FALSE(returned.load(oa::MemoryOrder::Acquire));

  ASSERT_TRUE(session.resume().isOk());
  ASSERT_TRUE(waiter.join().isOk());
  EXPECT_TRUE(returned.load(oa::MemoryOrder::Acquire));
  EXPECT_TRUE(mayBegin.load(oa::MemoryOrder::Acquire));
  EXPECT_EQ(session.state(), oa::TrainingState::Running);
}

TEST(TrainingSession, IteratorPublishesBoundedSnapshotsAndStop) {
  if (not vkTestEngineOk())
    GTEST_SKIP();
  oa::OptimizerNoOp optimizer;
  oa::ItTraining training(testEngine(), optimizer,
                        oa::ItTrainingConfig{.totalSteps = 3});
  oa::TrainingSession session(training, oa::TrainingSessionConfig{
                                          .snapshotCapacity = 2,
                                      });
  oa::TrainingViewerSource viewer(session, {
                                             .historyCapacity = 2,
                                             .maxMetricPlots = 4,
                                         });
  session.publishMetric("accuracy", 0.75);

  ASSERT_TRUE(session.tryBeginStep());
  auto loss = oa::FnMatrix::full(oa::MatrixShape{1}, 0.5F);
  training.next(loss);
  auto snapshot = session.latestSnapshot();
  ASSERT_TRUE(snapshot.hasValue());
  EXPECT_EQ(snapshot->step, 1);
  EXPECT_FLOAT_EQ(snapshot->loss, 0.5F);
  ASSERT_EQ(snapshot->metrics.size(), 1U);
  EXPECT_EQ(snapshot->metrics[0].name, "accuracy");
  EXPECT_DOUBLE_EQ(snapshot->metrics[0].value, 0.75);
  ASSERT_TRUE(viewer.update(16.0F).isOk());
  const auto viewedSnapshot = viewer.latestSnapshot();
  ASSERT_TRUE(viewedSnapshot.hasValue());
  EXPECT_EQ(viewedSnapshot->step, 1);
  EXPECT_EQ(viewer.metricSeriesCount(), 4U);
  EXPECT_EQ(viewer.metricSampleCount("loss"), 1U);
  EXPECT_EQ(viewer.metricSampleCount("accuracy"), 1U);

  ASSERT_TRUE(session.stop().isOk());
  EXPECT_FALSE(session.tryBeginStep());
  EXPECT_EQ(session.state(), oa::TrainingState::Stopping);
  ASSERT_TRUE(training.finish().isOk());
  EXPECT_EQ(session.state(), oa::TrainingState::Completed);
  ASSERT_TRUE(session.latestSnapshot().hasValue());
  EXPECT_EQ(session.latestSnapshot()->state, oa::TrainingState::Completed);
}

TEST(TrainingSession, McpAdapterUsesTheExistingSafePointAuthority) {
  if (not vkTestEngineOk())
    GTEST_SKIP();
  oa::OptimizerNoOp optimizer;
  optimizer.setLr(1.0e-3F);
  oa::ItTraining training(testEngine(), optimizer,
                        oa::ItTrainingConfig{.totalSteps = 2});
  oa::TrainingSession session(training);
  oa::McpServer server;
  ASSERT_TRUE(oa::McpTraining::registerTools(server, session).isOk());

  const std::string list = handleMcp(
      server,
      R"({"jsonrpc":"2.0","id":1,"method":"tools/list","params":{"_meta":{"io.modelcontextprotocol/protocolVersion":"2026-07-28","io.modelcontextprotocol/clientCapabilities":{}}}})");
  EXPECT_NE(list.find(R"("name":"training_status")"), std::string::npos);
  EXPECT_NE(list.find(R"("name":"training_set_parameter")"), std::string::npos);
  EXPECT_EQ(list.find("training_stop"), std::string::npos);
  EXPECT_EQ(list.find("training_request_rebuild"), std::string::npos);
  EXPECT_EQ(list.find("training_preview"), std::string::npos);

  const std::string initial =
      handleMcp(server, mcpToolRequest(2, "training_status", "{}"));
  EXPECT_NE(initial.find(R"("revision":1,"state":"running")"),
            std::string::npos);

  const std::string pause = handleMcp(
      server, mcpToolRequest(3, "training_pause", R"({"expectedRevision":1})"));
  EXPECT_NE(pause.find(R"("sequence":1)"), std::string::npos);
  EXPECT_EQ(session.state(), oa::TrainingState::Running);
  ASSERT_TRUE(session.poll().isOk());
  EXPECT_EQ(session.state(), oa::TrainingState::Paused);
  EXPECT_EQ(session.revision(), 2U);
  const std::string paused =
      handleMcp(server, mcpToolRequest(4, "training_status", "{}"));
  EXPECT_NE(paused.find(R"("revision":2,"state":"paused")"), std::string::npos);

  const std::string set = handleMcp(
      server,
      mcpToolRequest(
          5, "training_set_parameter",
          R"({"name":"learning_rate","value":0.0005,"expectedRevision":2})"));
  EXPECT_NE(set.find(R"("sequence":2)"), std::string::npos);
  ASSERT_TRUE(session.poll().isOk());
  EXPECT_FLOAT_EQ(optimizer.getLr(), 5.0e-4F);

  const std::string audit = handleMcp(
      server, mcpToolRequest(6, "training_results", R"({"afterSequence":0})"));
  EXPECT_NE(audit.find(R"("sequence":1,"revision":2,"disposition":"applied")"),
            std::string::npos);
  EXPECT_NE(audit.find(R"("sequence":2,"revision":3,"disposition":"applied")"),
            std::string::npos);

  const std::string invalid = handleMcp(
      server,
      mcpToolRequest(7, "training_set_parameter",
                     R"({"name":"learning_rate","value":{"unsafe":true}})"));
  EXPECT_NE(invalid.find(R"("isError":true)"), std::string::npos);
  EXPECT_NE(
      invalid.find("value must be a boolean, integer, finite number or string"),
      std::string::npos);
}

TEST(TrainingSession, ViewerPromotesOnlyLatestCompletedPreview) {
  if (not vkTestEngineOk())
    GTEST_SKIP();
  auto *engine = testEnginePtr();
  ASSERT_NE(engine, nullptr);
  oa::OptimizerNoOp optimizer;
  oa::ItTraining training(testEngine(), optimizer,
                        oa::ItTrainingConfig{.totalSteps = 1});
  oa::TrainingSession session(training);
  oa::TrainingViewerSource viewer(session);
  oa::Vec<oa::U8> firstPixels(8U * 8U * 4U, 0U);
  auto firstResult = oa::FnTexture::fromPixels(
      *engine, oa::Span<const oa::U8>(firstPixels.data(), firstPixels.size()), 8,
      8);
  ASSERT_TRUE(firstResult.isOk()) << firstResult.getStatus().toString();
  auto first = oa::makeShared<oa::Texture>();
  *first = oa::move(*firstResult);
  oa::Vec<oa::U8> latestPixels(16U * 16U * 4U, 0U);
  auto latestResult = oa::FnTexture::fromPixels(
      *engine, oa::Span<const oa::U8>(latestPixels.data(), latestPixels.size()), 16,
      16);
  ASSERT_TRUE(latestResult.isOk()) << latestResult.getStatus().toString();
  auto latest = oa::makeShared<oa::Texture>();
  *latest = oa::move(*latestResult);

  ASSERT_TRUE(viewer
                  .publishPreview({
                      .texture = first,
                      .label = "first",
                      .step = 1,
                  })
                  .isOk());
  ASSERT_TRUE(viewer
                  .publishPreview({
                      .texture = latest,
                      .label = "latest",
                      .step = 2,
                  })
                  .isOk());
  EXPECT_FALSE(viewer.latestPreview().hasValue());
  ASSERT_TRUE(viewer.update(16.0F).isOk());
  const auto preview = viewer.latestPreview();
  ASSERT_TRUE(preview.hasValue());
  EXPECT_EQ(preview->label, "latest");
  EXPECT_EQ(preview->step, 2);
  EXPECT_EQ(preview->texture->width(), 16);
}

TEST(TrainingProgram, CapturedMutableStepProgressesAcrossReplays) {
  if (not vkTestEngineOk())
    GTEST_SKIP();

  auto &ctx = oa::ExecutionSession::getActive();
  oa::Matrix param = oa::FnMatrix::full(oa::MatrixShape{4}, 1.0F);
  oa::Matrix grad = oa::FnMatrix::full(oa::MatrixShape{4}, 0.25F);
  oa::Matrix momentum;
  ASSERT_TRUE(testSubmitAndWait(ctx).isOk());

  oa::FnOptim::sgdStep(param, momentum, grad, 0.1F, 0.0F, 0.0F);
  ASSERT_EQ(ctx.nodeCount(), 1U);

  oa::TrainingProgram program;
  ASSERT_TRUE(program.capture(ctx.engine()).isOk());
  EXPECT_TRUE(program.isCaptured());
  EXPECT_EQ(program.nodeCount(), 1U);
  EXPECT_EQ(ctx.nodeCount(), 0U);

  for (oa::U32 i = 0; i < 3; ++i)
    ASSERT_TRUE(program.replay().isOk());
  ASSERT_TRUE(program.wait().isOk());
  for (oa::I64 i = 0; i < param.numElements(); ++i) {
    EXPECT_NEAR(param.at(i), 0.925F, 1e-6F);
  }
  ASSERT_TRUE(program.reset().isOk());
}

TEST(TrainingProgram, DroppedReplayRetiresWithoutDestructorWait) {
  if (not vkTestEngineOk())
    GTEST_SKIP();

  auto &ctx = oa::ExecutionSession::getActive();
  oa::WeakPtr<oavk::Buffer> inputStorage;
  {
    oa::Matrix input = oa::FnMatrix::full(oa::MatrixShape{4096}, 1.0F);
    oa::Matrix increment = oa::FnMatrix::full(oa::MatrixShape{4096}, 0.25F);
    ASSERT_TRUE(testSubmitAndWait(ctx).isOk());

    oa::FnMatrix::addInPlace(input, increment);
    auto program = oa::makeUnique<oa::TrainingProgram>();
    ASSERT_TRUE(program->capture(ctx.engine()).isOk());
    inputStorage = oa::WeakPtr<oavk::Buffer>(oa::MatrixAccess::storageOwner(input));
    ASSERT_TRUE(program->replay().isOk());
    // program destruction must not host-wait. Its compiled graph and input
    // owners transfer to oa::Engine retirement until exact replay completion.
  }
  EXPECT_FALSE(inputStorage.expired());

  // The first submission is ordered after the retired replay and its explicit
  // wait proves completion. The second submission gives the engine a collection
  // boundary at which releasing the retired graph is safe.
  {
    auto pulse = oa::FnMatrix::full(oa::MatrixShape{1}, 1.0F);
    (void)pulse;
    ASSERT_TRUE(testSubmitAndWait(ctx).isOk());
  }
  {
    auto pulse = oa::FnMatrix::full(oa::MatrixShape{1}, 2.0F);
    (void)pulse;
    ASSERT_TRUE(testSubmitAndWait(ctx).isOk());
  }
  EXPECT_TRUE(inputStorage.expired());
}

TEST(TrainingProgram, CaptureOwnsSemanticAndExecutableRecordingTogether) {
  if (not vkTestEngineOk())
    GTEST_SKIP();

  auto &ctx = oa::ExecutionSession::getActive();
  ctx.clear();
  const auto a = oa::FnMatrix::empty({2, 3});
  const auto b = oa::FnMatrix::empty({2, 3});
  ctx.clear();
  const auto sum = oa::FnMatrix::add(a, b);
  ASSERT_EQ(ctx.nodeCount(), 1U);
  ASSERT_EQ(ctx.semanticGraph()->operationCount(), 1U);

  oa::TrainingProgram program;
  const auto unrelated = oa::FnMatrix::empty({2, 3});
  const oa::Matrix *invalidOutputs[] = {&unrelated};
  oa::TrainingProgramOptions invalidOptions;
  invalidOptions.observedOutputs = {invalidOutputs, 1U};
  const auto invalidStatus = program.capture(ctx.engine(), invalidOptions);
  EXPECT_FALSE(invalidStatus.isOk());
  EXPECT_EQ(invalidStatus.getCode(), oa::StatusCode::InvalidArgument);
  EXPECT_EQ(ctx.nodeCount(), 1U);
  EXPECT_EQ(ctx.semanticGraph()->operationCount(), 1U);

  const oa::Matrix *observedOutputs[] = {&sum};
  oa::TrainingProgramOptions options;
  options.observedOutputs = {observedOutputs, 1U};
  ASSERT_TRUE(program.capture(ctx.engine(), options).isOk());
  EXPECT_EQ(program.nodeCount(), 1U);
  EXPECT_EQ(program.semanticOpCount(), 1U);
  EXPECT_EQ(program.capturedResourceCount(), 3U);
  ASSERT_TRUE(program.semanticGraph().validate().isOk());
  EXPECT_EQ(program.semanticGraph().operations()[0].name,
            oa::detail::opRegistry::FnMatrix::add.name);
  const auto bindings = program.semanticStorageBindings();
  ASSERT_EQ(bindings.size(), 3U);
  for (oa::U32 index = 0; index < bindings.size(); ++index) {
    EXPECT_EQ(bindings[index].value, index);
    EXPECT_EQ(bindings[index].resource, index);
    EXPECT_FALSE(bindings[index].stableReplayInput);
  }
  EXPECT_TRUE(bindings[0].semanticExternal);
  EXPECT_TRUE(bindings[1].semanticExternal);
  EXPECT_FALSE(bindings[2].semanticExternal);
  EXPECT_FALSE(bindings[0].observedOutput);
  EXPECT_FALSE(bindings[1].observedOutput);
  EXPECT_TRUE(bindings[2].observedOutput);
  const auto resources = program.capturedResources();
  ASSERT_EQ(resources.size(), 3U);
  EXPECT_TRUE(resources[0].semanticExternal);
  EXPECT_TRUE(resources[1].semanticExternal);
  EXPECT_TRUE(resources[2].observedOutput);
  EXPECT_TRUE(resources[2].isExternallyLive());
  EXPECT_EQ(program.aliasCandidateCount(), 0U);
  EXPECT_EQ(program.plannedAliasGroupCount(), 0U);
  EXPECT_EQ(program.potentialAliasSavings(), 0U);
  EXPECT_EQ(ctx.nodeCount(), 0U);
  EXPECT_EQ(ctx.semanticGraph()->operationCount(), 0U);
  EXPECT_EQ(ctx.semanticGraph()->valueCount(), 0U);

  const auto semanticReport =
      testStdString(program.semanticDebugReportJson("captured-training-step"));
  const auto executionReport =
      testStdString(program.debugReportJson("captured-training-step"));
  const auto compilationReport =
      testStdString(program.compilationDebugReportJson("captured-training-step"));
  const auto compilationReportAgain =
      testStdString(program.compilationDebugReportJson("captured-training-step"));
  EXPECT_NE(semanticReport.find("\"schema\": \"oa.semantic_graph.v2\""),
            std::string::npos);
  EXPECT_NE(semanticReport.find("\"name\": \"oa::FnMatrix::add\""),
            std::string::npos);
  EXPECT_NE(executionReport.find("\"semantic_operations\": [0]"),
            std::string::npos);
  EXPECT_EQ(compilationReport, compilationReportAgain);
  EXPECT_NE(
      compilationReport.find("\"schema\": \"oa.training_compilation.v2\""),
      std::string::npos);
  EXPECT_NE(compilationReport.find(
                "\"stage\": \"decomposition\", \"state\": \"analyzed\""),
            std::string::npos);
  EXPECT_NE(
      compilationReport.find("\"stage\": \"fusion\", \"state\": \"analyzed\""),
      std::string::npos);
  EXPECT_NE(compilationReport.find("\"dnn_plan\": {"), std::string::npos);
  EXPECT_NE(compilationReport.find("\"source_operation_count\": 1"),
            std::string::npos);
  EXPECT_NE(compilationReport.find("\"captured_operation_count\": 1"),
            std::string::npos);
  EXPECT_NE(compilationReport.find("\"partition_count\": 1"),
            std::string::npos);
  EXPECT_NE(compilationReport.find("\"recognized_partition_count\": 0"),
            std::string::npos);
  EXPECT_NE(compilationReport.find("\"lowering_analysis\": {"),
            std::string::npos);
  EXPECT_NE(compilationReport.find("\"direct_operation_count\": 1"),
            std::string::npos);
  EXPECT_NE(compilationReport.find("\"decomposed_operation_count\": 0"),
            std::string::npos);
  EXPECT_NE(compilationReport.find("\"fused_operation_count\": 0"),
            std::string::npos);
  EXPECT_NE(compilationReport.find("\"fused_node_count\": 0"),
            std::string::npos);
  EXPECT_NE(compilationReport.find(
                "\"stage\": \"kernel_selection\", \"state\": \"inherited\""),
            std::string::npos);
  EXPECT_NE(compilationReport.find(
                "\"stage\": \"memory_planning\", \"state\": \"analyzed\""),
            std::string::npos);
  EXPECT_NE(compilationReport.find("\"materialized\": false"),
            std::string::npos);
  EXPECT_NE(compilationReport.find(
                "\"stage\": \"command_recording\", \"state\": \"applied\""),
            std::string::npos);
  const auto stages = program.compilationStages();
  ASSERT_EQ(stages.size(), 11U);
  EXPECT_EQ(stages[0].stage, oa::TrainingCompilationStage::SemanticValidation);
  EXPECT_EQ(stages[1].stage, oa::TrainingCompilationStage::ReplaySafety);
  EXPECT_EQ(stages[2].stage, oa::TrainingCompilationStage::Decomposition);
  EXPECT_EQ(stages[2].state, oa::TrainingCompilationState::Analyzed);
  EXPECT_EQ(stages[3].state, oa::TrainingCompilationState::Analyzed);
  EXPECT_EQ(program.loweringAnalysis().operationCount(), 1U);
  EXPECT_EQ(program.loweringAnalysis().directOpCount(), 1U);
  EXPECT_EQ(program.loweringAnalysis().decomposedOpCount(), 0U);
  EXPECT_EQ(program.loweringAnalysis().fusedOpCount(), 0U);
  EXPECT_EQ(program.loweringAnalysis().fusedNodeCount(), 0U);
  EXPECT_EQ(stages[8].stage, oa::TrainingCompilationStage::MemoryPlanning);
  EXPECT_EQ(stages[8].state, oa::TrainingCompilationState::Analyzed);
  EXPECT_EQ(stages[9].stage,
            oa::TrainingCompilationStage::SynchronizationPlanning);
  EXPECT_EQ(stages[10].stage, oa::TrainingCompilationStage::CommandRecording);

  // A new recording starts a new semantic SSA namespace instead of appending
  // to the graph consumed by the captured program.
  const auto next = oa::FnMatrix::add(a, b);
  (void)next;
  ASSERT_EQ(ctx.semanticGraph()->operationCount(), 1U);
  EXPECT_EQ(ctx.semanticGraph()->operations()[0].id, 0U);
  ctx.clear();
  ASSERT_TRUE(program.reset().isOk());
}

TEST(TrainingProgram, AutomaticallyCapturesSemanticDnnPartitions) {
  if (not vkTestEngineOk())
    GTEST_SKIP();

  auto &ctx = oa::ExecutionSession::getActive();
  ctx.clear();
  const auto input = oa::FnMatrix::empty({4, 8}, oa::ScalarType::Float32);
  const auto qWeight = oa::FnMatrix::empty({8, 8}, oa::ScalarType::Float32);
  const auto kWeight = oa::FnMatrix::empty({8, 8}, oa::ScalarType::Float32);
  const auto vWeight = oa::FnMatrix::empty({8, 8}, oa::ScalarType::Float32);
  ctx.clear();
  const auto q = oa::FnMatrix::matMulNt(input, qWeight, oa::MatMulPrecision::Fp32);
  const auto k = oa::FnMatrix::matMulNt(input, kWeight, oa::MatMulPrecision::Fp32);
  const auto v = oa::FnMatrix::matMulNt(input, vWeight, oa::MatMulPrecision::Fp32);
  ASSERT_EQ(ctx.semanticGraph()->operationCount(), 3U);

  oa::TrainingProgram program;
  const oa::Matrix *observed[] = {&q, &k, &v};
  oa::TrainingProgramOptions options;
  options.observedOutputs = observed;
  ASSERT_TRUE(program.capture(ctx.engine(), options).isOk());
  const auto report =
      testStdString(program.compilationDebugReportJson("automatic-dnn-capture"));
  EXPECT_NE(report.find("\"stage\": \"fusion\", \"state\": \"analyzed\", "
                        "\"input_count\": 3, \"output_count\": 1"),
            std::string::npos);
  EXPECT_NE(report.find("\"source_operation_count\": 3"), std::string::npos);
  EXPECT_NE(report.find("\"captured_operation_count\": 3"), std::string::npos);
  EXPECT_NE(report.find("\"partition_count\": 1"), std::string::npos);
  EXPECT_NE(report.find("\"recognized_partition_count\": 1"),
            std::string::npos);
  ASSERT_TRUE(program.reset().isOk());
  ctx.clear();
}

TEST(TrainingProgram, RejectedCapturePreservesBothSourceGraphs) {
  if (not vkTestEngineOk())
    GTEST_SKIP();

  auto &ctx = oa::ExecutionSession::getActive();
  ctx.clear();
  const auto a = oa::FnMatrix::empty({4});
  const auto b = oa::FnMatrix::empty({4});
  const auto m = oa::FnMatrix::empty({4});
  const auto v = oa::FnMatrix::empty({4});
  ctx.clear();
  const auto sum = oa::FnMatrix::add(a, b);

  struct AdamPush {
    oa::U32 Count;
    oa::F32 lr;
    oa::F32 beta1;
    oa::F32 beta2;
    oa::F32 eps;
    oa::U32 step;
  } push{4, 1e-3F, 0.9F, 0.999F, 1e-8F, 1};
  oa::BufferAccess access[] = {
      oa::BufferAccess::ReadWrite,
      oa::BufferAccess::Read,
      oa::BufferAccess::ReadWrite,
      oa::BufferAccess::ReadWrite,
  };
  ctx.add( "Adam", {&sum, &b, &m, &v},
                       oa::Span<oa::BufferAccess>(access, 4), &push, sizeof(push),
                       1);
  ASSERT_EQ(ctx.nodeCount(), 2U);
  ASSERT_EQ(ctx.semanticGraph()->operationCount(), 1U);

  oa::TrainingProgram program;
  const auto status = program.capture(ctx.engine());
  ASSERT_FALSE(status.isOk());
  EXPECT_EQ(status.getCode(), oa::StatusCode::FailedPrecondition);
  EXPECT_FALSE(program.isCaptured());
  EXPECT_EQ(program.nodeCount(), 0U);
  EXPECT_EQ(program.semanticOpCount(), 0U);
  const auto failedStages = program.compilationStages();
  ASSERT_EQ(failedStages.size(), 2U);
  EXPECT_EQ(failedStages[0].stage,
            oa::TrainingCompilationStage::SemanticValidation);
  EXPECT_EQ(failedStages[0].state, oa::TrainingCompilationState::Applied);
  EXPECT_EQ(failedStages[1].stage, oa::TrainingCompilationStage::ReplaySafety);
  EXPECT_EQ(failedStages[1].state, oa::TrainingCompilationState::Failed);
  const auto failedReport =
      testStdString(program.compilationDebugReportJson("rejected-training-step"));
  EXPECT_NE(failedReport.find("\"captured\": false"), std::string::npos);
  EXPECT_NE(
      failedReport.find("\"stage\": \"replay_safety\", \"state\": \"failed\""),
      std::string::npos);
  EXPECT_EQ(ctx.nodeCount(), 2U);
  EXPECT_EQ(ctx.semanticGraph()->operationCount(), 1U);
  EXPECT_EQ(ctx.semanticGraph()->operations()[0].name,
            oa::detail::opRegistry::FnMatrix::add.name);
  ctx.clear();
}

TEST(TrainingProgram, CapturedPhiloxAdvancesWithoutFreezingRandomValues) {
  if (not vkTestEngineOk())
    GTEST_SKIP();

  auto &ctx = oa::ExecutionSession::getActive();
  oa::Matrix shape = oa::FnMatrix::zeros(oa::MatrixShape{8});
  ASSERT_TRUE(testSubmitAndWait(ctx).isOk());
  [[maybe_unused]] oa::Matrix random =
      oa::FnMatrix::philoxUniform(shape, 0.0F, 1.0F, 42);
  const oa::U32 eagerNodes = ctx.nodeCount();
  ASSERT_GT(eagerNodes, 0U);

  oa::TrainingProgram program;
  ASSERT_TRUE(program.capture(ctx.engine()).isOk());
  EXPECT_EQ(ctx.nodeCount(), 0U);
  EXPECT_EQ(program.nodeCount(), eagerNodes + 1U); // RNG + counter advance

  ASSERT_TRUE(program.replay().isOk());
  ASSERT_TRUE(program.wait().isOk());
  oa::F32 first[8]{};
  std::memcpy(first, random.data(), sizeof(first));
  ASSERT_TRUE(program.replay().isOk());
  ASSERT_TRUE(program.wait().isOk());
  oa::F32 second[8]{};
  std::memcpy(second, random.data(), sizeof(second));

  oa::Bool changed = false;
  for (oa::U32 i = 0; i < 8; ++i) {
    EXPECT_GE(first[i], 0.0F);
    EXPECT_LT(first[i], 1.0F);
    EXPECT_GE(second[i], 0.0F);
    EXPECT_LT(second[i], 1.0F);
    changed = changed or first[i] != second[i];
  }
  EXPECT_TRUE(changed);
  ASSERT_TRUE(program.reset().isOk());
}

TEST(TrainingProgram, IteratorRecordsTwiceThenReplaysFixedShapeStep) {
  if (not vkTestEngineOk())
    GTEST_SKIP();

  auto &ctx = oa::ExecutionSession::getActive();
  oa::Matrix param = oa::FnMatrix::full(oa::MatrixShape{4}, 1.0F);
  oa::Matrix grad = oa::FnMatrix::full(oa::MatrixShape{4}, 0.25F);
  oa::Matrix momentum;
  ASSERT_TRUE(testSubmitAndWait(ctx).isOk());

  oa::OptimizerNoOp optimizer;
  oa::TrainingProgram program;
  oa::ItTraining iter(testEngine(), optimizer,
                    oa::ItTrainingConfig{
                        .totalSteps = 4,
                        .program = &program,
                    });
  oa::I32 prepareCalls = 0;
  oa::I32 recordCalls = 0;
  while (not iter.isDone()) {
    iter.step([&] { ++prepareCalls; },
              [&] {
                ++recordCalls;
                oa::FnOptim::sgdStep(param, momentum, grad, 0.1F, 0.0F, 0.0F);
                iter.recordLoss(param);
              });
  }
  ASSERT_TRUE(iter.finish().isOk());

  EXPECT_TRUE(program.isCaptured());
  EXPECT_EQ(prepareCalls, 4);
  EXPECT_EQ(recordCalls, 2); // eager warm-up + capture, never rebuilt afterward
  EXPECT_NEAR(param.at(0), 0.9F, 1e-6F);
  EXPECT_NEAR(iter.lastLoss(), 0.9F, 1e-6F);
  EXPECT_EQ(iter.gpuTimingStats().count, 3);
  EXPECT_GT(iter.lastGpuMs(), 0.0);
}

TEST(TrainingProgram, TwoPhaseStepClassifiesStableReplayInputs) {
  if (not vkTestEngineOk())
    GTEST_SKIP();

  auto &ctx = oa::ExecutionSession::getActive();
  oa::OptimizerNoOp optimizer;
  oa::TrainingProgram program;
  oa::ItTraining iter(testEngine(), optimizer,
                    oa::ItTrainingConfig{
                        .totalSteps = 3,
                        .program = &program,
                    });
  oa::Matrix input;
  oa::Matrix loss;
  while (not iter.isDone()) {
    iter.step(
        [&] {
          input = oa::FnMatrix::empty(oa::MatrixShape{4}, oa::ScalarType::Float32,
                                    oa::MemoryPlacement::HostUpload);
          auto *values = input.dataAs<oa::F32>();
          ASSERT_NE(values, nullptr);
          for (oa::U32 index = 0; index < 4U; ++index)
            values[index] = 1.0F;
        },
        [&] {
          oa::Matrix temporary1 = oa::FnMatrix::add(input, input);
          oa::Matrix temporary2 = oa::FnMatrix::add(temporary1, input);
          oa::Matrix temporary3 = oa::FnMatrix::add(temporary2, input);
          // The schema-owned reduction contributes the final semantic value;
          // observing it keeps the same physical output externally live.
          loss = oa::FnMatrix::sum(temporary3);
          iter.recordLoss(loss);
        });
  }
  ASSERT_TRUE(iter.finish().isOk());

  ASSERT_TRUE(program.isCaptured());
  const auto bindings = program.semanticStorageBindings();
  ASSERT_EQ(bindings.size(), 5U);
  EXPECT_TRUE(bindings[0].semanticExternal);
  EXPECT_TRUE(bindings[0].stableReplayInput);
  for (oa::U32 index = 1; index + 1U < bindings.size(); ++index) {
    EXPECT_FALSE(bindings[index].semanticExternal);
    EXPECT_FALSE(bindings[index].stableReplayInput);
    EXPECT_FALSE(bindings[index].observedOutput);
  }
  EXPECT_FALSE(bindings[4].semanticExternal);
  EXPECT_FALSE(bindings[4].stableReplayInput);
  EXPECT_TRUE(bindings[4].observedOutput);
  const auto resources = program.capturedResources();
  ASSERT_EQ(resources.size(), 5U);
  EXPECT_TRUE(resources[0].stableReplayInput);
  EXPECT_TRUE(resources[1].aliasCandidate);
  EXPECT_TRUE(resources[2].aliasCandidate);
  EXPECT_TRUE(resources[3].aliasCandidate);
  EXPECT_TRUE(resources[1].aliasMaterialized);
  EXPECT_FALSE(resources[2].aliasMaterialized);
  EXPECT_TRUE(resources[3].aliasMaterialized);
  EXPECT_TRUE(resources[4].observedOutput);
  EXPECT_TRUE(resources[0].isExternallyLive());
  EXPECT_FALSE(resources[1].isExternallyLive());
  EXPECT_TRUE(resources[4].isExternallyLive());
  EXPECT_EQ(program.aliasCandidateCount(), 3U);
  EXPECT_EQ(program.plannedAliasGroupCount(), 1U);
  EXPECT_EQ(program.potentialAliasSavings(), 4U * sizeof(oa::F32));
  EXPECT_EQ(program.materializedAliasSavings(), 4U * sizeof(oa::F32));
  EXPECT_EQ(program.stats().aliasBarrierCount, 1U);
  EXPECT_FLOAT_EQ(loss.item(), 16.0F);
  const auto report =
      testStdString(program.compilationDebugReportJson("captured-memory-analysis"));
  EXPECT_NE(report.find("\"candidate_count\": 3"), std::string::npos);
  EXPECT_NE(report.find("\"materialization_eligible_count\": 3"),
            std::string::npos);
  EXPECT_NE(report.find("\"alias_group_count\": 1"), std::string::npos);
  EXPECT_NE(report.find("\"potential_savings_bytes\": 16"), std::string::npos);
  EXPECT_NE(report.find("\"materialized_savings_bytes\": 16"),
            std::string::npos);
  EXPECT_NE(report.find("\"materialized\": true"), std::string::npos);
  const auto stages = program.compilationStages();
  ASSERT_EQ(stages.size(), 11U);
  EXPECT_EQ(stages[8].stage, oa::TrainingCompilationStage::MemoryPlanning);
  EXPECT_EQ(stages[8].state, oa::TrainingCompilationState::Applied);
  EXPECT_EQ(ctx.stableExternalResourceCount(), 1U);
  EXPECT_EQ(ctx.stableTransientResourceCount(), 0U);
}

TEST(TrainingProgram, RetainedTransientMatrixPreventsAliasMaterialization) {
  if (not vkTestEngineOk())
    GTEST_SKIP();

  oa::OptimizerNoOp optimizer;
  oa::TrainingProgram program;
  oa::ItTraining iter(testEngine(), optimizer,
                    oa::ItTrainingConfig{
                        .totalSteps = 3,
                        .program = &program,
                    });
  oa::Matrix input;
  oa::Matrix retained1;
  oa::Matrix retained2;
  oa::Matrix retained3;
  oa::Matrix loss;
  while (not iter.isDone()) {
    iter.step(
        [&] {
          input = oa::FnMatrix::empty(oa::MatrixShape{4}, oa::ScalarType::Float32,
                                    oa::MemoryPlacement::HostUpload);
          auto *values = input.dataAs<oa::F32>();
          ASSERT_NE(values, nullptr);
          for (oa::U32 index = 0; index < 4U; ++index)
            values[index] = 1.0F;
        },
        [&] {
          retained1 = oa::FnMatrix::add(input, input);
          retained2 = oa::FnMatrix::add(retained1, input);
          retained3 = oa::FnMatrix::add(retained2, input);
          loss = oa::FnMatrix::sum(retained3);
          iter.recordLoss(loss);
        });
  }
  ASSERT_TRUE(iter.finish().isOk());

  EXPECT_TRUE(program.isCaptured());
  EXPECT_EQ(program.aliasCandidateCount(), 3U);
  EXPECT_EQ(program.potentialAliasSavings(), 4U * sizeof(oa::F32));
  EXPECT_EQ(program.materializedAliasSavings(), 0U);
  for (oa::U32 index = 1; index <= 3U; ++index) {
    EXPECT_FALSE(program.capturedResources()[index].aliasMaterialized);
  }
  const auto report =
      testStdString(program.compilationDebugReportJson("retained-transient"));
  EXPECT_NE(report.find("\"materialization_eligible_count\": 0"),
            std::string::npos);
  EXPECT_NE(report.find("\"fallback_reason\": \"\""), std::string::npos);
  const auto stages = program.compilationStages();
  ASSERT_EQ(stages.size(), 11U);
  EXPECT_EQ(stages[8].stage, oa::TrainingCompilationStage::MemoryPlanning);
  EXPECT_EQ(stages[8].state, oa::TrainingCompilationState::Analyzed);
  EXPECT_FLOAT_EQ(loss.item(), 16.0F);
}

TEST(TrainingProgram, SinglePhaseStepConservativelyRetainsStableResources) {
  if (not vkTestEngineOk())
    GTEST_SKIP();

  auto &ctx = oa::ExecutionSession::getActive();
  oa::OptimizerNoOp optimizer;
  oa::ItTraining iter(testEngine(), optimizer,
                    oa::ItTrainingConfig{.totalSteps = 2});
  oa::Matrix first;
  oa::Matrix second;
  while (not iter.isDone()) {
    iter.step([&] {
      first = oa::FnMatrix::empty(oa::MatrixShape{4});
      second = oa::FnMatrix::full(oa::MatrixShape{4}, 1.0F);
      iter.recordLoss(second);
    });
  }
  ASSERT_TRUE(iter.finish().isOk());

  EXPECT_EQ(ctx.stableExternalResourceCount(), 2U);
  EXPECT_EQ(ctx.stableTransientResourceCount(), 0U);
}

TEST(TrainingProgram, ExplicitRecaptureRecordsNewProgramOnce) {
  if (not vkTestEngineOk())
    GTEST_SKIP();

  auto &ctx = oa::ExecutionSession::getActive();
  oa::Matrix param = oa::FnMatrix::full(oa::MatrixShape{4}, 1.0F);
  oa::Matrix grad = oa::FnMatrix::full(oa::MatrixShape{4}, 0.25F);
  oa::Matrix momentum;
  ASSERT_TRUE(testSubmitAndWait(ctx).isOk());

  oa::OptimizerNoOp optimizer;
  oa::TrainingProgram program;
  oa::ItTraining iter(testEngine(), optimizer,
                    oa::ItTrainingConfig{
                        .totalSteps = 4,
                        .program = &program,
                    });
  oa::I32 recordCalls = 0;
  while (not iter.isDone()) {
    if (iter.index() == 3)
      ASSERT_TRUE(iter.requestProgramRecapture().isOk());
    iter.step([] {},
              [&] {
                ++recordCalls;
                oa::FnOptim::sgdStep(param, momentum, grad, 0.1F, 0.0F, 0.0F);
                iter.recordLoss(param);
              });
  }
  ASSERT_TRUE(iter.finish().isOk());

  EXPECT_TRUE(program.isCaptured());
  EXPECT_EQ(recordCalls, 3); // warm-up, initial capture, explicit recapture
  EXPECT_NEAR(param.at(0), 0.9F, 1e-6F);
}

TEST(TrainingProgram, AdamWMatchesReferenceAndAcceptsReplayLrUpdate) {
  if (not vkTestEngineOk())
    GTEST_SKIP();

  auto &ctx = oa::ExecutionSession::getActive();
  oa::Parameter param;
  param.name = "weight";
  param.data = oa::FnMatrix::full(oa::MatrixShape{4}, 1.0F);
  param.data.setRequiresGrad(true);
  oa::FnMatrix::fillInPlace(param.grad(), 0.25F);
  ASSERT_TRUE(testSubmitAndWait(ctx).isOk());

  oa::Vec<oa::Parameter *> params;
  params.pushBack(&param);
  constexpr oa::F32 beta1 = 0.9F;
  constexpr oa::F32 beta2 = 0.999F;
  constexpr oa::F32 eps = 1e-8F;
  constexpr oa::F32 decay = 0.01F;
  oa::AdamW optimizer(params, 0.1F, beta1, beta2, eps, decay);
  oa::TrainingProgram program;
  oa::ItTraining iter(testEngine(), optimizer,
                    oa::ItTrainingConfig{
                        .totalSteps = 4,
                        .program = &program,
                    });
  while (not iter.isDone()) {
    iter.step([] {}, [&] { iter.recordLoss(param.data); });
    if (iter.index() == 2)
      optimizer.setLr(0.05F);
  }
  ASSERT_TRUE(iter.finish().isOk());

  oa::F32 expected = 1.0F;
  oa::F32 m = 0.0F;
  oa::F32 v = 0.0F;
  for (oa::I32 step = 1; step <= 4; ++step) {
    const oa::F32 lr = step <= 2 ? 0.1F : 0.05F;
    expected -= lr * decay * expected;
    m = beta1 * m + (1.0F - beta1) * 0.25F;
    v = beta2 * v + (1.0F - beta2) * 0.25F * 0.25F;
    const oa::F32 mHat = m / (1.0F - std::pow(beta1, static_cast<oa::F32>(step)));
    const oa::F32 vHat = v / (1.0F - std::pow(beta2, static_cast<oa::F32>(step)));
    expected -= lr * mHat / (std::sqrt(vHat) + eps);
  }

  EXPECT_TRUE(program.isCaptured());
  EXPECT_EQ(optimizer.getStep(), 4U);
  EXPECT_NEAR(param.data.at(0), expected, 2e-5F);
  EXPECT_NEAR(iter.lastLoss(), expected, 2e-5F);
}

TEST(TrainingMetrics, LossIsCountedExactlyOncePerStep) {
  if (not vkTestEngineOk())
    GTEST_SKIP();

  oa::OptimizerNoOp opt;
  oa::MetricLoss lossMetric;
  oa::MetricLoss lastLossMetric("loss", oa::MetricLoss::Mode::Last);
  SampleRecorder recorder;

  oa::ItTraining iter(testEngine(), opt,
                    oa::ItTrainingConfig{
                        .totalSteps = 5,
                        .batchSize = 4,
                        .metrics = {&lossMetric, &lastLossMetric},
                        .callbacks = {&recorder},
                    });

  while (not iter.isDone()) {
    const oa::F32 value = static_cast<oa::F32>(iter.stepCount());
    iter.next(oa::FnMatrix::full(oa::MatrixShape{1}, value));
  }
  ASSERT_TRUE(iter.finish().isOk());

  ASSERT_EQ(recorder.lossSteps.size(), 5U);
  EXPECT_EQ(recorder.lossSteps[0], 1);
  EXPECT_EQ(recorder.lossSteps[1], 2);
  EXPECT_EQ(recorder.lossSteps[2], 3);
  EXPECT_EQ(recorder.lossSteps[3], 4);
  EXPECT_EQ(recorder.lossSteps[4], 5);
  EXPECT_FLOAT_EQ(recorder.lossValues[0], 1.0F);
  EXPECT_FLOAT_EQ(recorder.lossValues[1], 2.0F);
  EXPECT_FLOAT_EQ(recorder.lossValues[2], 3.0F);
  EXPECT_FLOAT_EQ(recorder.lossValues[3], 4.0F);
  EXPECT_FLOAT_EQ(recorder.lossValues[4], 5.0F);
  EXPECT_DOUBLE_EQ(lossMetric.result(), 3.0);
  EXPECT_DOUBLE_EQ(lastLossMetric.result(), 5.0);
  EXPECT_EQ(iter.epochLossCount(), 5);
  EXPECT_DOUBLE_EQ(iter.epochMeanLoss(), 3.0);
}

TEST(TrainingMetrics, WorkloadRatesHaveUnambiguousUnits) {
  if (not vkTestEngineOk())
    GTEST_SKIP();

  oa::OptimizerNoOp opt;
  oa::ItTraining iter(testEngine(), opt,
                    oa::ItTrainingConfig{
                        .totalSteps = 4,
                        .batchSize = 8,
                        .sequenceLength = 4,
                        .sequenceUnit = "token",
                        .sourceUnitsPerSample = 4.0,
                        .sourceUnit = "byte",
                    });

  while (not iter.isDone()) {
    iter.next(oa::FnMatrix::full(oa::MatrixShape{1}, 1.0F));
  }
  ASSERT_TRUE(iter.finish().isOk());

  EXPECT_GT(iter.wallMsPerStep(), 0.0);
  EXPECT_GT(iter.wallSamplesPerSecond(), 0.0);
  EXPECT_GT(iter.wallUnitsPerSecond(), 0.0);
  EXPECT_NEAR(iter.wallUnitsPerSecond(), iter.wallSamplesPerSecond() * 4.0,
              iter.wallSamplesPerSecond() * 0.01);
  EXPECT_EQ(iter.totalSamples(), 32);
  EXPECT_EQ(iter.totalUnits(), 128);
  EXPECT_EQ(iter.totalSourceUnits(), 128);
  EXPECT_NEAR(iter.wallSourceUnitsPerSecond(),
              iter.wallSamplesPerSecond() * 4.0,
              iter.wallSamplesPerSecond() * 0.01);
}

TEST(TrainingMetrics, VariableSourceWorkOverridesFixedRate) {
  if (not vkTestEngineOk())
    GTEST_SKIP();
  oa::OptimizerNoOp opt;
  oa::ItTraining iter(testEngine(), opt,
                    oa::ItTrainingConfig{
                        .totalSteps = 3,
                        .batchSize = 2,
                        .sequenceLength = 4,
                        .sequenceUnit = "token",
                        .sourceUnit = "byte",
                    });
  const oa::I64 bytes[] = {11, 13, 17};
  while (not iter.isDone()) {
    iter.recordSourceUnits(bytes[iter.index() - 1]);
    iter.next(oa::FnMatrix::full(oa::MatrixShape{1}, 1.0F));
  }
  ASSERT_TRUE(iter.finish().isOk());
  EXPECT_EQ(iter.totalSourceUnits(), 41);
  EXPECT_GT(iter.wallSourceUnitsPerSecond(), 0.0);
}

TEST(TrainingMetrics, BatchDefaultsToOneSample) {
  if (not vkTestEngineOk())
    GTEST_SKIP();
  oa::OptimizerNoOp opt;
  oa::ItTraining iter(testEngine(), opt, oa::ItTrainingConfig{.totalSteps = 3});
  while (not iter.isDone())
    iter.next(oa::FnMatrix::full(oa::MatrixShape{1}, 1.0F));
  ASSERT_TRUE(iter.finish().isOk());
  EXPECT_EQ(iter.cfg().batchSize, 1);
  EXPECT_EQ(iter.totalSamples(), 3);
}

TEST(TrainingMetrics, LastLossRenderOmitsAggregationSuffix) {
  oa::MetricLoss meanLoss("recon");
  meanLoss.update(0.25F);
  meanLoss.update(0.75F);
  char buffer[64]{};
  ASSERT_GT(meanLoss.render(buffer, sizeof(buffer), false), 0);
  EXPECT_STREQ(buffer, "recon: 0.5");
  EXPECT_EQ(std::strchr(buffer, '('), nullptr);
}

TEST(TrainingMetrics, LiveValueFormattingExpandsOnlyWhenRoundedValueStalls) {
  oa::MetricLoss lastLoss("recon", oa::MetricLoss::Mode::Last);
  char buffer[64]{};
  lastLoss.update(0.23444F);
  ASSERT_GT(lastLoss.render(buffer, sizeof(buffer), false), 0);
  EXPECT_STREQ(buffer, "recon: 0.2344");
  lastLoss.update(0.234449F);
  ASSERT_GT(lastLoss.render(buffer, sizeof(buffer), false), 0);
  EXPECT_STREQ(buffer, "recon: 0.23445");

  oa::MetricValueFormatter smallValue;
  ASSERT_GT(smallValue.format(buffer, sizeof(buffer), 4.95e-5), 0);
  EXPECT_STREQ(buffer, "0.0000495");
}

TEST(TrainingMetrics, AccuracyReducesLogitsOnGpu) {
  if (not vkTestEngineOk())
    GTEST_SKIP();
  const std::vector<oa::F32> logitsHost = {
      4, 1, 0, 0, 5, 1, 0, 2, 3, 3, 2, 1,
  };
  const std::vector<oa::U8> labelsHost = {0, 1, 1,
                                        0}; // row 2 intentionally wrong
  auto logits = oa::FnMatrix::fromBytes(
      oa::Span<const oa::U8>(reinterpret_cast<const oa::U8 *>(logitsHost.data()),
                         logitsHost.size() * sizeof(oa::F32)),
      oa::MatrixShape{4, 3}, oa::ScalarType::Float32);
  auto labels = oa::FnMatrix::fromBytes(
      oa::Span<const oa::U8>(labelsHost.data(), labelsHost.size()),
      oa::MatrixShape{4}, oa::ScalarType::UInt8);
  oa::MetricAccuracy accuracy;
  accuracy.update(logits, labels);
  EXPECT_DOUBLE_EQ(accuracy.result(), 0.75);
  EXPECT_EQ(accuracy.count(), 4);
}

// ─── Callback Lifecycle ───────────────────────────────────────────────────

class LifecycleRecorder final : public oa::CbTraining {
public:
  void onTrainBegin(oa::ItTraining &) override { ++trainBeginCount; }
  void onEpochBegin(oa::ItTraining &) override { ++epochBeginCount; }
  void onStepEnd(oa::ItTraining &) override { ++stepEndCount; }
  void onEpochEnd(oa::ItTraining &) override { ++epochEndCount; }
  void onTrainEnd(oa::ItTraining &) override { ++trainEndCount; }

  int trainBeginCount = 0;
  int epochBeginCount = 0;
  int stepEndCount = 0;
  int epochEndCount = 0;
  int trainEndCount = 0;
};

TEST(TrainingMetrics, CallbackLifecycleOrder) {
  if (not vkTestEngineOk())
    GTEST_SKIP();

  oa::OptimizerNoOp opt;
  LifecycleRecorder rec;
  oa::ItTraining iter(testEngine(), opt,
                    oa::ItTrainingConfig{
                        .totalSteps = 6,
                        .stepsPerEpoch = 3,
                        .batchSize = 2,
                        .callbacks = {&rec},
                    });

  while (not iter.isDone()) {
    iter.next(oa::FnMatrix::full(oa::MatrixShape{1}, 1.0F));
  }
  ASSERT_TRUE(iter.finish().isOk());

  EXPECT_EQ(rec.trainBeginCount, 1);
  EXPECT_EQ(rec.epochBeginCount, 2); // epoch 1 and epoch 2
  EXPECT_EQ(rec.stepEndCount, 6);
  EXPECT_EQ(rec.epochEndCount, 2); // after step 3 and step 6
  EXPECT_EQ(rec.trainEndCount, 1);
}

TEST(TrainingMetrics, ValidationRunsOncePerEpochAndPublishesLastLoss) {
  if (not vkTestEngineOk())
    GTEST_SKIP();

  oa::OptimizerNoOp opt;
  oa::I32 evalCount = 0;
  oa::CbValidation validation([&evalCount](oa::ItTraining &inIter) {
    ++evalCount;
    return oa::ValidationResult{
        .loss = 1.0 / static_cast<oa::F64>(inIter.epoch()),
        .batches = 2,
        .samples = 8,
    };
  });
  oa::ItTraining iter(testEngine(), opt,
                    oa::ItTrainingConfig{
                        .totalSteps = 6,
                        .stepsPerEpoch = 3,
                        .batchSize = 4,
                        .callbacks = {&validation},
                    });

  while (not iter.isDone()) {
    iter.next(oa::FnMatrix::full(oa::MatrixShape{1}, 1.0F));
  }
  ASSERT_TRUE(iter.finish().isOk());

  EXPECT_EQ(evalCount, 2);
  EXPECT_DOUBLE_EQ(validation.metric().result(), 0.5);
  EXPECT_EQ(validation.lastResult().batches, 2);
  EXPECT_EQ(validation.lastResult().samples, 8);
  EXPECT_GE(validation.lastSeconds(), 0.0);
}

TEST(TrainingMetrics, PartialFinalEpochStillCloses) {
  if (not vkTestEngineOk())
    GTEST_SKIP();
  oa::OptimizerNoOp opt;
  LifecycleRecorder rec;
  oa::ItTraining iter(testEngine(), opt,
                    oa::ItTrainingConfig{
                        .totalSteps = 5,
                        .stepsPerEpoch = 3,
                        .callbacks = {&rec},
                    });
  while (not iter.isDone())
    iter.next(oa::FnMatrix::full(oa::MatrixShape{1}, 1.0F));
  ASSERT_TRUE(iter.finish().isOk());
  EXPECT_EQ(iter.totalEpochs(), 2);
  EXPECT_EQ(rec.epochBeginCount, 2);
  EXPECT_EQ(rec.epochEndCount, 2);
  EXPECT_EQ(iter.stepsInCurrentEpoch(), 2);
}

// ─── epoch Boundary Sampling ──────────────────────────────────────────────

class EpochBoundaryRecorder final : public oa::CbTraining {
public:
  void onStepEnd(oa::ItTraining &inIter) override {
    if (inIter.hasLossSample()) {
      epochAtSample.pushBack(inIter.epoch());
      stepInEpochAtSample.pushBack(inIter.stepInEpoch());
      lossAtSample.pushBack(inIter.lastLoss());
    }
  }
  void onEpochEnd(oa::ItTraining &inIter) override {
    epochEndCount.pushBack(static_cast<oa::F32>(inIter.epoch()));
    meanLossAtEpochEnd.pushBack(static_cast<oa::F32>(inIter.epochMeanLoss()));
  }

  oa::Vec<oa::I64> epochAtSample;
  oa::Vec<oa::I64> stepInEpochAtSample;
  oa::Vec<oa::F32> lossAtSample;
  oa::Vec<oa::F32> epochEndCount;
  oa::Vec<oa::F32> meanLossAtEpochEnd;
};

TEST(TrainingMetrics, FirstFinalAndEpochBoundaryAreSampled) {
  if (not vkTestEngineOk())
    GTEST_SKIP();

  oa::OptimizerNoOp opt;
  EpochBoundaryRecorder rec;
  oa::ItTraining iter(testEngine(), opt,
                    oa::ItTrainingConfig{
                        .totalSteps = 9,
                        .stepsPerEpoch = 3,
                        .batchSize = 2,
                        .callbacks = {&rec},
                    });

  while (not iter.isDone()) {
    const oa::F32 value = static_cast<oa::F32>(iter.stepCount());
    iter.next(oa::FnMatrix::full(oa::MatrixShape{1}, value));
  }
  ASSERT_TRUE(iter.finish().isOk());

  ASSERT_EQ(rec.lossAtSample.size(), 9U);
  EXPECT_EQ(rec.stepInEpochAtSample[0], 1); // step 1
  EXPECT_EQ(rec.stepInEpochAtSample[1], 2); // step 2
  EXPECT_EQ(rec.stepInEpochAtSample[2], 3); // step 3 (boundary)
  EXPECT_EQ(rec.stepInEpochAtSample[3], 1); // step 4
  EXPECT_EQ(rec.stepInEpochAtSample[4], 2); // step 5
  EXPECT_EQ(rec.stepInEpochAtSample[5], 3); // step 6 (boundary)
  EXPECT_EQ(rec.stepInEpochAtSample[6], 1); // step 7
  EXPECT_EQ(rec.stepInEpochAtSample[7], 2); // step 8
  EXPECT_EQ(rec.stepInEpochAtSample[8], 3); // step 9 (final+boundary)

  // verify final step was actually captured
  EXPECT_FLOAT_EQ(rec.lossAtSample.back(), 9.0F);
}

TEST(TrainingMetrics, EpochResetDoesNotDiscardOrDuplicateBoundarySample) {
  if (not vkTestEngineOk())
    GTEST_SKIP();

  oa::OptimizerNoOp opt;
  EpochBoundaryRecorder rec;
  oa::ItTraining iter(testEngine(), opt,
                    oa::ItTrainingConfig{
                        .totalSteps = 8,
                        .stepsPerEpoch = 4,
                        .batchSize = 2,
                        .callbacks = {&rec},
                    });

  while (not iter.isDone()) {
    const oa::F32 value = static_cast<oa::F32>(iter.stepCount());
    iter.next(oa::FnMatrix::full(oa::MatrixShape{1}, value));
  }
  ASSERT_TRUE(iter.finish().isOk());

  // Every completed step is sampled.
  // epoch 1: steps 1,2,3,4  -> mean = (1+2+3+4)/4 = 2.5
  // epoch 2: steps 5,6,7,8  -> mean = (5+6+7+8)/4 = 6.5
  ASSERT_EQ(rec.lossAtSample.size(), 8U);
  ASSERT_EQ(rec.epochEndCount.size(), 2U);

  EXPECT_FLOAT_EQ(rec.meanLossAtEpochEnd[0], 2.5F);
  EXPECT_FLOAT_EQ(rec.meanLossAtEpochEnd[1], 6.5F);

  // verify no boundary sample leaked into wrong epoch
  for (size_t i = 0; i < 4; ++i)
    EXPECT_EQ(rec.epochAtSample[i], 1);
  for (size_t i = 4; i < 8; ++i)
    EXPECT_EQ(rec.epochAtSample[i], 2);
}

TEST(TrainingMetrics, EarlyExitAlreadyHasExactFinalSample) {
  if (not vkTestEngineOk())
    GTEST_SKIP();

  oa::OptimizerNoOp opt;
  oa::ItTraining iter(testEngine(), opt,
                    oa::ItTrainingConfig{
                        .totalSteps = 10, // declare 10 but only run 2
                        .batchSize = 2,
                    });

  // run only 2 steps.
  // Must call isDone() before each next() so index_ advances properly.
  int step = 0;
  while (not iter.isDone() and step < 2) {
    iter.next(oa::FnMatrix::full(oa::MatrixShape{1},
                               static_cast<oa::F32>(iter.stepCount())));
    ++step;
  }

  // The completed step is already exact before finish.
  EXPECT_EQ(iter.lastLossStep(), 2);
  EXPECT_FLOAT_EQ(iter.lastLoss(), 2.0F);

  ASSERT_TRUE(iter.finish().isOk());

  // finish does not manufacture another sample.
  EXPECT_EQ(iter.lastLossStep(), 2);
  EXPECT_FLOAT_EQ(iter.lastLoss(), 2.0F);
}
