// oa::ExecutableGraph Test Suite — Comprehensive testing of CPU and GPU execution paths
//
// Validates graph construction, synchronization, compiled replay, context
// batching, correctness, performance instrumentation and edge cases.
//
// Test structure (similar to TestAutograd):
// 1. Basic functionality tests (Add, execute, compile, replay)
// 2. Correctness tests (one-shot, compiled replay and context batching)
// 3. Edge cases (empty graphs, single node, large graphs)
// 4. Performance benchmarks (CPU overhead, throughput)
// 5. memory analysis (aliasing, lifetimes)

#include "../../oaTest.h"
#include <oa/runtime/executableGraph.h>
#include <oa/runtime/dnn.h>
#include <oa/runtime/semanticGraph.h>
#include <oa/runtime/executionSession.h>
#include <oa/runtime/timer.h>
#include <oa/runtime/engine.h>
#include <oa/runtime/eventAccess.h>
#include <oa/runtime/engine/allocatorAccess.h>
#include <oa/runtime/engine/bindlessAccess.h>
#include <oa/runtime/engine/engineAccess.h>
#include <oa/runtime/engine/submissionAccess.h>
#include <oa/core/op.h>
#include <oa/core/fnMatrix.h>
#include <oa/core/fnmatrix/fnMatrixInternal.h>
#include <oa/core/matrixAccess.h>
#include <oa/core/memory.h>
#include <oa/ml/autograd.h>
#include <oa/ml/fnLoss.h>
#include <oa/ml/fnMatrix.h>

#include <chrono>
#include <cstring>
#include <cstdio>

// =============================================================================
// HELPERS
// =============================================================================

static void printBar() {
	fprintf(stderr, "  ────────────────────────────────────────────────────────────────\n");
}

static void printHeader(const char* inSection) {
	fprintf(stderr, "\n");
	printBar();
	fprintf(stderr, "  %s\n", inSection);
	printBar();
}

template<typename F>
static double measureUs(oa::I32 inWarmup, oa::I32 inIters, F&& inFunc) {
	for (oa::I32 i = 0; i < inWarmup; ++i) inFunc();

	auto start = std::chrono::high_resolution_clock::now();
	for (oa::I32 i = 0; i < inIters; ++i) inFunc();
	auto end = std::chrono::high_resolution_clock::now();

	double totalUs = std::chrono::duration<double, std::micro>(end - start).count();
	return totalUs / inIters;
}

// Build a chain of scale dispatches for testing
static void buildChainGraph(
	oa::ExecutableGraph& outGraph,
	oa::Vec<oavk::Buffer>& inBufs,
	oa::U32 inNumDispatches)
{
	struct { oa::U32 N; oa::F32 scale; } pc = {256, 1.001f};

	for (oa::U32 i = 0; i < inNumDispatches; ++i) {
		oavk::Buffer bufs[] = {inBufs[i], inBufs[i + 1]};
		oa::BufferAccess acc[] = {oa::BufferAccess::Read, oa::BufferAccess::Write};
		outGraph.add("Scale", bufs, acc, &pc, sizeof(pc), 1);
	}
}

// =============================================================================
// BASIC FUNCTIONALITY TESTS
// =============================================================================

TEST(ExecutableGraph, DispatchDescriptorCopiesAllMetadata) {
	oavk::Buffer buffers[2];
	buffers[0].buffer = reinterpret_cast<void*>(0x1000);
	buffers[0].bindlessIndex = 7;
	buffers[1].buffer = reinterpret_cast<void*>(0x2000);
	buffers[1].bindlessIndex = 11;
	oa::BufferAccess access[] = {
		oa::BufferAccess::Read, oa::BufferAccess::Write};
	struct Push { oa::U32 count; oa::F32 scale; } push{64, 2.0F};
	const oa::U32 semanticOps[] = {7U};

	oa::ComputeDispatchDesc desc;
	desc.operation = "scale";
	desc.semanticOps = semanticOps;
	desc.implementationId = 0x1234U;
	desc.opContractHash = 0x5678U;
	desc.problemContractHash = 0x789aU;
	desc.kernelContentHash = 0x9abcU;
	desc.kernel = "Scale";
	desc.buffers = buffers;
	desc.access = access;
	desc.pushData = &push;
	desc.pushSize = sizeof(push);
	desc.dtype = 1;
	desc.groupsX = 3;
	desc.groupsY = 2;
	desc.queue = oa::QueueHint::AsyncCompute;

	oa::ExecutableGraph graph;
	graph.add(desc);
	ASSERT_EQ(graph.nodeCount(), 1U);

	// Descriptor storage is non-owning, but the graph node must be a complete
	// owning snapshot before Record/Add returns.
	push.count = 0;
	buffers[0].bindlessIndex = 99;
	const auto nodes = graph.nodes();
	ASSERT_EQ(nodes.size(), 1U);
	EXPECT_EQ(nodes[0].operation, "scale");
	ASSERT_EQ(nodes[0].semanticOps.size(), 1U);
	EXPECT_EQ(nodes[0].semanticOps[0], 7U);
	EXPECT_EQ(nodes[0].implementationId, 0x1234U);
	EXPECT_EQ(nodes[0].opContractHash, 0x5678U);
	EXPECT_EQ(nodes[0].problemContractHash, 0x789aU);
	EXPECT_EQ(nodes[0].kernelContentHash, 0x9abcU);
	EXPECT_EQ(nodes[0].shader, "Scale");
	EXPECT_EQ(nodes[0].buffers[0].bindlessIndex, 7U);
	EXPECT_EQ(nodes[0].dtype, 1U);
	EXPECT_EQ(nodes[0].groupsX, 3U);
	EXPECT_EQ(nodes[0].groupsY, 2U);
	EXPECT_EQ(nodes[0].queue, oa::QueueHint::AsyncCompute);
	Push copied{};
	std::memcpy(&copied, nodes[0].pushData, sizeof(copied));
	EXPECT_EQ(copied.count, 64U);
	EXPECT_FLOAT_EQ(copied.scale, 2.0F);
}

TEST(ExecutableGraph, SemanticOperationContractsComeFromSchema) {
	EXPECT_EQ(oa::detail::opRegistry::FnMatrix::add.name, "oa::FnMatrix::add");
	EXPECT_NE(oa::detail::opRegistry::FnMatrix::add.hash, 0U);
	EXPECT_EQ(oa::detail::opRegistry::FnMatrix::add.shapeRule, oa::OpShapeRule::Broadcast);
	EXPECT_EQ(oa::detail::opRegistry::FnMatrix::add.lowering, oa::OpLowering::Dispatch);
	EXPECT_EQ(oa::detail::opRegistry::FnMatrix::add.mutatedInputMask, 0U);
	EXPECT_EQ(oa::detail::opRegistry::FnMatrix::add.aliasInputForOutput(0),
		oa::OpContract::NoAliasInput);
	EXPECT_EQ(oa::detail::opRegistry::FnMatrix::add.controlFlow,
		oa::OpControlFlow::StraightLine);
	EXPECT_EQ(
		oa::detail::opRegistry::FnMatrix::addInPlace.name,
		"oa::FnMatrix::addInPlace");
	EXPECT_NE(oa::detail::opRegistry::FnMatrix::addInPlace.hash, 0U);
	EXPECT_TRUE(oa::detail::opRegistry::FnMatrix::addInPlace.mutatesInput(0U));
	EXPECT_EQ(oa::detail::opRegistry::FnMatrix::addInPlace.aliasInputForOutput(0U), 0U);
	const oa::OpContract* matrixMutations[] = {
		&oa::detail::opRegistry::FnMatrix::subInPlace,
		&oa::detail::opRegistry::FnMatrix::mulInPlace,
		&oa::detail::opRegistry::FnMatrix::divInPlace,
	};
	for (const auto* mutation : matrixMutations) {
		EXPECT_NE(mutation->hash, 0U);
		EXPECT_TRUE(mutation->mutatesInput(0U));
		EXPECT_EQ(mutation->aliasInputForOutput(0U), 0U);
		EXPECT_EQ(mutation->shapeRule, oa::OpShapeRule::Broadcast);
	}
	EXPECT_EQ(oa::detail::opRegistry::FnMatrix::scale.name, "oa::FnMatrix::scale");
	EXPECT_NE(oa::detail::opRegistry::FnMatrix::scale.hash, 0U);
	EXPECT_EQ(oa::detail::opRegistry::FnMatrix::scale.attributeCount, 1U);
	EXPECT_NE(oa::detail::opRegistry::FnMatrix::scale.attributeSignatureHash, 0U);
	EXPECT_EQ(
		oa::detail::opRegistry::FnMatrix::matMulNt.name,
		"oa::FnMatrix::matMulNt");
	EXPECT_NE(oa::detail::opRegistry::FnMatrix::matMulNt.hash, 0U);
	EXPECT_EQ(oa::detail::opRegistry::FnMatrix::matMulNt.shapeRule, oa::OpShapeRule::MatMulNt);
	EXPECT_EQ(oa::detail::opRegistry::FnMatrix::matMulNt.lowering, oa::OpLowering::Gemm);
	EXPECT_EQ(oa::detail::opRegistry::FnMatrix::copy.name, "oa::FnMatrix::copy");
	EXPECT_EQ(oa::detail::opRegistry::FnMatrix::copy.shapeRule,
		oa::OpShapeRule::MatchInput);
	EXPECT_EQ(oa::detail::opRegistry::FnMatrix::copy.aliasInputForOutput(0U),
		oa::OpContract::NoAliasInput);
	EXPECT_EQ(oa::detail::opRegistry::FnMatrix::copy.differentiation,
		oa::OpDifferentiation::Reverse);
	EXPECT_EQ(oa::detail::opRegistry::FnMatrix::slice.name,
		"oa::FnMatrix::slice");
	EXPECT_EQ(oa::detail::opRegistry::FnMatrix::slice.shapeRule,
		oa::OpShapeRule::Explicit);
	EXPECT_EQ(oa::detail::opRegistry::FnMatrix::slice.attributeCount, 3U);
	EXPECT_EQ(oa::detail::opRegistry::FnMatrix::slice.aliasInputForOutput(0U),
		oa::OpContract::NoAliasInput);
	EXPECT_TRUE(oa::detail::opRegistry::FnMatrix::concat.hasVariadicInputs());
	EXPECT_FALSE(oa::detail::opRegistry::FnMatrix::concat.hasVariadicOutputs());
	EXPECT_EQ(oa::detail::opRegistry::FnMatrix::concat.inputCount, 0U);
	EXPECT_EQ(
		oa::detail::opRegistry::FnMatrix::concat.minimumVariadicInputCount, 1U);
	EXPECT_EQ(
		oa::detail::opRegistry::FnMatrix::concat.inputKindAt(9U),
		oa::OpValueKind::Matrix);
	EXPECT_TRUE(oa::detail::opRegistry::FnMatrix::split.hasVariadicOutputs());
	EXPECT_FALSE(oa::detail::opRegistry::FnMatrix::split.hasVariadicInputs());
	EXPECT_EQ(oa::detail::opRegistry::FnMatrix::split.outputCount, 0U);
	EXPECT_EQ(
		oa::detail::opRegistry::FnMatrix::split.minimumVariadicOutputCount, 1U);
	EXPECT_EQ(
		oa::detail::opRegistry::FnMatrix::split.outputKindAt(9U),
		oa::OpValueKind::Matrix);
	EXPECT_EQ(oa::detail::opRegistry::FnMatrix::transpose.name,
		"oa::FnMatrix::transpose");
	EXPECT_EQ(oa::detail::opRegistry::FnMatrix::transpose.shapeRule,
		oa::OpShapeRule::Explicit);
	EXPECT_EQ(oa::detail::opRegistry::FnMatrix::transpose.attributeCount, 2U);
	EXPECT_EQ(oa::detail::opRegistry::FnMatrix::transpose.aliasInputForOutput(0U),
		oa::OpContract::NoAliasInput);
	EXPECT_EQ(oa::detail::opRegistry::FnMatrix::transpose.differentiation,
		oa::OpDifferentiation::Reverse);
	EXPECT_EQ(oa::detail::opRegistry::FnMatrix::fill.name,
		"oa::FnMatrix::fill");
	EXPECT_EQ(oa::detail::opRegistry::FnMatrix::fill.inputCount, 0U);
	EXPECT_EQ(oa::detail::opRegistry::FnMatrix::fill.outputCount, 1U);
	EXPECT_EQ(oa::detail::opRegistry::FnMatrix::fill.attributeCount, 2U);
	EXPECT_EQ(oa::detail::opRegistry::FnMatrix::fillInPlace.name,
		"oa::FnMatrix::fillInPlace");
	EXPECT_TRUE(oa::detail::opRegistry::FnMatrix::fillInPlace.mutatesInput(0U));
	EXPECT_EQ(
		oa::detail::opRegistry::FnMatrix::fillInPlace.aliasInputForOutput(0U),
		0U);
	EXPECT_EQ(oa::detail::opRegistry::FnMatrix::philoxUniform.attributeCount,
		3U);
	EXPECT_EQ(oa::detail::opRegistry::FnMatrix::philoxNormal.attributeCount,
		3U);
	EXPECT_EQ(
		oa::detail::opRegistry::FnMatrix::linearWeightBiasBwd.name,
		"oa::FnMatrix::linearWeightBiasBwd");
	EXPECT_EQ(oa::detail::opRegistry::FnMatrix::linearWeightBiasBwd.inputCount, 2U);
	EXPECT_EQ(oa::detail::opRegistry::FnMatrix::linearWeightBiasBwd.outputCount, 2U);
	EXPECT_EQ(oa::detail::opRegistry::FnMatrix::linearWeightBiasBwd.shapeRule,
		oa::OpShapeRule::Explicit);
	EXPECT_EQ(oa::detail::opRegistry::FnMatrix::linearWeightBiasBwd.lowering,
		oa::OpLowering::Dispatch);
	EXPECT_EQ(oa::detail::opRegistry::FnMatrix::topK.outputCount, 2U);
	EXPECT_EQ(oa::detail::opRegistry::FnMatrix::moeExpertPlan.outputCount, 6U);
	EXPECT_EQ(oa::detail::opRegistry::FnMatrix::compactRows.outputCount, 4U);
	EXPECT_TRUE(
		oa::detail::opRegistry::FnMatrix::compactRowsBwd.isInputOptional(3U));
	EXPECT_TRUE(
		oa::detail::opRegistry::FnMatrix::scatterRows.isInputOptional(4U));
	EXPECT_EQ(
		oa::detail::opRegistry::FnMatrix::moeRoutingBiasUpdate.outputCount, 0U);
	EXPECT_TRUE(
		oa::detail::opRegistry::FnMatrix::moeRoutingBiasUpdate.mutatesInput(1U));
	EXPECT_NE(oa::detail::opRegistry::FnMatrix::add.hash, oa::detail::opRegistry::FnMatrix::matMulNt.hash);
}

TEST(ExecutableGraph, SemanticGraphOwnsHandleFreeOperationTopology) {
	oa::SemanticGraph graph;
	oa::SemanticValueDesc inputA;
	inputA.name = "input_a";
	inputA.shape = {2, 3};
	inputA.external = true;
	oa::SemanticValueDesc inputB = inputA;
	inputB.name = "input_b";
	oa::SemanticValueDesc output;
	output.name = "sum";
	output.shape = {2, 3};
	output.isVirtual = true;

	auto a = graph.addValue(inputA);
	auto b = graph.addValue(inputB);
	auto sum = graph.addValue(output);
	ASSERT_TRUE(a.isOk());
	ASSERT_TRUE(b.isOk());
	ASSERT_TRUE(sum.isOk());
	const oa::U32 inputs[] = {a.getValue(), b.getValue()};
	const oa::U32 outputs[] = {sum.getValue()};
	auto operation = graph.addOp(
		oa::detail::opRegistry::FnMatrix::add, inputs, outputs);
	ASSERT_TRUE(operation.isOk()) << operation.getStatus().getMessage();
	EXPECT_EQ(operation.getValue(), 0U);
	ASSERT_TRUE(graph.attachAutograd(operation.getValue(), 0U, 17U).isOk());
	ASSERT_TRUE(graph.validate().isOk());
	ASSERT_EQ(graph.operationCount(), 1U);
	ASSERT_EQ(graph.valueCount(), 3U);
	ASSERT_EQ(graph.autograd().size(), 1U);
	EXPECT_EQ(graph.autograd()[0].forwardOp, operation.getValue());
	EXPECT_EQ(graph.autograd()[0].output, sum.getValue());
	EXPECT_EQ(graph.autograd()[0].outputIndex, 0U);
	EXPECT_EQ(graph.autograd()[0].sequence, 17U);
	EXPECT_EQ(graph.findValue(sum.getValue())->producer, operation.getValue());

	const auto operations = graph.operations();
	ASSERT_EQ(operations[0].accesses.size(), 3U);
	EXPECT_EQ(operations[0].accesses[0].mode, oa::SemanticAccessMode::Read);
	EXPECT_EQ(operations[0].accesses[2].mode, oa::SemanticAccessMode::Write);

	const auto first = graph.debugReportJson("pilot");
	const auto second = graph.debugReportJson("pilot");
	EXPECT_EQ(first, second);
	const auto text = first.stdStr();
	EXPECT_NE(text.find("\"schema\": \"oa.semantic_graph.v2\""),
		std::string::npos);
	EXPECT_NE(text.find("\"name\": \"oa::FnMatrix::add\""), std::string::npos);
	EXPECT_NE(text.find("\"lowering\": \"dispatch\""), std::string::npos);
	EXPECT_NE(text.find("\"mode\": \"write\""), std::string::npos);
	EXPECT_NE(text.find("\"control_flow\": \"straight_line\""),
		std::string::npos);
	EXPECT_NE(text.find("\"forward_operation\": 0"), std::string::npos);
	EXPECT_NE(text.find("\"sequence\": 17"), std::string::npos);
	EXPECT_EQ(text.find("VkBuffer"), std::string::npos);
}

TEST(ExecutableGraph, SemanticGraphOwnsMetadataViewTopology) {
	oa::SemanticGraph graph;
	oa::SemanticValueDesc source;
	source.name = "source";
	source.shape = {2, 3};
	source.strides[0] = 3;
	source.strides[1] = 1;
	source.external = true;
	oa::SemanticValueDesc view;
	view.name = "view";
	view.shape = {3, 2};
	view.strides[0] = 2;
	view.strides[1] = 1;

	const auto sourceValue = graph.addValue(source);
	const auto viewValue = graph.addValue(view);
	ASSERT_TRUE(sourceValue.isOk());
	ASSERT_TRUE(viewValue.isOk());
	ASSERT_TRUE(graph.addView(
		sourceValue.getValue(), viewValue.getValue(), 16).isOk());
	EXPECT_EQ(graph.viewCount(), 1U);
	ASSERT_TRUE(graph.validate().isOk());

	const auto* recorded = graph.findValue(viewValue.getValue());
	ASSERT_NE(recorded, nullptr);
	EXPECT_EQ(recorded->viewSource, sourceValue.getValue());
	EXPECT_EQ(recorded->viewByteOffset, 16);
	EXPECT_EQ(recorded->producer, oa::invalidSemanticOpId);
	EXPECT_EQ(recorded->strides[0], 2);
	EXPECT_EQ(recorded->strides[1], 1);

	const auto report = graph.debugReportJson("view");
	const auto text = report.stdStr();
	EXPECT_NE(text.find("\"strides\": [2, 1]"), std::string::npos);
	EXPECT_NE(text.find("\"view_source\": 0"), std::string::npos);
	EXPECT_NE(text.find("\"view_byte_offset\": 16"), std::string::npos);

	EXPECT_EQ(graph.addView(
		sourceValue.getValue(), viewValue.getValue(), 16).getCode(),
		oa::StatusCode::AlreadyExists);
	EXPECT_EQ(graph.addView(
		viewValue.getValue(), sourceValue.getValue(), 0).getCode(),
		oa::StatusCode::InvalidArgument);

	oa::SemanticValueDesc preassigned = view;
	preassigned.viewSource = sourceValue.getValue();
	EXPECT_EQ(graph.addValue(preassigned).getStatus().getCode(),
		oa::StatusCode::InvalidArgument);
}

TEST(ExecutableGraph, SemanticGraphRejectsContractAndSsaViolations) {
	oa::SemanticGraph graph;
	oa::SemanticValueDesc matrix;
	matrix.shape = {4};
	auto a = graph.addValue(matrix);
	auto b = graph.addValue(matrix);
	auto out = graph.addValue(matrix);
	ASSERT_TRUE(a.isOk());
	ASSERT_TRUE(b.isOk());
	ASSERT_TRUE(out.isOk());

	const oa::U32 wrongArity[] = {a.getValue()};
	const oa::U32 outputs[] = {out.getValue()};
	auto rejectedArity = graph.addOp(
		oa::detail::opRegistry::FnMatrix::add, wrongArity, outputs);
	ASSERT_FALSE(rejectedArity.isOk());
	EXPECT_EQ(rejectedArity.getStatus().getCode(), oa::StatusCode::InvalidArgument);

	const oa::U32 inputs[] = {a.getValue(), b.getValue()};
	auto accepted = graph.addOp(oa::detail::opRegistry::FnMatrix::add, inputs, outputs);
	ASSERT_TRUE(accepted.isOk());
	EXPECT_EQ(graph.attachAutograd(accepted.getValue(), 0U, 0U).getCode(),
		oa::StatusCode::InvalidArgument);
	EXPECT_EQ(graph.attachAutograd(accepted.getValue(), 1U, 1U).getCode(),
		oa::StatusCode::OutOfRange);
	ASSERT_TRUE(graph.attachAutograd(accepted.getValue(), 0U, 1U).isOk());
	EXPECT_EQ(graph.attachAutograd(accepted.getValue(), 0U, 2U).getCode(),
		oa::StatusCode::AlreadyExists);
	EXPECT_EQ(graph.completeAutograd(accepted.getValue(), 2U, 1U, 0U).getCode(),
		oa::StatusCode::NotFound);
	EXPECT_EQ(graph.completeAutograd(accepted.getValue(), 1U, 2U, 1U).getCode(),
		oa::StatusCode::OutOfRange);
	ASSERT_TRUE(graph.completeAutograd(
		accepted.getValue(), 1U, 1U, 0U).isOk());
	EXPECT_EQ(graph.completeAutograd(
		accepted.getValue(), 1U, 1U, 0U).getCode(),
		oa::StatusCode::AlreadyExists);
	auto duplicateProducer = graph.addOp(
		oa::detail::opRegistry::FnMatrix::add, inputs, outputs);
	ASSERT_FALSE(duplicateProducer.isOk());
	EXPECT_EQ(duplicateProducer.getStatus().getCode(), oa::StatusCode::AlreadyExists);
	EXPECT_EQ(graph.operationCount(), 1U);

	auto conditionalOut = graph.addValue(matrix);
	ASSERT_TRUE(conditionalOut.isOk());
	oa::OpContract conditional = oa::detail::opRegistry::FnMatrix::add;
	conditional.name = "ConditionalAdd";
	conditional.hash = 0xa5e2b4ff2b15dc61ULL;
	conditional.controlFlow = oa::OpControlFlow::Conditional;
	const oa::U32 conditionalOutputs[] = {conditionalOut.getValue()};
	const oa::U32 dependencies[] = {accepted.getValue()};
	auto controlled = graph.addOp(
		conditional, inputs, conditionalOutputs, dependencies);
	ASSERT_TRUE(controlled.isOk());
	ASSERT_EQ(graph.operations()[1].controlDependencies.size(), 1U);
	EXPECT_EQ(graph.operations()[1].controlDependencies[0], accepted.getValue());
	EXPECT_EQ(graph.operations()[1].controlFlow,
		oa::OpControlFlow::Conditional);
	ASSERT_TRUE(graph.validate().isOk());
}

TEST(ExecutableGraph, SemanticGraphAcceptsHomogeneousVariadicTails) {
	oa::SemanticGraph graph;
	oa::SemanticValueDesc matrix;
	matrix.shape = {2, 3};
	const auto a = graph.addValue(matrix);
	const auto b = graph.addValue(matrix);
	const auto c = graph.addValue(matrix);
	const auto concatenated = graph.addValue(matrix);
	const auto first = graph.addValue(matrix);
	const auto second = graph.addValue(matrix);
	const auto third = graph.addValue(matrix);
	ASSERT_TRUE(a.isOk());
	ASSERT_TRUE(b.isOk());
	ASSERT_TRUE(c.isOk());
	ASSERT_TRUE(concatenated.isOk());
	ASSERT_TRUE(first.isOk());
	ASSERT_TRUE(second.isOk());
	ASSERT_TRUE(third.isOk());

	const oa::OpAttribute attributes[] = {
		oa::OpAttribute::fromSignedInteger("dim", 1),
	};
	const oa::U32 concatOutputs[] = {concatenated.getValue()};
	const auto rejectedConcat = graph.addOp(
		oa::detail::opRegistry::FnMatrix::concat,
		oa::Span<const oa::U32>{}, concatOutputs, {}, attributes);
	ASSERT_FALSE(rejectedConcat.isOk());
	EXPECT_EQ(
		rejectedConcat.getStatus().getCode(), oa::StatusCode::InvalidArgument);

	const oa::U32 concatInputs[] = {
		a.getValue(), b.getValue(), c.getValue()};
	const auto concat = graph.addOp(
		oa::detail::opRegistry::FnMatrix::concat,
		concatInputs, concatOutputs, {}, attributes);
	ASSERT_TRUE(concat.isOk()) << concat.getStatus().getMessage();

	const oa::U32 splitInputs[] = {concatenated.getValue()};
	const oa::U32 splitOutputs[] = {
		first.getValue(), second.getValue(), third.getValue()};
	const auto split = graph.addOp(
		oa::detail::opRegistry::FnMatrix::split,
		splitInputs, splitOutputs, {}, attributes);
	ASSERT_TRUE(split.isOk()) << split.getStatus().getMessage();
	ASSERT_TRUE(graph.validate().isOk());
	ASSERT_EQ(graph.operations().size(), 2U);
	EXPECT_EQ(graph.operations()[0].inputs.size(), 3U);
	EXPECT_EQ(graph.operations()[0].outputs.size(), 1U);
	EXPECT_EQ(graph.operations()[1].inputs.size(), 1U);
	EXPECT_EQ(graph.operations()[1].outputs.size(), 3U);
}

TEST(ExecutableGraph, SemanticGraphPreservesAbsentOptionalInputs) {
	oa::SemanticGraph graph;
	oa::SemanticValueDesc matrix;
	matrix.shape = {2, 3};
	matrix.external = true;
	auto x = graph.addValue(matrix);
	auto weight = graph.addValue(matrix);
	matrix.external = false;
	matrix.shape = {2, 4};
	auto out = graph.addValue(matrix);
	ASSERT_TRUE(x.isOk());
	ASSERT_TRUE(weight.isOk());
	ASSERT_TRUE(out.isOk());

	const oa::U32 inputs[] = {
		x.getValue(),
		weight.getValue(),
		oa::invalidSemanticValueId,
	};
	const oa::U32 outputs[] = {out.getValue()};
	const auto operation = graph.addOp(
		oa::detail::opRegistry::FnMatrix::linear, inputs, outputs);
	ASSERT_TRUE(operation.isOk()) << operation.getStatus().getMessage();
	ASSERT_TRUE(graph.validate().isOk());
	ASSERT_EQ(graph.operationCount(), 1U);
	const auto& recorded = graph.operations()[0];
	EXPECT_EQ(recorded.optionalInputMask, 0x04U);
	ASSERT_EQ(recorded.inputs.size(), 3U);
	EXPECT_EQ(recorded.inputs[2], oa::invalidSemanticValueId);
	ASSERT_EQ(recorded.accesses.size(), 3U);

	const auto report = graph.debugReportJson("optional-input").stdStr();
	EXPECT_NE(report.find("\"inputs\": [0, 1, null]"), std::string::npos);

	auto required = oa::detail::opRegistry::FnMatrix::linear;
	required.name = "RequiredBiasLinear";
	required.hash = 0x2fbc22401b70c693ULL;
	required.optionalInputMask = 0U;
	matrix.shape = {2, 4};
	auto rejectedOut = graph.addValue(matrix);
	ASSERT_TRUE(rejectedOut.isOk());
	const oa::U32 rejectedOutputs[] = {rejectedOut.getValue()};
	const auto rejected = graph.addOp(
		required, inputs, rejectedOutputs);
	ASSERT_FALSE(rejected.isOk());
	EXPECT_EQ(rejected.getStatus().getCode(), oa::StatusCode::InvalidArgument);
}

TEST(ExecutableGraph, SemanticGraphRecordsOutputlessMutation) {
	oa::SemanticGraph graph;
	oa::SemanticValueDesc selection;
	selection.shape = {4, 8};
	selection.external = true;
	oa::SemanticValueDesc bias;
	bias.shape = {8};
	bias.external = true;
	const auto selectionValue = graph.addValue(selection);
	const auto biasValue = graph.addValue(bias);
	ASSERT_TRUE(selectionValue.isOk());
	ASSERT_TRUE(biasValue.isOk());

	const oa::U32 inputs[] = {
		selectionValue.getValue(),
		biasValue.getValue(),
	};
	const oa::OpAttribute attributes[] = {
		oa::OpAttribute::fromSignedInteger("expertsPerToken", 2),
		oa::OpAttribute::fromFloat("gamma", 0.1),
	};
	const auto operation = graph.addOp(
		oa::detail::opRegistry::FnMatrix::moeRoutingBiasUpdate,
		inputs, {}, {}, attributes);
	ASSERT_TRUE(operation.isOk()) << operation.getStatus().getMessage();
	ASSERT_TRUE(graph.validate().isOk());
	ASSERT_EQ(graph.operationCount(), 1U);
	const auto& recorded = graph.operations()[0];
	EXPECT_TRUE(recorded.outputs.empty());
	ASSERT_EQ(recorded.mutatedInputs.size(), 1U);
	EXPECT_EQ(recorded.mutatedInputs[0], biasValue.getValue());
	ASSERT_EQ(recorded.accesses.size(), 2U);
	EXPECT_EQ(recorded.accesses[1].value, biasValue.getValue());
	EXPECT_EQ(recorded.accesses[1].mode, oa::SemanticAccessMode::ReadWrite);
}

TEST(ExecutableGraph, SemanticGraphOwnsTypedOperationAttributes) {
	oa::SemanticGraph graph;
	oa::SemanticValueDesc input;
	input.shape = {2, 3};
	input.external = true;
	oa::SemanticValueDesc output = input;
	output.external = false;
	output.isVirtual = true;
	const auto inputValue = graph.addValue(input);
	const auto outputValue = graph.addValue(output);
	ASSERT_TRUE(inputValue.isOk());
	ASSERT_TRUE(outputValue.isOk());
	const oa::U32 inputs[] = {inputValue.getValue()};
	const oa::U32 outputs[] = {outputValue.getValue()};
	const oa::OpAttribute attributes[] = {
		oa::OpAttribute::fromBoolean("enabled", true),
		oa::OpAttribute::fromSignedInteger("axis", -2),
		oa::OpAttribute::fromUnsignedInteger("seed", 17U),
		oa::OpAttribute::fromFloat("epsilon", 1.0e-5),
		oa::OpAttribute::fromString("label", oa::String("pilot")),
		oa::OpAttribute::fromShape("target", oa::MatrixShape{2, 3}),
		oa::OpAttribute::fromEnum(
			"mode", oa::String("oa::InterpolationMode::Bilinear")),
	};
	const oa::OpContract contract{
		.name = "TypedAttributeTest",
		.hash = 0x43e084f8994e12d1ULL,
		.inputKinds = static_cast<oa::U32>(oa::OpValueKind::Matrix),
		.outputKinds = static_cast<oa::U32>(oa::OpValueKind::Matrix),
		.inputCount = 1U,
		.outputCount = 1U,
		.attributeCount = 7U,
		.attributeSignatureHash = oa::opAttributeSignatureHash(attributes),
		.shapeRule = oa::OpShapeRule::MatchInput,
		.dtypeRule = oa::OpDtypeRule::MatchInput,
		.differentiation = oa::OpDifferentiation::None,
		.lowering = oa::OpLowering::Dispatch,
		.effects = oa::OpEffect::ReadInputs
			| oa::OpEffect::WriteOutputs,
	};
	const oa::OpAttribute wrongAttributes[] = {
		oa::OpAttribute::fromBoolean("wrongName", true),
		oa::OpAttribute::fromSignedInteger("axis", -2),
		oa::OpAttribute::fromUnsignedInteger("seed", 17U),
		oa::OpAttribute::fromFloat("epsilon", 1.0e-5),
		oa::OpAttribute::fromString("label", oa::String("pilot")),
		oa::OpAttribute::fromShape("target", oa::MatrixShape{2, 3}),
		oa::OpAttribute::fromEnum(
			"mode", oa::String("oa::InterpolationMode::Bilinear")),
	};
	const auto rejected = graph.addOp(
		contract, inputs, outputs, {}, wrongAttributes);
	ASSERT_FALSE(rejected.isOk());
	EXPECT_EQ(rejected.getStatus().getCode(), oa::StatusCode::InvalidArgument);

	const auto operation = graph.addOp(
		contract, inputs, outputs, {}, attributes);
	ASSERT_TRUE(operation.isOk()) << operation.getStatus().getMessage();
	ASSERT_TRUE(graph.validate().isOk());
	ASSERT_EQ(graph.operations()[0].attributes.size(), 7U);
	EXPECT_EQ(graph.operations()[0].attributes[5].shape,
		(oa::MatrixShape{2, 3}));
	const auto report = graph.debugReportJson("typed-attributes").stdStr();
	EXPECT_NE(report.find("\"kind\": \"boolean\", \"value\": true"),
		std::string::npos);
	EXPECT_NE(report.find("\"kind\": \"signed_integer\", \"value\": -2"),
		std::string::npos);
	EXPECT_NE(report.find("\"kind\": \"shape\", \"value\": [2, 3]"),
		std::string::npos);
	EXPECT_NE(report.find(
		"\"kind\": \"enum\", \"value\": "
		"\"oa::InterpolationMode::Bilinear\""), std::string::npos);
}

TEST(ExecutableGraph, SemanticLoweringAnalysisClassifiesDecompositionAndDebt) {
	oa::SemanticGraph semantic;
	oa::SemanticValueDesc matrix;
	matrix.shape = {4};
	auto valueA = semantic.addValue(matrix);
	auto valueB = semantic.addValue(matrix);
	auto sum = semantic.addValue(matrix);
	auto output = semantic.addValue(matrix);
	ASSERT_TRUE(valueA.isOk());
	ASSERT_TRUE(valueB.isOk());
	ASSERT_TRUE(sum.isOk());
	ASSERT_TRUE(output.isOk());
	const oa::U32 firstInputs[] = {
		valueA.getValue(), valueB.getValue(),
	};
	const oa::U32 firstOutputs[] = {sum.getValue()};
	auto firstOperation = semantic.addOp(
		oa::detail::opRegistry::FnMatrix::add, firstInputs, firstOutputs);
	ASSERT_TRUE(firstOperation.isOk());
	const oa::U32 secondInputs[] = {
		sum.getValue(), valueB.getValue(),
	};
	const oa::U32 secondOutputs[] = {output.getValue()};
	auto secondOperation = semantic.addOp(
		oa::detail::opRegistry::FnMatrix::add, secondInputs, secondOutputs);
	ASSERT_TRUE(secondOperation.isOk());

	oa::ExecutableGraph executable;
	oa::ComputeDispatchDesc direct;
	direct.kernel = "AddPart";
	direct.operation = oa::detail::opRegistry::FnMatrix::add.name;
	direct.opContractHash = oa::detail::opRegistry::FnMatrix::add.hash;
	const oa::U32 firstProvenance[] = {
		firstOperation.getValue(),
	};
	direct.semanticOps = firstProvenance;
	executable.add(direct);
	executable.add(direct);
	const oa::U32 secondProvenance[] = {
		secondOperation.getValue(),
	};
	direct.semanticOps = secondProvenance;
	executable.add(direct);
	oa::ComputeDispatchDesc compatibility;
	compatibility.kernel = "CompatibilityOnly";
	executable.add(compatibility);

	auto analyzed = oa::analyzeSemanticLowering(semantic, executable);
	ASSERT_TRUE(analyzed.isOk()) << analyzed.getStatus().getMessage();
	const auto& result = analyzed.getValue();
	EXPECT_EQ(result.operationCount(), 2U);
	EXPECT_EQ(result.schemaOwnedNodeCount(), 3U);
	EXPECT_EQ(result.compatibilityNodeCount(), 1U);
	EXPECT_EQ(result.directOpCount(), 1U);
	EXPECT_EQ(result.decomposedOpCount(), 1U);
	EXPECT_EQ(result.fusedOpCount(), 0U);
	EXPECT_EQ(result.fusedNodeCount(), 0U);
	EXPECT_EQ(result.maximumNodesPerOp(), 2U);
	EXPECT_EQ(result.maximumOpsPerNode(), 1U);
	EXPECT_EQ(result.executableNodeCount(firstOperation.getValue()), 2U);
	EXPECT_EQ(result.executableNodeCount(secondOperation.getValue()), 1U);
	EXPECT_EQ(result.executableNodeCount(7U), 0U);
}

TEST(ExecutableGraph, SemanticLoweringAnalysisTracksFusionProvenance) {
	oa::SemanticGraph semantic;
	oa::SemanticValueDesc matrix;
	matrix.shape = {4};
	auto valueA = semantic.addValue(matrix);
	auto valueB = semantic.addValue(matrix);
	auto intermediate = semantic.addValue(matrix);
	auto output = semantic.addValue(matrix);
	ASSERT_TRUE(valueA.isOk());
	ASSERT_TRUE(valueB.isOk());
	ASSERT_TRUE(intermediate.isOk());
	ASSERT_TRUE(output.isOk());
	const oa::U32 firstInputs[] = {
		valueA.getValue(), valueB.getValue(),
	};
	const oa::U32 firstOutputs[] = {intermediate.getValue()};
	auto firstOperation = semantic.addOp(
		oa::detail::opRegistry::FnMatrix::add, firstInputs, firstOutputs);
	ASSERT_TRUE(firstOperation.isOk());
	const oa::U32 secondInputs[] = {
		intermediate.getValue(), valueB.getValue(),
	};
	const oa::U32 secondOutputs[] = {output.getValue()};
	auto secondOperation = semantic.addOp(
		oa::detail::opRegistry::FnMatrix::add, secondInputs, secondOutputs);
	ASSERT_TRUE(secondOperation.isOk());

	const oa::U32 fusedProvenance[] = {
		firstOperation.getValue(), secondOperation.getValue(),
	};
	oa::ComputeDispatchDesc fused;
	fused.kernel = "FusedAddChain";
	fused.operation = "FusedAddChain";
	fused.semanticOps = fusedProvenance;
	oa::ExecutableGraph executable;
	executable.add(fused);

	auto analyzed = oa::analyzeSemanticLowering(semantic, executable);
	ASSERT_TRUE(analyzed.isOk()) << analyzed.getStatus().getMessage();
	const auto& result = analyzed.getValue();
	EXPECT_EQ(result.operationCount(), 2U);
	EXPECT_EQ(result.schemaOwnedNodeCount(), 1U);
	EXPECT_EQ(result.compatibilityNodeCount(), 0U);
	EXPECT_EQ(result.directOpCount(), 0U);
	EXPECT_EQ(result.decomposedOpCount(), 0U);
	EXPECT_EQ(result.fusedOpCount(), 2U);
	EXPECT_EQ(result.fusedNodeCount(), 1U);
	EXPECT_EQ(result.maximumNodesPerOp(), 1U);
	EXPECT_EQ(result.maximumOpsPerNode(), 2U);
	EXPECT_EQ(result.executableNodeCount(firstOperation.getValue()), 1U);
	EXPECT_EQ(result.executableNodeCount(secondOperation.getValue()), 1U);
}

TEST(ExecutableGraph, MultiAddExecutesSchemaOwnedFusionAndRemainder) {
	auto* engine = testEnginePtr();
	ASSERT_NE(engine, nullptr);
	auto& ctx = oa::ExecutionSession::getActive();
	ctx.clear();
	constexpr oa::U32 PairCount = 6U;
	constexpr oa::U32 elementCount = 64U;
	oa::Vec<oa::Matrix> destinations;
	oa::Vec<oa::Matrix> sources;
	for (oa::U32 pair = 0U; pair < PairCount; ++pair) {
		destinations.pushBack(oa::FnMatrix::empty(
			{elementCount}, oa::ScalarType::Float32,
			oa::MemoryPlacement::HostUpload));
		sources.pushBack(oa::FnMatrix::empty(
			{elementCount}, oa::ScalarType::Float32,
			oa::MemoryPlacement::HostUpload));
		ASSERT_TRUE(destinations.back().hasStorage());
		ASSERT_TRUE(sources.back().hasStorage());
		for (oa::U32 element = 0U; element < elementCount; ++element) {
			destinations.back().dataAs<oa::F32>()[element] =
				static_cast<oa::F32>(pair + 1U);
			sources.back().dataAs<oa::F32>()[element] =
				static_cast<oa::F32>((pair + 1U) * 10U);
		}
		ASSERT_TRUE(oa::EngineAllocatorAccess::get(*engine).flushHostBuffer(
			oa::MatrixAccess::descriptor(destinations.back()), 0,
			oa::MatrixAccess::descriptor(destinations.back()).size));
		ASSERT_TRUE(oa::EngineAllocatorAccess::get(*engine).flushHostBuffer(
			oa::MatrixAccess::descriptor(sources.back()), 0,
			oa::MatrixAccess::descriptor(sources.back()).size));
	}
	ctx.clear();

	oa::FnMatrix::multiAdd(
		destinations.span(),
		oa::Span<const oa::Matrix>(sources.data(), sources.size()));
	ASSERT_EQ(ctx.nodeCount(), 3U);
	ASSERT_NE(ctx.semanticGraph(), nullptr);
	ASSERT_EQ(ctx.semanticGraph()->operationCount(), PairCount);
	const auto nodes = ctx.graph()->nodes();
	ASSERT_EQ(nodes[0].semanticOps.size(), 4U);
	EXPECT_EQ(nodes[0].operation, "MultiMatrixAdd");
	EXPECT_EQ(nodes[1].semanticOps.size(), 1U);
	EXPECT_EQ(nodes[2].semanticOps.size(), 1U);
	const auto analyzed = oa::analyzeSemanticLowering(
		*ctx.semanticGraph(), *ctx.graph());
	ASSERT_TRUE(analyzed.isOk()) << analyzed.getStatus().getMessage();
	EXPECT_EQ(analyzed.getValue().fusedOpCount(), 4U);
	EXPECT_EQ(analyzed.getValue().fusedNodeCount(), 1U);
	EXPECT_EQ(analyzed.getValue().directOpCount(), 2U);
	EXPECT_EQ(analyzed.getValue().maximumOpsPerNode(), 4U);

	ASSERT_TRUE(testSubmitAndWait(ctx).isOk());
	for (oa::U32 pair = 0U; pair < PairCount; ++pair) {
		ASSERT_TRUE(oa::EngineAllocatorAccess::get(*engine).invalidateHostBuffer(
			oa::MatrixAccess::descriptor(destinations[pair]), 0,
			oa::MatrixAccess::descriptor(destinations[pair]).size));
		for (oa::U32 element = 0U; element < elementCount; ++element) {
			EXPECT_FLOAT_EQ(destinations[pair].dataAs<const oa::F32>()[element],
				static_cast<oa::F32>((pair + 1U) * 11U));
		}
	}
	ctx.clear();
}

TEST(ExecutableGraph, BiasAddExecutesSchemaOwnedDecompositionAndBackward) {
	auto* engine = testEnginePtr();
	ASSERT_NE(engine, nullptr);
	auto& ctx = oa::ExecutionSession::getActive();
	ctx.clear();
	auto input = oa::FnMatrix::empty(
		{2, 4}, oa::ScalarType::Float32, oa::MemoryPlacement::HostUpload);
	auto bias = oa::FnMatrix::empty(
		{4}, oa::ScalarType::Float32, oa::MemoryPlacement::HostUpload);
	ASSERT_TRUE(input.hasStorage());
	ASSERT_TRUE(bias.hasStorage());
	for (oa::U32 element = 0U; element < 8U; ++element) {
		input.dataAs<oa::F32>()[element] = static_cast<oa::F32>(element + 1U);
	}
	for (oa::U32 element = 0U; element < 4U; ++element) {
		bias.dataAs<oa::F32>()[element] =
			static_cast<oa::F32>((element + 1U) * 10U);
	}
	ASSERT_TRUE(oa::EngineAllocatorAccess::get(*engine).flushHostBuffer(
		oa::MatrixAccess::descriptor(input), 0,
		oa::MatrixAccess::descriptor(input).size));
	ASSERT_TRUE(oa::EngineAllocatorAccess::get(*engine).flushHostBuffer(
		oa::MatrixAccess::descriptor(bias), 0,
		oa::MatrixAccess::descriptor(bias).size));
	input.setRequiresGrad(true);
	bias.setRequiresGrad(true);
	ctx.clear();

	oa::GradientTape tape;
	auto output = oa::FnMatrix::biasAdd(input, bias);
	ASSERT_EQ(ctx.nodeCount(), 2U);
	ASSERT_NE(ctx.semanticGraph(), nullptr);
	ASSERT_EQ(ctx.semanticGraph()->operationCount(), 1U);
	const auto nodes = ctx.graph()->nodes();
	ASSERT_EQ(nodes.size(), 2U);
	for (const auto& node : nodes) {
		EXPECT_EQ(node.operation, oa::detail::opRegistry::FnMatrix::biasAdd.name);
		EXPECT_EQ(node.opContractHash,
			oa::detail::opRegistry::FnMatrix::biasAdd.hash);
		ASSERT_EQ(node.semanticOps.size(), 1U);
		EXPECT_EQ(node.semanticOps[0], 0U);
	}
	EXPECT_EQ(nodes[0].shader, "Copy");
	EXPECT_EQ(nodes[1].shader, "BiasAdd");
	const auto analyzed = oa::analyzeSemanticLowering(
		*ctx.semanticGraph(), *ctx.graph());
	ASSERT_TRUE(analyzed.isOk()) << analyzed.getStatus().getMessage();
	EXPECT_EQ(analyzed.getValue().directOpCount(), 0U);
	EXPECT_EQ(analyzed.getValue().decomposedOpCount(), 1U);
	EXPECT_EQ(analyzed.getValue().fusedOpCount(), 0U);
	EXPECT_EQ(analyzed.getValue().maximumNodesPerOp(), 2U);
	EXPECT_EQ(analyzed.getValue().executableNodeCount(0U), 2U);

	auto loss = oa::FnMatrix::sum(output);
	tape.backward(loss);
	ASSERT_TRUE(testSubmitAndWait(ctx).isOk());
	oa::F32 outputValues[8]{};
	ASSERT_TRUE(oa::FnMatrix::copyToHost(
		output, outputValues, sizeof(outputValues)).isOk());
	for (oa::U32 element = 0U; element < 8U; ++element) {
		EXPECT_FLOAT_EQ(outputValues[element],
			static_cast<oa::F32>(element + 1U) +
			static_cast<oa::F32>(((element % 4U) + 1U) * 10U));
	}
	oa::F32 inputGrad[8]{};
	oa::F32 biasGrad[4]{};
	ASSERT_TRUE(oa::FnMatrix::copyToHost(
		input.gradMatrix(), inputGrad, sizeof(inputGrad)).isOk());
	ASSERT_TRUE(oa::FnMatrix::copyToHost(
		bias.gradMatrix(), biasGrad, sizeof(biasGrad)).isOk());
	for (const auto gradient : inputGrad) EXPECT_FLOAT_EQ(gradient, 1.0F);
	for (const auto gradient : biasGrad) EXPECT_FLOAT_EQ(gradient, 2.0F);
	ctx.clear();
}

TEST(ExecutableGraph, ExecutionRejectsIncompleteOrMismatchedSemanticLowering) {
	auto& ctx = oa::ExecutionSession::getActive();
	oa::ExecutionSession::RecordingScope scope(ctx);
	ctx.clear();
	auto a = oa::FnMatrix::empty({4});
	auto b = oa::FnMatrix::empty({4});
	auto out = oa::FnMatrix::empty({4});

	const auto semantic = ctx.recordOp(
		oa::detail::opRegistry::FnMatrix::add, {&a, &b}, {&out});
	ASSERT_TRUE(semantic.isOk());
	auto status = testSubmitAndWait(ctx);
	EXPECT_EQ(status.getCode(), oa::StatusCode::FailedPrecondition);
	EXPECT_EQ(ctx.nodeCount(), 0U);
	EXPECT_EQ(ctx.semanticGraph()->operationCount(), 0U);

	const auto semanticForSubmit = ctx.recordOp(
		oa::detail::opRegistry::FnMatrix::add, {&a, &b}, {&out});
	ASSERT_TRUE(semanticForSubmit.isOk());
	const auto incompleteSubmit = ctx.submit();
	EXPECT_FALSE(incompleteSubmit.isOk());
	EXPECT_EQ(incompleteSubmit.getStatus().getCode(),
		oa::StatusCode::FailedPrecondition);
	EXPECT_EQ(ctx.nodeCount(), 0U);
	EXPECT_EQ(ctx.semanticGraph()->operationCount(), 0U);

	oa::ComputeDispatchDesc dangling;
	dangling.kernel = "TestKernelThatDoesNotExist";
	dangling.operation = oa::detail::opRegistry::FnMatrix::add.name;
	dangling.opContractHash = oa::detail::opRegistry::FnMatrix::add.hash;
	const oa::U32 danglingProvenance[] = {7U};
	dangling.semanticOps = danglingProvenance;
	ASSERT_TRUE(ctx.record( dangling).isOk());
	status = testSubmitAndWait(ctx);
	EXPECT_EQ(status.getCode(), oa::StatusCode::OutOfRange);

	const auto semanticAgain = ctx.recordOp(
		oa::detail::opRegistry::FnMatrix::add, {&a, &b}, {&out});
	ASSERT_TRUE(semanticAgain.isOk());
	oa::ComputeDispatchDesc mismatched;
	mismatched.kernel = "TestKernelThatDoesNotExist";
	mismatched.operation = "WrongSemanticIdentity";
	mismatched.opContractHash = oa::detail::opRegistry::FnMatrix::add.hash;
	const oa::U32 mismatchedProvenance[] = {
		semanticAgain.getValue(),
	};
	mismatched.semanticOps = mismatchedProvenance;
	ASSERT_TRUE(ctx.record( mismatched).isOk());
	status = testSubmitAndWait(ctx);
	EXPECT_EQ(status.getCode(), oa::StatusCode::FailedPrecondition);
	EXPECT_EQ(ctx.nodeCount(), 0U);
	EXPECT_EQ(ctx.semanticGraph()->operationCount(), 0U);

	const oa::U32 duplicateProvenance[] = {0U, 0U};
	oa::ComputeDispatchDesc duplicate;
	duplicate.kernel = "TestKernelThatDoesNotExist";
	duplicate.operation = "DuplicateSemanticProvenance";
	duplicate.semanticOps = duplicateProvenance;
	status = ctx.record( duplicate);
	EXPECT_EQ(status.getCode(), oa::StatusCode::AlreadyExists);
	EXPECT_EQ(ctx.nodeCount(), 0U);
}

TEST(ExecutableGraph, SchemaContractsDriveValidationAndShapeInference) {
	auto& ctx = oa::ExecutionSession::getActive();
	ctx.clear();
	const auto broadcastA = oa::FnMatrix::empty({2, 3, 4}, oa::ScalarType::Float32);
	const auto broadcastB = oa::FnMatrix::empty({1, 4}, oa::ScalarType::Float32);
	auto broadcast = oa::inferBinaryOpShape(
		oa::detail::opRegistry::FnMatrix::add, broadcastA, broadcastB);
	ASSERT_TRUE(broadcast.isOk());
	EXPECT_EQ(broadcast.getValue(), (oa::MatrixShape{2, 3, 4}));

	const auto wrongDtype = oa::FnMatrix::empty({1, 4}, oa::ScalarType::UInt32);
	const auto dtypeStatus = oa::validateBinaryOp(
		oa::detail::opRegistry::FnMatrix::add, broadcastA, wrongDtype);
	EXPECT_EQ(dtypeStatus.getCode(), oa::StatusCode::DtypeMismatch);

	const auto weight = oa::FnMatrix::empty({5, 4}, oa::ScalarType::Float32);
	auto matmul = oa::inferBinaryOpShape(
		oa::detail::opRegistry::FnMatrix::matMulNt, broadcastA, weight);
	ASSERT_TRUE(matmul.isOk());
	EXPECT_EQ(matmul.getValue(), (oa::MatrixShape{2, 3, 5}));

	const auto wrongWeight = oa::FnMatrix::empty({5, 6}, oa::ScalarType::Float32);
	const auto shapeStatus = oa::validateBinaryOp(
		oa::detail::opRegistry::FnMatrix::matMulNt, broadcastA, wrongWeight);
	EXPECT_EQ(shapeStatus.getCode(), oa::StatusCode::ShapeMismatch);
	ctx.clear();
}

TEST(ExecutableGraph, MetadataViewsPreserveDataflowWithoutExecutableNodes) {
	auto& ctx = oa::ExecutionSession::getActive();
	ctx.clear();
	const auto source = oa::FnMatrix::empty(
		{2, 3}, oa::ScalarType::Float32);
	ctx.clear();

	const auto view = oa::FnMatrix::reshape(source, {3, 2});
	ASSERT_TRUE(view.hasStorage());
	ASSERT_NE(ctx.semanticGraph(), nullptr);
	EXPECT_EQ(ctx.nodeCount(), 0U);
	EXPECT_EQ(ctx.semanticGraph()->operationCount(), 0U);
	EXPECT_EQ(ctx.semanticGraph()->valueCount(), 2U);
	EXPECT_EQ(ctx.semanticGraph()->viewCount(), 1U);
	ASSERT_TRUE(ctx.semanticGraph()->validate().isOk());
	const auto values = ctx.semanticGraph()->values();
	EXPECT_TRUE(values[0].external);
	EXPECT_FALSE(values[1].external);
	EXPECT_EQ(values[1].viewSource, values[0].id);
	EXPECT_EQ(values[1].shape, (oa::MatrixShape{3, 2}));
	EXPECT_EQ(values[1].strides[0], 2);
	EXPECT_EQ(values[1].strides[1], 1);

	const auto scaled = oa::FnMatrix::scale(view, 2.0F);
	(void)scaled;
	EXPECT_EQ(ctx.nodeCount(), 1U);
	ASSERT_EQ(ctx.semanticGraph()->operationCount(), 1U);
	ASSERT_EQ(ctx.semanticGraph()->valueCount(), 3U);
	const auto& operation = ctx.semanticGraph()->operations()[0];
	ASSERT_EQ(operation.inputs.size(), 1U);
	EXPECT_EQ(operation.inputs[0], values[1].id);
	ASSERT_TRUE(ctx.semanticGraph()->validate().isOk());
	ASSERT_TRUE(oa::validateSemanticLowering(
		*ctx.semanticGraph(), *ctx.graph()).isOk());

	ctx.clear();
	const oa::I32 dimensions[] = {1, 0};
	const auto permuted = source.permute(dimensions);
	ASSERT_TRUE(permuted.hasStorage());
	EXPECT_EQ(ctx.nodeCount(), 0U);
	ASSERT_EQ(ctx.semanticGraph()->valueCount(), 2U);
	ASSERT_EQ(ctx.semanticGraph()->viewCount(), 1U);
	const auto permutedValues = ctx.semanticGraph()->values();
	EXPECT_EQ(permutedValues[1].shape, (oa::MatrixShape{3, 2}));
	EXPECT_EQ(permutedValues[1].strides[0], 1);
	EXPECT_EQ(permutedValues[1].strides[1], 3);
	ASSERT_TRUE(ctx.semanticGraph()->validate().isOk());
	ctx.clear();
}

TEST(ExecutableGraph, PilotFunctionsRecordSemanticContracts) {
	auto& ctx = oa::ExecutionSession::getActive();
	ctx.clear();
	const auto a = oa::FnMatrix::empty({2, 3}, oa::ScalarType::Float32);
	const auto b = oa::FnMatrix::empty({2, 3}, oa::ScalarType::Float32);
	ctx.clear();
	const auto sum = oa::FnMatrix::add(a, b);
	(void)sum;
	ASSERT_EQ(ctx.nodeCount(), 1U);
	auto nodes = ctx.graph()->nodes();
	EXPECT_EQ(nodes[0].operation, oa::detail::opRegistry::FnMatrix::add.name);
	ASSERT_EQ(nodes[0].semanticOps.size(), 1U);
	EXPECT_EQ(nodes[0].semanticOps[0], 0U);
	EXPECT_EQ(nodes[0].opContractHash, oa::detail::opRegistry::FnMatrix::add.hash);
	EXPECT_EQ(nodes[0].problemContractHash, 0U);
	ASSERT_NE(ctx.semanticGraph(), nullptr);
	ASSERT_EQ(ctx.semanticGraph()->operationCount(), 1U);
	EXPECT_EQ(ctx.semanticGraph()->operations()[0].name,
		oa::detail::opRegistry::FnMatrix::add.name);
	EXPECT_EQ(ctx.semanticGraph()->valueCount(), 3U);

	ctx.clear();
	constexpr oa::F32 ScaleValue = 1.25F;
	const auto scaled = oa::FnMatrix::scale(a, ScaleValue);
	(void)scaled;
	ASSERT_EQ(ctx.nodeCount(), 1U);
	nodes = ctx.graph()->nodes();
	EXPECT_EQ(nodes[0].operation, oa::detail::opRegistry::FnMatrix::scale.name);
	ASSERT_EQ(nodes[0].semanticOps.size(), 1U);
	EXPECT_EQ(nodes[0].opContractHash, oa::detail::opRegistry::FnMatrix::scale.hash);
	ASSERT_EQ(ctx.semanticGraph()->operationCount(), 1U);
	const auto& scaleOperation = ctx.semanticGraph()->operations()[0];
	ASSERT_EQ(scaleOperation.attributes.size(), 1U);
	EXPECT_EQ(scaleOperation.attributes[0].name, "scalar");
	EXPECT_EQ(scaleOperation.attributes[0].kind,
		oa::OpAttributeKind::Float);
	EXPECT_DOUBLE_EQ(scaleOperation.attributes[0].floatVal, ScaleValue);
	const auto scaleReport = ctx.semanticGraph()->debugReportJson("scale");
	EXPECT_NE(scaleReport.stdStr().find(
		"\"attributes\": [{\"name\": \"scalar\", \"kind\": \"float\", "
		"\"value\": 1.25}]"), std::string::npos);

	ctx.clear();
	const auto weight = oa::FnMatrix::empty({4, 3}, oa::ScalarType::Float32);
	ctx.clear();
	const auto product = oa::FnMatrix::matMulNt(
		a, weight, oa::MatMulPrecision::Fp32);
	(void)product;
	ASSERT_EQ(ctx.nodeCount(), 1U);
	nodes = ctx.graph()->nodes();
	EXPECT_EQ(nodes[0].operation, oa::detail::opRegistry::FnMatrix::matMulNt.name);
	ASSERT_EQ(nodes[0].semanticOps.size(), 1U);
	EXPECT_EQ(nodes[0].semanticOps[0], 0U);
	EXPECT_EQ(nodes[0].opContractHash, oa::detail::opRegistry::FnMatrix::matMulNt.hash);
	EXPECT_NE(nodes[0].problemContractHash, 0U);
	ASSERT_EQ(ctx.semanticGraph()->operationCount(), 1U);
	EXPECT_EQ(ctx.semanticGraph()->operations()[0].name,
		oa::detail::opRegistry::FnMatrix::matMulNt.name);
	EXPECT_EQ(ctx.semanticGraph()->operations()[0].lowering,
		oa::OpLowering::Gemm);
	EXPECT_EQ(ctx.semanticGraph()->valueCount(), 3U);

	ctx.clear();
	const auto copied = oa::FnMatrix::copy(a);
	(void)copied;
	ASSERT_EQ(ctx.nodeCount(), 1U);
	nodes = ctx.graph()->nodes();
	EXPECT_EQ(nodes[0].operation, oa::detail::opRegistry::FnMatrix::copy.name);
	ASSERT_EQ(nodes[0].semanticOps.size(), 1U);
	EXPECT_EQ(nodes[0].semanticOps[0], 0U);
	EXPECT_EQ(nodes[0].opContractHash,
		oa::detail::opRegistry::FnMatrix::copy.hash);
	ASSERT_EQ(ctx.semanticGraph()->operationCount(), 1U);
	EXPECT_EQ(ctx.semanticGraph()->operations()[0].name,
		oa::detail::opRegistry::FnMatrix::copy.name);
	EXPECT_EQ(ctx.semanticGraph()->valueCount(), 2U);

	ctx.clear();
	const auto sliced = oa::FnMatrix::slice(a, 1, 1, 3);
	(void)sliced;
	ASSERT_EQ(ctx.nodeCount(), 1U);
	nodes = ctx.graph()->nodes();
	EXPECT_EQ(nodes[0].operation, oa::detail::opRegistry::FnMatrix::slice.name);
	ASSERT_EQ(nodes[0].semanticOps.size(), 1U);
	EXPECT_EQ(nodes[0].opContractHash,
		oa::detail::opRegistry::FnMatrix::slice.hash);
	ASSERT_EQ(ctx.semanticGraph()->operationCount(), 1U);
	const auto& sliceOperation = ctx.semanticGraph()->operations()[0];
	EXPECT_EQ(sliceOperation.name,
		oa::detail::opRegistry::FnMatrix::slice.name);
	ASSERT_EQ(sliceOperation.attributes.size(), 3U);
	EXPECT_EQ(sliceOperation.attributes[0].signedInteger, 1);
	EXPECT_EQ(sliceOperation.attributes[1].signedInteger, 1);
	EXPECT_EQ(sliceOperation.attributes[2].signedInteger, 3);

	ctx.clear();
	const auto transposed = oa::FnMatrix::transpose(a, 0, 1);
	(void)transposed;
	ASSERT_EQ(ctx.nodeCount(), 1U);
	nodes = ctx.graph()->nodes();
	EXPECT_EQ(nodes[0].operation,
		oa::detail::opRegistry::FnMatrix::transpose.name);
	ASSERT_EQ(nodes[0].semanticOps.size(), 1U);
	EXPECT_EQ(nodes[0].opContractHash,
		oa::detail::opRegistry::FnMatrix::transpose.hash);
	ASSERT_EQ(ctx.semanticGraph()->operationCount(), 1U);
	const auto& transposeOperation = ctx.semanticGraph()->operations()[0];
	EXPECT_EQ(transposeOperation.name,
		oa::detail::opRegistry::FnMatrix::transpose.name);
	ASSERT_EQ(transposeOperation.attributes.size(), 2U);
	EXPECT_EQ(transposeOperation.attributes[0].signedInteger, 0);
	EXPECT_EQ(transposeOperation.attributes[1].signedInteger, 1);

	ctx.clear();
	const auto filled = oa::FnMatrix::fill(oa::MatrixShape{2, 3}, 2.5F);
	ASSERT_EQ(ctx.nodeCount(), 1U);
	nodes = ctx.graph()->nodes();
	EXPECT_EQ(nodes[0].operation, oa::detail::opRegistry::FnMatrix::fill.name);
	EXPECT_EQ(nodes[0].opContractHash,
		oa::detail::opRegistry::FnMatrix::fill.hash);
	ASSERT_EQ(ctx.semanticGraph()->operationCount(), 1U);
	const auto& fillOperation = ctx.semanticGraph()->operations()[0];
	EXPECT_TRUE(fillOperation.inputs.empty());
	ASSERT_EQ(fillOperation.attributes.size(), 2U);
	EXPECT_FLOAT_EQ(
		static_cast<oa::F32>(fillOperation.attributes[0].floatVal), 2.5F);
	EXPECT_EQ(fillOperation.attributes[1].shape, oa::MatrixShape({2, 3}));

	ctx.clear();
	auto fillTarget = filled;
	oa::FnMatrix::fillInPlace(fillTarget, -3.0F);
	ASSERT_EQ(ctx.nodeCount(), 1U);
	nodes = ctx.graph()->nodes();
	EXPECT_EQ(nodes[0].operation,
		oa::detail::opRegistry::FnMatrix::fillInPlace.name);
	EXPECT_EQ(nodes[0].opContractHash,
		oa::detail::opRegistry::FnMatrix::fillInPlace.hash);
	ASSERT_EQ(ctx.semanticGraph()->operationCount(), 1U);
	const auto& fillInPlaceOperation = ctx.semanticGraph()->operations()[0];
	ASSERT_EQ(fillInPlaceOperation.mutatedInputs.size(), 1U);
	ASSERT_EQ(fillInPlaceOperation.aliases.size(), 1U);
	EXPECT_EQ(fillInPlaceOperation.aliases[0].input,
		fillInPlaceOperation.inputs[0]);
	EXPECT_EQ(fillInPlaceOperation.aliases[0].output,
		fillInPlaceOperation.outputs[0]);

	ctx.clear();
	constexpr oa::U64 kSeed = 0x12345678abcdef01ULL;
	const auto uniform = oa::FnMatrix::philoxUniform(a, -2.0F, 4.0F, kSeed);
	(void)uniform;
	ASSERT_EQ(ctx.nodeCount(), 1U);
	nodes = ctx.graph()->nodes();
	EXPECT_EQ(nodes[0].operation,
		oa::detail::opRegistry::FnMatrix::philoxUniform.name);
	EXPECT_EQ(nodes[0].opContractHash,
		oa::detail::opRegistry::FnMatrix::philoxUniform.hash);
	const auto& uniformOperation = ctx.semanticGraph()->operations()[0];
	ASSERT_EQ(uniformOperation.attributes.size(), 3U);
	EXPECT_DOUBLE_EQ(uniformOperation.attributes[0].floatVal, -2.0);
	EXPECT_DOUBLE_EQ(uniformOperation.attributes[1].floatVal, 4.0);
	EXPECT_EQ(uniformOperation.attributes[2].unsignedInteger, kSeed);

	ctx.clear();
	const auto normal = oa::FnMatrix::philoxNormal(a, 1.5F, 0.25F, kSeed);
	(void)normal;
	ASSERT_EQ(ctx.nodeCount(), 1U);
	nodes = ctx.graph()->nodes();
	EXPECT_EQ(nodes[0].operation,
		oa::detail::opRegistry::FnMatrix::philoxNormal.name);
	EXPECT_EQ(nodes[0].opContractHash,
		oa::detail::opRegistry::FnMatrix::philoxNormal.hash);
	const auto& normalOperation = ctx.semanticGraph()->operations()[0];
	ASSERT_EQ(normalOperation.attributes.size(), 3U);
	EXPECT_DOUBLE_EQ(normalOperation.attributes[0].floatVal, 1.5);
	EXPECT_DOUBLE_EQ(normalOperation.attributes[1].floatVal, 0.25);
	EXPECT_EQ(normalOperation.attributes[2].unsignedInteger, kSeed);
	ctx.clear();
}

TEST(ExecutableGraph, VariadicConcatAndSplitOwnComposedLoweringAndAutograd) {
	auto& ctx = oa::ExecutionSession::getActive();
	ctx.clear();
	auto a = oa::FnMatrix::empty({1, 1}, oa::ScalarType::Float32);
	auto b = oa::FnMatrix::empty({1, 2}, oa::ScalarType::Float32);
	auto c = oa::FnMatrix::empty({1, 3}, oa::ScalarType::Float32);
	auto source = oa::FnMatrix::empty({1, 6}, oa::ScalarType::Float32);
	a.setRequiresGrad(true);
	b.setRequiresGrad(true);
	c.setRequiresGrad(true);
	source.setRequiresGrad(true);
	ctx.clear();

	oa::GradientTape tape;
	oa::Matrix concatInputs[] = {a, b, c};
	const auto concatenated = oa::FnMatrix::concat(
		oa::Span<oa::Matrix>(concatInputs, 3U), 1);
	ASSERT_FALSE(concatenated.isEmpty());
	const auto* concatGraph = ctx.semanticGraph();
	ASSERT_NE(concatGraph, nullptr);
	ASSERT_TRUE(concatGraph->validate().isOk());
	ASSERT_EQ(concatGraph->operationCount(), 1U);
	const auto& concat = concatGraph->operations()[0];
	EXPECT_EQ(concat.name, oa::detail::opRegistry::FnMatrix::concat.name);
	ASSERT_EQ(concat.inputs.size(), 3U);
	ASSERT_EQ(concat.outputs.size(), 1U);
	ASSERT_EQ(concat.attributes.size(), 1U);
	EXPECT_EQ(concat.attributes[0].signedInteger, 1);
	ASSERT_EQ(ctx.nodeCount(), 3U);
	for (const auto& node : ctx.graph()->nodes()) {
		EXPECT_EQ(node.operation, oa::detail::opRegistry::FnMatrix::concat.name);
		EXPECT_EQ(
			node.opContractHash,
			oa::detail::opRegistry::FnMatrix::concat.hash);
		ASSERT_EQ(node.semanticOps.size(), 1U);
		EXPECT_EQ(node.semanticOps[0], concat.id);
	}
	ASSERT_EQ(concatGraph->autograd().size(), 1U);
	EXPECT_EQ(concatGraph->autograd()[0].outputIndex, 0U);
	ASSERT_TRUE(concatenated.getGradFn());
	EXPECT_EQ(
		concatenated.getGradFn()->forwardSemanticOp_, concat.id);
	const auto concatAnalysis = oa::analyzeSemanticLowering(
		*concatGraph, *ctx.graph());
	ASSERT_TRUE(concatAnalysis.isOk())
		<< concatAnalysis.getStatus().getMessage();
	EXPECT_EQ(concatAnalysis.getValue().decomposedOpCount(), 1U);

	ctx.clear();
	oa::I64 sizes[] = {1, 2, 3};
	const auto splits = oa::FnMatrix::split(
		source, oa::Span<oa::I64>(sizes, 3U), 1);
	ASSERT_EQ(splits.size(), 3U);
	const auto* splitGraph = ctx.semanticGraph();
	ASSERT_NE(splitGraph, nullptr);
	ASSERT_TRUE(splitGraph->validate().isOk());
	ASSERT_EQ(splitGraph->operationCount(), 1U);
	const auto& split = splitGraph->operations()[0];
	EXPECT_EQ(split.name, oa::detail::opRegistry::FnMatrix::split.name);
	ASSERT_EQ(split.inputs.size(), 1U);
	ASSERT_EQ(split.outputs.size(), 3U);
	ASSERT_EQ(split.attributes.size(), 1U);
	EXPECT_EQ(split.attributes[0].signedInteger, 1);
	ASSERT_EQ(ctx.nodeCount(), 3U);
	for (const auto& node : ctx.graph()->nodes()) {
		EXPECT_EQ(node.operation, oa::detail::opRegistry::FnMatrix::split.name);
		EXPECT_EQ(
			node.opContractHash,
			oa::detail::opRegistry::FnMatrix::split.hash);
		ASSERT_EQ(node.semanticOps.size(), 1U);
		EXPECT_EQ(node.semanticOps[0], split.id);
	}
	ASSERT_EQ(splitGraph->autograd().size(), 3U);
	for (oa::U32 index = 0U; index < splits.size(); ++index) {
		ASSERT_TRUE(splits[index].getGradFn());
		EXPECT_EQ(
			splits[index].getGradFn()->forwardSemanticOp_, split.id);
		EXPECT_EQ(
			splits[index].getGradFn()->forwardSemanticOutput_, index);
		EXPECT_EQ(splitGraph->autograd()[index].outputIndex, index);
	}
	const auto splitAnalysis = oa::analyzeSemanticLowering(
		*splitGraph, *ctx.graph());
	ASSERT_TRUE(splitAnalysis.isOk())
		<< splitAnalysis.getStatus().getMessage();
	EXPECT_EQ(splitAnalysis.getValue().decomposedOpCount(), 1U);
	ctx.clear();
}

TEST(ExecutableGraph, AttentionSchemaWaveOwnsExecutableProvenance) {
	auto& ctx = oa::ExecutionSession::getActive();
	ctx.clear();
	const auto a = oa::FnMatrix::empty({2, 3, 4}, oa::ScalarType::Float32);
	const auto b = oa::FnMatrix::empty({2, 4, 5}, oa::ScalarType::Float32);
	ctx.clear();

	const auto product = oa::FnMatrix::bmm(a, b);
	ASSERT_FALSE(product.isEmpty());
	ASSERT_EQ(ctx.nodeCount(), 1U);
	ASSERT_EQ(ctx.semanticGraph()->operationCount(), 1U);
	const auto& bmm = ctx.semanticGraph()->operations()[0];
	EXPECT_EQ(bmm.name, oa::detail::opRegistry::FnMatrix::bmm.name);
	EXPECT_EQ(bmm.inputs.size(), 2U);
	EXPECT_EQ(bmm.outputs.size(), 1U);
	EXPECT_EQ(ctx.graph()->nodes()[0].opContractHash,
		oa::detail::opRegistry::FnMatrix::bmm.hash);
	ASSERT_TRUE(oa::validateSemanticLowering(
		*ctx.semanticGraph(), *ctx.graph()).isOk());

	ctx.clear();
	const auto bNt = oa::FnMatrix::empty({2, 5, 4}, oa::ScalarType::Float32);
	ctx.clear();
	const auto productNt = oa::FnMatrix::bmmNt(a, bNt);
	ASSERT_FALSE(productNt.isEmpty());
	ASSERT_EQ(ctx.nodeCount(), 1U);
	ASSERT_EQ(ctx.semanticGraph()->operationCount(), 1U);
	const auto& bmmNt = ctx.semanticGraph()->operations()[0];
	EXPECT_EQ(bmmNt.name, oa::detail::opRegistry::FnMatrix::bmmNt.name);
	EXPECT_EQ(bmmNt.inputs.size(), 2U);
	EXPECT_EQ(bmmNt.outputs.size(), 1U);
	EXPECT_EQ(ctx.graph()->nodes()[0].opContractHash,
		oa::detail::opRegistry::FnMatrix::bmmNt.hash);
	ASSERT_TRUE(oa::validateSemanticLowering(
		*ctx.semanticGraph(), *ctx.graph()).isOk());

	ctx.clear();
	const auto tokens = oa::FnMatrix::empty({6, 8}, oa::ScalarType::Float32);
	ctx.clear();
	const auto split = oa::FnMatrix::splitHeads(tokens, 2, 3, 2);
	const auto merged = oa::FnMatrix::mergeHeads(split, 2, 3, 2);
	ASSERT_FALSE(split.isEmpty());
	ASSERT_FALSE(merged.isEmpty());
	ASSERT_EQ(ctx.nodeCount(), 2U);
	ASSERT_EQ(ctx.semanticGraph()->operationCount(), 2U);
	EXPECT_EQ(ctx.semanticGraph()->operations()[0].name,
		oa::detail::opRegistry::FnMatrix::splitHeads.name);
	EXPECT_EQ(ctx.semanticGraph()->operations()[1].name,
		oa::detail::opRegistry::FnMatrix::mergeHeads.name);
	for (const auto& operation : ctx.semanticGraph()->operations()) {
		ASSERT_EQ(operation.attributes.size(), 3U);
		EXPECT_EQ(operation.attributes[0].signedInteger, 2);
		EXPECT_EQ(operation.attributes[1].signedInteger, 3);
		EXPECT_EQ(operation.attributes[2].signedInteger, 2);
	}
	ASSERT_TRUE(oa::validateSemanticLowering(
		*ctx.semanticGraph(), *ctx.graph()).isOk());

	ctx.clear();
	const auto q = oa::FnMatrix::empty({2, 4, 8}, oa::ScalarType::Float32);
	const auto k = oa::FnMatrix::empty({2, 4, 8}, oa::ScalarType::Float32);
	const auto v = oa::FnMatrix::empty({2, 4, 8}, oa::ScalarType::Float32);
	ctx.clear();
	constexpr oa::F32 scale = 0.5F;
	const auto flash = oa::FnMatrix::flashAttentionCausal(q, k, v, scale);
	ASSERT_FALSE(flash.isEmpty());
	ASSERT_EQ(ctx.nodeCount(), 1U);
	ASSERT_EQ(ctx.semanticGraph()->operationCount(), 1U);
	const auto& attention = ctx.semanticGraph()->operations()[0];
	EXPECT_EQ(attention.name,
		oa::detail::opRegistry::FnMatrix::flashAttentionCausal.name);
	EXPECT_EQ(attention.inputs.size(), 3U);
	// The second output is private FP32 log-sum-exp state consumed by backward.
	EXPECT_EQ(attention.outputs.size(), 2U);
	ASSERT_EQ(attention.attributes.size(), 1U);
	EXPECT_DOUBLE_EQ(attention.attributes[0].floatVal, scale);
	EXPECT_EQ(ctx.graph()->nodes()[0].opContractHash,
		oa::detail::opRegistry::FnMatrix::flashAttentionCausal.hash);
	ASSERT_TRUE(oa::validateSemanticLowering(
		*ctx.semanticGraph(), *ctx.graph()).isOk());
	ctx.clear();
}

TEST(ExecutableGraph, LinearParamAdjointNarrowRouteOwnsSemanticProvenance) {
	auto& ctx = oa::ExecutionSession::getActive();
	ctx.clear();
	const auto input = oa::FnMatrix::empty({1024, 32}, oa::ScalarType::Float32);
	const auto gradOutput = oa::FnMatrix::empty({1024, 32}, oa::ScalarType::Float32);
	ctx.clear();

	const auto result = oa::FnMatrix::linearWeightBiasBwd(input, gradOutput);
	ASSERT_FALSE(result.gradWeight.isEmpty());
	ASSERT_FALSE(result.gradBias.isEmpty());
	ASSERT_EQ(ctx.nodeCount(), 1U);
	ASSERT_EQ(ctx.semanticGraph()->operationCount(), 1U);
	const auto& operation = ctx.semanticGraph()->operations()[0];
	EXPECT_EQ(operation.name,
		oa::detail::opRegistry::FnMatrix::linearWeightBiasBwd.name);
	EXPECT_EQ(operation.inputs.size(), 2U);
	EXPECT_EQ(operation.outputs.size(), 2U);
	const auto& node = ctx.graph()->nodes()[0];
	EXPECT_EQ(node.shader, "LinearWeightBiasBwdRows32");
	EXPECT_EQ(node.opContractHash,
		oa::detail::opRegistry::FnMatrix::linearWeightBiasBwd.hash);
	ASSERT_EQ(node.semanticOps.size(), 1U);
	EXPECT_EQ(node.semanticOps[0], operation.id);
	ASSERT_TRUE(oa::validateSemanticLowering(
		*ctx.semanticGraph(), *ctx.graph()).isOk());
	ctx.clear();
}

TEST(ExecutableGraph, MoeSchemaWaveOwnsGroupedLowering) {
	auto& ctx = oa::ExecutionSession::getActive();
	ctx.clear();
	const auto expertIndices = oa::FnMatrix::empty({2, 2}, oa::ScalarType::Int32);
	const auto offsets = oa::FnMatrix::empty({3}, oa::ScalarType::UInt32);
	const auto packedIndices = oa::FnMatrix::empty({4}, oa::ScalarType::UInt32);
	const auto inverse = oa::FnMatrix::empty({4}, oa::ScalarType::UInt32);
	const auto packedSlot = oa::FnMatrix::empty({4}, oa::ScalarType::UInt32);
	auto probabilities = oa::FnMatrix::empty({2, 2}, oa::ScalarType::Float32);
	auto groupedInput = oa::FnMatrix::empty({4, 3}, oa::ScalarType::Float32);
	auto groupedWeight = oa::FnMatrix::empty({2, 2, 3}, oa::ScalarType::Float32);
	auto groupedBias = oa::FnMatrix::empty({2, 2}, oa::ScalarType::Float32);
	const auto groupedDOut = oa::FnMatrix::empty({4, 2}, oa::ScalarType::Float32);
	auto packedValues = oa::FnMatrix::empty({4, 2}, oa::ScalarType::Float32);
	auto routeGate = oa::FnMatrix::empty({2, 2}, oa::ScalarType::Float32);
	const auto tokenDOut = oa::FnMatrix::empty({2, 2}, oa::ScalarType::Float32);
	auto tokenValues = oa::FnMatrix::empty({2, 3}, oa::ScalarType::Float32);
	auto scatterSource = oa::FnMatrix::empty({4, 2}, oa::ScalarType::Float32);
	probabilities.setRequiresGrad(true);
	groupedInput.setRequiresGrad(true);
	groupedWeight.setRequiresGrad(true);
	groupedBias.setRequiresGrad(true);
	packedValues.setRequiresGrad(true);
	routeGate.setRequiresGrad(true);
	tokenValues.setRequiresGrad(true);
	scatterSource.setRequiresGrad(true);
	ctx.clear();

	oa::GradientTape tape;
	const auto weights = oa::FnMatrix::moeRouteWeights(
		probabilities, expertIndices);
	const auto probabilityGrad = oa::FnMatrix::moeRouteWeightsBwd(
		routeGate, probabilities, expertIndices, weights);
	const auto grouped = oa::FnMatrix::groupedGemmM(
		groupedInput, groupedWeight, offsets);
	const auto groupedGrad = oa::FnMatrix::groupedGemmMBwd(
		groupedDOut, groupedInput, groupedWeight, offsets);
	const auto linear = oa::FnMatrix::groupedLinearM(
		groupedInput, groupedWeight, groupedBias, offsets);
	const auto linearGrad = oa::FnMatrix::groupedLinearMBwd(
		groupedDOut, groupedInput, groupedWeight, offsets);
	const auto biasGrad = oa::FnMatrix::groupedLinearMBiasBwd(
		groupedDOut, offsets, 2);
	const auto combined = oa::FnMatrix::moeCombine(
		packedValues, routeGate, inverse, packedSlot);
	const auto combinedGrad = oa::FnMatrix::moeCombineBwd(
		tokenDOut, packedValues, routeGate, inverse, packedSlot);
	const auto gathered = oa::FnMatrix::moeGather(
		tokenValues, packedIndices, inverse);
	const auto gatheredGrad = oa::FnMatrix::moeGatherBwd(
		gathered, inverse, 2);
	const auto scattered = oa::FnMatrix::scatterAddRows(
		scatterSource, packedIndices, 2);
	(void)probabilityGrad;
	(void)grouped;
	(void)groupedGrad;
	(void)linear;
	(void)linearGrad;
	(void)biasGrad;
	(void)combined;
	(void)combinedGrad;
	(void)gatheredGrad;
	(void)scattered;

	const oa::OpContract* expected[] = {
		&oa::detail::opRegistry::FnMatrix::moeRouteWeights,
		&oa::detail::opRegistry::FnMatrix::moeRouteWeightsBwd,
		&oa::detail::opRegistry::FnMatrix::groupedGemmM,
		&oa::detail::opRegistry::FnMatrix::groupedGemmMBwd,
		&oa::detail::opRegistry::FnMatrix::groupedLinearM,
		&oa::detail::opRegistry::FnMatrix::groupedLinearMBwd,
		&oa::detail::opRegistry::FnMatrix::groupedLinearMBiasBwd,
		&oa::detail::opRegistry::FnMatrix::moeCombine,
		&oa::detail::opRegistry::FnMatrix::moeCombineBwd,
		&oa::detail::opRegistry::FnMatrix::moeGather,
		&oa::detail::opRegistry::FnMatrix::moeGatherBwd,
		&oa::detail::opRegistry::FnMatrix::scatterAddRows,
	};
	constexpr oa::U32 ExpectedOperationCount = 12U;
	ASSERT_NE(ctx.semanticGraph(), nullptr);
	ASSERT_TRUE(ctx.semanticGraph()->validate().isOk());
	const auto operations = ctx.semanticGraph()->operations();
	ASSERT_EQ(operations.size(), ExpectedOperationCount);
	for (oa::U32 index = 0U; index < ExpectedOperationCount; ++index) {
		EXPECT_EQ(operations[index].name, expected[index]->name);
		EXPECT_EQ(operations[index].contractHash, expected[index]->hash);
	}
	EXPECT_EQ(operations[3].outputs.size(), 2U);
	EXPECT_EQ(operations[5].outputs.size(), 3U);
	EXPECT_EQ(operations[8].outputs.size(), 2U);
	ASSERT_EQ(operations[6].attributes.size(), 1U);
	EXPECT_EQ(operations[6].attributes[0].signedInteger, 2);
	ASSERT_EQ(operations[10].attributes.size(), 1U);
	EXPECT_EQ(operations[10].attributes[0].signedInteger, 2);
	ASSERT_EQ(operations[11].attributes.size(), 1U);
	EXPECT_EQ(operations[11].attributes[0].signedInteger, 2);
	ASSERT_EQ(ctx.semanticGraph()->autograd().size(), 6U);

	ASSERT_EQ(ctx.nodeCount(), 14U);
	for (const auto& node : ctx.graph()->nodes()) {
		ASSERT_EQ(node.semanticOps.size(), 1U);
		EXPECT_NE(node.opContractHash, 0U);
	}
	const auto analyzed = oa::analyzeSemanticLowering(
		*ctx.semanticGraph(), *ctx.graph());
	ASSERT_TRUE(analyzed.isOk()) << analyzed.getStatus().getMessage();
	EXPECT_EQ(analyzed.getValue().directOpCount(), 10U);
	EXPECT_EQ(analyzed.getValue().decomposedOpCount(), 2U);
	EXPECT_EQ(analyzed.getValue().fusedOpCount(), 0U);
	ctx.clear();
}

TEST(ExecutableGraph, RecurrentSchemaWaveOwnsCellAndScanLowering) {
	auto& ctx = oa::ExecutionSession::getActive();
	ctx.clear();
	constexpr oa::I32 Batch = 1;
	constexpr oa::I32 sequenceLength = 2;
	constexpr oa::I32 hiddenSize = 2;

	const auto hidden = oa::FnMatrix::empty(
		{Batch, hiddenSize}, oa::ScalarType::Float32);
	const auto gradHidden = oa::FnMatrix::empty(
		{Batch, hiddenSize}, oa::ScalarType::Float32);

	const auto gruGatesI = oa::FnMatrix::empty(
		{Batch * sequenceLength, 3 * hiddenSize}, oa::ScalarType::Float32);
	const auto gruGatesH = oa::FnMatrix::empty(
		{Batch, 3 * hiddenSize}, oa::ScalarType::Float32);
	const auto gruWeight = oa::FnMatrix::empty(
		{3 * hiddenSize, hiddenSize}, oa::ScalarType::Float32);
	const auto gruBias = oa::FnMatrix::empty(
		{3 * hiddenSize}, oa::ScalarType::Float32);
	const auto gruDOut = oa::FnMatrix::empty(
		{Batch, sequenceLength, hiddenSize}, oa::ScalarType::Float32);
	const auto gruHprev = oa::FnMatrix::empty(
		{Batch, sequenceLength, hiddenSize}, oa::ScalarType::Float32);

	const auto rnnGatesI = oa::FnMatrix::empty(
		{Batch * sequenceLength, hiddenSize}, oa::ScalarType::Float32);
	const auto rnnCellGatesI = oa::FnMatrix::empty(
		{Batch, hiddenSize}, oa::ScalarType::Float32);
	const auto rnnGatesH = oa::FnMatrix::empty(
		{Batch, hiddenSize}, oa::ScalarType::Float32);
	const auto rnnWeight = oa::FnMatrix::empty(
		{hiddenSize, hiddenSize}, oa::ScalarType::Float32);
	const auto rnnBias = oa::FnMatrix::empty(
		{hiddenSize}, oa::ScalarType::Float32);
	const auto rnnDOut = oa::FnMatrix::empty(
		{Batch, sequenceLength, hiddenSize}, oa::ScalarType::Float32);
	const auto rnnHprev = oa::FnMatrix::empty(
		{Batch, sequenceLength, hiddenSize}, oa::ScalarType::Float32);
	ctx.clear();

	const auto gruPointwise = oa::FnMatrix::gruCellPointwise(
		gruGatesI, gruGatesH, hidden, hiddenSize, 0, 1);
	const auto gruPointwiseGrad = oa::FnMatrix::gruCellPointwiseBwd(
		gruGatesI, gruGatesH, hidden, gradHidden, hiddenSize, 0, 1);
	const auto gruLinear = oa::FnMatrix::gruCellLinear(
		gruGatesI, hidden, gruWeight, hiddenSize, 0, 1, gruBias);
	const auto gruScan = oa::FnMatrix::gruScan(
		gruGatesI, gruWeight, hiddenSize, sequenceLength, Batch, gruBias);
	const auto gruScanGrad = oa::FnMatrix::gruScanBwd(
		gruDOut, gruGatesI, gruHprev, gruWeight, hiddenSize,
		sequenceLength, Batch, gruBias);

	const auto rnnPointwise = oa::FnMatrix::rnnCellPointwise(
		rnnCellGatesI, rnnGatesH);
	const auto rnnPointwiseGrad = oa::FnMatrix::rnnCellPointwiseBwd(
		rnnGatesI, rnnGatesH, gradHidden, hiddenSize, 0, 1);
	const auto rnnLinear = oa::FnMatrix::rnnCellLinear(
		rnnGatesI, hidden, rnnWeight, 0, 1, rnnBias);
	const auto rnnScan = oa::FnMatrix::rnnScan(
		rnnGatesI, rnnWeight, hiddenSize, sequenceLength, Batch, rnnBias);
	const auto rnnScanGrad = oa::FnMatrix::rnnScanBwd(
		rnnDOut, rnnGatesI, rnnHprev, rnnWeight, hiddenSize,
		sequenceLength, Batch, rnnBias);
	(void)gruPointwise;
	(void)gruPointwiseGrad;
	(void)gruLinear;
	(void)gruScan;
	(void)gruScanGrad;
	(void)rnnPointwise;
	(void)rnnPointwiseGrad;
	(void)rnnLinear;
	(void)rnnScan;
	(void)rnnScanGrad;

	const oa::OpContract* expected[] = {
		&oa::detail::opRegistry::FnMatrix::gruCellPointwise,
		&oa::detail::opRegistry::FnMatrix::gruCellPointwiseBwd,
		&oa::detail::opRegistry::FnMatrix::gruCellLinear,
		&oa::detail::opRegistry::FnMatrix::gruScan,
		&oa::detail::opRegistry::FnMatrix::gruScanBwd,
		&oa::detail::opRegistry::FnMatrix::rnnCellPointwise,
		&oa::detail::opRegistry::FnMatrix::rnnCellPointwiseBwd,
		&oa::detail::opRegistry::FnMatrix::rnnCellLinear,
		&oa::detail::opRegistry::FnMatrix::rnnScan,
		&oa::detail::opRegistry::FnMatrix::rnnScanBwd,
	};
	constexpr oa::U32 ExpectedOperationCount = 10U;
	constexpr oa::U32 ExpectedNodeCount[] = {
		1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U,
	};

	ASSERT_NE(ctx.semanticGraph(), nullptr);
	ASSERT_TRUE(ctx.semanticGraph()->validate().isOk());
	const auto operations = ctx.semanticGraph()->operations();
	ASSERT_EQ(operations.size(), ExpectedOperationCount);
	for (oa::U32 index = 0U; index < ExpectedOperationCount; ++index) {
		EXPECT_EQ(operations[index].name, expected[index]->name);
		EXPECT_EQ(operations[index].contractHash, expected[index]->hash);
		oa::U32 ownedNodes = 0U;
		for (const auto& node : ctx.graph()->nodes()) {
			for (const auto semanticOp : node.semanticOps) {
				if (semanticOp == operations[index].id) ++ownedNodes;
			}
		}
		EXPECT_EQ(ownedNodes, ExpectedNodeCount[index]);
	}

	EXPECT_EQ(operations[2].optionalInputMask, 0x8U);
	EXPECT_NE(operations[2].inputs[3], oa::invalidSemanticValueId);
	EXPECT_EQ(operations[3].optionalInputMask, 0x4U);
	EXPECT_NE(operations[3].inputs[2], oa::invalidSemanticValueId);
	EXPECT_EQ(operations[4].optionalInputMask, 0x10U);
	EXPECT_NE(operations[4].inputs[4], oa::invalidSemanticValueId);
	EXPECT_EQ(operations[7].optionalInputMask, 0x8U);
	EXPECT_NE(operations[7].inputs[3], oa::invalidSemanticValueId);
	EXPECT_EQ(operations[8].optionalInputMask, 0x4U);
	EXPECT_NE(operations[8].inputs[2], oa::invalidSemanticValueId);
	EXPECT_EQ(operations[9].optionalInputMask, 0x10U);
	EXPECT_NE(operations[9].inputs[4], oa::invalidSemanticValueId);

	EXPECT_EQ(operations[0].attributes.size(), 3U);
	EXPECT_EQ(operations[1].attributes.size(), 3U);
	EXPECT_EQ(operations[2].attributes.size(), 3U);
	EXPECT_EQ(operations[3].attributes.size(), 3U);
	EXPECT_EQ(operations[4].attributes.size(), 3U);
	EXPECT_EQ(operations[5].attributes.size(), 0U);
	EXPECT_EQ(operations[6].attributes.size(), 3U);
	EXPECT_EQ(operations[7].attributes.size(), 2U);
	EXPECT_EQ(operations[8].attributes.size(), 3U);
	EXPECT_EQ(operations[9].attributes.size(), 3U);

	ASSERT_TRUE(oa::validateSemanticLowering(
		*ctx.semanticGraph(), *ctx.graph()).isOk());
	const auto analyzed = oa::analyzeSemanticLowering(
		*ctx.semanticGraph(), *ctx.graph());
	ASSERT_TRUE(analyzed.isOk()) << analyzed.getStatus().getMessage();
	EXPECT_EQ(analyzed.getValue().directOpCount(), 10U);
	EXPECT_EQ(analyzed.getValue().decomposedOpCount(), 0U);
	EXPECT_EQ(analyzed.getValue().fusedOpCount(), 0U);
	ctx.clear();
}

TEST(ExecutableGraph, ConvolutionAndNormSchemaWaveOwnsEveryLowering) {
	auto& ctx = oa::ExecutionSession::getActive();
	ctx.clear();

	const auto conv1dInput = oa::FnMatrix::empty(
		{1, 1, 4}, oa::ScalarType::Float32);
	const auto conv1dWeight = oa::FnMatrix::empty(
		{1, 1, 2}, oa::ScalarType::Float32);
	const auto conv1dBias = oa::FnMatrix::empty({1}, oa::ScalarType::Float32);
	const auto conv1dGrad = oa::FnMatrix::empty(
		{1, 1, 3}, oa::ScalarType::Float32);

	const auto conv2dInput = oa::FnMatrix::empty(
		{1, 1, 3, 3}, oa::ScalarType::Float32);
	const auto conv2dWeight = oa::FnMatrix::empty(
		{1, 1, 2, 2}, oa::ScalarType::Float32);
	const auto conv2dGrad = oa::FnMatrix::empty(
		{1, 1, 2, 2}, oa::ScalarType::Float32);
	const auto convTransposeGrad = oa::FnMatrix::empty(
		{1, 1, 3, 3}, oa::ScalarType::Float32);

	const auto normInput = oa::FnMatrix::empty(
		{2, 2}, oa::ScalarType::Float32);
	const auto normWeight = oa::FnMatrix::empty({2}, oa::ScalarType::Float32);
	const auto normBias = oa::FnMatrix::empty({2}, oa::ScalarType::Float32);
	const auto normGate = oa::FnMatrix::empty(
		{2, 2}, oa::ScalarType::Float32);
	const auto normGrad = oa::FnMatrix::empty(
		{2, 2}, oa::ScalarType::Float32);

	const auto channelInput = oa::FnMatrix::empty(
		{1, 2, 2}, oa::ScalarType::Float32);
	const auto channelWeight = oa::FnMatrix::empty({2}, oa::ScalarType::Float32);
	const auto channelBias = oa::FnMatrix::empty({2}, oa::ScalarType::Float32);
	const auto channelGrad = oa::FnMatrix::empty(
		{1, 2, 2}, oa::ScalarType::Float32);
	ctx.clear();

	const auto conv1d = oa::FnMatrix::conv1dGemm(
		conv1dInput, conv1dWeight, conv1dBias, 1, 0, 1);
	const auto conv1dRelu = oa::FnMatrix::conv1dReluGemm(
		conv1dInput, conv1dWeight, conv1dBias, 1, 0, 1);
	const auto columns = oa::FnMatrix::im2Col1d(conv1dInput, 2, 1, 0, 1);
	const auto folded = oa::FnMatrix::col2Im1d(
		columns, 1, 1, 4, 2, 1, 0, 1, 3);
	const auto conv1dDataGrad = oa::FnMatrix::conv1dBwdData(
		conv1dGrad, conv1dWeight, 1, 0, 1, {1, 1, 4});
	const auto conv1dWeightGrad = oa::FnMatrix::conv1dBwdWeight(
		conv1dInput, conv1dGrad, conv1dWeight, 1, 0, 1);

	const auto conv2dDataGrad = oa::FnMatrix::conv2dBwdData(
		conv2dGrad, conv2dWeight, 1, 0, {1, 1, 3, 3});
	const auto conv2dWeightGrad = oa::FnMatrix::conv2dBwdWeight(
		conv2dInput, conv2dGrad, conv2dWeight, 1, 0);
	const auto convTranspose = oa::FnMatrix::convTranspose2d(
		conv2dGrad, conv2dWeight, conv1dBias, 1, 0);
	const auto convTransposeDataGrad = oa::FnMatrix::convTranspose2dBwdData(
		convTransposeGrad, conv2dWeight, 1, 0, {1, 1, 2, 2});
	const auto convTransposeWeightGrad =
		oa::FnMatrix::convTranspose2dBwdWeight(
			conv2dGrad, convTransposeGrad, conv2dWeight, 1, 0);

	const auto gatedNorm = oa::FnMatrix::rmsNormGated(
		normInput, normWeight, normBias, normGate, 1e-5F, true);
	const auto gatedNormGrad = oa::FnMatrix::rmsNormGatedBwd(
		normInput, normWeight, normBias, normGate, normGrad, 1e-5F);

	const auto channelNorm = oa::FnMatrix::channelNorm(
		channelInput, channelWeight, channelBias, 1, 2, 2, 1e-5F);
	const auto channelNormGrad = oa::FnMatrix::channelNormBwd(
		channelInput, channelWeight, channelGrad, 1, 2, 2, 1e-5F);
	const auto channelNormRelu = oa::FnMatrix::channelNormRelu(
		channelInput, channelWeight, channelBias, 1, 2, 2, 1e-5F);
	const auto channelNormReluGrad = oa::FnMatrix::channelNormReluBwd(
		channelInput, channelWeight, channelNormRelu, channelGrad,
		1, 2, 2, 1e-5F);
	(void)conv1d;
	(void)conv1dRelu;
	(void)folded;
	(void)conv1dDataGrad;
	(void)conv1dWeightGrad;
	(void)conv2dDataGrad;
	(void)conv2dWeightGrad;
	(void)convTranspose;
	(void)convTransposeDataGrad;
	(void)convTransposeWeightGrad;
	(void)gatedNorm;
	(void)gatedNormGrad;
	(void)channelNorm;
	(void)channelNormGrad;
	(void)channelNormReluGrad;

	const oa::OpContract* expected[] = {
		&oa::detail::opRegistry::FnMatrix::conv1dGemm,
		&oa::detail::opRegistry::FnMatrix::conv1dReluGemm,
		&oa::detail::opRegistry::FnMatrix::im2Col1d,
		&oa::detail::opRegistry::FnMatrix::col2Im1d,
		&oa::detail::opRegistry::FnMatrix::conv1dBwdData,
		&oa::detail::opRegistry::FnMatrix::conv1dBwdWeight,
		&oa::detail::opRegistry::FnMatrix::conv2dBwdData,
		&oa::detail::opRegistry::FnMatrix::conv2dBwdWeight,
		&oa::detail::opRegistry::FnMatrix::convTranspose2d,
		&oa::detail::opRegistry::FnMatrix::convTranspose2dBwdData,
		&oa::detail::opRegistry::FnMatrix::convTranspose2dBwdWeight,
		&oa::detail::opRegistry::FnMatrix::rmsNormGated,
		&oa::detail::opRegistry::FnMatrix::rmsNormGatedBwd,
		&oa::detail::opRegistry::FnMatrix::channelNorm,
		&oa::detail::opRegistry::FnMatrix::channelNormBwd,
		&oa::detail::opRegistry::FnMatrix::channelNormRelu,
		&oa::detail::opRegistry::FnMatrix::channelNormReluBwd,
	};
	constexpr oa::U32 ExpectedOperationCount = 17U;
	constexpr oa::U32 ExpectedNodeCount[] = {
		4U, 5U, 1U, 1U, 1U, 1U, 1U, 1U, 3U,
		1U, 1U, 1U, 3U, 1U, 3U, 1U, 3U,
	};

	ASSERT_NE(ctx.semanticGraph(), nullptr);
	ASSERT_TRUE(ctx.semanticGraph()->validate().isOk());
	const auto operations = ctx.semanticGraph()->operations();
	ASSERT_EQ(operations.size(), ExpectedOperationCount);
	for (oa::U32 index = 0U; index < ExpectedOperationCount; ++index) {
		EXPECT_EQ(operations[index].name, expected[index]->name);
		EXPECT_EQ(operations[index].contractHash, expected[index]->hash);
		EXPECT_EQ(
			operations[index].attributes.size(), expected[index]->attributeCount);
		oa::U32 ownedNodes = 0U;
		for (const auto& node : ctx.graph()->nodes()) {
			for (const auto semanticOp : node.semanticOps) {
				if (semanticOp == operations[index].id) ++ownedNodes;
			}
		}
		EXPECT_EQ(ownedNodes, ExpectedNodeCount[index])
			<< operations[index].name.data();
	}
	EXPECT_EQ(operations[0].optionalInputMask, 0x4U);
	EXPECT_NE(operations[0].inputs[2], oa::invalidSemanticValueId);
	EXPECT_EQ(operations[1].optionalInputMask, 0x4U);
	EXPECT_NE(operations[1].inputs[2], oa::invalidSemanticValueId);

	ASSERT_TRUE(oa::validateSemanticLowering(
		*ctx.semanticGraph(), *ctx.graph()).isOk());
	const auto analyzed = oa::analyzeSemanticLowering(
		*ctx.semanticGraph(), *ctx.graph());
	ASSERT_TRUE(analyzed.isOk()) << analyzed.getStatus().getMessage();
	EXPECT_EQ(analyzed.getValue().directOpCount(), 11U);
	EXPECT_EQ(analyzed.getValue().decomposedOpCount(), 6U);
	EXPECT_EQ(analyzed.getValue().fusedOpCount(), 0U);
	ctx.clear();
}

TEST(ExecutableGraph, StateSpaceAndVqSchemaWaveOwnsEveryLowering) {
	auto& ctx = oa::ExecutionSession::getActive();
	ctx.clear();

	const oa::SsmConfig scanConfig{
		.batch = 1U,
		.seqLen = 2U,
		.nHeads = 1U,
		.headDim = 2U,
		.stateSize = 2U,
		.numRopeAngles = 1U,
		.hasZ = 1U,
		.hasD = 1U,
	};
	const oa::SsmConfig stepConfig{
		.batch = 1U,
		.seqLen = 1U,
		.nHeads = 1U,
		.headDim = 2U,
		.stateSize = 2U,
		.numRopeAngles = 1U,
		.hasZ = 1U,
		.hasD = 1U,
	};
	const oa::SsmConfig mimoScanConfig{
		.batch = 1U, .seqLen = 2U, .nHeads = 1U, .nGroups = 1U,
		.headDim = 2U, .stateSize = 2U, .numRopeAngles = 1U,
		.mimoRank = 2U, .hasZ = 1U, .hasD = 1U, .hasOutNorm = 1U,
	};
	const oa::SsmConfig mimoStepConfig{
		.batch = 1U, .seqLen = 1U, .nHeads = 1U, .nGroups = 1U,
		.headDim = 2U, .stateSize = 2U, .numRopeAngles = 1U,
		.mimoRank = 2U, .hasZ = 1U, .hasD = 1U, .hasOutNorm = 1U,
	};
	const oa::Mamba3PreprocessConfig preprocessConfig{
		.dInner = 2,
		.dState = 2,
		.nHeads = 1,
		.numRopeAngles = 1,
		.nGroups = 1,
		.mimoRank = 1,
		.eps = 1.0e-5F,
		.dtMin = 1.0e-3F,
		.dtMax = 1.0e-1F,
		.aFloor = 1.0e-4F,
	};

	const auto c = oa::FnMatrix::empty({1, 2, 1, 2}, oa::ScalarType::Float32);
	const auto b = oa::FnMatrix::empty({1, 2, 1, 2}, oa::ScalarType::Float32);
	const auto x = oa::FnMatrix::empty({1, 2, 1, 2}, oa::ScalarType::Float32);
	const auto z = oa::FnMatrix::empty({1, 2, 1, 2}, oa::ScalarType::Float32);
	const auto adt = oa::FnMatrix::empty({1, 2, 1}, oa::ScalarType::Float32);
	const auto dt = oa::FnMatrix::empty({1, 2, 1}, oa::ScalarType::Float32);
	const auto trap = oa::FnMatrix::empty({1, 2, 1}, oa::ScalarType::Float32);
	const auto angle = oa::FnMatrix::empty({1, 2, 1}, oa::ScalarType::Float32);
	const auto cBias = oa::FnMatrix::empty({1, 2}, oa::ScalarType::Float32);
	const auto bBias = oa::FnMatrix::empty({1, 2}, oa::ScalarType::Float32);
	const auto d = oa::FnMatrix::empty({1}, oa::ScalarType::Float32);
	const auto dOut = oa::FnMatrix::empty({1, 2, 1, 2}, oa::ScalarType::Float32);

	const auto stepC = oa::FnMatrix::empty({1, 1, 1, 2}, oa::ScalarType::Float32);
	const auto stepB = oa::FnMatrix::empty({1, 1, 1, 2}, oa::ScalarType::Float32);
	const auto stepX = oa::FnMatrix::empty({1, 1, 1, 2}, oa::ScalarType::Float32);
	const auto stepZ = oa::FnMatrix::empty({1, 1, 1, 2}, oa::ScalarType::Float32);
	const auto stepAdt = oa::FnMatrix::empty({1, 1, 1}, oa::ScalarType::Float32);
	const auto stepDt = oa::FnMatrix::empty({1, 1, 1}, oa::ScalarType::Float32);
	const auto stepTrap = oa::FnMatrix::empty({1, 1, 1}, oa::ScalarType::Float32);
	const auto stepAngle = oa::FnMatrix::empty({1, 1, 1}, oa::ScalarType::Float32);
	auto ssmState = oa::FnMatrix::empty({1, 1, 2, 2}, oa::ScalarType::Float32);
	auto angleState = oa::FnMatrix::empty({1, 1, 1}, oa::ScalarType::Float32);
	auto kState = oa::FnMatrix::empty({1, 1, 2}, oa::ScalarType::Float32);
	auto vState = oa::FnMatrix::empty({1, 1, 2}, oa::ScalarType::Float32);
	const auto mimoC = oa::FnMatrix::empty({1, 2, 2, 2}, oa::ScalarType::Float32);
	const auto mimoB = oa::FnMatrix::empty({1, 2, 2, 2}, oa::ScalarType::Float32);
	const auto mimoStepC = oa::FnMatrix::empty({1, 1, 2, 2}, oa::ScalarType::Float32);
	const auto mimoStepB = oa::FnMatrix::empty({1, 1, 2, 2}, oa::ScalarType::Float32);
	const auto mimoCBias = oa::FnMatrix::empty({1, 2, 2}, oa::ScalarType::Float32);
	const auto mimoBBias = oa::FnMatrix::empty({1, 2, 2}, oa::ScalarType::Float32);
	const auto mimoX = oa::FnMatrix::empty({1, 2, 2}, oa::ScalarType::Float32);
	const auto mimoZ = oa::FnMatrix::empty({1, 2, 2}, oa::ScalarType::Float32);
	const auto mimoO = oa::FnMatrix::empty({1, 2, 2}, oa::ScalarType::Float32);
	const auto mimoNorm = oa::FnMatrix::empty({1, 2}, oa::ScalarType::Float32);
	auto mimoSsmState = oa::FnMatrix::empty({1, 1, 2, 2}, oa::ScalarType::Float32);
	auto mimoAngleState = oa::FnMatrix::empty({1, 1, 1}, oa::ScalarType::Float32);
	auto mimoKState = oa::FnMatrix::empty({1, 1, 2, 2}, oa::ScalarType::Float32);
	auto mimoVState = oa::FnMatrix::empty({1, 1, 2, 2}, oa::ScalarType::Float32);

	const auto projected = oa::FnMatrix::empty({2, 12}, oa::ScalarType::Float32);
	const auto dtBias = oa::FnMatrix::empty({1}, oa::ScalarType::Float32);
	const auto dZ = oa::FnMatrix::empty({2, 2}, oa::ScalarType::Float32);
	const auto dX = oa::FnMatrix::empty({2, 2}, oa::ScalarType::Float32);
	const auto dBh = oa::FnMatrix::empty({2, 2}, oa::ScalarType::Float32);
	const auto dCh = oa::FnMatrix::empty({2, 2}, oa::ScalarType::Float32);
	const auto dDt = oa::FnMatrix::empty({2, 1}, oa::ScalarType::Float32);
	const auto dAdt = oa::FnMatrix::empty({2, 1}, oa::ScalarType::Float32);
	const auto dTrap = oa::FnMatrix::empty({2, 1}, oa::ScalarType::Float32);
	const auto dAngle = oa::FnMatrix::empty({2, 1}, oa::ScalarType::Float32);

	const auto heavyInput = oa::FnMatrix::empty({2, 2}, oa::ScalarType::Float32);
	const auto ze = oa::FnMatrix::empty({2, 2}, oa::ScalarType::Float32);
	auto codebook = oa::FnMatrix::empty({2, 2}, oa::ScalarType::Float32);
	auto embedSum = oa::FnMatrix::empty({2, 2}, oa::ScalarType::Float32);
	auto clusterSize = oa::FnMatrix::empty({2}, oa::ScalarType::Float32);
	ctx.clear();

	const auto mambaSiso = oa::FnMatrix::mamba3Siso(
		c, b, x, z, adt, dt, trap, angle, cBias, bBias, d, scanConfig);
	const auto mambaStep = oa::FnMatrix::mamba3SisoStep(
		stepC, stepB, stepX, stepZ, stepAdt, stepDt, stepTrap, stepAngle,
		cBias, bBias, d, ssmState, angleState, kState, vState, stepConfig);
	const auto mambaBwd = oa::FnMatrix::mamba3SisoBwd(
		dOut, c, b, x, z, adt, dt, trap, angle, cBias, bBias, d, scanConfig);
	const auto mambaMimo = oa::FnMatrix::mamba3Mimo(
		mimoC, mimoB, x, z, adt, dt, trap, angle, mimoCBias, mimoBBias, d,
		mimoX, mimoZ, mimoO, mimoNorm, mimoScanConfig);
	const auto mambaMimoBwd = oa::FnMatrix::mamba3MimoBwd(
		dOut, mimoC, mimoB, x, z, adt, dt, trap, angle, mimoCBias,
		mimoBBias, d, mimoX, mimoZ, mimoO, mimoNorm, mimoScanConfig);
	const auto mambaMimoStep = oa::FnMatrix::mamba3MimoStep(
		mimoStepC, mimoStepB, stepX, stepZ, stepAdt, stepDt, stepTrap,
		stepAngle, mimoCBias, mimoBBias, d, mimoX, mimoZ, mimoO, mimoNorm,
		mimoSsmState, mimoAngleState, mimoKState, mimoVState, mimoStepConfig);
	const auto mambaPreprocess =
		oa::FnMatrix::mamba3Preprocess(projected, dtBias, preprocessConfig);
	const auto mambaPreprocessBwd = oa::FnMatrix::mamba3PreprocessBwd(
		projected, dtBias, dZ, dX, dBh, dCh, dDt, dAdt, dTrap, dAngle,
		preprocessConfig);

	const auto empyrealmSiso = oa::FnMatrix::empyrealmSiso(
		c, b, x, z, adt, dt, trap, angle, cBias, bBias, d, scanConfig);
	const auto empyrealmStep = oa::FnMatrix::empyrealmSisoStep(
		stepC, stepB, stepX, stepZ, stepAdt, stepDt, stepTrap, stepAngle,
		cBias, bBias, d, ssmState, angleState, kState, vState, stepConfig);
	const auto empyrealmBwd = oa::FnMatrix::empyrealmSisoBwd(
		dOut, c, b, x, z, adt, dt, trap, angle, cBias, bBias, d, scanConfig);
	const auto empyrealmAdt =
		oa::FnMatrix::empyrealmAdt(dAdt, dDt, 1.0e-4F);
	const auto empyrealmAdtBwd =
		oa::FnMatrix::empyrealmAdtBwd(dDt, dAdt, dDt, 1.0e-4F);
	const auto empyrealmDt =
		oa::FnMatrix::empyrealmDt(dDt, 1.0e-3F, 1.0e-1F);
	const auto empyrealmDtBwd =
		oa::FnMatrix::empyrealmDtBwd(dDt, dDt, 1.0e-3F, 1.0e-1F);
	const auto empyrealmDtAdt = oa::FnMatrix::empyrealmDtAdt(
		dDt, dAdt, 1.0e-3F, 1.0e-1F, 1.0e-4F);
	const auto empyrealmPreprocess =
		oa::FnMatrix::empyrealmPreprocess(projected, dtBias, preprocessConfig);
	const auto empyrealmPreprocessBwd = oa::FnMatrix::empyrealmPreprocessBwd(
		projected, dtBias, dZ, dX, dBh, dCh, dDt, dAdt, dTrap, dAngle,
		preprocessConfig);

	const auto heavy = oa::FnMatrix::heavyTailActivation(heavyInput);
	const auto assigned = oa::FnMatrix::vqAssign(ze, codebook);
	oa::FnMatrix::vqEmaUpdate(
		ze, assigned.idx, embedSum, clusterSize, codebook,
		0.99F, 1.0e-5F, 1.0F, 17U, false);
	(void)mambaSiso;
	(void)mambaStep;
	(void)mambaBwd;
	(void)mambaMimo;
	(void)mambaMimoBwd;
	(void)mambaMimoStep;
	(void)mambaPreprocess;
	(void)mambaPreprocessBwd;
	(void)empyrealmSiso;
	(void)empyrealmStep;
	(void)empyrealmBwd;
	(void)empyrealmAdt;
	(void)empyrealmAdtBwd;
	(void)empyrealmDt;
	(void)empyrealmDtBwd;
	(void)empyrealmDtAdt;
	(void)empyrealmPreprocess;
	(void)empyrealmPreprocessBwd;
	(void)heavy;

	const oa::OpContract* expected[] = {
		&oa::detail::opRegistry::FnMatrix::mamba3Siso,
		&oa::detail::opRegistry::FnMatrix::mamba3SisoStep,
		&oa::detail::opRegistry::FnMatrix::mamba3SisoBwd,
		&oa::detail::opRegistry::FnMatrix::mamba3Mimo,
		&oa::detail::opRegistry::FnMatrix::mamba3MimoBwd,
		&oa::detail::opRegistry::FnMatrix::mamba3MimoStep,
		&oa::detail::opRegistry::FnMatrix::mamba3Preprocess,
		&oa::detail::opRegistry::FnMatrix::mamba3PreprocessBwd,
		&oa::detail::opRegistry::FnMatrix::empyrealmSiso,
		&oa::detail::opRegistry::FnMatrix::empyrealmSisoStep,
		&oa::detail::opRegistry::FnMatrix::empyrealmSisoBwd,
		&oa::detail::opRegistry::FnMatrix::empyrealmAdt,
		&oa::detail::opRegistry::FnMatrix::empyrealmAdtBwd,
		&oa::detail::opRegistry::FnMatrix::empyrealmDt,
		&oa::detail::opRegistry::FnMatrix::empyrealmDtBwd,
		&oa::detail::opRegistry::FnMatrix::empyrealmDtAdt,
		&oa::detail::opRegistry::FnMatrix::empyrealmPreprocess,
		&oa::detail::opRegistry::FnMatrix::empyrealmPreprocessBwd,
		&oa::detail::opRegistry::FnMatrix::heavyTailActivation,
		&oa::detail::opRegistry::FnMatrix::vqAssign,
		&oa::detail::opRegistry::FnMatrix::vqEmaUpdate,
	};
	constexpr oa::U32 ExpectedOperationCount = 21U;
	constexpr oa::U32 ExpectedNodeCount[] = {
		1U, 1U, 5U, 1U, 5U, 1U, 1U, 1U, 1U,
		1U, 7U, 1U, 1U, 1U, 1U, 1U, 1U, 1U,
		6U, 1U, 1U,
	};

	ASSERT_NE(ctx.semanticGraph(), nullptr);
	ASSERT_TRUE(ctx.semanticGraph()->validate().isOk());
	const auto operations = ctx.semanticGraph()->operations();
	ASSERT_EQ(operations.size(), ExpectedOperationCount);
	for (oa::U32 index = 0U; index < ExpectedOperationCount; ++index) {
		EXPECT_EQ(operations[index].name, expected[index]->name);
		EXPECT_EQ(operations[index].contractHash, expected[index]->hash);
		EXPECT_EQ(
			operations[index].attributes.size(), expected[index]->attributeCount);
		oa::U32 ownedNodes = 0U;
		for (const auto& node : ctx.graph()->nodes()) {
			for (const auto semanticOp : node.semanticOps) {
				if (semanticOp == operations[index].id) ++ownedNodes;
			}
		}
		EXPECT_EQ(ownedNodes, ExpectedNodeCount[index])
			<< operations[index].name.data();
	}
	EXPECT_EQ(operations[1].mutatedInputs.size(), 4U);
	EXPECT_EQ(operations[5].mutatedInputs.size(), 4U);
	EXPECT_EQ(operations[9].mutatedInputs.size(), 4U);
	EXPECT_EQ(operations[20].mutatedInputs.size(), 3U);
	EXPECT_EQ(operations[20].outputs.size(), 0U);

	ASSERT_TRUE(oa::validateSemanticLowering(
		*ctx.semanticGraph(), *ctx.graph()).isOk());
	const auto analyzed = oa::analyzeSemanticLowering(
		*ctx.semanticGraph(), *ctx.graph());
	ASSERT_TRUE(analyzed.isOk()) << analyzed.getStatus().getMessage();
	EXPECT_EQ(analyzed.getValue().directOpCount(), 17U);
	EXPECT_EQ(analyzed.getValue().decomposedOpCount(), 4U);
	EXPECT_EQ(analyzed.getValue().fusedOpCount(), 0U);
	ctx.clear();
}

TEST(ExecutableGraph, ElementwiseSchemaWaveRecordsEveryContract) {
	auto& ctx = oa::ExecutionSession::getActive();
	ctx.clear();
	const auto a = oa::FnMatrix::empty({2, 3}, oa::ScalarType::Float32);
	const auto b = oa::FnMatrix::empty({2, 3}, oa::ScalarType::Float32);
	ctx.clear();

	const auto sub = oa::FnMatrix::sub(a, b);
	const auto mul = oa::FnMatrix::mul(a, b);
	const auto div = oa::FnMatrix::div(a, b);
	const auto neg = oa::FnMatrix::neg(a);
	const auto abs = oa::FnMatrix::abs(a);
	const auto log = oa::FnMatrix::log(a);
	const auto sqrt = oa::FnMatrix::sqrt(a);
	const auto pow = oa::FnMatrix::pow(a, 2.0F);
	const auto addScalar = oa::FnMatrix::addScalar(a, 1.0F);
	const auto subScalar = oa::FnMatrix::subScalar(a, 1.0F);
	const auto divScalar = oa::FnMatrix::divScalar(a, 2.0F);
	const auto exp = oa::FnMatrix::exp(a);
	const auto sin = oa::FnMatrix::sin(a);
	const auto cos = oa::FnMatrix::cos(a);
	const auto reciprocal = oa::FnMatrix::reciprocal(a);
	const auto clampMax = oa::FnMatrix::clampMax(a, 1.0F);
	const auto clampMin = oa::FnMatrix::clampMin(a, -1.0F);
	(void)sub;
	(void)mul;
	(void)div;
	(void)neg;
	(void)abs;
	(void)log;
	(void)sqrt;
	(void)pow;
	(void)addScalar;
	(void)subScalar;
	(void)divScalar;
	(void)exp;
	(void)sin;
	(void)cos;
	(void)reciprocal;
	(void)clampMax;
	(void)clampMin;

	const oa::OpContract* expected[] = {
		&oa::detail::opRegistry::FnMatrix::sub,
		&oa::detail::opRegistry::FnMatrix::mul,
		&oa::detail::opRegistry::FnMatrix::div,
		&oa::detail::opRegistry::FnMatrix::neg,
		&oa::detail::opRegistry::FnMatrix::abs,
		&oa::detail::opRegistry::FnMatrix::log,
		&oa::detail::opRegistry::FnMatrix::sqrt,
		&oa::detail::opRegistry::FnMatrix::pow,
		&oa::detail::opRegistry::FnMatrix::addScalar,
		&oa::detail::opRegistry::FnMatrix::subScalar,
		&oa::detail::opRegistry::FnMatrix::divScalar,
		&oa::detail::opRegistry::FnMatrix::exp,
		&oa::detail::opRegistry::FnMatrix::sin,
		&oa::detail::opRegistry::FnMatrix::cos,
		&oa::detail::opRegistry::FnMatrix::reciprocal,
		&oa::detail::opRegistry::FnMatrix::clampMax,
		&oa::detail::opRegistry::FnMatrix::clampMin,
	};
	constexpr oa::U32 ExpectedCount = 17U;
	ASSERT_EQ(ctx.nodeCount(), ExpectedCount);
	ASSERT_NE(ctx.semanticGraph(), nullptr);
	ASSERT_TRUE(ctx.semanticGraph()->validate().isOk());
	ASSERT_EQ(ctx.semanticGraph()->operationCount(), ExpectedCount);
	const auto nodes = ctx.graph()->nodes();
	const auto operations = ctx.semanticGraph()->operations();
	oa::U32 attributedOperations = 0U;
	for (oa::U32 index = 0U; index < ExpectedCount; ++index) {
		EXPECT_EQ(operations[index].name, expected[index]->name);
		EXPECT_EQ(operations[index].contractHash, expected[index]->hash);
		EXPECT_EQ(operations[index].attributes.size(),
			expected[index]->attributeCount);
		if (not operations[index].attributes.empty()) ++attributedOperations;
		EXPECT_EQ(nodes[index].operation, expected[index]->name);
		EXPECT_EQ(nodes[index].opContractHash, expected[index]->hash);
		ASSERT_EQ(nodes[index].semanticOps.size(), 1U);
		EXPECT_EQ(nodes[index].semanticOps[0], operations[index].id);
	}
	EXPECT_EQ(attributedOperations, 6U);
	const auto analyzed = oa::analyzeSemanticLowering(
		*ctx.semanticGraph(), *ctx.graph());
	ASSERT_TRUE(analyzed.isOk()) << analyzed.getStatus().getMessage();
	EXPECT_EQ(analyzed.getValue().directOpCount(), ExpectedCount);
	EXPECT_EQ(analyzed.getValue().decomposedOpCount(), 0U);
	EXPECT_EQ(analyzed.getValue().fusedOpCount(), 0U);
	ctx.clear();
}

TEST(ExecutableGraph, ActivationSchemaWaveRecordsEveryContract) {
	auto& ctx = oa::ExecutionSession::getActive();
	ctx.clear();
	const auto a = oa::FnMatrix::empty({2, 6}, oa::ScalarType::Float32);
	const auto b = oa::FnMatrix::empty({2, 6}, oa::ScalarType::Float32);
	ctx.clear();

	const auto gelu = oa::FnMatrix::gelu(a);
	const auto silu = oa::FnMatrix::silu(a);
	const auto relu = oa::FnMatrix::relu(a);
	const auto tanh = oa::FnMatrix::tanh(a);
	const auto sigmoid = oa::FnMatrix::sigmoid(a);
	const auto leakyRelu = oa::FnMatrix::leakyRelu(a, 0.125F);
	const auto elu = oa::FnMatrix::elu(a, 1.5F);
	const auto mish = oa::FnMatrix::mish(a);
	const auto swiglu = oa::FnMatrix::swiglu(a, b);
	const auto siluMul = oa::FnMatrix::siluMul(a, 3U);
	const auto geglu = oa::FnMatrix::geglu(a, 3U);
	const auto softplus = oa::FnMatrix::softplus(a);
	(void)gelu;
	(void)silu;
	(void)relu;
	(void)tanh;
	(void)sigmoid;
	(void)leakyRelu;
	(void)elu;
	(void)mish;
	(void)swiglu;
	(void)siluMul;
	(void)geglu;
	(void)softplus;

	const oa::OpContract* expected[] = {
		&oa::detail::opRegistry::FnMatrix::gelu,
		&oa::detail::opRegistry::FnMatrix::silu,
		&oa::detail::opRegistry::FnMatrix::relu,
		&oa::detail::opRegistry::FnMatrix::tanh,
		&oa::detail::opRegistry::FnMatrix::sigmoid,
		&oa::detail::opRegistry::FnMatrix::leakyRelu,
		&oa::detail::opRegistry::FnMatrix::elu,
		&oa::detail::opRegistry::FnMatrix::mish,
		&oa::detail::opRegistry::FnMatrix::swiglu,
		&oa::detail::opRegistry::FnMatrix::siluMul,
		&oa::detail::opRegistry::FnMatrix::geglu,
		&oa::detail::opRegistry::FnMatrix::softplus,
	};
	constexpr oa::U32 ExpectedCount = 12U;
	ASSERT_EQ(ctx.nodeCount(), ExpectedCount);
	ASSERT_NE(ctx.semanticGraph(), nullptr);
	ASSERT_TRUE(ctx.semanticGraph()->validate().isOk());
	ASSERT_EQ(ctx.semanticGraph()->operationCount(), ExpectedCount);
	const auto nodes = ctx.graph()->nodes();
	const auto operations = ctx.semanticGraph()->operations();
	for (oa::U32 index = 0U; index < ExpectedCount; ++index) {
		EXPECT_EQ(operations[index].name, expected[index]->name);
		EXPECT_EQ(operations[index].contractHash, expected[index]->hash);
		EXPECT_EQ(nodes[index].operation, expected[index]->name);
		EXPECT_EQ(nodes[index].opContractHash, expected[index]->hash);
		ASSERT_EQ(nodes[index].semanticOps.size(), 1U);
		EXPECT_EQ(nodes[index].semanticOps[0], operations[index].id);
	}
	ASSERT_EQ(operations[5].attributes.size(), 1U);
	EXPECT_DOUBLE_EQ(operations[5].attributes[0].floatVal, 0.125);
	ASSERT_EQ(operations[6].attributes.size(), 1U);
	EXPECT_DOUBLE_EQ(operations[6].attributes[0].floatVal, 1.5);
	ASSERT_EQ(operations[9].attributes.size(), 1U);
	EXPECT_EQ(operations[9].attributes[0].unsignedInteger, 3U);
	ASSERT_EQ(operations[10].attributes.size(), 1U);
	EXPECT_EQ(operations[10].attributes[0].unsignedInteger, 3U);
	EXPECT_EQ(siluMul.getShape(), (oa::MatrixShape{2, 3}));
	EXPECT_EQ(geglu.getShape(), (oa::MatrixShape{2, 3}));

	const auto analyzed = oa::analyzeSemanticLowering(
		*ctx.semanticGraph(), *ctx.graph());
	ASSERT_TRUE(analyzed.isOk()) << analyzed.getStatus().getMessage();
	EXPECT_EQ(analyzed.getValue().directOpCount(), ExpectedCount);
	EXPECT_EQ(analyzed.getValue().decomposedOpCount(), 0U);
	EXPECT_EQ(analyzed.getValue().fusedOpCount(), 0U);
	ctx.clear();
}

TEST(ExecutableGraph, LinearSchemaRecordsOptionalBiasAndFusedEpilogues) {
	auto& ctx = oa::ExecutionSession::getActive();
	ctx.clear();
	const auto input = oa::FnMatrix::empty({2, 3, 4}, oa::ScalarType::Float32);
	const auto weight = oa::FnMatrix::empty({5, 4}, oa::ScalarType::Float32);
	const auto bias = oa::FnMatrix::empty({5}, oa::ScalarType::Float32);
	ctx.clear();

	const auto withoutBias = oa::FnMatrix::linear(input, weight);
	const auto withBias = oa::FnMatrix::linear(input, weight, bias);
	const auto withRelu = oa::FnMatrix::linearRelu(input, weight, bias);
	const auto withGelu = oa::FnMatrix::linearGelu(input, weight, bias);

	EXPECT_EQ(withoutBias.getShape(), (oa::MatrixShape{2, 3, 5}));
	EXPECT_EQ(withBias.getShape(), (oa::MatrixShape{2, 3, 5}));
	EXPECT_EQ(withRelu.getShape(), (oa::MatrixShape{2, 3, 5}));
	EXPECT_EQ(withGelu.getShape(), (oa::MatrixShape{2, 3, 5}));
	ASSERT_NE(ctx.semanticGraph(), nullptr);
	ASSERT_TRUE(ctx.semanticGraph()->validate().isOk());

	const auto operations = ctx.semanticGraph()->operations();
	const oa::OpContract* expected[] = {
		&oa::detail::opRegistry::FnMatrix::linear,
		&oa::detail::opRegistry::FnMatrix::linear,
		&oa::detail::opRegistry::FnMatrix::linearRelu,
		&oa::detail::opRegistry::FnMatrix::linearGelu,
	};
	ASSERT_EQ(operations.size(), 4U);
	for (oa::U32 index = 0U; index < 4U; ++index) {
		EXPECT_EQ(operations[index].name, expected[index]->name);
		EXPECT_EQ(operations[index].contractHash, expected[index]->hash);
		EXPECT_EQ(operations[index].inputs.size(), 3U);
		EXPECT_EQ(operations[index].outputs.size(), 1U);
		EXPECT_EQ(operations[index].lowering, oa::OpLowering::Gemm);
	}
	EXPECT_EQ(operations[0].optionalInputMask, 0x04U);
	EXPECT_EQ(operations[0].inputs[2], oa::invalidSemanticValueId);
	EXPECT_NE(operations[1].inputs[2], oa::invalidSemanticValueId);

	const auto nodes = ctx.graph()->nodes();
	ASSERT_EQ(nodes.size(), 4U);
	for (oa::U32 index = 0U; index < 4U; ++index) {
		EXPECT_EQ(nodes[index].operation, expected[index]->name);
		EXPECT_EQ(nodes[index].opContractHash, expected[index]->hash);
		ASSERT_EQ(nodes[index].semanticOps.size(), 1U);
		EXPECT_EQ(nodes[index].semanticOps[0], operations[index].id);
		EXPECT_NE(nodes[index].problemContractHash, 0U);
	}
	const auto analyzed = oa::analyzeSemanticLowering(
		*ctx.semanticGraph(), *ctx.graph());
	ASSERT_TRUE(analyzed.isOk()) << analyzed.getStatus().getMessage();
	EXPECT_EQ(analyzed.getValue().directOpCount(), 4U);
	EXPECT_EQ(analyzed.getValue().decomposedOpCount(), 0U);
	EXPECT_EQ(analyzed.getValue().fusedOpCount(), 0U);
	ctx.clear();
}

TEST(ExecutableGraph, GatherSchemaPairRecordsForwardAndAdjointContracts) {
	auto& ctx = oa::ExecutionSession::getActive();
	ctx.clear();
	const auto table = oa::FnMatrix::empty({4, 3}, oa::ScalarType::Float32);
	const auto indices = oa::FnMatrix::empty({2}, oa::ScalarType::UInt32);
	ctx.clear();

	const auto gathered = oa::FnMatrix::gather(table, indices);
	const auto gradTable = oa::FnMatrix::gatherBwd(indices, gathered, 4, 3);
	(void)gradTable;

	ASSERT_NE(ctx.semanticGraph(), nullptr);
	ASSERT_TRUE(ctx.semanticGraph()->validate().isOk());
	const auto operations = ctx.semanticGraph()->operations();
	ASSERT_EQ(operations.size(), 2U);
	EXPECT_EQ(operations[0].name,
		oa::detail::opRegistry::FnMatrix::gather.name);
	EXPECT_EQ(operations[1].name,
		oa::detail::opRegistry::FnMatrix::gatherBwd.name);
	ASSERT_EQ(operations[1].attributes.size(), 2U);
	EXPECT_EQ(operations[1].attributes[0].signedInteger, 4);
	EXPECT_EQ(operations[1].attributes[1].signedInteger, 3);

	const auto nodes = ctx.graph()->nodes();
	ASSERT_EQ(nodes.size(), 2U);
	for (oa::U32 index = 0U; index < 2U; ++index) {
		ASSERT_EQ(nodes[index].semanticOps.size(), 1U);
		EXPECT_EQ(nodes[index].semanticOps[0], operations[index].id);
		EXPECT_EQ(nodes[index].opContractHash,
			operations[index].contractHash);
	}
	const auto analyzed = oa::analyzeSemanticLowering(
		*ctx.semanticGraph(), *ctx.graph());
	ASSERT_TRUE(analyzed.isOk()) << analyzed.getStatus().getMessage();
	EXPECT_EQ(analyzed.getValue().directOpCount(), 2U);
	EXPECT_EQ(analyzed.getValue().decomposedOpCount(), 0U);
	ctx.clear();
}

TEST(ExecutableGraph, ReductionSchemaFamilyRecordsContractsAndDimensions) {
	auto& ctx = oa::ExecutionSession::getActive();
	ctx.clear();
	const auto input = oa::FnMatrix::empty({2, 3}, oa::ScalarType::Float32);
	ctx.clear();

	const auto fullSum = oa::FnMatrix::sum(input);
	const auto axisSum = oa::FnMatrix::sum(input, 1);
	const auto fullMax = oa::FnMatrix::max(input);
	(void)fullSum;
	(void)axisSum;
	(void)fullMax;

	ASSERT_NE(ctx.semanticGraph(), nullptr);
	ASSERT_TRUE(ctx.semanticGraph()->validate().isOk());
	const auto operations = ctx.semanticGraph()->operations();
	ASSERT_EQ(operations.size(), 3U);
	EXPECT_EQ(operations[0].name, oa::detail::opRegistry::FnMatrix::sum.name);
	EXPECT_EQ(operations[0].contractHash, oa::detail::opRegistry::FnMatrix::sum.hash);
	ASSERT_EQ(operations[0].attributes.size(), 1U);
	EXPECT_EQ(operations[0].attributes[0].name, "dim");
	EXPECT_EQ(operations[0].attributes[0].kind,
		oa::OpAttributeKind::SignedInteger);
	EXPECT_EQ(operations[0].attributes[0].signedInteger, -1);
	EXPECT_EQ(operations[1].name, oa::detail::opRegistry::FnMatrix::sum.name);
	EXPECT_EQ(operations[1].contractHash, oa::detail::opRegistry::FnMatrix::sum.hash);
	ASSERT_EQ(operations[1].attributes.size(), 1U);
	EXPECT_EQ(operations[1].attributes[0].signedInteger, 1);
	EXPECT_EQ(operations[2].name, oa::detail::opRegistry::FnMatrix::max.name);
	EXPECT_EQ(operations[2].contractHash, oa::detail::opRegistry::FnMatrix::max.hash);
	EXPECT_TRUE(operations[2].attributes.empty());

	oa::U32 ownedNodeCount = 0U;
	for (const auto& node : ctx.graph()->nodes()) {
		if (node.semanticOps.empty()) continue;
		ASSERT_EQ(node.semanticOps.size(), 1U);
		const auto operationId = node.semanticOps[0];
		ASSERT_LT(operationId, operations.size());
		EXPECT_EQ(node.operation, operations[operationId].name);
		EXPECT_EQ(node.opContractHash,
			operations[operationId].contractHash);
		++ownedNodeCount;
	}
	EXPECT_EQ(ownedNodeCount, 3U);
	const auto analyzed = oa::analyzeSemanticLowering(
		*ctx.semanticGraph(), *ctx.graph());
	ASSERT_TRUE(analyzed.isOk()) << analyzed.getStatus().getMessage();
	EXPECT_EQ(analyzed.getValue().directOpCount(), 3U);
	EXPECT_EQ(analyzed.getValue().decomposedOpCount(), 0U);
	EXPECT_EQ(analyzed.getValue().fusedOpCount(), 0U);
	ctx.clear();
}

TEST(ExecutableGraph, NormalizationSchemaSliceRecordsEpsilonAndProvenance) {
	auto& ctx = oa::ExecutionSession::getActive();
	ctx.clear();
	const auto input = oa::FnMatrix::empty({2, 3}, oa::ScalarType::Float32);
	const auto weight = oa::FnMatrix::empty({3}, oa::ScalarType::Float32);
	const auto bias = oa::FnMatrix::empty({3}, oa::ScalarType::Float32);
	ctx.clear();

	const auto layer = oa::FnMatrix::layerNorm(input, weight, bias, 0.125F);
	const auto rms = oa::FnMatrix::rmsNorm(input, weight, 0.25F);
	(void)layer;
	(void)rms;

	ASSERT_NE(ctx.semanticGraph(), nullptr);
	ASSERT_TRUE(ctx.semanticGraph()->validate().isOk());
	const auto operations = ctx.semanticGraph()->operations();
	ASSERT_EQ(operations.size(), 2U);
	EXPECT_EQ(operations[0].name, oa::detail::opRegistry::FnMatrix::layerNorm.name);
	EXPECT_EQ(operations[0].contractHash,
		oa::detail::opRegistry::FnMatrix::layerNorm.hash);
	ASSERT_EQ(operations[0].attributes.size(), 1U);
	EXPECT_EQ(operations[0].attributes[0].name, "eps");
	EXPECT_EQ(operations[0].attributes[0].kind,
		oa::OpAttributeKind::Float);
	EXPECT_EQ(operations[0].attributes[0].floatVal, 0.125);
	EXPECT_EQ(operations[1].name, oa::detail::opRegistry::FnMatrix::rmsNorm.name);
	EXPECT_EQ(operations[1].contractHash, oa::detail::opRegistry::FnMatrix::rmsNorm.hash);
	ASSERT_EQ(operations[1].attributes.size(), 1U);
	EXPECT_EQ(operations[1].attributes[0].name, "eps");
	EXPECT_EQ(operations[1].attributes[0].kind,
		oa::OpAttributeKind::Float);
	EXPECT_EQ(operations[1].attributes[0].floatVal, 0.25);

	oa::U32 ownedNodeCount = 0U;
	for (const auto& node : ctx.graph()->nodes()) {
		if (node.semanticOps.empty()) continue;
		ASSERT_EQ(node.semanticOps.size(), 1U);
		const auto operationId = node.semanticOps[0];
		ASSERT_LT(operationId, operations.size());
		EXPECT_EQ(node.operation, operations[operationId].name);
		EXPECT_EQ(node.opContractHash,
			operations[operationId].contractHash);
		++ownedNodeCount;
	}
	EXPECT_EQ(ownedNodeCount, 2U);
	const auto analyzed = oa::analyzeSemanticLowering(
		*ctx.semanticGraph(), *ctx.graph());
	ASSERT_TRUE(analyzed.isOk()) << analyzed.getStatus().getMessage();
	EXPECT_EQ(analyzed.getValue().directOpCount(), 2U);
	EXPECT_EQ(analyzed.getValue().decomposedOpCount(), 0U);
	EXPECT_EQ(analyzed.getValue().fusedOpCount(), 0U);
	ctx.clear();
}

TEST(ExecutableGraph, SoftmaxSchemaFamilyRecordsDimensionsAndProvenance) {
	auto& ctx = oa::ExecutionSession::getActive();
	ctx.clear();
	const auto input = oa::FnMatrix::empty({2, 3, 4}, oa::ScalarType::Float32);
	ctx.clear();

	const auto softmax = oa::FnMatrix::softmax(input, 1);
	const auto logSoftmax = oa::FnMatrix::logSoftmax(input, 0);
	(void)softmax;
	(void)logSoftmax;

	ASSERT_NE(ctx.semanticGraph(), nullptr);
	ASSERT_TRUE(ctx.semanticGraph()->validate().isOk());
	const auto operations = ctx.semanticGraph()->operations();
	ASSERT_EQ(operations.size(), 2U);
	EXPECT_EQ(operations[0].name, oa::detail::opRegistry::FnMatrix::softmax.name);
	EXPECT_EQ(operations[0].contractHash, oa::detail::opRegistry::FnMatrix::softmax.hash);
	ASSERT_EQ(operations[0].attributes.size(), 1U);
	EXPECT_EQ(operations[0].attributes[0].name, "dim");
	EXPECT_EQ(operations[0].attributes[0].kind,
		oa::OpAttributeKind::SignedInteger);
	EXPECT_EQ(operations[0].attributes[0].signedInteger, 1);
	EXPECT_EQ(operations[1].name, oa::detail::opRegistry::FnMatrix::logSoftmax.name);
	EXPECT_EQ(operations[1].contractHash,
		oa::detail::opRegistry::FnMatrix::logSoftmax.hash);
	ASSERT_EQ(operations[1].attributes.size(), 1U);
	EXPECT_EQ(operations[1].attributes[0].name, "dim");
	EXPECT_EQ(operations[1].attributes[0].kind,
		oa::OpAttributeKind::SignedInteger);
	EXPECT_EQ(operations[1].attributes[0].signedInteger, 0);

	oa::U32 ownedNodeCount = 0U;
	for (const auto& node : ctx.graph()->nodes()) {
		if (node.semanticOps.empty()) continue;
		ASSERT_EQ(node.semanticOps.size(), 1U);
		const auto operationId = node.semanticOps[0];
		ASSERT_LT(operationId, operations.size());
		EXPECT_EQ(node.operation, operations[operationId].name);
		EXPECT_EQ(node.opContractHash,
			operations[operationId].contractHash);
		++ownedNodeCount;
	}
	EXPECT_EQ(ownedNodeCount, 2U);
	const auto analyzed = oa::analyzeSemanticLowering(
		*ctx.semanticGraph(), *ctx.graph());
	ASSERT_TRUE(analyzed.isOk()) << analyzed.getStatus().getMessage();
	EXPECT_EQ(analyzed.getValue().directOpCount(), 2U);
	EXPECT_EQ(analyzed.getValue().decomposedOpCount(), 0U);
	EXPECT_EQ(analyzed.getValue().fusedOpCount(), 0U);
	ctx.clear();
}

TEST(ExecutableGraph, MeanSchemaRecordsOneDecomposedAxisOperation) {
	auto& ctx = oa::ExecutionSession::getActive();
	ctx.clear();
	const auto input = oa::FnMatrix::empty({2, 3, 4}, oa::ScalarType::Float32);
	ctx.clear();

	const auto output = oa::FnMatrix::mean(input, 1);
	ASSERT_EQ(output.getShape(), (oa::MatrixShape{2, 1, 4}));
	ASSERT_NE(ctx.semanticGraph(), nullptr);
	ASSERT_TRUE(ctx.semanticGraph()->validate().isOk());
	const auto operations = ctx.semanticGraph()->operations();
	ASSERT_EQ(operations.size(), 1U);
	EXPECT_EQ(operations[0].name, oa::detail::opRegistry::FnMatrix::mean.name);
	EXPECT_EQ(operations[0].contractHash, oa::detail::opRegistry::FnMatrix::mean.hash);
	ASSERT_EQ(operations[0].attributes.size(), 1U);
	EXPECT_EQ(operations[0].attributes[0].name, "dim");
	EXPECT_EQ(operations[0].attributes[0].kind,
		oa::OpAttributeKind::SignedInteger);
	EXPECT_EQ(operations[0].attributes[0].signedInteger, 1);

	const auto& nodes = ctx.graph()->nodes();
	ASSERT_EQ(nodes.size(), 2U);
	EXPECT_EQ(nodes[0].shader, "SumDim");
	EXPECT_EQ(nodes[1].shader, "Scale");
	for (const auto& node : nodes) {
		ASSERT_EQ(node.semanticOps.size(), 1U);
		EXPECT_EQ(node.semanticOps[0], 0U);
		EXPECT_EQ(node.operation, oa::detail::opRegistry::FnMatrix::mean.name);
		EXPECT_EQ(node.opContractHash, oa::detail::opRegistry::FnMatrix::mean.hash);
	}
	const auto analyzed = oa::analyzeSemanticLowering(
		*ctx.semanticGraph(), *ctx.graph());
	ASSERT_TRUE(analyzed.isOk()) << analyzed.getStatus().getMessage();
	EXPECT_EQ(analyzed.getValue().directOpCount(), 0U);
	EXPECT_EQ(analyzed.getValue().decomposedOpCount(), 1U);
	EXPECT_EQ(analyzed.getValue().maximumNodesPerOp(), 2U);
	ctx.clear();
}

TEST(ExecutableGraph, CrossEntropySchemaOwnsClassificationAndMeanDecomposition) {
	auto& ctx = oa::ExecutionSession::getActive();
	ctx.clear();
	auto logits = oa::FnMatrix::empty({2, 3}, oa::ScalarType::Float32);
	const auto targets = oa::FnMatrix::empty({2}, oa::ScalarType::UInt32);
	logits.setRequiresGrad(true);
	ctx.clear();

	oa::GradientTape tape;
	const auto loss = oa::FnLoss::crossEntropy(logits, targets);
	ASSERT_FALSE(loss.isEmpty());
	EXPECT_EQ(loss.getShape(), (oa::MatrixShape{1}));
	EXPECT_EQ(loss.getDtype(), oa::ScalarType::Float32);

	const auto* semantic = ctx.semanticGraph();
	ASSERT_NE(semantic, nullptr);
	ASSERT_TRUE(semantic->validate().isOk());
	const auto operations = semantic->operations();
	ASSERT_EQ(operations.size(), 1U);
	EXPECT_EQ(operations[0].name, oa::detail::opRegistry::FnLoss::crossEntropy.name);
	EXPECT_EQ(operations[0].contractHash,
		oa::detail::opRegistry::FnLoss::crossEntropy.hash);
	EXPECT_TRUE(operations[0].attributes.empty());
	ASSERT_EQ(operations[0].inputs.size(), 2U);
	ASSERT_EQ(operations[0].outputs.size(), 1U);
	const auto* outputValue = semantic->findValue(operations[0].outputs[0]);
	ASSERT_NE(outputValue, nullptr);
	EXPECT_EQ(outputValue->shape, (oa::MatrixShape{1}));
	EXPECT_EQ(outputValue->dtype, oa::ScalarType::Float32);

	const auto& nodes = ctx.graph()->nodes();
	ASSERT_EQ(nodes.size(), 3U);
	EXPECT_EQ(nodes[0].shader, "CrossEntropy");
	EXPECT_EQ(nodes[1].shader, "Sum");
	EXPECT_EQ(nodes[2].shader, "Scale");
	for (const auto& node : nodes) {
		EXPECT_EQ(node.operation, oa::detail::opRegistry::FnLoss::crossEntropy.name);
		EXPECT_EQ(node.opContractHash,
			oa::detail::opRegistry::FnLoss::crossEntropy.hash);
		ASSERT_EQ(node.semanticOps.size(), 1U);
		EXPECT_EQ(node.semanticOps[0], operations[0].id);
	}

	ASSERT_EQ(semantic->autograd().size(), 1U);
	EXPECT_EQ(semantic->autograd()[0].forwardOp, operations[0].id);
	ASSERT_TRUE(loss.getGradFn());
	EXPECT_EQ(loss.getGradFn()->forwardSemanticOp_, operations[0].id);
	const auto analyzed = oa::analyzeSemanticLowering(*semantic, *ctx.graph());
	ASSERT_TRUE(analyzed.isOk()) << analyzed.getStatus().getMessage();
	EXPECT_EQ(analyzed.getValue().directOpCount(), 0U);
	EXPECT_EQ(analyzed.getValue().decomposedOpCount(), 1U);
	EXPECT_EQ(analyzed.getValue().maximumNodesPerOp(), 3U);
	ctx.clear();
}

TEST(ExecutableGraph, CrossEntropySplitsRowsAcrossPortableDispatchDimensions) {
	auto& ctx = oa::ExecutionSession::getActive();
	ctx.clear();
	constexpr oa::I64 kRows = 65536;
	const auto logits = oa::FnMatrix::empty({kRows, 1}, oa::ScalarType::Float32);
	const auto targets = oa::FnMatrix::empty({kRows}, oa::ScalarType::UInt32);

	const auto loss = oa::FnLoss::crossEntropy(logits, targets);
	ASSERT_FALSE(loss.isEmpty());
	const auto& forwardNodes = ctx.graph()->nodes();
	ASSERT_EQ(forwardNodes.size(), 3U);
	EXPECT_EQ(forwardNodes[0].shader, "CrossEntropy");
	EXPECT_EQ(forwardNodes[0].groupsX, 65535U);
	EXPECT_EQ(forwardNodes[0].groupsY, 2U);

	ctx.clear();
	const auto gradient = oa::FnLoss::crossEntropyBwd(logits, targets);
	ASSERT_FALSE(gradient.isEmpty());
	const auto& backwardNodes = ctx.graph()->nodes();
	ASSERT_EQ(backwardNodes.size(), 1U);
	EXPECT_EQ(backwardNodes[0].shader, "CrossEntropyBwd");
	EXPECT_EQ(backwardNodes[0].groupsX, 65535U);
	EXPECT_EQ(backwardNodes[0].groupsY, 2U);
	ctx.clear();
}

TEST(ExecutableGraph, SemanticPilotPreservesDataflowAcrossOperations) {
	auto& ctx = oa::ExecutionSession::getActive();
	ctx.clear();
	const auto a = oa::FnMatrix::empty({2, 3}, oa::ScalarType::Float32);
	const auto b = oa::FnMatrix::empty({2, 3}, oa::ScalarType::Float32);
	const auto weight = oa::FnMatrix::empty({4, 3}, oa::ScalarType::Float32);
	ctx.clear();
	const auto sum = oa::FnMatrix::add(a, b);
	const auto product = oa::FnMatrix::matMulNt(
		sum, weight, oa::MatMulPrecision::Fp32);
	(void)product;

	const auto* semantic = ctx.semanticGraph();
	ASSERT_NE(semantic, nullptr);
	ASSERT_TRUE(semantic->validate().isOk());
	ASSERT_EQ(semantic->operationCount(), 2U);
	ASSERT_EQ(semantic->valueCount(), 5U);
	const auto operations = semantic->operations();
	ASSERT_EQ(operations[0].outputs.size(), 1U);
	ASSERT_EQ(operations[1].inputs.size(), 2U);
	EXPECT_EQ(operations[0].outputs[0], operations[1].inputs[0]);
	EXPECT_EQ(semantic->findValue(operations[1].inputs[0])->producer, 0U);
	const auto executable = ctx.graph()->nodes();
	ASSERT_EQ(executable.size(), 2U);
	ASSERT_EQ(executable[0].semanticOps.size(), 1U);
	ASSERT_EQ(executable[1].semanticOps.size(), 1U);
	EXPECT_EQ(executable[0].semanticOps[0], operations[0].id);
	EXPECT_EQ(executable[1].semanticOps[0], operations[1].id);
	ctx.clear();
}

TEST(ExecutableGraph, SemanticRecordingVersionsInPlaceStorage) {
	auto& ctx = oa::ExecutionSession::getActive();
	ctx.clear();
	const auto value = oa::FnMatrix::empty({4}, oa::ScalarType::Float32);
	ctx.clear();

	const oa::OpContract inPlace{
		.name = "TestInPlace",
		.hash = 0x91c4d14f09f6c7b1ULL,
		.inputKinds = static_cast<oa::U32>(oa::OpValueKind::Matrix),
		.outputKinds = static_cast<oa::U32>(oa::OpValueKind::Matrix),
		.inputCount = 1,
		.outputCount = 1,
		.effects = oa::OpEffect::ReadInputs | oa::OpEffect::WriteOutputs,
		.mutatedInputMask = 0x01U,
		.outputAliasInputs = 0xfffffff0U,
		.controlFlow = oa::OpControlFlow::StraightLine,
	};
	ASSERT_TRUE(ctx.recordOp( inPlace, {&value}, {&value}).isOk());
	ASSERT_TRUE(ctx.recordOp( inPlace, {&value}, {&value}).isOk());

	const auto* semantic = ctx.semanticGraph();
	ASSERT_NE(semantic, nullptr);
	ASSERT_TRUE(semantic->validate().isOk());
	ASSERT_EQ(semantic->valueCount(), 3U);
	ASSERT_EQ(semantic->operationCount(), 2U);
	const auto operations = semantic->operations();
	EXPECT_EQ(operations[0].inputs[0], 0U);
	EXPECT_EQ(operations[0].outputs[0], 1U);
	EXPECT_EQ(operations[1].inputs[0], 1U);
	EXPECT_EQ(operations[1].outputs[0], 2U);
	ASSERT_EQ(operations[0].mutatedInputs.size(), 1U);
	EXPECT_EQ(operations[0].mutatedInputs[0], 0U);
	ASSERT_EQ(operations[0].aliases.size(), 1U);
	EXPECT_EQ(operations[0].aliases[0].input, 0U);
	EXPECT_EQ(operations[0].aliases[0].output, 1U);
	EXPECT_EQ(operations[0].accesses[0].mode, oa::SemanticAccessMode::ReadWrite);
	ctx.clear();
}

TEST(ExecutableGraph, SemanticRecordingPreservesNonMatrixValueKinds) {
	auto& ctx = oa::ExecutionSession::getActive();
	ctx.clear();
	const auto input = oa::FnMatrix::empty({1, 16}, oa::ScalarType::Float32);
	const auto output = oa::FnMatrix::empty({1, 16}, oa::ScalarType::Float32);
	ctx.clear();

	const auto audioKind = static_cast<oa::U32>(oa::OpValueKind::Audio);
	const oa::OpContract contract{
		.name = "TestAudioSemanticValue",
		.hash = 0x4529d843d83ea1c1ULL,
		.inputKinds = audioKind,
		.outputKinds = audioKind,
		.inputCount = 1,
		.outputCount = 1,
		.effects = oa::OpEffect::ReadInputs | oa::OpEffect::WriteOutputs,
	};
	const auto recorded = ctx.recordOp(
		contract, {&input}, {&output});
	ASSERT_TRUE(recorded.isOk()) << recorded.getStatus().getMessage();

	const auto* semantic = ctx.semanticGraph();
	ASSERT_NE(semantic, nullptr);
	ASSERT_TRUE(semantic->validate().isOk());
	ASSERT_EQ(semantic->valueCount(), 2U);
	EXPECT_EQ(semantic->values()[0].kind, oa::OpValueKind::Audio);
	EXPECT_EQ(semantic->values()[1].kind, oa::OpValueKind::Audio);
	auto dnnPlan = oa::DnnPlanner::plan(*semantic);
	ASSERT_TRUE(dnnPlan.isOk()) << dnnPlan.getStatus().getMessage();
	EXPECT_NE(dnnPlan.getValue().graphHash, 0U);
	EXPECT_EQ(dnnPlan.getValue().sourceOpCount, 1U);
	EXPECT_EQ(dnnPlan.getValue().capturedOpCount, 0U);
	EXPECT_TRUE(dnnPlan.getValue().partitions.empty());
	ctx.clear();
}

TEST(ExecutableGraph, FailedLoweringDiscardsBothGraphRepresentations) {
	auto& ctx = oa::ExecutionSession::getActive();
	ctx.clear();
	const auto value = oa::FnMatrix::empty({4}, oa::ScalarType::Float32);
	ctx.clear();

	const oa::OpContract contract{
		.name = "TestMissingLowering",
		.hash = 0x8c8bdd2db4c59ec3ULL,
		.inputKinds = static_cast<oa::U32>(oa::OpValueKind::Matrix),
		.outputKinds = static_cast<oa::U32>(oa::OpValueKind::Matrix),
		.inputCount = 1,
		.outputCount = 1,
		.effects = oa::OpEffect::ReadInputs | oa::OpEffect::WriteOutputs,
	};
	ASSERT_TRUE(ctx.recordOp( contract, {&value}, {&value}).isOk());

	oa::ComputeDispatchDesc missing;
	missing.kernel = "TestKernelThatDoesNotExist";
	ASSERT_TRUE(ctx.record( missing).isOk());
	ASSERT_EQ(ctx.nodeCount(), 1U);
	ASSERT_EQ(ctx.semanticGraph()->operationCount(), 1U);

	const auto status = testSubmitAndWait(ctx);
	ASSERT_FALSE(status.isOk());
	EXPECT_EQ(ctx.nodeCount(), 0U);
	EXPECT_EQ(ctx.semanticGraph()->operationCount(), 0U);
	EXPECT_EQ(ctx.semanticGraph()->valueCount(), 0U);
	ctx.clear();
}

TEST(ExecutableGraph, RecordingFailureRollsBackBothGraphsAndStaysSticky) {
	auto& ctx = oa::ExecutionSession::getActive();
	ctx.clear();
	const auto a = oa::FnMatrix::empty({4}, oa::ScalarType::Float32);
	const auto b = oa::FnMatrix::empty({4}, oa::ScalarType::Float32);
	ctx.clear();

	const auto prefix = oa::FnMatrix::add(a, b);
	ASSERT_FALSE(prefix.isEmpty());
	ASSERT_EQ(ctx.nodeCount(), 1U);
	ASSERT_EQ(ctx.semanticGraph()->operationCount(), 1U);

	oa::ComputeDispatchDesc invalid;
	invalid.kernel = "Scale";
	oavk::Buffer invalidBuffers[] = {oa::MatrixAccess::descriptor(a)};
	invalid.buffers = oa::Span<oavk::Buffer>(invalidBuffers, 1U);
	const auto failure = ctx.record( invalid);
	ASSERT_FALSE(failure.isOk());
	EXPECT_EQ(failure.getCode(), oa::StatusCode::InvalidArgument);
	EXPECT_EQ(ctx.nodeCount(), 0U);
	EXPECT_EQ(ctx.semanticGraph()->operationCount(), 0U);
	EXPECT_EQ(ctx.semanticGraph()->valueCount(), 0U);

	const auto rejected = ctx.recordOp(
		oa::detail::opRegistry::FnMatrix::add, {&a, &b}, {&prefix});
	ASSERT_FALSE(rejected.isOk());
	EXPECT_EQ(rejected.getStatus().getCode(), failure.getCode());
	EXPECT_EQ(rejected.getStatus().getMessage(), failure.getMessage());
	EXPECT_EQ(ctx.nodeCount(), 0U);
	EXPECT_EQ(ctx.semanticGraph()->operationCount(), 0U);

	const auto surfaced = testSubmitAndWait(ctx);
	EXPECT_EQ(surfaced.getCode(), failure.getCode());
	EXPECT_EQ(surfaced.getMessage(), failure.getMessage());

	const auto recovered = oa::FnMatrix::add(a, b);
	ASSERT_FALSE(recovered.isEmpty());
	ASSERT_EQ(ctx.nodeCount(), 1U);
	ASSERT_EQ(ctx.semanticGraph()->operationCount(), 1U);
	ASSERT_TRUE(testSubmitAndWait(ctx).isOk());
	ctx.clear();
}

TEST(ExecutableGraph, MatrixDispatchPreflightFailureAbortsRecordingTransaction) {
	auto& ctx = oa::ExecutionSession::getActive();
	ctx.clear();
	const auto input = oa::FnMatrix::empty({4}, oa::ScalarType::Float32);
	const auto output = oa::FnMatrix::empty({4}, oa::ScalarType::Float32);
	ctx.clear();

	const auto prefix = oa::FnMatrix::scale(input, 3.0F);
	ASSERT_FALSE(prefix.isEmpty());
	ASSERT_EQ(ctx.nodeCount(), 1U);
	ASSERT_EQ(ctx.semanticGraph()->operationCount(), 1U);

	oa::MatrixDispatchDesc invalid;
	invalid.dispatch.kernel = "Scale";
	oavk::Buffer rawBuffers[] = {oa::MatrixAccess::descriptor(input)};
	invalid.dispatch.buffers = oa::Span<oavk::Buffer>(rawBuffers, 1U);
	const oa::Matrix* matrices[] = {&input, &output};
	invalid.matrices = oa::Span<const oa::Matrix* const>(matrices, 2U);
	const auto failure = ctx.record( invalid);
	ASSERT_FALSE(failure.isOk());
	EXPECT_EQ(failure.getCode(), oa::StatusCode::InvalidArgument);
	EXPECT_EQ(ctx.nodeCount(), 0U);
	EXPECT_EQ(ctx.semanticGraph()->operationCount(), 0U);
	EXPECT_EQ(ctx.semanticGraph()->valueCount(), 0U);

	const auto surfaced = ctx.submit();
	ASSERT_FALSE(surfaced.isOk());
	EXPECT_EQ(surfaced.getStatus().getCode(), failure.getCode());
	EXPECT_EQ(surfaced.getStatus().getMessage(), failure.getMessage());
	ctx.clear();
}

TEST(ExecutableGraph, SemanticAuthoringFailureRollsBackPartialValues) {
	auto& ctx = oa::ExecutionSession::getActive();
	ctx.clear();
	const auto input = oa::FnMatrix::empty({4}, oa::ScalarType::Float32);
	const auto output = oa::FnMatrix::empty({4}, oa::ScalarType::Float32);
	const oa::Matrix missing;
	ctx.clear();

	const auto matrixKind = static_cast<oa::U32>(oa::OpValueKind::Matrix);
	const oa::OpContract contract{
		.name = "TestTwoOutputAuthoringFailure",
		.hash = 0x78f48ced67af915bULL,
		.inputKinds = matrixKind,
		.outputKinds = matrixKind | (matrixKind << 4U),
		.inputCount = 1,
		.outputCount = 2,
		.effects = oa::OpEffect::ReadInputs | oa::OpEffect::WriteOutputs,
	};
	const auto failure = ctx.recordOp(
		contract, {&input}, {&output, &missing});
	ASSERT_FALSE(failure.isOk());
	EXPECT_EQ(failure.getStatus().getCode(), oa::StatusCode::InvalidArgument);
	EXPECT_EQ(ctx.nodeCount(), 0U);
	EXPECT_EQ(ctx.semanticGraph()->operationCount(), 0U);
	EXPECT_EQ(ctx.semanticGraph()->valueCount(), 0U);

	const auto surfaced = testSubmitAndWait(ctx);
	EXPECT_EQ(surfaced.getCode(), failure.getStatus().getCode());
	EXPECT_EQ(surfaced.getMessage(), failure.getStatus().getMessage());
	ctx.clear();
}

TEST(ExecutableGraph, RecordingFailureCancelsUnsubmittedBatchPrefix) {
	auto& ctx = oa::ExecutionSession::getActive();
	ctx.clear();
	const auto input = oa::FnMatrix::ones({8}, oa::ScalarType::Float32);
	const auto destination = oa::FnMatrix::zeros({8}, oa::ScalarType::Float32);
	ASSERT_TRUE(testSubmitAndWait(ctx).isOk());
	ctx.clear();

	struct Push { oa::U32 count; oa::F32 scale; } push{8U, 7.0F};
	oa::BufferAccess access[] = {oa::BufferAccess::Read, oa::BufferAccess::Write};
	ctx.add( "Scale", {&input, &destination}, access, &push, sizeof(push), 1U);
	ASSERT_GT(ctx.nodeCount(), 0U);

	oa::ComputeDispatchDesc invalid;
	invalid.kernel = "Scale";
	oavk::Buffer invalidBuffers[] = {oa::MatrixAccess::descriptor(input)};
	invalid.buffers = oa::Span<oavk::Buffer>(invalidBuffers, 1U);
	const auto failure = ctx.record( invalid);
	ASSERT_FALSE(failure.isOk());
	EXPECT_EQ(ctx.nodeCount(), 0U);
	EXPECT_EQ(ctx.semanticGraph()->operationCount(), 0U);

	const auto submitted = ctx.submit();
	ASSERT_FALSE(submitted.isOk());
	EXPECT_EQ(submitted.getStatus().getCode(), failure.getCode());
	EXPECT_EQ(submitted.getStatus().getMessage(), failure.getMessage());
	ASSERT_TRUE(testSubmitAndWait(ctx).isOk());

	oa::F32 values[8]{};
	ASSERT_TRUE(oa::FnMatrix::copyToHost(
		destination, values, sizeof(values)).isOk());
	for (const oa::F32 value : values) EXPECT_FLOAT_EQ(value, 0.0F);
	ctx.clear();
}

TEST(ExecutableGraph, ExplicitRecordingSubmitAndWait) {
	auto& ctx = oa::ExecutionSession::getActive();
	ctx.clear();
	const oa::F32 aValues[6]{1.0F, 1.0F, 1.0F, 1.0F, 1.0F, 1.0F};
	const oa::F32 bValues[6]{2.0F, 2.0F, 2.0F, 2.0F, 2.0F, 2.0F};
	const auto a = oa::FnMatrix::fromBytes(
		oa::Span<const oa::U8>(reinterpret_cast<const oa::U8*>(aValues), sizeof(aValues)),
		{2, 3}, oa::ScalarType::Float32);
	const auto b = oa::FnMatrix::fromBytes(
		oa::Span<const oa::U8>(reinterpret_cast<const oa::U8*>(bValues), sizeof(bValues)),
		{2, 3}, oa::ScalarType::Float32);
	oa::Matrix sum;
	{
		oa::ExecutionSession::RecordingScope recording(ctx);
		sum = oa::FnMatrix::add(a, b);
	}
	// Leaving a recording scope only restores recorder selection.
	ASSERT_EQ(ctx.nodeCount(), 1U);
	auto submitted = ctx.submit();
	ASSERT_TRUE(submitted.isOk()) << submitted.getStatus().getMessage();
	ASSERT_TRUE(submitted.getValue().isValid());
	const auto rejected = ctx.submit();
	ASSERT_FALSE(rejected.isOk());
	EXPECT_EQ(rejected.getStatus().getCode(), oa::StatusCode::FailedPrecondition);
	const auto invalidWait = ctx.wait(oa::Event{});
	EXPECT_FALSE(invalidWait.isOk());
	EXPECT_EQ(invalidWait.getCode(), oa::StatusCode::InvalidArgument);
	const auto secondBatch = ctx.submit();
	EXPECT_FALSE(secondBatch.isOk());
	EXPECT_EQ(
		secondBatch.getStatus().getCode(),
		oa::StatusCode::FailedPrecondition);
	ASSERT_TRUE(ctx.wait(submitted.getValue()).isOk());
	const auto staleWait = ctx.wait(submitted.getValue());
	EXPECT_FALSE(staleWait.isOk());
	EXPECT_EQ(staleWait.getCode(), oa::StatusCode::FailedPrecondition);
	const auto emptySubmit = ctx.submit();
	EXPECT_FALSE(emptySubmit.isOk());
	EXPECT_EQ(emptySubmit.getStatus().getCode(), oa::StatusCode::FailedPrecondition);

	oa::F32 host[6]{};
	ASSERT_TRUE(oa::FnMatrix::copyToHost(sum, host, sizeof(host)).isOk());
	for (const auto value : host) EXPECT_FLOAT_EQ(value, 3.0F);
	ctx.clear();
}

TEST(ExecutableGraph, EngineCaptureCompilesIsolatedReusablePlan) {
	auto* engine = testEnginePtr();
	ASSERT_NE(engine, nullptr);
	auto* previous = oa::ExecutionSession::getActivePtr();
	ASSERT_NE(previous, nullptr);
	auto& defaultContext = oa::ExecutionSession::getActive();
	defaultContext.clear();

	const oa::F32 aValues[6]{1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F};
	const oa::F32 bValues[6]{6.0F, 5.0F, 4.0F, 3.0F, 2.0F, 1.0F};
	const auto a = oa::FnMatrix::fromBytes(
		oa::Span<const oa::U8>(
			reinterpret_cast<const oa::U8*>(aValues), sizeof(aValues)),
		{2, 3}, oa::ScalarType::Float32);
	const auto b = oa::FnMatrix::fromBytes(
		oa::Span<const oa::U8>(
			reinterpret_cast<const oa::U8*>(bValues), sizeof(bValues)),
		{2, 3}, oa::ScalarType::Float32);

	oa::Matrix sum;
	auto captured = engine->capture([&]() {
		sum = oa::FnMatrix::add(a, b);
	});
	ASSERT_TRUE(captured.isOk()) << captured.getStatus().getMessage();
	EXPECT_EQ(oa::ExecutionSession::getActivePtr(), previous);
	EXPECT_EQ(defaultContext.nodeCount(), 0U);

	oa::ExecutionPlan plan = oa::move(captured).getValue();
	ASSERT_TRUE(plan.isCompiled());
	EXPECT_EQ(plan.nodeCount(), 1U);
	EXPECT_NE(plan.dnnGraphHash(), 0U);
	EXPECT_EQ(plan.dnnSourceOpCount(), 1U);
	EXPECT_EQ(plan.dnnCapturedOpCount(), 1U);
	EXPECT_EQ(plan.dnnPartitionCount(), 1U);
	EXPECT_EQ(plan.dnnRecognizedPartitionCount(), 0U);
	oa::ExecutionPlan moved = oa::move(plan);
	EXPECT_FALSE(plan.isCompiled());
	EXPECT_EQ(plan.nodeCount(), 0U);
	EXPECT_EQ(plan.dnnGraphHash(), 0U);
	ASSERT_TRUE(moved.isCompiled());
	EXPECT_EQ(moved.dnnPartitionCount(), 1U);

	auto first = engine->submit(moved);
	ASSERT_TRUE(first.isOk()) << first.getStatus().getMessage();
	ASSERT_TRUE(first.getValue().isValid());
	auto second = engine->submit(moved);
	ASSERT_TRUE(second.isOk()) << second.getStatus().getMessage();
	ASSERT_TRUE(second.getValue().isValid());
	EXPECT_FALSE(first.getValue().isSameCompletion(second.getValue()));
	EXPECT_GT(second.getValue().value(), first.getValue().value());
	ASSERT_TRUE(engine->wait(second.getValue()).isOk());

	oa::F32 host[6]{};
	ASSERT_TRUE(oa::FnMatrix::copyToHost(sum, host, sizeof(host)).isOk());
	for (const oa::F32 value : host) EXPECT_FLOAT_EQ(value, 7.0F);
	defaultContext.clear();
}

TEST(ExecutableGraph, EngineCaptureAutomaticallyPlansSemanticDnnRegions) {
	auto* engine = testEnginePtr();
	ASSERT_NE(engine, nullptr);
	auto& defaultContext = oa::ExecutionSession::getActive();
	defaultContext.clear();

	const auto input = oa::FnMatrix::empty({4, 8}, oa::ScalarType::Float32);
	const auto qWeight = oa::FnMatrix::empty({8, 8}, oa::ScalarType::Float32);
	const auto kWeight = oa::FnMatrix::empty({8, 8}, oa::ScalarType::Float32);
	const auto vWeight = oa::FnMatrix::empty({8, 8}, oa::ScalarType::Float32);
	oa::Matrix q;
	oa::Matrix k;
	oa::Matrix v;
	auto captured = engine->capture([&]() {
		q = oa::FnMatrix::matMulNt(input, qWeight, oa::MatMulPrecision::Fp32);
		k = oa::FnMatrix::matMulNt(input, kWeight, oa::MatMulPrecision::Fp32);
		v = oa::FnMatrix::matMulNt(input, vWeight, oa::MatMulPrecision::Fp32);
	});
	ASSERT_TRUE(captured.isOk()) << captured.getStatus().getMessage();
	auto plan = oa::move(captured).getValue();
	EXPECT_EQ(plan.nodeCount(), 3U);
	EXPECT_NE(plan.dnnGraphHash(), 0U);
	EXPECT_EQ(plan.dnnSourceOpCount(), 3U);
	EXPECT_EQ(plan.dnnCapturedOpCount(), 3U);
	EXPECT_EQ(plan.dnnPartitionCount(), 1U);
	EXPECT_EQ(plan.dnnRecognizedPartitionCount(), 1U);

	auto submitted = engine->submit(plan);
	ASSERT_TRUE(submitted.isOk()) << submitted.getStatus().getMessage();
	ASSERT_TRUE(engine->wait(submitted.getValue()).isOk());
	defaultContext.clear();
}

TEST(ExecutableGraph, EngineWaitKeepsUnrelatedEagerSubmissionPending) {
	auto* engine = testEnginePtr();
	ASSERT_NE(engine, nullptr);
	auto& defaultContext = oa::ExecutionSession::getActive();
	defaultContext.clear();

	const oa::F32 aValues[6]{1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F};
	const oa::F32 bValues[6]{6.0F, 5.0F, 4.0F, 3.0F, 2.0F, 1.0F};
	const auto a = oa::FnMatrix::fromBytes(
		oa::Span<const oa::U8>(
			reinterpret_cast<const oa::U8*>(aValues), sizeof(aValues)),
		{2, 3}, oa::ScalarType::Float32);
	const auto b = oa::FnMatrix::fromBytes(
		oa::Span<const oa::U8>(
			reinterpret_cast<const oa::U8*>(bValues), sizeof(bValues)),
		{2, 3}, oa::ScalarType::Float32);

	oa::Matrix planOutput;
	auto captured = engine->capture([&]() {
		planOutput = oa::FnMatrix::add(a, b);
	});
	ASSERT_TRUE(captured.isOk()) << captured.getStatus().getMessage();
	auto plan = oa::move(captured).getValue();
	auto planSubmission = engine->submit(plan);
	ASSERT_TRUE(planSubmission.isOk())
		<< planSubmission.getStatus().getMessage();

	auto eagerOutput = oa::FnMatrix::add(a, b);
	auto eagerSubmission = engine->submit();
	ASSERT_TRUE(eagerSubmission.isOk())
		<< eagerSubmission.getStatus().getMessage();

	// Waiting for a reusable-plan event must not reclaim the unrelated eager
	// batch. Only its exact completion event owns that state transition.
	ASSERT_TRUE(engine->wait(planSubmission.getValue()).isOk());
	const auto blockedSubmit = engine->submit();
	ASSERT_FALSE(blockedSubmit.isOk());
	EXPECT_EQ(
		blockedSubmit.getStatus().getCode(),
		oa::StatusCode::FailedPrecondition);
	ASSERT_TRUE(engine->wait(eagerSubmission.getValue()).isOk());

	const oa::Status invalidWait = engine->wait(oa::Event{});
	EXPECT_EQ(invalidWait.getCode(), oa::StatusCode::InvalidArgument);
	oavk::Device foreignDevice;
	const oa::Event foreignEvent = oa::EventAccess::create(
		foreignDevice,
		oa::EventAccess::timelineSemaphore(planSubmission.getValue()),
		planSubmission.getValue().value(),
		planSubmission.getValue().queueFamily());
	const oa::Status foreignWait = engine->wait(foreignEvent);
	EXPECT_EQ(foreignWait.getCode(), oa::StatusCode::InvalidArgument);

	oa::F32 planHost[6]{};
	oa::F32 eagerHost[6]{};
	ASSERT_TRUE(oa::FnMatrix::copyToHost(
		planOutput, planHost, sizeof(planHost)).isOk());
	ASSERT_TRUE(oa::FnMatrix::copyToHost(
		eagerOutput, eagerHost, sizeof(eagerHost)).isOk());
	for (oa::I32 index = 0; index < 6; ++index) {
		EXPECT_FLOAT_EQ(planHost[index], 7.0F);
		EXPECT_FLOAT_EQ(eagerHost[index], 7.0F);
	}
	defaultContext.clear();
}

TEST(ExecutableGraph, EngineCaptureRejectsInvalidWorkAndRestoresRecorder) {
	auto* engine = testEnginePtr();
	ASSERT_NE(engine, nullptr);
	auto* previous = oa::ExecutionSession::getActivePtr();
	ASSERT_NE(previous, nullptr);

	const auto noCallback = engine->capture(oa::Fn<void()>{});
	ASSERT_FALSE(noCallback.isOk());
	EXPECT_EQ(noCallback.getStatus().getCode(), oa::StatusCode::InvalidArgument);
	EXPECT_EQ(oa::ExecutionSession::getActivePtr(), previous);

	const auto noWork = engine->capture([]() {});
	ASSERT_FALSE(noWork.isOk());
	EXPECT_EQ(
		noWork.getStatus().getCode(),
		oa::StatusCode::FailedPrecondition);
	EXPECT_EQ(oa::ExecutionSession::getActivePtr(), previous);

	const auto a = oa::FnMatrix::empty({4});
	const auto b = oa::FnMatrix::empty({4});
	auto output = oa::FnMatrix::empty({4});
	oa::Status semanticStatus;
	const auto incompleteLowering = engine->capture([&]() {
		auto semantic = oa::ExecutionSession::getActive().recordOp(
			oa::detail::opRegistry::FnMatrix::add,
			{&a, &b},
			{&output});
		semanticStatus = semantic.isOk()
			? oa::Status::ok()
			: semantic.getStatus();
	});
	ASSERT_TRUE(semanticStatus.isOk()) << semanticStatus.getMessage();
	ASSERT_FALSE(incompleteLowering.isOk());
	EXPECT_EQ(
		incompleteLowering.getStatus().getCode(),
		oa::StatusCode::FailedPrecondition);
	EXPECT_EQ(
		incompleteLowering.getStatus().getMessage(),
		"semantic operation has no executable lowering");
	EXPECT_EQ(oa::ExecutionSession::getActivePtr(), previous);

	oa::ExecutionPlan empty;
	const auto uncompiled = engine->submit(empty);
	ASSERT_FALSE(uncompiled.isOk());
	EXPECT_EQ(
		uncompiled.getStatus().getCode(),
		oa::StatusCode::FailedPrecondition);
}

TEST(ExecutableGraph, ExecutionPlanDestructionRetiresSubmittedWork) {
	auto* engine = testEnginePtr();
	ASSERT_NE(engine, nullptr);
	auto target = oa::FnMatrix::ones({256}, oa::ScalarType::Float32);
	ASSERT_FLOAT_EQ(target.at(0), 1.0F);

	oa::Event completion;
	{
		auto captured = engine->capture([&]() {
			target.zero();
		});
		ASSERT_TRUE(captured.isOk()) << captured.getStatus().getMessage();
		auto plan = oa::move(captured).getValue();
		auto submitted = engine->submit(plan);
		ASSERT_TRUE(submitted.isOk()) << submitted.getStatus().getMessage();
		completion = submitted.getValue();
	}

	// Plan destruction transferred the in-flight graph to engine retirement.
	// The exact event remains a valid completion boundary and no destructor wait
	// was needed to preserve the submitted work.
	ASSERT_TRUE(completion.isValid());
	ASSERT_TRUE(completion.wait().isOk());
	auto* collectionProbe = oa::EngineSubmissionAccess::acquireStream(*engine);
	ASSERT_NE(collectionProbe, nullptr);
	oa::EngineSubmissionAccess::releaseStream(*engine, collectionProbe);
	EXPECT_FLOAT_EQ(target.at(0), 0.0F);
}

TEST(ExecutableGraph, ContextDestructionDiscardsUnsubmittedRecording) {
	auto* engine = testEnginePtr();
	ASSERT_NE(engine, nullptr);
	auto* previous = oa::ExecutionSession::getActivePtr();
	ASSERT_NE(previous, nullptr);

	auto target = oa::FnMatrix::ones({4}, oa::ScalarType::Float32);
	EXPECT_FLOAT_EQ(target.at(0), 1.0F);

	auto* temporary = new oa::ExecutionSession(engine);
	{
		oa::ExecutionSession::RecordingScope recording(*temporary);
		target.zero();
		ASSERT_GT(temporary->nodeCount(), 0U);
	}
	delete temporary;

	// Destruction must not turn the abandoned recording into GPU work.
	EXPECT_FLOAT_EQ(target.at(0), 1.0F);
}

TEST(ExecutableGraph, ContextBatchesAreIsolatedAndRejectForeignEvents) {
	auto* engine = testEnginePtr();
	ASSERT_NE(engine, nullptr);
	auto* previous = oa::ExecutionSession::getActivePtr();
	ASSERT_NE(previous, nullptr);

	const oa::F32 aValues[4]{1.0F, 2.0F, 3.0F, 4.0F};
	const oa::F32 bValues[4]{3.0F, 4.0F, 5.0F, 6.0F};
	const auto a = oa::FnMatrix::fromBytes(
		oa::Span<const oa::U8>(reinterpret_cast<const oa::U8*>(aValues), sizeof(aValues)),
		{4}, oa::ScalarType::Float32);
	const auto b = oa::FnMatrix::fromBytes(
		oa::Span<const oa::U8>(reinterpret_cast<const oa::U8*>(bValues), sizeof(bValues)),
		{4}, oa::ScalarType::Float32);

	auto* first = new oa::ExecutionSession(engine);
	auto* second = new oa::ExecutionSession(engine);
	oa::Matrix firstResult;
	oa::Matrix secondResult;
	oa::Event firstCompletion;
	oa::Event secondCompletion;
	{
		oa::ExecutionSession::RecordingScope recording(*first);
		firstResult = oa::FnMatrix::add(a, a);
		auto submitted = first->submit();
		ASSERT_TRUE(submitted.isOk());
		firstCompletion = submitted.getValue();
	}
	{
		oa::ExecutionSession::RecordingScope recording(*second);
		secondResult = oa::FnMatrix::add(b, b);
		auto submitted = second->submit();
		ASSERT_TRUE(submitted.isOk());
		secondCompletion = submitted.getValue();
	}

	ASSERT_TRUE(firstCompletion.isValid());
	ASSERT_TRUE(secondCompletion.isValid());
	EXPECT_FALSE(firstCompletion.isSameCompletion(secondCompletion));

	const auto foreignFirst = first->wait(secondCompletion);
	EXPECT_FALSE(foreignFirst.isOk());
	EXPECT_EQ(foreignFirst.getCode(), oa::StatusCode::InvalidArgument);
	const auto foreignSecond = second->wait(firstCompletion);
	EXPECT_FALSE(foreignSecond.isOk());
	EXPECT_EQ(foreignSecond.getCode(), oa::StatusCode::InvalidArgument);
	ASSERT_TRUE(first->wait(firstCompletion).isOk());
	ASSERT_TRUE(second->wait(secondCompletion).isOk());

	oa::F32 firstHost[4]{};
	oa::F32 secondHost[4]{};
	ASSERT_TRUE(oa::FnMatrix::copyToHost(
		firstResult, firstHost, sizeof(firstHost)).isOk());
	ASSERT_TRUE(oa::FnMatrix::copyToHost(
		secondResult, secondHost, sizeof(secondHost)).isOk());
	for (oa::U32 index = 0; index < 4U; ++index) {
		EXPECT_FLOAT_EQ(firstHost[index], aValues[index] * 2.0F);
		EXPECT_FLOAT_EQ(secondHost[index], bValues[index] * 2.0F);
	}

	delete first;
	delete second;
	oa::ExecutionSession::setActive(previous);
}

TEST(ExecutableGraph, SubmittedContextDestructionRetiresWithoutLosingWork) {
	auto* engine = testEnginePtr();
	ASSERT_NE(engine, nullptr);
	auto* previous = oa::ExecutionSession::getActivePtr();
	ASSERT_NE(previous, nullptr);

	auto target = oa::FnMatrix::ones({256}, oa::ScalarType::Float32);
	ASSERT_FLOAT_EQ(target.at(0), 1.0F);
	auto* temporary = new oa::ExecutionSession(engine);
	oa::Event completion;
	{
		oa::ExecutionSession::RecordingScope recording(*temporary);
		target.zero();
		auto submitted = temporary->submit();
		ASSERT_TRUE(submitted.isOk());
		completion = submitted.getValue();
	}
	delete temporary;

	// The event and graph resources are owned by engine retirement after the
	// context is gone. Waiting the exact event still observes the submitted work.
	ASSERT_TRUE(completion.isValid());
	ASSERT_TRUE(completion.wait().isOk());
	auto* collectionProbe = oa::EngineSubmissionAccess::acquireStream(*engine);
	ASSERT_NE(collectionProbe, nullptr);
	oa::EngineSubmissionAccess::releaseStream(*engine, collectionProbe);
	EXPECT_FLOAT_EQ(target.at(0), 0.0F);
	oa::ExecutionSession::setActive(previous);
}

TEST(ExecutableGraph, DebugReportIsDeterministicAndHandleFree) {
	oavk::Buffer a;
	a.buffer = reinterpret_cast<void*>(0x11110000);
	a.size = 256;
	oavk::Buffer b;
	b.buffer = reinterpret_cast<void*>(0x22220000);
	b.size = 512;
	oavk::Buffer buffers[] = {a, b};
	oa::BufferAccess access[] = {oa::BufferAccess::Read, oa::BufferAccess::Write};
	const oa::U32 semanticOps[] = {5U};

	oa::ComputeDispatchDesc desc;
	desc.operation = "MatMulNt";
	desc.semanticOps = semanticOps;
	desc.implementationId = 0x42U;
	desc.opContractHash = 0x99U;
	desc.problemContractHash = 0x100U;
	desc.kernelContentHash = 0x123U;
	desc.kernel = "GemmTiledFp32_64x64x16";
	desc.buffers = buffers;
	desc.access = access;
	desc.groupsX = 4;
	desc.groupsY = 8;

	oa::ExecutableGraph graph;
	graph.add(desc);
	oavk::Buffer secondBuffers[] = {b, a};
	oa::BufferAccess secondAccess[] = {oa::BufferAccess::Read, oa::BufferAccess::Write};
	oa::ComputeDispatchDesc secondDesc;
	secondDesc.operation = "Silu";
	secondDesc.kernel = "Silu";
	secondDesc.buffers = secondBuffers;
	secondDesc.access = secondAccess;
	graph.add(secondDesc);
	const oa::String first = graph.debugReportJson("unit-matmul");
	const oa::String second = graph.debugReportJson("unit-matmul");
	EXPECT_EQ(first, second);
	const auto text = first.stdStr();
	EXPECT_NE(text.find("\"schema\": \"oa.execution_graph.v3\""), std::string::npos);
	EXPECT_NE(text.find("\"operation\": \"MatMulNt\""), std::string::npos);
	EXPECT_NE(text.find("\"semantic_operations\": [5]"), std::string::npos);
	EXPECT_NE(text.find("\"implementation_id\": \"0x0000000000000042\""),
		std::string::npos);
	EXPECT_NE(text.find("\"operation_contract_hash\": \"0x0000000000000099\""),
		std::string::npos);
	EXPECT_NE(text.find("\"problem_contract_hash\": \"0x0000000000000100\""),
		std::string::npos);
	EXPECT_NE(text.find("\"kernel\": \"GemmTiledFp32_64x64x16\""),
		std::string::npos);
	EXPECT_NE(text.find("\"resource\": 0"), std::string::npos);
	EXPECT_NE(text.find("\"access\": \"write\""), std::string::npos);
	EXPECT_NE(text.find("\"reason\": \"read_after_write\""), std::string::npos);
	EXPECT_NE(text.find("\"scope\": \"buffer\""), std::string::npos);
	EXPECT_NE(text.find("\"source_nodes\": [0, 0]"), std::string::npos);
	EXPECT_NE(text.find("\"destination_node\": 1"), std::string::npos);
	EXPECT_NE(text.find("\"source_stages\": [\"compute_shader\"]"),
		std::string::npos);
	EXPECT_NE(text.find("\"destination_accesses\": [\"shader_storage_read\"]"),
		std::string::npos);
	EXPECT_NE(text.find("\"reason\": \"host_readback\""), std::string::npos);
	EXPECT_EQ(text.find("11110000"), std::string::npos);
	EXPECT_EQ(text.find("22220000"), std::string::npos);
}

TEST(ExecutableGraph, ContextRecordRejectsMalformedDescriptor) {
	auto& ctx = oa::ExecutionSession::getActive();
	ctx.clear();

	oavk::Buffer buffer;
	oa::ComputeDispatchDesc desc;
	desc.kernel = "MalformedDescriptorTest";
	desc.buffers = oa::Span<oavk::Buffer>(&buffer, 1);
	// No access annotation for one buffer: this must fail before graph append.

	const auto status = ctx.record( desc);
	EXPECT_FALSE(status.isOk());
	EXPECT_EQ(ctx.nodeCount(), 0U);
	ctx.clear();
}

TEST(ExecutableGraph, ContextRecordRejectsDescriptorBeyondLiveDeviceLimit) {
	auto& ctx = oa::ExecutionSession::getActive();
	auto& engine = ctx.engine();
	ctx.clear();
	auto sourceResult = oa::EngineResourceAccess::allocBuffer(engine, 16U);
	auto destinationResult = oa::EngineResourceAccess::allocBuffer(engine, 16U);
	ASSERT_TRUE(sourceResult.isOk() and destinationResult.isOk());
	auto source = std::move(*sourceResult);
	auto destination = std::move(*destinationResult);
	oavk::Buffer buffers[] = {source, destination};
	oa::BufferAccess access[] = {
		oa::BufferAccess::Read, oa::BufferAccess::Write};
	struct { oa::U32 count; oa::F32 scale; } push{4U, 2.0F};
	oa::ComputeDispatchDesc desc;
	desc.kernel = "Scale";
	desc.buffers = buffers;
	desc.access = access;
	desc.pushData = &push;
	desc.pushSize = sizeof(push);
	desc.groupsX = 1U;

	auto& maximum =
		oa::EngineDeviceAccess::get(engine).info.hardware.maxStorageBufferRangeBytes;
	const oa::U64 savedMaximum = maximum;
	ASSERT_GT(savedMaximum, 8U);
	maximum = 8U;
	const auto status = ctx.record( desc);
	EXPECT_EQ(status.getCode(), oa::StatusCode::OutOfRange);
	EXPECT_EQ(ctx.nodeCount(), 0U);

	maximum = savedMaximum;
	ctx.clear();
	oa::EngineResourceAccess::freeBuffer(engine, source);
	oa::EngineResourceAccess::freeBuffer(engine, destination);
}

TEST(ExecutableGraph, ContextRecordRejectsDirectGroupsBeyondLiveDeviceLimit) {
	auto& context = oa::ExecutionSession::getActive();
	auto& engine = context.engine();
	context.clear();
	auto sourceResult = oa::EngineResourceAccess::allocBuffer(engine, 16U);
	auto destinationResult = oa::EngineResourceAccess::allocBuffer(engine, 16U);
	ASSERT_TRUE(sourceResult.isOk() and destinationResult.isOk());
	auto source = std::move(*sourceResult);
	auto destination = std::move(*destinationResult);
	oavk::Buffer buffers[] = {source, destination};
	oa::BufferAccess access[] = {
		oa::BufferAccess::Read, oa::BufferAccess::Write};
	struct { oa::U32 count; oa::F32 scale; } push{4U, 2.0F};
	oa::ComputeDispatchDesc desc;
	desc.kernel = "Scale";
	desc.buffers = buffers;
	desc.access = access;
	desc.pushData = &push;
	desc.pushSize = sizeof(push);
	desc.groupsX = 2U;

	auto& maximum =
		oa::EngineDeviceAccess::get(engine).info.hardware.maxComputeWorkGroupCountX;
	const oa::U32 savedMaximum = maximum;
	ASSERT_GT(savedMaximum, 1U);
	maximum = 1U;
	const auto rejected = context.record( desc);
	EXPECT_EQ(rejected.getCode(), oa::StatusCode::OutOfRange);
	EXPECT_EQ(context.nodeCount(), 0U);

	maximum = savedMaximum;
	context.clear();
	desc.groupsX = 1U;
	EXPECT_TRUE(context.record( desc).isOk());
	EXPECT_EQ(context.nodeCount(), 1U);

	context.clear();
	oa::EngineResourceAccess::freeBuffer(engine, source);
	oa::EngineResourceAccess::freeBuffer(engine, destination);
}

TEST(ExecutableGraph, ContextRecordRejectsMalformedIndirectDescriptor) {
	auto& ctx = oa::ExecutionSession::getActive();
	ctx.clear();

	oa::ComputeDispatchDesc desc;
	desc.kernel = "MalformedIndirectDescriptorTest";
	desc.indirect = true;
	desc.indirectBuffer.buffer = reinterpret_cast<void*>(0x1000);
	desc.indirectBuffer.size = 3 * sizeof(oa::U32);
	desc.indirectBuffer.flags = OA_VK_BUFFER_FLAG_INDIRECT_DISPATCH;
	desc.indirectBuffer.allocatorIdentity =
		oa::EngineAllocatorAccess::get(testEngine()).allocator;
	desc.indirectOffset = 2;

	auto status = ctx.record( desc);
	EXPECT_EQ(status.getCode(), oa::StatusCode::InvalidArgument);
	EXPECT_EQ(ctx.nodeCount(), 0U);
	ctx.clear();

	desc.indirectOffset = 4;
	status = ctx.record( desc);
	EXPECT_EQ(status.getCode(), oa::StatusCode::OutOfRange);
	EXPECT_EQ(ctx.nodeCount(), 0U);
	ctx.clear();

	desc.indirectOffset = ~oa::U64{0} - 3U;
	status = ctx.record( desc);
	EXPECT_EQ(status.getCode(), oa::StatusCode::OutOfRange);
	EXPECT_EQ(ctx.nodeCount(), 0U);
	ctx.clear();

	desc.indirectOffset = 0;
	desc.indirectBuffer.allocatorIdentity = &desc;
	status = ctx.record( desc);
	EXPECT_EQ(status.getCode(), oa::StatusCode::InvalidArgument);
	EXPECT_EQ(ctx.nodeCount(), 0U);
	ctx.clear();

	desc.indirect = false;
	desc.indirectBuffer.allocatorIdentity = nullptr;
	desc.indirectOffset = 0;
	status = ctx.record( desc);
	EXPECT_EQ(status.getCode(), oa::StatusCode::InvalidArgument);
	EXPECT_EQ(ctx.nodeCount(), 0U);
	ctx.clear();
}

TEST(ExecutableGraph, ContextMatrixRecordRetainsIndirectArguments) {
	auto& ctx = oa::ExecutionSession::getActive();
	ctx.clear();

	oa::Matrix input;
	auto& inputStorage = oa::MatrixAccess::storageOwner(input);
	inputStorage = oa::SharedPtr<oavk::Buffer>(new oavk::Buffer());
	inputStorage->buffer = reinterpret_cast<void*>(0x3000);
	inputStorage->size = 64;
	inputStorage->bindlessIndex = 13;
	inputStorage->allocatorIdentity =
		oa::EngineAllocatorAccess::get(testEngine()).allocator;
	oa::Matrix args;
	auto& argsStorage = oa::MatrixAccess::storageOwner(args);
	argsStorage = oa::SharedPtr<oavk::Buffer>(new oavk::Buffer());
	argsStorage->buffer = reinterpret_cast<void*>(0x4000);
	argsStorage->size = 3 * sizeof(oa::U32);
	argsStorage->bindlessIndex = 17;
	argsStorage->flags = OA_VK_BUFFER_FLAG_INDIRECT_DISPATCH;
	argsStorage->allocatorIdentity =
		oa::EngineAllocatorAccess::get(testEngine()).allocator;

	const oa::Matrix* matrices[] = {&input, &args};
	oa::BufferAccess access[] = {oa::BufferAccess::Read, oa::BufferAccess::Read};
	struct { oa::U32 N; oa::F32 scale; } push{64, 1.0F};
	oa::MatrixDispatchDesc desc;
	desc.dispatch.kernel = "Scale";
	desc.dispatch.access = access;
	desc.dispatch.pushData = &push;
	desc.dispatch.pushSize = sizeof(push);
	desc.matrices = matrices;
	desc.indirectArgs = &args;
	ASSERT_TRUE(ctx.record( desc).isOk());

	ASSERT_EQ(ctx.nodeCount(), 1U);
	const auto nodes = ctx.graph()->nodes();
	ASSERT_EQ(nodes.size(), 1U);
	EXPECT_TRUE(nodes[0].indirect);
	EXPECT_EQ(nodes[0].indirectBuffer.buffer, argsStorage->buffer);
	ASSERT_EQ(nodes[0].bufferOwners.size(), 2U);
	EXPECT_TRUE(static_cast<bool>(nodes[0].bufferOwners[1]));
	ctx.clear();

	const oa::Matrix* missingOwner[] = {&input};
	desc.matrices = missingOwner;
	access[0] = oa::BufferAccess::Read;
	desc.dispatch.access = oa::Span<oa::BufferAccess>(access, 1);
	EXPECT_FALSE(ctx.record( desc).isOk());
	EXPECT_EQ(ctx.nodeCount(), 0U);
}

TEST(ExecutableGraph, ContextMatrixAddCapturesOwnershipAndDtype) {
	auto& ctx = oa::ExecutionSession::getActive();
	ctx.clear();

	oa::Matrix input;
	auto& inputStorage = oa::MatrixAccess::storageOwner(input);
	inputStorage = oa::SharedPtr<oavk::Buffer>(new oavk::Buffer());
	inputStorage->buffer = reinterpret_cast<void*>(0x3000);
	inputStorage->size = 128U;
	inputStorage->bindlessIndex = 13;
	inputStorage->allocatorIdentity =
		oa::EngineAllocatorAccess::get(testEngine()).allocator;
	oa::MatrixAccess::dtype(input) = oa::ScalarType::BFloat16;
	oa::Matrix output;
	auto& outputStorage = oa::MatrixAccess::storageOwner(output);
	outputStorage = oa::SharedPtr<oavk::Buffer>(new oavk::Buffer());
	outputStorage->buffer = reinterpret_cast<void*>(0x4000);
	outputStorage->size = 128U;
	outputStorage->bindlessIndex = 17;
	outputStorage->allocatorIdentity =
		oa::EngineAllocatorAccess::get(testEngine()).allocator;

	oa::BufferAccess access[] = {oa::BufferAccess::Read, oa::BufferAccess::Write};
	struct Push { oa::U32 count; oa::F32 scale; } push{32, 1.0F};
	ctx.add( "Scale", {&input, &output}, access, &push, sizeof(push), 1);

	ASSERT_EQ(ctx.nodeCount(), 1U);
	const auto nodes = ctx.graph()->nodes();
	ASSERT_EQ(nodes.size(), 1U);
	EXPECT_EQ(nodes[0].dtype, 1U);
	ASSERT_EQ(nodes[0].bufferOwners.size(), 2U);
	EXPECT_TRUE(static_cast<bool>(nodes[0].bufferOwners[0]));
	EXPECT_TRUE(static_cast<bool>(nodes[0].bufferOwners[1]));
	EXPECT_EQ(nodes[0].buffers[0].bindlessIndex, 13U);
	EXPECT_EQ(nodes[0].buffers[1].bindlessIndex, 17U);
	ctx.clear();
}

TEST(ExecutableGraph, SystemInfo) {
	fprintf(stderr, "\n");
	fprintf(stderr, "  ╔═══════════════════════════════════════════════════════════════╗\n");
	fprintf(stderr, "  ║       oa::ExecutableGraph TEST SUITE — graph & replay Paths      ║\n");
	fprintf(stderr, "  ╚═══════════════════════════════════════════════════════════════╝\n");

	auto* rt = testEnginePtr();
	if (rt) {
		const oa::StringView deviceName = rt->deviceName();
		const oa::StringView vendorName = rt->deviceVendorName();
		fprintf(stderr, "\n  GPU: %.*s (%.*s)\n",
			static_cast<int>(deviceName.size()), deviceName.data(),
			static_cast<int>(vendorName.size()), vendorName.data());
		const oa::DeviceType deviceType = rt->deviceType();
		fprintf(stderr, "  type: %s, VRAM: %llu MB\n",
			deviceType == oa::DeviceType::VkDiscrete ? "Discrete" :
			deviceType == oa::DeviceType::VkIntegrated ? "Integrated" :
			deviceType == oa::DeviceType::VkCpu ? "software" : "Unknown",
			static_cast<unsigned long long>(rt->deviceVramBytes() / (1024 * 1024)));
	}
	fprintf(stderr, "\n");
}

TEST(ExecutableGraph, DeviceAdmissionCanaryUsesIndependentKnownAnswers) {
	if (not vkTestEngineOk()) GTEST_SKIP();
	auto& engine = testEngine();
	oa::ExecutionSession::forEngine(engine).clear();

	oa::DeviceCanaryReport report;
	const auto status = oa::DeviceCanary::run(engine, report);
	ASSERT_TRUE(status.isOk()) << status.toString().cStr();
	EXPECT_TRUE(report.passed());
	ASSERT_EQ(report.checks.size(), 5U);
	for (const auto& check : report.checks) {
		EXPECT_TRUE(check.passed) << check.name.cStr();
		EXPECT_NE(check.expectedHash, 0U);
		EXPECT_NE(check.actualHash, 0U);
	}
	const auto first = report.debugReportJson().stdStr();
	const auto second = report.debugReportJson().stdStr();
	EXPECT_EQ(first, second);
	EXPECT_NE(first.find("\"schema\": \"oa.device_canary.v1\""),
		std::string::npos);
	EXPECT_NE(first.find("\"passed\": true"), std::string::npos);
	EXPECT_NE(first.find("\"name\": \"fp32_matmul_irregular\""),
		std::string::npos);
}

TEST(ExecutableGraph, BasicAddAndExecute) {
	auto* rt = testEnginePtr();
	ASSERT_NE(rt, nullptr);

	auto srcRes = oa::EngineAllocatorAccess::get(*rt).allocHostVisible(256 * sizeof(oa::F32));
	ASSERT_TRUE(srcRes.isOk());
	auto src = srcRes.getValue();
	ASSERT_NE(oa::EngineBindlessAccess::registerBuffer(*rt, src), OA_BINDLESS_INVALID);

	auto dstRes = oa::EngineAllocatorAccess::get(*rt).allocHostVisible(256 * sizeof(oa::F32));
	ASSERT_TRUE(dstRes.isOk());
	auto dst = dstRes.getValue();
	ASSERT_NE(oa::EngineBindlessAccess::registerBuffer(*rt, dst), OA_BINDLESS_INVALID);

	oa::F32* data = static_cast<oa::F32*>(src.mappedPtr);
	for (oa::I32 i = 0; i < 256; ++i) data[i] = static_cast<oa::F32>(i);

	struct { oa::U32 N; oa::F32 scale; } push{256, 3.0f};
	oavk::Buffer bufs[] = {src, dst};
	oa::BufferAccess access[] = {oa::BufferAccess::Read, oa::BufferAccess::Write};

	oa::ExecutableGraph graph;
	graph.add("Scale", bufs, access, &push, sizeof(push), (256 + 255) / 256);
	EXPECT_EQ(graph.nodeCount(), 1u);

	auto status = graph.execute(*rt);
	if (status.isOk()) {
		oa::F32* out = static_cast<oa::F32*>(dst.mappedPtr);
		for (oa::I32 i = 0; i < 256; ++i) {
			EXPECT_NEAR(out[i], static_cast<oa::F32>(i) * 3.0f, 1e-3f);
		}
	}

	oa::EngineBindlessAccess::deregisterBuffer(*rt, src);
	oa::EngineAllocatorAccess::get(*rt).free(src);
	oa::EngineBindlessAccess::deregisterBuffer(*rt, dst);
	oa::EngineAllocatorAccess::get(*rt).free(dst);
}

TEST(ExecutableGraph, CompileRejectsDirectGroupsBeyondLiveDeviceLimit) {
	auto* runtime = testEnginePtr();
	ASSERT_NE(runtime, nullptr);
	auto sourceResult = oa::EngineResourceAccess::allocBuffer(*runtime, 16U);
	auto destinationResult = oa::EngineResourceAccess::allocBuffer(*runtime, 16U);
	ASSERT_TRUE(sourceResult.isOk() and destinationResult.isOk());
	auto source = std::move(*sourceResult);
	auto destination = std::move(*destinationResult);
	oavk::Buffer buffers[] = {source, destination};
	oa::BufferAccess access[] = {
		oa::BufferAccess::Read, oa::BufferAccess::Write};
	struct { oa::U32 count; oa::F32 scale; } push{4U, 2.0F};
	oa::ExecutableGraph graph;
	graph.add(
		"Scale", buffers, access, &push, sizeof(push),
		2U, 1U, 1U);

	auto& maximum = oa::EngineDeviceAccess::get(
		*runtime).info.hardware.maxComputeWorkGroupCountX;
	const oa::U32 savedMaximum = maximum;
	ASSERT_GT(savedMaximum, 1U);
	struct RestoreLimit {
		oa::U32& Value;
		oa::U32 Saved;
		~RestoreLimit() { Value = Saved; }
	} restoreMaximum{maximum, savedMaximum};
	maximum = 1U;
	const auto rejected = graph.compile(*runtime);
	EXPECT_EQ(rejected.getCode(), oa::StatusCode::OutOfRange);
	EXPECT_FALSE(graph.isCompiled());

	maximum = savedMaximum;
	const auto accepted = graph.compile(*runtime);
	ASSERT_TRUE(accepted.isOk()) << accepted.getMessage();
	EXPECT_TRUE(graph.isCompiled());

	graph.reset(*runtime);
	oa::EngineResourceAccess::freeBuffer(*runtime, source);
	oa::EngineResourceAccess::freeBuffer(*runtime, destination);
}

TEST(ExecutableGraph, CompileRejectsMalformedIndirectDescriptorBeforeRecording) {
	auto* runtime = testEnginePtr();
	ASSERT_NE(runtime, nullptr);
	auto sourceResult = oa::EngineResourceAccess::allocBuffer(*runtime, 16U);
	auto destinationResult = oa::EngineResourceAccess::allocBuffer(*runtime, 16U);
	auto argumentResult = oa::EngineResourceAccess::allocBuffer(*runtime, 3U * sizeof(oa::U32));
	ASSERT_TRUE(sourceResult.isOk()
		and destinationResult.isOk()
		and argumentResult.isOk());
	auto source = std::move(*sourceResult);
	auto destination = std::move(*destinationResult);
	auto arguments = std::move(*argumentResult);
	oavk::Buffer buffers[] = {source, destination};
	oa::BufferAccess access[] = {
		oa::BufferAccess::Read, oa::BufferAccess::Write};
	struct { oa::U32 count; oa::F32 scale; } push{4U, 2.0F};

	oa::ExecutableGraph graph;
	graph.addIndirect(
		"Scale", buffers, access, &push, sizeof(push),
		arguments, ~oa::U64{0} - 3U);
	const auto overflowRejected = graph.compile(*runtime);
	EXPECT_EQ(overflowRejected.getCode(), oa::StatusCode::OutOfRange);
	EXPECT_FALSE(graph.isCompiled());

	graph.clearNodes();
	oavk::Buffer unsupported = arguments;
	unsupported.flags &= ~OA_VK_BUFFER_FLAG_INDIRECT_DISPATCH;
	graph.addIndirect(
		"Scale", buffers, access, &push, sizeof(push), unsupported, 0U);
	const auto usageRejected = graph.compile(*runtime);
	EXPECT_EQ(usageRejected.getCode(), oa::StatusCode::InvalidArgument);
	EXPECT_FALSE(graph.isCompiled());

	graph.clearNodes();
	graph.addIndirect(
		"Scale", buffers, access, &push, sizeof(push), arguments, 0U);
	const auto accepted = graph.compile(*runtime);
	ASSERT_TRUE(accepted.isOk()) << accepted.getMessage();
	EXPECT_TRUE(graph.isCompiled());
	EXPECT_FALSE(graph.lastCompileReused());

	// Capability and device-ownership metadata are part of graph cache
	// identity. Replacing only that metadata must reach compile preflight,
	// rather than reusing the valid command buffer recorded above.
	graph.clearNodes();
	graph.addIndirect(
		"Scale", buffers, access, &push, sizeof(push), unsupported, 0U);
	const auto cachedUsageRejected = graph.compile(*runtime);
	EXPECT_EQ(cachedUsageRejected.getCode(), oa::StatusCode::InvalidArgument);
	EXPECT_FALSE(graph.isCompiled());
	EXPECT_FALSE(graph.lastCompileReused());

	graph.reset(*runtime);
	oa::EngineResourceAccess::freeBuffer(*runtime, source);
	oa::EngineResourceAccess::freeBuffer(*runtime, destination);
	oa::EngineResourceAccess::freeBuffer(*runtime, arguments);
}

TEST(ExecutableGraph, CompileAndReplay) {
	auto* rt = testEnginePtr();
	ASSERT_NE(rt, nullptr);

	auto srcRes = oa::EngineAllocatorAccess::get(*rt).allocHostVisible(256 * sizeof(oa::F32));
	ASSERT_TRUE(srcRes.isOk());
	auto src = srcRes.getValue();
	ASSERT_NE(oa::EngineBindlessAccess::registerBuffer(*rt, src), OA_BINDLESS_INVALID);

	auto midRes = oa::EngineAllocatorAccess::get(*rt).allocHostVisible(256 * sizeof(oa::F32));
	ASSERT_TRUE(midRes.isOk());
	auto mid = midRes.getValue();
	ASSERT_NE(oa::EngineBindlessAccess::registerBuffer(*rt, mid), OA_BINDLESS_INVALID);

	auto dstRes = oa::EngineAllocatorAccess::get(*rt).allocHostVisible(256 * sizeof(oa::F32));
	ASSERT_TRUE(dstRes.isOk());
	auto dst = dstRes.getValue();
	ASSERT_NE(oa::EngineBindlessAccess::registerBuffer(*rt, dst), OA_BINDLESS_INVALID);

	oa::F32* data = static_cast<oa::F32*>(src.mappedPtr);
	for (oa::I32 i = 0; i < 256; ++i) data[i] = static_cast<oa::F32>(i + 1);

	struct { oa::U32 N; oa::F32 scale; } push1{256, 2.0f};
	struct { oa::U32 N; oa::F32 scale; } push2{256, 0.5f};

	oavk::Buffer bufs1[] = {src, mid};
	oa::BufferAccess access1[] = {oa::BufferAccess::Read, oa::BufferAccess::Write};
	oavk::Buffer bufs2[] = {mid, dst};
	oa::BufferAccess access2[] = {oa::BufferAccess::Read, oa::BufferAccess::Write};

	oa::ExecutableGraph graph;
	graph.add("Scale", bufs1, access1, &push1, sizeof(push1), 1);
	graph.add("Scale", bufs2, access2, &push2, sizeof(push2), 1);
	EXPECT_EQ(graph.nodeCount(), 2u);
	EXPECT_FALSE(graph.isCompiled());

	auto compileStatus = graph.compile(*rt);
	if (!compileStatus.isOk()) {
		graph.reset(*rt);
		oa::EngineBindlessAccess::deregisterBuffer(*rt, src);
		oa::EngineAllocatorAccess::get(*rt).free(src);
		oa::EngineBindlessAccess::deregisterBuffer(*rt, mid);
		oa::EngineAllocatorAccess::get(*rt).free(mid);
		oa::EngineBindlessAccess::deregisterBuffer(*rt, dst);
		oa::EngineAllocatorAccess::get(*rt).free(dst);
		GTEST_SKIP() << "scale shader not loaded";
	}
	EXPECT_TRUE(graph.isCompiled());

	// replay 5 times
	for (oa::I32 rep = 0; rep < 5; ++rep) {
		for (oa::I32 i = 0; i < 256; ++i) data[i] = static_cast<oa::F32>(i + 1);

		auto status = graph.replay(*rt);
		ASSERT_TRUE(status.isOk()) << "replay " << rep << " failed";
		// replay() is non-blocking (submits primaryCb_ with a timeline semaphore
		// and returns). results are undefined until the replay completes — the
		// documented contract is to waitForPendingReplay()/sync() before reading.
		ASSERT_TRUE(graph.waitForPendingReplay(*rt).isOk())
			<< "wait for replay " << rep << " failed";

		oa::F32* out = static_cast<oa::F32*>(dst.mappedPtr);
		for (oa::I32 i = 0; i < 256; ++i) {
			EXPECT_NEAR(out[i], static_cast<oa::F32>(i + 1) * 1.0f, 1e-3f)
				<< "mismatch at i=" << i << " rep=" << rep;
		}
	}

	graph.reset(*rt);
	oa::EngineBindlessAccess::deregisterBuffer(*rt, src);
	oa::EngineAllocatorAccess::get(*rt).free(src);
	oa::EngineBindlessAccess::deregisterBuffer(*rt, mid);
	oa::EngineAllocatorAccess::get(*rt).free(mid);
	oa::EngineBindlessAccess::deregisterBuffer(*rt, dst);
	oa::EngineAllocatorAccess::get(*rt).free(dst);
}

TEST(ExecutableGraph, DestructorRetiresPendingReplayWithoutWaiting) {
	auto* rt = testEnginePtr();
	ASSERT_NE(rt, nullptr);

	auto sourceResult = oa::EngineResourceAccess::allocBuffer(
		*rt, 64U * sizeof(oa::F32));
	auto destinationResult = oa::EngineResourceAccess::allocBuffer(
		*rt, 64U * sizeof(oa::F32));
	ASSERT_TRUE(sourceResult.isOk());
	ASSERT_TRUE(destinationResult.isOk());
	auto source = sourceResult.getValue();
	auto destination = destinationResult.getValue();
	auto* sourceData = static_cast<oa::F32*>(source.mappedPtr);
	for (oa::U32 i = 0; i < 64U; ++i) sourceData[i] = static_cast<oa::F32>(i + 1U);
	ASSERT_TRUE(oa::EngineAllocatorAccess::get(*rt).flushHostBuffer(
		source, 0U, source.size));

	struct Push {
		oa::U32 count;
		oa::F32 scale;
	} push{64U, 2.0F};
	oavk::Buffer buffers[] = {source, destination};
	oa::BufferAccess access[] = {
		oa::BufferAccess::Read, oa::BufferAccess::Write};
	oa::Event completion;
	{
		oa::ExecutableGraph graph;
		graph.add("Scale", buffers, access, &push, sizeof(push), 1U);
		ASSERT_TRUE(graph.compile(*rt).isOk());
		ASSERT_TRUE(graph.replay(*rt).isOk());
		completion = graph.lastCompletion(*rt);
		ASSERT_TRUE(completion.isValid());
	} // pending resources transfer to engine retirement; this never host-waits.

	ASSERT_TRUE(completion.wait().isOk());
	oa::EngineAccess(*rt).collectRetiredExecutionPlans();
	ASSERT_TRUE(oa::EngineAllocatorAccess::get(*rt).invalidateHostBuffer(
		destination, 0U, destination.size));
	const auto* output = static_cast<const oa::F32*>(destination.mappedPtr);
	for (oa::U32 i = 0; i < 64U; ++i) {
		EXPECT_NEAR(output[i], static_cast<oa::F32>(i + 1U) * 2.0F, 1e-4F);
	}

	oa::EngineResourceAccess::freeBuffer(*rt, source);
	oa::EngineResourceAccess::freeBuffer(*rt, destination);
}

TEST(ExecutableGraph, TimedReplayRequiresWaitBeforeQueryReuse) {
	auto* rt = testEnginePtr();
	ASSERT_NE(rt, nullptr);

	auto srcRes = oa::EngineResourceAccess::allocBuffer(*rt, 64 * sizeof(oa::F32));
	auto dstRes = oa::EngineResourceAccess::allocBuffer(*rt, 64 * sizeof(oa::F32));
	ASSERT_TRUE(srcRes.isOk() && dstRes.isOk());
	auto src = std::move(*srcRes);
	auto dst = std::move(*dstRes);
	for (oa::U32 i = 0; i < 64; ++i) {
		static_cast<oa::F32*>(src.mappedPtr)[i] = static_cast<oa::F32>(i);
	}
	ASSERT_TRUE(oa::EngineAllocatorAccess::get(*rt).flushHostBuffer(src, 0, src.size));

	struct { oa::U32 N; oa::F32 scale; } push{64, 2.0F};
	oavk::Buffer bufs[] = {src, dst};
	oa::BufferAccess access[] = {oa::BufferAccess::Read, oa::BufferAccess::Write};
	oa::ExecutableGraph graph;
	graph.add("Scale", bufs, access, &push, sizeof(push), 1);
	graph.setReplayTimingEnabled(true);
	ASSERT_TRUE(graph.compile(*rt).isOk());
	ASSERT_TRUE(graph.replay(*rt).isOk());
	EXPECT_FALSE(graph.replay(*rt).isOk());
	ASSERT_TRUE(graph.waitForPendingReplay(*rt).isOk());
	EXPECT_GT(graph.lastReplayGpuMs(), 0.0);
	ASSERT_TRUE(graph.replay(*rt).isOk());
	ASSERT_TRUE(graph.waitForPendingReplay(*rt).isOk());

	graph.reset(*rt);
	oa::EngineResourceAccess::freeBuffer(*rt, src);
	oa::EngineResourceAccess::freeBuffer(*rt, dst);
}

TEST(ExecutableGraph, HazardPlannerTracksReadBeforeWrite) {
	auto* rt = testEnginePtr();
	ASSERT_NE(rt, nullptr);

	auto xRes = oa::EngineResourceAccess::allocBuffer(*rt, 256 * sizeof(oa::F32));
	auto aRes = oa::EngineResourceAccess::allocBuffer(*rt, 256 * sizeof(oa::F32));
	auto bRes = oa::EngineResourceAccess::allocBuffer(*rt, 256 * sizeof(oa::F32));
	ASSERT_TRUE(xRes.isOk() && aRes.isOk() && bRes.isOk());
	auto x = xRes.getValue();
	auto a = aRes.getValue();
	auto b = bRes.getValue();

	struct { oa::U32 N; oa::F32 scale; } push{256, 1.0F};
	oa::ExecutableGraph graph;
	{
		oavk::Buffer bufs[] = {x, a};
		oa::BufferAccess access[] = {
			oa::BufferAccess::Read, oa::BufferAccess::Write};
		graph.add("Scale", bufs, access, &push, sizeof(push), 1);
	}
	{
		// X was only read by the first node. A writer still needs an execution
		// dependency even though there is no preceding in-graph writer for X.
		oavk::Buffer bufs[] = {b, x};
		oa::BufferAccess access[] = {
			oa::BufferAccess::Read, oa::BufferAccess::Write};
		graph.add("Scale", bufs, access, &push, sizeof(push), 1);
	}

	ASSERT_TRUE(graph.compile(*rt).isOk());
	const auto stats = graph.getStats();
	EXPECT_EQ(stats.warBarrierCount, 1U);
	EXPECT_GE(stats.barrierCount, 1U);

	graph.reset(*rt);

	oa::ExecutableGraph indirectGraph;
	{
		oavk::Buffer bufs[] = {x, a};
		oa::BufferAccess access[] = {
			oa::BufferAccess::Read, oa::BufferAccess::Write};
		indirectGraph.add("Scale", bufs, access, &push, sizeof(push), 1);
	}
	{
		oavk::Buffer bufs[] = {x, b};
		oa::BufferAccess access[] = {
			oa::BufferAccess::Read, oa::BufferAccess::Write};
		indirectGraph.addIndirect(
			"Scale", bufs, access, &push, sizeof(push), a, 0);
	}
	ASSERT_TRUE(indirectGraph.compile(*rt).isOk());
	EXPECT_EQ(indirectGraph.getStats().indirectBarrierCount, 1U);
	indirectGraph.reset(*rt);

	oa::EngineResourceAccess::freeBuffer(*rt, x);
	oa::EngineResourceAccess::freeBuffer(*rt, a);
	oa::EngineResourceAccess::freeBuffer(*rt, b);
}

TEST(ExecutableGraph, IndirectArgumentHazardAndCacheIdentity) {
	auto* rt = testEnginePtr();
	ASSERT_NE(rt, nullptr);

	constexpr oa::U32 N = 64;
	auto srcRes = oa::EngineResourceAccess::allocBuffer(*rt, N * sizeof(oa::F32));
	auto dstRes = oa::EngineResourceAccess::allocBuffer(*rt, N * sizeof(oa::F32));
	auto argsRes = oa::EngineResourceAccess::allocBuffer(*rt, 6 * sizeof(oa::U32));
	ASSERT_TRUE(srcRes.isOk() && dstRes.isOk() && argsRes.isOk());
	auto src = std::move(*srcRes);
	auto dst = std::move(*dstRes);
	auto args = std::move(*argsRes);

	auto* srcData = static_cast<oa::F32*>(src.mappedPtr);
	auto* dstData = static_cast<oa::F32*>(dst.mappedPtr);
	auto* dispatchArgs = static_cast<oa::U32*>(args.mappedPtr);
	for (oa::U32 i = 0; i < N; ++i) {
		srcData[i] = static_cast<oa::F32>(i + 1);
		dstData[i] = 0.0F;
	}
	dispatchArgs[0] = 1; dispatchArgs[1] = 1; dispatchArgs[2] = 1;
	dispatchArgs[3] = 0; dispatchArgs[4] = 1; dispatchArgs[5] = 1;
	ASSERT_TRUE(oa::EngineAllocatorAccess::get(*rt).flushHostBuffer(src, 0, src.size));
	ASSERT_TRUE(oa::EngineAllocatorAccess::get(*rt).flushHostBuffer(dst, 0, dst.size));
	ASSERT_TRUE(oa::EngineAllocatorAccess::get(*rt).flushHostBuffer(args, 0, args.size));

	struct { oa::U32 N; oa::F32 scale; } push{N, 2.0F};
	oavk::Buffer bufs[] = {src, dst};
	oa::BufferAccess access[] = {
		oa::BufferAccess::Read, oa::BufferAccess::Write};

	oa::ExecutableGraph graph;
	graph.addIndirect("Scale", bufs, access, &push, sizeof(push), args, 0);
	ASSERT_TRUE(graph.execute(*rt).isOk());
	EXPECT_NEAR(dstData[0], 2.0F, 1e-3F);
	for (oa::U32 i = 0; i < N; ++i) dstData[i] = 0.0F;
	ASSERT_TRUE(oa::EngineAllocatorAccess::get(*rt).flushHostBuffer(dst, 0, dst.size));
	ASSERT_TRUE(graph.compile(*rt).isOk());
	ASSERT_TRUE(graph.replay(*rt).isOk());
	ASSERT_TRUE(graph.waitForPendingReplay(*rt).isOk());
	EXPECT_NEAR(dstData[0], 2.0F, 1e-3F);

	// The only topology change is the indirect offset. compile must not reuse
	// the old command buffer: the second command has groupCountX=0 and skips.
	for (oa::U32 i = 0; i < N; ++i) dstData[i] = 0.0F;
	ASSERT_TRUE(oa::EngineAllocatorAccess::get(*rt).flushHostBuffer(dst, 0, dst.size));
	graph.clearNodes();
	graph.addIndirect("Scale", bufs, access, &push, sizeof(push), args,
		3 * sizeof(oa::U32));
	ASSERT_TRUE(graph.compile(*rt).isOk());
	ASSERT_TRUE(graph.replay(*rt).isOk());
	ASSERT_TRUE(graph.waitForPendingReplay(*rt).isOk());
	EXPECT_NEAR(dstData[0], 0.0F, 1e-6F);

	const auto lifetimes = graph.computeLifetimes();
	EXPECT_EQ(lifetimes.size(), 3U); // src, dst, and indirect argument buffer

	graph.reset(*rt);
	oa::EngineResourceAccess::freeBuffer(*rt, src);
	oa::EngineResourceAccess::freeBuffer(*rt, dst);
	oa::EngineResourceAccess::freeBuffer(*rt, args);
}

TEST(ExecutableGraph, EmptyGraph) {
	auto* rt = testEnginePtr();
	ASSERT_NE(rt, nullptr);

	oa::ExecutableGraph graph;
	EXPECT_EQ(graph.nodeCount(), 0u);
	EXPECT_TRUE(graph.execute(*rt).isOk());
	EXPECT_TRUE(graph.compile(*rt).isOk());
	EXPECT_TRUE(graph.isCompiled());
	EXPECT_TRUE(graph.replay(*rt).isOk());

	auto lifetimes = graph.computeLifetimes();
	EXPECT_TRUE(lifetimes.empty());

	auto groups = graph.computeAliasGroups();
	EXPECT_TRUE(groups.empty());

	auto stats = graph.getStats();
	EXPECT_EQ(stats.dispatchCount, 0u);
	EXPECT_EQ(stats.barrierCount, 0u);

	graph.reset(*rt);
}

TEST(ExecutableGraph, StructuredKernelSelectionsAreCountedAndDumped) {
	oa::ExecutableGraph graph;
	const auto add = [&](oa::StringView inKernel, oa::KernelSelectionKind inSelection) {
		oa::ComputeDispatchDesc dispatch;
		dispatch.kernel = inKernel;
		dispatch.kernelSelection = inSelection;
		graph.add(dispatch);
	};
	add("DirectSelection", oa::KernelSelectionKind::Direct);
	add("PrecisionFallback", oa::KernelSelectionKind::PrecisionFallback);
	add("LayoutFallback", oa::KernelSelectionKind::LayoutFallback);
	add("NaiveFallback", oa::KernelSelectionKind::NaiveFallback);
	add("UnclassifiedFixedDispatch", oa::KernelSelectionKind::Unspecified);

	const auto stats = graph.getStats();
	EXPECT_EQ(stats.dispatchCount, 5U);
	EXPECT_EQ(stats.kernelSelectionCount, 4U);
	EXPECT_EQ(stats.kernelFallbackCount, 3U);
	EXPECT_EQ(stats.precisionFallbackCount, 1U);
	EXPECT_EQ(stats.layoutFallbackCount, 1U);
	EXPECT_EQ(stats.naiveFallbackCount, 1U);

	const auto report = graph.debugReportJson("selection-contract");
	EXPECT_NE(report.view().stdView().find("\"kernel_selections\": 4"),
		std::string::npos);
	EXPECT_NE(report.view().stdView().find("\"kernel_fallbacks\": 3"),
		std::string::npos);
	EXPECT_NE(report.view().stdView().find(
		"\"kernel_selection\": \"precision_fallback\""), std::string::npos);
	EXPECT_NE(report.view().stdView().find(
		"\"kernel_selection\": \"layout_fallback\""), std::string::npos);
	EXPECT_NE(report.view().stdView().find(
		"\"kernel_selection\": \"naive_fallback\""), std::string::npos);
	EXPECT_NE(report.view().stdView().find(
		"\"kernel_selection\": null"), std::string::npos);
}

TEST(ExecutableGraph, HostReadbackBarrierIsAnExplicitCompletionPolicy) {
	auto* rt = testEnginePtr();
	ASSERT_NE(rt, nullptr);

	auto srcRes = oa::EngineResourceAccess::allocBuffer(*rt, 256 * sizeof(oa::F32));
	auto dstRes = oa::EngineResourceAccess::allocBuffer(*rt, 256 * sizeof(oa::F32));
	ASSERT_TRUE(srcRes.isOk() && dstRes.isOk());
	auto src = std::move(*srcRes);
	auto dst = std::move(*dstRes);

	struct { oa::U32 N; oa::F32 scale; } push{256, 1.0F};
	oavk::Buffer buffers[] = {src, dst};
	oa::BufferAccess access[] = {oa::BufferAccess::Read, oa::BufferAccess::Write};

	oa::ExecutableGraph graph;
	graph.add("Scale", buffers, access, &push, sizeof(push), 1);
	ASSERT_TRUE(graph.compile(*rt).isOk());
	EXPECT_EQ(graph.getStats().hostBarrierCount, 1U);

	graph.setHostReadbackRequired(false);
	EXPECT_FALSE(graph.isCompiled());
	ASSERT_TRUE(graph.compile(*rt).isOk());
	EXPECT_EQ(graph.getStats().hostBarrierCount, 0U);

	graph.reset(*rt);
	oa::EngineResourceAccess::freeBuffer(*rt, src);
	oa::EngineResourceAccess::freeBuffer(*rt, dst);
}

TEST(ExecutableGraph, ExplicitSubmissionUsesExactBoundaryAndReusesStaticGraph) {
	auto* rt = testEnginePtr();
	ASSERT_NE(rt, nullptr);
	auto& ctx = oa::ExecutionSession::getActive();
	ctx.clear();

	constexpr oa::U32 N = 256;
	auto src = oa::FnMatrix::empty(oa::MatrixShape{N}, oa::ScalarType::Float32);
	auto mid = oa::FnMatrix::empty(oa::MatrixShape{N}, oa::ScalarType::Float32);
	auto dst = oa::FnMatrix::empty(oa::MatrixShape{N}, oa::ScalarType::Float32);
	auto unrelatedSrc = oa::FnMatrix::empty(oa::MatrixShape{N}, oa::ScalarType::Float32);
	auto unrelatedDst = oa::FnMatrix::empty(oa::MatrixShape{N}, oa::ScalarType::Float32);
	ASSERT_TRUE(src.hasStorage() && mid.hasStorage() && dst.hasStorage());
	ASSERT_TRUE(unrelatedSrc.hasStorage() && unrelatedDst.hasStorage());

	for (oa::U32 i = 0; i < N; ++i) {
		src.dataAs<oa::F32>()[i] = static_cast<oa::F32>(i + 1);
		unrelatedSrc.dataAs<oa::F32>()[i] = static_cast<oa::F32>(N - i);
		dst.dataAs<oa::F32>()[i] = 0.0F;
	}
	ASSERT_TRUE(oa::EngineAllocatorAccess::get(*rt).flushHostBuffer(
		oa::MatrixAccess::descriptor(src), 0, src.byteSize()));
	ASSERT_TRUE(oa::EngineAllocatorAccess::get(*rt).flushHostBuffer(
		oa::MatrixAccess::descriptor(unrelatedSrc), 0, unrelatedSrc.byteSize()));
	ASSERT_TRUE(oa::EngineAllocatorAccess::get(*rt).flushHostBuffer(
		oa::MatrixAccess::descriptor(dst), 0, dst.byteSize()));

	oa::BufferAccess access[] = {oa::BufferAccess::Read, oa::BufferAccess::Write};
	auto recordScale = [&](const oa::Matrix& inSrc, oa::Matrix& outDst, oa::F32 inScale) {
		struct { oa::U32 count; oa::F32 scale; } push{N, inScale};
		ctx.add( "Scale", {&inSrc, &outDst}, access, &push, sizeof(push), 1);
	};
	auto submitGraph = [&]() {
		recordScale(src, mid, 2.0F);
		recordScale(unrelatedSrc, unrelatedDst, 3.0F);
		recordScale(mid, dst, 4.0F);
		auto completion = ctx.submit();
		EXPECT_TRUE(completion.isOk());
		ASSERT_TRUE(completion.getValue().isValid());
		EXPECT_TRUE(ctx.wait(completion.getValue()).isOk());
	};

	submitGraph();
	auto first = ctx.lastExecutionStats();
	EXPECT_EQ(first.graphCount, 1U);
	EXPECT_EQ(first.submissionCount, 1U);
	EXPECT_EQ(first.dispatchCount, 3U);
	EXPECT_EQ(first.nodeCount, first.dispatchCount);
	EXPECT_GT(first.intraGraphBarrierCount, 0U);
	EXPECT_GT(first.referencedBufferBytes, 0U);
	EXPECT_EQ(first.boundaryBarrierCount, 0U);
	EXPECT_EQ(first.hostBarrierCount, 1U);
	for (oa::U32 i = 0; i < N; ++i) {
		EXPECT_NEAR(dst.dataAs<oa::F32>()[i], static_cast<oa::F32>(i + 1) * 8.0F, 1e-3F);
	}

	// The same stable buffers, topology and push constants must reuse the
	// compiled graph on the next explicit submission.
	submitGraph();
	const auto second = ctx.lastExecutionStats();
	EXPECT_EQ(second.graphCount, 1U);
	EXPECT_EQ(second.submissionCount, 1U);
	EXPECT_EQ(second.dispatchCount, 3U);
	EXPECT_EQ(second.compileCacheHits, 1U);
	EXPECT_EQ(second.boundaryBarrierCount, 0U);
	EXPECT_EQ(second.hostBarrierCount, 1U);

	ctx.clear();
}

// =============================================================================
// GPU COMPILATION TESTS (phase 2)
TEST(ExecutableGraph, BarrierOverhead) {
	auto* rt = testEnginePtr();
	ASSERT_NE(rt, nullptr);

	printHeader("BARRIER ANALYSIS — CPU PLANNING COST");
	fprintf(stderr, "  %-12s %12s  %12s  %12s\n",
		"Dispatches", "Barriers", "Barrier%", "CPU Cost");
	printBar();

	struct Config { oa::U32 n; const char* label; };
	Config configs[] = {
		{6,  "6 dispatches"},
		{12, "12 dispatches"},
		{25, "25 dispatches"},
		{50, "50 dispatches"},
	};

	for (auto& cfg : configs) {
		oa::U32 N = cfg.n;
		oa::Vec<oavk::Buffer> bufs(N + 1);
		for (oa::U32 i = 0; i <= N; ++i) {
			auto res = oa::EngineAllocatorAccess::get(*rt).allocHostVisible(256 * sizeof(oa::F32));
			ASSERT_TRUE(res.isOk());
			bufs[i] = res.getValue();
			ASSERT_NE(oa::EngineBindlessAccess::registerBuffer(*rt, bufs[i]), OA_BINDLESS_INVALID);
		}

		oa::ExecutableGraph graph;
		buildChainGraph(graph, bufs, N);

		auto compileStatus = graph.compile(*rt);
		ASSERT_TRUE(compileStatus.isOk()) << compileStatus.toString().cStr();

		auto stats = graph.getStats();
		double barrierPct = N > 0 ? (100.0 * stats.barrierCount / N) : 0.0;

		// Estimate CPU cost of barrier computation
		// Each barrier requires hashmap lookup + comparison (~50ns on modern CPU)
		double cpuCostUs = stats.barrierCount * 0.05;  // 50ns per barrier

		fprintf(stderr, "  %-12s %10u    %10.0f%%   %10.1f µs\n",
			cfg.label, stats.barrierCount, barrierPct, cpuCostUs);

		graph.reset(*rt);
		for (auto& b : bufs) {
			oa::EngineBindlessAccess::deregisterBuffer(*rt, b);
			oa::EngineAllocatorAccess::get(*rt).free(b);
		}
	}

	fprintf(stderr, "\n  Barriers are planned on the CPU and encoded once during graph compilation.\n");
}

// =============================================================================
// MEMORY ANALYSIS
// =============================================================================

TEST(ExecutableGraph, MemoryAliasing) {
	auto* rt = testEnginePtr();
	ASSERT_NE(rt, nullptr);

	printHeader("MEMORY ALIASING — Potential VRAM savings");
	fprintf(stderr, "  %-12s %12s  %12s  %12s  %8s\n",
		"Dispatches", "total bufs", "alias groups", "Savings", "Pct");
	printBar();

	struct Config { oa::U32 n; const char* label; };
	Config configs[] = {
		{6,  "6 dispatches"},
		{12, "12 dispatches"},
		{25, "25 dispatches"},
	};

	for (auto& cfg : configs) {
		oa::U32 N = cfg.n;
		oa::Vec<oavk::Buffer> bufs(N + 1);
		for (oa::U32 i = 0; i <= N; ++i) {
			auto res = oa::EngineAllocatorAccess::get(*rt).allocHostVisible(4096);
			ASSERT_TRUE(res.isOk());
			bufs[i] = res.getValue();
		}

		oa::ExecutableGraph graph;
		buildChainGraph(graph, bufs, N);

		auto stats = graph.getStats();
		auto groups = graph.computeAliasGroups();
		auto lifetimes = graph.computeLifetimes();

		double savingsPct = stats.totalBufferBytes > 0
			? (100.0 * stats.potentialAliasSavings / stats.totalBufferBytes) : 0.0;

		fprintf(stderr, "  %-12s %10zu    %10zu    %8llu B   %5.1f%%\n",
			cfg.label,
			lifetimes.size(), groups.size(),
			static_cast<unsigned long long>(stats.potentialAliasSavings),
			savingsPct);

		for (auto& b : bufs) oa::EngineAllocatorAccess::get(*rt).free(b);
	}
}

TEST(ExecutableGraph, AllocatorBackedAliasesExecuteCorrectly) {
	auto* rt = testEnginePtr();
	ASSERT_NE(rt, nullptr);
	constexpr oa::U32 N = 256;
	oa::Vec<oa::Matrix> matrices;
	oa::Vec<oavk::Buffer> buffers;
	for (oa::U32 i = 0; i < 5U; ++i) {
		matrices.pushBack(oa::FnMatrix::empty(
			{static_cast<oa::I64>(N)}, oa::ScalarType::Float32,
			oa::MemoryPlacement::HostUpload));
		ASSERT_TRUE(matrices.back().hasStorage());
		buffers.pushBack(oa::MatrixAccess::descriptor(matrices.back()));
	}
	auto* input = matrices[0].dataAs<oa::F32>();
	for (oa::U32 i = 0; i < N; ++i) input[i] = static_cast<oa::F32>(i + 1U);
	ASSERT_TRUE(oa::EngineAllocatorAccess::get(*rt).flushHostBuffer(buffers[0], 0, buffers[0].size));

	oa::ExecutableGraph graph;
	buildChainGraph(graph, buffers, 4);
	oa::Matrix* eligible[] = {&matrices[1], &matrices[3]};
	const auto beforeAlias = oa::EngineAllocatorAccess::get(*rt).getStats();
	ASSERT_TRUE(graph.materializeAliases(*rt, eligible).isOk());
	const auto afterAlias = oa::EngineAllocatorAccess::get(*rt).getStats();
	EXPECT_EQ(graph.materializedAliasSavings(), N * sizeof(oa::F32));
	EXPECT_GE(beforeAlias.allocationBytes, afterAlias.allocationBytes + N * sizeof(oa::F32));
	EXPECT_EQ(oa::MatrixAccess::descriptor(matrices[1]).placement,
		oa::MemoryPlacement::HostUpload);
	EXPECT_EQ(matrices[1].data(), matrices[3].data());
	ASSERT_TRUE(graph.compile(*rt).isOk());
	// matrices 1 and 3 are distinct VkBuffer handles over one allocation.
	// Their non-overlapping logical lifetimes still require one global memory
	// dependency when the physical bytes are handed from the first alias to the
	// second; a per-buffer barrier cannot scope both handles.
	EXPECT_EQ(graph.getStats().aliasBarrierCount, 1U);
	const auto debugReport = graph.debugReportJson("allocator-alias").stdStr();
	EXPECT_NE(debugReport.find("\"scope\": \"memory_alias\""), std::string::npos);
	EXPECT_NE(debugReport.find("\"reason\": \"read_after_write\""),
		std::string::npos);
	EXPECT_NE(debugReport.find("\"ownership_transfer\": false"),
		std::string::npos);
	ASSERT_TRUE(graph.replay(*rt).isOk());
	ASSERT_TRUE(graph.waitForPendingReplay(*rt).isOk());
	ASSERT_TRUE(oa::EngineAllocatorAccess::get(*rt).invalidateHostBuffer(
		oa::MatrixAccess::descriptor(matrices[4]), 0,
		oa::MatrixAccess::descriptor(matrices[4]).size));
	const auto* output = matrices[4].dataAs<const oa::F32>();
	const oa::F32 factor = 1.001F * 1.001F * 1.001F * 1.001F;
	for (oa::U32 i = 0; i < N; ++i) {
		EXPECT_NEAR(output[i], static_cast<oa::F32>(i + 1U) * factor, 1e-4F);
	}

	graph.reset(*rt);
}

TEST(ExecutableGraph, AliasMaterializationRejectsExternallyOwnedTransient) {
	auto* rt = testEnginePtr();
	ASSERT_NE(rt, nullptr);
	constexpr oa::U32 N = 64;
	oa::Vec<oa::Matrix> matrices;
	oa::Vec<oavk::Buffer> buffers;
	for (oa::U32 i = 0; i < 5U; ++i) {
		matrices.pushBack(oa::FnMatrix::empty(
			{static_cast<oa::I64>(N)}, oa::ScalarType::Float32,
			oa::MemoryPlacement::HostUpload));
		buffers.pushBack(oa::MatrixAccess::descriptor(matrices.back()));
	}
	oa::Matrix retainedView = matrices[1];
	oa::ExecutableGraph graph;
	buildChainGraph(graph, buffers, 4);
	oa::Matrix* eligible[] = {&matrices[1], &matrices[3]};
	auto status = graph.materializeAliases(*rt, eligible);
	EXPECT_FALSE(status.isOk());
	EXPECT_EQ(status.getCode(), oa::StatusCode::FailedPrecondition);
	EXPECT_EQ(oa::MatrixAccess::descriptor(retainedView).buffer,
		oa::MatrixAccess::descriptor(matrices[1]).buffer);
}

TEST(ExecutableGraph, AllocatorBackedDeviceLocalAliasesExecuteCorrectly) {
	auto* rt = testEnginePtr();
	ASSERT_NE(rt, nullptr);
	constexpr oa::U32 N = 256;
	oa::Vec<oa::Matrix> matrices;
	oa::Vec<oavk::Buffer> buffers;
	for (oa::U32 i = 0; i < 5U; ++i) {
		const auto placement = (i == 0U or i == 4U)
			? oa::MemoryPlacement::HostUpload : oa::MemoryPlacement::DeviceLocal;
		matrices.pushBack(oa::FnMatrix::empty(
			{static_cast<oa::I64>(N)}, oa::ScalarType::Float32, placement));
		ASSERT_TRUE(matrices.back().hasStorage());
		buffers.pushBack(oa::MatrixAccess::descriptor(matrices.back()));
	}
	auto* input = matrices[0].dataAs<oa::F32>();
	for (oa::U32 i = 0; i < N; ++i) input[i] = static_cast<oa::F32>(i + 1U);
	ASSERT_TRUE(oa::EngineAllocatorAccess::get(*rt).flushHostBuffer(buffers[0], 0, buffers[0].size));

	oa::ExecutableGraph graph;
	buildChainGraph(graph, buffers, 4);
	oa::Matrix* eligible[] = {&matrices[1], &matrices[3]};
	ASSERT_TRUE(graph.materializeAliases(*rt, eligible).isOk());
	EXPECT_EQ(oa::MatrixAccess::descriptor(matrices[1]).placement,
		oa::MemoryPlacement::DeviceLocal);
	EXPECT_EQ(oa::MatrixAccess::descriptor(matrices[3]).placement,
		oa::MemoryPlacement::DeviceLocal);
	EXPECT_EQ(matrices[1].data(), nullptr);
	ASSERT_TRUE(graph.execute(*rt).isOk());
	ASSERT_TRUE(oa::EngineAllocatorAccess::get(*rt).invalidateHostBuffer(
		oa::MatrixAccess::descriptor(matrices[4]), 0,
		oa::MatrixAccess::descriptor(matrices[4]).size));
	const auto* output = matrices[4].dataAs<const oa::F32>();
	const oa::F32 factor = 1.001F * 1.001F * 1.001F * 1.001F;
	for (oa::U32 i = 0; i < N; ++i) {
		EXPECT_NEAR(output[i], static_cast<oa::F32>(i + 1U) * factor, 1e-4F);
	}
	graph.reset(*rt);
}

TEST(DnnPlanner, PartitionsQkvProjectionGroupGatedFfnAndFallback) {
	oa::DnnGraph graph;
	auto addMatrix = [&](oa::U32 id, oa::MatrixShape shape, bool external) {
		ASSERT_TRUE(graph.addMatrix({.id = id, .shape = shape,
			.dtype = oa::ScalarType::Float32, .external = external,
			.isVirtual = not external}).isOk());
	};
	addMatrix(0, {8, 16}, true);       // shared activation
	addMatrix(1, {16, 16}, true); addMatrix(2, {16, 16}, true); addMatrix(3, {16, 16}, true);
	addMatrix(4, {8, 16}, false); addMatrix(5, {8, 16}, false); addMatrix(6, {8, 16}, false);
	addMatrix(7, {16, 16}, true); addMatrix(8, {16, 16}, true);
	addMatrix(9, {8, 16}, false); addMatrix(10, {8, 16}, false);
	addMatrix(11, {8, 16}, false); addMatrix(12, {8, 16}, false);
	addMatrix(13, {8, 16}, true);

	auto op = [&](oa::DnnOpType type, std::initializer_list<oa::U32> inputs,
		std::initializer_list<oa::U32> outputs) {
		oa::DnnOpDesc desc; desc.type = type;
		desc.inputs = inputs; desc.outputs = outputs;
		ASSERT_TRUE(graph.addOp(desc).isOk());
	};
	op(oa::DnnOpType::Matmul, {0, 1}, {4});
	op(oa::DnnOpType::Matmul, {0, 2}, {5});
	op(oa::DnnOpType::Matmul, {0, 3}, {6});
	op(oa::DnnOpType::Matmul, {0, 7}, {9});
	op(oa::DnnOpType::Matmul, {0, 8}, {10});
	op(oa::DnnOpType::Silu, {9}, {11});
	op(oa::DnnOpType::Multiply, {11, 10}, {12});
	op(oa::DnnOpType::Add, {12, 0}, {13});

	auto result = oa::DnnPlanner::plan(graph);
	ASSERT_TRUE(result.isOk()) << result.getStatus().getMessage().data();
	const auto& plan = result.getValue();
	ASSERT_EQ(plan.partitions.size(), 3U);
	EXPECT_EQ(plan.partitions[0].engine, oa::DnnEngineType::QkvProjectionGroup);
	EXPECT_EQ(plan.partitions[0].ops.size(), 3U);
	EXPECT_EQ(plan.partitions[1].engine, oa::DnnEngineType::GatedFfn);
	EXPECT_EQ(plan.partitions[1].ops.size(), 4U);
	EXPECT_EQ(plan.partitions[2].engine, oa::DnnEngineType::Portable);
	EXPECT_NE(plan.graphHash, 0U);
}

TEST(DnnPlanner, RejectsUseBeforeProducer) {
	oa::DnnGraph invalidShape;
	EXPECT_FALSE(invalidShape.addMatrix({
		.id = 0, .shape = {-2, -3}, .external = true}).isOk());

	oa::DnnGraph graph;
	ASSERT_TRUE(graph.addMatrix({.id = 0, .shape = {2, 2}, .external = false}).isOk());
	ASSERT_TRUE(graph.addMatrix({.id = 1, .shape = {2, 2}, .external = false}).isOk());
	oa::DnnOpDesc op; op.type = oa::DnnOpType::Relu; op.inputs = {0}; op.outputs = {1};
	ASSERT_TRUE(graph.addOp(op).isOk());
	EXPECT_FALSE(graph.validate().isOk());
}

TEST(DnnPlanner, GraphHashIncludesEpilogueAndPolicyContracts) {
	auto makePlan = [](oa::GemmEpilogue epilogue, oa::Bool allowRecompute) {
		oa::DnnGraph graph;
		EXPECT_TRUE(graph.addMatrix({.id = 0, .shape = {2, 3}, .external = true}).isOk());
		EXPECT_TRUE(graph.addMatrix({.id = 1, .shape = {4, 3}, .external = true}).isOk());
		EXPECT_TRUE(graph.addMatrix({.id = 2, .shape = {2, 4}}).isOk());
		oa::DnnOpDesc op;
		op.type = oa::DnnOpType::Matmul;
		op.inputs = {0, 1};
		op.outputs = {2};
		op.epilogue = epilogue;
		EXPECT_TRUE(graph.addOp(op).isOk());
		oa::DnnPolicy policy;
		policy.allowRecompute = allowRecompute;
		return oa::DnnPlanner::plan(graph, policy);
	};

	auto raw = makePlan(oa::GemmEpilogue::None, true);
	auto fused = makePlan(oa::GemmEpilogue::BiasGelu, true);
	auto noRecompute = makePlan(oa::GemmEpilogue::BiasGelu, false);
	ASSERT_TRUE(raw.isOk());
	ASSERT_TRUE(fused.isOk());
	ASSERT_TRUE(noRecompute.isOk());
	EXPECT_NE(raw.getValue().graphHash, fused.getValue().graphHash);
	EXPECT_NE(fused.getValue().graphHash, noRecompute.getValue().graphHash);
	EXPECT_EQ(fused.getValue().recognizedPartitionCount, 1U);
}

TEST(DnnPlanner, SemanticCaptureUsesSchemaOwnedOptionalEpilogueRole) {
	auto makePlan = [](oa::Bool withBias, oa::Bool withAutograd) {
		oa::SemanticGraph graph;
		oa::SemanticValueDesc xDesc;
		xDesc.name = "x";
		xDesc.shape = {2, 3};
		xDesc.external = true;
		oa::SemanticValueDesc weightDesc = xDesc;
		weightDesc.name = "weight";
		weightDesc.shape = {4, 3};
		oa::SemanticValueDesc biasDesc = xDesc;
		biasDesc.name = "bias";
		biasDesc.shape = {4};
		oa::SemanticValueDesc outputDesc = xDesc;
		outputDesc.name = "output";
		outputDesc.shape = {2, 4};
		outputDesc.external = false;
		const auto x = graph.addValue(xDesc);
		const auto weight = graph.addValue(weightDesc);
		const auto bias = graph.addValue(biasDesc);
		const auto output = graph.addValue(outputDesc);
		EXPECT_TRUE(x.isOk());
		EXPECT_TRUE(weight.isOk());
		EXPECT_TRUE(bias.isOk());
		EXPECT_TRUE(output.isOk());
		const oa::U32 inputs[] = {
			x.getValue(),
			weight.getValue(),
			withBias ? bias.getValue() : oa::invalidSemanticValueId,
		};
		const oa::U32 outputs[] = {output.getValue()};
		auto operation = graph.addOp(
			oa::detail::opRegistry::FnMatrix::linear,
			inputs, outputs);
		EXPECT_TRUE(operation.isOk());
		if (withAutograd and operation.isOk()) {
			EXPECT_TRUE(graph.attachAutograd(
				operation.getValue(), 0U, 1U).isOk());
		}
		return oa::DnnPlanner::plan(graph);
	};

	auto withoutBias = makePlan(false, false);
	auto withBias = makePlan(true, false);
	auto withAutograd = makePlan(true, true);
	ASSERT_TRUE(withoutBias.isOk());
	ASSERT_TRUE(withBias.isOk());
	ASSERT_TRUE(withAutograd.isOk());
	ASSERT_EQ(withoutBias.getValue().partitions.size(), 1U);
	ASSERT_EQ(withBias.getValue().partitions.size(), 1U);
	EXPECT_EQ(
		withoutBias.getValue().partitions[0].engine,
		oa::DnnEngineType::Portable);
	EXPECT_EQ(
		withBias.getValue().partitions[0].engine,
		oa::DnnEngineType::BlasLtEpilogue);
	EXPECT_EQ(withBias.getValue().recognizedPartitionCount, 1U);
	EXPECT_NE(
		withBias.getValue().graphHash,
		withAutograd.getValue().graphHash);
}
// =============================================================================

// main is provided by MlTestMain.cpp (shared test infrastructure)
