// OaComputeGraph Test Suite — Comprehensive testing of CPU and GPU execution paths
//
// Validates graph construction, synchronization, compiled replay, context
// batching, correctness, performance instrumentation and edge cases.
//
// Test Structure (similar to TestAutograd):
// 1. Basic functionality tests (Add, Execute, Compile, Replay)
// 2. Correctness tests (one-shot, compiled replay and context batching)
// 3. Edge cases (empty graphs, single node, large graphs)
// 4. Performance benchmarks (CPU overhead, throughput)
// 5. Memory analysis (aliasing, lifetimes)

#include "../../OaTest.h"
#include <Oa/Runtime/ComputeGraph.h>
#include <Oa/Runtime/SemanticGraph.h>
#include <Oa/Runtime/Context.h>
#include <Oa/Runtime/GpuTimer.h>
#include <Oa/Runtime/Engine.h>
#include <Oa/Core/Operation.h>
#include <Oa/Core/FnMatrix.h>
#include <Oa/Core/Memory.h>
#include <Oa/Ml/Autograd.h>
#include <Oa/Ml/FnLoss.h>
#include <Oa/Ml/FnMatrix.h>

#include <chrono>
#include <cstring>
#include <cstdio>

// =============================================================================
// HELPERS
// =============================================================================

static void PrintBar() {
	fprintf(stderr, "  ────────────────────────────────────────────────────────────────\n");
}

static void PrintHeader(const char* InSection) {
	fprintf(stderr, "\n");
	PrintBar();
	fprintf(stderr, "  %s\n", InSection);
	PrintBar();
}

template<typename F>
static double MeasureUs(OaI32 InWarmup, OaI32 InIters, F&& InFunc) {
	for (OaI32 i = 0; i < InWarmup; ++i) InFunc();
	
	auto start = std::chrono::high_resolution_clock::now();
	for (OaI32 i = 0; i < InIters; ++i) InFunc();
	auto end = std::chrono::high_resolution_clock::now();
	
	double totalUs = std::chrono::duration<double, std::micro>(end - start).count();
	return totalUs / InIters;
}

// Build a chain of scale dispatches for testing
static void BuildChainGraph(
	OaComputeGraph& OutGraph,
	OaVec<OaVkBuffer>& InBufs,
	OaI32 InNumDispatches)
{
	struct { OaU32 N; OaF32 Scale; } pc = {256, 1.001f};
	
	for (OaI32 i = 0; i < InNumDispatches; ++i) {
		OaVkBuffer bufs[] = {InBufs[i], InBufs[i + 1]};
		OaBufferAccess acc[] = {OaBufferAccess::Read, OaBufferAccess::Write};
		OutGraph.Add("Scale", bufs, acc, &pc, sizeof(pc), 1);
	}
}

// =============================================================================
// BASIC FUNCTIONALITY TESTS
// =============================================================================

TEST(ComputeGraph, DispatchDescriptorCopiesAllMetadata) {
	OaVkBuffer buffers[2];
	buffers[0].Buffer = reinterpret_cast<void*>(0x1000);
	buffers[0].BindlessIndex = 7;
	buffers[1].Buffer = reinterpret_cast<void*>(0x2000);
	buffers[1].BindlessIndex = 11;
	OaBufferAccess access[] = {
		OaBufferAccess::Read, OaBufferAccess::Write};
	struct Push { OaU32 Count; OaF32 Scale; } push{64, 2.0F};
	const OaSemanticOperationId semanticOperations[] = {7U};

	OaComputeDispatchDesc desc;
	desc.Operation = "Scale";
	desc.SemanticOperations = semanticOperations;
	desc.ImplementationId = 0x1234U;
	desc.OperationContractHash = 0x5678U;
	desc.ProblemContractHash = 0x789aU;
	desc.KernelContentHash = 0x9abcU;
	desc.Kernel = "Scale";
	desc.Buffers = buffers;
	desc.Access = access;
	desc.PushData = &push;
	desc.PushSize = sizeof(push);
	desc.Dtype = 1;
	desc.GroupsX = 3;
	desc.GroupsY = 2;
	desc.Queue = OaQueueHint::AsyncCompute;
	desc.NodeIndex = 4;

	OaComputeGraph graph;
	graph.Add(desc);
	ASSERT_EQ(graph.NodeCount(), 1U);

	// Descriptor storage is non-owning, but the graph node must be a complete
	// owning snapshot before Record/Add returns.
	push.Count = 0;
	buffers[0].BindlessIndex = 99;
	const auto nodes = graph.Nodes();
	ASSERT_EQ(nodes.Size(), 1U);
	EXPECT_EQ(nodes[0].Operation, "Scale");
	ASSERT_EQ(nodes[0].SemanticOperations.Size(), 1U);
	EXPECT_EQ(nodes[0].SemanticOperations[0], 7U);
	EXPECT_EQ(nodes[0].ImplementationId, 0x1234U);
	EXPECT_EQ(nodes[0].OperationContractHash, 0x5678U);
	EXPECT_EQ(nodes[0].ProblemContractHash, 0x789aU);
	EXPECT_EQ(nodes[0].KernelContentHash, 0x9abcU);
	EXPECT_EQ(nodes[0].Shader, "Scale");
	EXPECT_EQ(nodes[0].Buffers[0].BindlessIndex, 7U);
	EXPECT_EQ(nodes[0].Dtype, 1U);
	EXPECT_EQ(nodes[0].GroupsX, 3U);
	EXPECT_EQ(nodes[0].GroupsY, 2U);
	EXPECT_EQ(nodes[0].Queue, OaQueueHint::AsyncCompute);
	EXPECT_EQ(nodes[0].NodeIndex, 4U);
	Push copied{};
	std::memcpy(&copied, nodes[0].PushData, sizeof(copied));
	EXPECT_EQ(copied.Count, 64U);
	EXPECT_FLOAT_EQ(copied.Scale, 2.0F);
}

TEST(ComputeGraph, SemanticOperationContractsComeFromSchema) {
	EXPECT_EQ(OaOperationRegistry::Add.Name, "Add");
	EXPECT_NE(OaOperationRegistry::Add.Hash, 0U);
	EXPECT_EQ(OaOperationRegistry::Add.ShapeRule, OaOperationShapeRule::Broadcast);
	EXPECT_EQ(OaOperationRegistry::Add.Lowering, OaOperationLowering::Dispatch);
	EXPECT_EQ(OaOperationRegistry::Add.MutatedInputMask, 0U);
	EXPECT_EQ(OaOperationRegistry::Add.AliasInputForOutput(0),
		OaOperationContract::NoAliasInput);
	EXPECT_EQ(OaOperationRegistry::Add.ControlFlow,
		OaOperationControlFlow::StraightLine);
	EXPECT_EQ(OaOperationRegistry::AddInPlace.Name, "AddInPlace");
	EXPECT_NE(OaOperationRegistry::AddInPlace.Hash, 0U);
	EXPECT_TRUE(OaOperationRegistry::AddInPlace.MutatesInput(0U));
	EXPECT_EQ(OaOperationRegistry::AddInPlace.AliasInputForOutput(0U), 0U);
	EXPECT_EQ(OaOperationRegistry::Scale.Name, "Scale");
	EXPECT_NE(OaOperationRegistry::Scale.Hash, 0U);
	EXPECT_EQ(OaOperationRegistry::Scale.AttributeCount, 1U);
	EXPECT_NE(OaOperationRegistry::Scale.AttributeSignatureHash, 0U);
	EXPECT_EQ(OaOperationRegistry::MatMulNt.Name, "MatMulNt");
	EXPECT_NE(OaOperationRegistry::MatMulNt.Hash, 0U);
	EXPECT_EQ(OaOperationRegistry::MatMulNt.ShapeRule, OaOperationShapeRule::MatMulNt);
	EXPECT_EQ(OaOperationRegistry::MatMulNt.Lowering, OaOperationLowering::Gemm);
	EXPECT_EQ(OaOperationRegistry::LinearWeightBiasBwd.Name, "LinearWeightBiasBwd");
	EXPECT_EQ(OaOperationRegistry::LinearWeightBiasBwd.InputCount, 2U);
	EXPECT_EQ(OaOperationRegistry::LinearWeightBiasBwd.OutputCount, 2U);
	EXPECT_EQ(OaOperationRegistry::LinearWeightBiasBwd.ShapeRule,
		OaOperationShapeRule::Explicit);
	EXPECT_EQ(OaOperationRegistry::LinearWeightBiasBwd.Lowering,
		OaOperationLowering::Dispatch);
	EXPECT_NE(OaOperationRegistry::Add.Hash, OaOperationRegistry::MatMulNt.Hash);
}

TEST(ComputeGraph, SemanticGraphOwnsHandleFreeOperationTopology) {
	OaSemanticGraph graph;
	OaSemanticValueDesc inputA;
	inputA.Name = "input_a";
	inputA.Shape = {2, 3};
	inputA.External = true;
	OaSemanticValueDesc inputB = inputA;
	inputB.Name = "input_b";
	OaSemanticValueDesc output;
	output.Name = "sum";
	output.Shape = {2, 3};
	output.Virtual = true;

	auto a = graph.AddValue(inputA);
	auto b = graph.AddValue(inputB);
	auto sum = graph.AddValue(output);
	ASSERT_TRUE(a.IsOk());
	ASSERT_TRUE(b.IsOk());
	ASSERT_TRUE(sum.IsOk());
	const OaSemanticValueId inputs[] = {a.GetValue(), b.GetValue()};
	const OaSemanticValueId outputs[] = {sum.GetValue()};
	auto operation = graph.AddOperation(
		OaOperationRegistry::Add, inputs, outputs);
	ASSERT_TRUE(operation.IsOk()) << operation.GetStatus().GetMessage();
	EXPECT_EQ(operation.GetValue(), 0U);
	ASSERT_TRUE(graph.AttachAutograd(operation.GetValue(), 0U, 17U).IsOk());
	ASSERT_TRUE(graph.Validate().IsOk());
	ASSERT_EQ(graph.OperationCount(), 1U);
	ASSERT_EQ(graph.ValueCount(), 3U);
	ASSERT_EQ(graph.Autograd().Size(), 1U);
	EXPECT_EQ(graph.Autograd()[0].ForwardOperation, operation.GetValue());
	EXPECT_EQ(graph.Autograd()[0].Output, sum.GetValue());
	EXPECT_EQ(graph.Autograd()[0].OutputIndex, 0U);
	EXPECT_EQ(graph.Autograd()[0].Sequence, 17U);
	EXPECT_EQ(graph.FindValue(sum.GetValue())->Producer, operation.GetValue());

	const auto operations = graph.Operations();
	ASSERT_EQ(operations[0].Accesses.Size(), 3U);
	EXPECT_EQ(operations[0].Accesses[0].Mode, OaSemanticAccessMode::Read);
	EXPECT_EQ(operations[0].Accesses[2].Mode, OaSemanticAccessMode::Write);

	const auto first = graph.DebugReportJson("pilot");
	const auto second = graph.DebugReportJson("pilot");
	EXPECT_EQ(first, second);
	const auto text = first.StdStr();
	EXPECT_NE(text.find("\"schema\": \"oa.semantic_graph.v2\""),
		std::string::npos);
	EXPECT_NE(text.find("\"name\": \"Add\""), std::string::npos);
	EXPECT_NE(text.find("\"lowering\": \"dispatch\""), std::string::npos);
	EXPECT_NE(text.find("\"mode\": \"write\""), std::string::npos);
	EXPECT_NE(text.find("\"control_flow\": \"straight_line\""),
		std::string::npos);
	EXPECT_NE(text.find("\"forward_operation\": 0"), std::string::npos);
	EXPECT_NE(text.find("\"sequence\": 17"), std::string::npos);
	EXPECT_EQ(text.find("VkBuffer"), std::string::npos);
}

TEST(ComputeGraph, SemanticGraphRejectsContractAndSsaViolations) {
	OaSemanticGraph graph;
	OaSemanticValueDesc matrix;
	matrix.Shape = {4};
	auto a = graph.AddValue(matrix);
	auto b = graph.AddValue(matrix);
	auto out = graph.AddValue(matrix);
	ASSERT_TRUE(a.IsOk());
	ASSERT_TRUE(b.IsOk());
	ASSERT_TRUE(out.IsOk());

	const OaSemanticValueId wrongArity[] = {a.GetValue()};
	const OaSemanticValueId outputs[] = {out.GetValue()};
	auto rejectedArity = graph.AddOperation(
		OaOperationRegistry::Add, wrongArity, outputs);
	ASSERT_FALSE(rejectedArity.IsOk());
	EXPECT_EQ(rejectedArity.GetStatus().GetCode(), OaStatusCode::InvalidArgument);

	const OaSemanticValueId inputs[] = {a.GetValue(), b.GetValue()};
	auto accepted = graph.AddOperation(OaOperationRegistry::Add, inputs, outputs);
	ASSERT_TRUE(accepted.IsOk());
	EXPECT_EQ(graph.AttachAutograd(accepted.GetValue(), 0U, 0U).GetCode(),
		OaStatusCode::InvalidArgument);
	EXPECT_EQ(graph.AttachAutograd(accepted.GetValue(), 1U, 1U).GetCode(),
		OaStatusCode::OutOfRange);
	ASSERT_TRUE(graph.AttachAutograd(accepted.GetValue(), 0U, 1U).IsOk());
	EXPECT_EQ(graph.AttachAutograd(accepted.GetValue(), 0U, 2U).GetCode(),
		OaStatusCode::AlreadyExists);
	EXPECT_EQ(graph.CompleteAutograd(accepted.GetValue(), 2U, 1U, 0U).GetCode(),
		OaStatusCode::NotFound);
	EXPECT_EQ(graph.CompleteAutograd(accepted.GetValue(), 1U, 2U, 1U).GetCode(),
		OaStatusCode::OutOfRange);
	ASSERT_TRUE(graph.CompleteAutograd(
		accepted.GetValue(), 1U, 1U, 0U).IsOk());
	EXPECT_EQ(graph.CompleteAutograd(
		accepted.GetValue(), 1U, 1U, 0U).GetCode(),
		OaStatusCode::AlreadyExists);
	auto duplicateProducer = graph.AddOperation(
		OaOperationRegistry::Add, inputs, outputs);
	ASSERT_FALSE(duplicateProducer.IsOk());
	EXPECT_EQ(duplicateProducer.GetStatus().GetCode(), OaStatusCode::AlreadyExists);
	EXPECT_EQ(graph.OperationCount(), 1U);

	auto conditionalOut = graph.AddValue(matrix);
	ASSERT_TRUE(conditionalOut.IsOk());
	OaOperationContract conditional = OaOperationRegistry::Add;
	conditional.Name = "ConditionalAdd";
	conditional.Hash = 0xa5e2b4ff2b15dc61ULL;
	conditional.ControlFlow = OaOperationControlFlow::Conditional;
	const OaSemanticValueId conditionalOutputs[] = {conditionalOut.GetValue()};
	const OaSemanticOperationId dependencies[] = {accepted.GetValue()};
	auto controlled = graph.AddOperation(
		conditional, inputs, conditionalOutputs, dependencies);
	ASSERT_TRUE(controlled.IsOk());
	ASSERT_EQ(graph.Operations()[1].ControlDependencies.Size(), 1U);
	EXPECT_EQ(graph.Operations()[1].ControlDependencies[0], accepted.GetValue());
	EXPECT_EQ(graph.Operations()[1].ControlFlow,
		OaOperationControlFlow::Conditional);
	ASSERT_TRUE(graph.Validate().IsOk());
}

TEST(ComputeGraph, SemanticGraphOwnsTypedOperationAttributes) {
	OaSemanticGraph graph;
	OaSemanticValueDesc input;
	input.Shape = {2, 3};
	input.External = true;
	OaSemanticValueDesc output = input;
	output.External = false;
	output.Virtual = true;
	const auto inputValue = graph.AddValue(input);
	const auto outputValue = graph.AddValue(output);
	ASSERT_TRUE(inputValue.IsOk());
	ASSERT_TRUE(outputValue.IsOk());
	const OaSemanticValueId inputs[] = {inputValue.GetValue()};
	const OaSemanticValueId outputs[] = {outputValue.GetValue()};
	const OaOperationAttribute attributes[] = {
		OaOperationAttribute::FromBoolean("Enabled", true),
		OaOperationAttribute::FromSignedInteger("Axis", -2),
		OaOperationAttribute::FromUnsignedInteger("Seed", 17U),
		OaOperationAttribute::FromFloat("Epsilon", 1.0e-5),
		OaOperationAttribute::FromString("Label", OaString("pilot")),
		OaOperationAttribute::FromShape("Target", OaMatrixShape{2, 3}),
		OaOperationAttribute::FromEnum(
			"Mode", OaString("OaInterpolationMode::Bilinear")),
	};
	const OaOperationContract contract{
		.Name = "TypedAttributeTest",
		.Hash = 0x43e084f8994e12d1ULL,
		.InputKinds = static_cast<OaU32>(OaOperationValueKind::Matrix),
		.OutputKinds = static_cast<OaU32>(OaOperationValueKind::Matrix),
		.InputCount = 1U,
		.OutputCount = 1U,
		.AttributeCount = 7U,
		.AttributeSignatureHash = OaOperationAttributeSignatureHash(attributes),
		.ShapeRule = OaOperationShapeRule::MatchInput,
		.DtypeRule = OaOperationDtypeRule::MatchInput,
		.Differentiation = OaOperationDifferentiation::None,
		.Lowering = OaOperationLowering::Dispatch,
		.Effects = OaOperationEffect::ReadInputs
			| OaOperationEffect::WriteOutputs,
	};
	const OaOperationAttribute wrongAttributes[] = {
		OaOperationAttribute::FromBoolean("WrongName", true),
		OaOperationAttribute::FromSignedInteger("Axis", -2),
		OaOperationAttribute::FromUnsignedInteger("Seed", 17U),
		OaOperationAttribute::FromFloat("Epsilon", 1.0e-5),
		OaOperationAttribute::FromString("Label", OaString("pilot")),
		OaOperationAttribute::FromShape("Target", OaMatrixShape{2, 3}),
		OaOperationAttribute::FromEnum(
			"Mode", OaString("OaInterpolationMode::Bilinear")),
	};
	const auto rejected = graph.AddOperation(
		contract, inputs, outputs, {}, wrongAttributes);
	ASSERT_FALSE(rejected.IsOk());
	EXPECT_EQ(rejected.GetStatus().GetCode(), OaStatusCode::InvalidArgument);

	const auto operation = graph.AddOperation(
		contract, inputs, outputs, {}, attributes);
	ASSERT_TRUE(operation.IsOk()) << operation.GetStatus().GetMessage();
	ASSERT_TRUE(graph.Validate().IsOk());
	ASSERT_EQ(graph.Operations()[0].Attributes.Size(), 7U);
	EXPECT_EQ(graph.Operations()[0].Attributes[5].Shape,
		(OaMatrixShape{2, 3}));
	const auto report = graph.DebugReportJson("typed-attributes").StdStr();
	EXPECT_NE(report.find("\"kind\": \"boolean\", \"value\": true"),
		std::string::npos);
	EXPECT_NE(report.find("\"kind\": \"signed_integer\", \"value\": -2"),
		std::string::npos);
	EXPECT_NE(report.find("\"kind\": \"shape\", \"value\": [2, 3]"),
		std::string::npos);
	EXPECT_NE(report.find(
		"\"kind\": \"enum\", \"value\": "
		"\"OaInterpolationMode::Bilinear\""), std::string::npos);
}

TEST(ComputeGraph, SemanticLoweringAnalysisClassifiesDecompositionAndDebt) {
	OaSemanticGraph semantic;
	OaSemanticValueDesc matrix;
	matrix.Shape = {4};
	auto valueA = semantic.AddValue(matrix);
	auto valueB = semantic.AddValue(matrix);
	auto sum = semantic.AddValue(matrix);
	auto output = semantic.AddValue(matrix);
	ASSERT_TRUE(valueA.IsOk());
	ASSERT_TRUE(valueB.IsOk());
	ASSERT_TRUE(sum.IsOk());
	ASSERT_TRUE(output.IsOk());
	const OaSemanticValueId firstInputs[] = {
		valueA.GetValue(), valueB.GetValue(),
	};
	const OaSemanticValueId firstOutputs[] = {sum.GetValue()};
	auto firstOperation = semantic.AddOperation(
		OaOperationRegistry::Add, firstInputs, firstOutputs);
	ASSERT_TRUE(firstOperation.IsOk());
	const OaSemanticValueId secondInputs[] = {
		sum.GetValue(), valueB.GetValue(),
	};
	const OaSemanticValueId secondOutputs[] = {output.GetValue()};
	auto secondOperation = semantic.AddOperation(
		OaOperationRegistry::Add, secondInputs, secondOutputs);
	ASSERT_TRUE(secondOperation.IsOk());

	OaComputeGraph executable;
	OaComputeDispatchDesc direct;
	direct.Kernel = "AddPart";
	direct.Operation = OaOperationRegistry::Add.Name;
	direct.OperationContractHash = OaOperationRegistry::Add.Hash;
	const OaSemanticOperationId firstProvenance[] = {
		firstOperation.GetValue(),
	};
	direct.SemanticOperations = firstProvenance;
	executable.Add(direct);
	executable.Add(direct);
	const OaSemanticOperationId secondProvenance[] = {
		secondOperation.GetValue(),
	};
	direct.SemanticOperations = secondProvenance;
	executable.Add(direct);
	OaComputeDispatchDesc compatibility;
	compatibility.Kernel = "CompatibilityOnly";
	executable.Add(compatibility);

	auto analyzed = OaAnalyzeSemanticLowering(semantic, executable);
	ASSERT_TRUE(analyzed.IsOk()) << analyzed.GetStatus().GetMessage();
	const auto& result = analyzed.GetValue();
	EXPECT_EQ(result.OperationCount(), 2U);
	EXPECT_EQ(result.SchemaOwnedNodeCount(), 3U);
	EXPECT_EQ(result.CompatibilityNodeCount(), 1U);
	EXPECT_EQ(result.DirectOperationCount(), 1U);
	EXPECT_EQ(result.DecomposedOperationCount(), 1U);
	EXPECT_EQ(result.FusedOperationCount(), 0U);
	EXPECT_EQ(result.FusedNodeCount(), 0U);
	EXPECT_EQ(result.MaximumNodesPerOperation(), 2U);
	EXPECT_EQ(result.MaximumOperationsPerNode(), 1U);
	EXPECT_EQ(result.ExecutableNodeCount(firstOperation.GetValue()), 2U);
	EXPECT_EQ(result.ExecutableNodeCount(secondOperation.GetValue()), 1U);
	EXPECT_EQ(result.ExecutableNodeCount(7U), 0U);
}

TEST(ComputeGraph, SemanticLoweringAnalysisTracksFusionProvenance) {
	OaSemanticGraph semantic;
	OaSemanticValueDesc matrix;
	matrix.Shape = {4};
	auto valueA = semantic.AddValue(matrix);
	auto valueB = semantic.AddValue(matrix);
	auto intermediate = semantic.AddValue(matrix);
	auto output = semantic.AddValue(matrix);
	ASSERT_TRUE(valueA.IsOk());
	ASSERT_TRUE(valueB.IsOk());
	ASSERT_TRUE(intermediate.IsOk());
	ASSERT_TRUE(output.IsOk());
	const OaSemanticValueId firstInputs[] = {
		valueA.GetValue(), valueB.GetValue(),
	};
	const OaSemanticValueId firstOutputs[] = {intermediate.GetValue()};
	auto firstOperation = semantic.AddOperation(
		OaOperationRegistry::Add, firstInputs, firstOutputs);
	ASSERT_TRUE(firstOperation.IsOk());
	const OaSemanticValueId secondInputs[] = {
		intermediate.GetValue(), valueB.GetValue(),
	};
	const OaSemanticValueId secondOutputs[] = {output.GetValue()};
	auto secondOperation = semantic.AddOperation(
		OaOperationRegistry::Add, secondInputs, secondOutputs);
	ASSERT_TRUE(secondOperation.IsOk());

	const OaSemanticOperationId fusedProvenance[] = {
		firstOperation.GetValue(), secondOperation.GetValue(),
	};
	OaComputeDispatchDesc fused;
	fused.Kernel = "FusedAddChain";
	fused.Operation = "FusedAddChain";
	fused.SemanticOperations = fusedProvenance;
	OaComputeGraph executable;
	executable.Add(fused);

	auto analyzed = OaAnalyzeSemanticLowering(semantic, executable);
	ASSERT_TRUE(analyzed.IsOk()) << analyzed.GetStatus().GetMessage();
	const auto& result = analyzed.GetValue();
	EXPECT_EQ(result.OperationCount(), 2U);
	EXPECT_EQ(result.SchemaOwnedNodeCount(), 1U);
	EXPECT_EQ(result.CompatibilityNodeCount(), 0U);
	EXPECT_EQ(result.DirectOperationCount(), 0U);
	EXPECT_EQ(result.DecomposedOperationCount(), 0U);
	EXPECT_EQ(result.FusedOperationCount(), 2U);
	EXPECT_EQ(result.FusedNodeCount(), 1U);
	EXPECT_EQ(result.MaximumNodesPerOperation(), 1U);
	EXPECT_EQ(result.MaximumOperationsPerNode(), 2U);
	EXPECT_EQ(result.ExecutableNodeCount(firstOperation.GetValue()), 1U);
	EXPECT_EQ(result.ExecutableNodeCount(secondOperation.GetValue()), 1U);
}

TEST(ComputeGraph, MultiAddExecutesSchemaOwnedFusionAndRemainder) {
	auto* engine = OaEngine::GetGlobal();
	ASSERT_NE(engine, nullptr);
	auto& ctx = OaContext::GetDefault();
	ctx.Clear();
	constexpr OaU32 PairCount = 6U;
	constexpr OaU32 ElementCount = 64U;
	OaVec<OaMatrix> destinations;
	OaVec<OaMatrix> sources;
	for (OaU32 pair = 0U; pair < PairCount; ++pair) {
		destinations.PushBack(OaFnMatrix::Empty(
			{ElementCount}, OaScalarType::Float32,
			OaMemoryPlacement::HostUpload));
		sources.PushBack(OaFnMatrix::Empty(
			{ElementCount}, OaScalarType::Float32,
			OaMemoryPlacement::HostUpload));
		ASSERT_TRUE(destinations.Back().HasStorage());
		ASSERT_TRUE(sources.Back().HasStorage());
		for (OaU32 element = 0U; element < ElementCount; ++element) {
			destinations.Back().DataAs<OaF32>()[element] =
				static_cast<OaF32>(pair + 1U);
			sources.Back().DataAs<OaF32>()[element] =
				static_cast<OaF32>((pair + 1U) * 10U);
		}
		ASSERT_TRUE(engine->Allocator.FlushHostBuffer(
			destinations.Back().GetVkBuffer(), 0,
			destinations.Back().GetVkBuffer().Size));
		ASSERT_TRUE(engine->Allocator.FlushHostBuffer(
			sources.Back().GetVkBuffer(), 0,
			sources.Back().GetVkBuffer().Size));
	}
	ctx.Clear();

	OaFnMatrix::MultiAdd(
		destinations.Span(),
		OaSpan<const OaMatrix>(sources.Data(), sources.Size()));
	ASSERT_EQ(ctx.NodeCount(), 3U);
	ASSERT_NE(ctx.SemanticGraph(), nullptr);
	ASSERT_EQ(ctx.SemanticGraph()->OperationCount(), PairCount);
	const auto nodes = ctx.Graph()->Nodes();
	ASSERT_EQ(nodes[0].SemanticOperations.Size(), 4U);
	EXPECT_EQ(nodes[0].Operation, "MultiMatrixAdd");
	EXPECT_EQ(nodes[1].SemanticOperations.Size(), 1U);
	EXPECT_EQ(nodes[2].SemanticOperations.Size(), 1U);
	const auto analyzed = OaAnalyzeSemanticLowering(
		*ctx.SemanticGraph(), *ctx.Graph());
	ASSERT_TRUE(analyzed.IsOk()) << analyzed.GetStatus().GetMessage();
	EXPECT_EQ(analyzed.GetValue().FusedOperationCount(), 4U);
	EXPECT_EQ(analyzed.GetValue().FusedNodeCount(), 1U);
	EXPECT_EQ(analyzed.GetValue().DirectOperationCount(), 2U);
	EXPECT_EQ(analyzed.GetValue().MaximumOperationsPerNode(), 4U);

	ASSERT_TRUE(ctx.Execute().IsOk());
	ASSERT_TRUE(ctx.Sync().IsOk());
	for (OaU32 pair = 0U; pair < PairCount; ++pair) {
		ASSERT_TRUE(engine->Allocator.InvalidateHostBuffer(
			destinations[pair].GetVkBuffer(), 0,
			destinations[pair].GetVkBuffer().Size));
		for (OaU32 element = 0U; element < ElementCount; ++element) {
			EXPECT_FLOAT_EQ(destinations[pair].DataAs<const OaF32>()[element],
				static_cast<OaF32>((pair + 1U) * 11U));
		}
	}
	ctx.Clear();
}

TEST(ComputeGraph, BiasAddExecutesSchemaOwnedDecompositionAndBackward) {
	auto* engine = OaEngine::GetGlobal();
	ASSERT_NE(engine, nullptr);
	auto& ctx = OaContext::GetDefault();
	ctx.Clear();
	auto input = OaFnMatrix::Empty(
		{2, 4}, OaScalarType::Float32, OaMemoryPlacement::HostUpload);
	auto bias = OaFnMatrix::Empty(
		{4}, OaScalarType::Float32, OaMemoryPlacement::HostUpload);
	ASSERT_TRUE(input.HasStorage());
	ASSERT_TRUE(bias.HasStorage());
	for (OaU32 element = 0U; element < 8U; ++element) {
		input.DataAs<OaF32>()[element] = static_cast<OaF32>(element + 1U);
	}
	for (OaU32 element = 0U; element < 4U; ++element) {
		bias.DataAs<OaF32>()[element] =
			static_cast<OaF32>((element + 1U) * 10U);
	}
	ASSERT_TRUE(engine->Allocator.FlushHostBuffer(
		input.GetVkBuffer(), 0, input.GetVkBuffer().Size));
	ASSERT_TRUE(engine->Allocator.FlushHostBuffer(
		bias.GetVkBuffer(), 0, bias.GetVkBuffer().Size));
	input.SetRequiresGrad(true);
	bias.SetRequiresGrad(true);
	ctx.Clear();

	OaGradientTape tape;
	auto output = OaFnMatrix::BiasAdd(input, bias);
	ASSERT_EQ(ctx.NodeCount(), 2U);
	ASSERT_NE(ctx.SemanticGraph(), nullptr);
	ASSERT_EQ(ctx.SemanticGraph()->OperationCount(), 1U);
	const auto nodes = ctx.Graph()->Nodes();
	ASSERT_EQ(nodes.Size(), 2U);
	for (const auto& node : nodes) {
		EXPECT_EQ(node.Operation, OaOperationRegistry::BiasAdd.Name);
		EXPECT_EQ(node.OperationContractHash,
			OaOperationRegistry::BiasAdd.Hash);
		ASSERT_EQ(node.SemanticOperations.Size(), 1U);
		EXPECT_EQ(node.SemanticOperations[0], 0U);
	}
	EXPECT_EQ(nodes[0].Shader, "Copy");
	EXPECT_EQ(nodes[1].Shader, "BiasAdd");
	const auto analyzed = OaAnalyzeSemanticLowering(
		*ctx.SemanticGraph(), *ctx.Graph());
	ASSERT_TRUE(analyzed.IsOk()) << analyzed.GetStatus().GetMessage();
	EXPECT_EQ(analyzed.GetValue().DirectOperationCount(), 0U);
	EXPECT_EQ(analyzed.GetValue().DecomposedOperationCount(), 1U);
	EXPECT_EQ(analyzed.GetValue().FusedOperationCount(), 0U);
	EXPECT_EQ(analyzed.GetValue().MaximumNodesPerOperation(), 2U);
	EXPECT_EQ(analyzed.GetValue().ExecutableNodeCount(0U), 2U);

	auto loss = OaFnMatrix::Sum(output);
	tape.Backward(loss);
	ASSERT_TRUE(ctx.Execute().IsOk());
	ASSERT_TRUE(ctx.Sync().IsOk());
	OaF32 outputValues[8]{};
	ASSERT_TRUE(OaFnMatrix::CopyToHost(
		output, outputValues, sizeof(outputValues)).IsOk());
	for (OaU32 element = 0U; element < 8U; ++element) {
		EXPECT_FLOAT_EQ(outputValues[element],
			static_cast<OaF32>(element + 1U) +
			static_cast<OaF32>(((element % 4U) + 1U) * 10U));
	}
	OaF32 inputGrad[8]{};
	OaF32 biasGrad[4]{};
	ASSERT_TRUE(OaFnMatrix::CopyToHost(
		input.GradMatrix(), inputGrad, sizeof(inputGrad)).IsOk());
	ASSERT_TRUE(OaFnMatrix::CopyToHost(
		bias.GradMatrix(), biasGrad, sizeof(biasGrad)).IsOk());
	for (const auto gradient : inputGrad) EXPECT_FLOAT_EQ(gradient, 1.0F);
	for (const auto gradient : biasGrad) EXPECT_FLOAT_EQ(gradient, 2.0F);
	ctx.Clear();
}

TEST(ComputeGraph, ExecutionRejectsIncompleteOrMismatchedSemanticLowering) {
	auto& ctx = OaContext::GetDefault();
	OaContext::RecordingScope scope(ctx);
	ctx.Clear();
	auto a = OaFnMatrix::Empty({4});
	auto b = OaFnMatrix::Empty({4});
	auto out = OaFnMatrix::Empty({4});

	const auto semantic = ctx.RecordOperation(
		OaOperationRegistry::Add, {&a, &b}, {&out});
	ASSERT_TRUE(semantic.IsOk());
	auto status = ctx.Execute();
	EXPECT_EQ(status.GetCode(), OaStatusCode::FailedPrecondition);
	EXPECT_EQ(ctx.NodeCount(), 0U);
	EXPECT_EQ(ctx.SemanticGraph()->OperationCount(), 0U);

	const auto semanticForSubmit = ctx.RecordOperation(
		OaOperationRegistry::Add, {&a, &b}, {&out});
	ASSERT_TRUE(semanticForSubmit.IsOk());
	const auto incompleteSubmit = ctx.Submit();
	EXPECT_FALSE(incompleteSubmit.IsOk());
	EXPECT_EQ(incompleteSubmit.GetStatus().GetCode(),
		OaStatusCode::FailedPrecondition);
	EXPECT_EQ(ctx.NodeCount(), 0U);
	EXPECT_EQ(ctx.SemanticGraph()->OperationCount(), 0U);

	OaComputeDispatchDesc dangling;
	dangling.Kernel = "OaTestKernelThatDoesNotExist";
	dangling.Operation = OaOperationRegistry::Add.Name;
	dangling.OperationContractHash = OaOperationRegistry::Add.Hash;
	const OaSemanticOperationId danglingProvenance[] = {7U};
	dangling.SemanticOperations = danglingProvenance;
	ASSERT_TRUE(ctx.Record(dangling).IsOk());
	status = ctx.Execute();
	EXPECT_EQ(status.GetCode(), OaStatusCode::OutOfRange);

	const auto semanticAgain = ctx.RecordOperation(
		OaOperationRegistry::Add, {&a, &b}, {&out});
	ASSERT_TRUE(semanticAgain.IsOk());
	OaComputeDispatchDesc mismatched;
	mismatched.Kernel = "OaTestKernelThatDoesNotExist";
	mismatched.Operation = "WrongSemanticIdentity";
	mismatched.OperationContractHash = OaOperationRegistry::Add.Hash;
	const OaSemanticOperationId mismatchedProvenance[] = {
		semanticAgain.GetValue(),
	};
	mismatched.SemanticOperations = mismatchedProvenance;
	ASSERT_TRUE(ctx.Record(mismatched).IsOk());
	status = ctx.Execute();
	EXPECT_EQ(status.GetCode(), OaStatusCode::FailedPrecondition);
	EXPECT_EQ(ctx.NodeCount(), 0U);
	EXPECT_EQ(ctx.SemanticGraph()->OperationCount(), 0U);

	const OaSemanticOperationId duplicateProvenance[] = {0U, 0U};
	OaComputeDispatchDesc duplicate;
	duplicate.Kernel = "OaTestKernelThatDoesNotExist";
	duplicate.Operation = "DuplicateSemanticProvenance";
	duplicate.SemanticOperations = duplicateProvenance;
	status = ctx.Record(duplicate);
	EXPECT_EQ(status.GetCode(), OaStatusCode::AlreadyExists);
	EXPECT_EQ(ctx.NodeCount(), 0U);
}

TEST(ComputeGraph, SchemaContractsDriveValidationAndShapeInference) {
	auto& ctx = OaContext::GetDefault();
	ctx.Clear();
	const auto broadcastA = OaFnMatrix::Empty({2, 3, 4}, OaScalarType::Float32);
	const auto broadcastB = OaFnMatrix::Empty({1, 4}, OaScalarType::Float32);
	auto broadcast = OaInferBinaryOperationShape(
		OaOperationRegistry::Add, broadcastA, broadcastB);
	ASSERT_TRUE(broadcast.IsOk());
	EXPECT_EQ(broadcast.GetValue(), (OaMatrixShape{2, 3, 4}));

	const auto wrongDtype = OaFnMatrix::Empty({1, 4}, OaScalarType::UInt32);
	const auto dtypeStatus = OaValidateBinaryOperation(
		OaOperationRegistry::Add, broadcastA, wrongDtype);
	EXPECT_EQ(dtypeStatus.GetCode(), OaStatusCode::DtypeMismatch);

	const auto weight = OaFnMatrix::Empty({5, 4}, OaScalarType::Float32);
	auto matmul = OaInferBinaryOperationShape(
		OaOperationRegistry::MatMulNt, broadcastA, weight);
	ASSERT_TRUE(matmul.IsOk());
	EXPECT_EQ(matmul.GetValue(), (OaMatrixShape{2, 3, 5}));

	const auto wrongWeight = OaFnMatrix::Empty({5, 6}, OaScalarType::Float32);
	const auto shapeStatus = OaValidateBinaryOperation(
		OaOperationRegistry::MatMulNt, broadcastA, wrongWeight);
	EXPECT_EQ(shapeStatus.GetCode(), OaStatusCode::ShapeMismatch);
	ctx.Clear();
}

TEST(ComputeGraph, PilotFunctionsRecordSemanticContracts) {
	auto& ctx = OaContext::GetDefault();
	ctx.Clear();
	const auto a = OaFnMatrix::Empty({2, 3}, OaScalarType::Float32);
	const auto b = OaFnMatrix::Empty({2, 3}, OaScalarType::Float32);
	ctx.Clear();
	const auto sum = OaFnMatrix::Add(a, b);
	(void)sum;
	ASSERT_EQ(ctx.NodeCount(), 1U);
	auto nodes = ctx.Graph()->Nodes();
	EXPECT_EQ(nodes[0].Operation, OaOperationRegistry::Add.Name);
	ASSERT_EQ(nodes[0].SemanticOperations.Size(), 1U);
	EXPECT_EQ(nodes[0].SemanticOperations[0], 0U);
	EXPECT_EQ(nodes[0].OperationContractHash, OaOperationRegistry::Add.Hash);
	EXPECT_EQ(nodes[0].ProblemContractHash, 0U);
	ASSERT_NE(ctx.SemanticGraph(), nullptr);
	ASSERT_EQ(ctx.SemanticGraph()->OperationCount(), 1U);
	EXPECT_EQ(ctx.SemanticGraph()->Operations()[0].Name,
		OaOperationRegistry::Add.Name);
	EXPECT_EQ(ctx.SemanticGraph()->ValueCount(), 3U);

	ctx.Clear();
	constexpr OaF32 ScaleValue = 1.25F;
	const auto scaled = OaFnMatrix::Scale(a, ScaleValue);
	(void)scaled;
	ASSERT_EQ(ctx.NodeCount(), 1U);
	nodes = ctx.Graph()->Nodes();
	EXPECT_EQ(nodes[0].Operation, OaOperationRegistry::Scale.Name);
	ASSERT_EQ(nodes[0].SemanticOperations.Size(), 1U);
	EXPECT_EQ(nodes[0].OperationContractHash, OaOperationRegistry::Scale.Hash);
	ASSERT_EQ(ctx.SemanticGraph()->OperationCount(), 1U);
	const auto& scaleOperation = ctx.SemanticGraph()->Operations()[0];
	ASSERT_EQ(scaleOperation.Attributes.Size(), 1U);
	EXPECT_EQ(scaleOperation.Attributes[0].Name, "Scalar");
	EXPECT_EQ(scaleOperation.Attributes[0].Kind,
		OaOperationAttributeKind::Float);
	EXPECT_DOUBLE_EQ(scaleOperation.Attributes[0].Float, ScaleValue);
	const auto scaleReport = ctx.SemanticGraph()->DebugReportJson("scale");
	EXPECT_NE(scaleReport.StdStr().find(
		"\"attributes\": [{\"name\": \"Scalar\", \"kind\": \"float\", "
		"\"value\": 1.25}]"), std::string::npos);

	ctx.Clear();
	const auto weight = OaFnMatrix::Empty({4, 3}, OaScalarType::Float32);
	ctx.Clear();
	const auto product = OaFnMatrix::MatMulNt(
		a, weight, OaMatMulPrecision::Fp32);
	(void)product;
	ASSERT_EQ(ctx.NodeCount(), 1U);
	nodes = ctx.Graph()->Nodes();
	EXPECT_EQ(nodes[0].Operation, OaOperationRegistry::MatMulNt.Name);
	ASSERT_EQ(nodes[0].SemanticOperations.Size(), 1U);
	EXPECT_EQ(nodes[0].SemanticOperations[0], 0U);
	EXPECT_EQ(nodes[0].OperationContractHash, OaOperationRegistry::MatMulNt.Hash);
	EXPECT_NE(nodes[0].ProblemContractHash, 0U);
	ASSERT_EQ(ctx.SemanticGraph()->OperationCount(), 1U);
	EXPECT_EQ(ctx.SemanticGraph()->Operations()[0].Name,
		OaOperationRegistry::MatMulNt.Name);
	EXPECT_EQ(ctx.SemanticGraph()->Operations()[0].Lowering,
		OaOperationLowering::Gemm);
	EXPECT_EQ(ctx.SemanticGraph()->ValueCount(), 3U);
	ctx.Clear();
}

TEST(ComputeGraph, ElementwiseSchemaWaveRecordsEveryContract) {
	auto& ctx = OaContext::GetDefault();
	ctx.Clear();
	const auto a = OaFnMatrix::Empty({2, 3}, OaScalarType::Float32);
	const auto b = OaFnMatrix::Empty({2, 3}, OaScalarType::Float32);
	ctx.Clear();

	const auto sub = OaFnMatrix::Sub(a, b);
	const auto mul = OaFnMatrix::Mul(a, b);
	const auto div = OaFnMatrix::Div(a, b);
	const auto neg = OaFnMatrix::Neg(a);
	const auto abs = OaFnMatrix::Abs(a);
	const auto log = OaFnMatrix::Log(a);
	const auto sqrt = OaFnMatrix::Sqrt(a);
	const auto pow = OaFnMatrix::Pow(a, 2.0F);
	const auto addScalar = OaFnMatrix::AddScalar(a, 1.0F);
	const auto subScalar = OaFnMatrix::SubScalar(a, 1.0F);
	const auto divScalar = OaFnMatrix::DivScalar(a, 2.0F);
	const auto exp = OaFnMatrix::Exp(a);
	const auto sin = OaFnMatrix::Sin(a);
	const auto cos = OaFnMatrix::Cos(a);
	const auto reciprocal = OaFnMatrix::Reciprocal(a);
	const auto clampMax = OaFnMatrix::ClampMax(a, 1.0F);
	const auto clampMin = OaFnMatrix::ClampMin(a, -1.0F);
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

	const OaOperationContract* expected[] = {
		&OaOperationRegistry::Sub,
		&OaOperationRegistry::Mul,
		&OaOperationRegistry::Div,
		&OaOperationRegistry::Neg,
		&OaOperationRegistry::Abs,
		&OaOperationRegistry::Log,
		&OaOperationRegistry::Sqrt,
		&OaOperationRegistry::Pow,
		&OaOperationRegistry::AddScalar,
		&OaOperationRegistry::SubScalar,
		&OaOperationRegistry::DivScalar,
		&OaOperationRegistry::Exp,
		&OaOperationRegistry::Sin,
		&OaOperationRegistry::Cos,
		&OaOperationRegistry::Reciprocal,
		&OaOperationRegistry::ClampMax,
		&OaOperationRegistry::ClampMin,
	};
	constexpr OaU32 ExpectedCount = 17U;
	ASSERT_EQ(ctx.NodeCount(), ExpectedCount);
	ASSERT_NE(ctx.SemanticGraph(), nullptr);
	ASSERT_TRUE(ctx.SemanticGraph()->Validate().IsOk());
	ASSERT_EQ(ctx.SemanticGraph()->OperationCount(), ExpectedCount);
	const auto nodes = ctx.Graph()->Nodes();
	const auto operations = ctx.SemanticGraph()->Operations();
	OaU32 attributedOperations = 0U;
	for (OaU32 index = 0U; index < ExpectedCount; ++index) {
		EXPECT_EQ(operations[index].Name, expected[index]->Name);
		EXPECT_EQ(operations[index].ContractHash, expected[index]->Hash);
		EXPECT_EQ(operations[index].Attributes.Size(),
			expected[index]->AttributeCount);
		if (not operations[index].Attributes.Empty()) ++attributedOperations;
		EXPECT_EQ(nodes[index].Operation, expected[index]->Name);
		EXPECT_EQ(nodes[index].OperationContractHash, expected[index]->Hash);
		ASSERT_EQ(nodes[index].SemanticOperations.Size(), 1U);
		EXPECT_EQ(nodes[index].SemanticOperations[0], operations[index].Id);
	}
	EXPECT_EQ(attributedOperations, 6U);
	const auto analyzed = OaAnalyzeSemanticLowering(
		*ctx.SemanticGraph(), *ctx.Graph());
	ASSERT_TRUE(analyzed.IsOk()) << analyzed.GetStatus().GetMessage();
	EXPECT_EQ(analyzed.GetValue().DirectOperationCount(), ExpectedCount);
	EXPECT_EQ(analyzed.GetValue().DecomposedOperationCount(), 0U);
	EXPECT_EQ(analyzed.GetValue().FusedOperationCount(), 0U);
	ctx.Clear();
}

TEST(ComputeGraph, ReductionSchemaFamilyRecordsContractsAndDimensions) {
	auto& ctx = OaContext::GetDefault();
	ctx.Clear();
	const auto input = OaFnMatrix::Empty({2, 3}, OaScalarType::Float32);
	ctx.Clear();

	const auto fullSum = OaFnMatrix::Sum(input);
	const auto axisSum = OaFnMatrix::Sum(input, 1);
	const auto fullMax = OaFnMatrix::Max(input);
	(void)fullSum;
	(void)axisSum;
	(void)fullMax;

	ASSERT_NE(ctx.SemanticGraph(), nullptr);
	ASSERT_TRUE(ctx.SemanticGraph()->Validate().IsOk());
	const auto operations = ctx.SemanticGraph()->Operations();
	ASSERT_EQ(operations.Size(), 3U);
	EXPECT_EQ(operations[0].Name, OaOperationRegistry::Sum.Name);
	EXPECT_EQ(operations[0].ContractHash, OaOperationRegistry::Sum.Hash);
	ASSERT_EQ(operations[0].Attributes.Size(), 1U);
	EXPECT_EQ(operations[0].Attributes[0].Name, "Dim");
	EXPECT_EQ(operations[0].Attributes[0].Kind,
		OaOperationAttributeKind::SignedInteger);
	EXPECT_EQ(operations[0].Attributes[0].SignedInteger, -1);
	EXPECT_EQ(operations[1].Name, OaOperationRegistry::Sum.Name);
	EXPECT_EQ(operations[1].ContractHash, OaOperationRegistry::Sum.Hash);
	ASSERT_EQ(operations[1].Attributes.Size(), 1U);
	EXPECT_EQ(operations[1].Attributes[0].SignedInteger, 1);
	EXPECT_EQ(operations[2].Name, OaOperationRegistry::Max.Name);
	EXPECT_EQ(operations[2].ContractHash, OaOperationRegistry::Max.Hash);
	EXPECT_TRUE(operations[2].Attributes.Empty());

	OaU32 ownedNodeCount = 0U;
	for (const auto& node : ctx.Graph()->Nodes()) {
		if (node.SemanticOperations.Empty()) continue;
		ASSERT_EQ(node.SemanticOperations.Size(), 1U);
		const auto operationId = node.SemanticOperations[0];
		ASSERT_LT(operationId, operations.Size());
		EXPECT_EQ(node.Operation, operations[operationId].Name);
		EXPECT_EQ(node.OperationContractHash,
			operations[operationId].ContractHash);
		++ownedNodeCount;
	}
	EXPECT_EQ(ownedNodeCount, 3U);
	const auto analyzed = OaAnalyzeSemanticLowering(
		*ctx.SemanticGraph(), *ctx.Graph());
	ASSERT_TRUE(analyzed.IsOk()) << analyzed.GetStatus().GetMessage();
	EXPECT_EQ(analyzed.GetValue().DirectOperationCount(), 3U);
	EXPECT_EQ(analyzed.GetValue().DecomposedOperationCount(), 0U);
	EXPECT_EQ(analyzed.GetValue().FusedOperationCount(), 0U);
	ctx.Clear();
}

TEST(ComputeGraph, NormalizationSchemaSliceRecordsEpsilonAndProvenance) {
	auto& ctx = OaContext::GetDefault();
	ctx.Clear();
	const auto input = OaFnMatrix::Empty({2, 3}, OaScalarType::Float32);
	const auto weight = OaFnMatrix::Empty({3}, OaScalarType::Float32);
	const auto bias = OaFnMatrix::Empty({3}, OaScalarType::Float32);
	ctx.Clear();

	const auto layer = OaFnMatrix::LayerNorm(input, weight, bias, 0.125F);
	const auto rms = OaFnMatrix::RmsNorm(input, weight, 0.25F);
	(void)layer;
	(void)rms;

	ASSERT_NE(ctx.SemanticGraph(), nullptr);
	ASSERT_TRUE(ctx.SemanticGraph()->Validate().IsOk());
	const auto operations = ctx.SemanticGraph()->Operations();
	ASSERT_EQ(operations.Size(), 2U);
	EXPECT_EQ(operations[0].Name, OaOperationRegistry::LayerNorm.Name);
	EXPECT_EQ(operations[0].ContractHash,
		OaOperationRegistry::LayerNorm.Hash);
	ASSERT_EQ(operations[0].Attributes.Size(), 1U);
	EXPECT_EQ(operations[0].Attributes[0].Name, "Eps");
	EXPECT_EQ(operations[0].Attributes[0].Kind,
		OaOperationAttributeKind::Float);
	EXPECT_EQ(operations[0].Attributes[0].Float, 0.125);
	EXPECT_EQ(operations[1].Name, OaOperationRegistry::RmsNorm.Name);
	EXPECT_EQ(operations[1].ContractHash, OaOperationRegistry::RmsNorm.Hash);
	ASSERT_EQ(operations[1].Attributes.Size(), 1U);
	EXPECT_EQ(operations[1].Attributes[0].Name, "Eps");
	EXPECT_EQ(operations[1].Attributes[0].Kind,
		OaOperationAttributeKind::Float);
	EXPECT_EQ(operations[1].Attributes[0].Float, 0.25);

	OaU32 ownedNodeCount = 0U;
	for (const auto& node : ctx.Graph()->Nodes()) {
		if (node.SemanticOperations.Empty()) continue;
		ASSERT_EQ(node.SemanticOperations.Size(), 1U);
		const auto operationId = node.SemanticOperations[0];
		ASSERT_LT(operationId, operations.Size());
		EXPECT_EQ(node.Operation, operations[operationId].Name);
		EXPECT_EQ(node.OperationContractHash,
			operations[operationId].ContractHash);
		++ownedNodeCount;
	}
	EXPECT_EQ(ownedNodeCount, 2U);
	const auto analyzed = OaAnalyzeSemanticLowering(
		*ctx.SemanticGraph(), *ctx.Graph());
	ASSERT_TRUE(analyzed.IsOk()) << analyzed.GetStatus().GetMessage();
	EXPECT_EQ(analyzed.GetValue().DirectOperationCount(), 2U);
	EXPECT_EQ(analyzed.GetValue().DecomposedOperationCount(), 0U);
	EXPECT_EQ(analyzed.GetValue().FusedOperationCount(), 0U);
	ctx.Clear();
}

TEST(ComputeGraph, SoftmaxSchemaFamilyRecordsDimensionsAndProvenance) {
	auto& ctx = OaContext::GetDefault();
	ctx.Clear();
	const auto input = OaFnMatrix::Empty({2, 3, 4}, OaScalarType::Float32);
	ctx.Clear();

	const auto softmax = OaFnMatrix::Softmax(input, 1);
	const auto logSoftmax = OaFnMatrix::LogSoftmax(input, 0);
	(void)softmax;
	(void)logSoftmax;

	ASSERT_NE(ctx.SemanticGraph(), nullptr);
	ASSERT_TRUE(ctx.SemanticGraph()->Validate().IsOk());
	const auto operations = ctx.SemanticGraph()->Operations();
	ASSERT_EQ(operations.Size(), 2U);
	EXPECT_EQ(operations[0].Name, OaOperationRegistry::Softmax.Name);
	EXPECT_EQ(operations[0].ContractHash, OaOperationRegistry::Softmax.Hash);
	ASSERT_EQ(operations[0].Attributes.Size(), 1U);
	EXPECT_EQ(operations[0].Attributes[0].Name, "Dim");
	EXPECT_EQ(operations[0].Attributes[0].Kind,
		OaOperationAttributeKind::SignedInteger);
	EXPECT_EQ(operations[0].Attributes[0].SignedInteger, 1);
	EXPECT_EQ(operations[1].Name, OaOperationRegistry::LogSoftmax.Name);
	EXPECT_EQ(operations[1].ContractHash,
		OaOperationRegistry::LogSoftmax.Hash);
	ASSERT_EQ(operations[1].Attributes.Size(), 1U);
	EXPECT_EQ(operations[1].Attributes[0].Name, "Dim");
	EXPECT_EQ(operations[1].Attributes[0].Kind,
		OaOperationAttributeKind::SignedInteger);
	EXPECT_EQ(operations[1].Attributes[0].SignedInteger, 0);

	OaU32 ownedNodeCount = 0U;
	for (const auto& node : ctx.Graph()->Nodes()) {
		if (node.SemanticOperations.Empty()) continue;
		ASSERT_EQ(node.SemanticOperations.Size(), 1U);
		const auto operationId = node.SemanticOperations[0];
		ASSERT_LT(operationId, operations.Size());
		EXPECT_EQ(node.Operation, operations[operationId].Name);
		EXPECT_EQ(node.OperationContractHash,
			operations[operationId].ContractHash);
		++ownedNodeCount;
	}
	EXPECT_EQ(ownedNodeCount, 2U);
	const auto analyzed = OaAnalyzeSemanticLowering(
		*ctx.SemanticGraph(), *ctx.Graph());
	ASSERT_TRUE(analyzed.IsOk()) << analyzed.GetStatus().GetMessage();
	EXPECT_EQ(analyzed.GetValue().DirectOperationCount(), 2U);
	EXPECT_EQ(analyzed.GetValue().DecomposedOperationCount(), 0U);
	EXPECT_EQ(analyzed.GetValue().FusedOperationCount(), 0U);
	ctx.Clear();
}

TEST(ComputeGraph, MeanSchemaRecordsOneDecomposedAxisOperation) {
	auto& ctx = OaContext::GetDefault();
	ctx.Clear();
	const auto input = OaFnMatrix::Empty({2, 3, 4}, OaScalarType::Float32);
	ctx.Clear();

	const auto output = OaFnMatrix::Mean(input, 1);
	ASSERT_EQ(output.GetShape(), (OaMatrixShape{2, 1, 4}));
	ASSERT_NE(ctx.SemanticGraph(), nullptr);
	ASSERT_TRUE(ctx.SemanticGraph()->Validate().IsOk());
	const auto operations = ctx.SemanticGraph()->Operations();
	ASSERT_EQ(operations.Size(), 1U);
	EXPECT_EQ(operations[0].Name, OaOperationRegistry::Mean.Name);
	EXPECT_EQ(operations[0].ContractHash, OaOperationRegistry::Mean.Hash);
	ASSERT_EQ(operations[0].Attributes.Size(), 1U);
	EXPECT_EQ(operations[0].Attributes[0].Name, "Dim");
	EXPECT_EQ(operations[0].Attributes[0].Kind,
		OaOperationAttributeKind::SignedInteger);
	EXPECT_EQ(operations[0].Attributes[0].SignedInteger, 1);

	const auto& nodes = ctx.Graph()->Nodes();
	ASSERT_EQ(nodes.Size(), 2U);
	EXPECT_EQ(nodes[0].Shader, "SumDim");
	EXPECT_EQ(nodes[1].Shader, "Scale");
	for (const auto& node : nodes) {
		ASSERT_EQ(node.SemanticOperations.Size(), 1U);
		EXPECT_EQ(node.SemanticOperations[0], 0U);
		EXPECT_EQ(node.Operation, OaOperationRegistry::Mean.Name);
		EXPECT_EQ(node.OperationContractHash, OaOperationRegistry::Mean.Hash);
	}
	const auto analyzed = OaAnalyzeSemanticLowering(
		*ctx.SemanticGraph(), *ctx.Graph());
	ASSERT_TRUE(analyzed.IsOk()) << analyzed.GetStatus().GetMessage();
	EXPECT_EQ(analyzed.GetValue().DirectOperationCount(), 0U);
	EXPECT_EQ(analyzed.GetValue().DecomposedOperationCount(), 1U);
	EXPECT_EQ(analyzed.GetValue().MaximumNodesPerOperation(), 2U);
	ctx.Clear();
}

TEST(ComputeGraph, CrossEntropySchemaOwnsClassificationAndMeanDecomposition) {
	auto& ctx = OaContext::GetDefault();
	ctx.Clear();
	auto logits = OaFnMatrix::Empty({2, 3}, OaScalarType::Float32);
	const auto targets = OaFnMatrix::Empty({2}, OaScalarType::UInt32);
	logits.SetRequiresGrad(true);
	ctx.Clear();

	OaGradientTape tape;
	const auto loss = OaFnLoss::CrossEntropy(logits, targets);
	ASSERT_FALSE(loss.IsEmpty());
	EXPECT_EQ(loss.GetShape(), (OaMatrixShape{1}));
	EXPECT_EQ(loss.GetDtype(), OaScalarType::Float32);

	const auto* semantic = ctx.SemanticGraph();
	ASSERT_NE(semantic, nullptr);
	ASSERT_TRUE(semantic->Validate().IsOk());
	const auto operations = semantic->Operations();
	ASSERT_EQ(operations.Size(), 1U);
	EXPECT_EQ(operations[0].Name, OaOperationRegistry::CrossEntropy.Name);
	EXPECT_EQ(operations[0].ContractHash,
		OaOperationRegistry::CrossEntropy.Hash);
	EXPECT_TRUE(operations[0].Attributes.Empty());
	ASSERT_EQ(operations[0].Inputs.Size(), 2U);
	ASSERT_EQ(operations[0].Outputs.Size(), 1U);
	const auto* outputValue = semantic->FindValue(operations[0].Outputs[0]);
	ASSERT_NE(outputValue, nullptr);
	EXPECT_EQ(outputValue->Shape, (OaMatrixShape{1}));
	EXPECT_EQ(outputValue->Dtype, OaScalarType::Float32);

	const auto& nodes = ctx.Graph()->Nodes();
	ASSERT_EQ(nodes.Size(), 3U);
	EXPECT_EQ(nodes[0].Shader, "CrossEntropy");
	EXPECT_EQ(nodes[1].Shader, "Sum");
	EXPECT_EQ(nodes[2].Shader, "Scale");
	for (const auto& node : nodes) {
		EXPECT_EQ(node.Operation, OaOperationRegistry::CrossEntropy.Name);
		EXPECT_EQ(node.OperationContractHash,
			OaOperationRegistry::CrossEntropy.Hash);
		ASSERT_EQ(node.SemanticOperations.Size(), 1U);
		EXPECT_EQ(node.SemanticOperations[0], operations[0].Id);
	}

	ASSERT_EQ(semantic->Autograd().Size(), 1U);
	EXPECT_EQ(semantic->Autograd()[0].ForwardOperation, operations[0].Id);
	ASSERT_TRUE(loss.GetGradFn());
	EXPECT_EQ(loss.GetGradFn()->ForwardSemanticOperation_, operations[0].Id);
	const auto analyzed = OaAnalyzeSemanticLowering(*semantic, *ctx.Graph());
	ASSERT_TRUE(analyzed.IsOk()) << analyzed.GetStatus().GetMessage();
	EXPECT_EQ(analyzed.GetValue().DirectOperationCount(), 0U);
	EXPECT_EQ(analyzed.GetValue().DecomposedOperationCount(), 1U);
	EXPECT_EQ(analyzed.GetValue().MaximumNodesPerOperation(), 3U);
	ctx.Clear();
}

TEST(ComputeGraph, CrossEntropySplitsRowsAcrossPortableDispatchDimensions) {
	auto& ctx = OaContext::GetDefault();
	ctx.Clear();
	constexpr OaI64 kRows = 65536;
	const auto logits = OaFnMatrix::Empty({kRows, 1}, OaScalarType::Float32);
	const auto targets = OaFnMatrix::Empty({kRows}, OaScalarType::UInt32);

	const auto loss = OaFnLoss::CrossEntropy(logits, targets);
	ASSERT_FALSE(loss.IsEmpty());
	const auto& forwardNodes = ctx.Graph()->Nodes();
	ASSERT_EQ(forwardNodes.Size(), 3U);
	EXPECT_EQ(forwardNodes[0].Shader, "CrossEntropy");
	EXPECT_EQ(forwardNodes[0].GroupsX, 65535U);
	EXPECT_EQ(forwardNodes[0].GroupsY, 2U);

	ctx.Clear();
	const auto gradient = OaFnLoss::CrossEntropyBwd(logits, targets);
	ASSERT_FALSE(gradient.IsEmpty());
	const auto& backwardNodes = ctx.Graph()->Nodes();
	ASSERT_EQ(backwardNodes.Size(), 1U);
	EXPECT_EQ(backwardNodes[0].Shader, "CrossEntropyBwd");
	EXPECT_EQ(backwardNodes[0].GroupsX, 65535U);
	EXPECT_EQ(backwardNodes[0].GroupsY, 2U);
	ctx.Clear();
}

TEST(ComputeGraph, SemanticPilotPreservesDataflowAcrossOperations) {
	auto& ctx = OaContext::GetDefault();
	ctx.Clear();
	const auto a = OaFnMatrix::Empty({2, 3}, OaScalarType::Float32);
	const auto b = OaFnMatrix::Empty({2, 3}, OaScalarType::Float32);
	const auto weight = OaFnMatrix::Empty({4, 3}, OaScalarType::Float32);
	ctx.Clear();
	const auto sum = OaFnMatrix::Add(a, b);
	const auto product = OaFnMatrix::MatMulNt(
		sum, weight, OaMatMulPrecision::Fp32);
	(void)product;

	const auto* semantic = ctx.SemanticGraph();
	ASSERT_NE(semantic, nullptr);
	ASSERT_TRUE(semantic->Validate().IsOk());
	ASSERT_EQ(semantic->OperationCount(), 2U);
	ASSERT_EQ(semantic->ValueCount(), 5U);
	const auto operations = semantic->Operations();
	ASSERT_EQ(operations[0].Outputs.Size(), 1U);
	ASSERT_EQ(operations[1].Inputs.Size(), 2U);
	EXPECT_EQ(operations[0].Outputs[0], operations[1].Inputs[0]);
	EXPECT_EQ(semantic->FindValue(operations[1].Inputs[0])->Producer, 0U);
	const auto executable = ctx.Graph()->Nodes();
	ASSERT_EQ(executable.Size(), 2U);
	ASSERT_EQ(executable[0].SemanticOperations.Size(), 1U);
	ASSERT_EQ(executable[1].SemanticOperations.Size(), 1U);
	EXPECT_EQ(executable[0].SemanticOperations[0], operations[0].Id);
	EXPECT_EQ(executable[1].SemanticOperations[0], operations[1].Id);
	ctx.Clear();
}

TEST(ComputeGraph, SemanticRecordingVersionsInPlaceStorage) {
	auto& ctx = OaContext::GetDefault();
	ctx.Clear();
	const auto value = OaFnMatrix::Empty({4}, OaScalarType::Float32);
	ctx.Clear();

	const OaOperationContract inPlace{
		.Name = "TestInPlace",
		.Hash = 0x91c4d14f09f6c7b1ULL,
		.InputKinds = static_cast<OaU32>(OaOperationValueKind::Matrix),
		.OutputKinds = static_cast<OaU32>(OaOperationValueKind::Matrix),
		.InputCount = 1,
		.OutputCount = 1,
		.Effects = OaOperationEffect::ReadInputs | OaOperationEffect::WriteOutputs,
		.MutatedInputMask = 0x01U,
		.OutputAliasInputs = 0xfffffff0U,
		.ControlFlow = OaOperationControlFlow::StraightLine,
	};
	ASSERT_TRUE(ctx.RecordOperation(inPlace, {&value}, {&value}).IsOk());
	ASSERT_TRUE(ctx.RecordOperation(inPlace, {&value}, {&value}).IsOk());

	const auto* semantic = ctx.SemanticGraph();
	ASSERT_NE(semantic, nullptr);
	ASSERT_TRUE(semantic->Validate().IsOk());
	ASSERT_EQ(semantic->ValueCount(), 3U);
	ASSERT_EQ(semantic->OperationCount(), 2U);
	const auto operations = semantic->Operations();
	EXPECT_EQ(operations[0].Inputs[0], 0U);
	EXPECT_EQ(operations[0].Outputs[0], 1U);
	EXPECT_EQ(operations[1].Inputs[0], 1U);
	EXPECT_EQ(operations[1].Outputs[0], 2U);
	ASSERT_EQ(operations[0].MutatedInputs.Size(), 1U);
	EXPECT_EQ(operations[0].MutatedInputs[0], 0U);
	ASSERT_EQ(operations[0].Aliases.Size(), 1U);
	EXPECT_EQ(operations[0].Aliases[0].Input, 0U);
	EXPECT_EQ(operations[0].Aliases[0].Output, 1U);
	EXPECT_EQ(operations[0].Accesses[0].Mode, OaSemanticAccessMode::ReadWrite);
	ctx.Clear();
}

TEST(ComputeGraph, FailedLoweringDiscardsBothGraphRepresentations) {
	auto& ctx = OaContext::GetDefault();
	ctx.Clear();
	const auto value = OaFnMatrix::Empty({4}, OaScalarType::Float32);
	ctx.Clear();

	const OaOperationContract contract{
		.Name = "TestMissingLowering",
		.Hash = 0x8c8bdd2db4c59ec3ULL,
		.InputKinds = static_cast<OaU32>(OaOperationValueKind::Matrix),
		.OutputKinds = static_cast<OaU32>(OaOperationValueKind::Matrix),
		.InputCount = 1,
		.OutputCount = 1,
		.Effects = OaOperationEffect::ReadInputs | OaOperationEffect::WriteOutputs,
	};
	ASSERT_TRUE(ctx.RecordOperation(contract, {&value}, {&value}).IsOk());

	OaComputeDispatchDesc missing;
	missing.Kernel = "OaTestKernelThatDoesNotExist";
	ASSERT_TRUE(ctx.Record(missing).IsOk());
	ASSERT_EQ(ctx.NodeCount(), 1U);
	ASSERT_EQ(ctx.SemanticGraph()->OperationCount(), 1U);

	const auto status = ctx.Execute();
	ASSERT_FALSE(status.IsOk());
	EXPECT_EQ(ctx.NodeCount(), 0U);
	EXPECT_EQ(ctx.SemanticGraph()->OperationCount(), 0U);
	EXPECT_EQ(ctx.SemanticGraph()->ValueCount(), 0U);
	ctx.Clear();
}

TEST(ComputeGraph, RecordingFailureRollsBackBothGraphsAndStaysSticky) {
	auto& ctx = OaContext::GetDefault();
	ctx.Clear();
	const auto a = OaFnMatrix::Empty({4}, OaScalarType::Float32);
	const auto b = OaFnMatrix::Empty({4}, OaScalarType::Float32);
	ctx.Clear();

	const auto prefix = OaFnMatrix::Add(a, b);
	ASSERT_FALSE(prefix.IsEmpty());
	ASSERT_EQ(ctx.NodeCount(), 1U);
	ASSERT_EQ(ctx.SemanticGraph()->OperationCount(), 1U);

	OaComputeDispatchDesc invalid;
	invalid.Kernel = "Scale";
	OaVkBuffer invalidBuffers[] = {a.GetVkBuffer()};
	invalid.Buffers = OaSpan<OaVkBuffer>(invalidBuffers, 1U);
	const auto failure = ctx.Record(invalid);
	ASSERT_FALSE(failure.IsOk());
	EXPECT_EQ(failure.GetCode(), OaStatusCode::InvalidArgument);
	EXPECT_EQ(ctx.NodeCount(), 0U);
	EXPECT_EQ(ctx.SemanticGraph()->OperationCount(), 0U);
	EXPECT_EQ(ctx.SemanticGraph()->ValueCount(), 0U);

	const auto rejected = ctx.RecordOperation(
		OaOperationRegistry::Add, {&a, &b}, {&prefix});
	ASSERT_FALSE(rejected.IsOk());
	EXPECT_EQ(rejected.GetStatus().GetCode(), failure.GetCode());
	EXPECT_EQ(rejected.GetStatus().GetMessage(), failure.GetMessage());
	EXPECT_EQ(ctx.NodeCount(), 0U);
	EXPECT_EQ(ctx.SemanticGraph()->OperationCount(), 0U);

	const auto surfaced = ctx.Execute();
	EXPECT_EQ(surfaced.GetCode(), failure.GetCode());
	EXPECT_EQ(surfaced.GetMessage(), failure.GetMessage());

	const auto recovered = OaFnMatrix::Add(a, b);
	ASSERT_FALSE(recovered.IsEmpty());
	ASSERT_EQ(ctx.NodeCount(), 1U);
	ASSERT_EQ(ctx.SemanticGraph()->OperationCount(), 1U);
	ASSERT_TRUE(ctx.Execute().IsOk());
	ASSERT_TRUE(ctx.Sync().IsOk());
	ctx.Clear();
}

TEST(ComputeGraph, MatrixDispatchPreflightFailureAbortsRecordingTransaction) {
	auto& ctx = OaContext::GetDefault();
	ctx.Clear();
	const auto input = OaFnMatrix::Empty({4}, OaScalarType::Float32);
	const auto output = OaFnMatrix::Empty({4}, OaScalarType::Float32);
	ctx.Clear();

	const auto prefix = OaFnMatrix::Scale(input, 3.0F);
	ASSERT_FALSE(prefix.IsEmpty());
	ASSERT_EQ(ctx.NodeCount(), 1U);
	ASSERT_EQ(ctx.SemanticGraph()->OperationCount(), 1U);

	OaMatrixDispatchDesc invalid;
	invalid.Dispatch.Kernel = "Scale";
	OaVkBuffer rawBuffers[] = {input.GetVkBuffer()};
	invalid.Dispatch.Buffers = OaSpan<OaVkBuffer>(rawBuffers, 1U);
	const OaMatrix* matrices[] = {&input, &output};
	invalid.Matrices = OaSpan<const OaMatrix* const>(matrices, 2U);
	const auto failure = ctx.Record(invalid);
	ASSERT_FALSE(failure.IsOk());
	EXPECT_EQ(failure.GetCode(), OaStatusCode::InvalidArgument);
	EXPECT_EQ(ctx.NodeCount(), 0U);
	EXPECT_EQ(ctx.SemanticGraph()->OperationCount(), 0U);
	EXPECT_EQ(ctx.SemanticGraph()->ValueCount(), 0U);

	const auto surfaced = ctx.Submit();
	ASSERT_FALSE(surfaced.IsOk());
	EXPECT_EQ(surfaced.GetStatus().GetCode(), failure.GetCode());
	EXPECT_EQ(surfaced.GetStatus().GetMessage(), failure.GetMessage());
	ctx.Clear();
}

TEST(ComputeGraph, SemanticAuthoringFailureRollsBackPartialValues) {
	auto& ctx = OaContext::GetDefault();
	ctx.Clear();
	const auto input = OaFnMatrix::Empty({4}, OaScalarType::Float32);
	const auto output = OaFnMatrix::Empty({4}, OaScalarType::Float32);
	const OaMatrix missing;
	ctx.Clear();

	const auto matrixKind = static_cast<OaU32>(OaOperationValueKind::Matrix);
	const OaOperationContract contract{
		.Name = "TestTwoOutputAuthoringFailure",
		.Hash = 0x78f48ced67af915bULL,
		.InputKinds = matrixKind,
		.OutputKinds = matrixKind | (matrixKind << 4U),
		.InputCount = 1,
		.OutputCount = 2,
		.Effects = OaOperationEffect::ReadInputs | OaOperationEffect::WriteOutputs,
	};
	const auto failure = ctx.RecordOperation(
		contract, {&input}, {&output, &missing});
	ASSERT_FALSE(failure.IsOk());
	EXPECT_EQ(failure.GetStatus().GetCode(), OaStatusCode::InvalidArgument);
	EXPECT_EQ(ctx.NodeCount(), 0U);
	EXPECT_EQ(ctx.SemanticGraph()->OperationCount(), 0U);
	EXPECT_EQ(ctx.SemanticGraph()->ValueCount(), 0U);

	const auto surfaced = ctx.Sync();
	EXPECT_EQ(surfaced.GetCode(), failure.GetStatus().GetCode());
	EXPECT_EQ(surfaced.GetMessage(), failure.GetStatus().GetMessage());
	ctx.Clear();
}

TEST(ComputeGraph, RecordingFailureCancelsUnsubmittedBatchPrefix) {
	auto& ctx = OaContext::GetDefault();
	ctx.Clear();
	const auto input = OaFnMatrix::Ones({8}, OaScalarType::Float32);
	const auto destination = OaFnMatrix::Zeros({8}, OaScalarType::Float32);
	ASSERT_TRUE(ctx.Execute().IsOk());
	ASSERT_TRUE(ctx.Sync().IsOk());
	ctx.Clear();

	struct Push { OaU32 Count; OaF32 Scale; } push{8U, 7.0F};
	OaBufferAccess access[] = {OaBufferAccess::Read, OaBufferAccess::Write};
	ctx.Add("Scale", {&input, &destination}, access, &push, sizeof(push), 1U);
	ASSERT_TRUE(ctx.ExecuteInAsyncBatch().IsOk());
	ASSERT_TRUE(ctx.IsAsyncBatchActive());

	OaComputeDispatchDesc invalid;
	invalid.Kernel = "Scale";
	OaVkBuffer invalidBuffers[] = {input.GetVkBuffer()};
	invalid.Buffers = OaSpan<OaVkBuffer>(invalidBuffers, 1U);
	const auto failure = ctx.Record(invalid);
	ASSERT_FALSE(failure.IsOk());
	EXPECT_FALSE(ctx.IsAsyncBatchActive());
	EXPECT_EQ(ctx.NodeCount(), 0U);
	EXPECT_EQ(ctx.SemanticGraph()->OperationCount(), 0U);

	const auto submitted = ctx.SubmitBatch();
	ASSERT_FALSE(submitted.IsOk());
	EXPECT_EQ(submitted.GetStatus().GetCode(), failure.GetCode());
	EXPECT_EQ(submitted.GetStatus().GetMessage(), failure.GetMessage());
	ASSERT_TRUE(ctx.Sync().IsOk());

	OaF32 values[8]{};
	ASSERT_TRUE(OaFnMatrix::CopyToHost(
		destination, values, sizeof(values)).IsOk());
	for (const OaF32 value : values) EXPECT_FLOAT_EQ(value, 0.0F);
	ctx.Clear();
}

TEST(ComputeGraph, ExplicitRecordingSubmitAndWait) {
	auto& ctx = OaContext::GetDefault();
	ctx.Clear();
	const OaF32 aValues[6]{1.0F, 1.0F, 1.0F, 1.0F, 1.0F, 1.0F};
	const OaF32 bValues[6]{2.0F, 2.0F, 2.0F, 2.0F, 2.0F, 2.0F};
	const auto a = OaFnMatrix::FromBytes(
		OaSpan<const OaU8>(reinterpret_cast<const OaU8*>(aValues), sizeof(aValues)),
		{2, 3}, OaScalarType::Float32);
	const auto b = OaFnMatrix::FromBytes(
		OaSpan<const OaU8>(reinterpret_cast<const OaU8*>(bValues), sizeof(bValues)),
		{2, 3}, OaScalarType::Float32);
	OaMatrix sum;
	{
		OaContext::RecordingScope recording(ctx);
		sum = OaFnMatrix::Add(a, b);
	}
	// Leaving a recording scope only restores recorder selection.
	ASSERT_EQ(ctx.NodeCount(), 1U);
	auto submitted = ctx.Submit();
	ASSERT_TRUE(submitted.IsOk()) << submitted.GetStatus().GetMessage();
	ASSERT_TRUE(submitted.GetValue().IsValid());
	const auto rejected = ctx.Submit();
	ASSERT_FALSE(rejected.IsOk());
	EXPECT_EQ(rejected.GetStatus().GetCode(), OaStatusCode::FailedPrecondition);
	const auto invalidWait = ctx.Wait(OaEvent{});
	EXPECT_FALSE(invalidWait.IsOk());
	EXPECT_EQ(invalidWait.GetCode(), OaStatusCode::InvalidArgument);
	const auto secondBatch = ctx.BeginAsyncBatch();
	EXPECT_FALSE(secondBatch.IsOk());
	EXPECT_EQ(secondBatch.GetCode(), OaStatusCode::FailedPrecondition);
	ASSERT_TRUE(ctx.Wait(submitted.GetValue()).IsOk());
	const auto staleWait = ctx.Wait(submitted.GetValue());
	EXPECT_FALSE(staleWait.IsOk());
	EXPECT_EQ(staleWait.GetCode(), OaStatusCode::FailedPrecondition);
	const auto emptySubmit = ctx.SubmitBatch();
	EXPECT_FALSE(emptySubmit.IsOk());
	EXPECT_EQ(emptySubmit.GetStatus().GetCode(), OaStatusCode::FailedPrecondition);

	OaF32 host[6]{};
	ASSERT_TRUE(OaFnMatrix::CopyToHost(sum, host, sizeof(host)).IsOk());
	for (const auto value : host) EXPECT_FLOAT_EQ(value, 3.0F);
	ctx.Clear();
}

TEST(ComputeGraph, ContextDestructionDiscardsUnsubmittedRecording) {
	auto* engine = OaEngine::GetGlobal();
	ASSERT_NE(engine, nullptr);
	auto* previous = OaContext::GetDefaultPtr();
	ASSERT_NE(previous, nullptr);

	auto target = OaFnMatrix::Ones({4}, OaScalarType::Float32);
	EXPECT_FLOAT_EQ(target.At(0), 1.0F);

	auto* temporary = OaContext::Create(engine);
	{
		OaContext::RecordingScope recording(*temporary);
		target.Zero();
		ASSERT_GT(temporary->NodeCount(), 0U);
		ASSERT_TRUE(temporary->ExecuteInAsyncBatch().IsOk());
		ASSERT_TRUE(temporary->IsAsyncBatchActive());
	}
	delete temporary;

	// Destruction must not turn the abandoned recording into GPU work.
	EXPECT_FLOAT_EQ(target.At(0), 1.0F);
}

TEST(ComputeGraph, ContextBatchesAreIsolatedAndRejectForeignEvents) {
	auto* engine = OaEngine::GetGlobal();
	ASSERT_NE(engine, nullptr);
	auto* previous = OaContext::GetDefaultPtr();
	ASSERT_NE(previous, nullptr);

	const OaF32 aValues[4]{1.0F, 2.0F, 3.0F, 4.0F};
	const OaF32 bValues[4]{3.0F, 4.0F, 5.0F, 6.0F};
	const auto a = OaFnMatrix::FromBytes(
		OaSpan<const OaU8>(reinterpret_cast<const OaU8*>(aValues), sizeof(aValues)),
		{4}, OaScalarType::Float32);
	const auto b = OaFnMatrix::FromBytes(
		OaSpan<const OaU8>(reinterpret_cast<const OaU8*>(bValues), sizeof(bValues)),
		{4}, OaScalarType::Float32);

	auto* first = OaContext::Create(engine);
	auto* second = OaContext::Create(engine);
	OaMatrix firstResult;
	OaMatrix secondResult;
	{
		OaContext::RecordingScope recording(*first);
		firstResult = OaFnMatrix::Add(a, a);
		ASSERT_TRUE(first->ExecuteInAsyncBatch().IsOk());
	}
	{
		OaContext::RecordingScope recording(*second);
		secondResult = OaFnMatrix::Add(b, b);
		ASSERT_TRUE(second->ExecuteInAsyncBatch().IsOk());
	}
	ASSERT_TRUE(first->IsAsyncBatchActive());
	ASSERT_TRUE(second->IsAsyncBatchActive());

	auto firstCompletion = first->SubmitBatch();
	auto secondCompletion = second->SubmitBatch();
	ASSERT_TRUE(firstCompletion.IsOk());
	ASSERT_TRUE(secondCompletion.IsOk());
	ASSERT_TRUE(firstCompletion.GetValue().IsValid());
	ASSERT_TRUE(secondCompletion.GetValue().IsValid());
	EXPECT_FALSE(firstCompletion.GetValue().IsSameCompletion(
		secondCompletion.GetValue()));

	const auto foreignFirst = first->Wait(secondCompletion.GetValue());
	EXPECT_FALSE(foreignFirst.IsOk());
	EXPECT_EQ(foreignFirst.GetCode(), OaStatusCode::InvalidArgument);
	const auto foreignSecond = second->Wait(firstCompletion.GetValue());
	EXPECT_FALSE(foreignSecond.IsOk());
	EXPECT_EQ(foreignSecond.GetCode(), OaStatusCode::InvalidArgument);
	ASSERT_TRUE(first->Wait(firstCompletion.GetValue()).IsOk());
	ASSERT_TRUE(second->Wait(secondCompletion.GetValue()).IsOk());

	OaF32 firstHost[4]{};
	OaF32 secondHost[4]{};
	ASSERT_TRUE(OaFnMatrix::CopyToHost(
		firstResult, firstHost, sizeof(firstHost)).IsOk());
	ASSERT_TRUE(OaFnMatrix::CopyToHost(
		secondResult, secondHost, sizeof(secondHost)).IsOk());
	for (OaU32 index = 0; index < 4U; ++index) {
		EXPECT_FLOAT_EQ(firstHost[index], aValues[index] * 2.0F);
		EXPECT_FLOAT_EQ(secondHost[index], bValues[index] * 2.0F);
	}

	delete first;
	delete second;
	OaContext::SetDefault(previous);
}

TEST(ComputeGraph, SubmittedContextDestructionRetiresWithoutLosingWork) {
	auto* engine = OaEngine::GetGlobal();
	ASSERT_NE(engine, nullptr);
	auto* previous = OaContext::GetDefaultPtr();
	ASSERT_NE(previous, nullptr);

	auto target = OaFnMatrix::Ones({256}, OaScalarType::Float32);
	ASSERT_FLOAT_EQ(target.At(0), 1.0F);
	auto* temporary = OaContext::Create(engine);
	OaEvent completion;
	{
		OaContext::RecordingScope recording(*temporary);
		target.Zero();
		ASSERT_TRUE(temporary->ExecuteInAsyncBatch().IsOk());
		auto submitted = temporary->SubmitBatch();
		ASSERT_TRUE(submitted.IsOk());
		completion = submitted.GetValue();
	}
	delete temporary;

	// The event and graph resources are owned by engine retirement after the
	// context is gone. Waiting the exact event still observes the submitted work.
	ASSERT_TRUE(completion.IsValid());
	ASSERT_TRUE(completion.Wait().IsOk());
	auto* collectionProbe = engine->AcquireStream();
	ASSERT_NE(collectionProbe, nullptr);
	engine->ReleaseStream(collectionProbe);
	EXPECT_FLOAT_EQ(target.At(0), 0.0F);
	OaContext::SetDefault(previous);
}

TEST(ComputeGraph, DebugReportIsDeterministicAndHandleFree) {
	OaVkBuffer a;
	a.Buffer = reinterpret_cast<void*>(0x11110000);
	a.Size = 256;
	OaVkBuffer b;
	b.Buffer = reinterpret_cast<void*>(0x22220000);
	b.Size = 512;
	OaVkBuffer buffers[] = {a, b};
	OaBufferAccess access[] = {OaBufferAccess::Read, OaBufferAccess::Write};
	const OaSemanticOperationId semanticOperations[] = {5U};

	OaComputeDispatchDesc desc;
	desc.Operation = "MatMulNt";
	desc.SemanticOperations = semanticOperations;
	desc.ImplementationId = 0x42U;
	desc.OperationContractHash = 0x99U;
	desc.ProblemContractHash = 0x100U;
	desc.KernelContentHash = 0x123U;
	desc.Kernel = "GemmTiledFp32_64x64x16";
	desc.Buffers = buffers;
	desc.Access = access;
	desc.GroupsX = 4;
	desc.GroupsY = 8;

	OaComputeGraph graph;
	graph.Add(desc);
	OaVkBuffer secondBuffers[] = {b, a};
	OaBufferAccess secondAccess[] = {OaBufferAccess::Read, OaBufferAccess::Write};
	OaComputeDispatchDesc secondDesc;
	secondDesc.Operation = "Silu";
	secondDesc.Kernel = "Silu";
	secondDesc.Buffers = secondBuffers;
	secondDesc.Access = secondAccess;
	graph.Add(secondDesc);
	const OaString first = graph.DebugReportJson("unit-matmul");
	const OaString second = graph.DebugReportJson("unit-matmul");
	EXPECT_EQ(first, second);
	const auto text = first.StdStr();
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

TEST(ComputeGraph, ContextRecordRejectsMalformedDescriptor) {
	auto& ctx = OaContext::GetDefault();
	ctx.Clear();

	OaVkBuffer buffer;
	OaComputeDispatchDesc desc;
	desc.Kernel = "MalformedDescriptorTest";
	desc.Buffers = OaSpan<OaVkBuffer>(&buffer, 1);
	// No access annotation for one buffer: this must fail before graph append.

	const auto status = ctx.Record(desc);
	EXPECT_FALSE(status.IsOk());
	EXPECT_EQ(ctx.NodeCount(), 0U);
	ctx.Clear();
}

TEST(ComputeGraph, ContextRecordRejectsMalformedIndirectDescriptor) {
	auto& ctx = OaContext::GetDefault();
	ctx.Clear();

	OaComputeDispatchDesc desc;
	desc.Kernel = "MalformedIndirectDescriptorTest";
	desc.Indirect = true;
	desc.IndirectBuffer.Buffer = reinterpret_cast<void*>(0x1000);
	desc.IndirectBuffer.Size = 3 * sizeof(OaU32);
	desc.IndirectOffset = 2;

	auto status = ctx.Record(desc);
	EXPECT_FALSE(status.IsOk());
	EXPECT_EQ(ctx.NodeCount(), 0U);

	desc.IndirectOffset = 4;
	status = ctx.Record(desc);
	EXPECT_FALSE(status.IsOk());
	EXPECT_EQ(ctx.NodeCount(), 0U);

	desc.Indirect = false;
	desc.IndirectOffset = 0;
	status = ctx.Record(desc);
	EXPECT_FALSE(status.IsOk());
	EXPECT_EQ(ctx.NodeCount(), 0U);
	ctx.Clear();
}

TEST(ComputeGraph, ContextMatrixRecordRetainsIndirectArguments) {
	auto& ctx = OaContext::GetDefault();
	ctx.Clear();

	OaMatrix input;
	input.VkBuf_ = OaSharedPtr<OaVkBuffer>(new OaVkBuffer());
	input.VkBuf_->Buffer = reinterpret_cast<void*>(0x3000);
	input.VkBuf_->Size = 64;
	input.VkBuf_->BindlessIndex = 13;
	OaMatrix args;
	args.VkBuf_ = OaSharedPtr<OaVkBuffer>(new OaVkBuffer());
	args.VkBuf_->Buffer = reinterpret_cast<void*>(0x4000);
	args.VkBuf_->Size = 3 * sizeof(OaU32);
	args.VkBuf_->BindlessIndex = 17;

	const OaMatrix* matrices[] = {&input, &args};
	OaBufferAccess access[] = {OaBufferAccess::Read, OaBufferAccess::Read};
	struct { OaU32 N; OaF32 Scale; } push{64, 1.0F};
	OaMatrixDispatchDesc desc;
	desc.Dispatch.Kernel = "Scale";
	desc.Dispatch.Access = access;
	desc.Dispatch.PushData = &push;
	desc.Dispatch.PushSize = sizeof(push);
	desc.Matrices = matrices;
	desc.IndirectArgs = &args;
	ASSERT_TRUE(ctx.Record(desc).IsOk());

	ASSERT_EQ(ctx.NodeCount(), 1U);
	const auto nodes = ctx.Graph()->Nodes();
	ASSERT_EQ(nodes.Size(), 1U);
	EXPECT_TRUE(nodes[0].Indirect);
	EXPECT_EQ(nodes[0].IndirectBuffer.Buffer, args.VkBuf_->Buffer);
	ASSERT_EQ(nodes[0].BufferOwners.Size(), 2U);
	EXPECT_TRUE(static_cast<bool>(nodes[0].BufferOwners[1]));
	ctx.Clear();

	const OaMatrix* missingOwner[] = {&input};
	desc.Matrices = missingOwner;
	access[0] = OaBufferAccess::Read;
	desc.Dispatch.Access = OaSpan<OaBufferAccess>(access, 1);
	EXPECT_FALSE(ctx.Record(desc).IsOk());
	EXPECT_EQ(ctx.NodeCount(), 0U);
}

TEST(ComputeGraph, ContextMatrixAddCapturesOwnershipAndDtype) {
	auto& ctx = OaContext::GetDefault();
	ctx.Clear();

	OaMatrix input;
	input.VkBuf_ = OaSharedPtr<OaVkBuffer>(new OaVkBuffer());
	input.VkBuf_->Buffer = reinterpret_cast<void*>(0x3000);
	input.VkBuf_->BindlessIndex = 13;
	input.Dtype_ = OaScalarType::BFloat16;
	OaMatrix output;
	output.VkBuf_ = OaSharedPtr<OaVkBuffer>(new OaVkBuffer());
	output.VkBuf_->Buffer = reinterpret_cast<void*>(0x4000);
	output.VkBuf_->BindlessIndex = 17;

	OaBufferAccess access[] = {OaBufferAccess::Read, OaBufferAccess::Write};
	struct Push { OaU32 Count; OaF32 Scale; } push{32, 1.0F};
	ctx.Add("Scale", {&input, &output}, access, &push, sizeof(push), 1);

	ASSERT_EQ(ctx.NodeCount(), 1U);
	const auto nodes = ctx.Graph()->Nodes();
	ASSERT_EQ(nodes.Size(), 1U);
	EXPECT_EQ(nodes[0].Dtype, 1U);
	ASSERT_EQ(nodes[0].BufferOwners.Size(), 2U);
	EXPECT_TRUE(static_cast<bool>(nodes[0].BufferOwners[0]));
	EXPECT_TRUE(static_cast<bool>(nodes[0].BufferOwners[1]));
	EXPECT_EQ(nodes[0].Buffers[0].BindlessIndex, 13U);
	EXPECT_EQ(nodes[0].Buffers[1].BindlessIndex, 17U);
	ctx.Clear();
}

TEST(ComputeGraph, SystemInfo) {
	fprintf(stderr, "\n");
	fprintf(stderr, "  ╔═══════════════════════════════════════════════════════════════╗\n");
	fprintf(stderr, "  ║       OaComputeGraph TEST SUITE — Graph & Replay Paths      ║\n");
	fprintf(stderr, "  ╚═══════════════════════════════════════════════════════════════╝\n");
	
	auto* rt = OaEngine::GetGlobal();
	if (rt) {
		fprintf(stderr, "\n  GPU: %s (%s)\n",
			rt->Device.Info.Hardware.DeviceName.c_str(), 
			rt->Device.Info.Hardware.VendorName.c_str());
		fprintf(stderr, "  Type: %s, VRAM: %llu MB\n",
			rt->Device.Info.Hardware.DeviceType == OaDeviceType::VkDiscrete ? "Discrete" :
			rt->Device.Info.Hardware.DeviceType == OaDeviceType::VkIntegrated ? "Integrated" :
			rt->Device.Info.Hardware.DeviceType == OaDeviceType::VkCpu ? "Software" : "Unknown",
			static_cast<unsigned long long>(rt->Device.Info.Hardware.VramBytes / (1024 * 1024)));
	}
	fprintf(stderr, "\n");
}

TEST(ComputeGraph, DeviceAdmissionCanaryUsesIndependentKnownAnswers) {
	if (not OaVkTestEngineOk()) GTEST_SKIP();
	auto& engine = *OaEngine::GetGlobal();
	engine.GetContext().Clear();

	OaDeviceCanaryReport report;
	const auto status = OaDeviceCanary::Run(engine, report);
	ASSERT_TRUE(status.IsOk()) << status.ToString().CStr();
	EXPECT_TRUE(report.Passed());
	ASSERT_EQ(report.Checks.Size(), 5U);
	for (const auto& check : report.Checks) {
		EXPECT_TRUE(check.Passed) << check.Name.CStr();
		EXPECT_NE(check.ExpectedHash, 0U);
		EXPECT_NE(check.ActualHash, 0U);
	}
	const auto first = report.DebugReportJson().StdStr();
	const auto second = report.DebugReportJson().StdStr();
	EXPECT_EQ(first, second);
	EXPECT_NE(first.find("\"schema\": \"oa.device_canary.v1\""),
		std::string::npos);
	EXPECT_NE(first.find("\"passed\": true"), std::string::npos);
	EXPECT_NE(first.find("\"name\": \"fp32_matmul_irregular\""),
		std::string::npos);
}

TEST(ComputeGraph, BasicAddAndExecute) {
	auto* rt = OaEngine::GetGlobal();
	ASSERT_NE(rt, nullptr);
	
	auto srcRes = rt->Allocator.AllocHostVisible(256 * sizeof(OaF32));
	ASSERT_TRUE(srcRes.IsOk());
	auto src = srcRes.GetValue();
	rt->RegisterBuffer(src);
	
	auto dstRes = rt->Allocator.AllocHostVisible(256 * sizeof(OaF32));
	ASSERT_TRUE(dstRes.IsOk());
	auto dst = dstRes.GetValue();
	rt->RegisterBuffer(dst);
	
	OaF32* data = static_cast<OaF32*>(src.MappedPtr);
	for (OaI32 i = 0; i < 256; ++i) data[i] = static_cast<OaF32>(i);
	
	struct { OaU32 N; OaF32 Scale; } push{256, 3.0f};
	OaVkBuffer bufs[] = {src, dst};
	OaBufferAccess access[] = {OaBufferAccess::Read, OaBufferAccess::Write};
	
	OaComputeGraph graph;
	graph.Add("Scale", bufs, access, &push, sizeof(push), (256 + 255) / 256);
	EXPECT_EQ(graph.NodeCount(), 1u);
	
	auto status = graph.Execute(*rt);
	if (status.IsOk()) {
		OaF32* out = static_cast<OaF32*>(dst.MappedPtr);
		for (OaI32 i = 0; i < 256; ++i) {
			EXPECT_NEAR(out[i], static_cast<OaF32>(i) * 3.0f, 1e-3f);
		}
	}
	
	rt->DeregisterBuffer(src);
	rt->Allocator.Free(src);
	rt->DeregisterBuffer(dst);
	rt->Allocator.Free(dst);
}

TEST(ComputeGraph, CompileAndReplay) {
	auto* rt = OaEngine::GetGlobal();
	ASSERT_NE(rt, nullptr);
	
	auto srcRes = rt->Allocator.AllocHostVisible(256 * sizeof(OaF32));
	ASSERT_TRUE(srcRes.IsOk());
	auto src = srcRes.GetValue();
	rt->RegisterBuffer(src);
	
	auto midRes = rt->Allocator.AllocHostVisible(256 * sizeof(OaF32));
	ASSERT_TRUE(midRes.IsOk());
	auto mid = midRes.GetValue();
	rt->RegisterBuffer(mid);
	
	auto dstRes = rt->Allocator.AllocHostVisible(256 * sizeof(OaF32));
	ASSERT_TRUE(dstRes.IsOk());
	auto dst = dstRes.GetValue();
	rt->RegisterBuffer(dst);
	
	OaF32* data = static_cast<OaF32*>(src.MappedPtr);
	for (OaI32 i = 0; i < 256; ++i) data[i] = static_cast<OaF32>(i + 1);
	
	struct { OaU32 N; OaF32 Scale; } push1{256, 2.0f};
	struct { OaU32 N; OaF32 Scale; } push2{256, 0.5f};
	
	OaVkBuffer bufs1[] = {src, mid};
	OaBufferAccess access1[] = {OaBufferAccess::Read, OaBufferAccess::Write};
	OaVkBuffer bufs2[] = {mid, dst};
	OaBufferAccess access2[] = {OaBufferAccess::Read, OaBufferAccess::Write};
	
	OaComputeGraph graph;
	graph.Add("Scale", bufs1, access1, &push1, sizeof(push1), 1);
	graph.Add("Scale", bufs2, access2, &push2, sizeof(push2), 1);
	EXPECT_EQ(graph.NodeCount(), 2u);
	EXPECT_FALSE(graph.IsCompiled());
	
	auto compileStatus = graph.Compile(*rt);
	if (!compileStatus.IsOk()) {
		graph.Destroy(rt->Device);
		rt->DeregisterBuffer(src);
		rt->Allocator.Free(src);
		rt->DeregisterBuffer(mid);
		rt->Allocator.Free(mid);
		rt->DeregisterBuffer(dst);
		rt->Allocator.Free(dst);
		GTEST_SKIP() << "scale shader not loaded";
	}
	EXPECT_TRUE(graph.IsCompiled());
	
	// Replay 5 times
	for (OaI32 rep = 0; rep < 5; ++rep) {
		for (OaI32 i = 0; i < 256; ++i) data[i] = static_cast<OaF32>(i + 1);
		
		auto status = graph.Replay(*rt);
		ASSERT_TRUE(status.IsOk()) << "replay " << rep << " failed";
		// Replay() is non-blocking (submits PrimaryCb_ with a timeline semaphore
		// and returns). Results are undefined until the replay completes — the
		// documented contract is to WaitForPendingReplay()/Sync() before reading.
		ASSERT_TRUE(graph.WaitForPendingReplay(rt->Device).IsOk())
			<< "wait for replay " << rep << " failed";

		OaF32* out = static_cast<OaF32*>(dst.MappedPtr);
		for (OaI32 i = 0; i < 256; ++i) {
			EXPECT_NEAR(out[i], static_cast<OaF32>(i + 1) * 1.0f, 1e-3f)
				<< "mismatch at i=" << i << " rep=" << rep;
		}
	}
	
	graph.Destroy(rt->Device);
	rt->DeregisterBuffer(src);
	rt->Allocator.Free(src);
	rt->DeregisterBuffer(mid);
	rt->Allocator.Free(mid);
	rt->DeregisterBuffer(dst);
	rt->Allocator.Free(dst);
}

TEST(ComputeGraph, TimedReplayRequiresWaitBeforeQueryReuse) {
	auto* rt = OaEngine::GetGlobal();
	ASSERT_NE(rt, nullptr);

	auto srcRes = rt->AllocBuffer(64 * sizeof(OaF32));
	auto dstRes = rt->AllocBuffer(64 * sizeof(OaF32));
	ASSERT_TRUE(srcRes.IsOk() && dstRes.IsOk());
	auto src = std::move(*srcRes);
	auto dst = std::move(*dstRes);
	for (OaU32 i = 0; i < 64; ++i) {
		static_cast<OaF32*>(src.MappedPtr)[i] = static_cast<OaF32>(i);
	}
	ASSERT_TRUE(rt->Allocator.FlushHostBuffer(src, 0, src.Size));

	struct { OaU32 N; OaF32 Scale; } push{64, 2.0F};
	OaVkBuffer bufs[] = {src, dst};
	OaBufferAccess access[] = {OaBufferAccess::Read, OaBufferAccess::Write};
	OaComputeGraph graph;
	graph.Add("Scale", bufs, access, &push, sizeof(push), 1);
	graph.SetReplayTimingEnabled(true);
	ASSERT_TRUE(graph.Compile(*rt).IsOk());
	ASSERT_TRUE(graph.Replay(*rt).IsOk());
	EXPECT_FALSE(graph.Replay(*rt).IsOk());
	ASSERT_TRUE(graph.WaitForPendingReplay(rt->Device).IsOk());
	EXPECT_GT(graph.LastReplayGpuMs(), 0.0);
	ASSERT_TRUE(graph.Replay(*rt).IsOk());
	ASSERT_TRUE(graph.WaitForPendingReplay(rt->Device).IsOk());

	graph.Destroy(rt->Device);
	rt->FreeBuffer(src);
	rt->FreeBuffer(dst);
}

TEST(ComputeGraph, HazardPlannerTracksReadBeforeWrite) {
	auto* rt = OaEngine::GetGlobal();
	ASSERT_NE(rt, nullptr);

	auto xRes = rt->Allocator.AllocHostVisible(256 * sizeof(OaF32));
	auto aRes = rt->Allocator.AllocHostVisible(256 * sizeof(OaF32));
	auto bRes = rt->Allocator.AllocHostVisible(256 * sizeof(OaF32));
	ASSERT_TRUE(xRes.IsOk() && aRes.IsOk() && bRes.IsOk());
	auto x = xRes.GetValue();
	auto a = aRes.GetValue();
	auto b = bRes.GetValue();

	struct { OaU32 N; OaF32 Scale; } push{256, 1.0F};
	OaComputeGraph graph;
	{
		OaVkBuffer bufs[] = {x, a};
		OaBufferAccess access[] = {
			OaBufferAccess::Read, OaBufferAccess::Write};
		graph.Add("Scale", bufs, access, &push, sizeof(push), 1);
	}
	{
		// X was only read by the first node. A writer still needs an execution
		// dependency even though there is no preceding in-graph writer for X.
		OaVkBuffer bufs[] = {b, x};
		OaBufferAccess access[] = {
			OaBufferAccess::Read, OaBufferAccess::Write};
		graph.Add("Scale", bufs, access, &push, sizeof(push), 1);
	}

	ASSERT_TRUE(graph.Compile(*rt).IsOk());
	const auto stats = graph.GetStats();
	EXPECT_EQ(stats.WarBarrierCount, 1U);
	EXPECT_GE(stats.BarrierCount, 1U);

	graph.Destroy(rt->Device);

	OaComputeGraph indirectGraph;
	{
		OaVkBuffer bufs[] = {x, a};
		OaBufferAccess access[] = {
			OaBufferAccess::Read, OaBufferAccess::Write};
		indirectGraph.Add("Scale", bufs, access, &push, sizeof(push), 1);
	}
	{
		OaVkBuffer bufs[] = {x, b};
		OaBufferAccess access[] = {
			OaBufferAccess::Read, OaBufferAccess::Write};
		indirectGraph.AddIndirect(
			"Scale", bufs, access, &push, sizeof(push), a, 0);
	}
	ASSERT_TRUE(indirectGraph.Compile(*rt).IsOk());
	EXPECT_EQ(indirectGraph.GetStats().IndirectBarrierCount, 1U);
	indirectGraph.Destroy(rt->Device);

	rt->Allocator.Free(x);
	rt->Allocator.Free(a);
	rt->Allocator.Free(b);
}

TEST(ComputeGraph, IndirectArgumentHazardAndCacheIdentity) {
	auto* rt = OaEngine::GetGlobal();
	ASSERT_NE(rt, nullptr);

	constexpr OaU32 N = 64;
	auto srcRes = rt->AllocBuffer(N * sizeof(OaF32));
	auto dstRes = rt->AllocBuffer(N * sizeof(OaF32));
	auto argsRes = rt->AllocBuffer(6 * sizeof(OaU32));
	ASSERT_TRUE(srcRes.IsOk() && dstRes.IsOk() && argsRes.IsOk());
	auto src = std::move(*srcRes);
	auto dst = std::move(*dstRes);
	auto args = std::move(*argsRes);

	auto* srcData = static_cast<OaF32*>(src.MappedPtr);
	auto* dstData = static_cast<OaF32*>(dst.MappedPtr);
	auto* dispatchArgs = static_cast<OaU32*>(args.MappedPtr);
	for (OaU32 i = 0; i < N; ++i) {
		srcData[i] = static_cast<OaF32>(i + 1);
		dstData[i] = 0.0F;
	}
	dispatchArgs[0] = 1; dispatchArgs[1] = 1; dispatchArgs[2] = 1;
	dispatchArgs[3] = 0; dispatchArgs[4] = 1; dispatchArgs[5] = 1;
	ASSERT_TRUE(rt->Allocator.FlushHostBuffer(src, 0, src.Size));
	ASSERT_TRUE(rt->Allocator.FlushHostBuffer(dst, 0, dst.Size));
	ASSERT_TRUE(rt->Allocator.FlushHostBuffer(args, 0, args.Size));

	struct { OaU32 N; OaF32 Scale; } push{N, 2.0F};
	OaVkBuffer bufs[] = {src, dst};
	OaBufferAccess access[] = {
		OaBufferAccess::Read, OaBufferAccess::Write};

	OaComputeGraph graph;
	graph.AddIndirect("Scale", bufs, access, &push, sizeof(push), args, 0);
	ASSERT_TRUE(graph.Execute(*rt).IsOk());
	EXPECT_NEAR(dstData[0], 2.0F, 1e-3F);
	for (OaU32 i = 0; i < N; ++i) dstData[i] = 0.0F;
	ASSERT_TRUE(rt->Allocator.FlushHostBuffer(dst, 0, dst.Size));
	ASSERT_TRUE(graph.Compile(*rt).IsOk());
	ASSERT_TRUE(graph.Replay(*rt).IsOk());
	ASSERT_TRUE(graph.WaitForPendingReplay(rt->Device).IsOk());
	EXPECT_NEAR(dstData[0], 2.0F, 1e-3F);

	// The only topology change is the indirect offset. Compile must not reuse
	// the old command buffer: the second command has groupCountX=0 and skips.
	for (OaU32 i = 0; i < N; ++i) dstData[i] = 0.0F;
	ASSERT_TRUE(rt->Allocator.FlushHostBuffer(dst, 0, dst.Size));
	graph.ClearNodes();
	graph.AddIndirect("Scale", bufs, access, &push, sizeof(push), args,
		3 * sizeof(OaU32));
	ASSERT_TRUE(graph.Compile(*rt).IsOk());
	ASSERT_TRUE(graph.Replay(*rt).IsOk());
	ASSERT_TRUE(graph.WaitForPendingReplay(rt->Device).IsOk());
	EXPECT_NEAR(dstData[0], 0.0F, 1e-6F);

	const auto lifetimes = graph.ComputeLifetimes();
	EXPECT_EQ(lifetimes.Size(), 3U); // src, dst, and indirect argument buffer

	graph.Destroy(rt->Device);
	rt->FreeBuffer(src);
	rt->FreeBuffer(dst);
	rt->FreeBuffer(args);
}

TEST(ComputeGraph, EmptyGraph) {
	auto* rt = OaEngine::GetGlobal();
	ASSERT_NE(rt, nullptr);
	
	OaComputeGraph graph;
	EXPECT_EQ(graph.NodeCount(), 0u);
	EXPECT_TRUE(graph.Execute(*rt).IsOk());
	EXPECT_TRUE(graph.Compile(*rt).IsOk());
	EXPECT_TRUE(graph.IsCompiled());
	EXPECT_TRUE(graph.Replay(*rt).IsOk());
	
	auto lifetimes = graph.ComputeLifetimes();
	EXPECT_TRUE(lifetimes.Empty());
	
	auto groups = graph.ComputeAliasGroups();
	EXPECT_TRUE(groups.Empty());
	
	auto stats = graph.GetStats();
	EXPECT_EQ(stats.DispatchCount, 0u);
	EXPECT_EQ(stats.BarrierCount, 0u);
	
	graph.Destroy(rt->Device);
}

TEST(ComputeGraph, HostReadbackBarrierIsAnExplicitCompletionPolicy) {
	auto* rt = OaEngine::GetGlobal();
	ASSERT_NE(rt, nullptr);

	auto srcRes = rt->AllocBuffer(256 * sizeof(OaF32));
	auto dstRes = rt->AllocBuffer(256 * sizeof(OaF32));
	ASSERT_TRUE(srcRes.IsOk() && dstRes.IsOk());
	auto src = std::move(*srcRes);
	auto dst = std::move(*dstRes);

	struct { OaU32 N; OaF32 Scale; } push{256, 1.0F};
	OaVkBuffer buffers[] = {src, dst};
	OaBufferAccess access[] = {OaBufferAccess::Read, OaBufferAccess::Write};

	OaComputeGraph graph;
	graph.Add("Scale", buffers, access, &push, sizeof(push), 1);
	ASSERT_TRUE(graph.Compile(*rt).IsOk());
	EXPECT_EQ(graph.GetStats().HostBarrierCount, 1U);

	graph.SetHostReadbackRequired(false);
	EXPECT_FALSE(graph.IsCompiled());
	ASSERT_TRUE(graph.Compile(*rt).IsOk());
	EXPECT_EQ(graph.GetStats().HostBarrierCount, 0U);

	graph.Destroy(rt->Device);
	rt->FreeBuffer(src);
	rt->FreeBuffer(dst);
}

TEST(ComputeGraph, ContextBatchUsesExactBoundariesAndReusesStaticGraphs) {
	auto* rt = OaEngine::GetGlobal();
	ASSERT_NE(rt, nullptr);
	auto& ctx = OaContext::GetDefault();
	ctx.Clear();

	constexpr OaU32 N = 256;
	auto src = OaFnMatrix::Empty(OaMatrixShape{N}, OaScalarType::Float32);
	auto mid = OaFnMatrix::Empty(OaMatrixShape{N}, OaScalarType::Float32);
	auto dst = OaFnMatrix::Empty(OaMatrixShape{N}, OaScalarType::Float32);
	auto unrelatedSrc = OaFnMatrix::Empty(OaMatrixShape{N}, OaScalarType::Float32);
	auto unrelatedDst = OaFnMatrix::Empty(OaMatrixShape{N}, OaScalarType::Float32);
	ASSERT_TRUE(src.HasStorage() && mid.HasStorage() && dst.HasStorage());
	ASSERT_TRUE(unrelatedSrc.HasStorage() && unrelatedDst.HasStorage());

	for (OaU32 i = 0; i < N; ++i) {
		src.DataAs<OaF32>()[i] = static_cast<OaF32>(i + 1);
		unrelatedSrc.DataAs<OaF32>()[i] = static_cast<OaF32>(N - i);
		dst.DataAs<OaF32>()[i] = 0.0F;
	}
	ASSERT_TRUE(rt->Allocator.FlushHostBuffer(src.GetVkBuffer(), 0, src.ByteSize()));
	ASSERT_TRUE(rt->Allocator.FlushHostBuffer(
		unrelatedSrc.GetVkBuffer(), 0, unrelatedSrc.ByteSize()));
	ASSERT_TRUE(rt->Allocator.FlushHostBuffer(dst.GetVkBuffer(), 0, dst.ByteSize()));

	OaBufferAccess access[] = {OaBufferAccess::Read, OaBufferAccess::Write};
	auto recordScale = [&](const OaMatrix& InSrc, OaMatrix& OutDst, OaF32 InScale) {
		struct { OaU32 Count; OaF32 Scale; } push{N, InScale};
		ctx.Add("Scale", {&InSrc, &OutDst}, access, &push, sizeof(push), 1);
	};
	auto executeBatch = [&]() {
		recordScale(src, mid, 2.0F);
		EXPECT_TRUE(ctx.ExecuteInAsyncBatch().IsOk());
		recordScale(unrelatedSrc, unrelatedDst, 3.0F);
		EXPECT_TRUE(ctx.ExecuteInAsyncBatch().IsOk());
		recordScale(mid, dst, 4.0F);
		EXPECT_TRUE(ctx.ExecuteInAsyncBatch().IsOk());
		auto completion = ctx.SubmitBatch();
		EXPECT_TRUE(completion.IsOk());
		ASSERT_TRUE(completion.GetValue().IsValid());
		EXPECT_TRUE(ctx.Wait(completion.GetValue()).IsOk());
	};

	executeBatch();
	auto first = ctx.LastExecutionStats();
	EXPECT_EQ(first.GraphCount, 3U);
	EXPECT_EQ(first.BoundaryBarrierCount, 1U);
	EXPECT_EQ(first.HostBarrierCount, 1U);
	for (OaU32 i = 0; i < N; ++i) {
		EXPECT_NEAR(dst.DataAs<OaF32>()[i], static_cast<OaF32>(i + 1) * 8.0F, 1e-3F);
	}

	// The same stable buffers, topology and push constants must reuse all three
	// compiled secondary command buffers on the next step.
	executeBatch();
	const auto second = ctx.LastExecutionStats();
	EXPECT_EQ(second.GraphCount, 3U);
	EXPECT_EQ(second.CompileCacheHits, 3U);
	EXPECT_EQ(second.BoundaryBarrierCount, 1U);
	EXPECT_EQ(second.HostBarrierCount, 1U);

	ctx.Clear();
}

// =============================================================================
// GPU COMPILATION TESTS (Phase 2)
TEST(ComputeGraph, BarrierOverhead) {
	auto* rt = OaEngine::GetGlobal();
	ASSERT_NE(rt, nullptr);
	
	PrintHeader("BARRIER ANALYSIS — CPU PLANNING COST");
	fprintf(stderr, "  %-12s %12s  %12s  %12s\n",
		"Dispatches", "Barriers", "Barrier%", "CPU Cost");
	PrintBar();
	
	struct Config { OaI32 N; const char* Label; };
	Config configs[] = {
		{6,  "6 dispatches"},
		{12, "12 dispatches"},
		{25, "25 dispatches"},
		{50, "50 dispatches"},
	};
	
	for (auto& cfg : configs) {
		OaI32 N = cfg.N;
		OaVec<OaVkBuffer> bufs(N + 1);
		for (OaI32 i = 0; i <= N; ++i) {
			auto res = rt->Allocator.AllocHostVisible(256 * sizeof(OaF32));
			ASSERT_TRUE(res.IsOk());
			bufs[i] = res.GetValue();
		}
		
		OaComputeGraph graph;
		BuildChainGraph(graph, bufs, N);
		
		auto compileStatus = graph.Compile(*rt);
		if (!compileStatus.IsOk()) {
			for (auto& b : bufs) rt->Allocator.Free(b);
			GTEST_SKIP() << "scale shader not loaded";
		}
		
		auto stats = graph.GetStats();
		double barrierPct = N > 0 ? (100.0 * stats.BarrierCount / N) : 0.0;
		
		// Estimate CPU cost of barrier computation
		// Each barrier requires hashmap lookup + comparison (~50ns on modern CPU)
		double cpuCostUs = stats.BarrierCount * 0.05;  // 50ns per barrier
		
		fprintf(stderr, "  %-12s %10u    %10.0f%%   %10.1f µs\n",
			cfg.Label, stats.BarrierCount, barrierPct, cpuCostUs);
		
		graph.Destroy(rt->Device);
		for (auto& b : bufs) rt->Allocator.Free(b);
	}
	
	fprintf(stderr, "\n  Barriers are planned on the CPU and encoded once during graph compilation.\n");
}

// =============================================================================
// MEMORY ANALYSIS
// =============================================================================

TEST(ComputeGraph, MemoryAliasing) {
	auto* rt = OaEngine::GetGlobal();
	ASSERT_NE(rt, nullptr);
	
	PrintHeader("MEMORY ALIASING — Potential VRAM savings");
	fprintf(stderr, "  %-12s %12s  %12s  %12s  %8s\n",
		"Dispatches", "Total bufs", "Alias groups", "Savings", "Pct");
	PrintBar();
	
	struct Config { OaI32 N; const char* Label; };
	Config configs[] = {
		{6,  "6 dispatches"},
		{12, "12 dispatches"},
		{25, "25 dispatches"},
	};
	
	for (auto& cfg : configs) {
		OaI32 N = cfg.N;
		OaVec<OaVkBuffer> bufs(N + 1);
		for (OaI32 i = 0; i <= N; ++i) {
			auto res = rt->Allocator.AllocHostVisible(4096);
			ASSERT_TRUE(res.IsOk());
			bufs[i] = res.GetValue();
		}
		
		OaComputeGraph graph;
		BuildChainGraph(graph, bufs, N);
		
		auto stats = graph.GetStats();
		auto groups = graph.ComputeAliasGroups();
		auto lifetimes = graph.ComputeLifetimes();
		
		double savingsPct = stats.TotalBufferBytes > 0
			? (100.0 * stats.PotentialAliasSavings / stats.TotalBufferBytes) : 0.0;
		
		fprintf(stderr, "  %-12s %10zu    %10zu    %8llu B   %5.1f%%\n",
			cfg.Label,
			lifetimes.Size(), groups.Size(),
			static_cast<unsigned long long>(stats.PotentialAliasSavings),
			savingsPct);
		
		for (auto& b : bufs) rt->Allocator.Free(b);
	}
}

TEST(ComputeGraph, AllocatorBackedAliasesExecuteCorrectly) {
	auto* rt = OaEngine::GetGlobal();
	ASSERT_NE(rt, nullptr);
	constexpr OaU32 N = 256;
	OaVec<OaMatrix> matrices;
	OaVec<OaVkBuffer> buffers;
	for (OaU32 i = 0; i < 5U; ++i) {
		matrices.PushBack(OaFnMatrix::Empty(
			{static_cast<OaI64>(N)}, OaScalarType::Float32,
			OaMemoryPlacement::HostUpload));
		ASSERT_TRUE(matrices.Back().HasStorage());
		buffers.PushBack(matrices.Back().GetVkBuffer());
	}
	auto* input = matrices[0].DataAs<OaF32>();
	for (OaU32 i = 0; i < N; ++i) input[i] = static_cast<OaF32>(i + 1U);
	ASSERT_TRUE(rt->Allocator.FlushHostBuffer(buffers[0], 0, buffers[0].Size));

	OaComputeGraph graph;
	BuildChainGraph(graph, buffers, 4);
	OaMatrix* eligible[] = {&matrices[1], &matrices[3]};
	const auto beforeAlias = rt->Allocator.GetStats();
	ASSERT_TRUE(graph.MaterializeAliases(*rt, eligible).IsOk());
	const auto afterAlias = rt->Allocator.GetStats();
	EXPECT_EQ(graph.MaterializedAliasSavings(), N * sizeof(OaF32));
	EXPECT_GE(beforeAlias.AllocationBytes, afterAlias.AllocationBytes + N * sizeof(OaF32));
	EXPECT_EQ(matrices[1].GetVkBuffer().Placement, OaMemoryPlacement::HostUpload);
	EXPECT_EQ(matrices[1].Data(), matrices[3].Data());
	ASSERT_TRUE(graph.Compile(*rt).IsOk());
	// Matrices 1 and 3 are distinct VkBuffer handles over one allocation.
	// Their non-overlapping logical lifetimes still require one global memory
	// dependency when the physical bytes are handed from the first alias to the
	// second; a per-buffer barrier cannot scope both handles.
	EXPECT_EQ(graph.GetStats().AliasBarrierCount, 1U);
	const auto debugReport = graph.DebugReportJson("allocator-alias").StdStr();
	EXPECT_NE(debugReport.find("\"scope\": \"memory_alias\""), std::string::npos);
	EXPECT_NE(debugReport.find("\"reason\": \"read_after_write\""),
		std::string::npos);
	EXPECT_NE(debugReport.find("\"ownership_transfer\": false"),
		std::string::npos);
	ASSERT_TRUE(graph.Replay(*rt).IsOk());
	ASSERT_TRUE(graph.WaitForPendingReplay(rt->Device).IsOk());
	ASSERT_TRUE(rt->Allocator.InvalidateHostBuffer(
		matrices[4].GetVkBuffer(), 0, matrices[4].GetVkBuffer().Size));
	const auto* output = matrices[4].DataAs<const OaF32>();
	const OaF32 factor = 1.001F * 1.001F * 1.001F * 1.001F;
	for (OaU32 i = 0; i < N; ++i) {
		EXPECT_NEAR(output[i], static_cast<OaF32>(i + 1U) * factor, 1e-4F);
	}

	graph.Destroy(rt->Device);
}

TEST(ComputeGraph, AliasMaterializationRejectsExternallyOwnedTransient) {
	auto* rt = OaEngine::GetGlobal();
	ASSERT_NE(rt, nullptr);
	constexpr OaU32 N = 64;
	OaVec<OaMatrix> matrices;
	OaVec<OaVkBuffer> buffers;
	for (OaU32 i = 0; i < 5U; ++i) {
		matrices.PushBack(OaFnMatrix::Empty(
			{static_cast<OaI64>(N)}, OaScalarType::Float32,
			OaMemoryPlacement::HostUpload));
		buffers.PushBack(matrices.Back().GetVkBuffer());
	}
	OaMatrix retainedView = matrices[1];
	OaComputeGraph graph;
	BuildChainGraph(graph, buffers, 4);
	OaMatrix* eligible[] = {&matrices[1], &matrices[3]};
	auto status = graph.MaterializeAliases(*rt, eligible);
	EXPECT_FALSE(status.IsOk());
	EXPECT_EQ(status.GetCode(), OaStatusCode::FailedPrecondition);
	EXPECT_EQ(retainedView.GetVkBuffer().Buffer, matrices[1].GetVkBuffer().Buffer);
}

TEST(ComputeGraph, AllocatorBackedDeviceLocalAliasesExecuteCorrectly) {
	auto* rt = OaEngine::GetGlobal();
	ASSERT_NE(rt, nullptr);
	constexpr OaU32 N = 256;
	OaVec<OaMatrix> matrices;
	OaVec<OaVkBuffer> buffers;
	for (OaU32 i = 0; i < 5U; ++i) {
		const auto placement = (i == 0U or i == 4U)
			? OaMemoryPlacement::HostUpload : OaMemoryPlacement::DeviceLocal;
		matrices.PushBack(OaFnMatrix::Empty(
			{static_cast<OaI64>(N)}, OaScalarType::Float32, placement));
		ASSERT_TRUE(matrices.Back().HasStorage());
		buffers.PushBack(matrices.Back().GetVkBuffer());
	}
	auto* input = matrices[0].DataAs<OaF32>();
	for (OaU32 i = 0; i < N; ++i) input[i] = static_cast<OaF32>(i + 1U);
	ASSERT_TRUE(rt->Allocator.FlushHostBuffer(buffers[0], 0, buffers[0].Size));

	OaComputeGraph graph;
	BuildChainGraph(graph, buffers, 4);
	OaMatrix* eligible[] = {&matrices[1], &matrices[3]};
	ASSERT_TRUE(graph.MaterializeAliases(*rt, eligible).IsOk());
	EXPECT_EQ(matrices[1].GetVkBuffer().Placement, OaMemoryPlacement::DeviceLocal);
	EXPECT_EQ(matrices[3].GetVkBuffer().Placement, OaMemoryPlacement::DeviceLocal);
	EXPECT_EQ(matrices[1].Data(), nullptr);
	ASSERT_TRUE(graph.Execute(*rt).IsOk());
	ASSERT_TRUE(rt->Allocator.InvalidateHostBuffer(
		matrices[4].GetVkBuffer(), 0, matrices[4].GetVkBuffer().Size));
	const auto* output = matrices[4].DataAs<const OaF32>();
	const OaF32 factor = 1.001F * 1.001F * 1.001F * 1.001F;
	for (OaU32 i = 0; i < N; ++i) {
		EXPECT_NEAR(output[i], static_cast<OaF32>(i + 1U) * factor, 1e-4F);
	}
	graph.Destroy(rt->Device);
}

TEST(OaDnnPlanner, PartitionsPackedQkvGatedFfnAndFallback) {
	OaDnnGraph graph;
	auto addMatrix = [&](OaDnnMatrixId id, OaMatrixShape shape, bool external) {
		ASSERT_TRUE(graph.AddMatrix({.Id = id, .Shape = shape,
			.Dtype = OaScalarType::Float32, .External = external,
			.Virtual = not external}).IsOk());
	};
	addMatrix(0, {8, 16}, true);       // shared activation
	addMatrix(1, {16, 16}, true); addMatrix(2, {16, 16}, true); addMatrix(3, {16, 16}, true);
	addMatrix(4, {8, 16}, false); addMatrix(5, {8, 16}, false); addMatrix(6, {8, 16}, false);
	addMatrix(7, {16, 16}, true); addMatrix(8, {16, 16}, true);
	addMatrix(9, {8, 16}, false); addMatrix(10, {8, 16}, false);
	addMatrix(11, {8, 16}, false); addMatrix(12, {8, 16}, false);
	addMatrix(13, {8, 16}, true);

	auto op = [&](OaDnnOpType type, std::initializer_list<OaDnnMatrixId> inputs,
		std::initializer_list<OaDnnMatrixId> outputs) {
		OaDnnOpDesc desc; desc.Type = type;
		desc.Inputs = inputs; desc.Outputs = outputs;
		ASSERT_TRUE(graph.AddOp(desc).IsOk());
	};
	op(OaDnnOpType::Matmul, {0, 1}, {4});
	op(OaDnnOpType::Matmul, {0, 2}, {5});
	op(OaDnnOpType::Matmul, {0, 3}, {6});
	op(OaDnnOpType::Matmul, {0, 7}, {9});
	op(OaDnnOpType::Matmul, {0, 8}, {10});
	op(OaDnnOpType::Silu, {9}, {11});
	op(OaDnnOpType::Multiply, {11, 10}, {12});
	op(OaDnnOpType::Add, {12, 0}, {13});

	auto result = OaDnnPlanner::Plan(graph);
	ASSERT_TRUE(result.IsOk()) << result.GetStatus().GetMessage().Data();
	const auto& plan = result.GetValue();
	ASSERT_EQ(plan.Partitions.Size(), 3U);
	EXPECT_EQ(plan.Partitions[0].Engine, OaDnnEngineType::PackedQkv);
	EXPECT_EQ(plan.Partitions[0].Ops.Size(), 3U);
	EXPECT_EQ(plan.Partitions[1].Engine, OaDnnEngineType::GatedFfn);
	EXPECT_EQ(plan.Partitions[1].Ops.Size(), 4U);
	EXPECT_EQ(plan.Partitions[2].Engine, OaDnnEngineType::Portable);
	EXPECT_NE(plan.GraphHash, 0U);
}

TEST(OaDnnPlanner, RejectsUseBeforeProducer) {
	OaDnnGraph graph;
	ASSERT_TRUE(graph.AddMatrix({.Id = 0, .Shape = {2, 2}, .External = false}).IsOk());
	ASSERT_TRUE(graph.AddMatrix({.Id = 1, .Shape = {2, 2}, .External = false}).IsOk());
	OaDnnOpDesc op; op.Type = OaDnnOpType::Relu; op.Inputs = {0}; op.Outputs = {1};
	ASSERT_TRUE(graph.AddOp(op).IsOk());
	EXPECT_FALSE(graph.Validate().IsOk());
}
// =============================================================================

// Main is provided by MlTestMain.cpp (shared test infrastructure)
