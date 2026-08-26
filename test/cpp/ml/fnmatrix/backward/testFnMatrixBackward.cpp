// Test suite for oa::FnMatrix backward operations
// Tests gradient computation for neural network operations

#include <oa/oa.h>
#include <oa/ml/autograd.h>
#include <oa/runtime/executionSession.h>
#include <oa/runtime/spirv.h>
#include <oaTest.h>
#include <cmath>
#include <cstdlib>
#include <numeric>

// ─── Helper Functions ───────────────────────────────────────────────────────

static float relu(float x) { return x > 0.0f ? x : 0.0f; }
static float reluGrad(float x) { return x > 0.0f ? 1.0f : 0.0f; }

static float tanh(float x) { return std::tanh(x); }
static float tanhGrad(float y) { return 1.0f - y * y; }

static float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }
static float sigmoidGrad(float y) { return y * (1.0f - y); }

static float leakyRelu(float x, float alpha = 0.01f) { return x > 0.0f ? x : alpha * x; }
static float leakyReluGrad(float x, float alpha = 0.01f) { return x > 0.0f ? 1.0f : alpha; }

static float elu(float x, float alpha = 1.0f) { return x > 0.0f ? x : alpha * (std::exp(x) - 1.0f); }
static float eluGrad(float y, float alpha = 1.0f) { return y > 0.0f ? 1.0f : y + alpha; }

static oa::Matrix matrixFromHost(const std::vector<float>& inData, const oa::MatrixShape& inShape) {
	return oa::FnMatrix::fromBytes(
		oa::Span<const oa::U8>(reinterpret_cast<const oa::U8*>(inData.data()),
			inData.size() * sizeof(float)), inShape);
}

static std::vector<float> matrixToHost(const oa::Matrix& inMatrix) {
	std::vector<float> result(static_cast<size_t>(inMatrix.numElements()));
	EXPECT_TRUE(oa::FnMatrix::copyToHost(
		inMatrix, result.data(), result.size() * sizeof(float)).isOk());
	return result;
}

static std::vector<float> axisSoftmaxReference(
	const std::vector<float>& inInput, oa::U32 inOuter, oa::U32 inDim,
	oa::U32 inInner, bool inLog) {
	std::vector<float> result(inInput.size());
	for (oa::U32 outer = 0; outer < inOuter; ++outer) {
		for (oa::U32 inner = 0; inner < inInner; ++inner) {
			const oa::U32 base = outer * inDim * inInner + inner;
			float maximum = -INFINITY;
			for (oa::U32 axis = 0; axis < inDim; ++axis) {
				maximum = std::max(maximum, inInput[base + axis * inInner]);
			}
			float sum = 0.0F;
			for (oa::U32 axis = 0; axis < inDim; ++axis) {
				sum += std::exp(inInput[base + axis * inInner] - maximum);
			}
			for (oa::U32 axis = 0; axis < inDim; ++axis) {
				const oa::U32 index = base + axis * inInner;
				const float shifted = inInput[index] - maximum;
				result[index] = inLog ? shifted - std::log(sum)
					: std::exp(shifted) / sum;
			}
		}
	}
	return result;
}

static std::vector<float> axisSoftmaxGradientReference(
	const std::vector<float>& inOutput, const std::vector<float>& inUpstream,
	oa::U32 inOuter, oa::U32 inDim, oa::U32 inInner, bool inLog) {
	std::vector<float> result(inOutput.size());
	for (oa::U32 outer = 0; outer < inOuter; ++outer) {
		for (oa::U32 inner = 0; inner < inInner; ++inner) {
			const oa::U32 base = outer * inDim * inInner + inner;
			float reduced = 0.0F;
			for (oa::U32 axis = 0; axis < inDim; ++axis) {
				const oa::U32 index = base + axis * inInner;
				reduced += inLog ? inUpstream[index]
					: inUpstream[index] * inOutput[index];
			}
			for (oa::U32 axis = 0; axis < inDim; ++axis) {
				const oa::U32 index = base + axis * inInner;
				result[index] = inLog
					? inUpstream[index] - std::exp(inOutput[index]) * reduced
					: inOutput[index] * (inUpstream[index] - reduced);
			}
		}
	}
	return result;
}

TEST(FnMatrixBackward, SchemaPilotsAttachTapeToSemanticOperations) {
	auto& ctx = oa::ExecutionSession::getActive();
	oa::ExecutionSession::RecordingScope scope(ctx);
	ctx.clear();
	auto a = oa::FnMatrix::empty({2, 3});
	auto b = oa::FnMatrix::empty({2, 3});
	auto weight = oa::FnMatrix::empty({4, 3});
	a.setRequiresGrad(true);
	b.setRequiresGrad(true);
	weight.setRequiresGrad(true);
	// setRequiresGrad creates persistent gradient storage. Its initialization is
	// unrelated to this provenance-only recording and is deliberately discarded.
	ctx.clear();

	oa::GradientTape tape;
	auto sum = oa::FnMatrix::add(a, b);
	auto product = oa::FnMatrix::matMulNt(sum, weight);
	ASSERT_FALSE(product.isEmpty());
	auto loss = oa::FnMatrix::sum(product, -1);
	tape.backward(loss);

	const auto* graph = ctx.semanticGraph();
	ASSERT_NE(graph, nullptr);
	ASSERT_TRUE(graph->validate().isOk());
	ASSERT_GE(graph->operationCount(), 6U);
	ASSERT_EQ(graph->autograd().size(), 3U);
	EXPECT_EQ(graph->autograd()[0].forwardOp, 0U);
	EXPECT_EQ(graph->autograd()[1].forwardOp, 1U);
	EXPECT_EQ(graph->autograd()[2].forwardOp, 2U);
	EXPECT_LT(graph->autograd()[0].sequence, graph->autograd()[1].sequence);
	EXPECT_LT(graph->autograd()[1].sequence, graph->autograd()[2].sequence);
	EXPECT_TRUE(graph->autograd()[0].backwardExpanded);
	EXPECT_EQ(graph->autograd()[0].backwardOpCount, 0U);
	EXPECT_TRUE(graph->autograd()[1].backwardExpanded);
	EXPECT_EQ(graph->autograd()[1].backwardOpCount, 5U);
	EXPECT_TRUE(graph->autograd()[2].backwardExpanded);
	EXPECT_EQ(graph->autograd()[2].backwardOpCount, 2U);

	const auto sumNode = sum.getGradFn();
	const auto productNode = product.getGradFn();
	const auto lossNode = loss.getGradFn();
	ASSERT_TRUE(sumNode);
	ASSERT_TRUE(productNode);
	ASSERT_TRUE(lossNode);
	EXPECT_EQ(sumNode->forwardSemanticOp_, 0U);
	EXPECT_EQ(productNode->forwardSemanticOp_, 1U);
	EXPECT_EQ(lossNode->forwardSemanticOp_, 2U);
	EXPECT_EQ(graph->autograd()[0].output,
		graph->operations()[0].outputs[0]);
	EXPECT_EQ(graph->autograd()[1].output,
		graph->operations()[1].outputs[0]);
	EXPECT_EQ(graph->autograd()[2].output,
		graph->operations()[2].outputs[0]);
	const auto matmulBackwardFirst =
		graph->autograd()[1].backwardFirstOp;
	ASSERT_LT(matmulBackwardFirst + 4U, graph->operationCount());
	oa::U32 matmulCount = 0;
	oa::U32 transposeCount = 0;
	for (oa::U32 offset = 0; offset < 5U; ++offset) {
		const auto& operation =
			graph->operations()[matmulBackwardFirst + offset];
		EXPECT_EQ(operation.backwardOf, 1U);
		EXPECT_EQ(operation.backwardSequence, graph->autograd()[1].sequence);
		if (operation.name == oa::detail::opRegistry::FnMatrix::matMulNt.name) {
			++matmulCount;
		} else if (
			operation.name == oa::detail::opRegistry::FnMatrix::transpose.name)
		{
			++transposeCount;
		} else {
			ADD_FAILURE() << "unexpected Core MatMulNt adjoint operation: "
				<< operation.name;
		}
	}
	EXPECT_EQ(matmulCount, 2U);
	EXPECT_EQ(transposeCount, 3U);
	const auto sumBackwardFirst = graph->autograd()[2].backwardFirstOp;
	ASSERT_LT(sumBackwardFirst + 1U, graph->operationCount());
	EXPECT_EQ(graph->operations()[sumBackwardFirst].name,
		oa::detail::opRegistry::FnMatrix::fillInPlace.name);
	EXPECT_EQ(graph->operations()[sumBackwardFirst + 1U].name,
		oa::detail::opRegistry::FnMatrix::mul.name);
	EXPECT_EQ(graph->operations()[sumBackwardFirst].backwardOf, 2U);
	EXPECT_EQ(graph->operations()[sumBackwardFirst + 1U].backwardOf, 2U);

	const auto report = testStdString(graph->debugReportJson("autograd-pilot"));
	EXPECT_NE(report.find("\"autograd\""), std::string::npos);
	EXPECT_NE(report.find("\"backward_of\": 1"), std::string::npos);
	EXPECT_NE(report.find("\"backward_operation_count\": 5"),
		std::string::npos);
	EXPECT_NE(report.find("\"backward_operation_count\": 2"),
		std::string::npos);
	EXPECT_NE(report.find("\"backward_operation_count\": 0"),
		std::string::npos);
	EXPECT_EQ(report.find("VkBuffer"), std::string::npos);
	ctx.clear();
}

// These are architecture-pilot tests, not only numerical kernel tests: Add and
// MatMulNt are hand-written lowerings whose autograd attachment is generated
// from their operation schemas. They fail if that generated policy disappears.
TEST(FnMatrixBackward, SchemaAttachAddSameShape) {
	auto& ctx = oa::ExecutionSession::getActive();
	oa::ExecutionSession::RecordingScope scope(ctx);
	auto a = matrixFromHost({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2});
	auto b = matrixFromHost({5.0f, 6.0f, 7.0f, 8.0f}, {2, 2});
	a.setRequiresGrad(true);
	b.setRequiresGrad(true);

	oa::GradientTape tape;
	auto loss = oa::FnMatrix::sum(oa::FnMatrix::add(a, b), -1);
	tape.backward(loss);
	ASSERT_TRUE(testSubmitAndWait(ctx).isOk());

	for (float grad : matrixToHost(a.gradMatrix())) EXPECT_NEAR(grad, 1.0f, 1e-5f);
	for (float grad : matrixToHost(b.gradMatrix())) EXPECT_NEAR(grad, 1.0f, 1e-5f);
}

TEST(FnMatrixBackward, SchemaAttachAddBroadcastReduction) {
	auto& ctx = oa::ExecutionSession::getActive();
	oa::ExecutionSession::RecordingScope scope(ctx);
	auto a = matrixFromHost({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {2, 3});
	auto b = matrixFromHost({0.5f, 1.0f, 1.5f}, {3});
	a.setRequiresGrad(true);
	b.setRequiresGrad(true);

	oa::GradientTape tape;
	auto loss = oa::FnMatrix::sum(oa::FnMatrix::add(a, b), -1);
	tape.backward(loss);
	ASSERT_TRUE(testSubmitAndWait(ctx).isOk());

	for (float grad : matrixToHost(a.gradMatrix())) EXPECT_NEAR(grad, 1.0f, 1e-5f);
	for (float grad : matrixToHost(b.gradMatrix())) EXPECT_NEAR(grad, 2.0f, 1e-5f);
}

TEST(FnMatrixBackward, SchemaAttachMatMulNtRank2) {
	auto& ctx = oa::ExecutionSession::getActive();
	oa::ExecutionSession::RecordingScope scope(ctx);
	const std::vector<float> aHost = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
	const std::vector<float> bHost = {
		1.0f, 2.0f, 3.0f,
		4.0f, 5.0f, 6.0f,
		7.0f, 8.0f, 9.0f,
		10.0f, 11.0f, 12.0f};
	auto a = matrixFromHost(aHost, {2, 3});
	auto b = matrixFromHost(bHost, {4, 3});
	a.setRequiresGrad(true);
	b.setRequiresGrad(true);

	oa::GradientTape tape;
	auto loss = oa::FnMatrix::sum(oa::FnMatrix::matMulNt(a, b), -1);
	tape.backward(loss);
	ASSERT_TRUE(testSubmitAndWait(ctx).isOk());

	const auto gradA = matrixToHost(a.gradMatrix());
	const auto gradB = matrixToHost(b.gradMatrix());
	for (size_t m = 0; m < 2; ++m) {
		for (size_t k = 0; k < 3; ++k) {
			float expected = 0.0f;
			for (size_t n = 0; n < 4; ++n) expected += bHost[n * 3 + k];
			EXPECT_NEAR(gradA[m * 3 + k], expected, 1e-4f);
		}
	}
	for (size_t n = 0; n < 4; ++n) {
		for (size_t k = 0; k < 3; ++k) {
			const float expected = aHost[k] + aHost[3 + k];
			EXPECT_NEAR(gradB[n * 3 + k], expected, 1e-4f);
		}
	}
}

// ─── Simple Activation backward Tests ──────────────────────────────────────

TEST(FnMatrixBackward, ReluBwd) {
	auto& ctx = oa::ExecutionSession::getActive();
	
	std::vector<float> input_data = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
	std::vector<float> grad_output_data = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
	
	auto input = oa::FnMatrix::fromBytes(oa::Span<const oa::U8>(reinterpret_cast<const oa::U8*>(input_data.data()), input_data.size() * sizeof(float)), oa::MatrixShape{5});
	auto forward_output = oa::FnMatrix::relu(input);
	auto grad_output = oa::FnMatrix::fromBytes(oa::Span<const oa::U8>(reinterpret_cast<const oa::U8*>(grad_output_data.data()), grad_output_data.size() * sizeof(float)), oa::MatrixShape{5});
	
	auto grad_input = oa::FnMatrix::reluBwd(forward_output, grad_output);
	[[maybe_unused]] auto exec_result = testSubmitAndWait(ctx);
	
	std::vector<float> result(5);
	[[maybe_unused]] auto copy_result_1 = oa::FnMatrix::copyToHost(grad_input, result.data(), result.size() * sizeof(float));
	
	// CPU reference
	std::vector<float> expected(5);
	for (size_t i = 0; i < 5; i++) {
		expected[i] = reluGrad(input_data[i]) * grad_output_data[i];
	}
	
	for (size_t i = 0; i < 5; i++) {
		EXPECT_NEAR(result[i], expected[i], 1e-5f);
	}
}

TEST(FnMatrixBackward, TanhBwd) {
	auto& ctx = oa::ExecutionSession::getActive();
	
	std::vector<float> input_data = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
	std::vector<float> grad_output_data = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
	
	auto input = oa::FnMatrix::fromBytes(oa::Span<const oa::U8>(reinterpret_cast<const oa::U8*>(input_data.data()), input_data.size() * sizeof(float)), oa::MatrixShape{5});
	auto forward_output = oa::FnMatrix::tanh(input);
	auto grad_output = oa::FnMatrix::fromBytes(oa::Span<const oa::U8>(reinterpret_cast<const oa::U8*>(grad_output_data.data()), grad_output_data.size() * sizeof(float)), oa::MatrixShape{5});
	
	auto grad_input = oa::FnMatrix::tanhBwd(forward_output, grad_output);
	[[maybe_unused]] auto exec_result = testSubmitAndWait(ctx);
	
	std::vector<float> result(5);
	[[maybe_unused]] auto copy_result_2 = oa::FnMatrix::copyToHost(grad_input, result.data(), result.size() * sizeof(float));
	
	std::vector<float> forward_host(5);
	[[maybe_unused]] auto copy_result_3 = oa::FnMatrix::copyToHost(forward_output, forward_host.data(), forward_host.size() * sizeof(float));
	
	for (size_t i = 0; i < 5; i++) {
		float expected = tanhGrad(forward_host[i]) * grad_output_data[i];
		EXPECT_NEAR(result[i], expected, 1e-5f);
	}
}

TEST(FnMatrixBackward, SigmoidBwd) {
	auto& ctx = oa::ExecutionSession::getActive();
	
	std::vector<float> input_data = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
	std::vector<float> grad_output_data = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
	
	auto input = oa::FnMatrix::fromBytes(oa::Span<const oa::U8>(reinterpret_cast<const oa::U8*>(input_data.data()), input_data.size() * sizeof(float)), oa::MatrixShape{5});
	auto forward_output = oa::FnMatrix::sigmoid(input);
	auto grad_output = oa::FnMatrix::fromBytes(oa::Span<const oa::U8>(reinterpret_cast<const oa::U8*>(grad_output_data.data()), grad_output_data.size() * sizeof(float)), oa::MatrixShape{5});
	
	auto grad_input = oa::FnMatrix::sigmoidBwd(forward_output, grad_output);
	[[maybe_unused]] auto exec_result = testSubmitAndWait(ctx);
	
	std::vector<float> result(5);
	[[maybe_unused]] auto copy_result_4 = oa::FnMatrix::copyToHost(grad_input, result.data(), result.size() * sizeof(float));
	
	std::vector<float> forward_host(5);
	[[maybe_unused]] auto copy_result_5 = oa::FnMatrix::copyToHost(forward_output, forward_host.data(), forward_host.size() * sizeof(float));
	
	for (size_t i = 0; i < 5; i++) {
		float expected = sigmoidGrad(forward_host[i]) * grad_output_data[i];
		EXPECT_NEAR(result[i], expected, 1e-5f);
	}
}

TEST(FnMatrixBackward, GeluBwd) {
	auto& ctx = oa::ExecutionSession::getActive();
	
	std::vector<float> input_data = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
	std::vector<float> grad_output_data = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
	
	auto input = oa::FnMatrix::fromBytes(oa::Span<const oa::U8>(reinterpret_cast<const oa::U8*>(input_data.data()), input_data.size() * sizeof(float)), oa::MatrixShape{5});
	auto forward_output = oa::FnMatrix::gelu(input);
	auto grad_output = oa::FnMatrix::fromBytes(oa::Span<const oa::U8>(reinterpret_cast<const oa::U8*>(grad_output_data.data()), grad_output_data.size() * sizeof(float)), oa::MatrixShape{5});
	
	auto grad_input = oa::FnMatrix::geluBwd(forward_output, grad_output);
	[[maybe_unused]] auto exec_result = testSubmitAndWait(ctx);
	
	std::vector<float> result(5);
	[[maybe_unused]] auto copy_result_6 = oa::FnMatrix::copyToHost(grad_input, result.data(), result.size() * sizeof(float));
	
	for (size_t i = 0; i < 5; i++) {
		EXPECT_TRUE(std::isfinite(result[i]));
	}
}

TEST(FnMatrixBackward, SiluBwd) {
	auto& ctx = oa::ExecutionSession::getActive();
	
	std::vector<float> input_data = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
	std::vector<float> grad_output_data = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
	
	auto input = oa::FnMatrix::fromBytes(oa::Span<const oa::U8>(reinterpret_cast<const oa::U8*>(input_data.data()), input_data.size() * sizeof(float)), oa::MatrixShape{5});
	auto forward_output = oa::FnMatrix::silu(input);
	auto grad_output = oa::FnMatrix::fromBytes(oa::Span<const oa::U8>(reinterpret_cast<const oa::U8*>(grad_output_data.data()), grad_output_data.size() * sizeof(float)), oa::MatrixShape{5});
	
	auto grad_input = oa::FnMatrix::siluBwd(forward_output, grad_output);
	[[maybe_unused]] auto exec_result = testSubmitAndWait(ctx);
	
	std::vector<float> result(5);
	[[maybe_unused]] auto copy_result_7 = oa::FnMatrix::copyToHost(grad_input, result.data(), result.size() * sizeof(float));
	
	for (size_t i = 0; i < 5; i++) {
		EXPECT_TRUE(std::isfinite(result[i]));
	}
}

TEST(FnMatrixBackward, SoftplusBwd) {
	auto& ctx = oa::ExecutionSession::getActive();
	
	std::vector<float> input_data = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
	std::vector<float> grad_output_data = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
	
	auto input = oa::FnMatrix::fromBytes(oa::Span<const oa::U8>(reinterpret_cast<const oa::U8*>(input_data.data()), input_data.size() * sizeof(float)), oa::MatrixShape{5});
	auto forward_output = oa::FnMatrix::softplus(input);
	auto grad_output = oa::FnMatrix::fromBytes(oa::Span<const oa::U8>(reinterpret_cast<const oa::U8*>(grad_output_data.data()), grad_output_data.size() * sizeof(float)), oa::MatrixShape{5});
	
	auto grad_input = oa::FnMatrix::softplusBwd(forward_output, grad_output);
	[[maybe_unused]] auto exec_result = testSubmitAndWait(ctx);
	
	std::vector<float> result(5);
	[[maybe_unused]] auto copy_result_8 = oa::FnMatrix::copyToHost(grad_input, result.data(), result.size() * sizeof(float));
	
	for (size_t i = 0; i < 5; i++) {
		EXPECT_TRUE(std::isfinite(result[i]));
	}
}

TEST(FnMatrixBackward, LeakyReluBwd) {
	auto& ctx = oa::ExecutionSession::getActive();
	
	std::vector<float> input_data = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
	std::vector<float> grad_output_data = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
	
	auto input = oa::FnMatrix::fromBytes(oa::Span<const oa::U8>(reinterpret_cast<const oa::U8*>(input_data.data()), input_data.size() * sizeof(float)), oa::MatrixShape{5});
	auto forward_output = oa::FnMatrix::leakyRelu(input, 0.01f);
	auto grad_output = oa::FnMatrix::fromBytes(oa::Span<const oa::U8>(reinterpret_cast<const oa::U8*>(grad_output_data.data()), grad_output_data.size() * sizeof(float)), oa::MatrixShape{5});
	
	auto grad_input = oa::FnMatrix::leakyReluBwd(forward_output, grad_output);
	[[maybe_unused]] auto exec_result = testSubmitAndWait(ctx);
	
	std::vector<float> result(5);
	[[maybe_unused]] auto copy_result_9 = oa::FnMatrix::copyToHost(grad_input, result.data(), result.size() * sizeof(float));
	
	for (size_t i = 0; i < 5; i++) {
		float expected = leakyReluGrad(input_data[i]) * grad_output_data[i];
		EXPECT_NEAR(result[i], expected, 1e-5f);
	}
}

TEST(FnMatrixBackward, EluBwd) {
	auto& ctx = oa::ExecutionSession::getActive();
	
	std::vector<float> input_data = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
	std::vector<float> grad_output_data = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
	
	auto input = oa::FnMatrix::fromBytes(oa::Span<const oa::U8>(reinterpret_cast<const oa::U8*>(input_data.data()), input_data.size() * sizeof(float)), oa::MatrixShape{5});
	auto forward_output = oa::FnMatrix::elu(input, 1.0f);
	auto grad_output = oa::FnMatrix::fromBytes(oa::Span<const oa::U8>(reinterpret_cast<const oa::U8*>(grad_output_data.data()), grad_output_data.size() * sizeof(float)), oa::MatrixShape{5});
	
	auto grad_input = oa::FnMatrix::eluBwd(forward_output, grad_output);
	[[maybe_unused]] auto exec_result = testSubmitAndWait(ctx);
	
	std::vector<float> result(5);
	[[maybe_unused]] auto copy_result_10 = oa::FnMatrix::copyToHost(grad_input, result.data(), result.size() * sizeof(float));
	
	std::vector<float> forward_host(5);
	[[maybe_unused]] auto copy_result_11 = oa::FnMatrix::copyToHost(forward_output, forward_host.data(), forward_host.size() * sizeof(float));
	
	for (size_t i = 0; i < 5; i++) {
		float expected = eluGrad(forward_host[i]) * grad_output_data[i];
		EXPECT_NEAR(result[i], expected, 1e-4f);
	}
}

// ─── Non-default-alpha autograd regression ─────────────────────────────────
// The value-level *Bwd above always ran with the default alpha. The bug these
// Guard: the generated GradLeakyRelu/GradElu node dropped alpha, so the tape
// always backpropagated with the defaulted slope (0.01 / 1.0). With a NON-default
// alpha the negative-side analytic grad diverged from finite differences. Now alpha
// is threaded through the autograd node (ctor_args=["inAlpha"]); these must match FD.

// Central-difference gradcheck of a leaf input through the autograd tape.
static void alphaAutogradGradcheck(
	oa::ExecutionSession& inCtx, oa::Matrix& inInput, const std::function<oa::Matrix()>& inForward,
	const oa::Matrix& inTarget, const char* inName)
{
	oa::GradientTape tape;
	auto out  = inForward();
	auto loss = oa::FnLoss::mse(out, inTarget);
	tape.backward(loss);
	(void)testSubmitAndWait(inCtx);

	auto analytic = inInput.gradMatrix();
	const oa::F32* ana = analytic.dataAs<const oa::F32>();

	auto lossFunc = [&]() -> oa::F32 {
		oa::GradNo noGrad;
		auto o = inForward();
		auto l = oa::FnLoss::mse(o, inTarget);
		(void)testSubmitAndWait(inCtx);
		return l.dataAs<const oa::F32>()[0];
	};

	oa::F32* data = inInput.dataAs<oa::F32>();
	const oa::I64 n = inInput.numElements();
	const oa::F32 eps = 1e-3f;
	int failed = 0, nonTrivial = 0;
	for (oa::I64 i = 0; i < n; ++i) {
		const oa::F32 orig = data[i];
		data[i] = orig + eps; (void)testSubmitAndWait(inCtx); const oa::F32 lp = lossFunc();
		data[i] = orig - eps; (void)testSubmitAndWait(inCtx); const oa::F32 lm = lossFunc();
		data[i] = orig; (void)testSubmitAndWait(inCtx);
		const oa::F32 num = (lp - lm) / (2.0f * eps);
		const oa::F32 tol = 2e-3f + 2e-2f * std::abs(num);
		if (std::abs(num - ana[i]) > tol) {
			++failed;
			printf("  [%s] idx %lld: analytic=%.6f numerical=%.6f MISMATCH\n",
				inName, static_cast<long long>(i), ana[i], num);
		}
		if (std::abs(num) > 5e-4f) ++nonTrivial;
	}
	EXPECT_EQ(failed, 0) << inName << ": autograd grad disagrees with finite differences";
	EXPECT_GE(nonTrivial, 3) << inName << ": gradients all ~0 — vacuous check";
}

// ─── SPIR-V push-constant reflection (backs the bindless-contract assert) ──
// The Record debug assert is only as good as the reflected block size. Pin the
// exact reflected sizes for representative kernel shapes so the check can never
// silently go vacuous (e.g. reflection regressing to always-0). Each shader's
// PushConstants struct = [one uint buffer-index per bound buffer] ++ [host tail].
TEST(FnMatrixBackward, SpirvPushBlockReflection) {
	// binary {a_idx,b_idx,out_idx,count}                 = 4*4 = 16
	EXPECT_EQ(oavk::spirvPushConstantBlockSizeByName("Add"), 16u);
	// unary {x_idx,out_idx,count}                        = 3*4 = 12
	EXPECT_EQ(oavk::spirvPushConstantBlockSizeByName("Sigmoid"), 12u);
	// unary_scalar {x_idx,out_idx,count,alpha}           = 4*4 = 16
	EXPECT_EQ(oavk::spirvPushConstantBlockSizeByName("LeakyRelu"), 16u);
	EXPECT_EQ(oavk::spirvPushConstantBlockSizeByName("Elu"), 16u);
	// hand-written bwd {x_idx,grad_out_idx,grad_in_idx,count,alpha} = 5*4 = 20
	EXPECT_EQ(oavk::spirvPushConstantBlockSizeByName("LeakyReluBwd"), 20u);
	// unknown kernel → 0 (conservative; assert is skipped)
	EXPECT_EQ(oavk::spirvPushConstantBlockSizeByName("NoSuchKernel_xyz"), 0u);
}

TEST(FnMatrixBackward, LeakyReluAutogradNonDefaultAlpha) {
	auto& ctx = oa::ExecutionSession::getActive();
	oa::ExecutionSession::RecordingScope scope(ctx);
	const oa::F32 alpha = 0.25f;  // far from the 0.01 default → exposes a dropped alpha

	std::vector<float> x = {-2.0f, -1.3f, -0.4f, 0.5f, 1.7f, 2.5f};
	std::vector<float> t = { 0.3f, -0.7f,  0.9f, 0.1f, -1.1f, 0.6f};
	auto input  = oa::FnMatrix::fromBytes(oa::Span<const oa::U8>(reinterpret_cast<const oa::U8*>(x.data()), x.size() * sizeof(float)), oa::MatrixShape{6});
	auto target = oa::FnMatrix::fromBytes(oa::Span<const oa::U8>(reinterpret_cast<const oa::U8*>(t.data()), t.size() * sizeof(float)), oa::MatrixShape{6});
	input.setRequiresGrad(true);

	alphaAutogradGradcheck(ctx, input, [&]() { return oa::FnMatrix::leakyRelu(input, alpha); },
		target, "leakyRelu(alpha=0.25)");
}

TEST(FnMatrixBackward, EluAutogradNonDefaultAlpha) {
	auto& ctx = oa::ExecutionSession::getActive();
	oa::ExecutionSession::RecordingScope scope(ctx);
	const oa::F32 alpha = 0.5f;  // != 1.0 default → exposes a dropped alpha on y<=0

	std::vector<float> x = {-2.0f, -1.3f, -0.4f, 0.5f, 1.7f, 2.5f};
	std::vector<float> t = { 0.3f, -0.7f,  0.9f, 0.1f, -1.1f, 0.6f};
	auto input  = oa::FnMatrix::fromBytes(oa::Span<const oa::U8>(reinterpret_cast<const oa::U8*>(x.data()), x.size() * sizeof(float)), oa::MatrixShape{6});
	auto target = oa::FnMatrix::fromBytes(oa::Span<const oa::U8>(reinterpret_cast<const oa::U8*>(t.data()), t.size() * sizeof(float)), oa::MatrixShape{6});
	input.setRequiresGrad(true);

	alphaAutogradGradcheck(ctx, input, [&]() { return oa::FnMatrix::elu(input, alpha); },
		target, "elu(alpha=0.5)");
}

TEST(FnMatrixBackward, MishBwd) {
	auto& ctx = oa::ExecutionSession::getActive();

	std::vector<float> input_data = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
	std::vector<float> grad_output_data = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f};

	auto input = oa::FnMatrix::fromBytes(oa::Span<const oa::U8>(reinterpret_cast<const oa::U8*>(input_data.data()), input_data.size() * sizeof(float)), oa::MatrixShape{5});
	auto forward_output = oa::FnMatrix::mish(input);
	auto grad_output = oa::FnMatrix::fromBytes(oa::Span<const oa::U8>(reinterpret_cast<const oa::U8*>(grad_output_data.data()), grad_output_data.size() * sizeof(float)), oa::MatrixShape{5});

	auto grad_input = oa::FnMatrix::mishBwd(forward_output, grad_output);
	[[maybe_unused]] auto exec_result = testSubmitAndWait(ctx);

	std::vector<float> result(5);
	[[maybe_unused]] auto copy_result_12 = oa::FnMatrix::copyToHost(grad_input, result.data(), result.size() * sizeof(float));

	for (size_t i = 0; i < 5; i++) {
		EXPECT_TRUE(std::isfinite(result[i]));
	}
}

TEST(FnMatrixBackward, SoftmaxFamilyPreservesSelectedAxisInAutograd) {
	auto& ctx = oa::ExecutionSession::getActive();
	constexpr oa::U32 kOuter = 2U;
	constexpr oa::U32 kDim = 3U;
	constexpr oa::U32 kInner = 4U;
	std::vector<float> inputData(kOuter * kDim * kInner);
	std::vector<float> upstreamData(inputData.size());
	for (oa::U32 i = 0; i < inputData.size(); ++i) {
		inputData[i] = static_cast<float>(static_cast<oa::I32>((i * 7U) % 17U) - 8) * 0.17F;
		upstreamData[i] = static_cast<float>(static_cast<oa::I32>((i * 5U) % 11U) - 5) * 0.09F;
	}

	auto runCase = [&](bool inLog) {
		ctx.clear();
		auto input = matrixFromHost(inputData, {kOuter, kDim, kInner});
		auto upstream = matrixFromHost(upstreamData, {kOuter, kDim, kInner});
		input.setRequiresGrad(true);
		oa::GradientTape tape;
		auto output = inLog
			? oa::FnMatrix::logSoftmax(input, 1)
			: oa::FnMatrix::softmax(input, 1);
		ASSERT_FALSE(output.isEmpty());
		auto loss = oa::FnMatrix::sum(oa::FnMatrix::mul(output, upstream), -1);
		tape.backward(loss);
		ASSERT_TRUE(testSubmitAndWait(ctx).isOk());

		const auto expectedOutput = axisSoftmaxReference(
			inputData, kOuter, kDim, kInner, inLog);
		const auto expectedGradient = axisSoftmaxGradientReference(
			expectedOutput, upstreamData, kOuter, kDim, kInner, inLog);
		const auto outputData = matrixToHost(output);
		const auto gradientData = matrixToHost(input.gradMatrix());
		for (oa::U32 i = 0; i < inputData.size(); ++i) {
			EXPECT_NEAR(outputData[i], expectedOutput[i], 2e-5F)
				<< (inLog ? "LogSoftmax" : "Softmax") << " output i=" << i;
			EXPECT_NEAR(gradientData[i], expectedGradient[i], 3e-5F)
				<< (inLog ? "LogSoftmax" : "Softmax") << " gradient i=" << i;
		}
	};
	runCase(false);
	runCase(true);
	ctx.clear();
}

TEST(FnMatrixBackward, MeanPreservesSelectedAxisInAutograd) {
	auto& ctx = oa::ExecutionSession::getActive();
	ctx.clear();
	constexpr oa::U32 kOuter = 2U;
	constexpr oa::U32 kDim = 3U;
	constexpr oa::U32 kInner = 4U;
	std::vector<float> inputData(kOuter * kDim * kInner);
	std::vector<float> upstreamData(kOuter * kInner);
	for (oa::U32 i = 0; i < static_cast<oa::U32>(inputData.size()); ++i) {
		inputData[i] = static_cast<float>(static_cast<oa::I32>(i) - 9) * 0.13F;
	}
	for (oa::U32 i = 0; i < static_cast<oa::U32>(upstreamData.size()); ++i) {
		upstreamData[i] = static_cast<float>(static_cast<oa::I32>(i) - 3) * 0.21F;
	}

	auto input = matrixFromHost(inputData, {kOuter, kDim, kInner});
	auto upstream = matrixFromHost(upstreamData, {kOuter, 1, kInner});
	input.setRequiresGrad(true);
	oa::GradientTape tape;
	auto output = oa::FnMatrix::mean(input, 1);
	ASSERT_EQ(output.getShape(), (oa::MatrixShape{kOuter, 1, kInner}));
	auto loss = oa::FnMatrix::sum(oa::FnMatrix::mul(output, upstream), -1);
	tape.backward(loss);
	ASSERT_TRUE(testSubmitAndWait(ctx).isOk());

	const auto outputData = matrixToHost(output);
	const auto gradientData = matrixToHost(input.gradMatrix());
	for (oa::U32 outer = 0; outer < kOuter; ++outer) {
		for (oa::U32 inner = 0; inner < kInner; ++inner) {
			const oa::U32 outputIndex = outer * kInner + inner;
			float expectedOutput = 0.0F;
			for (oa::U32 axis = 0; axis < kDim; ++axis) {
				expectedOutput += inputData[
					outer * kDim * kInner + axis * kInner + inner];
			}
			expectedOutput /= static_cast<float>(kDim);
			EXPECT_NEAR(outputData[outputIndex], expectedOutput, 1e-6F);
			for (oa::U32 axis = 0; axis < kDim; ++axis) {
				const oa::U32 inputIndex =
					outer * kDim * kInner + axis * kInner + inner;
				EXPECT_NEAR(gradientData[inputIndex],
					upstreamData[outputIndex] / static_cast<float>(kDim), 1e-6F);
			}
		}
	}
	ctx.clear();
}



// ─── Pooling backward ───────────────────────────────────────────────────────

TEST(FnMatrixBackward, MaxPool2dBwd) {
	// input: [1, 1, 4, 4] with distinct values so each 2x2 window has a clear max.
	std::vector<float> input_data = {
		1.0f, 2.0f, 3.0f, 4.0f,
		5.0f, 6.0f, 7.0f, 8.0f,
		9.0f, 10.0f, 11.0f, 12.0f,
		13.0f, 14.0f, 15.0f, 16.0f
	};
	auto input = oa::FnMatrix::fromBytes(oa::Span<const oa::U8>(reinterpret_cast<const oa::U8*>(input_data.data()), input_data.size() * sizeof(float)), oa::MatrixShape{1, 1, 4, 4});

	// kernel=2, stride=2 → output [1,1,2,2]; maxes are at 6,8,14,16 (flat 5,7,13,15).
	auto pool_result = oa::FnMatrix::maxPool2d(input, 2, 2, 0);
	std::vector<float> grad_output_data = {1.0f, 2.0f, 3.0f, 4.0f};
	auto grad_output = oa::FnMatrix::fromBytes(oa::Span<const oa::U8>(reinterpret_cast<const oa::U8*>(grad_output_data.data()), grad_output_data.size() * sizeof(float)), oa::MatrixShape{1, 1, 2, 2});

	auto grad_input = oa::FnMatrix::maxPool2dBwd(input, pool_result.indices, grad_output, 2, 2, 0);
	auto& ctx = oa::ExecutionSession::getActive();
	(void)testSubmitAndWait(ctx);

	const oa::F32* g = grad_input.dataAs<const oa::F32>();
	// Each grad_output value must land on exactly its window's argmax input position.
	const int maxFlat[4] = {5, 7, 13, 15};
	const float expected[4] = {1.0f, 2.0f, 3.0f, 4.0f};
	int nonZero = 0;
	for (int i = 0; i < 16; ++i) if (g[i] != 0.0f) ++nonZero;
	EXPECT_EQ(nonZero, 4) << "gradient must flow to exactly the 4 max positions";
	for (int k = 0; k < 4; ++k)
		EXPECT_NEAR(g[maxFlat[k]], expected[k], 1e-5f)
			<< "grad routed to wrong position / value at window " << k;
}

TEST(FnMatrixBackward, AvgPool2dBwd) {
	auto& ctx = oa::ExecutionSession::getActive();
	
	// input: [1, 1, 4, 4]
	std::vector<float> input_data(16, 1.0f);
	auto input = oa::FnMatrix::fromBytes(oa::Span<const oa::U8>(reinterpret_cast<const oa::U8*>(input_data.data()), input_data.size() * sizeof(float)), oa::MatrixShape{1, 1, 4, 4});
	
	// grad output: [1, 1, 2, 2]
	std::vector<float> grad_output_data = {1.0f, 1.0f, 1.0f, 1.0f};
	auto grad_output = oa::FnMatrix::fromBytes(oa::Span<const oa::U8>(reinterpret_cast<const oa::U8*>(grad_output_data.data()), grad_output_data.size() * sizeof(float)), oa::MatrixShape{1, 1, 2, 2});
	
	// AvgPool2dBwd now requires pooling parameters (kernel=2, stride=2, padding=0)
	auto grad_input = oa::FnMatrix::avgPool2dBwd(input, grad_output, 2, 2, 0);
	[[maybe_unused]] auto exec_result = testSubmitAndWait(ctx);
	
	std::vector<float> result(16);
	[[maybe_unused]] auto copy_result_15 = oa::FnMatrix::copyToHost(grad_input, result.data(), result.size() * sizeof(float));
	
	// Each output gradient should be distributed equally to 4 input positions
	for (size_t i = 0; i < 16; i++) {
		EXPECT_NEAR(result[i], 0.25f, 1e-5f);
	}
}

// ─── Linear backward ────────────────────────────────────────────────────────

TEST(FnMatrixBackward, LinearDataBwd) {
	auto& ctx = oa::ExecutionSession::getActive();
	
	// grad_output: [2, 3], weight: [3, 4]
	std::vector<float> grad_output_data = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
	std::vector<float> weight_data = {
		1.0f, 2.0f, 3.0f, 4.0f,
		5.0f, 6.0f, 7.0f, 8.0f,
		9.0f, 10.0f, 11.0f, 12.0f
	};
	
	auto grad_output = oa::FnMatrix::fromBytes(oa::Span<const oa::U8>(reinterpret_cast<const oa::U8*>(grad_output_data.data()), grad_output_data.size() * sizeof(float)), oa::MatrixShape{2, 3});
	auto weight = oa::FnMatrix::fromBytes(oa::Span<const oa::U8>(reinterpret_cast<const oa::U8*>(weight_data.data()), weight_data.size() * sizeof(float)), oa::MatrixShape{3, 4});
	
	auto grad_input = oa::FnMatrix::linearDataBwd(grad_output, weight);
	[[maybe_unused]] auto exec_result = testSubmitAndWait(ctx);
	
	EXPECT_EQ(grad_input.getShape(), (oa::MatrixShape{2, 4}));
	
	std::vector<float> result(8);
	[[maybe_unused]] auto copy_result_16 = oa::FnMatrix::copyToHost(grad_input, result.data(), result.size() * sizeof(float));
	
	for (size_t i = 0; i < 8; i++) {
		EXPECT_TRUE(std::isfinite(result[i]));
	}
}

TEST(FnMatrixBackward, LinearWeightBwd) {
	auto& ctx = oa::ExecutionSession::getActive();
	
	// input: [2, 4], grad_output: [2, 3]
	std::vector<float> input_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
	std::vector<float> grad_output_data = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
	
	auto input = oa::FnMatrix::fromBytes(oa::Span<const oa::U8>(reinterpret_cast<const oa::U8*>(input_data.data()), input_data.size() * sizeof(float)), oa::MatrixShape{2, 4});
	auto grad_output = oa::FnMatrix::fromBytes(oa::Span<const oa::U8>(reinterpret_cast<const oa::U8*>(grad_output_data.data()), grad_output_data.size() * sizeof(float)), oa::MatrixShape{2, 3});
	
	auto grad_weight = oa::FnMatrix::linearWeightBwd(input, grad_output);
	[[maybe_unused]] auto exec_result = testSubmitAndWait(ctx);
	
	EXPECT_EQ(grad_weight.getShape(), (oa::MatrixShape{3, 4}));
	
	std::vector<float> result(12);
	[[maybe_unused]] auto copy_result_17 = oa::FnMatrix::copyToHost(grad_weight, result.data(), result.size() * sizeof(float));
	
	for (size_t i = 0; i < 12; i++) {
		EXPECT_TRUE(std::isfinite(result[i]));
	}
}

TEST(FnMatrixBackward, LinearWeightBiasBwd) {
	auto& ctx = oa::ExecutionSession::getActive();
	
	// input: [2, 4], grad_output: [2, 3]
	std::vector<float> input_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
	std::vector<float> grad_output_data = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
	
	auto input = oa::FnMatrix::fromBytes(oa::Span<const oa::U8>(reinterpret_cast<const oa::U8*>(input_data.data()), input_data.size() * sizeof(float)), oa::MatrixShape{2, 4});
	auto grad_output = oa::FnMatrix::fromBytes(oa::Span<const oa::U8>(reinterpret_cast<const oa::U8*>(grad_output_data.data()), grad_output_data.size() * sizeof(float)), oa::MatrixShape{2, 3});
	
	auto [grad_weight, grad_bias] = oa::FnMatrix::linearWeightBiasBwd(input, grad_output);
	[[maybe_unused]] auto exec_result = testSubmitAndWait(ctx);
	
	EXPECT_EQ(grad_weight.getShape(), (oa::MatrixShape{3, 4}));
	EXPECT_EQ(grad_bias.getShape(), oa::MatrixShape{3});
	
	std::vector<float> weight_result(12);
	std::vector<float> bias_result(3);
	[[maybe_unused]] auto copy_result_18 = oa::FnMatrix::copyToHost(grad_weight, weight_result.data(), weight_result.size() * sizeof(float));
	[[maybe_unused]] auto copy_result_19 = oa::FnMatrix::copyToHost(grad_bias, bias_result.data(), bias_result.size() * sizeof(float));
	
	for (size_t i = 0; i < 12; i++) {
		EXPECT_TRUE(std::isfinite(weight_result[i]));
	}
	for (size_t i = 0; i < 3; i++) {
		EXPECT_TRUE(std::isfinite(bias_result[i]));
	}
}

TEST(FnMatrixBackward, LinearWeightBiasBwd_LargeShape) {
	auto& ctx = oa::ExecutionSession::getActive();

	// Large shape exercises the scalar LinearWeightBiasBwd kernel.
	constexpr oa::U32 M = 64;
	constexpr oa::U32 N = 64;
	constexpr oa::U32 K = 64;
	
	std::vector<float> input_data(M * K);
	std::vector<float> grad_output_data(M * N);
	for (oa::U32 i = 0; i < M * K; ++i) input_data[i] = static_cast<float>(i % 7) * 0.1F;
	for (oa::U32 i = 0; i < M * N; ++i) grad_output_data[i] = static_cast<float>(i % 5) * 0.2F;
	
	auto input = oa::FnMatrix::fromBytes(oa::Span<const oa::U8>(reinterpret_cast<const oa::U8*>(input_data.data()), input_data.size() * sizeof(float)), oa::MatrixShape{M, K});
	auto grad_output = oa::FnMatrix::fromBytes(oa::Span<const oa::U8>(reinterpret_cast<const oa::U8*>(grad_output_data.data()), grad_output_data.size() * sizeof(float)), oa::MatrixShape{M, N});
	
	auto [grad_weight, grad_bias] = oa::FnMatrix::linearWeightBiasBwd(input, grad_output);
	[[maybe_unused]] auto exec_result = testSubmitAndWait(ctx);
	
	EXPECT_EQ(grad_weight.getShape(), (oa::MatrixShape{N, K}));
	EXPECT_EQ(grad_bias.getShape(), oa::MatrixShape{N});
	
	std::vector<float> weight_result(N * K);
	std::vector<float> bias_result(N);
	[[maybe_unused]] auto copy_weight = oa::FnMatrix::copyToHost(grad_weight, weight_result.data(), weight_result.size() * sizeof(float));
	[[maybe_unused]] auto copy_bias = oa::FnMatrix::copyToHost(grad_bias, bias_result.data(), bias_result.size() * sizeof(float));
	
	// CPU reference: gradWeight[n, k] = sum_m gradOut[m, n] * input[m, k]
	std::vector<float> weight_ref(N * K, 0.0F);
	std::vector<float> bias_ref(N, 0.0F);
	for (oa::U32 n = 0; n < N; ++n) {
		for (oa::U32 k = 0; k < K; ++k) {
			float sum = 0.0F;
			for (oa::U32 m = 0; m < M; ++m) {
				sum += grad_output_data[m * N + n] * input_data[m * K + k];
			}
			weight_ref[n * K + k] = sum;
		}
		float bias_sum = 0.0F;
		for (oa::U32 m = 0; m < M; ++m) {
			bias_sum += grad_output_data[m * N + n];
		}
		bias_ref[n] = bias_sum;
	}
	
	for (oa::U32 i = 0; i < N * K; ++i) {
		EXPECT_NEAR(weight_result[i], weight_ref[i], 0.1F) << "i=" << i;
		EXPECT_TRUE(std::isfinite(weight_result[i]));
	}
	for (oa::U32 i = 0; i < N; ++i) {
		EXPECT_NEAR(bias_result[i], bias_ref[i], 0.1F) << "i=" << i;
		EXPECT_TRUE(std::isfinite(bias_result[i]));
	}
}

TEST(FnMatrixBackward, LinearWeightBiasBwd_Rows32Route) {
	auto& ctx = oa::ExecutionSession::getActive();
	constexpr oa::U32 M = 517, N = 37, K = 35;
	std::vector<float> inputData(M * K);
	std::vector<float> gradOutputData(M * N);
	for (oa::U32 i = 0; i < M * K; ++i)
		inputData[i] = (static_cast<float>(i % 19U) - 9.0F) * 0.015625F;
	for (oa::U32 i = 0; i < M * N; ++i)
		gradOutputData[i] = (static_cast<float>(i % 23U) - 11.0F) * 0.0078125F;

	auto input = oa::FnMatrix::fromBytes(
		oa::Span<const oa::U8>(reinterpret_cast<const oa::U8*>(inputData.data()),
			inputData.size() * sizeof(float)), {M, K});
	auto gradOutput = oa::FnMatrix::fromBytes(
		oa::Span<const oa::U8>(reinterpret_cast<const oa::U8*>(gradOutputData.data()),
			gradOutputData.size() * sizeof(float)), {M, N});
	auto [gradWeight, gradBias] = oa::FnMatrix::linearWeightBiasBwd(input, gradOutput);
	ASSERT_TRUE(testSubmitAndWait(ctx).isOk());

	std::vector<float> weight(N * K), bias(N);
	ASSERT_TRUE(oa::FnMatrix::copyToHost(
		gradWeight, weight.data(), weight.size() * sizeof(float)).isOk());
	ASSERT_TRUE(oa::FnMatrix::copyToHost(
		gradBias, bias.data(), bias.size() * sizeof(float)).isOk());
	for (oa::U32 n = 0; n < N; ++n) {
		float biasRef = 0.0F;
		for (oa::U32 m = 0; m < M; ++m)
			biasRef += gradOutputData[m * N + n];
		EXPECT_NEAR(bias[n], biasRef, 2e-3F) << "bias n=" << n;
		for (oa::U32 k = 0; k < K; ++k) {
			float weightRef = 0.0F;
			for (oa::U32 m = 0; m < M; ++m)
				weightRef += gradOutputData[m * N + n] * inputData[m * K + k];
			EXPECT_NEAR(weight[n * K + k], weightRef, 2e-3F)
				<< "weight n=" << n << " k=" << k;
		}
	}
}

TEST(FnMatrixBackward, LinearWeightBiasBwd_TiledRouteOddShape) {
	auto& ctx = oa::ExecutionSession::getActive();
	constexpr oa::U32 M = 67, N = 97, K = 95;
	std::vector<float> inputData(M * K);
	std::vector<float> gradOutputData(M * N);
	for (oa::U32 i = 0; i < M * K; ++i)
		inputData[i] = (static_cast<float>(i % 17) - 8.0F) * 0.03125F;
	for (oa::U32 i = 0; i < M * N; ++i)
		gradOutputData[i] = (static_cast<float>(i % 13) - 6.0F) * 0.0625F;

	auto input = oa::FnMatrix::fromBytes(
		oa::Span<const oa::U8>(reinterpret_cast<const oa::U8*>(inputData.data()),
			inputData.size() * sizeof(float)), {M, K});
	auto gradOutput = oa::FnMatrix::fromBytes(
		oa::Span<const oa::U8>(reinterpret_cast<const oa::U8*>(gradOutputData.data()),
			gradOutputData.size() * sizeof(float)), {M, N});
	auto [gradWeight, gradBias] = oa::FnMatrix::linearWeightBiasBwd(input, gradOutput);
	ASSERT_TRUE(testSubmitAndWait(ctx).isOk());

	std::vector<float> weight(N * K), bias(N);
	ASSERT_TRUE(oa::FnMatrix::copyToHost(
		gradWeight, weight.data(), weight.size() * sizeof(float)).isOk());
	ASSERT_TRUE(oa::FnMatrix::copyToHost(
		gradBias, bias.data(), bias.size() * sizeof(float)).isOk());
	for (oa::U32 n = 0; n < N; ++n) {
		float biasRef = 0.0F;
		for (oa::U32 m = 0; m < M; ++m)
			biasRef += gradOutputData[m * N + n];
		EXPECT_NEAR(bias[n], biasRef, 2e-4F) << "bias n=" << n;
		for (oa::U32 k = 0; k < K; ++k) {
			float weightRef = 0.0F;
			for (oa::U32 m = 0; m < M; ++m)
				weightRef += gradOutputData[m * N + n] * inputData[m * K + k];
			EXPECT_NEAR(weight[n * K + k], weightRef, 2e-4F)
				<< "weight n=" << n << " k=" << k;
		}
	}
}

// ─── Normalization backward (numerical gradient verification) ────────────────
//
// These replace earlier empty stubs. They run the FULL autograd path
// (forward → MSE → tape.backward) and compare analytical parameter gradients to
// central finite differences. Crucially they use rows>1 and rank-3 [B,T,C]
// inputs: LayerNormBwd/RmsNormBwd write a per-row dw_contrib buffer of size
// rows*cols and the host must column-sum it. An earlier bug allocated that
// buffer as just [cols] (heap overflow for rows>1) and skipped the reduction,
// so dWeight/dBias were wrong for any batch>1 — invisible to rows==1 tests.

static void normForceFp32() { setenv("OA_GEMM_FORCE_FP32", "1", 1); }

static bool normGradClose(oa::F32 inA, oa::F32 inN, oa::F32 inAtol = 2e-3F, oa::F32 inRtol = 3e-2F) {
	return std::abs(inA - inN) <= (inAtol + (inRtol * std::abs(inN)));
}

// run a gradcheck over every element of inParam given a loss-producing closure.
// Returns {numChecked, numFailed, numNonTrivial}.
struct NormGradStats { int checked = 0; int failed = 0; int nonTrivial = 0; };
static NormGradStats normGradCheck(
	oa::ExecutionSession& inCtx, const std::function<oa::F32()>& inLossFunc,
	oa::Matrix& inParam, const oa::F32* inAnalytical, const char* inName)
{
	NormGradStats s;
	oa::F32* data = inParam.dataAs<oa::F32>();
	const oa::I64 n = inParam.numElements();
	const oa::F32 eps = 1e-2f;
	printf("  [%s] %lld elements\n", inName, static_cast<long long>(n));
	for (oa::I64 i = 0; i < n; ++i) {
		const oa::F32 orig = data[i];
		data[i] = orig + eps; (void)testSubmitAndWait(inCtx);
		const oa::F32 lp = inLossFunc();
		data[i] = orig - eps; (void)testSubmitAndWait(inCtx);
		const oa::F32 lm = inLossFunc();
		data[i] = orig; (void)testSubmitAndWait(inCtx);
		const oa::F32 numerical = (lp - lm) / (2.0f * eps);
		const oa::F32 analytical = inAnalytical[i];
		const bool close = normGradClose(analytical, numerical);
		++s.checked;
		if (not close) ++s.failed;
		if (std::abs(numerical) > 5e-4f) ++s.nonTrivial;  // > FD noise → real signal
		if (not close)
			printf("    idx %lld: analytical=%.6f numerical=%.6f  MISMATCH\n",
				static_cast<long long>(i), analytical, numerical);
	}
	return s;
}

TEST(FnMatrixBackward, LayerNormBwd) {
	normForceFp32();
	oa::FnMatrix::setRngSeed(7);
	auto& ctx = oa::ExecutionSession::getActive();
	oa::ExecutionSession::RecordingScope scope(ctx);

	// rank-3 [B,T,C] = [2,3,4] → rows=6, cols=4. Exercises the rows>1 dw_contrib
	// path AND the dBias rank-3 reduction (must sum over B and T, not just B).
	constexpr oa::I32 B = 2, T = 3, C = 4;
	// A deliberately non-default epsilon proves the forward value is retained by
	// the generated autograd attachment instead of backward hard-coding 1e-5.
	constexpr oa::F32 kForwardEps = 0.01F;

	auto x = oa::FnMatrix::scale(oa::FnMatrix::randN(oa::MatrixShape{B, T, C}, oa::ScalarType::Float32), 1.5f);
	auto weight = oa::FnMatrix::scale(oa::FnMatrix::randN(oa::MatrixShape{C}, oa::ScalarType::Float32), 0.5f);
	auto bias   = oa::FnMatrix::scale(oa::FnMatrix::randN(oa::MatrixShape{C}, oa::ScalarType::Float32), 0.5f);
	for (auto* m : {&x, &weight, &bias}) { m->setRequiresGrad(true); }

	auto target = oa::FnMatrix::randN(oa::MatrixShape{B, T, C}, oa::ScalarType::Float32);

	oa::GradientTape tape;
	auto out  = oa::FnMatrix::layerNorm(x, weight, bias, kForwardEps);
	auto loss = oa::FnLoss::mse(out, target);
	tape.backward(loss);
	(void)testSubmitAndWait(ctx);

	auto dWeight = weight.gradMatrix();
	auto dBias   = bias.gradMatrix();
	// Shapes must match the params (not [T,C] from a partial reduction).
	EXPECT_EQ(dWeight.numElements(), C);
	EXPECT_EQ(dBias.numElements(), C);

	auto lossFunc = [&]() -> oa::F32 {
		oa::GradNo noGrad;
		auto o = oa::FnMatrix::layerNorm(x, weight, bias, kForwardEps);
		auto l = oa::FnLoss::mse(o, target);
		(void)testSubmitAndWait(ctx);
		return l.dataAs<const oa::F32>()[0];
	};

	printf("\nLayerNorm [B=%d,T=%d,C=%d] gradcheck:\n", B, T, C);
	auto sw = normGradCheck(ctx, lossFunc, weight, dWeight.dataAs<const oa::F32>(), "weight");
	auto sb = normGradCheck(ctx, lossFunc, bias,   dBias.dataAs<const oa::F32>(),   "bias");
	const int failed = sw.failed + sb.failed;
	const int nonTrivial = sw.nonTrivial + sb.nonTrivial;
	printf("LayerNorm gradcheck: %d/%d pass, %d non-trivial\n",
		(sw.checked + sb.checked) - failed, sw.checked + sb.checked, nonTrivial);
	EXPECT_EQ(failed, 0) << "LayerNorm weight/bias gradient mismatch";
	EXPECT_GE(nonTrivial, 3) << "gradients all ~0 — vacuous check";
}

TEST(FnMatrixBackward, RmsNormBwd) {
	normForceFp32();
	oa::FnMatrix::setRngSeed(11);
	auto& ctx = oa::ExecutionSession::getActive();
	oa::ExecutionSession::RecordingScope scope(ctx);

	// rank-3 [B,T,C] = [2,3,4] → rows=6, cols=4 (rows>1 dw_contrib path).
	constexpr oa::I32 B = 2, T = 3, C = 4;
	// This non-default value catches any forward/backward epsilon mismatch.
	constexpr oa::F32 kForwardEps = 0.01F;

	auto x = oa::FnMatrix::scale(oa::FnMatrix::randN(oa::MatrixShape{B, T, C}, oa::ScalarType::Float32), 1.5f);
	auto weight = oa::FnMatrix::scale(oa::FnMatrix::randN(oa::MatrixShape{C}, oa::ScalarType::Float32), 0.5f);
	x.setRequiresGrad(true);
	weight.setRequiresGrad(true);

	auto target = oa::FnMatrix::randN(oa::MatrixShape{B, T, C}, oa::ScalarType::Float32);

	oa::GradientTape tape;
	auto out  = oa::FnMatrix::rmsNorm(x, weight, kForwardEps);
	auto loss = oa::FnLoss::mse(out, target);
	tape.backward(loss);
	(void)testSubmitAndWait(ctx);

	auto dWeight = weight.gradMatrix();
	EXPECT_EQ(dWeight.numElements(), C);

	auto lossFunc = [&]() -> oa::F32 {
		oa::GradNo noGrad;
		auto o = oa::FnMatrix::rmsNorm(x, weight, kForwardEps);
		auto l = oa::FnLoss::mse(o, target);
		(void)testSubmitAndWait(ctx);
		return l.dataAs<const oa::F32>()[0];
	};

	printf("\nRmsNorm [B=%d,T=%d,C=%d] gradcheck:\n", B, T, C);
	auto sw = normGradCheck(ctx, lossFunc, weight, dWeight.dataAs<const oa::F32>(), "weight");
	printf("RmsNorm gradcheck: %d/%d pass, %d non-trivial\n",
		sw.checked - sw.failed, sw.checked, sw.nonTrivial);
	EXPECT_EQ(sw.failed, 0) << "RmsNorm weight gradient mismatch";
	EXPECT_GE(sw.nonTrivial, 3) << "gradients all ~0 — vacuous check";
}

TEST(FnMatrixBackward, ResidualRmsNormMatchesUnfusedAutograd) {
	auto& ctx = oa::ExecutionSession::getActive();
	oa::ExecutionSession::RecordingScope scope(ctx);
	const oa::MatrixShape shape{2, 3};
	const oa::MatrixShape weightShape{3};
	constexpr oa::F32 eps = 0.01F;

	auto fusedA = matrixFromHost(
		{0.2F, -0.4F, 0.7F, 1.1F, -0.8F, 0.3F}, shape);
	auto fusedB = matrixFromHost(
		{-0.1F, 0.6F, 0.2F, -0.5F, 0.9F, 0.4F}, shape);
	auto fusedWeight = matrixFromHost({0.8F, 1.2F, -0.7F}, weightShape);
	for (auto* matrix : {&fusedA, &fusedB, &fusedWeight}) {
		matrix->setRequiresGrad(true);
	}

	oa::GradientTape fusedTape;
	auto fused = oa::FnMatrix::residualRmsNorm(
		fusedA, fusedB, fusedWeight, eps);
	auto fusedLoss = oa::FnMatrix::sum(
		oa::FnMatrix::add(fused.out, fused.residual), -1);
	fusedTape.backward(fusedLoss);
	ASSERT_TRUE(testSubmitAndWait(ctx).isOk());

	const auto fusedDA = matrixToHost(fusedA.gradMatrix());
	const auto fusedDB = matrixToHost(fusedB.gradMatrix());
	const auto fusedDW = matrixToHost(fusedWeight.gradMatrix());

	auto referenceA = matrixFromHost(
		{0.2F, -0.4F, 0.7F, 1.1F, -0.8F, 0.3F}, shape);
	auto referenceB = matrixFromHost(
		{-0.1F, 0.6F, 0.2F, -0.5F, 0.9F, 0.4F}, shape);
	auto referenceWeight = matrixFromHost(
		{0.8F, 1.2F, -0.7F}, weightShape);
	for (auto* matrix : {&referenceA, &referenceB, &referenceWeight}) {
		matrix->setRequiresGrad(true);
	}

	oa::GradientTape referenceTape;
	auto referenceResidual = oa::FnMatrix::add(referenceA, referenceB);
	auto referenceOut = oa::FnMatrix::rmsNorm(
		referenceResidual, referenceWeight, eps);
	auto referenceLoss = oa::FnMatrix::sum(
		oa::FnMatrix::add(referenceOut, referenceResidual), -1);
	referenceTape.backward(referenceLoss);
	ASSERT_TRUE(testSubmitAndWait(ctx).isOk());

	const auto expectNear = [](const auto& actual, const auto& expected) {
		ASSERT_EQ(actual.size(), expected.size());
		for (oa::Usize index = 0; index < actual.size(); ++index) {
			EXPECT_NEAR(actual[index], expected[index], 2e-4F)
				<< "gradient mismatch at index " << index;
		}
	};
	expectNear(fusedDA, matrixToHost(referenceA.gradMatrix()));
	expectNear(fusedDB, matrixToHost(referenceB.gradMatrix()));
	expectNear(fusedDW, matrixToHost(referenceWeight.gradMatrix()));
}

// ─── Embedding backward ─────────────────────────────────────────────────────

TEST(FnMatrixBackward, GatherBwd) {
	auto& ctx = oa::ExecutionSession::getActive();

	// Embedding backward = scatter-add of the upstream gradient rows into the
	// table rows named by the (integer) gather indices. indices {0,2,1} route
	// grad row 0 → table[0], grad row 1 → table[2], grad row 2 → table[1];
	// table rows 3,4 are never named and must stay zero.
	constexpr oa::I32 kVocab = 5, kEmbed = 4;
	std::vector<oa::U32> indices_data = {0, 2, 1};
	std::vector<float> grad_output_data = {
		1.0f, 1.0f, 1.0f, 1.0f,   // → table[0]
		2.0f, 2.0f, 2.0f, 2.0f,   // → table[2]
		3.0f, 3.0f, 3.0f, 3.0f,   // → table[1]
	};

	auto indices = oa::FnMatrix::fromBytes(
		oa::Span<const oa::U8>(reinterpret_cast<const oa::U8*>(indices_data.data()),
			indices_data.size() * sizeof(oa::U32)),
		oa::MatrixShape{3}, oa::ScalarType::UInt32);
	auto grad_output = oa::FnMatrix::fromBytes(
		oa::Span<const oa::U8>(reinterpret_cast<const oa::U8*>(grad_output_data.data()),
			grad_output_data.size() * sizeof(float)),
		oa::MatrixShape{3, 4});

	auto grad_table = oa::FnMatrix::gatherBwd(indices, grad_output, kVocab, kEmbed);
	ASSERT_TRUE(testSubmitAndWait(ctx).isOk());

	EXPECT_EQ(grad_table.getShape()[0], kVocab);
	EXPECT_EQ(grad_table.getShape()[1], kEmbed);

	std::vector<float> result(kVocab * kEmbed);
	ASSERT_TRUE(oa::FnMatrix::copyToHost(grad_table, result.data(),
		result.size() * sizeof(float)).isOk());

	const float expected[kVocab][kEmbed] = {
		{1, 1, 1, 1},   // table[0] ← grad row 0
		{3, 3, 3, 3},   // table[1] ← grad row 2
		{2, 2, 2, 2},   // table[2] ← grad row 1
		{0, 0, 0, 0},   // untouched
		{0, 0, 0, 0},   // untouched
	};
	for (int v = 0; v < kVocab; ++v)
		for (int e = 0; e < kEmbed; ++e)
			EXPECT_NEAR(result[v * kEmbed + e], expected[v][e], 1e-5f)
				<< "table[" << v << "][" << e << "]";
}

// ─── Complex backward operations ────────────────────────────────────────────

TEST(FnMatrixBackward, SwigluBwd) {
	auto& ctx = oa::ExecutionSession::getActive();
	
	std::vector<float> gate_data = {1.0f, 2.0f, 3.0f, 4.0f};
	std::vector<float> up_data = {5.0f, 6.0f, 7.0f, 8.0f};
	std::vector<float> out_data = {1.0f, 2.0f, 3.0f, 4.0f};
	std::vector<float> grad_output_data = {1.0f, 1.0f, 1.0f, 1.0f};
	
	auto gate = oa::FnMatrix::fromBytes(oa::Span<const oa::U8>(reinterpret_cast<const oa::U8*>(gate_data.data()), gate_data.size() * sizeof(float)), oa::MatrixShape{4});
	auto up = oa::FnMatrix::fromBytes(oa::Span<const oa::U8>(reinterpret_cast<const oa::U8*>(up_data.data()), up_data.size() * sizeof(float)), oa::MatrixShape{4});
	auto out = oa::FnMatrix::fromBytes(oa::Span<const oa::U8>(reinterpret_cast<const oa::U8*>(out_data.data()), out_data.size() * sizeof(float)), oa::MatrixShape{4});
	auto grad_output = oa::FnMatrix::fromBytes(oa::Span<const oa::U8>(reinterpret_cast<const oa::U8*>(grad_output_data.data()), grad_output_data.size() * sizeof(float)), oa::MatrixShape{4});
	
	auto [gate_grad, up_grad] = oa::FnMatrix::swigluBwd(gate, up, out, grad_output);
	[[maybe_unused]] auto exec_result = testSubmitAndWait(ctx);
	
	EXPECT_EQ(gate_grad.getShape(), oa::MatrixShape{4});
	EXPECT_EQ(up_grad.getShape(), oa::MatrixShape{4});
	
	std::vector<float> gate_result(4);
	std::vector<float> up_result(4);
	[[maybe_unused]] auto copy_result_24 = oa::FnMatrix::copyToHost(gate_grad, gate_result.data(), gate_result.size() * sizeof(float));
	[[maybe_unused]] auto copy_result_25 = oa::FnMatrix::copyToHost(up_grad, up_result.data(), up_result.size() * sizeof(float));
	
	for (size_t i = 0; i < 4; i++) {
		EXPECT_TRUE(std::isfinite(gate_result[i]));
		EXPECT_TRUE(std::isfinite(up_result[i]));
	}
}

// Real finite-difference gradcheck of SiluMulBwd against the forward SiluMul.
// The forward (flat split) computes y[i] = siLU(x[i]) * x[i+N] for i in [0,N), with
// input x = [gate(N) ; up(N)]. The backward takes the INPUT (not the un-invertible
// output) and must match d/dx of loss = sum_i dY[i]*y[i]. Was a vacuous isfinite
// check; the kernel didn't even exist (not compiled / not registered) until now.
TEST(FnMatrixBackward, SiluMulBwd) {
	auto& ctx = oa::ExecutionSession::getActive();
	const oa::U32 N = 8;            // intermediate size
	const oa::U32 M = 2 * N;        // input length (gate||up)

	std::vector<float> x = {-1.5f, -0.5f, 0.25f, 1.0f, 2.0f, -2.0f, 0.75f, -0.1f,
	                         0.4f,  1.3f, -0.8f, 0.6f, -1.2f, 0.9f, 2.5f, -0.3f};  // M=16
	std::vector<float> dY(N);
	for (oa::U32 i = 0; i < N; ++i) dY[i] = 0.5f + 0.1f * static_cast<float>(i);

	auto upload = [](const std::vector<float>& v) {
		return oa::FnMatrix::fromBytes(oa::Span<const oa::U8>(reinterpret_cast<const oa::U8*>(v.data()),
			v.size() * sizeof(float)), oa::MatrixShape{static_cast<oa::I64>(v.size())});
	};

	// loss(x) = sum_{i<N} dY[i] * siluMul(x)[i]
	auto lossOf = [&](const std::vector<float>& xv) -> double {
		oa::GradNo noGrad;
		auto xm = upload(xv);
		auto y = oa::FnMatrix::siluMul(xm, N);
		(void)testSubmitAndWait(ctx);
		std::vector<float> yh(N);
		EXPECT_TRUE(oa::FnMatrix::copyToHost(y, yh.data(), N * sizeof(float)).isOk());
		double s = 0.0;
		for (oa::U32 i = 0; i < N; ++i) s += static_cast<double>(dY[i]) * static_cast<double>(yh[i]);
		return s;
	};

	// Analytic gradient via the kernel under test.
	auto xm = upload(x);
	auto dYm = upload(dY);
	auto gradInput = oa::FnMatrix::siluMulBwd(xm, dYm);
	(void)testSubmitAndWait(ctx);
	std::vector<float> analytic(M);
	(void)oa::FnMatrix::copyToHost(gradInput, analytic.data(), M * sizeof(float));

	// Central differences vs analytic for every input element.
	const float eps = 1e-3f;
	for (oa::U32 j = 0; j < M; ++j) {
		std::vector<float> xp = x, xn = x;
		xp[j] += eps; xn[j] -= eps;
		double num = (lossOf(xp) - lossOf(xn)) / (2.0 * static_cast<double>(eps));
		EXPECT_NEAR(static_cast<double>(analytic[j]), num,
			2e-2 + 2e-2 * std::abs(num)) << "grad mismatch at input index " << j;
	}
}

// Real finite-difference gradcheck of GegluBwd against the forward Geglu.
// forward (flat split): y[i] = x[i] * GELU(x[i+N]) for i in [0,N), input x =
// [up(N) ; gate(N)]. backward takes the INPUT and must match d/dx of
// loss = sum_i dY[i]*y[i]. Was vacuous isfinite; kernel didn't exist until now.
TEST(FnMatrixBackward, GegluBwd) {
	auto& ctx = oa::ExecutionSession::getActive();
	const oa::U32 N = 8;
	const oa::U32 M = 2 * N;

	std::vector<float> x = {-1.5f, -0.5f, 0.25f, 1.0f, 2.0f, -2.0f, 0.75f, -0.1f,
	                         0.4f,  1.3f, -0.8f, 0.6f, -1.2f, 0.9f, 2.5f, -0.3f};
	std::vector<float> dY(N);
	for (oa::U32 i = 0; i < N; ++i) dY[i] = 0.5f + 0.1f * static_cast<float>(i);

	auto upload = [](const std::vector<float>& v) {
		return oa::FnMatrix::fromBytes(oa::Span<const oa::U8>(reinterpret_cast<const oa::U8*>(v.data()),
			v.size() * sizeof(float)), oa::MatrixShape{static_cast<oa::I64>(v.size())});
	};

	auto lossOf = [&](const std::vector<float>& xv) -> double {
		oa::GradNo noGrad;
		auto xm = upload(xv);
		auto y = oa::FnMatrix::geglu(xm, N);
		(void)testSubmitAndWait(ctx);
		std::vector<float> yh(N);
		EXPECT_TRUE(oa::FnMatrix::copyToHost(y, yh.data(), N * sizeof(float)).isOk());
		double s = 0.0;
		for (oa::U32 i = 0; i < N; ++i) s += static_cast<double>(dY[i]) * static_cast<double>(yh[i]);
		return s;
	};

	auto xm = upload(x);
	auto dYm = upload(dY);
	auto gradInput = oa::FnMatrix::gegluBwd(xm, dYm);
	(void)testSubmitAndWait(ctx);
	std::vector<float> analytic(M);
	(void)oa::FnMatrix::copyToHost(gradInput, analytic.data(), M * sizeof(float));

	const float eps = 1e-3f;
	for (oa::U32 j = 0; j < M; ++j) {
		std::vector<float> xp = x, xn = x;
		xp[j] += eps; xn[j] -= eps;
		double num = (lossOf(xp) - lossOf(xn)) / (2.0 * static_cast<double>(eps));
		EXPECT_NEAR(static_cast<double>(analytic[j]), num,
			2e-2 + 2e-2 * std::abs(num)) << "grad mismatch at input index " << j;
	}
}

// Full-reduction Max backward: grad routes only to the element equal to the max.
// (Previously this test merely checked isfinite — vacuous; the kernel was in fact
// reading buffers/push fields the host never supplied. The SPIR-V push-block
// assert in Record now makes that class of mismatch impossible.)
//
// The earlier "fails in-suite, passes standalone" flakiness had TWO independent
// root causes, both fixed: (1) a real bug in the forward Max.slang cross-wave
// reduction (it folded only wave-leaders 0,1,2,4 and dropped 3,5,6,7 — global max
// lost when it landed in a dropped wave; rewritten as a subgroup-size-independent
// tree reduction mirroring Sum); and (2) the preceding SiluMulBwd test recorded
// an op with no shader, whose failed compile permanently bricked the shared default
// context so every later submission returned zeros. oa::ExecutionSession now resets the
// graph on compile/replay failure so one bad kernel can't poison the context.
TEST(FnMatrixBackward, MaxBwd) {
	auto& ctx = oa::ExecutionSession::getActive();

	std::vector<float> input_data = {1.0f, 4.0f, 3.0f, 2.0f};  // max is index 1
	auto input = oa::FnMatrix::fromBytes(oa::Span<const oa::U8>(reinterpret_cast<const oa::U8*>(input_data.data()), input_data.size() * sizeof(float)), oa::MatrixShape{4});

	std::vector<float> grad_data = {2.5f};  // upstream scalar grad
	auto gradOut = oa::FnMatrix::fromBytes(oa::Span<const oa::U8>(reinterpret_cast<const oa::U8*>(grad_data.data()), grad_data.size() * sizeof(float)), oa::MatrixShape{1});

	auto maxVal = oa::FnMatrix::max(input);
	// Materialize the forward max before the value-API backward reads it (OA is
	// deferred; MaxBwd consumes maxVal's contents, so it must be flushed first).
	[[maybe_unused]] auto e0 = testSubmitAndWait(ctx);
	auto gradInput = oa::FnMatrix::maxBwd(input, maxVal, gradOut);
	[[maybe_unused]] auto e = testSubmitAndWait(ctx);

	std::vector<float> result(4);
	[[maybe_unused]] auto c = oa::FnMatrix::copyToHost(gradInput, result.data(), result.size() * sizeof(float));

	// Only the argmax (index 1) receives the gradient; everything else is 0.
	EXPECT_NEAR(result[0], 0.0f, 1e-6f);
	EXPECT_NEAR(result[1], 2.5f, 1e-6f);
	EXPECT_NEAR(result[2], 0.0f, 1e-6f);
	EXPECT_NEAR(result[3], 0.0f, 1e-6f);
}

// Max is now differentiable end-to-end: tape backward must match the analytic
// scatter (grad to the max element only). Re-enabled with the Max.slang
// reduction fix (see MaxBwd note above).
TEST(FnMatrixBackward, MaxAutogradTape) {
	auto& ctx = oa::ExecutionSession::getActive();
	oa::ExecutionSession::RecordingScope scope(ctx);

	std::vector<float> x = {-1.0f, 0.5f, 3.0f, 2.0f, -4.0f};  // max at index 2
	auto input = oa::FnMatrix::fromBytes(oa::Span<const oa::U8>(reinterpret_cast<const oa::U8*>(x.data()), x.size() * sizeof(float)), oa::MatrixShape{5});
	input.setRequiresGrad(true);

	oa::GradientTape tape;
	auto m = oa::FnMatrix::max(input);
	auto loss = oa::FnMatrix::scale(m, 3.0f);  // d loss / d max = 3
	(void)testSubmitAndWait(ctx);   // materialize forward before backward
	tape.backward(loss);
	(void)testSubmitAndWait(ctx);

	auto g = input.gradMatrix();
	std::vector<float> grad(5);
	[[maybe_unused]] auto c = oa::FnMatrix::copyToHost(g, grad.data(), grad.size() * sizeof(float));

	for (size_t i = 0; i < 5; ++i)
		EXPECT_NEAR(grad[i], (i == 2) ? 3.0f : 0.0f, 1e-6f) << "i=" << i;
}

// ─── Vision backward ────────────────────────────────────────────────────────

TEST(FnMatrixBackward, UpsampleBwd_Nearest) {
	auto& ctx = oa::ExecutionSession::getActive();
	
	// input: [1, 1, 2, 2]
	std::vector<float> input_data = {1.0f, 2.0f, 3.0f, 4.0f};
	auto input = oa::FnMatrix::fromBytes(oa::Span<const oa::U8>(reinterpret_cast<const oa::U8*>(input_data.data()), input_data.size() * sizeof(float)), oa::MatrixShape{1, 1, 2, 2});
	
	// grad output: [1, 1, 4, 4] (upsampled 2x)
	std::vector<float> grad_output_data(16, 1.0f);
	auto grad_output = oa::FnMatrix::fromBytes(oa::Span<const oa::U8>(reinterpret_cast<const oa::U8*>(grad_output_data.data()), grad_output_data.size() * sizeof(float)), oa::MatrixShape{1, 1, 4, 4});
	
	auto grad_input = oa::FnMatrix::upsampleBwd(input, grad_output, 2, false);
	[[maybe_unused]] auto exec_result = testSubmitAndWait(ctx);
	
	EXPECT_EQ(grad_input.getShape(), (oa::MatrixShape{1, 1, 2, 2}));
	
	std::vector<float> result(4);
	[[maybe_unused]] auto copy_result_29 = oa::FnMatrix::copyToHost(grad_input, result.data(), result.size() * sizeof(float));
	
	// Each input pixel should accumulate gradients from 4 output pixels
	for (size_t i = 0; i < 4; i++) {
		EXPECT_NEAR(result[i], 4.0f, 1e-5f);
	}
}

TEST(FnMatrixBackward, UpsampleBwd_Bilinear) {
	auto& ctx = oa::ExecutionSession::getActive();
	
	// input: [1, 1, 2, 2]
	std::vector<float> input_data = {1.0f, 2.0f, 3.0f, 4.0f};
	auto input = oa::FnMatrix::fromBytes(oa::Span<const oa::U8>(reinterpret_cast<const oa::U8*>(input_data.data()), input_data.size() * sizeof(float)), oa::MatrixShape{1, 1, 2, 2});
	
	// grad output: [1, 1, 4, 4] (upsampled 2x)
	std::vector<float> grad_output_data(16, 1.0f);
	auto grad_output = oa::FnMatrix::fromBytes(oa::Span<const oa::U8>(reinterpret_cast<const oa::U8*>(grad_output_data.data()), grad_output_data.size() * sizeof(float)), oa::MatrixShape{1, 1, 4, 4});
	
	auto grad_input = oa::FnMatrix::upsampleBwd(input, grad_output, 2, true);
	[[maybe_unused]] auto exec_result = testSubmitAndWait(ctx);
	
	EXPECT_EQ(grad_input.getShape(), (oa::MatrixShape{1, 1, 2, 2}));
	
	std::vector<float> result(4);
	[[maybe_unused]] auto copy_result_30 = oa::FnMatrix::copyToHost(grad_input, result.data(), result.size() * sizeof(float));
	
	for (size_t i = 0; i < 4; i++) {
		EXPECT_TRUE(std::isfinite(result[i]));
	}
}
