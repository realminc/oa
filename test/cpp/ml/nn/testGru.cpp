// ═══════════════════════════════════════════════════════════════════════════
// OA Test — GRU (Gated Recurrent Unit) comprehensive testing
// ═══════════════════════════════════════════════════════════════════════════

#include "../../oaTest.h"
#include <oa/ml.h>
#include <oa/ml/nn.h>
#include <oa/ml/autograd.h>
#include <oa/runtime/engine.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <string>
#include <vector>

// ─── Helper Functions ────────────────────────────────────────────────────────

static oa::F32 computeRelativeError(const oa::Matrix& inA, const oa::Matrix& inB) {
	OA_ASSERT(inA.getShape() == inB.getShape());
	auto& ctx = oa::ExecutionSession::getActive();
	auto diff = oa::FnMatrix::sub(inA, inB);
	auto diffSq = oa::FnMatrix::mul(diff, diff);
	auto diffNorm = oa::FnMatrix::sqrt(oa::FnMatrix::sum(diffSq));
	auto aSq = oa::FnMatrix::mul(inA, inA);
	auto aNorm = oa::FnMatrix::sqrt(oa::FnMatrix::sum(aSq));
	(void)testSubmitAndWait(ctx);
	oa::F32 diffVal = diffNorm.item();
	oa::F32 aVal = aNorm.item();
	return diffVal / (aVal + 1e-8f);
}

[[maybe_unused]] static void syncCtx() {
	auto& ctx = oa::ExecutionSession::getActive();
	(void)testSubmitAndWait(ctx);
}

// Fill a matrix in place from a deterministic, bounded host pattern (keeps the
// GRU recurrent dynamics stable so finite-difference grad-check is meaningful).
static void fillDeterministic(oa::Matrix& m, float scale, double phase) {
	std::vector<float> v(static_cast<size_t>(m.numElements()));
	for (size_t i = 0; i < v.size(); ++i) {
		v[i] = scale * static_cast<float>(std::sin(0.37 * static_cast<double>(i) + phase));
	}
	m = oa::FnMatrix::fromBytes(
		oa::Span<const oa::U8>(reinterpret_cast<const oa::U8*>(v.data()), v.size() * sizeof(float)),
		m.getShape());
}

// Bulk host download of a (synced) float32 matrix.
static std::vector<float> downloadF32(const oa::Matrix& m) {
	syncCtx();
	const float* p = m.dataAs<const float>();
	return std::vector<float>(p, p + m.numElements());
}

// ─── Test Cases ──────────────────────────────────────────────────────────────

TEST(Gru, GruCellConstruction) {
	auto& ctx = oa::ExecutionSession::getActive();
	
	// Test basic construction
	oa::GruCell cell1(10, 20, true);
	EXPECT_EQ(cell1.inputSize(), 10);
	EXPECT_EQ(cell1.hiddenSize(), 20);
	EXPECT_EQ(cell1.parameters().size(), 4);  // weight_ih, weight_hh, bias_ih, bias_hh
	
	// Test without bias
	oa::GruCell cell2(10, 20, false);
	EXPECT_EQ(cell2.parameters().size(), 2);  // weight_ih, weight_hh only
	
	// verify parameter shapes
	const auto& params = cell1.parameters();
	EXPECT_EQ(params[0].data.getShape(), (oa::MatrixShape{3 * 20, 10}));  // weight_ih
	EXPECT_EQ(params[1].data.getShape(), (oa::MatrixShape{3 * 20, 20}));  // weight_hh
	EXPECT_EQ(params[2].data.getShape(), oa::MatrixShape{3 * 20});      // bias_ih
	EXPECT_EQ(params[3].data.getShape(), oa::MatrixShape{3 * 20});      // bias_hh
}

TEST(Gru, GruCellZeroState) {
	auto& ctx = oa::ExecutionSession::getActive();
	oa::GruCell cell(10, 20);
	
	auto h0 = cell.zeroState(5);
	EXPECT_EQ(h0.getShape(), (oa::MatrixShape{5, 20}));
	
	// verify it's actually zeros
	oa::F32 sum = oa::FnMatrix::sum(h0).item();
	EXPECT_NEAR(sum, 0.0f, 1e-6f);
}

TEST(Gru, GruCellForwardSingleStep) {
	auto& ctx = oa::ExecutionSession::getActive();
	oa::GruCell cell(10, 20);
	
	// Create input: [batch=2, input_size=10]
	auto input = oa::FnMatrix::randN(oa::MatrixShape{2, 10}, oa::ScalarType::Float32);
	auto hidden = cell.zeroState(2);
	
	// Single step
	auto output = cell.step(input, hidden);
	
	// execute and sync to get actual results
	(void)testSubmitAndWait(ctx);
	
	EXPECT_EQ(output.getShape(), (oa::MatrixShape{2, 20}));
	
	// output should be different from zero state
	auto diff = oa::FnMatrix::sub(output, hidden);
	auto diffSq = oa::FnMatrix::mul(diff, diff);
	auto diffNorm = oa::FnMatrix::sqrt(oa::FnMatrix::sum(diffSq));
	(void)testSubmitAndWait(ctx);
	oa::F32 diffNormVal = diffNorm.item();
	EXPECT_GT(diffNormVal, 1e-3f);
}

TEST(Gru, GruCellForwardWithoutHidden) {
	auto& ctx = oa::ExecutionSession::getActive();
	oa::GruCell cell(10, 20);
	
	auto input = oa::FnMatrix::randN(oa::MatrixShape{3, 10}, oa::ScalarType::Float32);
	auto output = cell.forward(input);
	
	EXPECT_EQ(output.getShape(), (oa::MatrixShape{3, 20}));
}

TEST(Gru, GruCellOutputRange) {
	auto& ctx = oa::ExecutionSession::getActive();
	oa::GruCell cell(10, 20);
	
	auto input = oa::FnMatrix::randN(oa::MatrixShape{2, 10}, oa::ScalarType::Float32);
	auto hidden = cell.zeroState(2);
	auto output = cell.step(input, hidden);
	
	// GRU output should be bounded by tanh activation (approximately [-1, 1])
	auto absOutput = oa::FnMatrix::abs(output);
	oa::F32 maxVal = oa::FnMatrix::max(absOutput).item();
	EXPECT_LT(maxVal, 2.0f);  // Allow some margin
}

TEST(Gru, GruMultiLayerConstruction) {
	auto& ctx = oa::ExecutionSession::getActive();
	
	oa::Gru gru(10, 20, 3, true);
	EXPECT_EQ(gru.inputSize(), 10);
	EXPECT_EQ(gru.hiddenSize(), 20);
	EXPECT_EQ(gru.numLayers(), 3);
	
	// Should have 3 layers registered as children
	EXPECT_EQ(gru.children().size(), 3);
}

TEST(Gru, GruForwardSequence) {
	auto& ctx = oa::ExecutionSession::getActive();
	oa::Gru gru(10, 20, 2);
	
	// input: [batch=2, seq_len=5, input_size=10]
	auto input = oa::FnMatrix::randN(oa::MatrixShape{2, 5, 10}, oa::ScalarType::Float32);
	auto output = gru.forward(input);
	
	// output: [batch=2, seq_len=5, hidden_size=20]
	EXPECT_EQ(output.getShape(), (oa::MatrixShape{2, 5, 20}));
}

TEST(Gru, GruSingleLayerSequence) {
	auto& ctx = oa::ExecutionSession::getActive();
	oa::Gru gru(10, 20, 1);
	
	auto input = oa::FnMatrix::randN(oa::MatrixShape{3, 7, 10}, oa::ScalarType::Float32);
	auto output = gru.forward(input);
	
	EXPECT_EQ(output.getShape(), (oa::MatrixShape{3, 7, 20}));
}

TEST(Gru, DecomposedForwardIsCausal) {
	// Mobile drivers use the decomposed route when the whole-sequence backward
	// shader exceeds their compiler/resource limits. Two inputs with an identical
	// prefix must produce identical prefix outputs; otherwise teacher-forced loss
	// can look excellent while autoregressive generation reads future padding.
	const char* previous = std::getenv("OA_DISABLE_GRU_SCAN");
	const std::string previousValue = previous != nullptr ? previous : "";
	setenv("OA_DISABLE_GRU_SCAN", "1", 1);

	constexpr oa::I32 batch = 1;
	constexpr oa::I32 sequence = 8;
	constexpr oa::I32 inputSize = 5;
	constexpr oa::I32 hiddenSize = 7;
	constexpr oa::I32 prefix = 4;
	oa::Gru gru(inputSize, hiddenSize, 1);

	std::vector<float> a(batch * sequence * inputSize);
	std::vector<float> b(a.size());
	for (oa::Usize index = 0; index < a.size(); ++index) {
		a[index] = 0.15F * static_cast<float>(
			std::sin(0.31 * static_cast<double>(index) + 0.2));
		b[index] = a[index];
	}
	for (oa::I32 time = prefix; time < sequence; ++time) {
		for (oa::I32 feature = 0; feature < inputSize; ++feature) {
			const oa::Usize index = static_cast<oa::Usize>(time * inputSize + feature);
			b[index] = -0.23F * static_cast<float>(
				std::cos(0.19 * static_cast<double>(index) + 0.4));
		}
	}

	auto inputA = oa::FnMatrix::fromBytes(
		oa::Span<const oa::U8>(reinterpret_cast<const oa::U8*>(a.data()),
			a.size() * sizeof(float)),
		oa::MatrixShape{batch, sequence, inputSize});
	auto outputA = gru.forward(inputA);
	const auto valuesA = downloadF32(outputA);
	auto inputB = oa::FnMatrix::fromBytes(
		oa::Span<const oa::U8>(reinterpret_cast<const oa::U8*>(b.data()),
			b.size() * sizeof(float)),
		oa::MatrixShape{batch, sequence, inputSize});
	auto outputB = gru.forward(inputB);
	const auto valuesB = downloadF32(outputB);

	for (oa::I32 time = 0; time < prefix; ++time) {
		for (oa::I32 hidden = 0; hidden < hiddenSize; ++hidden) {
			const oa::Usize index = static_cast<oa::Usize>(time * hiddenSize + hidden);
			EXPECT_NEAR(valuesA[index], valuesB[index], 1e-6F)
				<< "future suffix changed decomposed GRU output at time " << time
				<< ", hidden " << hidden;
		}
	}

	if (previous != nullptr) {
		setenv("OA_DISABLE_GRU_SCAN", previousValue.c_str(), 1);
	} else {
		unsetenv("OA_DISABLE_GRU_SCAN");
	}
}

TEST(Gru, GruStepByStep) {
	auto& ctx = oa::ExecutionSession::getActive();
	oa::Gru gru(10, 20, 1);
	
	const oa::I32 batch = 2;
	const oa::I32 seqLen = 5;
	
	// Manual step-by-step processing
	auto hidden = gru.zeroState(batch, 0);
	oa::Vec<oa::Matrix> outputs;
	
	for (oa::I32 t = 0; t < seqLen; ++t) {
		auto xt = oa::FnMatrix::randN(oa::MatrixShape{batch, 10}, oa::ScalarType::Float32);
		hidden = gru.step(xt, hidden, 0);
		outputs.pushBack(hidden);
	}
	
	EXPECT_EQ(outputs.size(), seqLen);
	EXPECT_EQ(outputs[0].getShape(), (oa::MatrixShape{batch, 20}));
}

TEST(Gru, GruBatchSizeVariation) {
	auto& ctx = oa::ExecutionSession::getActive();
	oa::Gru gru(10, 20, 2);
	
	// Test different batch sizes
	for (oa::I32 batch : {1, 2, 4, 8, 16}) {
		auto input = oa::FnMatrix::randN(oa::MatrixShape{batch, 3, 10}, oa::ScalarType::Float32);
		auto output = gru.forward(input);
		EXPECT_EQ(output.getShape(), (oa::MatrixShape{batch, 3, 20}));
	}
}

TEST(Gru, GruSequenceLengthVariation) {
	auto& ctx = oa::ExecutionSession::getActive();
	oa::Gru gru(10, 20, 1);
	
	// Test different sequence lengths
	for (oa::I32 seqLen : {1, 5, 10, 20, 50}) {
		auto input = oa::FnMatrix::randN(oa::MatrixShape{2, seqLen, 10}, oa::ScalarType::Float32);
		auto output = gru.forward(input);
		EXPECT_EQ(output.getShape(), (oa::MatrixShape{2, seqLen, 20}));
	}
}

TEST(Gru, GruHiddenSizeVariation) {
	auto& ctx = oa::ExecutionSession::getActive();
	
	for (oa::I32 hiddenSize : {8, 16, 32, 64, 128}) {
		oa::Gru gru(10, hiddenSize, 1);
		auto input = oa::FnMatrix::randN(oa::MatrixShape{2, 5, 10}, oa::ScalarType::Float32);
		auto output = gru.forward(input);
		EXPECT_EQ(output.getShape(), (oa::MatrixShape{2, 5, hiddenSize}));
	}
}

TEST(Gru, GruDeterminism) {
	auto& ctx = oa::ExecutionSession::getActive();
	oa::Gru gru(10, 20, 2);
	
	auto input = oa::FnMatrix::randN(oa::MatrixShape{2, 5, 10}, oa::ScalarType::Float32);
	
	// run twice with same input
	auto output1 = gru.forward(input);
	auto output2 = gru.forward(input);
	
	// Should produce identical results
	oa::F32 relError = computeRelativeError(output1, output2);
	EXPECT_LT(relError, 1e-6f);
}

TEST(Gru, GruGateActivations) {
	auto& ctx = oa::ExecutionSession::getActive();
	oa::GruCell cell(10, 20);
	
	// Test that gates are properly activated
	auto input = oa::FnMatrix::randN(oa::MatrixShape{2, 10}, oa::ScalarType::Float32);
	auto hidden = oa::FnMatrix::randN(oa::MatrixShape{2, 20}, oa::ScalarType::Float32);
	
	auto output = cell.step(input, hidden);
	(void)testSubmitAndWait(ctx);
	
	// output should be different from both input and hidden
	oa::F32 diffFromHidden = computeRelativeError(output, hidden);
	EXPECT_GT(diffFromHidden, 1e-3f);
}

TEST(Gru, GruNumericalStability) {
	auto& ctx = oa::ExecutionSession::getActive();
	oa::Gru gru(10, 20, 2);
	
	// Test with extreme values
	auto largeInput = oa::FnMatrix::full(oa::MatrixShape{2, 5, 10}, 10.0f, oa::ScalarType::Float32);
	auto output1 = gru.forward(largeInput);
	
	// Check for NaN or Inf
	auto absOutput = oa::FnMatrix::abs(output1);
	oa::F32 maxVal = oa::FnMatrix::max(absOutput).item();
	EXPECT_FALSE(std::isnan(maxVal));
	EXPECT_FALSE(std::isinf(maxVal));
	
	// Test with small values
	auto smallInput = oa::FnMatrix::full(oa::MatrixShape{2, 5, 10}, 1e-6f, oa::ScalarType::Float32);
	auto output2 = gru.forward(smallInput);
	
	absOutput = oa::FnMatrix::abs(output2);
	maxVal = oa::FnMatrix::max(absOutput).item();
	EXPECT_FALSE(std::isnan(maxVal));
	EXPECT_FALSE(std::isinf(maxVal));
}

TEST(Gru, GruParameterCount) {
	auto& ctx = oa::ExecutionSession::getActive();
	
	// Single layer: 3 * hidden * (input + hidden + 2*bias)
	oa::Gru gru1(10, 20, 1, true);
	oa::I64 expectedParams1 = 3 * 20 * (10 + 20 + 2);
	EXPECT_EQ(gru1.numParameters(), expectedParams1);
	
	// Without bias
	oa::Gru gru2(10, 20, 1, false);
	oa::I64 expectedParams2 = 3 * 20 * (10 + 20);
	EXPECT_EQ(gru2.numParameters(), expectedParams2);
	
	// Multi-layer
	oa::Gru gru3(10, 20, 3, true);
	oa::I64 expectedParams3 = 3 * 20 * (10 + 20 + 2) + 2 * (3 * 20 * (20 + 20 + 2));
	EXPECT_EQ(gru3.numParameters(), expectedParams3);
}

TEST(Gru, GruResetGate) {
	auto& ctx = oa::ExecutionSession::getActive();
	oa::GruCell cell(10, 20);
	
	// Test that reset gate affects hidden state
	auto input = oa::FnMatrix::randN(oa::MatrixShape{2, 10}, oa::ScalarType::Float32);
	auto hidden1 = oa::FnMatrix::randN(oa::MatrixShape{2, 20}, oa::ScalarType::Float32);
	auto hidden2 = oa::FnMatrix::zeros(oa::MatrixShape{2, 20}, oa::ScalarType::Float32);
	
	auto output1 = cell.step(input, hidden1);
	auto output2 = cell.step(input, hidden2);
	(void)testSubmitAndWait(ctx);
	
	// outputs should be different due to different hidden states
	oa::F32 relError = computeRelativeError(output1, output2);
	EXPECT_GT(relError, 1e-3f);
}

TEST(Gru, GruUpdateGate) {
	auto& ctx = oa::ExecutionSession::getActive();
	oa::GruCell cell(10, 20);
	
	// Multiple steps should show temporal dependency
	auto input = oa::FnMatrix::randN(oa::MatrixShape{2, 10}, oa::ScalarType::Float32);
	auto hidden = cell.zeroState(2);
	
	auto output1 = cell.step(input, hidden);
	auto output2 = cell.step(input, output1);
	auto output3 = cell.step(input, output2);
	(void)testSubmitAndWait(ctx);
	
	// Each output should be different
	oa::F32 diff12 = computeRelativeError(output1, output2);
	oa::F32 diff23 = computeRelativeError(output2, output3);
	
	EXPECT_GT(diff12, 1e-3f);
	EXPECT_GT(diff23, 1e-3f);
}

TEST(Gru, GruMemoryPersistence) {
	auto& ctx = oa::ExecutionSession::getActive();
	oa::Gru gru(10, 20, 1);
	
	// Test that GRU maintains memory across sequence
	auto input = oa::FnMatrix::randN(oa::MatrixShape{1, 10, 10}, oa::ScalarType::Float32);
	auto output = gru.forward(input);
	(void)testSubmitAndWait(ctx);
	
	// first and last outputs should be different
	auto first = oa::FnMatrix::slice(output, 1, 0, 1);
	auto last = oa::FnMatrix::slice(output, 1, 9, 10);
	(void)testSubmitAndWait(ctx);
	
	oa::F32 relError = computeRelativeError(first, last);
	EXPECT_GT(relError, 1e-3f);
}

TEST(Gru, GruEdgeCaseEmptyBatch) {
	auto& ctx = oa::ExecutionSession::getActive();
	oa::Gru gru(10, 20, 1);
	
	// Batch size of 1 (edge case)
	auto input = oa::FnMatrix::randN(oa::MatrixShape{1, 5, 10}, oa::ScalarType::Float32);
	auto output = gru.forward(input);
	
	EXPECT_EQ(output.getShape(), (oa::MatrixShape{1, 5, 20}));
}

TEST(Gru, GruEdgeCaseSingleTimestep) {
	auto& ctx = oa::ExecutionSession::getActive();
	oa::Gru gru(10, 20, 1);
	
	// sequence length of 1
	auto input = oa::FnMatrix::randN(oa::MatrixShape{2, 1, 10}, oa::ScalarType::Float32);
	auto output = gru.forward(input);
	
	EXPECT_EQ(output.getShape(), (oa::MatrixShape{2, 1, 20}));
}

TEST(Gru, GruEdgeCaseMinimalDimensions) {
	auto& ctx = oa::ExecutionSession::getActive();
	
	// Minimal dimensions
	oa::Gru gru(1, 1, 1);
	auto input = oa::FnMatrix::randN(oa::MatrixShape{1, 1, 1}, oa::ScalarType::Float32);
	auto output = gru.forward(input);
	
	EXPECT_EQ(output.getShape(), (oa::MatrixShape{1, 1, 1}));
}

TEST(Gru, GruSaveLoad) {
	auto& ctx = oa::ExecutionSession::getActive();
	oa::Gru gru1(10, 20, 2);
	
	// save
	auto status = gru1.save(testEngine(), "/tmp/test_gru.oam");
	EXPECT_TRUE(status.isOk());
	
	// Create new GRU and load
	oa::Gru gru2(10, 20, 2);
	status = gru2.load(testEngine(), "/tmp/test_gru.oam");
	EXPECT_TRUE(status.isOk());
	
	// Test that loaded model produces same output
	auto input = oa::FnMatrix::randN(oa::MatrixShape{2, 5, 10}, oa::ScalarType::Float32);
	auto output1 = gru1.forward(input);
	auto output2 = gru2.forward(input);
	(void)testSubmitAndWait(ctx);
	
	oa::F32 relError = computeRelativeError(output1, output2);
	EXPECT_LT(relError, 1e-3f);  // Relaxed tolerance for save/load
}

// ─── autograd gradient check vs a CPU FP64 reference ─────────────────────────
// The single biggest risk for a hand-rolled recurrent cell is a wrong/missing
// backward. Finite-differencing the *vulkan* forward is unreliable at tiny GEMM
// sizes (precision floor makes the FD chaotic), so instead we build an
// independent FP64 CPU reference of the cell and compare the framework's
// analytic autograd grads against FP64 central differences of loss = sum(h_new).
TEST(Gru, GruCellGradientCheck) {
	const oa::I32 B = 2, inSize = 4, H = 5;
	const oa::I32 G = 3 * H;

	auto gen = [](oa::I32 n, double scale, double phase, std::vector<double>& out) {
		out.resize(static_cast<size_t>(n));
		for (oa::I32 i = 0; i < n; ++i) out[static_cast<size_t>(i)] = scale * std::sin(0.37 * static_cast<double>(i) + phase);
	};
	std::vector<double> wih, whh, bih, bhh, xin, hin;
	gen(G * inSize, 0.3, 0.1, wih);  // weight_ih [G, inSize]
	gen(G * H,      0.3, 0.2, whh);  // weight_hh [G, H]
	gen(G,          0.1, 0.3, bih);  // bias_ih  [G]
	gen(G,          0.1, 0.4, bhh);  // bias_hh  [G]
	gen(B * inSize, 0.5, 0.5, xin);  // input    [B, inSize]
	gen(B * H,      0.4, 0.6, hin);  // hidden   [B, H]

	auto toMat = [](const std::vector<double>& v, const oa::MatrixShape& s) {
		std::vector<float> f(v.size());
		for (size_t i = 0; i < v.size(); ++i) f[i] = static_cast<float>(v[i]);
		return oa::FnMatrix::fromBytes(
			oa::Span<const oa::U8>(reinterpret_cast<const oa::U8*>(f.data()), f.size() * sizeof(float)), s);
	};

	oa::GruCell cell(inSize, H, true);
	auto& P = cell.parameters();
	ASSERT_EQ(P.size(), 4u);
	P[0].data = toMat(wih, oa::MatrixShape{G, inSize});
	P[1].data = toMat(whh, oa::MatrixShape{G, H});
	P[2].data = toMat(bih, oa::MatrixShape{G});
	P[3].data = toMat(bhh, oa::MatrixShape{G});
	for (auto& p : P) { p.data.setRequiresGrad(true); p.grad() = p.data.gradMatrix(); }
	auto input = toMat(xin, oa::MatrixShape{B, inSize}); input.setRequiresGrad(true);
	auto hidden = toMat(hin, oa::MatrixShape{B, H}); hidden.setRequiresGrad(true);

	// Framework analytic backward.
	oa::F32 fwLoss = 0.0f;
	oa::Matrix fwOut;
	{
		oa::GradientTape tape;
		fwOut = cell.step(input, hidden);
		auto loss = oa::FnMatrix::sum(fwOut);
		syncCtx();
		fwLoss = loss.item();
		tape.backward(loss);
		syncCtx();
	}

	// CPU FP64 forward of the cell (matches oa::GruCell::step + bias). loss = sum(h_new).
	auto sig = [](double v) { return 1.0 / (1.0 + std::exp(-v)); };
	auto cpuLoss = [&](const std::vector<double>& w, const std::vector<double>& u,
	                   const std::vector<double>& bi, const std::vector<double>& bu,
	                   const std::vector<double>& x, const std::vector<double>& h) -> double {
		double loss = 0.0;
		for (oa::I32 b = 0; b < B; ++b) {
			for (oa::I32 j = 0; j < H; ++j) {
				double ir = bi[0 * H + j], iz = bi[1 * H + j], in = bi[2 * H + j];
				for (oa::I32 k = 0; k < inSize; ++k) {
					ir += x[b * inSize + k] * w[(0 * H + j) * inSize + k];
					iz += x[b * inSize + k] * w[(1 * H + j) * inSize + k];
					in += x[b * inSize + k] * w[(2 * H + j) * inSize + k];
				}
				double hr = bu[0 * H + j], hz = bu[1 * H + j], hn = bu[2 * H + j];
				for (oa::I32 k = 0; k < H; ++k) {
					hr += h[b * H + k] * u[(0 * H + j) * H + k];
					hz += h[b * H + k] * u[(1 * H + j) * H + k];
					hn += h[b * H + k] * u[(2 * H + j) * H + k];
				}
				const double r = sig(ir + hr), z = sig(iz + hz), n = std::tanh(in + r * hn);
				loss += (1.0 - z) * n + z * h[b * H + j];
			}
		}
		return loss;
	};
	printf("  FWD CHECK: framework=%.8f cpu=%.8f diff=%.3e\n",
		static_cast<double>(fwLoss), cpuLoss(wih, whh, bih, bhh, xin, hin),
		static_cast<double>(fwLoss) - cpuLoss(wih, whh, bih, bhh, xin, hin));
	// Per-element forward comparison to localize the bug.
	{
		std::vector<float> fw = downloadF32(fwOut);
		auto cpuOut = [&](oa::I32 b, oa::I32 j) -> double {
			double ir = bih[0 * H + j], iz = bih[1 * H + j], in = bih[2 * H + j];
			for (oa::I32 k = 0; k < inSize; ++k) {
				ir += xin[b * inSize + k] * wih[(0 * H + j) * inSize + k];
				iz += xin[b * inSize + k] * wih[(1 * H + j) * inSize + k];
				in += xin[b * inSize + k] * wih[(2 * H + j) * inSize + k];
			}
			double hr = bhh[0 * H + j], hz = bhh[1 * H + j], hn = bhh[2 * H + j];
			for (oa::I32 k = 0; k < H; ++k) {
				hr += hin[b * H + k] * whh[(0 * H + j) * H + k];
				hz += hin[b * H + k] * whh[(1 * H + j) * H + k];
				hn += hin[b * H + k] * whh[(2 * H + j) * H + k];
			}
			double r = sig(ir + hr), z = sig(iz + hz), n = std::tanh(in + r * hn);
			return (1.0 - z) * n + z * hin[b * H + j];
		};
		for (oa::I32 b = 0; b < B; ++b)
			for (oa::I32 j = 0; j < H; ++j) {
				double ir = bih[0 * H + j], iz = bih[1 * H + j], inc = bih[2 * H + j];
				for (oa::I32 k = 0; k < inSize; ++k) {
					ir += xin[b * inSize + k] * wih[(0 * H + j) * inSize + k];
					iz += xin[b * inSize + k] * wih[(1 * H + j) * inSize + k];
					inc += xin[b * inSize + k] * wih[(2 * H + j) * inSize + k];
				}
				double hr = bhh[0 * H + j], hz = bhh[1 * H + j], hn = bhh[2 * H + j];
				for (oa::I32 k = 0; k < H; ++k) {
					hr += hin[b * H + k] * whh[(0 * H + j) * H + k];
					hz += hin[b * H + k] * whh[(1 * H + j) * H + k];
					hn += hin[b * H + k] * whh[(2 * H + j) * H + k];
				}
				double r = sig(ir + hr), z = sig(iz + hz), nc = std::tanh(inc + r * hn);
				printf("    [%d,%d] fw=%.6f  cpu: r=%.4f z=%.4f n=%.6f out=%.6f\n",
					b, j, fw[b * H + j], r, z, nc, (1.0 - z) * nc + z * hin[b * H + j]);
			}
	}
	const double eps = 1e-6;
	auto fdGrad = [&](std::vector<double>& v, oa::I32 idx) -> double {
		const double o = v[static_cast<size_t>(idx)];
		v[static_cast<size_t>(idx)] = o + eps; const double lp = cpuLoss(wih, whh, bih, bhh, xin, hin);
		v[static_cast<size_t>(idx)] = o - eps; const double lm = cpuLoss(wih, whh, bih, bhh, xin, hin);
		v[static_cast<size_t>(idx)] = o;
		return (lp - lm) / (2.0 * eps);
	};

	// compare analytic (framework, FP32) against reference (CPU, FP64).
	// The framework backward runs FP32 GEMMs, so on these tiny cell sizes there is
	// a ~1e-3 absolute noise floor that dominates small-magnitude gradients. A
	// genuine backward bug, by contrast, shows a *large* absolute deviation (wrong
	// sign / missing factor), so we require BOTH a meaningful absolute and relative
	// error to flag an element.
	auto cmp = [&](const std::vector<float>& ana, std::vector<double>& v, const char* name) {
		const oa::I32 n = static_cast<oa::I32>(v.size());
		int bad = 0; double maxRel = 0.0, maxAbs = 0.0;
		for (oa::I32 i = 0; i < n; ++i) {
			const double num = fdGrad(v, i);
			const double a   = ana[static_cast<size_t>(i)];
			const double absErr = std::fabs(num - a);
			const double mag = std::max(std::fabs(num), std::fabs(a));
			maxAbs = std::max(maxAbs, absErr);
			if (mag > 1e-6) maxRel = std::max(maxRel, absErr / mag);
			if (mag < 1e-6) continue;
			// Real bug ⇒ large abs AND large rel. FP32 noise ⇒ small abs on tiny grads.
			if (absErr > 2e-3 && absErr / mag > 0.05) {
				if (bad < 8) printf("  %s[%d] ref=%.6f analytic=%.6f abs=%.2e rel=%.4f\n", name, i, num, a, absErr, absErr / mag);
				++bad;
			}
		}
		printf("  %-10s checked %d elems, maxAbs=%.2e maxRel=%.2e bad=%d\n", name, n, maxAbs, maxRel, bad);
		return bad;
	};

	const int badWih = cmp(downloadF32(P[0].grad()), wih, "weight_ih");
	const int badWhh = cmp(downloadF32(P[1].grad()), whh, "weight_hh");
	const int badBih = cmp(downloadF32(P[2].grad()), bih, "bias_ih");
	const int badBhh = cmp(downloadF32(P[3].grad()), bhh, "bias_hh");
	const int badIn  = cmp(downloadF32(input.gradMatrix()),  xin, "input");
	const int badH   = cmp(downloadF32(hidden.gradMatrix()), hin, "hidden");

	EXPECT_EQ(badWih, 0);
	EXPECT_EQ(badWhh, 0);
	EXPECT_EQ(badBih, 0);
	EXPECT_EQ(badBhh, 0);
	EXPECT_EQ(badIn, 0);
	EXPECT_EQ(badH, 0);
}
