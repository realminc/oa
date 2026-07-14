// ═══════════════════════════════════════════════════════════════════════════
// OA Test — GRU (Gated Recurrent Unit) comprehensive testing
// ═══════════════════════════════════════════════════════════════════════════

#include "../../OaTest.h"
#include <Oa/Ml.h>
#include <Oa/Ml/Nn.h>
#include <Oa/Ml/Autograd.h>
#include <Oa/Runtime/Engine.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <string>
#include <vector>

// ─── Helper Functions ────────────────────────────────────────────────────────

static OaF32 ComputeRelativeError(const OaMatrix& InA, const OaMatrix& InB) {
	OA_ASSERT(InA.GetShape() == InB.GetShape());
	auto& ctx = OaContext::GetDefault();
	auto diff = OaFnMatrix::Sub(InA, InB);
	auto diffSq = OaFnMatrix::Mul(diff, diff);
	auto diffNorm = OaFnMatrix::Sqrt(OaFnMatrix::Sum(diffSq));
	auto aSq = OaFnMatrix::Mul(InA, InA);
	auto aNorm = OaFnMatrix::Sqrt(OaFnMatrix::Sum(aSq));
	(void)ctx.Execute();
	(void)ctx.Sync();
	OaF32 diffVal = diffNorm.Item();
	OaF32 aVal = aNorm.Item();
	return diffVal / (aVal + 1e-8f);
}

[[maybe_unused]] static void SyncCtx() {
	auto& ctx = OaContext::GetDefault();
	(void)ctx.Execute();
	(void)ctx.Sync();
}

// Fill a matrix in place from a deterministic, bounded host pattern (keeps the
// GRU recurrent dynamics stable so finite-difference grad-check is meaningful).
static void FillDeterministic(OaMatrix& m, float scale, double phase) {
	std::vector<float> v(static_cast<size_t>(m.NumElements()));
	for (size_t i = 0; i < v.size(); ++i) {
		v[i] = scale * static_cast<float>(std::sin(0.37 * static_cast<double>(i) + phase));
	}
	m = OaFnMatrix::FromBytes(
		OaSpan<const OaU8>(reinterpret_cast<const OaU8*>(v.data()), v.size() * sizeof(float)),
		m.GetShape());
}

// Bulk host download of a (synced) float32 matrix.
static std::vector<float> DownloadF32(const OaMatrix& m) {
	SyncCtx();
	const float* p = m.DataAs<const float>();
	return std::vector<float>(p, p + m.NumElements());
}

// ─── Test Cases ──────────────────────────────────────────────────────────────

TEST(Gru, GruCellConstruction) {
	auto& ctx = OaContext::GetDefault();
	
	// Test basic construction
	OaGruCell cell1(10, 20, true);
	EXPECT_EQ(cell1.InputSize(), 10);
	EXPECT_EQ(cell1.HiddenSize(), 20);
	EXPECT_EQ(cell1.Parameters().Size(), 4);  // weight_ih, weight_hh, bias_ih, bias_hh
	
	// Test without bias
	OaGruCell cell2(10, 20, false);
	EXPECT_EQ(cell2.Parameters().Size(), 2);  // weight_ih, weight_hh only
	
	// Verify parameter shapes
	const auto& params = cell1.Parameters();
	EXPECT_EQ(params[0].Data.GetShape(), (OaMatrixShape{3 * 20, 10}));  // weight_ih
	EXPECT_EQ(params[1].Data.GetShape(), (OaMatrixShape{3 * 20, 20}));  // weight_hh
	EXPECT_EQ(params[2].Data.GetShape(), OaMatrixShape{3 * 20});      // bias_ih
	EXPECT_EQ(params[3].Data.GetShape(), OaMatrixShape{3 * 20});      // bias_hh
}

TEST(Gru, GruCellZeroState) {
	auto& ctx = OaContext::GetDefault();
	OaGruCell cell(10, 20);
	
	auto h0 = cell.ZeroState(5);
	EXPECT_EQ(h0.GetShape(), (OaMatrixShape{5, 20}));
	
	// Verify it's actually zeros
	OaF32 sum = OaFnMatrix::Sum(h0).Item();
	EXPECT_NEAR(sum, 0.0f, 1e-6f);
}

TEST(Gru, GruCellForwardSingleStep) {
	auto& ctx = OaContext::GetDefault();
	OaGruCell cell(10, 20);
	
	// Create input: [batch=2, input_size=10]
	auto input = OaFnMatrix::RandN(OaMatrixShape{2, 10}, OaScalarType::Float32);
	auto hidden = cell.ZeroState(2);
	
	// Single step
	auto output = cell.Step(input, hidden);
	
	// Execute and sync to get actual results
	(void)ctx.Execute();
	(void)ctx.Sync();
	
	EXPECT_EQ(output.GetShape(), (OaMatrixShape{2, 20}));
	
	// Output should be different from zero state
	auto diff = OaFnMatrix::Sub(output, hidden);
	auto diffSq = OaFnMatrix::Mul(diff, diff);
	auto diffNorm = OaFnMatrix::Sqrt(OaFnMatrix::Sum(diffSq));
	(void)ctx.Execute();
	(void)ctx.Sync();
	OaF32 diffNormVal = diffNorm.Item();
	EXPECT_GT(diffNormVal, 1e-3f);
}

TEST(Gru, GruCellForwardWithoutHidden) {
	auto& ctx = OaContext::GetDefault();
	OaGruCell cell(10, 20);
	
	auto input = OaFnMatrix::RandN(OaMatrixShape{3, 10}, OaScalarType::Float32);
	auto output = cell.Forward(input);
	
	EXPECT_EQ(output.GetShape(), (OaMatrixShape{3, 20}));
}

TEST(Gru, GruCellOutputRange) {
	auto& ctx = OaContext::GetDefault();
	OaGruCell cell(10, 20);
	
	auto input = OaFnMatrix::RandN(OaMatrixShape{2, 10}, OaScalarType::Float32);
	auto hidden = cell.ZeroState(2);
	auto output = cell.Step(input, hidden);
	
	// GRU output should be bounded by tanh activation (approximately [-1, 1])
	auto absOutput = OaFnMatrix::Abs(output);
	OaF32 maxVal = OaFnMatrix::Max(absOutput).Item();
	EXPECT_LT(maxVal, 2.0f);  // Allow some margin
}

TEST(Gru, GruMultiLayerConstruction) {
	auto& ctx = OaContext::GetDefault();
	
	OaGru gru(10, 20, 3, true);
	EXPECT_EQ(gru.InputSize(), 10);
	EXPECT_EQ(gru.HiddenSize(), 20);
	EXPECT_EQ(gru.NumLayers(), 3);
	
	// Should have 3 layers registered as children
	EXPECT_EQ(gru.Children().Size(), 3);
}

TEST(Gru, GruForwardSequence) {
	auto& ctx = OaContext::GetDefault();
	OaGru gru(10, 20, 2);
	
	// Input: [batch=2, seq_len=5, input_size=10]
	auto input = OaFnMatrix::RandN(OaMatrixShape{2, 5, 10}, OaScalarType::Float32);
	auto output = gru.Forward(input);
	
	// Output: [batch=2, seq_len=5, hidden_size=20]
	EXPECT_EQ(output.GetShape(), (OaMatrixShape{2, 5, 20}));
}

TEST(Gru, GruSingleLayerSequence) {
	auto& ctx = OaContext::GetDefault();
	OaGru gru(10, 20, 1);
	
	auto input = OaFnMatrix::RandN(OaMatrixShape{3, 7, 10}, OaScalarType::Float32);
	auto output = gru.Forward(input);
	
	EXPECT_EQ(output.GetShape(), (OaMatrixShape{3, 7, 20}));
}

TEST(Gru, DecomposedForwardIsCausal) {
	// Mobile drivers use the decomposed route when the whole-sequence backward
	// shader exceeds their compiler/resource limits. Two inputs with an identical
	// prefix must produce identical prefix outputs; otherwise teacher-forced loss
	// can look excellent while autoregressive generation reads future padding.
	const char* previous = std::getenv("OA_DISABLE_GRU_SCAN");
	const std::string previousValue = previous != nullptr ? previous : "";
	setenv("OA_DISABLE_GRU_SCAN", "1", 1);

	constexpr OaI32 batch = 1;
	constexpr OaI32 sequence = 8;
	constexpr OaI32 inputSize = 5;
	constexpr OaI32 hiddenSize = 7;
	constexpr OaI32 prefix = 4;
	OaGru gru(inputSize, hiddenSize, 1);

	std::vector<float> a(batch * sequence * inputSize);
	std::vector<float> b(a.size());
	for (OaUsize index = 0; index < a.size(); ++index) {
		a[index] = 0.15F * static_cast<float>(
			std::sin(0.31 * static_cast<double>(index) + 0.2));
		b[index] = a[index];
	}
	for (OaI32 time = prefix; time < sequence; ++time) {
		for (OaI32 feature = 0; feature < inputSize; ++feature) {
			const OaUsize index = static_cast<OaUsize>(time * inputSize + feature);
			b[index] = -0.23F * static_cast<float>(
				std::cos(0.19 * static_cast<double>(index) + 0.4));
		}
	}

	auto inputA = OaFnMatrix::FromBytes(
		OaSpan<const OaU8>(reinterpret_cast<const OaU8*>(a.data()),
			a.size() * sizeof(float)),
		OaMatrixShape{batch, sequence, inputSize});
	auto outputA = gru.Forward(inputA);
	const auto valuesA = DownloadF32(outputA);
	auto inputB = OaFnMatrix::FromBytes(
		OaSpan<const OaU8>(reinterpret_cast<const OaU8*>(b.data()),
			b.size() * sizeof(float)),
		OaMatrixShape{batch, sequence, inputSize});
	auto outputB = gru.Forward(inputB);
	const auto valuesB = DownloadF32(outputB);

	for (OaI32 time = 0; time < prefix; ++time) {
		for (OaI32 hidden = 0; hidden < hiddenSize; ++hidden) {
			const OaUsize index = static_cast<OaUsize>(time * hiddenSize + hidden);
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
	auto& ctx = OaContext::GetDefault();
	OaGru gru(10, 20, 1);
	
	const OaI32 batch = 2;
	const OaI32 seqLen = 5;
	
	// Manual step-by-step processing
	auto hidden = gru.ZeroState(batch, 0);
	OaVec<OaMatrix> outputs;
	
	for (OaI32 t = 0; t < seqLen; ++t) {
		auto xt = OaFnMatrix::RandN(OaMatrixShape{batch, 10}, OaScalarType::Float32);
		hidden = gru.Step(xt, hidden, 0);
		outputs.PushBack(hidden);
	}
	
	EXPECT_EQ(outputs.Size(), seqLen);
	EXPECT_EQ(outputs[0].GetShape(), (OaMatrixShape{batch, 20}));
}

TEST(Gru, GruBatchSizeVariation) {
	auto& ctx = OaContext::GetDefault();
	OaGru gru(10, 20, 2);
	
	// Test different batch sizes
	for (OaI32 batch : {1, 2, 4, 8, 16}) {
		auto input = OaFnMatrix::RandN(OaMatrixShape{batch, 3, 10}, OaScalarType::Float32);
		auto output = gru.Forward(input);
		EXPECT_EQ(output.GetShape(), (OaMatrixShape{batch, 3, 20}));
	}
}

TEST(Gru, GruSequenceLengthVariation) {
	auto& ctx = OaContext::GetDefault();
	OaGru gru(10, 20, 1);
	
	// Test different sequence lengths
	for (OaI32 seqLen : {1, 5, 10, 20, 50}) {
		auto input = OaFnMatrix::RandN(OaMatrixShape{2, seqLen, 10}, OaScalarType::Float32);
		auto output = gru.Forward(input);
		EXPECT_EQ(output.GetShape(), (OaMatrixShape{2, seqLen, 20}));
	}
}

TEST(Gru, GruHiddenSizeVariation) {
	auto& ctx = OaContext::GetDefault();
	
	for (OaI32 hiddenSize : {8, 16, 32, 64, 128}) {
		OaGru gru(10, hiddenSize, 1);
		auto input = OaFnMatrix::RandN(OaMatrixShape{2, 5, 10}, OaScalarType::Float32);
		auto output = gru.Forward(input);
		EXPECT_EQ(output.GetShape(), (OaMatrixShape{2, 5, hiddenSize}));
	}
}

TEST(Gru, GruDeterminism) {
	auto& ctx = OaContext::GetDefault();
	OaGru gru(10, 20, 2);
	
	auto input = OaFnMatrix::RandN(OaMatrixShape{2, 5, 10}, OaScalarType::Float32);
	
	// Run twice with same input
	auto output1 = gru.Forward(input);
	auto output2 = gru.Forward(input);
	
	// Should produce identical results
	OaF32 relError = ComputeRelativeError(output1, output2);
	EXPECT_LT(relError, 1e-6f);
}

TEST(Gru, GruGateActivations) {
	auto& ctx = OaContext::GetDefault();
	OaGruCell cell(10, 20);
	
	// Test that gates are properly activated
	auto input = OaFnMatrix::RandN(OaMatrixShape{2, 10}, OaScalarType::Float32);
	auto hidden = OaFnMatrix::RandN(OaMatrixShape{2, 20}, OaScalarType::Float32);
	
	auto output = cell.Step(input, hidden);
	(void)ctx.Execute();
	(void)ctx.Sync();
	
	// Output should be different from both input and hidden
	OaF32 diffFromHidden = ComputeRelativeError(output, hidden);
	EXPECT_GT(diffFromHidden, 1e-3f);
}

TEST(Gru, GruNumericalStability) {
	auto& ctx = OaContext::GetDefault();
	OaGru gru(10, 20, 2);
	
	// Test with extreme values
	auto largeInput = OaFnMatrix::Full(OaMatrixShape{2, 5, 10}, 10.0f, OaScalarType::Float32);
	auto output1 = gru.Forward(largeInput);
	
	// Check for NaN or Inf
	auto absOutput = OaFnMatrix::Abs(output1);
	OaF32 maxVal = OaFnMatrix::Max(absOutput).Item();
	EXPECT_FALSE(std::isnan(maxVal));
	EXPECT_FALSE(std::isinf(maxVal));
	
	// Test with small values
	auto smallInput = OaFnMatrix::Full(OaMatrixShape{2, 5, 10}, 1e-6f, OaScalarType::Float32);
	auto output2 = gru.Forward(smallInput);
	
	absOutput = OaFnMatrix::Abs(output2);
	maxVal = OaFnMatrix::Max(absOutput).Item();
	EXPECT_FALSE(std::isnan(maxVal));
	EXPECT_FALSE(std::isinf(maxVal));
}

TEST(Gru, GruParameterCount) {
	auto& ctx = OaContext::GetDefault();
	
	// Single layer: 3 * hidden * (input + hidden + 2*bias)
	OaGru gru1(10, 20, 1, true);
	OaI64 expectedParams1 = 3 * 20 * (10 + 20 + 2);
	EXPECT_EQ(gru1.NumParameters(), expectedParams1);
	
	// Without bias
	OaGru gru2(10, 20, 1, false);
	OaI64 expectedParams2 = 3 * 20 * (10 + 20);
	EXPECT_EQ(gru2.NumParameters(), expectedParams2);
	
	// Multi-layer
	OaGru gru3(10, 20, 3, true);
	OaI64 expectedParams3 = 3 * 20 * (10 + 20 + 2) + 2 * (3 * 20 * (20 + 20 + 2));
	EXPECT_EQ(gru3.NumParameters(), expectedParams3);
}

TEST(Gru, GruResetGate) {
	auto& ctx = OaContext::GetDefault();
	OaGruCell cell(10, 20);
	
	// Test that reset gate affects hidden state
	auto input = OaFnMatrix::RandN(OaMatrixShape{2, 10}, OaScalarType::Float32);
	auto hidden1 = OaFnMatrix::RandN(OaMatrixShape{2, 20}, OaScalarType::Float32);
	auto hidden2 = OaFnMatrix::Zeros(OaMatrixShape{2, 20}, OaScalarType::Float32);
	
	auto output1 = cell.Step(input, hidden1);
	auto output2 = cell.Step(input, hidden2);
	(void)ctx.Execute();
	(void)ctx.Sync();
	
	// Outputs should be different due to different hidden states
	OaF32 relError = ComputeRelativeError(output1, output2);
	EXPECT_GT(relError, 1e-3f);
}

TEST(Gru, GruUpdateGate) {
	auto& ctx = OaContext::GetDefault();
	OaGruCell cell(10, 20);
	
	// Multiple steps should show temporal dependency
	auto input = OaFnMatrix::RandN(OaMatrixShape{2, 10}, OaScalarType::Float32);
	auto hidden = cell.ZeroState(2);
	
	auto output1 = cell.Step(input, hidden);
	auto output2 = cell.Step(input, output1);
	auto output3 = cell.Step(input, output2);
	(void)ctx.Execute();
	(void)ctx.Sync();
	
	// Each output should be different
	OaF32 diff12 = ComputeRelativeError(output1, output2);
	OaF32 diff23 = ComputeRelativeError(output2, output3);
	
	EXPECT_GT(diff12, 1e-3f);
	EXPECT_GT(diff23, 1e-3f);
}

TEST(Gru, GruMemoryPersistence) {
	auto& ctx = OaContext::GetDefault();
	OaGru gru(10, 20, 1);
	
	// Test that GRU maintains memory across sequence
	auto input = OaFnMatrix::RandN(OaMatrixShape{1, 10, 10}, OaScalarType::Float32);
	auto output = gru.Forward(input);
	(void)ctx.Execute();
	(void)ctx.Sync();
	
	// First and last outputs should be different
	auto first = OaFnMatrix::Slice(output, 1, 0, 1);
	auto last = OaFnMatrix::Slice(output, 1, 9, 10);
	(void)ctx.Execute();
	(void)ctx.Sync();
	
	OaF32 relError = ComputeRelativeError(first, last);
	EXPECT_GT(relError, 1e-3f);
}

TEST(Gru, GruEdgeCaseEmptyBatch) {
	auto& ctx = OaContext::GetDefault();
	OaGru gru(10, 20, 1);
	
	// Batch size of 1 (edge case)
	auto input = OaFnMatrix::RandN(OaMatrixShape{1, 5, 10}, OaScalarType::Float32);
	auto output = gru.Forward(input);
	
	EXPECT_EQ(output.GetShape(), (OaMatrixShape{1, 5, 20}));
}

TEST(Gru, GruEdgeCaseSingleTimestep) {
	auto& ctx = OaContext::GetDefault();
	OaGru gru(10, 20, 1);
	
	// Sequence length of 1
	auto input = OaFnMatrix::RandN(OaMatrixShape{2, 1, 10}, OaScalarType::Float32);
	auto output = gru.Forward(input);
	
	EXPECT_EQ(output.GetShape(), (OaMatrixShape{2, 1, 20}));
}

TEST(Gru, GruEdgeCaseMinimalDimensions) {
	auto& ctx = OaContext::GetDefault();
	
	// Minimal dimensions
	OaGru gru(1, 1, 1);
	auto input = OaFnMatrix::RandN(OaMatrixShape{1, 1, 1}, OaScalarType::Float32);
	auto output = gru.Forward(input);
	
	EXPECT_EQ(output.GetShape(), (OaMatrixShape{1, 1, 1}));
}

TEST(Gru, GruSaveLoad) {
	auto& ctx = OaContext::GetDefault();
	OaGru gru1(10, 20, 2);
	
	// Save
	auto status = gru1.Save("/tmp/test_gru.oam");
	EXPECT_TRUE(status.IsOk());
	
	// Create new GRU and load
	OaGru gru2(10, 20, 2);
	status = gru2.Load("/tmp/test_gru.oam");
	EXPECT_TRUE(status.IsOk());
	
	// Test that loaded model produces same output
	auto input = OaFnMatrix::RandN(OaMatrixShape{2, 5, 10}, OaScalarType::Float32);
	auto output1 = gru1.Forward(input);
	auto output2 = gru2.Forward(input);
	(void)ctx.Execute();
	(void)ctx.Sync();
	
	OaF32 relError = ComputeRelativeError(output1, output2);
	EXPECT_LT(relError, 1e-3f);  // Relaxed tolerance for save/load
}

// ─── Autograd gradient check vs a CPU FP64 reference ─────────────────────────
// The single biggest risk for a hand-rolled recurrent cell is a wrong/missing
// backward. Finite-differencing the *Vulkan* forward is unreliable at tiny GEMM
// sizes (precision floor makes the FD chaotic), so instead we build an
// independent FP64 CPU reference of the cell and compare the framework's
// analytic autograd grads against FP64 central differences of loss = Sum(h_new).
TEST(Gru, GruCellGradientCheck) {
	const OaI32 B = 2, InSize = 4, H = 5;
	const OaI32 G = 3 * H;

	auto gen = [](OaI32 n, double scale, double phase, std::vector<double>& out) {
		out.resize(static_cast<size_t>(n));
		for (OaI32 i = 0; i < n; ++i) out[static_cast<size_t>(i)] = scale * std::sin(0.37 * static_cast<double>(i) + phase);
	};
	std::vector<double> wih, whh, bih, bhh, xin, hin;
	gen(G * InSize, 0.3, 0.1, wih);  // weight_ih [G, InSize]
	gen(G * H,      0.3, 0.2, whh);  // weight_hh [G, H]
	gen(G,          0.1, 0.3, bih);  // bias_ih  [G]
	gen(G,          0.1, 0.4, bhh);  // bias_hh  [G]
	gen(B * InSize, 0.5, 0.5, xin);  // input    [B, InSize]
	gen(B * H,      0.4, 0.6, hin);  // hidden   [B, H]

	auto toMat = [](const std::vector<double>& v, const OaMatrixShape& s) {
		std::vector<float> f(v.size());
		for (size_t i = 0; i < v.size(); ++i) f[i] = static_cast<float>(v[i]);
		return OaFnMatrix::FromBytes(
			OaSpan<const OaU8>(reinterpret_cast<const OaU8*>(f.data()), f.size() * sizeof(float)), s);
	};

	OaGruCell cell(InSize, H, true);
	auto& P = cell.Parameters();
	ASSERT_EQ(P.Size(), 4u);
	P[0].Data = toMat(wih, OaMatrixShape{G, InSize});
	P[1].Data = toMat(whh, OaMatrixShape{G, H});
	P[2].Data = toMat(bih, OaMatrixShape{G});
	P[3].Data = toMat(bhh, OaMatrixShape{G});
	for (auto& p : P) { p.Data.SetRequiresGrad(true); p.Grad() = p.Data.GradMatrix(); }
	auto input = toMat(xin, OaMatrixShape{B, InSize}); input.SetRequiresGrad(true);
	auto hidden = toMat(hin, OaMatrixShape{B, H}); hidden.SetRequiresGrad(true);

	// Framework analytic backward.
	OaF32 fwLoss = 0.0f;
	OaMatrix fwOut;
	{
		OaGradientTape tape;
		fwOut = cell.Step(input, hidden);
		auto loss = OaFnMatrix::Sum(fwOut);
		SyncCtx();
		fwLoss = loss.Item();
		tape.Backward(loss);
		SyncCtx();
	}

	// CPU FP64 forward of the cell (matches OaGruCell::Step + bias). loss = sum(h_new).
	auto sig = [](double v) { return 1.0 / (1.0 + std::exp(-v)); };
	auto cpuLoss = [&](const std::vector<double>& w, const std::vector<double>& u,
	                   const std::vector<double>& bi, const std::vector<double>& bu,
	                   const std::vector<double>& x, const std::vector<double>& h) -> double {
		double loss = 0.0;
		for (OaI32 b = 0; b < B; ++b) {
			for (OaI32 j = 0; j < H; ++j) {
				double ir = bi[0 * H + j], iz = bi[1 * H + j], in = bi[2 * H + j];
				for (OaI32 k = 0; k < InSize; ++k) {
					ir += x[b * InSize + k] * w[(0 * H + j) * InSize + k];
					iz += x[b * InSize + k] * w[(1 * H + j) * InSize + k];
					in += x[b * InSize + k] * w[(2 * H + j) * InSize + k];
				}
				double hr = bu[0 * H + j], hz = bu[1 * H + j], hn = bu[2 * H + j];
				for (OaI32 k = 0; k < H; ++k) {
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
		std::vector<float> fw = DownloadF32(fwOut);
		auto cpuOut = [&](OaI32 b, OaI32 j) -> double {
			double ir = bih[0 * H + j], iz = bih[1 * H + j], in = bih[2 * H + j];
			for (OaI32 k = 0; k < InSize; ++k) {
				ir += xin[b * InSize + k] * wih[(0 * H + j) * InSize + k];
				iz += xin[b * InSize + k] * wih[(1 * H + j) * InSize + k];
				in += xin[b * InSize + k] * wih[(2 * H + j) * InSize + k];
			}
			double hr = bhh[0 * H + j], hz = bhh[1 * H + j], hn = bhh[2 * H + j];
			for (OaI32 k = 0; k < H; ++k) {
				hr += hin[b * H + k] * whh[(0 * H + j) * H + k];
				hz += hin[b * H + k] * whh[(1 * H + j) * H + k];
				hn += hin[b * H + k] * whh[(2 * H + j) * H + k];
			}
			double r = sig(ir + hr), z = sig(iz + hz), n = std::tanh(in + r * hn);
			return (1.0 - z) * n + z * hin[b * H + j];
		};
		for (OaI32 b = 0; b < B; ++b)
			for (OaI32 j = 0; j < H; ++j) {
				double ir = bih[0 * H + j], iz = bih[1 * H + j], inc = bih[2 * H + j];
				for (OaI32 k = 0; k < InSize; ++k) {
					ir += xin[b * InSize + k] * wih[(0 * H + j) * InSize + k];
					iz += xin[b * InSize + k] * wih[(1 * H + j) * InSize + k];
					inc += xin[b * InSize + k] * wih[(2 * H + j) * InSize + k];
				}
				double hr = bhh[0 * H + j], hz = bhh[1 * H + j], hn = bhh[2 * H + j];
				for (OaI32 k = 0; k < H; ++k) {
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
	auto fdGrad = [&](std::vector<double>& v, OaI32 idx) -> double {
		const double o = v[static_cast<size_t>(idx)];
		v[static_cast<size_t>(idx)] = o + eps; const double lp = cpuLoss(wih, whh, bih, bhh, xin, hin);
		v[static_cast<size_t>(idx)] = o - eps; const double lm = cpuLoss(wih, whh, bih, bhh, xin, hin);
		v[static_cast<size_t>(idx)] = o;
		return (lp - lm) / (2.0 * eps);
	};

	// Compare analytic (framework, FP32) against reference (CPU, FP64).
	// The framework backward runs FP32 GEMMs, so on these tiny cell sizes there is
	// a ~1e-3 absolute noise floor that dominates small-magnitude gradients. A
	// genuine backward bug, by contrast, shows a *large* absolute deviation (wrong
	// sign / missing factor), so we require BOTH a meaningful absolute and relative
	// error to flag an element.
	auto cmp = [&](const std::vector<float>& ana, std::vector<double>& v, const char* name) {
		const OaI32 n = static_cast<OaI32>(v.size());
		int bad = 0; double maxRel = 0.0, maxAbs = 0.0;
		for (OaI32 i = 0; i < n; ++i) {
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

	const int badWih = cmp(DownloadF32(P[0].Grad()), wih, "weight_ih");
	const int badWhh = cmp(DownloadF32(P[1].Grad()), whh, "weight_hh");
	const int badBih = cmp(DownloadF32(P[2].Grad()), bih, "bias_ih");
	const int badBhh = cmp(DownloadF32(P[3].Grad()), bhh, "bias_hh");
	const int badIn  = cmp(DownloadF32(input.GradMatrix()),  xin, "input");
	const int badH   = cmp(DownloadF32(hidden.GradMatrix()), hin, "hidden");

	EXPECT_EQ(badWih, 0);
	EXPECT_EQ(badWhh, 0);
	EXPECT_EQ(badBih, 0);
	EXPECT_EQ(badBhh, 0);
	EXPECT_EQ(badIn, 0);
	EXPECT_EQ(badH, 0);
}
