// Test/Ml/nn/TestRope.cpp
// Coverage for oa::Rope (Rotary position Embedding): a CPU forward-parity check and
// a finite-difference gradcheck of the autograd backward (GradRoPE → RopeApplyBwd).
//
// Before this file RoPE had NO numerical verification — neither the forward rotation
// nor the backward existed in any registered test. RoPE is a pure orthogonal rotation
// (no GEMM, no bf16), so fp32 finite differences are clean and OA_GEMM_FORCE_FP32 is
// not needed.

#include <gtest/gtest.h>
#include "../../oaTest.h"
#include <oa/ml.h>
#include <oa/ml/nn.h>
#include <oa/ml/autograd.h>
#include <oa/runtime/executionSession.h>
#include <cmath>
#include <vector>

static oa::Matrix matFromVec(const std::vector<float>& inV, oa::MatrixShape inShape) {
	return oa::FnMatrix::fromBytes(
		{reinterpret_cast<const oa::U8*>(inV.data()), inV.size() * sizeof(float)},
		inShape, oa::ScalarType::Float32);
}

static std::vector<float> downloadF32(const oa::Matrix& inM) {
	auto& ctx = oa::ExecutionSession::getActive();
	(void)testSubmitAndWait(ctx);
	const oa::I64 n = inM.numElements();
	std::vector<float> out(static_cast<size_t>(n));
	const oa::F32* src = inM.dataAs<const oa::F32>();
	for (oa::I64 i = 0; i < n; ++i) out[static_cast<size_t>(i)] = src[i];
	return out;
}

// Reference forward — mirrors RopeApply.slang exactly (llama/Granite half-split pairs).
static std::vector<float> cpuRope(const std::vector<float>& inX, int inT, int inHeads,
                                  int inHeadDim, float inThetaBase) {
	const int D = inHeads * inHeadDim;
	const int half = inHeadDim / 2;
	std::vector<float> out = inX;
	for (int t = 0; t < inT; ++t) {
		for (int h = 0; h < inHeads; ++h) {
			for (int i = 0; i < half; ++i) {
				const float freq = static_cast<float>(t) *
					std::pow(inThetaBase, -2.0f * static_cast<float>(i) / static_cast<float>(inHeadDim));
				const float c = std::cos(freq);
				const float s = std::sin(freq);
				const int d0 = t * D + h * inHeadDim + i;
				const int d1 = d0 + half;
				const float x0 = inX[static_cast<size_t>(d0)];
				const float x1 = inX[static_cast<size_t>(d1)];
				out[static_cast<size_t>(d0)] = x0 * c - x1 * s;
				out[static_cast<size_t>(d1)] = x0 * s + x1 * c;
			}
		}
	}
	return out;
}

TEST(TestRope, ForwardMatchesCpu) {
	const int T = 5, H = 2, headDim = 4;
	const int D = H * headDim;
	const float theta = 10000.0f;

	std::vector<float> x(static_cast<size_t>(T * D));
	auto f = [](int s) { return std::sin(0.7f * s + 0.4f) * 0.8f; };
	for (size_t i = 0; i < x.size(); ++i) x[i] = f(static_cast<int>(i));

	oa::Rope rope(H, headDim, theta);
	auto y = rope.forward(matFromVec(x, oa::MatrixShape{T, D}));
	auto gpu = downloadF32(y);
	auto ref = cpuRope(x, T, H, headDim, theta);

	float maxErr = 0.0f;
	for (size_t i = 0; i < ref.size(); ++i) {
		maxErr = std::max(maxErr, std::abs(gpu[i] - ref[i]));
		EXPECT_NEAR(gpu[i], ref[i], 1e-4f) << "i=" << i;
	}
	std::cerr << "RoPE forward vs CPU max abs err = " << maxErr << "\n";

	// rotation is norm-preserving per (i, i+half) pair → total ||y|| == ||x||.
	double nx = 0.0, ny = 0.0;
	for (size_t i = 0; i < x.size(); ++i) { nx += (double)x[i] * x[i]; ny += (double)gpu[i] * gpu[i]; }
	EXPECT_NEAR(std::sqrt(nx), std::sqrt(ny), 1e-3) << "RoPE must preserve norm";
}

// Finite-difference gradcheck of GradRoPE: analytic input grad (RopeApplyBwd) vs
// central differences of MSE(rope(x), target).
TEST(TestRope, BackwardGradcheck) {
	const int T = 4, H = 2, headDim = 4;
	const int D = H * headDim;
	const float theta = 10000.0f;

	auto& ctx = oa::ExecutionSession::getActive();
	oa::ExecutionSession::RecordingScope scope(ctx);

	std::vector<float> x(static_cast<size_t>(T * D)), tgt(static_cast<size_t>(T * D));
	auto f = [](int s) { return std::sin(0.7f * s + 0.4f) * 0.8f; };
	for (size_t i = 0; i < x.size(); ++i) { x[i] = f(static_cast<int>(i)); tgt[i] = f(static_cast<int>(i) + 17); }

	oa::Rope rope(H, headDim, theta);
	auto target = matFromVec(tgt, oa::MatrixShape{T, D});

	auto input = matFromVec(x, oa::MatrixShape{T, D});
	input.setRequiresGrad(true);

	oa::GradientTape tape;
	auto y = rope.forward(input);
	auto loss = oa::FnLoss::mse(y, target);
	tape.backward(loss);
	(void)testSubmitAndWait(ctx);

	auto analytic = downloadF32(input.gradMatrix());
	ASSERT_EQ(analytic.size(), x.size());

	// Numerical gradient: perturb input host memory, re-run forward without autograd.
	oa::F32* xData = input.dataAs<oa::F32>();
	const float eps = 1e-3f;
	auto scalarLoss = [&]() -> double {
		oa::GradNo noGrad;
		auto yy = rope.forward(input);
		auto l = oa::FnLoss::mse(yy, target);
		(void)testSubmitAndWait(ctx);
		return (double)l.dataAs<const oa::F32>()[0];
	};

	float maxErr = 0.0f;
	int nonTrivial = 0;
	for (size_t i = 0; i < x.size(); ++i) {
		const float orig = xData[i];
		xData[i] = orig + eps; (void)testSubmitAndWait(ctx); const double lp = scalarLoss();
		xData[i] = orig - eps; (void)testSubmitAndWait(ctx); const double lm = scalarLoss();
		xData[i] = orig; (void)testSubmitAndWait(ctx);
		const float num = (float)((lp - lm) / (2.0 * eps));
		const float ana = analytic[i];
		const float tol = 2e-3f + 2e-2f * std::abs(num);
		maxErr = std::max(maxErr, std::abs(num - ana));
		if (std::abs(num) > 1e-4f) ++nonTrivial;
		EXPECT_NEAR(num, ana, tol) << "grad[" << i << "] num=" << num << " ana=" << ana;
	}
	std::cerr << "RoPE backward gradcheck max abs err = " << maxErr << "\n";
	EXPECT_GE(nonTrivial, 4) << "gradients all ~0 — check is vacuous";
}
