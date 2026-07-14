// Test/Ml/Nn/TestRope.cpp
// Coverage for OaRoPE (Rotary Position Embedding): a CPU forward-parity check and
// a finite-difference gradcheck of the autograd backward (OaGradRoPE → RopeApplyBwd).
//
// Before this file RoPE had NO numerical verification — neither the forward rotation
// nor the backward existed in any registered test. RoPE is a pure orthogonal rotation
// (no GEMM, no bf16), so fp32 finite differences are clean and OA_GEMM_FORCE_FP32 is
// not needed.

#include <gtest/gtest.h>
#include <Oa/Ml.h>
#include <Oa/Ml/Nn.h>
#include <Oa/Ml/Autograd.h>
#include <Oa/Runtime/Context.h>
#include <cmath>
#include <vector>

static OaMatrix MatFromVec(const std::vector<float>& InV, OaMatrixShape InShape) {
	return OaFnMatrix::FromBytes(
		{reinterpret_cast<const OaU8*>(InV.data()), InV.size() * sizeof(float)},
		InShape, OaScalarType::Float32);
}

static std::vector<float> DownloadF32(const OaMatrix& InM) {
	auto& ctx = OaContext::GetDefault();
	(void)ctx.Execute();
	(void)ctx.Sync();
	const OaI64 n = InM.NumElements();
	std::vector<float> out(static_cast<size_t>(n));
	const OaF32* src = InM.DataAs<const OaF32>();
	for (OaI64 i = 0; i < n; ++i) out[static_cast<size_t>(i)] = src[i];
	return out;
}

// Reference forward — mirrors RopeApply.slang exactly (llama/Granite half-split pairs).
static std::vector<float> CpuRope(const std::vector<float>& InX, int InT, int InHeads,
                                  int InHeadDim, float InThetaBase) {
	const int D = InHeads * InHeadDim;
	const int half = InHeadDim / 2;
	std::vector<float> out = InX;
	for (int t = 0; t < InT; ++t) {
		for (int h = 0; h < InHeads; ++h) {
			for (int i = 0; i < half; ++i) {
				const float freq = static_cast<float>(t) *
					std::pow(InThetaBase, -2.0f * static_cast<float>(i) / static_cast<float>(InHeadDim));
				const float c = std::cos(freq);
				const float s = std::sin(freq);
				const int d0 = t * D + h * InHeadDim + i;
				const int d1 = d0 + half;
				const float x0 = InX[static_cast<size_t>(d0)];
				const float x1 = InX[static_cast<size_t>(d1)];
				out[static_cast<size_t>(d0)] = x0 * c - x1 * s;
				out[static_cast<size_t>(d1)] = x0 * s + x1 * c;
			}
		}
	}
	return out;
}

TEST(TestRope, ForwardMatchesCpu) {
	const int T = 5, H = 2, HeadDim = 4;
	const int D = H * HeadDim;
	const float theta = 10000.0f;

	std::vector<float> x(static_cast<size_t>(T * D));
	auto f = [](int s) { return std::sin(0.7f * s + 0.4f) * 0.8f; };
	for (size_t i = 0; i < x.size(); ++i) x[i] = f(static_cast<int>(i));

	OaRoPE rope(H, HeadDim, theta);
	auto y = rope.Forward(MatFromVec(x, OaMatrixShape{T, D}));
	auto gpu = DownloadF32(y);
	auto ref = CpuRope(x, T, H, HeadDim, theta);

	float maxErr = 0.0f;
	for (size_t i = 0; i < ref.size(); ++i) {
		maxErr = std::max(maxErr, std::abs(gpu[i] - ref[i]));
		EXPECT_NEAR(gpu[i], ref[i], 1e-4f) << "i=" << i;
	}
	std::cerr << "RoPE forward vs CPU max abs err = " << maxErr << "\n";

	// Rotation is norm-preserving per (i, i+half) pair → total ||y|| == ||x||.
	double nx = 0.0, ny = 0.0;
	for (size_t i = 0; i < x.size(); ++i) { nx += (double)x[i] * x[i]; ny += (double)gpu[i] * gpu[i]; }
	EXPECT_NEAR(std::sqrt(nx), std::sqrt(ny), 1e-3) << "RoPE must preserve norm";
}

// Finite-difference gradcheck of OaGradRoPE: analytic input grad (RopeApplyBwd) vs
// central differences of MSE(rope(x), target).
TEST(TestRope, BackwardGradcheck) {
	const int T = 4, H = 2, HeadDim = 4;
	const int D = H * HeadDim;
	const float theta = 10000.0f;

	auto& ctx = OaContext::GetDefault();
	OaContext::Scope scope(ctx);

	std::vector<float> x(static_cast<size_t>(T * D)), tgt(static_cast<size_t>(T * D));
	auto f = [](int s) { return std::sin(0.7f * s + 0.4f) * 0.8f; };
	for (size_t i = 0; i < x.size(); ++i) { x[i] = f(static_cast<int>(i)); tgt[i] = f(static_cast<int>(i) + 17); }

	OaRoPE rope(H, HeadDim, theta);
	auto target = MatFromVec(tgt, OaMatrixShape{T, D});

	auto input = MatFromVec(x, OaMatrixShape{T, D});
	input.SetRequiresGrad(true);

	OaGradientTape tape;
	auto y = rope.Forward(input);
	auto loss = OaFnLoss::Mse(y, target);
	tape.Backward(loss);
	(void)ctx.Execute();
	(void)ctx.Sync();

	auto analytic = DownloadF32(input.GradMatrix());
	ASSERT_EQ(analytic.size(), x.size());

	// Numerical gradient: perturb input host memory, re-run forward without autograd.
	OaF32* xData = input.DataAs<OaF32>();
	const float eps = 1e-3f;
	auto scalarLoss = [&]() -> double {
		OaGradNo noGrad;
		auto yy = rope.Forward(input);
		auto l = OaFnLoss::Mse(yy, target);
		(void)ctx.Execute();
		(void)ctx.Sync();
		return (double)l.DataAs<const OaF32>()[0];
	};

	float maxErr = 0.0f;
	int nonTrivial = 0;
	for (size_t i = 0; i < x.size(); ++i) {
		const float orig = xData[i];
		xData[i] = orig + eps; (void)ctx.Sync(); const double lp = scalarLoss();
		xData[i] = orig - eps; (void)ctx.Sync(); const double lm = scalarLoss();
		xData[i] = orig; (void)ctx.Sync();
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
