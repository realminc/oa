// Test/Ml/Nn/TestLinearGelu.cpp
// Correctness for the fused OaLinear + GELU path (OaFnMatrix::LinearGelu +
// OaGradLinearGelu). The fused forward must match the unfused Gelu(Linear(x))
// reference, and the fused backward (which recomputes the pre-activation for
// GeluBwd) must match finite-difference numerical gradients.

#include "../../OaTest.h"
#include <gtest/gtest.h>
#include <Oa/Ml.h>
#include <Oa/Ml/Autograd.h>
#include <Oa/Runtime/Context.h>
#include <cmath>
#include <cstdlib>

// Finite-difference checking is only valid in fp32 (bf16's ~3-digit mantissa
// swallows the perturbations). Same rationale as TestGruNumericalGrad.
static void ForceFp32Gemm() { setenv("OA_GEMM_FORCE_FP32", "1", 1); }

static bool GradClose(OaF32 InAnalytical, OaF32 InNumerical,
	OaF32 InAtol = 2e-3F, OaF32 InRtol = 2e-2F) {
	return std::abs(InAnalytical - InNumerical) <= (InAtol + (InRtol * std::abs(InNumerical)));
}

static OaF32 ComputeNumericalGradient(
	std::function<OaF32()> InLossFunc, OaMatrix& InParam, OaI32 InIndex,
	OaF32 InEpsilon = 1e-2f) {
	auto& ctx = OaContext::GetDefault();
	OaF32* data = InParam.DataAs<OaF32>();
	OaF32 original = data[InIndex];
	data[InIndex] = original + InEpsilon;
	(void)ctx.Sync();
	OaF32 lossPlus = InLossFunc();
	data[InIndex] = original - InEpsilon;
	(void)ctx.Sync();
	OaF32 lossMinus = InLossFunc();
	data[InIndex] = original;
	(void)ctx.Sync();
	return (lossPlus - lossMinus) / (2.0f * InEpsilon);
}

// The fused LinearGelu forward must equal the unfused Gelu(Linear(x, W, b)) using
// the SAME weights — i.e. the fused GEMM epilogue computes the right GELU.
TEST(LinearGelu, ForwardMatchesUnfused) {
	constexpr OaI32 kRows = 8;
	constexpr OaI32 kIn   = 16;
	constexpr OaI32 kOut  = 12;

	ForceFp32Gemm();
	auto& ctx = OaContext::GetDefault();
	OaContext::RecordingScope scope(ctx);

	OaLinear linear(kIn, kOut);
	linear.SetActivation(OaActivation::Gelu);
	for (auto& p : linear.Parameters()) {
		p.Data = OaFnMatrix::Scale(OaFnMatrix::RandN(p.Data.GetShape(), OaScalarType::Float32), 0.5f);
	}

	auto x = OaFnMatrix::Scale(OaFnMatrix::RandN(OaMatrixShape{kRows, kIn}, OaScalarType::Float32), 0.5f);

	OaMatrix fused, reference;
	{
		OaGradNo noGrad;  // pure forward, no graph
		fused = linear.Forward(x);
		// Unfused reference with identical weights.
		auto pre = OaFnMatrix::Linear(x, linear.Parameters()[0].Data, linear.Parameters()[1].Data);
		reference = OaFnMatrix::Gelu(pre);
		(void)ctx.Execute();
		(void)ctx.Sync();
	}

	ASSERT_EQ(fused.NumElements(), reference.NumElements());
	const OaF32* f = fused.DataAs<const OaF32>();
	const OaF32* r = reference.DataAs<const OaF32>();
	OaF32 maxAbsDiff = 0.0f;
	for (OaI64 i = 0; i < fused.NumElements(); ++i) {
		maxAbsDiff = std::max(maxAbsDiff, std::abs(f[i] - r[i]));
	}
	printf("\nLinearGelu fused vs unfused: max|diff| = %.6g\n", maxAbsDiff);
	EXPECT_LT(maxAbsDiff, 1e-3f) << "fused LinearGelu forward diverges from Gelu(Linear(x))";
}

// The fused backward (OaGradLinearGelu recomputes pre-activation for GeluBwd)
// must match finite-difference numerical gradients on both weight and bias.
TEST(LinearGelu, BackwardNumericalGrad) {
	constexpr OaI32 kRows = 4;
	constexpr OaI32 kIn   = 6;
	constexpr OaI32 kOut  = 5;

	ForceFp32Gemm();
	OaFnMatrix::SetRngSeed(2024);
	auto& ctx = OaContext::GetDefault();
	OaContext::RecordingScope scope(ctx);

	OaLinear linear(kIn, kOut);
	linear.SetActivation(OaActivation::Gelu);
	for (auto& p : linear.Parameters()) {
		p.Data = OaFnMatrix::Scale(OaFnMatrix::RandN(p.Data.GetShape(), OaScalarType::Float32), 0.5f);
		p.Data.SetRequiresGrad(true);
	}

	auto input = OaFnMatrix::Scale(OaFnMatrix::RandN(OaMatrixShape{kRows, kIn}, OaScalarType::Float32), 1.0f);
	input.SetRequiresGrad(true);
	auto target = OaFnMatrix::RandN(OaMatrixShape{kRows, kOut}, OaScalarType::Float32);

	OaGradientTape tape;
	auto output = linear.Forward(input);
	auto loss = OaFnLoss::Mse(output, target);
	tape.Backward(loss);
	(void)ctx.Execute();
	(void)ctx.Sync();

	struct ParamCheck { const char* Name; OaI32 Index; };
	const ParamCheck checks[] = {{"weight", 0}, {"bias", 1}};

	OaI32 numChecked = 0, numFailed = 0, numNonTrivial = 0;
	for (const auto& chk : checks) {
		auto  analytical = linear.Parameters()[chk.Index].Grad().DataAs<const OaF32>();
		auto& weight     = linear.Parameters()[chk.Index].Data;
		const OaI64 n    = weight.NumElements();

		printf("\nLinearGelu backward grad check (%s):\n", chk.Name);
		printf("Index | Analytical | Numerical  | Close?\n");
		for (OaI64 idx = 0; idx < n; ++idx) {
			auto lossFunc = [&]() -> OaF32 {
				OaGradNo noGrad;
				auto out = linear.Forward(input);
				auto l = OaFnLoss::Mse(out, target);
				(void)ctx.Execute();
				(void)ctx.Sync();
				return l.DataAs<const OaF32>()[0];
			};
			OaF32 num = ComputeNumericalGradient(lossFunc, weight, static_cast<OaI32>(idx));
			OaF32 ana = analytical[idx];
			const bool close = GradClose(ana, num);
			++numChecked;
			if (not close) ++numFailed;
			if (std::abs(num) > 1e-4F) ++numNonTrivial;
			if (std::abs(num) > 5e-5F || not close) {
				printf("%5lld | %10.6f | %10.6f | %s\n",
					static_cast<long long>(idx), ana, num, close ? "yes" : "NO");
			}
		}
	}
	printf("\nLinearGelu GradCheck: %d/%d within tol, %d non-trivial\n",
		numChecked - numFailed, numChecked, numNonTrivial);

	EXPECT_EQ(numFailed, 0) << "fused LinearGelu gradient failed numerical check";
	EXPECT_GE(numNonTrivial, 3) << "gradients all ~0 — check is vacuous";
}

// Regression test for the SMEM-overflow bug (commit 1398f5e):
// OaFnMatrix::Linear with NLP-scale shapes (M=1024, N=32, K=32) routes through
// AddLinear → GemmBiasCmSgBf16 when N%16==0 and BF16 CoopMat is available.
// The fused kernel's smOut[BM*BN] = 64KB exceeded the 48KB SMEM limit, causing
// a silent hang on first dispatch. This test does NOT force FP32, so it
// exercises the actual BF16 fused path that NLP tutorials hit.
TEST(LinearGelu, FusedBf16NlpShapesNoHang) {
	auto& ctx = OaContext::GetDefault();
	OaContext::RecordingScope scope(ctx);

	// NLP-scale shapes: large batch, small output dim (divisible by 16).
	struct Shape { OaI32 M; OaI32 N; OaI32 K; };
	static const Shape kShapes[] = {
		{1024, 32, 32}, {512, 64, 32}, {1024, 32, 64}, {256, 16, 16},
	};

	for (const auto& sh : kShapes) {
		auto x = OaFnMatrix::RandN(OaMatrixShape{sh.M, sh.K}, OaScalarType::Float32);
		auto w = OaFnMatrix::RandN(OaMatrixShape{sh.N, sh.K}, OaScalarType::Float32);
		auto b = OaFnMatrix::RandN(OaMatrixShape{sh.N}, OaScalarType::Float32);

		auto y = OaFnMatrix::Linear(x, w, b);
		(void)ctx.Execute();
		(void)ctx.Sync();

		OaExpectShape(y, {sh.M, sh.N});
		OaExpectFinite(y);
	}
}
