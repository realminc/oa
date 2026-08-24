// Test/Ml/nn/TestLinearGelu.cpp
// Correctness for the fused oa::Linear activation paths. Each fused forward must
// match the corresponding activation(Linear(x)) reference, and each backward
// must match finite-difference numerical gradients.

#include "../../oaTest.h"
#include <gtest/gtest.h>
#include <oa/ml.h>
#include <oa/ml/autograd.h>
#include <oa/runtime/executionSession.h>
#include <cmath>
#include <cstdlib>

// Finite-difference checking is only valid in fp32 (bf16's ~3-digit mantissa
// swallows the perturbations). Same rationale as TestGruNumericalGrad.
static void forceFp32Gemm() { setenv("OA_GEMM_FORCE_FP32", "1", 1); }

static bool gradClose(oa::F32 inAnalytical, oa::F32 inNumerical,
	oa::F32 inAtol = 2e-3F, oa::F32 inRtol = 2e-2F) {
	return std::abs(inAnalytical - inNumerical) <= (inAtol + (inRtol * std::abs(inNumerical)));
}

static oa::F32 computeNumericalGradient(
	std::function<oa::F32()> inLossFunc, oa::Matrix& inParam, oa::I32 inIndex,
	oa::F32 inEpsilon = 1e-2f) {
	auto& ctx = oa::ExecutionSession::getActive();
	oa::F32* data = inParam.dataAs<oa::F32>();
	oa::F32 original = data[inIndex];
	data[inIndex] = original + inEpsilon;
	(void)testSubmitAndWait(ctx);
	oa::F32 lossPlus = inLossFunc();
	data[inIndex] = original - inEpsilon;
	(void)testSubmitAndWait(ctx);
	oa::F32 lossMinus = inLossFunc();
	data[inIndex] = original;
	(void)testSubmitAndWait(ctx);
	return (lossPlus - lossMinus) / (2.0f * inEpsilon);
}

// The fused LinearGelu forward must equal the unfused Gelu(Linear(x, W, b)) using
// the SAME weights — i.e. the fused GEMM epilogue computes the right GELU.
TEST(LinearGelu, ForwardMatchesUnfused) {
	constexpr oa::I32 kRows = 8;
	constexpr oa::I32 kIn   = 16;
	constexpr oa::I32 kOut  = 12;

	forceFp32Gemm();
	auto& ctx = oa::ExecutionSession::getActive();
	oa::ExecutionSession::RecordingScope scope(ctx);

	oa::Linear linear(kIn, kOut);
	linear.setActivation(oa::Activation::Gelu);
	for (auto& p : linear.parameters()) {
		p.data = oa::FnMatrix::scale(oa::FnMatrix::randN(p.data.getShape(), oa::ScalarType::Float32), 0.5f);
	}

	auto x = oa::FnMatrix::scale(oa::FnMatrix::randN(oa::MatrixShape{kRows, kIn}, oa::ScalarType::Float32), 0.5f);

	oa::Matrix fused, reference;
	{
		oa::GradNo noGrad;  // pure forward, no graph
		fused = linear.forward(x);
		// Unfused reference with identical weights.
		auto pre = oa::FnMatrix::linear(x, linear.parameters()[0].data, linear.parameters()[1].data);
		reference = oa::FnMatrix::gelu(pre);
		(void)testSubmitAndWait(ctx);
	}

	ASSERT_EQ(fused.numElements(), reference.numElements());
	const oa::F32* f = fused.dataAs<const oa::F32>();
	const oa::F32* r = reference.dataAs<const oa::F32>();
	oa::F32 maxAbsDiff = 0.0f;
	for (oa::I64 i = 0; i < fused.numElements(); ++i) {
		maxAbsDiff = std::max(maxAbsDiff, std::abs(f[i] - r[i]));
	}
	printf("\nLinearGelu fused vs unfused: max|diff| = %.6g\n", maxAbsDiff);
	EXPECT_LT(maxAbsDiff, 1e-3f) << "fused LinearGelu forward diverges from Gelu(Linear(x))";
}

// The fused backward (GradLinearGelu recomputes pre-activation for GeluBwd)
// must match finite-difference numerical gradients on both weight and bias.
TEST(LinearGelu, BackwardNumericalGrad) {
	constexpr oa::I32 kRows = 4;
	constexpr oa::I32 kIn   = 6;
	constexpr oa::I32 kOut  = 5;

	forceFp32Gemm();
	oa::FnMatrix::setRngSeed(2024);
	auto& ctx = oa::ExecutionSession::getActive();
	oa::ExecutionSession::RecordingScope scope(ctx);

	oa::Linear linear(kIn, kOut);
	linear.setActivation(oa::Activation::Gelu);
	for (auto& p : linear.parameters()) {
		p.data = oa::FnMatrix::scale(oa::FnMatrix::randN(p.data.getShape(), oa::ScalarType::Float32), 0.5f);
		p.data.setRequiresGrad(true);
	}

	auto input = oa::FnMatrix::scale(oa::FnMatrix::randN(oa::MatrixShape{kRows, kIn}, oa::ScalarType::Float32), 1.0f);
	input.setRequiresGrad(true);
	auto target = oa::FnMatrix::randN(oa::MatrixShape{kRows, kOut}, oa::ScalarType::Float32);

	oa::GradientTape tape;
	auto output = linear.forward(input);
	auto loss = oa::FnLoss::mse(output, target);
	tape.backward(loss);
	(void)testSubmitAndWait(ctx);

	struct ParamCheck { const char* name; oa::Usize index; };
	const ParamCheck checks[] = {{"weight", 0}, {"bias", 1}};

	oa::I32 numChecked = 0, numFailed = 0, numNonTrivial = 0;
	for (const auto& chk : checks) {
		auto  analytical = linear.parameters()[chk.index].grad().dataAs<const oa::F32>();
		auto& weight     = linear.parameters()[chk.index].data;
		const oa::I64 n    = weight.numElements();

		printf("\nLinearGelu backward grad check (%s):\n", chk.name);
		printf("index | Analytical | Numerical  | Close?\n");
		for (oa::I64 idx = 0; idx < n; ++idx) {
			auto lossFunc = [&]() -> oa::F32 {
				oa::GradNo noGrad;
				auto out = linear.forward(input);
				auto l = oa::FnLoss::mse(out, target);
				(void)testSubmitAndWait(ctx);
				return l.dataAs<const oa::F32>()[0];
			};
			oa::F32 num = computeNumericalGradient(lossFunc, weight, static_cast<oa::I32>(idx));
			oa::F32 ana = analytical[idx];
			const bool close = gradClose(ana, num);
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

TEST(LinearSilu, ForwardMatchesUnfused) {
	constexpr oa::I32 kRows = 8;
	constexpr oa::I32 kIn   = 16;
	constexpr oa::I32 kOut  = 12;

	forceFp32Gemm();
	auto& ctx = oa::ExecutionSession::getActive();
	oa::ExecutionSession::RecordingScope scope(ctx);

	oa::Linear linear(kIn, kOut);
	linear.setActivation(oa::Activation::Silu);
	for (auto& p : linear.parameters()) {
		p.data = oa::FnMatrix::scale(
			oa::FnMatrix::randN(p.data.getShape(), oa::ScalarType::Float32), 0.5F);
	}

	auto x = oa::FnMatrix::scale(
		oa::FnMatrix::randN(oa::MatrixShape{kRows, kIn}, oa::ScalarType::Float32), 0.5F);

	oa::Matrix fused, reference;
	{
		oa::GradNo noGrad;
		fused = linear.forward(x);
		auto pre = oa::FnMatrix::linear(
			x, linear.parameters()[0].data, linear.parameters()[1].data);
		reference = oa::FnMatrix::silu(pre);
		(void)testSubmitAndWait(ctx);
	}

	ASSERT_EQ(fused.numElements(), reference.numElements());
	const oa::F32* f = fused.dataAs<const oa::F32>();
	const oa::F32* r = reference.dataAs<const oa::F32>();
	oa::F32 maxAbsDiff = 0.0F;
	for (oa::I64 i = 0; i < fused.numElements(); ++i) {
		maxAbsDiff = std::max(maxAbsDiff, std::abs(f[i] - r[i]));
	}
	EXPECT_LT(maxAbsDiff, 1e-3F)
		<< "fused LinearSilu forward diverges from Silu(Linear(x))";
}

TEST(LinearSilu, BackwardNumericalGrad) {
	constexpr oa::I32 kRows = 4;
	constexpr oa::I32 kIn   = 6;
	constexpr oa::I32 kOut  = 5;

	forceFp32Gemm();
	oa::FnMatrix::setRngSeed(2025);
	auto& ctx = oa::ExecutionSession::getActive();
	oa::ExecutionSession::RecordingScope scope(ctx);

	oa::Linear linear(kIn, kOut);
	linear.setActivation(oa::Activation::Silu);
	for (auto& p : linear.parameters()) {
		p.data = oa::FnMatrix::scale(
			oa::FnMatrix::randN(p.data.getShape(), oa::ScalarType::Float32), 0.5F);
		p.data.setRequiresGrad(true);
	}

	auto input = oa::FnMatrix::scale(
		oa::FnMatrix::randN(oa::MatrixShape{kRows, kIn}, oa::ScalarType::Float32), 1.0F);
	input.setRequiresGrad(true);
	auto target = oa::FnMatrix::randN(
		oa::MatrixShape{kRows, kOut}, oa::ScalarType::Float32);

	oa::GradientTape tape;
	auto output = linear.forward(input);
	auto loss = oa::FnLoss::mse(output, target);
	tape.backward(loss);
	(void)testSubmitAndWait(ctx);

	struct ParamCheck { const char* name; oa::Usize index; };
	const ParamCheck checks[] = {{"weight", 0}, {"bias", 1}};

	oa::I32 numChecked = 0, numFailed = 0, numNonTrivial = 0;
	for (const auto& chk : checks) {
		auto analytical = linear.parameters()[chk.index].grad().dataAs<const oa::F32>();
		auto& parameter = linear.parameters()[chk.index].data;
		const oa::I64 count = parameter.numElements();
		for (oa::I64 idx = 0; idx < count; ++idx) {
			auto lossFunc = [&]() -> oa::F32 {
				oa::GradNo noGrad;
				auto out = linear.forward(input);
				auto currentLoss = oa::FnLoss::mse(out, target);
				(void)testSubmitAndWait(ctx);
				return currentLoss.dataAs<const oa::F32>()[0];
			};
			const oa::F32 numerical = computeNumericalGradient(
				lossFunc, parameter, static_cast<oa::I32>(idx));
			const oa::F32 analytic = analytical[idx];
			const bool close = gradClose(analytic, numerical);
			++numChecked;
			if (not close) ++numFailed;
			if (std::abs(numerical) > 1e-4F) ++numNonTrivial;
		}
	}

	EXPECT_EQ(numChecked, 35);
	EXPECT_EQ(numFailed, 0)
		<< "fused LinearSilu gradient failed numerical check";
	EXPECT_GE(numNonTrivial, 3) << "gradients all ~0 — check is vacuous";
}

// Regression test for the SMEM-overflow bug (commit 1398f5e):
// oa::FnMatrix::linear with NLP-scale shapes (M=1024, N=32, K=32) routes through
// addLinear → GemmBiasCmSgBf16 when N%16==0 and BF16 CoopMat is available.
// The fused kernel's smOut[BM*BN] = 64KB exceeded the 48KB SMEM limit, causing
// a silent hang on first dispatch. This test does NOT force FP32, so it
// exercises the actual BF16 fused path that NLP tutorials hit.
TEST(LinearGelu, FusedBf16NlpShapesNoHang) {
	auto& ctx = oa::ExecutionSession::getActive();
	oa::ExecutionSession::RecordingScope scope(ctx);

	// NLP-scale shapes: large batch, small output dim (divisible by 16).
	struct Shape { oa::I32 m; oa::I32 n; oa::I32 k; };
	static const Shape kShapes[] = {
		{1024, 32, 32}, {512, 64, 32}, {1024, 32, 64}, {256, 16, 16},
	};

	for (const auto& sh : kShapes) {
		auto x = oa::FnMatrix::randN(oa::MatrixShape{sh.m, sh.k}, oa::ScalarType::Float32);
		auto w = oa::FnMatrix::randN(oa::MatrixShape{sh.n, sh.k}, oa::ScalarType::Float32);
		auto b = oa::FnMatrix::randN(oa::MatrixShape{sh.n}, oa::ScalarType::Float32);

		auto y = oa::FnMatrix::linear(x, w, b);
		(void)testSubmitAndWait(ctx);

		expectShape(y, {sh.m, sh.n});
		expectFinite(y);
	}
}
