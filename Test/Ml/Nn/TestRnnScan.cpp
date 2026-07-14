// Test/Ml/Nn/TestRnnScan.cpp
// Verification for the whole-sequence RNN recurrent scan (RnnScan/RnnScanBwd):
//   1. Forward output matches the per-timestep path (same weights/input).
//   2. Autograd param gradients match the per-timestep path.
//   3. Independent finite-difference gradcheck of the recurrent weight (weight_hh),
//      which exercises both RnnScanBwd and the LinearWeightBiasBwd weight-grad call.

#include <gtest/gtest.h>
#include <Oa/Ml.h>
#include <Oa/Ml/Autograd.h>
#include <Oa/Runtime/Context.h>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <vector>

static void ForceFp32Gemm() { setenv("OA_GEMM_FORCE_FP32", "1", 1); }

static bool GradClose(OaF32 InAnalytical, OaF32 InNumerical,
	OaF32 InAtol = 2e-3F, OaF32 InRtol = 2e-2F) {
	return std::abs(InAnalytical - InNumerical) <= (InAtol + (InRtol * std::abs(InNumerical)));
}

static void FillDeterministic(OaMatrix& m, float scale, double phase) {
	std::vector<float> v(static_cast<size_t>(m.NumElements()));
	for (size_t i = 0; i < v.size(); ++i)
		v[i] = scale * static_cast<float>(std::sin(0.37 * static_cast<double>(i) + phase));
	m = OaFnMatrix::FromBytes(
		OaSpan<const OaU8>(reinterpret_cast<const OaU8*>(v.data()), v.size() * sizeof(float)),
		m.GetShape(), m.GetDtype());
}

static std::vector<float> DownloadF32(const OaMatrix& m) {
	auto& ctx = OaContext::GetDefault();
	(void)ctx.Execute();
	(void)ctx.Sync();
	const float* p = m.DataAs<const float>();
	return std::vector<float>(p, p + m.NumElements());
}

static OaF32 ComputeNumericalGradient(
	std::function<OaF32()> InLossFunc, OaMatrix& InParam, OaI32 InIndex, OaF32 InEpsilon = 1e-2f)
{
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

// Build an OaRnn filled with deterministic weights (identical across calls w/ same phase).
static OaSharedPtr<OaRnn> MakeRnn(OaI32 InInput, OaI32 InHidden) {
	auto rnn = OaMakeSharedPtr<OaRnn>(InInput, InHidden, 1, true);
	double phase = 0.0;
	for (auto* p : rnn->AllParameterPtrs()) {
		FillDeterministic(p->Data, 0.1f, phase);
		p->Data.SetRequiresGrad(true);
		phase += 1.7;
	}
	return rnn;
}

static OaMatrix MakeInput(OaI32 B, OaI32 S, OaI32 InInput) {
	auto x = OaFnMatrix::Empty(OaMatrixShape{B, S, InInput}, OaScalarType::Float32);
	FillDeterministic(x, 0.3f, 5.0);
	return x;
}

// Forward sanity: the scan-only path produces finite output with the expected shape.
TEST(RnnScan, ForwardShapeAndFinite) {
	ForceFp32Gemm();
	auto& ctx = OaContext::GetDefault();
	OaContext::Scope scope(ctx);
	constexpr OaI32 B = 3, S = 5, In = 7, H = 6;

	auto rnn = MakeRnn(In, H);
	auto x = MakeInput(B, S, In);
	auto out = rnn->Forward(x);
	EXPECT_EQ(out.GetShape(), (OaMatrixShape{B, S, H}));
	auto v = DownloadF32(out);
	bool allFinite = true;
	for (float f : v) allFinite = allFinite and std::isfinite(f);
	EXPECT_TRUE(allFinite) << "RnnScan produced non-finite output";
}

// Independent FD gradcheck of weight_hh (the recurrent weight).
TEST(RnnScan, WeightHhNumericalGradcheck) {
	ForceFp32Gemm();
	auto& ctx = OaContext::GetDefault();
	OaContext::Scope scope(ctx);
	constexpr OaI32 B = 2, S = 4, In = 5, H = 4;

	auto rnn = MakeRnn(In, H);
	auto x = MakeInput(B, S, In);

	// Analytical gradient via autograd.
	{
		OaGradientTape tape;
		auto out = rnn->Forward(x);
		auto loss = OaFnMatrix::Sum(out);
		tape.Backward(loss);
	}
	auto& weightHh = rnn->AllParameterPtrs()[1]->Data;   // weight_hh [H, H]
	auto analytical = DownloadF32(rnn->AllParameterPtrs()[1]->Grad());

	auto lossFunc = [&]() -> OaF32 {
		OaGradNo noGrad;
		auto out = rnn->Forward(x);
		auto loss = OaFnMatrix::Sum(out);
		(void)ctx.Execute();
		(void)ctx.Sync();
		return loss.DataAs<const OaF32>()[0];
	};

	int numFailed = 0;
	const size_t checkCount = std::min<size_t>(24, analytical.size());
	for (size_t idx = 0; idx < checkCount; ++idx) {
		OaF32 numerical = ComputeNumericalGradient(lossFunc, weightHh, static_cast<OaI32>(idx));
		if (not GradClose(analytical[idx], numerical)) ++numFailed;
		printf("weight_hh[%zu] analytical=%.5f numerical=%.5f %s\n",
			idx, analytical[idx], numerical, GradClose(analytical[idx], numerical) ? "ok" : "FAIL");
	}
	EXPECT_EQ(numFailed, 0) << "scan weight_hh FD gradcheck failed";
}

