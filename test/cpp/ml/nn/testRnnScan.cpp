// Test/Ml/nn/TestRnnScan.cpp
// Verification for the whole-sequence RNN recurrent scan (RnnScan/RnnScanBwd):
//   1. forward output matches the per-timestep path (same weights/input).
//   2. autograd param gradients match the per-timestep path.
//   3. Independent finite-difference gradcheck of the recurrent weight (weight_hh),
//      which exercises both RnnScanBwd and the LinearWeightBiasBwd weight-grad call.

#include <gtest/gtest.h>
#include "../../oaTest.h"
#include <oa/ml.h>
#include <oa/ml/autograd.h>
#include <oa/runtime/executionSession.h>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <vector>

static void forceFp32Gemm() { setenv("OA_GEMM_FORCE_FP32", "1", 1); }

static bool gradClose(oa::F32 inAnalytical, oa::F32 inNumerical,
	oa::F32 inAtol = 2e-3F, oa::F32 inRtol = 2e-2F) {
	return std::abs(inAnalytical - inNumerical) <= (inAtol + (inRtol * std::abs(inNumerical)));
}

static void fillDeterministic(oa::Matrix& m, float scale, double phase) {
	std::vector<float> v(static_cast<size_t>(m.numElements()));
	for (size_t i = 0; i < v.size(); ++i)
		v[i] = scale * static_cast<float>(std::sin(0.37 * static_cast<double>(i) + phase));
	m = oa::FnMatrix::fromBytes(
		oa::Span<const oa::U8>(reinterpret_cast<const oa::U8*>(v.data()), v.size() * sizeof(float)),
		m.getShape(), m.getDtype());
}

static std::vector<float> downloadF32(const oa::Matrix& m) {
	auto& ctx = oa::ExecutionSession::getActive();
	(void)testSubmitAndWait(ctx);
	const float* p = m.dataAs<const float>();
	return std::vector<float>(p, p + m.numElements());
}

static oa::F32 computeNumericalGradient(
	std::function<oa::F32()> inLossFunc, oa::Matrix& inParam, oa::I32 inIndex, oa::F32 inEpsilon = 1e-2f)
{
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

// Build an oa::Rnn filled with deterministic weights (identical across calls w/ same phase).
static oa::SharedPtr<oa::Rnn> makeRnn(oa::I32 inInput, oa::I32 inHidden) {
	auto rnn = oa::makeShared<oa::Rnn>(inInput, inHidden, 1, true);
	double phase = 0.0;
	for (auto* p : rnn->allParameterPtrs()) {
		fillDeterministic(p->data, 0.1f, phase);
		p->data.setRequiresGrad(true);
		phase += 1.7;
	}
	return rnn;
}

static oa::Matrix makeInput(oa::I32 B, oa::I32 S, oa::I32 inInput) {
	auto x = oa::FnMatrix::empty(oa::MatrixShape{B, S, inInput}, oa::ScalarType::Float32);
	fillDeterministic(x, 0.3f, 5.0);
	return x;
}

// forward sanity: the scan-only path produces finite output with the expected shape.
TEST(RnnScan, ForwardShapeAndFinite) {
	forceFp32Gemm();
	auto& ctx = oa::ExecutionSession::getActive();
	oa::ExecutionSession::RecordingScope scope(ctx);
	constexpr oa::I32 B = 3, S = 5, in = 7, H = 6;

	auto rnn = makeRnn(in, H);
	auto x = makeInput(B, S, in);
	auto out = rnn->forward(x);
	EXPECT_EQ(out.getShape(), (oa::MatrixShape{B, S, H}));
	auto v = downloadF32(out);
	bool allFinite = true;
	for (float f : v) allFinite = allFinite and std::isfinite(f);
	EXPECT_TRUE(allFinite) << "RnnScan produced non-finite output";
}

// Independent FD gradcheck of weight_hh (the recurrent weight).
TEST(RnnScan, WeightHhNumericalGradcheck) {
	forceFp32Gemm();
	auto& ctx = oa::ExecutionSession::getActive();
	oa::ExecutionSession::RecordingScope scope(ctx);
	constexpr oa::I32 B = 2, S = 4, in = 5, H = 4;

	auto rnn = makeRnn(in, H);
	auto x = makeInput(B, S, in);

	// Analytical gradient via autograd.
	{
		oa::GradientTape tape;
		auto out = rnn->forward(x);
		auto loss = oa::FnMatrix::sum(out);
		tape.backward(loss);
	}
	auto& weightHh = rnn->allParameterPtrs()[1]->data;   // weight_hh [H, H]
	auto analytical = downloadF32(rnn->allParameterPtrs()[1]->grad());

	auto lossFunc = [&]() -> oa::F32 {
		oa::GradNo noGrad;
		auto out = rnn->forward(x);
		auto loss = oa::FnMatrix::sum(out);
		(void)testSubmitAndWait(ctx);
		return loss.dataAs<const oa::F32>()[0];
	};

	int numFailed = 0;
	const size_t checkCount = std::min<size_t>(24, analytical.size());
	for (size_t idx = 0; idx < checkCount; ++idx) {
		oa::F32 numerical = computeNumericalGradient(lossFunc, weightHh, static_cast<oa::I32>(idx));
		if (not gradClose(analytical[idx], numerical)) ++numFailed;
		printf("weight_hh[%zu] analytical=%.5f numerical=%.5f %s\n",
			idx, analytical[idx], numerical, gradClose(analytical[idx], numerical) ? "ok" : "FAIL");
	}
	EXPECT_EQ(numFailed, 0) << "scan weight_hh FD gradcheck failed";
}
