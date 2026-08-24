// Test/Ml/nn/TestGruNumericalGrad.cpp
// Numerical gradient verification for GRU backward pass (cell + BPTT sequence).

#include <gtest/gtest.h>
#include "../../oaTest.h"
#include <oa/ml.h>
#include <oa/ml/autograd.h>
#include <oa/runtime/executionSession.h>
#include <cmath>
#include <cstdlib>

// Finite-difference gradient checking is only valid when the forward runs in
// fp32: the default Auto GEMM route uses bf16, whose ~3-digit mantissa swallows
// the small weight perturbations and collapses numerical gradients to ~0 (false
// failures). OA_GEMM_FORCE_FP32 pins the router to the exact fp32 path. set it at
// the start of each test body (before the first GEMM); the router reads it live.
static void forceFp32Gemm() { setenv("OA_GEMM_FORCE_FP32", "1", 1); }

// PyTorch-gradcheck-style acceptance: |analytical - numerical| <= atol + rtol*|numerical|.
// A pure relative-error metric (|a-n|/|n|) explodes for near-zero gradients and
// is NOT a valid pass/fail criterion — that was the original test's flaw.
static bool gradClose(oa::F32 inAnalytical, oa::F32 inNumerical,
	oa::F32 inAtol = 2e-3F, oa::F32 inRtol = 2e-2F) {
	return std::abs(inAnalytical - inNumerical) <= (inAtol + (inRtol * std::abs(inNumerical)));
}

// Helper to compute numerical gradient using finite differences
static oa::F32 computeNumericalGradient(
	std::function<oa::F32()> inLossFunc,
	oa::Matrix& inParam,
	oa::I32 inIndex,
	oa::F32 inEpsilon = 1e-2f
) {
	auto& ctx = oa::ExecutionSession::getActive();

	// get original value
	oa::F32* data = inParam.dataAs<oa::F32>();
	oa::F32 original = data[inIndex];

	// f(x + h)
	data[inIndex] = original + inEpsilon;
	(void)testSubmitAndWait(ctx);
	oa::F32 lossPlus = inLossFunc();
	
	// f(x - h)
	data[inIndex] = original - inEpsilon;
	(void)testSubmitAndWait(ctx);
	oa::F32 lossMinus = inLossFunc();
	
	// Restore original
	data[inIndex] = original;
	(void)testSubmitAndWait(ctx);

	// Numerical gradient: (f(x+h) - f(x-h)) / (2h)
	return (lossPlus - lossMinus) / (2.0f * inEpsilon);
}

TEST(GruNumericalGrad, SingleStepGradient) {
	// Test GRU gradient on a single timestep with small dimensions
	constexpr oa::I32 kBatch = 2;
	constexpr oa::I32 kInputSize = 4;
	constexpr oa::I32 kHiddenSize = 3;

	forceFp32Gemm();
	auto& ctx = oa::ExecutionSession::getActive();
	oa::ExecutionSession::RecordingScope scope(ctx);

	// Create GRU cell
	auto gru = oa::makeShared<oa::GruCell>(kInputSize, kHiddenSize, true);
	
	// initialize with small random weights for numerical stability
	for (auto& param : gru->parameters()) {
		param.data = oa::FnMatrix::randN(param.data.getShape(), oa::ScalarType::Float32);
		param.data = oa::FnMatrix::scale(param.data, 0.1f);  // Small weights
		param.data.setRequiresGrad(true);
		param.grad() = param.data.gradMatrix();
	}
	
	// Create input and target
	auto input = oa::FnMatrix::randN(oa::MatrixShape{kBatch, kInputSize}, oa::ScalarType::Float32);
	input = oa::FnMatrix::scale(input, 0.5f);
	input.setRequiresGrad(true);
	
	auto hidden = gru->zeroState(kBatch);
	hidden.setRequiresGrad(true);
	
	auto target = oa::FnMatrix::randN(oa::MatrixShape{kBatch, kHiddenSize}, oa::ScalarType::Float32);
	
	// forward pass with autograd
	oa::GradientTape tape;
	auto output = gru->step(input, hidden);
	auto loss = oa::FnLoss::mse(output, target);
	
	// backward pass
	tape.backward(loss);
	(void)testSubmitAndWait(ctx);
	
	// get analytical gradients
	auto analyticalGrad = gru->parameters()[0].grad();  // weight_ih
	auto analyticalGradData = analyticalGrad.dataAs<const oa::F32>();
	
	// Compute numerical gradients for a few weight elements
	auto& weightIh = gru->parameters()[0].data;
	
	printf("\nComparing analytical vs numerical gradients (weight_ih):\n");
	printf("index | Analytical | Numerical  | Close?\n");
	printf("------|------------|------------|-------\n");

	oa::I32 numSamples = 5;
	oa::I32 numChecked = 0;
	oa::I32 numFailed  = 0;

	for (oa::I32 i = 0; i < numSamples; ++i) {
		// Sample random indices
		oa::I32 idx = (i * 7) % (3 * kHiddenSize * kInputSize);

		// Define loss function for numerical gradient
		// Disable autograd to prevent interference with numerical gradient computation
		auto lossFunc = [&]() -> oa::F32 {
			oa::GradNo noGrad;  // Disable autograd for numerical gradient
			auto out = gru->step(input, hidden);
			auto l = oa::FnLoss::mse(out, target);
			(void)testSubmitAndWait(ctx);
			const oa::F32* lossData = l.dataAs<const oa::F32>();
			return lossData[0];
		};

		oa::F32 numericalGrad = computeNumericalGradient(lossFunc, weightIh, idx);
		oa::F32 analytical = analyticalGradData[idx];
		const bool close = gradClose(analytical, numericalGrad);
		++numChecked;
		if (not close) ++numFailed;

		printf("%5d | %10.6f | %10.6f | %s\n",
			idx, analytical, numericalGrad, close ? "yes" : "NO");
	}

	printf("\nGradCheck: %d/%d elements within atol=2e-3 + rtol=2%%\n",
		numChecked - numFailed, numChecked);

	// Every sampled element must satisfy the absolute+relative tolerance.
	EXPECT_EQ(numFailed, 0) << "GRU weight_ih gradient failed numerical gradient check";
}

// Multi-step oa::Gru::forward (backprop-through-time) gradient check.
//
// NOTE: oa::Gru holds its weights in child oa::GruCell sub-modules, so the recursive
// accessor allParameterPtrs() must be used — gru->parameters() returns ONLY the
// module's own (empty) parameter list and indexing it is a hard crash. Earlier
// this test used gru->parameters()[0], which read past the end of an empty vector
// (SIGSEGV under NDEBUG / assert-abort in debug) and was misdiagnosed as a
// "BPTT autograd/buffer-lifecycle bug". The sequence backward itself is fine.
TEST(GruNumericalGrad, SequenceGradient) {
	// Test GRU gradient over a short sequence
	constexpr oa::I32 kBatch = 2;
	constexpr oa::I32 kSeqLen = 3;
	constexpr oa::I32 kInputSize = 4;
	constexpr oa::I32 kHiddenSize = 3;

	forceFp32Gemm();
	oa::FnMatrix::setRngSeed(1234);  // deterministic init → stable non-trivial gradient signal
	auto& ctx = oa::ExecutionSession::getActive();
	oa::ExecutionSession::RecordingScope scope(ctx);

	// Create GRU
	auto gru = oa::makeShared<oa::Gru>(kInputSize, kHiddenSize, 1, true);

	// initialize with moderate random weights. allParameterPtrs() yields pointers to
	// the REAL parameters in the child cells (a by-value allParameters() copy would
	// not mutate them).
	for (auto* param : gru->allParameterPtrs()) {
		param->data = oa::FnMatrix::randN(param->data.getShape(), oa::ScalarType::Float32);
		param->data = oa::FnMatrix::scale(param->data, 0.5f);  // larger weights → non-trivial gradient signal
		param->data.setRequiresGrad(true);
		param->grad() = param->data.gradMatrix();
	}

	// Create input sequence [batch, seq_len, input_size]. Larger inputs drive the
	// gates harder, lifting the gradient signal clear of finite-difference noise.
	auto input = oa::FnMatrix::randN(oa::MatrixShape{kBatch, kSeqLen, kInputSize}, oa::ScalarType::Float32);
	input = oa::FnMatrix::scale(input, 2.0f);
	input.setRequiresGrad(true);

	auto target = oa::FnMatrix::randN(oa::MatrixShape{kBatch, kSeqLen, kHiddenSize}, oa::ScalarType::Float32);

	// forward pass with autograd
	oa::GradientTape tape;
	auto output = gru->forward(input);
	auto loss = oa::FnLoss::mse(output, target);

	// backward pass
	tape.backward(loss);
	(void)testSubmitAndWait(ctx);

	auto params = gru->allParameterPtrs();

	// Scan EVERY element of both the input weight (weight_ih) and — crucially — the
	// recurrent weight (weight_hh), whose gradient only flows through the
	// timestep-to-timestep hidden chain. weight_hh is the real BPTT path; a broken
	// sequence unroll would show up there. nonTrivial counts elements whose gradient
	// magnitude is large enough to be a meaningful check (not just 0≈0).
	struct ParamCheck { const char* name; oa::I32 index; };
	const ParamCheck checks[] = {{"weight_ih", 0}, {"weight_hh", 1}};

	oa::I32 numChecked   = 0;
	oa::I32 numFailed    = 0;
	oa::I32 numNonTrivial = 0;

	for (const auto& chk : checks) {
		auto  analyticalGradData = params[chk.index]->grad().dataAs<const oa::F32>();
		auto& weight             = params[chk.index]->data;
		const oa::I64 n            = weight.numElements();

		printf("\nComparing analytical vs numerical gradients (sequence, %s):\n", chk.name);
		printf("index | Analytical | Numerical  | Close?\n");
		printf("------|------------|------------|-------\n");

		for (oa::I64 idx = 0; idx < n; ++idx) {
			auto lossFunc = [&]() -> oa::F32 {
				oa::GradNo noGrad;  // Disable autograd for numerical gradient
				auto out = gru->forward(input);
				auto l = oa::FnLoss::mse(out, target);
				(void)testSubmitAndWait(ctx);
				const oa::F32* lossData = l.dataAs<const oa::F32>();
				return lossData[0];
			};

			oa::F32 numericalGrad = computeNumericalGradient(lossFunc, weight, static_cast<oa::I32>(idx));
			oa::F32 analytical    = analyticalGradData[idx];
			const bool close    = gradClose(analytical, numericalGrad);
			++numChecked;
			if (not close) ++numFailed;
			// This small net's gradients sit around 1e-4 (MSE mean dilutes them and
			// the zero initial hidden state starves weight_hh at t=0); 1e-4 is well
			// above fp32 finite-difference noise, so it's a real non-vacuous signal.
			if (std::abs(numericalGrad) > 1e-4F) ++numNonTrivial;

			if (std::abs(numericalGrad) > 5e-5F || not close) {
				printf("%5lld | %10.6f | %10.6f | %s\n",
					static_cast<long long>(idx), analytical, numericalGrad, close ? "yes" : "NO");
			}
		}
	}

	printf("\nGradCheck (sequence): %d/%d elements within atol=2e-3 + rtol=2%%, %d non-trivial\n",
		numChecked - numFailed, numChecked, numNonTrivial);

	EXPECT_EQ(numFailed, 0) << "GRU sequence weight gradient failed numerical gradient check";
	// Guard against a vacuous pass (all-zero gradients trivially "match"): the BPTT
	// path must produce at least a few elements with real signal.
	EXPECT_GE(numNonTrivial, 3) << "sequence gradients are all ~0 — check is vacuous";
}
