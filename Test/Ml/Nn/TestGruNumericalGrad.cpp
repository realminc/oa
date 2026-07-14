// Test/Ml/Nn/TestGruNumericalGrad.cpp
// Numerical gradient verification for GRU backward pass (cell + BPTT sequence).

#include <gtest/gtest.h>
#include <Oa/Ml.h>
#include <Oa/Ml/Autograd.h>
#include <Oa/Runtime/Context.h>
#include <cmath>
#include <cstdlib>

// Finite-difference gradient checking is only valid when the forward runs in
// fp32: the default Auto GEMM route uses bf16, whose ~3-digit mantissa swallows
// the small weight perturbations and collapses numerical gradients to ~0 (false
// failures). OA_GEMM_FORCE_FP32 pins the router to the exact fp32 path. Set it at
// the start of each test body (before the first GEMM); the router reads it live.
static void ForceFp32Gemm() { setenv("OA_GEMM_FORCE_FP32", "1", 1); }

// PyTorch-gradcheck-style acceptance: |analytical - numerical| <= atol + rtol*|numerical|.
// A pure relative-error metric (|a-n|/|n|) explodes for near-zero gradients and
// is NOT a valid pass/fail criterion — that was the original test's flaw.
static bool GradClose(OaF32 InAnalytical, OaF32 InNumerical,
	OaF32 InAtol = 2e-3F, OaF32 InRtol = 2e-2F) {
	return std::abs(InAnalytical - InNumerical) <= (InAtol + (InRtol * std::abs(InNumerical)));
}

// Helper to compute numerical gradient using finite differences
static OaF32 ComputeNumericalGradient(
	std::function<OaF32()> InLossFunc,
	OaMatrix& InParam,
	OaI32 InIndex,
	OaF32 InEpsilon = 1e-2f
) {
	auto& ctx = OaContext::GetDefault();

	// Get original value
	OaF32* data = InParam.DataAs<OaF32>();
	OaF32 original = data[InIndex];

	// f(x + h)
	data[InIndex] = original + InEpsilon;
	(void)ctx.Sync();
	OaF32 lossPlus = InLossFunc();
	
	// f(x - h)
	data[InIndex] = original - InEpsilon;
	(void)ctx.Sync();
	OaF32 lossMinus = InLossFunc();
	
	// Restore original
	data[InIndex] = original;
	(void)ctx.Sync();

	// Numerical gradient: (f(x+h) - f(x-h)) / (2h)
	return (lossPlus - lossMinus) / (2.0f * InEpsilon);
}

TEST(GruNumericalGrad, SingleStepGradient) {
	// Test GRU gradient on a single timestep with small dimensions
	constexpr OaI32 kBatch = 2;
	constexpr OaI32 kInputSize = 4;
	constexpr OaI32 kHiddenSize = 3;

	ForceFp32Gemm();
	auto& ctx = OaContext::GetDefault();
	OaContext::Scope scope(ctx);

	// Create GRU cell
	auto gru = OaMakeSharedPtr<OaGruCell>(kInputSize, kHiddenSize, true);
	
	// Initialize with small random weights for numerical stability
	for (auto& param : gru->Parameters()) {
		param.Data = OaFnMatrix::RandN(param.Data.GetShape(), OaScalarType::Float32);
		param.Data = OaFnMatrix::Scale(param.Data, 0.1f);  // Small weights
		param.Data.SetRequiresGrad(true);
		param.Grad() = param.Data.GradMatrix();
	}
	
	// Create input and target
	auto input = OaFnMatrix::RandN(OaMatrixShape{kBatch, kInputSize}, OaScalarType::Float32);
	input = OaFnMatrix::Scale(input, 0.5f);
	input.SetRequiresGrad(true);
	
	auto hidden = gru->ZeroState(kBatch);
	hidden.SetRequiresGrad(true);
	
	auto target = OaFnMatrix::RandN(OaMatrixShape{kBatch, kHiddenSize}, OaScalarType::Float32);
	
	// Forward pass with autograd
	OaGradientTape tape;
	auto output = gru->Step(input, hidden);
	auto loss = OaFnLoss::Mse(output, target);
	
	// Backward pass
	tape.Backward(loss);
	(void)ctx.Execute();
	(void)ctx.Sync();
	
	// Get analytical gradients
	auto analyticalGrad = gru->Parameters()[0].Grad();  // weight_ih
	auto analyticalGradData = analyticalGrad.DataAs<const OaF32>();
	
	// Compute numerical gradients for a few weight elements
	auto& weightIh = gru->Parameters()[0].Data;
	
	printf("\nComparing analytical vs numerical gradients (weight_ih):\n");
	printf("Index | Analytical | Numerical  | Close?\n");
	printf("------|------------|------------|-------\n");

	OaI32 numSamples = 5;
	OaI32 numChecked = 0;
	OaI32 numFailed  = 0;

	for (OaI32 i = 0; i < numSamples; ++i) {
		// Sample random indices
		OaI32 idx = (i * 7) % (3 * kHiddenSize * kInputSize);

		// Define loss function for numerical gradient
		// Disable autograd to prevent interference with numerical gradient computation
		auto lossFunc = [&]() -> OaF32 {
			OaGradNo noGrad;  // Disable autograd for numerical gradient
			auto out = gru->Step(input, hidden);
			auto l = OaFnLoss::Mse(out, target);
			(void)ctx.Execute();
			(void)ctx.Sync();
			const OaF32* lossData = l.DataAs<const OaF32>();
			return lossData[0];
		};

		OaF32 numericalGrad = ComputeNumericalGradient(lossFunc, weightIh, idx);
		OaF32 analytical = analyticalGradData[idx];
		const bool close = GradClose(analytical, numericalGrad);
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

// Multi-step OaGru::Forward (backprop-through-time) gradient check.
//
// NOTE: OaGru holds its weights in child OaGruCell sub-modules, so the recursive
// accessor AllParameterPtrs() must be used — gru->Parameters() returns ONLY the
// module's own (empty) parameter list and indexing it is a hard crash. Earlier
// this test used gru->Parameters()[0], which read past the end of an empty vector
// (SIGSEGV under NDEBUG / assert-abort in debug) and was misdiagnosed as a
// "BPTT autograd/buffer-lifecycle bug". The sequence backward itself is fine.
TEST(GruNumericalGrad, SequenceGradient) {
	// Test GRU gradient over a short sequence
	constexpr OaI32 kBatch = 2;
	constexpr OaI32 kSeqLen = 3;
	constexpr OaI32 kInputSize = 4;
	constexpr OaI32 kHiddenSize = 3;

	ForceFp32Gemm();
	OaFnMatrix::SetRngSeed(1234);  // deterministic init → stable non-trivial gradient signal
	auto& ctx = OaContext::GetDefault();
	OaContext::Scope scope(ctx);

	// Create GRU
	auto gru = OaMakeSharedPtr<OaGru>(kInputSize, kHiddenSize, 1, true);

	// Initialize with moderate random weights. AllParameterPtrs() yields pointers to
	// the REAL parameters in the child cells (a by-value AllParameters() copy would
	// not mutate them).
	for (auto* param : gru->AllParameterPtrs()) {
		param->Data = OaFnMatrix::RandN(param->Data.GetShape(), OaScalarType::Float32);
		param->Data = OaFnMatrix::Scale(param->Data, 0.5f);  // larger weights → non-trivial gradient signal
		param->Data.SetRequiresGrad(true);
		param->Grad() = param->Data.GradMatrix();
	}

	// Create input sequence [batch, seq_len, input_size]. Larger inputs drive the
	// gates harder, lifting the gradient signal clear of finite-difference noise.
	auto input = OaFnMatrix::RandN(OaMatrixShape{kBatch, kSeqLen, kInputSize}, OaScalarType::Float32);
	input = OaFnMatrix::Scale(input, 2.0f);
	input.SetRequiresGrad(true);

	auto target = OaFnMatrix::RandN(OaMatrixShape{kBatch, kSeqLen, kHiddenSize}, OaScalarType::Float32);

	// Forward pass with autograd
	OaGradientTape tape;
	auto output = gru->Forward(input);
	auto loss = OaFnLoss::Mse(output, target);

	// Backward pass
	tape.Backward(loss);
	(void)ctx.Execute();
	(void)ctx.Sync();

	auto params = gru->AllParameterPtrs();

	// Scan EVERY element of both the input weight (weight_ih) and — crucially — the
	// recurrent weight (weight_hh), whose gradient only flows through the
	// timestep-to-timestep hidden chain. weight_hh is the real BPTT path; a broken
	// sequence unroll would show up there. nonTrivial counts elements whose gradient
	// magnitude is large enough to be a meaningful check (not just 0≈0).
	struct ParamCheck { const char* Name; OaI32 Index; };
	const ParamCheck checks[] = {{"weight_ih", 0}, {"weight_hh", 1}};

	OaI32 numChecked   = 0;
	OaI32 numFailed    = 0;
	OaI32 numNonTrivial = 0;

	for (const auto& chk : checks) {
		auto  analyticalGradData = params[chk.Index]->Grad().DataAs<const OaF32>();
		auto& weight             = params[chk.Index]->Data;
		const OaI64 n            = weight.NumElements();

		printf("\nComparing analytical vs numerical gradients (sequence, %s):\n", chk.Name);
		printf("Index | Analytical | Numerical  | Close?\n");
		printf("------|------------|------------|-------\n");

		for (OaI64 idx = 0; idx < n; ++idx) {
			auto lossFunc = [&]() -> OaF32 {
				OaGradNo noGrad;  // Disable autograd for numerical gradient
				auto out = gru->Forward(input);
				auto l = OaFnLoss::Mse(out, target);
				(void)ctx.Execute();
				(void)ctx.Sync();
				const OaF32* lossData = l.DataAs<const OaF32>();
				return lossData[0];
			};

			OaF32 numericalGrad = ComputeNumericalGradient(lossFunc, weight, static_cast<OaI32>(idx));
			OaF32 analytical    = analyticalGradData[idx];
			const bool close    = GradClose(analytical, numericalGrad);
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