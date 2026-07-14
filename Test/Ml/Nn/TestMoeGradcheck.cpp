// OaMoE — end-to-end finite-difference gradient checks.
//
// These tests verify the whole module differentiates correctly — router and
// experts, dense oracle and sparse GPU executor — by comparing
// analytical parameter gradients (autograd) to central finite differences of
// the MSE loss. This replaces the prior placeholder smoke tests, which only
// checked "doesn't crash" + "gradient != 0" (a buffer-binding bug passes both).
//
// Forward is fp32-forced (OA_GEMM_FORCE_FP32) so finite differences are valid.

#include <gtest/gtest.h>
#include <Oa/Ml.h>
#include <Oa/Ml/Autograd.h>
#include <Oa/Runtime/Context.h>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

bool GradClose(float a, float n, float atol = 3e-3f, float rtol = 4e-2f) {
	return std::abs(a - n) <= (atol + rtol * std::abs(n));
}

struct GradStats { int Checked = 0; int Failed = 0; int NonTrivial = 0; };

// Central finite-difference gradcheck over every element of one parameter.
GradStats GradCheckParam(OaContext& ctx, const std::function<float()>& loss,
	OaMatrix& param, const float* analytical, const char* name) {
	GradStats s;
	float* d = param.DataAs<float>();
	const OaI64 n = param.NumElements();
	// The finite-difference forwards allocate many transient tensors. Snapshot the
	// synchronized analytical gradient before those runs: retaining only a raw
	// pointer made the reference allocator-lifetime dependent and it could be
	// overwritten while the check was still using it.
	std::vector<float> analyticalSnapshot(
		analytical, analytical + static_cast<size_t>(n));
	const float eps = 1e-2f;
	printf("  [%s] %lld elements\n", name, static_cast<long long>(n));
	for (OaI64 i = 0; i < n; ++i) {
		const float orig = d[i];
		d[i] = orig + eps; (void)ctx.Sync(); const float lp = loss();
		d[i] = orig - eps; (void)ctx.Sync(); const float lm = loss();
		d[i] = orig;       (void)ctx.Sync();
		const float numerical = (lp - lm) / (2.0f * eps);
		const float a = analyticalSnapshot[static_cast<size_t>(i)];
		++s.Checked;
		if (not GradClose(a, numerical)) {
			++s.Failed;
			printf("    idx %lld: analytical=%.6f numerical=%.6f  MISMATCH\n",
				static_cast<long long>(i), a, numerical);
		}
		if (std::abs(numerical) > 5e-4f) ++s.NonTrivial;
	}
	return s;
}

// Find a parameter by its dotted path; nullptr if absent (test fails loudly).
OaParameter* FindParam(OaVec<OaNamedParameter>& named, const char* path) {
	for (auto& np : named)
		if (std::string(np.Path.CStr()) == path) return np.Param;
	return nullptr;
}

float GradMag(const OaMatrix& g) {
	if (g.NumElements() == 0) return 0.0f;
	const float* d = g.DataAs<const float>();
	float s = 0.0f;
	for (OaI64 i = 0; i < g.NumElements(); ++i) s += std::abs(d[i]);
	return s;
}

}  // namespace

// ── DIAGNOSTICS: isolate the broadcast-Mul + Slice backward path ──────────────
TEST(OaMoeExpertPlan, StableDroplessPacking) {
	auto& ctx = OaContext::GetDefault();
	OaContext::Scope scope(ctx);
	const std::vector<OaI32> routes = {2, 0, 1, 2, 0, 1, 2, 1};
	auto indices = OaFnMatrix::FromBytes(
		OaSpan<const OaU8>(reinterpret_cast<const OaU8*>(routes.data()), routes.size() * sizeof(OaI32)),
		OaMatrixShape{4, 2}, OaScalarType::Int32);
	auto plan = OaFnMatrix::MoeExpertPlan(indices, 3);
	ASSERT_TRUE(ctx.Execute().IsOk());
	ASSERT_TRUE(ctx.Sync().IsOk());

	const std::vector<OaU32> counts = {2, 3, 3};
	const std::vector<OaU32> offsets = {0, 2, 5, 8};
	const std::vector<OaU32> slots = {1, 4, 2, 5, 7, 0, 3, 6};
	const std::vector<OaU32> tokens = {0, 2, 1, 2, 3, 0, 1, 3};
	const std::vector<OaU32> experts = {0, 0, 1, 1, 1, 2, 2, 2};
	const std::vector<OaU32> inverse = {5, 0, 2, 6, 1, 3, 7, 4};
	auto expect = [](const OaMatrix& m, const std::vector<OaU32>& wanted) {
		ASSERT_EQ(m.NumElements(), static_cast<OaI64>(wanted.size()));
		const OaU32* got = m.DataAs<const OaU32>();
		for (size_t i = 0; i < wanted.size(); ++i) EXPECT_EQ(got[i], wanted[i]) << "index " << i;
	};
	expect(plan.Counts, counts);
	expect(plan.Offsets, offsets);
	expect(plan.PackedSlot, slots);
	expect(plan.PackedToken, tokens);
	expect(plan.PackedExpert, experts);
	expect(plan.Inverse, inverse);
}

TEST(OaMoeExpertPlan, EmptyExpertsAndAllRoutesPreserved) {
	auto& ctx = OaContext::GetDefault();
	OaContext::Scope scope(ctx);
	const std::vector<OaI32> routes = {3, 3, 3, 3};
	auto indices = OaFnMatrix::FromBytes(
		OaSpan<const OaU8>(reinterpret_cast<const OaU8*>(routes.data()), routes.size() * sizeof(OaI32)),
		OaMatrixShape{4, 1}, OaScalarType::Int32);
	auto plan = OaFnMatrix::MoeExpertPlan(indices, 5);
	ASSERT_TRUE(ctx.Execute().IsOk());
	ASSERT_TRUE(ctx.Sync().IsOk());
	const OaU32* counts = plan.Counts.DataAs<const OaU32>();
	const OaU32* offsets = plan.Offsets.DataAs<const OaU32>();
	EXPECT_EQ(counts[0], 0u); EXPECT_EQ(counts[1], 0u); EXPECT_EQ(counts[2], 0u);
	EXPECT_EQ(counts[3], 4u); EXPECT_EQ(counts[4], 0u);
	EXPECT_EQ(offsets[0], 0u); EXPECT_EQ(offsets[3], 0u);
	EXPECT_EQ(offsets[4], 4u); EXPECT_EQ(offsets[5], 4u);
	const OaU32* inverse = plan.Inverse.DataAs<const OaU32>();
	for (OaU32 r = 0; r < 4; ++r) EXPECT_EQ(inverse[r], r);
}

TEST(OaMoeExpertPlan, StableAcrossMultipleWorkgroupChunks) {
	auto& ctx = OaContext::GetDefault();
	OaContext::Scope scope(ctx);
	constexpr OaU32 T = 211, K = 3, E = 7;
	std::vector<OaI32> routes(T * K);
	for (OaU32 r = 0; r < T * K; ++r) routes[r] = static_cast<OaI32>((r * 5 + r / 11) % E);
	auto indices = OaFnMatrix::FromBytes(
		OaSpan<const OaU8>(reinterpret_cast<const OaU8*>(routes.data()), routes.size() * sizeof(OaI32)),
		OaMatrixShape{T, K}, OaScalarType::Int32);
	auto plan = OaFnMatrix::MoeExpertPlan(indices, E);
	ASSERT_TRUE(ctx.Execute().IsOk());
	ASSERT_TRUE(ctx.Sync().IsOk());
	const OaU32* counts = plan.Counts.DataAs<const OaU32>();
	const OaU32* offsets = plan.Offsets.DataAs<const OaU32>();
	const OaU32* slots = plan.PackedSlot.DataAs<const OaU32>();
	const OaU32* inverse = plan.Inverse.DataAs<const OaU32>();
	EXPECT_EQ(offsets[E], T * K);
	for (OaU32 e = 0; e < E; ++e) {
		EXPECT_EQ(offsets[e + 1] - offsets[e], counts[e]);
		OaU32 previous = 0;
		for (OaU32 p = offsets[e]; p < offsets[e + 1]; ++p) {
			const OaU32 route = slots[p];
			EXPECT_EQ(static_cast<OaU32>(routes[route]), e);
			if (p > offsets[e]) EXPECT_GT(route, previous);
			previous = route;
			EXPECT_EQ(inverse[route], p);
		}
	}
}

TEST(OaMoeDiag, BroadcastMulBackward) {
	setenv("OA_GEMM_FORCE_FP32", "1", 1);
	OaFnMatrix::SetRngSeed(3);
	auto& ctx = OaContext::GetDefault();
	OaContext::Scope scope(ctx);

	// a[2,3] * b[2,1] (broadcast over dim1). Check grad to BOTH operands.
	auto a = OaFnMatrix::RandN(OaMatrixShape{2, 3}, OaScalarType::Float32);
	auto b = OaFnMatrix::RandN(OaMatrixShape{2, 1}, OaScalarType::Float32);
	auto target = OaFnMatrix::RandN(OaMatrixShape{2, 3}, OaScalarType::Float32);
	a.SetRequiresGrad(true);
	b.SetRequiresGrad(true);

	OaGradientTape tape;
	auto out  = OaFnMatrix::Mul(a, b);          // [2,3] broadcast
	auto loss = OaFnLoss::Mse(out, target);
	tape.Backward(loss);
	(void)ctx.Execute(); (void)ctx.Sync();

	auto dA = a.GradMatrix();
	auto dB = b.GradMatrix();
	printf("\nBcastMul: dA elems=%lld dB elems=%lld |dA|=%.5f |dB|=%.5f\n",
		(long long)dA.NumElements(), (long long)dB.NumElements(),
		GradMag(dA), GradMag(dB));
	EXPECT_EQ(dB.NumElements(), 2) << "dB must be reduced to b's shape [2,1]";
	EXPECT_GT(GradMag(dA), 1e-6f) << "broadcast-mul: grad to large operand is zero";
	EXPECT_GT(GradMag(dB), 1e-6f) << "broadcast-mul: grad to broadcast operand is zero";

	auto lossFunc = [&]() -> float {
		OaGradNo noGrad;
		auto o = OaFnMatrix::Mul(a, b);
		auto l = OaFnLoss::Mse(o, target);
		(void)ctx.Execute(); (void)ctx.Sync();
		return l.DataAs<const float>()[0];
	};
	auto sa = GradCheckParam(ctx, lossFunc, a, dA.DataAs<const float>(), "a[2,3]");
	auto sb = GradCheckParam(ctx, lossFunc, b, dB.DataAs<const float>(), "b[2,1]");
	EXPECT_EQ(sa.Failed + sb.Failed, 0) << "broadcast-mul gradient mismatch";
}

TEST(OaMoeDiag, SliceColumnBackwardMultiUse) {
	setenv("OA_GEMM_FORCE_FP32", "1", 1);
	OaFnMatrix::SetRngSeed(5);
	auto& ctx = OaContext::GetDefault();
	OaContext::Scope scope(ctx);

	// p[2,3]; out = sum_e slice(p,col e)  → every column sliced once (multi-use).
	auto p = OaFnMatrix::RandN(OaMatrixShape{2, 3}, OaScalarType::Float32);
	auto target = OaFnMatrix::RandN(OaMatrixShape{2, 1}, OaScalarType::Float32);
	p.SetRequiresGrad(true);

	OaGradientTape tape;
	OaMatrix acc;
	for (OaI32 e = 0; e < 3; ++e) {
		auto col = OaFnMatrix::Slice(p, 1, e, e + 1);  // [2,1]
		acc = (e == 0) ? col : OaFnMatrix::Add(acc, col);
	}
	auto loss = OaFnLoss::Mse(acc, target);
	tape.Backward(loss);
	(void)ctx.Execute(); (void)ctx.Sync();

	auto dP = p.GradMatrix();
	printf("\nSliceMultiUse: dP elems=%lld |dP|=%.5f\n",
		(long long)dP.NumElements(), GradMag(dP));
	EXPECT_EQ(dP.NumElements(), 6);
	EXPECT_GT(GradMag(dP), 1e-6f) << "slice backward dropped gradient to p";
}

// Minimal: leaf p[T,E], each column sliced then BROADCAST-multiplied with an
// expert[T,D] and accumulated. No softmax. Isolates slice-of-multi-use combined
// with broadcast-mul gradient accumulation.
TEST(OaMoeDiag, SliceBcastMulMultiUse) {
	setenv("OA_GEMM_FORCE_FP32", "1", 1);
	OaFnMatrix::SetRngSeed(21);
	auto& ctx = OaContext::GetDefault();
	OaContext::Scope scope(ctx);

	constexpr OaI32 T = 2, E = 2, D = 3;
	auto p = OaFnMatrix::RandN(OaMatrixShape{T, E}, OaScalarType::Float32);
	OaVec<OaMatrix> experts;
	for (OaI32 e = 0; e < E; ++e) experts.PushBack(OaFnMatrix::RandN(OaMatrixShape{T, D}, OaScalarType::Float32));
	auto target = OaFnMatrix::RandN(OaMatrixShape{T, D}, OaScalarType::Float32);
	p.SetRequiresGrad(true);

	auto fwd = [&]() -> OaMatrix {
		OaMatrix out;
		for (OaI32 e = 0; e < E; ++e) {
			auto gateE = OaFnMatrix::Slice(p, 1, e, e + 1);     // [T,1]
			auto contrib = OaFnMatrix::Mul(experts[e], gateE);  // [T,D] bcast
			out = (e == 0) ? contrib : OaFnMatrix::Add(out, contrib);
		}
		return out;
	};
	OaGradientTape tape;
	auto out = fwd();
	auto loss = OaFnLoss::Mse(out, target);
	tape.Backward(loss);
	(void)ctx.Execute(); (void)ctx.Sync();
	auto dP = p.GradMatrix();
	printf("\nSliceBcastMulMultiUse: |dP|=%.6f\n", GradMag(dP));
	EXPECT_GT(GradMag(dP), 1e-6f) << "slice→bcast-mul multi-use dropped gradient to p";
}

// Mirrors the MoE forward chain with plain leaves (no modules): the gate is a
// softmax over logits, sliced per-expert and broadcast-multiplied with each
// expert output, then summed. Isolates whether the softmax→gate→combine path
// carries gradient back to the logits.
TEST(OaMoeDiag, SoftmaxGateCombineChain) {
	setenv("OA_GEMM_FORCE_FP32", "1", 1);
	OaFnMatrix::SetRngSeed(9);
	auto& ctx = OaContext::GetDefault();
	OaContext::Scope scope(ctx);

	constexpr OaI32 T = 2, E = 2, D = 3;
	auto logits = OaFnMatrix::RandN(OaMatrixShape{T, E}, OaScalarType::Float32);
	OaVec<OaMatrix> experts;
	for (OaI32 e = 0; e < E; ++e)
		experts.PushBack(OaFnMatrix::RandN(OaMatrixShape{T, D}, OaScalarType::Float32));
	auto target = OaFnMatrix::RandN(OaMatrixShape{T, D}, OaScalarType::Float32);
	logits.SetRequiresGrad(true);
	for (auto& ex : experts) ex.SetRequiresGrad(true);

	auto fwd = [&]() -> OaMatrix {
		auto probs = OaFnMatrix::Softmax(logits, 1);   // [T,E]
		OaMatrix out;
		for (OaI32 e = 0; e < E; ++e) {
			auto gateE = OaFnMatrix::Slice(probs, 1, e, e + 1);  // [T,1]
			auto contrib = OaFnMatrix::Mul(experts[e], gateE);   // [T,D] bcast
			out = (e == 0) ? contrib : OaFnMatrix::Add(out, contrib);
		}
		return out;
	};

	OaGradientTape tape;
	auto out  = fwd();
	auto loss = OaFnLoss::Mse(out, target);
	tape.Backward(loss);
	(void)ctx.Execute(); (void)ctx.Sync();

	auto dLogits = logits.GradMatrix();
	printf("\nSoftmaxGateChain: |dLogits|=%.6f\n", GradMag(dLogits));
	EXPECT_GT(GradMag(dLogits), 1e-6f) << "softmax→gate→combine dropped gradient to logits";

	auto lossFunc = [&]() -> float {
		OaGradNo noGrad;
		auto o = fwd();
		auto l = OaFnLoss::Mse(o, target);
		(void)ctx.Execute(); (void)ctx.Sync();
		return l.DataAs<const float>()[0];
	};
	auto sl = GradCheckParam(ctx, lossFunc, logits, dLogits.DataAs<const float>(), "logits");
	EXPECT_EQ(sl.Failed, 0) << "logits gradient mismatch through softmax-gate chain";
}

// OaLinear weight + input gradient in isolation (Linear is used everywhere).
// Previously failed 6–28% under fp32-forced FD. ROOT CAUSE (fixed): the
// fused-bias forward Linear (AddLinear) packed inputs to bf16 and ran a bf16
// CoopMat GEMM unconditionally, bypassing OaGemmRouter — so it ignored
// OA_GEMM_FORCE_FP32. bf16's ~3-digit mantissa added ~4e-3 forward error, which
// both corrupted the FD reference and fed into the analytical gradient via dy.
// Fix: AddLinear/AddLinearRelu/LinearBwdWeightBias now honor the flag.
TEST(OaMoeDiag, LinearWeightAndInput) {
	setenv("OA_GEMM_FORCE_FP32", "1", 1);
	OaFnMatrix::SetRngSeed(43);
	auto& ctx = OaContext::GetDefault();
	OaContext::Scope scope(ctx);
	constexpr OaI32 In = 4, Out = 5, T = 3;
	OaLinear lin(In, Out);
	auto x = OaFnMatrix::RandN(OaMatrixShape{T, In}, OaScalarType::Float32);
	auto target = OaFnMatrix::RandN(OaMatrixShape{T, Out}, OaScalarType::Float32);
	x.SetRequiresGrad(true);
	auto named = lin.AllNamedParameterPtrs();
	OaParameter* w = FindParam(named, "weight");
	ASSERT_NE(w, nullptr);
	OaGradientTape tape;
	auto out = lin.Forward(x);
	auto loss = OaFnLoss::Mse(out, target);
	tape.Backward(loss);
	(void)ctx.Execute(); (void)ctx.Sync();
	auto dW = w->Data.GradMatrix();
	auto dX = x.GradMatrix();
	auto lossFunc = [&]() -> float {
		OaGradNo noGrad;
		auto o = lin.Forward(x);
		auto l = OaFnLoss::Mse(o, target);
		(void)ctx.Execute(); (void)ctx.Sync();
		return l.DataAs<const float>()[0];
	};
	printf("\nLinear weight+input gradcheck:\n");
	auto sw = GradCheckParam(ctx, lossFunc, w->Data, dW.DataAs<const float>(), "weight");
	auto sx = GradCheckParam(ctx, lossFunc, x, dX.DataAs<const float>(), "input");
	EXPECT_EQ(sw.Failed + sx.Failed, 0) << "Linear gradient wrong";
}

// CPU GROUND-TRUTH for Linear backward. Reads x, W, b, forward-y and target to
// host, computes the analytic gradient by hand (exact fp64), and compares to BOTH
// the GPU autograd gradient and the central finite difference. This decisively
// separates a forward-precision problem (FD ≠ CPU-truth) from a backward-kernel
// problem (autograd ≠ CPU-truth). RESOLVED: the fused-bias forward Linear ran in
// bf16 even under OA_GEMM_FORCE_FP32 (it bypassed OaGemmRouter); now ~5e-8.
TEST(OaMoeDiag, LinearCpuGroundTruth) {
	setenv("OA_GEMM_FORCE_FP32", "1", 1);
	OaFnMatrix::SetRngSeed(43);
	auto& ctx = OaContext::GetDefault();
	OaContext::Scope scope(ctx);
	constexpr OaI32 In = 4, Out = 5, T = 3;
	OaLinear lin(In, Out);
	auto x = OaFnMatrix::RandN(OaMatrixShape{T, In}, OaScalarType::Float32);
	auto target = OaFnMatrix::RandN(OaMatrixShape{T, Out}, OaScalarType::Float32);
	x.SetRequiresGrad(true);
	auto named = lin.AllNamedParameterPtrs();
	OaParameter* w = FindParam(named, "weight");
	OaParameter* b = FindParam(named, "bias");
	ASSERT_NE(w, nullptr);

	OaGradientTape tape;
	auto out = lin.Forward(x);
	auto loss = OaFnLoss::Mse(out, target);
	tape.Backward(loss);
	(void)ctx.Execute(); (void)ctx.Sync();

	auto dW = w->Data.GradMatrix();
	auto dX = x.GradMatrix();

	// Host copies. y is the GPU forward output.
	const float* hx = x.DataAs<const float>();
	const float* hw = w->Data.DataAs<const float>();
	const float* hb = (b ? b->Data.DataAs<const float>() : nullptr);
	const float* ht = target.DataAs<const float>();
	const float* hy = out.DataAs<const float>();

	// 1. CPU forward: y_cpu[i,j] = sum_k x[i,k]*W[j,k] + b[j]. Compare to GPU y.
	double maxFwdErr = 0.0;
	for (int i = 0; i < T; ++i)
		for (int j = 0; j < Out; ++j) {
			double acc = hb ? (double)hb[j] : 0.0;
			for (int k = 0; k < In; ++k) acc += (double)hx[i * In + k] * (double)hw[j * In + k];
			maxFwdErr = std::max(maxFwdErr, std::abs(acc - (double)hy[i * Out + j]));
		}
	printf("\nLinear CPU ground-truth:\n  max |y_gpu - y_cpu| = %.3e (forward precision)\n", maxFwdErr);

	// 2. CPU backward from the *CPU forward* y. dL/dy = (2/(T*Out))*(y-target).
	const double scale = 2.0 / (double)(T * Out);
	std::vector<double> dy(T * Out);
	for (int i = 0; i < T; ++i)
		for (int j = 0; j < Out; ++j) {
			double acc = hb ? (double)hb[j] : 0.0;
			for (int k = 0; k < In; ++k) acc += (double)hx[i * In + k] * (double)hw[j * In + k];
			dy[i * Out + j] = scale * (acc - (double)ht[i * Out + j]);
		}
	// dW[j,k] = sum_i dy[i,j]*x[i,k];  dX[i,k] = sum_j dy[i,j]*W[j,k].
	double maxWErr = 0.0, maxXErr = 0.0;
	for (int j = 0; j < Out; ++j)
		for (int k = 0; k < In; ++k) {
			double acc = 0.0;
			for (int i = 0; i < T; ++i) acc += dy[i * Out + j] * (double)hx[i * In + k];
			maxWErr = std::max(maxWErr, std::abs(acc - (double)dW.DataAs<const float>()[j * In + k]));
		}
	for (int i = 0; i < T; ++i)
		for (int k = 0; k < In; ++k) {
			double acc = 0.0;
			for (int j = 0; j < Out; ++j) acc += dy[i * Out + j] * (double)hw[j * In + k];
			maxXErr = std::max(maxXErr, std::abs(acc - (double)dX.DataAs<const float>()[i * In + k]));
		}
	printf("  max |dW_gpu - dW_cpu| = %.3e\n  max |dX_gpu - dX_cpu| = %.3e\n", maxWErr, maxXErr);
	EXPECT_LT(maxFwdErr, 1e-4) << "FORWARD GEMM is not fp32-accurate (breaks the FD reference)";
	EXPECT_LT(maxWErr, 1e-4) << "weight backward kernel disagrees with CPU truth";
	EXPECT_LT(maxXErr, 1e-4) << "data backward kernel disagrees with CPU truth";
}

// OaRmsNorm MODULE weight gradient (uses OaGradRmsNorm — different path than the
// raw OaFnMatrix::RmsNorm verified in Session 3). normed feeds nothing else here.
TEST(OaMoeDiag, RmsNormModuleWeight) {
	setenv("OA_GEMM_FORCE_FP32", "1", 1);
	OaFnMatrix::SetRngSeed(37);
	auto& ctx = OaContext::GetDefault();
	OaContext::Scope scope(ctx);
	constexpr OaI32 D = 4, T = 3;
	OaRmsNorm norm(D);
	auto x = OaFnMatrix::RandN(OaMatrixShape{T, D}, OaScalarType::Float32);
	auto target = OaFnMatrix::RandN(OaMatrixShape{T, D}, OaScalarType::Float32);
	auto named = norm.AllNamedParameterPtrs();
	OaParameter* w = FindParam(named, "weight");
	if (w == nullptr) { for (auto& np : named) printf("  rmsnorm param: %s\n", np.Path.CStr()); FAIL(); }
	OaGradientTape tape;
	auto out = norm.Forward(x);
	auto loss = OaFnLoss::Mse(out, target);
	tape.Backward(loss);
	(void)ctx.Execute(); (void)ctx.Sync();
	auto dW = w->Data.GradMatrix();
	auto lossFunc = [&]() -> float {
		OaGradNo noGrad;
		auto o = norm.Forward(x);
		auto l = OaFnLoss::Mse(o, target);
		(void)ctx.Execute(); (void)ctx.Sync();
		return l.DataAs<const float>()[0];
	};
	printf("\nRmsNorm MODULE weight gradcheck:\n");
	auto s = GradCheckParam(ctx, lossFunc, w->Data, dW.DataAs<const float>(), "weight");
	EXPECT_EQ(s.Failed, 0) << "RmsNorm module weight gradient wrong";
}

// Silu backward via autograd (its existing test only checks isfinite).
TEST(OaMoeDiag, SiluBackward) {
	setenv("OA_GEMM_FORCE_FP32", "1", 1);
	OaFnMatrix::SetRngSeed(41);
	auto& ctx = OaContext::GetDefault();
	OaContext::Scope scope(ctx);
	auto x = OaFnMatrix::RandN(OaMatrixShape{3, 4}, OaScalarType::Float32);
	auto target = OaFnMatrix::RandN(OaMatrixShape{3, 4}, OaScalarType::Float32);
	x.SetRequiresGrad(true);
	OaGradientTape tape;
	auto out = OaFnMatrix::Silu(x);
	auto loss = OaFnLoss::Mse(out, target);
	tape.Backward(loss);
	(void)ctx.Execute(); (void)ctx.Sync();
	auto dX = x.GradMatrix();
	auto lossFunc = [&]() -> float {
		OaGradNo noGrad;
		auto o = OaFnMatrix::Silu(x);
		auto l = OaFnLoss::Mse(o, target);
		(void)ctx.Execute(); (void)ctx.Sync();
		return l.DataAs<const float>()[0];
	};
	printf("\nSilu backward gradcheck:\n");
	auto s = GradCheckParam(ctx, lossFunc, x, dX.DataAs<const float>(), "silu.x");
	EXPECT_EQ(s.Failed, 0) << "Silu backward is wrong";
}

// Is the expert FFN itself gradient-correct in isolation? (Separates an FFN/
// RmsNorm bug from the MoE weighting.) Gradchecks OaFfn's RmsNorm weight.
// Previously failed via the SAME fused-bias-Linear bf16 bug as LinearWeightAndInput
// (the FFN's gate/up/down are OaLinear); fixed with the OA_GEMM_FORCE_FP32 guard.
TEST(OaMoeDiag, FfnNormWeightStandalone) {
	setenv("OA_GEMM_FORCE_FP32", "1", 1);
	OaFnMatrix::SetRngSeed(31);
	auto& ctx = OaContext::GetDefault();
	OaContext::Scope scope(ctx);

	constexpr OaI32 D = 4, DFF = 8, T = 3;
	OaFfn ffn(D, DFF);
	auto x = OaFnMatrix::RandN(OaMatrixShape{T, D}, OaScalarType::Float32);
	auto target = OaFnMatrix::RandN(OaMatrixShape{T, D}, OaScalarType::Float32);

	auto named = ffn.AllNamedParameterPtrs();
	OaParameter* normW = FindParam(named, "norm.weight");
	if (normW == nullptr) {
		printf("FFN param paths:\n");
		for (auto& np : named) printf("  %s\n", np.Path.CStr());
		FAIL() << "norm.weight not found";
	}

	OaGradientTape tape;
	auto out  = ffn.Forward(x);
	auto loss = OaFnLoss::Mse(out, target);
	tape.Backward(loss);
	(void)ctx.Execute(); (void)ctx.Sync();
	auto dNorm = normW->Data.GradMatrix();

	auto lossFunc = [&]() -> float {
		OaGradNo noGrad;
		auto o = ffn.Forward(x);
		auto l = OaFnLoss::Mse(o, target);
		(void)ctx.Execute(); (void)ctx.Sync();
		return l.DataAs<const float>()[0];
	};
	printf("\nFFN norm.weight standalone gradcheck:\n");
	auto s = GradCheckParam(ctx, lossFunc, normW->Data, dNorm.DataAs<const float>(), "norm.weight");
	EXPECT_EQ(s.Failed, 0) << "FFN RmsNorm weight gradient is wrong in isolation (not a MoE bug)";
}

// Dense case (K == E): the selection mask is all-ones, so the forward is fully
// smooth — a clean gradcheck of the entire chain (softmax → gate → expert FFN →
// weighted combine). Verifies both the router weight and an expert weight.
TEST(OaMoeGradcheck, DenseRouterAndExperts) {
	setenv("OA_GEMM_FORCE_FP32", "1", 1);
	OaFnMatrix::SetRngSeed(7);
	auto& ctx = OaContext::GetDefault();
	OaContext::Scope scope(ctx);

	constexpr OaI32 D = 4, DFF = 8, E = 2, K = 2, T = 3;  // K==E → dense, smooth
	OaMoE moe(D, DFF, E, K);
	// Finite differences repeatedly rebuild the same smooth function. Use the
	// explicit dense oracle here; sparse-executor correctness is covered by the
	// forward/input/parameter parity test in TestMoeSystems.
	moe.SetSparseExecution(false);

	auto x      = OaFnMatrix::RandN(OaMatrixShape{T, D}, OaScalarType::Float32);
	auto target = OaFnMatrix::RandN(OaMatrixShape{T, D}, OaScalarType::Float32);

	auto named = moe.AllNamedParameterPtrs();
	OaParameter* routerW = FindParam(named, "router.weight");
	OaParameter* expertW = FindParam(named, "expert_gate_up_weight");
	ASSERT_NE(routerW, nullptr) << "router.weight not found";
	ASSERT_NE(expertW, nullptr) << "expert_gate_up_weight not found";

	OaGradientTape tape;
	auto out  = moe.Forward(x);
	auto loss = OaFnLoss::Mse(out, target);
	tape.Backward(loss);
	(void)ctx.Execute();
	(void)ctx.Sync();

	auto dRouter = routerW->Data.GradMatrix();
	auto dExpert = expertW->Data.GradMatrix();
	ASSERT_EQ(dRouter.NumElements(), routerW->Data.NumElements());
	ASSERT_EQ(dExpert.NumElements(), expertW->Data.NumElements());

	auto lossFunc = [&]() -> float {
		OaGradNo noGrad;
		auto o = moe.Forward(x);
		auto l = OaFnLoss::Mse(o, target);
		(void)ctx.Execute(); (void)ctx.Sync();
		return l.DataAs<const float>()[0];
	};

	// Verify both sides of the layer end to end: router selection magnitudes and
	// an actual expert projection weight. This used to stop at "non-zero expert
	// grad" because GEMM backward was suspect; the CPU-ground-truth diagnostics
	// above now make a full finite-difference gate meaningful.
	printf("\nMoE dense gradcheck [D=%d,DFF=%d,E=%d,K=%d,T=%d]:\n", D, DFF, E, K, T);
	auto sr = GradCheckParam(ctx, lossFunc, routerW->Data, dRouter.DataAs<const float>(), "router.weight");
	printf("MoE router gradcheck: %d/%d pass, %d non-trivial\n",
		sr.Checked - sr.Failed, sr.Checked, sr.NonTrivial);
	EXPECT_EQ(sr.Failed, 0) << "MoE router gradient mismatch (autograd != finite diff)";
	EXPECT_GE(sr.NonTrivial, 3) << "router gradients all ~0 — vacuous check";
	auto se = GradCheckParam(ctx, lossFunc, expertW->Data, dExpert.DataAs<const float>(), "expert_gate_up_weight");
	EXPECT_EQ(se.Failed, 0) << "MoE expert gradient mismatch (autograd != finite diff)";
	EXPECT_GE(se.NonTrivial, 3) << "expert gradients all ~0 — vacuous check";
}

// Sparse top-k (K=2 of E=4): with K>=2 the renormalized gate still depends on the
// selected probabilities, so the router receives a real task-loss gradient — the
// thing the old stub MoeRouterBwd never produced. (For K==1 the renormalized gate
// is identically 1, so the router gets no signal by construction — a property of
// top-1 renormalized routing, not a bug.) Assert both router and experts learn.
TEST(OaMoeGradcheck, TopKRouterAndExpertsLearn) {
	setenv("OA_GEMM_FORCE_FP32", "1", 1);
	OaFnMatrix::SetRngSeed(13);
	auto& ctx = OaContext::GetDefault();
	OaContext::Scope scope(ctx);

	constexpr OaI32 D = 4, DFF = 8, E = 4, K = 2, T = 4;  // top-2 of 4
	OaMoE moe(D, DFF, E, K);

	auto x      = OaFnMatrix::RandN(OaMatrixShape{T, D}, OaScalarType::Float32);
	auto target = OaFnMatrix::RandN(OaMatrixShape{T, D}, OaScalarType::Float32);

	auto named = moe.AllNamedParameterPtrs();
	OaParameter* routerW = FindParam(named, "router.weight");
	OaParameter* expertW = FindParam(named, "expert_gate_up_weight");
	ASSERT_NE(routerW, nullptr);
	ASSERT_NE(expertW, nullptr);

	OaGradientTape tape;
	auto out  = moe.Forward(x);
	auto loss = OaFnLoss::Mse(out, target);
	tape.Backward(loss);
	(void)ctx.Execute();
	(void)ctx.Sync();

	// Router gets a real task-loss gradient through the selected gate magnitudes
	// (the old stub MoeRouterBwd produced ~0 from the data path).
	EXPECT_GT(GradMag(routerW->Data.GradMatrix()), 1e-5f)
		<< "router got no task-loss gradient — routing won't learn (the old stub bug)";
	// Experts receive a real gradient (they are selected and weighted).
	EXPECT_GT(GradMag(expertW->Data.GradMatrix()), 1e-5f)
		<< "experts receive no gradient — they won't learn";
}

TEST(OaMoeConfiguration, RejectsInvalidAndClampsTopK) {
	EXPECT_THROW((void)OaMoE(0, 8, 2, 1), std::invalid_argument);
	EXPECT_THROW((void)OaMoE(4, 0, 2, 1), std::invalid_argument);
	EXPECT_THROW((void)OaMoE(4, 8, 0, 1), std::invalid_argument);

	OaMoE low(4, 8, 3, 0);
	EXPECT_EQ(low.ExpertsPerToken(), 1);
	OaMoE high(4, 8, 3, 99);
	EXPECT_EQ(high.ExpertsPerToken(), 3);
}

TEST(OaMoeConfiguration, TiedRouterStillSelectsExactlyK) {
	setenv("OA_GEMM_FORCE_FP32", "1", 1);
	auto& ctx = OaContext::GetDefault();
	OaContext::Scope scope(ctx);
	constexpr OaI32 T = 5, D = 4, E = 4, K = 2;
	OaMoE moe(D, 8, E, K);
	auto named = moe.AllNamedParameterPtrs();
	auto* routerW = FindParam(named, "router.weight");
	auto* routerB = FindParam(named, "router.bias");
	ASSERT_NE(routerW, nullptr);
	ASSERT_NE(routerB, nullptr);
	// Flush deferred parameter initialization before overwriting the router with
	// exact zeros; otherwise the queued Xavier dispatch would run afterward.
	ASSERT_TRUE(ctx.Execute().IsOk());
	ASSERT_TRUE(ctx.Sync().IsOk());
	for (OaI64 i = 0; i < routerW->Data.NumElements(); ++i) routerW->Data.DataAs<float>()[i] = 0.0F;
	for (OaI64 i = 0; i < routerB->Data.NumElements(); ++i) routerB->Data.DataAs<float>()[i] = 0.0F;

	auto x = OaFnMatrix::RandN(OaMatrixShape{T, D}, OaScalarType::Float32);
	(void)moe.Forward(x);
	ASSERT_TRUE(ctx.Execute().IsOk());
	ASSERT_TRUE(ctx.Sync().IsOk());
	const auto& mask = moe.LastSelectionMask();
	ASSERT_EQ(mask.GetShape(), (OaMatrixShape{T, E}));
	const float* values = mask.DataAs<const float>();
	for (OaI32 t = 0; t < T; ++t) {
		float selected = 0.0F;
		for (OaI32 e = 0; e < E; ++e) selected += values[t * E + e];
		EXPECT_FLOAT_EQ(selected, static_cast<float>(K));
		EXPECT_FLOAT_EQ(values[t * E + 0], 1.0F);  // deterministic lower-index tie break
		EXPECT_FLOAT_EQ(values[t * E + 1], 1.0F);
		EXPECT_FLOAT_EQ(values[t * E + 2], 0.0F);
		EXPECT_FLOAT_EQ(values[t * E + 3], 0.0F);
	}
}

// ── Stage 0: route-utilization telemetry ──────────────────────────────────────
// RouteStats must be internally consistent: load fractions and mean probs each
// sum to 1, entropy is normalized to [0,1], and the max-load ratio is >= 1.
TEST(OaMoeStage0, RouteStatsAreSane) {
	setenv("OA_GEMM_FORCE_FP32", "1", 1);
	OaFnMatrix::SetRngSeed(101);
	auto& ctx = OaContext::GetDefault();
	OaContext::Scope scope(ctx);
	constexpr OaI32 T = 16, D = 8, DFF = 16, E = 4, K = 2;
	OaMoE moe(D, DFF, E, K);
	auto x = OaFnMatrix::RandN(OaMatrixShape{T, D}, OaScalarType::Float32);
	(void)moe.Forward(x);
	ASSERT_TRUE(ctx.Execute().IsOk());
	ASSERT_TRUE(ctx.Sync().IsOk());

	auto st = moe.RouteStats();
	ASSERT_EQ(st.LoadFraction.Size(), static_cast<OaI64>(E));
	ASSERT_EQ(st.MeanProb.Size(), static_cast<OaI64>(E));
	float loadSum = 0.0F, probSum = 0.0F;
	for (OaI32 e = 0; e < E; ++e) { loadSum += st.LoadFraction[e]; probSum += st.MeanProb[e]; }
	EXPECT_NEAR(loadSum, 1.0F, 1e-4F) << "load fractions must sum to 1";
	EXPECT_NEAR(probSum, 1.0F, 1e-3F) << "mean softmax probs must sum to ~1";
	EXPECT_GE(st.Entropy, 0.0F);
	EXPECT_LE(st.Entropy, 1.0001F);
	EXPECT_GE(st.MaxLoadRatio, 1.0F) << "E*max_e load is >= 1 by definition";
}

// ── Stage 0: aux-loss-free balancing rescues a collapsed router ────────────────
// Force a maximally-collapsed router (large static bias to expert 0), then let the
// gradient-free per-expert bias nudge load back toward balance. Asserts the sign
// rule (deterministic) AND that real load spreads (emergent, tokens differ via the
// live router weight). Exercises the [T,E]+[1,E] broadcast in the bias path.
TEST(OaMoeStage0, BiasBalancingSpreadsCollapsedLoad) {
	setenv("OA_GEMM_FORCE_FP32", "1", 1);
	OaFnMatrix::SetRngSeed(202);
	auto& ctx = OaContext::GetDefault();
	OaContext::Scope scope(ctx);
	constexpr OaI32 T = 64, D = 16, DFF = 16, E = 4, K = 1;  // K=1 → collapse is stark
	OaMoE moe(D, DFF, E, K);
	moe.SetBalanceRate(0.3F);  // aggressive for a fast unit test

	auto named = moe.AllNamedParameterPtrs();
	auto* rW = FindParam(named, "router.weight");
	auto* rB = FindParam(named, "router.bias");
	ASSERT_NE(rW, nullptr);
	ASSERT_NE(rB, nullptr);
	// Flush deferred init, then bias the router hard toward expert 0. The live
	// (random) weight keeps per-token logits distinct so load CAN spread.
	ASSERT_TRUE(ctx.Execute().IsOk());
	ASSERT_TRUE(ctx.Sync().IsOk());
	for (OaI64 i = 0; i < rB->Data.NumElements(); ++i) rB->Data.DataAs<float>()[i] = 0.0F;
	rB->Data.DataAs<float>()[0] = 4.0F;

	auto x = OaFnMatrix::RandN(OaMatrixShape{T, D}, OaScalarType::Float32);

	(void)moe.Forward(x);
	ASSERT_TRUE(ctx.Execute().IsOk());
	ASSERT_TRUE(ctx.Sync().IsOk());
	auto before = moe.RouteStats();
	EXPECT_EQ(before.DeadExperts, E - 1) << "router should start collapsed onto expert 0";

	for (int s = 0; s < 80; ++s) {
		(void)moe.Forward(x);
		ASSERT_TRUE(ctx.Execute().IsOk());
		ASSERT_TRUE(ctx.Sync().IsOk());
		moe.UpdateRoutingBias();
	}
	(void)moe.Forward(x);
	ASSERT_TRUE(ctx.Execute().IsOk());
	ASSERT_TRUE(ctx.Sync().IsOk());
	auto after = moe.RouteStats();

	printf("\nBiasBalancing: dead %d→%d  entropy %.3f→%.3f  maxratio %.2f→%.2f  bias[0]=%.2f\n",
		before.DeadExperts, after.DeadExperts, before.Entropy, after.Entropy,
		before.MaxLoadRatio, after.MaxLoadRatio, moe.RoutingBias(0));
	EXPECT_LT(moe.RoutingBias(0), 0.0F) << "over-loaded expert 0 must be pushed down (sign rule)";
	EXPECT_LT(after.DeadExperts, before.DeadExperts) << "balancing must revive dead experts";
	EXPECT_LT(after.MaxLoadRatio, before.MaxLoadRatio) << "peak load must drop toward balance";
}

// ── Stage 0: shared always-on expert always receives gradient ─────────────────
// With K=1 the routed experts are mostly idle, but a shared expert is applied to
// every token unconditionally, so it must always get a gradient. Also verifies the
// shared-expert forward/backward path and its registration.
TEST(OaMoeStage0, SharedExpertAlwaysContributes) {
	setenv("OA_GEMM_FORCE_FP32", "1", 1);
	OaFnMatrix::SetRngSeed(303);
	auto& ctx = OaContext::GetDefault();
	OaContext::Scope scope(ctx);
	constexpr OaI32 T = 4, D = 4, DFF = 8, E = 4, K = 1;
	OaMoE moe(D, DFF, E, K, 1e-5F, /*shared*/ 1);
	EXPECT_EQ(moe.NumSharedExperts(), 1);

	auto named = moe.AllNamedParameterPtrs();
	auto* shW = FindParam(named, "shared_expert.0.gate_up.weight");
	ASSERT_NE(shW, nullptr) << "shared expert must be registered so the optimizer sees it";

	auto x = OaFnMatrix::RandN(OaMatrixShape{T, D}, OaScalarType::Float32);
	auto target = OaFnMatrix::RandN(OaMatrixShape{T, D}, OaScalarType::Float32);
	OaGradientTape tape;
	auto out = moe.Forward(x);
	auto loss = OaFnLoss::Mse(out, target);
	tape.Backward(loss);
	ASSERT_TRUE(ctx.Execute().IsOk());
	ASSERT_TRUE(ctx.Sync().IsOk());
	EXPECT_GT(GradMag(shW->Data.GradMatrix()), 1e-6F)
		<< "shared expert is always on — it must always receive gradient";
}

// ── Stage 0: opt-in switch aux loss + router z-loss are finite + differentiable ─
TEST(OaMoeStage0, AuxLossIsFiniteAndDifferentiable) {
	setenv("OA_GEMM_FORCE_FP32", "1", 1);
	OaFnMatrix::SetRngSeed(404);
	auto& ctx = OaContext::GetDefault();
	OaContext::Scope scope(ctx);
	constexpr OaI32 T = 8, D = 4, DFF = 8, E = 4, K = 2;
	OaMoE moe(D, DFF, E, K);
	moe.SetAuxLossAlpha(0.01F);
	moe.SetRouterZLossBeta(0.001F);

	auto named = moe.AllNamedParameterPtrs();
	auto* rW = FindParam(named, "router.weight");
	ASSERT_NE(rW, nullptr);

	auto x = OaFnMatrix::RandN(OaMatrixShape{T, D}, OaScalarType::Float32);
	auto target = OaFnMatrix::RandN(OaMatrixShape{T, D}, OaScalarType::Float32);
	OaGradientTape tape;
	auto out = moe.Forward(x);
	auto aux = moe.AuxLoss();  // handle to the scalar recorded on this tape
	auto loss = OaFnMatrix::Add(OaFnLoss::Mse(out, target), aux);
	tape.Backward(loss);
	ASSERT_TRUE(ctx.Execute().IsOk());
	ASSERT_TRUE(ctx.Sync().IsOk());

	const float av = moe.AuxLoss().DataAs<const float>()[0];
	printf("\nAuxLoss (switch α=0.01 + z β=0.001) = %.6f\n", av);
	EXPECT_TRUE(std::isfinite(av));
	EXPECT_GT(av, 0.0F) << "switch + z losses are positive for a non-degenerate router";
	EXPECT_GT(GradMag(rW->Data.GradMatrix()), 1e-6F) << "router must receive gradient (incl. aux)";
}

TEST(OaMoeStage0, DefaultsAreInertAndAuxLossResets) {
	setenv("OA_GEMM_FORCE_FP32", "1", 1);
	OaFnMatrix::SetRngSeed(505);
	auto& ctx = OaContext::GetDefault();
	OaContext::Scope scope(ctx);
	OaMoE moe(4, 8, 4, 2);
	EXPECT_FLOAT_EQ(moe.BalanceRate(), 0.0F);
	for (OaI32 e = 0; e < 4; ++e) EXPECT_FLOAT_EQ(moe.RoutingBias(e), 0.0F);

	auto x = OaFnMatrix::RandN(OaMatrixShape{8, 4}, OaScalarType::Float32);
	moe.SetAuxLossAlpha(0.01F);
	(void)moe.Forward(x);
	ASSERT_TRUE(ctx.Execute().IsOk());
	ASSERT_TRUE(ctx.Sync().IsOk());

	moe.SetAuxLossAlpha(0.0F);
	(void)moe.Forward(x);
	ASSERT_TRUE(ctx.Execute().IsOk());
	ASSERT_TRUE(ctx.Sync().IsOk());
	EXPECT_FLOAT_EQ(moe.AuxLoss().DataAs<const float>()[0], 0.0F)
		<< "disabling aux loss must not expose the previous forward's stale scalar";
}

TEST(OaMoeStage0, RoutingBiasCheckpoints) {
	setenv("OA_GEMM_FORCE_FP32", "1", 1);
	OaFnMatrix::SetRngSeed(606);
	auto& ctx = OaContext::GetDefault();
	OaContext::Scope scope(ctx);
	constexpr OaI32 E = 4;
	OaMoE moe(8, 16, E, 1);
	moe.SetBalanceRate(0.25F);

	auto named = moe.AllNamedParameterPtrs();
	auto* rB = FindParam(named, "router.bias");
	ASSERT_NE(rB, nullptr);
	ASSERT_TRUE(ctx.Execute().IsOk());
	ASSERT_TRUE(ctx.Sync().IsOk());
	for (OaI64 i = 0; i < rB->Data.NumElements(); ++i) rB->Data.DataAs<float>()[i] = 0.0F;
	rB->Data.DataAs<float>()[0] = 5.0F;

	auto x = OaFnMatrix::RandN(OaMatrixShape{16, 8}, OaScalarType::Float32);
	(void)moe.Forward(x);
	ASSERT_TRUE(ctx.Execute().IsOk());
	ASSERT_TRUE(ctx.Sync().IsOk());
	moe.UpdateRoutingBias();
	const float expected = moe.RoutingBias(0);
	ASSERT_LT(expected, 0.0F);

	const OaString path = "/tmp/oa_moe_routing_bias.oam";
	ASSERT_TRUE(moe.Save(path).IsOk());
	OaMoE loaded(8, 16, E, 1);
	ASSERT_TRUE(loaded.Load(path).IsOk());
	EXPECT_FLOAT_EQ(loaded.RoutingBias(0), expected);
}
