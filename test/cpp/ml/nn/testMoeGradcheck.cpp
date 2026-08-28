// oa::Moe — end-to-end finite-difference gradient checks.
//
// These tests verify the whole module differentiates correctly — router and
// experts, dense oracle and sparse GPU executor — by comparing
// analytical parameter gradients (autograd) to central finite differences of
// the MSE loss. This replaces the prior placeholder smoke tests, which only
// checked "doesn't crash" + "gradient != 0" (a buffer-binding bug passes both).
//
// forward is fp32-forced (OA_GEMM_FORCE_FP32) so finite differences are valid.

#include <gtest/gtest.h>
#include "../../oaTest.h"
#include <oa/ml.h>
#include <oa/ml/autograd.h>
#include <oa/runtime/executionSession.h>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

bool gradClose(float a, float n, float atol = 3e-3f, float rtol = 4e-2f) {
	return std::abs(a - n) <= (atol + rtol * std::abs(n));
}

struct GradStats { int checked = 0; int failed = 0; int nonTrivial = 0; };

// Central finite-difference gradcheck over every element of one parameter.
GradStats gradCheckParam(oa::ExecutionSession& ctx, const std::function<float()>& loss,
	oa::Matrix& param, const float* analytical, const char* name) {
	GradStats s;
	float* d = param.dataAs<float>();
	const oa::I64 n = param.numElements();
	// The finite-difference forwards allocate many transient tensors. snapshot the
	// synchronized analytical gradient before those runs: retaining only a raw
	// pointer made the reference allocator-lifetime dependent and it could be
	// overwritten while the check was still using it.
	std::vector<float> analyticalSnapshot(
		analytical, analytical + static_cast<size_t>(n));
	const float eps = 1e-2f;
	printf("  [%s] %lld elements\n", name, static_cast<long long>(n));
	for (oa::I64 i = 0; i < n; ++i) {
		const float orig = d[i];
		d[i] = orig + eps; (void)testSubmitAndWait(ctx); const float lp = loss();
		d[i] = orig - eps; (void)testSubmitAndWait(ctx); const float lm = loss();
		d[i] = orig;       (void)testSubmitAndWait(ctx);
		const float numerical = (lp - lm) / (2.0f * eps);
		const float a = analyticalSnapshot[static_cast<size_t>(i)];
		++s.checked;
		if (not gradClose(a, numerical)) {
			++s.failed;
			printf("    idx %lld: analytical=%.6f numerical=%.6f  MISMATCH\n",
				static_cast<long long>(i), a, numerical);
		}
		if (std::abs(numerical) > 5e-4f) ++s.nonTrivial;
	}
	return s;
}

// find a parameter by its dotted path; nullptr if absent (test fails loudly).
oa::Parameter* findParam(oa::Vector<oa::NamedParameter>& named, const char* path) {
	for (auto& np : named)
		if (std::string(np.path.cStr()) == path) return np.param;
	return nullptr;
}

float gradMag(const oa::Matrix& g) {
	if (g.numElements() == 0) return 0.0f;
	const float* d = g.dataAs<const float>();
	float s = 0.0f;
	for (oa::I64 i = 0; i < g.numElements(); ++i) s += std::abs(d[i]);
	return s;
}

}  // namespace

// ── DIAGNOSTICS: isolate the broadcast-Mul + Slice backward path ──────────────
TEST(MoeExpertPlan, StableDroplessPacking) {
	auto& ctx = oa::ExecutionSession::getActive();
	oa::ExecutionSession::RecordingScope scope(ctx);
	const std::vector<oa::I32> routes = {2, 0, 1, 2, 0, 1, 2, 1};
	auto indices = oa::FnMatrix::fromBytes(
		oa::Span<const oa::U8>(reinterpret_cast<const oa::U8*>(routes.data()), routes.size() * sizeof(oa::I32)),
		oa::MatrixShape{4, 2}, oa::ScalarType::Int32);
	auto plan = oa::FnMatrix::moeExpertPlan(indices, 3);
	ASSERT_TRUE(testSubmitAndWait(ctx).isOk());

	const std::vector<oa::U32> counts = {2, 3, 3};
	const std::vector<oa::U32> offsets = {0, 2, 5, 8};
	const std::vector<oa::U32> slots = {1, 4, 2, 5, 7, 0, 3, 6};
	const std::vector<oa::U32> tokens = {0, 2, 1, 2, 3, 0, 1, 3};
	const std::vector<oa::U32> experts = {0, 0, 1, 1, 1, 2, 2, 2};
	const std::vector<oa::U32> inverse = {5, 0, 2, 6, 1, 3, 7, 4};
	auto expect = [](const oa::Matrix& m, const std::vector<oa::U32>& wanted) {
		ASSERT_EQ(m.numElements(), static_cast<oa::I64>(wanted.size()));
		const oa::U32* got = m.dataAs<const oa::U32>();
		for (size_t i = 0; i < wanted.size(); ++i) EXPECT_EQ(got[i], wanted[i]) << "index " << i;
	};
	expect(plan.counts, counts);
	expect(plan.offsets, offsets);
	expect(plan.packedSlot, slots);
	expect(plan.packedToken, tokens);
	expect(plan.packedExpert, experts);
	expect(plan.inverse, inverse);
}

TEST(MoeExpertPlan, EmptyExpertsAndAllRoutesPreserved) {
	auto& ctx = oa::ExecutionSession::getActive();
	oa::ExecutionSession::RecordingScope scope(ctx);
	const std::vector<oa::I32> routes = {3, 3, 3, 3};
	auto indices = oa::FnMatrix::fromBytes(
		oa::Span<const oa::U8>(reinterpret_cast<const oa::U8*>(routes.data()), routes.size() * sizeof(oa::I32)),
		oa::MatrixShape{4, 1}, oa::ScalarType::Int32);
	auto plan = oa::FnMatrix::moeExpertPlan(indices, 5);
	ASSERT_TRUE(testSubmitAndWait(ctx).isOk());
	const oa::U32* counts = plan.counts.dataAs<const oa::U32>();
	const oa::U32* offsets = plan.offsets.dataAs<const oa::U32>();
	EXPECT_EQ(counts[0], 0u); EXPECT_EQ(counts[1], 0u); EXPECT_EQ(counts[2], 0u);
	EXPECT_EQ(counts[3], 4u); EXPECT_EQ(counts[4], 0u);
	EXPECT_EQ(offsets[0], 0u); EXPECT_EQ(offsets[3], 0u);
	EXPECT_EQ(offsets[4], 4u); EXPECT_EQ(offsets[5], 4u);
	const oa::U32* inverse = plan.inverse.dataAs<const oa::U32>();
	for (oa::U32 r = 0; r < 4; ++r) EXPECT_EQ(inverse[r], r);
}

TEST(MoeExpertPlan, StableAcrossMultipleWorkgroupChunks) {
	auto& ctx = oa::ExecutionSession::getActive();
	oa::ExecutionSession::RecordingScope scope(ctx);
	constexpr oa::U32 T = 211, K = 3, E = 7;
	std::vector<oa::I32> routes(T * K);
	for (oa::U32 r = 0; r < T * K; ++r) routes[r] = static_cast<oa::I32>((r * 5 + r / 11) % E);
	auto indices = oa::FnMatrix::fromBytes(
		oa::Span<const oa::U8>(reinterpret_cast<const oa::U8*>(routes.data()), routes.size() * sizeof(oa::I32)),
		oa::MatrixShape{T, K}, oa::ScalarType::Int32);
	auto plan = oa::FnMatrix::moeExpertPlan(indices, E);
	ASSERT_TRUE(testSubmitAndWait(ctx).isOk());
	const oa::U32* counts = plan.counts.dataAs<const oa::U32>();
	const oa::U32* offsets = plan.offsets.dataAs<const oa::U32>();
	const oa::U32* slots = plan.packedSlot.dataAs<const oa::U32>();
	const oa::U32* inverse = plan.inverse.dataAs<const oa::U32>();
	EXPECT_EQ(offsets[E], T * K);
	for (oa::U32 e = 0; e < E; ++e) {
		EXPECT_EQ(offsets[e + 1] - offsets[e], counts[e]);
		oa::U32 previous = 0;
		for (oa::U32 p = offsets[e]; p < offsets[e + 1]; ++p) {
			const oa::U32 route = slots[p];
			EXPECT_EQ(static_cast<oa::U32>(routes[route]), e);
			if (p > offsets[e]) EXPECT_GT(route, previous);
			previous = route;
			EXPECT_EQ(inverse[route], p);
		}
	}
}

TEST(MoeDiag, BroadcastMulBackward) {
	setenv("OA_GEMM_FORCE_FP32", "1", 1);
	oa::FnMatrix::setRngSeed(3);
	auto& ctx = oa::ExecutionSession::getActive();
	oa::ExecutionSession::RecordingScope scope(ctx);

	// a[2,3] * b[2,1] (broadcast over dim1). Check grad to BOTH operands.
	auto a = oa::FnMatrix::randN(oa::MatrixShape{2, 3}, oa::ScalarType::Float32);
	auto b = oa::FnMatrix::randN(oa::MatrixShape{2, 1}, oa::ScalarType::Float32);
	auto target = oa::FnMatrix::randN(oa::MatrixShape{2, 3}, oa::ScalarType::Float32);
	a.setRequiresGrad(true);
	b.setRequiresGrad(true);

	oa::GradientTape tape;
	auto out  = oa::FnMatrix::mul(a, b);          // [2,3] broadcast
	auto loss = oa::FnLoss::mse(out, target);
	tape.backward(loss);
	(void)testSubmitAndWait(ctx);

	auto dA = a.gradMatrix();
	auto dB = b.gradMatrix();
	printf("\nBcastMul: dA elems=%lld dB elems=%lld |dA|=%.5f |dB|=%.5f\n",
		(long long)dA.numElements(), (long long)dB.numElements(),
		gradMag(dA), gradMag(dB));
	EXPECT_EQ(dB.numElements(), 2) << "dB must be reduced to b's shape [2,1]";
	EXPECT_GT(gradMag(dA), 1e-6f) << "broadcast-mul: grad to large operand is zero";
	EXPECT_GT(gradMag(dB), 1e-6f) << "broadcast-mul: grad to broadcast operand is zero";

	auto lossFunc = [&]() -> float {
		oa::GradNo noGrad;
		auto o = oa::FnMatrix::mul(a, b);
		auto l = oa::FnLoss::mse(o, target);
		(void)testSubmitAndWait(ctx);
		return l.dataAs<const float>()[0];
	};
	auto sa = gradCheckParam(ctx, lossFunc, a, dA.dataAs<const float>(), "a[2,3]");
	auto sb = gradCheckParam(ctx, lossFunc, b, dB.dataAs<const float>(), "b[2,1]");
	EXPECT_EQ(sa.failed + sb.failed, 0) << "broadcast-mul gradient mismatch";
}

TEST(MoeDiag, SliceColumnBackwardMultiUse) {
	setenv("OA_GEMM_FORCE_FP32", "1", 1);
	oa::FnMatrix::setRngSeed(5);
	auto& ctx = oa::ExecutionSession::getActive();
	oa::ExecutionSession::RecordingScope scope(ctx);

	// p[2,3]; out = sum_e slice(p,col e)  → every column sliced once (multi-use).
	auto p = oa::FnMatrix::randN(oa::MatrixShape{2, 3}, oa::ScalarType::Float32);
	auto target = oa::FnMatrix::randN(oa::MatrixShape{2, 1}, oa::ScalarType::Float32);
	p.setRequiresGrad(true);

	oa::GradientTape tape;
	oa::Matrix acc;
	for (oa::I32 e = 0; e < 3; ++e) {
		auto col = oa::FnMatrix::slice(p, 1, e, e + 1);  // [2,1]
		acc = (e == 0) ? col : oa::FnMatrix::add(acc, col);
	}
	auto loss = oa::FnLoss::mse(acc, target);
	tape.backward(loss);
	(void)testSubmitAndWait(ctx);

	auto dP = p.gradMatrix();
	printf("\nSliceMultiUse: dP elems=%lld |dP|=%.5f\n",
		(long long)dP.numElements(), gradMag(dP));
	EXPECT_EQ(dP.numElements(), 6);
	EXPECT_GT(gradMag(dP), 1e-6f) << "slice backward dropped gradient to p";
}

// Minimal: leaf p[T,E], each column sliced then BROADCAST-multiplied with an
// expert[T,D] and accumulated. No softmax. Isolates slice-of-multi-use combined
// with broadcast-mul gradient accumulation.
TEST(MoeDiag, SliceBcastMulMultiUse) {
	setenv("OA_GEMM_FORCE_FP32", "1", 1);
	oa::FnMatrix::setRngSeed(21);
	auto& ctx = oa::ExecutionSession::getActive();
	oa::ExecutionSession::RecordingScope scope(ctx);

	constexpr oa::I32 T = 2, E = 2, D = 3;
	auto p = oa::FnMatrix::randN(oa::MatrixShape{T, E}, oa::ScalarType::Float32);
	oa::Vector<oa::Matrix> experts;
	for (oa::I32 e = 0; e < E; ++e) experts.pushBack(oa::FnMatrix::randN(oa::MatrixShape{T, D}, oa::ScalarType::Float32));
	auto target = oa::FnMatrix::randN(oa::MatrixShape{T, D}, oa::ScalarType::Float32);
	p.setRequiresGrad(true);

	auto fwd = [&]() -> oa::Matrix {
		oa::Matrix out;
		for (oa::I32 e = 0; e < E; ++e) {
			auto gateE = oa::FnMatrix::slice(p, 1, e, e + 1);     // [T,1]
			auto contrib = oa::FnMatrix::mul(experts[e], gateE);  // [T,D] bcast
			out = (e == 0) ? contrib : oa::FnMatrix::add(out, contrib);
		}
		return out;
	};
	oa::GradientTape tape;
	auto out = fwd();
	auto loss = oa::FnLoss::mse(out, target);
	tape.backward(loss);
	(void)testSubmitAndWait(ctx);
	auto dP = p.gradMatrix();
	printf("\nSliceBcastMulMultiUse: |dP|=%.6f\n", gradMag(dP));
	EXPECT_GT(gradMag(dP), 1e-6f) << "slice→bcast-mul multi-use dropped gradient to p";
}

// Mirrors the MoE forward chain with plain leaves (no modules): the gate is a
// softmax over logits, sliced per-expert and broadcast-multiplied with each
// expert output, then summed. Isolates whether the softmax→gate→combine path
// carries gradient back to the logits.
TEST(MoeDiag, SoftmaxGateCombineChain) {
	setenv("OA_GEMM_FORCE_FP32", "1", 1);
	oa::FnMatrix::setRngSeed(9);
	auto& ctx = oa::ExecutionSession::getActive();
	oa::ExecutionSession::RecordingScope scope(ctx);

	constexpr oa::I32 T = 2, E = 2, D = 3;
	auto logits = oa::FnMatrix::randN(oa::MatrixShape{T, E}, oa::ScalarType::Float32);
	oa::Vector<oa::Matrix> experts;
	for (oa::I32 e = 0; e < E; ++e)
		experts.pushBack(oa::FnMatrix::randN(oa::MatrixShape{T, D}, oa::ScalarType::Float32));
	auto target = oa::FnMatrix::randN(oa::MatrixShape{T, D}, oa::ScalarType::Float32);
	logits.setRequiresGrad(true);
	for (auto& ex : experts) ex.setRequiresGrad(true);

	auto fwd = [&]() -> oa::Matrix {
		auto probs = oa::FnMatrix::softmax(logits, 1);   // [T,E]
		oa::Matrix out;
		for (oa::I32 e = 0; e < E; ++e) {
			auto gateE = oa::FnMatrix::slice(probs, 1, e, e + 1);  // [T,1]
			auto contrib = oa::FnMatrix::mul(experts[e], gateE);   // [T,D] bcast
			out = (e == 0) ? contrib : oa::FnMatrix::add(out, contrib);
		}
		return out;
	};

	oa::GradientTape tape;
	auto out  = fwd();
	auto loss = oa::FnLoss::mse(out, target);
	tape.backward(loss);
	(void)testSubmitAndWait(ctx);

	auto dLogits = logits.gradMatrix();
	printf("\nSoftmaxGateChain: |dLogits|=%.6f\n", gradMag(dLogits));
	EXPECT_GT(gradMag(dLogits), 1e-6f) << "softmax→gate→combine dropped gradient to logits";

	auto lossFunc = [&]() -> float {
		oa::GradNo noGrad;
		auto o = fwd();
		auto l = oa::FnLoss::mse(o, target);
		(void)testSubmitAndWait(ctx);
		return l.dataAs<const float>()[0];
	};
	auto sl = gradCheckParam(ctx, lossFunc, logits, dLogits.dataAs<const float>(), "logits");
	EXPECT_EQ(sl.failed, 0) << "logits gradient mismatch through softmax-gate chain";
}

// oa::Linear weight + input gradient in isolation (Linear is used everywhere).
// Previously failed 6–28% under fp32-forced FD. ROOT CAUSE (fixed): the
// fused-bias forward Linear packed inputs to bf16 and ran a bf16
// CoopMat GEMM unconditionally, bypassing oa::GemmRouter — so it ignored
// OA_GEMM_FORCE_FP32. bf16's ~3-digit mantissa added ~4e-3 forward error, which
// both corrupted the FD reference and fed into the analytical gradient via dy.
// Fix: canonical GEMM lowering and LinearBwdWeightBias now honor the flag.
TEST(MoeDiag, LinearWeightAndInput) {
	setenv("OA_GEMM_FORCE_FP32", "1", 1);
	oa::FnMatrix::setRngSeed(43);
	auto& ctx = oa::ExecutionSession::getActive();
	oa::ExecutionSession::RecordingScope scope(ctx);
	constexpr oa::I32 inFeatures = 4, outFeatures = 5, tokenCount = 3;
	oa::Linear lin(inFeatures, outFeatures);
	auto x = oa::FnMatrix::randN(
		oa::MatrixShape{tokenCount, inFeatures}, oa::ScalarType::Float32);
	auto target = oa::FnMatrix::randN(
		oa::MatrixShape{tokenCount, outFeatures}, oa::ScalarType::Float32);
	x.setRequiresGrad(true);
	auto named = lin.allNamedParameterPtrs();
	oa::Parameter* w = findParam(named, "weight");
	ASSERT_NE(w, nullptr);
	oa::GradientTape tape;
	auto out = lin.forward(x);
	auto loss = oa::FnLoss::mse(out, target);
	tape.backward(loss);
	(void)testSubmitAndWait(ctx);
	auto dW = w->data.gradMatrix();
	auto dX = x.gradMatrix();
	auto lossFunc = [&]() -> float {
		oa::GradNo noGrad;
		auto o = lin.forward(x);
		auto l = oa::FnLoss::mse(o, target);
		(void)testSubmitAndWait(ctx);
		return l.dataAs<const float>()[0];
	};
	printf("\nLinear weight+input gradcheck:\n");
	auto sw = gradCheckParam(ctx, lossFunc, w->data, dW.dataAs<const float>(), "weight");
	auto sx = gradCheckParam(ctx, lossFunc, x, dX.dataAs<const float>(), "input");
	EXPECT_EQ(sw.failed + sx.failed, 0) << "Linear gradient wrong";
}

// CPU GROUND-TRUTH for Linear backward. Reads x, W, b, forward-y and target to
// host, computes the analytic gradient by hand (exact fp64), and compares to BOTH
// the GPU autograd gradient and the central finite difference. This decisively
// separates a forward-precision problem (FD ≠ CPU-truth) from a backward-kernel
// problem (autograd ≠ CPU-truth). RESOLVED: the fused-bias forward Linear ran in
// bf16 even under OA_GEMM_FORCE_FP32 (it bypassed oa::GemmRouter); now ~5e-8.
TEST(MoeDiag, LinearCpuGroundTruth) {
	setenv("OA_GEMM_FORCE_FP32", "1", 1);
	oa::FnMatrix::setRngSeed(43);
	auto& ctx = oa::ExecutionSession::getActive();
	oa::ExecutionSession::RecordingScope scope(ctx);
	constexpr oa::I32 inFeatures = 4, outFeatures = 5, tokenCount = 3;
	oa::Linear lin(inFeatures, outFeatures);
	auto x = oa::FnMatrix::randN(
		oa::MatrixShape{tokenCount, inFeatures}, oa::ScalarType::Float32);
	auto target = oa::FnMatrix::randN(
		oa::MatrixShape{tokenCount, outFeatures}, oa::ScalarType::Float32);
	x.setRequiresGrad(true);
	auto named = lin.allNamedParameterPtrs();
	oa::Parameter* w = findParam(named, "weight");
	oa::Parameter* b = findParam(named, "bias");
	ASSERT_NE(w, nullptr);

	oa::GradientTape tape;
	auto out = lin.forward(x);
	auto loss = oa::FnLoss::mse(out, target);
	tape.backward(loss);
	(void)testSubmitAndWait(ctx);

	auto dW = w->data.gradMatrix();
	auto dX = x.gradMatrix();

	// Host copies. y is the GPU forward output.
	const float* hx = x.dataAs<const float>();
	const float* hw = w->data.dataAs<const float>();
	const float* hb = (b ? b->data.dataAs<const float>() : nullptr);
	const float* ht = target.dataAs<const float>();
	const float* hy = out.dataAs<const float>();

	// 1. CPU forward: y_cpu[i,j] = sum_k x[i,k]*W[j,k] + b[j]. compare to GPU y.
	double maxFwdErr = 0.0;
	for (int i = 0; i < tokenCount; ++i)
		for (int j = 0; j < outFeatures; ++j) {
			double acc = hb ? (double)hb[j] : 0.0;
			for (int k = 0; k < inFeatures; ++k)
				acc += (double)hx[i * inFeatures + k] * (double)hw[j * inFeatures + k];
			maxFwdErr = std::max(
				maxFwdErr, std::abs(acc - (double)hy[i * outFeatures + j]));
		}
	printf("\nLinear CPU ground-truth:\n  max |y_gpu - y_cpu| = %.3e (forward precision)\n", maxFwdErr);

	// 2. CPU backward from the *CPU forward* y. dL/dy = (2/(T*out))*(y-target).
	const double scale = 2.0 / (double)(tokenCount * outFeatures);
	std::vector<double> dy(static_cast<size_t>(tokenCount * outFeatures));
	for (int i = 0; i < tokenCount; ++i)
		for (int j = 0; j < outFeatures; ++j) {
			double acc = hb ? (double)hb[j] : 0.0;
			for (int k = 0; k < inFeatures; ++k)
				acc += (double)hx[i * inFeatures + k] * (double)hw[j * inFeatures + k];
			dy[static_cast<size_t>(i * outFeatures + j)] =
				scale * (acc - (double)ht[i * outFeatures + j]);
		}
	// dW[j,k] = sum_i dy[i,j]*x[i,k];  dX[i,k] = sum_j dy[i,j]*W[j,k].
	double maxWErr = 0.0, maxXErr = 0.0;
	for (int j = 0; j < outFeatures; ++j)
		for (int k = 0; k < inFeatures; ++k) {
			double acc = 0.0;
			for (int i = 0; i < tokenCount; ++i)
				acc += dy[static_cast<size_t>(i * outFeatures + j)] *
					(double)hx[i * inFeatures + k];
			maxWErr = std::max(maxWErr,
				std::abs(acc - (double)dW.dataAs<const float>()[j * inFeatures + k]));
		}
	for (int i = 0; i < tokenCount; ++i)
		for (int k = 0; k < inFeatures; ++k) {
			double acc = 0.0;
			for (int j = 0; j < outFeatures; ++j)
				acc += dy[static_cast<size_t>(i * outFeatures + j)] *
					(double)hw[j * inFeatures + k];
			maxXErr = std::max(maxXErr,
				std::abs(acc - (double)dX.dataAs<const float>()[i * inFeatures + k]));
		}
	printf("  max |dW_gpu - dW_cpu| = %.3e\n  max |dX_gpu - dX_cpu| = %.3e\n", maxWErr, maxXErr);
	EXPECT_LT(maxFwdErr, 1e-4) << "FORWARD GEMM is not fp32-accurate (breaks the FD reference)";
	EXPECT_LT(maxWErr, 1e-4) << "weight backward kernel disagrees with CPU truth";
	EXPECT_LT(maxXErr, 1e-4) << "data backward kernel disagrees with CPU truth";
}

// oa::RmsNorm MODULE weight gradient (uses GradRmsNorm — different path than the
// raw oa::FnMatrix::rmsNorm verified in session 3). normed feeds nothing else here.
TEST(MoeDiag, RmsNormModuleWeight) {
	setenv("OA_GEMM_FORCE_FP32", "1", 1);
	oa::FnMatrix::setRngSeed(37);
	auto& ctx = oa::ExecutionSession::getActive();
	oa::ExecutionSession::RecordingScope scope(ctx);
	constexpr oa::I32 D = 4, T = 3;
	oa::RmsNorm norm(D);
	auto x = oa::FnMatrix::randN(oa::MatrixShape{T, D}, oa::ScalarType::Float32);
	auto target = oa::FnMatrix::randN(oa::MatrixShape{T, D}, oa::ScalarType::Float32);
	auto named = norm.allNamedParameterPtrs();
	oa::Parameter* w = findParam(named, "weight");
	if (w == nullptr) { for (auto& np : named) printf("  rmsnorm param: %s\n", np.path.cStr()); FAIL(); }
	oa::GradientTape tape;
	auto out = norm.forward(x);
	auto loss = oa::FnLoss::mse(out, target);
	tape.backward(loss);
	(void)testSubmitAndWait(ctx);
	auto dW = w->data.gradMatrix();
	auto lossFunc = [&]() -> float {
		oa::GradNo noGrad;
		auto o = norm.forward(x);
		auto l = oa::FnLoss::mse(o, target);
		(void)testSubmitAndWait(ctx);
		return l.dataAs<const float>()[0];
	};
	printf("\nRmsNorm MODULE weight gradcheck:\n");
	auto s = gradCheckParam(ctx, lossFunc, w->data, dW.dataAs<const float>(), "weight");
	EXPECT_EQ(s.failed, 0) << "RmsNorm module weight gradient wrong";
}

// Silu backward via autograd (its existing test only checks isfinite).
TEST(MoeDiag, SiluBackward) {
	setenv("OA_GEMM_FORCE_FP32", "1", 1);
	oa::FnMatrix::setRngSeed(41);
	auto& ctx = oa::ExecutionSession::getActive();
	oa::ExecutionSession::RecordingScope scope(ctx);
	auto x = oa::FnMatrix::randN(oa::MatrixShape{3, 4}, oa::ScalarType::Float32);
	auto target = oa::FnMatrix::randN(oa::MatrixShape{3, 4}, oa::ScalarType::Float32);
	x.setRequiresGrad(true);
	oa::GradientTape tape;
	auto out = oa::FnMatrix::silu(x);
	auto loss = oa::FnLoss::mse(out, target);
	tape.backward(loss);
	(void)testSubmitAndWait(ctx);
	auto dX = x.gradMatrix();
	auto lossFunc = [&]() -> float {
		oa::GradNo noGrad;
		auto o = oa::FnMatrix::silu(x);
		auto l = oa::FnLoss::mse(o, target);
		(void)testSubmitAndWait(ctx);
		return l.dataAs<const float>()[0];
	};
	printf("\nSilu backward gradcheck:\n");
	auto s = gradCheckParam(ctx, lossFunc, x, dX.dataAs<const float>(), "silu.x");
	EXPECT_EQ(s.failed, 0) << "Silu backward is wrong";
}

// Is the expert FFN itself gradient-correct in isolation? (Separates an FFN/
// RmsNorm bug from the MoE weighting.) Gradchecks oa::Ffn's RmsNorm weight.
// Previously failed via the SAME fused-bias-Linear bf16 bug as LinearWeightAndInput
// (the FFN's gate/up/down are oa::Linear); fixed with the OA_GEMM_FORCE_FP32 guard.
TEST(MoeDiag, FfnNormWeightStandalone) {
	setenv("OA_GEMM_FORCE_FP32", "1", 1);
	oa::FnMatrix::setRngSeed(31);
	auto& ctx = oa::ExecutionSession::getActive();
	oa::ExecutionSession::RecordingScope scope(ctx);

	constexpr oa::I32 D = 4, DFF = 8, T = 3;
	oa::Ffn ffn(D, DFF);
	auto x = oa::FnMatrix::randN(oa::MatrixShape{T, D}, oa::ScalarType::Float32);
	auto target = oa::FnMatrix::randN(oa::MatrixShape{T, D}, oa::ScalarType::Float32);

	auto named = ffn.allNamedParameterPtrs();
	oa::Parameter* normW = findParam(named, "norm.weight");
	if (normW == nullptr) {
		printf("FFN param paths:\n");
		for (auto& np : named) printf("  %s\n", np.path.cStr());
		FAIL() << "norm.weight not found";
	}

	oa::GradientTape tape;
	auto out  = ffn.forward(x);
	auto loss = oa::FnLoss::mse(out, target);
	tape.backward(loss);
	(void)testSubmitAndWait(ctx);
	auto dNorm = normW->data.gradMatrix();

	auto lossFunc = [&]() -> float {
		oa::GradNo noGrad;
		auto o = ffn.forward(x);
		auto l = oa::FnLoss::mse(o, target);
		(void)testSubmitAndWait(ctx);
		return l.dataAs<const float>()[0];
	};
	printf("\nFFN norm.weight standalone gradcheck:\n");
	auto s = gradCheckParam(ctx, lossFunc, normW->data, dNorm.dataAs<const float>(), "norm.weight");
	EXPECT_EQ(s.failed, 0) << "FFN RmsNorm weight gradient is wrong in isolation (not a MoE bug)";
}

// Dense case (K == E): the selection mask is all-ones, so the forward is fully
// smooth — a clean gradcheck of the entire chain (softmax → gate → expert FFN →
// weighted combine). Verifies both the router weight and an expert weight.
TEST(MoeGradcheck, DenseRouterAndExperts) {
	setenv("OA_GEMM_FORCE_FP32", "1", 1);
	oa::FnMatrix::setRngSeed(7);
	auto& ctx = oa::ExecutionSession::getActive();
	oa::ExecutionSession::RecordingScope scope(ctx);

	constexpr oa::I32 D = 4, DFF = 8, E = 2, K = 2, T = 3;  // K==E → dense, smooth
	oa::Moe moe(D, DFF, E, K);
	// Finite differences repeatedly rebuild the same smooth function. Use the
	// explicit dense oracle here; sparse-executor correctness is covered by the
	// forward/input/parameter parity test in TestMoeSystems.
	moe.setSparseExecution(false);

	auto x      = oa::FnMatrix::randN(oa::MatrixShape{T, D}, oa::ScalarType::Float32);
	auto target = oa::FnMatrix::randN(oa::MatrixShape{T, D}, oa::ScalarType::Float32);

	auto named = moe.allNamedParameterPtrs();
	oa::Parameter* routerW = findParam(named, "router.weight");
	oa::Parameter* expertW = findParam(named, "expert_gate_up_weight");
	ASSERT_NE(routerW, nullptr) << "router.weight not found";
	ASSERT_NE(expertW, nullptr) << "expert_gate_up_weight not found";

	oa::GradientTape tape;
	auto out  = moe.forward(x);
	auto loss = oa::FnLoss::mse(out, target);
	tape.backward(loss);
	(void)testSubmitAndWait(ctx);

	auto dRouter = routerW->data.gradMatrix();
	auto dExpert = expertW->data.gradMatrix();
	ASSERT_EQ(dRouter.numElements(), routerW->data.numElements());
	ASSERT_EQ(dExpert.numElements(), expertW->data.numElements());

	auto lossFunc = [&]() -> float {
		oa::GradNo noGrad;
		auto o = moe.forward(x);
		auto l = oa::FnLoss::mse(o, target);
		(void)testSubmitAndWait(ctx);
		return l.dataAs<const float>()[0];
	};

	// verify both sides of the layer end to end: router selection magnitudes and
	// an actual expert projection weight. This used to stop at "non-zero expert
	// grad" because GEMM backward was suspect; the CPU-ground-truth diagnostics
	// above now make a full finite-difference gate meaningful.
	printf("\nMoE dense gradcheck [D=%d,DFF=%d,E=%d,K=%d,T=%d]:\n", D, DFF, E, K, T);
	auto sr = gradCheckParam(ctx, lossFunc, routerW->data, dRouter.dataAs<const float>(), "router.weight");
	printf("MoE router gradcheck: %d/%d pass, %d non-trivial\n",
		sr.checked - sr.failed, sr.checked, sr.nonTrivial);
	EXPECT_EQ(sr.failed, 0) << "MoE router gradient mismatch (autograd != finite diff)";
	EXPECT_GE(sr.nonTrivial, 3) << "router gradients all ~0 — vacuous check";
	auto se = gradCheckParam(ctx, lossFunc, expertW->data, dExpert.dataAs<const float>(), "expert_gate_up_weight");
	EXPECT_EQ(se.failed, 0) << "MoE expert gradient mismatch (autograd != finite diff)";
	EXPECT_GE(se.nonTrivial, 3) << "expert gradients all ~0 — vacuous check";
}

// Sparse top-k (K=2 of E=4): with K>=2 the renormalized gate still depends on the
// selected probabilities, so the router receives a real task-loss gradient — the
// thing the old stub MoeRouterBwd never produced. (For K==1 the renormalized gate
// is identically 1, so the router gets no signal by construction — a property of
// top-1 renormalized routing, not a bug.) Assert both router and experts learn.
TEST(MoeGradcheck, TopKRouterAndExpertsLearn) {
	setenv("OA_GEMM_FORCE_FP32", "1", 1);
	oa::FnMatrix::setRngSeed(13);
	auto& ctx = oa::ExecutionSession::getActive();
	oa::ExecutionSession::RecordingScope scope(ctx);

	constexpr oa::I32 D = 4, DFF = 8, E = 4, K = 2, T = 4;  // top-2 of 4
	oa::Moe moe(D, DFF, E, K);

	auto x      = oa::FnMatrix::randN(oa::MatrixShape{T, D}, oa::ScalarType::Float32);
	auto target = oa::FnMatrix::randN(oa::MatrixShape{T, D}, oa::ScalarType::Float32);

	auto named = moe.allNamedParameterPtrs();
	oa::Parameter* routerW = findParam(named, "router.weight");
	oa::Parameter* expertW = findParam(named, "expert_gate_up_weight");
	ASSERT_NE(routerW, nullptr);
	ASSERT_NE(expertW, nullptr);

	oa::GradientTape tape;
	auto out  = moe.forward(x);
	auto loss = oa::FnLoss::mse(out, target);
	tape.backward(loss);
	(void)testSubmitAndWait(ctx);

	// Router gets a real task-loss gradient through the selected gate magnitudes
	// (the old stub MoeRouterBwd produced ~0 from the data path).
	EXPECT_GT(gradMag(routerW->data.gradMatrix()), 1e-5f)
		<< "router got no task-loss gradient — routing won't learn (the old stub bug)";
	// Experts receive a real gradient (they are selected and weighted).
	EXPECT_GT(gradMag(expertW->data.gradMatrix()), 1e-5f)
		<< "experts receive no gradient — they won't learn";
}

TEST(MoeConfiguration, RejectsInvalidAndClampsTopK) {
	EXPECT_DEATH((void)oa::Moe(0, 8, 2, 1), "OA contract failed");
	EXPECT_DEATH((void)oa::Moe(4, 0, 2, 1), "OA contract failed");
	EXPECT_DEATH((void)oa::Moe(4, 8, 0, 1), "OA contract failed");

	oa::Moe low(4, 8, 3, 0);
	EXPECT_EQ(low.expertsPerToken(), 1);
	oa::Moe high(4, 8, 3, 99);
	EXPECT_EQ(high.expertsPerToken(), 3);
}

TEST(MoeConfiguration, TiedRouterStillSelectsExactlyK) {
	setenv("OA_GEMM_FORCE_FP32", "1", 1);
	auto& ctx = oa::ExecutionSession::getActive();
	oa::ExecutionSession::RecordingScope scope(ctx);
	constexpr oa::I32 T = 5, D = 4, E = 4, K = 2;
	oa::Moe moe(D, 8, E, K);
	auto named = moe.allNamedParameterPtrs();
	auto* routerW = findParam(named, "router.weight");
	auto* routerB = findParam(named, "router.bias");
	ASSERT_NE(routerW, nullptr);
	ASSERT_NE(routerB, nullptr);
	// flush deferred parameter initialization before overwriting the router with
	// exact zeros; otherwise the queued Xavier dispatch would run afterward.
	ASSERT_TRUE(testSubmitAndWait(ctx).isOk());
	for (oa::I64 i = 0; i < routerW->data.numElements(); ++i) routerW->data.dataAs<float>()[i] = 0.0F;
	for (oa::I64 i = 0; i < routerB->data.numElements(); ++i) routerB->data.dataAs<float>()[i] = 0.0F;

	auto x = oa::FnMatrix::randN(oa::MatrixShape{T, D}, oa::ScalarType::Float32);
	(void)moe.forward(x);
	ASSERT_TRUE(testSubmitAndWait(ctx).isOk());
	const auto& mask = moe.lastSelectionMask();
	ASSERT_EQ(mask.getShape(), (oa::MatrixShape{T, E}));
	const float* values = mask.dataAs<const float>();
	for (oa::I32 t = 0; t < T; ++t) {
		float selected = 0.0F;
		for (oa::I32 e = 0; e < E; ++e) selected += values[t * E + e];
		EXPECT_FLOAT_EQ(selected, static_cast<float>(K));
		EXPECT_FLOAT_EQ(values[t * E + 0], 1.0F);  // deterministic lower-index tie break
		EXPECT_FLOAT_EQ(values[t * E + 1], 1.0F);
		EXPECT_FLOAT_EQ(values[t * E + 2], 0.0F);
		EXPECT_FLOAT_EQ(values[t * E + 3], 0.0F);
	}
}

// ── stage 0: route-utilization telemetry ──────────────────────────────────────
// routeStats must be internally consistent: load fractions and mean probs each
// sum to 1, entropy is normalized to [0,1], and the max-load ratio is >= 1.
TEST(MoeStage0, RouteStatsAreSane) {
	setenv("OA_GEMM_FORCE_FP32", "1", 1);
	oa::FnMatrix::setRngSeed(101);
	auto& ctx = oa::ExecutionSession::getActive();
	oa::ExecutionSession::RecordingScope scope(ctx);
	constexpr oa::I32 T = 16, D = 8, DFF = 16, E = 4, K = 2;
	oa::Moe moe(D, DFF, E, K);
	auto x = oa::FnMatrix::randN(oa::MatrixShape{T, D}, oa::ScalarType::Float32);
	(void)moe.forward(x);
	ASSERT_TRUE(testSubmitAndWait(ctx).isOk());

	auto st = moe.routeStats();
	ASSERT_EQ(st.loadFraction.size(), static_cast<oa::I64>(E));
	ASSERT_EQ(st.meanProb.size(), static_cast<oa::I64>(E));
	float loadSum = 0.0F, probSum = 0.0F;
	for (oa::I32 e = 0; e < E; ++e) { loadSum += st.loadFraction[e]; probSum += st.meanProb[e]; }
	EXPECT_NEAR(loadSum, 1.0F, 1e-4F) << "load fractions must sum to 1";
	EXPECT_NEAR(probSum, 1.0F, 1e-3F) << "mean softmax probs must sum to ~1";
	EXPECT_GE(st.entropy, 0.0F);
	EXPECT_LE(st.entropy, 1.0001F);
	EXPECT_GE(st.maxLoadRatio, 1.0F) << "E*max_e load is >= 1 by definition";
}

// ── stage 0: aux-loss-free balancing rescues a collapsed router ────────────────
// Force a maximally-collapsed router (large static bias to expert 0), then let the
// gradient-free per-expert bias nudge load back toward balance. Asserts the sign
// rule (deterministic) AND that real load spreads (emergent, tokens differ via the
// live router weight). Exercises the [T,E]+[1,E] broadcast in the bias path.
TEST(MoeStage0, BiasBalancingSpreadsCollapsedLoad) {
	setenv("OA_GEMM_FORCE_FP32", "1", 1);
	oa::FnMatrix::setRngSeed(202);
	auto& ctx = oa::ExecutionSession::getActive();
	oa::ExecutionSession::RecordingScope scope(ctx);
	constexpr oa::I32 T = 64, D = 16, DFF = 16, E = 4, K = 1;  // K=1 → collapse is stark
	oa::Moe moe(D, DFF, E, K);
	moe.setBalanceRate(0.3F);  // aggressive for a fast unit test

	auto named = moe.allNamedParameterPtrs();
	auto* rW = findParam(named, "router.weight");
	auto* rB = findParam(named, "router.bias");
	ASSERT_NE(rW, nullptr);
	ASSERT_NE(rB, nullptr);
	// flush deferred init, then bias the router hard toward expert 0. The live
	// (random) weight keeps per-token logits distinct so load CAN spread.
	ASSERT_TRUE(testSubmitAndWait(ctx).isOk());
	for (oa::I64 i = 0; i < rB->data.numElements(); ++i) rB->data.dataAs<float>()[i] = 0.0F;
	rB->data.dataAs<float>()[0] = 4.0F;

	auto x = oa::FnMatrix::randN(oa::MatrixShape{T, D}, oa::ScalarType::Float32);

	(void)moe.forward(x);
	ASSERT_TRUE(testSubmitAndWait(ctx).isOk());
	auto before = moe.routeStats();
	EXPECT_EQ(before.deadExperts, E - 1) << "router should start collapsed onto expert 0";

	for (int s = 0; s < 80; ++s) {
		(void)moe.forward(x);
		ASSERT_TRUE(testSubmitAndWait(ctx).isOk());
		moe.updateRoutingBias();
	}
	(void)moe.forward(x);
	ASSERT_TRUE(testSubmitAndWait(ctx).isOk());
	auto after = moe.routeStats();

	printf("\nBiasBalancing: dead %d→%d  entropy %.3f→%.3f  maxratio %.2f→%.2f  bias[0]=%.2f\n",
		before.deadExperts, after.deadExperts, before.entropy, after.entropy,
		before.maxLoadRatio, after.maxLoadRatio, moe.routingBias(0));
	EXPECT_LT(moe.routingBias(0), 0.0F) << "over-loaded expert 0 must be pushed down (sign rule)";
	EXPECT_LT(after.deadExperts, before.deadExperts) << "balancing must revive dead experts";
	EXPECT_LT(after.maxLoadRatio, before.maxLoadRatio) << "peak load must drop toward balance";
}

// ── stage 0: shared always-on expert always receives gradient ─────────────────
// With K=1 the routed experts are mostly idle, but a shared expert is applied to
// every token unconditionally, so it must always get a gradient. Also verifies the
// shared-expert forward/backward path and its registration.
TEST(MoeStage0, SharedExpertAlwaysContributes) {
	setenv("OA_GEMM_FORCE_FP32", "1", 1);
	oa::FnMatrix::setRngSeed(303);
	auto& ctx = oa::ExecutionSession::getActive();
	oa::ExecutionSession::RecordingScope scope(ctx);
	constexpr oa::I32 T = 4, D = 4, DFF = 8, E = 4, K = 1;
	oa::Moe moe(D, DFF, E, K, 1e-5F, /*shared*/ 1);
	EXPECT_EQ(moe.numSharedExperts(), 1);

	auto named = moe.allNamedParameterPtrs();
	auto* shW = findParam(named, "shared_expert.0.gate_weight");
	ASSERT_NE(shW, nullptr) << "shared expert must be registered so the optimizer sees it";

	auto x = oa::FnMatrix::randN(oa::MatrixShape{T, D}, oa::ScalarType::Float32);
	auto target = oa::FnMatrix::randN(oa::MatrixShape{T, D}, oa::ScalarType::Float32);
	oa::GradientTape tape;
	auto out = moe.forward(x);
	auto loss = oa::FnLoss::mse(out, target);
	tape.backward(loss);
	ASSERT_TRUE(testSubmitAndWait(ctx).isOk());
	EXPECT_GT(gradMag(shW->data.gradMatrix()), 1e-6F)
		<< "shared expert is always on — it must always receive gradient";
}

// ── stage 0: opt-in switch aux loss + router z-loss are finite + differentiable ─
TEST(MoeStage0, AuxLossIsFiniteAndDifferentiable) {
	setenv("OA_GEMM_FORCE_FP32", "1", 1);
	oa::FnMatrix::setRngSeed(404);
	auto& ctx = oa::ExecutionSession::getActive();
	oa::ExecutionSession::RecordingScope scope(ctx);
	constexpr oa::I32 T = 8, D = 4, DFF = 8, E = 4, K = 2;
	oa::Moe moe(D, DFF, E, K);
	moe.setAuxLossAlpha(0.01F);
	moe.setRouterZLossBeta(0.001F);

	auto named = moe.allNamedParameterPtrs();
	auto* rW = findParam(named, "router.weight");
	ASSERT_NE(rW, nullptr);

	auto x = oa::FnMatrix::randN(oa::MatrixShape{T, D}, oa::ScalarType::Float32);
	auto target = oa::FnMatrix::randN(oa::MatrixShape{T, D}, oa::ScalarType::Float32);
	oa::GradientTape tape;
	auto out = moe.forward(x);
	auto aux = moe.auxLoss();  // handle to the scalar recorded on this tape
	auto loss = oa::FnMatrix::add(oa::FnLoss::mse(out, target), aux);
	tape.backward(loss);
	ASSERT_TRUE(testSubmitAndWait(ctx).isOk());

	const float av = moe.auxLoss().dataAs<const float>()[0];
	printf("\nAuxLoss (switch α=0.01 + z β=0.001) = %.6f\n", av);
	EXPECT_TRUE(std::isfinite(av));
	EXPECT_GT(av, 0.0F) << "switch + z losses are positive for a non-degenerate router";
	EXPECT_GT(gradMag(rW->data.gradMatrix()), 1e-6F) << "router must receive gradient (incl. aux)";
}

TEST(MoeStage0, DefaultsAreInertAndAuxLossResets) {
	setenv("OA_GEMM_FORCE_FP32", "1", 1);
	oa::FnMatrix::setRngSeed(505);
	auto& ctx = oa::ExecutionSession::getActive();
	oa::ExecutionSession::RecordingScope scope(ctx);
	oa::Moe moe(4, 8, 4, 2);
	EXPECT_FLOAT_EQ(moe.balanceRate(), 0.0F);
	for (oa::I32 e = 0; e < 4; ++e) EXPECT_FLOAT_EQ(moe.routingBias(e), 0.0F);

	auto x = oa::FnMatrix::randN(oa::MatrixShape{8, 4}, oa::ScalarType::Float32);
	moe.setAuxLossAlpha(0.01F);
	(void)moe.forward(x);
	ASSERT_TRUE(testSubmitAndWait(ctx).isOk());

	moe.setAuxLossAlpha(0.0F);
	(void)moe.forward(x);
	ASSERT_TRUE(testSubmitAndWait(ctx).isOk());
	EXPECT_FLOAT_EQ(moe.auxLoss().dataAs<const float>()[0], 0.0F)
		<< "disabling aux loss must not expose the previous forward's stale scalar";
}

TEST(MoeStage0, RoutingBiasCheckpoints) {
	setenv("OA_GEMM_FORCE_FP32", "1", 1);
	oa::FnMatrix::setRngSeed(606);
	auto& ctx = oa::ExecutionSession::getActive();
	oa::ExecutionSession::RecordingScope scope(ctx);
	constexpr oa::I32 E = 4;
	oa::Moe moe(8, 16, E, 1);
	moe.setBalanceRate(0.25F);

	auto named = moe.allNamedParameterPtrs();
	auto* rB = findParam(named, "router.bias");
	ASSERT_NE(rB, nullptr);
	ASSERT_TRUE(testSubmitAndWait(ctx).isOk());
	for (oa::I64 i = 0; i < rB->data.numElements(); ++i) rB->data.dataAs<float>()[i] = 0.0F;
	rB->data.dataAs<float>()[0] = 5.0F;

	auto x = oa::FnMatrix::randN(oa::MatrixShape{16, 8}, oa::ScalarType::Float32);
	(void)moe.forward(x);
	ASSERT_TRUE(testSubmitAndWait(ctx).isOk());
	moe.updateRoutingBias();
	const float expected = moe.routingBias(0);
	ASSERT_LT(expected, 0.0F);

	const oa::String path = "/tmp/oa_moe_routing_bias.oam";
	ASSERT_TRUE(moe.save(testEngine(), path).isOk());
	oa::Moe loaded(8, 16, E, 1);
	ASSERT_TRUE(loaded.load(testEngine(), path).isOk());
	EXPECT_FLOAT_EQ(loaded.routingBias(0), expected);
}
