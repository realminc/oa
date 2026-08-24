// OA example — Alm
//
// A port of MotionGPT into OA: a two-stage model that generates
// human motion as a stream of discrete tokens.
//
//   stage 1  VQ-VAE tokenizer   — motion to discrete tokens
//   stage 2  Autoregressive LM  — token generation
//   generate                    — sample tokens, decode, USD output
//
// run tests:
//   ./Alm --gtest_filter="Alm.*"

#include "oaTest.h"
#include <ml/nn/alm/almConfig.h>
#include <ml/nn/alm/almAg.h>
#include <ml/nn/alm/almTokenizerAg.h>
#include <ml/nn/alm/almPriorAg.h>
#include <anim/usd.h>
#include <oa/core/filesystem.h>
#include <oa/core/paths.h>
#include <oa/core/vlm.h>
#include <data/dsHumanMl3d.h>
#include <oa/ml.h>
#include <oa/ml/autograd.h>
#include <oa/ml/fnOptim.h>
#include <oa/ml/modelFile.h>
#include <oa/runtime/executionSession.h>
#include <rig/skeleton.h>
#include <rig/skeletonUsd.h>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>
#include <chrono>
#include <functional>

// Copy a matrix to host FP32. Safe for BF16/FP16 storage models.
static oa::Vec<oa::F32> hostFloatData(const oa::Matrix& inMatrix) {
	auto& ctx = oa::ExecutionSession::getActive();
	if (inMatrix.getDtype() == oa::ScalarType::Float32) {
		(void)testSubmitAndWait(ctx);
		const oa::F32* p = inMatrix.dataAs<const oa::F32>();
		return oa::Vec<oa::F32>(p, p + inMatrix.numElements());
	}
	oa::Matrix f32 = oa::FnMatrix::empty(inMatrix.getShape(), oa::ScalarType::Float32);
	oa::FnMatrix::castInto(inMatrix, f32);
	(void)testSubmitAndWait(ctx);
	const oa::F32* p = f32.dataAs<const oa::F32>();
	return oa::Vec<oa::F32>(p, p + f32.numElements());
}

// Dot product on host FP32. Safe for BF16/FP16 storage models.
static double hostDot(const oa::Matrix& inA, const oa::Matrix& inB) {
	auto a = hostFloatData(inA);
	auto b = hostFloatData(inB);
	OA_ASSERT(a.size() == b.size() && "HostDot: size mismatch");
	double s = 0.0;
	for (size_t i = 0; i < a.size(); ++i) { s += static_cast<double>(a[i]) * static_cast<double>(b[i]); }
	return s;
}

// True if every element is finite. Safe for BF16/FP16 storage models.
static bool hostAllFinite(const oa::Matrix& inMatrix) {
	auto h = hostFloatData(inMatrix);
	for (size_t i = 0; i < h.size(); ++i) { if (not std::isfinite(h[i])) return false; }
	return true;
}

// Config implementation

oa::AlmDatasetConfig oa::AlmDatasetConfig::fromEnv() {
	oa::AlmDatasetConfig cfg;
	cfg.corpus = "cmp";
	cfg.dataDir = oa::Paths::data("humanMl3d/Cmp").string();
	cfg.split = "train";
	cfg.maxClips = 0;
	
	// Override from environment
	if (const char* corpus = std::getenv("OA_MOTION_CORPUS")) {
		cfg.corpus = corpus;
	}
	if (const char* dataDir = std::getenv("OA_MOTION_DATA")) {
		cfg.dataDir = dataDir;
	}
	if (const char* split = std::getenv("OA_MOTION_SPLIT")) {
		cfg.split = split;
	}
	if (const char* maxClips = std::getenv("OA_MOTION_MAX_CLIPS")) {
		cfg.maxClips = std::atoi(maxClips);
	}
	
	return cfg;
}

// Test cases

TEST(Alm, ConfigTest) {
	auto datasetCfg = oa::AlmDatasetConfig::fromEnv();
	std::printf("Dataset: %s\n", datasetCfg.corpus.cStr());
	std::printf("dataDir: %s\n", datasetCfg.dataDir.cStr());
	
	oa::AlmTokenizerConfig tokCfg;
	std::printf("tokenizer: inputDim=%d, numCodes=%d\n", tokCfg.inputDim, tokCfg.numCodes);
	
	oa::AlmPriorConfig lmCfg;
	lmCfg.syncVocab(tokCfg.numCodes);
	std::printf("LM: vocabSize=%d, dModel=%d, numHeads=%d\n",
		lmCfg.vocabSize, lmCfg.dModel, lmCfg.numHeads);
	
	EXPECT_EQ(lmCfg.vocabSize, tokCfg.numCodes + 3);
}

// Encode → quantize → Decode round-trip: the 8× temporal downsample must show up in
// the token count (T → T/Factor) and the decode must restore the frame count (T), with
// finite values throughout.
TEST(Alm, TokenizerRoundTripShape) {
	oa::AlmTokenizerConfig cfg;
	cfg.inputDim = 48; cfg.width = 64; cfg.codeDim = 32; cfg.numCodes = 64;
	cfg.downT = 3; cfg.depth = 1;                       // factor 8
	auto tok = oa::makeShared<oa::AlmTokenizerAg>(cfg);
	ASSERT_EQ(tok->downsampleFactor(), 8);

	const oa::I32 B = 2;
	const oa::I32 T = 32;                                 // multiple of 8 → 4 tokens/seq
	auto& ctx = oa::ExecutionSession::getActive();
	auto x = oa::FnMatrix::randN(oa::MatrixShape{B, T, cfg.inputDim});
	auto z = tok->encode(x, B, T);                      // [B·T/8, codeDim]
	(void)testSubmitAndWait(ctx);
	EXPECT_EQ(z.size(0), static_cast<oa::I64>(B) * (T / 8));
	EXPECT_EQ(z.size(1), cfg.codeDim);

	auto q   = tok->quantize(z);
	auto rec = tok->decode(q.quantized, B, T / 8);      // [B·T, inputDim]
	(void)testSubmitAndWait(ctx);
	EXPECT_EQ(rec.size(0), static_cast<oa::I64>(B) * T);
	EXPECT_EQ(rec.size(1), cfg.inputDim);
	ASSERT_TRUE(hostAllFinite(rec)) << "decoded tensor has non-finite values";
	ctx.clear();
}

// exact gradient check for the new oa::ConvTranspose1d op via the bilinear identity:
// y = convT(x; W) is bilinear, so for any cotangent g,
//   <y, g> == <x, ∂/∂x> == <W, ∂/∂W>   i.e.  sum(y*g) == sum(x*x.grad) == sum(W*W.grad).
// This validates BOTH the dX (adjoint = Conv1d) and dW (= Conv1dBwdWeight) paths with no
// finite-difference epsilon. If the adjoint were wrong, these three would disagree.
TEST(Alm, ConvTranspose1dGradCheck) {
	oa::FnMatrix::setRngSeed(123);
	const oa::I32 inC = 3, outC = 2, K = 4, S = 2, P = 1, B = 2, L = 5;
	auto ct = oa::makeShared<oa::ConvTranspose1d>(inC, outC, K, S, P);
	auto& ctx = oa::ExecutionSession::getActive();

	auto x = oa::FnMatrix::randN(oa::MatrixShape{B, inC, L});
	x.setRequiresGrad(true);
	auto& W = ct->parameters()[0].data;             // [inC, outC, K]
	auto g = oa::FnMatrix::randN(oa::MatrixShape{B, outC, (L - 1) * S - 2 * P + K});

	oa::GradientTape tape;
	auto y = ct->forward(x);
	auto loss = oa::FnMatrix::sum(oa::FnMatrix::mul(y, g));
	tape.backward(loss);
	(void)testSubmitAndWait(ctx);

	const double sYg = static_cast<double>(loss.at(0));
	auto dx = x.gradMatrix();
	auto dW = W.gradMatrix();
	double sXdx = hostDot(x, dx);
	double sWdW = hostDot(W, dW);
	std::printf("ConvT grad-check: <y,g>=%.8f  <x,dx>=%.8f  <W,dW>=%.8f\n", sYg, sXdx, sWdW);
	EXPECT_NEAR(sXdx, sYg, 1e-3 * (1.0 + std::abs(sYg))) << "dX adjoint wrong";
	EXPECT_NEAR(sWdW, sYg, 1e-3 * (1.0 + std::abs(sYg))) << "dW wrong";
	ctx.clear();
}

// Same bilinear grad-check for the STOCK oa::Conv1d (the suspect behind every conv-VQ
// NaN in OA). With zero bias, y = Conv1d(x,W) is bilinear, so
//   sum(y*g) == sum(x*x.grad) == sum(W*W.grad).  Checked at stride 1 AND stride 2.
static void convGradCheckImpl(oa::I32 S) {
	oa::FnMatrix::setRngSeed(321);
	const oa::I32 inC = 3, outC = 4, K = 3, P = 1, B = 2, L = 8;
	auto cv = oa::makeShared<oa::Conv1d>(inC, outC, K, S, P);
	auto& ctx = oa::ExecutionSession::getActive();
	auto x = oa::FnMatrix::randN(oa::MatrixShape{B, inC, L});
	x.setRequiresGrad(true);
	auto& W = cv->parameters()[0].data;
	auto y0 = cv->forward(x);                            // realize to read output length
	(void)testSubmitAndWait(ctx);
	const oa::I64 Lout = y0.size(2);
	auto g = oa::FnMatrix::randN(oa::MatrixShape{B, outC, Lout});

	oa::GradientTape tape;
	auto y = cv->forward(x);
	auto loss = oa::FnMatrix::sum(oa::FnMatrix::mul(y, g));
	tape.backward(loss);
	(void)testSubmitAndWait(ctx);
	const double sYg = static_cast<double>(loss.at(0));
	auto dx = x.gradMatrix();  auto dW = W.gradMatrix();
	double sXdx = hostDot(x, dx);
	double sWdW = hostDot(W, dW);
	std::printf("Conv1d(stride=%d) grad-check: <y,g>=%.8f  <x,dx>=%.8f  <W,dW>=%.8f\n", S, sYg, sXdx, sWdW);
	EXPECT_NEAR(sXdx, sYg, 1e-3 * (1.0 + std::abs(sYg))) << "Conv1d stride " << S << " dX wrong";
	EXPECT_NEAR(sWdW, sYg, 1e-3 * (1.0 + std::abs(sYg))) << "Conv1d stride " << S << " dW wrong";
	ctx.clear();
}
TEST(Alm, Conv1dGradCheckStride1) { convGradCheckImpl(1); }
TEST(Alm, Conv1dGradCheckStride2) { convGradCheckImpl(2); }

// conv1dGemm (im2col + tensor-core matmul) gradient self-consistency at tokenizer
// shapes. The bilinear grad-check (zero bias) validates the Im2Col1d adjoint plus
// the composed reshape/transpose/Linear gradient with no finite-difference eps.
// (forward correctness vs a CPU reference is covered by
// NN.conv1dGemmMatchesCpuReference in TestNnKernels.)
static void conv1dGemmParityImpl(oa::I32 S) {
	oa::FnMatrix::setRngSeed(4242);
	const oa::I32 B = 8, inC = 96, outC = 128, K = 3, P = 1, L = 64;
	auto& ctx = oa::ExecutionSession::getActive();

	// Isolated Im2Col1d grad-check: cols is linear in x, so <cols,g> == <x,dx>.
	{
		auto xi = oa::FnMatrix::randN(oa::MatrixShape{B, inC, L});
		xi.setRequiresGrad(true);
		oa::GradientTape t2;
		auto cols = oa::FnMatrix::im2Col1d(xi, K, S, P, 1);
		auto gc = oa::FnMatrix::randN(cols.getShape());
		auto lc = oa::FnMatrix::sum(oa::FnMatrix::mul(cols, gc));
		t2.backward(lc);
		(void)testSubmitAndWait(ctx);
		const double sc = static_cast<double>(lc.at(0));
		auto dxi = xi.gradMatrix();
		double sXdxi = hostDot(xi, dxi);
		std::printf("Im2Col1d-only grad (S=%d): <cols,g>=%.6f <x,dx>=%.6f\n", S, sc, sXdxi);
		ctx.clear();
	}

	// Bilinear grad-check (zero bias): sum(y*g) == <x,dx> == <w,dw>.
	auto xg = oa::FnMatrix::randN(oa::MatrixShape{B, inC, L});
	xg.setRequiresGrad(true);
	auto wg = oa::FnMatrix::randN(oa::MatrixShape{outC, inC, K});
	wg.setRequiresGrad(true);
	auto zb = oa::FnMatrix::zeros(oa::MatrixShape{outC});

	oa::GradientTape tape;
	auto y = oa::FnMatrix::conv1dGemm(xg, wg, zb, S, P, 1);
	auto g = oa::FnMatrix::randN(y.getShape());
	auto loss = oa::FnMatrix::sum(oa::FnMatrix::mul(y, g));
	tape.backward(loss);
	(void)testSubmitAndWait(ctx);

	const double sYg = static_cast<double>(loss.at(0));
	auto dx = xg.gradMatrix();
	auto dw = wg.gradMatrix();
	double sXdx = hostDot(xg, dx);
	double sWdW = hostDot(wg, dw);
	std::printf("Conv1dGemm grad-check (S=%d): <y,g>=%.6f <x,dx>=%.6f <w,dw>=%.6f\n", S, sYg, sXdx, sWdW);
	EXPECT_NEAR(sXdx, sYg, 1e-3 * (1.0 + std::abs(sYg))) << "Im2Col1d dX adjoint wrong";
	EXPECT_NEAR(sWdW, sYg, 1e-3 * (1.0 + std::abs(sYg))) << "Conv1dGemm dW wrong";
	ctx.clear();
}
TEST(Alm, Conv1dGemmParityStride1) { conv1dGemmParityImpl(1); }
TEST(Alm, Conv1dGemmParityStride2) { conv1dGemmParityImpl(2); }

// Perf: scalar direct Conv1d vs im2col+GEMM, at the tokenizer's workhorse shapes.
// Reports:
//   GPU ms/fwd   — isolated GPU compute (batched N copies, one submit/sync, GPU timer)
//   wall ms/step — realistic training step (fwd + backward + sync per iter)
//   sps          — samples/sec = B / (wall ms/step / 1000), B=32
namespace {
double nowMs() {
	return std::chrono::duration<double, std::milli>(
		std::chrono::high_resolution_clock::now().time_since_epoch()).count();
}
// Batched-wall ms/iter ≈ isolated GPU compute: record N copies into one command
// buffer (oa::GradNo, no grad graph), a single execute+Sync, wall/N. With the GPU
// pipeline kept full, per-call CPU submit + sync-drain overhead is amortized to
// ~0, so this separates real compute from the per-step sync stall.
double batchedMsPerFwd(const std::function<oa::Matrix()>& inFn, oa::I32 inN) {
	auto& ctx = oa::ExecutionSession::getActive();
	{
		oa::GradNo nog;
		for (oa::I32 i = 0; i < 5; ++i) { auto y = inFn(); (void)y; }  // warmup
		(void)testSubmitAndWait(ctx); ctx.clear();
		double t0 = nowMs();
		for (oa::I32 i = 0; i < inN; ++i) { auto y = inFn(); (void)y; }
		(void)testSubmitAndWait(ctx);
		double per = (nowMs() - t0) / inN;
		ctx.clear();
		return per;
	}
}
double wallMsPerStep(const std::function<oa::Matrix()>& inFwd, oa::I32 inN) {
	auto& ctx = oa::ExecutionSession::getActive();
	for (oa::I32 i = 0; i < 5; ++i) {
		oa::GradientTape tape; auto y = inFwd(); auto g = oa::FnMatrix::randN(y.getShape());
		auto loss = oa::FnMatrix::sum(oa::FnMatrix::mul(y, g)); tape.backward(loss);
		(void)testSubmitAndWait(ctx); ctx.clear();
	}
	double t0 = nowMs();
	for (oa::I32 i = 0; i < inN; ++i) {
		oa::GradientTape tape; auto y = inFwd(); auto g = oa::FnMatrix::randN(y.getShape());
		auto loss = oa::FnMatrix::sum(oa::FnMatrix::mul(y, g)); tape.backward(loss);
		(void)testSubmitAndWait(ctx); ctx.clear();
	}
	return (nowMs() - t0) / inN;
}
void convPerfImpl(const char* inTag, oa::I32 inC, oa::I32 outC, oa::I32 K, oa::I32 S, oa::I32 P) {
	auto& ctx = oa::ExecutionSession::getActive();
	const oa::I32 B = 32, L = 64;
	auto x = oa::FnMatrix::randN(oa::MatrixShape{B, inC, L});
	auto w = oa::FnMatrix::randN(oa::MatrixShape{outC, inC, K});
	auto b = oa::FnMatrix::randN(oa::MatrixShape{outC});
	const int n = 50;
	auto fwdGemm = [&]{ return oa::FnMatrix::conv1dReluGemm(x, w, b, S, P, 1); };
	double gpuG = batchedMsPerFwd(fwdGemm, n);
	double stepG = wallMsPerStep(fwdGemm, n);
	double spsG = 1000.0 * B / stepG;
	std::printf("[perf %s] inC=%d outC=%d K=%d S=%d | batched ms/fwd=%.4f | wall ms/step=%.4f | sps=%.0f\n",
		inTag, inC, outC, K, S, gpuG, stepG, spsG);
	(void)ctx;
}
}
TEST(Alm, Conv1dGemmPerf) {
	const char* prec = testEngine().getPrecision() == oa::Precision::BF16 ? "BF16" : "FP32";
	std::printf("[perf] engine precision = %s\n", prec);
	convPerfImpl("W384-K3", 384, 384, 3, 1, 1);   // res-block workhorse (x12 fwd)
	convPerfImpl("W384-K4S2", 384, 384, 4, 2, 1); // strided down-conv (per stage)
	convPerfImpl("in263-W384", 263, 384, 3, 1, 1); // enc_in
}

// Full-tokenizer-step perf: scalar vs GEMM at training config shapes.
// Measures the real end-to-end sps win from wiring Conv1dGemm into the tokenizer.
// One training step = Encode + quantize + Decode + Mse + backward + optimizer step.
// The conv kernels dominate; the GEMM path routes them through the tiled matmul
// stack instead of the scalar direct-conv loop.
TEST(Alm, TokenizerStepPerfGemm) {
	oa::FnMatrix::setRngSeed(42);
	auto& ctx = oa::ExecutionSession::getActive();
	const oa::I32 B = 8, T = 64;
	const oa::I32 inDim = 263;

	auto runSteps = [&](oa::I32 NumSteps) -> double {
		oa::AlmTokenizerConfig cfg;
		cfg.inputDim = inDim;
		cfg.width = 384;
		cfg.codeDim = 384;
		cfg.numCodes = 128;
		cfg.downT = 2;
		cfg.depth = 2;
		auto tok = oa::makeShared<oa::AlmTokenizerAg>(cfg);
		const oa::I32 tokLen = T / tok->downsampleFactor();

		// Synthetic batch.
		std::vector<float> xh(static_cast<size_t>(B) * T * inDim);
		oa::U64 rng = 0xDEADULL;
		for (auto& v : xh) {
			rng = (rng * 6364136223846793005ULL) + 1;
			v = static_cast<float>(static_cast<oa::U32>(rng >> 40)) / static_cast<float>(1 << 24);
		}
		auto X = oa::FnMatrix::fromBytes(
			oa::Span<const oa::U8>(reinterpret_cast<const oa::U8*>(xh.data()), xh.size() * sizeof(float)),
			oa::MatrixShape{B, T, inDim}, oa::ScalarType::Float32);
		auto xFlat = X.reshape(oa::MatrixShape{static_cast<oa::I64>(B) * T, inDim});

		// seed codebook.
		{ auto z0 = tok->encode(X, B, T); tok->seed(z0); ctx.clear(); }

		auto params = tok->allParameterPtrs();
		auto opt = oa::makeUnique<oa::AdamW>(params, 1e-4F, 0.9F, 0.99F, 1e-8F, 0.01F);

		// warmup (3 steps).
		for (oa::I32 s = 0; s < 3; ++s) {
			ctx.clear();
			oa::GradientTape tape; opt->zeroGrad();
			auto z = tok->encode(X, B, T);
			auto q = tok->quantize(z);
			auto rec = tok->decode(q.quantized, B, tokLen);
			auto loss = oa::FnLoss::mse(rec, xFlat) + q.commitLoss;
			tape.backward(loss);
			opt->step();
			tok->emaUpdate(q);
			(void)testSubmitAndWait(ctx);
		}

		// Timed.
		double t0 = nowMs();
		for (oa::I32 s = 0; s < NumSteps; ++s) {
			ctx.clear();
			oa::GradientTape tape; opt->zeroGrad();
			auto z = tok->encode(X, B, T);
			auto q = tok->quantize(z);
			auto rec = tok->decode(q.quantized, B, tokLen);
			auto loss = oa::FnLoss::mse(rec, xFlat) + q.commitLoss;
			tape.backward(loss);
			opt->step();
			tok->emaUpdate(q);
			(void)testSubmitAndWait(ctx);
		}
		double total = nowMs() - t0;
		opt.reset();
		tok.reset();
		ctx.clear();
		return total / NumSteps;
	};

	const oa::I32 steps = 10;
	double msGemm = runSteps(steps);
	double spsGemm = 1000.0 * B / msGemm;
	std::printf("[tokenizer-step-perf] B=%d T=%d W=384 downT=2 depth=2\n"
	            "  gemm: %.2f ms/step  sps=%.0f\n",
	            B, T, msGemm, spsGemm);
}

// Most atomic isolation: a SINGLE Conv1d trained to the identity (target == input).
// gradients are verified correct, so SGD at a small lr MUST reduce MSE. If it climbs,
// the bug is the conv weight update / optimizer step on 3D conv weights — not the graph.
TEST(Alm, SingleConvIdentity) {
	oa::FnMatrix::setRngSeed(4);
	const oa::I32 C = 8, L = 16, B = 16;
	auto cv = oa::makeShared<oa::Conv1d>(C, C, 3, 1, 1);
	auto& ctx = oa::ExecutionSession::getActive();
	auto X = oa::FnMatrix::randN(oa::MatrixShape{B, C, L});
	(void)testSubmitAndWait(ctx);   // realize X before the loop
	auto params = cv->allParameterPtrs();
	auto opt = oa::makeUnique<oa::AdamW>(params, 0.01F);
	oa::F32 first = 0.0F, last = 0.0F;
	for (oa::I32 s = 1; s <= 60; ++s) {
		ctx.clear();
		oa::GradientTape tape; opt->zeroGrad();
		auto y  = cv->forward(X);
		auto y2 = y.reshape(oa::MatrixShape{B, static_cast<oa::I64>(C) * L});
		auto x2 = X.reshape(oa::MatrixShape{B, static_cast<oa::I64>(C) * L});
		auto loss = oa::FnLoss::mse(y2, x2);
		tape.backward(loss);
		opt->step();
		(void)testSubmitAndWait(ctx);
		const float lv = loss.at(0);
		if (s == 1) {
			auto y2H = hostFloatData(y2);
			auto x2H = hostFloatData(x2);
			double sy = 0.0, sx = 0.0, sd = 0.0;
			const oa::I64 n = y2.numElements();
			for (oa::I64 i = 0; i < n; ++i) {
				sy += std::abs(static_cast<double>(y2H[i]));
				sx += std::abs(static_cast<double>(x2H[i]));
				const double d = static_cast<double>(y2H[i]) - static_cast<double>(x2H[i]);
				sd += d * d;
			}
			std::printf("  [1conv DIAG] n=%lld sum|y|=%.6f sum|x|=%.6f manualMSE=%.8f OaMse=%.8f\n",
				static_cast<long long>(n), sy, sx, sd / static_cast<double>(n), static_cast<double>(lv));
		}
		if (s == 1) first = lv;
		last = lv;
		if (s == 1 || s % 15 == 0) std::printf("  [1conv] step %2d | mse %.8f\n", s, static_cast<double>(lv));
	}
	EXPECT_LT(last, first) << "single Conv1d must learn identity";
	ctx.clear();
}

TEST(Alm, SingleConvTransposeIdentity) {
	oa::FnMatrix::setRngSeed(7);
	const oa::I32 C = 8, Lin = 16, B = 16;
	const oa::I32 K = 3, S = 1, P = 1;
	const oa::I32 Lout = (Lin - 1) * S - 2 * P + K;  // = 16
	auto ct = oa::makeShared<oa::ConvTranspose1d>(C, C, K, S, P);
	auto& ctx = oa::ExecutionSession::getActive();
	auto X = oa::FnMatrix::randN(oa::MatrixShape{B, C, Lin});
	(void)testSubmitAndWait(ctx);
	auto params = ct->allParameterPtrs();
	auto opt = oa::makeUnique<oa::AdamW>(params, 0.01F);
	oa::F32 first = 0.0F, last = 0.0F;
	for (oa::I32 s = 1; s <= 60; ++s) {
		ctx.clear();
		oa::GradientTape tape; opt->zeroGrad();
		auto y  = ct->forward(X);
		auto y2 = y.reshape(oa::MatrixShape{B, static_cast<oa::I64>(C) * Lout});
		auto x2 = X.reshape(oa::MatrixShape{B, static_cast<oa::I64>(C) * Lin});
		auto loss = oa::FnLoss::mse(y2, x2);
		tape.backward(loss);
		opt->step();
		(void)testSubmitAndWait(ctx);
		const float lv = loss.at(0);
		if (s == 1) first = lv;
		last = lv;
		if (s == 1 || s % 15 == 0) std::printf("  [1convt] step %2d | mse %.8f\n", s, static_cast<double>(lv));
	}
	EXPECT_LT(last, first) << "single ConvTranspose1d must learn identity";
	ctx.clear();
}

TEST(Alm, ConvAutoEncoderIdentity) {
	oa::FnMatrix::setRngSeed(11);
	const oa::I32 C = 8, L = 16, B = 16;
	const oa::I32 K = 3, S = 1, P = 1;
	auto enc = oa::makeShared<oa::Conv1d>(C, C, K, S, P);
	auto dec = oa::makeShared<oa::ConvTranspose1d>(C, C, K, S, P);
	auto& ctx = oa::ExecutionSession::getActive();
	auto X = oa::FnMatrix::randN(oa::MatrixShape{B, C, L});
	(void)testSubmitAndWait(ctx);
	oa::Vec<oa::Parameter*> params;
	auto ep = enc->allParameterPtrs();
	auto dp = dec->allParameterPtrs();
	for (auto* p : ep) params.pushBack(p);
	for (auto* p : dp) params.pushBack(p);
	auto opt = oa::makeUnique<oa::AdamW>(params, 0.01F);
	oa::F32 first = 0.0F, last = 0.0F;
	for (oa::I32 s = 1; s <= 60; ++s) {
		ctx.clear();
		oa::GradientTape tape; opt->zeroGrad();
		auto z   = enc->forward(X);
		auto y   = dec->forward(z);
		auto y2  = y.reshape(oa::MatrixShape{B, static_cast<oa::I64>(C) * L});
		auto x2  = X.reshape(oa::MatrixShape{B, static_cast<oa::I64>(C) * L});
		auto loss = oa::FnLoss::mse(y2, x2);
		tape.backward(loss);
		opt->step();
		(void)testSubmitAndWait(ctx);
		const float lv = loss.at(0);
		if (s == 1) first = lv;
		last = lv;
		if (s == 1 || s % 15 == 0) std::printf("  [convae] step %2d | mse %.8f\n", s, static_cast<double>(lv));
	}
	EXPECT_LT(last, first) << "Conv1d→ConvTranspose1d autoencoder must learn identity";
	ctx.clear();
}

TEST(Alm, ConvAutoEncoderStride2) {
	oa::FnMatrix::setRngSeed(13);
	const oa::I32 C = 8, L = 16, B = 16;
	const oa::I32 K = 4, S = 2, P = 1;
	auto enc = oa::makeShared<oa::Conv1d>(C, C, K, S, P);
	auto dec = oa::makeShared<oa::ConvTranspose1d>(C, C, K, S, P);
	auto& ctx = oa::ExecutionSession::getActive();
	auto X = oa::FnMatrix::randN(oa::MatrixShape{B, C, L});
	(void)testSubmitAndWait(ctx);
	oa::Vec<oa::Parameter*> params;
	auto ep = enc->allParameterPtrs();
	auto dp = dec->allParameterPtrs();
	for (auto* p : ep) params.pushBack(p);
	for (auto* p : dp) params.pushBack(p);
	auto opt = oa::makeUnique<oa::AdamW>(params, 0.01F);
	oa::F32 first = 0.0F, last = 0.0F;
	for (oa::I32 s = 1; s <= 60; ++s) {
		ctx.clear();
		oa::GradientTape tape; opt->zeroGrad();
		auto z   = enc->forward(X);
		auto y   = dec->forward(z);
		auto y2  = y.reshape(oa::MatrixShape{B, static_cast<oa::I64>(C) * L});
		auto x2  = X.reshape(oa::MatrixShape{B, static_cast<oa::I64>(C) * L});
		auto loss = oa::FnLoss::mse(y2, x2);
		tape.backward(loss);
		opt->step();
		(void)testSubmitAndWait(ctx);
		const float lv = loss.at(0);
		if (s == 1) first = lv;
		last = lv;
		if (s == 1 || s % 15 == 0) std::printf("  [convae2] step %2d | mse %.8f\n", s, static_cast<double>(lv));
	}
	EXPECT_LT(last, first) << "Conv1d→ConvTranspose1d stride-2 autoencoder must learn identity";
	ctx.clear();
}

TEST(Alm, DeepConvAutoEncoder) {
	oa::FnMatrix::setRngSeed(17);
	const oa::I32 inC = 3, W = 8, CodeC = 4, L = 16, B = 16;
	auto& ctx = oa::ExecutionSession::getActive();
	auto X = oa::FnMatrix::randN(oa::MatrixShape{B, inC, L});
	(void)testSubmitAndWait(ctx);
	oa::Vec<oa::Parameter*> params;
	auto collect = [&](const oa::SharedPtr<oa::Module>& m) {
		for (auto* p : m->allParameterPtrs()) params.pushBack(p);
	};
	auto encIn  = oa::makeShared<oa::Conv1d>(inC, W, 3, 1, 1);   collect(encIn);
	auto encDown = oa::makeShared<oa::Conv1d>(W, W, 4, 2, 1);     collect(encDown);
	auto encOut  = oa::makeShared<oa::Conv1d>(W, CodeC, 3, 1, 1); collect(encOut);
	auto decIn   = oa::makeShared<oa::Conv1d>(CodeC, W, 3, 1, 1); collect(decIn);
	auto decUp   = oa::makeShared<oa::ConvTranspose1d>(W, W, 4, 2, 1); collect(decUp);
	auto decOut  = oa::makeShared<oa::Conv1d>(W, inC, 3, 1, 1);   collect(decOut);
	auto opt = oa::makeUnique<oa::AdamW>(params, 0.01F);
	oa::F32 first = 0.0F, last = 0.0F;
	for (oa::I32 s = 1; s <= 60; ++s) {
		ctx.clear();
		oa::GradientTape tape; opt->zeroGrad();
		auto h = oa::FnMatrix::relu(encIn->forward(X));
		h = oa::FnMatrix::relu(encDown->forward(h));
		auto z = encOut->forward(h);
		auto h2 = oa::FnMatrix::relu(decIn->forward(z));
		h2 = oa::FnMatrix::relu(decUp->forward(h2));
		auto y = decOut->forward(h2);
		auto y2 = y.reshape(oa::MatrixShape{B, static_cast<oa::I64>(inC) * L});
		auto x2 = X.reshape(oa::MatrixShape{B, static_cast<oa::I64>(inC) * L});
		auto loss = oa::FnLoss::mse(y2, x2);
		tape.backward(loss);
		opt->step();
		(void)testSubmitAndWait(ctx);
		const float lv = loss.at(0);
		if (s == 1) first = lv;
		last = lv;
		if (s == 1 || s % 15 == 0) std::printf("  [deepae] step %2d | mse %.8f\n", s, static_cast<double>(lv));
	}
	EXPECT_LT(last, first) << "deep conv autoencoder must learn identity";
	ctx.clear();
}

TEST(Alm, DeepConvAutoEncoderNorm) {
	oa::FnMatrix::setRngSeed(17);
	const oa::I32 inC = 3, W = 8, CodeC = 4, L = 16, B = 16;
	auto& ctx = oa::ExecutionSession::getActive();
	auto X = oa::FnMatrix::randN(oa::MatrixShape{B, inC, L});
	(void)testSubmitAndWait(ctx);
	oa::Vec<oa::Parameter*> params;
	auto collect = [&](const oa::SharedPtr<oa::Module>& m) {
		for (auto* p : m->allParameterPtrs()) params.pushBack(p);
	};
	auto encIn  = oa::makeShared<oa::Conv1d>(inC, W, 3, 1, 1);   collect(encIn);
	auto encDown = oa::makeShared<oa::Conv1d>(W, W, 4, 2, 1);     collect(encDown);
	auto encOut  = oa::makeShared<oa::Conv1d>(W, CodeC, 3, 1, 1); collect(encOut);
	auto decIn   = oa::makeShared<oa::Conv1d>(CodeC, W, 3, 1, 1); collect(decIn);
	auto decUp   = oa::makeShared<oa::ConvTranspose1d>(W, W, 4, 2, 1); collect(decUp);
	auto decOut  = oa::makeShared<oa::Conv1d>(W, inC, 3, 1, 1);   collect(decOut);
	auto ln1 = oa::makeShared<oa::LayerNorm>(W); collect(ln1);
	auto ln2 = oa::makeShared<oa::LayerNorm>(W); collect(ln2);
	auto ln3 = oa::makeShared<oa::LayerNorm>(W); collect(ln3);
	auto ln4 = oa::makeShared<oa::LayerNorm>(W); collect(ln4);
	auto normC = [&](const oa::SharedPtr<oa::LayerNorm>& ln, const oa::Matrix& h) -> oa::Matrix {
		auto t = oa::FnMatrix::transpose(h, 1, 2);  // [B, T, C]
		auto n = ln->forward(t);
		return oa::FnMatrix::transpose(n, 1, 2);    // [B, C, T]
	};
	auto opt = oa::makeUnique<oa::AdamW>(params, 0.01F);
	oa::F32 first = 0.0F, last = 0.0F;
	for (oa::I32 s = 1; s <= 60; ++s) {
		ctx.clear();
		oa::GradientTape tape; opt->zeroGrad();
		auto h = oa::FnMatrix::relu(normC(ln1, encIn->forward(X)));
		h = oa::FnMatrix::relu(normC(ln2, encDown->forward(h)));
		auto z = encOut->forward(h);
		auto h2 = oa::FnMatrix::relu(normC(ln3, decIn->forward(z)));
		h2 = oa::FnMatrix::relu(normC(ln4, decUp->forward(h2)));
		auto y = decOut->forward(h2);
		auto y2 = y.reshape(oa::MatrixShape{B, static_cast<oa::I64>(inC) * L});
		auto x2 = X.reshape(oa::MatrixShape{B, static_cast<oa::I64>(inC) * L});
		auto loss = oa::FnLoss::mse(y2, x2);
		tape.backward(loss);
		opt->step();
		(void)testSubmitAndWait(ctx);
		const float lv = loss.at(0);
		if (s == 1) first = lv;
		last = lv;
		if (s == 1 || s % 15 == 0) std::printf("  [deepaen] step %2d | mse %.8f\n", s, static_cast<double>(lv));
	}
	EXPECT_LT(last, first) << "deep conv autoencoder with LayerNorm must learn identity";
	ctx.clear();
}
// inconsistency; if this also climbs, the loop/loss/optimizer harness is the bug.
TEST(Alm, LinearAeSanity) {
	oa::FnMatrix::setRngSeed(11);
	const oa::I32 D = 48, H = 32, B = 64;
	auto enc = oa::makeShared<oa::Linear>(D, H);
	auto dec = oa::makeShared<oa::Linear>(H, D);
	auto& ctx = oa::ExecutionSession::getActive();
	std::vector<float> xh(static_cast<size_t>(B) * D);
	{ oa::U64 r = 5; for (auto& v : xh) { r = (r * 6364136223846793005ULL) + 1; v = std::sin(0.01F * static_cast<float>(static_cast<oa::U32>(r >> 40))); } }
	auto X = oa::FnMatrix::fromBytes(oa::Span<const oa::U8>(reinterpret_cast<const oa::U8*>(xh.data()), xh.size() * sizeof(float)),
		oa::MatrixShape{B, D}, oa::ScalarType::Float32);
	oa::Vec<oa::Parameter*> params;
	for (auto* p : enc->allParameterPtrs()) params.pushBack(p);
	for (auto* p : dec->allParameterPtrs()) params.pushBack(p);
	auto opt = oa::makeUnique<oa::AdamW>(params, 1e-3F);
	oa::F32 first = 0.0F, last = 0.0F;
	for (oa::I32 s = 1; s <= 80; ++s) {
		ctx.clear();
		oa::GradientTape tape; opt->zeroGrad();
		auto rec = dec->forward(enc->forward(X));
		auto loss = oa::FnLoss::mse(rec, X);
		tape.backward(loss);
		opt->step();
		(void)testSubmitAndWait(ctx);
		const float lv = loss.at(0);
		if (s == 1) first = lv;
		last = lv;
		if (s == 1 || s % 20 == 0) std::printf("  [linAE] step %2d | mse %.8f\n", s, static_cast<double>(lv));
	}
	EXPECT_LT(last, first) << "linear AE harness must descend";
	ctx.clear();
}

// DECISIVE root-cause tool: is the COMPOSED end-to-end gradient a descent direction?
// Per-op bilinear checks pass, but the full encode→decode→MSE graph could compose a
// wrong gradient. Line-search: from a fixed init compute loss L0 + grads, then for a
// range of step sizes h evaluate L(W − h·grad) (forward only). If NO h gives L < L0,
// the composed gradient is NOT downhill → a composition/accumulation bug in the conv
// graph (transpose/reshape/rmsnorm/convT grad), NOT mere conditioning.
TEST(Alm, ComposedDescentCheck) {
	oa::FnMatrix::setRngSeed(7);
	oa::AlmTokenizerConfig cfg;
	cfg.inputDim = 48; cfg.width = 64; cfg.codeDim = 32; cfg.numCodes = 64;
	cfg.downT = 1; cfg.depth = 0;                       // minimal: no res blocks, factor 2
	auto tok = oa::makeShared<oa::AlmTokenizerAg>(cfg);
	auto& ctx = oa::ExecutionSession::getActive();
	const oa::I32 B = 8, Tw = 16;
	std::vector<float> xh(static_cast<size_t>(B) * Tw * cfg.inputDim);
	for (oa::I32 b = 0; b < B; ++b) for (oa::I32 t = 0; t < Tw; ++t) for (oa::I32 c = 0; c < cfg.inputDim; ++c)
		xh[(static_cast<size_t>(b) * Tw + t) * cfg.inputDim + c] = std::sin(0.2F * static_cast<float>(t) + 0.3F * static_cast<float>(c));
	auto X = oa::FnMatrix::fromBytes(oa::Span<const oa::U8>(reinterpret_cast<const oa::U8*>(xh.data()), xh.size() * sizeof(float)),
		oa::MatrixShape{B, Tw, cfg.inputDim}, oa::ScalarType::Float32);
	auto Xflat = X.reshape(oa::MatrixShape{static_cast<oa::I64>(B) * Tw, cfg.inputDim});
	const oa::I32 fac = tok->downsampleFactor();
	auto params = tok->allParameterPtrs();

	// L0 + grads (bypass VQ to isolate the pure conv AE).
	ctx.clear();
	oa::GradientTape tape;
	auto rec0 = tok->decode(tok->encode(X, B, Tw), B, Tw / fac);
	auto loss0 = oa::FnLoss::mse(rec0, Xflat);
	tape.backward(loss0);
	(void)testSubmitAndWait(ctx);
	const double L0 = static_cast<double>(loss0.at(0));
	std::printf("ComposedDescent: L0 = %.8f, params=%lld\n", L0, static_cast<long long>(params.size()));

	oa::Vec<oa::Matrix> W0, G;
	for (auto* p : params) { W0.pushBack(p->data.clone()); G.pushBack(p->data.gradMatrix().clone()); }
	(void)testSubmitAndWait(ctx);

	bool anyDown = false;
	const double hs[] = {1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8};
	for (double h : hs) {
		for (oa::I64 i = 0; i < params.size(); ++i) {
			auto Wn = oa::FnMatrix::add(W0[i], oa::FnMatrix::scale(G[i], static_cast<oa::F32>(-h)));
			params[i]->data.copyFrom(Wn);
		}
		(void)testSubmitAndWait(ctx);
		ctx.clear();
		auto rec = tok->decode(tok->encode(X, B, Tw), B, Tw / fac);
		auto l = oa::FnLoss::mse(rec, Xflat);
		(void)testSubmitAndWait(ctx);
		const double Lh = static_cast<double>(l.at(0));
		std::printf("  h=%.0e  L=%.8f  %s\n", h, Lh, Lh < L0 ? "DOWN ✓" : "up");
		if (Lh < L0) anyDown = true;
		for (oa::I64 i = 0; i < params.size(); ++i) params[i]->data.copyFrom(W0[i]);
		(void)testSubmitAndWait(ctx);
	}
	EXPECT_TRUE(anyDown) << "NO step size reduced loss → composed gradient is not a descent direction";
	ctx.clear();
}

// The NaN-fix / learns-to-reconstruct check (todo 2): a short synchronous training loop
// on a fixed smooth synthetic batch must drive recon MSE down while staying finite.
// Defaults to depth=0, downT=1 (minimal stable conv VQ-VAE) because the full T2M-GPT
// depth needs much smaller lr and longer tuning; env OA_MG_DEPTH / OA_MG_DOWNT override.
TEST(Alm, TokenizerLearnsRecon) {
	oa::FnMatrix::setRngSeed(7);
	auto envI0 = [](const char* n, oa::I32 d) { const char* e = std::getenv(n); return (e && *e) ? static_cast<oa::I32>(std::atoi(e)) : d; };
	oa::AlmTokenizerConfig cfg;
	cfg.inputDim = 48; cfg.width = 96; cfg.codeDim = 32; cfg.numCodes = 64;
	cfg.downT = envI0("OA_MG_DOWNT", 1); cfg.depth = envI0("OA_MG_DEPTH", 0); cfg.commitBeta = 0.25F;
	auto tok = oa::makeShared<oa::AlmTokenizerAg>(cfg);

	// Window length chosen so one batch seeds the codebook (B·Tw/Factor ≥ numCodes).
	const oa::I32 B  = 8;
	const oa::I32 Tw = 64;                                // 8·64/2 = 256 tokens >= numCodes
	auto& ctx = oa::ExecutionSession::getActive();

	// env knobs for fast diagnosis without rebuilds.
	auto envF = [](const char* n, oa::F32 d) { const char* e = std::getenv(n); return (e && *e) ? static_cast<oa::F32>(std::atof(e)) : d; };
	auto envI = [](const char* n, oa::I32 d) { const char* e = std::getenv(n); return (e && *e) ? static_cast<oa::I32>(std::atoi(e)) : d; };
	// Diverse synthetic DATASET (not one repeated batch): dsFrames of multi-frequency,
	// channel-varying content → a rich, learnable, non-degenerate manifold. Fresh random
	// mini-batch each step, matching the real tokenizer's window-sampling contract.
	// OA_MG_FRESH=0 reverts to the old single-fixed-batch repro.
	const bool  fresh = envI("OA_MG_FRESH", 0) != 0;
	const oa::I32 dsFrames = 2048;
	std::vector<float> ds(static_cast<size_t>(dsFrames) * cfg.inputDim);
	for (oa::I32 t = 0; t < dsFrames; ++t)
		for (oa::I32 c = 0; c < cfg.inputDim; ++c) {
			const float f1 = 0.04F + 0.011F * static_cast<float>(c % 7);
			const float f2 = 0.13F + 0.007F * static_cast<float>(c % 5);
			ds[(static_cast<size_t>(t) * cfg.inputDim) + c] =
				(0.6F * std::sin((f1 * static_cast<float>(t)) + (0.3F * static_cast<float>(c))))
				+ (0.4F * std::sin((f2 * static_cast<float>(t)) + (0.7F * static_cast<float>(c))));
		}
	oa::U64 rng = 0x1234ABCDULL;
	auto sample = [&](oa::Matrix& outX, oa::Matrix& outXflat) {
		std::vector<float> hb(static_cast<size_t>(B) * Tw * cfg.inputDim);
		for (oa::I32 b = 0; b < B; ++b) {
			rng = (rng * 6364136223846793005ULL) + 1442695040888963407ULL;
			const oa::I32 start = static_cast<oa::I32>((rng >> 33) % static_cast<oa::U64>(dsFrames - Tw));
			for (oa::I32 t = 0; t < Tw; ++t)
				for (oa::I32 c = 0; c < cfg.inputDim; ++c)
					hb[((static_cast<size_t>(b) * Tw + t) * cfg.inputDim) + c] =
						ds[(static_cast<size_t>(start + t) * cfg.inputDim) + c];
		}
		outX = oa::FnMatrix::fromBytes(oa::Span<const oa::U8>(reinterpret_cast<const oa::U8*>(hb.data()), hb.size() * sizeof(float)),
			oa::MatrixShape{B, Tw, cfg.inputDim}, oa::ScalarType::Float32);
		outXflat = outX.reshape(oa::MatrixShape{static_cast<oa::I64>(B) * Tw, cfg.inputDim});
	};
	oa::Matrix X, Xflat;
	sample(X, Xflat);

	// seed codebook from one warm batch of latents (B·Tw/Factor >= numCodes).
	{ auto z0 = tok->encode(X, B, Tw); tok->seed(z0); ctx.clear(); }

	const oa::F32 lr     = envF("OA_MG_LR", 1e-6F);
	const oa::I32 steps  = envI("OA_MG_STEPS", 300);
	const bool  bypass = envI("OA_MG_BYPASS", 0) != 0;

	auto params = tok->allParameterPtrs();
	std::printf("Optimizer params: %lld tensors\n", static_cast<long long>(params.size()));
	oa::UniquePtr<oa::Optimizer> opt = oa::makeUnique<oa::AdamW>(params, lr, 0.9F, 0.999F, 1e-8F, 0.01F);
	oa::F32 first = 0.0F;
	oa::F32 last  = 0.0F;
	for (oa::I32 s = 1; s <= steps; ++s) {
		ctx.clear();
		if (fresh) sample(X, Xflat);                         // fresh mini-batch each step
		oa::GradientTape tape; opt->zeroGrad();
		auto z   = tok->encode(X, B, Tw);
		oa::ResidualVqResult q;
		oa::Matrix zq;
		if (bypass) { zq = z; }
		else        { q = tok->quantize(z); zq = q.quantized; }
		auto rec = tok->decode(zq, B, Tw / tok->downsampleFactor());
		auto recon = oa::FnLoss::mse(rec, Xflat);
		auto loss  = bypass ? recon : (recon + q.commitLoss);
		tape.backward(loss);
		opt->step();
		if (!bypass) tok->emaUpdate(q);
		(void)testSubmitAndWait(ctx);
		const float lv = recon.at(0);
		if (s == 1) first = lv;
		last = lv;
		if (s == 1 || s % 10 == 0 || s == steps)
			std::printf("  [tok] step %3d | recon %.8f | commit %.8f\n", s, static_cast<double>(lv),
				static_cast<double>(bypass ? 0.0F : q.commitLoss.at(0)));
		ASSERT_TRUE(std::isfinite(lv)) << "diverged at step " << s;
	}
	std::printf("tokenizer recon: %.8f -> %.8f\n", static_cast<double>(first), static_cast<double>(last));
	EXPECT_LT(last, first) << "tokenizer did not learn to reconstruct";
	ctx.clear();
}

TEST(Alm, LmStub) {
	oa::AlmPriorConfig cfg;
	cfg.dModel = 256;
	cfg.numLayers = 2;  // Small for testing
	cfg.vocabSize = 515;
	
	auto lm = oa::makeShared<oa::AlmPriorAg>(cfg);
	std::printf("LM created (stub)\n");
	
	// TODO: Test forward/generate when implemented
	EXPECT_TRUE(lm != nullptr);
	EXPECT_EQ(lm->config().vocabSize, 515);
}

TEST(Alm, GenerateStub) {
	oa::AlmTokenizerConfig tokCfg;
	oa::AlmPriorConfig lmCfg;
	lmCfg.syncVocab(tokCfg.numCodes);
	
	auto tokenizer = oa::makeShared<oa::AlmTokenizerAg>(tokCfg);
	auto lm = oa::makeShared<oa::AlmPriorAg>(lmCfg);
	
	std::printf("tokenizer and LM created\n");
	
	// generate tokens unconditionally (starts from [SOM]).
	oa::Matrix generatedTokens = lm->generate(1, 1.0F, 0, 0.0F, 16);
	std::printf("generated tokens shape: [%lld, %lld]\n",
		static_cast<long long>(generatedTokens.size(0)), static_cast<long long>(generatedTokens.size(1)));
	
	// Decode to motion.
	oa::Matrix motion = lm->decodeToMotion(generatedTokens, *tokenizer);
	std::printf("Decoded motion shape: [%lld, %lld]\n",
		static_cast<long long>(motion.size(0)), static_cast<long long>(motion.size(1)));
	
	// Minimal USD export: treat motion as per-frame per-joint translations.
	// This is a placeholder skeleton; a real rig uses the canonical 272-dim pose.
	const oa::I32 frames = static_cast<oa::I32>(motion.size(0));
	const oa::I32 inputDim = static_cast<oa::I32>(motion.size(1));
	const oa::I32 joints = inputDim / 3;
	if (frames > 0 and joints > 0) {
		auto motionHost = hostFloatData(motion);
		const oa::F32* m = motionHost.data();
		oa::UsdSkelClip clip;
		clip.frameCount = frames;
		clip.fps = 30.0F;
		clip.upAxis = 2;
		clip.jointPaths.reserve(joints);
		clip.bindTransforms.reserve(joints);
		clip.restTransforms.reserve(joints);
		oa::String path = "root";
		for (oa::I32 j = 0; j < joints; ++j) {
			clip.jointPaths.pushBack(path);
			clip.bindTransforms.pushBack(oa::vlm::Mat4::identity());
			clip.restTransforms.pushBack(oa::vlm::Mat4::identity());
			char buf[32];
			std::snprintf(buf, sizeof(buf), "/j%d", j);
			path = path + buf;
		}
		clip.translations.reserve(static_cast<oa::I64>(frames) * joints);
		clip.rotations.reserve(static_cast<oa::I64>(frames) * joints);
		for (oa::I32 f = 0; f < frames; ++f) {
			for (oa::I32 j = 0; j < joints; ++j) {
				const oa::I64 base = static_cast<oa::I64>(f) * inputDim + static_cast<oa::I64>(j * 3);
				clip.translations.pushBack({.x = m[base], .y = m[base + 1], .z = m[base + 2]});
				clip.rotations.pushBack({.x = 0.0F, .y = 0.0F, .z = 0.0F, .w = 1.0F});
			}
		}
		oa::Path usdPath("var/alm/Alm_GenerateStub.usda");
		(void)oa::Filesystem::createDirectories(usdPath.parentPath());
		auto st = oa::Usd::writeUsda(usdPath, clip, "rig");
		std::printf("USD export: %s\n", st.isOk() ? "ok" : st.toString().cStr());
		EXPECT_TRUE(st.isOk());
	}
	
	EXPECT_TRUE(lm != nullptr);
	EXPECT_TRUE(tokenizer != nullptr);
}

TEST(Alm, LmLearnsNextToken) {
	oa::FnMatrix::setRngSeed(11);
	auto& ctx = oa::ExecutionSession::getActive();

	// stage 1: minimal stable tokenizer.
	oa::AlmTokenizerConfig tokCfg;
	tokCfg.inputDim = 48; tokCfg.width = 96; tokCfg.codeDim = 32; tokCfg.numCodes = 64;
	tokCfg.downT = 1; tokCfg.depth = 0; tokCfg.commitBeta = 0.25F;
	auto tok = oa::makeShared<oa::AlmTokenizerAg>(tokCfg);

	oa::AlmPriorConfig lmCfg;
	lmCfg.syncVocab(tokCfg.numCodes);

	const oa::I32 B  = 4;   // batched sequences; block-diagonal mask keeps each sequence causal
	const oa::I32 Tw = 128; // B*Tw/Factor must be >= numCodes for VQ seed
	const oa::I32 tokLen = Tw / tok->downsampleFactor();   // 64
	std::vector<float> ds(static_cast<size_t>(B * Tw * tokCfg.inputDim));
	auto sampleDs = [&]() {
		for (oa::I32 b = 0; b < B; ++b) {
			for (oa::I32 t = 0; t < Tw; ++t) {
				const float tt = static_cast<float>(t) / static_cast<float>(Tw);
				for (oa::I32 c = 0; c < tokCfg.inputDim; ++c) {
					const float freq = 1.0F + static_cast<float>(c % 8) * 0.5F;
					const float phase = static_cast<float>(b) * 0.3F + static_cast<float>(c) * 0.1F;
					const float start = static_cast<float>(b) * 0.7F + static_cast<float>(c % 3) * 0.2F;
					ds[((static_cast<size_t>(b) * Tw + t) * tokCfg.inputDim) + c] =
						start + std::sin(6.2831853F * (freq * tt + phase)) * 0.5F;
				}
			}
		}
	};
	sampleDs();
	oa::Matrix X = oa::FnMatrix::fromBytes(
		oa::Span<const oa::U8>(reinterpret_cast<const oa::U8*>(ds.data()), ds.size() * sizeof(float)),
		oa::MatrixShape{B, Tw, tokCfg.inputDim}, oa::ScalarType::Float32);
	{ auto z0 = tok->encode(X, B, Tw); tok->seed(z0); ctx.clear(); }

	auto tokParams = tok->allParameterPtrs();
	auto tokOpt = oa::makeUnique<oa::AdamW>(tokParams, 1e-6F, 0.9F, 0.999F, 1e-8F, 0.01F);
	for (oa::I32 s = 1; s <= 100; ++s) {
		ctx.clear();
		oa::GradientTape tape; tokOpt->zeroGrad();
		auto z = tok->encode(X, B, Tw);
		auto q = tok->quantize(z);
		auto rec = tok->decode(q.quantized, B, tokLen);
		auto recon = oa::FnLoss::l1(rec, X.reshape(oa::MatrixShape{static_cast<oa::I64>(B) * Tw, tokCfg.inputDim}));
		auto loss = recon + q.commitLoss;
		tape.backward(loss);
		tokOpt->step();
		tok->emaUpdate(q);
		(void)testSubmitAndWait(ctx);
	}
	ctx.clear();

	// tokenize the fixed dataset: [B, tokLen].
	oa::Matrix tokenIds = tok->tokenize(X, B, Tw)[0].reshape(oa::MatrixShape{B, tokLen});
	(void)testSubmitAndWait(ctx);
	const oa::I32* ids = tokenIds.dataAs<const oa::I32>();

	// Build LM inputs [SOM, c0, ..., cN] and targets [c0, ..., cN, EOM].
	std::vector<oa::I32> inputHost(static_cast<size_t>(B) * (tokLen + 1));
	std::vector<oa::I32> targetHost(static_cast<size_t>(B) * (tokLen + 1));
	for (oa::I32 b = 0; b < B; ++b) {
		const size_t inRow = static_cast<size_t>(b) * (tokLen + 1);
		const size_t outRow = static_cast<size_t>(b) * tokLen;
		inputHost[inRow] = lmCfg.somToken;
		for (oa::I32 t = 0; t < tokLen; ++t) {
			oa::I32 code = ids[outRow + t];
			inputHost[inRow + 1 + t] = code;
			targetHost[inRow + t] = code;
		}
		targetHost[inRow + tokLen] = lmCfg.eomToken;
	}
	oa::Matrix inputIds = oa::FnMatrix::fromBytes(
		oa::Span<const oa::U8>(reinterpret_cast<const oa::U8*>(inputHost.data()), inputHost.size() * sizeof(oa::U32)),
		oa::MatrixShape{B, tokLen + 1}, oa::ScalarType::UInt32);
	oa::Matrix targetIds = oa::FnMatrix::fromBytes(
		oa::Span<const oa::U8>(reinterpret_cast<const oa::U8*>(targetHost.data()), targetHost.size() * sizeof(oa::U32)),
		oa::MatrixShape{B, tokLen + 1}, oa::ScalarType::UInt32);

	// stage 2: small AR transformer.
	lmCfg.dModel = 128; lmCfg.numLayers = 2; lmCfg.dFfn = 256;
	auto lm = oa::makeShared<oa::AlmPriorAg>(lmCfg);
	(void)testSubmitAndWait(ctx);  // flush LM initialization
	auto lmParams = lm->allParameterPtrs();
	auto lmOpt = oa::makeUnique<oa::AdamW>(lmParams, 1e-3F, 0.9F, 0.999F, 1e-8F, 0.01F);

	oa::F32 firstLoss = 0.0F;
	oa::F32 lastLoss = 0.0F;
	for (oa::I32 s = 1; s <= 200; ++s) {
		ctx.clear();
		oa::GradientTape tape; lmOpt->zeroGrad();
		auto logits = lm->forward(inputIds);                       // [B, tokLen+1, Vocab]
		auto logitsFlat = logits.reshape(oa::MatrixShape{static_cast<oa::I64>(B) * (tokLen + 1), lmCfg.vocabSize});
		auto targetFlat = targetIds.reshape(oa::MatrixShape{static_cast<oa::I64>(B) * (tokLen + 1)});
		auto ce = oa::FnLoss::crossEntropy(logitsFlat, targetFlat);
		tape.backward(ce);
		lmOpt->step();
		(void)testSubmitAndWait(ctx);
		const float lv = ce.at(0);
		if (s == 1) firstLoss = lv;
		lastLoss = lv;
		if (s == 1 || s % 50 == 0 || s == 200)
			std::printf("  [lm] step %3d | ce %.8f\n", s, static_cast<double>(lv));
		ASSERT_TRUE(std::isfinite(lv)) << "LM diverged at step " << s;
	}
	std::printf("LM cross-entropy: %.8f -> %.8f\n", static_cast<double>(firstLoss), static_cast<double>(lastLoss));
	EXPECT_LT(lastLoss, firstLoss) << "LM did not learn next-token prediction";
	lmOpt.reset();
	lm.reset();
	tokOpt.reset();
	tok.reset();
	ctx.clear();
}

// exact trainalm LM shape on the iGPU configuration. This is intentionally a
// short fixed-batch regression: it verifies that B=64, T=65 and the full
// 3-layer D=192 graph produce gradients and update the model, while reporting
// unambiguous wall latency and token throughput.
TEST(Alm, LmProductionShapeUpdates) {
	oa::FnMatrix::setRngSeed(42);
	auto& ctx = oa::ExecutionSession::getActive();
	constexpr oa::I32 B = 64;
	constexpr oa::I32 TokenLen = 64;
	constexpr oa::I32 T = TokenLen + 1;
	constexpr oa::I32 steps = 4;

	oa::AlmPriorConfig cfg;
	cfg.syncVocab(512);
	cfg.dModel = 192;
	cfg.numHeads = 6;
	cfg.numLayers = 3;
	cfg.dFfn = 512;
	cfg.seqLen = T;

	std::vector<oa::I32> input(static_cast<size_t>(B) * T);
	std::vector<oa::I32> target(static_cast<size_t>(B) * T);
	for (oa::I32 b = 0; b < B; ++b) {
		const size_t row = static_cast<size_t>(b) * T;
		input[row] = cfg.somToken;
		for (oa::I32 t = 0; t < TokenLen; ++t) {
			const oa::I32 code = (b * 7 + t * 3) % cfg.numCodes;
			input[row + 1 + static_cast<size_t>(t)] = code;
			target[row + static_cast<size_t>(t)] = code;
		}
		target[row + TokenLen] = cfg.eomToken;
	}
	auto inputIds = oa::FnMatrix::fromBytes(
		oa::Span<const oa::U8>(reinterpret_cast<const oa::U8*>(input.data()), input.size() * sizeof(oa::I32)),
		oa::MatrixShape{B, T}, oa::ScalarType::Int32);
	auto targetIds = oa::FnMatrix::fromBytes(
		oa::Span<const oa::U8>(reinterpret_cast<const oa::U8*>(target.data()), target.size() * sizeof(oa::I32)),
		oa::MatrixShape{B, T}, oa::ScalarType::Int32);

	auto lm = oa::makeShared<oa::AlmPriorAg>(cfg);
	(void)testSubmitAndWait(ctx);
	auto params = lm->allParameterPtrs();
	oa::AdamW opt(params, 1e-4F, 0.9F, 0.99F, 1e-8F, 0.01F);

	oa::F32 first = 0.0F;
	oa::F32 last = 0.0F;
	const auto begin = std::chrono::steady_clock::now();
	for (oa::I32 step = 1; step <= steps; ++step) {
		ctx.clear();
		opt.zeroGrad();
		oa::GradientTape tape;
		auto logits = lm->forward(inputIds);
		auto ce = oa::FnLoss::crossEntropy(
			logits.reshape(oa::MatrixShape{static_cast<oa::I64>(B) * T, cfg.vocabSize}),
			targetIds.reshape(oa::MatrixShape{static_cast<oa::I64>(B) * T}));
		tape.backward(ce);
		opt.step();
		ASSERT_TRUE(testSubmitAndWait(ctx).isOk());
		last = ce.at(0);
		if (step == 1) first = last;
		std::printf("  [production-shape] step %d/%d | ce %.8f\n",
			step, steps, static_cast<double>(last));
		ASSERT_TRUE(std::isfinite(last));
	}
	const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
	const double msPerStep = seconds * 1000.0 / steps;
	const double seqPerSec = static_cast<double>(B) * steps / seconds;
	const double tokensPerSec = static_cast<double>(B) * T * steps / seconds;
	std::printf("Production shape: CE %.8f -> %.8f | %.2f ms/step | %.2f seq/s | %.2f tks\n",
		static_cast<double>(first), static_cast<double>(last), msPerStep, seqPerSec, tokensPerSec);
	EXPECT_LT(last, first) << "production-shape OaAlm did not update";
	ctx.clear();
}

// A causal Transformer's logits at position p must not depend on later tokens.
// This also verifies that one block can safely change runtime sequence length:
// every prefix rebuilds the mask while reusing exactly the same model weights.
TEST(Alm, LmDynamicPrefixMatchesFullForward) {
	oa::FnMatrix::setRngSeed(23);
	auto& ctx = oa::ExecutionSession::getActive();

	oa::AlmPriorConfig lmCfg;
	lmCfg.syncVocab(16);
	lmCfg.dModel = 64; lmCfg.numLayers = 2; lmCfg.dFfn = 128;
	lmCfg.seqLen = 13;
	lmCfg.maxGenLen = 20;

	auto lm = oa::makeShared<oa::AlmPriorAg>(lmCfg);
	(void)testSubmitAndWait(ctx);

	constexpr oa::I32 B = 3;
	constexpr oa::I32 T = 13;
	std::vector<oa::U32> fullIds(static_cast<size_t>(B) * T);
	for (oa::I32 b = 0; b < B; ++b) {
		for (oa::I32 t = 0; t < T; ++t) {
			fullIds[static_cast<size_t>(b) * T + t] = static_cast<oa::U32>((b * 5 + t * 3) % lmCfg.numCodes);
		}
	}
	auto fullInput = oa::FnMatrix::fromBytes(
		oa::Span<const oa::U8>(reinterpret_cast<const oa::U8*>(fullIds.data()), fullIds.size() * sizeof(oa::U32)),
		oa::MatrixShape{B, T}, oa::ScalarType::UInt32);
	auto fullLogits = hostFloatData(lm->forward(fullInput));

	for (const oa::I32 prefixLen : {1, 2, 7, T}) {
		ctx.clear();
		std::vector<oa::U32> prefix(static_cast<size_t>(B) * prefixLen);
		for (oa::I32 b = 0; b < B; ++b) {
			for (oa::I32 t = 0; t < prefixLen; ++t) {
				prefix[static_cast<size_t>(b) * prefixLen + t] = fullIds[static_cast<size_t>(b) * T + t];
			}
		}
		auto prefixInput = oa::FnMatrix::fromBytes(
			oa::Span<const oa::U8>(reinterpret_cast<const oa::U8*>(prefix.data()), prefix.size() * sizeof(oa::U32)),
			oa::MatrixShape{B, prefixLen}, oa::ScalarType::UInt32);
		auto prefixLogits = hostFloatData(lm->forward(prefixInput));

		oa::F32 maxError = 0.0F;
		for (oa::I32 b = 0; b < B; ++b) {
			for (oa::I32 v = 0; v < lmCfg.vocabSize; ++v) {
				const oa::I64 fullIdx = (static_cast<oa::I64>(b) * T + prefixLen - 1) * lmCfg.vocabSize + v;
				const oa::I64 prefixIdx = (static_cast<oa::I64>(b) * prefixLen + prefixLen - 1) * lmCfg.vocabSize + v;
				maxError = std::max(maxError, std::abs(fullLogits[fullIdx] - prefixLogits[prefixIdx]));
			}
		}
		std::printf("  [dynamic-prefix] T=%d max error %.8g\n", prefixLen, static_cast<double>(maxError));
		EXPECT_LT(maxError, 1e-4F);
	}

	lm.reset();
	ctx.clear();
}

// Frozen semantic features are a real part of the ALM graph: they become one
// learned causal prefix token, alter motion logits, and train the projection.
TEST(Alm, LmFrozenTextPrefixConditionsAndBackpropagates) {
	oa::FnMatrix::setRngSeed(27);
	auto& ctx = oa::ExecutionSession::getActive();

	oa::AlmPriorConfig cfg;
	cfg.syncVocab(16);
	cfg.dModel = 32; cfg.numLayers = 1; cfg.dFfn = 64;
	cfg.textFeatureDim = 4;
	cfg.seqLen = 6;       // one text prefix + five motion-token positions
	cfg.maxSeqLen = 8;
	auto lm = oa::makeShared<oa::AlmPriorAg>(cfg);

	const std::vector<oa::I32> ids = {
		cfg.somToken, 1, 2, 3, 4,
		cfg.somToken, 1, 2, 3, 4};
	const std::vector<oa::I32> targets = {
		1, 2, 3, 4, cfg.eomToken,
		1, 2, 3, 4, cfg.eomToken};
	const std::vector<float> textA = {
		1.0F, 0.0F, 0.0F, 0.0F,
		1.0F, 0.0F, 0.0F, 0.0F};
	const std::vector<float> textB = {
		0.0F, 1.0F, 0.5F, -0.5F,
		0.0F, 1.0F, 0.5F, -0.5F};
	auto input = oa::FnMatrix::fromBytes(
		oa::Span<const oa::U8>(reinterpret_cast<const oa::U8*>(ids.data()), ids.size() * sizeof(oa::I32)),
		oa::MatrixShape{2, 5}, oa::ScalarType::Int32);
	auto target = oa::FnMatrix::fromBytes(
		oa::Span<const oa::U8>(reinterpret_cast<const oa::U8*>(targets.data()), targets.size() * sizeof(oa::I32)),
		oa::MatrixShape{10}, oa::ScalarType::Int32);
	auto featureA = oa::FnMatrix::fromBytes(
		oa::Span<const oa::U8>(reinterpret_cast<const oa::U8*>(textA.data()), textA.size() * sizeof(float)),
		oa::MatrixShape{2, 4}, oa::ScalarType::Float32);
	auto featureB = oa::FnMatrix::fromBytes(
		oa::Span<const oa::U8>(reinterpret_cast<const oa::U8*>(textB.data()), textB.size() * sizeof(float)),
		oa::MatrixShape{2, 4}, oa::ScalarType::Float32);

	auto logitsA = hostFloatData(lm->forwardConditioned(input, featureA));
	ctx.clear();
	auto logitsBMatrix = lm->forwardConditioned(input, featureB);
	auto logitsB = hostFloatData(logitsBMatrix);
	ASSERT_EQ(logitsA.size(), logitsB.size());
	oa::F32 maxPromptDelta = 0.0F;
	for (oa::Usize i = 0; i < logitsA.size(); ++i) {
		maxPromptDelta = std::max(maxPromptDelta, std::abs(logitsA[i] - logitsB[i]));
	}
	EXPECT_GT(maxPromptDelta, 1e-6F) << "different frozen text features did not affect motion logits";
	ctx.clear();
	auto generated = lm->generateConditioned(featureA, 1.0F, 1, 0.0F, 7);
	EXPECT_EQ(generated.size(0), 2);
	EXPECT_LE(generated.size(1), 8);  // [MOTION_BOS] plus at most seven samples

	ctx.clear();
	oa::GradientTape tape;
	auto trainLogits = lm->forwardConditioned(input, featureB).reshape(
		oa::MatrixShape{10, cfg.vocabSize});
	auto loss = oa::FnLoss::crossEntropy(trainLogits, target);
	tape.backward(loss);
	ASSERT_TRUE(testSubmitAndWait(ctx).isOk());

	oa::Parameter* projectionWeight = nullptr;
	for (auto named : lm->allNamedParameterPtrs()) {
		if (named.path == "text_projection.weight") projectionWeight = named.param;
	}
	ASSERT_NE(projectionWeight, nullptr);
	ASSERT_FALSE(projectionWeight->grad().isEmpty());
	double gradL1 = 0.0;
	for (const auto value : hostFloatData(projectionWeight->grad())) gradL1 += std::abs(value);
	EXPECT_GT(gradL1, 1e-8) << "motion loss did not train the text projection";
	const oa::String checkpointPath = "/tmp/alm_conditioned_contract.oam";
	ASSERT_TRUE(lm->save(ctx.engine(), checkpointPath).isOk());
	auto checkpoint = oa::ModelFile::load(checkpointPath);
	ASSERT_TRUE(checkpoint.isOk());
	const auto& saved = checkpoint.getValue();
	const auto* savedProjection = saved.findWeight("text_projection.weight");
	ASSERT_NE(savedProjection, nullptr);
	EXPECT_EQ(savedProjection->rank, 2);
	EXPECT_EQ(savedProjection->shape[0], static_cast<oa::U64>(cfg.dModel));
	EXPECT_EQ(savedProjection->shape[1], static_cast<oa::U64>(cfg.textFeatureDim));
	std::remove(checkpointPath.cStr());
	std::printf("  [text-prefix] max logit delta %.8g · projection grad L1 %.8g\n",
		static_cast<double>(maxPromptDelta), gradL1);
	ctx.clear();
}

TEST(Alm, LmFfnPoliciesForward) {
	oa::FnMatrix::setRngSeed(29);
	auto& ctx = oa::ExecutionSession::getActive();
	constexpr oa::I32 B = 2;
	constexpr oa::I32 T = 5;
	std::vector<oa::U32> ids(static_cast<size_t>(B) * T);
	for (oa::I32 i = 0; i < B * T; ++i) ids[static_cast<size_t>(i)] = static_cast<oa::U32>(i % 16);
	auto input = oa::FnMatrix::fromBytes(
		oa::Span<const oa::U8>(reinterpret_cast<const oa::U8*>(ids.data()), ids.size() * sizeof(oa::U32)),
		oa::MatrixShape{B, T}, oa::ScalarType::UInt32);

	for (const oa::AlmFfnType policy : {oa::AlmFfnType::Dense, oa::AlmFfnType::Moe, oa::AlmFfnType::Hybrid}) {
		oa::AlmPriorConfig cfg;
		cfg.syncVocab(16);
		cfg.dModel = 32; cfg.numLayers = 2; cfg.dFfn = 32; cfg.seqLen = T;
		cfg.ffnType = policy; cfg.moeNumExperts = 2; cfg.moeExpertsPerToken = 1; cfg.moeEvery = 2;
		auto lm = oa::makeShared<oa::AlmPriorAg>(cfg);
		auto logits = lm->forward(input);
		EXPECT_EQ(logits.size(0), B);
		EXPECT_EQ(logits.size(1), T);
		EXPECT_EQ(logits.size(2), cfg.vocabSize);
		EXPECT_TRUE(hostAllFinite(logits));
		ctx.clear();
	}
}

TEST(Alm, LmCheckpointRoundtrip) {
	oa::FnMatrix::setRngSeed(31);
	auto& ctx = oa::ExecutionSession::getActive();
	oa::AlmPriorConfig cfg;
	cfg.syncVocab(16);
	cfg.dModel = 32; cfg.numLayers = 2; cfg.dFfn = 64; cfg.seqLen = 6;

	std::vector<oa::I32> ids = {cfg.somToken, 1, 2, 3, 4, 5, cfg.somToken, 5, 4, 3, 2, 1};
	std::vector<oa::I32> targets = {1, 2, 3, 4, 5, cfg.eomToken, 5, 4, 3, 2, 1, cfg.eomToken};
	auto input = oa::FnMatrix::fromBytes(
		oa::Span<const oa::U8>(reinterpret_cast<const oa::U8*>(ids.data()), ids.size() * sizeof(oa::I32)),
		oa::MatrixShape{2, 6}, oa::ScalarType::Int32);
	auto target = oa::FnMatrix::fromBytes(
		oa::Span<const oa::U8>(reinterpret_cast<const oa::U8*>(targets.data()), targets.size() * sizeof(oa::I32)),
		oa::MatrixShape{12}, oa::ScalarType::Int32);

	auto original = oa::makeShared<oa::AlmPriorAg>(cfg);
	auto originalParams = original->allParameterPtrs();
	auto originalOpt = oa::makeUnique<oa::AdamW>(originalParams, 1e-3F);
	{
		oa::GradientTape tape;
		originalOpt->zeroGrad();
		auto logits = original->forward(input).reshape(oa::MatrixShape{12, cfg.vocabSize});
		auto ce = oa::FnLoss::crossEntropy(logits, target);
		tape.backward(ce);
		originalOpt->step();
		(void)testSubmitAndWait(ctx);
	}
	ctx.clear();
	auto before = hostFloatData(original->forward(input));
	const oa::String path = "/tmp/alm_transformer_roundtrip.oam";
	ASSERT_TRUE(original->save(ctx.engine(), path, *originalOpt).isOk());

	auto reloaded = oa::makeShared<oa::AlmPriorAg>(cfg);
	auto reloadedParams = reloaded->allParameterPtrs();
	auto reloadedOpt = oa::makeUnique<oa::AdamW>(reloadedParams, 1e-3F);
	ASSERT_TRUE(reloaded->load(ctx.engine(), path, *reloadedOpt).isOk());
	(void)testSubmitAndWait(ctx);
	ctx.clear();
	auto after = hostFloatData(reloaded->forward(input));
	ASSERT_EQ(before.size(), after.size());
	oa::F32 maxError = 0.0F;
	for (oa::Usize i = 0; i < before.size(); ++i) maxError = std::max(maxError, std::abs(before[i] - after[i]));
	std::printf("LM checkpoint round-trip max logit error %.8g\n", static_cast<double>(maxError));
	EXPECT_EQ(maxError, 0.0F);
	EXPECT_EQ(reloadedOpt->getStep(), originalOpt->getStep());
	std::remove(path.cStr());
	reloadedOpt.reset(); reloaded.reset(); originalOpt.reset(); original.reset();
	ctx.clear();
}

TEST(Alm, BundleRoundtrip) {
	oa::FnMatrix::setRngSeed(37);
	auto& ctx = oa::ExecutionSession::getActive();
	oa::AlmAgConfig cfg;
	cfg.tokenizer.inputDim = 6;
	cfg.tokenizer.width = 8;
	cfg.tokenizer.codeDim = 8;
	cfg.tokenizer.numCodes = 8;
	cfg.tokenizer.downT = 1;
	cfg.tokenizer.depth = 1;
	cfg.prior.syncVocab(cfg.tokenizer.numCodes);
	cfg.prior.dModel = 8;
	cfg.prior.numHeads = 2;
	cfg.prior.numLayers = 1;
	cfg.prior.dFfn = 16;
	cfg.prior.textFeatureDim = 4;
	cfg.prior.seqLen = 4;
	cfg.prior.maxSeqLen = 8;
	cfg.prior.maxGenLen = 7;
	cfg.textEncoder = "oa/test-clip";

	auto original = oa::makeShared<oa::AlmAg>(cfg);
	const std::vector<oa::I32> ids = {8, 1, 2, 3};
	const std::vector<oa::F32> text = {0.25F, -0.5F, 0.75F, 1.0F};
	auto input = oa::FnMatrix::fromBytes(
		oa::Span<const oa::U8>(reinterpret_cast<const oa::U8*>(ids.data()), ids.size() * sizeof(oa::I32)),
		oa::MatrixShape{1, 4}, oa::ScalarType::Int32);
	auto feature = oa::FnMatrix::fromBytes(
		oa::Span<const oa::U8>(reinterpret_cast<const oa::U8*>(text.data()), text.size() * sizeof(oa::F32)),
		oa::MatrixShape{1, 4}, oa::ScalarType::Float32);
	auto before = hostFloatData(original->forwardConditioned(input, feature));

	const oa::String path = "/tmp/alm_ag_bundle_roundtrip.oam";
	ASSERT_TRUE(original->saveBundle(ctx.engine(), path).isOk());
	auto raw = oa::ModelFile::load(path);
	ASSERT_TRUE(raw.isOk());
	EXPECT_STREQ(raw.getValue().config.architecture, "OaAlmAg");
	EXPECT_NE(raw.getValue().findWeight("tokenizer.enc_in.weight"), nullptr);
	EXPECT_NE(raw.getValue().findWeight("prior.text_projection.weight"), nullptr);

	auto loaded = oa::AlmAg::loadBundle(ctx.engine(), path);
	ASSERT_TRUE(loaded.isOk()) << loaded.getStatus().getMessage().cStr();
	auto reloaded = std::move(loaded).getValue();
	EXPECT_EQ(reloaded->config().tokenizer.numCodes, cfg.tokenizer.numCodes);
	EXPECT_EQ(reloaded->config().prior.textFeatureDim, cfg.prior.textFeatureDim);
	EXPECT_EQ(reloaded->config().prior.numHeads, cfg.prior.numHeads);
	EXPECT_EQ(reloaded->config().textEncoder, cfg.textEncoder);
	auto after = hostFloatData(reloaded->forwardConditioned(input, feature));
	ASSERT_EQ(before.size(), after.size());
	for (oa::Usize i = 0; i < before.size(); ++i) EXPECT_EQ(before[i], after[i]);
	std::remove(path.cStr());
	ctx.clear();
}

// ─── throughput benchmarks (samples/sec) at the var/config/Alm.yaml scale ───
// run fp32, then repeat with --bf16, and compare the printed ms/step + sps.
// These do NOT assert correctness — they measure steady-state training throughput.
// The startup log line "precision=FP32|BF16" identifies which run is which.

// These benches deliberately run the REFERENCE full-scale model at batch 128 — the whole
// point is the real throughput number. On an integrated GPU (shared system RAM) the
// full-scale model + autograd tape exhausts host memory and OOMs the box, and a throughput
// figure from hardware that can't hold the model is meaningless anyway. Skip there.
static bool oaBenchNeedsDiscreteGpu() {
	const oa::DeviceType dt = testEngine().deviceType();
	return dt == oa::DeviceType::VkIntegrated || dt == oa::DeviceType::VkCpu
	    || dt == oa::DeviceType::Host;
}

TEST(Alm, LmTrainBench) {
	if (oaBenchNeedsDiscreteGpu())
		GTEST_SKIP() << "full-scale throughput bench needs a discrete GPU (shared-RAM iGPU OOMs)";
	oa::FnMatrix::setRngSeed(42);
	auto& ctx = oa::ExecutionSession::getActive();

	oa::AlmPriorConfig lmCfg;
	lmCfg.syncVocab(512);                       // matches yaml num_codes
	lmCfg.dModel = 384; lmCfg.numLayers = 6; lmCfg.dFfn = 1536;

	const oa::I32 B = 128, T = 64;                // batch, lm_seq_len (yaml)
	std::vector<oa::U32> inHost(static_cast<size_t>(B) * (T + 1));
	std::vector<oa::U32> tgtHost(static_cast<size_t>(B) * (T + 1));
	for (oa::I32 b = 0; b < B; ++b) {
		const size_t row = static_cast<size_t>(b) * (T + 1);
		inHost[row] = static_cast<oa::U32>(lmCfg.somToken);
		for (oa::I32 t = 0; t < T; ++t) {
			const oa::U32 c = static_cast<oa::U32>((b * 7u + t * 3u) % 512u);
			inHost[row + 1 + t] = c;
			tgtHost[row + t] = c;
		}
		tgtHost[row + T] = static_cast<oa::U32>(lmCfg.eomToken);
	}
	oa::Matrix inputIds = oa::FnMatrix::fromBytes(
		oa::Span<const oa::U8>(reinterpret_cast<const oa::U8*>(inHost.data()), inHost.size() * sizeof(oa::U32)),
		oa::MatrixShape{B, T + 1}, oa::ScalarType::UInt32);
	oa::Matrix targetIds = oa::FnMatrix::fromBytes(
		oa::Span<const oa::U8>(reinterpret_cast<const oa::U8*>(tgtHost.data()), tgtHost.size() * sizeof(oa::U32)),
		oa::MatrixShape{B, T + 1}, oa::ScalarType::UInt32);

	auto lm = oa::makeShared<oa::AlmPriorAg>(lmCfg);
	(void)testSubmitAndWait(ctx);
	auto lmParams = lm->allParameterPtrs();
	auto lmOpt = oa::makeUnique<oa::AdamW>(lmParams, 1e-4F, 0.9F, 0.999F, 1e-8F, 0.01F);

	const int warmup = 3, timed = 10;
	auto stepOnce = [&]() {
		ctx.clear();
		oa::GradientTape tape; lmOpt->zeroGrad();
		auto logits = lm->forward(inputIds);
		auto lf = logits.reshape(oa::MatrixShape{static_cast<oa::I64>(B) * (T + 1), lmCfg.vocabSize});
		auto tf = targetIds.reshape(oa::MatrixShape{static_cast<oa::I64>(B) * (T + 1)});
		auto ce = oa::FnLoss::crossEntropy(lf, tf);
		tape.backward(ce);
		lmOpt->step();
		(void)testSubmitAndWait(ctx);
	};
	for (int i = 0; i < warmup; ++i) stepOnce();
	auto t0 = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < timed; ++i) stepOnce();
	auto t1 = std::chrono::high_resolution_clock::now();
	const double dt = std::chrono::duration<double>(t1 - t0).count();
	const double msStep = dt / timed * 1000.0;
	const double sps = static_cast<double>(B) * timed / dt;
	const char* prec = oa::FnMatrix::weightDtype() == oa::ScalarType::Float32 ? "fp32" : "bf16";
	std::printf("  [lm bench %s] 6L D=384 B=%d T=%d | %.2f ms/step | %.1f seq/s (%.1fK tok/s)\n",
		prec, B, T, msStep, sps, sps * T / 1000.0);
	lmOpt.reset(); lm.reset(); ctx.clear();
}

TEST(Alm, TokTrainBench) {
	if (oaBenchNeedsDiscreteGpu())
		GTEST_SKIP() << "full-scale throughput bench needs a discrete GPU (shared-RAM iGPU OOMs)";
	oa::FnMatrix::setRngSeed(42);
	auto& ctx = oa::ExecutionSession::getActive();

	oa::AlmTokenizerConfig tokCfg;
	tokCfg.inputDim = 263; tokCfg.width = 512; tokCfg.codeDim = 512; tokCfg.numCodes = 512;
	tokCfg.downT = 2; tokCfg.depth = 3;
	tokCfg.commitBeta = 0.25F; tokCfg.emaDecay = 0.99F; tokCfg.deadThresh = 2.0F;
	auto tok = oa::makeShared<oa::AlmTokenizerAg>(tokCfg);

	const oa::I32 B = 128, T = 64;                // batch, seq_len (yaml)
	const oa::I32 tokLen = T / tok->downsampleFactor();
	std::vector<float> ds(static_cast<size_t>(B) * T * tokCfg.inputDim);
	for (oa::I32 b = 0; b < B; ++b) {
		for (oa::I32 t = 0; t < T; ++t) {
			const float tt = static_cast<float>(t) / static_cast<float>(T);
			for (oa::I32 c = 0; c < tokCfg.inputDim; ++c) {
				const float freq = 1.0F + static_cast<float>(c % 8) * 0.5F;
				ds[((static_cast<size_t>(b) * T + t) * tokCfg.inputDim) + c] =
					static_cast<float>(b) * 0.7F + std::sin(6.2831853F * freq * tt) * 0.5F;
			}
		}
	}
	oa::Matrix X = oa::FnMatrix::fromBytes(
		oa::Span<const oa::U8>(reinterpret_cast<const oa::U8*>(ds.data()), ds.size() * sizeof(float)),
		oa::MatrixShape{B, T, tokCfg.inputDim}, oa::ScalarType::Float32);
	{ auto z0 = tok->encode(X, B, T); tok->seed(z0); ctx.clear(); }   // seed codebook (2048 latents ≥ 512)

	auto tokParams = tok->allParameterPtrs();
	auto tokOpt = oa::makeUnique<oa::AdamW>(tokParams, 2e-4F, 0.9F, 0.99F, 1e-8F, 0.0F);

	const int warmup = 3, timed = 8;
	auto stepOnce = [&]() {
		ctx.clear();
		oa::GradientTape tape; tokOpt->zeroGrad();
		auto z = tok->encode(X, B, T);
		auto q = tok->quantize(z);
		auto rec = tok->decode(q.quantized, B, tokLen);
		auto recon = oa::FnLoss::smoothL1(rec, X.reshape(oa::MatrixShape{static_cast<oa::I64>(B) * T, tokCfg.inputDim}));
		auto loss = recon + q.commitLoss;
		tape.backward(loss);
		tokOpt->step();
		tok->emaUpdate(q);
		(void)testSubmitAndWait(ctx);
	};
	for (int i = 0; i < warmup; ++i) stepOnce();
	auto t0 = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < timed; ++i) stepOnce();
	auto t1 = std::chrono::high_resolution_clock::now();
	const double dt = std::chrono::duration<double>(t1 - t0).count();
	const double msStep = dt / timed * 1000.0;
	const double sps = static_cast<double>(B) * timed / dt;
	const char* prec = oa::FnMatrix::weightDtype() == oa::ScalarType::Float32 ? "fp32" : "bf16";
	std::printf("  [tok bench %s] W=512 263d 3L downT=2 B=%d T=%d | %.2f ms/step | %.1f clips/s\n",
		prec, B, T, msStep, sps);
	tokOpt.reset(); tok.reset(); ctx.clear();
}

// iGPU-sized SMOKE tokenizer config shared by the two CMP training tests. NOT the
// reference 512³ model — those tests verify scale-invariant properties (the tokenizer
// LEARNS; the codebook does NOT collapse), so a light net that fits an integrated GPU
// is the correct vehicle and a full run finishes in seconds rather than never.
//   • The conv stack runs through the im2col + tiled-matmul path (the retired scalar
//     direct-conv was ~20× slower at tokenizer shapes) — the biggest iGPU-tractability lever.
//   • width/codeDim/depth trimmed for speed; numCodes stays 512 because codebook health
//     at the real K is precisely the property under test.
//   • commitBeta/emaDecay/deadThresh are the VQ-collapse-safe values (§12.1).
static oa::AlmTokenizerConfig cmpSmokeTokenizerCfg(oa::I32 inFeatDim) {
	oa::AlmTokenizerConfig cfg;
	cfg.inputDim    = inFeatDim;  // 263 for CMP/HumanML3D
	cfg.width       = 128;
	cfg.codeDim     = 128;
	cfg.numCodes    = 512;
	cfg.downT       = 2;
	cfg.depth       = 2;
	cfg.commitBeta  = 0.25F;
	cfg.emaDecay    = 0.99F;
	cfg.deadThresh  = 2.0F;
	return cfg;
}

// codebook health: tokenize a batch, histogram code usage →
// active-code % + perplexity (effective #codes used = exp(-Σ p·log p)). VQ collapse
// (a few dead-heavy codes) is the #1 cause of blurry/mode-collapsed motion; MotionGPT's
// mgpt_vq.py logs this and we did not (phase A4). perplexity is scale-robust; active-%
// needs tokens ≫ codes to be meaningful, so tokenize as many clips as fit.
// Returns perplexity (effective #codes in use) — a scale-robust collapse metric: a
// collapsed codebook drives it toward 1.0, so callers can assert on it as a hard gate.
static double printCodebookHealth(oa::AlmTokenizerAg& inTok, const oa::Matrix& inX,
                                oa::I32 inBatch, oa::I32 inFrames, oa::I32 inNumCodes) {
	auto& ctx = oa::ExecutionSession::getActive();
	auto ids = inTok.tokenize(inX, inBatch, inFrames)[0];
	(void)testSubmitAndWait(ctx);
	const oa::I64 n = ids.numElements();
	const oa::I32* p = ids.dataAs<const oa::I32>();
	std::vector<oa::I64> hist(static_cast<size_t>(inNumCodes), 0);
	for (oa::I64 i = 0; i < n; ++i) {
		const oa::I32 c = p[i];
		if (c >= 0 && c < inNumCodes) ++hist[static_cast<size_t>(c)];
	}
	oa::I32 active = 0;
	double H = 0.0;
	for (oa::I32 c = 0; c < inNumCodes; ++c) {
		if (hist[static_cast<size_t>(c)] == 0) continue;
		++active;
		const double pr = static_cast<double>(hist[static_cast<size_t>(c)]) / static_cast<double>(n);
		H -= pr * std::log(pr);
	}
	const double perplexity = std::exp(H);
	std::printf("  [codebook] %d/%d active (%.1f%%) | perplexity %.1f | %lld tokens\n",
	            active, inNumCodes, 100.0 * active / inNumCodes, perplexity,
	            static_cast<long long>(n));
	ctx.clear();
	return perplexity;
}

// save/load round-trip: a trained tokenizer's conv weights AND its EMA codebook
// must persist. seed + move the codebook off init, tokenize, save, reload into a
// fresh module, tokenize again — identical token ids prove BOTH the conv weights
// (Encode) and the codebook (assignment) survived. If only gradient params were
// saved and the EMA codebook were dropped, the reload would retokenize differently
// and this fails.
TEST(Alm, TokenizerSaveLoadRoundtrip) {
	oa::FnMatrix::setRngSeed(11);
	auto& ctx = oa::ExecutionSession::getActive();
	auto cfg = cmpSmokeTokenizerCfg(32);            // small inputDim for speed
	const oa::I32 B = 64, T = 64;                      // B·(T/Factor)=1024 ≥ numCodes(512) to seed
	const oa::I32 tokLen = T / (1 << cfg.downT);

	// Deterministic synthetic batch.
	std::vector<float> xh(static_cast<size_t>(B) * T * cfg.inputDim);
	oa::U64 rng = 0x1234ULL;
	for (auto& v : xh) {
		rng = (rng * 6364136223846793005ULL) + 1;
		v = (static_cast<float>(static_cast<oa::U32>(rng >> 40)) / static_cast<float>(1 << 24)) - 0.5F;
	}
	auto X = oa::FnMatrix::fromBytes(
		oa::Span<const oa::U8>(reinterpret_cast<const oa::U8*>(xh.data()), xh.size() * sizeof(float)),
		oa::MatrixShape{B, T, cfg.inputDim}, oa::ScalarType::Float32);

	auto tok = oa::makeShared<oa::AlmTokenizerAg>(cfg);
	{ auto z0 = tok->encode(X, B, T); tok->seed(z0); ctx.clear(); }   // data-dependent codebook seed
	for (int s = 0; s < 3; ++s) {                                      // move the EMA codebook off the seed
		auto z = tok->encode(X, B, T);
		auto q = tok->quantize(z);
		tok->emaUpdate(q);
		(void)testSubmitAndWait(ctx); ctx.clear();
	}

	auto readIds = [&](const oa::Vec<oa::Matrix>& inIdx) {
		std::vector<std::vector<oa::I32>> out;
		for (const auto& m : inIdx) {
			std::vector<oa::I32> h(static_cast<size_t>(m.numElements()));
			(void)oa::FnMatrix::copyToHost(m, h.data(), h.size() * sizeof(oa::I32));
			out.push_back(std::move(h));
		}
		return out;
	};

	auto idsA = tok->tokenize(X, B, T);
	(void)testSubmitAndWait(ctx);
	auto hostA = readIds(idsA);
	ctx.clear();

	const oa::String path = "/tmp/oa_mg_tok_roundtrip.oam";
	ASSERT_TRUE(tok->save(ctx.engine(), path).isOk()) << "tokenizer save failed";

	auto tok2 = oa::makeShared<oa::AlmTokenizerAg>(cfg);   // fresh random init
	ASSERT_TRUE(tok2->load(ctx.engine(), path).isOk()) << "tokenizer load failed";
	auto idsB = tok2->tokenize(X, B, T);
	(void)testSubmitAndWait(ctx);
	auto hostB = readIds(idsB);
	ctx.clear();

	ASSERT_EQ(hostA.size(), hostB.size()) << "level count mismatch";
	oa::I64 total = 0, mismatch = 0;
	for (size_t lvl = 0; lvl < hostA.size(); ++lvl) {
		ASSERT_EQ(hostA[lvl].size(), hostB[lvl].size());
		for (size_t i = 0; i < hostA[lvl].size(); ++i) {
			++total;
			if (hostA[lvl][i] != hostB[lvl][i]) ++mismatch;
		}
	}
	std::printf("TokenizerSaveLoadRoundtrip: %lld/%lld tokens identical after save+load (%zu levels)\n",
		static_cast<long long>(total - mismatch), static_cast<long long>(total), hostA.size());
	EXPECT_EQ(mismatch, 0) << "reloaded tokenizer retokenized differently — conv weights or EMA codebook not persisted";
	(void)tokLen;
}

TEST(Alm, TokenizerLearnsCmp) {
	oa::FnMatrix::setRngSeed(7);
	auto& ctx = oa::ExecutionSession::getActive();

	const oa::String dsPath = oa::Paths::data("humanMl3d/Cmp").string();
	oa::DsCombatMotionProcessed ds(dsPath, "train", /*inMaxClips=*/128);
	ASSERT_TRUE(ds.ok()) << "Failed to load CMP from " << dsPath.cStr();
	ASSERT_GE(ds.numClips(), 128) << "Not enough clips in dataset";

	auto tokCfg = cmpSmokeTokenizerCfg(ds.featDim());
	auto tok = oa::makeShared<oa::AlmTokenizerAg>(tokCfg);

	const oa::I32 B = 8;
	const oa::I32 T = 64;
	const oa::I32 tokLen = T / tok->downsampleFactor();  // 16

	// seed the codebook: need >= numCodes latent rows. With factor=4 and
	// numCodes=512, use B_seed=64 clips × T=64 frames → 1024 latents.
	const oa::I32 Bseed = 64;
	{
		std::vector<float> seed(static_cast<size_t>(Bseed * T) * ds.featDim());
		for (oa::I32 b = 0; b < Bseed; ++b) {
			const oa::I32 clipIdx = b % ds.numClips();
			const oa::I32 frames = static_cast<oa::I32>(ds.clipFrames(clipIdx));
			const oa::I32 start = frames > T ? (frames - T) / 2 : 0;
			const oa::F32* src = ds.clipData(clipIdx) + start * ds.featDim();
			float* dst = seed.data() + static_cast<size_t>(b) * T * ds.featDim();
			const oa::I32 copyFrames = std::min(T, frames);
			std::memcpy(dst, src, static_cast<size_t>(copyFrames) * ds.featDim() * sizeof(float));
		}
		auto seedX = oa::FnMatrix::fromBytes(
			oa::Span<const oa::U8>(reinterpret_cast<const oa::U8*>(seed.data()), seed.size() * sizeof(float)),
			oa::MatrixShape{Bseed, T, ds.featDim()}, oa::ScalarType::Float32);
		auto z0 = tok->encode(seedX, Bseed, T);
		tok->seed(z0);
		ctx.clear();
	}

	auto tokParams = tok->allParameterPtrs();
	auto tokOpt = oa::makeUnique<oa::AdamW>(tokParams, 1e-4F, 0.9F, 0.99F, 1e-8F, 0.01F);

	oa::F32 firstLoss = 0.0F;
	oa::F32 lastLoss = 0.0F;
	for (oa::I32 s = 1; s <= 500; ++s) {
		ctx.clear();
		oa::GradientTape tape; tokOpt->zeroGrad();

		std::vector<float> batch(static_cast<size_t>(B * T) * ds.featDim());
		for (oa::I32 b = 0; b < B; ++b) {
			const oa::I32 clipIdx = b % ds.numClips();
			const oa::I32 frames = static_cast<oa::I32>(ds.clipFrames(clipIdx));
			const oa::I32 start = frames > T ? (s * 17 + b * 31) % (frames - T) : 0;
			const oa::F32* src = ds.clipData(clipIdx) + start * ds.featDim();
			float* dst = batch.data() + static_cast<size_t>(b) * T * ds.featDim();
			const oa::I32 copyFrames = std::min(T, frames);
			std::memcpy(dst, src, static_cast<size_t>(copyFrames) * ds.featDim() * sizeof(float));
		}
		auto X = oa::FnMatrix::fromBytes(
			oa::Span<const oa::U8>(reinterpret_cast<const oa::U8*>(batch.data()), batch.size() * sizeof(float)),
			oa::MatrixShape{B, T, ds.featDim()}, oa::ScalarType::Float32);

		auto z = tok->encode(X, B, T);
		auto q = tok->quantize(z);
		auto rec = tok->decode(q.quantized, B, tokLen);
		auto xFlat = X.reshape(oa::MatrixShape{static_cast<oa::I64>(B) * T, ds.featDim()});
		auto recon = oa::FnLoss::smoothL1(rec, xFlat);

		// velocity loss: SmoothL1 on frame-to-frame differences (improves motion smoothness).
		// LAMBDA_VELOCITY=0.5 (reference: config_h3d_stage2.yaml)
		auto rec3d   = rec.reshape(oa::MatrixShape{B, T, ds.featDim()});
		auto xFlat3d = X;  // already [B, T, D]
		auto recVel   = oa::FnMatrix::sub(oa::FnMatrix::slice(rec3d, 1, 1, T),   oa::FnMatrix::slice(rec3d, 1, 0, T - 1));
		auto xVel     = oa::FnMatrix::sub(oa::FnMatrix::slice(xFlat3d, 1, 1, T), oa::FnMatrix::slice(xFlat3d, 1, 0, T - 1));
		auto velLoss  = oa::FnLoss::smoothL1(recVel.reshape(oa::MatrixShape{static_cast<oa::I64>(B) * (T - 1), ds.featDim()}),
		                                   xVel.reshape(oa::MatrixShape{static_cast<oa::I64>(B) * (T - 1), ds.featDim()}));

		auto loss = recon + oa::FnMatrix::scale(velLoss, 0.5F) + q.commitLoss;
		tape.backward(loss);
		tokOpt->step();
		tok->emaUpdate(q);
		(void)testSubmitAndWait(ctx);

		const float lv = loss.at(0);
		if (s == 1) firstLoss = lv;
		lastLoss = lv;
		if (s == 1 || s % 100 == 0 || s == 500)
			std::printf("  [h3d] step %3d | loss %.8f\n", s, static_cast<double>(lv));
		ASSERT_TRUE(std::isfinite(lv)) << "tokenizer diverged at step " << s;
	}
	std::printf("CMP tokenizer loss: %.8f -> %.8f\n", static_cast<double>(firstLoss), static_cast<double>(lastLoss));
	EXPECT_LT(lastLoss, firstLoss) << "tokenizer did not learn CMP reconstruction";

	// codebook health (§12.1): tokenize all clips (fixed T window) → active% + perplexity.
	{
		const oa::I32 bh = std::min<oa::I32>(128, static_cast<oa::I32>(ds.numClips()));
		std::vector<float> hb(static_cast<size_t>(bh * T) * ds.featDim());
		for (oa::I32 b = 0; b < bh; ++b) {
			const oa::I32 frames = static_cast<oa::I32>(ds.clipFrames(b));
			const oa::I32 start = frames > T ? (frames - T) / 2 : 0;
			const oa::F32* src = ds.clipData(b) + start * ds.featDim();
			float* dst = hb.data() + static_cast<size_t>(b) * T * ds.featDim();
			std::memcpy(dst, src, static_cast<size_t>(std::min(T, frames)) * ds.featDim() * sizeof(float));
		}
		auto Xh = oa::FnMatrix::fromBytes(
			oa::Span<const oa::U8>(reinterpret_cast<const oa::U8*>(hb.data()), hb.size() * sizeof(float)),
			oa::MatrixShape{bh, T, ds.featDim()}, oa::ScalarType::Float32);
		const double perplexity = printCodebookHealth(*tok, Xh, bh, T, tokCfg.numCodes);
		// Hard collapse gate: a healthy VQ spreads tokens over many codes (perplexity ≫ 1);
		// centroid collapse pins every latent on one code (perplexity → 1). The threshold is
		// far below observed-healthy and far above collapse, so it catches the regression
		// without flaking on run-to-run variance.
		EXPECT_GT(perplexity, 8.0) << "VQ codebook collapsed (near-degenerate token usage)";
	}

	tokOpt.reset();
	tok.reset();
	ctx.clear();
}

TEST(Alm, HumanMl3dLoads) {
	const oa::String dsPath = oa::Paths::data("humanMl3d/HumanML3D").string();
	oa::DsHumanMl3d ds(dsPath, "train", /*inMaxClips=*/4);
	ASSERT_TRUE(ds.ok()) << "Failed to load HumanML3D from " << dsPath.cStr();
	ASSERT_GE(ds.numClips(), 1) << "HumanML3D dataset has no clips";
	EXPECT_EQ(ds.featDim(), 263) << "HumanML3D must use 263-dim SMPL-22 features";
	EXPECT_EQ(ds.numJoints(), 22) << "HumanML3D must use 22-joint SMPL skeleton";
	for (oa::I32 i = 0; i < ds.numClips(); ++i) {
		const oa::I32 frames = ds.clipFrames(i);
		EXPECT_GT(frames, 0) << "clip " << i << " has no frames";
		const oa::F32* data = ds.clipData(i);
		float first = data[0];
		EXPECT_TRUE(std::isfinite(first)) << "clip " << i << " contains non-finite data";
		const auto& captions = ds.clipCaptions(i);
		if (not captions.empty()) {
			EXPECT_EQ(ds.clipCaptions(i)[0].text, captions[0].text);
			for (const auto& caption : captions) {
				EXPECT_FALSE(caption.text.empty());
				EXPECT_TRUE(std::isfinite(caption.startSec));
				EXPECT_TRUE(std::isfinite(caption.endSec));
			}
		}
	}
	std::printf("HumanML3D: loaded %d clips (%d-dim, %d-joint)\n", ds.numClips(), ds.featDim(), ds.numJoints());
}

TEST(Alm, CmpLoadsAllCaptions) {
	const oa::String dsPath = oa::Paths::data("humanMl3d/Cmp").string();
	oa::DsCombatMotionProcessed ds(dsPath, "train", /*inMaxClips=*/8);
	ASSERT_TRUE(ds.ok());
	ASSERT_GE(ds.numClips(), 1);
	for (oa::I32 i = 0; i < ds.numClips(); ++i) {
		const auto& captions = ds.clipCaptions(i);
		EXPECT_GE(captions.size(), 3u) << "CMP clip " << ds.clipId(i).cStr();
		ASSERT_FALSE(captions.empty());
		EXPECT_EQ(ds.clipCaptions(i)[0].text, captions[0].text);
		for (const auto& caption : captions) {
			EXPECT_FALSE(caption.text.empty());
			EXPECT_TRUE(std::isfinite(caption.startSec));
			EXPECT_TRUE(std::isfinite(caption.endSec));
		}
	}
}

TEST(Alm, HumanMl3dReferenceInverse) {
	constexpr oa::I32 frames = 2;
	constexpr oa::I32 featDim = 263;
	constexpr oa::I32 joints = 22;
	std::vector<oa::F32> features(static_cast<size_t>(frames) * featDim, 0.0F);
	features[1] = 1.0F;  // frame-0 root X velocity, applied to frame 1
	features[2] = 2.0F;  // frame-0 root Z velocity, applied to frame 1
	features[3] = 0.5F;
	features[featDim + 3] = 0.75F;
	for (oa::I32 t = 0; t < frames; ++t) {
		for (oa::I32 j = 1; j < joints; ++j) {
			const size_t base = static_cast<size_t>(t) * featDim + 4
				+ static_cast<size_t>(j - 1) * 3;
			features[base] = static_cast<oa::F32>(j) * 0.1F;
			features[base + 1] = static_cast<oa::F32>(j) * 0.2F;
			features[base + 2] = static_cast<oa::F32>(j) * -0.1F;
		}
	}

	auto world = oa::humanMl3dRecoverWorldJoints(
		oa::Span<const oa::F32>(features.data(), features.size()), frames, featDim);
	ASSERT_EQ(world.size(), static_cast<oa::Usize>(frames * joints * 3));
	EXPECT_FLOAT_EQ(world[0], 0.0F);
	EXPECT_FLOAT_EQ(world[1], 0.5F);
	EXPECT_FLOAT_EQ(world[2], 0.0F);
	const size_t frame1 = static_cast<size_t>(joints) * 3;
	EXPECT_FLOAT_EQ(world[frame1], 1.0F);
	EXPECT_FLOAT_EQ(world[frame1 + 1], 0.75F);
	EXPECT_FLOAT_EQ(world[frame1 + 2], 2.0F);
	EXPECT_NEAR(world[frame1 + 3], 1.1F, 1e-6F);
	EXPECT_NEAR(world[frame1 + 4], 0.2F, 1e-6F);
	EXPECT_NEAR(world[frame1 + 5], 1.9F, 1e-6F);

	EXPECT_DOUBLE_EQ(oa::humanMl3dMpjpeCm(
		oa::Span<const oa::F32>(world.data(), world.size()),
		oa::Span<const oa::F32>(world.data(), world.size())), 0.0);
	auto shifted = world;
	for (oa::Usize i = 0; i < shifted.size(); i += 3) shifted[i] += 0.01F;
	EXPECT_NEAR(oa::humanMl3dMpjpeCm(
		oa::Span<const oa::F32>(shifted.data(), shifted.size()),
		oa::Span<const oa::F32>(world.data(), world.size())), 1.0, 1e-4);
}

TEST(Alm, MaskedCrossEntropyAutograd) {
	const std::vector<float> logitsHost = {
		1.0f, 2.0f, 3.0f, 3.0f, 1.0f, 0.0f,
		9.0f, 8.0f, 7.0f, -2.0f, 0.0f, 2.0f};
	const std::vector<oa::I32> targetsHost = {2, 0, 0, 1};
	const std::vector<float> maskHost = {1.0f, 1.0f, 0.0f, 0.0f};
	auto logits = oa::FnMatrix::fromBytes(
		oa::Span<const oa::U8>(reinterpret_cast<const oa::U8*>(logitsHost.data()),
			logitsHost.size() * sizeof(float)), oa::MatrixShape{4, 3}, oa::ScalarType::Float32);
	auto targets = oa::FnMatrix::fromBytes(
		oa::Span<const oa::U8>(reinterpret_cast<const oa::U8*>(targetsHost.data()),
			targetsHost.size() * sizeof(oa::I32)), oa::MatrixShape{4}, oa::ScalarType::Int32);
	auto mask = oa::FnMatrix::fromBytes(
		oa::Span<const oa::U8>(reinterpret_cast<const oa::U8*>(maskHost.data()),
			maskHost.size() * sizeof(float)), oa::MatrixShape{4}, oa::ScalarType::Float32);
	logits.setRequiresGrad(true);
	oa::GradientTape tape;
	auto loss = oa::FnLoss::maskedCrossEntropy(logits, targets, mask, 2);
	tape.backward(loss);
	auto& ctx = oa::ExecutionSession::getActive();
	ASSERT_TRUE(testSubmitAndWait(ctx).isOk());
	EXPECT_TRUE(std::isfinite(loss.at(0)));
	const auto& grad = logits.gradMatrix();
	for (oa::I32 i = 2 * 3; i < 4 * 3; ++i) EXPECT_FLOAT_EQ(grad.at(i), 0.0f);
	for (oa::I32 row = 0; row < 2; ++row) {
		float sum = 0.0f;
		for (oa::I32 col = 0; col < 3; ++col) sum += grad.at(row * 3 + col);
		EXPECT_NEAR(sum, 0.0f, 1e-5f);
	}
	ctx.clear();
}

TEST(Alm, LmLearnsCmpTokens) {
	oa::FnMatrix::setRngSeed(13);
	auto& ctx = oa::ExecutionSession::getActive();

	const oa::String dsPath = oa::Paths::data("humanMl3d/Cmp").string();
	oa::DsCombatMotionProcessed ds(dsPath, "train", /*inMaxClips=*/128);
	ASSERT_TRUE(ds.ok()) << "Failed to load CMP from " << dsPath.cStr();
	ASSERT_GE(ds.numClips(), 128) << "Not enough clips in dataset";

	// stage 1: train a tokenizer on the CMP corpus.
	auto tokCfg = cmpSmokeTokenizerCfg(ds.featDim());
	auto tok = oa::makeShared<oa::AlmTokenizerAg>(tokCfg);

	const oa::I32 B = 8;
	const oa::I32 T = 64;
	const oa::I32 tokLen = T / tok->downsampleFactor();  // 16

	// seed the codebook: need >= numCodes latent rows. With factor=4 and
	// numCodes=512, use B_seed=64 clips × T=64 frames → 1024 latents.
	const oa::I32 Bseed = 64;
	{
		std::vector<float> seed(static_cast<size_t>(Bseed * T) * ds.featDim());
		for (oa::I32 b = 0; b < Bseed; ++b) {
			const oa::I32 clipIdx = b % ds.numClips();
			const oa::I32 frames = static_cast<oa::I32>(ds.clipFrames(clipIdx));
			const oa::I32 start = frames > T ? (frames - T) / 2 : 0;
			const oa::F32* src = ds.clipData(clipIdx) + start * ds.featDim();
			float* dst = seed.data() + static_cast<size_t>(b) * T * ds.featDim();
			const oa::I32 copyFrames = std::min(T, frames);
			std::memcpy(dst, src, static_cast<size_t>(copyFrames) * ds.featDim() * sizeof(float));
		}
		auto seedX = oa::FnMatrix::fromBytes(
			oa::Span<const oa::U8>(reinterpret_cast<const oa::U8*>(seed.data()), seed.size() * sizeof(float)),
			oa::MatrixShape{Bseed, T, ds.featDim()}, oa::ScalarType::Float32);
		auto z0 = tok->encode(seedX, Bseed, T);
		tok->seed(z0);
		ctx.clear();
	}

	auto tokParams = tok->allParameterPtrs();
	auto tokOpt = oa::makeUnique<oa::AdamW>(tokParams, 1e-4F, 0.9F, 0.99F, 1e-8F, 0.01F);
	for (oa::I32 s = 1; s <= 500; ++s) {
		ctx.clear();
		oa::GradientTape tape; tokOpt->zeroGrad();
		std::vector<float> batch(static_cast<size_t>(B * T) * ds.featDim());
		for (oa::I32 b = 0; b < B; ++b) {
			const oa::I32 clipIdx = b % ds.numClips();
			const oa::I32 frames = static_cast<oa::I32>(ds.clipFrames(clipIdx));
			const oa::I32 start = frames > T ? (s * 17 + b * 31) % (frames - T) : 0;
			const oa::F32* src = ds.clipData(clipIdx) + start * ds.featDim();
			float* dst = batch.data() + static_cast<size_t>(b) * T * ds.featDim();
			const oa::I32 copyFrames = std::min(T, frames);
			std::memcpy(dst, src, static_cast<size_t>(copyFrames) * ds.featDim() * sizeof(float));
		}
		auto X = oa::FnMatrix::fromBytes(
			oa::Span<const oa::U8>(reinterpret_cast<const oa::U8*>(batch.data()), batch.size() * sizeof(float)),
			oa::MatrixShape{B, T, ds.featDim()}, oa::ScalarType::Float32);
		auto z = tok->encode(X, B, T);
		auto q = tok->quantize(z);
		auto rec = tok->decode(q.quantized, B, tokLen);
		auto xFlat = X.reshape(oa::MatrixShape{static_cast<oa::I64>(B) * T, ds.featDim()});
		auto recon = oa::FnLoss::smoothL1(rec, xFlat);

		// velocity loss: SmoothL1 on frame-to-frame differences (improves motion smoothness).
		// LAMBDA_VELOCITY=0.5 (reference: config_h3d_stage2.yaml)
		auto rec3d   = rec.reshape(oa::MatrixShape{B, T, ds.featDim()});
		auto xFlat3d = X;  // already [B, T, D]
		auto recVel   = oa::FnMatrix::sub(oa::FnMatrix::slice(rec3d, 1, 1, T),   oa::FnMatrix::slice(rec3d, 1, 0, T - 1));
		auto xVel     = oa::FnMatrix::sub(oa::FnMatrix::slice(xFlat3d, 1, 1, T), oa::FnMatrix::slice(xFlat3d, 1, 0, T - 1));
		auto velLoss  = oa::FnLoss::smoothL1(recVel.reshape(oa::MatrixShape{static_cast<oa::I64>(B) * (T - 1), ds.featDim()}),
		                                   xVel.reshape(oa::MatrixShape{static_cast<oa::I64>(B) * (T - 1), ds.featDim()}));

		auto loss = recon + oa::FnMatrix::scale(velLoss, 0.5F) + q.commitLoss;
		tape.backward(loss);
		tokOpt->step();
		tok->emaUpdate(q);
		(void)testSubmitAndWait(ctx);
		ASSERT_TRUE(std::isfinite(loss.at(0))) << "tokenizer diverged at step " << s;
	}
	tokOpt.reset();
	ctx.clear();

	// stage 2: tokenize each long clip and collect token sequences.
	const oa::I32 lmTokLen = 16;  // window of 16 tokens per sequence
	const oa::I32 minFrames = lmTokLen * tok->downsampleFactor();  // 128
	std::vector<std::vector<oa::I32>> tokenSequences;
	for (oa::I32 i = 0; i < ds.numClips(); ++i) {
		const oa::I32 frames = static_cast<oa::I32>(ds.clipFrames(i));
		if (frames < minFrames) continue;
		std::vector<float> clip(static_cast<size_t>(frames) * ds.featDim());
		std::memcpy(clip.data(), ds.clipData(i), clip.size() * sizeof(float));
		auto x = oa::FnMatrix::fromBytes(
			oa::Span<const oa::U8>(reinterpret_cast<const oa::U8*>(clip.data()), clip.size() * sizeof(float)),
			oa::MatrixShape{1, frames, ds.featDim()}, oa::ScalarType::Float32);
		auto ids = tok->tokenize(x, 1, frames)[0];
		(void)testSubmitAndWait(ctx);
		const oa::I64 n = ids.numElements();
		const oa::I32* p = ids.dataAs<const oa::I32>();
		tokenSequences.emplace_back(p, p + n);
		ctx.clear();
	}
	ASSERT_GE(static_cast<oa::I32>(tokenSequences.size()), B) << "Not enough long clips to build LM batches";
	std::printf("  [lm cmp] collected %zu token sequences (minFrames=%d)\n", tokenSequences.size(), minFrames);
	std::fflush(stdout);

	// stage 3: train the AR transformer on sliding windows.
	// iGPU-sized SMOKE LM (reference is D=384/L=6/DFF=1536). The iGPU shares system
	// RAM, so a full transformer + autograd tape OOMs the box; this proves next-token
	// learning at a footprint that fits.
	oa::AlmPriorConfig lmCfg;
	lmCfg.syncVocab(tokCfg.numCodes);
	lmCfg.dModel = 192; lmCfg.numLayers = 3; lmCfg.dFfn = 512;
	lmCfg.seqLen = lmTokLen + 1;
	auto lm = oa::makeShared<oa::AlmPriorAg>(lmCfg);
	(void)testSubmitAndWait(ctx);  // flush LM initialization
	auto lmParams = lm->allParameterPtrs();
	auto lmOpt = oa::makeUnique<oa::AdamW>(lmParams, 1e-4F, 0.9F, 0.99F, 1e-8F, 0.01F);

	oa::F32 firstLoss = 0.0F;
	oa::F32 lastLoss = 0.0F;
	for (oa::I32 s = 1; s <= 500; ++s) {
		ctx.clear();
		oa::GradientTape tape; lmOpt->zeroGrad();

		std::vector<oa::I32> inputHost(static_cast<size_t>(B) * (lmTokLen + 1));
		std::vector<oa::I32> targetHost(static_cast<size_t>(B) * (lmTokLen + 1));
		for (oa::I32 b = 0; b < B; ++b) {
			const auto& seq = tokenSequences[(s + b) % tokenSequences.size()];
			const oa::I32 maxStart = static_cast<oa::I32>(seq.size()) - lmTokLen;
			const oa::I32 start = maxStart > 0 ? (s * 23 + b * 17) % maxStart : 0;
			const size_t row = static_cast<size_t>(b) * (lmTokLen + 1);
			inputHost[row] = lmCfg.somToken;
			for (oa::I32 t = 0; t < lmTokLen; ++t) {
				const oa::I32 id = static_cast<oa::I32>(seq[start + t]);
				inputHost[row + 1 + t] = id;
				targetHost[row + t] = id;
			}
			targetHost[row + lmTokLen] = lmCfg.eomToken;
		}
		auto inputIds = oa::FnMatrix::fromBytes(
			oa::Span<const oa::U8>(reinterpret_cast<const oa::U8*>(inputHost.data()), inputHost.size() * sizeof(oa::U32)),
			oa::MatrixShape{B, lmTokLen + 1}, oa::ScalarType::UInt32);
		auto targetIds = oa::FnMatrix::fromBytes(
			oa::Span<const oa::U8>(reinterpret_cast<const oa::U8*>(targetHost.data()), targetHost.size() * sizeof(oa::U32)),
			oa::MatrixShape{B, lmTokLen + 1}, oa::ScalarType::UInt32);

		auto logits = lm->forward(inputIds);
		auto logitsFlat = logits.reshape(oa::MatrixShape{static_cast<oa::I64>(B) * (lmTokLen + 1), lmCfg.vocabSize});
		auto targetFlat = targetIds.reshape(oa::MatrixShape{static_cast<oa::I64>(B) * (lmTokLen + 1)});
		auto ce = oa::FnLoss::crossEntropy(logitsFlat, targetFlat);
		tape.backward(ce);
		lmOpt->step();
		(void)testSubmitAndWait(ctx);

		const float lv = ce.at(0);
		if (s == 1) firstLoss = lv;
		lastLoss = lv;
		if (s == 1 || s % 50 == 0 || s == 200)
			std::printf("  [lm cmp] step %3d | ce %.8f\n", s, static_cast<double>(lv));
		ASSERT_TRUE(std::isfinite(lv)) << "LM diverged at step " << s;
	}
	std::printf("CMP LM cross-entropy: %.8f -> %.8f\n", static_cast<double>(firstLoss), static_cast<double>(lastLoss));
	EXPECT_LT(lastLoss, firstLoss) << "LM did not learn next-token prediction on CMP tokens";

	// stage 4: end-to-end generation — sample multiple tokens streams and decode to motion.
	ctx.clear();
	const float temperatures[] = {1.0F, 2.0F, 3.0F};
	std::vector<std::vector<oa::I32>> genStreams;   // captured per-temp token streams (diversity guard)
	for (oa::I32 g = 0; g < 3; ++g) {
		auto generated = lm->generate(1, temperatures[g], 0, 0.9F, lmTokLen);
		auto motion = lm->decodeToMotion(generated, *tok);
		(void)testSubmitAndWait(ctx);

		std::vector<oa::I32> genHost(static_cast<size_t>(generated.numElements()));
		(void)oa::FnMatrix::copyToHost(generated, genHost.data(), genHost.size() * sizeof(oa::I32));
		genStreams.push_back(std::move(genHost));
		std::printf("  [lm cmp] generated motion %d (T=%.2f) shape: [%lld, %lld]\n", g,
			temperatures[g], static_cast<long long>(motion.size(0)), static_cast<long long>(motion.size(1)));
		EXPECT_EQ(motion.size(1), ds.featDim()) << "generated motion feature dim must match dataset";
		EXPECT_GT(motion.size(0), 0) << "generated motion must have frames";

		// denormalize features and recover world joint positions for USD export.
		char pathBuf[128];
		const oa::I32 frames = static_cast<oa::I32>(motion.size(0));
		const oa::I32 featDim = ds.featDim();
		auto motionHost = hostFloatData(motion);
		std::vector<float> featHost(motionHost.data(), motionHost.data() + motionHost.size());
		ds.denormalize(featHost.data(), frames);
		auto worldJoints = oa::humanMl3dRecoverWorldJoints(
			oa::Span<const oa::F32>(featHost.data(), featHost.size()), frames, featDim);
		auto skelClip = oa::usdClipFromWorldJoints(
			oa::skHumanMl3d(),
			oa::Span<const oa::F32>(worldJoints.data(), worldJoints.size()),
			frames, 20.0F, 1, 100.0F);
		std::snprintf(pathBuf, sizeof(pathBuf),
			"var/alm/Alm_LmLearnsCmpTokens_generated_%d_T%.1f.usda", g, temperatures[g]);
		oa::Path usdPath(pathBuf);
		auto usdSt = oa::Usd::writeUsda(usdPath, skelClip, "humanml3d");
		std::printf("  [lm cmp] saved generated skeleton %d to %s (%s)\n", g,
			usdPath.cStr(), usdSt.isOk() ? "ok" : usdSt.toString().cStr());
		EXPECT_TRUE(usdSt.isOk()) << "Failed to write generated motion .usda";
	}

	// Diversity guard: sampling at rising temperatures must not collapse to one
	// clip. Count distinct generated streams — mode-collapse (identical tokens
	// regardless of temperature) would leave only 1.
	oa::I32 distinct = 0;
	for (size_t i = 0; i < genStreams.size(); ++i) {
		bool isNew = true;
		for (size_t j = 0; j < i; ++j) { if (genStreams[j] == genStreams[i]) { isNew = false; break; } }
		if (isNew) ++distinct;
	}
	std::printf("  [lm cmp] generation diversity: %d/%zu distinct token streams across T=1/2/3\n",
		distinct, genStreams.size());
	EXPECT_GT(distinct, 1) << "generation mode-collapsed — all temperatures produced identical tokens";

	lmOpt.reset();
	lm.reset();
	tok.reset();
	ctx.clear();
}
