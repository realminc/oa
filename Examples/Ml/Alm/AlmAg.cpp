// OA Example — Alm
//
// A port of MotionGPT into OA: a two-stage model that generates
// human motion as a stream of discrete tokens.
//
//   Stage 1  VQ-VAE tokenizer   — motion to discrete tokens
//   Stage 2  Autoregressive LM  — token generation
//   Generate                    — sample tokens, decode, USD output
//
// Run tests:
//   ./Alm --gtest_filter="Alm.*"

#include "../../Test/OaTest.h"
#include <Ml/Nn/Alm/AlmConfig.h>
#include <Ml/Nn/Alm/AlmAg.h>
#include <Ml/Nn/Alm/AlmTokenizerAg.h>
#include <Ml/Nn/Alm/AlmPriorAg.h>
#include <Anim/Usd.h>
#include <Oa/Core/FileIo.h>
#include <Oa/Core/Vlm.h>
#include <Oa/Data/DsHumanMl3d.h>
#include <Oa/Ml.h>
#include <Oa/Ml/Autograd.h>
#include <Oa/Ml/FnOptim.h>
#include <Oa/Ml/Oam.h>
#include <Oa/Runtime/Context.h>
#include <Rig/Skeleton.h>
#include <Rig/SkeletonUsd.h>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>
#include <chrono>
#include <functional>

// Copy a matrix to host FP32. Safe for BF16/FP16 storage models.
static OaVec<OaF32> HostFloatData(const OaMatrix& InMatrix) {
	auto& ctx = OaContext::GetDefault();
	if (InMatrix.GetDtype() == OaScalarType::Float32) {
		(void)ctx.Execute(); (void)ctx.Sync();
		const OaF32* p = InMatrix.DataAs<const OaF32>();
		return OaVec<OaF32>(p, p + InMatrix.NumElements());
	}
	OaMatrix f32 = OaFnMatrix::Empty(InMatrix.GetShape(), OaScalarType::Float32);
	OaFnMatrix::CastInto(InMatrix, f32);
	(void)ctx.Execute(); (void)ctx.Sync();
	const OaF32* p = f32.DataAs<const OaF32>();
	return OaVec<OaF32>(p, p + f32.NumElements());
}

// Dot product on host FP32. Safe for BF16/FP16 storage models.
static double HostDot(const OaMatrix& InA, const OaMatrix& InB) {
	auto a = HostFloatData(InA);
	auto b = HostFloatData(InB);
	OA_ASSERT(a.Size() == b.Size() && "HostDot: size mismatch");
	double s = 0.0;
	for (size_t i = 0; i < a.Size(); ++i) { s += static_cast<double>(a[i]) * static_cast<double>(b[i]); }
	return s;
}

// True if every element is finite. Safe for BF16/FP16 storage models.
static bool HostAllFinite(const OaMatrix& InMatrix) {
	auto h = HostFloatData(InMatrix);
	for (size_t i = 0; i < h.Size(); ++i) { if (not std::isfinite(h[i])) return false; }
	return true;
}

// Config implementation

OaAlmDatasetConfig OaAlmDatasetConfig::FromEnv() {
	OaAlmDatasetConfig cfg;
	cfg.Corpus = "cmp";
	cfg.DataDir = "../dataset/gen/3d/anim/ds/Cmp";
	cfg.Split = "train";
	cfg.MaxClips = 0;
	
	// Override from environment
	if (const char* corpus = std::getenv("OA_MOTION_CORPUS")) {
		cfg.Corpus = corpus;
	}
	if (const char* dataDir = std::getenv("OA_MOTION_DATA")) {
		cfg.DataDir = dataDir;
	}
	if (const char* split = std::getenv("OA_MOTION_SPLIT")) {
		cfg.Split = split;
	}
	if (const char* maxClips = std::getenv("OA_MOTION_MAX_CLIPS")) {
		cfg.MaxClips = std::atoi(maxClips);
	}
	
	return cfg;
}

// Test cases

TEST(Alm, ConfigTest) {
	auto datasetCfg = OaAlmDatasetConfig::FromEnv();
	std::printf("Dataset: %s\n", datasetCfg.Corpus.CStr());
	std::printf("DataDir: %s\n", datasetCfg.DataDir.CStr());
	
	OaAlmTokenizerConfig tokCfg;
	std::printf("Tokenizer: InputDim=%d, NumCodes=%d\n", tokCfg.InputDim, tokCfg.NumCodes);
	
	OaAlmPriorConfig lmCfg;
	lmCfg.SyncVocab(tokCfg.NumCodes);
	std::printf("LM: VocabSize=%d, DModel=%d, NumHeads=%d\n",
		lmCfg.VocabSize, lmCfg.DModel, lmCfg.NumHeads);
	
	EXPECT_EQ(lmCfg.VocabSize, tokCfg.NumCodes + 3);
}

// Encode → Quantize → Decode round-trip: the 8× temporal downsample must show up in
// the token count (T → T/Factor) and the decode must restore the frame count (T), with
// finite values throughout.
TEST(Alm, TokenizerRoundTripShape) {
	OaAlmTokenizerConfig cfg;
	cfg.InputDim = 48; cfg.Width = 64; cfg.CodeDim = 32; cfg.NumCodes = 64;
	cfg.DownT = 3; cfg.Depth = 1;                       // factor 8
	auto tok = OaMakeSharedPtr<OaAlmTokenizerAg>(cfg);
	ASSERT_EQ(tok->DownsampleFactor(), 8);

	const OaI32 B = 2;
	const OaI32 T = 32;                                 // multiple of 8 → 4 tokens/seq
	auto& ctx = OaContext::GetDefault();
	auto x = OaFnMatrix::RandN(OaMatrixShape{B, T, cfg.InputDim});
	auto z = tok->Encode(x, B, T);                      // [B·T/8, CodeDim]
	(void)ctx.Execute(); (void)ctx.Sync();
	EXPECT_EQ(z.Size(0), static_cast<OaI64>(B) * (T / 8));
	EXPECT_EQ(z.Size(1), cfg.CodeDim);

	auto q   = tok->Quantize(z);
	auto rec = tok->Decode(q.Quantized, B, T / 8);      // [B·T, InputDim]
	(void)ctx.Execute(); (void)ctx.Sync();
	EXPECT_EQ(rec.Size(0), static_cast<OaI64>(B) * T);
	EXPECT_EQ(rec.Size(1), cfg.InputDim);
	ASSERT_TRUE(HostAllFinite(rec)) << "decoded tensor has non-finite values";
	ctx.Clear();
}

// Exact gradient check for the new OaConvTranspose1d op via the bilinear identity:
// y = ConvT(x; W) is bilinear, so for any cotangent g,
//   <y, g> == <x, ∂/∂x> == <W, ∂/∂W>   i.e.  sum(y*g) == sum(x*x.grad) == sum(W*W.grad).
// This validates BOTH the dX (adjoint = Conv1d) and dW (= Conv1dBwdWeight) paths with no
// finite-difference epsilon. If the adjoint were wrong, these three would disagree.
TEST(Alm, ConvTranspose1dGradCheck) {
	OaFnMatrix::SetRngSeed(123);
	const OaI32 inC = 3, outC = 2, K = 4, S = 2, P = 1, B = 2, L = 5;
	auto ct = OaMakeSharedPtr<OaConvTranspose1d>(inC, outC, K, S, P);
	auto& ctx = OaContext::GetDefault();

	auto x = OaFnMatrix::RandN(OaMatrixShape{B, inC, L});
	x.SetRequiresGrad(true);
	auto& W = ct->Parameters()[0].Data;             // [inC, outC, K]
	auto g = OaFnMatrix::RandN(OaMatrixShape{B, outC, (L - 1) * S - 2 * P + K});

	OaGradientTape tape;
	auto y = ct->Forward(x);
	auto loss = OaFnMatrix::Sum(OaFnMatrix::Mul(y, g));
	tape.Backward(loss);
	(void)ctx.Execute(); (void)ctx.Sync();

	const double sYg = static_cast<double>(loss.At(0));
	auto dx = x.GradMatrix();
	auto dW = W.GradMatrix();
	double sXdx = HostDot(x, dx);
	double sWdW = HostDot(W, dW);
	std::printf("ConvT grad-check: <y,g>=%.8f  <x,dx>=%.8f  <W,dW>=%.8f\n", sYg, sXdx, sWdW);
	EXPECT_NEAR(sXdx, sYg, 1e-3 * (1.0 + std::abs(sYg))) << "dX adjoint wrong";
	EXPECT_NEAR(sWdW, sYg, 1e-3 * (1.0 + std::abs(sYg))) << "dW wrong";
	ctx.Clear();
}

// Same bilinear grad-check for the STOCK OaConv1d (the suspect behind every conv-VQ
// NaN in OA). With zero bias, y = Conv1d(x,W) is bilinear, so
//   sum(y*g) == sum(x*x.grad) == sum(W*W.grad).  Checked at stride 1 AND stride 2.
static void ConvGradCheckImpl(OaI32 S) {
	OaFnMatrix::SetRngSeed(321);
	const OaI32 inC = 3, outC = 4, K = 3, P = 1, B = 2, L = 8;
	auto cv = OaMakeSharedPtr<OaConv1d>(inC, outC, K, S, P);
	auto& ctx = OaContext::GetDefault();
	auto x = OaFnMatrix::RandN(OaMatrixShape{B, inC, L});
	x.SetRequiresGrad(true);
	auto& W = cv->Parameters()[0].Data;
	auto y0 = cv->Forward(x);                            // realize to read output length
	(void)ctx.Execute(); (void)ctx.Sync();
	const OaI64 Lout = y0.Size(2);
	auto g = OaFnMatrix::RandN(OaMatrixShape{B, outC, Lout});

	OaGradientTape tape;
	auto y = cv->Forward(x);
	auto loss = OaFnMatrix::Sum(OaFnMatrix::Mul(y, g));
	tape.Backward(loss);
	(void)ctx.Execute(); (void)ctx.Sync();
	const double sYg = static_cast<double>(loss.At(0));
	auto dx = x.GradMatrix();  auto dW = W.GradMatrix();
	double sXdx = HostDot(x, dx);
	double sWdW = HostDot(W, dW);
	std::printf("Conv1d(stride=%d) grad-check: <y,g>=%.8f  <x,dx>=%.8f  <W,dW>=%.8f\n", S, sYg, sXdx, sWdW);
	EXPECT_NEAR(sXdx, sYg, 1e-3 * (1.0 + std::abs(sYg))) << "Conv1d stride " << S << " dX wrong";
	EXPECT_NEAR(sWdW, sYg, 1e-3 * (1.0 + std::abs(sYg))) << "Conv1d stride " << S << " dW wrong";
	ctx.Clear();
}
TEST(Alm, Conv1dGradCheckStride1) { ConvGradCheckImpl(1); }
TEST(Alm, Conv1dGradCheckStride2) { ConvGradCheckImpl(2); }

// Conv1dGemm (im2col + tensor-core matmul) gradient self-consistency at tokenizer
// shapes. The bilinear grad-check (zero bias) validates the Im2Col1d adjoint plus
// the composed reshape/transpose/Linear gradient with no finite-difference eps.
// (Forward correctness vs a CPU reference is covered by
// NN.Conv1dGemmMatchesCpuReference in TestNnKernels.)
static void Conv1dGemmParityImpl(OaI32 S) {
	OaFnMatrix::SetRngSeed(4242);
	const OaI32 B = 8, inC = 96, outC = 128, K = 3, P = 1, L = 64;
	auto& ctx = OaContext::GetDefault();

	// Isolated Im2Col1d grad-check: cols is linear in x, so <cols,g> == <x,dx>.
	{
		auto xi = OaFnMatrix::RandN(OaMatrixShape{B, inC, L});
		xi.SetRequiresGrad(true);
		OaGradientTape t2;
		auto cols = OaFnMatrix::Im2Col1d(xi, K, S, P, 1);
		auto gc = OaFnMatrix::RandN(cols.GetShape());
		auto lc = OaFnMatrix::Sum(OaFnMatrix::Mul(cols, gc));
		t2.Backward(lc);
		(void)ctx.Execute(); (void)ctx.Sync();
		const double sc = static_cast<double>(lc.At(0));
		auto dxi = xi.GradMatrix();
		double sXdxi = HostDot(xi, dxi);
		std::printf("Im2Col1d-only grad (S=%d): <cols,g>=%.6f <x,dx>=%.6f\n", S, sc, sXdxi);
		ctx.Clear();
	}

	// Bilinear grad-check (zero bias): sum(y*g) == <x,dx> == <w,dw>.
	auto xg = OaFnMatrix::RandN(OaMatrixShape{B, inC, L});
	xg.SetRequiresGrad(true);
	auto wg = OaFnMatrix::RandN(OaMatrixShape{outC, inC, K});
	wg.SetRequiresGrad(true);
	auto zb = OaFnMatrix::Zeros(OaMatrixShape{outC});

	OaGradientTape tape;
	auto y = OaFnMatrix::Conv1dGemm(xg, wg, zb, S, P, 1);
	auto g = OaFnMatrix::RandN(y.GetShape());
	auto loss = OaFnMatrix::Sum(OaFnMatrix::Mul(y, g));
	tape.Backward(loss);
	(void)ctx.Execute(); (void)ctx.Sync();

	const double sYg = static_cast<double>(loss.At(0));
	auto dx = xg.GradMatrix();
	auto dw = wg.GradMatrix();
	double sXdx = HostDot(xg, dx);
	double sWdW = HostDot(wg, dw);
	std::printf("Conv1dGemm grad-check (S=%d): <y,g>=%.6f <x,dx>=%.6f <w,dw>=%.6f\n", S, sYg, sXdx, sWdW);
	EXPECT_NEAR(sXdx, sYg, 1e-3 * (1.0 + std::abs(sYg))) << "Im2Col1d dX adjoint wrong";
	EXPECT_NEAR(sWdW, sYg, 1e-3 * (1.0 + std::abs(sYg))) << "Conv1dGemm dW wrong";
	ctx.Clear();
}
TEST(Alm, Conv1dGemmParityStride1) { Conv1dGemmParityImpl(1); }
TEST(Alm, Conv1dGemmParityStride2) { Conv1dGemmParityImpl(2); }

// Perf: scalar direct Conv1d vs im2col+GEMM, at the tokenizer's workhorse shapes.
// Reports:
//   GPU ms/fwd   — isolated GPU compute (batched N copies, one submit/sync, GPU timer)
//   wall ms/step — realistic training step (fwd + backward + sync per iter)
//   sps          — samples/sec = B / (wall ms/step / 1000), B=32
namespace {
double NowMs() {
	return std::chrono::duration<double, std::milli>(
		std::chrono::high_resolution_clock::now().time_since_epoch()).count();
}
// Batched-wall ms/iter ≈ isolated GPU compute: record N copies into one command
// buffer (OaGradNo, no grad graph), a single Execute+Sync, wall/N. With the GPU
// pipeline kept full, per-call CPU submit + sync-drain overhead is amortized to
// ~0, so this separates real compute from the per-step sync stall.
double BatchedMsPerFwd(const std::function<OaMatrix()>& InFn, OaI32 InN) {
	auto& ctx = OaContext::GetDefault();
	{
		OaGradNo nog;
		for (OaI32 i = 0; i < 5; ++i) { auto y = InFn(); (void)y; }  // warmup
		(void)ctx.Execute(); (void)ctx.Sync(); ctx.Clear();
		double t0 = NowMs();
		for (OaI32 i = 0; i < InN; ++i) { auto y = InFn(); (void)y; }
		(void)ctx.Execute(); (void)ctx.Sync();
		double per = (NowMs() - t0) / InN;
		ctx.Clear();
		return per;
	}
}
double WallMsPerStep(const std::function<OaMatrix()>& InFwd, OaI32 InN) {
	auto& ctx = OaContext::GetDefault();
	for (OaI32 i = 0; i < 5; ++i) {
		OaGradientTape tape; auto y = InFwd(); auto g = OaFnMatrix::RandN(y.GetShape());
		auto loss = OaFnMatrix::Sum(OaFnMatrix::Mul(y, g)); tape.Backward(loss);
		(void)ctx.Execute(); (void)ctx.Sync(); ctx.Clear();
	}
	double t0 = NowMs();
	for (OaI32 i = 0; i < InN; ++i) {
		OaGradientTape tape; auto y = InFwd(); auto g = OaFnMatrix::RandN(y.GetShape());
		auto loss = OaFnMatrix::Sum(OaFnMatrix::Mul(y, g)); tape.Backward(loss);
		(void)ctx.Execute(); (void)ctx.Sync(); ctx.Clear();
	}
	return (NowMs() - t0) / InN;
}
void ConvPerfImpl(const char* InTag, OaI32 inC, OaI32 outC, OaI32 K, OaI32 S, OaI32 P) {
	auto& ctx = OaContext::GetDefault();
	const OaI32 B = 32, L = 64;
	auto x = OaFnMatrix::RandN(OaMatrixShape{B, inC, L});
	auto w = OaFnMatrix::RandN(OaMatrixShape{outC, inC, K});
	auto b = OaFnMatrix::RandN(OaMatrixShape{outC});
	const int n = 50;
	auto fwdGemm = [&]{ return OaFnMatrix::Conv1dReluGemm(x, w, b, S, P, 1); };
	double gpuG = BatchedMsPerFwd(fwdGemm, n);
	double stepG = WallMsPerStep(fwdGemm, n);
	double spsG = 1000.0 * B / stepG;
	std::printf("[perf %s] inC=%d outC=%d K=%d S=%d | batched ms/fwd=%.4f | wall ms/step=%.4f | sps=%.0f\n",
		InTag, inC, outC, K, S, gpuG, stepG, spsG);
	(void)ctx;
}
}
TEST(Alm, Conv1dGemmPerf) {
	const char* prec = (OaEngine::GetGlobal() && OaEngine::GetGlobal()->GetPrecision() == OaPrecision::BF16) ? "BF16" : "FP32";
	std::printf("[perf] engine precision = %s\n", prec);
	ConvPerfImpl("W384-K3", 384, 384, 3, 1, 1);   // res-block workhorse (x12 fwd)
	ConvPerfImpl("W384-K4S2", 384, 384, 4, 2, 1); // strided down-conv (per stage)
	ConvPerfImpl("in263-W384", 263, 384, 3, 1, 1); // enc_in
}

// Full-tokenizer-step perf: scalar vs GEMM at training config shapes.
// Measures the real end-to-end sps win from wiring Conv1dGemm into the tokenizer.
// One training step = Encode + Quantize + Decode + Mse + backward + optimizer step.
// The conv kernels dominate; the GEMM path routes them through the tiled matmul
// stack instead of the scalar direct-conv loop.
TEST(Alm, TokenizerStepPerfGemm) {
	OaFnMatrix::SetRngSeed(42);
	auto& ctx = OaContext::GetDefault();
	const OaI32 B = 8, T = 64;
	const OaI32 InDim = 263;

	auto runSteps = [&](OaI32 NumSteps) -> double {
		OaAlmTokenizerConfig cfg;
		cfg.InputDim = InDim;
		cfg.Width = 384;
		cfg.CodeDim = 384;
		cfg.NumCodes = 128;
		cfg.DownT = 2;
		cfg.Depth = 2;
		auto tok = OaMakeSharedPtr<OaAlmTokenizerAg>(cfg);
		const OaI32 tokLen = T / tok->DownsampleFactor();

		// Synthetic batch.
		std::vector<float> xh(static_cast<size_t>(B) * T * InDim);
		OaU64 rng = 0xDEADULL;
		for (auto& v : xh) {
			rng = (rng * 6364136223846793005ULL) + 1;
			v = static_cast<float>(static_cast<OaU32>(rng >> 40)) / static_cast<float>(1 << 24);
		}
		auto X = OaFnMatrix::FromBytes(
			OaSpan<const OaU8>(reinterpret_cast<const OaU8*>(xh.data()), xh.size() * sizeof(float)),
			OaMatrixShape{B, T, InDim}, OaScalarType::Float32);
		auto xFlat = X.Reshape(OaMatrixShape{static_cast<OaI64>(B) * T, InDim});

		// Seed codebook.
		{ auto z0 = tok->Encode(X, B, T); tok->Seed(z0); ctx.Clear(); }

		auto params = tok->AllParameterPtrs();
		auto opt = OaMakeUniquePtr<OaAdamW>(params, 1e-4F, 0.9F, 0.99F, 1e-8F, 0.01F);

		// Warmup (3 steps).
		for (OaI32 s = 0; s < 3; ++s) {
			ctx.Clear();
			OaGradientTape tape; opt->ZeroGrad();
			auto z = tok->Encode(X, B, T);
			auto q = tok->Quantize(z);
			auto rec = tok->Decode(q.Quantized, B, tokLen);
			auto loss = OaFnLoss::Mse(rec, xFlat) + q.CommitLoss;
			tape.Backward(loss);
			opt->Step();
			tok->EmaUpdate(q);
			(void)ctx.Execute(); (void)ctx.Sync();
		}

		// Timed.
		double t0 = NowMs();
		for (OaI32 s = 0; s < NumSteps; ++s) {
			ctx.Clear();
			OaGradientTape tape; opt->ZeroGrad();
			auto z = tok->Encode(X, B, T);
			auto q = tok->Quantize(z);
			auto rec = tok->Decode(q.Quantized, B, tokLen);
			auto loss = OaFnLoss::Mse(rec, xFlat) + q.CommitLoss;
			tape.Backward(loss);
			opt->Step();
			tok->EmaUpdate(q);
			(void)ctx.Execute(); (void)ctx.Sync();
		}
		double total = NowMs() - t0;
		opt.Reset();
		tok.Reset();
		ctx.Clear();
		return total / NumSteps;
	};

	const OaI32 steps = 10;
	double msGemm = runSteps(steps);
	double spsGemm = 1000.0 * B / msGemm;
	std::printf("[tokenizer-step-perf] B=%d T=%d W=384 DownT=2 Depth=2\n"
	            "  gemm: %.2f ms/step  sps=%.0f\n",
	            B, T, msGemm, spsGemm);
}

// Most atomic isolation: a SINGLE Conv1d trained to the identity (target == input).
// Gradients are verified correct, so SGD at a small lr MUST reduce MSE. If it climbs,
// the bug is the conv weight update / optimizer step on 3D conv weights — not the graph.
TEST(Alm, SingleConvIdentity) {
	OaFnMatrix::SetRngSeed(4);
	const OaI32 C = 8, L = 16, B = 16;
	auto cv = OaMakeSharedPtr<OaConv1d>(C, C, 3, 1, 1);
	auto& ctx = OaContext::GetDefault();
	auto X = OaFnMatrix::RandN(OaMatrixShape{B, C, L});
	(void)ctx.Execute(); (void)ctx.Sync();   // realize X before the loop
	auto params = cv->AllParameterPtrs();
	auto opt = OaMakeUniquePtr<OaAdamW>(params, 0.01F);
	OaF32 first = 0.0F, last = 0.0F;
	for (OaI32 s = 1; s <= 60; ++s) {
		ctx.Clear();
		OaGradientTape tape; opt->ZeroGrad();
		auto y  = cv->Forward(X);
		auto y2 = y.Reshape(OaMatrixShape{B, static_cast<OaI64>(C) * L});
		auto x2 = X.Reshape(OaMatrixShape{B, static_cast<OaI64>(C) * L});
		auto loss = OaFnLoss::Mse(y2, x2);
		tape.Backward(loss);
		opt->Step();
		(void)ctx.Execute(); (void)ctx.Sync();
		const float lv = loss.At(0);
		if (s == 1) {
			auto y2H = HostFloatData(y2);
			auto x2H = HostFloatData(x2);
			double sy = 0.0, sx = 0.0, sd = 0.0;
			const OaI64 n = y2.NumElements();
			for (OaI64 i = 0; i < n; ++i) {
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
	ctx.Clear();
}

TEST(Alm, SingleConvTransposeIdentity) {
	OaFnMatrix::SetRngSeed(7);
	const OaI32 C = 8, Lin = 16, B = 16;
	const OaI32 K = 3, S = 1, P = 1;
	const OaI32 Lout = (Lin - 1) * S - 2 * P + K;  // = 16
	auto ct = OaMakeSharedPtr<OaConvTranspose1d>(C, C, K, S, P);
	auto& ctx = OaContext::GetDefault();
	auto X = OaFnMatrix::RandN(OaMatrixShape{B, C, Lin});
	(void)ctx.Execute(); (void)ctx.Sync();
	auto params = ct->AllParameterPtrs();
	auto opt = OaMakeUniquePtr<OaAdamW>(params, 0.01F);
	OaF32 first = 0.0F, last = 0.0F;
	for (OaI32 s = 1; s <= 60; ++s) {
		ctx.Clear();
		OaGradientTape tape; opt->ZeroGrad();
		auto y  = ct->Forward(X);
		auto y2 = y.Reshape(OaMatrixShape{B, static_cast<OaI64>(C) * Lout});
		auto x2 = X.Reshape(OaMatrixShape{B, static_cast<OaI64>(C) * Lin});
		auto loss = OaFnLoss::Mse(y2, x2);
		tape.Backward(loss);
		opt->Step();
		(void)ctx.Execute(); (void)ctx.Sync();
		const float lv = loss.At(0);
		if (s == 1) first = lv;
		last = lv;
		if (s == 1 || s % 15 == 0) std::printf("  [1convt] step %2d | mse %.8f\n", s, static_cast<double>(lv));
	}
	EXPECT_LT(last, first) << "single ConvTranspose1d must learn identity";
	ctx.Clear();
}

TEST(Alm, ConvAutoEncoderIdentity) {
	OaFnMatrix::SetRngSeed(11);
	const OaI32 C = 8, L = 16, B = 16;
	const OaI32 K = 3, S = 1, P = 1;
	auto enc = OaMakeSharedPtr<OaConv1d>(C, C, K, S, P);
	auto dec = OaMakeSharedPtr<OaConvTranspose1d>(C, C, K, S, P);
	auto& ctx = OaContext::GetDefault();
	auto X = OaFnMatrix::RandN(OaMatrixShape{B, C, L});
	(void)ctx.Execute(); (void)ctx.Sync();
	OaVec<OaParameter*> params;
	auto ep = enc->AllParameterPtrs();
	auto dp = dec->AllParameterPtrs();
	for (auto* p : ep) params.PushBack(p);
	for (auto* p : dp) params.PushBack(p);
	auto opt = OaMakeUniquePtr<OaAdamW>(params, 0.01F);
	OaF32 first = 0.0F, last = 0.0F;
	for (OaI32 s = 1; s <= 60; ++s) {
		ctx.Clear();
		OaGradientTape tape; opt->ZeroGrad();
		auto z   = enc->Forward(X);
		auto y   = dec->Forward(z);
		auto y2  = y.Reshape(OaMatrixShape{B, static_cast<OaI64>(C) * L});
		auto x2  = X.Reshape(OaMatrixShape{B, static_cast<OaI64>(C) * L});
		auto loss = OaFnLoss::Mse(y2, x2);
		tape.Backward(loss);
		opt->Step();
		(void)ctx.Execute(); (void)ctx.Sync();
		const float lv = loss.At(0);
		if (s == 1) first = lv;
		last = lv;
		if (s == 1 || s % 15 == 0) std::printf("  [convae] step %2d | mse %.8f\n", s, static_cast<double>(lv));
	}
	EXPECT_LT(last, first) << "Conv1d→ConvTranspose1d autoencoder must learn identity";
	ctx.Clear();
}

TEST(Alm, ConvAutoEncoderStride2) {
	OaFnMatrix::SetRngSeed(13);
	const OaI32 C = 8, L = 16, B = 16;
	const OaI32 K = 4, S = 2, P = 1;
	auto enc = OaMakeSharedPtr<OaConv1d>(C, C, K, S, P);
	auto dec = OaMakeSharedPtr<OaConvTranspose1d>(C, C, K, S, P);
	auto& ctx = OaContext::GetDefault();
	auto X = OaFnMatrix::RandN(OaMatrixShape{B, C, L});
	(void)ctx.Execute(); (void)ctx.Sync();
	OaVec<OaParameter*> params;
	auto ep = enc->AllParameterPtrs();
	auto dp = dec->AllParameterPtrs();
	for (auto* p : ep) params.PushBack(p);
	for (auto* p : dp) params.PushBack(p);
	auto opt = OaMakeUniquePtr<OaAdamW>(params, 0.01F);
	OaF32 first = 0.0F, last = 0.0F;
	for (OaI32 s = 1; s <= 60; ++s) {
		ctx.Clear();
		OaGradientTape tape; opt->ZeroGrad();
		auto z   = enc->Forward(X);
		auto y   = dec->Forward(z);
		auto y2  = y.Reshape(OaMatrixShape{B, static_cast<OaI64>(C) * L});
		auto x2  = X.Reshape(OaMatrixShape{B, static_cast<OaI64>(C) * L});
		auto loss = OaFnLoss::Mse(y2, x2);
		tape.Backward(loss);
		opt->Step();
		(void)ctx.Execute(); (void)ctx.Sync();
		const float lv = loss.At(0);
		if (s == 1) first = lv;
		last = lv;
		if (s == 1 || s % 15 == 0) std::printf("  [convae2] step %2d | mse %.8f\n", s, static_cast<double>(lv));
	}
	EXPECT_LT(last, first) << "Conv1d→ConvTranspose1d stride-2 autoencoder must learn identity";
	ctx.Clear();
}

TEST(Alm, DeepConvAutoEncoder) {
	OaFnMatrix::SetRngSeed(17);
	const OaI32 InC = 3, W = 8, CodeC = 4, L = 16, B = 16;
	auto& ctx = OaContext::GetDefault();
	auto X = OaFnMatrix::RandN(OaMatrixShape{B, InC, L});
	(void)ctx.Execute(); (void)ctx.Sync();
	OaVec<OaParameter*> params;
	auto collect = [&](const OaSharedPtr<OaModule>& m) {
		for (auto* p : m->AllParameterPtrs()) params.PushBack(p);
	};
	auto encIn  = OaMakeSharedPtr<OaConv1d>(InC, W, 3, 1, 1);   collect(encIn);
	auto encDown = OaMakeSharedPtr<OaConv1d>(W, W, 4, 2, 1);     collect(encDown);
	auto encOut  = OaMakeSharedPtr<OaConv1d>(W, CodeC, 3, 1, 1); collect(encOut);
	auto decIn   = OaMakeSharedPtr<OaConv1d>(CodeC, W, 3, 1, 1); collect(decIn);
	auto decUp   = OaMakeSharedPtr<OaConvTranspose1d>(W, W, 4, 2, 1); collect(decUp);
	auto decOut  = OaMakeSharedPtr<OaConv1d>(W, InC, 3, 1, 1);   collect(decOut);
	auto opt = OaMakeUniquePtr<OaAdamW>(params, 0.01F);
	OaF32 first = 0.0F, last = 0.0F;
	for (OaI32 s = 1; s <= 60; ++s) {
		ctx.Clear();
		OaGradientTape tape; opt->ZeroGrad();
		auto h = OaFnMatrix::Relu(encIn->Forward(X));
		h = OaFnMatrix::Relu(encDown->Forward(h));
		auto z = encOut->Forward(h);
		auto h2 = OaFnMatrix::Relu(decIn->Forward(z));
		h2 = OaFnMatrix::Relu(decUp->Forward(h2));
		auto y = decOut->Forward(h2);
		auto y2 = y.Reshape(OaMatrixShape{B, static_cast<OaI64>(InC) * L});
		auto x2 = X.Reshape(OaMatrixShape{B, static_cast<OaI64>(InC) * L});
		auto loss = OaFnLoss::Mse(y2, x2);
		tape.Backward(loss);
		opt->Step();
		(void)ctx.Execute(); (void)ctx.Sync();
		const float lv = loss.At(0);
		if (s == 1) first = lv;
		last = lv;
		if (s == 1 || s % 15 == 0) std::printf("  [deepae] step %2d | mse %.8f\n", s, static_cast<double>(lv));
	}
	EXPECT_LT(last, first) << "deep conv autoencoder must learn identity";
	ctx.Clear();
}

TEST(Alm, DeepConvAutoEncoderNorm) {
	OaFnMatrix::SetRngSeed(17);
	const OaI32 InC = 3, W = 8, CodeC = 4, L = 16, B = 16;
	auto& ctx = OaContext::GetDefault();
	auto X = OaFnMatrix::RandN(OaMatrixShape{B, InC, L});
	(void)ctx.Execute(); (void)ctx.Sync();
	OaVec<OaParameter*> params;
	auto collect = [&](const OaSharedPtr<OaModule>& m) {
		for (auto* p : m->AllParameterPtrs()) params.PushBack(p);
	};
	auto encIn  = OaMakeSharedPtr<OaConv1d>(InC, W, 3, 1, 1);   collect(encIn);
	auto encDown = OaMakeSharedPtr<OaConv1d>(W, W, 4, 2, 1);     collect(encDown);
	auto encOut  = OaMakeSharedPtr<OaConv1d>(W, CodeC, 3, 1, 1); collect(encOut);
	auto decIn   = OaMakeSharedPtr<OaConv1d>(CodeC, W, 3, 1, 1); collect(decIn);
	auto decUp   = OaMakeSharedPtr<OaConvTranspose1d>(W, W, 4, 2, 1); collect(decUp);
	auto decOut  = OaMakeSharedPtr<OaConv1d>(W, InC, 3, 1, 1);   collect(decOut);
	auto ln1 = OaMakeSharedPtr<OaLayerNorm>(W); collect(ln1);
	auto ln2 = OaMakeSharedPtr<OaLayerNorm>(W); collect(ln2);
	auto ln3 = OaMakeSharedPtr<OaLayerNorm>(W); collect(ln3);
	auto ln4 = OaMakeSharedPtr<OaLayerNorm>(W); collect(ln4);
	auto normC = [&](const OaSharedPtr<OaLayerNorm>& ln, const OaMatrix& h) -> OaMatrix {
		auto t = OaFnMatrix::Transpose(h, 1, 2);  // [B, T, C]
		auto n = ln->Forward(t);
		return OaFnMatrix::Transpose(n, 1, 2);    // [B, C, T]
	};
	auto opt = OaMakeUniquePtr<OaAdamW>(params, 0.01F);
	OaF32 first = 0.0F, last = 0.0F;
	for (OaI32 s = 1; s <= 60; ++s) {
		ctx.Clear();
		OaGradientTape tape; opt->ZeroGrad();
		auto h = OaFnMatrix::Relu(normC(ln1, encIn->Forward(X)));
		h = OaFnMatrix::Relu(normC(ln2, encDown->Forward(h)));
		auto z = encOut->Forward(h);
		auto h2 = OaFnMatrix::Relu(normC(ln3, decIn->Forward(z)));
		h2 = OaFnMatrix::Relu(normC(ln4, decUp->Forward(h2)));
		auto y = decOut->Forward(h2);
		auto y2 = y.Reshape(OaMatrixShape{B, static_cast<OaI64>(InC) * L});
		auto x2 = X.Reshape(OaMatrixShape{B, static_cast<OaI64>(InC) * L});
		auto loss = OaFnLoss::Mse(y2, x2);
		tape.Backward(loss);
		opt->Step();
		(void)ctx.Execute(); (void)ctx.Sync();
		const float lv = loss.At(0);
		if (s == 1) first = lv;
		last = lv;
		if (s == 1 || s % 15 == 0) std::printf("  [deepaen] step %2d | mse %.8f\n", s, static_cast<double>(lv));
	}
	EXPECT_LT(last, first) << "deep conv autoencoder with LayerNorm must learn identity";
	ctx.Clear();
}
// inconsistency; if this also climbs, the loop/loss/optimizer harness is the bug.
TEST(Alm, LinearAeSanity) {
	OaFnMatrix::SetRngSeed(11);
	const OaI32 D = 48, H = 32, B = 64;
	auto enc = OaMakeSharedPtr<OaLinear>(D, H);
	auto dec = OaMakeSharedPtr<OaLinear>(H, D);
	auto& ctx = OaContext::GetDefault();
	std::vector<float> xh(static_cast<size_t>(B) * D);
	{ OaU64 r = 5; for (auto& v : xh) { r = (r * 6364136223846793005ULL) + 1; v = std::sin(0.01F * static_cast<float>(static_cast<OaU32>(r >> 40))); } }
	auto X = OaFnMatrix::FromBytes(OaSpan<const OaU8>(reinterpret_cast<const OaU8*>(xh.data()), xh.size() * sizeof(float)),
		OaMatrixShape{B, D}, OaScalarType::Float32);
	OaVec<OaParameter*> params;
	for (auto* p : enc->AllParameterPtrs()) params.PushBack(p);
	for (auto* p : dec->AllParameterPtrs()) params.PushBack(p);
	auto opt = OaMakeUniquePtr<OaAdamW>(params, 1e-3F);
	OaF32 first = 0.0F, last = 0.0F;
	for (OaI32 s = 1; s <= 80; ++s) {
		ctx.Clear();
		OaGradientTape tape; opt->ZeroGrad();
		auto rec = dec->Forward(enc->Forward(X));
		auto loss = OaFnLoss::Mse(rec, X);
		tape.Backward(loss);
		opt->Step();
		(void)ctx.Execute(); (void)ctx.Sync();
		const float lv = loss.At(0);
		if (s == 1) first = lv;
		last = lv;
		if (s == 1 || s % 20 == 0) std::printf("  [linAE] step %2d | mse %.8f\n", s, static_cast<double>(lv));
	}
	EXPECT_LT(last, first) << "linear AE harness must descend";
	ctx.Clear();
}

// DECISIVE root-cause tool: is the COMPOSED end-to-end gradient a descent direction?
// Per-op bilinear checks pass, but the full encode→decode→MSE graph could compose a
// wrong gradient. Line-search: from a fixed init compute loss L0 + grads, then for a
// range of step sizes h evaluate L(W − h·grad) (forward only). If NO h gives L < L0,
// the composed gradient is NOT downhill → a composition/accumulation bug in the conv
// graph (transpose/reshape/rmsnorm/convT grad), NOT mere conditioning.
TEST(Alm, ComposedDescentCheck) {
	OaFnMatrix::SetRngSeed(7);
	OaAlmTokenizerConfig cfg;
	cfg.InputDim = 48; cfg.Width = 64; cfg.CodeDim = 32; cfg.NumCodes = 64;
	cfg.DownT = 1; cfg.Depth = 0;                       // minimal: no res blocks, factor 2
	auto tok = OaMakeSharedPtr<OaAlmTokenizerAg>(cfg);
	auto& ctx = OaContext::GetDefault();
	const OaI32 B = 8, Tw = 16;
	std::vector<float> xh(static_cast<size_t>(B) * Tw * cfg.InputDim);
	for (OaI32 b = 0; b < B; ++b) for (OaI32 t = 0; t < Tw; ++t) for (OaI32 c = 0; c < cfg.InputDim; ++c)
		xh[(static_cast<size_t>(b) * Tw + t) * cfg.InputDim + c] = std::sin(0.2F * static_cast<float>(t) + 0.3F * static_cast<float>(c));
	auto X = OaFnMatrix::FromBytes(OaSpan<const OaU8>(reinterpret_cast<const OaU8*>(xh.data()), xh.size() * sizeof(float)),
		OaMatrixShape{B, Tw, cfg.InputDim}, OaScalarType::Float32);
	auto Xflat = X.Reshape(OaMatrixShape{static_cast<OaI64>(B) * Tw, cfg.InputDim});
	const OaI32 fac = tok->DownsampleFactor();
	auto params = tok->AllParameterPtrs();

	// L0 + grads (bypass VQ to isolate the pure conv AE).
	ctx.Clear();
	OaGradientTape tape;
	auto rec0 = tok->Decode(tok->Encode(X, B, Tw), B, Tw / fac);
	auto loss0 = OaFnLoss::Mse(rec0, Xflat);
	tape.Backward(loss0);
	(void)ctx.Execute(); (void)ctx.Sync();
	const double L0 = static_cast<double>(loss0.At(0));
	std::printf("ComposedDescent: L0 = %.8f, params=%lld\n", L0, static_cast<long long>(params.Size()));

	OaVec<OaMatrix> W0, G;
	for (auto* p : params) { W0.PushBack(p->Data.Clone()); G.PushBack(p->Data.GradMatrix().Clone()); }
	(void)ctx.Execute(); (void)ctx.Sync();

	bool anyDown = false;
	const double hs[] = {1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8};
	for (double h : hs) {
		for (OaI64 i = 0; i < params.Size(); ++i) {
			auto Wn = OaFnMatrix::Add(W0[i], OaFnMatrix::Scale(G[i], static_cast<OaF32>(-h)));
			params[i]->Data.CopyFrom(Wn);
		}
		(void)ctx.Execute(); (void)ctx.Sync();
		ctx.Clear();
		auto rec = tok->Decode(tok->Encode(X, B, Tw), B, Tw / fac);
		auto l = OaFnLoss::Mse(rec, Xflat);
		(void)ctx.Execute(); (void)ctx.Sync();
		const double Lh = static_cast<double>(l.At(0));
		std::printf("  h=%.0e  L=%.8f  %s\n", h, Lh, Lh < L0 ? "DOWN ✓" : "up");
		if (Lh < L0) anyDown = true;
		for (OaI64 i = 0; i < params.Size(); ++i) params[i]->Data.CopyFrom(W0[i]);
		(void)ctx.Execute(); (void)ctx.Sync();
	}
	EXPECT_TRUE(anyDown) << "NO step size reduced loss → composed gradient is not a descent direction";
	ctx.Clear();
}

// The NaN-fix / learns-to-reconstruct check (todo 2): a short synchronous training loop
// on a fixed smooth synthetic batch must drive recon MSE down while staying finite.
// Defaults to Depth=0, DownT=1 (minimal stable conv VQ-VAE) because the full T2M-GPT
// depth needs much smaller lr and longer tuning; env OA_MG_DEPTH / OA_MG_DOWNT override.
TEST(Alm, TokenizerLearnsRecon) {
	OaFnMatrix::SetRngSeed(7);
	auto envI0 = [](const char* n, OaI32 d) { const char* e = std::getenv(n); return (e && *e) ? static_cast<OaI32>(std::atoi(e)) : d; };
	OaAlmTokenizerConfig cfg;
	cfg.InputDim = 48; cfg.Width = 96; cfg.CodeDim = 32; cfg.NumCodes = 64;
	cfg.DownT = envI0("OA_MG_DOWNT", 1); cfg.Depth = envI0("OA_MG_DEPTH", 0); cfg.CommitBeta = 0.25F;
	auto tok = OaMakeSharedPtr<OaAlmTokenizerAg>(cfg);

	// Window length chosen so one batch seeds the codebook (B·Tw/Factor ≥ NumCodes).
	const OaI32 B  = 8;
	const OaI32 Tw = 64;                                // 8·64/2 = 256 tokens >= NumCodes
	auto& ctx = OaContext::GetDefault();

	// Env knobs for fast diagnosis without rebuilds.
	auto envF = [](const char* n, OaF32 d) { const char* e = std::getenv(n); return (e && *e) ? static_cast<OaF32>(std::atof(e)) : d; };
	auto envI = [](const char* n, OaI32 d) { const char* e = std::getenv(n); return (e && *e) ? static_cast<OaI32>(std::atoi(e)) : d; };
	// Diverse synthetic DATASET (not one repeated batch): dsFrames of multi-frequency,
	// channel-varying content → a rich, learnable, non-degenerate manifold. Fresh random
	// mini-batch each step, matching the real tokenizer's window-sampling contract.
	// OA_MG_FRESH=0 reverts to the old single-fixed-batch repro.
	const bool  fresh = envI("OA_MG_FRESH", 0) != 0;
	const OaI32 dsFrames = 2048;
	std::vector<float> ds(static_cast<size_t>(dsFrames) * cfg.InputDim);
	for (OaI32 t = 0; t < dsFrames; ++t)
		for (OaI32 c = 0; c < cfg.InputDim; ++c) {
			const float f1 = 0.04F + 0.011F * static_cast<float>(c % 7);
			const float f2 = 0.13F + 0.007F * static_cast<float>(c % 5);
			ds[(static_cast<size_t>(t) * cfg.InputDim) + c] =
				(0.6F * std::sin((f1 * static_cast<float>(t)) + (0.3F * static_cast<float>(c))))
				+ (0.4F * std::sin((f2 * static_cast<float>(t)) + (0.7F * static_cast<float>(c))));
		}
	OaU64 rng = 0x1234ABCDULL;
	auto sample = [&](OaMatrix& OutX, OaMatrix& OutXflat) {
		std::vector<float> hb(static_cast<size_t>(B) * Tw * cfg.InputDim);
		for (OaI32 b = 0; b < B; ++b) {
			rng = (rng * 6364136223846793005ULL) + 1442695040888963407ULL;
			const OaI32 start = static_cast<OaI32>((rng >> 33) % static_cast<OaU64>(dsFrames - Tw));
			for (OaI32 t = 0; t < Tw; ++t)
				for (OaI32 c = 0; c < cfg.InputDim; ++c)
					hb[((static_cast<size_t>(b) * Tw + t) * cfg.InputDim) + c] =
						ds[(static_cast<size_t>(start + t) * cfg.InputDim) + c];
		}
		OutX = OaFnMatrix::FromBytes(OaSpan<const OaU8>(reinterpret_cast<const OaU8*>(hb.data()), hb.size() * sizeof(float)),
			OaMatrixShape{B, Tw, cfg.InputDim}, OaScalarType::Float32);
		OutXflat = OutX.Reshape(OaMatrixShape{static_cast<OaI64>(B) * Tw, cfg.InputDim});
	};
	OaMatrix X, Xflat;
	sample(X, Xflat);

	// Seed codebook from one warm batch of latents (B·Tw/Factor >= NumCodes).
	{ auto z0 = tok->Encode(X, B, Tw); tok->Seed(z0); ctx.Clear(); }

	const OaF32 lr     = envF("OA_MG_LR", 1e-6F);
	const OaI32 steps  = envI("OA_MG_STEPS", 300);
	const bool  bypass = envI("OA_MG_BYPASS", 0) != 0;

	auto params = tok->AllParameterPtrs();
	std::printf("Optimizer params: %lld tensors\n", static_cast<long long>(params.Size()));
	OaUniquePtr<OaOptimizer> opt = OaMakeUniquePtr<OaAdamW>(params, lr, 0.9F, 0.999F, 1e-8F, 0.01F);
	OaF32 first = 0.0F;
	OaF32 last  = 0.0F;
	for (OaI32 s = 1; s <= steps; ++s) {
		ctx.Clear();
		if (fresh) sample(X, Xflat);                         // fresh mini-batch each step
		OaGradientTape tape; opt->ZeroGrad();
		auto z   = tok->Encode(X, B, Tw);
		OaResidualVqResult q;
		OaMatrix zq;
		if (bypass) { zq = z; }
		else        { q = tok->Quantize(z); zq = q.Quantized; }
		auto rec = tok->Decode(zq, B, Tw / tok->DownsampleFactor());
		auto recon = OaFnLoss::Mse(rec, Xflat);
		auto loss  = bypass ? recon : (recon + q.CommitLoss);
		tape.Backward(loss);
		opt->Step();
		if (!bypass) tok->EmaUpdate(q);
		(void)ctx.Execute(); (void)ctx.Sync();
		const float lv = recon.At(0);
		if (s == 1) first = lv;
		last = lv;
		if (s == 1 || s % 10 == 0 || s == steps)
			std::printf("  [tok] step %3d | recon %.8f | commit %.8f\n", s, static_cast<double>(lv),
				static_cast<double>(bypass ? 0.0F : q.CommitLoss.At(0)));
		ASSERT_TRUE(std::isfinite(lv)) << "diverged at step " << s;
	}
	std::printf("Tokenizer recon: %.8f -> %.8f\n", static_cast<double>(first), static_cast<double>(last));
	EXPECT_LT(last, first) << "tokenizer did not learn to reconstruct";
	ctx.Clear();
}

TEST(Alm, LmStub) {
	OaAlmPriorConfig cfg;
	cfg.DModel = 256;
	cfg.NumLayers = 2;  // Small for testing
	cfg.VocabSize = 515;
	
	auto lm = OaMakeSharedPtr<OaAlmPriorAg>(cfg);
	std::printf("LM created (stub)\n");
	
	// TODO: Test forward/generate when implemented
	EXPECT_TRUE(lm != nullptr);
	EXPECT_EQ(lm->GetConfig().VocabSize, 515);
}

TEST(Alm, GenerateStub) {
	OaAlmTokenizerConfig tokCfg;
	OaAlmPriorConfig lmCfg;
	lmCfg.SyncVocab(tokCfg.NumCodes);
	
	auto tokenizer = OaMakeSharedPtr<OaAlmTokenizerAg>(tokCfg);
	auto lm = OaMakeSharedPtr<OaAlmPriorAg>(lmCfg);
	
	std::printf("Tokenizer and LM created\n");
	
	// Generate tokens unconditionally (starts from [SOM]).
	OaMatrix generatedTokens = lm->Generate(1, 1.0F, 0, 0.0F, 16);
	std::printf("Generated tokens shape: [%lld, %lld]\n",
		static_cast<long long>(generatedTokens.Size(0)), static_cast<long long>(generatedTokens.Size(1)));
	
	// Decode to motion.
	OaMatrix motion = lm->DecodeToMotion(generatedTokens, *tokenizer);
	std::printf("Decoded motion shape: [%lld, %lld]\n",
		static_cast<long long>(motion.Size(0)), static_cast<long long>(motion.Size(1)));
	
	// Minimal USD export: treat motion as per-frame per-joint translations.
	// This is a placeholder skeleton; a real rig uses the canonical 272-dim pose.
	const OaI32 frames = static_cast<OaI32>(motion.Size(0));
	const OaI32 inputDim = static_cast<OaI32>(motion.Size(1));
	const OaI32 joints = inputDim / 3;
	if (frames > 0 and joints > 0) {
		auto motionHost = HostFloatData(motion);
		const OaF32* m = motionHost.Data();
		OaUsdSkelClip clip;
		clip.FrameCount = frames;
		clip.Fps = 30.0F;
		clip.UpAxis = 2;
		clip.JointPaths.Reserve(joints);
		clip.BindTransforms.Reserve(joints);
		clip.RestTransforms.Reserve(joints);
		OaString path = "root";
		for (OaI32 j = 0; j < joints; ++j) {
			clip.JointPaths.PushBack(path);
			clip.BindTransforms.PushBack(VlmMat4::Identity());
			clip.RestTransforms.PushBack(VlmMat4::Identity());
			char buf[32];
			std::snprintf(buf, sizeof(buf), "/j%d", j);
			path = path + buf;
		}
		clip.Translations.Reserve(static_cast<OaI64>(frames) * joints);
		clip.Rotations.Reserve(static_cast<OaI64>(frames) * joints);
		for (OaI32 f = 0; f < frames; ++f) {
			for (OaI32 j = 0; j < joints; ++j) {
				const OaI64 base = static_cast<OaI64>(f) * inputDim + static_cast<OaI64>(j * 3);
				clip.Translations.PushBack({.X = m[base], .Y = m[base + 1], .Z = m[base + 2]});
				clip.Rotations.PushBack({.X = 0.0F, .Y = 0.0F, .Z = 0.0F, .W = 1.0F});
			}
		}
		OaPath usdPath("var/alm/Alm_GenerateStub.usda");
		(void)OaFileIo::CreateDirectories(OaFileIo::GetParent(usdPath));
		auto st = OaUsd::WriteUsda(usdPath, clip, "rig");
		std::printf("USD export: %s\n", st.IsOk() ? "ok" : st.ToString().CStr());
		EXPECT_TRUE(st.IsOk());
	}
	
	EXPECT_TRUE(lm != nullptr);
	EXPECT_TRUE(tokenizer != nullptr);
}

TEST(Alm, LmLearnsNextToken) {
	OaFnMatrix::SetRngSeed(11);
	auto& ctx = OaContext::GetDefault();

	// Stage 1: minimal stable tokenizer.
	OaAlmTokenizerConfig tokCfg;
	tokCfg.InputDim = 48; tokCfg.Width = 96; tokCfg.CodeDim = 32; tokCfg.NumCodes = 64;
	tokCfg.DownT = 1; tokCfg.Depth = 0; tokCfg.CommitBeta = 0.25F;
	auto tok = OaMakeSharedPtr<OaAlmTokenizerAg>(tokCfg);

	OaAlmPriorConfig lmCfg;
	lmCfg.SyncVocab(tokCfg.NumCodes);

	const OaI32 B  = 4;   // batched sequences; block-diagonal mask keeps each sequence causal
	const OaI32 Tw = 128; // B*Tw/Factor must be >= NumCodes for VQ seed
	const OaI32 tokLen = Tw / tok->DownsampleFactor();   // 64
	std::vector<float> ds(static_cast<size_t>(B * Tw * tokCfg.InputDim));
	auto sampleDs = [&]() {
		for (OaI32 b = 0; b < B; ++b) {
			for (OaI32 t = 0; t < Tw; ++t) {
				const float tt = static_cast<float>(t) / static_cast<float>(Tw);
				for (OaI32 c = 0; c < tokCfg.InputDim; ++c) {
					const float freq = 1.0F + static_cast<float>(c % 8) * 0.5F;
					const float phase = static_cast<float>(b) * 0.3F + static_cast<float>(c) * 0.1F;
					const float start = static_cast<float>(b) * 0.7F + static_cast<float>(c % 3) * 0.2F;
					ds[((static_cast<size_t>(b) * Tw + t) * tokCfg.InputDim) + c] =
						start + std::sin(6.2831853F * (freq * tt + phase)) * 0.5F;
				}
			}
		}
	};
	sampleDs();
	OaMatrix X = OaFnMatrix::FromBytes(
		OaSpan<const OaU8>(reinterpret_cast<const OaU8*>(ds.data()), ds.size() * sizeof(float)),
		OaMatrixShape{B, Tw, tokCfg.InputDim}, OaScalarType::Float32);
	{ auto z0 = tok->Encode(X, B, Tw); tok->Seed(z0); ctx.Clear(); }

	auto tokParams = tok->AllParameterPtrs();
	auto tokOpt = OaMakeUniquePtr<OaAdamW>(tokParams, 1e-6F, 0.9F, 0.999F, 1e-8F, 0.01F);
	for (OaI32 s = 1; s <= 100; ++s) {
		ctx.Clear();
		OaGradientTape tape; tokOpt->ZeroGrad();
		auto z = tok->Encode(X, B, Tw);
		auto q = tok->Quantize(z);
		auto rec = tok->Decode(q.Quantized, B, tokLen);
		auto recon = OaFnLoss::L1(rec, X.Reshape(OaMatrixShape{static_cast<OaI64>(B) * Tw, tokCfg.InputDim}));
		auto loss = recon + q.CommitLoss;
		tape.Backward(loss);
		tokOpt->Step();
		tok->EmaUpdate(q);
		(void)ctx.Execute(); (void)ctx.Sync();
	}
	ctx.Clear();

	// Tokenize the fixed dataset: [B, tokLen].
	OaMatrix tokenIds = tok->Tokenize(X, B, Tw)[0].Reshape(OaMatrixShape{B, tokLen});
	(void)ctx.Execute(); (void)ctx.Sync();
	const OaI32* ids = tokenIds.DataAs<const OaI32>();

	// Build LM inputs [SOM, c0, ..., cN] and targets [c0, ..., cN, EOM].
	std::vector<OaI32> inputHost(static_cast<size_t>(B) * (tokLen + 1));
	std::vector<OaI32> targetHost(static_cast<size_t>(B) * (tokLen + 1));
	for (OaI32 b = 0; b < B; ++b) {
		const size_t inRow = static_cast<size_t>(b) * (tokLen + 1);
		const size_t outRow = static_cast<size_t>(b) * tokLen;
		inputHost[inRow] = lmCfg.SomToken;
		for (OaI32 t = 0; t < tokLen; ++t) {
			OaI32 code = ids[outRow + t];
			inputHost[inRow + 1 + t] = code;
			targetHost[inRow + t] = code;
		}
		targetHost[inRow + tokLen] = lmCfg.EomToken;
	}
	OaMatrix inputIds = OaFnMatrix::FromBytes(
		OaSpan<const OaU8>(reinterpret_cast<const OaU8*>(inputHost.data()), inputHost.size() * sizeof(OaU32)),
		OaMatrixShape{B, tokLen + 1}, OaScalarType::UInt32);
	OaMatrix targetIds = OaFnMatrix::FromBytes(
		OaSpan<const OaU8>(reinterpret_cast<const OaU8*>(targetHost.data()), targetHost.size() * sizeof(OaU32)),
		OaMatrixShape{B, tokLen + 1}, OaScalarType::UInt32);

	// Stage 2: small AR transformer.
	lmCfg.DModel = 128; lmCfg.NumLayers = 2; lmCfg.DFfn = 256;
	auto lm = OaMakeSharedPtr<OaAlmPriorAg>(lmCfg);
	(void)ctx.Execute(); (void)ctx.Sync();  // flush LM initialization
	auto lmParams = lm->AllParameterPtrs();
	auto lmOpt = OaMakeUniquePtr<OaAdamW>(lmParams, 1e-3F, 0.9F, 0.999F, 1e-8F, 0.01F);

	OaF32 firstLoss = 0.0F;
	OaF32 lastLoss = 0.0F;
	for (OaI32 s = 1; s <= 200; ++s) {
		ctx.Clear();
		OaGradientTape tape; lmOpt->ZeroGrad();
		auto logits = lm->Forward(inputIds);                       // [B, tokLen+1, Vocab]
		auto logitsFlat = logits.Reshape(OaMatrixShape{static_cast<OaI64>(B) * (tokLen + 1), lmCfg.VocabSize});
		auto targetFlat = targetIds.Reshape(OaMatrixShape{static_cast<OaI64>(B) * (tokLen + 1)});
		auto ce = OaFnLoss::CrossEntropy(logitsFlat, targetFlat);
		tape.Backward(ce);
		lmOpt->Step();
		(void)ctx.Execute(); (void)ctx.Sync();
		const float lv = ce.At(0);
		if (s == 1) firstLoss = lv;
		lastLoss = lv;
		if (s == 1 || s % 50 == 0 || s == 200)
			std::printf("  [lm] step %3d | ce %.8f\n", s, static_cast<double>(lv));
		ASSERT_TRUE(std::isfinite(lv)) << "LM diverged at step " << s;
	}
	std::printf("LM cross-entropy: %.8f -> %.8f\n", static_cast<double>(firstLoss), static_cast<double>(lastLoss));
	EXPECT_LT(lastLoss, firstLoss) << "LM did not learn next-token prediction";
	lmOpt.Reset();
	lm.Reset();
	tokOpt.Reset();
	tok.Reset();
	ctx.Clear();
}

// Exact trainalm LM shape on the iGPU configuration. This is intentionally a
// short fixed-batch regression: it verifies that B=64, T=65 and the full
// 3-layer D=192 graph produce gradients and update the model, while reporting
// unambiguous wall latency and token throughput.
TEST(Alm, LmProductionShapeUpdates) {
	OaFnMatrix::SetRngSeed(42);
	auto& ctx = OaContext::GetDefault();
	constexpr OaI32 B = 64;
	constexpr OaI32 TokenLen = 64;
	constexpr OaI32 T = TokenLen + 1;
	constexpr OaI32 Steps = 4;

	OaAlmPriorConfig cfg;
	cfg.SyncVocab(512);
	cfg.DModel = 192;
	cfg.NumHeads = 6;
	cfg.NumLayers = 3;
	cfg.DFfn = 512;
	cfg.SeqLen = T;

	std::vector<OaI32> input(static_cast<size_t>(B) * T);
	std::vector<OaI32> target(static_cast<size_t>(B) * T);
	for (OaI32 b = 0; b < B; ++b) {
		const size_t row = static_cast<size_t>(b) * T;
		input[row] = cfg.SomToken;
		for (OaI32 t = 0; t < TokenLen; ++t) {
			const OaI32 code = (b * 7 + t * 3) % cfg.NumCodes;
			input[row + 1 + static_cast<size_t>(t)] = code;
			target[row + static_cast<size_t>(t)] = code;
		}
		target[row + TokenLen] = cfg.EomToken;
	}
	auto inputIds = OaFnMatrix::FromBytes(
		OaSpan<const OaU8>(reinterpret_cast<const OaU8*>(input.data()), input.size() * sizeof(OaI32)),
		OaMatrixShape{B, T}, OaScalarType::Int32);
	auto targetIds = OaFnMatrix::FromBytes(
		OaSpan<const OaU8>(reinterpret_cast<const OaU8*>(target.data()), target.size() * sizeof(OaI32)),
		OaMatrixShape{B, T}, OaScalarType::Int32);

	auto lm = OaMakeSharedPtr<OaAlmPriorAg>(cfg);
	(void)ctx.Execute(); (void)ctx.Sync();
	auto params = lm->AllParameterPtrs();
	OaAdamW opt(params, 1e-4F, 0.9F, 0.99F, 1e-8F, 0.01F);

	OaF32 first = 0.0F;
	OaF32 last = 0.0F;
	const auto begin = std::chrono::steady_clock::now();
	for (OaI32 step = 1; step <= Steps; ++step) {
		ctx.Clear();
		opt.ZeroGrad();
		OaGradientTape tape;
		auto logits = lm->Forward(inputIds);
		auto ce = OaFnLoss::CrossEntropy(
			logits.Reshape(OaMatrixShape{static_cast<OaI64>(B) * T, cfg.VocabSize}),
			targetIds.Reshape(OaMatrixShape{static_cast<OaI64>(B) * T}));
		tape.Backward(ce);
		opt.Step();
		ASSERT_TRUE(ctx.Execute().IsOk());
		ASSERT_TRUE(ctx.Sync().IsOk());
		last = ce.At(0);
		if (step == 1) first = last;
		std::printf("  [production-shape] step %d/%d | ce %.8f\n",
			step, Steps, static_cast<double>(last));
		ASSERT_TRUE(std::isfinite(last));
	}
	const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
	const double msPerStep = seconds * 1000.0 / Steps;
	const double seqPerSec = static_cast<double>(B) * Steps / seconds;
	const double tokensPerSec = static_cast<double>(B) * T * Steps / seconds;
	std::printf("Production shape: CE %.8f -> %.8f | %.2f ms/step | %.2f seq/s | %.2f tks\n",
		static_cast<double>(first), static_cast<double>(last), msPerStep, seqPerSec, tokensPerSec);
	EXPECT_LT(last, first) << "production-shape OaAlm did not update";
	ctx.Clear();
}

// A causal Transformer's logits at position p must not depend on later tokens.
// This also verifies that one block can safely change runtime sequence length:
// every prefix rebuilds the mask while reusing exactly the same model weights.
TEST(Alm, LmDynamicPrefixMatchesFullForward) {
	OaFnMatrix::SetRngSeed(23);
	auto& ctx = OaContext::GetDefault();

	OaAlmPriorConfig lmCfg;
	lmCfg.SyncVocab(16);
	lmCfg.DModel = 64; lmCfg.NumLayers = 2; lmCfg.DFfn = 128;
	lmCfg.SeqLen = 13;
	lmCfg.MaxGenLen = 20;

	auto lm = OaMakeSharedPtr<OaAlmPriorAg>(lmCfg);
	(void)ctx.Execute(); (void)ctx.Sync();

	constexpr OaI32 B = 3;
	constexpr OaI32 T = 13;
	std::vector<OaU32> fullIds(static_cast<size_t>(B) * T);
	for (OaI32 b = 0; b < B; ++b) {
		for (OaI32 t = 0; t < T; ++t) {
			fullIds[static_cast<size_t>(b) * T + t] = static_cast<OaU32>((b * 5 + t * 3) % lmCfg.NumCodes);
		}
	}
	auto fullInput = OaFnMatrix::FromBytes(
		OaSpan<const OaU8>(reinterpret_cast<const OaU8*>(fullIds.data()), fullIds.size() * sizeof(OaU32)),
		OaMatrixShape{B, T}, OaScalarType::UInt32);
	auto fullLogits = HostFloatData(lm->Forward(fullInput));

	for (const OaI32 prefixLen : {1, 2, 7, T}) {
		ctx.Clear();
		std::vector<OaU32> prefix(static_cast<size_t>(B) * prefixLen);
		for (OaI32 b = 0; b < B; ++b) {
			for (OaI32 t = 0; t < prefixLen; ++t) {
				prefix[static_cast<size_t>(b) * prefixLen + t] = fullIds[static_cast<size_t>(b) * T + t];
			}
		}
		auto prefixInput = OaFnMatrix::FromBytes(
			OaSpan<const OaU8>(reinterpret_cast<const OaU8*>(prefix.data()), prefix.size() * sizeof(OaU32)),
			OaMatrixShape{B, prefixLen}, OaScalarType::UInt32);
		auto prefixLogits = HostFloatData(lm->Forward(prefixInput));

		OaF32 maxError = 0.0F;
		for (OaI32 b = 0; b < B; ++b) {
			for (OaI32 v = 0; v < lmCfg.VocabSize; ++v) {
				const OaI64 fullIdx = (static_cast<OaI64>(b) * T + prefixLen - 1) * lmCfg.VocabSize + v;
				const OaI64 prefixIdx = (static_cast<OaI64>(b) * prefixLen + prefixLen - 1) * lmCfg.VocabSize + v;
				maxError = std::max(maxError, std::abs(fullLogits[fullIdx] - prefixLogits[prefixIdx]));
			}
		}
		std::printf("  [dynamic-prefix] T=%d max error %.8g\n", prefixLen, static_cast<double>(maxError));
		EXPECT_LT(maxError, 1e-4F);
	}

	lm.Reset();
	ctx.Clear();
}

// Frozen semantic features are a real part of the ALM graph: they become one
// learned causal prefix token, alter motion logits, and train the projection.
TEST(Alm, LmFrozenTextPrefixConditionsAndBackpropagates) {
	OaFnMatrix::SetRngSeed(27);
	auto& ctx = OaContext::GetDefault();

	OaAlmPriorConfig cfg;
	cfg.SyncVocab(16);
	cfg.DModel = 32; cfg.NumLayers = 1; cfg.DFfn = 64;
	cfg.TextFeatureDim = 4;
	cfg.SeqLen = 6;       // one text prefix + five motion-token positions
	cfg.MaxSeqLen = 8;
	auto lm = OaMakeSharedPtr<OaAlmPriorAg>(cfg);

	const std::vector<OaI32> ids = {
		cfg.SomToken, 1, 2, 3, 4,
		cfg.SomToken, 1, 2, 3, 4};
	const std::vector<OaI32> targets = {
		1, 2, 3, 4, cfg.EomToken,
		1, 2, 3, 4, cfg.EomToken};
	const std::vector<float> textA = {
		1.0F, 0.0F, 0.0F, 0.0F,
		1.0F, 0.0F, 0.0F, 0.0F};
	const std::vector<float> textB = {
		0.0F, 1.0F, 0.5F, -0.5F,
		0.0F, 1.0F, 0.5F, -0.5F};
	auto input = OaFnMatrix::FromBytes(
		OaSpan<const OaU8>(reinterpret_cast<const OaU8*>(ids.data()), ids.size() * sizeof(OaI32)),
		OaMatrixShape{2, 5}, OaScalarType::Int32);
	auto target = OaFnMatrix::FromBytes(
		OaSpan<const OaU8>(reinterpret_cast<const OaU8*>(targets.data()), targets.size() * sizeof(OaI32)),
		OaMatrixShape{10}, OaScalarType::Int32);
	auto featureA = OaFnMatrix::FromBytes(
		OaSpan<const OaU8>(reinterpret_cast<const OaU8*>(textA.data()), textA.size() * sizeof(float)),
		OaMatrixShape{2, 4}, OaScalarType::Float32);
	auto featureB = OaFnMatrix::FromBytes(
		OaSpan<const OaU8>(reinterpret_cast<const OaU8*>(textB.data()), textB.size() * sizeof(float)),
		OaMatrixShape{2, 4}, OaScalarType::Float32);

	auto logitsA = HostFloatData(lm->ForwardConditioned(input, featureA));
	ctx.Clear();
	auto logitsBMatrix = lm->ForwardConditioned(input, featureB);
	auto logitsB = HostFloatData(logitsBMatrix);
	ASSERT_EQ(logitsA.Size(), logitsB.Size());
	OaF32 maxPromptDelta = 0.0F;
	for (OaUsize i = 0; i < logitsA.Size(); ++i) {
		maxPromptDelta = std::max(maxPromptDelta, std::abs(logitsA[i] - logitsB[i]));
	}
	EXPECT_GT(maxPromptDelta, 1e-6F) << "different frozen text features did not affect motion logits";
	ctx.Clear();
	auto generated = lm->GenerateConditioned(featureA, 1.0F, 1, 0.0F, 7);
	EXPECT_EQ(generated.Size(0), 2);
	EXPECT_LE(generated.Size(1), 8);  // [MOTION_BOS] plus at most seven samples

	ctx.Clear();
	OaGradientTape tape;
	auto trainLogits = lm->ForwardConditioned(input, featureB).Reshape(
		OaMatrixShape{10, cfg.VocabSize});
	auto loss = OaFnLoss::CrossEntropy(trainLogits, target);
	tape.Backward(loss);
	ASSERT_TRUE(ctx.Execute().IsOk());
	ASSERT_TRUE(ctx.Sync().IsOk());

	OaParameter* projectionWeight = nullptr;
	for (auto named : lm->AllNamedParameterPtrs()) {
		if (named.Path == "text_projection.weight") projectionWeight = named.Param;
	}
	ASSERT_NE(projectionWeight, nullptr);
	ASSERT_FALSE(projectionWeight->Grad().IsEmpty());
	double gradL1 = 0.0;
	for (const auto value : HostFloatData(projectionWeight->Grad())) gradL1 += std::abs(value);
	EXPECT_GT(gradL1, 1e-8) << "motion loss did not train the text projection";
	const OaString checkpointPath = "/tmp/alm_conditioned_contract.oam";
	ASSERT_TRUE(lm->Save(checkpointPath).IsOk());
	auto checkpoint = OamModel::Load(checkpointPath);
	ASSERT_TRUE(checkpoint.IsOk());
	const auto& saved = checkpoint.GetValue();
	const auto* savedProjection = saved.FindWeight("text_projection.weight");
	ASSERT_NE(savedProjection, nullptr);
	EXPECT_EQ(savedProjection->Rank, 2);
	EXPECT_EQ(savedProjection->Shape[0], static_cast<OaU64>(cfg.DModel));
	EXPECT_EQ(savedProjection->Shape[1], static_cast<OaU64>(cfg.TextFeatureDim));
	std::remove(checkpointPath.CStr());
	std::printf("  [text-prefix] max logit delta %.8g · projection grad L1 %.8g\n",
		static_cast<double>(maxPromptDelta), gradL1);
	ctx.Clear();
}

TEST(Alm, LmFfnPoliciesForward) {
	OaFnMatrix::SetRngSeed(29);
	auto& ctx = OaContext::GetDefault();
	constexpr OaI32 B = 2;
	constexpr OaI32 T = 5;
	std::vector<OaU32> ids(static_cast<size_t>(B) * T);
	for (OaI32 i = 0; i < B * T; ++i) ids[static_cast<size_t>(i)] = static_cast<OaU32>(i % 16);
	auto input = OaFnMatrix::FromBytes(
		OaSpan<const OaU8>(reinterpret_cast<const OaU8*>(ids.data()), ids.size() * sizeof(OaU32)),
		OaMatrixShape{B, T}, OaScalarType::UInt32);

	for (const OaAlmFfnType policy : {OaAlmFfnType::Dense, OaAlmFfnType::Moe, OaAlmFfnType::Hybrid}) {
		OaAlmPriorConfig cfg;
		cfg.SyncVocab(16);
		cfg.DModel = 32; cfg.NumLayers = 2; cfg.DFfn = 32; cfg.SeqLen = T;
		cfg.FfnType = policy; cfg.MoeNumExperts = 2; cfg.MoeExpertsPerToken = 1; cfg.MoeEvery = 2;
		auto lm = OaMakeSharedPtr<OaAlmPriorAg>(cfg);
		auto logits = lm->Forward(input);
		EXPECT_EQ(logits.Size(0), B);
		EXPECT_EQ(logits.Size(1), T);
		EXPECT_EQ(logits.Size(2), cfg.VocabSize);
		EXPECT_TRUE(HostAllFinite(logits));
		ctx.Clear();
	}
}

TEST(Alm, LmCheckpointRoundtrip) {
	OaFnMatrix::SetRngSeed(31);
	auto& ctx = OaContext::GetDefault();
	OaAlmPriorConfig cfg;
	cfg.SyncVocab(16);
	cfg.DModel = 32; cfg.NumLayers = 2; cfg.DFfn = 64; cfg.SeqLen = 6;

	std::vector<OaI32> ids = {cfg.SomToken, 1, 2, 3, 4, 5, cfg.SomToken, 5, 4, 3, 2, 1};
	std::vector<OaI32> targets = {1, 2, 3, 4, 5, cfg.EomToken, 5, 4, 3, 2, 1, cfg.EomToken};
	auto input = OaFnMatrix::FromBytes(
		OaSpan<const OaU8>(reinterpret_cast<const OaU8*>(ids.data()), ids.size() * sizeof(OaI32)),
		OaMatrixShape{2, 6}, OaScalarType::Int32);
	auto target = OaFnMatrix::FromBytes(
		OaSpan<const OaU8>(reinterpret_cast<const OaU8*>(targets.data()), targets.size() * sizeof(OaI32)),
		OaMatrixShape{12}, OaScalarType::Int32);

	auto original = OaMakeSharedPtr<OaAlmPriorAg>(cfg);
	auto originalParams = original->AllParameterPtrs();
	auto originalOpt = OaMakeUniquePtr<OaAdamW>(originalParams, 1e-3F);
	{
		OaGradientTape tape;
		originalOpt->ZeroGrad();
		auto logits = original->Forward(input).Reshape(OaMatrixShape{12, cfg.VocabSize});
		auto ce = OaFnLoss::CrossEntropy(logits, target);
		tape.Backward(ce);
		originalOpt->Step();
		(void)ctx.Execute(); (void)ctx.Sync();
	}
	ctx.Clear();
	auto before = HostFloatData(original->Forward(input));
	const OaString path = "/tmp/alm_transformer_roundtrip.oam";
	ASSERT_TRUE(original->Save(path, *originalOpt).IsOk());

	auto reloaded = OaMakeSharedPtr<OaAlmPriorAg>(cfg);
	auto reloadedParams = reloaded->AllParameterPtrs();
	auto reloadedOpt = OaMakeUniquePtr<OaAdamW>(reloadedParams, 1e-3F);
	ASSERT_TRUE(reloaded->Load(path, *reloadedOpt).IsOk());
	(void)ctx.Execute(); (void)ctx.Sync();
	ctx.Clear();
	auto after = HostFloatData(reloaded->Forward(input));
	ASSERT_EQ(before.Size(), after.Size());
	OaF32 maxError = 0.0F;
	for (OaUsize i = 0; i < before.Size(); ++i) maxError = std::max(maxError, std::abs(before[i] - after[i]));
	std::printf("LM checkpoint round-trip max logit error %.8g\n", static_cast<double>(maxError));
	EXPECT_EQ(maxError, 0.0F);
	EXPECT_EQ(reloadedOpt->GetStep(), originalOpt->GetStep());
	std::remove(path.CStr());
	reloadedOpt.Reset(); reloaded.Reset(); originalOpt.Reset(); original.Reset();
	ctx.Clear();
}

TEST(Alm, BundleRoundtrip) {
	OaFnMatrix::SetRngSeed(37);
	auto& ctx = OaContext::GetDefault();
	OaAlmAgConfig cfg;
	cfg.Tokenizer.InputDim = 6;
	cfg.Tokenizer.Width = 8;
	cfg.Tokenizer.CodeDim = 8;
	cfg.Tokenizer.NumCodes = 8;
	cfg.Tokenizer.DownT = 1;
	cfg.Tokenizer.Depth = 1;
	cfg.Prior.SyncVocab(cfg.Tokenizer.NumCodes);
	cfg.Prior.DModel = 8;
	cfg.Prior.NumHeads = 2;
	cfg.Prior.NumLayers = 1;
	cfg.Prior.DFfn = 16;
	cfg.Prior.TextFeatureDim = 4;
	cfg.Prior.SeqLen = 4;
	cfg.Prior.MaxSeqLen = 8;
	cfg.Prior.MaxGenLen = 7;
	cfg.TextEncoder = "oa/test-clip";

	auto original = OaMakeSharedPtr<OaAlmAg>(cfg);
	const std::vector<OaI32> ids = {8, 1, 2, 3};
	const std::vector<OaF32> text = {0.25F, -0.5F, 0.75F, 1.0F};
	auto input = OaFnMatrix::FromBytes(
		OaSpan<const OaU8>(reinterpret_cast<const OaU8*>(ids.data()), ids.size() * sizeof(OaI32)),
		OaMatrixShape{1, 4}, OaScalarType::Int32);
	auto feature = OaFnMatrix::FromBytes(
		OaSpan<const OaU8>(reinterpret_cast<const OaU8*>(text.data()), text.size() * sizeof(OaF32)),
		OaMatrixShape{1, 4}, OaScalarType::Float32);
	auto before = HostFloatData(original->ForwardConditioned(input, feature));

	const OaString path = "/tmp/alm_ag_bundle_roundtrip.oam";
	ASSERT_TRUE(original->SaveBundle(path).IsOk());
	auto raw = OamModel::Load(path);
	ASSERT_TRUE(raw.IsOk());
	EXPECT_STREQ(raw.GetValue().Config.Architecture, "OaAlmAg");
	EXPECT_NE(raw.GetValue().FindWeight("tokenizer.enc_in.weight"), nullptr);
	EXPECT_NE(raw.GetValue().FindWeight("prior.text_projection.weight"), nullptr);

	auto loaded = OaAlmAg::LoadBundle(path);
	ASSERT_TRUE(loaded.IsOk()) << loaded.GetStatus().GetMessage().CStr();
	auto reloaded = std::move(loaded).GetValue();
	EXPECT_EQ(reloaded->Config().Tokenizer.NumCodes, cfg.Tokenizer.NumCodes);
	EXPECT_EQ(reloaded->Config().Prior.TextFeatureDim, cfg.Prior.TextFeatureDim);
	EXPECT_EQ(reloaded->Config().Prior.NumHeads, cfg.Prior.NumHeads);
	EXPECT_EQ(reloaded->Config().TextEncoder, cfg.TextEncoder);
	auto after = HostFloatData(reloaded->ForwardConditioned(input, feature));
	ASSERT_EQ(before.Size(), after.Size());
	for (OaUsize i = 0; i < before.Size(); ++i) EXPECT_EQ(before[i], after[i]);
	std::remove(path.CStr());
	ctx.Clear();
}

// ─── Throughput benchmarks (samples/sec) at the var/config/Alm.yaml scale ───
// Run fp32, then repeat with --bf16, and compare the printed ms/step + sps.
// These do NOT assert correctness — they measure steady-state training throughput.
// The startup log line "precision=FP32|BF16" identifies which run is which.

// These benches deliberately run the REFERENCE full-scale model at batch 128 — the whole
// point is the real throughput number. On an integrated GPU (shared system RAM) the
// full-scale model + autograd tape exhausts host memory and OOMs the box, and a throughput
// figure from hardware that can't hold the model is meaningless anyway. Skip there.
static bool OaBenchNeedsDiscreteGpu() {
	auto* rt = OaEngine::GetGlobal();
	if (rt == nullptr) return true;   // no engine → nothing to benchmark
	const OaDeviceType dt = rt->Device.Info.Hardware.DeviceType;
	return dt == OaDeviceType::VkIntegrated || dt == OaDeviceType::VkCpu
	    || dt == OaDeviceType::Host;
}

TEST(Alm, LmTrainBench) {
	if (OaBenchNeedsDiscreteGpu())
		GTEST_SKIP() << "full-scale throughput bench needs a discrete GPU (shared-RAM iGPU OOMs)";
	OaFnMatrix::SetRngSeed(42);
	auto& ctx = OaContext::GetDefault();

	OaAlmPriorConfig lmCfg;
	lmCfg.SyncVocab(512);                       // matches yaml num_codes
	lmCfg.DModel = 384; lmCfg.NumLayers = 6; lmCfg.DFfn = 1536;

	const OaI32 B = 128, T = 64;                // batch, lm_seq_len (yaml)
	std::vector<OaU32> inHost(static_cast<size_t>(B) * (T + 1));
	std::vector<OaU32> tgtHost(static_cast<size_t>(B) * (T + 1));
	for (OaI32 b = 0; b < B; ++b) {
		const size_t row = static_cast<size_t>(b) * (T + 1);
		inHost[row] = static_cast<OaU32>(lmCfg.SomToken);
		for (OaI32 t = 0; t < T; ++t) {
			const OaU32 c = static_cast<OaU32>((b * 7u + t * 3u) % 512u);
			inHost[row + 1 + t] = c;
			tgtHost[row + t] = c;
		}
		tgtHost[row + T] = static_cast<OaU32>(lmCfg.EomToken);
	}
	OaMatrix inputIds = OaFnMatrix::FromBytes(
		OaSpan<const OaU8>(reinterpret_cast<const OaU8*>(inHost.data()), inHost.size() * sizeof(OaU32)),
		OaMatrixShape{B, T + 1}, OaScalarType::UInt32);
	OaMatrix targetIds = OaFnMatrix::FromBytes(
		OaSpan<const OaU8>(reinterpret_cast<const OaU8*>(tgtHost.data()), tgtHost.size() * sizeof(OaU32)),
		OaMatrixShape{B, T + 1}, OaScalarType::UInt32);

	auto lm = OaMakeSharedPtr<OaAlmPriorAg>(lmCfg);
	(void)ctx.Execute(); (void)ctx.Sync();
	auto lmParams = lm->AllParameterPtrs();
	auto lmOpt = OaMakeUniquePtr<OaAdamW>(lmParams, 1e-4F, 0.9F, 0.999F, 1e-8F, 0.01F);

	const int warmup = 3, timed = 10;
	auto stepOnce = [&]() {
		ctx.Clear();
		OaGradientTape tape; lmOpt->ZeroGrad();
		auto logits = lm->Forward(inputIds);
		auto lf = logits.Reshape(OaMatrixShape{static_cast<OaI64>(B) * (T + 1), lmCfg.VocabSize});
		auto tf = targetIds.Reshape(OaMatrixShape{static_cast<OaI64>(B) * (T + 1)});
		auto ce = OaFnLoss::CrossEntropy(lf, tf);
		tape.Backward(ce);
		lmOpt->Step();
		(void)ctx.Execute(); (void)ctx.Sync();
	};
	for (int i = 0; i < warmup; ++i) stepOnce();
	auto t0 = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < timed; ++i) stepOnce();
	auto t1 = std::chrono::high_resolution_clock::now();
	const double dt = std::chrono::duration<double>(t1 - t0).count();
	const double msStep = dt / timed * 1000.0;
	const double sps = static_cast<double>(B) * timed / dt;
	const char* prec = OaFnMatrix::GetWeightDtype() == OaScalarType::Float32 ? "fp32" : "bf16";
	std::printf("  [lm bench %s] 6L D=384 B=%d T=%d | %.2f ms/step | %.1f seq/s (%.1fK tok/s)\n",
		prec, B, T, msStep, sps, sps * T / 1000.0);
	lmOpt.Reset(); lm.Reset(); ctx.Clear();
}

TEST(Alm, TokTrainBench) {
	if (OaBenchNeedsDiscreteGpu())
		GTEST_SKIP() << "full-scale throughput bench needs a discrete GPU (shared-RAM iGPU OOMs)";
	OaFnMatrix::SetRngSeed(42);
	auto& ctx = OaContext::GetDefault();

	OaAlmTokenizerConfig tokCfg;
	tokCfg.InputDim = 263; tokCfg.Width = 512; tokCfg.CodeDim = 512; tokCfg.NumCodes = 512;
	tokCfg.DownT = 2; tokCfg.Depth = 3;
	tokCfg.CommitBeta = 0.25F; tokCfg.EmaDecay = 0.99F; tokCfg.DeadThresh = 2.0F;
	auto tok = OaMakeSharedPtr<OaAlmTokenizerAg>(tokCfg);

	const OaI32 B = 128, T = 64;                // batch, seq_len (yaml)
	const OaI32 tokLen = T / tok->DownsampleFactor();
	std::vector<float> ds(static_cast<size_t>(B) * T * tokCfg.InputDim);
	for (OaI32 b = 0; b < B; ++b) {
		for (OaI32 t = 0; t < T; ++t) {
			const float tt = static_cast<float>(t) / static_cast<float>(T);
			for (OaI32 c = 0; c < tokCfg.InputDim; ++c) {
				const float freq = 1.0F + static_cast<float>(c % 8) * 0.5F;
				ds[((static_cast<size_t>(b) * T + t) * tokCfg.InputDim) + c] =
					static_cast<float>(b) * 0.7F + std::sin(6.2831853F * freq * tt) * 0.5F;
			}
		}
	}
	OaMatrix X = OaFnMatrix::FromBytes(
		OaSpan<const OaU8>(reinterpret_cast<const OaU8*>(ds.data()), ds.size() * sizeof(float)),
		OaMatrixShape{B, T, tokCfg.InputDim}, OaScalarType::Float32);
	{ auto z0 = tok->Encode(X, B, T); tok->Seed(z0); ctx.Clear(); }   // seed codebook (2048 latents ≥ 512)

	auto tokParams = tok->AllParameterPtrs();
	auto tokOpt = OaMakeUniquePtr<OaAdamW>(tokParams, 2e-4F, 0.9F, 0.99F, 1e-8F, 0.0F);

	const int warmup = 3, timed = 8;
	auto stepOnce = [&]() {
		ctx.Clear();
		OaGradientTape tape; tokOpt->ZeroGrad();
		auto z = tok->Encode(X, B, T);
		auto q = tok->Quantize(z);
		auto rec = tok->Decode(q.Quantized, B, tokLen);
		auto recon = OaFnLoss::SmoothL1(rec, X.Reshape(OaMatrixShape{static_cast<OaI64>(B) * T, tokCfg.InputDim}));
		auto loss = recon + q.CommitLoss;
		tape.Backward(loss);
		tokOpt->Step();
		tok->EmaUpdate(q);
		(void)ctx.Execute(); (void)ctx.Sync();
	};
	for (int i = 0; i < warmup; ++i) stepOnce();
	auto t0 = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < timed; ++i) stepOnce();
	auto t1 = std::chrono::high_resolution_clock::now();
	const double dt = std::chrono::duration<double>(t1 - t0).count();
	const double msStep = dt / timed * 1000.0;
	const double sps = static_cast<double>(B) * timed / dt;
	const char* prec = OaFnMatrix::GetWeightDtype() == OaScalarType::Float32 ? "fp32" : "bf16";
	std::printf("  [tok bench %s] W=512 263d 3L DownT=2 B=%d T=%d | %.2f ms/step | %.1f clips/s\n",
		prec, B, T, msStep, sps);
	tokOpt.Reset(); tok.Reset(); ctx.Clear();
}

// iGPU-sized SMOKE tokenizer config shared by the two CMP training tests. NOT the
// reference 512³ model — those tests verify scale-invariant properties (the tokenizer
// LEARNS; the codebook does NOT collapse), so a light net that fits an integrated GPU
// is the correct vehicle and a full run finishes in seconds rather than never.
//   • The conv stack runs through the im2col + tiled-matmul path (the retired scalar
//     direct-conv was ~20× slower at tokenizer shapes) — the biggest iGPU-tractability lever.
//   • Width/CodeDim/Depth trimmed for speed; NumCodes stays 512 because codebook health
//     at the real K is precisely the property under test.
//   • CommitBeta/EmaDecay/DeadThresh are the VQ-collapse-safe values (§12.1).
static OaAlmTokenizerConfig CmpSmokeTokenizerCfg(OaI32 InFeatDim) {
	OaAlmTokenizerConfig cfg;
	cfg.InputDim    = InFeatDim;  // 263 for CMP/HumanML3D
	cfg.Width       = 128;
	cfg.CodeDim     = 128;
	cfg.NumCodes    = 512;
	cfg.DownT       = 2;
	cfg.Depth       = 2;
	cfg.CommitBeta  = 0.25F;
	cfg.EmaDecay    = 0.99F;
	cfg.DeadThresh  = 2.0F;
	return cfg;
}

// Codebook health: tokenize a batch, histogram code usage →
// active-code % + perplexity (effective #codes used = exp(-Σ p·log p)). VQ collapse
// (a few dead-heavy codes) is the #1 cause of blurry/mode-collapsed motion; MotionGPT's
// mgpt_vq.py logs this and we did not (Phase A4). Perplexity is scale-robust; active-%
// needs tokens ≫ codes to be meaningful, so tokenize as many clips as fit.
// Returns perplexity (effective #codes in use) — a scale-robust collapse metric: a
// collapsed codebook drives it toward 1.0, so callers can assert on it as a hard gate.
static double PrintCodebookHealth(OaAlmTokenizerAg& InTok, const OaMatrix& InX,
                                OaI32 InBatch, OaI32 InFrames, OaI32 InNumCodes) {
	auto& ctx = OaContext::GetDefault();
	auto ids = InTok.Tokenize(InX, InBatch, InFrames)[0];
	(void)ctx.Execute(); (void)ctx.Sync();
	const OaI64 n = ids.NumElements();
	const OaI32* p = ids.DataAs<const OaI32>();
	std::vector<OaI64> hist(static_cast<size_t>(InNumCodes), 0);
	for (OaI64 i = 0; i < n; ++i) {
		const OaI32 c = p[i];
		if (c >= 0 && c < InNumCodes) ++hist[static_cast<size_t>(c)];
	}
	OaI32 active = 0;
	double H = 0.0;
	for (OaI32 c = 0; c < InNumCodes; ++c) {
		if (hist[static_cast<size_t>(c)] == 0) continue;
		++active;
		const double pr = static_cast<double>(hist[static_cast<size_t>(c)]) / static_cast<double>(n);
		H -= pr * std::log(pr);
	}
	const double perplexity = std::exp(H);
	std::printf("  [codebook] %d/%d active (%.1f%%) | perplexity %.1f | %lld tokens\n",
	            active, InNumCodes, 100.0 * active / InNumCodes, perplexity,
	            static_cast<long long>(n));
	ctx.Clear();
	return perplexity;
}

// Save/Load round-trip: a trained tokenizer's conv weights AND its EMA codebook
// must persist. Seed + move the codebook off init, tokenize, save, reload into a
// fresh module, tokenize again — identical token ids prove BOTH the conv weights
// (Encode) and the codebook (assignment) survived. If only gradient params were
// saved and the EMA codebook were dropped, the reload would retokenize differently
// and this fails.
TEST(Alm, TokenizerSaveLoadRoundtrip) {
	OaFnMatrix::SetRngSeed(11);
	auto& ctx = OaContext::GetDefault();
	auto cfg = CmpSmokeTokenizerCfg(32);            // small InputDim for speed
	const OaI32 B = 64, T = 64;                      // B·(T/Factor)=1024 ≥ NumCodes(512) to seed
	const OaI32 tokLen = T / (1 << cfg.DownT);

	// Deterministic synthetic batch.
	std::vector<float> xh(static_cast<size_t>(B) * T * cfg.InputDim);
	OaU64 rng = 0x1234ULL;
	for (auto& v : xh) {
		rng = (rng * 6364136223846793005ULL) + 1;
		v = (static_cast<float>(static_cast<OaU32>(rng >> 40)) / static_cast<float>(1 << 24)) - 0.5F;
	}
	auto X = OaFnMatrix::FromBytes(
		OaSpan<const OaU8>(reinterpret_cast<const OaU8*>(xh.data()), xh.size() * sizeof(float)),
		OaMatrixShape{B, T, cfg.InputDim}, OaScalarType::Float32);

	auto tok = OaMakeSharedPtr<OaAlmTokenizerAg>(cfg);
	{ auto z0 = tok->Encode(X, B, T); tok->Seed(z0); ctx.Clear(); }   // data-dependent codebook seed
	for (int s = 0; s < 3; ++s) {                                      // move the EMA codebook off the seed
		auto z = tok->Encode(X, B, T);
		auto q = tok->Quantize(z);
		tok->EmaUpdate(q);
		(void)ctx.Execute(); (void)ctx.Sync(); ctx.Clear();
	}

	auto readIds = [&](const OaVec<OaMatrix>& InIdx) {
		std::vector<std::vector<OaI32>> out;
		for (const auto& m : InIdx) {
			std::vector<OaI32> h(static_cast<size_t>(m.NumElements()));
			(void)OaFnMatrix::CopyToHost(m, h.data(), h.size() * sizeof(OaI32));
			out.push_back(std::move(h));
		}
		return out;
	};

	auto idsA = tok->Tokenize(X, B, T);
	(void)ctx.Execute(); (void)ctx.Sync();
	auto hostA = readIds(idsA);
	ctx.Clear();

	const OaString path = "/tmp/oa_mg_tok_roundtrip.oam";
	ASSERT_TRUE(tok->Save(path).IsOk()) << "tokenizer Save failed";

	auto tok2 = OaMakeSharedPtr<OaAlmTokenizerAg>(cfg);   // fresh random init
	ASSERT_TRUE(tok2->Load(path).IsOk()) << "tokenizer Load failed";
	auto idsB = tok2->Tokenize(X, B, T);
	(void)ctx.Execute(); (void)ctx.Sync();
	auto hostB = readIds(idsB);
	ctx.Clear();

	ASSERT_EQ(hostA.size(), hostB.size()) << "level count mismatch";
	OaI64 total = 0, mismatch = 0;
	for (size_t lvl = 0; lvl < hostA.size(); ++lvl) {
		ASSERT_EQ(hostA[lvl].size(), hostB[lvl].size());
		for (size_t i = 0; i < hostA[lvl].size(); ++i) {
			++total;
			if (hostA[lvl][i] != hostB[lvl][i]) ++mismatch;
		}
	}
	std::printf("TokenizerSaveLoadRoundtrip: %lld/%lld tokens identical after Save+Load (%zu levels)\n",
		static_cast<long long>(total - mismatch), static_cast<long long>(total), hostA.size());
	EXPECT_EQ(mismatch, 0) << "reloaded tokenizer retokenized differently — conv weights or EMA codebook not persisted";
	(void)tokLen;
}

TEST(Alm, TokenizerLearnsCmp) {
	OaFnMatrix::SetRngSeed(7);
	auto& ctx = OaContext::GetDefault();

	const OaString dsPath = "../dataset/gen/3d/anim/ds/Cmp";
	OaDsCmp ds(dsPath, "train", /*InMaxClips=*/128);
	ASSERT_TRUE(ds.Ok()) << "Failed to load CMP from " << dsPath.CStr();
	ASSERT_GE(ds.NumClips(), 128) << "Not enough clips in dataset";

	auto tokCfg = CmpSmokeTokenizerCfg(ds.FeatDim());
	auto tok = OaMakeSharedPtr<OaAlmTokenizerAg>(tokCfg);

	const OaI32 B = 8;
	const OaI32 T = 64;
	const OaI32 tokLen = T / tok->DownsampleFactor();  // 16

	// Seed the codebook: need >= NumCodes latent rows. With factor=4 and
	// NumCodes=512, use B_seed=64 clips × T=64 frames → 1024 latents.
	const OaI32 Bseed = 64;
	{
		std::vector<float> seed(static_cast<size_t>(Bseed * T) * ds.FeatDim());
		for (OaI32 b = 0; b < Bseed; ++b) {
			const OaI32 clipIdx = b % ds.NumClips();
			const OaI32 frames = static_cast<OaI32>(ds.ClipFrames(clipIdx));
			const OaI32 start = frames > T ? (frames - T) / 2 : 0;
			const OaF32* src = ds.ClipData(clipIdx) + start * ds.FeatDim();
			float* dst = seed.data() + static_cast<size_t>(b) * T * ds.FeatDim();
			const OaI32 copyFrames = std::min(T, frames);
			std::memcpy(dst, src, static_cast<size_t>(copyFrames) * ds.FeatDim() * sizeof(float));
		}
		auto seedX = OaFnMatrix::FromBytes(
			OaSpan<const OaU8>(reinterpret_cast<const OaU8*>(seed.data()), seed.size() * sizeof(float)),
			OaMatrixShape{Bseed, T, ds.FeatDim()}, OaScalarType::Float32);
		auto z0 = tok->Encode(seedX, Bseed, T);
		tok->Seed(z0);
		ctx.Clear();
	}

	auto tokParams = tok->AllParameterPtrs();
	auto tokOpt = OaMakeUniquePtr<OaAdamW>(tokParams, 1e-4F, 0.9F, 0.99F, 1e-8F, 0.01F);

	OaF32 firstLoss = 0.0F;
	OaF32 lastLoss = 0.0F;
	for (OaI32 s = 1; s <= 500; ++s) {
		ctx.Clear();
		OaGradientTape tape; tokOpt->ZeroGrad();

		std::vector<float> batch(static_cast<size_t>(B * T) * ds.FeatDim());
		for (OaI32 b = 0; b < B; ++b) {
			const OaI32 clipIdx = b % ds.NumClips();
			const OaI32 frames = static_cast<OaI32>(ds.ClipFrames(clipIdx));
			const OaI32 start = frames > T ? (s * 17 + b * 31) % (frames - T) : 0;
			const OaF32* src = ds.ClipData(clipIdx) + start * ds.FeatDim();
			float* dst = batch.data() + static_cast<size_t>(b) * T * ds.FeatDim();
			const OaI32 copyFrames = std::min(T, frames);
			std::memcpy(dst, src, static_cast<size_t>(copyFrames) * ds.FeatDim() * sizeof(float));
		}
		auto X = OaFnMatrix::FromBytes(
			OaSpan<const OaU8>(reinterpret_cast<const OaU8*>(batch.data()), batch.size() * sizeof(float)),
			OaMatrixShape{B, T, ds.FeatDim()}, OaScalarType::Float32);

		auto z = tok->Encode(X, B, T);
		auto q = tok->Quantize(z);
		auto rec = tok->Decode(q.Quantized, B, tokLen);
		auto xFlat = X.Reshape(OaMatrixShape{static_cast<OaI64>(B) * T, ds.FeatDim()});
		auto recon = OaFnLoss::SmoothL1(rec, xFlat);

		// Velocity loss: SmoothL1 on frame-to-frame differences (improves motion smoothness).
		// LAMBDA_VELOCITY=0.5 (reference: config_h3d_stage2.yaml)
		auto rec3d   = rec.Reshape(OaMatrixShape{B, T, ds.FeatDim()});
		auto xFlat3d = X;  // already [B, T, D]
		auto recVel   = OaFnMatrix::Sub(OaFnMatrix::Slice(rec3d, 1, 1, T),   OaFnMatrix::Slice(rec3d, 1, 0, T - 1));
		auto xVel     = OaFnMatrix::Sub(OaFnMatrix::Slice(xFlat3d, 1, 1, T), OaFnMatrix::Slice(xFlat3d, 1, 0, T - 1));
		auto velLoss  = OaFnLoss::SmoothL1(recVel.Reshape(OaMatrixShape{static_cast<OaI64>(B) * (T - 1), ds.FeatDim()}),
		                                   xVel.Reshape(OaMatrixShape{static_cast<OaI64>(B) * (T - 1), ds.FeatDim()}));

		auto loss = recon + OaFnMatrix::Scale(velLoss, 0.5F) + q.CommitLoss;
		tape.Backward(loss);
		tokOpt->Step();
		tok->EmaUpdate(q);
		(void)ctx.Execute(); (void)ctx.Sync();

		const float lv = loss.At(0);
		if (s == 1) firstLoss = lv;
		lastLoss = lv;
		if (s == 1 || s % 100 == 0 || s == 500)
			std::printf("  [h3d] step %3d | loss %.8f\n", s, static_cast<double>(lv));
		ASSERT_TRUE(std::isfinite(lv)) << "Tokenizer diverged at step " << s;
	}
	std::printf("CMP tokenizer loss: %.8f -> %.8f\n", static_cast<double>(firstLoss), static_cast<double>(lastLoss));
	EXPECT_LT(lastLoss, firstLoss) << "Tokenizer did not learn CMP reconstruction";

	// Codebook health (§12.1): tokenize all clips (fixed T window) → active% + perplexity.
	{
		const OaI32 Bh = std::min<OaI32>(128, static_cast<OaI32>(ds.NumClips()));
		std::vector<float> hb(static_cast<size_t>(Bh * T) * ds.FeatDim());
		for (OaI32 b = 0; b < Bh; ++b) {
			const OaI32 frames = static_cast<OaI32>(ds.ClipFrames(b));
			const OaI32 start = frames > T ? (frames - T) / 2 : 0;
			const OaF32* src = ds.ClipData(b) + start * ds.FeatDim();
			float* dst = hb.data() + static_cast<size_t>(b) * T * ds.FeatDim();
			std::memcpy(dst, src, static_cast<size_t>(std::min(T, frames)) * ds.FeatDim() * sizeof(float));
		}
		auto Xh = OaFnMatrix::FromBytes(
			OaSpan<const OaU8>(reinterpret_cast<const OaU8*>(hb.data()), hb.size() * sizeof(float)),
			OaMatrixShape{Bh, T, ds.FeatDim()}, OaScalarType::Float32);
		const double perplexity = PrintCodebookHealth(*tok, Xh, Bh, T, tokCfg.NumCodes);
		// Hard collapse gate: a healthy VQ spreads tokens over many codes (perplexity ≫ 1);
		// centroid collapse pins every latent on one code (perplexity → 1). The threshold is
		// far below observed-healthy and far above collapse, so it catches the regression
		// without flaking on run-to-run variance.
		EXPECT_GT(perplexity, 8.0) << "VQ codebook collapsed (near-degenerate token usage)";
	}

	tokOpt.Reset();
	tok.Reset();
	ctx.Clear();
}

TEST(Alm, HumanMl3dLoads) {
	const OaString dsPath = "../dataset/gen/3d/anim/ds/HumanML3D/HumanML3D";
	OaDsHumanMl3d ds(dsPath, "train", /*InMaxClips=*/4);
	ASSERT_TRUE(ds.Ok()) << "Failed to load HumanML3D from " << dsPath.CStr();
	ASSERT_GE(ds.NumClips(), 1) << "HumanML3D dataset has no clips";
	EXPECT_EQ(ds.FeatDim(), 263) << "HumanML3D must use 263-dim SMPL-22 features";
	EXPECT_EQ(ds.NumJoints(), 22) << "HumanML3D must use 22-joint SMPL skeleton";
	for (OaI32 i = 0; i < ds.NumClips(); ++i) {
		const OaI32 frames = ds.ClipFrames(i);
		EXPECT_GT(frames, 0) << "Clip " << i << " has no frames";
		const OaF32* data = ds.ClipData(i);
		float first = data[0];
		EXPECT_TRUE(std::isfinite(first)) << "Clip " << i << " contains non-finite data";
		const auto& captions = ds.ClipCaptions(i);
		if (not captions.Empty()) {
			EXPECT_EQ(ds.ClipText(i), captions[0].Text);
			for (const auto& caption : captions) {
				EXPECT_FALSE(caption.Text.Empty());
				EXPECT_TRUE(std::isfinite(caption.StartSec));
				EXPECT_TRUE(std::isfinite(caption.EndSec));
			}
		}
	}
	std::printf("HumanML3D: loaded %d clips (%d-dim, %d-joint)\n", ds.NumClips(), ds.FeatDim(), ds.NumJoints());
}

TEST(Alm, CmpLoadsAllCaptions) {
	const OaString dsPath = "../dataset/gen/3d/anim/ds/Cmp";
	OaDsCmp ds(dsPath, "train", /*InMaxClips=*/8);
	ASSERT_TRUE(ds.Ok());
	ASSERT_GE(ds.NumClips(), 1);
	for (OaI32 i = 0; i < ds.NumClips(); ++i) {
		const auto& captions = ds.ClipCaptions(i);
		EXPECT_GE(captions.Size(), 3u) << "CMP clip " << ds.ClipId(i).CStr();
		ASSERT_FALSE(captions.Empty());
		EXPECT_EQ(ds.ClipText(i), captions[0].Text);
		for (const auto& caption : captions) {
			EXPECT_FALSE(caption.Text.Empty());
			EXPECT_TRUE(std::isfinite(caption.StartSec));
			EXPECT_TRUE(std::isfinite(caption.EndSec));
		}
	}
}

TEST(Alm, HumanMl3dReferenceInverse) {
	constexpr OaI32 frames = 2;
	constexpr OaI32 featDim = 263;
	constexpr OaI32 joints = 22;
	std::vector<OaF32> features(static_cast<size_t>(frames) * featDim, 0.0F);
	features[1] = 1.0F;  // frame-0 root X velocity, applied to frame 1
	features[2] = 2.0F;  // frame-0 root Z velocity, applied to frame 1
	features[3] = 0.5F;
	features[featDim + 3] = 0.75F;
	for (OaI32 t = 0; t < frames; ++t) {
		for (OaI32 j = 1; j < joints; ++j) {
			const size_t base = static_cast<size_t>(t) * featDim + 4
				+ static_cast<size_t>(j - 1) * 3;
			features[base] = static_cast<OaF32>(j) * 0.1F;
			features[base + 1] = static_cast<OaF32>(j) * 0.2F;
			features[base + 2] = static_cast<OaF32>(j) * -0.1F;
		}
	}

	auto world = OaHumanMl3dRecoverWorldJoints(
		OaSpan<const OaF32>(features.data(), features.size()), frames, featDim);
	ASSERT_EQ(world.Size(), static_cast<OaUsize>(frames * joints * 3));
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

	EXPECT_DOUBLE_EQ(OaHumanMl3dMpjpeCm(
		OaSpan<const OaF32>(world.Data(), world.Size()),
		OaSpan<const OaF32>(world.Data(), world.Size())), 0.0);
	auto shifted = world;
	for (OaUsize i = 0; i < shifted.Size(); i += 3) shifted[i] += 0.01F;
	EXPECT_NEAR(OaHumanMl3dMpjpeCm(
		OaSpan<const OaF32>(shifted.Data(), shifted.Size()),
		OaSpan<const OaF32>(world.Data(), world.Size())), 1.0, 1e-4);
}

TEST(Alm, MaskedCrossEntropyAutograd) {
	const std::vector<float> logitsHost = {
		1.0f, 2.0f, 3.0f, 3.0f, 1.0f, 0.0f,
		9.0f, 8.0f, 7.0f, -2.0f, 0.0f, 2.0f};
	const std::vector<OaI32> targetsHost = {2, 0, 0, 1};
	const std::vector<float> maskHost = {1.0f, 1.0f, 0.0f, 0.0f};
	auto logits = OaFnMatrix::FromBytes(
		OaSpan<const OaU8>(reinterpret_cast<const OaU8*>(logitsHost.data()),
			logitsHost.size() * sizeof(float)), OaMatrixShape{4, 3}, OaScalarType::Float32);
	auto targets = OaFnMatrix::FromBytes(
		OaSpan<const OaU8>(reinterpret_cast<const OaU8*>(targetsHost.data()),
			targetsHost.size() * sizeof(OaI32)), OaMatrixShape{4}, OaScalarType::Int32);
	auto mask = OaFnMatrix::FromBytes(
		OaSpan<const OaU8>(reinterpret_cast<const OaU8*>(maskHost.data()),
			maskHost.size() * sizeof(float)), OaMatrixShape{4}, OaScalarType::Float32);
	logits.SetRequiresGrad(true);
	OaGradientTape tape;
	auto loss = OaFnLoss::MaskedCrossEntropy(logits, targets, mask, 2);
	tape.Backward(loss);
	auto& ctx = OaContext::GetDefault();
	ASSERT_TRUE(ctx.Execute().IsOk());
	ASSERT_TRUE(ctx.Sync().IsOk());
	EXPECT_TRUE(std::isfinite(loss.At(0)));
	const auto& grad = logits.GradMatrix();
	for (OaI32 i = 2 * 3; i < 4 * 3; ++i) EXPECT_FLOAT_EQ(grad.At(i), 0.0f);
	for (OaI32 row = 0; row < 2; ++row) {
		float sum = 0.0f;
		for (OaI32 col = 0; col < 3; ++col) sum += grad.At(row * 3 + col);
		EXPECT_NEAR(sum, 0.0f, 1e-5f);
	}
	ctx.Clear();
}

TEST(Alm, LmLearnsCmpTokens) {
	OaFnMatrix::SetRngSeed(13);
	auto& ctx = OaContext::GetDefault();

	const OaString dsPath = "../dataset/gen/3d/anim/ds/Cmp";
	OaDsCmp ds(dsPath, "train", /*InMaxClips=*/128);
	ASSERT_TRUE(ds.Ok()) << "Failed to load CMP from " << dsPath.CStr();
	ASSERT_GE(ds.NumClips(), 128) << "Not enough clips in dataset";

	// Stage 1: train a tokenizer on the CMP corpus.
	auto tokCfg = CmpSmokeTokenizerCfg(ds.FeatDim());
	auto tok = OaMakeSharedPtr<OaAlmTokenizerAg>(tokCfg);

	const OaI32 B = 8;
	const OaI32 T = 64;
	const OaI32 tokLen = T / tok->DownsampleFactor();  // 16

	// Seed the codebook: need >= NumCodes latent rows. With factor=4 and
	// NumCodes=512, use B_seed=64 clips × T=64 frames → 1024 latents.
	const OaI32 Bseed = 64;
	{
		std::vector<float> seed(static_cast<size_t>(Bseed * T) * ds.FeatDim());
		for (OaI32 b = 0; b < Bseed; ++b) {
			const OaI32 clipIdx = b % ds.NumClips();
			const OaI32 frames = static_cast<OaI32>(ds.ClipFrames(clipIdx));
			const OaI32 start = frames > T ? (frames - T) / 2 : 0;
			const OaF32* src = ds.ClipData(clipIdx) + start * ds.FeatDim();
			float* dst = seed.data() + static_cast<size_t>(b) * T * ds.FeatDim();
			const OaI32 copyFrames = std::min(T, frames);
			std::memcpy(dst, src, static_cast<size_t>(copyFrames) * ds.FeatDim() * sizeof(float));
		}
		auto seedX = OaFnMatrix::FromBytes(
			OaSpan<const OaU8>(reinterpret_cast<const OaU8*>(seed.data()), seed.size() * sizeof(float)),
			OaMatrixShape{Bseed, T, ds.FeatDim()}, OaScalarType::Float32);
		auto z0 = tok->Encode(seedX, Bseed, T);
		tok->Seed(z0);
		ctx.Clear();
	}

	auto tokParams = tok->AllParameterPtrs();
	auto tokOpt = OaMakeUniquePtr<OaAdamW>(tokParams, 1e-4F, 0.9F, 0.99F, 1e-8F, 0.01F);
	for (OaI32 s = 1; s <= 500; ++s) {
		ctx.Clear();
		OaGradientTape tape; tokOpt->ZeroGrad();
		std::vector<float> batch(static_cast<size_t>(B * T) * ds.FeatDim());
		for (OaI32 b = 0; b < B; ++b) {
			const OaI32 clipIdx = b % ds.NumClips();
			const OaI32 frames = static_cast<OaI32>(ds.ClipFrames(clipIdx));
			const OaI32 start = frames > T ? (s * 17 + b * 31) % (frames - T) : 0;
			const OaF32* src = ds.ClipData(clipIdx) + start * ds.FeatDim();
			float* dst = batch.data() + static_cast<size_t>(b) * T * ds.FeatDim();
			const OaI32 copyFrames = std::min(T, frames);
			std::memcpy(dst, src, static_cast<size_t>(copyFrames) * ds.FeatDim() * sizeof(float));
		}
		auto X = OaFnMatrix::FromBytes(
			OaSpan<const OaU8>(reinterpret_cast<const OaU8*>(batch.data()), batch.size() * sizeof(float)),
			OaMatrixShape{B, T, ds.FeatDim()}, OaScalarType::Float32);
		auto z = tok->Encode(X, B, T);
		auto q = tok->Quantize(z);
		auto rec = tok->Decode(q.Quantized, B, tokLen);
		auto xFlat = X.Reshape(OaMatrixShape{static_cast<OaI64>(B) * T, ds.FeatDim()});
		auto recon = OaFnLoss::SmoothL1(rec, xFlat);

		// Velocity loss: SmoothL1 on frame-to-frame differences (improves motion smoothness).
		// LAMBDA_VELOCITY=0.5 (reference: config_h3d_stage2.yaml)
		auto rec3d   = rec.Reshape(OaMatrixShape{B, T, ds.FeatDim()});
		auto xFlat3d = X;  // already [B, T, D]
		auto recVel   = OaFnMatrix::Sub(OaFnMatrix::Slice(rec3d, 1, 1, T),   OaFnMatrix::Slice(rec3d, 1, 0, T - 1));
		auto xVel     = OaFnMatrix::Sub(OaFnMatrix::Slice(xFlat3d, 1, 1, T), OaFnMatrix::Slice(xFlat3d, 1, 0, T - 1));
		auto velLoss  = OaFnLoss::SmoothL1(recVel.Reshape(OaMatrixShape{static_cast<OaI64>(B) * (T - 1), ds.FeatDim()}),
		                                   xVel.Reshape(OaMatrixShape{static_cast<OaI64>(B) * (T - 1), ds.FeatDim()}));

		auto loss = recon + OaFnMatrix::Scale(velLoss, 0.5F) + q.CommitLoss;
		tape.Backward(loss);
		tokOpt->Step();
		tok->EmaUpdate(q);
		(void)ctx.Execute(); (void)ctx.Sync();
		ASSERT_TRUE(std::isfinite(loss.At(0))) << "Tokenizer diverged at step " << s;
	}
	tokOpt.Reset();
	ctx.Clear();

	// Stage 2: tokenize each long clip and collect token sequences.
	const OaI32 lmTokLen = 16;  // window of 16 tokens per sequence
	const OaI32 minFrames = lmTokLen * tok->DownsampleFactor();  // 128
	std::vector<std::vector<OaI32>> tokenSequences;
	for (OaI32 i = 0; i < ds.NumClips(); ++i) {
		const OaI32 frames = static_cast<OaI32>(ds.ClipFrames(i));
		if (frames < minFrames) continue;
		std::vector<float> clip(static_cast<size_t>(frames) * ds.FeatDim());
		std::memcpy(clip.data(), ds.ClipData(i), clip.size() * sizeof(float));
		auto x = OaFnMatrix::FromBytes(
			OaSpan<const OaU8>(reinterpret_cast<const OaU8*>(clip.data()), clip.size() * sizeof(float)),
			OaMatrixShape{1, frames, ds.FeatDim()}, OaScalarType::Float32);
		auto ids = tok->Tokenize(x, 1, frames)[0];
		(void)ctx.Execute(); (void)ctx.Sync();
		const OaI64 n = ids.NumElements();
		const OaI32* p = ids.DataAs<const OaI32>();
		tokenSequences.emplace_back(p, p + n);
		ctx.Clear();
	}
	ASSERT_GE(static_cast<OaI32>(tokenSequences.size()), B) << "Not enough long clips to build LM batches";
	std::printf("  [lm cmp] collected %zu token sequences (minFrames=%d)\n", tokenSequences.size(), minFrames);
	std::fflush(stdout);

	// Stage 3: train the AR transformer on sliding windows.
	// iGPU-sized SMOKE LM (reference is D=384/L=6/DFF=1536). The iGPU shares system
	// RAM, so a full transformer + autograd tape OOMs the box; this proves next-token
	// learning at a footprint that fits.
	OaAlmPriorConfig lmCfg;
	lmCfg.SyncVocab(tokCfg.NumCodes);
	lmCfg.DModel = 192; lmCfg.NumLayers = 3; lmCfg.DFfn = 512;
	lmCfg.SeqLen = lmTokLen + 1;
	auto lm = OaMakeSharedPtr<OaAlmPriorAg>(lmCfg);
	(void)ctx.Execute(); (void)ctx.Sync();  // flush LM initialization
	auto lmParams = lm->AllParameterPtrs();
	auto lmOpt = OaMakeUniquePtr<OaAdamW>(lmParams, 1e-4F, 0.9F, 0.99F, 1e-8F, 0.01F);

	OaF32 firstLoss = 0.0F;
	OaF32 lastLoss = 0.0F;
	for (OaI32 s = 1; s <= 500; ++s) {
		ctx.Clear();
		OaGradientTape tape; lmOpt->ZeroGrad();

		std::vector<OaI32> inputHost(static_cast<size_t>(B) * (lmTokLen + 1));
		std::vector<OaI32> targetHost(static_cast<size_t>(B) * (lmTokLen + 1));
		for (OaI32 b = 0; b < B; ++b) {
			const auto& seq = tokenSequences[(s + b) % tokenSequences.size()];
			const OaI32 maxStart = static_cast<OaI32>(seq.size()) - lmTokLen;
			const OaI32 start = maxStart > 0 ? (s * 23 + b * 17) % maxStart : 0;
			const size_t row = static_cast<size_t>(b) * (lmTokLen + 1);
			inputHost[row] = lmCfg.SomToken;
			for (OaI32 t = 0; t < lmTokLen; ++t) {
				const OaI32 id = static_cast<OaI32>(seq[start + t]);
				inputHost[row + 1 + t] = id;
				targetHost[row + t] = id;
			}
			targetHost[row + lmTokLen] = lmCfg.EomToken;
		}
		auto inputIds = OaFnMatrix::FromBytes(
			OaSpan<const OaU8>(reinterpret_cast<const OaU8*>(inputHost.data()), inputHost.size() * sizeof(OaU32)),
			OaMatrixShape{B, lmTokLen + 1}, OaScalarType::UInt32);
		auto targetIds = OaFnMatrix::FromBytes(
			OaSpan<const OaU8>(reinterpret_cast<const OaU8*>(targetHost.data()), targetHost.size() * sizeof(OaU32)),
			OaMatrixShape{B, lmTokLen + 1}, OaScalarType::UInt32);

		auto logits = lm->Forward(inputIds);
		auto logitsFlat = logits.Reshape(OaMatrixShape{static_cast<OaI64>(B) * (lmTokLen + 1), lmCfg.VocabSize});
		auto targetFlat = targetIds.Reshape(OaMatrixShape{static_cast<OaI64>(B) * (lmTokLen + 1)});
		auto ce = OaFnLoss::CrossEntropy(logitsFlat, targetFlat);
		tape.Backward(ce);
		lmOpt->Step();
		(void)ctx.Execute(); (void)ctx.Sync();

		const float lv = ce.At(0);
		if (s == 1) firstLoss = lv;
		lastLoss = lv;
		if (s == 1 || s % 50 == 0 || s == 200)
			std::printf("  [lm cmp] step %3d | ce %.8f\n", s, static_cast<double>(lv));
		ASSERT_TRUE(std::isfinite(lv)) << "LM diverged at step " << s;
	}
	std::printf("CMP LM cross-entropy: %.8f -> %.8f\n", static_cast<double>(firstLoss), static_cast<double>(lastLoss));
	EXPECT_LT(lastLoss, firstLoss) << "LM did not learn next-token prediction on CMP tokens";

	// Stage 4: end-to-end generation — sample multiple tokens streams and decode to motion.
	ctx.Clear();
	const float temperatures[] = {1.0F, 2.0F, 3.0F};
	std::vector<std::vector<OaI32>> genStreams;   // captured per-temp token streams (diversity guard)
	for (OaI32 g = 0; g < 3; ++g) {
		auto generated = lm->Generate(1, temperatures[g], 0, 0.9F, lmTokLen);
		auto motion = lm->DecodeToMotion(generated, *tok);
		(void)ctx.Execute(); (void)ctx.Sync();

		std::vector<OaI32> genHost(static_cast<size_t>(generated.NumElements()));
		(void)OaFnMatrix::CopyToHost(generated, genHost.data(), genHost.size() * sizeof(OaI32));
		genStreams.push_back(std::move(genHost));
		std::printf("  [lm cmp] generated motion %d (T=%.2f) shape: [%lld, %lld]\n", g,
			temperatures[g], static_cast<long long>(motion.Size(0)), static_cast<long long>(motion.Size(1)));
		EXPECT_EQ(motion.Size(1), ds.FeatDim()) << "Generated motion feature dim must match dataset";
		EXPECT_GT(motion.Size(0), 0) << "Generated motion must have frames";

		// Denormalize features and recover world joint positions for USD export.
		char pathBuf[128];
		const OaI32 frames = static_cast<OaI32>(motion.Size(0));
		const OaI32 featDim = ds.FeatDim();
		auto motionHost = HostFloatData(motion);
		std::vector<float> featHost(motionHost.Data(), motionHost.Data() + motionHost.Size());
		ds.Denormalize(featHost.data(), frames);
		auto worldJoints = OaHumanMl3dRecoverWorldJoints(
			OaSpan<const OaF32>(featHost.data(), featHost.size()), frames, featDim);
		auto skelClip = OaUsdClipFromWorldJoints(
			OaSkHumanMl3d(),
			OaSpan<const OaF32>(worldJoints.Data(), worldJoints.Size()),
			frames, 20.0F, 1, 100.0F);
		std::snprintf(pathBuf, sizeof(pathBuf),
			"var/alm/Alm_LmLearnsCmpTokens_generated_%d_T%.1f.usda", g, temperatures[g]);
		OaPath usdPath(pathBuf);
		auto usdSt = OaUsd::WriteUsda(usdPath, skelClip, "humanml3d");
		std::printf("  [lm cmp] saved generated skeleton %d to %s (%s)\n", g,
			usdPath.CStr(), usdSt.IsOk() ? "ok" : usdSt.ToString().CStr());
		EXPECT_TRUE(usdSt.IsOk()) << "Failed to write generated motion .usda";
	}

	// Diversity guard: sampling at rising temperatures must not collapse to one
	// clip. Count distinct generated streams — mode-collapse (identical tokens
	// regardless of temperature) would leave only 1.
	OaI32 distinct = 0;
	for (size_t i = 0; i < genStreams.size(); ++i) {
		bool isNew = true;
		for (size_t j = 0; j < i; ++j) { if (genStreams[j] == genStreams[i]) { isNew = false; break; } }
		if (isNew) ++distinct;
	}
	std::printf("  [lm cmp] generation diversity: %d/%zu distinct token streams across T=1/2/3\n",
		distinct, genStreams.size());
	EXPECT_GT(distinct, 1) << "generation mode-collapsed — all temperatures produced identical tokens";

	lmOpt.Reset();
	lm.Reset();
	tok.Reset();
	ctx.Clear();
}
