// Dense-Transformer component benchmark for the canonical NLP tutorial shape.
// Deliberately not registered with ctest: run it only for controlled profiling.

#include "../../OaTest.h"

#include <Oa/Core/PerfStat.h>
#include <Oa/Ml.h>
#include <Oa/Ml/Autograd.h>
#include <Oa/Runtime/Engine.h>
#include <Oa/Runtime/GpuTimer.h>

#include <cmath>
#include <cstdio>
#include <stdexcept>

namespace {

template <typename Enqueue>
OaPerfStat Measure(OaEngine& InEngine, const char* InName, Enqueue&& InEnqueue,
	OaU32 InWarmup = 5, OaU32 InSamples = 25) {
	OaGpuTimer timer;
	if (auto status = timer.Init(InEngine, InName); not status.IsOk()) {
		throw std::runtime_error("BenchTransformerComponents timer initialization failed");
	}
	OaPerfStat stat(InName, InSamples, InWarmup);
	auto& ctx = OaContext::GetDefault();
	for (OaU32 i = 0; i < InWarmup + InSamples; ++i) {
		InEnqueue();
		auto submitted = ctx.Submit(&timer);
		if (not submitted.IsOk()) {
			throw std::runtime_error("BenchTransformerComponents execution failed");
		}
		if (auto status = ctx.Wait(submitted.GetValue()); not status.IsOk()) {
			throw std::runtime_error("BenchTransformerComponents completion failed");
		}
		stat.Push(timer.ReadbackMs(InEngine.Device));
	}
	timer.Destroy(InEngine.Device);
	return stat;
}

void Print(const char* InName, const OaPerfStat& InStat) {
	std::printf("  %-28s mean=%7.4f ms  p50=%7.4f  p95=%7.4f\n",
		InName, InStat.Mean(), InStat.P50(), InStat.P95());
}

void MeasureAttentionForwardPair(
	OaEngine& InEngine, const char* InPrefix,
	OaI32 InBatch, OaI32 InHeads, OaI32 InSeqLen, OaI32 InHeadDim) {
	auto& ctx = OaContext::GetDefault();
	const OaI32 batchHeads = InBatch * InHeads;
	auto q = OaFnMatrix::RandN({batchHeads, InSeqLen, InHeadDim});
	auto k = OaFnMatrix::RandN({batchHeads, InSeqLen, InHeadDim});
	auto v = OaFnMatrix::RandN({batchHeads, InSeqLen, InHeadDim});
	auto mask = OaFnMatrix::CausalMask(OaFnMatrix::Zeros(
		{batchHeads, InSeqLen, InSeqLen}, OaScalarType::Float32));
	ASSERT_TRUE(ctx.Execute().IsOk());
	ASSERT_TRUE(ctx.Sync().IsOk());
	ctx.Clear();
	const OaF32 scale = 1.0F / std::sqrt(static_cast<OaF32>(InHeadDim));
	OaVec<OaMatrix> keep;
	char standardName[96], flashName[96], standardTimer[96], flashTimer[96];
	std::snprintf(standardName, sizeof(standardName), "%s standard forward", InPrefix);
	std::snprintf(flashName, sizeof(flashName), "%s flash forward", InPrefix);
	std::snprintf(standardTimer, sizeof(standardTimer), "%s_standard_forward", InPrefix);
	std::snprintf(flashTimer, sizeof(flashTimer), "%s_flash_forward", InPrefix);
	Print(standardName, Measure(InEngine, standardTimer, [&] {
		auto scores = OaFnMatrix::Bmm(q, OaFnMatrix::Transpose(k, 1, 2));
		auto probability = OaFnMatrix::SoftmaxScaledMasked(
			scores.Reshape({static_cast<OaI64>(batchHeads) * InSeqLen, InSeqLen}),
			mask.Reshape({static_cast<OaI64>(batchHeads) * InSeqLen, InSeqLen}), scale);
		keep = {OaFnMatrix::Bmm(
			probability.Reshape({batchHeads, InSeqLen, InSeqLen}), v)};
	}, 5, 25));
	Print(flashName, Measure(InEngine, flashTimer, [&] {
		keep = {OaFnMatrix::FlashAttentionCausal(q, k, v, scale)};
	}, 5, 25));
}

void MeasureAttentionTrainingPair(
	OaEngine& InEngine, const char* InPrefix,
	OaI32 InBatch, OaI32 InHeads, OaI32 InSeqLen, OaI32 InHeadDim) {
	auto& ctx = OaContext::GetDefault();
	const OaI32 batchHeads = InBatch * InHeads;
	auto q = OaFnMatrix::RandN({batchHeads, InSeqLen, InHeadDim}); q.SetRequiresGrad(true);
	auto k = OaFnMatrix::RandN({batchHeads, InSeqLen, InHeadDim}); k.SetRequiresGrad(true);
	auto v = OaFnMatrix::RandN({batchHeads, InSeqLen, InHeadDim}); v.SetRequiresGrad(true);
	auto mask = OaFnMatrix::CausalMask(OaFnMatrix::Zeros(
		{batchHeads, InSeqLen, InSeqLen}, OaScalarType::Float32));
	ASSERT_TRUE(ctx.Execute().IsOk());
	ASSERT_TRUE(ctx.Sync().IsOk());
	ctx.Clear();
	const OaF32 scale = 1.0F / std::sqrt(static_cast<OaF32>(InHeadDim));
	OaVec<OaMatrix> keep;
	char standardName[96], flashName[96], standardTimer[96], flashTimer[96];
	std::snprintf(standardName, sizeof(standardName), "%s standard train", InPrefix);
	std::snprintf(flashName, sizeof(flashName), "%s flash train", InPrefix);
	std::snprintf(standardTimer, sizeof(standardTimer), "%s_standard_train", InPrefix);
	std::snprintf(flashTimer, sizeof(flashTimer), "%s_flash_train", InPrefix);
	auto zeroLeaves = [&] { q.ZeroGrad(); k.ZeroGrad(); v.ZeroGrad(); };
	Print(standardName, Measure(InEngine, standardTimer, [&] {
		zeroLeaves();
		OaGradientTape tape;
		auto scores = OaFnMatrix::Bmm(q, OaFnMatrix::Transpose(k, 1, 2));
		auto probability = OaFnMatrix::SoftmaxScaledMasked(
			scores.Reshape({static_cast<OaI64>(batchHeads) * InSeqLen, InSeqLen}),
			mask.Reshape({static_cast<OaI64>(batchHeads) * InSeqLen, InSeqLen}), scale);
		auto output = OaFnMatrix::Bmm(
			probability.Reshape({batchHeads, InSeqLen, InSeqLen}), v);
		tape.Backward(OaFnMatrix::Mean(output));
		keep = {output, q.GradMatrix(), k.GradMatrix(), v.GradMatrix()};
	}, 2, 8));
	Print(flashName, Measure(InEngine, flashTimer, [&] {
		zeroLeaves();
		OaGradientTape tape;
		auto output = OaFnMatrix::FlashAttentionCausal(q, k, v, scale);
		tape.Backward(OaFnMatrix::Mean(output));
		keep = {output, q.GradMatrix(), k.GradMatrix(), v.GradMatrix()};
	}, 2, 8));
}

OaFnMatrix::OaLinearWeightBiasBwdResult LinearWeightBiasBwdTiledForBench(
	const OaMatrix& InInput, const OaMatrix& InGradOutput) {
	const OaU32 M = static_cast<OaU32>(InInput.Size(0));
	const OaU32 K = static_cast<OaU32>(InInput.Size(1));
	const OaU32 N = static_cast<OaU32>(InGradOutput.Size(1));
	auto dw = OaFnMatrix::Empty({N, K}, InInput.GetDtype());
	auto db = OaFnMatrix::Empty({N}, InGradOutput.GetDtype());
	struct Push { OaU32 M, N, K; } push{M, N, K};
	OaBufferAccess access[] = {OaBufferAccess::Read, OaBufferAccess::Read,
		OaBufferAccess::Write, OaBufferAccess::Write};
	OaContext::GetDefault().Add("LinearWeightBiasBwdTiled",
		{&InGradOutput, &InInput, &dw, &db}, access, &push, sizeof(push),
		(N + 31) / 32, (K + 31) / 32, 1);
	return {.GradWeight = dw, .GradBias = db};
}

OaFnMatrix::OaLinearWeightBiasBwdResult LinearWeightBiasBwdScalarForBench(
	const OaMatrix& InInput, const OaMatrix& InGradOutput) {
	const OaU32 M = static_cast<OaU32>(InInput.Size(0));
	const OaU32 K = static_cast<OaU32>(InInput.Size(1));
	const OaU32 N = static_cast<OaU32>(InGradOutput.Size(1));
	auto dw = OaFnMatrix::Empty({N, K}, InInput.GetDtype());
	auto db = OaFnMatrix::Empty({N}, InGradOutput.GetDtype());
	struct Push { OaU32 M, N, K, Total; } push{M, N, K, N * K + N};
	OaBufferAccess access[] = {OaBufferAccess::Read, OaBufferAccess::Read,
		OaBufferAccess::Write, OaBufferAccess::Write};
	OaContext::GetDefault().Add("LinearWeightBiasBwd",
		{&InGradOutput, &InInput, &dw, &db}, access, &push, sizeof(push),
		(push.Total + 255) / 256, 1, 1);
	return {.GradWeight = dw, .GradBias = db};
}

TEST(BenchTransformerComponents, AlmLinearShape) {
	if (not OaVkTestEngineOk()) GTEST_SKIP();
	auto& engine = *OaEngine::GetGlobal();
	auto& ctx = OaContext::GetDefault();
	OaContext::RecordingScope scope(ctx);
	OaGradNo noGrad;

	// Current Iris-Xe presentation configuration: B=64, 64 motion tokens plus
	// the two sequence boundary tokens, D=192 and DFF=512. This is deliberately
	// the live train shape rather than the dGPU-only 384/1536 reference model.
	constexpr OaI32 B = 64, S = 66, T = B * S, D = 192, FF = 512;
	auto x = OaFnMatrix::RandXavier({T, D});
	auto dModel = OaFnMatrix::RandXavier({T, D});
	auto dFfn = OaFnMatrix::RandXavier({T, FF});
	auto modelW = OaFnMatrix::RandXavier({D, D});
	auto ffnW = OaFnMatrix::RandXavier({FF, D});
	ASSERT_TRUE(ctx.Execute().IsOk());
	ASSERT_TRUE(ctx.Sync().IsOk());

	constexpr OaU32 warmup = 3, samples = 10;
	OaVec<OaMatrix> keep;
	Print("ALM Linear data backward", Measure(engine, "alm_linear_data_bwd", [&] {
		keep = {OaFnMatrix::LinearDataBwd(dModel, modelW)};
	}, warmup, samples));
	Print("ALM Linear param backward", Measure(engine, "alm_linear_param_bwd", [&] {
		auto dw = OaFnMatrix::LinearWeightBiasBwd(x, dModel);
		keep = {dw.GradWeight, dw.GradBias};
	}, warmup, samples));
	Print("ALM scalar param backward", Measure(engine, "alm_linear_param_bwd_scalar", [&] {
		auto dw = LinearWeightBiasBwdScalarForBench(x, dModel);
		keep = {dw.GradWeight, dw.GradBias};
	}, warmup, samples));
	Print("ALM tiled param backward", Measure(engine, "alm_linear_param_bwd_tiled", [&] {
		auto dw = LinearWeightBiasBwdTiledForBench(x, dModel);
		keep = {dw.GradWeight, dw.GradBias};
	}, warmup, samples));
	Print("ALM Q/K/V backward", Measure(engine, "alm_qkv_bwd", [&] {
		auto qdx = OaFnMatrix::LinearDataBwd(dModel, modelW);
		auto qdw = OaFnMatrix::LinearWeightBiasBwd(x, dModel);
		auto kdx = OaFnMatrix::LinearDataBwd(dModel, modelW);
		auto kdw = OaFnMatrix::LinearWeightBiasBwd(x, dModel);
		auto vdx = OaFnMatrix::LinearDataBwd(dModel, modelW);
		auto vdw = OaFnMatrix::LinearWeightBiasBwd(x, dModel);
		keep = {qdx, qdw.GradWeight, qdw.GradBias, kdx, kdw.GradWeight,
			kdw.GradBias, vdx, vdw.GradWeight, vdw.GradBias};
	}, warmup, samples));
	Print("ALM FFN data backward", Measure(engine, "alm_ffn_data_bwd", [&] {
		keep = {OaFnMatrix::LinearDataBwd(dFfn, ffnW)};
	}, warmup, samples));
	Print("ALM FFN param backward", Measure(engine, "alm_ffn_param_bwd", [&] {
		auto dw = OaFnMatrix::LinearWeightBiasBwd(x, dFfn);
		keep = {dw.GradWeight, dw.GradBias};
	}, warmup, samples));
	Print("ALM FFN tiled param bwd", Measure(engine, "alm_ffn_param_bwd_tiled", [&] {
		auto dw = LinearWeightBiasBwdTiledForBench(x, dFfn);
		keep = {dw.GradWeight, dw.GradBias};
	}, warmup, samples));
}

TEST(BenchTransformerComponents, LinearParamCrossover) {
	if (not OaVkTestEngineOk()) GTEST_SKIP();
	auto& engine = *OaEngine::GetGlobal();
	auto& ctx = OaContext::GetDefault();
	OaContext::RecordingScope scope(ctx);
	OaGradNo noGrad;
	struct Shape { OaI32 M, N, K; };
	constexpr Shape shapes[] = {
		{64, 192, 192}, {256, 192, 192}, {512, 192, 192},
		{1024, 32, 32}, {1024, 64, 32}, {1024, 64, 64},
		{1024, 96, 96}, {1024, 192, 192},
		{2048, 64, 64}, {2048, 64, 192}, {2048, 96, 96}, {2048, 192, 192},
		{3072, 64, 64}, {3072, 96, 96},
		{4224, 64, 64}, {4224, 96, 96}, {4224, 128, 128},
		{4224, 192, 192}, {4224, 512, 192},
	};
	for (const auto [M, N, K] : shapes) {
		auto x = OaFnMatrix::RandXavier({M, K});
		auto dy = OaFnMatrix::RandXavier({M, N});
		ASSERT_TRUE(ctx.Execute().IsOk());
		ASSERT_TRUE(ctx.Sync().IsOk());
		OaVec<OaMatrix> keep;
		char scalarName[64], tiledName[64], scalarTimer[64], tiledTimer[64];
		std::snprintf(scalarName, sizeof(scalarName), "scalar M%d N%d K%d", M, N, K);
		std::snprintf(tiledName, sizeof(tiledName), "tiled M%d N%d K%d", M, N, K);
		std::snprintf(scalarTimer, sizeof(scalarTimer), "linear_param_scalar_%d_%d_%d", M, N, K);
		std::snprintf(tiledTimer, sizeof(tiledTimer), "linear_param_tiled_%d_%d_%d", M, N, K);
		Print(scalarName, Measure(engine, scalarTimer, [&] {
			auto dw = LinearWeightBiasBwdScalarForBench(x, dy);
			keep = {dw.GradWeight, dw.GradBias};
		}, 2, 6));
		Print(tiledName, Measure(engine, tiledTimer, [&] {
			auto dw = LinearWeightBiasBwdTiledForBench(x, dy);
			keep = {dw.GradWeight, dw.GradBias};
		}, 2, 6));
	}
}
} // namespace

TEST(BenchTransformerComponents, NlpShape) {
	if (not OaVkTestEngineOk()) GTEST_SKIP();
	auto& engine = *OaEngine::GetGlobal();
	auto& ctx = OaContext::GetDefault();
	OaContext::RecordingScope scope(ctx);
	OaGradNo noGrad;

	constexpr OaI32 B = 64, S = 16, T = B * S, D = 32, FF = 64;
	constexpr OaI32 H = 1, BH = B * H, P = D / H;
	auto x = OaFnMatrix::RandXavier({T, D});
	auto lnW = OaFnMatrix::Ones({D});
	auto lnB = OaFnMatrix::Zeros({D});
	auto qW = OaFnMatrix::RandXavier({D, D});
	auto kW = OaFnMatrix::RandXavier({D, D});
	auto vW = OaFnMatrix::RandXavier({D, D});
	auto oW = OaFnMatrix::RandXavier({D, D});
	auto qB = OaFnMatrix::Zeros({D});
	auto kB = OaFnMatrix::Zeros({D});
	auto vB = OaFnMatrix::Zeros({D});
	auto oB = OaFnMatrix::Zeros({D});
	auto f1W = OaFnMatrix::RandXavier({FF, D});
	auto f1B = OaFnMatrix::Zeros({FF});
	auto f2W = OaFnMatrix::RandXavier({D, FF});
	auto f2B = OaFnMatrix::Zeros({D});

	auto q = OaFnMatrix::RandXavier({BH, S, P});
	auto k = OaFnMatrix::RandXavier({BH, S, P});
	auto v = OaFnMatrix::RandXavier({BH, S, P});
	auto scores = OaFnMatrix::RandXavier({BH, S, S});
	auto mask = OaFnMatrix::Zeros({BH * S, S});
	auto attn = OaFnMatrix::RandXavier({BH, S, S});
	auto context = OaFnMatrix::RandXavier({BH, S, P});
	auto ffPre = OaFnMatrix::RandXavier({T, FF});
	auto hidden = OaFnMatrix::RandXavier({T, FF});
	auto dFlat = OaFnMatrix::RandXavier({T, D});
	auto dHidden = OaFnMatrix::RandXavier({T, FF});
	auto dContext = OaFnMatrix::RandXavier({BH, S, P});
	auto dAttn = OaFnMatrix::RandXavier({BH * S, S});
	ASSERT_TRUE(ctx.Execute().IsOk());
	ASSERT_TRUE(ctx.Sync().IsOk());

	OaVec<OaMatrix> keep;
	Print("layer norm", Measure(engine, "transformer_ln", [&] {
		keep = {OaFnMatrix::LayerNorm(x, lnW, lnB, 1e-5F)};
	}));
	Print("three Q/K/V projections", Measure(engine, "transformer_qkv", [&] {
		keep = {OaFnMatrix::Linear(x, qW, qB), OaFnMatrix::Linear(x, kW, kB),
			OaFnMatrix::Linear(x, vW, vB)};
	}));
	Print("score transpose + BMM", Measure(engine, "transformer_score", [&] {
		auto kt = OaFnMatrix::Transpose(k, 1, 2);
		keep = {kt, OaFnMatrix::Bmm(q, kt)};
	}));
	Print("scaled masked softmax", Measure(engine, "transformer_softmax", [&] {
		keep = {OaFnMatrix::SoftmaxScaledMasked(
			scores.Reshape({BH * S, S}), mask, 1.0F / std::sqrt(static_cast<OaF32>(P)))};
	}));
	Print("context BMM", Measure(engine, "transformer_context", [&] {
		keep = {OaFnMatrix::Bmm(attn, v)};
	}));
	Print("output projection", Measure(engine, "transformer_out", [&] {
		keep = {OaFnMatrix::Linear(context.Reshape({T, D}), oW, oB)};
	}));
	Print("inference FFN LinearGelu", Measure(engine, "transformer_ffn1_infer", [&] {
		keep = {OaFnMatrix::LinearGelu(x, f1W, f1B)};
	}));
	Print("training FFN Linear + Gelu", Measure(engine, "transformer_ffn1_train", [&] {
		auto pre = OaFnMatrix::Linear(x, f1W, f1B);
		keep = {pre, OaFnMatrix::Gelu(pre)};
	}));
	Print("FFN down projection", Measure(engine, "transformer_ffn2", [&] {
		keep = {OaFnMatrix::Linear(hidden, f2W, f2B)};
	}));
	Print("residual add", Measure(engine, "transformer_residual", [&] {
		keep = {OaFnMatrix::Add(x, dFlat)};
	}));

	Print("complete attention", Measure(engine, "transformer_attention", [&] {
		auto xn = OaFnMatrix::LayerNorm(x, lnW, lnB, 1e-5F);
		auto q1 = OaFnMatrix::SplitHeads(OaFnMatrix::Linear(xn, qW, qB), B, S, H);
		auto k1 = OaFnMatrix::SplitHeads(OaFnMatrix::Linear(xn, kW, kB), B, S, H);
		auto v1 = OaFnMatrix::SplitHeads(OaFnMatrix::Linear(xn, vW, vB), B, S, H);
		auto score = OaFnMatrix::Bmm(q1, OaFnMatrix::Transpose(k1, 1, 2));
		auto prob = OaFnMatrix::SoftmaxScaledMasked(score.Reshape({BH * S, S}), mask,
			1.0F / std::sqrt(static_cast<OaF32>(P)));
		auto cv = OaFnMatrix::Bmm(prob.Reshape({BH, S, S}), v1);
		auto merged = OaFnMatrix::MergeHeads(cv, B, S, H);
		keep = {OaFnMatrix::Add(x, OaFnMatrix::Linear(merged, oW, oB))};
	}));
	Print("complete dense block", Measure(engine, "transformer_block", [&] {
		auto xn = OaFnMatrix::LayerNorm(x, lnW, lnB, 1e-5F);
		auto q1 = OaFnMatrix::SplitHeads(OaFnMatrix::Linear(xn, qW, qB), B, S, H);
		auto k1 = OaFnMatrix::SplitHeads(OaFnMatrix::Linear(xn, kW, kB), B, S, H);
		auto v1 = OaFnMatrix::SplitHeads(OaFnMatrix::Linear(xn, vW, vB), B, S, H);
		auto score = OaFnMatrix::Bmm(q1, OaFnMatrix::Transpose(k1, 1, 2));
		auto prob = OaFnMatrix::SoftmaxScaledMasked(score.Reshape({BH * S, S}), mask,
			1.0F / std::sqrt(static_cast<OaF32>(P)));
		auto cv = OaFnMatrix::Bmm(prob.Reshape({BH, S, S}), v1);
		auto merged = OaFnMatrix::MergeHeads(cv, B, S, H);
		auto residual = OaFnMatrix::Add(x, OaFnMatrix::Linear(merged, oW, oB));
		auto fn = OaFnMatrix::LayerNorm(residual, lnW, lnB, 1e-5F);
		auto ff = OaFnMatrix::LinearGelu(fn, f1W, f1B);
		keep = {OaFnMatrix::Add(residual, OaFnMatrix::Linear(ff, f2W, f2B))};
	}));

	Print("Q/K/V projection backward", Measure(engine, "transformer_qkv_bwd", [&] {
		auto qdx = OaFnMatrix::LinearDataBwd(dFlat, qW);
		auto qdw = OaFnMatrix::LinearWeightBiasBwd(x, dFlat);
		auto kdx = OaFnMatrix::LinearDataBwd(dFlat, kW);
		auto kdw = OaFnMatrix::LinearWeightBiasBwd(x, dFlat);
		auto vdx = OaFnMatrix::LinearDataBwd(dFlat, vW);
		auto vdw = OaFnMatrix::LinearWeightBiasBwd(x, dFlat);
		keep = {qdx, qdw.GradWeight, qdw.GradBias, kdx, kdw.GradWeight,
			kdw.GradBias, vdx, vdw.GradWeight, vdw.GradBias};
	}));
	Print("Linear data backward", Measure(engine, "transformer_linear_data_bwd", [&] {
		keep = {OaFnMatrix::LinearDataBwd(dFlat, qW)};
	}));
	Print("Linear weight+bias backward", Measure(engine, "transformer_linear_param_bwd", [&] {
		auto dw = OaFnMatrix::LinearWeightBiasBwd(x, dFlat);
		keep = {dw.GradWeight, dw.GradBias};
	}));
	Print("tiled weight+bias backward", Measure(engine, "transformer_linear_param_bwd_tiled", [&] {
		auto dw = LinearWeightBiasBwdTiledForBench(x, dFlat);
		keep = {dw.GradWeight, dw.GradBias};
	}));
	Print("attention core backward", Measure(engine, "transformer_attn_bwd", [&] {
		auto da = OaFnMatrix::Bmm(dContext, OaFnMatrix::Transpose(v, 1, 2));
		auto dv = OaFnMatrix::Bmm(OaFnMatrix::Transpose(attn, 1, 2), dContext);
		auto ds = OaFnMatrix::SoftmaxScaledMaskedBwd(
			attn.Reshape({BH * S, S}), dAttn, 1.0F / std::sqrt(static_cast<OaF32>(P)));
		auto ds3 = ds.Reshape({BH, S, S});
		auto dq = OaFnMatrix::Bmm(ds3, k);
		auto dk = OaFnMatrix::Bmm(OaFnMatrix::Transpose(ds3, 1, 2), q);
		keep = {da, dv, ds, dq, dk};
	}));
	Print("FFN1 backward saved pre", Measure(engine, "transformer_ffn1_bwd_saved", [&] {
		auto dz = OaFnMatrix::GeluBwd(ffPre, dHidden);
		auto dx = OaFnMatrix::LinearDataBwd(dz, f1W);
		auto dw = OaFnMatrix::LinearWeightBiasBwd(x, dz);
		keep = {dz, dx, dw.GradWeight, dw.GradBias};
	}));
	Print("legacy FFN1 bwd recompute", Measure(engine, "transformer_ffn1_bwd_recompute", [&] {
		auto pre = OaFnMatrix::Linear(x, f1W, f1B);
		auto dz = OaFnMatrix::GeluBwd(pre, dHidden);
		auto dx = OaFnMatrix::LinearDataBwd(dz, f1W);
		auto dw = OaFnMatrix::LinearWeightBiasBwd(x, dz);
		keep = {pre, dz, dx, dw.GradWeight, dw.GradBias};
	}));
}

TEST(BenchTransformerComponents, FlashAttentionForward) {
	if (not OaVkTestEngineOk()) GTEST_SKIP();
	auto& engine = *OaEngine::GetGlobal();
	auto& ctx = OaContext::GetDefault();
	OaContext::RecordingScope scope(ctx);
	OaGradNo noGrad;
	MeasureAttentionForwardPair(engine, "NLP B64 H1 S16 Dh32", 64, 1, 16, 32);
	MeasureAttentionForwardPair(engine, "B64 H1 S32 Dh32", 64, 1, 32, 32);
	MeasureAttentionForwardPair(engine, "B64 H1 S64 Dh32", 64, 1, 64, 32);
	MeasureAttentionForwardPair(engine, "ALM B64 H6 S66 Dh32", 64, 6, 66, 32);
}

TEST(BenchTransformerComponents, FlashAttentionTraining) {
	if (not OaVkTestEngineOk()) GTEST_SKIP();
	auto& engine = *OaEngine::GetGlobal();
	auto& ctx = OaContext::GetDefault();
	OaContext::RecordingScope scope(ctx);
	MeasureAttentionTrainingPair(engine, "NLP B64 H1 S16 Dh32", 64, 1, 16, 32);
	MeasureAttentionTrainingPair(engine, "ALM B64 H6 S66 Dh32", 64, 6, 66, 32);
}
