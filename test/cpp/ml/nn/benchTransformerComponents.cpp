// Dense-Transformer component benchmark for the canonical NLP tutorial shape.
// Deliberately not registered with ctest: run it only for controlled profiling.

#include "../../oaTest.h"

#include <oa/core/perfStat.h>
#include <oa/ml.h>
#include <oa/ml/autograd.h>
#include <oa/runtime/executionSession.h>
#include <oa/runtime/engine.h>
#include <oa/runtime/timer.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <utility>

namespace {

template <typename Enqueue>
oa::PerfStat measure(oa::Engine& inEngine, const char* inName, Enqueue&& inEnqueue,
	oa::U32 inWarmup = 5, oa::U32 inSamples = 25) {
	oa::Timer timer;
	if (auto status = timer.init(inEngine, inName); not status.isOk()) {
		throw std::runtime_error("BenchTransformerComponents timer initialization failed");
	}
	oa::PerfStat stat(inName, inSamples, inWarmup);
	auto& ctx = oa::ExecutionSession::getActive();
	for (oa::U32 i = 0; i < inWarmup + inSamples; ++i) {
		inEnqueue();
		auto submitted = ctx.submit(&timer);
		if (not submitted.isOk()) {
			throw std::runtime_error("BenchTransformerComponents execution failed");
		}
		if (auto status = ctx.wait(submitted.getValue()); not status.isOk()) {
			throw std::runtime_error("BenchTransformerComponents completion failed");
		}
		stat.push(*timer.commit(inEngine));
	}
	return stat;
}

void printStat(const char* inName, const oa::PerfStat& inStat) {
	std::printf("  %-28s mean=%7.4f ms  p50=%7.4f  p95=%7.4f\n",
		inName, inStat.mean(), inStat.p50(), inStat.p95());
}

void setEnvironmentFlag(const char* inName, bool inEnabled) {
#ifdef _WIN32
	_putenv_s(inName, inEnabled ? "1" : "");
#else
	if (inEnabled) setenv(inName, "1", 1);
	else unsetenv(inName);
#endif
}

struct FeaturePairStats {
	oa::PerfStat baseline;
	oa::PerfStat optimized;
	oa::PerfStat pairedGain;
};

template <typename Enqueue>
FeaturePairStats measureFeaturePair(
	oa::Engine& inEngine, Enqueue&& inEnqueue, const char* inDisableFlag,
	const char* inTimerName,
	oa::U32 inWarmup = 4, oa::U32 inSamples = 20) {
	oa::Timer timer;
	if (auto status = timer.init(inEngine, inTimerName);
		not status.isOk()) {
		throw std::runtime_error("feature-pair timer initialization failed");
	}
	FeaturePairStats stats {
		oa::PerfStat("transformer_block_baseline", inSamples, inWarmup),
		oa::PerfStat("transformer_block_optimized", inSamples, inWarmup),
		oa::PerfStat("transformer_block_paired_gain", inSamples, inWarmup),
	};
	auto& ctx = oa::ExecutionSession::getActive();
	auto run = [&](bool useOptimized, oa::PerfStat& stat) {
		setEnvironmentFlag(inDisableFlag, not useOptimized);
		inEnqueue();
		auto submitted = ctx.submit(&timer);
		if (not submitted.isOk()) {
			throw std::runtime_error("feature-pair execution failed");
		}
		if (auto status = ctx.wait(submitted.getValue()); not status.isOk()) {
			throw std::runtime_error("feature-pair completion failed");
		}
		const oa::F64 elapsedMs = *timer.commit(inEngine);
		stat.push(elapsedMs);
		return elapsedMs;
	};
	for (oa::U32 i = 0; i < inWarmup + inSamples; ++i) {
		oa::F64 baselineMs  = 0.0;
		oa::F64 optimizedMs = 0.0;
		if ((i & 1U) == 0U) {
			baselineMs  = run(false, stats.baseline);
			optimizedMs = run(true, stats.optimized);
		} else {
			optimizedMs = run(true, stats.optimized);
			baselineMs  = run(false, stats.baseline);
		}
		stats.pairedGain.push(100.0 * (baselineMs - optimizedMs) / baselineMs);
	}
	setEnvironmentFlag(inDisableFlag, false);
	return stats;
}

template <typename EnqueueBaseline, typename EnqueueOptimized>
FeaturePairStats measureExplicitPair(
	oa::Engine& inEngine, EnqueueBaseline&& inEnqueueBaseline,
	EnqueueOptimized&& inEnqueueOptimized, const char* inTimerName,
	oa::U32 inWarmup = 4, oa::U32 inSamples = 20) {
	oa::Timer timer;
	if (auto status = timer.init(inEngine, inTimerName);
		not status.isOk()) {
		throw std::runtime_error("explicit-pair timer initialization failed");
	}
	FeaturePairStats stats {
		oa::PerfStat("explicit_pair_baseline", inSamples, inWarmup),
		oa::PerfStat("explicit_pair_optimized", inSamples, inWarmup),
		oa::PerfStat("explicit_pair_gain", inSamples, inWarmup),
	};
	auto& ctx = oa::ExecutionSession::getActive();
	auto run = [&](auto&& enqueue, oa::PerfStat& stat) {
		enqueue();
		auto submitted = ctx.submit(&timer);
		if (not submitted.isOk()) {
			throw std::runtime_error("explicit-pair execution failed");
		}
		if (auto status = ctx.wait(submitted.getValue()); not status.isOk()) {
			throw std::runtime_error("explicit-pair completion failed");
		}
		const oa::F64 elapsedMs = *timer.commit(inEngine);
		stat.push(elapsedMs);
		return elapsedMs;
	};
	for (oa::U32 i = 0; i < inWarmup + inSamples; ++i) {
		oa::F64 baselineMs  = 0.0;
		oa::F64 optimizedMs = 0.0;
		if ((i & 1U) == 0U) {
			baselineMs = run(inEnqueueBaseline, stats.baseline);
			optimizedMs = run(inEnqueueOptimized, stats.optimized);
		} else {
			optimizedMs = run(inEnqueueOptimized, stats.optimized);
			baselineMs = run(inEnqueueBaseline, stats.baseline);
		}
		stats.pairedGain.push(100.0 * (baselineMs - optimizedMs) / baselineMs);
	}
	return stats;
}

void measureAttentionForwardPair(
	oa::Engine& inEngine, const char* inPrefix, const char* inMetricId,
	oa::I32 inBatch, oa::I32 inHeads, oa::I32 inSeqLen, oa::I32 inHeadDim) {
	auto& ctx = oa::ExecutionSession::getActive();
	const oa::I32 batchHeads = inBatch * inHeads;
	auto q = oa::FnMatrix::randN({batchHeads, inSeqLen, inHeadDim});
	auto k = oa::FnMatrix::randN({batchHeads, inSeqLen, inHeadDim});
	auto v = oa::FnMatrix::randN({batchHeads, inSeqLen, inHeadDim});
	auto mask = oa::FnMatrix::causalMask(oa::FnMatrix::zeros(
		{batchHeads, inSeqLen, inSeqLen}, oa::ScalarType::Float32));
	ASSERT_TRUE(testSubmitAndWait(ctx).isOk());
	ctx.clear();
	const oa::F32 scale = 1.0F / std::sqrt(static_cast<oa::F32>(inHeadDim));
	oa::Vector<oa::Matrix> keep;
	char standardName[96], flashName[96], flashTimer[96];
	std::snprintf(standardName, sizeof(standardName), "%s standard forward", inPrefix);
	std::snprintf(flashName, sizeof(flashName), "%s flash forward", inPrefix);
	std::snprintf(flashTimer, sizeof(flashTimer), "%s_flash_forward", inPrefix);
	auto enqueueStandard = [&] {
		auto scores = oa::FnMatrix::bmmNt(q, k);
		auto probability = oa::FnMatrix::softmaxScaledMasked(
			scores.reshape({static_cast<oa::I64>(batchHeads) * inSeqLen, inSeqLen}),
			mask.reshape({static_cast<oa::I64>(batchHeads) * inSeqLen, inSeqLen}), scale);
		keep = {oa::FnMatrix::bmm(
			probability.reshape({batchHeads, inSeqLen, inSeqLen}), v)};
	};
	auto enqueueFlash = [&] {
		keep = {oa::FnMatrix::flashAttentionCausal(q, k, v, scale)};
	};
	auto pair = measureExplicitPair(
		inEngine, enqueueStandard, enqueueFlash, flashTimer, 5, 25);
	printStat(standardName, pair.baseline);
	printStat(flashName, pair.optimized);
	std::printf(
		"  %-28s mean=%7.3f %%   p50=%7.3f  p95=%7.3f\n",
		"paired flash gain", pair.pairedGain.mean(),
		pair.pairedGain.p50(), pair.pairedGain.p95());
	std::printf(
		"OABENCH transformer.flash_forward.%s standard_p50_ms=%.6f "
		"flash_p50_ms=%.6f paired_flash_gain_p50_percent=%.3f\n",
		inMetricId, pair.baseline.p50(), pair.optimized.p50(),
		pair.pairedGain.p50());
}

void measureAttentionTrainingPair(
	oa::Engine& inEngine, const char* inPrefix,
	oa::I32 inBatch, oa::I32 inHeads, oa::I32 inSeqLen, oa::I32 inHeadDim) {
	auto& ctx = oa::ExecutionSession::getActive();
	const oa::I32 batchHeads = inBatch * inHeads;
	auto q = oa::FnMatrix::randN({batchHeads, inSeqLen, inHeadDim}); q.setRequiresGrad(true);
	auto k = oa::FnMatrix::randN({batchHeads, inSeqLen, inHeadDim}); k.setRequiresGrad(true);
	auto v = oa::FnMatrix::randN({batchHeads, inSeqLen, inHeadDim}); v.setRequiresGrad(true);
	auto mask = oa::FnMatrix::causalMask(oa::FnMatrix::zeros(
		{batchHeads, inSeqLen, inSeqLen}, oa::ScalarType::Float32));
	ASSERT_TRUE(testSubmitAndWait(ctx).isOk());
	ctx.clear();
	const oa::F32 scale = 1.0F / std::sqrt(static_cast<oa::F32>(inHeadDim));
	oa::Vector<oa::Matrix> keep;
	char standardName[96], flashName[96], standardTimer[96], flashTimer[96];
	std::snprintf(standardName, sizeof(standardName), "%s standard train", inPrefix);
	std::snprintf(flashName, sizeof(flashName), "%s flash train", inPrefix);
	std::snprintf(standardTimer, sizeof(standardTimer), "%s_standard_train", inPrefix);
	std::snprintf(flashTimer, sizeof(flashTimer), "%s_flash_train", inPrefix);
	auto zeroLeaves = [&] { q.zeroGrad(); k.zeroGrad(); v.zeroGrad(); };
	printStat(standardName, measure(inEngine, standardTimer, [&] {
		zeroLeaves();
		oa::GradientTape tape;
		auto scores = oa::FnMatrix::bmmNt(q, k);
		auto probability = oa::FnMatrix::softmaxScaledMasked(
			scores.reshape({static_cast<oa::I64>(batchHeads) * inSeqLen, inSeqLen}),
			mask.reshape({static_cast<oa::I64>(batchHeads) * inSeqLen, inSeqLen}), scale);
		auto output = oa::FnMatrix::bmm(
			probability.reshape({batchHeads, inSeqLen, inSeqLen}), v);
		tape.backward(oa::FnMatrix::mean(output));
		keep = {output, q.gradMatrix(), k.gradMatrix(), v.gradMatrix()};
	}, 2, 8));
	printStat(flashName, measure(inEngine, flashTimer, [&] {
		zeroLeaves();
		oa::GradientTape tape;
		auto output = oa::FnMatrix::flashAttentionCausal(q, k, v, scale);
		tape.backward(oa::FnMatrix::mean(output));
		keep = {output, q.gradMatrix(), k.gradMatrix(), v.gradMatrix()};
	}, 2, 8));
}

oa::LinearWeightBiasBwdResult linearWeightBiasBwdTiledForBench(
	const oa::Matrix& inInput, const oa::Matrix& inGradOutput) {
	const oa::U32 M = static_cast<oa::U32>(inInput.size(0));
	const oa::U32 K = static_cast<oa::U32>(inInput.size(1));
	const oa::U32 N = static_cast<oa::U32>(inGradOutput.size(1));
	auto dw = oa::FnMatrix::empty({N, K}, inInput.getDtype());
	auto db = oa::FnMatrix::empty({N}, inGradOutput.getDtype());
	struct Push { oa::U32 M, N, K; } push{M, N, K};
	oa::BufferAccess access[] = {oa::BufferAccess::Read, oa::BufferAccess::Read,
		oa::BufferAccess::Write, oa::BufferAccess::Write};
	oa::ExecutionSession::getActive().add( "LinearWeightBiasBwdTiled",
		{&inGradOutput, &inInput, &dw, &db}, access, &push, sizeof(push),
		(N + 31) / 32, (K + 31) / 32, 1);
	return {.gradWeight = dw, .gradBias = db};
}

oa::LinearWeightBiasBwdResult linearWeightBiasBwdScalarForBench(
	const oa::Matrix& inInput, const oa::Matrix& inGradOutput) {
	const oa::U32 M = static_cast<oa::U32>(inInput.size(0));
	const oa::U32 K = static_cast<oa::U32>(inInput.size(1));
	const oa::U32 N = static_cast<oa::U32>(inGradOutput.size(1));
	auto dw = oa::FnMatrix::empty({N, K}, inInput.getDtype());
	auto db = oa::FnMatrix::empty({N}, inGradOutput.getDtype());
	struct Push { oa::U32 M, N, K, total; } push{M, N, K, N * K + N};
	oa::BufferAccess access[] = {oa::BufferAccess::Read, oa::BufferAccess::Read,
		oa::BufferAccess::Write, oa::BufferAccess::Write};
	oa::ExecutionSession::getActive().add( "LinearWeightBiasBwd",
		{&inGradOutput, &inInput, &dw, &db}, access, &push, sizeof(push),
		(push.total + 255) / 256, 1, 1);
	return {.gradWeight = dw, .gradBias = db};
}

oa::LinearWeightBiasBwdResult linearWeightBiasBwdRows32ForBench(
	const oa::Matrix& inInput, const oa::Matrix& inGradOutput) {
	const oa::U32 M = static_cast<oa::U32>(inInput.size(0));
	const oa::U32 K = static_cast<oa::U32>(inInput.size(1));
	const oa::U32 N = static_cast<oa::U32>(inGradOutput.size(1));
	auto dw = oa::FnMatrix::empty({N, K}, inInput.getDtype());
	auto db = oa::FnMatrix::empty({N}, inGradOutput.getDtype());
	struct Push { oa::U32 M, N, K; } push{M, N, K};
	oa::BufferAccess access[] = {oa::BufferAccess::Read, oa::BufferAccess::Read,
		oa::BufferAccess::Write, oa::BufferAccess::Write};
	oa::ExecutionSession::getActive().add( "LinearWeightBiasBwdRows32",
		{&inGradOutput, &inInput, &dw, &db}, access, &push, sizeof(push),
		N, (K + 31) / 32, 1);
	return {.gradWeight = dw, .gradBias = db};
}

TEST(BenchTransformerComponents, AlmLinearShape) {
	if (not vkTestEngineOk()) GTEST_SKIP();
	auto& engine = testEngine();
	auto& ctx = oa::ExecutionSession::getActive();
	oa::ExecutionSession::RecordingScope scope(ctx);
	oa::GradNo noGrad;

	// current Iris-Xe presentation configuration: B=64, 64 motion tokens plus
	// the two sequence boundary tokens, D=192 and DFF=512. This is deliberately
	// the live train shape rather than the dGPU-only 384/1536 reference model.
	constexpr oa::I32 B = 64, S = 66, T = B * S, D = 192, FF = 512;
	auto x = oa::FnMatrix::randXavier({T, D});
	auto dModel = oa::FnMatrix::randXavier({T, D});
	auto dFfn = oa::FnMatrix::randXavier({T, FF});
	auto modelW = oa::FnMatrix::randXavier({D, D});
	auto ffnW = oa::FnMatrix::randXavier({FF, D});
	ASSERT_TRUE(testSubmitAndWait(ctx).isOk());

	constexpr oa::U32 warmup = 3, samples = 10;
	oa::Vector<oa::Matrix> keep;
	printStat("ALM Linear data backward", measure(engine, "alm_linear_data_bwd", [&] {
		keep = {oa::FnMatrix::linearDataBwd(dModel, modelW)};
	}, warmup, samples));
	printStat("ALM Linear param backward", measure(engine, "alm_linear_param_bwd", [&] {
		auto dw = oa::FnMatrix::linearWeightBiasBwd(x, dModel);
		keep = {dw.gradWeight, dw.gradBias};
	}, warmup, samples));
	printStat("ALM scalar param backward", measure(engine, "alm_linear_param_bwd_scalar", [&] {
		auto dw = linearWeightBiasBwdScalarForBench(x, dModel);
		keep = {dw.gradWeight, dw.gradBias};
	}, warmup, samples));
	printStat("ALM tiled param backward", measure(engine, "alm_linear_param_bwd_tiled", [&] {
		auto dw = linearWeightBiasBwdTiledForBench(x, dModel);
		keep = {dw.gradWeight, dw.gradBias};
	}, warmup, samples));
	printStat("ALM rows32 param backward", measure(engine, "alm_linear_param_bwd_rows32", [&] {
		auto dw = linearWeightBiasBwdRows32ForBench(x, dModel);
		keep = {dw.gradWeight, dw.gradBias};
	}, warmup, samples));
	printStat("ALM Q/K/V backward", measure(engine, "alm_qkv_bwd", [&] {
		auto qdx = oa::FnMatrix::linearDataBwd(dModel, modelW);
		auto qdw = oa::FnMatrix::linearWeightBiasBwd(x, dModel);
		auto kdx = oa::FnMatrix::linearDataBwd(dModel, modelW);
		auto kdw = oa::FnMatrix::linearWeightBiasBwd(x, dModel);
		auto vdx = oa::FnMatrix::linearDataBwd(dModel, modelW);
		auto vdw = oa::FnMatrix::linearWeightBiasBwd(x, dModel);
		keep = {qdx, qdw.gradWeight, qdw.gradBias, kdx, kdw.gradWeight,
			kdw.gradBias, vdx, vdw.gradWeight, vdw.gradBias};
	}, warmup, samples));
	printStat("ALM FFN data backward", measure(engine, "alm_ffn_data_bwd", [&] {
		keep = {oa::FnMatrix::linearDataBwd(dFfn, ffnW)};
	}, warmup, samples));
	printStat("ALM FFN param backward", measure(engine, "alm_ffn_param_bwd", [&] {
		auto dw = oa::FnMatrix::linearWeightBiasBwd(x, dFfn);
		keep = {dw.gradWeight, dw.gradBias};
	}, warmup, samples));
	printStat("ALM FFN tiled param bwd", measure(engine, "alm_ffn_param_bwd_tiled", [&] {
		auto dw = linearWeightBiasBwdTiledForBench(x, dFfn);
		keep = {dw.gradWeight, dw.gradBias};
	}, warmup, samples));
	printStat("ALM FFN rows32 param bwd", measure(engine, "alm_ffn_param_bwd_rows32", [&] {
		auto dw = linearWeightBiasBwdRows32ForBench(x, dFfn);
		keep = {dw.gradWeight, dw.gradBias};
	}, warmup, samples));
}

TEST(BenchTransformerComponents, LinearParamCrossover) {
	if (not vkTestEngineOk()) GTEST_SKIP();
	auto& engine = testEngine();
	auto& ctx = oa::ExecutionSession::getActive();
	oa::ExecutionSession::RecordingScope scope(ctx);
	oa::GradNo noGrad;
	struct Shape { oa::I32 M, N, K; };
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
		auto x = oa::FnMatrix::randXavier({M, K});
		auto dy = oa::FnMatrix::randXavier({M, N});
		ASSERT_TRUE(testSubmitAndWait(ctx).isOk());
		oa::Vector<oa::Matrix> keep;
		char scalarName[64], tiledName[64], rowsName[64];
		char scalarTimer[64], tiledTimer[64], rowsTimer[64];
		std::snprintf(scalarName, sizeof(scalarName), "scalar M%d N%d K%d", M, N, K);
		std::snprintf(tiledName, sizeof(tiledName), "tiled M%d N%d K%d", M, N, K);
		std::snprintf(rowsName, sizeof(rowsName), "rows32 M%d N%d K%d", M, N, K);
		std::snprintf(scalarTimer, sizeof(scalarTimer), "linear_param_scalar_%d_%d_%d", M, N, K);
		std::snprintf(tiledTimer, sizeof(tiledTimer), "linear_param_tiled_%d_%d_%d", M, N, K);
		std::snprintf(rowsTimer, sizeof(rowsTimer), "linear_param_rows32_%d_%d_%d", M, N, K);
		printStat(scalarName, measure(engine, scalarTimer, [&] {
			auto dw = linearWeightBiasBwdScalarForBench(x, dy);
			keep = {dw.gradWeight, dw.gradBias};
		}, 2, 6));
		printStat(tiledName, measure(engine, tiledTimer, [&] {
			auto dw = linearWeightBiasBwdTiledForBench(x, dy);
			keep = {dw.gradWeight, dw.gradBias};
		}, 2, 6));
		printStat(rowsName, measure(engine, rowsTimer, [&] {
			auto dw = linearWeightBiasBwdRows32ForBench(x, dy);
			keep = {dw.gradWeight, dw.gradBias};
		}, 2, 6));
	}
}
} // namespace

TEST(BenchTransformerComponents, NlpShape) {
	if (not vkTestEngineOk()) GTEST_SKIP();
	auto& engine = testEngine();
	auto& ctx = oa::ExecutionSession::getActive();
	oa::ExecutionSession::RecordingScope scope(ctx);
	oa::GradNo noGrad;

	constexpr oa::I32 B = 64, S = 16, T = B * S, D = 32, FF = 64;
	constexpr oa::I32 H = 1, BH = B * H, P = D / H;
	auto x = oa::FnMatrix::randXavier({T, D});
	auto lnW = oa::FnMatrix::ones({D});
	auto lnB = oa::FnMatrix::zeros({D});
	auto qW = oa::FnMatrix::randXavier({D, D});
	auto kW = oa::FnMatrix::randXavier({D, D});
	auto vW = oa::FnMatrix::randXavier({D, D});
	auto oW = oa::FnMatrix::randXavier({D, D});
	auto qB = oa::FnMatrix::zeros({D});
	auto kB = oa::FnMatrix::zeros({D});
	auto vB = oa::FnMatrix::zeros({D});
	auto oB = oa::FnMatrix::zeros({D});
	auto f1W = oa::FnMatrix::randXavier({FF, D});
	auto f1B = oa::FnMatrix::zeros({FF});
	auto f2W = oa::FnMatrix::randXavier({D, FF});
	auto f2B = oa::FnMatrix::zeros({D});

	auto q = oa::FnMatrix::randXavier({BH, S, P});
	auto k = oa::FnMatrix::randXavier({BH, S, P});
	auto v = oa::FnMatrix::randXavier({BH, S, P});
	auto scores = oa::FnMatrix::randXavier({BH, S, S});
	auto mask = oa::FnMatrix::zeros({BH * S, S});
	auto attn = oa::FnMatrix::randXavier({BH, S, S});
	auto context = oa::FnMatrix::randXavier({BH, S, P});
	auto ffPre = oa::FnMatrix::randXavier({T, FF});
	auto hidden = oa::FnMatrix::randXavier({T, FF});
	auto dFlat = oa::FnMatrix::randXavier({T, D});
	auto dHidden = oa::FnMatrix::randXavier({T, FF});
	auto dContext = oa::FnMatrix::randXavier({BH, S, P});
	auto dAttn = oa::FnMatrix::randXavier({BH * S, S});
	ASSERT_TRUE(testSubmitAndWait(ctx).isOk());

	oa::Vector<oa::Matrix> keep;
	printStat("layer norm", measure(engine, "transformer_ln", [&] {
		keep = {oa::FnMatrix::layerNorm(x, lnW, lnB, 1e-5F)};
	}));
	printStat("three Q/K/V projections", measure(engine, "transformer_qkv", [&] {
		keep = {oa::FnMatrix::linear(x, qW, qB), oa::FnMatrix::linear(x, kW, kB),
			oa::FnMatrix::linear(x, vW, vB)};
	}));
	printStat("score transpose + BMM", measure(engine, "transformer_score", [&] {
		auto kt = oa::FnMatrix::transpose(k, 1, 2);
		keep = {kt, oa::FnMatrix::bmm(q, kt)};
	}));
	printStat("score direct BMM NT", measure(engine, "transformer_score_nt", [&] {
		keep = {oa::FnMatrix::bmmNt(q, k)};
	}));
	auto scorePair = measureExplicitPair(
		engine,
		[&] {
			auto kt = oa::FnMatrix::transpose(k, 1, 2);
			keep = {kt, oa::FnMatrix::bmm(q, kt)};
		},
		[&] { keep = {oa::FnMatrix::bmmNt(q, k)}; },
		"transformer_score_bmm_nt_pair", 5, 25);
	printStat("score transpose baseline", scorePair.baseline);
	printStat("score direct BMM NT pair", scorePair.optimized);
	std::printf(
		"  %-28s mean=%7.3f %%   p50=%7.3f  p95=%7.3f\n",
		"paired direct-BMM gain", scorePair.pairedGain.mean(),
		scorePair.pairedGain.p50(), scorePair.pairedGain.p95());
	std::printf(
		"OABENCH transformer.bmm_nt baseline_p50_ms=%.6f "
		"optimized_p50_ms=%.6f paired_gain_p50_percent=%.3f\n",
		scorePair.baseline.p50(), scorePair.optimized.p50(),
		scorePair.pairedGain.p50());
	printStat("scaled masked softmax", measure(engine, "transformer_softmax", [&] {
		keep = {oa::FnMatrix::softmaxScaledMasked(
			scores.reshape({BH * S, S}), mask, 1.0F / std::sqrt(static_cast<oa::F32>(P)))};
	}));
	printStat("context BMM", measure(engine, "transformer_context", [&] {
		keep = {oa::FnMatrix::bmm(attn, v)};
	}));
	printStat("output projection", measure(engine, "transformer_out", [&] {
		keep = {oa::FnMatrix::linear(context.reshape({T, D}), oW, oB)};
	}));
	printStat("inference FFN LinearGelu", measure(engine, "transformer_ffn1_infer", [&] {
		keep = {oa::FnMatrix::linearGelu(x, f1W, f1B)};
	}));
	printStat("training FFN Linear + Gelu", measure(engine, "transformer_ffn1_train", [&] {
		auto pre = oa::FnMatrix::linear(x, f1W, f1B);
		keep = {pre, oa::FnMatrix::gelu(pre)};
	}));
	printStat("FFN down projection", measure(engine, "transformer_ffn2", [&] {
		keep = {oa::FnMatrix::linear(hidden, f2W, f2B)};
	}));
	printStat("residual add", measure(engine, "transformer_residual", [&] {
		keep = {oa::FnMatrix::add(x, dFlat)};
	}));

	printStat("complete attention", measure(engine, "transformer_attention", [&] {
		auto xn = oa::FnMatrix::layerNorm(x, lnW, lnB, 1e-5F);
		auto q1 = oa::FnMatrix::splitHeads(oa::FnMatrix::linear(xn, qW, qB), B, S, H);
		auto k1 = oa::FnMatrix::splitHeads(oa::FnMatrix::linear(xn, kW, kB), B, S, H);
		auto v1 = oa::FnMatrix::splitHeads(oa::FnMatrix::linear(xn, vW, vB), B, S, H);
		auto score = oa::FnMatrix::bmm(q1, oa::FnMatrix::transpose(k1, 1, 2));
		auto prob = oa::FnMatrix::softmaxScaledMasked(score.reshape({BH * S, S}), mask,
			1.0F / std::sqrt(static_cast<oa::F32>(P)));
		auto cv = oa::FnMatrix::bmm(prob.reshape({BH, S, S}), v1);
		auto merged = oa::FnMatrix::mergeHeads(cv, B, S, H);
		keep = {oa::FnMatrix::add(x, oa::FnMatrix::linear(merged, oW, oB))};
	}));
	auto enqueueDenseBlock = [&] {
		auto xn = oa::FnMatrix::layerNorm(x, lnW, lnB, 1e-5F);
		auto q1 = oa::FnMatrix::splitHeads(oa::FnMatrix::linear(xn, qW, qB), B, S, H);
		auto k1 = oa::FnMatrix::splitHeads(oa::FnMatrix::linear(xn, kW, kB), B, S, H);
		auto v1 = oa::FnMatrix::splitHeads(oa::FnMatrix::linear(xn, vW, vB), B, S, H);
		auto score = oa::FnMatrix::bmm(q1, oa::FnMatrix::transpose(k1, 1, 2));
		auto prob = oa::FnMatrix::softmaxScaledMasked(score.reshape({BH * S, S}), mask,
			1.0F / std::sqrt(static_cast<oa::F32>(P)));
		auto cv = oa::FnMatrix::bmm(prob.reshape({BH, S, S}), v1);
		auto merged = oa::FnMatrix::mergeHeads(cv, B, S, H);
		auto residual = oa::FnMatrix::add(x, oa::FnMatrix::linear(merged, oW, oB));
		auto fn = oa::FnMatrix::layerNorm(residual, lnW, lnB, 1e-5F);
		auto ff = oa::FnMatrix::linearGelu(fn, f1W, f1B);
		keep = {oa::FnMatrix::add(residual, oa::FnMatrix::linear(ff, f2W, f2B))};
	};
	printStat("complete dense block", measure(
		engine, "transformer_block", enqueueDenseBlock));

	auto enqueueQkvBackward = [&] {
		auto qdx = oa::FnMatrix::linearDataBwd(dFlat, qW);
		auto qdw = oa::FnMatrix::linearWeightBiasBwd(x, dFlat);
		auto kdx = oa::FnMatrix::linearDataBwd(dFlat, kW);
		auto kdw = oa::FnMatrix::linearWeightBiasBwd(x, dFlat);
		auto vdx = oa::FnMatrix::linearDataBwd(dFlat, vW);
		auto vdw = oa::FnMatrix::linearWeightBiasBwd(x, dFlat);
		keep = {qdx, qdw.gradWeight, qdw.gradBias, kdx, kdw.gradWeight,
			kdw.gradBias, vdx, vdw.gradWeight, vdw.gradBias};
	};
	printStat("Q/K/V projection backward", measure(
		engine, "transformer_qkv_bwd", enqueueQkvBackward));
	auto qkvBackwardPair = measureFeaturePair(
		engine, enqueueQkvBackward, "OA_DISABLE_LINEAR_PARAM_ROWS32",
		"transformer_qkv_bwd_rows32_pair");
	printStat("Q/K/V backward scalar", qkvBackwardPair.baseline);
	printStat("Q/K/V backward rows32", qkvBackwardPair.optimized);
	std::printf(
		"  %-28s mean=%7.3f %%   p50=%7.3f  p95=%7.3f\n",
		"paired rows32 gain", qkvBackwardPair.pairedGain.mean(),
		qkvBackwardPair.pairedGain.p50(), qkvBackwardPair.pairedGain.p95());
	std::printf(
		"OABENCH transformer.linear_param_rows32 baseline_p50_ms=%.6f "
		"optimized_p50_ms=%.6f paired_gain_p50_percent=%.3f\n",
		qkvBackwardPair.baseline.p50(), qkvBackwardPair.optimized.p50(),
		qkvBackwardPair.pairedGain.p50());
	printStat("Linear data backward", measure(engine, "transformer_linear_data_bwd", [&] {
		keep = {oa::FnMatrix::linearDataBwd(dFlat, qW)};
	}));
	printStat("Linear weight+bias backward", measure(engine, "transformer_linear_param_bwd", [&] {
		auto dw = oa::FnMatrix::linearWeightBiasBwd(x, dFlat);
		keep = {dw.gradWeight, dw.gradBias};
	}));
	printStat("tiled weight+bias backward", measure(engine, "transformer_linear_param_bwd_tiled", [&] {
		auto dw = linearWeightBiasBwdTiledForBench(x, dFlat);
		keep = {dw.gradWeight, dw.gradBias};
	}));
	printStat("rows32 weight+bias backward", measure(engine, "transformer_linear_param_bwd_rows32", [&] {
		auto dw = linearWeightBiasBwdRows32ForBench(x, dFlat);
		keep = {dw.gradWeight, dw.gradBias};
	}));
	printStat("attention core backward", measure(engine, "transformer_attn_bwd", [&] {
		auto da = oa::FnMatrix::bmm(dContext, oa::FnMatrix::transpose(v, 1, 2));
		auto dv = oa::FnMatrix::bmm(oa::FnMatrix::transpose(attn, 1, 2), dContext);
		auto ds = oa::FnMatrix::softmaxScaledMaskedBwd(
			attn.reshape({BH * S, S}), dAttn, 1.0F / std::sqrt(static_cast<oa::F32>(P)));
		auto ds3 = ds.reshape({BH, S, S});
		auto dq = oa::FnMatrix::bmm(ds3, k);
		auto dk = oa::FnMatrix::bmm(oa::FnMatrix::transpose(ds3, 1, 2), q);
		keep = {da, dv, ds, dq, dk};
	}));
	printStat("FFN1 backward saved pre", measure(engine, "transformer_ffn1_bwd_saved", [&] {
		auto dz = oa::FnMatrix::geluBwd(ffPre, dHidden);
		auto dx = oa::FnMatrix::linearDataBwd(dz, f1W);
		auto dw = oa::FnMatrix::linearWeightBiasBwd(x, dz);
		keep = {dz, dx, dw.gradWeight, dw.gradBias};
	}));
	printStat("legacy FFN1 bwd recompute", measure(engine, "transformer_ffn1_bwd_recompute", [&] {
		auto pre = oa::FnMatrix::linear(x, f1W, f1B);
		auto dz = oa::FnMatrix::geluBwd(pre, dHidden);
		auto dx = oa::FnMatrix::linearDataBwd(dz, f1W);
		auto dw = oa::FnMatrix::linearWeightBiasBwd(x, dz);
		keep = {pre, dz, dx, dw.gradWeight, dw.gradBias};
	}));

	if (std::getenv("OA_DISABLE_GEMM_ROUTE_CACHE") == nullptr) {
		auto cachePair = measureFeaturePair(
			engine, enqueueDenseBlock, "OA_DISABLE_GEMM_ROUTE_CACHE",
			"transformer_gemm_cache_pair");
		printStat("dense block heuristic", cachePair.baseline);
		printStat("dense block measured cache", cachePair.optimized);
		std::printf(
			"  %-28s mean=%7.3f %%   p50=%7.3f  p95=%7.3f\n",
			"paired route-cache gain", cachePair.pairedGain.mean(),
			cachePair.pairedGain.p50(), cachePair.pairedGain.p95());
		std::printf(
			"OABENCH transformer.gemm_cache heuristic_p50_ms=%.6f "
			"measured_p50_ms=%.6f paired_gain_p50_percent=%.3f\n",
			cachePair.baseline.p50(), cachePair.optimized.p50(),
			cachePair.pairedGain.p50());
	}

	if (std::getenv("OA_DISABLE_NARROW_ROW_KERNELS") == nullptr) {
		auto narrowPair = measureFeaturePair(
			engine, enqueueDenseBlock, "OA_DISABLE_NARROW_ROW_KERNELS",
			"transformer_narrow_row_pair");
		printStat("dense block general rows", narrowPair.baseline);
		printStat("dense block narrow rows", narrowPair.optimized);
		std::printf(
			"  %-28s mean=%7.3f %%   p50=%7.3f  p95=%7.3f\n",
			"paired narrow-row gain", narrowPair.pairedGain.mean(),
			narrowPair.pairedGain.p50(), narrowPair.pairedGain.p95());
		std::printf(
			"OABENCH transformer.narrow_rows baseline_p50_ms=%.6f "
			"optimized_p50_ms=%.6f paired_gain_p50_percent=%.3f\n",
			narrowPair.baseline.p50(), narrowPair.optimized.p50(),
			narrowPair.pairedGain.p50());
	}

	if (std::getenv("OA_DISABLE_TILED_BMM") == nullptr) {
		auto bmmPair = measureFeaturePair(
			engine, enqueueDenseBlock, "OA_DISABLE_TILED_BMM",
			"transformer_tiled_bmm_pair");
		printStat("dense block scalar BMM", bmmPair.baseline);
		printStat("dense block tiled BMM", bmmPair.optimized);
		std::printf(
			"  %-28s mean=%7.3f %%   p50=%7.3f  p95=%7.3f\n",
			"paired tiled-BMM gain", bmmPair.pairedGain.mean(),
			bmmPair.pairedGain.p50(), bmmPair.pairedGain.p95());
		std::printf(
			"OABENCH transformer.tiled_bmm baseline_p50_ms=%.6f "
			"optimized_p50_ms=%.6f paired_gain_p50_percent=%.3f\n",
			bmmPair.baseline.p50(), bmmPair.optimized.p50(),
			bmmPair.pairedGain.p50());
	}
}

TEST(BenchTransformerComponents, FlashAttentionForward) {
	if (not vkTestEngineOk()) GTEST_SKIP();
	auto& engine = testEngine();
	auto& ctx = oa::ExecutionSession::getActive();
	oa::ExecutionSession::RecordingScope scope(ctx);
	oa::GradNo noGrad;
	measureAttentionForwardPair(
		engine, "NLP B64 H1 S16 Dh32", "b64_h1_s16_dh32", 64, 1, 16, 32);
	measureAttentionForwardPair(
		engine, "B64 H1 S32 Dh32", "b64_h1_s32_dh32", 64, 1, 32, 32);
	measureAttentionForwardPair(
		engine, "B64 H1 S64 Dh32", "b64_h1_s64_dh32", 64, 1, 64, 32);
	measureAttentionForwardPair(
		engine, "ALM B64 H6 S66 Dh32", "b64_h6_s66_dh32", 64, 6, 66, 32);
	measureAttentionForwardPair(
		engine, "B16 H1 S128 Dh32", "b16_h1_s128_dh32", 16, 1, 128, 32);
	measureAttentionForwardPair(
		engine, "B4 H1 S256 Dh32", "b4_h1_s256_dh32", 4, 1, 256, 32);
	measureAttentionForwardPair(
		engine, "B1 H1 S512 Dh32", "b1_h1_s512_dh32", 1, 1, 512, 32);
}

TEST(BenchTransformerComponents, FlashAttentionTraining) {
	if (not vkTestEngineOk()) GTEST_SKIP();
	auto& engine = testEngine();
	auto& ctx = oa::ExecutionSession::getActive();
	oa::ExecutionSession::RecordingScope scope(ctx);
	measureAttentionTrainingPair(engine, "NLP B64 H1 S16 Dh32", 64, 1, 16, 32);
	measureAttentionTrainingPair(engine, "ALM B64 H6 S66 Dh32", 64, 6, 66, 32);
}
