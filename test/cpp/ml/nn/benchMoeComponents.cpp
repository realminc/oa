// Sparse-MoE component benchmark for the canonical NLP tutorial shape.

#include "../../oaTest.h"

#include <oa/core/perfStat.h>
#include <oa/ml.h>
#include <oa/ml/autograd.h>
#include <oa/ml/fnOptim.h>
#include <oa/runtime/engine.h>
#include <oa/runtime/timer.h>

#include <cstdio>
#include <stdexcept>
#include <vector>

namespace {

template <typename Enqueue>
oa::PerfStat measure(oa::Engine& inEngine, const char* inName, Enqueue&& inEnqueue) {
	constexpr oa::U32 warmup = 5, samples = 25;
	oa::Timer timer;
	if (auto status = timer.init(inEngine, inName); not status.isOk())
		throw std::runtime_error("BenchMoeComponents timer initialization failed");
	oa::PerfStat stat(inName, samples, warmup);
	auto& ctx = oa::ExecutionSession::getActive();
	for (oa::U32 i = 0; i < warmup + samples; ++i) {
		inEnqueue();
		auto submitted = ctx.submit(&timer);
		if (not submitted.isOk())
			throw std::runtime_error("BenchMoeComponents execution failed");
		if (auto status = ctx.wait(submitted.getValue()); not status.isOk())
			throw std::runtime_error("BenchMoeComponents completion failed");
		stat.push(*timer.commit(inEngine));
	}
	return stat;
}

void printStat(const char* inName, const oa::PerfStat& inStat) {
	std::printf("  %-25s mean=%7.4f ms  p50=%7.4f  p95=%7.4f\n",
		inName, inStat.mean(), inStat.p50(), inStat.p95());
}

} // namespace

TEST(BenchMoeComponents, NlpShape) {
	if (not vkTestEngineOk()) GTEST_SKIP();
	auto& engine = testEngine();
	auto& ctx = oa::ExecutionSession::getActive();
	oa::ExecutionSession::RecordingScope scope(ctx);
	oa::GradNo noGrad;

	constexpr oa::I32 T = 1024, D = 32, E = 4, K = 2, H = 16, R = T * K;
	std::vector<oa::I32> indexHost(R);
	for (oa::I32 t = 0; t < T; ++t) {
		indexHost[t * K] = t % E;
		indexHost[t * K + 1] = (t + 1) % E;
	}
	const std::vector<oa::U32> offsetHost = {0, 512, 1024, 1536, 2048};
	auto indices = oa::FnMatrix::fromBytes(oa::Span<const oa::U8>(
		reinterpret_cast<const oa::U8*>(indexHost.data()), indexHost.size() * sizeof(oa::I32)),
		{T, K}, oa::ScalarType::Int32);
	auto offsets = oa::FnMatrix::fromBytes(oa::Span<const oa::U8>(
		reinterpret_cast<const oa::U8*>(offsetHost.data()), offsetHost.size() * sizeof(oa::U32)),
		{E + 1}, oa::ScalarType::UInt32);
	auto probs = oa::FnMatrix::scale(oa::FnMatrix::ones({T, E}), 0.25F);
	auto mask = oa::FnMatrix::topKMask(indices, E);
	auto x = oa::FnMatrix::randXavier({T, D});
	auto packedX = oa::FnMatrix::randXavier({R, D});
	auto gateW = oa::FnMatrix::randXavier({E, 2 * H, D});
	auto gateB = oa::FnMatrix::zeros({E, 2 * H});
	auto downW = oa::FnMatrix::randXavier({E, D, H});
	auto downB = oa::FnMatrix::zeros({E, D});
	auto gateUp = oa::FnMatrix::randXavier({R, 2 * H});
	auto hidden = oa::FnMatrix::randXavier({R, H});
	auto packedOut = oa::FnMatrix::randXavier({R, D});
	auto routeGate = oa::FnMatrix::scale(oa::FnMatrix::ones({T, K}), 0.5F);
	auto dOut = oa::FnMatrix::randXavier({T, D});
	auto dPacked = oa::FnMatrix::randXavier({R, D});
	auto dHidden = oa::FnMatrix::randXavier({R, H});
	auto dGateUp = oa::FnMatrix::randXavier({R, 2 * H});
	auto dRouteGate = oa::FnMatrix::randXavier({T, K});
	auto gateWM = oa::FnMatrix::zeros(gateW.getShape());
	auto gateWV = oa::FnMatrix::zeros(gateW.getShape());
	auto gateBM = oa::FnMatrix::zeros(gateB.getShape());
	auto gateBV = oa::FnMatrix::zeros(gateB.getShape());
	auto downWM = oa::FnMatrix::zeros(downW.getShape());
	auto downWV = oa::FnMatrix::zeros(downW.getShape());
	auto downBM = oa::FnMatrix::zeros(downB.getShape());
	auto downBV = oa::FnMatrix::zeros(downB.getShape());
	auto gateWGrad = oa::FnMatrix::randXavier(gateW.getShape());
	auto gateBGrad = oa::FnMatrix::randXavier(gateB.getShape());
	auto downWGrad = oa::FnMatrix::randXavier(downW.getShape());
	auto downBGrad = oa::FnMatrix::randXavier(downB.getShape());
	auto plan = oa::FnMatrix::moeExpertPlan(indices, E);
	ASSERT_TRUE(testSubmitAndWait(ctx).isOk());

	oa::Vector<oa::Matrix> keep;
	printStat("legacy route normalize", measure(engine, "legacy_route", [&] {
		auto unnorm = oa::FnMatrix::mul(probs, mask);
		auto denom = oa::FnMatrix::sum(unnorm, 1);
		auto dense = oa::FnMatrix::div(unnorm, denom);
		auto selected = oa::FnMatrix::gatherLastDim(dense, indices);
		keep = {unnorm, denom, dense, selected};
	}));
	printStat("fused route normalize", measure(engine, "fused_route", [&] {
		keep = {oa::FnMatrix::moeRouteWeights(probs, indices)};
	}));
	printStat("expert plan", measure(engine, "expert_plan", [&] {
		auto p = oa::FnMatrix::moeExpertPlan(indices, E);
		keep = {p.counts, p.offsets, p.packedToken, p.packedSlot, p.inverse};
	}));
	printStat("pack tokens", measure(engine, "pack", [&] {
		keep = {oa::FnMatrix::moeGather(x, plan.packedToken, plan.inverse)};
	}));
	printStat("grouped gate/up", measure(engine, "gate_up", [&] {
		keep = {oa::FnMatrix::groupedLinearM(packedX, gateW, gateB, offsets)};
	}));
	printStat("SwiGLU", measure(engine, "swiglu", [&] {
		keep = {oa::FnMatrix::siluMul(gateUp, H)};
	}));
	printStat("grouped down", measure(engine, "down", [&] {
		keep = {oa::FnMatrix::groupedLinearM(hidden, downW, downB, offsets)};
	}));
	printStat("combine", measure(engine, "combine", [&] {
		keep = {oa::FnMatrix::moeCombine(
			packedOut, routeGate, plan.inverse, plan.packedSlot)};
	}));
	printStat("fused sparse chain", measure(engine, "sparse_chain", [&] {
		auto weights = oa::FnMatrix::moeRouteWeights(probs, indices);
		auto p = oa::FnMatrix::moeExpertPlan(indices, E);
		auto px = oa::FnMatrix::moeGather(x, p.packedToken, p.inverse);
		auto gu = oa::FnMatrix::groupedLinearM(px, gateW, gateB, p.offsets);
		auto h = oa::FnMatrix::siluMul(gu, H);
		auto po = oa::FnMatrix::groupedLinearM(h, downW, downB, p.offsets);
		auto out = oa::FnMatrix::moeCombine(po, weights, p.inverse, p.packedSlot);
		keep = {weights, p.offsets, p.packedToken, p.packedSlot, p.inverse,
			px, gu, h, po, out};
	}));

	std::printf("\nSparse backward and optimizer:\n");
	printStat("combine backward", measure(engine, "combine_bwd", [&] {
		auto bwd = oa::FnMatrix::moeCombineBwd(
			dOut, packedOut, routeGate, plan.inverse, plan.packedSlot);
		keep = {bwd.dPacked, bwd.dRouteGate};
	}));
	printStat("grouped down backward", measure(engine, "down_bwd", [&] {
		auto bwd = oa::FnMatrix::groupedLinearMBwd(dPacked, hidden, downW, offsets);
		keep = {bwd.dInput, bwd.dWeight, bwd.dBias};
	}));
	printStat("down data+weight bwd", measure(engine, "down_gemm_bwd", [&] {
		auto bwd = oa::FnMatrix::groupedGemmMBwd(dPacked, hidden, downW, offsets);
		keep = {bwd.dInput, bwd.dWeight};
	}));
	printStat("down bias backward", measure(engine, "down_bias_bwd", [&] {
		keep = {oa::FnMatrix::groupedLinearMBiasBwd(dPacked, offsets, E)};
	}));
	printStat("SwiGLU backward", measure(engine, "swiglu_bwd", [&] {
		keep = {oa::FnMatrix::siluMulBwd(gateUp, dHidden)};
	}));
	printStat("grouped gate/up backward", measure(engine, "gate_up_bwd", [&] {
		auto bwd = oa::FnMatrix::groupedLinearMBwd(dGateUp, packedX, gateW, offsets);
		keep = {bwd.dInput, bwd.dWeight, bwd.dBias};
	}));
	printStat("gate/up data+weight bwd", measure(engine, "gate_up_gemm_bwd", [&] {
		auto bwd = oa::FnMatrix::groupedGemmMBwd(dGateUp, packedX, gateW, offsets);
		keep = {bwd.dInput, bwd.dWeight};
	}));
	printStat("gate/up bias backward", measure(engine, "gate_up_bias_bwd", [&] {
		keep = {oa::FnMatrix::groupedLinearMBiasBwd(dGateUp, offsets, E)};
	}));
	printStat("legacy pack backward", measure(engine, "legacy_pack_bwd", [&] {
		keep = {oa::FnMatrix::gatherBwd(plan.packedToken, packedX, T, D)};
	}));
	printStat("atomic pack backward", measure(engine, "atomic_pack_bwd", [&] {
		keep = {oa::FnMatrix::scatterAddRows(packedX, plan.packedToken, T)};
	}));
	printStat("route weights backward", measure(engine, "route_bwd", [&] {
		keep = {oa::FnMatrix::moeRouteWeightsBwd(
			dRouteGate, probs, indices, routeGate)};
	}));
	printStat("complete sparse backward", measure(engine, "sparse_chain_bwd", [&] {
		auto combine = oa::FnMatrix::moeCombineBwd(
			dOut, packedOut, routeGate, plan.inverse, plan.packedSlot);
		auto down = oa::FnMatrix::groupedLinearMBwd(
			combine.dPacked, hidden, downW, offsets);
		auto dGu = oa::FnMatrix::siluMulBwd(gateUp, down.dInput);
		auto gate = oa::FnMatrix::groupedLinearMBwd(dGu, packedX, gateW, offsets);
		auto dx = oa::FnMatrix::scatterAddRows(gate.dInput, plan.packedToken, T);
		auto dProb = oa::FnMatrix::moeRouteWeightsBwd(
			combine.dRouteGate, probs, indices, routeGate);
		keep = {combine.dPacked, combine.dRouteGate, down.dInput, down.dWeight,
			down.dBias, dGu, gate.dInput, gate.dWeight, gate.dBias, dx, dProb};
	}));
	printStat("expert AdamW (4 tensors)", measure(engine, "expert_adamw", [&] {
		oa::FnOptim::AdamWParamSet sets[] = {
			{&gateW, &gateWM, &gateWV, &gateWGrad},
			{&gateB, &gateBM, &gateBV, &gateBGrad},
			{&downW, &downWM, &downWV, &downWGrad},
			{&downB, &downBM, &downBV, &downBGrad},
		};
		oa::FnOptim::adamWStepMany(oa::Span<const oa::FnOptim::AdamWParamSet>(sets, 4),
			1e-3F, 0.9F, 0.999F, 1e-8F, 0.01F, 1);
	}));
}
