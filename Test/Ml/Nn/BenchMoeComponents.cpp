// Sparse-MoE component benchmark for the canonical NLP tutorial shape.

#include "../../OaTest.h"

#include <Oa/Core/PerfStat.h>
#include <Oa/Ml.h>
#include <Oa/Ml/Autograd.h>
#include <Oa/Ml/FnOptim.h>
#include <Oa/Runtime/Engine.h>
#include <Oa/Runtime/GpuTimer.h>

#include <cstdio>
#include <stdexcept>
#include <vector>

namespace {

template <typename Enqueue>
OaPerfStat Measure(OaComputeEngine& InEngine, const char* InName, Enqueue&& InEnqueue) {
	constexpr OaU32 warmup = 5, samples = 25;
	OaGpuTimer timer;
	if (auto status = timer.Init(InEngine, InName); not status.IsOk())
		throw std::runtime_error("BenchMoeComponents timer initialization failed");
	OaPerfStat stat(InName, samples, warmup);
	auto& ctx = OaContext::GetDefault();
	for (OaU32 i = 0; i < warmup + samples; ++i) {
		InEnqueue();
		if (auto status = ctx.ExecuteAsync(&timer); not status.IsOk())
			throw std::runtime_error("BenchMoeComponents execution failed");
		stat.Push(timer.ReadbackMs(InEngine.Device));
	}
	timer.Destroy(InEngine.Device);
	return stat;
}

void Print(const char* InName, const OaPerfStat& InStat) {
	std::printf("  %-25s mean=%7.4f ms  p50=%7.4f  p95=%7.4f\n",
		InName, InStat.Mean(), InStat.P50(), InStat.P95());
}

} // namespace

TEST(BenchMoeComponents, NlpShape) {
	if (not OaVkTestEngineOk()) GTEST_SKIP();
	auto& engine = *OaComputeEngine::GetGlobal();
	auto& ctx = OaContext::GetDefault();
	OaContext::Scope scope(ctx);
	OaGradNo noGrad;

	constexpr OaI32 T = 1024, D = 32, E = 4, K = 2, H = 16, R = T * K;
	std::vector<OaI32> indexHost(R);
	for (OaI32 t = 0; t < T; ++t) {
		indexHost[t * K] = t % E;
		indexHost[t * K + 1] = (t + 1) % E;
	}
	const std::vector<OaU32> offsetHost = {0, 512, 1024, 1536, 2048};
	auto indices = OaFnMatrix::FromBytes(OaSpan<const OaU8>(
		reinterpret_cast<const OaU8*>(indexHost.data()), indexHost.size() * sizeof(OaI32)),
		{T, K}, OaScalarType::Int32);
	auto offsets = OaFnMatrix::FromBytes(OaSpan<const OaU8>(
		reinterpret_cast<const OaU8*>(offsetHost.data()), offsetHost.size() * sizeof(OaU32)),
		{E + 1}, OaScalarType::UInt32);
	auto probs = OaFnMatrix::Scale(OaFnMatrix::Ones({T, E}), 0.25F);
	auto mask = OaFnMatrix::TopKMask(indices, E);
	auto x = OaFnMatrix::RandXavier({T, D});
	auto packedX = OaFnMatrix::RandXavier({R, D});
	auto gateW = OaFnMatrix::RandXavier({E, 2 * H, D});
	auto gateB = OaFnMatrix::Zeros({E, 2 * H});
	auto downW = OaFnMatrix::RandXavier({E, D, H});
	auto downB = OaFnMatrix::Zeros({E, D});
	auto gateUp = OaFnMatrix::RandXavier({R, 2 * H});
	auto hidden = OaFnMatrix::RandXavier({R, H});
	auto packedOut = OaFnMatrix::RandXavier({R, D});
	auto routeGate = OaFnMatrix::Scale(OaFnMatrix::Ones({T, K}), 0.5F);
	auto dOut = OaFnMatrix::RandXavier({T, D});
	auto dPacked = OaFnMatrix::RandXavier({R, D});
	auto dHidden = OaFnMatrix::RandXavier({R, H});
	auto dGateUp = OaFnMatrix::RandXavier({R, 2 * H});
	auto dRouteGate = OaFnMatrix::RandXavier({T, K});
	auto gateWM = OaFnMatrix::Zeros(gateW.GetShape());
	auto gateWV = OaFnMatrix::Zeros(gateW.GetShape());
	auto gateBM = OaFnMatrix::Zeros(gateB.GetShape());
	auto gateBV = OaFnMatrix::Zeros(gateB.GetShape());
	auto downWM = OaFnMatrix::Zeros(downW.GetShape());
	auto downWV = OaFnMatrix::Zeros(downW.GetShape());
	auto downBM = OaFnMatrix::Zeros(downB.GetShape());
	auto downBV = OaFnMatrix::Zeros(downB.GetShape());
	auto gateWGrad = OaFnMatrix::RandXavier(gateW.GetShape());
	auto gateBGrad = OaFnMatrix::RandXavier(gateB.GetShape());
	auto downWGrad = OaFnMatrix::RandXavier(downW.GetShape());
	auto downBGrad = OaFnMatrix::RandXavier(downB.GetShape());
	auto plan = OaFnMatrix::MoeExpertPlan(indices, E);
	ASSERT_TRUE(ctx.Execute().IsOk());
	ASSERT_TRUE(ctx.Sync().IsOk());

	OaVec<OaMatrix> keep;
	Print("legacy route normalize", Measure(engine, "legacy_route", [&] {
		auto unnorm = OaFnMatrix::Mul(probs, mask);
		auto denom = OaFnMatrix::Sum(unnorm, 1);
		auto dense = OaFnMatrix::Div(unnorm, denom);
		auto selected = OaFnMatrix::GatherLastDim(dense, indices);
		keep = {unnorm, denom, dense, selected};
	}));
	Print("fused route normalize", Measure(engine, "fused_route", [&] {
		keep = {OaFnMatrix::MoeRouteWeights(probs, indices)};
	}));
	Print("expert plan", Measure(engine, "expert_plan", [&] {
		auto p = OaFnMatrix::MoeExpertPlan(indices, E);
		keep = {p.Counts, p.Offsets, p.PackedToken, p.PackedSlot, p.Inverse};
	}));
	Print("pack tokens", Measure(engine, "pack", [&] {
		keep = {OaFnMatrix::MoeGather(x, plan.PackedToken, plan.Inverse)};
	}));
	Print("grouped gate/up", Measure(engine, "gate_up", [&] {
		keep = {OaFnMatrix::GroupedLinearM(packedX, gateW, gateB, offsets)};
	}));
	Print("SwiGLU", Measure(engine, "swiglu", [&] {
		keep = {OaFnMatrix::SiluMul(gateUp, H)};
	}));
	Print("grouped down", Measure(engine, "down", [&] {
		keep = {OaFnMatrix::GroupedLinearM(hidden, downW, downB, offsets)};
	}));
	Print("combine", Measure(engine, "combine", [&] {
		keep = {OaFnMatrix::MoeCombine(
			packedOut, routeGate, plan.Inverse, plan.PackedSlot)};
	}));
	Print("fused sparse chain", Measure(engine, "sparse_chain", [&] {
		auto weights = OaFnMatrix::MoeRouteWeights(probs, indices);
		auto p = OaFnMatrix::MoeExpertPlan(indices, E);
		auto px = OaFnMatrix::MoeGather(x, p.PackedToken, p.Inverse);
		auto gu = OaFnMatrix::GroupedLinearM(px, gateW, gateB, p.Offsets);
		auto h = OaFnMatrix::SiluMul(gu, H);
		auto po = OaFnMatrix::GroupedLinearM(h, downW, downB, p.Offsets);
		auto out = OaFnMatrix::MoeCombine(po, weights, p.Inverse, p.PackedSlot);
		keep = {weights, p.Offsets, p.PackedToken, p.PackedSlot, p.Inverse,
			px, gu, h, po, out};
	}));

	std::printf("\nSparse backward and optimizer:\n");
	Print("combine backward", Measure(engine, "combine_bwd", [&] {
		auto bwd = OaFnMatrix::MoeCombineBwd(
			dOut, packedOut, routeGate, plan.Inverse, plan.PackedSlot);
		keep = {bwd.DPacked, bwd.DRouteGate};
	}));
	Print("grouped down backward", Measure(engine, "down_bwd", [&] {
		auto bwd = OaFnMatrix::GroupedLinearMBwd(dPacked, hidden, downW, offsets);
		keep = {bwd.DInput, bwd.DWeight, bwd.DBias};
	}));
	Print("down data+weight bwd", Measure(engine, "down_gemm_bwd", [&] {
		auto bwd = OaFnMatrix::GroupedGemmMBwd(dPacked, hidden, downW, offsets);
		keep = {bwd.DInput, bwd.DWeight};
	}));
	Print("down bias backward", Measure(engine, "down_bias_bwd", [&] {
		keep = {OaFnMatrix::GroupedLinearMBiasBwd(dPacked, offsets, E)};
	}));
	Print("SwiGLU backward", Measure(engine, "swiglu_bwd", [&] {
		keep = {OaFnMatrix::SiluMulBwd(gateUp, dHidden)};
	}));
	Print("grouped gate/up backward", Measure(engine, "gate_up_bwd", [&] {
		auto bwd = OaFnMatrix::GroupedLinearMBwd(dGateUp, packedX, gateW, offsets);
		keep = {bwd.DInput, bwd.DWeight, bwd.DBias};
	}));
	Print("gate/up data+weight bwd", Measure(engine, "gate_up_gemm_bwd", [&] {
		auto bwd = OaFnMatrix::GroupedGemmMBwd(dGateUp, packedX, gateW, offsets);
		keep = {bwd.DInput, bwd.DWeight};
	}));
	Print("gate/up bias backward", Measure(engine, "gate_up_bias_bwd", [&] {
		keep = {OaFnMatrix::GroupedLinearMBiasBwd(dGateUp, offsets, E)};
	}));
	Print("legacy pack backward", Measure(engine, "legacy_pack_bwd", [&] {
		keep = {OaFnMatrix::GatherBwd(plan.PackedToken, packedX, T, D)};
	}));
	Print("atomic pack backward", Measure(engine, "atomic_pack_bwd", [&] {
		keep = {OaFnMatrix::ScatterAddRows(packedX, plan.PackedToken, T)};
	}));
	Print("route weights backward", Measure(engine, "route_bwd", [&] {
		keep = {OaFnMatrix::MoeRouteWeightsBwd(
			dRouteGate, probs, indices, routeGate)};
	}));
	Print("complete sparse backward", Measure(engine, "sparse_chain_bwd", [&] {
		auto combine = OaFnMatrix::MoeCombineBwd(
			dOut, packedOut, routeGate, plan.Inverse, plan.PackedSlot);
		auto down = OaFnMatrix::GroupedLinearMBwd(
			combine.DPacked, hidden, downW, offsets);
		auto dGu = OaFnMatrix::SiluMulBwd(gateUp, down.DInput);
		auto gate = OaFnMatrix::GroupedLinearMBwd(dGu, packedX, gateW, offsets);
		auto dx = OaFnMatrix::ScatterAddRows(gate.DInput, plan.PackedToken, T);
		auto dProb = OaFnMatrix::MoeRouteWeightsBwd(
			combine.DRouteGate, probs, indices, routeGate);
		keep = {combine.DPacked, combine.DRouteGate, down.DInput, down.DWeight,
			down.DBias, dGu, gate.DInput, gate.DWeight, gate.DBias, dx, dProb};
	}));
	Print("expert AdamW (4 tensors)", Measure(engine, "expert_adamw", [&] {
		OaFnOptim::OaAdamWParamSet sets[] = {
			{&gateW, &gateWM, &gateWV, &gateWGrad},
			{&gateB, &gateBM, &gateBV, &gateBGrad},
			{&downW, &downWM, &downWV, &downWGrad},
			{&downB, &downBM, &downBV, &downBGrad},
		};
		OaFnOptim::AdamWStepMany(OaSpan<const OaFnOptim::OaAdamWParamSet>(sets, 4),
			1e-3F, 0.9F, 0.999F, 1e-8F, 0.01F, 1);
	}));
}
