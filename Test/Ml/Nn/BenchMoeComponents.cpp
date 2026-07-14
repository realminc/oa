// Sparse-MoE component benchmark for the canonical NLP tutorial shape.

#include "../../OaTest.h"

#include <Oa/Core/PerfStat.h>
#include <Oa/Ml.h>
#include <Oa/Ml/Autograd.h>
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
		keep = {OaFnMatrix::Gather(x, plan.PackedToken)};
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
		auto px = OaFnMatrix::Gather(x, p.PackedToken);
		auto gu = OaFnMatrix::GroupedLinearM(px, gateW, gateB, p.Offsets);
		auto h = OaFnMatrix::SiluMul(gu, H);
		auto po = OaFnMatrix::GroupedLinearM(h, downW, downB, p.Offsets);
		auto out = OaFnMatrix::MoeCombine(po, weights, p.Inverse, p.PackedSlot);
		keep = {weights, p.Offsets, p.PackedToken, p.PackedSlot, p.Inverse,
			px, gu, h, po, out};
	}));
}
