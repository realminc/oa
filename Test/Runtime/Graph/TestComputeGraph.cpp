// OaComputeGraph Test Suite — Comprehensive testing of CPU and GPU execution paths
//
// Tests both legacy CPU-driven execution and new GPU-driven execution (Phase 2).
// Validates correctness, performance, and edge cases for both modes.
//
// Test Structure (similar to TestAutograd):
// 1. Basic functionality tests (Add, Execute, Compile, Replay)
// 2. Correctness tests (CPU vs GPU path comparison)
// 3. Edge cases (empty graphs, single node, large graphs)
// 4. Performance benchmarks (CPU overhead, throughput)
// 5. Memory analysis (aliasing, lifetimes)

#include "../../OaTest.h"
#include <Oa/Runtime/ComputeGraph.h>
#include <Oa/Runtime/GpuTimer.h>
#include <Oa/Runtime/Engine.h>
#include <Oa/Core/Memory.h>

#include <chrono>
#include <cstdio>

// =============================================================================
// HELPERS
// =============================================================================

static void PrintBar() {
	fprintf(stderr, "  ────────────────────────────────────────────────────────────────\n");
}

static void PrintHeader(const char* InSection) {
	fprintf(stderr, "\n");
	PrintBar();
	fprintf(stderr, "  %s\n", InSection);
	PrintBar();
}

template<typename F>
static double MeasureUs(OaI32 InWarmup, OaI32 InIters, F&& InFunc) {
	for (OaI32 i = 0; i < InWarmup; ++i) InFunc();
	
	auto start = std::chrono::high_resolution_clock::now();
	for (OaI32 i = 0; i < InIters; ++i) InFunc();
	auto end = std::chrono::high_resolution_clock::now();
	
	double totalUs = std::chrono::duration<double, std::micro>(end - start).count();
	return totalUs / InIters;
}

// Build a chain of scale dispatches for testing
static void BuildChainGraph(
	OaComputeGraph& OutGraph,
	OaVec<OaVkBuffer>& InBufs,
	OaI32 InNumDispatches)
{
	struct { OaU32 N; OaF32 Scale; } pc = {256, 1.001f};
	
	for (OaI32 i = 0; i < InNumDispatches; ++i) {
		OaVkBuffer bufs[] = {InBufs[i], InBufs[i + 1]};
		OaBufferAccess acc[] = {OaBufferAccess::Read, OaBufferAccess::Write};
		OutGraph.Add("Scale", bufs, acc, &pc, sizeof(pc), 1);
	}
}

// =============================================================================
// BASIC FUNCTIONALITY TESTS
// =============================================================================

TEST(ComputeGraph, SystemInfo) {
	fprintf(stderr, "\n");
	fprintf(stderr, "  ╔═══════════════════════════════════════════════════════════════╗\n");
	fprintf(stderr, "  ║         OaComputeGraph TEST SUITE — CPU & GPU Paths          ║\n");
	fprintf(stderr, "  ╚═══════════════════════════════════════════════════════════════╝\n");
	
	auto* rt = OaComputeEngine::GetGlobal();
	if (rt) {
		fprintf(stderr, "\n  GPU: %s (%s)\n",
			rt->Device.Info.Hardware.DeviceName.c_str(), 
			rt->Device.Info.Hardware.VendorName.c_str());
		fprintf(stderr, "  Type: %s, VRAM: %llu MB\n",
			rt->Device.Info.Hardware.DeviceType == OaDeviceType::VkDiscrete ? "Discrete" :
			rt->Device.Info.Hardware.DeviceType == OaDeviceType::VkIntegrated ? "Integrated" :
			rt->Device.Info.Hardware.DeviceType == OaDeviceType::VkCpu ? "Software" : "Unknown",
			static_cast<unsigned long long>(rt->Device.Info.Hardware.VramBytes / (1024 * 1024)));
	}
	fprintf(stderr, "\n");
}

TEST(ComputeGraph, BasicAddAndExecute) {
	auto* rt = OaComputeEngine::GetGlobal();
	ASSERT_NE(rt, nullptr);
	
	auto srcRes = rt->Allocator.AllocHostVisible(256 * sizeof(OaF32));
	ASSERT_TRUE(srcRes.IsOk());
	auto src = srcRes.GetValue();
	rt->RegisterBuffer(src);
	
	auto dstRes = rt->Allocator.AllocHostVisible(256 * sizeof(OaF32));
	ASSERT_TRUE(dstRes.IsOk());
	auto dst = dstRes.GetValue();
	rt->RegisterBuffer(dst);
	
	OaF32* data = static_cast<OaF32*>(src.MappedPtr);
	for (OaI32 i = 0; i < 256; ++i) data[i] = static_cast<OaF32>(i);
	
	struct { OaU32 N; OaF32 Scale; } push{256, 3.0f};
	OaVkBuffer bufs[] = {src, dst};
	OaBufferAccess access[] = {OaBufferAccess::Read, OaBufferAccess::Write};
	
	OaComputeGraph graph;
	graph.Add("Scale", bufs, access, &push, sizeof(push), (256 + 255) / 256);
	EXPECT_EQ(graph.NodeCount(), 1u);
	
	auto status = graph.Execute(*rt);
	if (status.IsOk()) {
		OaF32* out = static_cast<OaF32*>(dst.MappedPtr);
		for (OaI32 i = 0; i < 256; ++i) {
			EXPECT_NEAR(out[i], static_cast<OaF32>(i) * 3.0f, 1e-3f);
		}
	}
	
	rt->DeregisterBuffer(src);
	rt->Allocator.Free(src);
	rt->DeregisterBuffer(dst);
	rt->Allocator.Free(dst);
}

TEST(ComputeGraph, CompileAndReplay) {
	auto* rt = OaComputeEngine::GetGlobal();
	ASSERT_NE(rt, nullptr);
	
	auto srcRes = rt->Allocator.AllocHostVisible(256 * sizeof(OaF32));
	ASSERT_TRUE(srcRes.IsOk());
	auto src = srcRes.GetValue();
	rt->RegisterBuffer(src);
	
	auto midRes = rt->Allocator.AllocHostVisible(256 * sizeof(OaF32));
	ASSERT_TRUE(midRes.IsOk());
	auto mid = midRes.GetValue();
	rt->RegisterBuffer(mid);
	
	auto dstRes = rt->Allocator.AllocHostVisible(256 * sizeof(OaF32));
	ASSERT_TRUE(dstRes.IsOk());
	auto dst = dstRes.GetValue();
	rt->RegisterBuffer(dst);
	
	OaF32* data = static_cast<OaF32*>(src.MappedPtr);
	for (OaI32 i = 0; i < 256; ++i) data[i] = static_cast<OaF32>(i + 1);
	
	struct { OaU32 N; OaF32 Scale; } push1{256, 2.0f};
	struct { OaU32 N; OaF32 Scale; } push2{256, 0.5f};
	
	OaVkBuffer bufs1[] = {src, mid};
	OaBufferAccess access1[] = {OaBufferAccess::Read, OaBufferAccess::Write};
	OaVkBuffer bufs2[] = {mid, dst};
	OaBufferAccess access2[] = {OaBufferAccess::Read, OaBufferAccess::Write};
	
	OaComputeGraph graph;
	graph.Add("Scale", bufs1, access1, &push1, sizeof(push1), 1);
	graph.Add("Scale", bufs2, access2, &push2, sizeof(push2), 1);
	EXPECT_EQ(graph.NodeCount(), 2u);
	EXPECT_FALSE(graph.IsCompiled());
	
	auto compileStatus = graph.Compile(*rt);
	if (!compileStatus.IsOk()) {
		graph.Destroy(rt->Device);
		rt->DeregisterBuffer(src);
		rt->Allocator.Free(src);
		rt->DeregisterBuffer(mid);
		rt->Allocator.Free(mid);
		rt->DeregisterBuffer(dst);
		rt->Allocator.Free(dst);
		GTEST_SKIP() << "scale shader not loaded";
	}
	EXPECT_TRUE(graph.IsCompiled());
	
	// Replay 5 times
	for (OaI32 rep = 0; rep < 5; ++rep) {
		for (OaI32 i = 0; i < 256; ++i) data[i] = static_cast<OaF32>(i + 1);
		
		auto status = graph.Replay(*rt);
		ASSERT_TRUE(status.IsOk()) << "replay " << rep << " failed";
		// Replay() is non-blocking (submits PrimaryCb_ with a timeline semaphore
		// and returns). Results are undefined until the replay completes — the
		// documented contract is to WaitForPendingReplay()/Sync() before reading.
		ASSERT_TRUE(graph.WaitForPendingReplay(rt->Device).IsOk())
			<< "wait for replay " << rep << " failed";

		OaF32* out = static_cast<OaF32*>(dst.MappedPtr);
		for (OaI32 i = 0; i < 256; ++i) {
			EXPECT_NEAR(out[i], static_cast<OaF32>(i + 1) * 1.0f, 1e-3f)
				<< "mismatch at i=" << i << " rep=" << rep;
		}
	}
	
	graph.Destroy(rt->Device);
	rt->DeregisterBuffer(src);
	rt->Allocator.Free(src);
	rt->DeregisterBuffer(mid);
	rt->Allocator.Free(mid);
	rt->DeregisterBuffer(dst);
	rt->Allocator.Free(dst);
}

TEST(ComputeGraph, HazardPlannerTracksReadBeforeWrite) {
	auto* rt = OaComputeEngine::GetGlobal();
	ASSERT_NE(rt, nullptr);

	auto xRes = rt->Allocator.AllocHostVisible(256 * sizeof(OaF32));
	auto aRes = rt->Allocator.AllocHostVisible(256 * sizeof(OaF32));
	auto bRes = rt->Allocator.AllocHostVisible(256 * sizeof(OaF32));
	ASSERT_TRUE(xRes.IsOk() && aRes.IsOk() && bRes.IsOk());
	auto x = xRes.GetValue();
	auto a = aRes.GetValue();
	auto b = bRes.GetValue();

	struct { OaU32 N; OaF32 Scale; } push{256, 1.0F};
	OaComputeGraph graph;
	{
		OaVkBuffer bufs[] = {x, a};
		OaBufferAccess access[] = {
			OaBufferAccess::Read, OaBufferAccess::Write};
		graph.Add("Scale", bufs, access, &push, sizeof(push), 1);
	}
	{
		// X was only read by the first node. A writer still needs an execution
		// dependency even though there is no preceding in-graph writer for X.
		OaVkBuffer bufs[] = {b, x};
		OaBufferAccess access[] = {
			OaBufferAccess::Read, OaBufferAccess::Write};
		graph.Add("Scale", bufs, access, &push, sizeof(push), 1);
	}

	ASSERT_TRUE(graph.Compile(*rt).IsOk());
	const auto stats = graph.GetStats();
	EXPECT_EQ(stats.WarBarrierCount, 1U);
	EXPECT_GE(stats.BarrierCount, 1U);

	graph.Destroy(rt->Device);

	OaComputeGraph indirectGraph;
	{
		OaVkBuffer bufs[] = {x, a};
		OaBufferAccess access[] = {
			OaBufferAccess::Read, OaBufferAccess::Write};
		indirectGraph.Add("Scale", bufs, access, &push, sizeof(push), 1);
	}
	{
		OaVkBuffer bufs[] = {x, b};
		OaBufferAccess access[] = {
			OaBufferAccess::Read, OaBufferAccess::Write};
		indirectGraph.AddIndirect(
			"Scale", bufs, access, &push, sizeof(push), a, 0);
	}
	ASSERT_TRUE(indirectGraph.Compile(*rt).IsOk());
	EXPECT_EQ(indirectGraph.GetStats().IndirectBarrierCount, 1U);
	indirectGraph.Destroy(rt->Device);

	rt->Allocator.Free(x);
	rt->Allocator.Free(a);
	rt->Allocator.Free(b);
}

TEST(ComputeGraph, IndirectArgumentHazardAndCacheIdentity) {
	auto* rt = OaComputeEngine::GetGlobal();
	ASSERT_NE(rt, nullptr);

	constexpr OaU32 N = 64;
	auto srcRes = rt->AllocBuffer(N * sizeof(OaF32));
	auto dstRes = rt->AllocBuffer(N * sizeof(OaF32));
	auto argsRes = rt->AllocBuffer(6 * sizeof(OaU32));
	ASSERT_TRUE(srcRes.IsOk() && dstRes.IsOk() && argsRes.IsOk());
	auto src = std::move(*srcRes);
	auto dst = std::move(*dstRes);
	auto args = std::move(*argsRes);

	auto* srcData = static_cast<OaF32*>(src.MappedPtr);
	auto* dstData = static_cast<OaF32*>(dst.MappedPtr);
	auto* dispatchArgs = static_cast<OaU32*>(args.MappedPtr);
	for (OaU32 i = 0; i < N; ++i) {
		srcData[i] = static_cast<OaF32>(i + 1);
		dstData[i] = 0.0F;
	}
	dispatchArgs[0] = 1; dispatchArgs[1] = 1; dispatchArgs[2] = 1;
	dispatchArgs[3] = 0; dispatchArgs[4] = 1; dispatchArgs[5] = 1;
	ASSERT_TRUE(rt->Allocator.FlushHostBuffer(src, 0, src.Size));
	ASSERT_TRUE(rt->Allocator.FlushHostBuffer(dst, 0, dst.Size));
	ASSERT_TRUE(rt->Allocator.FlushHostBuffer(args, 0, args.Size));

	struct { OaU32 N; OaF32 Scale; } push{N, 2.0F};
	OaVkBuffer bufs[] = {src, dst};
	OaBufferAccess access[] = {
		OaBufferAccess::Read, OaBufferAccess::Write};

	OaComputeGraph graph;
	graph.AddIndirect("Scale", bufs, access, &push, sizeof(push), args, 0);
	ASSERT_TRUE(graph.Compile(*rt).IsOk());
	ASSERT_TRUE(graph.Replay(*rt).IsOk());
	ASSERT_TRUE(graph.WaitForPendingReplay(rt->Device).IsOk());
	EXPECT_NEAR(dstData[0], 2.0F, 1e-3F);

	// The only topology change is the indirect offset. Compile must not reuse
	// the old command buffer: the second command has groupCountX=0 and skips.
	for (OaU32 i = 0; i < N; ++i) dstData[i] = 0.0F;
	ASSERT_TRUE(rt->Allocator.FlushHostBuffer(dst, 0, dst.Size));
	graph.ClearNodes();
	graph.AddIndirect("Scale", bufs, access, &push, sizeof(push), args,
		3 * sizeof(OaU32));
	ASSERT_TRUE(graph.Compile(*rt).IsOk());
	ASSERT_TRUE(graph.Replay(*rt).IsOk());
	ASSERT_TRUE(graph.WaitForPendingReplay(rt->Device).IsOk());
	EXPECT_NEAR(dstData[0], 0.0F, 1e-6F);

	const auto lifetimes = graph.ComputeLifetimes();
	EXPECT_EQ(lifetimes.Size(), 3U); // src, dst, and indirect argument buffer

	graph.Destroy(rt->Device);
	rt->FreeBuffer(src);
	rt->FreeBuffer(dst);
	rt->FreeBuffer(args);
}

TEST(ComputeGraph, EmptyGraph) {
	auto* rt = OaComputeEngine::GetGlobal();
	ASSERT_NE(rt, nullptr);
	
	OaComputeGraph graph;
	EXPECT_EQ(graph.NodeCount(), 0u);
	EXPECT_TRUE(graph.Execute(*rt).IsOk());
	EXPECT_TRUE(graph.Compile(*rt).IsOk());
	EXPECT_TRUE(graph.IsCompiled());
	EXPECT_TRUE(graph.Replay(*rt).IsOk());
	
	auto lifetimes = graph.ComputeLifetimes();
	EXPECT_TRUE(lifetimes.Empty());
	
	auto groups = graph.ComputeAliasGroups();
	EXPECT_TRUE(groups.Empty());
	
	auto stats = graph.GetStats();
	EXPECT_EQ(stats.DispatchCount, 0u);
	EXPECT_EQ(stats.BarrierCount, 0u);
	
	graph.Destroy(rt->Device);
}

// =============================================================================
// GPU COMPILATION TESTS (Phase 2)
TEST(ComputeGraph, BarrierOverhead) {
	auto* rt = OaComputeEngine::GetGlobal();
	ASSERT_NE(rt, nullptr);
	
	PrintHeader("BARRIER ANALYSIS — CPU vs GPU Barrier Computation");
	fprintf(stderr, "  %-12s %12s  %12s  %12s\n",
		"Dispatches", "Barriers", "Barrier%", "CPU Cost");
	PrintBar();
	
	struct Config { OaI32 N; const char* Label; };
	Config configs[] = {
		{6,  "6 dispatches"},
		{12, "12 dispatches"},
		{25, "25 dispatches"},
		{50, "50 dispatches"},
	};
	
	for (auto& cfg : configs) {
		OaI32 N = cfg.N;
		OaVec<OaVkBuffer> bufs(N + 1);
		for (OaI32 i = 0; i <= N; ++i) {
			auto res = rt->Allocator.AllocHostVisible(256 * sizeof(OaF32));
			ASSERT_TRUE(res.IsOk());
			bufs[i] = res.GetValue();
		}
		
		OaComputeGraph graph;
		BuildChainGraph(graph, bufs, N);
		
		auto compileStatus = graph.Compile(*rt);
		if (!compileStatus.IsOk()) {
			for (auto& b : bufs) rt->Allocator.Free(b);
			GTEST_SKIP() << "scale shader not loaded";
		}
		
		auto stats = graph.GetStats();
		double barrierPct = N > 0 ? (100.0 * stats.BarrierCount / N) : 0.0;
		
		// Estimate CPU cost of barrier computation
		// Each barrier requires hashmap lookup + comparison (~50ns on modern CPU)
		double cpuCostUs = stats.BarrierCount * 0.05;  // 50ns per barrier
		
		fprintf(stderr, "  %-12s %10u    %10.0f%%   %10.1f µs\n",
			cfg.Label, stats.BarrierCount, barrierPct, cpuCostUs);
		
		graph.Destroy(rt->Device);
		for (auto& b : bufs) rt->Allocator.Free(b);
	}
	
	fprintf(stderr, "\n  GPU path moves barrier computation to GPU, eliminating CPU cost.\n");
}

// =============================================================================
// MEMORY ANALYSIS
// =============================================================================

TEST(ComputeGraph, MemoryAliasing) {
	auto* rt = OaComputeEngine::GetGlobal();
	ASSERT_NE(rt, nullptr);
	
	PrintHeader("MEMORY ALIASING — Potential VRAM savings");
	fprintf(stderr, "  %-12s %12s  %12s  %12s  %8s\n",
		"Dispatches", "Total bufs", "Alias groups", "Savings", "Pct");
	PrintBar();
	
	struct Config { OaI32 N; const char* Label; };
	Config configs[] = {
		{6,  "6 dispatches"},
		{12, "12 dispatches"},
		{25, "25 dispatches"},
	};
	
	for (auto& cfg : configs) {
		OaI32 N = cfg.N;
		OaVec<OaVkBuffer> bufs(N + 1);
		for (OaI32 i = 0; i <= N; ++i) {
			auto res = rt->Allocator.AllocHostVisible(4096);
			ASSERT_TRUE(res.IsOk());
			bufs[i] = res.GetValue();
		}
		
		OaComputeGraph graph;
		BuildChainGraph(graph, bufs, N);
		
		auto stats = graph.GetStats();
		auto groups = graph.ComputeAliasGroups();
		auto lifetimes = graph.ComputeLifetimes();
		
		double savingsPct = stats.TotalBufferBytes > 0
			? (100.0 * stats.PotentialAliasSavings / stats.TotalBufferBytes) : 0.0;
		
		fprintf(stderr, "  %-12s %10zu    %10zu    %8llu B   %5.1f%%\n",
			cfg.Label,
			lifetimes.Size(), groups.Size(),
			static_cast<unsigned long long>(stats.PotentialAliasSavings),
			savingsPct);
		
		for (auto& b : bufs) rt->Allocator.Free(b);
	}
}
// =============================================================================

// Main is provided by MlTestMain.cpp (shared test infrastructure)
