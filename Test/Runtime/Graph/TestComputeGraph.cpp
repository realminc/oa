// OaComputeGraph Test Suite — Comprehensive testing of CPU and GPU execution paths
//
// Validates graph construction, synchronization, compiled replay, context
// batching, correctness, performance instrumentation and edge cases.
//
// Test Structure (similar to TestAutograd):
// 1. Basic functionality tests (Add, Execute, Compile, Replay)
// 2. Correctness tests (one-shot, compiled replay and context batching)
// 3. Edge cases (empty graphs, single node, large graphs)
// 4. Performance benchmarks (CPU overhead, throughput)
// 5. Memory analysis (aliasing, lifetimes)

#include "../../OaTest.h"
#include <Oa/Runtime/ComputeGraph.h>
#include <Oa/Runtime/Context.h>
#include <Oa/Runtime/GpuTimer.h>
#include <Oa/Runtime/Engine.h>
#include <Oa/Core/Memory.h>

#include <chrono>
#include <cstring>
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

TEST(ComputeGraph, DispatchDescriptorCopiesAllMetadata) {
	OaVkBuffer buffers[2];
	buffers[0].Buffer = reinterpret_cast<void*>(0x1000);
	buffers[0].BindlessIndex = 7;
	buffers[1].Buffer = reinterpret_cast<void*>(0x2000);
	buffers[1].BindlessIndex = 11;
	OaBufferAccess access[] = {
		OaBufferAccess::Read, OaBufferAccess::Write};
	struct Push { OaU32 Count; OaF32 Scale; } push{64, 2.0F};

	OaComputeDispatchDesc desc;
	desc.Kernel = "Scale";
	desc.Buffers = buffers;
	desc.Access = access;
	desc.PushData = &push;
	desc.PushSize = sizeof(push);
	desc.Dtype = 1;
	desc.GroupsX = 3;
	desc.GroupsY = 2;
	desc.Queue = OaQueueHint::AsyncCompute;
	desc.NodeIndex = 4;

	OaComputeGraph graph;
	graph.Add(desc);
	ASSERT_EQ(graph.NodeCount(), 1U);

	// Descriptor storage is non-owning, but the graph node must be a complete
	// owning snapshot before Record/Add returns.
	push.Count = 0;
	buffers[0].BindlessIndex = 99;
	const auto nodes = graph.Nodes();
	ASSERT_EQ(nodes.Size(), 1U);
	EXPECT_EQ(nodes[0].Shader, "Scale");
	EXPECT_EQ(nodes[0].Buffers[0].BindlessIndex, 7U);
	EXPECT_EQ(nodes[0].Dtype, 1U);
	EXPECT_EQ(nodes[0].GroupsX, 3U);
	EXPECT_EQ(nodes[0].GroupsY, 2U);
	EXPECT_EQ(nodes[0].Queue, OaQueueHint::AsyncCompute);
	EXPECT_EQ(nodes[0].NodeIndex, 4U);
	Push copied{};
	std::memcpy(&copied, nodes[0].PushData, sizeof(copied));
	EXPECT_EQ(copied.Count, 64U);
	EXPECT_FLOAT_EQ(copied.Scale, 2.0F);
}

TEST(ComputeGraph, ContextRecordRejectsMalformedDescriptor) {
	auto& ctx = OaContext::GetDefault();
	ctx.Clear();

	OaVkBuffer buffer;
	OaComputeDispatchDesc desc;
	desc.Kernel = "MalformedDescriptorTest";
	desc.Buffers = OaSpan<OaVkBuffer>(&buffer, 1);
	// No access annotation for one buffer: this must fail before graph append.

	const auto status = ctx.Record(desc);
	EXPECT_FALSE(status.IsOk());
	EXPECT_EQ(ctx.NodeCount(), 0U);
	ctx.Clear();
}

TEST(ComputeGraph, ContextMatrixAddCapturesOwnershipAndDtype) {
	auto& ctx = OaContext::GetDefault();
	ctx.Clear();

	OaMatrix input;
	input.VkBuf_ = OaSharedPtr<OaVkBuffer>(new OaVkBuffer());
	input.VkBuf_->Buffer = reinterpret_cast<void*>(0x3000);
	input.VkBuf_->BindlessIndex = 13;
	input.Dtype_ = OaScalarType::BFloat16;
	OaMatrix output;
	output.VkBuf_ = OaSharedPtr<OaVkBuffer>(new OaVkBuffer());
	output.VkBuf_->Buffer = reinterpret_cast<void*>(0x4000);
	output.VkBuf_->BindlessIndex = 17;

	OaBufferAccess access[] = {OaBufferAccess::Read, OaBufferAccess::Write};
	struct Push { OaU32 Count; } push{32};
	ctx.Add("Scale", {&input, &output}, access, &push, sizeof(push), 1);

	ASSERT_EQ(ctx.NodeCount(), 1U);
	const auto nodes = ctx.Graph()->Nodes();
	ASSERT_EQ(nodes.Size(), 1U);
	EXPECT_EQ(nodes[0].Dtype, 1U);
	ASSERT_EQ(nodes[0].BufferOwners.Size(), 2U);
	EXPECT_TRUE(static_cast<bool>(nodes[0].BufferOwners[0]));
	EXPECT_TRUE(static_cast<bool>(nodes[0].BufferOwners[1]));
	EXPECT_EQ(nodes[0].Buffers[0].BindlessIndex, 13U);
	EXPECT_EQ(nodes[0].Buffers[1].BindlessIndex, 17U);
	ctx.Clear();
}

TEST(ComputeGraph, SystemInfo) {
	fprintf(stderr, "\n");
	fprintf(stderr, "  ╔═══════════════════════════════════════════════════════════════╗\n");
	fprintf(stderr, "  ║       OaComputeGraph TEST SUITE — Graph & Replay Paths      ║\n");
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

TEST(ComputeGraph, TimedReplayRequiresWaitBeforeQueryReuse) {
	auto* rt = OaComputeEngine::GetGlobal();
	ASSERT_NE(rt, nullptr);

	auto srcRes = rt->AllocBuffer(64 * sizeof(OaF32));
	auto dstRes = rt->AllocBuffer(64 * sizeof(OaF32));
	ASSERT_TRUE(srcRes.IsOk() && dstRes.IsOk());
	auto src = std::move(*srcRes);
	auto dst = std::move(*dstRes);
	for (OaU32 i = 0; i < 64; ++i) {
		static_cast<OaF32*>(src.MappedPtr)[i] = static_cast<OaF32>(i);
	}
	ASSERT_TRUE(rt->Allocator.FlushHostBuffer(src, 0, src.Size));

	struct { OaU32 N; OaF32 Scale; } push{64, 2.0F};
	OaVkBuffer bufs[] = {src, dst};
	OaBufferAccess access[] = {OaBufferAccess::Read, OaBufferAccess::Write};
	OaComputeGraph graph;
	graph.Add("Scale", bufs, access, &push, sizeof(push), 1);
	graph.SetReplayTimingEnabled(true);
	ASSERT_TRUE(graph.Compile(*rt).IsOk());
	ASSERT_TRUE(graph.Replay(*rt).IsOk());
	EXPECT_FALSE(graph.Replay(*rt).IsOk());
	ASSERT_TRUE(graph.WaitForPendingReplay(rt->Device).IsOk());
	EXPECT_GT(graph.LastReplayGpuMs(), 0.0);
	ASSERT_TRUE(graph.Replay(*rt).IsOk());
	ASSERT_TRUE(graph.WaitForPendingReplay(rt->Device).IsOk());

	graph.Destroy(rt->Device);
	rt->FreeBuffer(src);
	rt->FreeBuffer(dst);
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
	ASSERT_TRUE(graph.Execute(*rt).IsOk());
	EXPECT_NEAR(dstData[0], 2.0F, 1e-3F);
	for (OaU32 i = 0; i < N; ++i) dstData[i] = 0.0F;
	ASSERT_TRUE(rt->Allocator.FlushHostBuffer(dst, 0, dst.Size));
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

TEST(ComputeGraph, HostReadbackBarrierIsAnExplicitCompletionPolicy) {
	auto* rt = OaComputeEngine::GetGlobal();
	ASSERT_NE(rt, nullptr);

	auto srcRes = rt->AllocBuffer(256 * sizeof(OaF32));
	auto dstRes = rt->AllocBuffer(256 * sizeof(OaF32));
	ASSERT_TRUE(srcRes.IsOk() && dstRes.IsOk());
	auto src = std::move(*srcRes);
	auto dst = std::move(*dstRes);

	struct { OaU32 N; OaF32 Scale; } push{256, 1.0F};
	OaVkBuffer buffers[] = {src, dst};
	OaBufferAccess access[] = {OaBufferAccess::Read, OaBufferAccess::Write};

	OaComputeGraph graph;
	graph.Add("Scale", buffers, access, &push, sizeof(push), 1);
	ASSERT_TRUE(graph.Compile(*rt).IsOk());
	EXPECT_EQ(graph.GetStats().HostBarrierCount, 1U);

	graph.SetHostReadbackRequired(false);
	EXPECT_FALSE(graph.IsCompiled());
	ASSERT_TRUE(graph.Compile(*rt).IsOk());
	EXPECT_EQ(graph.GetStats().HostBarrierCount, 0U);

	graph.Destroy(rt->Device);
	rt->FreeBuffer(src);
	rt->FreeBuffer(dst);
}

TEST(ComputeGraph, ContextBatchUsesExactBoundariesAndReusesStaticGraphs) {
	auto* rt = OaComputeEngine::GetGlobal();
	ASSERT_NE(rt, nullptr);
	auto& ctx = OaContext::GetDefault();
	ctx.Clear();

	constexpr OaU32 N = 256;
	auto src = OaFnMatrix::Empty(OaMatrixShape{N}, OaScalarType::Float32);
	auto mid = OaFnMatrix::Empty(OaMatrixShape{N}, OaScalarType::Float32);
	auto dst = OaFnMatrix::Empty(OaMatrixShape{N}, OaScalarType::Float32);
	auto unrelatedSrc = OaFnMatrix::Empty(OaMatrixShape{N}, OaScalarType::Float32);
	auto unrelatedDst = OaFnMatrix::Empty(OaMatrixShape{N}, OaScalarType::Float32);
	ASSERT_TRUE(src.HasStorage() && mid.HasStorage() && dst.HasStorage());
	ASSERT_TRUE(unrelatedSrc.HasStorage() && unrelatedDst.HasStorage());

	for (OaU32 i = 0; i < N; ++i) {
		src.DataAs<OaF32>()[i] = static_cast<OaF32>(i + 1);
		unrelatedSrc.DataAs<OaF32>()[i] = static_cast<OaF32>(N - i);
		dst.DataAs<OaF32>()[i] = 0.0F;
	}
	ASSERT_TRUE(rt->Allocator.FlushHostBuffer(src.GetVkBuffer(), 0, src.ByteSize()));
	ASSERT_TRUE(rt->Allocator.FlushHostBuffer(
		unrelatedSrc.GetVkBuffer(), 0, unrelatedSrc.ByteSize()));
	ASSERT_TRUE(rt->Allocator.FlushHostBuffer(dst.GetVkBuffer(), 0, dst.ByteSize()));

	OaBufferAccess access[] = {OaBufferAccess::Read, OaBufferAccess::Write};
	auto recordScale = [&](const OaMatrix& InSrc, OaMatrix& OutDst, OaF32 InScale) {
		struct { OaU32 Count; OaF32 Scale; } push{N, InScale};
		ctx.Add("Scale", {&InSrc, &OutDst}, access, &push, sizeof(push), 1);
	};
	auto executeBatch = [&]() {
		recordScale(src, mid, 2.0F);
		EXPECT_TRUE(ctx.ExecuteInAsyncBatch().IsOk());
		recordScale(unrelatedSrc, unrelatedDst, 3.0F);
		EXPECT_TRUE(ctx.ExecuteInAsyncBatch().IsOk());
		recordScale(mid, dst, 4.0F);
		EXPECT_TRUE(ctx.ExecuteInAsyncBatch().IsOk());
		EXPECT_TRUE(ctx.FlushAsyncBatch().IsOk());
		EXPECT_TRUE(ctx.Sync().IsOk());
	};

	executeBatch();
	auto first = ctx.LastExecutionStats();
	EXPECT_EQ(first.GraphCount, 3U);
	EXPECT_EQ(first.BoundaryBarrierCount, 1U);
	EXPECT_EQ(first.HostBarrierCount, 1U);
	for (OaU32 i = 0; i < N; ++i) {
		EXPECT_NEAR(dst.DataAs<OaF32>()[i], static_cast<OaF32>(i + 1) * 8.0F, 1e-3F);
	}

	// The same stable buffers, topology and push constants must reuse all three
	// compiled secondary command buffers on the next step.
	executeBatch();
	const auto second = ctx.LastExecutionStats();
	EXPECT_EQ(second.GraphCount, 3U);
	EXPECT_EQ(second.CompileCacheHits, 3U);
	EXPECT_EQ(second.BoundaryBarrierCount, 1U);
	EXPECT_EQ(second.HostBarrierCount, 1U);

	ctx.Clear();
}

// =============================================================================
// GPU COMPILATION TESTS (Phase 2)
TEST(ComputeGraph, BarrierOverhead) {
	auto* rt = OaComputeEngine::GetGlobal();
	ASSERT_NE(rt, nullptr);
	
	PrintHeader("BARRIER ANALYSIS — CPU PLANNING COST");
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
	
	fprintf(stderr, "\n  Barriers are planned on the CPU and encoded once during graph compilation.\n");
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

TEST(ComputeGraph, AllocatorBackedAliasesExecuteCorrectly) {
	auto* rt = OaComputeEngine::GetGlobal();
	ASSERT_NE(rt, nullptr);
	constexpr OaU32 N = 256;
	OaVec<OaVkBuffer> buffers(5);
	for (auto& buffer : buffers) {
		auto result = rt->Allocator.AllocHostVisible(N * sizeof(OaF32));
		ASSERT_TRUE(result.IsOk());
		buffer = result.GetValue();
		rt->RegisterBuffer(buffer);
	}
	auto* input = static_cast<OaF32*>(buffers[0].MappedPtr);
	for (OaU32 i = 0; i < N; ++i) input[i] = static_cast<OaF32>(i + 1U);
	ASSERT_TRUE(rt->Allocator.FlushHostBuffer(buffers[0], 0, buffers[0].Size));

	OaComputeGraph graph;
	BuildChainGraph(graph, buffers, 4);
	OaVkBuffer eligible[] = {buffers[1], buffers[3]};
	ASSERT_TRUE(graph.MaterializeAliases(*rt, eligible).IsOk());
	EXPECT_EQ(graph.MaterializedAliasSavings(), N * sizeof(OaF32));
	ASSERT_TRUE(graph.Execute(*rt).IsOk());
	ASSERT_TRUE(rt->Allocator.InvalidateHostBuffer(buffers[4], 0, buffers[4].Size));
	const auto* output = static_cast<const OaF32*>(buffers[4].MappedPtr);
	const OaF32 factor = 1.001F * 1.001F * 1.001F * 1.001F;
	for (OaU32 i = 0; i < N; ++i) {
		EXPECT_NEAR(output[i], static_cast<OaF32>(i + 1U) * factor, 1e-4F);
	}

	graph.Destroy(rt->Device);
	for (auto& buffer : buffers) {
		rt->DeregisterBuffer(buffer);
		rt->Allocator.Free(buffer);
	}
}

TEST(OaDnnPlanner, PartitionsPackedQkvGatedFfnAndFallback) {
	OaDnnGraph graph;
	auto addMatrix = [&](OaDnnMatrixId id, OaMatrixShape shape, bool external) {
		ASSERT_TRUE(graph.AddMatrix({.Id = id, .Shape = shape,
			.Dtype = OaScalarType::Float32, .External = external,
			.Virtual = not external}).IsOk());
	};
	addMatrix(0, {8, 16}, true);       // shared activation
	addMatrix(1, {16, 16}, true); addMatrix(2, {16, 16}, true); addMatrix(3, {16, 16}, true);
	addMatrix(4, {8, 16}, false); addMatrix(5, {8, 16}, false); addMatrix(6, {8, 16}, false);
	addMatrix(7, {16, 16}, true); addMatrix(8, {16, 16}, true);
	addMatrix(9, {8, 16}, false); addMatrix(10, {8, 16}, false);
	addMatrix(11, {8, 16}, false); addMatrix(12, {8, 16}, false);
	addMatrix(13, {8, 16}, true);

	auto op = [&](OaDnnOpType type, std::initializer_list<OaDnnMatrixId> inputs,
		std::initializer_list<OaDnnMatrixId> outputs) {
		OaDnnOpDesc desc; desc.Type = type;
		desc.Inputs = inputs; desc.Outputs = outputs;
		ASSERT_TRUE(graph.AddOp(desc).IsOk());
	};
	op(OaDnnOpType::Matmul, {0, 1}, {4});
	op(OaDnnOpType::Matmul, {0, 2}, {5});
	op(OaDnnOpType::Matmul, {0, 3}, {6});
	op(OaDnnOpType::Matmul, {0, 7}, {9});
	op(OaDnnOpType::Matmul, {0, 8}, {10});
	op(OaDnnOpType::Silu, {9}, {11});
	op(OaDnnOpType::Multiply, {11, 10}, {12});
	op(OaDnnOpType::Add, {12, 0}, {13});

	auto result = OaDnnPlanner::Plan(graph);
	ASSERT_TRUE(result.IsOk()) << result.GetStatus().GetMessage().Data();
	const auto& plan = result.GetValue();
	ASSERT_EQ(plan.Partitions.Size(), 3U);
	EXPECT_EQ(plan.Partitions[0].Engine, OaDnnEngineType::PackedQkv);
	EXPECT_EQ(plan.Partitions[0].Ops.Size(), 3U);
	EXPECT_EQ(plan.Partitions[1].Engine, OaDnnEngineType::GatedFfn);
	EXPECT_EQ(plan.Partitions[1].Ops.Size(), 4U);
	EXPECT_EQ(plan.Partitions[2].Engine, OaDnnEngineType::Portable);
	EXPECT_NE(plan.GraphHash, 0U);
}

TEST(OaDnnPlanner, RejectsUseBeforeProducer) {
	OaDnnGraph graph;
	ASSERT_TRUE(graph.AddMatrix({.Id = 0, .Shape = {2, 2}, .External = false}).IsOk());
	ASSERT_TRUE(graph.AddMatrix({.Id = 1, .Shape = {2, 2}, .External = false}).IsOk());
	OaDnnOpDesc op; op.Type = OaDnnOpType::Relu; op.Inputs = {0}; op.Outputs = {1};
	ASSERT_TRUE(graph.AddOp(op).IsOk());
	EXPECT_FALSE(graph.Validate().IsOk());
}
// =============================================================================

// Main is provided by MlTestMain.cpp (shared test infrastructure)
