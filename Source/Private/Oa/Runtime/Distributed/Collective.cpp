#include <Oa/Runtime/Collective.h>
#include <Oa/Runtime/Engine.h>
#include <Oa/Runtime/Topology.h>
#include <Oa/Runtime/OaVk.h>
#include <Oa/Core/Memory.h>
#include <Oa/Core/Simd.h>
#include <Oa/Core/Log.h>
#include <Oa/Core/Thread.h>

#include <algorithm>
#include <cmath>
#include <atomic>

// ─── CPU SIMD Reduce Helpers ──────────────────────────────────────────────

static void CpuReduceF32(OaF32* InOutAcc, const OaF32* InB, OaI64 InCount, OaReduceOp InOp) {
	switch (InOp) {
		case OaReduceOp::Sum:
			OaSimd::AddF32(InOutAcc, InB, InCount);
			break;
		case OaReduceOp::Max:
			for (OaI64 i = 0; i < InCount; ++i)
				InOutAcc[i] = std::fmax(InOutAcc[i], InB[i]);
			break;
		case OaReduceOp::Min:
			for (OaI64 i = 0; i < InCount; ++i)
				InOutAcc[i] = std::fmin(InOutAcc[i], InB[i]);
			break;
	}
}

// ─── Validation Helpers ──────────────────────────────────────────────────

static OaStatus ValidateBuffers(OaSpan<OaVkBuffer> InBufs) {
	if (InBufs.size() == 0)
		return OaStatus::Error(OaStatusCode::InvalidArgument, "empty buffer span");
	OaU64 size = InBufs[0].Size;
	for (OaU32 i = 0; i < InBufs.size(); ++i) {
		if (!InBufs[i].MappedPtr)
			return OaStatus::Error(OaStatusCode::InvalidArgument,
				"collective requires host-visible buffers (MappedPtr must be valid)");
		if (InBufs[i].Size != size)
			return OaStatus::Error(OaStatusCode::InvalidArgument,
				"all buffers must have the same size for collective ops");
	}
	return OaStatus::Ok();
}

// ─── Span-of-Buffers API ───────────────────────────────────────────────

OaStatus OaCollective::AllReduce(
	OaComputeEngine& InRt,
	OaSpan<OaVkBuffer> InOutBufs,
	OaReduceOp InOp)
{
	if (InOutBufs.size() <= 1) return OaStatus::Ok();
	OA_RETURN_IF_ERROR(ValidateBuffers(InOutBufs));

	OaU32 n = static_cast<OaU32>(InOutBufs.size());
	OaU64 size = InOutBufs[0].Size;
	OaI64 count = static_cast<OaI64>(size / sizeof(OaF32));

	if (n == 2) {
		// Flat reduce: accumulate buf[1] into buf[0], then copy buf[0] to buf[1].
		auto* acc = static_cast<OaF32*>(InOutBufs[0].MappedPtr);
		auto* src = static_cast<OaF32*>(InOutBufs[1].MappedPtr);

		OaDeviceMesh* mesh = InRt.GetMesh();
		if (mesh) {
			OA_RETURN_IF_ERROR(mesh->EnsureScratch(size));
			OaMemcpy(mesh->ScratchBuf.MappedPtr, src, size);
			CpuReduceF32(acc, static_cast<const OaF32*>(mesh->ScratchBuf.MappedPtr), count, InOp);
		} else {
			CpuReduceF32(acc, src, count, InOp);
		}
		OaMemcpy(InOutBufs[1].MappedPtr, InOutBufs[0].MappedPtr, size);
		return OaStatus::Ok();
	}

	// Ring AllReduce for N > 2
	if (size % (static_cast<OaU64>(n) * sizeof(OaF32)) != 0) {
		return OaStatus::Error(OaStatusCode::InvalidArgument,
			"AllReduce ring: buffer size must be divisible by (N * sizeof(f32))");
	}
	OaU64 chunkSize = size / n;
	OaU64 chunkCount = chunkSize / sizeof(OaF32);

	OaDeviceMesh* mesh = InRt.GetMesh();
	if (mesh) {
		OA_RETURN_IF_ERROR(mesh->EnsureScratch(chunkSize));
	}

	OaVec<OaU8> stageAllReduce(chunkSize);

	// Phase 1: Reduce-Scatter via ring
	for (OaU32 step = 0; step < n - 1; ++step) {
		for (OaU32 i = 0; i < n; ++i) {
			OaU32 recvFrom = (i - 1 + n) % n;
			OaU32 accChunk = (i - step + n) % n;

			OaU64 offset = accChunk * chunkSize;
			auto* dst = reinterpret_cast<OaF32*>(
				static_cast<OaU8*>(InOutBufs[i].MappedPtr) + offset);
			auto* incoming = static_cast<const OaU8*>(InOutBufs[recvFrom].MappedPtr) + offset;

			OaMemcpy(stageAllReduce.Data(), incoming, chunkSize);
			CpuReduceF32(dst, reinterpret_cast<const OaF32*>(stageAllReduce.Data()),
				static_cast<OaI64>(chunkCount), InOp);
		}
	}

	// Phase 2: AllGather via ring — node i has reduced chunk i, spread to all
	for (OaU32 step = 0; step < n - 1; ++step) {
		for (OaU32 i = 0; i < n; ++i) {
			OaU32 recvFrom = (i - 1 + n) % n;
			OaU32 copyChunk = (i - step - 1 + n) % n;

			OaU64 offset = copyChunk * chunkSize;
			auto* dst = static_cast<OaU8*>(InOutBufs[i].MappedPtr) + offset;
			auto* src2 = static_cast<const OaU8*>(InOutBufs[recvFrom].MappedPtr) + offset;

			OaMemcpy(stageAllReduce.Data(), src2, chunkSize);
			OaMemcpy(dst, stageAllReduce.Data(), chunkSize);
		}
	}

	return OaStatus::Ok();
}

OaStatus OaCollective::Broadcast(
	OaComputeEngine& InRt,
	OaSpan<OaVkBuffer> InOutBufs,
	OaU32 InSrcIdx)
{
	(void)InRt;
	if (InOutBufs.size() <= 1) return OaStatus::Ok();
	if (InSrcIdx >= InOutBufs.size())
		return OaStatus::Error(OaStatusCode::InvalidArgument, "InSrcIdx out of range");

	const auto& src = InOutBufs[InSrcIdx];
	if (!src.MappedPtr)
		return OaStatus::Error(OaStatusCode::InvalidArgument, "source buffer not host-visible");

	for (OaU32 i = 0; i < InOutBufs.size(); ++i) {
		if (i == InSrcIdx) continue;
		if (!InOutBufs[i].MappedPtr)
			return OaStatus::Error(OaStatusCode::InvalidArgument,
				"destination buffer not host-visible");
		OaMemcpy(InOutBufs[i].MappedPtr, src.MappedPtr, src.Size);
	}
	return OaStatus::Ok();
}

OaStatus OaCollective::AllGather(
	OaComputeEngine& InRt,
	OaSpan<const OaVkBuffer> InPartials,
	OaSpan<OaVkBuffer> OutFullBufs)
{
	(void)InRt;
	OaU32 n = static_cast<OaU32>(InPartials.size());
	if (n <= 1) {
		if (n == 1 && OutFullBufs.size() >= 1 && InPartials[0].MappedPtr && OutFullBufs[0].MappedPtr)
			OaMemcpy(OutFullBufs[0].MappedPtr, InPartials[0].MappedPtr, InPartials[0].Size);
		return OaStatus::Ok();
	}

	if (OutFullBufs.size() != InPartials.size())
		return OaStatus::Error(OaStatusCode::InvalidArgument,
			"AllGather: partials and full buffer counts must match");

	OaU64 partialSize = InPartials[0].Size;

	// Gather all partials into each full buffer
	for (OaU32 dst = 0; dst < n; ++dst) {
		if (!OutFullBufs[dst].MappedPtr)
			return OaStatus::Error(OaStatusCode::InvalidArgument, "output buffer not host-visible");
		for (OaU32 src = 0; src < n; ++src) {
			if (!InPartials[src].MappedPtr)
				return OaStatus::Error(OaStatusCode::InvalidArgument, "partial buffer not host-visible");
			OaU64 offset = src * partialSize;
			OaMemcpy(
				static_cast<OaU8*>(OutFullBufs[dst].MappedPtr) + offset,
				InPartials[src].MappedPtr,
				partialSize);
		}
	}
	return OaStatus::Ok();
}

OaStatus OaCollective::Scatter(
	OaComputeEngine& InRt,
	const OaVkBuffer& InFull,
	OaSpan<OaVkBuffer> OutPartials
) {
	(void)InRt;
	OaU32 n = static_cast<OaU32>(OutPartials.size());
	if (n == 0) return OaStatus::Ok();
	if (!InFull.MappedPtr)
		return OaStatus::Error(OaStatusCode::InvalidArgument, "source buffer not host-visible");

	OaU64 chunkSize = InFull.Size / n;

	for (OaU32 i = 0; i < n; ++i) {
		if (!OutPartials[i].MappedPtr)
			return OaStatus::Error(OaStatusCode::InvalidArgument, "partial buffer not host-visible");
		OaMemcpy(
			OutPartials[i].MappedPtr,
			static_cast<const OaU8*>(InFull.MappedPtr) + i * chunkSize,
			chunkSize);
	}
	return OaStatus::Ok();
}

OaStatus OaCollective::ReduceScatter(
	OaComputeEngine& InRt,
	OaSpan<OaVkBuffer> InOutBufs,
	OaReduceOp InOp)
{
	(void)InRt;
	OaU32 n = static_cast<OaU32>(InOutBufs.size());
	if (n <= 1) return OaStatus::Ok();
	OA_RETURN_IF_ERROR(ValidateBuffers(InOutBufs));

	OaU64 size = InOutBufs[0].Size;
	OaU64 chunkSize = size / n;
	OaI64 chunkCount = static_cast<OaI64>(chunkSize / sizeof(OaF32));

	if (size % (static_cast<OaU64>(n) * sizeof(OaF32)) != 0) {
		return OaStatus::Error(OaStatusCode::InvalidArgument,
			"ReduceScatter ring: buffer size must be divisible by (N * sizeof(f32))");
	}

	OaVec<OaU8> stageBuf(chunkSize);

	for (OaU32 step = 0; step < n - 1; ++step) {
		for (OaU32 i = 0; i < n; ++i) {
			OaU32 recvFrom = (i - 1 + n) % n;
			OaU32 accChunk = (i - step + n) % n;

			OaU64 offset = accChunk * chunkSize;
			auto* dst = reinterpret_cast<OaF32*>(
				static_cast<OaU8*>(InOutBufs[i].MappedPtr) + offset);
			auto* incoming = static_cast<const OaU8*>(InOutBufs[recvFrom].MappedPtr) + offset;

			OaMemcpy(stageBuf.Data(), incoming, chunkSize);
			CpuReduceF32(dst, reinterpret_cast<const OaF32*>(stageBuf.Data()), chunkCount, InOp);
		}
	}

	return OaStatus::Ok();
}

// ─── Async Variants (Overlap Pattern) ────────────────────────────────────

static OaThreadPool& GetCollectivePool() {
	OaThreadPoolConfig cfg;
	cfg.NumWorkers = 2;
	cfg.PinToCores = false;
	static OaThreadPool pool = OaThreadPool::Create(cfg);
	return pool;
}

OaResult<OaDispatchTicket> OaCollective::AllReduceAsync(
	OaComputeEngine& InRt,
	OaSpan<OaVkBuffer> InOutBufs,
	OaReduceOp InOp)
{
	if (InOutBufs.size() <= 1) {
		return OaDispatchTicket{};
	}

	auto semResult = OaVkTimelineSemaphore::Create(InRt.Device, 0);
	if (!semResult.IsOk()) return semResult.GetStatus();

	auto sem = std::move(semResult.GetValue());
	OaU64 signalValue = 1;

	OaVec<OaVkBuffer> bufsCopy(InOutBufs.begin(), InOutBufs.end());
	VkSemaphore rawSem = static_cast<VkSemaphore>(sem.Semaphore);
	VkDevice rawDev = static_cast<VkDevice>(InRt.Device.Device);
	auto* rtPtr = &InRt;

	GetCollectivePool().Submit([bufsCopy = std::move(bufsCopy), InOp, rawSem, rawDev, signalValue, rtPtr]() mutable {
		OaSpan<OaVkBuffer> span(bufsCopy.Data(), bufsCopy.Size());
		auto status = OaCollective::AllReduce(*rtPtr, span, InOp);
		if (!status.IsOk()) {
			OA_LOG_ERROR(OaLogComponent::Core, "async AllReduce failed: %s",
				status.GetMessage().c_str());
		}

		VkSemaphoreSignalInfo signalInfo{};
		signalInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO;
		signalInfo.semaphore = rawSem;
		signalInfo.value = signalValue;
		vkSignalSemaphore(rawDev, &signalInfo);
	});

	OaDispatchTicket ticket;
	ticket.Semaphore = sem;
	sem.Semaphore = nullptr;
	ticket.Value = signalValue;
	ticket.NodeIndex = 0;
	return ticket;
}

OaResult<OaDispatchTicket> OaCollective::BroadcastAsync(
	OaComputeEngine& InRt,
	OaSpan<OaVkBuffer> InOutBufs,
	OaU32 InSrcIdx)
{
	if (InOutBufs.size() <= 1) {
		return OaDispatchTicket{};
	}

	auto semResult = OaVkTimelineSemaphore::Create(InRt.Device, 0);
	if (!semResult.IsOk()) return semResult.GetStatus();

	auto sem = std::move(semResult.GetValue());
	OaU64 signalValue = 1;

	OaVec<OaVkBuffer> bufsCopy(InOutBufs.begin(), InOutBufs.end());
	VkSemaphore rawSem = static_cast<VkSemaphore>(sem.Semaphore);
	VkDevice rawDev = static_cast<VkDevice>(InRt.Device.Device);
	auto* rtPtr = &InRt;

	GetCollectivePool().Submit([bufsCopy = std::move(bufsCopy), InSrcIdx, rawSem, rawDev, signalValue, rtPtr]() mutable {
		OaSpan<OaVkBuffer> span(bufsCopy.Data(), bufsCopy.Size());
		auto status = OaCollective::Broadcast(*rtPtr, span, InSrcIdx);
		if (!status.IsOk()) {
			OA_LOG_ERROR(OaLogComponent::Core, "async Broadcast failed: %s",
				status.GetMessage().c_str());
		}

		VkSemaphoreSignalInfo signalInfo{};
		signalInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO;
		signalInfo.semaphore = rawSem;
		signalInfo.value = signalValue;
		vkSignalSemaphore(rawDev, &signalInfo);
	});

	OaDispatchTicket ticket;
	ticket.Semaphore = sem;
	sem.Semaphore = nullptr;
	ticket.Value = signalValue;
	ticket.NodeIndex = 0;
	return ticket;
}

// ─── Single-Buffer Backward-Compat API ───────────────────────────────────

OaStatus OaCollective::AllReduce(
	OaComputeEngine& InRt,
	OaVkBuffer& InOutBuf,
	OaReduceOp InOp,
	OaSpan<const OaU32> InNodes)
{
	(void)InRt;
	(void)InOutBuf;
	(void)InOp;
	if (InNodes.size() <= 1) return OaStatus::Ok();
	return OaStatus::Unimplemented("single-buffer AllReduce: use span-of-buffers API");
}

OaStatus OaCollective::Broadcast(
	OaComputeEngine& InRt,
	const OaVkBuffer& InSrc,
	OaU32 InSrcNode,
	OaSpan<const OaU32> InDstNodes)
{
	(void)InRt;
	(void)InSrc;
	(void)InSrcNode;
	if (InDstNodes.size() == 0) return OaStatus::Ok();
	return OaStatus::Unimplemented("single-buffer Broadcast: use span-of-buffers API");
}

OaStatus OaCollective::AllGather(
	OaComputeEngine& InRt,
	OaSpan<const OaVkBuffer> InPartials,
	OaVkBuffer& OutFull)
{
	(void)InRt;
	(void)OutFull;
	if (InPartials.size() <= 1) return OaStatus::Ok();
	return OaStatus::Unimplemented("single-buffer AllGather: use span-of-buffers API");
}

OaStatus OaCollective::Scatter(
	OaComputeEngine& InRt,
	const OaVkBuffer& InFull,
	OaU32 InSrcNode,
	OaSpan<OaVkBuffer> OutPartials)
{
	(void)InRt;
	(void)InFull;
	(void)InSrcNode;
	if (OutPartials.size() <= 1) return OaStatus::Ok();
	return OaStatus::Unimplemented("single-buffer Scatter: use span-of-buffers API");
}
