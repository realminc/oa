// OaCollective — Multi-device collective operations
//
// Span-of-buffers API: each buffer in the span is on a different device node.
// Transport-aware: auto-selects host-staging or DMA-BUF per link.
// Sync and async variants — async returns OaDispatchTicket for overlap.
//
// AllReduce: reduce across all devices, result on every device.
// Broadcast: copy from source to all destinations.
// AllGather: gather partial buffers from each device into full buffer on all.
// Scatter: split a buffer and distribute chunks to devices.
// ReduceScatter: reduce then scatter (ring reduce-scatter).

#pragma once

#include <Oa/Core/Types.h>
#include <Oa/Core/Status.h>
#include <Oa/Runtime/Allocator.h>
#include <Oa/Runtime/Scheduler.h>

class OaComputeEngine;

enum class OaCollectiveOp : OaU8 {
	AllReduce,
	AllGather,
	Broadcast,
	Scatter,
	ReduceScatter,
};

enum class OaReduceOp : OaU8 {
	Sum,
	Max,
	Min,
};

[[nodiscard]] constexpr OaStringView OaReduceOpName(OaReduceOp InOp) noexcept {
	switch (InOp) {
		case OaReduceOp::Sum: return "Sum";
		case OaReduceOp::Max: return "Max";
		case OaReduceOp::Min: return "Min";
		default:              return "Unknown";
	}
}

class OaCollective {
public:
	// ─── Span-of-buffers API (primary — one buffer per participating device) ──

	// Reduce InOutBufs across all devices, result available on every device.
	// InOutBufs[i].NodeIndex identifies which device each buffer lives on.
	// All buffers must be the same size.
	[[nodiscard]] static OaStatus AllReduce(
		OaComputeEngine& InRt,
		OaSpan<OaVkBuffer> InOutBufs,
		OaReduceOp InOp);

	// Broadcast InOutBufs[InSrcIdx] to all other buffers in the span.
	[[nodiscard]] static OaStatus Broadcast(
		OaComputeEngine& InRt,
		OaSpan<OaVkBuffer> InOutBufs,
		OaU32 InSrcIdx);

	// Gather partial buffers from each device into full buffers on all devices.
	// InPartials[i] has size total/N on device i. OutFullBufs[i] has full size on device i.
	[[nodiscard]] static OaStatus AllGather(
		OaComputeEngine& InRt,
		OaSpan<const OaVkBuffer> InPartials,
		OaSpan<OaVkBuffer> OutFullBufs);

	// Split InFull into N equal chunks, distribute chunk i to OutPartials[i].
	[[nodiscard]] static OaStatus Scatter(
		OaComputeEngine& InRt,
		const OaVkBuffer& InFull,
		OaSpan<OaVkBuffer> OutPartials);

	// ReduceScatter: reduce across all, then each device gets chunk i.
	// InOutBufs[i] has full size on input. On output, only the first 1/N is valid.
	[[nodiscard]] static OaStatus ReduceScatter(
		OaComputeEngine& InRt,
		OaSpan<OaVkBuffer> InOutBufs,
		OaReduceOp InOp);

	// ─── Async variants (overlap pattern) ───────────────────────────────────

	// Async AllReduce: returns ticket, caller waits before optimizer step.
	[[nodiscard]] static OaResult<OaDispatchTicket> AllReduceAsync(
		OaComputeEngine& InRt,
		OaSpan<OaVkBuffer> InOutBufs,
		OaReduceOp InOp);

	// Async Broadcast: returns ticket.
	[[nodiscard]] static OaResult<OaDispatchTicket> BroadcastAsync(
		OaComputeEngine& InRt,
		OaSpan<OaVkBuffer> InOutBufs,
		OaU32 InSrcIdx);

	// ─── Single-buffer backward-compat (single-device no-op) ────────────────

	[[nodiscard]] static OaStatus AllReduce(
		OaComputeEngine& InRt,
		OaVkBuffer& InOutBuf,
		OaReduceOp InOp,
		OaSpan<const OaU32> InNodes);

	[[nodiscard]] static OaStatus Broadcast(
		OaComputeEngine& InRt,
		const OaVkBuffer& InSrc,
		OaU32 InSrcNode,
		OaSpan<const OaU32> InDstNodes);

	[[nodiscard]] static OaStatus AllGather(
		OaComputeEngine& InRt,
		OaSpan<const OaVkBuffer> InPartials,
		OaVkBuffer& OutFull);

	[[nodiscard]] static OaStatus Scatter(
		OaComputeEngine& InRt,
		const OaVkBuffer& InFull,
		OaU32 InSrcNode,
		OaSpan<OaVkBuffer> OutPartials);
};
