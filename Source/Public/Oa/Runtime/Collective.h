// OaCollective — Multi-device collective operations
//
// Stateless synchronous host transformations. The current backend does not
// borrow OaEngine because it uses only the host-visible ranges supplied by the
// caller; a future device/queue implementation must use engine-owned execution.
// Span-of-buffers API: each buffer in the span is one logical participant.
// NodeIndex remains allocation metadata; this host backend does not route by it.
// The current correctness backend requires host-visible buffers and stages the
// complete result in host memory. Non-blocking and peer-to-peer variants remain
// unpublished until an engine-owned execution/event path exists.
//
// AllReduce: reduce across all devices, result on every device.
// Broadcast: copy from source to all destinations.
// AllGather: gather partial buffers from each device into full buffer on all.
// Scatter: split a buffer and distribute chunks to devices.
// ReduceScatter: reduce then scatter.

#pragma once

#include <Oa/Core/Types.h>
#include <Oa/Core/Status.h>
#include <Oa/Runtime/Allocator.h>

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

	// Reduce InOutBufs across all participants, result available in every buffer.
	// All buffers must be the same size.
	[[nodiscard]] static OaStatus AllReduce(
		OaSpan<OaVkBuffer> InOutBufs,
		OaReduceOp InOp);

	// Broadcast InOutBufs[InSrcIdx] to all other buffers in the span.
	[[nodiscard]] static OaStatus Broadcast(
		OaSpan<OaVkBuffer> InOutBufs,
		OaU32 InSrcIdx);

	// Gather partial buffers from each device into full buffers on all devices.
	// InPartials[i] has size total/N on device i. OutFullBufs[i] has full size on device i.
	[[nodiscard]] static OaStatus AllGather(
		OaSpan<const OaVkBuffer> InPartials,
		OaSpan<OaVkBuffer> OutFullBufs);

	// Split InFull into N equal chunks, distribute chunk i to OutPartials[i].
	[[nodiscard]] static OaStatus Scatter(
		const OaVkBuffer& InFull,
		OaSpan<OaVkBuffer> OutPartials);

	// ReduceScatter: reduce across all, then each device gets chunk i.
	// InOutBufs[i] has full size on input. On output, only the first 1/N is valid.
	[[nodiscard]] static OaStatus ReduceScatter(
		OaSpan<OaVkBuffer> InOutBufs,
		OaReduceOp InOp);

};
