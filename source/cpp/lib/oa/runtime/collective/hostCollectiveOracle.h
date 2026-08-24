// Independent synchronous host collective correctness oracle.
//
// This is private test/reference machinery, not a device collective service.
// It uses only the host-visible ranges supplied by the caller and stages the
// complete result in host memory. Future device or network collectives require
// a separate engine-borrowing session with exact oa::Event completion.
//
// AllReduce: reduce across all participants, result on every participant.
// Broadcast: copy from source to all destinations.
// AllGather: gather partial buffers into full buffers on all participants.
// Scatter: split a buffer and distribute chunks to participants.
// ReduceScatter: reduce then scatter.

#pragma once

#include <oa/core/types.h>
#include <oa/core/status.h>
#include <oa/runtime/allocator.h>

namespace oa {

enum class HostReduceOp : oa::U8 {
	Sum,
	Max,
	Min,
};

class HostCollectiveOracle {
public:
	// Reduce inOutBufs across all participants, result available in every buffer.
	// All buffers must be the same size.
	[[nodiscard]] static oa::Status allReduce(
		oa::Span<oavk::Buffer> inOutBufs,
		HostReduceOp inOp);

	// Broadcast inOutBufs[inSrcIdx] to all other buffers in the span.
	[[nodiscard]] static oa::Status broadcast(
		oa::Span<oavk::Buffer> inOutBufs,
		oa::U32 inSrcIdx);

	// Gather partial buffers into full buffers for all participants.
	// inPartials[i] has size total/N. outFullBufs[i] has full size.
	[[nodiscard]] static oa::Status allGather(
		oa::Span<const oavk::Buffer> inPartials,
		oa::Span<oavk::Buffer> outFullBufs);

	// split inFull into N equal chunks, distribute chunk i to outPartials[i].
	[[nodiscard]] static oa::Status scatter(
		const oavk::Buffer& inFull,
		oa::Span<oavk::Buffer> outPartials);

	// ReduceScatter: reduce across all, then each device gets chunk i.
	// inOutBufs[i] has full size on input. On output, only the first 1/N is valid.
	[[nodiscard]] static oa::Status reduceScatter(
		oa::Span<oavk::Buffer> inOutBufs,
		HostReduceOp inOp);

};

} // namespace oa
