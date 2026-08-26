#include "oa/runtime/collective/hostCollectiveOracle.h"
#include <oa/core/memory.h>
#include <oa/core/simd.h>
#include <oa/core/std/limits.h>
#include <oa/core/std/scalarMath.h>

namespace oa {

// ─── CPU SIMD Reduce Helpers ──────────────────────────────────────────────

static void cpuReduceF32(
	oa::F32* inOutAcc,
	const oa::F32* inB,
	oa::I64 inCount,
	HostReduceOp inOp) {
	switch (inOp) {
		case HostReduceOp::Sum:
			oa::FnSimd::addF32(inOutAcc, inB, inCount);
			break;
		case HostReduceOp::Max:
			for (oa::I64 i = 0; i < inCount; ++i)
				inOutAcc[i] = oa::fmax(inOutAcc[i], inB[i]);
			break;
		case HostReduceOp::Min:
			for (oa::I64 i = 0; i < inCount; ++i)
				inOutAcc[i] = oa::fmin(inOutAcc[i], inB[i]);
			break;
	}
}

// ─── Validation Helpers ──────────────────────────────────────────────────

static oa::Status validateEqualHostBuffers(oa::Span<oavk::Buffer> inBufs) {
	if (inBufs.empty()) {
		return oa::Status::error(oa::StatusCode::InvalidArgument, "empty buffer span");
	}
	const oa::U64 size = inBufs[0].size;
	if (size > static_cast<oa::U64>(
			oa::Limits<oa::Vec<oa::U8>::size_type>::max())) {
		return oa::Status::error(oa::StatusCode::InvalidArgument,
			"collective buffer size exceeds the host staging limit");
	}
	for (oa::U32 i = 0; i < inBufs.size(); ++i) {
		if (inBufs[i].size != size) {
			return oa::Status::error(oa::StatusCode::InvalidArgument,
				"all buffers must have the same size for collective ops");
		}
		if (size != 0 and inBufs[i].mappedPtr == nullptr) {
			return oa::Status::error(oa::StatusCode::InvalidArgument,
				"collective requires host-visible buffers (mappedPtr must be valid)");
		}
	}
	return oa::Status::ok();
}

static oa::Status validateReduceBuffers(
	oa::Span<oavk::Buffer> inBufs,
	HostReduceOp inOp) {
	OA_RETURN_IF_ERROR(validateEqualHostBuffers(inBufs));
	if (inOp != HostReduceOp::Sum
		and inOp != HostReduceOp::Max
		and inOp != HostReduceOp::Min) {
		return oa::Status::error(oa::StatusCode::InvalidArgument,
			"unsupported collective reduction operation");
	}
	if (inBufs[0].size % sizeof(oa::F32) != 0) {
		return oa::Status::error(oa::StatusCode::InvalidArgument,
			"collective reduction buffer size must be a multiple of sizeof(f32)");
	}
	if (inBufs[0].size / sizeof(oa::F32)
		> static_cast<oa::U64>(oa::Limits<oa::I64>::max())) {
		return oa::Status::error(oa::StatusCode::InvalidArgument,
			"collective reduction element count exceeds the SIMD backend limit");
	}
	return oa::Status::ok();
}

// ─── Span-of-buffers API ───────────────────────────────────────────────

oa::Status HostCollectiveOracle::allReduce(
	oa::Span<oavk::Buffer> inOutBufs,
	HostReduceOp inOp)
{
	OA_RETURN_IF_ERROR(validateReduceBuffers(inOutBufs, inOp));
	const oa::U64 size = inOutBufs[0].size;
	if (size == 0 or inOutBufs.size() == 1) return oa::Status::ok();

	const oa::I64 count = static_cast<oa::I64>(size / sizeof(oa::F32));
	oa::Vec<oa::F32> reduced(static_cast<oa::Vec<oa::F32>::size_type>(count));
	oa::memcpy(reduced.data(), inOutBufs[0].mappedPtr, size);
	for (oa::U32 i = 1; i < inOutBufs.size(); ++i) {
		cpuReduceF32(
			reduced.data(),
			static_cast<const oa::F32*>(inOutBufs[i].mappedPtr),
			count,
			inOp);
	}
	for (oavk::Buffer& buffer : inOutBufs) {
		oa::memcpy(buffer.mappedPtr, reduced.data(), size);
	}

	return oa::Status::ok();
}

oa::Status HostCollectiveOracle::broadcast(
	oa::Span<oavk::Buffer> inOutBufs,
	oa::U32 inSrcIdx)
{
	OA_RETURN_IF_ERROR(validateEqualHostBuffers(inOutBufs));
	if (inSrcIdx >= inOutBufs.size())
		return oa::Status::error(oa::StatusCode::InvalidArgument, "inSrcIdx out of range");

	const auto& src = inOutBufs[inSrcIdx];
	if (src.size == 0 or inOutBufs.size() == 1) return oa::Status::ok();

	oa::Vec<oa::U8> staged(static_cast<oa::Vec<oa::U8>::size_type>(src.size));
	oa::memcpy(staged.data(), src.mappedPtr, src.size);
	for (oa::U32 i = 0; i < inOutBufs.size(); ++i) {
		if (i == inSrcIdx) continue;
		oa::memcpy(inOutBufs[i].mappedPtr, staged.data(), src.size);
	}
	return oa::Status::ok();
}

oa::Status HostCollectiveOracle::allGather(
	oa::Span<const oavk::Buffer> inPartials,
	oa::Span<oavk::Buffer> outFullBufs)
{
	const oa::U32 n = static_cast<oa::U32>(inPartials.size());
	if (n == 0) {
		return oa::Status::error(oa::StatusCode::InvalidArgument,
			"AllGather: empty partial buffer span");
	}
	if (outFullBufs.size() != inPartials.size())
		return oa::Status::error(oa::StatusCode::InvalidArgument,
			"AllGather: partials and full buffer counts must match");

	const oa::U64 partialSize = inPartials[0].size;
	if (partialSize > oa::Limits<oa::U64>::max() / n) {
		return oa::Status::error(oa::StatusCode::InvalidArgument,
			"AllGather: total output size overflows");
	}
	const oa::U64 fullSize = partialSize * n;
	if (fullSize > static_cast<oa::U64>(
			oa::Limits<oa::Vec<oa::U8>::size_type>::max())) {
		return oa::Status::error(oa::StatusCode::InvalidArgument,
			"AllGather: output size exceeds the host staging limit");
	}
	for (oa::U32 i = 0; i < n; ++i) {
		if (inPartials[i].size != partialSize) {
			return oa::Status::error(oa::StatusCode::InvalidArgument,
				"AllGather: partial buffer sizes must match");
		}
		if (outFullBufs[i].size < fullSize) {
			return oa::Status::error(oa::StatusCode::InvalidArgument,
				"AllGather: output buffer is too small");
		}
		if (partialSize != 0 and inPartials[i].mappedPtr == nullptr) {
			return oa::Status::error(oa::StatusCode::InvalidArgument,
				"AllGather: partial buffer is not host-visible");
		}
		if (fullSize != 0 and outFullBufs[i].mappedPtr == nullptr) {
			return oa::Status::error(oa::StatusCode::InvalidArgument,
				"AllGather: output buffer is not host-visible");
		}
	}
	if (fullSize == 0) return oa::Status::ok();

	oa::Vec<oa::U8> gathered(static_cast<oa::Vec<oa::U8>::size_type>(fullSize));
	for (oa::U32 src = 0; src < n; ++src) {
		const oa::U64 offset = src * partialSize;
		oa::memcpy(gathered.data() + offset, inPartials[src].mappedPtr, partialSize);
	}
	for (oavk::Buffer& output : outFullBufs) {
		oa::memcpy(output.mappedPtr, gathered.data(), fullSize);
	}
	return oa::Status::ok();
}

oa::Status HostCollectiveOracle::scatter(
	const oavk::Buffer& inFull,
	oa::Span<oavk::Buffer> outPartials
) {
	const oa::U32 n = static_cast<oa::U32>(outPartials.size());
	if (n == 0) {
		return oa::Status::error(oa::StatusCode::InvalidArgument,
			"Scatter: empty output buffer span");
	}
	if (inFull.size % n != 0) {
		return oa::Status::error(oa::StatusCode::InvalidArgument,
			"Scatter: source size must be divisible by the output count");
	}
	if (inFull.size != 0 and inFull.mappedPtr == nullptr)
		return oa::Status::error(oa::StatusCode::InvalidArgument, "source buffer not host-visible");

	const oa::U64 chunkSize = inFull.size / n;
	for (const oavk::Buffer& partial : outPartials) {
		if (partial.size < chunkSize) {
			return oa::Status::error(oa::StatusCode::InvalidArgument,
				"Scatter: output buffer is too small");
		}
		if (chunkSize != 0 and partial.mappedPtr == nullptr) {
			return oa::Status::error(oa::StatusCode::InvalidArgument,
				"Scatter: output buffer is not host-visible");
		}
	}
	if (chunkSize == 0) return oa::Status::ok();

	oa::Vec<oa::U8> staged(static_cast<oa::Vec<oa::U8>::size_type>(inFull.size));
	oa::memcpy(staged.data(), inFull.mappedPtr, inFull.size);
	for (oa::U32 i = 0; i < n; ++i) {
		oa::memcpy(
			outPartials[i].mappedPtr,
			staged.data() + i * chunkSize,
			chunkSize);
	}
	return oa::Status::ok();
}

oa::Status HostCollectiveOracle::reduceScatter(
	oa::Span<oavk::Buffer> inOutBufs,
	HostReduceOp inOp)
{
	OA_RETURN_IF_ERROR(validateReduceBuffers(inOutBufs, inOp));
	const oa::U32 n = static_cast<oa::U32>(inOutBufs.size());
	const oa::U64 size = inOutBufs[0].size;
	if (size == 0 or n == 1) return oa::Status::ok();

	if (size % n != 0 or (size / n) % sizeof(oa::F32) != 0) {
		return oa::Status::error(oa::StatusCode::InvalidArgument,
			"ReduceScatter: buffer size must divide into whole f32 chunks");
	}
	const oa::U64 chunkSize = size / n;
	const oa::I64 count = static_cast<oa::I64>(size / sizeof(oa::F32));
	oa::Vec<oa::F32> reduced(static_cast<oa::Vec<oa::F32>::size_type>(count));
	oa::memcpy(reduced.data(), inOutBufs[0].mappedPtr, size);
	for (oa::U32 i = 1; i < n; ++i) {
		cpuReduceF32(
			reduced.data(),
			static_cast<const oa::F32*>(inOutBufs[i].mappedPtr),
			count,
			inOp);
	}
	for (oa::U32 i = 0; i < n; ++i) {
		oa::memcpy(
			inOutBufs[i].mappedPtr,
			reinterpret_cast<const oa::U8*>(reduced.data()) + i * chunkSize,
			chunkSize);
	}

	return oa::Status::ok();
}

} // namespace oa
