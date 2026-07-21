#include <Oa/Runtime/Collective.h>
#include <Oa/Core/Memory.h>
#include <Oa/Core/Simd.h>

#include <cmath>
#include <limits>

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

static OaStatus ValidateEqualHostBuffers(OaSpan<OaVkBuffer> InBufs) {
	if (InBufs.empty()) {
		return OaStatus::Error(OaStatusCode::InvalidArgument, "empty buffer span");
	}
	const OaU64 size = InBufs[0].Size;
	if (size > static_cast<OaU64>(
			std::numeric_limits<OaVec<OaU8>::size_type>::max())) {
		return OaStatus::Error(OaStatusCode::InvalidArgument,
			"collective buffer size exceeds the host staging limit");
	}
	for (OaU32 i = 0; i < InBufs.size(); ++i) {
		if (InBufs[i].Size != size) {
			return OaStatus::Error(OaStatusCode::InvalidArgument,
				"all buffers must have the same size for collective ops");
		}
		if (size != 0 and InBufs[i].MappedPtr == nullptr) {
			return OaStatus::Error(OaStatusCode::InvalidArgument,
				"collective requires host-visible buffers (MappedPtr must be valid)");
		}
	}
	return OaStatus::Ok();
}

static OaStatus ValidateReduceBuffers(
	OaSpan<OaVkBuffer> InBufs,
	OaReduceOp InOp) {
	OA_RETURN_IF_ERROR(ValidateEqualHostBuffers(InBufs));
	if (InOp != OaReduceOp::Sum
		and InOp != OaReduceOp::Max
		and InOp != OaReduceOp::Min) {
		return OaStatus::Error(OaStatusCode::InvalidArgument,
			"unsupported collective reduction operation");
	}
	if (InBufs[0].Size % sizeof(OaF32) != 0) {
		return OaStatus::Error(OaStatusCode::InvalidArgument,
			"collective reduction buffer size must be a multiple of sizeof(f32)");
	}
	if (InBufs[0].Size / sizeof(OaF32)
		> static_cast<OaU64>(std::numeric_limits<OaI64>::max())) {
		return OaStatus::Error(OaStatusCode::InvalidArgument,
			"collective reduction element count exceeds the SIMD backend limit");
	}
	return OaStatus::Ok();
}

// ─── Span-of-Buffers API ───────────────────────────────────────────────

OaStatus OaCollective::AllReduce(
	OaSpan<OaVkBuffer> InOutBufs,
	OaReduceOp InOp)
{
	OA_RETURN_IF_ERROR(ValidateReduceBuffers(InOutBufs, InOp));
	const OaU64 size = InOutBufs[0].Size;
	if (size == 0 or InOutBufs.size() == 1) return OaStatus::Ok();

	const OaI64 count = static_cast<OaI64>(size / sizeof(OaF32));
	OaVec<OaF32> reduced(static_cast<OaVec<OaF32>::size_type>(count));
	OaMemcpy(reduced.Data(), InOutBufs[0].MappedPtr, size);
	for (OaU32 i = 1; i < InOutBufs.size(); ++i) {
		CpuReduceF32(
			reduced.Data(),
			static_cast<const OaF32*>(InOutBufs[i].MappedPtr),
			count,
			InOp);
	}
	for (OaVkBuffer& buffer : InOutBufs) {
		OaMemcpy(buffer.MappedPtr, reduced.Data(), size);
	}

	return OaStatus::Ok();
}

OaStatus OaCollective::Broadcast(
	OaSpan<OaVkBuffer> InOutBufs,
	OaU32 InSrcIdx)
{
	OA_RETURN_IF_ERROR(ValidateEqualHostBuffers(InOutBufs));
	if (InSrcIdx >= InOutBufs.size())
		return OaStatus::Error(OaStatusCode::InvalidArgument, "InSrcIdx out of range");

	const auto& src = InOutBufs[InSrcIdx];
	if (src.Size == 0 or InOutBufs.size() == 1) return OaStatus::Ok();

	OaVec<OaU8> staged(static_cast<OaVec<OaU8>::size_type>(src.Size));
	OaMemcpy(staged.Data(), src.MappedPtr, src.Size);
	for (OaU32 i = 0; i < InOutBufs.size(); ++i) {
		if (i == InSrcIdx) continue;
		OaMemcpy(InOutBufs[i].MappedPtr, staged.Data(), src.Size);
	}
	return OaStatus::Ok();
}

OaStatus OaCollective::AllGather(
	OaSpan<const OaVkBuffer> InPartials,
	OaSpan<OaVkBuffer> OutFullBufs)
{
	const OaU32 n = static_cast<OaU32>(InPartials.size());
	if (n == 0) {
		return OaStatus::Error(OaStatusCode::InvalidArgument,
			"AllGather: empty partial buffer span");
	}
	if (OutFullBufs.size() != InPartials.size())
		return OaStatus::Error(OaStatusCode::InvalidArgument,
			"AllGather: partials and full buffer counts must match");

	const OaU64 partialSize = InPartials[0].Size;
	if (partialSize > std::numeric_limits<OaU64>::max() / n) {
		return OaStatus::Error(OaStatusCode::InvalidArgument,
			"AllGather: total output size overflows");
	}
	const OaU64 fullSize = partialSize * n;
	if (fullSize > static_cast<OaU64>(
			std::numeric_limits<OaVec<OaU8>::size_type>::max())) {
		return OaStatus::Error(OaStatusCode::InvalidArgument,
			"AllGather: output size exceeds the host staging limit");
	}
	for (OaU32 i = 0; i < n; ++i) {
		if (InPartials[i].Size != partialSize) {
			return OaStatus::Error(OaStatusCode::InvalidArgument,
				"AllGather: partial buffer sizes must match");
		}
		if (OutFullBufs[i].Size < fullSize) {
			return OaStatus::Error(OaStatusCode::InvalidArgument,
				"AllGather: output buffer is too small");
		}
		if (partialSize != 0 and InPartials[i].MappedPtr == nullptr) {
			return OaStatus::Error(OaStatusCode::InvalidArgument,
				"AllGather: partial buffer is not host-visible");
		}
		if (fullSize != 0 and OutFullBufs[i].MappedPtr == nullptr) {
			return OaStatus::Error(OaStatusCode::InvalidArgument,
				"AllGather: output buffer is not host-visible");
		}
	}
	if (fullSize == 0) return OaStatus::Ok();

	OaVec<OaU8> gathered(static_cast<OaVec<OaU8>::size_type>(fullSize));
	for (OaU32 src = 0; src < n; ++src) {
		const OaU64 offset = src * partialSize;
		OaMemcpy(gathered.Data() + offset, InPartials[src].MappedPtr, partialSize);
	}
	for (OaVkBuffer& output : OutFullBufs) {
		OaMemcpy(output.MappedPtr, gathered.Data(), fullSize);
	}
	return OaStatus::Ok();
}

OaStatus OaCollective::Scatter(
	const OaVkBuffer& InFull,
	OaSpan<OaVkBuffer> OutPartials
) {
	const OaU32 n = static_cast<OaU32>(OutPartials.size());
	if (n == 0) {
		return OaStatus::Error(OaStatusCode::InvalidArgument,
			"Scatter: empty output buffer span");
	}
	if (InFull.Size % n != 0) {
		return OaStatus::Error(OaStatusCode::InvalidArgument,
			"Scatter: source size must be divisible by the output count");
	}
	if (InFull.Size != 0 and InFull.MappedPtr == nullptr)
		return OaStatus::Error(OaStatusCode::InvalidArgument, "source buffer not host-visible");

	const OaU64 chunkSize = InFull.Size / n;
	for (const OaVkBuffer& partial : OutPartials) {
		if (partial.Size < chunkSize) {
			return OaStatus::Error(OaStatusCode::InvalidArgument,
				"Scatter: output buffer is too small");
		}
		if (chunkSize != 0 and partial.MappedPtr == nullptr) {
			return OaStatus::Error(OaStatusCode::InvalidArgument,
				"Scatter: output buffer is not host-visible");
		}
	}
	if (chunkSize == 0) return OaStatus::Ok();

	OaVec<OaU8> staged(static_cast<OaVec<OaU8>::size_type>(InFull.Size));
	OaMemcpy(staged.Data(), InFull.MappedPtr, InFull.Size);
	for (OaU32 i = 0; i < n; ++i) {
		OaMemcpy(
			OutPartials[i].MappedPtr,
			staged.Data() + i * chunkSize,
			chunkSize);
	}
	return OaStatus::Ok();
}

OaStatus OaCollective::ReduceScatter(
	OaSpan<OaVkBuffer> InOutBufs,
	OaReduceOp InOp)
{
	OA_RETURN_IF_ERROR(ValidateReduceBuffers(InOutBufs, InOp));
	const OaU32 n = static_cast<OaU32>(InOutBufs.size());
	const OaU64 size = InOutBufs[0].Size;
	if (size == 0 or n == 1) return OaStatus::Ok();

	if (size % n != 0 or (size / n) % sizeof(OaF32) != 0) {
		return OaStatus::Error(OaStatusCode::InvalidArgument,
			"ReduceScatter: buffer size must divide into whole f32 chunks");
	}
	const OaU64 chunkSize = size / n;
	const OaI64 count = static_cast<OaI64>(size / sizeof(OaF32));
	OaVec<OaF32> reduced(static_cast<OaVec<OaF32>::size_type>(count));
	OaMemcpy(reduced.Data(), InOutBufs[0].MappedPtr, size);
	for (OaU32 i = 1; i < n; ++i) {
		CpuReduceF32(
			reduced.Data(),
			static_cast<const OaF32*>(InOutBufs[i].MappedPtr),
			count,
			InOp);
	}
	for (OaU32 i = 0; i < n; ++i) {
		OaMemcpy(
			InOutBufs[i].MappedPtr,
			reinterpret_cast<const OaU8*>(reduced.Data()) + i * chunkSize,
			chunkSize);
	}

	return OaStatus::Ok();
}
