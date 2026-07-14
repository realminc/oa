#include <Oa/Core/ByteLogitsHost.h>

#include <cfloat>

static constexpr OaI32 kOaByteLogitClasses = 256;

OaVec<OaU8> OaByteLogitsArgmaxFromContiguousF32(OaI64 InSeqLen, const OaF32* InRowMajorLogits) {
	OaVec<OaU8> result;
	if (InSeqLen < 1 || !InRowMajorLogits) {
		return result;
	}
	result.resize(static_cast<OaUsize>(InSeqLen));
	for (OaI64 rowIdx = 0; rowIdx < InSeqLen; ++rowIdx) {
		const OaF32* row = InRowMajorLogits + rowIdx * kOaByteLogitClasses;
		OaF32 maxVal = -FLT_MAX;
		OaI32 maxIdx = 0;
		for (OaI32 colIdx = 0; colIdx < kOaByteLogitClasses; ++colIdx) {
			if (row[colIdx] > maxVal) {
				maxVal = row[colIdx];
				maxIdx = colIdx;
			}
		}
		result[static_cast<OaUsize>(rowIdx)] = static_cast<OaU8>(maxIdx);
	}
	return result;
}

OaResult<OaVec<OaU8>> OaByteLogitsArgmaxHost(const OaMatrix& InLogits) {
	if (InLogits.GetDtype() != OaScalarType::Float32) {
		return OaResult<OaVec<OaU8>>(
			OaStatus::InvalidArgument("OaByteLogitsArgmaxHost: logits must be Float32"));
	}
	const OaMemoryBlock host = InLogits.HostBlock();
	if (!host.Ptr) {
		return OaResult<OaVec<OaU8>>(
			OaStatus::InvalidArgument("OaByteLogitsArgmaxHost: missing host storage"));
	}
	const OaMatrixShape shape = InLogits.GetShape();
	if (!InLogits.GetStride().MatchesRowMajor(shape)) {
		return OaResult<OaVec<OaU8>>(
			OaStatus::InvalidArgument("OaByteLogitsArgmaxHost: need row-major stride"));
	}
	OaI64 seqLen = 0;
	if (shape.Rank == 2) {
		if (shape[1] != kOaByteLogitClasses) {
			return OaResult<OaVec<OaU8>>(OaStatus::InvalidArgument(
				"OaByteLogitsArgmaxHost: expected [seq, 256] logits"));
		}
		seqLen = shape[0];
	} else if (shape.Rank == 3) {
		if (shape[0] != 1 || shape[2] != kOaByteLogitClasses) {
			return OaResult<OaVec<OaU8>>(OaStatus::InvalidArgument(
				"OaByteLogitsArgmaxHost: expected [1, seq, 256] logits"));
		}
		seqLen = shape[1];
	} else {
		return OaResult<OaVec<OaU8>>(
			OaStatus::InvalidArgument("OaByteLogitsArgmaxHost: rank must be 2 or 3"));
	}
	if (seqLen < 1) {
		return OaResult<OaVec<OaU8>>(
			OaStatus::InvalidArgument("OaByteLogitsArgmaxHost: empty sequence"));
	}
	const OaU8* base = static_cast<const OaU8*>(host.Ptr) + InLogits.ByteOffset();
	const OaF32* logits = reinterpret_cast<const OaF32*>(base);
	return OaResult<OaVec<OaU8>>(OaByteLogitsArgmaxFromContiguousF32(seqLen, logits));
}
