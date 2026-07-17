// OaFnMatrix — OaMatrix view, reshape, and accessor methods.
//
// View, Reshape, Flatten, Unsqueeze, Squeeze, Permute, Transpose,
// Contiguous, Clone, CopyFrom, Item, At, Set, Zero.

#include <Oa/Core/Matrix.h>
#include <Oa/Core/FnMatrix.h>
#include <Oa/Core/Memory.h>
#include <Oa/Core/Status.h>
#include <Oa/Core/Types.h>
#include <Oa/Core/BufferAccess.h>
#include <Oa/Runtime/Context.h>
#include <Oa/Runtime/Dispatch.h>
#include <Oa/Runtime/Engine.h>
#include <Oa/Core/Validation.h>

#include <cassert>
#include <cstring>
static OaU32 DivCeil(OaU32 InA, OaU32 InB) { return (InA + InB - 1) / InB; }
static OaI64 OaMatrixFlatToElementOffset(
	const OaMatrixShape& InShape, const OaStride& InStride, OaI64 InFlatIdx
) {
	OaI64 remainder = InFlatIdx;
	OaI64 elemOff = 0;
	for (OaI32 dimIdx = 0; dimIdx < InShape.Rank; ++dimIdx) {
		OaI64 later = 1;
		for (OaI32 kk = dimIdx + 1; kk < InShape.Rank; ++kk) {
			later *= InShape.Dims[kk];
		}
		const OaI64 coord = later > 0 ? remainder / later : 0;
		if (later > 0) {
			remainder %= later;
		}
		elemOff += coord * InStride.StepElements(dimIdx);
	}
	return elemOff;
}

// OaMatrix: views, reshape, clone, accessors
OaMatrix OaMatrix::View(OaMatrixShape InNewShape) const {
	OaMatrix t = *this;
	OaMatrixShape origShape = t.Shape_;
	t.Shape_ = InNewShape;
	t.Stride_ = OaStride::RowMajor(InNewShape);

	t.SyncMatrixDescriptor();
	return t;
}

OaMatrix OaMatrix::Reshape(OaMatrixShape InNewShape) const { return View(InNewShape); }

OaMatrix OaMatrix::Flatten() const {
	return View(OaMatrixShape{NumElements()});
}

OaMatrix OaMatrix::Unsqueeze(OaI32 InDim) const {
	OaMatrixShape s;
	s.Rank = Shape_.Rank + 1;
	OaI32 j = 0;
	for (OaI32 i = 0; i < s.Rank; ++i) {
		s.Dims[i] = (i == InDim) ? 1 : Shape_.Dims[j++];
	}
	return View(s);
}

OaMatrix OaMatrix::Squeeze(OaI32 InDim) const {
	if (InDim < 0 or InDim >= Shape_.Rank or Shape_[InDim] != 1) return *this;
	OaMatrixShape outShape;
	OaStride outStride;
	outShape.Rank = Shape_.Rank - 1;
	outStride.Rank_ = outShape.Rank;
	OaI32 outIdx = 0;
	for (OaI32 i = 0; i < Shape_.Rank; ++i) {
		if (i == InDim) continue;
		outShape.Dims[outIdx] = Shape_.Dims[i];
		outStride.StepsElems_[outIdx] = Stride_.StepElements(i);
		++outIdx;
	}
	OaMatrix t = *this;
	t.Shape_ = outShape;
	t.Stride_ = outStride;
	t.SyncMatrixDescriptor();
	return t;
}

OaMatrix OaMatrix::Permute(OaSpan<const OaI32> InDims) const {
	if (static_cast<OaI32>(InDims.Size()) != Shape_.Rank) return *this;
	OaMatrixShape outShape;
	OaStride outStride;
	outShape.Rank = Shape_.Rank;
	outStride.Rank_ = Shape_.Rank;
	for (OaI32 i = 0; i < Shape_.Rank; ++i) {
		OaI32 srcDim = InDims[i];
		if (srcDim < 0 or srcDim >= Shape_.Rank) return *this;
		outShape.Dims[i] = Shape_.Dims[srcDim];
		outStride.StepsElems_[i] = Stride_.StepElements(srcDim);
	}
	OaMatrix t = *this;
	t.Shape_ = outShape;
	t.Stride_ = outStride;
	t.SyncMatrixDescriptor();
	return t;
}

OaMatrix OaMatrix::Transpose(OaI32 InDim0, OaI32 InDim1) const {
	// Rank-3 swap of the last two axes ([B,R,C] → [B,C,R]) — the channels-first case
	// conv / sequence layouts need. GPU-native materialize (the strided Permute view is
	// not stride-readable by the elementwise/reduction kernels). Other rank-3 axis pairs
	// and rank>3 are not yet needed; fall through to the no-op below.
	if (Shape_.Rank == 3 && ((InDim0 == 1 && InDim1 == 2) || (InDim0 == 2 && InDim1 == 1))) {
		auto& ctx = OaContext::GetDefault();
		OaI64 batch = Shape_[0], rows = Shape_[1], cols = Shape_[2];
		OaMatrix out = OaFnMatrix::Empty(OaMatrixShape{batch, cols, rows}, Dtype_);

		struct { OaU32 Batch; OaU32 Rows; OaU32 Cols; } push{
			static_cast<OaU32>(batch), static_cast<OaU32>(rows), static_cast<OaU32>(cols)
		};
		OaBufferAccess access[] = {OaBufferAccess::Read, OaBufferAccess::Write};
		// Tiled: one workgroup per (col-tile, row-tile, batch slice); numthreads(32,32,1).
		constexpr OaU32 kTile = 32;
		ctx.Add("TransposeBatched", {this, &out}, access, &push, sizeof(push),
			DivCeil(static_cast<OaU32>(cols), kTile),
			DivCeil(static_cast<OaU32>(rows), kTile),
			static_cast<OaU32>(batch));
		return out;
	}

	if (Shape_.Rank != 2) return *this;

	auto& ctx = OaContext::GetDefault();
	OaI64 rows = Shape_[0], cols = Shape_[1];
	OaMatrix out = OaFnMatrix::Empty(OaMatrixShape{cols, rows}, Dtype_);

	// TransposeTiled: SMEM-staged, coalesced reads AND writes, +1 bank-conflict padding.
	// numthreads(32,32,1); grid = (ceil(cols/TILE), ceil(rows/TILE)). Same kernel the GEMM
	// stack uses; strictly better than the naive scatter-write path it replaced.
	constexpr OaU32 kTile = 32;
	struct { OaU32 Rows; OaU32 Cols; } push{
		static_cast<OaU32>(rows), static_cast<OaU32>(cols)
	};
	OaBufferAccess access[] = {OaBufferAccess::Read, OaBufferAccess::Write};
	ctx.Add("TransposeTiled", {this, &out}, access, &push, sizeof(push),
		DivCeil(static_cast<OaU32>(cols), kTile), DivCeil(static_cast<OaU32>(rows), kTile));

	return out;
}

OaMatrix OaMatrix::Contiguous() const {
	if (Stride_.MatchesRowMajor(Shape_)) return Clone();

	auto out = OaFnMatrix::Empty(Shape_, Dtype_);
	if (not HasStorage() or not out.HasStorage()) return out;

	const OaUsize elSize = OaScalarSize(Dtype_);
	const auto* srcBytes = static_cast<const OaU8*>(Data());
	auto* dstBytes = static_cast<OaU8*>(out.Data());
	const OaI64 numEl = Shape_.NumElements();
	const OaStride dstStride = OaStride::RowMajor(Shape_);

	for (OaI64 flatIdx = 0; flatIdx < numEl; ++flatIdx) {
		OaI64 srcOffset = 0;
		OaI64 remaining = flatIdx;
		for (OaI32 dimIdx = 0; dimIdx < Shape_.Rank; ++dimIdx) {
			OaI64 dstStep = dstStride.StepElements(dimIdx);
			OaI64 coord = remaining / dstStep;
			remaining -= coord * dstStep;
			srcOffset += coord * Stride_.StepElements(dimIdx);
		}
		OaMemcpy(dstBytes + flatIdx * elSize, srcBytes + srcOffset * elSize, elSize);
	}
	return out;
}

// Note: OaMatrix::Clone() is implemented in DeviceMatrixFn.cpp

void OaMatrix::CopyFrom(const OaMatrix& InOther) {
	if (not (HasStorage() and InOther.HasStorage())) { return; }
	if (Dtype_ == InOther.Dtype_) {
		OaU32 count = static_cast<OaU32>(NumElements());
		struct { OaU32 Count; } push{count};
		OaBufferAccess access[] = {OaBufferAccess::Read, OaBufferAccess::Write};
		OaContext::GetDefault().Add(
			"Copy", {&InOther, this}, access, &push, sizeof(push), (count + 255U) / 256U);
	} else {
		OaFnMatrix::CastInto(InOther, *this);
	}
}

OaF32 OaMatrix::Item() const {
	assert(NumElements() == 1 && "Item() requires scalar tensor");
	return At(0);
}

OaF32 OaMatrix::At(OaI64 InIdx) const {
	assert(InIdx >= 0 and InIdx < NumElements());
	const OaI64 elemOff = OaMatrixFlatToElementOffset(Shape_, Stride_, InIdx);
	const OaI64 elemSz = static_cast<OaI64>(OaScalarSize(Dtype_));
	OaU8 cellBytes[sizeof(OaF32)]{};
	const auto* cell = static_cast<const char*>(Data());
	if (cell) {
		cell += elemOff * elemSz;
	} else if (auto* runtime = OaContext::GetDefault().GetEngine()) {
		auto executeStatus = OaContext::GetDefault().Execute();
		assert(executeStatus.IsOk());
		const auto status = runtime->ReadbackBuffer(
			GetVkBuffer(), ByteOffset_ + static_cast<OaU64>(elemOff * elemSz),
			cellBytes, static_cast<OaU64>(elemSz));
		assert(status.IsOk());
		cell = reinterpret_cast<const char*>(cellBytes);
	}
	assert(cell != nullptr);
	if (Dtype_ == OaScalarType::BFloat16) {
		return OaBf16ToF32(*reinterpret_cast<const OaU16*>(cell));
	}
	return *reinterpret_cast<const OaF32*>(cell);
}

void OaMatrix::Set(OaI64 InIdx, OaF32 InValue) {
	assert(InIdx >= 0 and InIdx < NumElements());
	const OaI64 elemOff = OaMatrixFlatToElementOffset(Shape_, Stride_, InIdx);
	const OaI64 elemSz = static_cast<OaI64>(OaScalarSize(Dtype_));
	auto* base = static_cast<char*>(Data());
	auto* cell = base ? base + elemOff * elemSz : nullptr;
	OaU8 encoded[sizeof(OaF32)]{};
	if (Dtype_ == OaScalarType::BFloat16) {
		*reinterpret_cast<OaU16*>(cell ? static_cast<void*>(cell) : static_cast<void*>(encoded)) =
			OaF32ToBf16(InValue);
	} else {
		*reinterpret_cast<OaF32*>(cell ? static_cast<void*>(cell) : static_cast<void*>(encoded)) = InValue;
	}
	if (cell) {
		if (auto* runtime = OaContext::GetDefault().GetEngine()) {
			(void)runtime->Allocator.FlushHostBuffer(
				GetVkBuffer(), ByteOffset_ + static_cast<OaU64>(elemOff * elemSz),
				static_cast<OaU64>(elemSz));
		}
		return;
	}
	if (auto* runtime = OaContext::GetDefault().GetEngine()) {
		auto executeStatus = OaContext::GetDefault().Execute();
		assert(executeStatus.IsOk());
		const auto status = runtime->UploadBuffer(
			GetVkBuffer(), ByteOffset_ + static_cast<OaU64>(elemOff * elemSz),
			encoded, static_cast<OaU64>(elemSz));
		assert(status.IsOk());
	}
}

void OaMatrix::Zero() {
	// GPU-resident tensors MUST be zeroed GPU-side: a host OaMemzero on the mapped pointer
	// is a silent no-op in the deferred model (the GPU buffer is the source of truth), which
	// was the long-standing optimizer-ZeroGrad footgun (OaSGD/OaAdam accumulated grads and
	// diverged). Record a Fill kernel instead so Zero() is correct for every caller.
	if (IsOnDevice() and VkBuf_ != nullptr) {
		OaFnMatrix::Fill(*this, 0.0F);
		return;
	}
	if (HasStorage()) {
		OaMemzero(Data(), static_cast<OaUsize>(ByteSize()));
	}
}
