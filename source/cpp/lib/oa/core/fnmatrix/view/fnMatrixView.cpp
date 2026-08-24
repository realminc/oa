// oa::FnMatrix — oa::Matrix view, reshape, and accessor methods.
//
// view, reshape, flatten, unsqueeze, squeeze, permute, transpose,
// contiguous, clone, copyFrom, item, at, set, zero.

#include <oa/core/matrix.h>
#include <oa/core/matrixAccess.h>
#include <oa/core/fnMatrix.h>
#include <oa/core/fnmatrix/fnMatrixInternal.h>
#include <oa/core/log.h>
#include <oa/core/memory.h>
#include <oa/core/status.h>
#include <oa/core/types.h>
#include <oa/core/bufferAccess.h>
#include <oa/runtime/executionSession.h>
#include <oa/runtime/engine/resourceAccess.h>
#include <oa/core/validation.h>

#include <cassert>
#include <cstring>

static void recordMatrixView(
	const oa::Matrix& inSource,
	const oa::Matrix& inView)
{
	auto* context = oa::ExecutionSession::getActivePtr();
	if (not context) return;
	const auto status = context->recordView(inSource, inView);
	if (not status.isOk()) {
		OaLogError(oa::LogComponent::Compute,
			"matrix view semantic recording failed: %s",
			status.getMessage().cStr());
	}
}

static oa::I64 matrixFlatToElementOffset(
	const oa::MatrixShape& inShape, const oa::Stride& inStride, oa::I64 inFlatIdx
) {
	oa::I64 remainder = inFlatIdx;
	oa::I64 elemOff = 0;
	for (oa::I32 dimIdx = 0; dimIdx < inShape.rank; ++dimIdx) {
		oa::I64 later = 1;
		for (oa::I32 kk = dimIdx + 1; kk < inShape.rank; ++kk) {
			later *= inShape.dims[static_cast<oa::Usize>(kk)];
		}
		const oa::I64 coord = later > 0 ? remainder / later : 0;
		if (later > 0) {
			remainder %= later;
		}
		elemOff += coord * inStride.stepElements(dimIdx);
	}
	return elemOff;
}

// oa::Matrix: views, reshape, clone, accessors
oa::Matrix oa::Matrix::view(oa::MatrixShape inNewShape) const {
	if (inNewShape.numElements() != numElements()) {
		OaLogError(oa::LogComponent::Compute,
			"view: element count must remain unchanged");
		return {};
	}
	oa::Matrix t = *this;
	t.shape_ = inNewShape;
	t.stride_ = oa::Stride::rowMajor(inNewShape);

	t.syncMatrixDescriptor();
	recordMatrixView(*this, t);
	return t;
}

oa::Matrix oa::Matrix::reshape(oa::MatrixShape inNewShape) const { return view(inNewShape); }

oa::Matrix oa::Matrix::flatten() const {
	return view(oa::MatrixShape{numElements()});
}

oa::Matrix oa::Matrix::unsqueeze(oa::I32 inDim) const {
	oa::MatrixShape s;
	s.rank = shape_.rank + 1;
	oa::I32 j = 0;
	for (oa::I32 i = 0; i < s.rank; ++i) {
		const auto outIdx = static_cast<oa::Usize>(i);
		s.dims[outIdx] = (i == inDim)
			? 1
			: shape_.dims[static_cast<oa::Usize>(j++)];
	}
	return view(s);
}

oa::Matrix oa::Matrix::squeeze(oa::I32 inDim) const {
	if (inDim < 0 or inDim >= shape_.rank or shape_[inDim] != 1) return *this;
	oa::MatrixShape outShape;
	oa::Stride outStride;
	outShape.rank = shape_.rank - 1;
	outStride.rank_ = outShape.rank;
	oa::I32 outIdx = 0;
	for (oa::I32 i = 0; i < shape_.rank; ++i) {
		if (i == inDim) continue;
		const auto outStorageIdx = static_cast<oa::Usize>(outIdx);
		outShape.dims[outStorageIdx] =
			shape_.dims[static_cast<oa::Usize>(i)];
		outStride.stepsElems_[outStorageIdx] = stride_.stepElements(i);
		++outIdx;
	}
	oa::Matrix t = *this;
	t.shape_ = outShape;
	t.stride_ = outStride;
	t.syncMatrixDescriptor();
	recordMatrixView(*this, t);
	return t;
}

oa::Matrix oa::Matrix::permute(oa::Span<const oa::I32> inDims) const {
	if (static_cast<oa::I32>(inDims.size()) != shape_.rank) return *this;
	oa::MatrixShape outShape;
	oa::Stride outStride;
	outShape.rank = shape_.rank;
	outStride.rank_ = shape_.rank;
	oa::Array<oa::Bool, OA_MAX_TENSOR_DIMS> used{};
	for (oa::I32 i = 0; i < shape_.rank; ++i) {
		const auto outStorageIdx = static_cast<oa::Usize>(i);
		oa::I32 srcDim = inDims[outStorageIdx];
		if (srcDim < 0) srcDim += shape_.rank;
		if (srcDim < 0 or srcDim >= shape_.rank
			or used[static_cast<oa::Usize>(srcDim)])
		{
			return {};
		}
		used[static_cast<oa::Usize>(srcDim)] = true;
		outShape.dims[outStorageIdx] =
			shape_.dims[static_cast<oa::Usize>(srcDim)];
		outStride.stepsElems_[outStorageIdx] =
			stride_.stepElements(srcDim);
	}
	oa::Matrix t = *this;
	t.shape_ = outShape;
	t.stride_ = outStride;
	t.syncMatrixDescriptor();
	recordMatrixView(*this, t);
	return t;
}

oa::Matrix oa::Matrix::transpose(oa::I32 inDim0, oa::I32 inDim1) const {
	return oa::FnMatrix::transpose(*this, inDim0, inDim1);
}

oa::Matrix oa::Matrix::contiguous() const {
	if (stride_.matchesRowMajor(shape_)) return clone();

	auto out = oa::FnMatrix::empty(shape_, dtype_);
	if (not hasStorage() or not out.hasStorage()) return out;

	const oa::Usize elSize = oa::scalarSize(dtype_);
	const auto* srcBytes = static_cast<const oa::U8*>(data());
	auto* dstBytes = static_cast<oa::U8*>(out.data());
	const oa::I64 numEl = shape_.numElements();
	const oa::Stride dstStride = oa::Stride::rowMajor(shape_);

	for (oa::I64 flatIdx = 0; flatIdx < numEl; ++flatIdx) {
		oa::I64 srcOffset = 0;
		oa::I64 remaining = flatIdx;
		for (oa::I32 dimIdx = 0; dimIdx < shape_.rank; ++dimIdx) {
			oa::I64 dstStep = dstStride.stepElements(dimIdx);
			oa::I64 coord = remaining / dstStep;
			remaining -= coord * dstStep;
			srcOffset += coord * stride_.stepElements(dimIdx);
		}
		oa::memcpy(
			dstBytes + static_cast<oa::Usize>(flatIdx) * elSize,
			srcBytes + static_cast<oa::Usize>(srcOffset) * elSize,
			elSize);
	}
	return out;
}

// Note: oa::Matrix::clone() is implemented in DeviceMatrixFn.cpp

void oa::Matrix::copyFrom(const oa::Matrix& inOther) {
	if (not (hasStorage() and inOther.hasStorage())) { return; }
	if (dtype_ == inOther.dtype_) {
		oa::U32 count = static_cast<oa::U32>(numElements());
		struct { oa::U32 Count; } push{count};
		oa::BufferAccess access[] = {oa::BufferAccess::Read, oa::BufferAccess::Write};
		oa::ExecutionSession::getActive().add(
			"Copy", {&inOther, this}, access, &push, sizeof(push), (count + 255U) / 256U);
	} else {
		oa::FnMatrix::castInto(inOther, *this);
	}
}

oa::F32 oa::Matrix::item() const {
	assert(numElements() == 1 && "item() requires scalar tensor");
	return at(0);
}

oa::F32 oa::Matrix::at(oa::I64 inIdx) const {
	assert(inIdx >= 0 and inIdx < numElements());
	// A host read is an observation boundary even when the allocation happens to
	// be persistently mapped. Recorded GPU writes must become visible before the
	// CPU dereferences that mapping; otherwise correctness depends on allocation
	// size (device-local matrices took the readback path below and were flushed,
	// while small host-visible matrices returned stale data).
	if (oa::ExecutionSession::getActivePtr()) {
		auto completionStatus =
			oa::FnMatrix::completeRecordedWork(oa::ExecutionSession::getActive());
		assert(completionStatus.isOk());
	}
	const oa::I64 elemOff = matrixFlatToElementOffset(shape_, stride_, inIdx);
	const oa::I64 elemSz = static_cast<oa::I64>(oa::scalarSize(dtype_));
	oa::U8 cellBytes[sizeof(oa::F32)]{};
	const auto* cell = static_cast<const char*>(data());
	if (cell) {
		cell += elemOff * elemSz;
	} else {
		auto& runtime = oa::ExecutionSession::getActive().engine();
		const auto status = oa::EngineResourceAccess::readbackBuffer(runtime,
			oa::MatrixAccess::descriptor(*this),
			byteOffset_ + static_cast<oa::U64>(elemOff * elemSz),
			cellBytes, static_cast<oa::U64>(elemSz));
		assert(status.isOk());
		cell = reinterpret_cast<const char*>(cellBytes);
	}
	assert(cell != nullptr);
	if (dtype_ == oa::ScalarType::BFloat16) {
		return oa::bf16ToF32(*reinterpret_cast<const oa::U16*>(cell));
	}
	return *reinterpret_cast<const oa::F32*>(cell);
}

void oa::Matrix::set(oa::I64 inIdx, oa::F32 inValue) {
	assert(inIdx >= 0 and inIdx < numElements());
	const oa::I64 elemOff = matrixFlatToElementOffset(shape_, stride_, inIdx);
	const oa::I64 elemSz = static_cast<oa::I64>(oa::scalarSize(dtype_));
	auto* base = static_cast<char*>(data());
	auto* cell = base ? base + elemOff * elemSz : nullptr;
	oa::U8 encoded[sizeof(oa::F32)]{};
	if (dtype_ == oa::ScalarType::BFloat16) {
		*reinterpret_cast<oa::U16*>(encoded) = oa::f32ToBf16(inValue);
	} else {
		*reinterpret_cast<oa::F32*>(encoded) = inValue;
	}
	if (not cell) {
		auto completionStatus =
			oa::FnMatrix::completeRecordedWork(oa::ExecutionSession::getActive());
		assert(completionStatus.isOk());
	}
	auto& runtime = oa::ExecutionSession::getActive().engine();
	const auto status = oa::EngineResourceAccess::uploadBuffer(runtime,
		oa::MatrixAccess::descriptor(*this),
		byteOffset_ + static_cast<oa::U64>(elemOff * elemSz),
		encoded, static_cast<oa::U64>(elemSz));
	assert(status.isOk());
}

void oa::Matrix::zero() {
	// GPU-resident tensors MUST be zeroed GPU-side: a host oa::memzero on the mapped pointer
	// is a silent no-op in the deferred model (the GPU buffer is the source of truth), which
	// was the long-standing optimizer-zeroGrad footgun (oa::Sgd/oa::Adam accumulated grads and
	// diverged). Record a Fill kernel instead so zero() is correct for every caller.
	if (isOnDevice() and vkBuf_ != nullptr) {
		oa::FnMatrix::fillInPlace(*this, 0.0F);
		return;
	}
	if (hasStorage()) {
		oa::memzero(data(), static_cast<oa::Usize>(byteSize()));
		markStorageMutation();
	}
}
