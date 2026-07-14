#include <Oa/Core/Matrix.h>
#include <Oa/Core/FnMatrix.h>
#include <Oa/Core/Memory.h>
#include <Oa/Core/Status.h>
#include <Oa/Core/Types.h>
#include <Oa/Runtime/Bindless.h>
#include <Oa/Runtime/Context.h>
#include <Oa/Runtime/Dispatch.h>
#include <Oa/Runtime/Engine.h>
#include <Oa/Runtime/ComputeGraph.h>
#include <Oa/Runtime/Stream.h>
#include <Oa/Runtime/Topology.h>
#include <Oa/Core/Log.h>

#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <random>

// ============================================================================
// OaMatrixShape broadcast helpers
// ============================================================================

OaResult<OaMatrixShape> OaMatrixShape::Broadcast(const OaMatrixShape& InOther) const {
	OaMatrixShape out;
	OaI32 maxRank = Rank > InOther.Rank ? Rank : InOther.Rank;
	out.Rank = maxRank;
	for (OaI32 i = 1; i <= maxRank; ++i) {
		OaI32 aIdx = Rank - i;
		OaI32 bIdx = InOther.Rank - i;
		OaI32 outIdx = maxRank - i;
		OaI64 aDim = (aIdx >= 0) ? Dims[aIdx] : 1;
		OaI64 bDim = (bIdx >= 0) ? InOther.Dims[bIdx] : 1;
		if (aDim == 1) {
			out.Dims[outIdx] = bDim;
		} else if (bDim == 1) {
			out.Dims[outIdx] = aDim;
		} else if (aDim == bDim) {
			out.Dims[outIdx] = aDim;
		} else {
			return OaResult<OaMatrixShape>(OaStatus::InvalidArgument("Shapes not broadcastable"));
		}
	}
	return OaResult<OaMatrixShape>(out);
}

OaStdArray<OaI64, OA_MAX_TENSOR_DIMS> OaMatrixShape::BroadcastStrides(const OaMatrixShape& InOut) const {
	OaStdArray<OaI64, OA_MAX_TENSOR_DIMS> strides;
	OaI32 pad = InOut.Rank - Rank;
	for (OaI32 d = 0; d < InOut.Rank; ++d) {
		if (d < pad) {
			strides[static_cast<std::size_t>(d)] = 0;
		} else {
			OaI32 selfIdx = d - pad;
			if (Dims[selfIdx] == 1) {
				strides[static_cast<std::size_t>(d)] = 0;
			} else {
				strides[static_cast<std::size_t>(d)] = Stride(selfIdx);
			}
		}
	}
	return strides;
}

// ============================================================================
// OaStride utility functions
// ============================================================================

OaStride OaStride::RowMajor(const OaMatrixShape& InShape) {
	OaStride strideOut;
	strideOut.Rank_ = InShape.Rank;
	for (OaI32 dimIdx = 0; dimIdx < InShape.Rank; ++dimIdx) {
		OaI64 step = 1;
		for (OaI32 later = dimIdx + 1; later < InShape.Rank; ++later) {
			step *= InShape.Dims[later];
		}
		strideOut.StepsElems_[dimIdx] = step;
	}
	return strideOut;
}

OaI64 OaStride::StepElements(OaI32 InDim) const {
	if (InDim < 0 || InDim >= Rank_) return 0;
	return StepsElems_[InDim];
}

bool OaStride::MatchesRowMajor(const OaMatrixShape& InShape) const {
	if (Rank_ != InShape.Rank) return false;
	OaStride expect = RowMajor(InShape);
	for (OaI32 dimIdx = 0; dimIdx < Rank_; ++dimIdx) {
		if (StepsElems_[dimIdx] != expect.StepsElems_[dimIdx]) return false;
	}
	return true;
}

// ============================================================================
// OaMatrix member functions
// ============================================================================

OaVkBuffer OaMatrix::GetVkBuffer() const {
	if (!VkBuf_) return {};
	return *VkBuf_;
}

const void* OaMatrix::Data() const {
	if (VkBuf_ and VkBuf_->MappedPtr) {
		const auto* base = static_cast<const char*>(VkBuf_->MappedPtr);
		return base + static_cast<ptrdiff_t>(ByteOffset_);
	}
	if (Data_.get()) {
		const auto* base = static_cast<const char*>(Data_.get());
		return base + static_cast<ptrdiff_t>(ByteOffset_);
	}
	return nullptr;
}

void* OaMatrix::Data() {
	return const_cast<void*>(static_cast<const OaMatrix*>(this)->Data());
}

bool OaMatrix::HasStorage() const {
	if (Shape_.Rank <= 0) return false;
	if (NumElements() <= 0) return false;
	if (VkBuf_) return VkBuf_->MappedPtr != nullptr;
	return Data_.get() != nullptr;
}

void OaMatrix::SyncMatrixDescriptor() const noexcept {
	auto& view = const_cast<OaMatrix&>(static_cast<const OaMatrix&>(*this));
	if (VkBuf_ and VkBuf_->BindlessIndex != OA_BINDLESS_INVALID) {
		view.HeapSlot_ = static_cast<OaI32>(VkBuf_->BindlessIndex);
	} else {
		view.HeapSlot_ = -1;
	}
	if (HasStorage()) {
		view.HostBlock_.Ptr = const_cast<void*>(Data());
		view.HostBlock_.SizeBytes = static_cast<OaU64>(ByteSize());
	} else {
		view.HostBlock_.Ptr = nullptr;
		view.HostBlock_.SizeBytes = 0;
	}
}

// ============================================================================
// OaMatrix shape+fill constructor — delegates to OaFnMatrix::Full so the
// braced-init form `OaMatrix m = {OaMatrixShape{3, 3}, 0.0F};` works.
// ============================================================================

OaMatrix::OaMatrix(OaMatrixShape InShape, OaF32 InFillValue, OaScalarType InDtype) {
	*this = OaFnMatrix::Full(InShape, static_cast<OaF64>(InFillValue), InDtype);
}

// ============================================================================
// OaMatrix operator overloads
// ============================================================================

// Arithmetic operators (element-wise)
OaMatrix OaMatrix::operator+(const OaMatrix& InOther) const {
	return OaFnMatrix::Add(*this, InOther);
}

OaMatrix OaMatrix::operator-(const OaMatrix& InOther) const {
	return OaFnMatrix::Sub(*this, InOther);
}

OaMatrix OaMatrix::operator*(const OaMatrix& InOther) const {
	return OaFnMatrix::Mul(*this, InOther);
}

OaMatrix OaMatrix::operator/(const OaMatrix& InOther) const {
	return OaFnMatrix::Div(*this, InOther);
}

// Scalar operators. Dispatch dedicated scalar kernels — zero temp allocation.
OaMatrix OaMatrix::operator+(OaF32 InScalar) const {
	return OaFnMatrix::AddScalar(*this, InScalar);
}

OaMatrix OaMatrix::operator-(OaF32 InScalar) const {
	return OaFnMatrix::SubScalar(*this, InScalar);
}

OaMatrix OaMatrix::operator*(OaF32 InScalar) const {
	return OaFnMatrix::Scale(*this, InScalar);
}

OaMatrix OaMatrix::operator/(OaF32 InScalar) const {
	return OaFnMatrix::DivScalar(*this, InScalar);
}

// Unary operators
OaMatrix OaMatrix::operator-() const {
	return OaFnMatrix::Neg(*this);
}

// Compound assignment operators
OaMatrix& OaMatrix::operator+=(const OaMatrix& InOther) {
	OaFnMatrix::AddInPlace(*this, InOther);
	return *this;
}

OaMatrix& OaMatrix::operator-=(const OaMatrix& InOther) {
	OaMatrix temp = OaFnMatrix::Sub(*this, InOther);
	*this = std::move(temp);
	return *this;
}

OaMatrix& OaMatrix::operator*=(const OaMatrix& InOther) {
	OaMatrix temp = OaFnMatrix::Mul(*this, InOther);
	*this = std::move(temp);
	return *this;
}

OaMatrix& OaMatrix::operator/=(const OaMatrix& InOther) {
	OaMatrix temp = OaFnMatrix::Div(*this, InOther);
	*this = std::move(temp);
	return *this;
}

OaMatrix& OaMatrix::operator+=(OaF32 InScalar) {
	OaFnMatrix::AddScalarInPlace(*this, InScalar);
	return *this;
}

OaMatrix& OaMatrix::operator-=(OaF32 InScalar) {
	OaFnMatrix::SubScalarInPlace(*this, InScalar);
	return *this;
}

OaMatrix& OaMatrix::operator*=(OaF32 InScalar) {
	OaFnMatrix::ScaleInPlace(*this, InScalar);
	return *this;
}

OaMatrix& OaMatrix::operator/=(OaF32 InScalar) {
	OaFnMatrix::DivScalarInPlace(*this, InScalar);
	return *this;
}


