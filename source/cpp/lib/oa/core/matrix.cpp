#include <oa/core/matrix.h>
#include <oa/core/fnMatrix.h>
#include <oa/runtime/allocator.h>

// ============================================================================
// oa::MatrixShape broadcast helpers
// ============================================================================

oa::Result<oa::MatrixShape> oa::MatrixShape::broadcast(const oa::MatrixShape& inOther) const {
	oa::MatrixShape out;
	oa::I32 maxRank = rank > inOther.rank ? rank : inOther.rank;
	out.rank = maxRank;
	for (oa::I32 i = 1; i <= maxRank; ++i) {
		oa::I32 aIdx   = rank - i;
		oa::I32 bIdx   = inOther.rank - i;
		oa::I32 outIdx = maxRank - i;
		oa::I64 aDim   = (aIdx >= 0) ? dims[static_cast<oa::Usize>(aIdx)] : 1;
		oa::I64 bDim   = (bIdx >= 0) ? inOther.dims[static_cast<oa::Usize>(bIdx)] : 1;
		const auto outStorageIdx = static_cast<oa::Usize>(outIdx);
		if (aDim == 1) {
			out.dims[outStorageIdx] = bDim;
		} else if (bDim == 1) {
			out.dims[outStorageIdx] = aDim;
		} else if (aDim == bDim) {
			out.dims[outStorageIdx] = aDim;
		} else {
			return oa::Result<oa::MatrixShape>(oa::Status::invalidArgument("Shapes not broadcastable"));
		}
	}
	return oa::Result<oa::MatrixShape>(out);
}

oa::Array<oa::I64, OA_MAX_TENSOR_DIMS> oa::MatrixShape::broadcastStrides(const oa::MatrixShape& inOut) const {
	oa::Array<oa::I64, OA_MAX_TENSOR_DIMS> strides;
	oa::I32 pad = inOut.rank - rank;
	for (oa::I32 d = 0; d < inOut.rank; ++d) {
		if (d < pad) {
			strides[static_cast<oa::Usize>(d)] = 0;
		} else {
			oa::I32 selfIdx = d - pad;
			if (dims[static_cast<oa::Usize>(selfIdx)] == 1) {
				strides[static_cast<oa::Usize>(d)] = 0;
			} else {
				strides[static_cast<oa::Usize>(d)] = stride(selfIdx);
			}
		}
	}
	return strides;
}

// ============================================================================
// oa::Stride utility functions
// ============================================================================

oa::Stride oa::Stride::rowMajor(const oa::MatrixShape& inShape) {
	Stride strideOut;
	strideOut.rank_ = inShape.rank;
	for (oa::I32 dimIdx = 0; dimIdx < inShape.rank; ++dimIdx) {
		oa::I64 step = 1;
		for (oa::I32 later = dimIdx + 1; later < inShape.rank; ++later) {
			step *= inShape.dims[static_cast<oa::Usize>(later)];
		}
		strideOut.stepsElems_[static_cast<oa::Usize>(dimIdx)] = step;
	}
	return strideOut;
}

oa::I64 oa::Stride::stepElements(oa::I32 inDim) const {
	if (inDim < 0 || inDim >= rank_) return 0;
	return stepsElems_[static_cast<oa::Usize>(inDim)];
}

bool oa::Stride::matchesRowMajor(const oa::MatrixShape& inShape) const {
	if (rank_ != inShape.rank) {
		return false;
	}
	Stride expect = rowMajor(inShape);
	for (oa::I32 dimIdx = 0; dimIdx < rank_; ++dimIdx) {
		const auto storageIdx = static_cast<oa::Usize>(dimIdx);
		if (stepsElems_[storageIdx] != expect.stepsElems_[storageIdx]) {
			return false;
		}
	}
	return true;
}

// ============================================================================
// oa::Matrix member functions
// ============================================================================

const void* oa::Matrix::data() const {
	if (vkBuf_) {
		if (vkBuf_->mappedPtr) {
			const auto* base = static_cast<const char*>(vkBuf_->mappedPtr);
			return base + static_cast<oa::Isize>(byteOffset_);
		}
		// A retained device matrix is deliberately made inert when its engine
		// closes. Do not fall through to the non-owning cached mapped pointer.
		return nullptr;
	}
	if (data_.get()) {
		const auto* base = static_cast<const char*>(data_.get());
		return base + static_cast<oa::Isize>(byteOffset_);
	}
	return nullptr;
}

void* oa::Matrix::data() {
	return const_cast<void*>(static_cast<const oa::Matrix*>(this)->data());
}

bool oa::Matrix::hasStorage() const {
	if (shape_.rank <= 0) return false;
	if (numElements() <= 0) return false;
	if (vkBuf_) return vkBuf_->buffer != nullptr;
	return data_.get() != nullptr;
}

oa::MemoryPlacement oa::Matrix::getMemoryPlacement() const {
	return vkBuf_ ? vkBuf_->placement : oa::MemoryPlacement::Auto;
}

oa::U64 oa::Matrix::observeStorageMutationVersion() const {
	return vkBuf_ ? vkBuf_->observeMutationVersion() : 0U;
}

oa::U64 oa::Matrix::currentStorageMutationVersion() const noexcept {
	return vkBuf_ ? vkBuf_->currentMutationVersion() : 0U;
}

void oa::Matrix::markStorageMutation() const noexcept {
	if (vkBuf_) vkBuf_->markMutation();
}

void oa::Matrix::syncMatrixDescriptor() const noexcept {
	auto& view = const_cast<oa::Matrix&>(static_cast<const oa::Matrix&>(*this));
	if (vkBuf_ && vkBuf_->bindlessIndex != UINT32_MAX) {
		view.heapSlot_ = static_cast<oa::I32>(vkBuf_->bindlessIndex);
	} else {
		view.heapSlot_ = -1;
	}
	if (data() != nullptr) {
		view.hostBlock_.ptr = const_cast<void*>(data());
		view.hostBlock_.sizeBytes = static_cast<oa::U64>(byteSize());
	} else {
		view.hostBlock_.ptr = nullptr;
		view.hostBlock_.sizeBytes = 0;
	}
}

// ============================================================================
// oa::Matrix shape+fill constructor — delegates to oa::FnMatrix::full so the
// braced-init form `oa::Matrix m = {oa::MatrixShape{3, 3}, 0.0F};` works.
// ============================================================================

oa::Matrix::Matrix(oa::MatrixShape inShape, oa::F32 inFillValue, oa::ScalarType inDtype) {
	*this = oa::FnMatrix::full(inShape, static_cast<oa::F64>(inFillValue), inDtype);
}

// ============================================================================
// oa::Matrix operator overloads
// ============================================================================

oa::Matrix oa::Matrix::operator+(const oa::Matrix& inOther) const {
	return oa::FnMatrix::add(*this, inOther);
}

oa::Matrix oa::Matrix::operator-(const oa::Matrix& inOther) const {
	return oa::FnMatrix::sub(*this, inOther);
}

oa::Matrix oa::Matrix::operator*(const oa::Matrix& inOther) const {
	return oa::FnMatrix::mul(*this, inOther);
}

oa::Matrix oa::Matrix::operator/(const oa::Matrix& inOther) const {
	return oa::FnMatrix::div(*this, inOther);
}

// scalar operators. Dispatch dedicated scalar kernels — zero temp allocation.
oa::Matrix oa::Matrix::operator+(oa::F32 inScalar) const {
	return oa::FnMatrix::addScalar(*this, inScalar);
}

oa::Matrix oa::Matrix::operator-(oa::F32 inScalar) const {
	return oa::FnMatrix::subScalar(*this, inScalar);
}

oa::Matrix oa::Matrix::operator*(oa::F32 inScalar) const {
	return oa::FnMatrix::scale(*this, inScalar);
}

oa::Matrix oa::Matrix::operator/(oa::F32 inScalar) const {
	return oa::FnMatrix::divScalar(*this, inScalar);
}

// Unary operators
oa::Matrix oa::Matrix::operator-() const {
	return oa::FnMatrix::neg(*this);
}

// Compound assignment operators
oa::Matrix& oa::Matrix::operator+=(const oa::Matrix& inOther) {
	oa::FnMatrix::addInPlace(*this, inOther);
	return *this;
}

oa::Matrix& oa::Matrix::operator-=(const oa::Matrix& inOther) {
	oa::FnMatrix::subInPlace(*this, inOther);
	return *this;
}

oa::Matrix& oa::Matrix::operator*=(const oa::Matrix& inOther) {
	oa::FnMatrix::mulInPlace(*this, inOther);
	return *this;
}

oa::Matrix& oa::Matrix::operator/=(const oa::Matrix& inOther) {
	oa::FnMatrix::divInPlace(*this, inOther);
	return *this;
}

oa::Matrix& oa::Matrix::operator+=(oa::F32 inScalar) {
	oa::FnMatrix::addScalarInPlace(*this, inScalar);
	return *this;
}

oa::Matrix& oa::Matrix::operator-=(oa::F32 inScalar) {
	oa::FnMatrix::subScalarInPlace(*this, inScalar);
	return *this;
}

oa::Matrix& oa::Matrix::operator*=(oa::F32 inScalar) {
	oa::FnMatrix::scaleInPlace(*this, inScalar);
	return *this;
}

oa::Matrix& oa::Matrix::operator/=(oa::F32 inScalar) {
	oa::FnMatrix::divScalarInPlace(*this, inScalar);
	return *this;
}
