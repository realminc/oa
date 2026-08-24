#pragma once

#include <oa/core/matrix.h>
#include <oa/runtime/buffer.h>


namespace oa {

// Private, zero-allocation bridge between the semantic matrix value and
// Runtime lowering. Public code observes shape, dtype, placement, and host
// accessibility through Matrix; only implementation code may manipulate the
// physical storage owner or cached descriptor metadata.
class MatrixAccess {
public:
	[[nodiscard]] static const oa::SharedPtr<oavk::Buffer>& storageOwner(const Matrix& inMatrix) noexcept {
		return inMatrix.vkBuf_;
	}

	[[nodiscard]] static oa::SharedPtr<oavk::Buffer>& storageOwner(Matrix& inMatrix) noexcept {
		return inMatrix.vkBuf_;
	}

	[[nodiscard]] static oavk::Buffer descriptor(const Matrix& inMatrix) {
		return inMatrix.vkBuf_ ? *inMatrix.vkBuf_ : oavk::Buffer{};
	}

	[[nodiscard]] static oa::SharedPtr<void>& hostOwner(Matrix& inMatrix) noexcept {
		return inMatrix.data_;
	}

	[[nodiscard]] static MatrixShape& shape(Matrix& inMatrix) noexcept {
		return inMatrix.shape_;
	}

	[[nodiscard]] static Stride& stride(Matrix& inMatrix) noexcept {
		return inMatrix.stride_;
	}

	[[nodiscard]] static oa::ScalarType& dtype(Matrix& inMatrix) noexcept {
		return inMatrix.dtype_;
	}

	[[nodiscard]] static oa::Device& device(Matrix& inMatrix) noexcept {
		return inMatrix.device_;
	}

	[[nodiscard]] static oa::U64& byteOffset(Matrix& inMatrix) noexcept {
		return inMatrix.byteOffset_;
	}

	[[nodiscard]] static oa::SharedPtr<AutogradMeta>& autograd(Matrix& inMatrix) noexcept {
		return inMatrix.autograd_;
	}

	[[nodiscard]] static const oa::SharedPtr<AutogradMeta>& autograd(const Matrix& inMatrix) noexcept {
		return inMatrix.autograd_;
	}

	static void syncDescriptor(const Matrix& inMatrix) noexcept {
		inMatrix.syncMatrixDescriptor();
	}
};

} // namespace oa
