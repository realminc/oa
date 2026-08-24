#pragma once

#include <oa/core/device.h>
#include <oa/core/matrixShape.h>
#include <oa/core/types.h>

namespace oavk { class Buffer; }

namespace oa {

class Engine;
class Matrix;
class MatrixAccess;
class GradNode;


// AutogradMeta — per-tensor autograd state (PyTorch-style nullptr-optimized).
// Inference tensors pay zero bytes: autograd_ stays nullptr.
//
// Invariants:
//   - leaf ⇔ !gradFn
//   - leaf with requiresGrad owns a persistent grad matrix (Tier 1)
//   - non-leaf has gradFn; grad is not populated unless retainGrad (v1: unsupported)
//
class AutogradMeta {
public:
	bool                    requiresGrad_ = false;  // leaf opt-in
	oa::SharedPtr<GradNode>   gradFn;                // non-leaf: producer node; nullptr ⇒ leaf
	oa::UniquePtr<Matrix>     grad;                  // Tier-1 persistent grad accumulator
	                                               // (heap-stored to break the Matrix↔meta type-completeness cycle).
	oa::U32                   outputNr     = 0;      // which output of gradFn I am

	// Constructors.
	AutogradMeta();
	AutogradMeta(const AutogradMeta&) = delete;

	// Destructors.
	~AutogradMeta();                               // out-of-line: Matrix complete in autograd.cpp

	// Operators.
	AutogradMeta& operator=(const AutogradMeta&) = delete;
};

class MemoryBlock {
public:
	// Host-visible span for init/readback/debug (not a second compute engine).

	// Data, class members.
	void* ptr = nullptr;
	oa::U64 sizeBytes = 0;
};


class Stride {
public:
	// Stride: step size in elements for each dimension (same layout as shape).

	// Data, class members.
	oa::Array<oa::I64, OA_MAX_TENSOR_DIMS> stepsElems_{};
	oa::I32 rank_ = 0;

	// Constructors.
	Stride() = default;

	// Methods.
	[[nodiscard]] static Stride rowMajor(const MatrixShape& inShape);

	[[nodiscard]] oa::I32 rank() const { return rank_; }
	[[nodiscard]] oa::I64 stepElements(oa::I32 inDim) const;
	[[nodiscard]] bool  matchesRowMajor(const MatrixShape& inShape) const;
};


/// A multidimensional semantic value backed by OA-managed storage.
///
/// A matrix carries shape, element stride, dtype, device, byte-offset, and
/// optional autograd metadata. Copying a matrix shares its storage and autograd
/// state. Metadata-only views also alias storage; use `clone()` when independent
/// storage is required.
///
/// Matrix operations use OA's Vulkan execution path. Direct host observation is
/// an explicit synchronization boundary, not a CPU compute fallback. Stateful
/// device ownership remains with `oa::Engine`; a matrix only retains the storage
/// required for its value and recorded graph lifetime.
class Matrix {
public:
	/// Construct an empty matrix without storage.
	Matrix() = default;

	/// Allocate and fill a matrix through `oa::FnMatrix::full`.
	///
	/// \param[in] inShape Logical dimensions of the matrix.
	/// \param[in] inFillValue Value written to every element.
	/// \param[in] inDtype Scalar representation used by the allocation.
	Matrix(
		MatrixShape inShape,
		oa::F32 inFillValue,
		oa::ScalarType inDtype = oa::ScalarType::Float32
	);

	/// Release this value's references without submitting or waiting for work.
	~Matrix() = default;

	/// Return a writable pointer when this matrix is host-accessible.
	///
	/// This function does not make a device-only allocation host-visible and does
	/// not synchronize recorded GPU work.
	///
	/// \return Pointer to this view's first byte, or `nullptr` when inaccessible.
	void* data();

	/// Return a read-only pointer when this matrix is host-accessible.
	///
	/// This function does not make a device-only allocation host-visible and does
	/// not synchronize recorded GPU work.
	///
	/// \return Pointer to this view's first byte, or `nullptr` when inaccessible.
	const void* data() const;

	/// Return `data()` cast to the requested element type.
	template<typename T> [[nodiscard]] T* dataAs() { return static_cast<T*>(data()); }

	/// Return const `data()` cast to the requested element type.
	template<typename T> [[nodiscard]] const T* dataAs() const {
		return static_cast<const T*>(data());
	}

	/// Return the number of logical dimensions.
	[[nodiscard]] oa::I32 rank() const { return shape_.rank; }

	/// Return the size of one logical dimension.
	///
	/// Negative indices count from the final dimension, so `-1` names the last
	/// dimension. The resolved index must be in range.
	///
	/// \param[in] inDim Dimension index to query.
	/// \return Number of elements along the resolved dimension.
	[[nodiscard]] oa::I64 size(oa::I32 inDim) const {
		if (inDim < 0) {
			inDim += shape_.rank;
		}
		return shape_[inDim];
	}

	/// Return the product of all logical dimensions.
	[[nodiscard]] oa::I64 numElements() const { return shape_.numElements(); }

	/// Return the logical element count multiplied by the scalar size in bytes.
	[[nodiscard]] oa::I64 byteSize() const {
		return numElements() * static_cast<oa::I64>(oa::scalarSize(dtype_));
	}

	/// Return the device identity associated with this matrix.
	[[nodiscard]] oa::Device getDevice() const { return device_; }

	/// Return whether this value has a non-empty shape and live backing storage.
	[[nodiscard]] bool hasStorage() const;

	/// Return whether this value has no usable backing storage.
	[[nodiscard]] bool isEmpty() const { return !hasStorage(); }

	/// Return whether `data()` exposes this matrix directly to the host.
	[[nodiscard]] bool isHostAccessible() const { return data() != nullptr; }

	/// Return the allocation's current memory-placement policy.
	[[nodiscard]] oa::MemoryPlacement getMemoryPlacement() const;

	/// Observe and initialize the shared saved-value mutation version.
	[[nodiscard]] oa::U64 observeStorageMutationVersion() const;

	/// Return the current shared saved-value mutation version.
	[[nodiscard]] oa::U64 currentStorageMutationVersion() const noexcept;

	/// Advance the shared mutation version after an in-place write.
	void markStorageMutation() const noexcept;

	/// Create a metadata-only row-major view with a new shape.
	///
	/// The new shape must contain exactly the same number of elements. The
	/// returned matrix shares storage and byte offset with this matrix.
	///
	/// \param[in] inNewShape Logical shape of the returned view.
	/// \return The storage-sharing view, or an empty matrix on count mismatch.
	[[nodiscard]] Matrix view(MatrixShape inNewShape) const;

	/// Create a storage-sharing view with a new shape.
	///
	/// This method currently has the same row-major element-count contract as
	/// `view()` and does not materialize a copy.
	///
	/// \param[in] inNewShape Logical shape of the returned view.
	/// \return The storage-sharing view, or an empty matrix on count mismatch.
	[[nodiscard]] Matrix reshape(MatrixShape inNewShape) const;

	/// Return a one-dimensional storage-sharing view.
	[[nodiscard]] Matrix flatten() const;

	/// Insert a size-one dimension without copying storage.
	///
	/// \param[in] inDim Position of the inserted dimension.
	/// \return A storage-sharing view with rank increased by one.
	[[nodiscard]] Matrix unsqueeze(oa::I32 inDim) const;

	/// Remove one size-one dimension without copying storage.
	///
	/// \param[in] inDim Dimension to remove.
	/// \return A storage-sharing view, or this matrix unchanged when the
	/// dimension is invalid or is not size one.
	[[nodiscard]] Matrix squeeze(oa::I32 inDim) const;

	/// Reorder dimensions and element strides without copying storage.
	///
	/// Negative dimension indices are accepted. Every source dimension must occur
	/// exactly once.
	///
	/// \param[in] inDims Source dimension for every output dimension.
	/// \return A strided storage-sharing view; an empty matrix for an invalid
	/// permutation, or this matrix unchanged when the rank does not match.
	[[nodiscard]] Matrix permute(oa::Span<const oa::I32> inDims) const;

	/// Exchange two dimensions through the canonical matrix operation path.
	///
	/// \param[in] inDim0 First dimension to exchange.
	/// \param[in] inDim1 Second dimension to exchange.
	/// \return A matrix with the requested dimensions exchanged.
	[[nodiscard]] Matrix transpose(oa::I32 inDim0, oa::I32 inDim1) const;

	/// Return an independently stored row-major matrix with the same values.
	///
	/// The current non-row-major path may cross a host-access boundary while
	/// materializing the result.
	///
	/// \return A contiguous copy, or an empty matrix when allocation fails.
	[[nodiscard]] Matrix contiguous() const;

	/// Copy this matrix into independent storage through `oa::FnMatrix::copy`.
	///
	/// \return The independent copy, or an empty matrix when this matrix is empty.
	[[nodiscard]] Matrix clone() const;

	/// Record a copy or dtype conversion into existing destination storage.
	///
	/// Both matrices must have storage and compatible element counts. Missing
	/// storage leaves this matrix unchanged.
	///
	/// \param[in] inOther Source values to copy.
	void copyFrom(const Matrix& inOther);

	/// Read a single-element matrix as FP32.
	///
	/// This host observation submits and waits for recorded work when required.
	/// The matrix must contain exactly one element.
	///
	/// \return The scalar value converted to FP32.
	[[nodiscard]] oa::F32 item() const;

	/// Read one logical flat element as FP32.
	///
	/// This host observation submits and waits for recorded work when required and
	/// performs device readback when storage is not host-accessible.
	///
	/// \param[in] inIdx Row-major logical flat index. It must be in range.
	/// \return The selected value converted to FP32.
	[[nodiscard]] oa::F32 at(oa::I64 inIdx) const;

	/// Write one logical flat element from FP32.
	///
	/// The index must be in range. Device-only storage uses an explicit upload
	/// after completing prior recorded work.
	///
	/// \param[in] inIdx Row-major logical flat index to update.
	/// \param[in] inValue Value converted to the matrix dtype and written.
	void set(oa::I64 inIdx, oa::F32 inValue);

	/// Set every element to zero through the placement-appropriate path.
	void zero();

	/// Copy a matrix descriptor while sharing its storage and autograd state.
	Matrix(const Matrix&) = default;

	/// Replace this descriptor with a storage-sharing copy of another matrix.
	Matrix& operator=(const Matrix&) = default;

	/// Move a matrix descriptor without submitting or waiting for device work.
	Matrix(Matrix&&) noexcept = default;

	/// Move-assign a matrix descriptor without submitting or waiting for work.
	Matrix& operator=(Matrix&&) noexcept = default;

	/// Return the element-wise sum using `oa::FnMatrix` broadcasting rules.
	[[nodiscard]] Matrix operator+(const Matrix& inOther) const;

	/// Return the element-wise difference using `oa::FnMatrix` broadcasting rules.
	[[nodiscard]] Matrix operator-(const Matrix& inOther) const;

	/// Return the element-wise product; this operator is not matrix multiplication.
	[[nodiscard]] Matrix operator*(const Matrix& inOther) const;

	/// Return the element-wise quotient using `oa::FnMatrix` broadcasting rules.
	[[nodiscard]] Matrix operator/(const Matrix& inOther) const;

	/// Return this matrix with a scalar added to every element.
	[[nodiscard]] Matrix operator+(oa::F32 inScalar) const;

	/// Return this matrix with a scalar subtracted from every element.
	[[nodiscard]] Matrix operator-(oa::F32 inScalar) const;

	/// Return this matrix with every element multiplied by a scalar.
	[[nodiscard]] Matrix operator*(oa::F32 inScalar) const;

	/// Return this matrix with every element divided by a scalar.
	[[nodiscard]] Matrix operator/(oa::F32 inScalar) const;

	/// Return the element-wise negation of this matrix.
	[[nodiscard]] Matrix operator-() const;

	/// Add a broadcast-compatible matrix into this storage in place.
	Matrix& operator+=(const Matrix& inOther);

	/// Subtract a broadcast-compatible matrix from this storage in place.
	Matrix& operator-=(const Matrix& inOther);

	/// Multiply this storage by a broadcast-compatible matrix in place.
	Matrix& operator*=(const Matrix& inOther);

	/// Divide this storage by a broadcast-compatible matrix in place.
	Matrix& operator/=(const Matrix& inOther);

	/// Add a scalar to every element of this storage in place.
	Matrix& operator+=(oa::F32 inScalar);

	/// Subtract a scalar from every element of this storage in place.
	Matrix& operator-=(oa::F32 inScalar);

	/// Multiply every element of this storage by a scalar in place.
	Matrix& operator*=(oa::F32 inScalar);

	/// Divide every element of this storage by a scalar in place.
	Matrix& operator/=(oa::F32 inScalar);

	/// Enable or disable leaf-gradient tracking.
	///
	/// Enabling tracking lazily allocates a persistent gradient accumulator.
	/// Disabling it preserves allocated gradient storage for reuse.
	///
	/// \param[in] inValue Whether this leaf should accumulate gradients.
	void setRequiresGrad(bool inValue);

	/// Return whether this matrix participates in autograd.
	[[nodiscard]] bool requiresGrad() const noexcept;

	/// Return whether this matrix has no producing gradient node.
	[[nodiscard]] bool isLeaf() const noexcept;

	/// Return the producing gradient node, or `nullptr` for a leaf.
	[[nodiscard]] oa::SharedPtr<GradNode> getGradFn() const noexcept;

	/// Return mutable autograd metadata, allocating it on first access.
	[[nodiscard]] AutogradMeta& mutAutograd();

	/// Replace shared autograd metadata before attaching a producer node.
	///
	/// \param[in] inRequiresGrad Initial tracking state for the detached metadata.
	void detachForGradAttach(bool inRequiresGrad);

	/// Return the persistent gradient accumulator, or an empty matrix when absent.
	[[nodiscard]] Matrix gradMatrix() const;

	/// Return the persistent gradient accumulator, allocating it when absent.
	[[nodiscard]] Matrix& mutGradMatrix();

	/// Defer addition of one contribution into this leaf's gradient accumulator.
	///
	/// \param[in] inContribution Gradient contribution with a compatible element
	/// count.
	void accumulateGrad(const Matrix& inContribution);

	/// Record a zero fill for the persistent gradient accumulator when present.
	void zeroGrad();

	/// Return the logical shape descriptor.
	[[nodiscard]] MatrixShape getShape() const { return shape_; }

	/// Return the element-stride descriptor.
	[[nodiscard]] const Stride& getStride() const { return stride_; }

	/// Return the scalar dtype stored by this matrix.
	[[nodiscard]] oa::ScalarType getDtype() const { return dtype_; }

	/// Return the current runtime bindless slot, or `-1` when none is bound.
	[[nodiscard]] oa::I32 heapSlot() const {
		if (vkBuf_) {
			syncMatrixDescriptor();
		}
		return heapSlot_;
	}

	/// Return this view's byte offset within shared storage.
	[[nodiscard]] oa::U64 byteOffset() const { return byteOffset_; }

	/// Return the current host-visible span, or an empty span when inaccessible.
	[[nodiscard]] MemoryBlock hostBlock() const {
		if (vkBuf_) {
			syncMatrixDescriptor();
		}
		return hostBlock_;
	}

	/// Return whether this matrix currently has a device bindless slot.
	[[nodiscard]] bool isOnDevice() const { return heapSlot() >= 0; }

	/// Synchronize cached descriptors and return this matrix as a const view.
	[[nodiscard]] const Matrix& asMatrixView() const {
		syncMatrixDescriptor();
		return *this;
	}


private:
	friend class MatrixAccess;
	void syncMatrixDescriptor() const noexcept;

	// GPU-specific members
	oa::SharedPtr<void>       data_   = nullptr;
	oa::Device                device_ {oa::DeviceType::VkDiscrete, 0};
	oa::SharedPtr<oavk::Buffer> vkBuf_  = nullptr;

	// base metadata
	MatrixShape   shape_     {};
	Stride        stride_    {};
	oa::ScalarType  dtype_     = oa::ScalarType::Float32;
	oa::I32         heapSlot_  = -1;
	oa::U64         byteOffset_= 0;
	MemoryBlock   hostBlock_ {};

	// autograd tape state — nullptr unless this tensor is tracked.
	// Owned via shared_ptr so Matrix copies (by value) share the same tape entry,
	// matching PyTorch's TensorImpl-shared AutogradMeta semantics.
	oa::SharedPtr<AutogradMeta> autograd_ = nullptr;
};

} // namespace oa
