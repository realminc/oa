#pragma once

#include <Oa/Core/Device.h>
#include <Oa/Core/MatrixShape.h>
#include <Oa/Core/Types.h>
#include <Oa/Runtime/Allocator.h>
#include <Oa/Runtime/MatmulTypes.h>

class OaComputeEngine;
class OaGradNode;   // Ml/Autograd.h — tape node; forward-declared so Core stays Ml-free.
class OaMatrix;

// OaAutogradMeta — per-tensor autograd state (PyTorch-style nullptr-optimized).
// Inference tensors pay zero bytes: OaMatrix::Autograd_ stays nullptr.
//
// Invariants:
//   - leaf ⇔ !GradFn
//   - leaf with RequiresGrad owns a persistent Grad matrix (Tier 1)
//   - non-leaf has GradFn; Grad is not populated unless RetainGrad (v1: unsupported)
//
class OaAutogradMeta {
public:
	bool                       RequiresGrad = false;  // leaf opt-in
	OaSharedPtr<OaGradNode>  GradFn;                // non-leaf: producer node; nullptr ⇒ leaf
	OaUniquePtr<OaMatrix>      Grad;                  // Tier-1 persistent grad accumulator
	                                                  // (heap-stored to break the OaMatrix↔meta type-completeness cycle).
	OaU32                      OutputNr     = 0;      // which output of GradFn I am

	OaAutogradMeta();
	~OaAutogradMeta();                                // out-of-line: OaMatrix complete in Autograd.cpp
	OaAutogradMeta(const OaAutogradMeta&) = delete;
	OaAutogradMeta& operator=(const OaAutogradMeta&) = delete;
};

class OaMemoryBlock {
public:
	// Host-visible span for init/readback/debug (not a second compute engine).

	// Data, class members.
	void* Ptr = nullptr;
	OaU64 SizeBytes = 0;
};


class OaStride {
public:
	// Stride: step size in elements for each dimension (same layout as shape).

	// Data, class members.
	OaArray<OaI64, OA_MAX_TENSOR_DIMS> StepsElems_{};
	OaI32 Rank_ = 0;

	// Constructors.
	OaStride() = default;

	// Methods.
	[[nodiscard]] static OaStride RowMajor(const OaMatrixShape& InShape);

	[[nodiscard]] OaI32 Rank() const { return Rank_; }
	[[nodiscard]] OaI64 StepElements(OaI32 InDim) const;
	[[nodiscard]] bool MatchesRowMajor(const OaMatrixShape& InShape) const;
};


class OaMatrix {
public:
	// GPU Matrix: Vulkan compute (pure GPU, no CPU fallback).

	// Constructors.
	OaMatrix() = default;
	// Construct + allocate + fill in one step (delegates to OaFnMatrix::Full).
	// Lets you write `OaMatrix m = {OaMatrixShape{3, 3}, 0.0F};` instead of
	// `OaMatrix m = OaFnMatrix::Full(OaMatrixShape{3, 3}, 0.0);`.
	OaMatrix(OaMatrixShape InShape, OaF32 InFillValue, OaScalarType InDtype = OaScalarType::Float32);
	// Destructors.
	~OaMatrix() = default;

	// Methods.
	[[nodiscard]] OaMatrix To(OaU32 InNodeIndex) const;

	void* Data();
	const void* Data() const;
	template<typename T> [[nodiscard]] T* DataAs() { return static_cast<T*>(Data()); }
	template<typename T> [[nodiscard]] const T* DataAs() const { return static_cast<const T*>(Data()); }

	[[nodiscard]] OaI32 Rank() const { return Shape_.Rank; }
	[[nodiscard]] OaI64 Size(OaI32 InDim) const {
		// Python-style negative indexing: -1 = last dim, -2 = second-to-last, etc.
		if (InDim < 0) InDim += Shape_.Rank;
		return Shape_[InDim];
	}
	[[nodiscard]] OaI64 NumElements() const { return Shape_.NumElements(); }
	[[nodiscard]] OaI64 ByteSize() const { return NumElements() * static_cast<OaI64>(OaScalarSize(Dtype_)); }
	[[nodiscard]] OaDevice GetDevice() const { return Device_; }
	[[nodiscard]] OaU32 NodeIndex() const { return static_cast<OaU32>(Device_.Index); }
	[[nodiscard]] bool HasStorage() const;
	[[nodiscard]] bool IsEmpty() const { return !HasStorage(); }
	[[nodiscard]] bool IsHostAccessible() const { return Data() != nullptr; }
	[[nodiscard]] OaMemoryPlacement GetMemoryPlacement() const;

	[[nodiscard]] OaVkBuffer GetVkBuffer() const;

	[[nodiscard]] OaMatrix View(OaMatrixShape InNewShape) const;
	[[nodiscard]] OaMatrix Reshape(OaMatrixShape InNewShape) const;
	[[nodiscard]] OaMatrix Flatten() const;
	[[nodiscard]] OaMatrix Unsqueeze(OaI32 InDim) const;
	[[nodiscard]] OaMatrix Squeeze(OaI32 InDim) const;
	[[nodiscard]] OaMatrix Permute(OaSpan<const OaI32> InDims) const;
	[[nodiscard]] OaMatrix Transpose(OaI32 InDim0, OaI32 InDim1) const;
	[[nodiscard]] OaMatrix Contiguous() const;

	[[nodiscard]] OaMatrix Clone() const;
	void CopyFrom(const OaMatrix& InOther);

	[[nodiscard]] OaF32 Item() const;
	[[nodiscard]] OaF32 At(OaI64 InIdx) const;
	void Set(OaI64 InIdx, OaF32 InValue);

	void Zero();

	// Operators.
	OaMatrix(const OaMatrix&) = default;
	OaMatrix& operator=(const OaMatrix&) = default;

	OaMatrix(OaMatrix&&) noexcept = default;
	OaMatrix& operator=(OaMatrix&&) noexcept = default;

	// Arithmetic operators
	[[nodiscard]] OaMatrix operator+(const OaMatrix& InOther) const;
	[[nodiscard]] OaMatrix operator-(const OaMatrix& InOther) const;
	[[nodiscard]] OaMatrix operator*(const OaMatrix& InOther) const;  // Element-wise multiplication
	[[nodiscard]] OaMatrix operator/(const OaMatrix& InOther) const;  // Element-wise division
	
	// Scalar operators
	[[nodiscard]] OaMatrix operator+(OaF32 InScalar) const;
	[[nodiscard]] OaMatrix operator-(OaF32 InScalar) const;
	[[nodiscard]] OaMatrix operator*(OaF32 InScalar) const;
	[[nodiscard]] OaMatrix operator/(OaF32 InScalar) const;
	
	// Unary operators
	[[nodiscard]] OaMatrix operator-() const;  // Negation
	
	// Compound assignment operators
	OaMatrix& operator+=(const OaMatrix& InOther);
	OaMatrix& operator-=(const OaMatrix& InOther);
	OaMatrix& operator*=(const OaMatrix& InOther);
	OaMatrix& operator/=(const OaMatrix& InOther);
	
	OaMatrix& operator+=(OaF32 InScalar);
	OaMatrix& operator-=(OaF32 InScalar);
	OaMatrix& operator*=(OaF32 InScalar);
	OaMatrix& operator/=(OaF32 InScalar);

	// Autograd — nullptr unless this tensor is on the tape (see OaAutograd.md).
	// SetRequiresGrad(true) on a leaf allocates a Tier-1 persistent Grad matrix.
	// Forward ops assign a non-null GradFn on outputs (via MutAutograd()) when
	// OaFnAutograd::IsEnabled() and any input RequiresGrad().
	void                              SetRequiresGrad(bool InValue);
	[[nodiscard]] bool                RequiresGrad() const noexcept;
	[[nodiscard]] bool                IsLeaf() const noexcept;
	[[nodiscard]] OaSharedPtr<OaGradNode> GetGradFn() const noexcept;
	[[nodiscard]] OaAutogradMeta&     MutAutograd();              // lazily allocates Autograd_
	void                              DetachForGradAttach(bool InRequiresGrad);  // replace shared/aliased Autograd_ with an independent meta (carries only the requires-grad flag) so attaching a gradfn to a view does not clobber the source
	[[nodiscard]] OaMatrix            GradMatrix() const;         // returns the persistent grad accumulator (empty if none)
	[[nodiscard]] OaMatrix&           MutGradMatrix();            // mutable lvalue ref to the persistent grad (allocates if absent)
	void                              AccumulateGrad(const OaMatrix& InContribution);  // grad += contribution (records ctx Add)
	void                              ZeroGrad();                                       // grad = 0 (records ctx Fill)

	// Accessors for base metadata (used by Record* functions that take const OaMatrix&)
	[[nodiscard]] OaMatrixShape GetShape() const { return Shape_; }
	[[nodiscard]] const OaStride& GetStride() const { return Stride_; }
	[[nodiscard]] OaScalarType GetDtype() const { return Dtype_; }
	[[nodiscard]] OaI32 HeapSlot() const { return HeapSlot_; }
	[[nodiscard]] OaU64 ByteOffset() const { return ByteOffset_; }
	[[nodiscard]] OaMemoryBlock HostBlock() const { return HostBlock_; }
	[[nodiscard]] bool IsOnDevice() const { return HeapSlot_ >= 0; }
	[[nodiscard]] const OaMatrix& AsMatrixView() const {
		SyncMatrixDescriptor();
		return *this;
	}

// Internal API - used by OaFnMatrix namespace functions. Do not access directly.
public:
	void SyncMatrixDescriptor() const noexcept;
	
	// GPU-specific members
	OaSharedPtr<void> Data_ = nullptr;
	OaDevice Device_{OaDeviceType::VkDiscrete, 0};
	OaSharedPtr<OaVkBuffer> VkBuf_ = nullptr;
	
	// Base metadata (inherited from old OaMatrix base class)
	OaMatrixShape Shape_{};
	OaStride Stride_{};
	OaScalarType Dtype_ = OaScalarType::Float32;
	OaI32 HeapSlot_ = -1;
	OaU64 ByteOffset_ = 0;
	OaMemoryBlock HostBlock_{};

	// Autograd tape state — nullptr unless this tensor is tracked.
	// Owned via shared_ptr so OaMatrix copies (by value) share the same tape entry,
	// matching PyTorch's TensorImpl-shared AutogradMeta semantics.
	OaSharedPtr<OaAutogradMeta>     Autograd_           = nullptr;

private:
};
