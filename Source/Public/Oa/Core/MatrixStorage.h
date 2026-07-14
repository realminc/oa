#pragma once

#include <Oa/Core/Matrix.h>
#include <Oa/Core/Status.h>


class OaMatrixStorage {
public:
	// Host-owned row-major buffer for OaMatrix views (debug, tests, CPU reference).

	// Constructors.
	OaMatrixStorage(OaMatrixStorage&& InOther) noexcept = default;

	// Destructors.
	~OaMatrixStorage() = default;

	// Methods.
	[[nodiscard]] OaMatrixStorage Copy() const;
	[[nodiscard]] OaMatrix View();
	[[nodiscard]] OaMatrix View() const;

	[[nodiscard]] const OaMatrixShape& GetShape() const { return Shape_; }
	[[nodiscard]] OaScalarType GetDtype() const { return Dtype_; }
	[[nodiscard]] OaI64 NumElements() const { return Shape_.NumElements(); }
	[[nodiscard]] OaU8* HeapData() { return Heap_.Empty() ? nullptr : Heap_.Data(); }
	[[nodiscard]] const OaU8* HeapData() const { return Heap_.Empty() ? nullptr : Heap_.Data(); }
	[[nodiscard]] OaUsize HeapByteCount() const noexcept { return Heap_.Size(); }

	[[nodiscard]] static OaMatrixStorage Empty(OaMatrixShape InShape, OaScalarType InDtype);
	[[nodiscard]] static OaMatrixStorage Zeros(OaMatrixShape InShape, OaScalarType InDtype);
	[[nodiscard]] static OaMatrixStorage Ones(OaMatrixShape InShape, OaScalarType InDtype);
	[[nodiscard]] static OaMatrixStorage Full(OaMatrixShape InShape, OaScalarType InDtype, OaF64 InFill);
	[[nodiscard]] static OaResult<OaMatrixStorage> FromBytes(
		OaMatrixShape InShape, OaScalarType InDtype, OaSpan<const OaU8> InData
	);

	// Operators.
	OaMatrixStorage& operator=(OaMatrixStorage&& InOther) noexcept = default;
	OaMatrixStorage(const OaMatrixStorage&) = delete;
	OaMatrixStorage& operator=(const OaMatrixStorage&) = delete;

private:
	// Data, class members.
	OaMatrixShape Shape_{};
	OaStride Stride_{};
	OaScalarType Dtype_ = OaScalarType::Float32;
	OaVec<OaU8> Heap_{};

	// Constructors.
	OaMatrixStorage() = default;

	// Methods.
	[[nodiscard]] static OaU64 ByteLength(OaMatrixShape InShape, OaScalarType InDtype);
	
};
