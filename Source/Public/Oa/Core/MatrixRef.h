#pragma once

#include <Oa/Core/Matrix.h>
#include <Oa/Core/MatrixStorage.h>
#include <Oa/Core/Status.h>


class OaMatrixRef {
public:
	// View of host `OaMatrix` or device `OaMatrix` (mapped buffer) for read/write access (not thread-safe).

	// Constructors.
	explicit OaMatrixRef(const OaMatrix& InMat);
	explicit OaMatrixRef(OaMatrix& InMat);

	// Methods.
	[[nodiscard]] OaMatrixShape Shape() const { return DevPtr_ ? DevPtr_->GetShape() : Mat_.GetShape(); }
	[[nodiscard]] OaScalarType Dtype() const { return DevPtr_ ? DevPtr_->GetDtype() : Mat_.GetDtype(); }
	[[nodiscard]] OaI64 NumElements() const {
		return DevPtr_ ? DevPtr_->NumElements() : Mat_.GetShape().NumElements();
	}

	[[nodiscard]] const void* DataPtr() const;
	[[nodiscard]] void* DataPtr();

	[[nodiscard]] OaResult<OaMatrixStorage> Eval() const;
	[[nodiscard]] OaResult<OaMatrixStorage> ToHostStorage() const { return Eval(); }

private:
	// Data, class members.
	OaMatrix* DevPtr_{nullptr};
	OaMatrix Mat_{};
	bool Writable_{false};
};

using OaMatrixRef = OaMatrixRef;
