#include <Oa/Core/MatrixRef.h>

#include <Oa/Core/Memory.h>
#include <Oa/Runtime/Allocator.h>

static OaStatus OaMxRefValidateHostContiguous(const OaMatrix& InM, OaI64& OutNumEl) {
	if (InM.IsOnDevice()) {
		return OaStatus::Error(OaStatusCode::FailedPrecondition, "OaMatrixRef: device matrix");
	}
	const OaMemoryBlock host = InM.HostBlock();
	if (!host.Ptr || host.SizeBytes == 0) {
		return OaStatus::Error(OaStatusCode::FailedPrecondition, "OaMatrixRef: missing host block");
	}
	const OaScalarType dtype = InM.GetDtype();
	const OaMatrixShape shape = InM.GetShape();
	if (!InM.GetStride().MatchesRowMajor(shape)) {
		return OaStatus::Error(OaStatusCode::FailedPrecondition, "OaMatrixRef: need row-major stride");
	}
	OutNumEl = shape.NumElements();
	const OaU64 need =
		InM.ByteOffset() + static_cast<OaU64>(OutNumEl) * static_cast<OaU64>(OaScalarSize(dtype));
	if (host.SizeBytes < need) {
		return OaStatus::Error(OaStatusCode::FailedPrecondition, "OaMatrixRef: host block too small");
	}
	return OaStatus::Ok();
}

static OaStatus OaMxRefValidateDeviceContiguous(const OaMatrix& InD, OaI64& OutNumEl) {
	if (!InD.HasStorage()) {
		return OaStatus::Error(
			OaStatusCode::FailedPrecondition, "OaMatrixRef: device matrix missing storage");
	}
	const OaMatrixShape shape = InD.GetShape();
	if (!InD.GetStride().MatchesRowMajor(shape)) {
		return OaStatus::Error(OaStatusCode::FailedPrecondition, "OaMatrixRef: device need row-major stride");
	}
	if (!InD.Data()) {
		return OaStatus::Error(OaStatusCode::FailedPrecondition, "OaMatrixRef: device null data");
	}
	OutNumEl = shape.NumElements();
	const OaScalarType dtype = InD.GetDtype();
	const OaU64 need =
		InD.ByteOffset() + static_cast<OaU64>(OutNumEl) * static_cast<OaU64>(OaScalarSize(dtype));
	const OaVkBuffer buf = InD.GetVkBuffer();
	if (buf.Size > 0 && need > buf.Size) {
		return OaStatus::Error(OaStatusCode::FailedPrecondition, "OaMatrixRef: device buffer too small");
	}
	return OaStatus::Ok();
}

OaMatrixRef::OaMatrixRef(const OaMatrix& InMat)
: Writable_(false) {
	if (const auto* dev = dynamic_cast<const OaMatrix*>(&InMat)) {
		DevPtr_ = const_cast<OaMatrix*>(dev);
		return;
	}
	Mat_ = InMat;
}

OaMatrixRef::OaMatrixRef(OaMatrix& InMat)
: Writable_(true) {
	if (auto* dev = dynamic_cast<OaMatrix*>(&InMat)) {
		DevPtr_ = dev;
		return;
	}
	Mat_ = InMat;
}

const void* OaMatrixRef::DataPtr() const {
	if (DevPtr_) {
		return DevPtr_->Data();
	}
	const OaU8* base = static_cast<const OaU8*>(Mat_.HostBlock().Ptr);
	if (!base) {
		return nullptr;
	}
	return base + Mat_.ByteOffset();
}

void* OaMatrixRef::DataPtr() {
	if (!Writable_) {
		return nullptr;
	}
	if (DevPtr_) {
		return DevPtr_->Data();
	}
	OaU8* base = static_cast<OaU8*>(Mat_.HostBlock().Ptr);
	if (!base) {
		return nullptr;
	}
	return base + Mat_.ByteOffset();
}

OaResult<OaMatrixStorage> OaMatrixRef::Eval() const {
	if (DevPtr_) {
		OaI64 numEl = 0;
		if (auto status = OaMxRefValidateDeviceContiguous(*DevPtr_, numEl); !status.IsOk()) {
			return OaResult<OaMatrixStorage>(std::move(status));
		}
		const OaScalarType dtype = DevPtr_->GetDtype();
		const OaMatrixShape shape = DevPtr_->GetShape();
		OaMatrixStorage out = OaMatrixStorage::Empty(shape, dtype);
		const OaU8* src = static_cast<const OaU8*>(DevPtr_->Data());
		const OaUsize byteCount = static_cast<OaUsize>(numEl) * static_cast<OaUsize>(OaScalarSize(dtype));
		OaMemcpy(out.HeapData(), src, byteCount);
		return OaResult<OaMatrixStorage>(std::move(out));
	}
	OaI64 numEl = 0;
	if (auto status = OaMxRefValidateHostContiguous(Mat_, numEl); !status.IsOk()) {
		return OaResult<OaMatrixStorage>(std::move(status));
	}
	const OaScalarType dtype = Mat_.GetDtype();
	const OaMatrixShape shape = Mat_.GetShape();
	OaMatrixStorage out = OaMatrixStorage::Empty(shape, dtype);
	const OaU8* src = static_cast<const OaU8*>(Mat_.HostBlock().Ptr) + Mat_.ByteOffset();
	const OaUsize byteCount = static_cast<OaUsize>(numEl) * static_cast<OaUsize>(OaScalarSize(dtype));
	OaMemcpy(out.HeapData(), src, byteCount);
	return OaResult<OaMatrixStorage>(std::move(out));
}
