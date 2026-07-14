#include <Oa/Core/MatrixStorage.h>

#include <Oa/Core/Memory.h>

OaU64 OaMatrixStorage::ByteLength(OaMatrixShape InShape, OaScalarType InDtype) {
	const OaI64 numEl = InShape.NumElements();
	const OaUsize elBytes = OaScalarSize(InDtype);
	return static_cast<OaU64>(numEl) * static_cast<OaU64>(elBytes);
}

OaMatrixStorage OaMatrixStorage::Empty(OaMatrixShape InShape, OaScalarType InDtype) {
	OaMatrixStorage out;
	out.Shape_ = InShape;
	out.Dtype_ = InDtype;
	out.Stride_ = OaStride::RowMajor(InShape);
	const OaU64 bytes = ByteLength(InShape, InDtype);
	out.Heap_.Resize(static_cast<OaUsize>(bytes));
	return out;
}

OaMatrixStorage OaMatrixStorage::Zeros(OaMatrixShape InShape, OaScalarType InDtype) {
	OaMatrixStorage out = Empty(InShape, InDtype);
	if (!out.Heap_.Empty()) {
		OaMemzero(out.Heap_.Data(), out.Heap_.Size());
	}
	return out;
}

static void OaMatrixStorageFillAll(OaMatrixStorage& InOut, OaF64 InFill) {
	const OaI64 numEl = InOut.NumElements();
	if (numEl <= 0) return;
	switch (InOut.GetDtype()) {
		case OaScalarType::Float32: {
			const OaF32 value = static_cast<OaF32>(InFill);
			OaF32* data = reinterpret_cast<OaF32*>(InOut.HeapData());
			for (OaI64 idx = 0; idx < numEl; ++idx) {
				data[idx] = value;
			}
			break;
		}
		case OaScalarType::Float64: {
			OaF64* data = reinterpret_cast<OaF64*>(InOut.HeapData());
			for (OaI64 idx = 0; idx < numEl; ++idx) {
				data[idx] = InFill;
			}
			break;
		}
		case OaScalarType::Int32: {
			const OaI32 value = static_cast<OaI32>(InFill);
			OaI32* data = reinterpret_cast<OaI32*>(InOut.HeapData());
			for (OaI64 idx = 0; idx < numEl; ++idx) {
				data[idx] = value;
			}
			break;
		}
		default:
			if (!InOut.HeapData()) break;
			OaMemzero(InOut.HeapData(), InOut.HeapByteCount());
			break;
	}
}

OaMatrixStorage OaMatrixStorage::Ones(OaMatrixShape InShape, OaScalarType InDtype) {
	OaMatrixStorage out = Empty(InShape, InDtype);
	OaMatrixStorageFillAll(out, 1.0);
	return out;
}

OaMatrixStorage OaMatrixStorage::Full(OaMatrixShape InShape, OaScalarType InDtype, OaF64 InFill) {
	OaMatrixStorage out = Empty(InShape, InDtype);
	OaMatrixStorageFillAll(out, InFill);
	return out;
}

OaResult<OaMatrixStorage> OaMatrixStorage::FromBytes(
	OaMatrixShape InShape, OaScalarType InDtype, OaSpan<const OaU8> InData) {
	const OaU64 need = ByteLength(InShape, InDtype);
	if (static_cast<OaU64>(InData.size()) < need) {
		return OaResult<OaMatrixStorage>(
			OaStatus::InvalidArgument("FromBytes: byte span smaller than tensor"));
	}
	OaMatrixStorage out = Empty(InShape, InDtype);
	if (need > 0) {
		OaMemcpy(out.Heap_.Data(), InData.data(), static_cast<OaUsize>(need));
	}
	return OaResult<OaMatrixStorage>(std::move(out));
}

OaMatrixStorage OaMatrixStorage::Copy() const {
	OaMatrixStorage out;
	out.Shape_ = Shape_;
	out.Stride_ = Stride_;
	out.Dtype_ = Dtype_;
	out.Heap_ = Heap_;
	return out;
}

OaMatrix OaMatrixStorage::View() {
	OaMatrix view;
	view.Shape_ = Shape_;
	view.Stride_ = Stride_;
	view.Dtype_ = Dtype_;
	view.HeapSlot_ = -1;
	view.ByteOffset_ = 0;
	view.HostBlock_.Ptr = Heap_.Empty() ? nullptr : Heap_.Data();
	view.HostBlock_.SizeBytes = static_cast<OaU64>(Heap_.Size());
	return view;
}

OaMatrix OaMatrixStorage::View() const {
	OaMatrix view;
	view.Shape_ = Shape_;
	view.Stride_ = Stride_;
	view.Dtype_ = Dtype_;
	view.HeapSlot_ = -1;
	view.ByteOffset_ = 0;
	view.HostBlock_.Ptr = Heap_.Empty() ? nullptr : const_cast<OaU8*>(Heap_.Data());
	view.HostBlock_.SizeBytes = static_cast<OaU64>(Heap_.Size());
	return view;
}
