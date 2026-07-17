// OaFnMatrix — Allocation, initialisation, host transfer, multi-device.
//
// Empty, Zeros, Ones, Full, Rand*, FromBytes, CausalMask, CopyToHost, EmptyOn
//
// RNG functions (Rand, RandN, RandXavier, etc.) use GPU-native Philox PRNG
// via PhiloxUniform/PhiloxNormal (see FnMatrixRng.cpp).
//
// Weight dtype configuration lives in FnMatrix.cpp. Allocation resolves the
// current engine through the selected OaContext.

#include <Oa/Core/Matrix.h>
#include <Oa/Core/FnMatrix.h>
#include <Oa/Core/Memory.h>
#include <Oa/Core/Status.h>
#include <Oa/Core/Types.h>
#include <Oa/Runtime/Bindless.h>
#include <Oa/Runtime/ComputeGraph.h>
#include <Oa/Runtime/Context.h>
#include <Oa/Runtime/Dispatch.h>
#include <Oa/Runtime/Engine.h>
#include <Oa/Runtime/Pool.h>
#include <Oa/Runtime/Stream.h>
#include <Oa/Runtime/Topology.h>

#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <cstring>

namespace {

OaComputeEngine* ActiveRuntime() {
	auto* context = OaContext::GetDefaultPtr();
	return context ? context->GetEngine() : nullptr;
}

} // namespace

// Allocators: Empty, Zeros, Ones, Full, Rand*, FromBytes, CausalMask
OaMatrix OaFnMatrix::Empty(
	OaMatrixShape InShape, OaScalarType InDtype, OaMemoryPlacement InPlacement) {

	OaMatrix t;
	t.Shape_ = InShape;
	t.Stride_ = OaStride::RowMajor(InShape);
	t.ByteOffset_ = 0;
	t.Dtype_ = InDtype;
	t.Device_ = OaDevice{OaDeviceType::VkDiscrete, 0};

	auto* rt = ActiveRuntime();
	if (not rt) {
		t.SyncMatrixDescriptor();
		return t;
	}

	OaI64 bytes = InShape.NumElements() * static_cast<OaI64>(OaScalarSize(InDtype));
	if (bytes <= 0) {
		t.SyncMatrixDescriptor();
		return t;
	}

	// The context owns the allocation policy. Normal calls use the engine cache;
	// repeatable training frames can additionally map allocation ordinals onto
	// stable VkBuffer/bindless slots for exact command-graph replay.
	auto buf = OaContext::GetDefault().AllocateMatrixBuffer(
		static_cast<OaU64>(bytes), InPlacement);
	if (not buf) {
		t.SyncMatrixDescriptor();
		return t;
	}
	buf->NodeIndex = 0;
	t.VkBuf_ = buf;
	t.Data_ = OaSharedPtr<void>(buf->MappedPtr, [](void*) {});
	t.SyncMatrixDescriptor();
	return t;
}

OaMatrix OaFnMatrix::Zeros(OaMatrixShape InShape, OaScalarType InDtype) {
	auto t = OaFnMatrix::Empty(InShape, InDtype);
	if (t.HasStorage()) {
		if (t.Data()) {
			OaMemzero(t.Data(), static_cast<OaUsize>(t.ByteSize()));
			if (auto* rt = ActiveRuntime()) {
				(void)rt->Allocator.FlushHostBuffer(
					t.GetVkBuffer(), 0, static_cast<OaU64>(t.ByteSize()));
			}
		} else {
			OaFnMatrix::Fill(t, 0.0F);
		}
	}
	return t;
}

OaMatrix OaFnMatrix::Ones(OaMatrixShape InShape, OaScalarType InDtype) {
	auto t = OaFnMatrix::Empty(InShape, InDtype);
	if (t.HasStorage()) {
		OaFnMatrix::Fill(t, 1.0f);
	}
	return t;
}

OaMatrix OaFnMatrix::Full(OaMatrixShape InShape, OaF64 InValue, OaScalarType InDtype) {
	auto t = OaFnMatrix::Empty(InShape, InDtype);
	if (t.HasStorage()) {
		OaFnMatrix::Fill(t, static_cast<OaF32>(InValue));
	}
	return t;
}

OaMatrix OaFnMatrix::Rand(OaMatrixShape InShape, OaScalarType InDtype) {
	// GPU-native: use PhiloxUniform for parallel generation
	auto t = OaFnMatrix::Empty(InShape, InDtype);
	return OaFnMatrix::PhiloxUniform(t, 0.0F, 1.0F, 0);
}

OaMatrix OaFnMatrix::RandN(OaMatrixShape InShape, OaScalarType InDtype) {
	// GPU-native: use PhiloxNormal for parallel generation
	// Kaiming/He normal initialization: N(0, √(2/fan_in))
	OaF32 std_dev = 1.0F;
	if (InShape.Rank == 2) {
		OaI64 fan_in = InShape[1];  // embed_dim for embeddings
		std_dev = std::sqrt(2.0F / static_cast<OaF32>(fan_in));
	} else {
		// For other shapes, use small std_dev
		std_dev = 0.02F;
	}
	
	auto t = OaFnMatrix::Empty(InShape, InDtype);
	return OaFnMatrix::PhiloxNormal(t, 0.0F, std_dev, 0);
}

OaMatrix OaFnMatrix::RandXavier(OaMatrixShape InShape, OaScalarType InDtype) {
	// GPU-native: use PhiloxUniform for parallel generation
	// PyTorch nn.Linear default: kaiming_uniform_(a=sqrt(5))
	OaF32 bound = 1.0F;
	if (InShape.Rank == 2) {
		OaI64 fan_in = InShape[1];  // For weight matrix [out_features, in_features]
		bound = std::sqrt(1.0F / static_cast<OaF32>(fan_in));
	} else {
		// For other shapes, use small uniform range
		bound = 0.1F;
	}
	
	auto t = OaFnMatrix::Empty(InShape, InDtype);
	return OaFnMatrix::PhiloxUniform(t, -bound, bound, 0);
}

OaMatrix OaFnMatrix::RandGlorotUniform(OaMatrixShape InShape, OaScalarType InDtype) {
	// GPU-native: use PhiloxUniform for parallel generation
	// Glorot/Xavier uniform: U(-√(6/(fan_in+fan_out)), √(6/(fan_in+fan_out)))
	OaF32 bound = 0.1F;
	if (InShape.Rank == 2) {
		OaI64 fan_out = InShape[0];
		OaI64 fan_in = InShape[1];
		bound = std::sqrt(6.0F / static_cast<OaF32>(fan_in + fan_out));
	}
	
	auto t = OaFnMatrix::Empty(InShape, InDtype);
	return OaFnMatrix::PhiloxUniform(t, -bound, bound, 0);
}

OaMatrix OaFnMatrix::RandKaimingUniform(OaMatrixShape InShape, OaScalarType InDtype) {
	// GPU-native: use PhiloxUniform for parallel generation
	// Kaiming/He uniform: U(-√(6/fan_in), √(6/fan_in))
	OaF32 bound = 0.1F;
	if (InShape.Rank == 2) {
		OaI64 fan_in = InShape[1];
		bound = std::sqrt(6.0F / static_cast<OaF32>(fan_in));
	}
	
	auto t = OaFnMatrix::Empty(InShape, InDtype);
	return OaFnMatrix::PhiloxUniform(t, -bound, bound, 0);
}

OaMatrix OaFnMatrix::FromBytes(OaSpan<const OaU8> InData, OaMatrixShape InShape, OaScalarType InDtype) {
	OaI64 numElements = InShape.NumElements();
	OaI64 expectedBytes = numElements * static_cast<OaI64>(OaScalarSize(InDtype));

	// Float32 accepts either raw F32 bytes (4×) or U8 bytes (1×) to convert
	bool isFloat32U8Input = (InDtype == OaScalarType::Float32 && InData.Size() == static_cast<OaUsize>(numElements));
	// BFloat16 accepts raw FP32 bytes (4×numElements) and truncates to BF16 on host.
	bool isFp32ToBf16 = (InDtype == OaScalarType::BFloat16
		and InData.Size() == static_cast<OaUsize>(numElements * 4));
	if (InData.Size() != static_cast<OaUsize>(expectedBytes) && !isFloat32U8Input && !isFp32ToBf16) {
		OA_LOG_ERROR(OaLogComponent::Core,
			"FromBytes: input size mismatch. Expected %lld bytes for shape, got %zu bytes",
			static_cast<long long>(expectedBytes), InData.Size());
		return OaMatrix{};
	}

	auto t = OaFnMatrix::Empty(InShape, InDtype);
	if (!t.HasStorage()) return t;
	const OaU8* uploadData = InData.Data();
	OaVec<OaU8> converted;
	if (isFloat32U8Input) {
		converted.Resize(static_cast<OaUsize>(expectedBytes));
		const OaU8* src = InData.Data();
		auto* dst = reinterpret_cast<OaF32*>(converted.Data());
		for (OaI64 i = 0; i < numElements; ++i) dst[i] = static_cast<OaF32>(src[i]);
		uploadData = converted.Data();
	} else if (isFp32ToBf16) {
		converted.Resize(static_cast<OaUsize>(expectedBytes));
		const auto* src = reinterpret_cast<const OaF32*>(InData.Data());
		auto* dst = reinterpret_cast<OaU16*>(converted.Data());
		for (OaI64 i = 0; i < numElements; ++i) dst[i] = OaF32ToBf16(src[i]);
		uploadData = converted.Data();
	}
	if (auto* rt = ActiveRuntime()) {
		const auto status = rt->UploadBuffer(
			t.GetVkBuffer(), 0, uploadData, static_cast<OaU64>(expectedBytes));
		if (!status.IsOk()) {
			OA_LOG_ERROR(OaLogComponent::Core, "FromBytes upload failed: %s",
				status.GetMessage().CStr());
			return {};
		}
	}
	return t;
}

OaMatrix OaFnMatrix::FromInt32(OaSpan<const OaI32> InData, OaMatrixShape InShape, OaScalarType InDtype) {
	OaI64 numElements = InShape.NumElements();
	if (InData.Size() != static_cast<OaUsize>(numElements)) {
		OA_LOG_ERROR(OaLogComponent::Core,
			"FromInt32: input size mismatch. Expected %lld elements for shape, got %zu elements",
			static_cast<long long>(numElements), InData.Size());
		return OaMatrix{};
	}

	auto t = OaFnMatrix::Empty(InShape, InDtype);
	if (!t.HasStorage()) return t;
	const void* uploadData = InData.Data();
	OaVec<OaF32> converted;
	if (InDtype == OaScalarType::Float32) {
		converted.Resize(static_cast<OaUsize>(numElements));
		for (OaI64 i = 0; i < numElements; ++i) {
			converted[static_cast<OaUsize>(i)] = static_cast<OaF32>(InData[static_cast<OaUsize>(i)]);
		}
		uploadData = converted.Data();
	} else if (InDtype != OaScalarType::Int32 and InDtype != OaScalarType::UInt32) {
		OA_LOG_ERROR(OaLogComponent::Core,
			"FromInt32: unsupported dtype %d, use Int32, UInt32, or Float32",
			static_cast<int>(InDtype));
		return {};
	}
	if (auto* rt = ActiveRuntime()) {
		const auto status = rt->UploadBuffer(
			t.GetVkBuffer(), 0, uploadData, static_cast<OaU64>(t.ByteSize()));
		if (!status.IsOk()) {
			OA_LOG_ERROR(OaLogComponent::Core, "FromInt32 upload failed: %s",
				status.GetMessage().CStr());
			return {};
		}
	}
	return t;
}

OaMatrix OaFnMatrix::CausalMask(OaI64 InSeqLen) {
	if (InSeqLen <= 0) return {};
	// Reuse the tensor form so mask construction remains deferred GPU work.
	// Callers such as MHA cache the result, so this is paid only on shape changes.
	return OaFnMatrix::CausalMask(
		OaFnMatrix::Zeros(OaMatrixShape{InSeqLen, InSeqLen}, OaScalarType::Float32));
}

// Host ↔ Device Transfer
OaStatus OaFnMatrix::CopyToHost(const OaMatrix& InSrc, void* OutHost, OaU64 InBytes) {
	if (!OutHost) {
		return OaStatus::Error(OaStatusCode::InvalidArgument, "CopyToHost: null output pointer");
	}
	if (!InSrc.HasStorage()) {
		return OaStatus::Error(OaStatusCode::InvalidArgument, "CopyToHost: source has no storage");
	}

	const OaU64 srcBytes = static_cast<OaU64>(InSrc.ByteSize());
	if (InBytes < srcBytes) {
		return OaStatus::Error(OaStatusCode::InvalidArgument, "CopyToHost: buffer too small");
	}

	// Flush any context-recorded ops so the source buffer reflects the latest
	// writes before we memcpy from host-visible storage. Execute() early-returns
	// when nothing is pending, so this is free for ops that already executed.
	auto flush = OaContext::GetDefault().Execute();
	if (not flush.IsOk()) {
		return flush;
	}
	if (auto* rt = ActiveRuntime()) {
		return rt->ReadbackBuffer(
			InSrc.GetVkBuffer(), InSrc.ByteOffset(), OutHost, srcBytes);
	}
	return OaStatus::Error(OaStatusCode::FailedPrecondition, "CopyToHost: runtime unavailable");
}

// Note: OaFnMatrix::Scalar() is implemented in DeviceMatrixFn.cpp
// Multi-device: EmptyOn
OaMatrix OaFnMatrix::EmptyOn(OaMatrixShape InShape, OaScalarType InDtype, OaU32 InNodeIndex) {
	auto* rt = ActiveRuntime();
	if (not rt or InNodeIndex == 0 or not rt->IsMultiDevice()) {
		OaMatrix t = OaFnMatrix::Empty(InShape, InDtype);
		t.Device_.Index = static_cast<OaI32>(InNodeIndex);
		t.SyncMatrixDescriptor();
		return t;
	}

	auto* node = rt->GetNode(InNodeIndex);
	if (not node) {
		OaMatrix t = OaFnMatrix::Empty(InShape, InDtype);
		t.Device_.Index = static_cast<OaI32>(InNodeIndex);
		t.SyncMatrixDescriptor();
		return t;
	}

	OaMatrix t;
	t.Shape_ = InShape;
	t.Stride_ = OaStride::RowMajor(InShape);
	t.ByteOffset_ = 0;
	t.Dtype_ = InDtype;
	t.Device_ = OaDevice{node->Device.Info.Hardware.DeviceType, static_cast<OaI32>(InNodeIndex)};

	OaI64 bytes = InShape.NumElements() * static_cast<OaI64>(OaScalarSize(InDtype));
	if (bytes <= 0) {
		t.SyncMatrixDescriptor();
		return t;
	}

	auto res = node->Allocator.AllocHostVisible(static_cast<OaU64>(bytes));
	if (not res) {
		t.SyncMatrixDescriptor();
		return t;
	}

	OaSharedPtr<OaVkBuffer> buf(
		new OaVkBuffer(std::move(*res)),
		[rt, lifetime = rt->GetLifetimeToken()](OaVkBuffer* InPtr) {
			if (not InPtr) return;
			if (not lifetime.Expired()) {
				rt->FreeBufferOnNode(*InPtr);
			}
			delete InPtr;
		});
	buf->NodeIndex = InNodeIndex;
	rt->RegisterBufferForOwnedNode(*buf);
	t.VkBuf_ = buf;
	t.Data_ = OaSharedPtr<void>(buf->MappedPtr, [](void*) {});
	t.SyncMatrixDescriptor();
	return t;
}

// Note: OaMatrix::To() is implemented in DeviceMatrixFn.cpp to avoid duplication
