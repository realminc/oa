// OaFnMatrix — Allocation, initialisation, host transfer, multi-device.
//
// Empty, Zeros, Ones, Full, Rand*, FromBytes, CausalMask, CopyToHost, EmptyOn
//
// RNG functions (Rand, RandN, RandXavier, etc.) use GPU-native Philox PRNG
// via PhiloxUniform/PhiloxNormal (see FnMatrixRng.cpp).
//
// Note: Runtime management (SetRuntime, GetRuntime, SetWeightDtype, GetWeightDtype)
// is in DeviceMatrixFn.cpp

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
#include <Oa/Runtime/RuntimeGlobal.h>
#include <Oa/Runtime/Stream.h>
#include <Oa/Runtime/Topology.h>

#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <cstring>

// Allocators: Empty, Zeros, Ones, Full, Rand*, FromBytes, CausalMask
OaMatrix OaFnMatrix::Empty(OaMatrixShape InShape, OaScalarType InDtype) {

	OaMatrix t;
	t.Shape_ = InShape;
	t.Stride_ = OaStride::RowMajor(InShape);
	t.ByteOffset_ = 0;
	t.Dtype_ = InDtype;
	t.Device_ = OaDevice{OaDeviceType::VkDiscrete, 0};

	auto* rt = OaRuntimeGlobal::GetRuntime();
	if (not rt) {
		t.SyncMatrixDescriptor();
		return t;
	}

	OaI64 bytes = InShape.NumElements() * static_cast<OaI64>(OaScalarSize(InDtype));
	if (bytes <= 0) {
		t.SyncMatrixDescriptor();
		return t;
	}

	// Engine's HostVisibleBufferCache provides best-fit buffer reuse after warm-up.
	auto res = rt->AllocBuffer(static_cast<OaU64>(bytes));
	if (not res) {
		t.SyncMatrixDescriptor();
		return t;
	}

	OaSharedPtr<OaVkBuffer> buf(
		new OaVkBuffer(std::move(*res)),
		[](OaVkBuffer* InPtr) {
			if (not InPtr) return;
			auto* rt = OaRuntimeGlobal::GetRuntime();
			if (rt) {
				rt->FreeBuffer(*InPtr);
			}
			delete InPtr;
		});
	buf->NodeIndex = 0;
	t.VkBuf_ = buf;
	t.Data_ = OaSharedPtr<void>(buf->MappedPtr, [](void*) {});
	t.SyncMatrixDescriptor();
	return t;
}

OaMatrix OaFnMatrix::Zeros(OaMatrixShape InShape, OaScalarType InDtype) {
	auto t = OaFnMatrix::Empty(InShape, InDtype);
	if (t.HasStorage()) {
		OaMemzero(t.Data(), static_cast<OaUsize>(t.ByteSize()));
		if (auto* rt = OaRuntimeGlobal::GetRuntime()) {
			(void)rt->Allocator.FlushHostBuffer(
				t.GetVkBuffer(), 0, static_cast<OaU64>(t.ByteSize()));
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
	if (t.Data()) {
		if (isFloat32U8Input) {
			// Convert U8 to F32
			const OaU8* src = InData.Data();
			OaF32* dst = static_cast<OaF32*>(t.Data());
			for (OaI64 i = 0; i < numElements; ++i) {
				dst[i] = static_cast<OaF32>(src[i]);
			}
		} else if (isFp32ToBf16) {
			// Convert FP32 host bytes to BF16 device storage (truncation).
			const OaF32* src = reinterpret_cast<const OaF32*>(InData.Data());
			OaU16* dst = static_cast<OaU16*>(t.Data());
			for (OaI64 i = 0; i < numElements; ++i) {
				dst[i] = OaF32ToBf16(src[i]);
			}
		} else {
			std::memcpy(t.Data(), InData.Data(), static_cast<OaUsize>(expectedBytes));
		}
		// Make the host write visible to a subsequent GPU read. No-op on coherent
		// memory, but VMA can hand out non-coherent HOST_VISIBLE memory under
		// pressure (seen deep in a long test run / training loop), and without the
		// flush the GPU reads stale zeros — silently corrupting the upload.
		if (auto* rt = OaRuntimeGlobal::GetRuntime()) {
			const OaVkBuffer vkb = t.GetVkBuffer();
			(void)rt->Allocator.FlushHostBuffer(vkb, 0, static_cast<OaU64>(t.ByteSize()));
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
	if (t.Data()) {
		if (InDtype == OaScalarType::Int32 or InDtype == OaScalarType::UInt32) {
			std::memcpy(t.Data(), InData.Data(), static_cast<OaUsize>(numElements * sizeof(OaI32)));
		} else if (InDtype == OaScalarType::Float32) {
			const OaI32* src = InData.Data();
			OaF32* dst = static_cast<OaF32*>(t.Data());
			for (OaI64 i = 0; i < numElements; ++i) {
				dst[i] = static_cast<OaF32>(src[i]);
			}
		} else {
			OA_LOG_ERROR(OaLogComponent::Core,
				"FromInt32: unsupported dtype %d, use Int32, UInt32, or Float32",
				static_cast<int>(InDtype));
			return OaMatrix{};
		}
		if (auto* rt = OaRuntimeGlobal::GetRuntime()) {
			(void)rt->Allocator.FlushHostBuffer(
				t.GetVkBuffer(), 0, static_cast<OaU64>(t.ByteSize()));
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
	if (auto* rt = OaRuntimeGlobal::GetRuntime()) {
		if (not rt->Allocator.InvalidateHostBuffer(
			InSrc.GetVkBuffer(), 0, srcBytes)) {
			return OaStatus::Error(
				OaStatusCode::VulkanError,
				"CopyToHost: mapped allocation invalidate failed");
		}
	}

	const void* srcData = InSrc.Data();
	if (!srcData) {
		return OaStatus::Error(OaStatusCode::InvalidArgument, "CopyToHost: source data is null");
	}

	OaMemcpy(OutHost, srcData, static_cast<OaUsize>(srcBytes));
	return OaStatus::Ok();
}

// Note: OaFnMatrix::Scalar() is implemented in DeviceMatrixFn.cpp
// Multi-device: EmptyOn
OaMatrix OaFnMatrix::EmptyOn(OaMatrixShape InShape, OaScalarType InDtype, OaU32 InNodeIndex) {
	auto* rt = OaRuntimeGlobal::GetRuntime();
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
		[](OaVkBuffer* InPtr) {
			if (not InPtr) return;
			auto* rt = OaRuntimeGlobal::GetRuntime();
			if (rt) {
				rt->FreeBuffer(*InPtr);
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
