// oa::FnMatrix — allocation, initialization, and host transfer.
//
// Empty, Zeros, Ones, Full, Rand*, FromBytes, CausalMask, CopyToHost
//
// RNG functions (Rand, RandN, RandXavier, etc.) use GPU-native Philox PRNG
// via PhiloxUniform/philoxNormal (see FnMatrixRng.cpp).
//
// weight dtype configuration lives in FnMatrix.cpp. allocation resolves the
// current engine through the selected oa::ExecutionSession.

#include <oa/core/matrix.h>
#include <oa/core/matrixAccess.h>
#include <oa/core/fnMatrix.h>
#include <oa/core/fnmatrix/fnMatrixInternal.h>
#include <oa/core/log.h>
#include <oa/core/std/memory.h>
#include <oa/core/status.h>
#include <oa/core/types.h>
#include <oa/runtime/executionSession.h>
#include <oa/runtime/engine/resourceAccess.h>
#include <oa/runtime/executionSession.h>

#include <assert.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

namespace {

oa::Engine* activeRuntime() {
	auto* context = oa::ExecutionSession::getActivePtr();
	return context ? &context->engine() : nullptr;
}

} // namespace

// Allocators: Empty, Zeros, Ones, Full, Rand*, FromBytes, CausalMask
oa::Matrix oa::FnMatrix::empty(
	oa::MatrixShape inShape, oa::ScalarType inDtype, oa::MemoryPlacement inPlacement) {

	oa::Matrix t;
	oa::MatrixAccess::shape(t) = inShape;
	oa::MatrixAccess::stride(t) = oa::Stride::rowMajor(inShape);
	oa::MatrixAccess::byteOffset(t) = 0;
	oa::MatrixAccess::dtype(t) = inDtype;
	oa::MatrixAccess::device(t) = oa::Device{oa::DeviceType::VkDiscrete, 0};

	auto* rt = activeRuntime();
	if (not rt) {
		oa::MatrixAccess::syncDescriptor(t);
		return t;
	}

	oa::I64 bytes = inShape.numElements() * static_cast<oa::I64>(oa::scalarSize(inDtype));
	if (bytes <= 0) {
		oa::MatrixAccess::syncDescriptor(t);
		return t;
	}

	// The execution session owns allocation policy. Repeatable training frames
	// map allocation ordinals onto stable VkBuffer/bindless slots for exact
	// command-graph replay without exposing memory policy through oa::ExecutionSession.
	auto& context = oa::ExecutionSession::getActive();
	auto buf = context.allocateMatrixBuffer(static_cast<oa::U64>(bytes), inPlacement);
	if (not buf) {
		oa::MatrixAccess::syncDescriptor(t);
		return t;
	}
	oa::MatrixAccess::storageOwner(t) = buf;
	oa::MatrixAccess::syncDescriptor(t);
	return t;
}

oa::Matrix oa::FnMatrix::zeros(oa::MatrixShape inShape, oa::ScalarType inDtype) {
	auto t = oa::FnMatrix::empty(inShape, inDtype);
	if (t.hasStorage()) {
		if (t.data()) {
			oa::memzero(t.data(), static_cast<oa::Usize>(t.byteSize()));
			if (auto* rt = activeRuntime()) {
				(void)oa::EngineResourceAccess::uploadBuffer(*rt,
					oa::MatrixAccess::descriptor(t), 0, t.data(),
					static_cast<oa::U64>(t.byteSize()));
			}
		} else {
			oa::FnMatrix::fillInPlace(t, 0.0F);
		}
	}
	return t;
}

oa::Matrix oa::FnMatrix::ones(oa::MatrixShape inShape, oa::ScalarType inDtype) {
	auto t = oa::FnMatrix::empty(inShape, inDtype);
	if (t.hasStorage()) {
		oa::FnMatrix::fillInPlace(t, 1.0f);
	}
	return t;
}

oa::Matrix oa::FnMatrix::full(oa::MatrixShape inShape, oa::F64 inValue, oa::ScalarType inDtype) {
	auto t = oa::FnMatrix::empty(inShape, inDtype);
	if (t.hasStorage()) {
		oa::FnMatrix::fillInPlace(t, static_cast<oa::F32>(inValue));
	}
	return t;
}

oa::Matrix oa::FnMatrix::rand(oa::MatrixShape inShape, oa::ScalarType inDtype) {
	// GPU-native: use PhiloxUniform for parallel generation
	auto t = oa::FnMatrix::empty(inShape, inDtype);
	return oa::FnMatrix::philoxUniform(t, 0.0F, 1.0F, 0);
}

oa::Matrix oa::FnMatrix::randN(oa::MatrixShape inShape, oa::ScalarType inDtype) {
	// GPU-native: use PhiloxNormal for parallel generation
	// Kaiming/He normal initialization: N(0, √(2/fan_in))
	oa::F32 std_dev = 1.0F;
	if (inShape.rank == 2) {
		oa::I64 fan_in = inShape[1];  // embed_dim for embeddings
		std_dev = oa::sqrt(2.0F / static_cast<oa::F32>(fan_in));
	} else {
		// For other shapes, use small std_dev
		std_dev = 0.02F;
	}

	auto t = oa::FnMatrix::empty(inShape, inDtype);
	return oa::FnMatrix::philoxNormal(t, 0.0F, std_dev, 0);
}

oa::Matrix oa::FnMatrix::randXavier(oa::MatrixShape inShape, oa::ScalarType inDtype) {
	// GPU-native: use PhiloxUniform for parallel generation
	// PyTorch nn.linear default: kaiming_uniform_(a=sqrt(5))
	oa::F32 bound = 1.0F;
	if (inShape.rank == 2) {
		oa::I64 fan_in = inShape[1];  // For weight matrix [out_features, in_features]
		bound = oa::sqrt(1.0F / static_cast<oa::F32>(fan_in));
	} else {
		// For other shapes, use small uniform range
		bound = 0.1F;
	}

	auto t = oa::FnMatrix::empty(inShape, inDtype);
	return oa::FnMatrix::philoxUniform(t, -bound, bound, 0);
}

oa::Matrix oa::FnMatrix::randGlorotUniform(oa::MatrixShape inShape, oa::ScalarType inDtype) {
	// GPU-native: use PhiloxUniform for parallel generation
	// Glorot/Xavier uniform: U(-√(6/(fan_in+fan_out)), √(6/(fan_in+fan_out)))
	oa::F32 bound = 0.1F;
	if (inShape.rank == 2) {
		oa::I64 fan_out = inShape[0];
		oa::I64 fan_in = inShape[1];
		bound = oa::sqrt(6.0F / static_cast<oa::F32>(fan_in + fan_out));
	}

	auto t = oa::FnMatrix::empty(inShape, inDtype);
	return oa::FnMatrix::philoxUniform(t, -bound, bound, 0);
}

oa::Matrix oa::FnMatrix::randKaimingUniform(oa::MatrixShape inShape, oa::ScalarType inDtype) {
	// GPU-native: use PhiloxUniform for parallel generation
	// Kaiming/He uniform: U(-√(6/fan_in), √(6/fan_in))
	oa::F32 bound = 0.1F;
	if (inShape.rank == 2) {
		oa::I64 fan_in = inShape[1];
		bound = oa::sqrt(6.0F / static_cast<oa::F32>(fan_in));
	}

	auto t = oa::FnMatrix::empty(inShape, inDtype);
	return oa::FnMatrix::philoxUniform(t, -bound, bound, 0);
}

oa::Matrix oa::FnMatrix::fromBytes(oa::Span<const oa::U8> inData, oa::MatrixShape inShape, oa::ScalarType inDtype) {
	oa::I64 numElements = inShape.numElements();
	oa::I64 expectedBytes = numElements * static_cast<oa::I64>(oa::scalarSize(inDtype));

	// Float32 accepts either raw F32 bytes (4×) or U8 bytes (1×) to convert
	bool isFloat32U8Input = (inDtype == oa::ScalarType::Float32 && inData.size() == static_cast<oa::Usize>(numElements));
	// BFloat16 accepts raw FP32 bytes (4×numElements) and truncates to BF16 on host.
	bool isFp32ToBf16 = (inDtype == oa::ScalarType::BFloat16
		and inData.size() == static_cast<oa::Usize>(numElements * 4));
	if (inData.size() != static_cast<oa::Usize>(expectedBytes) && !isFloat32U8Input && !isFp32ToBf16) {
		OaLogError(oa::LogComponent::Compute,
			"FromBytes: input size mismatch. expected %lld bytes for shape, got %zu bytes",
			static_cast<long long>(expectedBytes), inData.size());
		return oa::Matrix{};
	}

	auto t = oa::FnMatrix::empty(inShape, inDtype);
	if (!t.hasStorage()) return t;
	const oa::U8* uploadData = inData.data();
	oa::Vector<oa::U8> converted;
	if (isFloat32U8Input) {
		converted.resize(static_cast<oa::Usize>(expectedBytes));
		const oa::U8* src = inData.data();
		auto* dst = reinterpret_cast<oa::F32*>(converted.data());
		for (oa::I64 i = 0; i < numElements; ++i) dst[i] = static_cast<oa::F32>(src[i]);
		uploadData = converted.data();
	} else if (isFp32ToBf16) {
		converted.resize(static_cast<oa::Usize>(expectedBytes));
		const auto* src = reinterpret_cast<const oa::F32*>(inData.data());
		auto* dst = reinterpret_cast<oa::U16*>(converted.data());
		for (oa::I64 i = 0; i < numElements; ++i) dst[i] = oa::f32ToBf16(src[i]);
		uploadData = converted.data();
	}
	if (auto* rt = activeRuntime()) {
		const auto status = oa::EngineResourceAccess::uploadBuffer(*rt,
			oa::MatrixAccess::descriptor(t), 0, uploadData,
			static_cast<oa::U64>(expectedBytes));
		if (!status.isOk()) {
			OaLogError(oa::LogComponent::Compute, "FromBytes upload failed: %s",
				status.getMessage().cStr());
			return {};
		}
	}
	return t;
}

oa::Matrix oa::FnMatrix::fromInt32(oa::Span<const oa::I32> inData, oa::MatrixShape inShape, oa::ScalarType inDtype) {
	oa::I64 numElements = inShape.numElements();
	if (inData.size() != static_cast<oa::Usize>(numElements)) {
		OaLogError(oa::LogComponent::Compute,
			"FromInt32: input size mismatch. expected %lld elements for shape, got %zu elements",
			static_cast<long long>(numElements), inData.size());
		return oa::Matrix{};
	}

	auto t = oa::FnMatrix::empty(inShape, inDtype);
	if (!t.hasStorage()) return t;
	const void* uploadData = inData.data();
	oa::Vector<oa::F32> converted;
	if (inDtype == oa::ScalarType::Float32) {
		converted.resize(static_cast<oa::Usize>(numElements));
		for (oa::I64 i = 0; i < numElements; ++i) {
			converted[static_cast<oa::Usize>(i)] = static_cast<oa::F32>(inData[static_cast<oa::Usize>(i)]);
		}
		uploadData = converted.data();
	} else if (inDtype != oa::ScalarType::Int32 and inDtype != oa::ScalarType::UInt32) {
		OaLogError(oa::LogComponent::Compute,
			"FromInt32: unsupported dtype %d, use Int32, UInt32, or Float32",
			static_cast<int>(inDtype));
		return {};
	}
	if (auto* rt = activeRuntime()) {
		const auto status = oa::EngineResourceAccess::uploadBuffer(*rt,
			oa::MatrixAccess::descriptor(t), 0, uploadData,
			static_cast<oa::U64>(t.byteSize()));
		if (!status.isOk()) {
			OaLogError(oa::LogComponent::Compute, "FromInt32 upload failed: %s",
				status.getMessage().cStr());
			return {};
		}
	}
	return t;
}

oa::Matrix oa::FnMatrix::causalMask(oa::I64 inSeqLen) {
	if (inSeqLen <= 0) return {};
	// Reuse the tensor form so mask construction remains deferred GPU work.
	// Callers such as MHA cache the result, so this is paid only on shape changes.
	return oa::FnMatrix::causalMask(
		oa::FnMatrix::zeros(oa::MatrixShape{inSeqLen, inSeqLen}, oa::ScalarType::Float32));
}

// Host ↔ Device Transfer
oa::Status oa::FnMatrix::copyToHost(const oa::Matrix& inSrc, void* outHost, oa::U64 inBytes) {
	if (!outHost) {
		return oa::Status::error(oa::StatusCode::InvalidArgument, "CopyToHost: null output pointer");
	}
	if (!inSrc.hasStorage()) {
		return oa::Status::error(oa::StatusCode::InvalidArgument, "CopyToHost: source has no storage");
	}

	const oa::U64 srcBytes = static_cast<oa::U64>(inSrc.byteSize());
	if (inBytes < srcBytes) {
		return oa::Status::error(oa::StatusCode::InvalidArgument, "CopyToHost: buffer too small");
	}

	// flush any context-recorded ops so the source buffer reflects the latest
	// writes before we memcpy from host-visible storage. An empty recording is a
	// validated no-op; non-empty work completes through its exact event.
	auto flush = oa::FnMatrix::completeRecordedWork(oa::ExecutionSession::getActive());
	if (not flush.isOk()) {
		return flush;
	}
	if (auto* rt = activeRuntime()) {
		return oa::EngineResourceAccess::readbackBuffer(*rt,
			oa::MatrixAccess::descriptor(inSrc), inSrc.byteOffset(), outHost, srcBytes);
	}
	return oa::Status::error(oa::StatusCode::FailedPrecondition, "CopyToHost: runtime unavailable");
}

// Note: oa::FnMatrix::scalar() is implemented in DeviceMatrixFn.cpp
