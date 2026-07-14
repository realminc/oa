// FnMatrixQuant.cpp — Quantization and dequantization operations
//
// Implements OaFnMatrix::Quantize() and OaFnMatrix::Dequantize() for
// block-wise quantization formats (Q4_K, Q8_0, etc.).

#include <Oa/Core/FnMatrix.h>
#include <Oa/Core/QuantBlocks.h>
#include <Oa/Runtime/Context.h>
#include <Oa/Runtime/Dispatch.h>
#include <Oa/Core/Log.h>

namespace OaFnMatrix {

// Helper: Get kernel name for quantization
static const char* GetQuantizeKernelName(OaScalarType InTargetType) {
	switch (InTargetType) {
		case OaScalarType::Q8_0:  return "QuantizeQ8_0";
		case OaScalarType::Q8_1:  return "QuantizeQ8_1";
		case OaScalarType::Q4_0:  return "QuantizeQ4_0";
		case OaScalarType::Q4_1:  return "QuantizeQ4_1";
		case OaScalarType::Q5_0:  return "QuantizeQ5_0";
		case OaScalarType::Q5_1:  return "QuantizeQ5_1";
		case OaScalarType::Q2_K:  return "QuantizeQ2_K";
		case OaScalarType::Q3_K:  return "QuantizeQ3_K";
		case OaScalarType::Q4_K:  return "QuantizeQ4_K";
		case OaScalarType::Q5_K:  return "QuantizeQ5_K";
		case OaScalarType::Q6_K:  return "QuantizeQ6_K";
		case OaScalarType::Q8_K:  return "QuantizeQ8_K";
		default: return nullptr;
	}
}

// Helper: Get kernel name for dequantization
static const char* GetDequantizeKernelName(OaScalarType InQuantType) {
	switch (InQuantType) {
		case OaScalarType::Q8_0:  return "DequantizeQ8_0";
		case OaScalarType::Q8_1:  return "DequantizeQ8_1";
		case OaScalarType::Q4_0:  return "DequantizeQ4_0";
		case OaScalarType::Q4_1:  return "DequantizeQ4_1";
		case OaScalarType::Q5_0:  return "DequantizeQ5_0";
		case OaScalarType::Q5_1:  return "DequantizeQ5_1";
		case OaScalarType::Q2_K:  return "DequantizeQ2_K";
		case OaScalarType::Q3_K:  return "DequantizeQ3_K";
		case OaScalarType::Q4_K:  return "DequantizeQ4_K";
		case OaScalarType::Q5_K:  return "DequantizeQ5_K";
		case OaScalarType::Q6_K:  return "DequantizeQ6_K";
		case OaScalarType::Q8_K:  return "DequantizeQ8_K";
		default: return nullptr;
	}
}

OaMatrix Quantize(const OaMatrix& InTensor, OaScalarType InTargetType) {
	// Validate input
	if (InTensor.GetDtype() != OaScalarType::Float32) {
		OA_LOG_ERROR(OaLogComponent::Core, 
			"Quantize: Input must be FP32, got %s",
			OaScalarTypeToString(InTensor.GetDtype()));
		return OaMatrix();
	}
	
	if (!OaIsQuantized(InTargetType)) {
		OA_LOG_ERROR(OaLogComponent::Core,
			"Quantize: Target type %s is not a quantized format",
			OaScalarTypeToString(InTargetType));
		return OaMatrix();
	}
	
	const char* kernelName = GetQuantizeKernelName(InTargetType);
	if (!kernelName) {
		OA_LOG_ERROR(OaLogComponent::Core,
			"Quantize: No kernel available for %s (not implemented yet)",
			OaScalarTypeToString(InTargetType));
		return OaMatrix();
	}
	
	// Compute block layout
	const OaI64 numElements = InTensor.NumElements();
	const OaU32 blockSize = OaQuantBlockSize(InTargetType);
	const OaU32 blockBytes = OaQuantBlockBytes(InTargetType);
	
	if (numElements % blockSize != 0) {
		OA_LOG_ERROR(OaLogComponent::Core,
			"Quantize: Tensor size %lld must be divisible by block size %u for %s",
			static_cast<long long>(numElements), blockSize,
			OaScalarTypeToString(InTargetType));
		return OaMatrix();
	}
	
	const OaU32 numBlocks = static_cast<OaU32>(numElements / blockSize);
	
	// Allocate output buffer (quantized blocks)
	// Store as 1D tensor with custom dtype
	OaMatrix output;
	output.Shape_ = OaMatrixShape{numElements};
	output.Dtype_ = InTargetType;
	output.Device_ = InTensor.GetDevice();
	
	// Allocate storage for quantized blocks
	const OaU64 outputBytes = static_cast<OaU64>(numBlocks) * blockBytes;
	auto& ctx = OaContext::GetDefault();
	output.Storage_ = ctx.AllocateStorage(outputBytes);
	
	if (!output.Storage_) {
		OA_LOG_ERROR(OaLogComponent::Core,
			"Quantize: Failed to allocate %llu bytes for quantized output",
			static_cast<unsigned long long>(outputBytes));
		return OaMatrix();
	}
	
	// Dispatch quantization kernel
	struct {
		OaU32 input_idx;
		OaU32 output_idx;
		OaU32 num_blocks;
	} pc;
	
	pc.input_idx = InTensor.Storage_->BufferIndex;
	pc.output_idx = output.Storage_->BufferIndex;
	pc.num_blocks = numBlocks;
	
	OaVec<OaVkBuffer> buffers = {
		InTensor.GetVkBuffer(),
		output.GetVkBuffer()
	};
	
	// Dispatch with 256 threads per workgroup
	const OaU32 numWorkgroups = (numBlocks + 255) / 256;
	
	OaStatus status = OaVkDispatch::Run(
		ctx.GetRuntime(),
		kernelName,
		buffers,
		&pc,
		sizeof(pc),
		numWorkgroups,
		1,
		1
	);
	
	if (!status.IsOk()) {
		OA_LOG_ERROR(OaLogComponent::Core,
			"Quantize: Kernel dispatch failed: %s",
			status.GetMessage().Data());
		return OaMatrix();
	}
	
	return output;
}

OaMatrix Dequantize(const OaMatrix& InQuantized) {
	// Validate input
	if (!OaIsQuantized(InQuantized.GetDtype())) {
		OA_LOG_ERROR(OaLogComponent::Core,
			"Dequantize: Input type %s is not a quantized format",
			OaScalarTypeToString(InQuantized.GetDtype()));
		return OaMatrix();
	}
	
	const char* kernelName = GetDequantizeKernelName(InQuantized.GetDtype());
	if (!kernelName) {
		OA_LOG_ERROR(OaLogComponent::Core,
			"Dequantize: No kernel available for %s (not implemented yet)",
			OaScalarTypeToString(InQuantized.GetDtype()));
		return OaMatrix();
	}
	
	// Compute output size
	const OaI64 numElements = InQuantized.NumElements();
	const OaU32 blockSize = OaQuantBlockSize(InQuantized.GetDtype());
	const OaU32 blockBytes = OaQuantBlockBytes(InQuantized.GetDtype());
	const OaU32 numBlocks = static_cast<OaU32>(numElements / blockSize);
	
	// Allocate FP32 output
	OaMatrix output = Empty(InQuantized.GetShape(), OaScalarType::Float32);
	
	// Dispatch dequantization kernel
	struct {
		OaU32 input_idx;
		OaU32 output_idx;
		OaU32 num_blocks;
	} pc;
	
	pc.input_idx = InQuantized.Storage_->BufferIndex;
	pc.output_idx = output.Storage_->BufferIndex;
	pc.num_blocks = numBlocks;
	
	OaVec<OaVkBuffer> buffers = {
		InQuantized.GetVkBuffer(),
		output.GetVkBuffer()
	};
	
	// Dispatch with 256 threads per workgroup
	const OaU32 numWorkgroups = (numBlocks + 255) / 256;
	
	auto& ctx = OaContext::GetDefault();
	OaStatus status = OaVkDispatch::Run(
		ctx.GetRuntime(),
		kernelName,
		buffers,
		&pc,
		sizeof(pc),
		numWorkgroups,
		1,
		1
	);
	
	if (!status.IsOk()) {
		OA_LOG_ERROR(OaLogComponent::Core,
			"Dequantize: Kernel dispatch failed: %s",
			status.GetMessage().Data());
		return OaMatrix();
	}
	
	return output;
}

} // namespace OaFnMatrix

