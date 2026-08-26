// oa::FnMatrix — native symmetric Q4/Q8 block-plane operations.

#include <oa/ml/fnMatrix.h>
#include <oa/core/matrix.h>
#include <oa/core/fnMatrix.h>
#include <oa/core/log.h>
#include <oa/core/op.h>
#include <oa/core/status.h>
#include <oa/core/types.h>
#include <oa/core/bufferAccess.h>
#include <oa/core/matrixAccess.h>
#include <oa/runtime/executionSession.h>
#include <oa/core/std/limits.h>

#include "fnMatrixQuantInternal.h"
#include "../../quantMatrixAccess.h"

namespace {

constexpr oa::U32 Q4BlockSize = 32;
constexpr oa::U32 Q4BytesPerBlock = 16;
constexpr oa::U32 Q4WordsPerBlock = 4;
constexpr oa::U32 Q8BlockSize = 32;
constexpr oa::U32 Q8BytesPerBlock = 32;
constexpr oa::U32 Q8WordsPerBlock = 8;
constexpr oa::U32 QuantizeWorkgroupSize = 64;
constexpr oa::U32 ElementWorkgroupSize = 256;
constexpr oa::U64 MaxFloat32ElementCount =
	oa::Limits<oa::U32>::max() / sizeof(oa::F32);

enum class QuantPlaneFormat : oa::U8 {
	Q4,
	Q8,
};

[[nodiscard]] oa::U32 divCeil(oa::U32 inA, oa::U32 inB) {
	return inA / inB + (inA % inB != 0 ? 1U : 0U);
}

[[nodiscard]] bool isAdmittedCount(oa::I64 inCount, const char* inOperation) {
	if (inCount < 0
		or static_cast<oa::U64>(inCount) > MaxFloat32ElementCount)
	{
		OaLogError(oa::LogComponent::Ml,
			"%s: Float32 byte offsets must fit the UInt32 kernel ABI", inOperation);
		return false;
	}
	return true;
}

[[nodiscard]] bool isFloat32(const oa::Matrix& inMatrix) {
	return inMatrix.getDtype() == oa::ScalarType::Float32;
}

[[nodiscard]] bool tryMultiply(
	oa::U64 inA,
	oa::U64 inB,
	oa::U64& outProduct)
{
	if (inA != 0 and inB > oa::Limits<oa::U64>::max() / inA) {
		return false;
	}
	outProduct = inA * inB;
	return true;
}

// The semantic graph currently binds one primary storage resource per value.
// A quantized value owns two physical planes, so use its payload as the stable
// primary identity while recording the logical Float32 shape and dtype. The
// scale plane remains strongly owned by the executable nodes and therefore
// appears in capture's complete resource inventory; it is never exposed as a
// second public matrix value.
[[nodiscard]] oa::Matrix semanticQuantizedProxy(
	const oa::QuantMatrix& inValue)
{
	oa::Matrix proxy = oa::QuantMatrixAccess::payload(inValue);
	oa::MatrixAccess::shape(proxy) = inValue.getShape();
	oa::MatrixAccess::stride(proxy) = oa::Stride::rowMajor(inValue.getShape());
	oa::MatrixAccess::dtype(proxy) = oa::ScalarType::Float32;
	oa::MatrixAccess::autograd(proxy).reset();
	return proxy;
}

[[nodiscard]] oa::OpAttribute quantizationAttribute(
	oa::Quantization inQuantization)
{
	return oa::OpAttribute::fromEnum(
		"quantization", oa::String(oa::quantizationToString(inQuantization)));
}

[[nodiscard]] oa::Matrix matMulNtQuantized(
	const oa::Matrix& inInput,
	const oa::Matrix& inPayload,
	const oa::Matrix& inScale,
	oa::I64 inOutputFeatures,
	QuantPlaneFormat inFormat,
	const oa::OpContract& inContract,
	const char* inKernel)
{
	auto fail = [inKernel](const char* inMessage) {
		OaLogError(oa::LogComponent::Ml, "%s: %s", inKernel, inMessage);
		return oa::Matrix{};
	};
	if (inInput.rank() < 2) {
		return fail("input must have rank >= 2 with K in the last dimension");
	}
	if (inPayload.rank() != 1 or inScale.rank() != 1) {
		return fail("payload and scale planes must both be one-dimensional");
	}
	const oa::ScalarType payloadDtype = inFormat == QuantPlaneFormat::Q4
		? oa::ScalarType::UInt8
		: oa::ScalarType::Int8;
	if (not isFloat32(inInput)
		or inPayload.getDtype() != payloadDtype
		or not isFloat32(inScale))
	{
		return fail(inFormat == QuantPlaneFormat::Q4
			? "expected Float32 input, UInt8 payload, and Float32 scale"
			: "expected Float32 input, Int8 payload, and Float32 scale");
	}
	if (not inInput.getStride().matchesRowMajor(inInput.getShape())
		or not inPayload.getStride().matchesRowMajor(inPayload.getShape())
		or not inScale.getStride().matchesRowMajor(inScale.getShape())
		or inInput.byteOffset() != 0
		or inPayload.byteOffset() != 0
		or inScale.byteOffset() != 0)
	{
		return fail("input and quantization planes must be contiguous base views");
	}
	if (inOutputFeatures < 0
		or static_cast<oa::U64>(inOutputFeatures)
			> oa::Limits<oa::U32>::max())
	{
		return fail("OutputFeatures must fit the UInt32 kernel ABI");
	}

	const oa::I64 inputFeatures = inInput.size(inInput.rank() - 1);
	if (inputFeatures <= 0
		or static_cast<oa::U64>(inputFeatures)
			> oa::Limits<oa::U32>::max())
	{
		return fail("the input K dimension must be positive and fit UInt32");
	}

	oa::U64 rows = 1;
	for (oa::I32 dim = 0; dim < inInput.rank() - 1; ++dim) {
		const oa::I64 extent = inInput.size(dim);
		if (extent < 0) return fail("input dimensions must be non-negative");
		oa::U64 nextRows = 0;
		if (not tryMultiply(rows, static_cast<oa::U64>(extent), nextRows)) {
			return fail("flattened input row count overflows UInt64");
		}
		rows = nextRows;
	}
	if (rows > oa::Limits<oa::U32>::max()) {
		return fail("flattened input row count must fit UInt32");
	}

	const oa::U64 outputFeatures = static_cast<oa::U64>(inOutputFeatures);
	const oa::U64 k = static_cast<oa::U64>(inputFeatures);
	oa::U64 weightCount = 0;
	oa::U64 inputCount = 0;
	oa::U64 outputCount = 0;
	if (not tryMultiply(outputFeatures, k, weightCount)
		or not tryMultiply(rows, k, inputCount)
		or not tryMultiply(rows, outputFeatures, outputCount))
	{
		return fail("matrix dimensions overflow UInt64");
	}
	if (weightCount > MaxFloat32ElementCount
		or inputCount > MaxFloat32ElementCount
		or outputCount > MaxFloat32ElementCount)
	{
		return fail("matrix byte offsets exceed the UInt32 kernel ABI");
	}

	const oa::U64 blocks = weightCount / Q4BlockSize
		+ (weightCount % Q4BlockSize != 0 ? 1U : 0U);
	const oa::U64 bytesPerBlock = inFormat == QuantPlaneFormat::Q4
		? Q4BytesPerBlock
		: Q8BytesPerBlock;
	oa::U64 payloadBytes = 0;
	if (not tryMultiply(blocks, bytesPerBlock, payloadBytes)
		or payloadBytes > static_cast<oa::U64>(oa::Limits<oa::I64>::max()))
	{
		return fail("quantized payload size overflows the matrix ABI");
	}
	if (inPayload.numElements() != static_cast<oa::I64>(payloadBytes)
		or inScale.numElements() != static_cast<oa::I64>(blocks))
	{
		return fail("payload/scale shape does not match [OutputFeatures,K]");
	}

	auto outputShape = inInput.getShape();
	outputShape.dims[static_cast<oa::Usize>(outputShape.rank - 1)] =
		inOutputFeatures;
	oa::Matrix output = oa::FnMatrix::empty(outputShape, oa::ScalarType::Float32);
	if (rows == 0 or outputFeatures == 0) return output;
	if (not inInput.hasStorage()
		or not inPayload.hasStorage()
		or not inScale.hasStorage()
		or not output.hasStorage())
	{
		return fail("non-empty inputs and output require live engine storage");
	}

	auto& context = oa::ExecutionSession::getActive();
	const auto semantic = context.recordOp(
		inContract,
		{&inInput, &inPayload, &inScale},
		{&output},
		{oa::OpAttribute::fromSignedInteger(
			"outputFeatures", inOutputFeatures)});
	if (not semantic.isOk()) return {};

	struct {
		oa::U32 M;
		oa::U32 N;
		oa::U32 K;
	} push{
		static_cast<oa::U32>(rows),
		static_cast<oa::U32>(outputFeatures),
		static_cast<oa::U32>(k),
	};
	oa::BufferAccess access[] = {
		oa::BufferAccess::Read,
		oa::BufferAccess::Read,
		oa::BufferAccess::Read,
		oa::BufferAccess::Write,
	};
	context.add(
		inKernel,
		{&inInput, &inPayload, &inScale, &output},
		access,
		&push,
		sizeof(push),
		static_cast<oa::U32>(outputFeatures),
		static_cast<oa::U32>(rows),
		1,
		inContract.name,
		0,
		inContract.hash,
		0,
		0,
		semantic.getValue());
	return output;
}

} // namespace

oa::Matrix oa::FnMatrix::quantizeQ4(const oa::Matrix& inInput, const oa::Matrix& inScale) {
	auto& ctx = oa::ExecutionSession::getActive();
	const oa::I64 count = inInput.numElements();
	if (not isAdmittedCount(count, "QuantizeQ4")) return {};
	if (not isFloat32(inInput) or not isFloat32(inScale)) {
		OaLogError(oa::LogComponent::Ml,
			"QuantizeQ4: input and scale must use Float32 storage");
		return {};
	}
	const oa::U32 count32 = static_cast<oa::U32>(count);
	const oa::U32 numBlocks = divCeil(count32, Q4BlockSize);
	if (inScale.numElements() != static_cast<oa::I64>(numBlocks)) {
		OaLogError(oa::LogComponent::Ml,
			"QuantizeQ4: expected one scale per 32-value block");
		return {};
	}

	oa::Matrix out = oa::FnMatrix::empty(
		oa::MatrixShape{static_cast<oa::I64>(numBlocks) * Q4BytesPerBlock},
		oa::ScalarType::UInt8);
	if (count32 == 0) return out;
	const auto semantic = ctx.recordOp(
		oa::detail::opRegistry::FnMatrix::quantizeQ4,
		{&inInput, &inScale}, {&out});
	if (not semantic.isOk()) return {};

	struct {
		oa::U32 Count;
	} push{count32};

	oa::BufferAccess access[] = {oa::BufferAccess::Read, oa::BufferAccess::Write, oa::BufferAccess::Read};
	ctx.add( "QuantizeQ4", {&inInput, &out, &inScale}, access,
		&push, sizeof(push), divCeil(numBlocks * Q4WordsPerBlock, QuantizeWorkgroupSize),
		1, 1, oa::detail::opRegistry::FnMatrix::quantizeQ4.name, 0,
		oa::detail::opRegistry::FnMatrix::quantizeQ4.hash, 0, 0,
		semantic.getValue());

	return out;
}

oa::Matrix oa::FnMatrix::dequantizeQ4(
	const oa::Matrix& inInput, const oa::Matrix& inScale, oa::I64 inCount) {
	auto& ctx = oa::ExecutionSession::getActive();
	if (not isAdmittedCount(inCount, "DequantizeQ4")) return {};
	if (inInput.getDtype() != oa::ScalarType::UInt8 or not isFloat32(inScale)) {
		OaLogError(oa::LogComponent::Ml,
			"DequantizeQ4: payload must be UInt8 and scale must be Float32");
		return {};
	}
	const oa::U32 count32 = static_cast<oa::U32>(inCount);
	const oa::U32 numBlocks = divCeil(count32, Q4BlockSize);
	if (inScale.numElements() != static_cast<oa::I64>(numBlocks)
		or inInput.numElements()
			!= static_cast<oa::I64>(numBlocks) * Q4BytesPerBlock)
	{
		OaLogError(oa::LogComponent::Ml,
			"DequantizeQ4: payload/scale shape does not match Count");
		return {};
	}

	oa::Matrix out = oa::FnMatrix::empty(oa::MatrixShape{inCount}, oa::ScalarType::Float32);
	if (count32 == 0) return out;
	const auto semantic = ctx.recordOp(
		oa::detail::opRegistry::FnMatrix::dequantizeQ4,
		{&inInput, &inScale}, {&out},
		{oa::OpAttribute::fromSignedInteger("count", inCount)});
	if (not semantic.isOk()) return {};

	struct {
		oa::U32 Count;
	} push{count32};

	oa::BufferAccess access[] = {oa::BufferAccess::Read, oa::BufferAccess::Write, oa::BufferAccess::Read};
	ctx.add( "DequantizeQ4", {&inInput, &out, &inScale}, access,
		&push, sizeof(push), divCeil(count32, ElementWorkgroupSize),
		1, 1, oa::detail::opRegistry::FnMatrix::dequantizeQ4.name, 0,
		oa::detail::opRegistry::FnMatrix::dequantizeQ4.hash, 0, 0,
		semantic.getValue());

	return out;
}

oa::Matrix oa::FnMatrix::computeScaleQ4(const oa::Matrix& inInput) {
	auto& ctx = oa::ExecutionSession::getActive();
	const oa::I64 count = inInput.numElements();
	if (not isAdmittedCount(count, "ComputeScaleQ4")) return {};
	if (not isFloat32(inInput)) {
		OaLogError(oa::LogComponent::Ml,
			"ComputeScaleQ4: input must use Float32 storage");
		return {};
	}
	const oa::U32 count32 = static_cast<oa::U32>(count);
	const oa::U32 numBlocks = divCeil(count32, Q4BlockSize);

	oa::Matrix scale = oa::FnMatrix::empty(
		oa::MatrixShape{static_cast<oa::I64>(numBlocks)}, oa::ScalarType::Float32);
	if (count32 == 0) return scale;
	const auto semantic = ctx.recordOp(
		oa::detail::opRegistry::FnMatrix::computeScaleQ4,
		{&inInput}, {&scale});
	if (not semantic.isOk()) return {};

	struct {
		oa::U32 Count;
	} push{count32};

	oa::BufferAccess access[] = {oa::BufferAccess::Read, oa::BufferAccess::Write};
	ctx.add( "ComputeScaleQ4", {&inInput, &scale}, access, &push,
		sizeof(push), divCeil(numBlocks, QuantizeWorkgroupSize),
		1, 1, oa::detail::opRegistry::FnMatrix::computeScaleQ4.name, 0,
		oa::detail::opRegistry::FnMatrix::computeScaleQ4.hash, 0, 0,
		semantic.getValue());

	return scale;
}

oa::Matrix oa::FnMatrix::quantizeQ8(const oa::Matrix& inInput, const oa::Matrix& inScale) {
	auto& ctx = oa::ExecutionSession::getActive();
	const oa::I64 count = inInput.numElements();
	if (not isAdmittedCount(count, "QuantizeQ8")) return {};
	if (not isFloat32(inInput) or not isFloat32(inScale)) {
		OaLogError(oa::LogComponent::Ml,
			"QuantizeQ8: input and scale must use Float32 storage");
		return {};
	}
	const oa::U32 count32 = static_cast<oa::U32>(count);
	const oa::U32 numBlocks = divCeil(count32, Q8BlockSize);
	if (inScale.numElements() != static_cast<oa::I64>(numBlocks)) {
		OaLogError(oa::LogComponent::Ml,
			"QuantizeQ8: expected one scale per 32-value block");
		return {};
	}

	oa::Matrix out = oa::FnMatrix::empty(
		oa::MatrixShape{static_cast<oa::I64>(numBlocks) * Q8BytesPerBlock},
		oa::ScalarType::Int8);
	if (count32 == 0) return out;
	const auto semantic = ctx.recordOp(
		oa::detail::opRegistry::FnMatrix::quantizeQ8,
		{&inInput, &inScale}, {&out});
	if (not semantic.isOk()) return {};

	struct {
		oa::U32 Count;
	} push{count32};

	oa::BufferAccess access[] = {
		oa::BufferAccess::Read, oa::BufferAccess::Write, oa::BufferAccess::Read};
	ctx.add( "QuantizeQ8", {&inInput, &out, &inScale}, access,
		&push, sizeof(push), divCeil(numBlocks * Q8WordsPerBlock, QuantizeWorkgroupSize),
		1, 1, oa::detail::opRegistry::FnMatrix::quantizeQ8.name, 0,
		oa::detail::opRegistry::FnMatrix::quantizeQ8.hash, 0, 0,
		semantic.getValue());

	return out;
}

oa::Matrix oa::FnMatrix::dequantizeQ8(
	const oa::Matrix& inInput, const oa::Matrix& inScale, oa::I64 inCount)
{
	auto& ctx = oa::ExecutionSession::getActive();
	if (not isAdmittedCount(inCount, "DequantizeQ8")) return {};
	if (inInput.getDtype() != oa::ScalarType::Int8 or not isFloat32(inScale)) {
		OaLogError(oa::LogComponent::Ml,
			"DequantizeQ8: payload must be Int8 and scale must be Float32");
		return {};
	}
	const oa::U32 count32 = static_cast<oa::U32>(inCount);
	const oa::U32 numBlocks = divCeil(count32, Q8BlockSize);
	if (inScale.numElements() != static_cast<oa::I64>(numBlocks)
		or inInput.numElements()
			!= static_cast<oa::I64>(numBlocks) * Q8BytesPerBlock)
	{
		OaLogError(oa::LogComponent::Ml,
			"DequantizeQ8: payload/scale shape does not match Count");
		return {};
	}

	oa::Matrix out = oa::FnMatrix::empty(oa::MatrixShape{inCount}, oa::ScalarType::Float32);
	if (count32 == 0) return out;
	const auto semantic = ctx.recordOp(
		oa::detail::opRegistry::FnMatrix::dequantizeQ8,
		{&inInput, &inScale}, {&out},
		{oa::OpAttribute::fromSignedInteger("count", inCount)});
	if (not semantic.isOk()) return {};

	struct {
		oa::U32 Count;
	} push{count32};

	oa::BufferAccess access[] = {
		oa::BufferAccess::Read, oa::BufferAccess::Write, oa::BufferAccess::Read};
	ctx.add( "DequantizeQ8", {&inInput, &out, &inScale}, access,
		&push, sizeof(push), divCeil(count32, ElementWorkgroupSize),
		1, 1, oa::detail::opRegistry::FnMatrix::dequantizeQ8.name, 0,
		oa::detail::opRegistry::FnMatrix::dequantizeQ8.hash, 0, 0,
		semantic.getValue());

	return out;
}

oa::Matrix oa::FnMatrix::computeScaleQ8(const oa::Matrix& inInput) {
	auto& ctx = oa::ExecutionSession::getActive();
	const oa::I64 count = inInput.numElements();
	if (not isAdmittedCount(count, "ComputeScaleQ8")) return {};
	if (not isFloat32(inInput)) {
		OaLogError(oa::LogComponent::Ml,
			"ComputeScaleQ8: input must use Float32 storage");
		return {};
	}
	const oa::U32 count32 = static_cast<oa::U32>(count);
	const oa::U32 numBlocks = divCeil(count32, Q8BlockSize);

	oa::Matrix scale = oa::FnMatrix::empty(
		oa::MatrixShape{static_cast<oa::I64>(numBlocks)}, oa::ScalarType::Float32);
	if (count32 == 0) return scale;
	const auto semantic = ctx.recordOp(
		oa::detail::opRegistry::FnMatrix::computeScaleQ8,
		{&inInput}, {&scale});
	if (not semantic.isOk()) return {};

	struct {
		oa::U32 Count;
	} push{count32};

	oa::BufferAccess access[] = {oa::BufferAccess::Read, oa::BufferAccess::Write};
	ctx.add( "ComputeScaleQ8", {&inInput, &scale}, access,
		&push, sizeof(push), divCeil(numBlocks, QuantizeWorkgroupSize),
		1, 1, oa::detail::opRegistry::FnMatrix::computeScaleQ8.name, 0,
		oa::detail::opRegistry::FnMatrix::computeScaleQ8.hash, 0, 0,
		semantic.getValue());

	return scale;
}

oa::Matrix oa::FnMatrix::matMulNtQ4(
	const oa::Matrix& inInput,
	const oa::Matrix& inPayload,
	const oa::Matrix& inScale,
	oa::I64 inOutputFeatures)
{
	return matMulNtQuantized(
		inInput,
		inPayload,
		inScale,
		inOutputFeatures,
		QuantPlaneFormat::Q4,
		oa::detail::opRegistry::FnMatrix::matMulNtQ4,
		"MatMulNtQ4");
}

oa::Matrix oa::FnMatrix::matMulNtQ8(
	const oa::Matrix& inInput,
	const oa::Matrix& inPayload,
	const oa::Matrix& inScale,
	oa::I64 inOutputFeatures)
{
	return matMulNtQuantized(
		inInput,
		inPayload,
		inScale,
		inOutputFeatures,
		QuantPlaneFormat::Q8,
		oa::detail::opRegistry::FnMatrix::matMulNtQ8,
		"MatMulNtQ8");
}

oa::QuantMatrix oa::FnMatrix::quantize(
	const oa::Matrix& inInput,
	oa::Quantization inQuantization)
{
	if (inInput.isEmpty() or inInput.numElements() <= 0) {
		OaLogError(oa::LogComponent::Ml,
			"quantize: input must have non-empty Float32 storage");
		return {};
	}
	if (inInput.getDtype() != oa::ScalarType::Float32) {
		OaLogError(oa::LogComponent::Ml,
			"quantize: only Float32 source storage is admitted");
			return {};
	}

	auto& context = oa::ExecutionSession::getActive();
	oa::OpLoweringScope lowering(context);
	oa::Matrix scale;
	oa::Matrix payload;
	switch (inQuantization) {
		case oa::Quantization::Q4:
			scale = computeScaleQ4(inInput);
			if (not scale.isEmpty()) payload = quantizeQ4(inInput, scale);
			break;
		case oa::Quantization::Q8:
			scale = computeScaleQ8(inInput);
			if (not scale.isEmpty()) payload = quantizeQ8(inInput, scale);
			break;
		default:
			OaLogError(oa::LogComponent::Ml,
				"quantize: unsupported quantization value %u",
				static_cast<unsigned>(inQuantization));
			return {};
	}
	if (payload.isEmpty() or scale.isEmpty()) return {};
	auto output = oa::QuantMatrixAccess::make(
		oa::move(payload), oa::move(scale), inInput.getShape(), inQuantization);
	auto semanticOutput = semanticQuantizedProxy(output);
	const auto status = lowering.commit(
		oa::detail::opRegistry::FnMatrix::quantize,
		{&inInput}, {&semanticOutput}, {quantizationAttribute(inQuantization)});
	if (not status.isOk()) {
		OaLogError(oa::LogComponent::Ml,
			"quantize: semantic lowering failed: %s",
			status.getMessage().cStr());
		return {};
	}
	return output;
}

oa::Matrix oa::FnMatrix::dequantize(const oa::QuantMatrix& inInput) {
	if (inInput.isEmpty() or inInput.numElements() <= 0) {
		OaLogError(oa::LogComponent::Ml,
			"Dequantize: input must contain a non-empty quantized weight");
		return {};
	}
	const auto& payload = oa::QuantMatrixAccess::payload(inInput);
	const auto& scale = oa::QuantMatrixAccess::scale(inInput);
	auto& context = oa::ExecutionSession::getActive();
	oa::OpLoweringScope lowering(context);
	oa::Matrix output;
	switch (inInput.getQuantization()) {
		case oa::Quantization::Q4:
			output = dequantizeQ4(
				payload, scale, inInput.numElements());
			break;
		case oa::Quantization::Q8:
			output = dequantizeQ8(
				payload, scale, inInput.numElements());
			break;
		default:
			OaLogError(oa::LogComponent::Ml,
				"Dequantize: unsupported quantization value %u",
				static_cast<unsigned>(inInput.getQuantization()));
			return {};
	}
	if (output.isEmpty()) return {};
	oa::MatrixAccess::shape(output) = inInput.getShape();
	oa::MatrixAccess::stride(output) = oa::Stride::rowMajor(inInput.getShape());
	auto semanticInput = semanticQuantizedProxy(inInput);
	const auto status = lowering.commit(
		oa::detail::opRegistry::FnMatrix::dequantize,
		{&semanticInput}, {&output},
		{quantizationAttribute(inInput.getQuantization())});
	if (not status.isOk()) {
		OaLogError(oa::LogComponent::Ml,
			"Dequantize: semantic lowering failed: %s",
			status.getMessage().cStr());
		return {};
	}
	return output;
}

oa::Matrix oa::FnMatrix::matMulNt(
	const oa::Matrix& inInput,
	const oa::QuantMatrix& inWeight)
{
	if (inWeight.isEmpty() or inWeight.rank() != 2) {
		OaLogError(oa::LogComponent::Ml,
			"MatMulNt: quantized weight must have rank 2 [N,K]");
		return {};
	}
	if (inInput.rank() < 2
		or inInput.size(-1) != inWeight.size(1))
	{
		OaLogError(oa::LogComponent::Ml,
			"MatMulNt: input last dimension must match quantized weight K");
		return {};
	}
	const auto& payload = oa::QuantMatrixAccess::payload(inWeight);
	const auto& scale = oa::QuantMatrixAccess::scale(inWeight);
	auto& context = oa::ExecutionSession::getActive();
	oa::OpLoweringScope lowering(context);
	oa::Matrix output;
	switch (inWeight.getQuantization()) {
		case oa::Quantization::Q4:
			output = matMulNtQ4(
				inInput, payload, scale, inWeight.size(0));
			break;
		case oa::Quantization::Q8:
			output = matMulNtQ8(
				inInput, payload, scale, inWeight.size(0));
			break;
		default:
			OaLogError(oa::LogComponent::Ml,
				"MatMulNt: unsupported quantization value %u",
				static_cast<unsigned>(inWeight.getQuantization()));
			return {};
	}
	if (output.isEmpty()) return {};
	auto semanticWeight = semanticQuantizedProxy(inWeight);
	const auto status = lowering.commit(
		oa::detail::opRegistry::FnMatrix::matMulNtQuantized,
		{&inInput, &semanticWeight}, {&output},
		{quantizationAttribute(inWeight.getQuantization())});
	if (not status.isOk()) {
		OaLogError(oa::LogComponent::Ml,
			"MatMulNt: semantic lowering failed: %s",
			status.getMessage().cStr());
		return {};
	}
	return output;
}
