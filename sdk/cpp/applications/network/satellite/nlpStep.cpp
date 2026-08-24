#include "nlpStep.h"

#include <oa/core/fnMatrix.h>
#include <oa/ml/autograd.h>
#include <oa/ml/fnLoss.h>
#include <ml/nlpSuite.h>
#include <oa/ml/optim.h>
#include <oa/runtime/executionSession.h>
#include <oa/runtime/executionSession.h>
#include <oa/runtime/engine.h>
#include <oa/runtime/executionStats.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <exception>
#include <limits>

namespace {

constexpr oa::U32 kArgumentMagic = 0x31504c4eU; // "NLP1", little endian.
constexpr oa::U32 kArgumentVersion = 1U;
constexpr oa::U32 kGradientArgumentMagic = 0x31474c4eU; // "NLG1", little endian.
constexpr oa::U32 kGradientArgumentVersion = 1U;
constexpr oa::U32 kResultMagic = 0x3152534eU; // "NSR1", little endian.
constexpr oa::U32 kResultVersion = 1U;
constexpr oa::U32 kGradientResultMagic = 0x3152474eU; // "NGR1", little endian.
constexpr oa::U32 kGradientResultVersion = 1U;
constexpr oa::U32 kPredictedPositions =
	static_cast<oa::U32>(oa::NlpSuiteBatchSize * oa::NlpSuiteContextLength);
constexpr oa::U32 kVocabSize = 256U;

class StepArguments {
public:
	oa::U64 seed = 0;
	oa::U64 stepIndex = 0;
	oa::U32 batchOffset = 0;
	oa::Vec<oa::U32> sampleIndices;
};

void appendU32(oa::Vec<oa::Byte>& out, oa::U32 inValue) {
	for (oa::U32 shift = 0; shift < 32U; shift += 8U) {
		out.pushBack(static_cast<oa::Byte>((inValue >> shift) & 0xffU));
	}
}

void appendU64(oa::Vec<oa::Byte>& out, oa::U64 inValue) {
	for (oa::U32 shift = 0; shift < 64U; shift += 8U) {
		out.pushBack(static_cast<oa::Byte>((inValue >> shift) & 0xffU));
	}
}

void appendF32(oa::Vec<oa::Byte>& out, oa::F32 inValue) {
	oa::U32 bits = 0;
	std::memcpy(&bits, &inValue, sizeof(bits));
	appendU32(out, bits);
}

void appendHash(oa::Vec<oa::Byte>& out, const oa::Array<oa::Byte, 32>& inHash) {
	out.append(inHash.data(), inHash.size());
}

oa::U32 readU32(const oa::Byte* inData) {
	oa::U32 value = 0;
	for (oa::U32 i = 0; i < 4U; ++i) {
		value |= static_cast<oa::U32>(inData[i]) << (i * 8U);
	}
	return value;
}

oa::U64 readU64(const oa::Byte* inData) {
	oa::U64 value = 0;
	for (oa::U32 i = 0; i < 8U; ++i) {
		value |= static_cast<oa::U64>(inData[i]) << (i * 8U);
	}
	return value;
}

oa::F32 readF32(const oa::Byte* inData) {
	const oa::U32 bits = readU32(inData);
	oa::F32 value = 0.0F;
	std::memcpy(&value, &bits, sizeof(value));
	return value;
}

oa::Result<StepArguments> decodeArguments(oa::Span<const oa::Byte> inBytes) {
	constexpr oa::Usize kHeaderBytes = 28U;
	if (inBytes.size() < kHeaderBytes) {
		return oa::Status::error(oa::StatusCode::DataLoss,
			"satellite NLP step: argument header is truncated");
	}
	if (readU32(inBytes.data()) != kArgumentMagic
		or readU32(inBytes.data() + 4U) != kArgumentVersion)
	{
		return oa::Status::error(oa::StatusCode::FailedPrecondition,
			"satellite NLP step: argument version is unsupported");
	}
	StepArguments arguments;
	arguments.seed = readU64(inBytes.data() + 8U);
	arguments.stepIndex = readU64(inBytes.data() + 16U);
	const oa::U32 count = readU32(inBytes.data() + 24U);
	if (count != static_cast<oa::U32>(oa::NlpSuiteBatchSize)
		or inBytes.size() != kHeaderBytes + static_cast<oa::Usize>(count) * 4U)
	{
		return oa::Status::error(oa::StatusCode::DataLoss,
			"satellite NLP step: sample-index payload is invalid");
	}
	arguments.sampleIndices.reserve(count);
	for (oa::U32 i = 0; i < count; ++i) {
		arguments.sampleIndices.pushBack(readU32(
			inBytes.data() + kHeaderBytes + static_cast<oa::Usize>(i) * 4U));
	}
	return arguments;
}

oa::Result<StepArguments> decodeGradientArguments(oa::Span<const oa::Byte> inBytes) {
	constexpr oa::Usize kHeaderBytes = 32U;
	if (inBytes.size() < kHeaderBytes) {
		return oa::Status::error(oa::StatusCode::DataLoss,
			"satellite NLP gradient: argument header is truncated");
	}
	if (readU32(inBytes.data()) != kGradientArgumentMagic
		or readU32(inBytes.data() + 4U) != kGradientArgumentVersion)
	{
		return oa::Status::error(oa::StatusCode::FailedPrecondition,
			"satellite NLP gradient: argument version is unsupported");
	}
	StepArguments arguments;
	arguments.seed = readU64(inBytes.data() + 8U);
	arguments.stepIndex = readU64(inBytes.data() + 16U);
	arguments.batchOffset = readU32(inBytes.data() + 24U);
	const oa::U32 count = readU32(inBytes.data() + 28U);
	const oa::U32 globalBatch = static_cast<oa::U32>(oa::NlpSuiteBatchSize);
	if (count == 0U or arguments.batchOffset >= globalBatch
		or count > globalBatch - arguments.batchOffset
		or inBytes.size() != kHeaderBytes + static_cast<oa::Usize>(count) * 4U)
	{
		return oa::Status::error(oa::StatusCode::DataLoss,
			"satellite NLP gradient: microbatch payload is invalid");
	}
	arguments.sampleIndices.reserve(count);
	for (oa::U32 i = 0; i < count; ++i) {
		arguments.sampleIndices.pushBack(readU32(
			inBytes.data() + kHeaderBytes + static_cast<oa::Usize>(i) * 4U));
	}
	return arguments;
}

oa::Status validateSampleIndices(const StepArguments& inArguments) {
	if (inArguments.seed != oa::NlpSuiteRngSeed
		or inArguments.stepIndex >= static_cast<oa::U64>(oa::NlpSuiteTrainingSteps))
	{
		return oa::Status::error(oa::StatusCode::FailedPrecondition,
			"satellite NLP step: seed or step index is outside the frozen recipe");
	}
	auto expectedBatch = oa::satelliteBuildNlpBatch(inArguments.stepIndex);
	if (expectedBatch.isError()) return expectedBatch.getStatus();
	for (oa::U32 batch = 0;
		batch < static_cast<oa::U32>(inArguments.sampleIndices.size()); ++batch)
	{
		const oa::U32 globalBatch = inArguments.batchOffset + batch;
		if (inArguments.sampleIndices[batch]
			!= expectedBatch->sampleIndices[globalBatch])
		{
			return oa::Status::error(oa::StatusCode::FailedPrecondition,
				"satellite NLP step: sample indices do not match the frozen sampler");
		}
	}
	return oa::Status::ok();
}

oa::Result<oa::Vec<oa::F32>> decodeF32(oa::Span<const oa::Byte> inBytes) {
	if (inBytes.size() % sizeof(oa::F32) != 0U) {
		return oa::Status::error(oa::StatusCode::DataLoss,
			"satellite NLP step: FP32 payload has a partial element");
	}
	oa::Vec<oa::F32> values;
	values.reserve(inBytes.size() / sizeof(oa::F32));
	for (oa::Usize offset = 0; offset < inBytes.size(); offset += sizeof(oa::F32)) {
		values.pushBack(readF32(inBytes.data() + offset));
	}
	return values;
}

oa::Result<oa::Vec<oa::U32>> decodeU32(oa::Span<const oa::Byte> inBytes) {
	if (inBytes.size() % sizeof(oa::U32) != 0U) {
		return oa::Status::error(oa::StatusCode::DataLoss,
			"satellite NLP step: UInt32 payload has a partial element");
	}
	oa::Vec<oa::U32> values;
	values.reserve(inBytes.size() / sizeof(oa::U32));
	for (oa::Usize offset = 0; offset < inBytes.size(); offset += sizeof(oa::U32)) {
		values.pushBack(readU32(inBytes.data() + offset));
	}
	return values;
}

oa::Array<oa::Byte, 32> parameterLayoutHash(
	const oa::Vec<oa::NamedParameter>& inParameters)
{
	oa::Vec<oa::Byte> layout;
	for (const auto& named : inParameters) {
		appendU32(layout, static_cast<oa::U32>(named.path.size()));
		layout.append(reinterpret_cast<const oa::Byte*>(named.path.cStr()),
			named.path.size());
		layout.pushBack(static_cast<oa::Byte>(named.param->data.getDtype()));
		layout.pushBack(static_cast<oa::Byte>(named.param->data.rank()));
		for (oa::I32 dim = 0; dim < named.param->data.rank(); ++dim) {
			appendU64(layout, static_cast<oa::U64>(named.param->data.size(dim)));
		}
	}
	return oa::SatelliteProtocol::stableDigest(oa::Span<const oa::Byte>(
		layout.data(), layout.size()));
}

oa::Status loadParameters(
	oa::Vec<oa::NamedParameter>& inParameters,
	oa::Span<const oa::F32> inFlat)
{
	oa::Usize offset = 0;
	for (auto& named : inParameters) {
		if (named.param == nullptr or named.param->data.getDtype() != oa::ScalarType::Float32) {
			return oa::Status::error(oa::StatusCode::DtypeMismatch,
				"satellite NLP step: canonical model parameter is not FP32");
		}
		const oa::Usize count = static_cast<oa::Usize>(named.param->data.numElements());
		if (offset > inFlat.size() or count > inFlat.size() - offset) {
			return oa::Status::error(oa::StatusCode::ShapeMismatch,
				"satellite NLP step: flat parameter bundle is truncated");
		}
		auto loaded = oa::FnMatrix::fromBytes(
			oa::Span<const oa::Byte>(reinterpret_cast<const oa::Byte*>(
				inFlat.data() + offset), count * sizeof(oa::F32)),
			named.param->data.getShape(), oa::ScalarType::Float32);
		if (loaded.isEmpty()) {
			return oa::Status::error(oa::StatusCode::Internal,
				"satellite NLP step: parameter upload failed");
		}
		loaded.setRequiresGrad(true);
		named.param->data = oa::move(loaded);
		offset += count;
	}
	if (offset != inFlat.size()) {
		return oa::Status::error(oa::StatusCode::ShapeMismatch,
			"satellite NLP step: flat parameter bundle has trailing values");
	}
	return oa::Status::ok();
}

oa::Status copyF32(const oa::Matrix& inMatrix, oa::Vec<oa::F32>& out) {
	if (inMatrix.getDtype() != oa::ScalarType::Float32) {
		return oa::Status::error(oa::StatusCode::DtypeMismatch,
			"satellite NLP step: FP32 readback contract changed");
	}
	const oa::Usize oldSize = out.size();
	const oa::Usize count = static_cast<oa::Usize>(inMatrix.numElements());
	out.resize(oldSize + count);
	return oa::FnMatrix::copyToHost(
		inMatrix, out.data() + oldSize, count * sizeof(oa::F32));
}

oa::Vec<oa::Byte> encodeResult(const oa::SatelliteNlpStepResult& inResult) {
	oa::Vec<oa::Byte> bytes;
	bytes.reserve(256U
		+ (inResult.logitSample.size() + inResult.gradients.size()
			+ inResult.updatedParameters.size()) * sizeof(oa::F32));
	appendU32(bytes, kResultMagic);
	appendU32(bytes, kResultVersion);
	appendU64(bytes, inResult.seed);
	appendU64(bytes, inResult.stepIndex);
	appendU32(bytes, inResult.predictedPositions);
	appendU32(bytes, inResult.vocabSize);
	appendU32(bytes, static_cast<oa::U32>(inResult.gradients.size()));
	appendU32(bytes, static_cast<oa::U32>(inResult.logitSample.size()));
	appendF32(bytes, inResult.loss);
	appendF32(bytes, inResult.logitMin);
	appendF32(bytes, inResult.logitMax);
	appendF32(bytes, inResult.logitMean);
	appendF32(bytes, inResult.logitL2);
	appendHash(bytes, inResult.initialParameterHash);
	appendHash(bytes, inResult.inputHash);
	appendHash(bytes, inResult.targetHash);
	appendHash(bytes, inResult.parameterLayoutHash);
	appendHash(bytes, inResult.updatedParameterHash);
	for (const oa::F32 value : inResult.logitSample) appendF32(bytes, value);
	for (const oa::F32 value : inResult.gradients) appendF32(bytes, value);
	for (const oa::F32 value : inResult.updatedParameters) appendF32(bytes, value);
	return bytes;
}

oa::Vec<oa::Byte> encodeGradientResult(
	const oa::SatelliteNlpGradientResult& inResult)
{
	oa::Vec<oa::Byte> bytes;
	bytes.reserve(192U + inResult.gradients.size() * sizeof(oa::F32));
	appendU32(bytes, kGradientResultMagic);
	appendU32(bytes, kGradientResultVersion);
	appendU64(bytes, inResult.seed);
	appendU64(bytes, inResult.stepIndex);
	appendU32(bytes, inResult.batchOffset);
	appendU32(bytes, inResult.batchCount);
	appendU32(bytes, inResult.predictedPositions);
	appendU32(bytes, static_cast<oa::U32>(inResult.gradients.size()));
	appendF32(bytes, inResult.loss);
	appendHash(bytes, inResult.initialParameterHash);
	appendHash(bytes, inResult.inputHash);
	appendHash(bytes, inResult.targetHash);
	appendHash(bytes, inResult.parameterLayoutHash);
	for (const oa::F32 value : inResult.gradients) appendF32(bytes, value);
	return bytes;
}

oa::Vec<oa::Byte> buildProfile(
	oa::Engine& inEngine,
	oa::StringView inWorkload,
	oa::U32 inBatchOffset,
	oa::U32 inBatchCount)
{
	const auto& stats = inEngine.lastExecutionStats();
	char profile[1024]{};
	const int written = std::snprintf(profile, sizeof(profile),
		"workload=%.*s;batch_offset=%u;batch_count=%u;device=%.*s;driver=%.*s;"
		"vulkan=%.*s;precision=FP32;nodes=%u;dispatches=%u;graphs=%u;"
		"submissions=%u;kernel_selection_coverage=gemm-v1;kernel_selections=%u;"
		"kernel_fallbacks=%u;precision_fallbacks=%u;layout_fallbacks=%u;"
		"naive_fallbacks=%u",
		static_cast<int>(inWorkload.size()), inWorkload.data(),
		inBatchOffset, inBatchCount,
		static_cast<int>(inEngine.deviceName().size()), inEngine.deviceName().data(),
		static_cast<int>(inEngine.driverVersion().size()), inEngine.driverVersion().data(),
		static_cast<int>(inEngine.vulkanApiVersion().size()), inEngine.vulkanApiVersion().data(),
		stats.nodeCount, stats.dispatchCount, stats.graphCount, stats.submissionCount,
		stats.kernelSelectionCount, stats.kernelFallbackCount,
		stats.precisionFallbackCount, stats.layoutFallbackCount,
		stats.naiveFallbackCount);
	oa::Vec<oa::Byte> out;
	if (written > 0) {
		const oa::Usize count = std::min(
			static_cast<oa::Usize>(written), sizeof(profile) - 1U);
		out.append(reinterpret_cast<const oa::Byte*>(profile), count);
	}
	return out;
}

class Cursor {
public:
	explicit Cursor(oa::Span<const oa::Byte> inBytes) : bytes_(inBytes) {}

	[[nodiscard]] oa::Result<oa::U32> u32() {
		if (remaining() < 4U) return truncated();
		const oa::U32 value = readU32(bytes_.data() + offset_);
		offset_ += 4U;
		return value;
	}
	[[nodiscard]] oa::Result<oa::U64> u64() {
		if (remaining() < 8U) return truncated();
		const oa::U64 value = readU64(bytes_.data() + offset_);
		offset_ += 8U;
		return value;
	}
	[[nodiscard]] oa::Result<oa::F32> f32() {
		if (remaining() < 4U) return truncated();
		const oa::F32 value = readF32(bytes_.data() + offset_);
		offset_ += 4U;
		return value;
	}
	[[nodiscard]] oa::Status hash(oa::Array<oa::Byte, 32>& out) {
		if (remaining() < out.size()) return truncated();
		std::copy(bytes_.data() + offset_, bytes_.data() + offset_ + out.size(),
			out.data());
		offset_ += out.size();
		return oa::Status::ok();
	}
	[[nodiscard]] oa::Status f32Vector(oa::U32 inCount, oa::Vec<oa::F32>& out) {
		if (static_cast<oa::Usize>(inCount) > remaining() / sizeof(oa::F32)) {
			return truncated();
		}
		out.reserve(inCount);
		for (oa::U32 i = 0; i < inCount; ++i) {
			out.pushBack(readF32(bytes_.data() + offset_));
			offset_ += sizeof(oa::F32);
		}
		return oa::Status::ok();
	}
	[[nodiscard]] oa::Usize remaining() const { return bytes_.size() - offset_; }

private:
	[[nodiscard]] static oa::Status truncated() {
		return oa::Status::error(oa::StatusCode::DataLoss,
			"satellite NLP step: result payload is truncated");
	}
	oa::Span<const oa::Byte> bytes_;
	oa::Usize offset_ = 0;
};

class BatchExecution {
public:
	oa::Array<oa::Byte, 32> parameterLayoutHash{};
	oa::F32 loss = 0.0F;
	oa::F32 logitMin = 0.0F;
	oa::F32 logitMax = 0.0F;
	oa::F32 logitMean = 0.0F;
	oa::F32 logitL2 = 0.0F;
	oa::Vec<oa::F32> logitSample;
	oa::Vec<oa::F32> gradients;
	oa::Vec<oa::F32> updatedParameters;
	oa::Array<oa::Byte, 32> updatedParameterHash{};
	oa::Event completion;
};

oa::Result<BatchExecution> executeBatch(
	oa::Engine& inEngine,
	const oa::SatelliteNamedRequest& inRequest,
	const StepArguments& inArguments,
	oa::Bool inApplyUpdate,
	oa::Bool inCollectLogits)
{
	if (inEngine.getPrecision() != oa::Precision::FP32) {
		return oa::Status::error(oa::StatusCode::FailedPrecondition,
			"satellite NLP workload: frozen workload requires an FP32 engine");
	}
	OA_RETURN_IF_ERROR(validateSampleIndices(inArguments));
	const oa::U32 batchCount = static_cast<oa::U32>(inArguments.sampleIndices.size());
	const oa::U32 predictedPositions = batchCount
		* static_cast<oa::U32>(oa::NlpSuiteContextLength);
	const auto& parameterObject = inRequest.inputs[0];
	const auto& inputObject = inRequest.inputs[1];
	const auto& targetObject = inRequest.inputs[2];
	if (parameterObject.dtype != oa::ScalarType::Float32
		or parameterObject.shape.size() != 1U
		or parameterObject.shape[0] != oa::SatelliteNlpStepParameterCount
		or inputObject.dtype != oa::ScalarType::UInt32
		or targetObject.dtype != oa::ScalarType::UInt32
		or inputObject.shape.size() != 2U or targetObject.shape.size() != 2U
		or inputObject.shape[0] != batchCount
		or inputObject.shape[1] != oa::NlpSuiteContextLength
		or targetObject.shape[0] != batchCount
		or targetObject.shape[1] != oa::NlpSuiteContextLength
		or inRequest.expectedVersion != parameterObject.version
		or inRequest.expectedHash != parameterObject.hash)
	{
		return oa::Status::error(oa::StatusCode::FailedPrecondition,
			"satellite NLP workload: object shape, dtype, version, or hash is invalid");
	}
	auto parameterValues = decodeF32(parameterObject.data);
	auto inputValues = decodeU32(inputObject.data);
	auto targetValues = decodeU32(targetObject.data);
	if (parameterValues.isError()) return parameterValues.getStatus();
	if (inputValues.isError()) return inputValues.getStatus();
	if (targetValues.isError()) return targetValues.getStatus();
	if (parameterValues->size() != oa::SatelliteNlpStepParameterCount
		or inputValues->size() != predictedPositions
		or targetValues->size() != predictedPositions)
	{
		return oa::Status::error(oa::StatusCode::ShapeMismatch,
			"satellite NLP workload: decoded object sizes are invalid");
	}
	const auto* corpus = reinterpret_cast<const oa::U8*>(oa::NlpSuiteSampler::corpus());
	for (oa::U32 batch = 0; batch < batchCount; ++batch) {
		const oa::U32 start = inArguments.sampleIndices[batch];
		for (oa::U32 position = 0;
			position < static_cast<oa::U32>(oa::NlpSuiteContextLength); ++position)
		{
			const oa::Usize flat = static_cast<oa::Usize>(batch)
				* oa::NlpSuiteContextLength + position;
			if ((*inputValues)[flat] != corpus[start + position]
				or (*targetValues)[flat] != corpus[start + position + 1U])
			{
				return oa::Status::error(oa::StatusCode::FailedPrecondition,
					"satellite NLP workload: input or target differs from the frozen batch");
			}
		}
	}

	auto& context = oa::ExecutionSession::forEngine(inEngine);
	oa::ExecutionSession::RecordingScope recording(context);
	const oa::NlpSuiteRecipe recipe(
		oa::NlpArchitecture::Transformer, oa::NlpTokenizerKind::Byte);
	oa::NlpSuiteModel model(recipe);
	auto namedParameters = model.allNamedParameterPtrs();
	if (model.numParameters() != oa::SatelliteNlpStepParameterCount) {
		context.clear();
		return oa::Status::error(oa::StatusCode::FailedPrecondition,
			"satellite NLP workload: model layout changed without a version bump");
	}
	BatchExecution result;
	result.parameterLayoutHash = parameterLayoutHash(namedParameters);
	context.clear();
	const auto loaded = loadParameters(namedParameters, oa::Span<const oa::F32>(
		parameterValues->data(), parameterValues->size()));
	if (loaded.isError()) {
		context.clear();
		return loaded;
	}
	auto parameters = model.allParameterPtrs();
	oa::UniquePtr<oa::AdamW> optimizer;
	if (inApplyUpdate) {
		optimizer = oa::makeUnique<oa::AdamW>(parameters, recipe.learningRate());
		optimizer->zeroGrad();
	}
	const auto inputs = oa::FnMatrix::fromBytes(
		oa::Span<const oa::Byte>(reinterpret_cast<const oa::Byte*>(inputValues->data()),
			inputValues->size() * sizeof(oa::U32)),
		oa::MatrixShape{batchCount, oa::NlpSuiteContextLength}, oa::ScalarType::UInt32);
	const auto targets = oa::FnMatrix::fromBytes(
		oa::Span<const oa::Byte>(reinterpret_cast<const oa::Byte*>(targetValues->data()),
			targetValues->size() * sizeof(oa::U32)),
		oa::MatrixShape{batchCount, oa::NlpSuiteContextLength}, oa::ScalarType::UInt32);
	oa::GradientTape tape;
	const auto logits = model.forward(inputs);
	const auto loss = oa::FnLoss::crossEntropy(
		logits, targets.reshape(oa::MatrixShape{predictedPositions}));
	const auto backward = tape.tryBackward(loss);
	if (backward.isError()) {
		context.clear();
		return backward;
	}
	if (optimizer) optimizer->step();
	auto submitted = inEngine.submit();
	if (submitted.isError()) {
		context.clear();
		return submitted.getStatus();
	}
	const auto wait = inEngine.wait(*submitted);
	if (wait.isError()) return wait;
	if (not inEngine.ownsEvent(*submitted) or not submitted->isComplete()) {
		return oa::Status::error(oa::StatusCode::Internal,
			"satellite NLP workload: exact completion event provenance failed");
	}
	result.completion = *submitted;
	OA_RETURN_IF_ERROR(oa::FnMatrix::copyToHost(
		loss, &result.loss, sizeof(result.loss)));
	if (not std::isfinite(result.loss)) {
		return oa::Status::error(oa::StatusCode::DataLoss,
			"satellite NLP workload: loss is not finite");
	}
	if (inCollectLogits) {
		oa::Vec<oa::F32> allLogits;
		OA_RETURN_IF_ERROR(copyF32(logits, allLogits));
		if (allLogits.size() != static_cast<oa::Usize>(predictedPositions) * kVocabSize) {
			return oa::Status::error(oa::StatusCode::ShapeMismatch,
				"satellite NLP workload: logits shape changed without a version bump");
		}
		result.logitMin = std::numeric_limits<oa::F32>::infinity();
		result.logitMax = -std::numeric_limits<oa::F32>::infinity();
		oa::F64 sum = 0.0;
		oa::F64 sumSquares = 0.0;
		for (const oa::F32 value : allLogits) {
			if (not std::isfinite(value)) {
				return oa::Status::error(oa::StatusCode::DataLoss,
					"satellite NLP workload: logits contain a non-finite value");
			}
			result.logitMin = std::min(result.logitMin, value);
			result.logitMax = std::max(result.logitMax, value);
			sum += value;
			sumSquares += static_cast<oa::F64>(value) * value;
		}
		result.logitMean = static_cast<oa::F32>(
			sum / static_cast<oa::F64>(allLogits.size()));
		result.logitL2 = static_cast<oa::F32>(std::sqrt(sumSquares));
		result.logitSample.append(allLogits.data(), oa::SatelliteNlpStepLogitSampleCount);
	}
	result.gradients.reserve(oa::SatelliteNlpStepParameterCount);
	result.updatedParameters.reserve(oa::SatelliteNlpStepParameterCount);
	for (auto* parameter : parameters) {
		if (parameter == nullptr or parameter->grad().isEmpty()) {
			return oa::Status::error(oa::StatusCode::DataLoss,
				"satellite NLP workload: a model gradient is missing");
		}
		OA_RETURN_IF_ERROR(copyF32(parameter->grad(), result.gradients));
		if (inApplyUpdate) {
			OA_RETURN_IF_ERROR(copyF32(parameter->data, result.updatedParameters));
		}
	}
	if (result.gradients.size() != oa::SatelliteNlpStepParameterCount
		or (inApplyUpdate
			and result.updatedParameters.size() != oa::SatelliteNlpStepParameterCount))
	{
		return oa::Status::error(oa::StatusCode::DataLoss,
			"satellite NLP workload: parameter result count is invalid");
	}
	for (const oa::F32 value : result.gradients) {
		if (not std::isfinite(value)) {
			return oa::Status::error(oa::StatusCode::DataLoss,
				"satellite NLP workload: gradients contain a non-finite value");
		}
	}
	for (const oa::F32 value : result.updatedParameters) {
		if (not std::isfinite(value)) {
			return oa::Status::error(oa::StatusCode::DataLoss,
				"satellite NLP workload: updated parameters contain a non-finite value");
		}
	}
	if (inApplyUpdate) {
		result.updatedParameterHash = oa::satelliteHashF32(oa::Span<const oa::F32>(
			result.updatedParameters.data(), result.updatedParameters.size()));
	}
	return result;
}

} // namespace

oa::Array<oa::Byte, 32> oa::satelliteHashF32(oa::Span<const oa::F32> inValues) {
	oa::Vec<oa::Byte> bytes;
	bytes.reserve(inValues.size() * sizeof(oa::F32));
	for (const oa::F32 value : inValues) appendF32(bytes, value);
	return oa::SatelliteProtocol::stableDigest(oa::Span<const oa::Byte>(
		bytes.data(), bytes.size()));
}

oa::Array<oa::Byte, 32> oa::satelliteHashU32(oa::Span<const oa::U32> inValues) {
	oa::Vec<oa::Byte> bytes;
	bytes.reserve(inValues.size() * sizeof(oa::U32));
	for (const oa::U32 value : inValues) appendU32(bytes, value);
	return oa::SatelliteProtocol::stableDigest(oa::Span<const oa::Byte>(
		bytes.data(), bytes.size()));
}

oa::Result<oa::SatelliteNlpBatch> oa::satelliteBuildNlpBatch(oa::U64 inStepIndex) {
	if (inStepIndex >= static_cast<oa::U64>(oa::NlpSuiteTrainingSteps)) {
		return oa::Status::error(oa::StatusCode::OutOfRange,
			"satellite NLP batch: step index is outside the frozen recipe");
	}
	const oa::U64 corpusBytes = std::strlen(oa::NlpSuiteSampler::corpus());
	const oa::U64 context = static_cast<oa::U64>(oa::NlpSuiteContextLength);
	if (corpusBytes <= context + 1U) {
		return oa::Status::error(oa::StatusCode::FailedPrecondition,
			"satellite NLP batch: canonical corpus is too short");
	}
	const oa::U64 limit = corpusBytes - context - 1U;
	const oa::U64 cursor = (inStepIndex
		* static_cast<oa::U64>(oa::NlpSuiteBatchSize)) % limit;
	const auto* corpus = reinterpret_cast<const oa::U8*>(oa::NlpSuiteSampler::corpus());
	oa::SatelliteNlpBatch batch;
	batch.stepIndex = inStepIndex;
	batch.sampleIndices.reserve(oa::NlpSuiteBatchSize);
	batch.inputs.reserve(
		static_cast<oa::Usize>(oa::NlpSuiteBatchSize * oa::NlpSuiteContextLength));
	batch.targets.reserve(
		static_cast<oa::Usize>(oa::NlpSuiteBatchSize * oa::NlpSuiteContextLength));
	for (oa::U32 row = 0; row < static_cast<oa::U32>(oa::NlpSuiteBatchSize); ++row) {
		const oa::U32 start = static_cast<oa::U32>(
			(cursor + static_cast<oa::U64>(row) * 7U) % limit);
		batch.sampleIndices.pushBack(start);
		for (oa::U32 position = 0;
			position < static_cast<oa::U32>(oa::NlpSuiteContextLength); ++position)
		{
			batch.inputs.pushBack(corpus[start + position]);
			batch.targets.pushBack(corpus[start + position + 1U]);
		}
	}
	return batch;
}

oa::Array<oa::Byte, 32> oa::satelliteNlpParameterLayoutHash(
	oa::NlpSuiteModel& inModel)
{
	return parameterLayoutHash(inModel.allNamedParameterPtrs());
}

oa::Result<oa::U64> oa::satelliteStartNlpStep(
	oa::SatelliteClientSession& inClient,
	oa::U64 inParameters,
	oa::U64 inTokens,
	oa::U64 inTargets,
	oa::U64 inOutput,
	oa::U64 inSeed,
	oa::U64 inStepIndex,
	oa::Span<const oa::U32> inSampleIndices,
	oa::U64 inExpectedVersion,
	const oa::Array<oa::Byte, 32>& inExpectedHash)
{
	if (inSampleIndices.size() != static_cast<oa::Usize>(oa::NlpSuiteBatchSize)) {
		return oa::Status::invalidArgument(
			"satellite NLP step: sample-index count differs from the frozen batch");
	}
	if (inExpectedVersion == oa::U64Max) {
		return oa::Status::error(oa::StatusCode::OutOfRange,
			"satellite NLP step: result version would overflow");
	}
	oa::Vec<oa::Byte> arguments;
	appendU32(arguments, kArgumentMagic);
	appendU32(arguments, kArgumentVersion);
	appendU64(arguments, inSeed);
	appendU64(arguments, inStepIndex);
	appendU32(arguments, static_cast<oa::U32>(inSampleIndices.size()));
	for (const oa::U32 sample : inSampleIndices) appendU32(arguments, sample);
	const oa::U64 inputs[] = {inParameters, inTokens, inTargets};
	return inClient.startNamed(
		oa::SatelliteNlpStepOperation, inputs, inOutput,
		oa::Span<const oa::Byte>(arguments.data(), arguments.size()),
		inExpectedVersion, inExpectedHash);
}

oa::Result<oa::U64> oa::satelliteStartNlpGradient(
	oa::SatelliteClientSession& inClient,
	oa::U64 inParameters,
	oa::U64 inTokens,
	oa::U64 inTargets,
	oa::U64 inOutput,
	oa::U64 inSeed,
	oa::U64 inStepIndex,
	oa::U32 inBatchOffset,
	oa::Span<const oa::U32> inSampleIndices,
	oa::U64 inExpectedVersion,
	const oa::Array<oa::Byte, 32>& inExpectedHash)
{
	const oa::U32 globalBatch = static_cast<oa::U32>(oa::NlpSuiteBatchSize);
	if (inSampleIndices.size() > static_cast<oa::Usize>(globalBatch)) {
		return oa::Status::invalidArgument(
			"satellite NLP gradient: microbatch exceeds the frozen global batch");
	}
	const oa::U32 count = static_cast<oa::U32>(inSampleIndices.size());
	if (count == 0U or inBatchOffset >= globalBatch
		or count > globalBatch - inBatchOffset)
	{
		return oa::Status::invalidArgument(
			"satellite NLP gradient: microbatch is outside the frozen global batch");
	}
	oa::Vec<oa::Byte> arguments;
	appendU32(arguments, kGradientArgumentMagic);
	appendU32(arguments, kGradientArgumentVersion);
	appendU64(arguments, inSeed);
	appendU64(arguments, inStepIndex);
	appendU32(arguments, inBatchOffset);
	appendU32(arguments, count);
	for (const oa::U32 sample : inSampleIndices) appendU32(arguments, sample);
	const oa::U64 inputs[] = {inParameters, inTokens, inTargets};
	return inClient.startNamed(
		oa::SatelliteNlpGradientOperation, inputs, inOutput,
		oa::Span<const oa::Byte>(arguments.data(), arguments.size()),
		inExpectedVersion, inExpectedHash);
}

oa::Result<oa::SatelliteNamedResult> oa::satelliteExecuteNlpStep(
	oa::Engine& inEngine,
	const oa::SatelliteNamedRequest& inRequest)
{
	if (inRequest.operation != oa::SatelliteNlpStepOperation
		or inRequest.inputs.size() != 3U)
	{
		return oa::Status::invalidArgument(
			"satellite NLP step: request operation or input count is invalid");
	}
	try {
		auto arguments = decodeArguments(inRequest.arguments);
		if (arguments.isError()) return arguments.getStatus();
		auto execution = executeBatch(
			inEngine, inRequest, *arguments, true, true);
		if (execution.isError()) return execution.getStatus();

		oa::SatelliteNlpStepResult result;
		result.seed = arguments->seed;
		result.stepIndex = arguments->stepIndex;
		result.predictedPositions = kPredictedPositions;
		result.vocabSize = kVocabSize;
		result.loss = execution->loss;
		result.logitMin = execution->logitMin;
		result.logitMax = execution->logitMax;
		result.logitMean = execution->logitMean;
		result.logitL2 = execution->logitL2;
		result.initialParameterHash = inRequest.inputs[0].hash;
		result.inputHash = inRequest.inputs[1].hash;
		result.targetHash = inRequest.inputs[2].hash;
		result.parameterLayoutHash = execution->parameterLayoutHash;
		result.updatedParameterHash = execution->updatedParameterHash;
		result.logitSample = oa::move(execution->logitSample);
		result.gradients = oa::move(execution->gradients);
		result.updatedParameters = oa::move(execution->updatedParameters);

		oa::SatelliteNamedResult namedResult;
		namedResult.version = inRequest.expectedVersion + 1U;
		namedResult.dtype = oa::ScalarType::UInt8;
		namedResult.data = encodeResult(result);
		namedResult.shape.pushBack(static_cast<oa::I64>(namedResult.data.size()));
		namedResult.profile = buildProfile(
			inEngine, oa::SatelliteNlpStepOperation, 0U,
			static_cast<oa::U32>(oa::NlpSuiteBatchSize));
		namedResult.completion = execution->completion;
		return namedResult;
	} catch (const std::exception& exception) {
		oa::ExecutionSession::forEngine(inEngine).clear();
		return oa::Status::error(oa::StatusCode::Internal,
			oa::String("satellite NLP step: ") + exception.what());
	}
}
oa::Result<oa::SatelliteNamedResult> oa::satelliteExecuteNlpGradient(
	oa::Engine& inEngine,
	const oa::SatelliteNamedRequest& inRequest)
{
	if (inRequest.operation != oa::SatelliteNlpGradientOperation
		or inRequest.inputs.size() != 3U)
	{
		return oa::Status::invalidArgument(
			"satellite NLP gradient: request operation or input count is invalid");
	}
	try {
		auto arguments = decodeGradientArguments(inRequest.arguments);
		if (arguments.isError()) return arguments.getStatus();
		auto execution = executeBatch(
			inEngine, inRequest, *arguments, false, false);
		if (execution.isError()) return execution.getStatus();

		oa::SatelliteNlpGradientResult result;
		result.seed = arguments->seed;
		result.stepIndex = arguments->stepIndex;
		result.batchOffset = arguments->batchOffset;
		result.batchCount = static_cast<oa::U32>(arguments->sampleIndices.size());
		result.predictedPositions = result.batchCount
			* static_cast<oa::U32>(oa::NlpSuiteContextLength);
		result.loss = execution->loss;
		result.initialParameterHash = inRequest.inputs[0].hash;
		result.inputHash = inRequest.inputs[1].hash;
		result.targetHash = inRequest.inputs[2].hash;
		result.parameterLayoutHash = execution->parameterLayoutHash;
		result.gradients = oa::move(execution->gradients);

		oa::SatelliteNamedResult namedResult;
		namedResult.version = inRequest.expectedVersion;
		namedResult.dtype = oa::ScalarType::UInt8;
		namedResult.data = encodeGradientResult(result);
		namedResult.shape.pushBack(static_cast<oa::I64>(namedResult.data.size()));
		namedResult.profile = buildProfile(
			inEngine, oa::SatelliteNlpGradientOperation,
			arguments->batchOffset, result.batchCount);
		namedResult.completion = execution->completion;
		return namedResult;
	} catch (const std::exception& exception) {
		oa::ExecutionSession::forEngine(inEngine).clear();
		return oa::Status::error(oa::StatusCode::Internal,
			oa::String("satellite NLP gradient: ") + exception.what());
	}
}
oa::Result<oa::SatelliteNamedResult> oa::satelliteExecuteNlpWork(
	oa::Engine& inEngine,
	const oa::SatelliteNamedRequest& inRequest)
{
	if (inRequest.operation == oa::SatelliteNlpStepOperation) {
		return oa::satelliteExecuteNlpStep(inEngine, inRequest);
	}
	if (inRequest.operation == oa::SatelliteNlpGradientOperation) {
		return oa::satelliteExecuteNlpGradient(inEngine, inRequest);
	}
	return oa::Status::invalidArgument(
		"satellite NLP worker: operation is not admitted");
}

oa::Result<oa::SatelliteNlpStepResult> oa::satelliteDecodeNlpStepResult(
	oa::Span<const oa::Byte> inBytes)
{
	Cursor cursor(inBytes);
	auto magic = cursor.u32();
	auto version = cursor.u32();
	if (magic.isError() or version.isError()) {
		return oa::Status::error(oa::StatusCode::DataLoss,
			"satellite NLP step: result header is truncated");
	}
	if (*magic != kResultMagic or *version != kResultVersion) {
		return oa::Status::error(oa::StatusCode::FailedPrecondition,
			"satellite NLP step: result version is unsupported");
	}
	oa::SatelliteNlpStepResult result;
	auto seed = cursor.u64();
	auto step = cursor.u64();
	auto positions = cursor.u32();
	auto vocab = cursor.u32();
	auto parameterCount = cursor.u32();
	auto sampleCount = cursor.u32();
	auto loss = cursor.f32();
	auto minimum = cursor.f32();
	auto maximum = cursor.f32();
	auto mean = cursor.f32();
	auto l2 = cursor.f32();
	if (seed.isError() or step.isError() or positions.isError() or vocab.isError()
		or parameterCount.isError() or sampleCount.isError() or loss.isError()
		or minimum.isError() or maximum.isError() or mean.isError() or l2.isError())
	{
		return oa::Status::error(oa::StatusCode::DataLoss,
			"satellite NLP step: result scalar section is truncated");
	}
	if (*positions != kPredictedPositions or *vocab != kVocabSize
		or *parameterCount != oa::SatelliteNlpStepParameterCount
		or *sampleCount != oa::SatelliteNlpStepLogitSampleCount)
	{
		return oa::Status::error(oa::StatusCode::FailedPrecondition,
			"satellite NLP step: result shape contract changed");
	}
	result.seed = *seed;
	result.stepIndex = *step;
	result.predictedPositions = *positions;
	result.vocabSize = *vocab;
	result.loss = *loss;
	result.logitMin = *minimum;
	result.logitMax = *maximum;
	result.logitMean = *mean;
	result.logitL2 = *l2;
	OA_RETURN_IF_ERROR(cursor.hash(result.initialParameterHash));
	OA_RETURN_IF_ERROR(cursor.hash(result.inputHash));
	OA_RETURN_IF_ERROR(cursor.hash(result.targetHash));
	OA_RETURN_IF_ERROR(cursor.hash(result.parameterLayoutHash));
	OA_RETURN_IF_ERROR(cursor.hash(result.updatedParameterHash));
	OA_RETURN_IF_ERROR(cursor.f32Vector(*sampleCount, result.logitSample));
	OA_RETURN_IF_ERROR(cursor.f32Vector(*parameterCount, result.gradients));
	OA_RETURN_IF_ERROR(cursor.f32Vector(*parameterCount, result.updatedParameters));
	if (cursor.remaining() != 0U) {
		return oa::Status::error(oa::StatusCode::DataLoss,
			"satellite NLP step: result payload has trailing bytes");
	}
	return result;
}

oa::Result<oa::SatelliteNlpGradientResult>
oa::satelliteDecodeNlpGradientResult(oa::Span<const oa::Byte> inBytes)
{
	Cursor cursor(inBytes);
	auto magic = cursor.u32();
	auto version = cursor.u32();
	auto seed = cursor.u64();
	auto step = cursor.u64();
	auto batchOffset = cursor.u32();
	auto batchCount = cursor.u32();
	auto positions = cursor.u32();
	auto parameterCount = cursor.u32();
	auto loss = cursor.f32();
	if (magic.isError() or version.isError() or seed.isError() or step.isError()
		or batchOffset.isError() or batchCount.isError() or positions.isError()
		or parameterCount.isError() or loss.isError())
	{
		return oa::Status::error(oa::StatusCode::DataLoss,
			"satellite NLP gradient: result scalar section is truncated");
	}
	const oa::U32 globalBatch = static_cast<oa::U32>(oa::NlpSuiteBatchSize);
	if (*magic != kGradientResultMagic or *version != kGradientResultVersion
		or *batchCount == 0U or *batchOffset >= globalBatch
		or *batchCount > globalBatch - *batchOffset
		or *positions != *batchCount
			* static_cast<oa::U32>(oa::NlpSuiteContextLength)
		or *parameterCount != oa::SatelliteNlpStepParameterCount)
	{
		return oa::Status::error(oa::StatusCode::FailedPrecondition,
			"satellite NLP gradient: result shape or version contract changed");
	}
	oa::SatelliteNlpGradientResult result;
	result.seed = *seed;
	result.stepIndex = *step;
	result.batchOffset = *batchOffset;
	result.batchCount = *batchCount;
	result.predictedPositions = *positions;
	result.loss = *loss;
	OA_RETURN_IF_ERROR(cursor.hash(result.initialParameterHash));
	OA_RETURN_IF_ERROR(cursor.hash(result.inputHash));
	OA_RETURN_IF_ERROR(cursor.hash(result.targetHash));
	OA_RETURN_IF_ERROR(cursor.hash(result.parameterLayoutHash));
	OA_RETURN_IF_ERROR(cursor.f32Vector(*parameterCount, result.gradients));
	if (cursor.remaining() != 0U) {
		return oa::Status::error(oa::StatusCode::DataLoss,
			"satellite NLP gradient: result payload has trailing bytes");
	}
	return result;
}
