#include "nlpCoordinator.h"

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
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstring>
#include <exception>
#include <limits>
#include <string>
#include <string_view>
#include <thread>

namespace {

constexpr oa::U64 kParameterObject = 1U;
constexpr oa::U64 kInputObject = 2U;
constexpr oa::U64 kTargetObject = 3U;
constexpr oa::U64 kResultObject = 4U;
using CoordinatorClock = std::chrono::steady_clock;

class LocalGradient {
public:
	oa::F32 loss = 0.0F;
	oa::Vector<oa::F32> values;
};

class ContextFailureGuard {
public:
	explicit ContextFailureGuard(oa::ExecutionSession& inContext) : context_(&inContext) {}
	~ContextFailureGuard() {
		if (context_ != nullptr) context_->clear();
	}
	void release() { context_ = nullptr; }

private:
	oa::ExecutionSession* context_ = nullptr;
};

oa::Status submitAndWait(oa::Engine& inEngine) {
	auto submitted = inEngine.submit();
	if (submitted.isError()) return submitted.getStatus();
	OA_RETURN_IF_ERROR(inEngine.wait(*submitted));
	if (not inEngine.ownsEvent(*submitted) or not submitted->isComplete()) {
		return oa::Status::error(oa::StatusCode::Internal,
			"satellite NLP coordinator: exact local completion failed");
	}
	return oa::Status::ok();
}

oa::Status appendF32(const oa::Matrix& inMatrix, oa::Vector<oa::F32>& outValues) {
	if (inMatrix.getDtype() != oa::ScalarType::Float32) {
		return oa::Status::error(oa::StatusCode::DtypeMismatch,
			"satellite NLP coordinator: expected an FP32 matrix");
	}
	const oa::Usize oldSize = outValues.size();
	const oa::Usize count = static_cast<oa::Usize>(inMatrix.numElements());
	outValues.resize(oldSize + count);
	return oa::FnMatrix::copyToHost(
		inMatrix, outValues.data() + oldSize, count * sizeof(oa::F32));
}

oa::Result<oa::Vector<oa::F32>> copyParameters(oa::NlpSuiteModel& inModel) {
	oa::Vector<oa::F32> values;
	values.reserve(oa::SatelliteNlpStepParameterCount);
	for (const auto* parameter : inModel.allParameterPtrs()) {
		if (parameter == nullptr) {
			return oa::Status::error(oa::StatusCode::Internal,
				"satellite NLP coordinator: model parameter is null");
		}
		OA_RETURN_IF_ERROR(appendF32(parameter->data, values));
	}
	if (values.size() != oa::SatelliteNlpStepParameterCount) {
		return oa::Status::error(oa::StatusCode::FailedPrecondition,
			"satellite NLP coordinator: canonical parameter count changed");
	}
	return values;
}

oa::Result<LocalGradient> executeLocalGradient(
	oa::Engine& inEngine,
	oa::NlpSuiteModel& inModel,
	oa::AdamW& inOptimizer,
	oa::Span<const oa::U32> inInputs,
	oa::Span<const oa::U32> inTargets)
{
	if (inInputs.empty() or inInputs.size() != inTargets.size()
		or inInputs.size() % static_cast<oa::Usize>(oa::NlpSuiteContextLength) != 0U)
	{
		return oa::Status::invalidArgument(
			"satellite NLP coordinator: local microbatch shape is invalid");
	}
	const oa::I64 batchCount = static_cast<oa::I64>(
		inInputs.size() / static_cast<oa::Usize>(oa::NlpSuiteContextLength));
	inOptimizer.zeroGrad();
	const auto inputs = oa::FnMatrix::fromBytes(
		oa::Span<const oa::Byte>(reinterpret_cast<const oa::Byte*>(inInputs.data()),
			inInputs.size() * sizeof(oa::U32)),
		oa::MatrixShape{batchCount, oa::NlpSuiteContextLength},
		oa::ScalarType::UInt32);
	const auto targets = oa::FnMatrix::fromBytes(
		oa::Span<const oa::Byte>(reinterpret_cast<const oa::Byte*>(inTargets.data()),
			inTargets.size() * sizeof(oa::U32)),
		oa::MatrixShape{batchCount, oa::NlpSuiteContextLength},
		oa::ScalarType::UInt32);
	oa::GradientTape tape;
	const auto logits = inModel.forward(inputs);
	const auto loss = oa::FnLoss::crossEntropy(logits,
		targets.reshape(oa::MatrixShape{static_cast<oa::I64>(inTargets.size())}));
	OA_RETURN_IF_ERROR(tape.tryBackward(loss));
	OA_RETURN_IF_ERROR(submitAndWait(inEngine));
	LocalGradient result;
	OA_RETURN_IF_ERROR(oa::FnMatrix::copyToHost(
		loss, &result.loss, sizeof(result.loss)));
	if (not std::isfinite(result.loss)) {
		return oa::Status::error(oa::StatusCode::DataLoss,
			"satellite NLP coordinator: local loss is not finite");
	}
	result.values.reserve(oa::SatelliteNlpStepParameterCount);
	for (const auto* parameter : inModel.allParameterPtrs()) {
		if (parameter == nullptr or parameter->grad().isEmpty()) {
			return oa::Status::error(oa::StatusCode::DataLoss,
				"satellite NLP coordinator: local gradient is missing");
		}
		OA_RETURN_IF_ERROR(appendF32(parameter->grad(), result.values));
	}
	if (result.values.size() != oa::SatelliteNlpStepParameterCount) {
		return oa::Status::error(oa::StatusCode::DataLoss,
			"satellite NLP coordinator: local gradient count is invalid");
	}
	for (const oa::F32 value : result.values) {
		if (not std::isfinite(value)) {
			return oa::Status::error(oa::StatusCode::DataLoss,
				"satellite NLP coordinator: local gradient is not finite");
		}
	}
	return result;
}

oa::Status applyGradient(
	oa::Engine& inEngine,
	oa::NlpSuiteModel& inModel,
	oa::AdamW& inOptimizer,
	oa::Span<const oa::F32> inGradient)
{
	oa::Usize offset = 0U;
	for (auto* parameter : inModel.allParameterPtrs()) {
		if (parameter == nullptr) {
			return oa::Status::error(oa::StatusCode::Internal,
				"satellite NLP coordinator: model parameter is null");
		}
		const oa::Usize count = static_cast<oa::Usize>(parameter->data.numElements());
		if (offset > inGradient.size() or count > inGradient.size() - offset) {
			return oa::Status::error(oa::StatusCode::ShapeMismatch,
				"satellite NLP coordinator: reduced gradient is truncated");
		}
		parameter->grad() = oa::FnMatrix::fromBytes(
			oa::Span<const oa::Byte>(reinterpret_cast<const oa::Byte*>(
				inGradient.data() + offset), count * sizeof(oa::F32)),
			parameter->data.getShape(), oa::ScalarType::Float32);
		offset += count;
	}
	if (offset != inGradient.size()) {
		return oa::Status::error(oa::StatusCode::ShapeMismatch,
			"satellite NLP coordinator: reduced gradient has trailing values");
	}
	inOptimizer.step();
	return submitAndWait(inEngine);
}

oa::Status accumulateLocalSelection(
	const oa::Engine& inEngine,
	oa::SatelliteNlpCoordinatorReport& outReport)
{
	const auto& stats = inEngine.lastExecutionStats();
	if (stats.kernelSelectionCount == 0U or stats.kernelFallbackCount != 0U
		or outReport.localKernelSelections
			> oa::U64Max - stats.kernelSelectionCount
		or outReport.localKernelFallbacks
			> oa::U64Max - stats.kernelFallbackCount)
	{
		return oa::Status::error(oa::StatusCode::FailedPrecondition,
			"satellite NLP coordinator: local GEMM selection gate failed");
	}
	outReport.localKernelSelections += stats.kernelSelectionCount;
	outReport.localKernelFallbacks += stats.kernelFallbackCount;
	return oa::Status::ok();
}

oa::Result<oa::F32> executeStandaloneStep(
	oa::Engine& inEngine,
	oa::NlpSuiteModel& inModel,
	oa::AdamW& inOptimizer,
	const oa::SatelliteNlpBatch& inBatch)
{
	inOptimizer.zeroGrad();
	const auto inputs = oa::FnMatrix::fromBytes(
		oa::Span<const oa::Byte>(reinterpret_cast<const oa::Byte*>(inBatch.inputs.data()),
			inBatch.inputs.size() * sizeof(oa::U32)),
		oa::MatrixShape{oa::NlpSuiteBatchSize, oa::NlpSuiteContextLength},
		oa::ScalarType::UInt32);
	const auto targets = oa::FnMatrix::fromBytes(
		oa::Span<const oa::Byte>(reinterpret_cast<const oa::Byte*>(inBatch.targets.data()),
			inBatch.targets.size() * sizeof(oa::U32)),
		oa::MatrixShape{oa::NlpSuiteBatchSize, oa::NlpSuiteContextLength},
		oa::ScalarType::UInt32);
	oa::GradientTape tape;
	const auto logits = inModel.forward(inputs);
	const auto loss = oa::FnLoss::crossEntropy(logits,
		targets.reshape(oa::MatrixShape{
			static_cast<oa::I64>(inBatch.targets.size())}));
	OA_RETURN_IF_ERROR(tape.tryBackward(loss));
	inOptimizer.step();
	OA_RETURN_IF_ERROR(submitAndWait(inEngine));
	oa::F32 lossValue = 0.0F;
	OA_RETURN_IF_ERROR(oa::FnMatrix::copyToHost(
		loss, &lossValue, sizeof(lossValue)));
	if (not std::isfinite(lossValue)) {
		return oa::Status::error(oa::StatusCode::DataLoss,
			"satellite NLP coordinator: standalone loss is not finite");
	}
	return lossValue;
}

oa::Result<oa::U64> profileCounter(
	oa::Span<const oa::Byte> inProfile,
	std::string_view inKey)
{
	const std::string_view profile(
		reinterpret_cast<const char*>(inProfile.data()), inProfile.size());
	const auto position = profile.find(inKey);
	if (position == std::string_view::npos) {
		return oa::Status::error(oa::StatusCode::DataLoss,
			"satellite NLP coordinator: remote profile counter is missing");
	}
	const char* begin = profile.data() + position + inKey.size();
	const char* end = profile.data() + profile.size();
	oa::U64 value = 0U;
	const auto parsed = std::from_chars(begin, end, value);
	if (parsed.ec != std::errc{} or (parsed.ptr != end and *parsed.ptr != ';')) {
		return oa::Status::error(oa::StatusCode::DataLoss,
			"satellite NLP coordinator: remote profile counter is malformed");
	}
	return value;
}

oa::Result<oa::F32> evaluateAccuracy(
	oa::Engine& inEngine,
	oa::NlpSuiteModel& inModel,
	const oa::SatelliteNlpBatch& inBatch)
{
	const auto inputs = oa::FnMatrix::fromBytes(
		oa::Span<const oa::Byte>(reinterpret_cast<const oa::Byte*>(inBatch.inputs.data()),
			inBatch.inputs.size() * sizeof(oa::U32)),
		oa::MatrixShape{oa::NlpSuiteBatchSize, oa::NlpSuiteContextLength},
		oa::ScalarType::UInt32);
	const auto logits = inModel.forward(inputs);
	OA_RETURN_IF_ERROR(submitAndWait(inEngine));
	oa::Vector<oa::F32> host(static_cast<oa::Usize>(logits.numElements()));
	OA_RETURN_IF_ERROR(oa::FnMatrix::copyToHost(
		logits, host.data(), host.size() * sizeof(oa::F32)));
	const oa::Usize rows = inBatch.targets.size();
	if (host.size() != rows * 256U) {
		return oa::Status::error(oa::StatusCode::ShapeMismatch,
			"satellite NLP coordinator: evaluation logits shape is invalid");
	}
	oa::Usize correct = 0U;
	for (oa::Usize row = 0; row < rows; ++row) {
		oa::U32 best = 0U;
		for (oa::U32 column = 1U; column < 256U; ++column) {
			if (host[row * 256U + column] > host[row * 256U + best]) best = column;
		}
		if (best == inBatch.targets[row]) ++correct;
	}
	return static_cast<oa::F32>(correct) / static_cast<oa::F32>(rows);
}

oa::Result<oa::String> generateGreedy(
	oa::Engine& inEngine,
	oa::NlpSuiteModel& inModel,
	const oa::NlpSuiteRecipe& inRecipe)
{
	oa::NlpSuiteSampler sampler(inRecipe, 1);
	const oa::Usize contextLength = static_cast<oa::Usize>(inRecipe.contextLength());
	oa::Vector<oa::I32> context(contextLength, 0);
	const auto prompt = sampler.encode(oa::NlpSuiteGenerationPrompt);
	const oa::Usize copyCount = std::min(prompt.size(), contextLength);
	for (oa::Usize index = 0; index < copyCount; ++index) {
		context[index] = prompt[index];
	}
	oa::Usize filled = std::max<oa::Usize>(copyCount, 1U);
	oa::Usize logitRow = filled - 1U;
	oa::String output(oa::NlpSuiteGenerationPrompt);
	for (oa::I32 index = 0; index < oa::NlpSuiteGenerationSourceUnits; ++index) {
		const auto inputs = sampler.inputMatrix(context);
		const auto logits = inModel.forward(inputs);
		OA_RETURN_IF_ERROR(submitAndWait(inEngine));
		oa::Vector<oa::F32> host(static_cast<oa::Usize>(logits.numElements()));
		OA_RETURN_IF_ERROR(oa::FnMatrix::copyToHost(
			logits, host.data(), host.size() * sizeof(oa::F32)));
		if (host.size() != contextLength * 256U) {
			return oa::Status::error(oa::StatusCode::ShapeMismatch,
				"satellite NLP coordinator: generation logits shape is invalid");
		}
		oa::I32 next = 0;
		for (oa::I32 column = 1; column < 256; ++column) {
			if (host[logitRow * 256U + static_cast<oa::Usize>(column)]
				> host[logitRow * 256U + static_cast<oa::Usize>(next)])
			{
				next = column;
			}
		}
		output += static_cast<char>(static_cast<oa::U8>(next));
		if (filled < contextLength) {
			context[filled++] = next;
			logitRow = filled - 1U;
		} else {
			for (oa::Usize token = 1U; token < contextLength; ++token) {
				context[token - 1U] = context[token];
			}
			context[contextLength - 1U] = next;
			logitRow = contextLength - 1U;
		}
	}
	return output;
}

oa::Bool generationQuality(const oa::String& inGenerated) {
	const oa::Usize promptBytes = std::strlen(oa::NlpSuiteGenerationPrompt);
	if (inGenerated.size() != promptBytes
		+ static_cast<oa::Usize>(oa::NlpSuiteGenerationSourceUnits))
	{
		return false;
	}
	for (oa::Usize index = promptBytes; index < inGenerated.size(); ++index) {
		const auto value = static_cast<unsigned char>(inGenerated[index]);
		if (value != ' ' and (value < 'a' or value > 'z')) return false;
	}
	const std::string generated(inGenerated.data(), inGenerated.size());
	const std::string corpus = oa::NlpSuiteSampler::corpus();
	const std::string prompt = oa::NlpSuiteGenerationPrompt;
	const std::string continuation = generated.substr(promptBytes);
	oa::Usize bestPrefix = 0U;
	for (oa::Usize found = corpus.find(prompt);
		found != std::string::npos; found = corpus.find(prompt, found + 1U))
	{
		const oa::Usize start = found + prompt.size();
		oa::Usize matched = 0U;
		while (matched < continuation.size() and start + matched < corpus.size()
			and continuation[matched] == corpus[start + matched])
		{
			++matched;
		}
		bestPrefix = std::max(bestPrefix, matched);
	}
	constexpr oa::Usize ngram = 8U;
	oa::Usize supported = 0U;
	oa::Usize total = 0U;
	for (oa::Usize index = 0; index + ngram <= generated.size(); ++index) {
		++total;
		if (corpus.find(generated.substr(index, ngram)) != std::string::npos) {
			++supported;
		}
	}
	const oa::F32 coverage = total == 0U ? 0.0F
		: static_cast<oa::F32>(supported) / static_cast<oa::F32>(total);
	return bestPrefix >= 16U and coverage >= 0.90F;
}

oa::Status finalizeReport(
	oa::Engine& inEngine,
	oa::NlpSuiteModel& inModel,
	oa::AdamW& inOptimizer,
	const oa::NlpSuiteRecipe& inRecipe,
	const oa::SatelliteNlpBatch& inLastBatch,
	const oa::SatelliteNlpCoordinatorConfig& inConfig,
	oa::SatelliteNlpCoordinatorReport& outReport)
{
	outReport.optimizerStep = inOptimizer.getStep();
	auto parameterValues = copyParameters(inModel);
	if (parameterValues.isError()) return parameterValues.getStatus();
	outReport.finalParameterHash = oa::satelliteHashF32(oa::Span<const oa::F32>(
		parameterValues->data(), parameterValues->size()));
	auto accuracy = evaluateAccuracy(inEngine, inModel, inLastBatch);
	if (accuracy.isError()) return accuracy.getStatus();
	outReport.finalAccuracy = *accuracy;
	auto generated = generateGreedy(inEngine, inModel, inRecipe);
	if (generated.isError()) return generated.getStatus();
	outReport.generated = *generated;
	outReport.generationQualityPassed = generationQuality(outReport.generated);

	OA_RETURN_IF_ERROR(inModel.save(
		inEngine, inConfig.checkpointPath, inOptimizer));
	auto reloaded = oa::makeUnique<oa::NlpSuiteModel>(inRecipe);
	auto reloadParameters = reloaded->allParameterPtrs();
	auto reloadOptimizer = oa::makeUnique<oa::AdamW>(
		reloadParameters, inRecipe.learningRate());
	OA_RETURN_IF_ERROR(reloaded->load(
		inEngine, inConfig.checkpointPath, *reloadOptimizer));
	auto reloadedValues = copyParameters(*reloaded);
	if (reloadedValues.isError()) return reloadedValues.getStatus();
	const auto reloadedHash = oa::satelliteHashF32(oa::Span<const oa::F32>(
		reloadedValues->data(), reloadedValues->size()));
	if (reloadedHash != outReport.finalParameterHash
		or reloadOptimizer->getStep() != inOptimizer.getStep())
	{
		return oa::Status::error(oa::StatusCode::DataLoss,
			"satellite NLP coordinator: checkpoint state changed on reload");
	}
	auto reloadedGenerated = generateGreedy(inEngine, *reloaded, inRecipe);
	if (reloadedGenerated.isError()) return reloadedGenerated.getStatus();
	if (*reloadedGenerated != outReport.generated) {
		return oa::Status::error(oa::StatusCode::DataLoss,
			"satellite NLP coordinator: checkpoint generation changed on reload");
	}
	outReport.checkpointRoundTrip = true;
	if (outReport.completedSteps != inConfig.steps
		or outReport.optimizerStep != inConfig.steps
		or not std::isfinite(outReport.initialLoss)
		or not std::isfinite(outReport.finalLoss)
		or not std::isfinite(outReport.finalAccuracy)
		or not outReport.checkpointRoundTrip)
	{
		return oa::Status::error(oa::StatusCode::DataLoss,
			"satellite NLP coordinator: completion accounting is invalid");
	}
	if (inConfig.steps == static_cast<oa::U32>(oa::NlpSuiteTrainingSteps)
		and (outReport.finalLoss >= outReport.initialLoss
			or outReport.finalAccuracy < 0.85F
			or not outReport.generationQualityPassed))
	{
		return oa::Status::error(oa::StatusCode::FailedPrecondition,
			"satellite NLP coordinator: 300-step quality gate failed");
	}
	return oa::Status::ok();
}

oa::Status dropStepObjects(oa::SatelliteClientSession& inClient) {
	OA_RETURN_IF_ERROR(inClient.dropObject(kInputObject));
	OA_RETURN_IF_ERROR(inClient.dropObject(kTargetObject));
	return inClient.dropObject(kResultObject);
}

} // namespace

oa::Result<oa::SatelliteNlpCoordinatorReport> oa::satelliteRunSplitBatchNlp(
	oa::Engine& inEngine,
	oa::SatelliteClientSession& inClient,
	const oa::SatelliteNlpCoordinatorConfig& inConfig)
{
	if (inConfig.steps == 0U
		or inConfig.steps > static_cast<oa::U32>(oa::NlpSuiteTrainingSteps)
		or inConfig.localBatchCount == 0U
		or inConfig.localBatchCount >= static_cast<oa::U32>(oa::NlpSuiteBatchSize)
		or inConfig.initialVersion == 0U
		or inConfig.initialVersion > oa::U64Max - inConfig.steps
		or inConfig.checkpointPath.empty())
	{
		return oa::Status::invalidArgument(
			"satellite NLP coordinator: invalid steps, split, version, or checkpoint path");
	}
	if (inEngine.getPrecision() != oa::Precision::FP32) {
		return oa::Status::error(oa::StatusCode::FailedPrecondition,
			"satellite NLP coordinator: frozen workload requires FP32");
	}

	auto& context = oa::ExecutionSession::forEngine(inEngine);
	oa::ExecutionSession::RecordingScope recording(context);
	ContextFailureGuard failureGuard(context);
	try {
		const oa::NlpSuiteRecipe recipe(
			oa::NlpArchitecture::Transformer, oa::NlpTokenizerKind::Byte);
		oa::FnMatrix::setRngSeed(oa::NlpSuiteRngSeed);
		auto model = oa::makeUnique<oa::NlpSuiteModel>(recipe);
		if (model->numParameters() != oa::SatelliteNlpStepParameterCount) {
			context.clear();
			return oa::Status::error(oa::StatusCode::FailedPrecondition,
				"satellite NLP coordinator: canonical parameter count changed");
		}
		auto parameters = model->allParameterPtrs();
		auto optimizer = oa::makeUnique<oa::AdamW>(parameters, recipe.learningRate());
		OA_RETURN_IF_ERROR(submitAndWait(inEngine));
		auto parameterValues = copyParameters(*model);
		if (parameterValues.isError()) return parameterValues.getStatus();

		oa::SatelliteNlpCoordinatorReport report;
		report.parameterLayoutHash = oa::satelliteNlpParameterLayoutHash(*model);
		report.initialParameterHash = oa::satelliteHashF32(oa::Span<const oa::F32>(
			parameterValues->data(), parameterValues->size()));
		oa::U64 parameterVersion = inConfig.initialVersion;
		oa::Array<oa::Byte, 32> parameterHash = report.initialParameterHash;
		const oa::I64 parameterShape[] = {oa::SatelliteNlpStepParameterCount};
		const auto started = CoordinatorClock::now();
		OA_RETURN_IF_ERROR(inClient.putF32Versioned(
			kParameterObject, parameterVersion, parameterShape,
			oa::Span<const oa::F32>(parameterValues->data(), parameterValues->size())));

		oa::SatelliteNlpBatch lastBatch;
		const oa::U32 remoteCount = static_cast<oa::U32>(oa::NlpSuiteBatchSize)
			- inConfig.localBatchCount;
		const oa::Usize localElements = static_cast<oa::Usize>(inConfig.localBatchCount)
			* oa::NlpSuiteContextLength;
		const oa::Usize remoteElements = static_cast<oa::Usize>(remoteCount)
			* oa::NlpSuiteContextLength;
		const oa::F32 localWeight = static_cast<oa::F32>(inConfig.localBatchCount)
			/ static_cast<oa::F32>(oa::NlpSuiteBatchSize);
		const oa::F32 remoteWeight = static_cast<oa::F32>(remoteCount)
			/ static_cast<oa::F32>(oa::NlpSuiteBatchSize);
		const oa::I64 remoteShape[] = {remoteCount, oa::NlpSuiteContextLength};

		for (oa::U32 step = 0U; step < inConfig.steps; ++step) {
			auto batch = oa::satelliteBuildNlpBatch(step);
			if (batch.isError()) return batch.getStatus();
			lastBatch = *batch;
			const auto remoteInputs = oa::Span<const oa::U32>(
				batch->inputs.data() + localElements, remoteElements);
			const auto remoteTargets = oa::Span<const oa::U32>(
				batch->targets.data() + localElements, remoteElements);
			const auto remoteSamples = oa::Span<const oa::U32>(
				batch->sampleIndices.data() + inConfig.localBatchCount, remoteCount);
			OA_RETURN_IF_ERROR(inClient.putU32(
				kInputObject, remoteShape, remoteInputs));
			OA_RETURN_IF_ERROR(inClient.putU32(
				kTargetObject, remoteShape, remoteTargets));
			auto request = oa::satelliteStartNlpGradient(inClient,
				kParameterObject, kInputObject, kTargetObject, kResultObject,
				oa::NlpSuiteRngSeed, step, inConfig.localBatchCount, remoteSamples,
				parameterVersion, parameterHash);
			if (request.isError()) return request.getStatus();

			oa::SatelliteMessage waitedMessage;
			oa::Status remoteWaitStatus = oa::Status::ok();
			std::thread remoteWait;
			if (inConfig.overlapRemote) {
				remoteWait = std::thread([&] {
					try {
						auto waited = inClient.wait(*request);
						if (waited.isError()) {
							remoteWaitStatus = waited.getStatus();
						} else {
							waitedMessage = oa::move(*waited);
						}
					} catch (const std::exception& exception) {
						remoteWaitStatus = oa::Status::error(oa::StatusCode::Internal,
							oa::String("satellite NLP coordinator: remote wait failed: ")
								+ exception.what());
					}
				});
			}
			oa::Optional<LocalGradient> local;
			oa::Status localStatus = oa::Status::ok();
			oa::Status localSelectionStatus = oa::Status::ok();
			try {
				auto localResult = executeLocalGradient(
					inEngine, *model, *optimizer,
					oa::Span<const oa::U32>(batch->inputs.data(), localElements),
					oa::Span<const oa::U32>(batch->targets.data(), localElements));
				if (localResult.isError()) {
					localStatus = localResult.getStatus();
				} else {
					local = oa::move(*localResult);
					localSelectionStatus = accumulateLocalSelection(
						inEngine, report);
				}
			} catch (...) {
				if (remoteWait.joinable()) remoteWait.join();
				throw;
			}
			if (remoteWait.joinable()) {
				remoteWait.join();
			} else {
				auto waited = inClient.wait(*request);
				if (waited.isError()) remoteWaitStatus = waited.getStatus();
				else waitedMessage = oa::move(*waited);
			}
			if (localStatus.isError()) return localStatus;
			if (localSelectionStatus.isError()) return localSelectionStatus;
			if (remoteWaitStatus.isError()) return remoteWaitStatus;
			auto resultBytes = oa::SatelliteClientSession::readBytesResult(
				waitedMessage);
			if (resultBytes.isError()) return resultBytes.getStatus();
			auto remote = oa::satelliteDecodeNlpGradientResult(
				oa::Span<const oa::Byte>(resultBytes->data(), resultBytes->size()));
			if (remote.isError()) return remote.getStatus();
			if (remote->seed != oa::NlpSuiteRngSeed or remote->stepIndex != step
				or remote->batchOffset != inConfig.localBatchCount
				or remote->batchCount != remoteCount
				or remote->predictedPositions
					!= remoteCount * static_cast<oa::U32>(oa::NlpSuiteContextLength)
				or remote->initialParameterHash != parameterHash
				or remote->inputHash != oa::satelliteHashU32(remoteInputs)
				or remote->targetHash != oa::satelliteHashU32(remoteTargets)
				or remote->parameterLayoutHash != report.parameterLayoutHash
				or remote->gradients.size() != local->values.size()
				or not std::isfinite(remote->loss))
			{
				return oa::Status::error(oa::StatusCode::DataLoss,
					"satellite NLP coordinator: remote gradient identity is invalid");
			}
			auto profile = oa::SatelliteClientSession::readProfile(waitedMessage);
			if (profile.isError()) return profile.getStatus();
			const oa::Span<const oa::Byte> profileBytes(
				profile->data(), profile->size());
			auto selections = profileCounter(
				profileBytes, "kernel_selections=");
			auto fallbacks = profileCounter(
				profileBytes, "kernel_fallbacks=");
			if (selections.isError()) return selections.getStatus();
			if (fallbacks.isError()) return fallbacks.getStatus();
			if (*selections == 0U or *fallbacks != 0U
				or report.remoteKernelSelections > oa::U64Max - *selections)
			{
				return oa::Status::error(oa::StatusCode::FailedPrecondition,
					"satellite NLP coordinator: remote GEMM selection gate failed");
			}
			report.remoteKernelSelections += *selections;
			report.remoteKernelFallbacks += *fallbacks;
			OA_RETURN_IF_ERROR(dropStepObjects(inClient));

			oa::Vector<oa::F32> reduced(local->values.size());
			for (oa::Usize index = 0; index < reduced.size(); ++index) {
				reduced[index] = local->values[index] * localWeight
					+ remote->gradients[index] * remoteWeight;
				if (not std::isfinite(reduced[index])) {
					return oa::Status::error(oa::StatusCode::DataLoss,
						"satellite NLP coordinator: reduced gradient is not finite");
				}
			}
			const oa::F32 loss = local->loss * localWeight
				+ remote->loss * remoteWeight;
			if (step == 0U) report.initialLoss = loss;
			report.finalLoss = loss;
			OA_RETURN_IF_ERROR(applyGradient(inEngine, *model, *optimizer,
				oa::Span<const oa::F32>(reduced.data(), reduced.size())));
			parameterValues = copyParameters(*model);
			if (parameterValues.isError()) return parameterValues.getStatus();
			parameterHash = oa::satelliteHashF32(oa::Span<const oa::F32>(
				parameterValues->data(), parameterValues->size()));
			OA_RETURN_IF_ERROR(inClient.dropObject(kParameterObject));
			++parameterVersion;
			OA_RETURN_IF_ERROR(inClient.putF32Versioned(
				kParameterObject, parameterVersion, parameterShape,
				oa::Span<const oa::F32>(
					parameterValues->data(), parameterValues->size())));
			report.completedSteps = step + 1U;
		}

		report.finalParameterHash = parameterHash;
		report.completeWallMs = std::chrono::duration<oa::F64, std::milli>(
			CoordinatorClock::now() - started).count();
		OA_RETURN_IF_ERROR(inClient.dropObject(kParameterObject));
		OA_RETURN_IF_ERROR(finalizeReport(
			inEngine, *model, *optimizer, recipe, lastBatch, inConfig, report));
		failureGuard.release();
		return report;
	} catch (const std::exception& exception) {
		context.clear();
		return oa::Status::error(oa::StatusCode::Internal,
			oa::String("satellite NLP coordinator: ") + exception.what());
	}
}

oa::Result<oa::SatelliteNlpCoordinatorReport> oa::satelliteRunStandaloneNlp(
	oa::Engine& inEngine,
	const oa::SatelliteNlpCoordinatorConfig& inConfig)
{
	if (inConfig.steps == 0U
		or inConfig.steps > static_cast<oa::U32>(oa::NlpSuiteTrainingSteps)
		or inConfig.checkpointPath.empty())
	{
		return oa::Status::invalidArgument(
			"satellite NLP standalone: invalid steps or checkpoint path");
	}
	if (inEngine.getPrecision() != oa::Precision::FP32) {
		return oa::Status::error(oa::StatusCode::FailedPrecondition,
			"satellite NLP standalone: frozen workload requires FP32");
	}

	auto& context = oa::ExecutionSession::forEngine(inEngine);
	oa::ExecutionSession::RecordingScope recording(context);
	ContextFailureGuard failureGuard(context);
	try {
		const oa::NlpSuiteRecipe recipe(
			oa::NlpArchitecture::Transformer, oa::NlpTokenizerKind::Byte);
		oa::FnMatrix::setRngSeed(oa::NlpSuiteRngSeed);
		auto model = oa::makeUnique<oa::NlpSuiteModel>(recipe);
		if (model->numParameters() != oa::SatelliteNlpStepParameterCount) {
			return oa::Status::error(oa::StatusCode::FailedPrecondition,
				"satellite NLP standalone: canonical parameter count changed");
		}
		auto parameters = model->allParameterPtrs();
		auto optimizer = oa::makeUnique<oa::AdamW>(parameters, recipe.learningRate());
		OA_RETURN_IF_ERROR(submitAndWait(inEngine));
		auto initialParameters = copyParameters(*model);
		if (initialParameters.isError()) return initialParameters.getStatus();

		oa::SatelliteNlpCoordinatorReport report;
		report.parameterLayoutHash = oa::satelliteNlpParameterLayoutHash(*model);
		report.initialParameterHash = oa::satelliteHashF32(oa::Span<const oa::F32>(
			initialParameters->data(), initialParameters->size()));
		oa::SatelliteNlpBatch lastBatch;
		const auto started = CoordinatorClock::now();
		for (oa::U32 step = 0U; step < inConfig.steps; ++step) {
			auto batch = oa::satelliteBuildNlpBatch(step);
			if (batch.isError()) return batch.getStatus();
			lastBatch = *batch;
			auto loss = executeStandaloneStep(
				inEngine, *model, *optimizer, *batch);
			if (loss.isError()) return loss.getStatus();
			OA_RETURN_IF_ERROR(accumulateLocalSelection(inEngine, report));
			if (step == 0U) report.initialLoss = *loss;
			report.finalLoss = *loss;
			report.completedSteps = step + 1U;
		}
		report.completeWallMs = std::chrono::duration<oa::F64, std::milli>(
			CoordinatorClock::now() - started).count();
		OA_RETURN_IF_ERROR(finalizeReport(
			inEngine, *model, *optimizer, recipe, lastBatch, inConfig, report));
		failureGuard.release();
		return report;
	} catch (const std::exception& exception) {
		context.clear();
		return oa::Status::error(oa::StatusCode::Internal,
			oa::String("satellite NLP standalone: ") + exception.what());
	}
}
