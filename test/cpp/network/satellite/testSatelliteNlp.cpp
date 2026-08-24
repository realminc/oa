#include <gtest/gtest.h>

#include "../../oaTest.h"
#include <oa/crypto/keccak.h>
#include <oa/ml/autograd.h>
#include <oa/ml/fnLoss.h>
#include <ml/nlpSuite.h>
#include <oa/ml/optim.h>
#include <oa/network/satelliteSession.h>
#include <oa/runtime/executionSession.h>
#include <oa/runtime/executionSession.h>

#include "../../../sdk/cpp/applications/network/satellite/nlpStep.h"
#include "../../../sdk/cpp/applications/network/satellite/nlpCoordinator.h"

#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <thread>

namespace {

oa::Status satelliteMac(
	oa::Span<const oa::Byte> inKey,
	oa::Span<const oa::Byte> inData,
	oa::StringView inCustom,
	oa::Array<oa::Byte, 32>& outMac)
{
	return oa::kmac256(
		inKey.data(), inKey.size(), inData.data(), inData.size(),
		reinterpret_cast<const oa::Byte*>(inCustom.data()), inCustom.size(),
		outMac.data(), outMac.size());
}

oa::SatelliteSecret makeSecret() {
	oa::Array<oa::Byte, 32> bytes{};
	for (oa::Usize i = 0; i < bytes.size(); ++i) {
		bytes[i] = static_cast<oa::Byte>(0x31U + i);
	}
	auto secret = oa::SatelliteSecret::fromBytes(bytes);
	EXPECT_TRUE(secret.isOk());
	return oa::move(*secret);
}

oa::SatelliteSessionConfig makeConfig() {
	oa::SatelliteSessionConfig config(makeSecret(), satelliteMac);
	config.deviceName = oa::String(testEngine().deviceName());
	config.ioTimeoutMs = 10000U;
	config.limits.maxPayloadBytes = 1024U * 1024U;
	config.limits.maxResidentBytes = 4U * 1024U * 1024U;
	config.limits.maxObjects = 8U;
	config.limits.maxInflight = 1U;
	return config;
}

oa::Status copyF32(const oa::Matrix& inMatrix, oa::Vec<oa::F32>& out) {
	if (inMatrix.getDtype() != oa::ScalarType::Float32) {
		return oa::Status::error(oa::StatusCode::DtypeMismatch,
			"satellite NLP oracle: expected FP32 matrix");
	}
	const oa::Usize oldSize = out.size();
	const oa::Usize count = static_cast<oa::Usize>(inMatrix.numElements());
	out.resize(oldSize + count);
	return oa::FnMatrix::copyToHost(
		inMatrix, out.data() + oldSize, count * sizeof(oa::F32));
}

oa::Status loadParameters(
	oa::NlpSuiteModel& inModel,
	oa::Span<const oa::F32> inFlat)
{
	auto named = inModel.allNamedParameterPtrs();
	oa::Usize offset = 0;
	for (auto& entry : named) {
		const oa::Usize count = static_cast<oa::Usize>(entry.param->data.numElements());
		if (count > inFlat.size() - offset) {
			return oa::Status::error(oa::StatusCode::ShapeMismatch,
				"satellite NLP oracle: parameter bundle is truncated");
		}
		auto loaded = oa::FnMatrix::fromBytes(
			oa::Span<const oa::Byte>(reinterpret_cast<const oa::Byte*>(
				inFlat.data() + offset), count * sizeof(oa::F32)),
			entry.param->data.getShape(), oa::ScalarType::Float32);
		if (loaded.isEmpty()) {
			return oa::Status::error(oa::StatusCode::Internal,
				"satellite NLP oracle: parameter upload failed");
		}
		loaded.setRequiresGrad(true);
		entry.param->data = oa::move(loaded);
		offset += count;
	}
	if (offset != inFlat.size()) {
		return oa::Status::error(oa::StatusCode::ShapeMismatch,
			"satellite NLP oracle: parameter bundle has trailing values");
	}
	return oa::Status::ok();
}

void appendU32Le(oa::Vec<oa::Byte>& out, oa::U32 inValue) {
	for (oa::U32 shift = 0; shift < 32U; shift += 8U) {
		out.pushBack(static_cast<oa::Byte>((inValue >> shift) & 0xffU));
	}
}

void appendU64Le(oa::Vec<oa::Byte>& out, oa::U64 inValue) {
	for (oa::U32 shift = 0; shift < 64U; shift += 8U) {
		out.pushBack(static_cast<oa::Byte>((inValue >> shift) & 0xffU));
	}
}

oa::Vec<oa::Byte> encodeF32(oa::Span<const oa::F32> inValues) {
	oa::Vec<oa::Byte> bytes;
	bytes.reserve(inValues.size() * sizeof(oa::F32));
	for (const oa::F32 value : inValues) {
		oa::U32 bits = 0U;
		std::memcpy(&bits, &value, sizeof(bits));
		appendU32Le(bytes, bits);
	}
	return bytes;
}

oa::Vec<oa::Byte> encodeU32(oa::Span<const oa::U32> inValues) {
	oa::Vec<oa::Byte> bytes;
	bytes.reserve(inValues.size() * sizeof(oa::U32));
	for (const oa::U32 value : inValues) appendU32Le(bytes, value);
	return bytes;
}

oa::Vec<oa::Byte> encodeStepArguments(oa::Span<const oa::U32> inSampleIndices) {
	oa::Vec<oa::Byte> bytes;
	appendU32Le(bytes, 0x31504c4eU);
	appendU32Le(bytes, 1U);
	appendU64Le(bytes, oa::NlpSuiteRngSeed);
	appendU64Le(bytes, 0U);
	appendU32Le(bytes, static_cast<oa::U32>(inSampleIndices.size()));
	for (const oa::U32 sample : inSampleIndices) appendU32Le(bytes, sample);
	return bytes;
}

oa::Array<oa::Byte, 32> localParameterLayoutHash(
	const oa::Vec<oa::NamedParameter>& inParameters)
{
	oa::Vec<oa::Byte> layout;
	for (const auto& named : inParameters) {
		appendU32Le(layout, static_cast<oa::U32>(named.path.size()));
		layout.append(reinterpret_cast<const oa::Byte*>(named.path.cStr()),
			named.path.size());
		layout.pushBack(static_cast<oa::Byte>(named.param->data.getDtype()));
		layout.pushBack(static_cast<oa::Byte>(named.param->data.rank()));
		for (oa::I32 dim = 0; dim < named.param->data.rank(); ++dim) {
			appendU64Le(layout, static_cast<oa::U64>(named.param->data.size(dim)));
		}
	}
	return oa::SatelliteProtocol::stableDigest(oa::Span<const oa::Byte>(
		layout.data(), layout.size()));
}

class LocalStepResult {
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
};

oa::Result<LocalStepResult> runLocalOracle(
	oa::Engine& inEngine,
	oa::Span<const oa::F32> inParameters,
	oa::Span<const oa::U32> inInputs,
	oa::Span<const oa::U32> inTargets)
{
	auto& context = oa::ExecutionSession::forEngine(inEngine);
	oa::ExecutionSession::RecordingScope recording(context);
	const oa::NlpSuiteRecipe recipe(
		oa::NlpArchitecture::Transformer, oa::NlpTokenizerKind::Byte);
	oa::NlpSuiteModel model(recipe);
	LocalStepResult result;
	result.parameterLayoutHash = localParameterLayoutHash(
		model.allNamedParameterPtrs());
	context.clear();
	OA_RETURN_IF_ERROR(loadParameters(model, inParameters));
	auto parameters = model.allParameterPtrs();
	oa::AdamW optimizer(parameters, recipe.learningRate());
	optimizer.zeroGrad();
	const auto inputs = oa::FnMatrix::fromBytes(
		oa::Span<const oa::Byte>(reinterpret_cast<const oa::Byte*>(inInputs.data()),
			inInputs.size() * sizeof(oa::U32)),
		oa::MatrixShape{oa::NlpSuiteBatchSize, oa::NlpSuiteContextLength},
		oa::ScalarType::UInt32);
	const auto targets = oa::FnMatrix::fromBytes(
		oa::Span<const oa::Byte>(reinterpret_cast<const oa::Byte*>(inTargets.data()),
			inTargets.size() * sizeof(oa::U32)),
		oa::MatrixShape{oa::NlpSuiteBatchSize, oa::NlpSuiteContextLength},
		oa::ScalarType::UInt32);
	oa::GradientTape tape;
	const auto logits = model.forward(inputs);
	const auto loss = oa::FnLoss::crossEntropy(
		logits, targets.reshape(oa::MatrixShape{
			oa::NlpSuiteBatchSize * oa::NlpSuiteContextLength}));
	OA_RETURN_IF_ERROR(tape.tryBackward(loss));
	optimizer.step();
	auto submitted = inEngine.submit();
	if (submitted.isError()) {
		context.clear();
		return submitted.getStatus();
	}
	OA_RETURN_IF_ERROR(inEngine.wait(*submitted));
	if (not inEngine.ownsEvent(*submitted) or not submitted->isComplete()) {
		return oa::Status::error(oa::StatusCode::Internal,
			"satellite NLP oracle: completion provenance failed");
	}

	OA_RETURN_IF_ERROR(oa::FnMatrix::copyToHost(loss, &result.loss, sizeof(result.loss)));
	oa::Vec<oa::F32> allLogits;
	OA_RETURN_IF_ERROR(copyF32(logits, allLogits));
	result.logitMin = std::numeric_limits<oa::F32>::infinity();
	result.logitMax = -std::numeric_limits<oa::F32>::infinity();
	oa::F64 sum = 0.0;
	oa::F64 sumSquares = 0.0;
	for (const oa::F32 value : allLogits) {
		if (not std::isfinite(value)) {
			return oa::Status::error(oa::StatusCode::DataLoss,
				"satellite NLP oracle: non-finite logit");
		}
		result.logitMin = std::min(result.logitMin, value);
		result.logitMax = std::max(result.logitMax, value);
		sum += value;
		sumSquares += static_cast<oa::F64>(value) * value;
	}
	result.logitMean = static_cast<oa::F32>(
		sum / static_cast<oa::F64>(allLogits.size()));
	result.logitL2 = static_cast<oa::F32>(std::sqrt(sumSquares));
	result.logitSample.append(
		allLogits.data(), oa::SatelliteNlpStepLogitSampleCount);
	for (auto* parameter : parameters) {
		OA_RETURN_IF_ERROR(copyF32(parameter->grad(), result.gradients));
		OA_RETURN_IF_ERROR(copyF32(parameter->data, result.updatedParameters));
	}
	return result;
}

oa::Result<LocalStepResult> runLocalGradient(
	oa::Engine& inEngine,
	oa::Span<const oa::F32> inParameters,
	oa::Span<const oa::U32> inInputs,
	oa::Span<const oa::U32> inTargets)
{
	if (inInputs.size() == 0U or inInputs.size() != inTargets.size()
		or inInputs.size() % static_cast<oa::Usize>(oa::NlpSuiteContextLength) != 0U)
	{
		return oa::Status::invalidArgument(
			"satellite NLP oracle: invalid microbatch shape");
	}
	const oa::I64 batchCount = static_cast<oa::I64>(
		inInputs.size() / static_cast<oa::Usize>(oa::NlpSuiteContextLength));
	auto& context = oa::ExecutionSession::forEngine(inEngine);
	oa::ExecutionSession::RecordingScope recording(context);
	const oa::NlpSuiteRecipe recipe(
		oa::NlpArchitecture::Transformer, oa::NlpTokenizerKind::Byte);
	oa::NlpSuiteModel model(recipe);
	LocalStepResult result;
	result.parameterLayoutHash = localParameterLayoutHash(
		model.allNamedParameterPtrs());
	context.clear();
	OA_RETURN_IF_ERROR(loadParameters(model, inParameters));
	auto parameters = model.allParameterPtrs();
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
	const auto logits = model.forward(inputs);
	const auto loss = oa::FnLoss::crossEntropy(logits,
		targets.reshape(oa::MatrixShape{static_cast<oa::I64>(inTargets.size())}));
	OA_RETURN_IF_ERROR(tape.tryBackward(loss));
	auto submitted = inEngine.submit();
	if (submitted.isError()) {
		context.clear();
		return submitted.getStatus();
	}
	OA_RETURN_IF_ERROR(inEngine.wait(*submitted));
	OA_RETURN_IF_ERROR(oa::FnMatrix::copyToHost(
		loss, &result.loss, sizeof(result.loss)));
	for (auto* parameter : parameters) {
		OA_RETURN_IF_ERROR(copyF32(parameter->grad(), result.gradients));
	}
	return result;
}

oa::Result<oa::Vec<oa::F32>> applyAuthoritativeGradient(
	oa::Engine& inEngine,
	oa::Span<const oa::F32> inParameters,
	oa::Span<const oa::F32> inGradient)
{
	auto& context = oa::ExecutionSession::forEngine(inEngine);
	oa::ExecutionSession::RecordingScope recording(context);
	const oa::NlpSuiteRecipe recipe(
		oa::NlpArchitecture::Transformer, oa::NlpTokenizerKind::Byte);
	oa::NlpSuiteModel model(recipe);
	context.clear();
	OA_RETURN_IF_ERROR(loadParameters(model, inParameters));
	auto parameters = model.allParameterPtrs();
	oa::Usize offset = 0U;
	for (auto* parameter : parameters) {
		const oa::Usize count = static_cast<oa::Usize>(parameter->data.numElements());
		if (offset > inGradient.size() or count > inGradient.size() - offset) {
			context.clear();
			return oa::Status::error(oa::StatusCode::ShapeMismatch,
				"satellite NLP oracle: reduced gradient is truncated");
		}
		parameter->grad() = oa::FnMatrix::fromBytes(
			oa::Span<const oa::Byte>(reinterpret_cast<const oa::Byte*>(
				inGradient.data() + offset), count * sizeof(oa::F32)),
			parameter->data.getShape(), oa::ScalarType::Float32);
		offset += count;
	}
	if (offset != inGradient.size()) {
		context.clear();
		return oa::Status::error(oa::StatusCode::ShapeMismatch,
			"satellite NLP oracle: reduced gradient has trailing values");
	}
	oa::AdamW optimizer(parameters, recipe.learningRate());
	optimizer.step();
	auto submitted = inEngine.submit();
	if (submitted.isError()) {
		context.clear();
		return submitted.getStatus();
	}
	OA_RETURN_IF_ERROR(inEngine.wait(*submitted));
	oa::Vec<oa::F32> updated;
	updated.reserve(oa::SatelliteNlpStepParameterCount);
	for (auto* parameter : parameters) {
		OA_RETURN_IF_ERROR(copyF32(parameter->data, updated));
	}
	return updated;
}

oa::Status buildFrozenInputs(
	oa::Engine& inEngine,
	oa::Vec<oa::F32>& outParameters,
	oa::Vec<oa::U32>& outInputs,
	oa::Vec<oa::U32>& outTargets)
{
	auto& context = oa::ExecutionSession::forEngine(inEngine);
	oa::ExecutionSession::RecordingScope recording(context);
	oa::FnMatrix::setRngSeed(oa::NlpSuiteRngSeed);
	const oa::NlpSuiteRecipe recipe(
		oa::NlpArchitecture::Transformer, oa::NlpTokenizerKind::Byte);
	oa::NlpSuiteModel model(recipe);
	auto initialized = inEngine.submit();
	if (initialized.isError()) {
		context.clear();
		return initialized.getStatus();
	}
	OA_RETURN_IF_ERROR(inEngine.wait(*initialized));
	for (auto* parameter : model.allParameterPtrs()) {
		OA_RETURN_IF_ERROR(copyF32(parameter->data, outParameters));
	}
	if (outParameters.size() != oa::SatelliteNlpStepParameterCount) {
		return oa::Status::error(oa::StatusCode::FailedPrecondition,
			"satellite NLP oracle: canonical parameter count changed");
	}
	oa::NlpSuiteSampler sampler(recipe, oa::NlpSuiteBatchSize);
	oa::Matrix input;
	oa::Matrix target;
	sampler.next(input, target);
	outInputs.resize(static_cast<oa::Usize>(input.numElements()));
	outTargets.resize(static_cast<oa::Usize>(target.numElements()));
	OA_RETURN_IF_ERROR(oa::FnMatrix::copyToHost(
		input, outInputs.data(), outInputs.size() * sizeof(oa::U32)));
	OA_RETURN_IF_ERROR(oa::FnMatrix::copyToHost(
		target, outTargets.data(), outTargets.size() * sizeof(oa::U32)));
	return oa::Status::ok();
}

oa::Bool near(oa::F32 inA, oa::F32 inB, oa::F32 inAbs = 1e-5F, oa::F32 inRel = 1e-5F) {
	return std::abs(inA - inB) <= inAbs
		+ inRel * std::max(std::abs(inA), std::abs(inB));
}

void expectVectorNear(
	oa::Span<const oa::F32> inActual,
	oa::Span<const oa::F32> inExpected,
	oa::F32 inAbs,
	oa::F32 inRel)
{
	ASSERT_EQ(inActual.size(), inExpected.size());
	for (oa::Usize i = 0; i < inActual.size(); ++i) {
		ASSERT_TRUE(near(inActual[i], inExpected[i], inAbs, inRel))
			<< "mismatch at element " << i << ": " << inActual[i]
			<< " vs " << inExpected[i];
	}
}

oa::Bool hashIsZero(const oa::Array<oa::Byte, 32>& inHash) {
	for (const oa::Byte value : inHash) {
		if (value != 0U) return false;
	}
	return true;
}

} // namespace

TEST_VK(VkEngineTestFixture, SatelliteNlpReplicatedStepMatchesLocalOracle) {
	auto& engine = testEngine();
	oa::Vec<oa::F32> initialParameters;
	oa::Vec<oa::U32> inputs;
	oa::Vec<oa::U32> targets;
	ASSERT_TRUE(buildFrozenInputs(
		engine, initialParameters, inputs, targets).isOk());
	const oa::I64 parameterShape[] = {oa::SatelliteNlpStepParameterCount};
	const oa::I64 batchShape[] = {oa::NlpSuiteBatchSize, oa::NlpSuiteContextLength};
	const oa::Span<const oa::F32> initialParameterView(
		initialParameters.data(), initialParameters.size());
	const oa::Span<const oa::U32> inputView(inputs.data(), inputs.size());
	const oa::Span<const oa::U32> targetView(targets.data(), targets.size());
	const auto initialHash = oa::satelliteHashF32(initialParameterView);
	constexpr oa::U64 initialVersion = 7U;
	oa::Array<oa::U32, oa::NlpSuiteBatchSize> sampleIndices{};
	const oa::U32 limit = static_cast<oa::U32>(
		std::strlen(oa::NlpSuiteSampler::corpus()) - oa::NlpSuiteContextLength - 1U);
	for (oa::U32 i = 0; i < sampleIndices.size(); ++i) {
		sampleIndices[i] = (i * 7U) % limit;
	}

	// The workload boundary independently rejects a batch that does not match
	// the declared canonical sample indices, before recording GPU work.
	auto corruptedInputs = inputs;
	corruptedInputs[0] ^= 1U;
	auto parameterBytes = encodeF32(initialParameterView);
	auto inputBytes = encodeU32(oa::Span<const oa::U32>(
		corruptedInputs.data(), corruptedInputs.size()));
	auto targetBytes = encodeU32(targetView);
	auto arguments = encodeStepArguments(sampleIndices);
	oa::SatelliteNamedRequest invalidBatch;
	invalidBatch.operation = oa::String(oa::SatelliteNlpStepOperation);
	invalidBatch.arguments = oa::Span<const oa::Byte>(arguments.data(), arguments.size());
	invalidBatch.expectedVersion = 0U;
	invalidBatch.expectedHash = initialHash;
	oa::SatelliteObjectView parameterObject;
	parameterObject.id = 1U;
	parameterObject.dtype = oa::ScalarType::Float32;
	parameterObject.shape = parameterShape;
	parameterObject.data = oa::Span<const oa::Byte>(
		parameterBytes.data(), parameterBytes.size());
	parameterObject.hash = initialHash;
	oa::SatelliteObjectView inputObject;
	inputObject.id = 2U;
	inputObject.dtype = oa::ScalarType::UInt32;
	inputObject.shape = batchShape;
	inputObject.data = oa::Span<const oa::Byte>(inputBytes.data(), inputBytes.size());
	inputObject.hash = oa::satelliteHashU32(oa::Span<const oa::U32>(
		corruptedInputs.data(), corruptedInputs.size()));
	oa::SatelliteObjectView targetObject;
	targetObject.id = 3U;
	targetObject.dtype = oa::ScalarType::UInt32;
	targetObject.shape = batchShape;
	targetObject.data = oa::Span<const oa::Byte>(targetBytes.data(), targetBytes.size());
	targetObject.hash = oa::satelliteHashU32(targetView);
	invalidBatch.inputs.pushBack(parameterObject);
	invalidBatch.inputs.pushBack(inputObject);
	invalidBatch.inputs.pushBack(targetObject);
	auto invalidResult = oa::satelliteExecuteNlpStep(engine, invalidBatch);
	ASSERT_TRUE(invalidResult.isError());
	EXPECT_EQ(invalidResult.getStatus().getCode(), oa::StatusCode::FailedPrecondition);
	EXPECT_EQ(oa::ExecutionSession::forEngine(engine).nodeCount(), 0U);

	auto local = runLocalOracle(
		engine,
		oa::Span<const oa::F32>(initialParameters.data(), initialParameters.size()),
		oa::Span<const oa::U32>(inputs.data(), inputs.size()),
		oa::Span<const oa::U32>(targets.data(), targets.size()));
	ASSERT_TRUE(local.isOk()) << local.getStatus().toString();

	auto listenerResult = oa::TcpListener::bind("127.0.0.1", 0U, 8);
	ASSERT_TRUE(listenerResult.isOk());
	auto listener = oa::move(*listenerResult);
	std::atomic<oa::U32> handlerCalls{0U};
	auto serverConfig = makeConfig();
	serverConfig.namedOperations.pushBack(oa::String(oa::SatelliteNlpStepOperation));
	serverConfig.namedOperations.pushBack(oa::String(oa::SatelliteNlpGradientOperation));
	serverConfig.namedWork = [&](oa::Engine& inEngine,
		const oa::SatelliteNamedRequest& inRequest) {
		++handlerCalls;
		return oa::satelliteExecuteNlpWork(inEngine, inRequest);
	};
	oa::SatelliteServerSession serverSession(engine, oa::move(serverConfig));
	oa::Status serverStatus;
	std::thread server([&] {
		auto accepted = listener.accept();
		ASSERT_TRUE(accepted.isOk());
		serverStatus = serverSession.serve(oa::move(*accepted));
	});

	auto connected = oa::SatelliteClientSession::connect(
		"127.0.0.1", listener.port(), makeConfig());
	ASSERT_TRUE(connected.isOk()) << connected.getStatus().toString();
	auto client = oa::move(*connected);
	ASSERT_TRUE(client.putF32Versioned(
		1U, initialVersion, parameterShape, initialParameterView).isOk());
	ASSERT_TRUE(client.putU32(2U, batchShape, inputView).isOk());
	ASSERT_TRUE(client.putU32(3U, batchShape, targetView).isOk());
	auto overflowVersion = oa::satelliteStartNlpStep(client,
		1U, 2U, 3U, 4U, oa::NlpSuiteRngSeed, 0U, sampleIndices,
		oa::U64Max, initialHash);
	ASSERT_TRUE(overflowVersion.isError());
	EXPECT_EQ(overflowVersion.getStatus().getCode(), oa::StatusCode::OutOfRange);
	EXPECT_EQ(handlerCalls.load(), 0U);
	auto wrongHash = initialHash;
	wrongHash[0] ^= 0x01U;
	auto rejected = oa::satelliteStartNlpStep(client,
		1U, 2U, 3U, 4U, oa::NlpSuiteRngSeed, 0U, sampleIndices,
		initialVersion, wrongHash);
	ASSERT_TRUE(rejected.isError());
	EXPECT_EQ(rejected.getStatus().getCode(), oa::StatusCode::FailedPrecondition);
	EXPECT_EQ(handlerCalls.load(), 0U);
	EXPECT_EQ(client.dropObject(4U).getCode(), oa::StatusCode::NotFound);

	// A pre-submission cancellation must not invoke the workload or create its
	// destination. The same destination is then reusable for the proven step.
	auto cancelled = oa::satelliteStartNlpStep(client,
		1U, 2U, 3U, 4U, oa::NlpSuiteRngSeed, 0U, sampleIndices,
		initialVersion, initialHash);
	ASSERT_TRUE(cancelled.isOk());
	ASSERT_TRUE(client.cancel(*cancelled).isOk());
	EXPECT_EQ(handlerCalls.load(), 0U);
	EXPECT_EQ(client.dropObject(4U).getCode(), oa::StatusCode::NotFound);

	auto request = oa::satelliteStartNlpStep(client,
		1U, 2U, 3U, 4U, oa::NlpSuiteRngSeed, 0U, sampleIndices,
		initialVersion, initialHash);
	ASSERT_TRUE(request.isOk()) << request.getStatus().toString();
	auto waited = client.wait(*request);
	ASSERT_TRUE(waited.isOk()) << waited.getStatus().toString();
	auto bytes = oa::SatelliteClientSession::readBytesResult(*waited);
	ASSERT_TRUE(bytes.isOk()) << bytes.getStatus().toString();
	auto remote = oa::satelliteDecodeNlpStepResult(oa::Span<const oa::Byte>(
		bytes->data(), bytes->size()));
	ASSERT_TRUE(remote.isOk()) << remote.getStatus().toString();
	auto truncated = oa::satelliteDecodeNlpStepResult(oa::Span<const oa::Byte>(
		bytes->data(), bytes->size() - 1U));
	ASSERT_TRUE(truncated.isError());
	EXPECT_EQ(truncated.getStatus().getCode(), oa::StatusCode::DataLoss);
	auto trailingBytes = *bytes;
	trailingBytes.pushBack(0U);
	auto trailing = oa::satelliteDecodeNlpStepResult(oa::Span<const oa::Byte>(
		trailingBytes.data(), trailingBytes.size()));
	ASSERT_TRUE(trailing.isError());
	EXPECT_EQ(trailing.getStatus().getCode(), oa::StatusCode::DataLoss);
	auto wrongVersionBytes = *bytes;
	wrongVersionBytes[4] ^= 1U;
	auto wrongVersion = oa::satelliteDecodeNlpStepResult(oa::Span<const oa::Byte>(
		wrongVersionBytes.data(), wrongVersionBytes.size()));
	ASSERT_TRUE(wrongVersion.isError());
	EXPECT_EQ(wrongVersion.getStatus().getCode(), oa::StatusCode::FailedPrecondition);
	EXPECT_EQ(handlerCalls.load(), 1U);
	EXPECT_EQ(remote->seed, oa::NlpSuiteRngSeed);
	EXPECT_EQ(remote->stepIndex, 0U);
	EXPECT_EQ(remote->predictedPositions,
		static_cast<oa::U32>(oa::NlpSuiteBatchSize * oa::NlpSuiteContextLength));
	EXPECT_EQ(remote->vocabSize, 256U);
	EXPECT_EQ(remote->initialParameterHash, initialHash);
	EXPECT_EQ(remote->inputHash, oa::satelliteHashU32(inputView));
	EXPECT_EQ(remote->targetHash, oa::satelliteHashU32(targetView));
	EXPECT_FALSE(hashIsZero(remote->parameterLayoutHash));
	EXPECT_EQ(remote->parameterLayoutHash, local->parameterLayoutHash);
	EXPECT_EQ(remote->updatedParameterHash,
		oa::satelliteHashF32(oa::Span<const oa::F32>(
			local->updatedParameters.data(), local->updatedParameters.size())));
	EXPECT_TRUE(near(remote->loss, local->loss));
	EXPECT_TRUE(near(remote->logitMin, local->logitMin));
	EXPECT_TRUE(near(remote->logitMax, local->logitMax));
	EXPECT_TRUE(near(remote->logitMean, local->logitMean));
	EXPECT_TRUE(near(remote->logitL2, local->logitL2, 1e-4F, 1e-5F));
	expectVectorNear(
		oa::Span<const oa::F32>(remote->logitSample.data(), remote->logitSample.size()),
		oa::Span<const oa::F32>(local->logitSample.data(), local->logitSample.size()),
		1e-5F, 1e-5F);
	expectVectorNear(
		oa::Span<const oa::F32>(remote->gradients.data(), remote->gradients.size()),
		oa::Span<const oa::F32>(local->gradients.data(), local->gradients.size()),
		1e-5F, 1e-5F);
	expectVectorNear(
		oa::Span<const oa::F32>(remote->updatedParameters.data(),
			remote->updatedParameters.size()),
		oa::Span<const oa::F32>(local->updatedParameters.data(),
			local->updatedParameters.size()),
		1e-6F, 1e-6F);

	auto profile = oa::SatelliteClientSession::readProfile(*waited);
	ASSERT_TRUE(profile.isOk());
	const oa::String profileText(
		reinterpret_cast<const char*>(profile->data()), profile->size());
	EXPECT_NE(profileText.view().stdView().find("workload=nlp-byte-transformer-step-v1"),
		std::string::npos);
	EXPECT_NE(profileText.view().stdView().find("kernel_selection_coverage=gemm-v1"),
		std::string::npos);
	EXPECT_EQ(profileText.view().stdView().find("kernel_selections=0"),
		std::string::npos);
	EXPECT_NE(profileText.view().stdView().find("kernel_fallbacks=0"),
		std::string::npos);
	EXPECT_NE(profileText.view().stdView().find("precision_fallbacks=0"),
		std::string::npos);
	EXPECT_NE(profileText.view().stdView().find("layout_fallbacks=0"),
		std::string::npos);
	EXPECT_NE(profileText.view().stdView().find("naive_fallbacks=0"),
		std::string::npos);
	auto retained = client.getResult(*request);
	ASSERT_TRUE(retained.isOk());
	auto retainedBytes = oa::SatelliteClientSession::readBytesResult(*retained);
	ASSERT_TRUE(retainedBytes.isOk());
	EXPECT_EQ(*retainedBytes, *bytes);
	ASSERT_TRUE(client.dropObject(1U).isOk());
	ASSERT_TRUE(client.dropObject(2U).isOk());
	ASSERT_TRUE(client.dropObject(3U).isOk());
	ASSERT_TRUE(client.dropObject(4U).isOk());
	ASSERT_TRUE(client.close().isOk());
	server.join();

	EXPECT_TRUE(serverStatus.isOk()) << serverStatus.toString();
	EXPECT_TRUE(serverSession.lastGpuEventWasOwned());
	EXPECT_TRUE(serverSession.lastGpuEventCompleted());
	EXPECT_NE(serverSession.lastGpuEventValue(), 0U);
}

TEST_VK(VkEngineTestFixture,
	SatelliteNlpSplitBatchGradientAndAuthoritativeUpdateMatchFullBatch)
{
	auto& engine = testEngine();
	oa::Vec<oa::F32> initialParameters;
	oa::Vec<oa::U32> inputs;
	oa::Vec<oa::U32> targets;
	ASSERT_TRUE(buildFrozenInputs(
		engine, initialParameters, inputs, targets).isOk());
	const oa::Span<const oa::F32> initialParameterView(
		initialParameters.data(), initialParameters.size());
	const auto initialHash = oa::satelliteHashF32(initialParameterView);
	auto full = runLocalOracle(
		engine, initialParameterView,
		oa::Span<const oa::U32>(inputs.data(), inputs.size()),
		oa::Span<const oa::U32>(targets.data(), targets.size()));
	ASSERT_TRUE(full.isOk()) << full.getStatus().toString();

	constexpr oa::U32 localCount = oa::NlpSuiteBatchSize / 2U;
	constexpr oa::U32 remoteCount = oa::NlpSuiteBatchSize - localCount;
	constexpr oa::U32 context = oa::NlpSuiteContextLength;
	const oa::Usize localElements = static_cast<oa::Usize>(localCount) * context;
	const oa::Usize remoteElements = static_cast<oa::Usize>(remoteCount) * context;
	auto local = runLocalGradient(
		engine, initialParameterView,
		oa::Span<const oa::U32>(inputs.data(), localElements),
		oa::Span<const oa::U32>(targets.data(), localElements));
	ASSERT_TRUE(local.isOk()) << local.getStatus().toString();

	oa::Array<oa::U32, oa::NlpSuiteBatchSize> sampleIndices{};
	const oa::U32 limit = static_cast<oa::U32>(
		std::strlen(oa::NlpSuiteSampler::corpus()) - oa::NlpSuiteContextLength - 1U);
	for (oa::U32 i = 0; i < sampleIndices.size(); ++i) {
		sampleIndices[i] = (i * 7U) % limit;
	}
	const oa::I64 parameterShape[] = {oa::SatelliteNlpStepParameterCount};
	const oa::I64 remoteShape[] = {remoteCount, oa::NlpSuiteContextLength};
	const auto remoteInputs = oa::Span<const oa::U32>(
		inputs.data() + localElements, remoteElements);
	const auto remoteTargets = oa::Span<const oa::U32>(
		targets.data() + localElements, remoteElements);
	const auto remoteSamples = oa::Span<const oa::U32>(
		sampleIndices.data() + localCount, remoteCount);

	auto listenerResult = oa::TcpListener::bind("127.0.0.1", 0U, 8);
	ASSERT_TRUE(listenerResult.isOk());
	auto listener = oa::move(*listenerResult);
	std::atomic<oa::U32> handlerCalls{0U};
	auto serverConfig = makeConfig();
	serverConfig.namedOperations.pushBack(oa::String(oa::SatelliteNlpStepOperation));
	serverConfig.namedOperations.pushBack(oa::String(oa::SatelliteNlpGradientOperation));
	serverConfig.namedWork = [&](oa::Engine& inEngine,
		const oa::SatelliteNamedRequest& inRequest) {
		++handlerCalls;
		return oa::satelliteExecuteNlpWork(inEngine, inRequest);
	};
	oa::SatelliteServerSession serverSession(engine, oa::move(serverConfig));
	oa::Status serverStatus;
	std::thread server([&] {
		auto accepted = listener.accept();
		ASSERT_TRUE(accepted.isOk());
		serverStatus = serverSession.serve(oa::move(*accepted));
	});

	auto connected = oa::SatelliteClientSession::connect(
		"127.0.0.1", listener.port(), makeConfig());
	ASSERT_TRUE(connected.isOk()) << connected.getStatus().toString();
	auto client = oa::move(*connected);
	constexpr oa::U64 initialVersion = 11U;
	ASSERT_TRUE(client.putF32Versioned(
		1U, initialVersion, parameterShape, initialParameterView).isOk());
	ASSERT_TRUE(client.putU32(2U, remoteShape, remoteInputs).isOk());
	ASSERT_TRUE(client.putU32(3U, remoteShape, remoteTargets).isOk());
	EXPECT_TRUE(oa::satelliteStartNlpGradient(
		client, 1U, 2U, 3U, 4U, oa::NlpSuiteRngSeed, 0U,
		static_cast<oa::U32>(oa::NlpSuiteBatchSize), remoteSamples,
		initialVersion, initialHash).isError());
	auto request = oa::satelliteStartNlpGradient(
		client, 1U, 2U, 3U, 4U, oa::NlpSuiteRngSeed, 0U, localCount,
		remoteSamples, initialVersion, initialHash);
	ASSERT_TRUE(request.isOk()) << request.getStatus().toString();
	auto waited = client.wait(*request);
	ASSERT_TRUE(waited.isOk()) << waited.getStatus().toString();
	auto bytes = oa::SatelliteClientSession::readBytesResult(*waited);
	ASSERT_TRUE(bytes.isOk()) << bytes.getStatus().toString();
	auto remote = oa::satelliteDecodeNlpGradientResult(
		oa::Span<const oa::Byte>(bytes->data(), bytes->size()));
	ASSERT_TRUE(remote.isOk()) << remote.getStatus().toString();
	EXPECT_TRUE(oa::satelliteDecodeNlpGradientResult(
		oa::Span<const oa::Byte>(bytes->data(), bytes->size() - 1U)).isError());
	EXPECT_EQ(remote->seed, oa::NlpSuiteRngSeed);
	EXPECT_EQ(remote->stepIndex, 0U);
	EXPECT_EQ(remote->batchOffset, localCount);
	EXPECT_EQ(remote->batchCount, remoteCount);
	EXPECT_EQ(remote->predictedPositions, remoteCount * context);
	EXPECT_EQ(remote->initialParameterHash, initialHash);
	EXPECT_EQ(remote->inputHash, oa::satelliteHashU32(remoteInputs));
	EXPECT_EQ(remote->targetHash, oa::satelliteHashU32(remoteTargets));
	EXPECT_EQ(remote->parameterLayoutHash, local->parameterLayoutHash);

	const oa::F32 localWeight = static_cast<oa::F32>(localCount)
		/ static_cast<oa::F32>(oa::NlpSuiteBatchSize);
	const oa::F32 remoteWeight = static_cast<oa::F32>(remoteCount)
		/ static_cast<oa::F32>(oa::NlpSuiteBatchSize);
	EXPECT_TRUE(near(
		local->loss * localWeight + remote->loss * remoteWeight,
		full->loss, 2e-5F, 2e-5F));
	ASSERT_EQ(local->gradients.size(), remote->gradients.size());
	oa::Vec<oa::F32> reduced(local->gradients.size());
	for (oa::Usize i = 0; i < reduced.size(); ++i) {
		reduced[i] = local->gradients[i] * localWeight
			+ remote->gradients[i] * remoteWeight;
	}
	expectVectorNear(
		oa::Span<const oa::F32>(reduced.data(), reduced.size()),
		oa::Span<const oa::F32>(full->gradients.data(), full->gradients.size()),
		1e-3F, 2e-4F);
	auto authoritative = applyAuthoritativeGradient(
		engine, initialParameterView,
		oa::Span<const oa::F32>(reduced.data(), reduced.size()));
	ASSERT_TRUE(authoritative.isOk()) << authoritative.getStatus().toString();
	expectVectorNear(
		oa::Span<const oa::F32>(authoritative->data(), authoritative->size()),
		oa::Span<const oa::F32>(full->updatedParameters.data(),
			full->updatedParameters.size()),
		1e-3F, 2e-4F);

	auto profile = oa::SatelliteClientSession::readProfile(*waited);
	ASSERT_TRUE(profile.isOk());
	const oa::String profileText(
		reinterpret_cast<const char*>(profile->data()), profile->size());
	EXPECT_NE(profileText.view().stdView().find(
		"workload=nlp-byte-transformer-gradient-v1"), std::string::npos);
	EXPECT_NE(profileText.view().stdView().find("batch_offset=32;batch_count=32"),
		std::string::npos);
	EXPECT_NE(profileText.view().stdView().find("kernel_fallbacks=0"),
		std::string::npos);

	ASSERT_TRUE(client.dropObject(1U).isOk());
	ASSERT_TRUE(client.dropObject(4U).isOk());
	const auto authoritativeHash = oa::satelliteHashF32(
		oa::Span<const oa::F32>(authoritative->data(), authoritative->size()));
	ASSERT_TRUE(client.putF32Versioned(
		5U, initialVersion + 1U, parameterShape,
		oa::Span<const oa::F32>(authoritative->data(), authoritative->size())).isOk());
	auto stale = oa::satelliteStartNlpGradient(
		client, 5U, 2U, 3U, 6U, oa::NlpSuiteRngSeed, 0U, localCount,
		remoteSamples, initialVersion, authoritativeHash);
	ASSERT_TRUE(stale.isError());
	EXPECT_EQ(stale.getStatus().getCode(), oa::StatusCode::FailedPrecondition);
	EXPECT_EQ(handlerCalls.load(), 1U);
	auto acknowledged = oa::satelliteStartNlpGradient(
		client, 5U, 2U, 3U, 6U, oa::NlpSuiteRngSeed, 0U, localCount,
		remoteSamples, initialVersion + 1U, authoritativeHash);
	ASSERT_TRUE(acknowledged.isOk()) << acknowledged.getStatus().toString();
	ASSERT_TRUE(client.wait(*acknowledged).isOk());
	EXPECT_EQ(handlerCalls.load(), 2U);

	ASSERT_TRUE(client.dropObject(2U).isOk());
	ASSERT_TRUE(client.dropObject(3U).isOk());
	ASSERT_TRUE(client.dropObject(5U).isOk());
	ASSERT_TRUE(client.dropObject(6U).isOk());
	ASSERT_TRUE(client.close().isOk());
	server.join();
	EXPECT_TRUE(serverStatus.isOk()) << serverStatus.toString();
	EXPECT_TRUE(serverSession.lastGpuEventWasOwned());
	EXPECT_TRUE(serverSession.lastGpuEventCompleted());
}

TEST_VK(VkEngineTestFixture,
	SatelliteNlpCoordinatorPersistsOptimizerAndAcknowledgedSnapshots)
{
	auto firstBatch = oa::satelliteBuildNlpBatch(0U);
	ASSERT_TRUE(firstBatch.isOk()) << firstBatch.getStatus().toString();
	ASSERT_EQ(firstBatch->sampleIndices.size(), oa::NlpSuiteBatchSize);
	ASSERT_EQ(firstBatch->inputs.size(),
		static_cast<oa::Usize>(oa::NlpSuiteBatchSize * oa::NlpSuiteContextLength));
	ASSERT_EQ(firstBatch->targets.size(), firstBatch->inputs.size());
	EXPECT_TRUE(oa::satelliteBuildNlpBatch(oa::NlpSuiteTrainingSteps).isError());

	auto& engine = testEngine();
	auto listenerResult = oa::TcpListener::bind("127.0.0.1", 0U, 8);
	ASSERT_TRUE(listenerResult.isOk());
	auto listener = oa::move(*listenerResult);
	std::atomic<oa::U32> handlerCalls{0U};
	auto serverConfig = makeConfig();
	serverConfig.namedOperations.pushBack(oa::String(oa::SatelliteNlpStepOperation));
	serverConfig.namedOperations.pushBack(oa::String(oa::SatelliteNlpGradientOperation));
	serverConfig.namedWork = [&](oa::Engine& inEngine,
		const oa::SatelliteNamedRequest& inRequest) {
		++handlerCalls;
		return oa::satelliteExecuteNlpWork(inEngine, inRequest);
	};
	oa::SatelliteServerSession serverSession(engine, oa::move(serverConfig));
	oa::Status serverStatus;
	std::thread server([&] {
		auto accepted = listener.accept();
		ASSERT_TRUE(accepted.isOk());
		serverStatus = serverSession.serve(oa::move(*accepted));
	});

	auto connected = oa::SatelliteClientSession::connect(
		"127.0.0.1", listener.port(), makeConfig());
	ASSERT_TRUE(connected.isOk()) << connected.getStatus().toString();
	auto client = oa::move(*connected);
	const oa::String checkpoint = "/tmp/oa-satellite-nlp-coordinator-test.oam";
	oa::SatelliteNlpCoordinatorConfig coordinatorConfig;
	coordinatorConfig.steps = 2U;
	coordinatorConfig.localBatchCount = 32U;
	coordinatorConfig.initialVersion = 41U;
	coordinatorConfig.checkpointPath = checkpoint;
	auto report = oa::satelliteRunSplitBatchNlp(
		engine, client, coordinatorConfig);
	ASSERT_TRUE(report.isOk()) << report.getStatus().toString();
	EXPECT_EQ(report->completedSteps, 2U);
	EXPECT_EQ(report->optimizerStep, 2U);
	EXPECT_TRUE(std::isfinite(report->initialLoss));
	EXPECT_TRUE(std::isfinite(report->finalLoss));
	EXPECT_TRUE(std::isfinite(report->finalAccuracy));
	EXPECT_NE(report->initialParameterHash, report->finalParameterHash);
	EXPECT_GT(report->localKernelSelections, 0U);
	EXPECT_EQ(report->localKernelFallbacks, 0U);
	EXPECT_GT(report->remoteKernelSelections, 0U);
	EXPECT_EQ(report->remoteKernelFallbacks, 0U);
	EXPECT_TRUE(report->checkpointRoundTrip);
	EXPECT_EQ(report->generated.size(),
		std::strlen(oa::NlpSuiteGenerationPrompt)
			+ static_cast<oa::Usize>(oa::NlpSuiteGenerationSourceUnits));
	EXPECT_EQ(handlerCalls.load(), 2U);

	ASSERT_TRUE(client.close().isOk());
	server.join();
	EXPECT_TRUE(serverStatus.isOk()) << serverStatus.toString();
	EXPECT_TRUE(serverSession.lastGpuEventWasOwned());
	EXPECT_TRUE(serverSession.lastGpuEventCompleted());
	EXPECT_EQ(std::remove(checkpoint.cStr()), 0);
}

TEST_VK(VkEngineTestFixture,
	SatelliteNlpCoordinatorCompletesCanonical300StepQualityGate)
{
	auto& engine = testEngine();
	auto listenerResult = oa::TcpListener::bind("127.0.0.1", 0U, 8);
	ASSERT_TRUE(listenerResult.isOk());
	auto listener = oa::move(*listenerResult);
	std::atomic<oa::U32> handlerCalls{0U};
	auto serverConfig = makeConfig();
	serverConfig.namedOperations.pushBack(oa::String(oa::SatelliteNlpStepOperation));
	serverConfig.namedOperations.pushBack(oa::String(oa::SatelliteNlpGradientOperation));
	serverConfig.namedWork = [&](oa::Engine& inEngine,
		const oa::SatelliteNamedRequest& inRequest) {
		++handlerCalls;
		return oa::satelliteExecuteNlpWork(inEngine, inRequest);
	};
	oa::SatelliteServerSession serverSession(engine, oa::move(serverConfig));
	oa::Status serverStatus;
	std::thread server([&] {
		auto accepted = listener.accept();
		ASSERT_TRUE(accepted.isOk());
		serverStatus = serverSession.serve(oa::move(*accepted));
	});

	auto connected = oa::SatelliteClientSession::connect(
		"127.0.0.1", listener.port(), makeConfig());
	ASSERT_TRUE(connected.isOk()) << connected.getStatus().toString();
	auto client = oa::move(*connected);
	const oa::String checkpoint = "/tmp/oa-satellite-nlp-coordinator-full-test.oam";
	oa::SatelliteNlpCoordinatorConfig coordinatorConfig;
	coordinatorConfig.checkpointPath = checkpoint;
	auto report = oa::satelliteRunSplitBatchNlp(
		engine, client, coordinatorConfig);
	ASSERT_TRUE(report.isOk()) << report.getStatus().toString();
	EXPECT_EQ(report->completedSteps, oa::NlpSuiteTrainingSteps);
	EXPECT_EQ(report->optimizerStep, oa::NlpSuiteTrainingSteps);
	EXPECT_LT(report->finalLoss, report->initialLoss);
	EXPECT_GT(report->finalAccuracy, 0.85F);
	EXPECT_TRUE(report->generationQualityPassed);
	EXPECT_TRUE(report->checkpointRoundTrip);
	EXPECT_EQ(report->localKernelFallbacks, 0U);
	EXPECT_GT(report->localKernelSelections, 0U);
	EXPECT_EQ(report->remoteKernelFallbacks, 0U);
	EXPECT_GT(report->remoteKernelSelections, 0U);
	EXPECT_EQ(handlerCalls.load(), oa::NlpSuiteTrainingSteps);

	ASSERT_TRUE(client.close().isOk());
	server.join();
	EXPECT_TRUE(serverStatus.isOk()) << serverStatus.toString();
	EXPECT_TRUE(serverSession.lastGpuEventWasOwned());
	EXPECT_TRUE(serverSession.lastGpuEventCompleted());
	EXPECT_EQ(std::remove(checkpoint.cStr()), 0);
}

TEST_VK(VkEngineTestFixture,
	SatelliteNlpStandaloneCompletesCanonical300StepQualityGate)
{
	auto& engine = testEngine();
	const oa::String checkpoint = "/tmp/oa-satellite-nlp-standalone-full-test.oam";
	oa::SatelliteNlpCoordinatorConfig config;
	config.checkpointPath = checkpoint;
	auto report = oa::satelliteRunStandaloneNlp(engine, config);
	ASSERT_TRUE(report.isOk()) << report.getStatus().toString();
	EXPECT_EQ(report->completedSteps, oa::NlpSuiteTrainingSteps);
	EXPECT_EQ(report->optimizerStep, oa::NlpSuiteTrainingSteps);
	EXPECT_LT(report->finalLoss, report->initialLoss);
	EXPECT_GT(report->finalAccuracy, 0.85F);
	EXPECT_TRUE(report->generationQualityPassed);
	EXPECT_TRUE(report->checkpointRoundTrip);
	EXPECT_GT(report->localKernelSelections, 0U);
	EXPECT_EQ(report->localKernelFallbacks, 0U);
	EXPECT_EQ(report->remoteKernelSelections, 0U);
	EXPECT_EQ(report->remoteKernelFallbacks, 0U);
	EXPECT_EQ(std::remove(checkpoint.cStr()), 0);
}
