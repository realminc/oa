#pragma once

#include <oa/network/satelliteSession.h>

namespace oa {

class NlpSuiteModel;

inline const StringView SatelliteNlpStepOperation =
	"nlp-byte-transformer-step-v1";
inline const StringView SatelliteNlpGradientOperation =
	"nlp-byte-transformer-gradient-v1";
inline constexpr U32 SatelliteNlpStepParameterCount = 25760U;
inline constexpr U32 SatelliteNlpStepLogitSampleCount = 64U;

class SatelliteNlpBatch {
public:
	U64 stepIndex = 0;
	Vec<U32> sampleIndices;
	Vec<U32> inputs;
	Vec<U32> targets;
};

class SatelliteNlpStepResult {
public:
	U64 seed = 0;
	U64 stepIndex = 0;
	U32 predictedPositions = 0;
	U32 vocabSize = 0;
	F32 loss = 0.0F;
	F32 logitMin = 0.0F;
	F32 logitMax = 0.0F;
	F32 logitMean = 0.0F;
	F32 logitL2 = 0.0F;
	Array<Byte, 32> initialParameterHash{};
	Array<Byte, 32> inputHash{};
	Array<Byte, 32> targetHash{};
	Array<Byte, 32> parameterLayoutHash{};
	Array<Byte, 32> updatedParameterHash{};
	Vec<F32> logitSample;
	Vec<F32> gradients;
	Vec<F32> updatedParameters;
};

class SatelliteNlpGradientResult {
public:
	U64 seed = 0;
	U64 stepIndex = 0;
	U32 batchOffset = 0;
	U32 batchCount = 0;
	U32 predictedPositions = 0;
	F32 loss = 0.0F;
	Array<Byte, 32> initialParameterHash{};
	Array<Byte, 32> inputHash{};
	Array<Byte, 32> targetHash{};
	Array<Byte, 32> parameterLayoutHash{};
	Vec<F32> gradients;
};

[[nodiscard]] Result<SatelliteNamedResult> satelliteExecuteNlpStep(
	Engine& inEngine,
	const SatelliteNamedRequest& inRequest);

[[nodiscard]] Result<SatelliteNamedResult> satelliteExecuteNlpGradient(
	Engine& inEngine,
	const SatelliteNamedRequest& inRequest);

[[nodiscard]] Result<SatelliteNamedResult> satelliteExecuteNlpWork(
	Engine& inEngine,
	const SatelliteNamedRequest& inRequest);

[[nodiscard]] Result<U64> satelliteStartNlpStep(
	SatelliteClientSession& inClient,
	U64 inParameters,
	U64 inTokens,
	U64 inTargets,
	U64 inOutput,
	U64 inSeed,
	U64 inStepIndex,
	Span<const U32> inSampleIndices,
	U64 inExpectedVersion,
	const Array<Byte, 32>& inExpectedHash);

[[nodiscard]] Result<U64> satelliteStartNlpGradient(
	SatelliteClientSession& inClient,
	U64 inParameters,
	U64 inTokens,
	U64 inTargets,
	U64 inOutput,
	U64 inSeed,
	U64 inStepIndex,
	U32 inBatchOffset,
	Span<const U32> inSampleIndices,
	U64 inExpectedVersion,
	const Array<Byte, 32>& inExpectedHash);

[[nodiscard]] Result<SatelliteNlpStepResult> satelliteDecodeNlpStepResult(
	Span<const Byte> inBytes);

[[nodiscard]] Result<SatelliteNlpGradientResult> satelliteDecodeNlpGradientResult(
	Span<const Byte> inBytes);

[[nodiscard]] Array<Byte, 32> satelliteHashF32(Span<const F32> inValues);
[[nodiscard]] Array<Byte, 32> satelliteHashU32(Span<const U32> inValues);

[[nodiscard]] Result<SatelliteNlpBatch> satelliteBuildNlpBatch(U64 inStepIndex);
[[nodiscard]] Array<Byte, 32> satelliteNlpParameterLayoutHash(
	NlpSuiteModel& inModel);

} // namespace oa
