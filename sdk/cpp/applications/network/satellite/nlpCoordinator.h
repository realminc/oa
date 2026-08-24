#pragma once

#include <oa/core/types.h>
#include <oa/network/satelliteSession.h>

namespace oa {

class Engine;

class SatelliteNlpCoordinatorConfig {
public:
	U32 steps = 300U;
	U32 localBatchCount = 32U;
	U64 initialVersion = 1U;
	Bool overlapRemote = false;
	String checkpointPath;
};

class SatelliteNlpCoordinatorReport {
public:
	U32 completedSteps = 0U;
	U64 optimizerStep = 0U;
	F32 initialLoss = 0.0F;
	F32 finalLoss = 0.0F;
	F32 finalAccuracy = 0.0F;
	Array<Byte, 32> initialParameterHash{};
	Array<Byte, 32> finalParameterHash{};
	Array<Byte, 32> parameterLayoutHash{};
	U64 localKernelSelections = 0U;
	U64 localKernelFallbacks = 0U;
	U64 remoteKernelSelections = 0U;
	U64 remoteKernelFallbacks = 0U;
	Bool checkpointRoundTrip = false;
	Bool generationQualityPassed = false;
	String generated;
	F64 completeWallMs = 0.0;
};

[[nodiscard]] Result<SatelliteNlpCoordinatorReport> satelliteRunSplitBatchNlp(
	Engine& inEngine,
	SatelliteClientSession& inClient,
	const SatelliteNlpCoordinatorConfig& inConfig);

[[nodiscard]] Result<SatelliteNlpCoordinatorReport> satelliteRunStandaloneNlp(
	Engine& inEngine,
	const SatelliteNlpCoordinatorConfig& inConfig);

} // namespace oa
