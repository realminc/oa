// OA ML — Checkpoint Manager
//
// Path management, metric tracking, and rotation for OamModel (.oam) files.
// OaCheckpointManager wraps OaModule::Save/Load — the actual file format is
// OamModel (oam.h, magic "OAM\0") which supports sections, SPIR-V embedding,
// optimizer state, and training progress.
//
// Checkpoint naming:
//   var/model/dev/{ModelName}/
//     {ModelName}.oam                                          <- master (best)
//     checkpoint_{Context}/
//       {ModelName}_{Context}_step{N}_loss{V}.oam              <- incremental

#pragma once

#include <Oa/Core/Matrix.h>
#include <Oa/Core/FileIo.h>

class OaModule;
class OaOptimizer;

// OaCheckpointManager — auto-save best models with rotation
// Wraps OaModule::Save/Load to manage directory structure, path naming,
// metric tracking, and incremental-checkpoint rotation.

class OaCheckpointManagerConfig {
public:
	OaString Dir = OaFileIo::GetVarDir("model/dev").String();
	OaString ModelName = "OaModule";
	OaString Context;
	OaI32 MaxKeep = 5;
	bool SaveBest = true;
	OaString MetricName = "loss";
	bool LowerIsBetter = true;
};

class OaCheckpointManager {
public:
	explicit OaCheckpointManager(OaCheckpointManagerConfig InConfig = {});

	// Save model + optimizer state if the metric improved (or unconditionally
	// if InForce=true). Saves weights AND optimizer state (AdamW M/V/step,
	// etc.) into one .oam via OaModule::Save(path, opt).
	OaStatus MaybeSave(
		OaModule& InModel, OaOptimizer& InOpt,
		OaU64 InStep, OaF64 InMetric, bool InForce = false);

	// Save a resumable incremental checkpoint without changing the best-model
	// metric/master file. Used by mid-epoch saves: only a complete epoch metric
	// (preferably validation) may select the best model.
	OaStatus SaveIncremental(
		OaModule& InModel, OaOptimizer& InOpt,
		OaU64 InStep, OaF64 InMetric, const OaString& InMetricName = {});

	// Restore weights and optimizer state into an already-constructed
	// model/optimizer. Symmetric with MaybeSave.
	OaStatus LoadBestInto(OaModule& InOutModel, OaOptimizer& InOutOpt) const;
	OaStatus LoadLatestInto(OaModule& InOutModel, OaOptimizer& InOutOpt) const;

	[[nodiscard]] OaString GetModelDir() const;
	[[nodiscard]] OaString GetIncrementalDir() const;
	[[nodiscard]] OaString GetMasterPath() const;
	[[nodiscard]] bool IsBetter(OaF64 InMetric) const;
	[[nodiscard]] OaF64 GetBestMetric() const { return BestMetric_; }
	[[nodiscard]] const OaString& MetricName() const { return Config_.MetricName; }

private:
	[[nodiscard]] OaString BuildFilename(OaU64 InStep, OaF64 InMetric, const OaString& InMetricName = {}) const;
	void RotateCheckpoints();

	OaCheckpointManagerConfig Config_;
	OaF64 BestMetric_;

	class SavedCheckpoint {
	public:
		OaString Path;
		OaF64 Metric;
		OaU64 Step;
	};
	OaVec<SavedCheckpoint> Saved_;
};
