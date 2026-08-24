// OA ML — checkpoint Manager
//
// Path management, metric tracking, and rotation for ModelFile (.oam) files.
// CheckpointManager wraps Module::save/load — the actual file format is
// ModelFile (modelFile.h, magic "OAM\0") which supports sections,
// optimizer state, and training progress.
//
// checkpoint naming:
//   var/model/dev/{modelName}/
//     {modelName}.oam                                          <- master (best)
//     checkpoint_{context}/
//       {modelName}_{context}_step{N}_loss{V}.oam              <- incremental

#pragma once

#include <oa/core/matrix.h>
#include <oa/core/filesystem.h>
#include <oa/core/paths.h>

namespace oa {

class Module;
class Optimizer;
class Engine;

// CheckpointManager — auto-save best models with rotation.
// Wraps Module::save/load to manage directory structure, path naming,
// metric tracking, and incremental-checkpoint rotation.

class CheckpointManagerConfig {
public:
	oa::String dir = oa::Paths::var("model/dev").string();
	oa::String modelName = "Module";
	oa::String context;
	oa::I32 maxKeep = 5;
	bool saveBest = true;
	oa::String metricName = "loss";
	bool lowerIsBetter = true;
};

class CheckpointManager {
public:
	explicit CheckpointManager(
		oa::Engine& inEngine, CheckpointManagerConfig inConfig = {});

	// save model + optimizer state if the metric improved (or unconditionally
	// if inForce=true). Saves weights AND optimizer state (AdamW M/V/step,
	// etc.) into one .oam via Module::save(engine, path, opt).
	oa::Status maybeSave(
		oa::Module& inModel, oa::Optimizer& inOpt,
		oa::U64 inStep, oa::F64 inMetric, bool inForce = false);

	// save a resumable incremental checkpoint without changing the best-model
	// metric/master file. Used by mid-epoch saves: only a complete epoch metric
	// (preferably validation) may select the best model.
	oa::Status saveIncremental(
		oa::Module& inModel, oa::Optimizer& inOpt,
		oa::U64 inStep, oa::F64 inMetric, const oa::String& inMetricName = {});

	// Restore weights and optimizer state into an already-constructed
	// model/optimizer. Symmetric with maybeSave.
	oa::Status loadBestInto(oa::Module& inOutModel, oa::Optimizer& inOutOpt) const;
	oa::Status loadLatestInto(oa::Module& inOutModel, oa::Optimizer& inOutOpt) const;

	[[nodiscard]] oa::String modelDir() const;
	[[nodiscard]] oa::String incrementalDir() const;
	[[nodiscard]] oa::String masterPath() const;
	[[nodiscard]] bool isBetter(oa::F64 inMetric) const;
	[[nodiscard]] oa::F64 bestMetric() const { return bestMetric_; }
	[[nodiscard]] const oa::String& metricName() const { return config_.metricName; }

private:
	[[nodiscard]] oa::String buildFilename(
		oa::U64 inStep,
		oa::F64 inMetric,
		const oa::String& inMetricName = {}
	) const;
	void rotateCheckpoints();

	CheckpointManagerConfig config_;
	oa::Engine* engine_ = nullptr;
	oa::F64 bestMetric_;

	class SavedCheckpoint {
	public:
		oa::String path;
		oa::F64 metric;
		oa::U64 step;
	};
	oa::Vec<SavedCheckpoint> saved_;
};

} // namespace oa
