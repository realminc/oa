// oa::CheckpointManager — .oam path management, rotation, and best-tracking.
// actual save/load delegates to oa::Module::save/load (ModelFile "OAM\0" format).

#include <oa/ml/checkpoint.h>
#include <oa/ml/module.h>
#include <oa/ml/modelFile.h>
#include <oa/ml/optim.h>
#include <oa/core/log.h>

#include <algorithm>
#include <charconv>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>

static oa::Status ensureDirectoryTree(const oa::String& inPath) {
	if (inPath.empty()) return oa::Status::ok();

	oa::Usize pos = 1;
	while (pos <= inPath.size()) {
		const oa::Usize slash = inPath.find('/', pos);
		const oa::Usize end = (slash == oa::String::Npos) ? inPath.size() : slash;
		const oa::String prefix = inPath.substr(0, end);
		if (!prefix.empty()) {
			oa::Status st = oa::Filesystem::createDirectories(oa::Path(prefix));
			if (!st) {
				return oa::Status::error("create checkpoint directory failed: " + prefix + ": " + st.toString());
			}
		}
		if (slash == oa::String::Npos) break;
		pos = slash + 1;
	}

	return oa::Status::ok();
}

static oa::Status saveCheckpointFile(
	oa::Engine& inEngine,
	const oa::String& inPath,
	oa::Module& inModel,
	oa::Optimizer& inOpt,
	oa::U64 inStep,
	oa::F64 inMetric,
	const oa::String& inMetricName,
	bool inLowerIsBetter)
{
	if (inStep > static_cast<oa::U64>(std::numeric_limits<oa::I64>::max())) {
		return oa::Status::error(oa::StatusCode::OutOfRange,
			"checkpoint step exceeds the .oam progress range");
	}
	oa::ModelFile checkpoint;
	OA_RETURN_IF_ERROR(inModel.saveTo(inEngine, checkpoint));
	OA_RETURN_IF_ERROR(inOpt.saveTo(inEngine, checkpoint));
	checkpoint.progress.step = static_cast<oa::I64>(inStep);
	checkpoint.progress.lr = inOpt.getLr();
	checkpoint.progress.bestMetric = static_cast<oa::F32>(inMetric);
	checkpoint.progress.lowerIsBetter = inLowerIsBetter ? 1 : 0;
	std::memset(checkpoint.progress.metricName, 0,
		sizeof(checkpoint.progress.metricName));
	std::strncpy(checkpoint.progress.metricName, inMetricName.cStr(),
		sizeof(checkpoint.progress.metricName) - 1);
	return checkpoint.save(inPath);
}

static oa::Status restoreCheckpointFile(
	oa::Engine& inEngine,
	const oa::String& inPath,
	oa::Module& inOutModel,
	oa::Optimizer& inOutOpt,
	oa::U64 inExpectedStep,
	bool inCheckStep)
{
	auto loaded = oa::ModelFile::load(inPath);
	if (not loaded.isOk()) return loaded.getStatus();
	auto checkpoint = std::move(loaded).getValue();
	if (inCheckStep and checkpoint.formatVersion >= 2
		and (checkpoint.progress.step < 0
			or static_cast<oa::U64>(checkpoint.progress.step) != inExpectedStep))
	{
		return oa::Status::error(oa::StatusCode::CheckpointCorrupt,
			"checkpoint filename/progress step mismatch: " + inPath);
	}
	OA_RETURN_IF_ERROR(inOutOpt.validateLoad(checkpoint));
	OA_RETURN_IF_ERROR(inOutModel.loadFrom(inEngine, checkpoint));
	return inOutOpt.loadFrom(inEngine, checkpoint);
}

// oa::CheckpointManager

oa::CheckpointManager::CheckpointManager(
	oa::Engine& inEngine, oa::CheckpointManagerConfig inConfig)
	: config_(std::move(inConfig)), engine_(&inEngine) {
	bestMetric_ = config_.lowerIsBetter
		? std::numeric_limits<oa::F64>::max()
		: -std::numeric_limits<oa::F64>::max();
}

oa::String oa::CheckpointManager::modelDir() const {
	return config_.dir + "/" + config_.modelName;
}

oa::String oa::CheckpointManager::incrementalDir() const {
	const oa::String base = modelDir();
	if (config_.context.empty()) return base + "/checkpoint";
	return base + "/checkpoint_" + config_.context;
}

oa::String oa::CheckpointManager::masterPath() const {
	return modelDir() + "/" + config_.modelName + ".oam";
}

bool oa::CheckpointManager::isBetter(oa::F64 inMetric) const {
	return config_.lowerIsBetter
		? (inMetric < bestMetric_)
		: (inMetric > bestMetric_);
}

oa::String oa::CheckpointManager::buildFilename(
	oa::U64 inStep,
	oa::F64 inMetric,
	const oa::String& inMetricName
) const {
	char buf[512];
	const oa::String& metricName = inMetricName.empty() ? config_.metricName : inMetricName;
	if (config_.context.empty()) {
		snprintf(buf, sizeof(buf), "%s_step%llu_%s%.2f.oam",
			config_.modelName.cStr(),
			static_cast<unsigned long long>(inStep),
			metricName.cStr(), inMetric);
	} else {
		snprintf(buf, sizeof(buf), "%s_%s_step%llu_%s%.2f.oam",
			config_.modelName.cStr(), config_.context.cStr(),
			static_cast<unsigned long long>(inStep),
			metricName.cStr(), inMetric);
	}
	return oa::String(buf);
}

void oa::CheckpointManager::rotateCheckpoints() {
	if (config_.maxKeep <= 0) return;
	if (static_cast<oa::I32>(saved_.size()) <= config_.maxKeep) return;

	auto compare = [](const SavedCheckpoint& inA, const SavedCheckpoint& inB) {
		return inA.step > inB.step;  // newest first; pop the oldest from the back
	};
	std::sort(saved_.begin(), saved_.end(), compare);

	while (static_cast<oa::I32>(saved_.size()) > config_.maxKeep) {
		const auto& worst = saved_.back();
		(void)oa::Filesystem::removeFile(oa::Path(worst.path));
		OaLogDebug(oa::LogComponent::Ml, "Rotated: %s", worst.path.cStr());
		saved_.popBack();
	}
}

oa::Status oa::CheckpointManager::maybeSave(
	oa::Module& inModel, oa::Optimizer& inOpt,
	oa::U64 inStep, oa::F64 inMetric, bool inForce
) {
	const bool improved = isBetter(inMetric);
	if (not improved and not inForce) return oa::Status::ok();

	OA_RETURN_IF_ERROR(saveIncremental(inModel, inOpt, inStep, inMetric));

	if (improved) {
		if (config_.saveBest) {
			const oa::String masterPath = this->masterPath();
			OA_RETURN_IF_ERROR(saveCheckpointFile(*engine_, masterPath, inModel, inOpt,
				inStep, inMetric, config_.metricName, config_.lowerIsBetter));
			OaLogInfo(oa::LogComponent::Ml, "* Best: %s=%.4f -> %s",
				config_.metricName.cStr(), inMetric, masterPath.cStr());
		}
		bestMetric_ = inMetric;
	}

	return oa::Status::ok();
}

oa::Status oa::CheckpointManager::saveIncremental(
	oa::Module& inModel, oa::Optimizer& inOpt,
	oa::U64 inStep, oa::F64 inMetric, const oa::String& inMetricName
) {
	OA_RETURN_IF_ERROR(ensureDirectoryTree(incrementalDir()));
	const oa::String& metricName = inMetricName.empty() ? config_.metricName : inMetricName;
	const oa::String filename = buildFilename(inStep, inMetric, metricName);
	const oa::String path = incrementalDir() + "/" + filename;
	const oa::Status saveStatus = saveCheckpointFile(*engine_, path, inModel, inOpt,
		inStep, inMetric, metricName, config_.lowerIsBetter);
	if (not saveStatus.isOk()) {
		OaLogError(oa::LogComponent::Ml, "checkpoint save failed: %s", path.cStr());
		return saveStatus;
	}
	OaLogInfo(oa::LogComponent::Ml, "checkpoint: %s (%s=%.4f step=%llu)",
		filename.cStr(), metricName.cStr(), inMetric,
		static_cast<unsigned long long>(inStep));
	saved_.pushBack({path, inMetric, inStep});
	rotateCheckpoints();
	return oa::Status::ok();
}

// ─── oa::Module + oa::Optimizer load helpers ───────────────────────────────────

oa::Status oa::CheckpointManager::loadBestInto(oa::Module& inOutModel, oa::Optimizer& inOutOpt) const {
	const oa::String masterPath = this->masterPath();
	OaLogInfo(oa::LogComponent::Ml, "Loading best: %s", masterPath.cStr());
	return restoreCheckpointFile(
		*engine_, masterPath, inOutModel, inOutOpt, 0, false);
}

oa::Status oa::CheckpointManager::loadLatestInto(oa::Module& inOutModel, oa::Optimizer& inOutOpt) const {
	// Prefer the in-memory saved_ list when we have one (saved this session);
	// otherwise scan the incremental dir, matching the LoadLatest scan rules.
	oa::String latestPath;
	oa::U64 expectedStep = 0;
	bool found = false;
	if (not saved_.empty()) {
		const SavedCheckpoint* latest = &saved_[0];
		for (const auto& s : saved_) {
			if (s.step > latest->step) latest = &s;
		}
		latestPath = latest->path;
		expectedStep = latest->step;
		found = true;
	} else {
		const oa::String dir = incrementalDir();
		if (not oa::Filesystem::isDirectory(oa::Path(dir))) {
			return oa::Status::error(oa::StatusCode::NotFound, "No checkpoint dir: " + dir);
		}
		auto filesResult = oa::Filesystem::listFiles(oa::Path(dir), ".oam");
		if (not filesResult.isOk()) return filesResult.getStatus();

		for (const auto& filePath : filesResult.getValue()) {
			const oa::String name = filePath.stem().string() + filePath.extension().string();
			if (name == config_.modelName + ".oam") continue;
			const auto stepPos = name.find("_step");
			if (stepPos == oa::String::Npos) continue;
			const auto stepStart = stepPos + 5;
			const auto stepEnd = name.find('_', stepStart);
			if (stepEnd == oa::String::Npos) continue;
			const oa::String stepText = name.substr(stepStart, stepEnd - stepStart);
			oa::U64 step = 0;
			const char* first = stepText.cStr();
			const char* last = first + stepText.size();
			const auto parsed = std::from_chars(first, last, step);
			if (parsed.ec != std::errc{} or parsed.ptr != last) continue;
			if (not found or step > expectedStep) {
				expectedStep = step;
				latestPath = filePath.string();
				found = true;
			}
		}
		if (not found) {
			return oa::Status::error(oa::StatusCode::NotFound, "No checkpoints in " + dir);
		}
	}

	OaLogInfo(oa::LogComponent::Ml, "Loading latest: %s", latestPath.cStr());
	return restoreCheckpointFile(
		*engine_, latestPath, inOutModel, inOutOpt, expectedStep, true);
}
