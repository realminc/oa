// OaCheckpointManager — .oam path management, rotation, and best-tracking.
// Actual save/load delegates to OaModule::Save/Load (OamModel "OAM\0" format).

#include <Oa/Ml/Checkpoint.h>
#include <Oa/Ml/Module.h>
#include <Oa/Ml/Optim.h>
#include <Oa/Core/Log.h>

#include <algorithm>
#include <cstdio>
#include <string>

static OaStatus EnsureDirectoryTree(const OaString& InPath) {
	if (InPath.empty()) return OaStatus::Ok();

	OaUsize pos = 1;
	while (pos <= InPath.size()) {
		const OaUsize slash = InPath.find('/', pos);
		const OaUsize end = (slash == OaString::npos) ? InPath.size() : slash;
		const OaString prefix = InPath.substr(0, end);
		if (!prefix.empty()) {
			OaStatus st = OaFileIo::CreateDirectories(OaPath(prefix));
			if (!st) {
				return OaStatus::Error("create checkpoint directory failed: " + prefix + ": " + st.ToString());
			}
		}
		if (slash == OaString::npos) break;
		pos = slash + 1;
	}

	return OaStatus::Ok();
}

// OaCheckpointManager

OaCheckpointManager::OaCheckpointManager(OaCheckpointManagerConfig InConfig)
	: Config_(std::move(InConfig)) {
	BestMetric_ = Config_.LowerIsBetter
		? std::numeric_limits<OaF64>::max()
		: -std::numeric_limits<OaF64>::max();
}

OaString OaCheckpointManager::GetModelDir() const {
	return Config_.Dir + "/" + Config_.ModelName;
}

OaString OaCheckpointManager::GetIncrementalDir() const {
	const OaString base = GetModelDir();
	if (Config_.Context.empty()) return base + "/checkpoint";
	return base + "/checkpoint_" + Config_.Context;
}

OaString OaCheckpointManager::GetMasterPath() const {
	return GetModelDir() + "/" + Config_.ModelName + ".oam";
}

bool OaCheckpointManager::IsBetter(OaF64 InMetric) const {
	return Config_.LowerIsBetter
		? (InMetric < BestMetric_)
		: (InMetric > BestMetric_);
}

OaString OaCheckpointManager::BuildFilename(OaU64 InStep, OaF64 InMetric, const OaString& InMetricName) const {
	char buf[512];
	const OaString& metricName = InMetricName.empty() ? Config_.MetricName : InMetricName;
	if (Config_.Context.empty()) {
		snprintf(buf, sizeof(buf), "%s_step%llu_%s%.2f.oam",
			Config_.ModelName.c_str(),
			static_cast<unsigned long long>(InStep),
			metricName.c_str(), InMetric);
	} else {
		snprintf(buf, sizeof(buf), "%s_%s_step%llu_%s%.2f.oam",
			Config_.ModelName.c_str(), Config_.Context.c_str(),
			static_cast<unsigned long long>(InStep),
			metricName.c_str(), InMetric);
	}
	return OaString(buf);
}

void OaCheckpointManager::RotateCheckpoints() {
	if (Config_.MaxKeep <= 0) return;
	if (static_cast<OaI32>(Saved_.Size()) <= Config_.MaxKeep) return;

	auto compare = [](const SavedCheckpoint& InA, const SavedCheckpoint& InB) {
		return InA.Step > InB.Step;  // newest first; pop the oldest from the back
	};
	std::sort(Saved_.Begin(), Saved_.End(), compare);

	while (static_cast<OaI32>(Saved_.Size()) > Config_.MaxKeep) {
		const auto& worst = Saved_.Back();
		(void)OaFileIo::RemoveFile(OaPath(worst.Path));
		OA_LOG_DEBUG(OaLogComponent::ML, "Rotated: %s", worst.Path.c_str());
		Saved_.PopBack();
	}
}

OaStatus OaCheckpointManager::MaybeSave(
	OaModule& InModel, OaOptimizer& InOpt,
	OaU64 InStep, OaF64 InMetric, bool InForce
) {
	const bool improved = IsBetter(InMetric);
	if (not improved and not InForce) return OaStatus::Ok();

	OA_RETURN_IF_ERROR(SaveIncremental(InModel, InOpt, InStep, InMetric));

	if (improved) {
		BestMetric_ = InMetric;
		const OaString masterPath = GetMasterPath();
		const OaStatus masterStatus = InModel.Save(masterPath, InOpt);
		if (masterStatus.IsOk()) {
			OA_LOG_INFO(OaLogComponent::ML, "* Best: %s=%.4f -> %s",
				Config_.MetricName.c_str(), InMetric, masterPath.c_str());
		}
	}

	return OaStatus::Ok();
}

OaStatus OaCheckpointManager::SaveIncremental(
	OaModule& InModel, OaOptimizer& InOpt,
	OaU64 InStep, OaF64 InMetric, const OaString& InMetricName
) {
	OA_RETURN_IF_ERROR(EnsureDirectoryTree(GetIncrementalDir()));
	const OaString& metricName = InMetricName.empty() ? Config_.MetricName : InMetricName;
	const OaString filename = BuildFilename(InStep, InMetric, metricName);
	const OaString path = GetIncrementalDir() + "/" + filename;
	const OaStatus saveStatus = InModel.Save(path, InOpt);
	if (not saveStatus.IsOk()) {
		OA_LOG_ERROR(OaLogComponent::ML, "Checkpoint save failed: %s", path.c_str());
		return saveStatus;
	}
	OA_LOG_INFO(OaLogComponent::ML, "Checkpoint: %s (%s=%.4f step=%llu)",
		filename.c_str(), metricName.c_str(), InMetric,
		static_cast<unsigned long long>(InStep));
	Saved_.PushBack({path, InMetric, InStep});
	RotateCheckpoints();
	return OaStatus::Ok();
}

// ─── OaModule + OaOptimizer load helpers ───────────────────────────────────

OaStatus OaCheckpointManager::LoadBestInto(OaModule& InOutModel, OaOptimizer& InOutOpt) const {
	const OaString masterPath = GetMasterPath();
	OA_LOG_INFO(OaLogComponent::ML, "Loading best: %s", masterPath.c_str());
	return InOutModel.Load(masterPath, InOutOpt);
}

OaStatus OaCheckpointManager::LoadLatestInto(OaModule& InOutModel, OaOptimizer& InOutOpt) const {
	// Prefer the in-memory Saved_ list when we have one (saved this session);
	// otherwise scan the incremental dir, matching the LoadLatest scan rules.
	OaString latestPath;
	if (not Saved_.Empty()) {
		const SavedCheckpoint* latest = &Saved_[0];
		for (const auto& s : Saved_) {
			if (s.Step > latest->Step) latest = &s;
		}
		latestPath = latest->Path;
	} else {
		const OaString dir = GetIncrementalDir();
		if (not OaFileIo::IsDirectory(OaPath(dir))) {
			return OaStatus::Error(OaStatusCode::NotFound, "No checkpoint dir: " + dir);
		}
		auto filesResult = OaFileIo::ListFiles(OaPath(dir), ".oam");
		if (not filesResult.IsOk()) return filesResult.GetStatus();

		OaU64 latestStep = 0;
		for (const auto& filePath : filesResult.GetValue()) {
			const OaString name = OaFileIo::GetStem(filePath) + OaFileIo::GetExtension(filePath);
			if (name == Config_.ModelName + ".oam") continue;
			const auto stepPos = name.find("_step");
			if (stepPos == OaString::npos) continue;
			const auto stepStart = stepPos + 5;
			const auto stepEnd = name.find('_', stepStart);
			if (stepEnd == OaString::npos) continue;
			const OaU64 step = std::stoull(name.substr(stepStart, stepEnd - stepStart).StdStr());
			if (step > latestStep) {
				latestStep = step;
				latestPath = filePath.String();
			}
		}
		if (latestPath.Empty()) {
			return OaStatus::Error(OaStatusCode::NotFound, "No checkpoints in " + dir);
		}
	}

	OA_LOG_INFO(OaLogComponent::ML, "Loading latest: %s", latestPath.c_str());
	return InOutModel.Load(latestPath, InOutOpt);
}
