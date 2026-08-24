// TrainingViewerSource - non-owning oa::TrainingSession dashboard adapter.
//
// training owns the model, optimizer, graph and worker thread. The viewer only
// reads immutable snapshots and enqueues typed commands at the shared safe
// point. It can therefore attach to supervised, PPO, DQN or SAC training without
// another trainer or application abstraction.

#pragma once

#include <oa/ml/trainingSession.h>
#include <oa/runtime/event.h>
#include <oa/ui/image.h>
#include <oa/ui/viewer.h>

namespace oa {

struct TrainingViewerConfig {
	oa::String title = "OA training";
	oa::U32 historyCapacity = 512;
	oa::U32 maxMetricPlots = 6;
	bool showGpuTiming = true;
	bool showWallTiming = true;
	bool showPreview = true;
};

/// Immutable generated artifact handed from a training/evaluation producer to
/// the viewer. The texture allocation is shared; the timeline token prevents
/// presentation until the producing GPU batch has completed. Replacing a
/// pending frame drops it without ever blocking the training owner.
struct TrainingPreviewFrame {
	oa::SharedPtr<const oa::Texture> texture;
	oa::Event completion;
	oa::String label;
	oa::I64 step = 0;
};

class TrainingViewerSource final : public ViewerLiveSource {
public:
	explicit TrainingViewerSource(
		oa::TrainingSession& inSession,
		TrainingViewerConfig inConfig = {});
	~TrainingViewerSource() override;

	TrainingViewerSource(const TrainingViewerSource&) = delete;
	TrainingViewerSource& operator=(const TrainingViewerSource&) = delete;
	TrainingViewerSource(TrainingViewerSource&&) = delete;
	TrainingViewerSource& operator=(TrainingViewerSource&&) = delete;

	[[nodiscard]] ViewerLiveCapabilities capabilities() const noexcept override {
		return {};
	}
	oa::Status open(oa::Engine& inEngine) override;
	oa::Status init(
		InputSystem& inInput,
		oa::Fn<void(bool)> inCapturePointer) override;
	[[nodiscard]] oa::Status update(oa::F32 inDeltaMs) override;
	[[nodiscard]] oa::Status render(
		Ui& inUi,
		const TextAtlas& inTextAtlas,
		oa::U32 inWidth,
		oa::U32 inHeight) override;
	[[nodiscard]] oa::Status close() override;

	[[nodiscard]] oa::Status publishPreview(TrainingPreviewFrame inFrame);

	[[nodiscard]] oa::Optional<oa::TrainingSnapshot> latestSnapshot() const;
	[[nodiscard]] oa::Optional<TrainingPreviewFrame> latestPreview() const;
	[[nodiscard]] oa::U32 metricSeriesCount() const;
	[[nodiscard]] oa::U32 metricSampleCount(oa::StringView inName) const;

private:
	struct Impl;
	oa::UniquePtr<Impl> impl_;
};

}  // namespace oa
