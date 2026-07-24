// OaTrainingViewerSource - non-owning OaTrainingSession dashboard adapter.
//
// Training owns the model, optimizer, graph and worker thread. The viewer only
// reads immutable snapshots and enqueues typed commands at the shared safe
// point. It can therefore attach to supervised, PPO, DQN or SAC training without
// another trainer or application abstraction.

#pragma once

#include <Oa/Ml/TrainingSession.h>
#include <Oa/Runtime/Sync.h>
#include <Oa/Ui/Image.h>
#include <Oa/Ui/Viewer.h>

struct OaTrainingViewerConfig {
	OaString Title = "OA Training";
	OaU32 HistoryCapacity = 512;
	OaU32 MaxMetricPlots = 6;
	bool ShowGpuTiming = true;
	bool ShowWallTiming = true;
	bool ShowPreview = true;
};

/// Immutable generated artifact handed from a training/evaluation producer to
/// the viewer. The texture allocation is shared; the timeline token prevents
/// presentation until the producing GPU batch has completed. Replacing a
/// pending frame drops it without ever blocking the training owner.
struct OaTrainingPreviewFrame {
	OaSharedPtr<const OaTexture> Texture;
	OaCompletionToken Completion;
	OaString Label;
	OaI64 Step = 0;
};

class OaTrainingViewerSource final : public OaViewerLiveSource {
public:
	explicit OaTrainingViewerSource(
		OaTrainingSession& InSession,
		OaTrainingViewerConfig InConfig = {});
	~OaTrainingViewerSource() override;

	OaTrainingViewerSource(const OaTrainingViewerSource&) = delete;
	OaTrainingViewerSource& operator=(const OaTrainingViewerSource&) = delete;
	OaTrainingViewerSource(OaTrainingViewerSource&&) = delete;
	OaTrainingViewerSource& operator=(OaTrainingViewerSource&&) = delete;

	OaStatus Open(OaEngine& InEngine) override;
	OaStatus Init(
		OaInputSystem& InInput,
		OaFunc<void(bool)> InCapturePointer) override;
	void Update(OaF32 InDeltaMs) override;
	void Render(
		OaUi& InUi,
		const OaTextAtlas& InTextAtlas,
		OaU32 InWidth,
		OaU32 InHeight) override;
	[[nodiscard]] OaStatus MarkConsumed(
		const OaEvent& InCompletion) override;
	[[nodiscard]] OaStatus Close() override;

	[[nodiscard]] OaStatus PublishPreview(OaTrainingPreviewFrame InFrame);

	[[nodiscard]] OaOpt<OaTrainingSnapshot> LatestSnapshot() const;
	[[nodiscard]] OaOpt<OaTrainingPreviewFrame> LatestPreview() const;
	[[nodiscard]] OaU32 MetricSeriesCount() const;
	[[nodiscard]] OaU32 MetricSampleCount(OaStringView InName) const;

private:
	struct Impl;
	OaUniquePtr<Impl> Impl_;
};
