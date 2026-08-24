#pragma once

#include <oa/ui/viewer.h>

// Tutorial-local live source for the one OA Viewer application. It borrows the
// Viewer's engine, advances the deterministic scalar environment at its fixed
// policy timestep, and exposes the vulkan render target directly to oa::Ui.
class LunarLander3dViewerSource final : public oa::ViewerLiveSource {
public:
	explicit LunarLander3dViewerSource(oa::U32 inSampleCount = 1U);
	~LunarLander3dViewerSource() override;
	LunarLander3dViewerSource(
		const LunarLander3dViewerSource&) = delete;
	LunarLander3dViewerSource& operator=(
		const LunarLander3dViewerSource&) = delete;

	[[nodiscard]] oa::ViewerLiveCapabilities capabilities() const noexcept override {
		return {
			.receivesEvents = false,
			.publishesRenderDependency = true,
			.retainsConsumerCompletion = true,
		};
	}
	[[nodiscard]] oa::Status open(oa::Engine& inEngine) override;
	[[nodiscard]] oa::Status init(
		oa::InputSystem& inInput,
		oa::Fn<void(bool)> inCapturePointer) override;
	[[nodiscard]] oa::Status update(oa::F32 inDeltaMs) override;
	[[nodiscard]] oa::Status render(
		oa::Ui& inUi,
		const oa::TextAtlas& inTextAtlas,
		oa::U32 inWidth,
		oa::U32 inHeight) override;
	[[nodiscard]] oa::Result<oa::Event> renderReady() const override;
	[[nodiscard]] oa::Status markConsumed(
		const oa::Event& inCompletion) override;
	[[nodiscard]] oa::Status close() override;

private:
	class Impl;
	oa::UniquePtr<Impl> impl_;
};
