#pragma once

#include <Oa/Ui/Viewer.h>

// Tutorial-local live source for the one OA Viewer application. It borrows the
// Viewer's engine, advances the deterministic scalar environment at its fixed
// policy timestep, and exposes the Vulkan render target directly to OaUi.
class OaLunarLander3dViewerSource final : public OaViewerLiveSource {
public:
	OaLunarLander3dViewerSource();
	~OaLunarLander3dViewerSource() override;
	OaLunarLander3dViewerSource(
		const OaLunarLander3dViewerSource&) = delete;
	OaLunarLander3dViewerSource& operator=(
		const OaLunarLander3dViewerSource&) = delete;

	[[nodiscard]] OaStatus Open(OaEngine& InEngine) override;
	[[nodiscard]] OaStatus Init(
		OaInputSystem& InInput,
		OaFunc<void(bool)> InCapturePointer) override;
	void Update(OaF32 InDeltaMs) override;
	void Render(
		OaUi& InUi,
		const OaTextAtlas& InTextAtlas,
		OaU32 InWidth,
		OaU32 InHeight) override;
	[[nodiscard]] OaEvent RenderReady() const override;
	[[nodiscard]] OaStatus MarkConsumed(
		const OaEvent& InCompletion) override;
	[[nodiscard]] OaStatus Close() override;

private:
	class Impl;
	OaUniquePtr<Impl> Impl_;
};
