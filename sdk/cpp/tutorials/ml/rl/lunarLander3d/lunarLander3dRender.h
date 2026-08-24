#pragma once

#include <ml/rl/lunarLander3d.h>
#include <oa/render.h>

namespace oa { class Engine; }

// Tutorial-local scene adapter over the reusable oa::Renderer session. It owns
// Lunar-specific terrain, pad, lander geometry, palette, and camera defaults;
// target allocation, submission, synchronization, readback, and retirement
// remain source/Render responsibilities.
class LunarLander3dRenderConfig {
public:
	oa::U32 width_ = 256U;
	oa::U32 height_ = 192U;
	oa::U32 targetSlotCount_ = 3U;
	oa::U32 sampleCount_ = 1U;
	oa::vlm::Vec4 clearColor_{0.02F, 0.03F, 0.05F, 1.0F};
};

class LunarLander3dRenderSession {
public:
	LunarLander3dRenderSession(const LunarLander3dRenderSession&) = delete;
	LunarLander3dRenderSession& operator=(const LunarLander3dRenderSession&) = delete;
	LunarLander3dRenderSession(LunarLander3dRenderSession&&) = delete;
	LunarLander3dRenderSession& operator=(LunarLander3dRenderSession&&) = delete;
	~LunarLander3dRenderSession();

	[[nodiscard]] static oa::Result<oa::UniquePtr<LunarLander3dRenderSession>>
		create(
			oa::Engine& inEngine,
			const oa::LunarLander3dConfig& inLanderConfig,
			const oa::LunarTerrain& inTerrain,
			const LunarLander3dRenderConfig& inRenderConfig = {}
		);

	// CPU snapshot boundary. Terrain was copied at create(); state and camera
	// are copied here before any command is submitted.
	[[nodiscard]] oa::Status beginFrame(const oa::LunarLander3dState& inState, const oa::CameraState& inCamera);
	[[nodiscard]] oa::Result<oa::RenderFrame> submitFrame(oa::Span<const oa::Event> inDependencies = {});
	[[nodiscard]] oa::Status cancelFrame();

	// consumeReadback is an explicit host wait. abandonFrame never waits; a
	// sampled target remains retired until collect() observes its producer and
	// any registered consumer complete.
	[[nodiscard]] oa::Result<oa::RenderReadback> consumeReadback(const oa::RenderFrame& inFrame);
	// Registers the exact GPU completion that consumed image_. The consumer
	// must belong to this engine and the graphics queue family; cross-family
	// sampling requires an ownership-transfer path that this pilot does not yet
	// expose. The target is reusable only after both exact events complete.
	[[nodiscard]] oa::Status markConsumed(const oa::RenderFrame& inFrame, const oa::Event& inConsumer);
	[[nodiscard]] oa::Status abandonFrame(const oa::RenderFrame& inFrame);
	[[nodiscard]] oa::Status collect();

	// resize is non-waiting and succeeds only when every old-generation slot is
	// free. Successful resize invalidates every previously returned frame.
	[[nodiscard]] oa::Status resize(oa::U32 inWidth, oa::U32 inHeight);

	// Mandatory explicit shutdown boundary. It waits only exact outstanding
	// producer/consumer events, releases all target resources, and reports
	// failures.
	[[nodiscard]] oa::Status close();

	[[nodiscard]] static oa::CameraState defaultCamera(
		oa::U32 inWidth, oa::U32 inHeight);

private:
	LunarLander3dRenderSession() = default;
	class Impl;
	oa::UniquePtr<Impl> impl_;
};
