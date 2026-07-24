#pragma once

#include "LunarLander3d.h"

#include <Oa/Core/Status.h>
#include <Oa/Render/FnCamera.h>
#include <Oa/Runtime/OaVk.h>
#include <Oa/Runtime/Sync.h>

#include <vector>

class OaEngine;

// Experimental headless pilot for the scalar Lunar Lander tutorial. This is a
// tutorial-local session rather than a second public renderer abstraction or a
// completed Render A0. It borrows one OaEngine, which must outlive the session,
// and snapshots scalar terrain/state into bounded graphics-only target slots.
class OaLunarLander3dRenderConfig {
public:
	OaU32 Width_ = 256U;
	OaU32 Height_ = 192U;
	OaU32 TargetSlotCount_ = 3U;
	VlmVec4 ClearColor_{0.02F, 0.03F, 0.05F, 1.0F};
};

// Generation-safe handle for one submitted target slot. The Vulkan handles
// are non-owning and remain live only until this exact frame is consumed or
// abandoned. Producer_ is the exact graphics submission completion and must
// not be substituted with a later event from the same stream.
class OaLunarLander3dRenderFrame {
public:
	OaU32 Slot_ = 0U;
	OaU64 SlotGeneration_ = 0U;
	OaU64 TargetGeneration_ = 0U;
	OaU32 Width_ = 0U;
	OaU32 Height_ = 0U;
	VkImage Image_ = VK_NULL_HANDLE;
	VkImageView ImageView_ = VK_NULL_HANDLE;
	VkImageLayout ImageLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
	OaEvent Producer_;
};

// Host-owned copy returned only after waiting the exact producer event and
// invalidating both mapped readback allocations.
class OaLunarLander3dReadback {
public:
	OaU32 Width_ = 0U;
	OaU32 Height_ = 0U;
	std::vector<OaU8> ColorRgba8_;
	std::vector<OaF32> Depth32_;
};

class OaLunarLander3dRenderSession {
public:
	OaLunarLander3dRenderSession(
		const OaLunarLander3dRenderSession&) = delete;
	OaLunarLander3dRenderSession& operator=(
		const OaLunarLander3dRenderSession&) = delete;
	OaLunarLander3dRenderSession(
		OaLunarLander3dRenderSession&&) = delete;
	OaLunarLander3dRenderSession& operator=(
		OaLunarLander3dRenderSession&&) = delete;
	~OaLunarLander3dRenderSession();

	[[nodiscard]] static OaResult<OaUniquePtr<OaLunarLander3dRenderSession>>
		Create(
			OaEngine& InEngine,
			const OaLunarLander3dConfig& InLanderConfig,
			const OaLunarTerrain& InTerrain,
			const OaLunarLander3dRenderConfig& InRenderConfig = {});

	// CPU snapshot boundary. Terrain was copied at Create(); state and camera
	// are copied here before any command is submitted.
	[[nodiscard]] OaStatus BeginFrame(
		const OaLunarLander3dState& InState,
		const OaCameraState& InCamera);
	[[nodiscard]] OaResult<OaLunarLander3dRenderFrame> SubmitFrame(
		OaSpan<const OaEvent> InDependencies = {});
	[[nodiscard]] OaStatus CancelFrame();

	// ConsumeReadback is an explicit host wait. AbandonFrame never waits; a
	// sampled target remains retired until Collect() observes its producer and
	// any registered consumer complete.
	[[nodiscard]] OaResult<OaLunarLander3dReadback> ConsumeReadback(
		const OaLunarLander3dRenderFrame& InFrame);
	// Registers the exact GPU completion that consumed Image_. The consumer
	// must belong to this engine and the graphics queue family; cross-family
	// sampling requires an ownership-transfer path that this pilot does not yet
	// expose. The target is reusable only after both exact events complete.
	[[nodiscard]] OaStatus MarkConsumed(
		const OaLunarLander3dRenderFrame& InFrame,
		const OaEvent& InConsumer);
	[[nodiscard]] OaStatus AbandonFrame(
		const OaLunarLander3dRenderFrame& InFrame);
	[[nodiscard]] OaStatus Collect();

	// Resize is non-waiting and succeeds only when every old-generation slot is
	// free. Successful resize invalidates every previously returned frame.
	[[nodiscard]] OaStatus Resize(OaU32 InWidth, OaU32 InHeight);

	// Mandatory explicit shutdown boundary. It waits only exact outstanding
	// producer/consumer events, releases all target resources, and reports
	// failures.
	[[nodiscard]] OaStatus Close();

	[[nodiscard]] static OaCameraState DefaultCamera(
		OaU32 InWidth, OaU32 InHeight);

private:
	OaLunarLander3dRenderSession() = default;
	class Impl;
	OaUniquePtr<Impl> Impl_;
};
