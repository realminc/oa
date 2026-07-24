#pragma once

#include <Oa/Core/Status.h>

class OaEngineConfig;

// Process-level SDL video/Vulkan-loader lease used to admit presentation
// capabilities before OaEngine creation. Windows, surfaces, swapchains, and UI
// resources remain owned by each OaViewer session.
class OaViewerPlatformLease {
public:
	OaViewerPlatformLease() = default;
	~OaViewerPlatformLease();

	OaViewerPlatformLease(const OaViewerPlatformLease&) = delete;
	OaViewerPlatformLease& operator=(const OaViewerPlatformLease&) = delete;
	OaViewerPlatformLease(OaViewerPlatformLease&&) = delete;
	OaViewerPlatformLease& operator=(OaViewerPlatformLease&&) = delete;

	[[nodiscard]] OaStatus Acquire(OaEngineConfig* InOutEngineConfig = nullptr);
	void Release() noexcept;
	[[nodiscard]] bool IsAcquired() const noexcept { return Acquired_; }

private:
	bool Acquired_ = false;
};
