#pragma once

#include <oa/core/status.h>

namespace oa { class EngineConfig; }

// process-level SDL video/vulkan-loader lease used to admit presentation
// capabilities before oa::Engine creation. Windows, surfaces, swapchains, and UI
// resources remain owned by each Viewer session.
namespace oa {

class ViewerPlatformLease {
public:
	ViewerPlatformLease() = default;
	~ViewerPlatformLease();

	ViewerPlatformLease(const ViewerPlatformLease&) = delete;
	ViewerPlatformLease& operator=(const ViewerPlatformLease&) = delete;
	ViewerPlatformLease(ViewerPlatformLease&&) = delete;
	ViewerPlatformLease& operator=(ViewerPlatformLease&&) = delete;

	[[nodiscard]] oa::Status acquire(oa::EngineConfig* inOutEngineConfig = nullptr);
	void release() noexcept;
	[[nodiscard]] bool isAcquired() const noexcept { return acquired_; }

private:
	bool acquired_ = false;
};

}  // namespace oa
