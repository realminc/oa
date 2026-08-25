#pragma once

#include <oa/core/status.h>

namespace oa { class EngineConfig; }

// Process-level SDL video/vulkan-loader lease used to admit presentation
// capabilities before oa::Engine creation. Windows, surfaces, swapchains, and UI
// resources remain owned by the borrowing presentation session.
namespace oa {

class PresentationPlatformLease {
public:
	PresentationPlatformLease() = default;
	~PresentationPlatformLease();

	PresentationPlatformLease(const PresentationPlatformLease&) = delete;
	PresentationPlatformLease& operator=(const PresentationPlatformLease&) = delete;
	PresentationPlatformLease(PresentationPlatformLease&&) = delete;
	PresentationPlatformLease& operator=(PresentationPlatformLease&&) = delete;

	[[nodiscard]] oa::Status acquire(oa::EngineConfig* inOutEngineConfig = nullptr);
	void release() noexcept;
	[[nodiscard]] bool isAcquired() const noexcept { return acquired_; }

private:
	bool acquired_ = false;
};

} // namespace oa
