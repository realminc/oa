#pragma once

#include <Oa/Core/Types.h>
#include <Oa/Core/Status.h>

class OaVkDevice;

class OaVkFence {
public:
	void* Fence = nullptr;

	[[nodiscard]] static OaResult<OaVkFence> Create(const OaVkDevice& InDevice, OaBool InSignaled = false);
	void Destroy(const OaVkDevice& InDevice);
	[[nodiscard]] OaStatus Wait(const OaVkDevice& InDevice, OaU64 InTimeoutNs = UINT64_MAX);
	[[nodiscard]] OaBool IsSignaled(const OaVkDevice& InDevice) const;
	void Reset(const OaVkDevice& InDevice);
};

// Monotonically increasing counter semaphore (Vulkan 1.2 core).
// No reset needed — value only goes up. Host-waitable.
// One semaphore per stream for its entire lifetime.
class OaVkTimelineSemaphore {
public:
	void* Semaphore = nullptr;

	[[nodiscard]] static OaResult<OaVkTimelineSemaphore> Create(
		const OaVkDevice& InDevice, OaU64 InInitialValue = 0
	);
	void Destroy(const OaVkDevice& InDevice);

	[[nodiscard]] OaStatus Wait(const OaVkDevice& InDevice, OaU64 InValue, OaU64 InTimeoutNs = UINT64_MAX) const;
	[[nodiscard]] OaU64 GetValue(const OaVkDevice& InDevice) const;
};

// A GPU-side timeline dependency consumed by queue submission helpers.
// Keeping the semaphore and value together avoids accidentally host-waiting
// without establishing the device-memory dependency required across queues.
struct OaVkTimelineWait {
	const OaVkTimelineSemaphore* Semaphore = nullptr;
	OaU64 Value = 0;
};

// Non-owning completion value shared by compute, image, video and capture
// paths. A token is a GPU dependency first and a host wait handle second:
// pass TimelineWait() to another queue whenever possible, and call Wait()
// only at a true CPU boundary. The semaphore owner must outlive the token.
class OaCompletionToken {
public:
	OaCompletionToken() = default;
	OaCompletionToken(
		const OaVkDevice& InDevice,
		const OaVkTimelineSemaphore& InSemaphore,
		OaU64 InValue) noexcept
		: Device_(&InDevice), Semaphore_(&InSemaphore), Value_(InValue) {}

	[[nodiscard]] bool IsValid() const noexcept {
		return Device_ != nullptr && Semaphore_ != nullptr
			&& Semaphore_->Semaphore != nullptr && Value_ != 0U;
	}
	[[nodiscard]] OaStatus Wait(OaU64 InTimeoutNs = UINT64_MAX) const;
	[[nodiscard]] OaBool IsComplete() const;
	[[nodiscard]] OaVkTimelineWait TimelineWait() const noexcept {
		return {Semaphore_, Value_};
	}
	[[nodiscard]] const OaVkTimelineSemaphore* Semaphore() const noexcept {
		return Semaphore_;
	}
	[[nodiscard]] OaU64 Value() const noexcept { return Value_; }

private:
	const OaVkDevice* Device_ = nullptr;
	const OaVkTimelineSemaphore* Semaphore_ = nullptr;
	OaU64 Value_ = 0U;
};
