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
// only at a true CPU boundary. The underlying Vulkan semaphore and device must
// outlive the token; the OaVkTimelineSemaphore wrapper used to create it need
// not do so.
class OaCompletionToken {
public:
	OaCompletionToken() = default;
	OaCompletionToken(
		const OaVkDevice& InDevice,
		const OaVkTimelineSemaphore& InSemaphore,
		OaU64 InValue) noexcept
		: Device_(&InDevice), Value_(InValue) {
		Semaphore_.Semaphore = InSemaphore.Semaphore;
	}

	[[nodiscard]] bool IsValid() const noexcept {
		return Device_ != nullptr and Semaphore_.Semaphore != nullptr
			and Value_ != 0U;
	}
	[[nodiscard]] OaStatus Wait(OaU64 InTimeoutNs = UINT64_MAX) const;
	[[nodiscard]] OaBool IsComplete() const;
	[[nodiscard]] OaVkTimelineWait TimelineWait() const noexcept {
		return {Semaphore_.Semaphore != nullptr ? &Semaphore_ : nullptr, Value_};
	}
	[[nodiscard]] const OaVkTimelineSemaphore* Semaphore() const noexcept {
		return Semaphore_.Semaphore != nullptr ? &Semaphore_ : nullptr;
	}
	[[nodiscard]] OaU64 Value() const noexcept { return Value_; }
	[[nodiscard]] bool IsSameCompletion(
		const OaCompletionToken& InOther) const noexcept {
		return IsValid() and InOther.IsValid()
			and Device_ == InOther.Device_
			and Semaphore_.Semaphore == InOther.Semaphore_.Semaphore
			and Value_ == InOther.Value_;
	}

private:
	const OaVkDevice* Device_ = nullptr;
	// Non-owning handle snapshot. Keeping this view inside the token makes
	// copied events independent of the source wrapper object's address.
	OaVkTimelineSemaphore Semaphore_;
	OaU64 Value_ = 0U;
};

// Canonical completion name for new execution APIs. OaCompletionToken remains
// as a compatibility spelling while existing media consumers migrate.
using OaEvent = OaCompletionToken;
