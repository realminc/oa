#include <oa/runtime/sync.h>
#include <oa/runtime/device.h>
#include <vkl/vkl.h>

// ─── oavk::Fence ──────────────────────────────────────────────────────────────

oa::Result<oavk::Fence> oavk::Fence::create(const oavk::Device& inDevice, oa::Bool inSignaled) {
	VkFenceCreateInfo ci{};
	ci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
	if (inSignaled) ci.flags = VK_FENCE_CREATE_SIGNALED_BIT;

	VkFence fence = VK_NULL_HANDLE;
	VkResult r = inDevice.deviceDispatch.vkCreateFence(static_cast<VkDevice>(inDevice.device), &ci, nullptr, &fence);
	if (r != VK_SUCCESS) {
		return oa::Status::error(oa::StatusCode::VulkanError, "vkCreateFence failed");
	}

	oavk::Fence f;
	f.fence = fence;
	return f;
}

void oavk::Fence::destroy(const oavk::Device& inDevice) {
	if (fence) {
		inDevice.deviceDispatch.vkDestroyFence(static_cast<VkDevice>(inDevice.device),
			static_cast<VkFence>(fence), nullptr);
		fence = nullptr;
	}
}

oa::Status oavk::Fence::wait(const oavk::Device& inDevice, oa::U64 inTimeoutNs) {
	VkFence vkFence = static_cast<VkFence>(fence);
	VkResult r = inDevice.deviceDispatch.vkWaitForFences(
		static_cast<VkDevice>(inDevice.device), 1, &vkFence, VK_TRUE, inTimeoutNs);
	if (r == VK_TIMEOUT) return oa::Status::error(oa::StatusCode::Timeout, "fence wait timed out");
	if (r != VK_SUCCESS) return oa::Status::error(oa::StatusCode::VulkanError, "vkWaitForFences failed");
	return oa::Status::ok();
}

oa::Bool oavk::Fence::isSignaled(const oavk::Device& inDevice) const {
	VkFence vkFence = static_cast<VkFence>(fence);
	VkResult r = inDevice.deviceDispatch.vkGetFenceStatus(
		static_cast<VkDevice>(inDevice.device), vkFence);
	return r == VK_SUCCESS;
}

void oavk::Fence::reset(const oavk::Device& inDevice) {
	VkFence vkFence = static_cast<VkFence>(fence);
	inDevice.deviceDispatch.vkResetFences(
		static_cast<VkDevice>(inDevice.device), 1, &vkFence);
}

// ─── oavk::TimelineSemaphore ──────────────────────────────────────────────────

oa::Result<oavk::TimelineSemaphore> oavk::TimelineSemaphore::create(
	const oavk::Device& inDevice, oa::U64 inInitialValue)
{
	VkSemaphoreTypeCreateInfo typeCI{};
	typeCI.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
	typeCI.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
	typeCI.initialValue = inInitialValue;

	VkSemaphoreCreateInfo ci{};
	ci.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
	ci.pNext = &typeCI;

	VkSemaphore sem = VK_NULL_HANDLE;
	VkResult r = inDevice.deviceDispatch.vkCreateSemaphore(
		static_cast<VkDevice>(inDevice.device), &ci, nullptr, &sem);
	if (r != VK_SUCCESS) {
		return oa::Status::error(oa::StatusCode::VulkanError, "vkCreateSemaphore (timeline) failed");
	}

	oavk::TimelineSemaphore ts;
	ts.semaphore = sem;
	return ts;
}

void oavk::TimelineSemaphore::destroy(const oavk::Device& inDevice) {
	if (semaphore) {
		inDevice.deviceDispatch.vkDestroySemaphore(static_cast<VkDevice>(inDevice.device),
			static_cast<VkSemaphore>(semaphore), nullptr);
		semaphore = nullptr;
	}
}

oa::Status oavk::TimelineSemaphore::wait(
	const oavk::Device& inDevice, oa::U64 inValue, oa::U64 inTimeoutNs) const
{
	VkSemaphore sem = static_cast<VkSemaphore>(semaphore);
	VkSemaphoreWaitInfo wi{};
	wi.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO;
	wi.semaphoreCount = 1;
	wi.pSemaphores = &sem;
	wi.pValues = &inValue;

	VkResult r = inDevice.deviceDispatch.vkWaitSemaphores(
		static_cast<VkDevice>(inDevice.device), &wi, inTimeoutNs);
	if (r == VK_TIMEOUT) return oa::Status::error(oa::StatusCode::Timeout, "timeline semaphore wait timed out");
	if (r != VK_SUCCESS) return oa::Status::error(oa::StatusCode::VulkanError, "vkWaitSemaphores failed");
	return oa::Status::ok();
}

oa::U64 oavk::TimelineSemaphore::getValue(const oavk::Device& inDevice) const {
	oa::U64 value = 0;
	inDevice.deviceDispatch.vkGetSemaphoreCounterValue(
		static_cast<VkDevice>(inDevice.device),
		static_cast<VkSemaphore>(semaphore), &value);
	return value;
}

oa::Status oa::Event::wait(oa::U64 inTimeoutNs) const
{
	if (not isValid()) return oa::Status::ok();
	oavk::TimelineSemaphore semaphore{.semaphore = semaphore_};
	return semaphore.wait(
		*static_cast<const oavk::Device*>(device_), value_, inTimeoutNs);
}

oa::Bool oa::Event::isComplete() const
{
	if (not isValid()) return true;
	oavk::TimelineSemaphore semaphore{.semaphore = semaphore_};
	return semaphore.getValue(*static_cast<const oavk::Device*>(device_)) >= value_;
}
