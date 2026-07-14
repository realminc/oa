#include <Oa/Runtime/Sync.h>
#include <Oa/Runtime/Device.h>
#include <Oa/Runtime/OaVk.h>

// ─── OaVkFence ──────────────────────────────────────────────────────────────

OaResult<OaVkFence> OaVkFence::Create(const OaVkDevice& InDevice, OaBool InSignaled) {
	VkFenceCreateInfo ci{};
	ci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
	if (InSignaled) ci.flags = VK_FENCE_CREATE_SIGNALED_BIT;

	VkFence fence = VK_NULL_HANDLE;
	VkResult r = vkCreateFence(static_cast<VkDevice>(InDevice.Device), &ci, nullptr, &fence);
	if (r != VK_SUCCESS) {
		return OaStatus::Error(OaStatusCode::VulkanError, "vkCreateFence failed");
	}

	OaVkFence f;
	f.Fence = fence;
	return f;
}

void OaVkFence::Destroy(const OaVkDevice& InDevice) {
	if (Fence) {
		vkDestroyFence(static_cast<VkDevice>(InDevice.Device),
			static_cast<VkFence>(Fence), nullptr);
		Fence = nullptr;
	}
}

OaStatus OaVkFence::Wait(const OaVkDevice& InDevice, OaU64 InTimeoutNs) {
	VkFence fence = static_cast<VkFence>(Fence);
	VkResult r = vkWaitForFences(
		static_cast<VkDevice>(InDevice.Device), 1, &fence, VK_TRUE, InTimeoutNs);
	if (r == VK_TIMEOUT) return OaStatus::Error(OaStatusCode::Timeout, "fence wait timed out");
	if (r != VK_SUCCESS) return OaStatus::Error(OaStatusCode::VulkanError, "vkWaitForFences failed");
	return OaStatus::Ok();
}

OaBool OaVkFence::IsSignaled(const OaVkDevice& InDevice) const {
	VkFence fence = static_cast<VkFence>(Fence);
	VkResult r = vkGetFenceStatus(static_cast<VkDevice>(InDevice.Device), fence);
	return r == VK_SUCCESS;
}

void OaVkFence::Reset(const OaVkDevice& InDevice) {
	VkFence fence = static_cast<VkFence>(Fence);
	vkResetFences(static_cast<VkDevice>(InDevice.Device), 1, &fence);
}

// ─── OaVkTimelineSemaphore ──────────────────────────────────────────────────

OaResult<OaVkTimelineSemaphore> OaVkTimelineSemaphore::Create(
	const OaVkDevice& InDevice, OaU64 InInitialValue)
{
	VkSemaphoreTypeCreateInfo typeCI{};
	typeCI.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
	typeCI.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
	typeCI.initialValue = InInitialValue;

	VkSemaphoreCreateInfo ci{};
	ci.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
	ci.pNext = &typeCI;

	VkSemaphore sem = VK_NULL_HANDLE;
	VkResult r = vkCreateSemaphore(
		static_cast<VkDevice>(InDevice.Device), &ci, nullptr, &sem);
	if (r != VK_SUCCESS) {
		return OaStatus::Error(OaStatusCode::VulkanError, "vkCreateSemaphore (timeline) failed");
	}

	OaVkTimelineSemaphore ts;
	ts.Semaphore = sem;
	return ts;
}

void OaVkTimelineSemaphore::Destroy(const OaVkDevice& InDevice) {
	if (Semaphore) {
		vkDestroySemaphore(static_cast<VkDevice>(InDevice.Device),
			static_cast<VkSemaphore>(Semaphore), nullptr);
		Semaphore = nullptr;
	}
}

OaStatus OaVkTimelineSemaphore::Wait(
	const OaVkDevice& InDevice, OaU64 InValue, OaU64 InTimeoutNs) const
{
	VkSemaphore sem = static_cast<VkSemaphore>(Semaphore);
	VkSemaphoreWaitInfo wi{};
	wi.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO;
	wi.semaphoreCount = 1;
	wi.pSemaphores = &sem;
	wi.pValues = &InValue;

	VkResult r = vkWaitSemaphores(
		static_cast<VkDevice>(InDevice.Device), &wi, InTimeoutNs);
	if (r == VK_TIMEOUT) return OaStatus::Error(OaStatusCode::Timeout, "timeline semaphore wait timed out");
	if (r != VK_SUCCESS) return OaStatus::Error(OaStatusCode::VulkanError, "vkWaitSemaphores failed");
	return OaStatus::Ok();
}

OaU64 OaVkTimelineSemaphore::GetValue(const OaVkDevice& InDevice) const {
	OaU64 value = 0;
	vkGetSemaphoreCounterValue(
		static_cast<VkDevice>(InDevice.Device),
		static_cast<VkSemaphore>(Semaphore), &value);
	return value;
}

OaStatus OaCompletionToken::Wait(OaU64 InTimeoutNs) const
{
	if (not IsValid()) return OaStatus::Ok();
	return Semaphore_->Wait(*Device_, Value_, InTimeoutNs);
}

OaBool OaCompletionToken::IsComplete() const
{
	return not IsValid() || Semaphore_->GetValue(*Device_) >= Value_;
}
