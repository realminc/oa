#pragma once

#include <Oa/Runtime/Device.h>

// Pure queue-handle classification shared by the submission dispatcher,
// presentation's unavoidable raw WSI calls, and focused route tests. Alias
// priority is deliberate: one physical VkQueue must always map to one mutex.
enum class OaQueueSubmitRoute : OaU8 {
	Unknown,
	Compute,
	AsyncCompute,
	Transfer,
	Graphics,
	Present,
};

[[nodiscard]] inline OaQueueSubmitRoute OaClassifyQueueSubmitRoute(
	const OaVkQueues& InQueues,
	const void* InQueue) noexcept
{
	if (InQueue == nullptr) return OaQueueSubmitRoute::Unknown;
	if (InQueue == InQueues.ComputeQueue) return OaQueueSubmitRoute::Compute;
	if (InQueues.HasAsyncCompute and InQueue == InQueues.AsyncComputeQueue) {
		return OaQueueSubmitRoute::AsyncCompute;
	}
	if (InQueue == InQueues.TransferQueue) return OaQueueSubmitRoute::Transfer;
	if (InQueue == InQueues.GraphicsQueue) return OaQueueSubmitRoute::Graphics;
	if (InQueue == InQueues.PresentQueue) return OaQueueSubmitRoute::Present;
	return OaQueueSubmitRoute::Unknown;
}
