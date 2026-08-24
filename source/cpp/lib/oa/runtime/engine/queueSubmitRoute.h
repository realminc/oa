#pragma once

#include <oa/runtime/device.h>

// Pure queue-handle classification shared by the submission dispatcher,
// presentation's unavoidable raw WSI calls, and focused route tests. alias
// priority is deliberate: one physical VkQueue must always map to one mutex.
namespace oavk {

enum class QueueSubmitRoute : oa::U8 {
	Unknown,
	Compute,
	AsyncCompute,
	Transfer,
	Graphics,
	Present,
};

[[nodiscard]] inline QueueSubmitRoute classifyQueueSubmitRoute(
	const oavk::Queues& inQueues,
	const void* inQueue) noexcept
{
	if (inQueue == nullptr) return QueueSubmitRoute::Unknown;
	if (inQueue == inQueues.computeQueue) return QueueSubmitRoute::Compute;
	if (inQueues.hasAsyncCompute and inQueue == inQueues.asyncComputeQueue) {
		return QueueSubmitRoute::AsyncCompute;
	}
	if (inQueue == inQueues.transferQueue) return QueueSubmitRoute::Transfer;
	if (inQueue == inQueues.graphicsQueue) return QueueSubmitRoute::Graphics;
	if (inQueue == inQueues.presentQueue) return QueueSubmitRoute::Present;
	return QueueSubmitRoute::Unknown;
}

} // namespace oavk
