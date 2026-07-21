#pragma once

#include <Oa/Core/Types.h>
#include <Oa/Core/Status.h>
#include <Oa/Runtime/Allocator.h>

class OaEngine;

// GPU work item: shader name index + push constant offset + dispatch dimensions.
// Packed into a GPU-readable buffer for persistent kernel consumption.
class OaWorkItem {
public:
	OaU32 GroupsX = 0;
	OaU32 GroupsY = 0;
	OaU32 GroupsZ = 0;
	OaU32 PushOffset = 0;
};

// Persistent kernel work queue — CPU-visible ring buffer that long-running
// GPU shaders poll for work items. Enables GPU-side scheduling without
// CPU re-submission overhead.
//
// Usage:
//   auto wq = OaWorkQueue::Create(rt, 256);
//   wq.Enqueue({64, 1, 1, 0});
//   wq.Flush(rt);
//   // GPU shader reads WorkItems from the queue buffer
//   wq.Destroy(rt);
//
// The queue buffer layout (host-visible):
//   [0]     = Head (CPU writes, GPU reads)
//   [1]     = Tail (GPU writes, CPU reads)
//   [2]     = Capacity
//   [3]     = Padding
//   [4..N]  = OaWorkItem ring entries
class OaWorkQueue {
public:
	OaVkBuffer QueueBuffer;
	OaVkBuffer PushBuffer;
	OaU32 Capacity = 0;
	OaU32 Head = 0;

	[[nodiscard]] static OaResult<OaWorkQueue> Create(
		OaEngine& InRt, OaU32 InCapacity, OaU32 InMaxPushBytes = 4096);
	void Destroy(OaEngine& InRt);

	// Enqueue a work item. Returns false if queue is full.
	[[nodiscard]] bool Enqueue(OaWorkItem InItem);

	// Write push constant data into the push buffer at InOffset.
	void WritePush(OaU32 InOffset, const void* InData, OaU32 InSize);

	// Flush: update the GPU-visible head pointer so the persistent kernel sees new work.
	void Flush();

	// Check how many items the GPU has consumed (reads tail pointer).
	[[nodiscard]] OaU32 Consumed() const;
};
