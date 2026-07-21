#include <Oa/Runtime/Persistent.h>
#include <Oa/Runtime/Engine.h>
#include <Oa/Core/Memory.h>

struct OaWorkQueueHeader {
	OaU32 Head;
	OaU32 Tail;
	OaU32 Capacity;
	OaU32 Pad;
};

OaResult<OaWorkQueue> OaWorkQueue::Create(
	OaEngine& InRt, OaU32 InCapacity, OaU32 InMaxPushBytes)
{
	OaU64 queueBytes = sizeof(OaWorkQueueHeader) + InCapacity * sizeof(OaWorkItem);
	auto queueBuf = InRt.AllocBuffer(queueBytes);
	if (!queueBuf.IsOk()) return queueBuf.GetStatus();

	auto pushBuf = InRt.AllocBuffer(InMaxPushBytes);
	if (!pushBuf.IsOk()) {
		InRt.FreeBuffer(queueBuf.GetValue());
		return pushBuf.GetStatus();
	}

	auto& qb = queueBuf.GetValue();
	auto* header = static_cast<OaWorkQueueHeader*>(qb.MappedPtr);
	header->Head = 0;
	header->Tail = 0;
	header->Capacity = InCapacity;
	header->Pad = 0;

	OaWorkQueue wq;
	wq.QueueBuffer = qb;
	wq.PushBuffer = pushBuf.GetValue();
	wq.Capacity = InCapacity;
	wq.Head = 0;
	return wq;
}

void OaWorkQueue::Destroy(OaEngine& InRt) {
	InRt.FreeBuffer(QueueBuffer);
	InRt.FreeBuffer(PushBuffer);
	Capacity = 0;
	Head = 0;
}

bool OaWorkQueue::Enqueue(OaWorkItem InItem) {
	auto* header = static_cast<OaWorkQueueHeader*>(QueueBuffer.MappedPtr);
	OaU32 tail = header->Tail;
	OaU32 next = (Head + 1) % Capacity;
	if (next == tail) return false;

	auto* items = reinterpret_cast<OaWorkItem*>(
		static_cast<OaU8*>(QueueBuffer.MappedPtr) + sizeof(OaWorkQueueHeader));
	items[Head] = InItem;
	Head = next;
	return true;
}

void OaWorkQueue::WritePush(OaU32 InOffset, const void* InData, OaU32 InSize) {
	auto* dst = static_cast<OaU8*>(PushBuffer.MappedPtr) + InOffset;
	OaMemcpy(dst, InData, InSize);
}

void OaWorkQueue::Flush() {
	auto* header = static_cast<OaWorkQueueHeader*>(QueueBuffer.MappedPtr);
	header->Head = Head;
}

OaU32 OaWorkQueue::Consumed() const {
	auto* header = static_cast<const OaWorkQueueHeader*>(QueueBuffer.MappedPtr);
	return header->Tail;
}
