#include <Oa/Runtime/Pool.h>
#include <Oa/Core/Memory.h>

OaStatus OaVkInferencePool::Init(OaVma& InAllocator, const OaVkInferencePoolConfig& InConfig) {
	Config_ = InConfig;

	if (Config_.NumLayers == 0 || Config_.ActivationSize == 0) {
		return OaStatus::Error(OaStatusCode::InvalidArgument, "pool requires NumLayers > 0 and ActivationSize > 0");
	}

	// Single contiguous buffer: [NumLayers][MaxBatchSize] x ActivationSize
	OaU64 totalActBytes = static_cast<OaU64>(Config_.NumLayers) * Config_.MaxBatchSize * Config_.ActivationSize;
	auto actResult = InAllocator.AllocBar(totalActBytes);
	if (!actResult.IsOk()) {
		return actResult.GetStatus();
	}
	ActivationBuf_ = actResult.GetValue();
	OaMemzero(ActivationBuf_.MappedPtr, totalActBytes);

	// Output ring buffer
	if (Config_.OutputRingBytes > 0) {
		auto ringResult = InAllocator.AllocHostVisible(Config_.OutputRingBytes);
		if (!ringResult.IsOk()) {
			InAllocator.Free(ActivationBuf_);
			return ringResult.GetStatus();
		}
		OutputRing_ = ringResult.GetValue();

		// Each slot holds one token's worth of output (e.g. vocab_size * sizeof(float))
		// For now, fixed slot size = ActivationSize (overrideable later)
		SlotBytes_ = Config_.ActivationSize;
		RingCapacity_ = SlotBytes_ > 0 ? static_cast<OaU32>(Config_.OutputRingBytes / SlotBytes_) : 0;
		RingHead_ = 0;
		RingTail_ = 0;
	}

	return OaStatus::Ok();
}

void OaVkInferencePool::Destroy(OaVma& InAllocator) {
	if (ActivationBuf_.Buffer) {
		InAllocator.Free(ActivationBuf_);
	}
	if (OutputRing_.Buffer) {
		InAllocator.Free(OutputRing_);
	}
	Config_ = {};
	RingHead_ = 0;
	RingTail_ = 0;
	RingCapacity_ = 0;
	SlotBytes_ = 0;
}

void* OaVkInferencePool::GetActivation(OaU32 InLayer, OaU32 InBatchSlot) const {
	OaU64 offset = (static_cast<OaU64>(InLayer) * Config_.MaxBatchSize + InBatchSlot) * Config_.ActivationSize;
	return static_cast<OaU8*>(ActivationBuf_.MappedPtr) + offset;
}

void* OaVkInferencePool::NextOutputSlot() {
	if (RingCapacity_ == 0 || !OutputRing_.MappedPtr) return nullptr;
	OaU64 offset = static_cast<OaU64>(RingHead_) * SlotBytes_;
	return static_cast<OaU8*>(OutputRing_.MappedPtr) + offset;
}

void OaVkInferencePool::ConsumeOutputSlot() {
	if (RingCapacity_ == 0) return;
	RingHead_ = (RingHead_ + 1) % RingCapacity_;
}

OaStatus OaVkInferencePool::PinWeights(OaVkBuffer& InWeightBuffer) {
	// VK_EXT_memory_priority is applied at allocation time via VMA flags.
	// For existing buffers, this is best-effort. The priority hint can only
	// be set during vmaCreateBuffer — re-binding with priority requires
	// reallocating. For now, this is a no-op placeholder that returns Ok.
	// Future: AllocPinned() variant that sets priority=1.0 at alloc time.
	(void)InWeightBuffer;
	return OaStatus::Ok();
}

void OaVkInferencePool::ResetSession() {
	if (ActivationBuf_.MappedPtr) {
		OaMemzero(ActivationBuf_.MappedPtr, ActivationBuf_.Size);
	}
	RingHead_ = 0;
	RingTail_ = 0;
}
