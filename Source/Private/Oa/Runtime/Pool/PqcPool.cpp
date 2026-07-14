#include <Oa/Runtime/Pool.h>
#include <Oa/Core/Memory.h>

OaStatus OaVkPqcPool::Init(OaVma& InAllocator, const OaVkPqcPoolConfig& InConfig) {
	Config_ = InConfig;

	OaU32 nttCap = Config_.MaxConcurrentSigns > 64 ? 64 : Config_.MaxConcurrentSigns;
	OaU32 hashCap = Config_.MaxConcurrentHashes > 64 ? 64 : Config_.MaxConcurrentHashes;

	OaU64 alignedSlotSize = OaAlignUp(Config_.NttSlotSize, OaUsize(256));
	OaU64 nttTotal = alignedSlotSize * static_cast<OaU64>(nttCap);
	auto nttResult = InAllocator.AllocHostVisible(nttTotal);
	if (!nttResult.IsOk()) {
		return nttResult.GetStatus();
	}
	NttBuf_ = nttResult.GetValue();
	NttBitmap_ = nttCap == 64 ? ~OaU64(0) : ((OaU64(1) << nttCap) - 1);

	OaU64 hashTotal = Config_.HashSlotSize * static_cast<OaU64>(hashCap);
	auto hashResult = InAllocator.AllocHostVisible(hashTotal);
	if (!hashResult.IsOk()) {
		InAllocator.Free(NttBuf_);
		return hashResult.GetStatus();
	}
	HashBuf_ = hashResult.GetValue();
	HashBitmap_ = hashCap == 64 ? ~OaU64(0) : ((OaU64(1) << hashCap) - 1);

	KeyCapacity_ = OaU64(64) * 1024;
	auto keyResult = InAllocator.AllocHostVisible(KeyCapacity_);
	if (!keyResult.IsOk()) {
		InAllocator.Free(NttBuf_);
		InAllocator.Free(HashBuf_);
		return keyResult.GetStatus();
	}
	KeyBuf_ = keyResult.GetValue();
	KeyOffset_ = 0;

	return OaStatus::Ok();
}

void OaVkPqcPool::Destroy(OaVma& InAllocator) {
	if (KeyBuf_.MappedPtr != nullptr && KeyBuf_.Size > 0) {
		OaSecureBuffer tempSec(KeyBuf_.MappedPtr, KeyBuf_.Size);
		tempSec.Reset();
	}

	if (NttBuf_.Buffer != nullptr) {
		InAllocator.Free(NttBuf_);
	}
	if (HashBuf_.Buffer != nullptr) {
		InAllocator.Free(HashBuf_);
	}
	if (KeyBuf_.Buffer != nullptr) {
		InAllocator.Free(KeyBuf_);
	}

	Config_ = {};
	NttBitmap_ = 0;
	HashBitmap_ = 0;
	KeyOffset_ = 0;
	KeyCapacity_ = 0;
}

void* OaVkPqcPool::AllocNttSlot() {
	if (NttBitmap_ == 0) {
		return nullptr;
	}
	OaU32 slot;
	__asm__ __volatile__("tzcntq %1, %q0" : "=r"(slot) : "r"(NttBitmap_));
	NttBitmap_ &= NttBitmap_ - 1;
	OaU64 alignedSlotSize = OaAlignUp(Config_.NttSlotSize, OaUsize(256));
	return static_cast<OaU8*>(NttBuf_.MappedPtr) + (static_cast<OaU64>(slot) * alignedSlotSize);
}

void OaVkPqcPool::FreeNttSlot(void* InPtr) {
	OaU64 alignedSlotSize = OaAlignUp(Config_.NttSlotSize, OaUsize(256));
	OaUsize offset = static_cast<OaU8*>(InPtr) - static_cast<OaU8*>(NttBuf_.MappedPtr);
	OaU32 slot = static_cast<OaU32>(offset / alignedSlotSize);
	NttBitmap_ |= (OaU64(1) << slot);
}

void* OaVkPqcPool::AllocHashSlot() {
	if (HashBitmap_ == 0) {
		return nullptr;
	}
	OaU32 slot;
	__asm__ __volatile__("tzcntq %1, %q0" : "=r"(slot) : "r"(HashBitmap_));
	HashBitmap_ &= HashBitmap_ - 1;
	return static_cast<OaU8*>(HashBuf_.MappedPtr) + (static_cast<OaU64>(slot) * Config_.HashSlotSize);
}

void OaVkPqcPool::FreeHashSlot(void* InPtr) {
	OaUsize offset = static_cast<OaU8*>(InPtr) - static_cast<OaU8*>(HashBuf_.MappedPtr);
	OaU32 slot = static_cast<OaU32>(offset / Config_.HashSlotSize);
	HashBitmap_ |= (OaU64(1) << slot);
}

OaSecureBuffer OaVkPqcPool::AllocKeyBuffer(OaU64 InSize) {
	OaU64 aligned = OaAlignUp(InSize, OaUsize(64));
	if (KeyOffset_ + aligned > KeyCapacity_) {
		return OaSecureBuffer{};
	}
	void* ptr = static_cast<OaU8*>(KeyBuf_.MappedPtr) + KeyOffset_;
	KeyOffset_ += aligned;
	return OaSecureBuffer(ptr, InSize);
}

OaU32 OaVkPqcPool::NttSlotsUsed() const {
	OaU32 cap = Config_.MaxConcurrentSigns > 64 ? 64 : Config_.MaxConcurrentSigns;
	OaU64 count;
	__asm__ __volatile__("popcntq %1, %0" : "=r"(count) : "r"(NttBitmap_));
	return cap - static_cast<OaU32>(count);
}

OaU32 OaVkPqcPool::HashSlotsUsed() const {
	OaU32 cap = Config_.MaxConcurrentHashes > 64 ? 64 : Config_.MaxConcurrentHashes;
	OaU64 count;
	__asm__ __volatile__("popcntq %1, %0" : "=r"(count) : "r"(HashBitmap_));
	return cap - static_cast<OaU32>(count);
}
