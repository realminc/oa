// OaVkInferencePool — Pre-allocated activation pool for inference
// OaVkPqcPool — Specialized memory pool for post-quantum crypto

#pragma once

#include <Oa/Core/Types.h>
#include <Oa/Core/Status.h>
#include <Oa/Core/MatrixShape.h>
#include <Oa/Runtime/Allocator.h>
#include <Oa/Crypto/Buffer.h>

// Forward declarations
class OaComputeEngine;
class OaMatrix;
enum class OaScalarType : OaU8;

// OaVkInferencePool
// Eliminates all per-token allocator calls. Activation pointers are stable
// for the session lifetime, enabling pre-recorded Vulkan command buffers.
// Ring buffer for streaming output tokens.

class OaVkInferencePoolConfig {
public:
	OaU32 NumLayers = 0;
	OaU64 ActivationSize = 0;
	OaU32 MaxBatchSize = 1;
	OaU64 OutputRingBytes = 0;
	OaBool PinWeights = true;
};

class OaVkInferencePool {
public:
	OaVkInferencePool() = default;
	~OaVkInferencePool() = default;

	OaVkInferencePool(const OaVkInferencePool&) = delete;
	OaVkInferencePool& operator=(const OaVkInferencePool&) = delete;
	OaVkInferencePool(OaVkInferencePool&&) noexcept = default;
	OaVkInferencePool& operator=(OaVkInferencePool&&) noexcept = default;

	[[nodiscard]] OaStatus Init(OaVma& InAllocator, const OaVkInferencePoolConfig& InConfig);
	void Destroy(OaVma& InAllocator);

	[[nodiscard]] void* GetActivation(OaU32 InLayer, OaU32 InBatchSlot) const;

	[[nodiscard]] void* NextOutputSlot();
	void ConsumeOutputSlot();

	// Best-effort VK_EXT_memory_priority hint (priority=1.0)
	[[nodiscard]] OaStatus PinWeights(OaVkBuffer& InWeightBuffer);

	void ResetSession();

	[[nodiscard]] OaU32 NumLayers() const { return Config_.NumLayers; }
	[[nodiscard]] OaU32 MaxBatchSize() const { return Config_.MaxBatchSize; }
	[[nodiscard]] OaU64 ActivationSize() const { return Config_.ActivationSize; }
	[[nodiscard]] OaU64 TotalActivationBytes() const { return ActivationBuf_.Size; }
	[[nodiscard]] OaU64 TotalOutputRingBytes() const { return OutputRing_.Size; }

private:
	OaVkInferencePoolConfig Config_{};
	OaVkBuffer ActivationBuf_{};
	OaVkBuffer OutputRing_{};
	OaU32 RingHead_ = 0;
	OaU32 RingTail_ = 0;
	OaU32 RingCapacity_ = 0;
	OaU64 SlotBytes_ = 0;
};

// OaVkPqcPool
// NTT slab: MaxConcurrentSigns x 1KB slots, 256B-aligned.
// Hash slab: MaxConcurrentHashes x 64B slots for Keccak state.
// Key alloc: returns OaSecureBuffer (mlock'd, zeroed on free).

class OaVkPqcPoolConfig {
public:
	OaU32 MaxConcurrentSigns = 64;
	OaU32 MaxConcurrentHashes = 64;
	OaU64 NttSlotSize = 1024;
	OaU64 HashSlotSize = 200;
};

class OaVkPqcPool {
public:
	OaVkPqcPool() = default;
	~OaVkPqcPool() = default;

	OaVkPqcPool(const OaVkPqcPool&) = delete;
	OaVkPqcPool& operator=(const OaVkPqcPool&) = delete;

	[[nodiscard]] OaStatus Init(OaVma& InAllocator, const OaVkPqcPoolConfig& InConfig);
	void Destroy(OaVma& InAllocator);

	[[nodiscard]] void* AllocNttSlot();
	void FreeNttSlot(void* InPtr);

	[[nodiscard]] void* AllocHashSlot();
	void FreeHashSlot(void* InPtr);

	[[nodiscard]] OaSecureBuffer AllocKeyBuffer(OaU64 InSize);

	[[nodiscard]] OaU32 NttSlotsUsed() const;
	[[nodiscard]] OaU32 HashSlotsUsed() const;

private:
	OaVkPqcPoolConfig Config_{};
	OaVkBuffer NttBuf_{};
	OaVkBuffer HashBuf_{};
	OaVkBuffer KeyBuf_{};
	OaU64 NttBitmap_ = 0;
	OaU64 HashBitmap_ = 0;
	OaU64 KeyOffset_ = 0;
	OaU64 KeyCapacity_ = 0;
};
