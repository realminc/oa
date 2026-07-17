// ═══════════════════════════════════════════════════════════════════════════════
// OA - HIGH-PERFORMANCE MEMORY OPERATIONS
// ═══════════════════════════════════════════════════════════════════════════════
//
// Fully inlined small-copy dispatch with compiler-generated fixed-size moves.
//
// Strategy:
//   1-256B: fixed-size overlapping blocks, emitted by the compiler for the
//             selected target ISA without unaligned typed accesses or raw asm
//   >256B:   platform memcpy (IFUNC/ERMS/vector implementation on glibc)
//   explicit streaming: OaMemcpyStream, only when the destination will not be
//             consumed soon and bypassing the cache is part of the contract
//   OA_MEMCPY_NT_PREFETCH: optional NTA prefetch distance for experiments
//             (default 0; max 8192)
//
// The size dispatch matters when the caller's size is dynamic. For a compile-
// time constant size, both OaMemcpy and std::memcpy reduce to the same moves.
//
// ═══════════════════════════════════════════════════════════════════════════════

#pragma once

#define OA_TYPES_H_SKIP_REST
#include <Oa/Core/Types.h>
#undef OA_TYPES_H_SKIP_REST
#include <atomic>
#include <cstring>

// Explicit non-temporal copy implementation (defined in Memory.cpp).
void* OaMemcpyNT(void* InDst, const void* InSrc, OaUsize InSize);

namespace OaMemoryDetail {

template <OaUsize Size>
__attribute__((always_inline))
inline void CopyBlock(OaByte* InDst, const OaByte* InSrc) {
	static_assert(Size == 1 || Size == 2 || Size == 4 || Size == 8
		|| Size == 16 || Size == 32);
	std::memcpy(InDst, InSrc, Size);
}

} // namespace OaMemoryDetail

// ═══════════════════════════════════════════════════════════════════════════════
// OaMemcpy — Zero-overhead for all sizes
// ═══════════════════════════════════════════════════════════════════════════════

__attribute__((always_inline))
inline void* OaMemcpy(void* __restrict__ InDst, const void* __restrict__ InSrc, OaUsize InSize) {
	if (__builtin_expect(InSize == 0, 0)) return InDst;
	// Let the compiler emit its optimal single sequence when the call site knows
	// the size. The branches below are specifically for dynamic-size callers.
	if (__builtin_constant_p(InSize)) return std::memcpy(InDst, InSrc, InSize);

	OaByte* Dst = static_cast<OaByte*>(InDst);
	const OaByte* Src = static_cast<const OaByte*>(InSrc);
	using OaMemoryDetail::CopyBlock;

	// Fixed-size std::memcpy is a compiler primitive, not a libc call. Keeping
	// these accesses expressed as copies also makes unaligned data legal C++.
	if (__builtin_expect(InSize <= 16, 1)) {
		if (InSize >= 8) {
			CopyBlock<8>(Dst, Src);
			CopyBlock<8>(Dst + InSize - 8, Src + InSize - 8);
		} else if (InSize >= 4) {
			CopyBlock<4>(Dst, Src);
			CopyBlock<4>(Dst + InSize - 4, Src + InSize - 4);
		} else if (InSize >= 2) {
			CopyBlock<2>(Dst, Src);
			CopyBlock<2>(Dst + InSize - 2, Src + InSize - 2);
		} else {
			CopyBlock<1>(Dst, Src);
		}
		return InDst;
	}

	if (InSize <= 32) {
		CopyBlock<16>(Dst, Src);
		CopyBlock<16>(Dst + InSize - 16, Src + InSize - 16);
		return InDst;
	}

	if (InSize <= 64) {
		CopyBlock<32>(Dst, Src);
		CopyBlock<32>(Dst + InSize - 32, Src + InSize - 32);
		return InDst;
	}

	if (InSize <= 128) {
		CopyBlock<32>(Dst, Src);
		CopyBlock<32>(Dst + 32, Src + 32);
		CopyBlock<32>(Dst + InSize - 64, Src + InSize - 64);
		CopyBlock<32>(Dst + InSize - 32, Src + InSize - 32);
		return InDst;
	}

	if (InSize <= 256) {
		CopyBlock<32>(Dst, Src);
		CopyBlock<32>(Dst + 32, Src + 32);
		CopyBlock<32>(Dst + 64, Src + 64);
		CopyBlock<32>(Dst + 96, Src + 96);
		CopyBlock<32>(Dst + InSize - 128, Src + InSize - 128);
		CopyBlock<32>(Dst + InSize - 96, Src + InSize - 96);
		CopyBlock<32>(Dst + InSize - 64, Src + InSize - 64);
		CopyBlock<32>(Dst + InSize - 32, Src + InSize - 32);
		return InDst;
	}

	return std::memcpy(InDst, InSrc, InSize);
}

// ═══════════════════════════════════════════════════════════════════════════════
// OaMemcpyStream — Explicit non-temporal cache policy
// ═══════════════════════════════════════════════════════════════════════════════

inline void* OaMemcpyStream(void* InDst, const void* InSrc, OaUsize InSize) {
	if (InSize == 0 || InDst == InSrc) return InDst;
	return OaMemcpyNT(InDst, InSrc, InSize);
}

// ═══════════════════════════════════════════════════════════════════════════════
// MEMSET / MEMZERO / MEMCMP (defined in memory.cpp)
// ═══════════════════════════════════════════════════════════════════════════════

inline void* OaMemset(void* InDst, OaI32 InValue, OaUsize InSize) {
	return std::memset(InDst, InValue, InSize);
}

inline void* OaMemzero(void* InDst, OaUsize InSize) {
	return std::memset(InDst, 0, InSize);
}

inline OaI32 OaMemcmp(const void* InA, const void* InB, OaUsize InSize) {
	return std::memcmp(InA, InB, InSize);
}

OaBool OaMemEqual(const void* InA, const void* InB, OaUsize InSize);

// ═══════════════════════════════════════════════════════════════════════════════
// ALIGNED ALLOCATION
// ═══════════════════════════════════════════════════════════════════════════════

void* OaAlignedAlloc(OaUsize InSize, OaUsize InAlignment = 64);
void OaAlignedFree(void* InPtr);

// ═══════════════════════════════════════════════════════════════════════════════
// PREFETCH / CACHE CONTROL
// ═══════════════════════════════════════════════════════════════════════════════

inline void OaPrefetchL1(const void* InPtr)    { __builtin_prefetch(InPtr, 0, 3); }
inline void OaPrefetchL2(const void* InPtr)    { __builtin_prefetch(InPtr, 0, 2); }
inline void OaPrefetchWrite(void* InPtr)       { __builtin_prefetch(InPtr, 1, 3); }
inline void OaPrefetchNTA(const void* InPtr)   { __builtin_prefetch(InPtr, 0, 0); }
inline void OaMemoryFence() { __sync_synchronize(); }
inline void OaStoreFence()  { std::atomic_thread_fence(std::memory_order_release); }
inline void OaLoadFence()   { std::atomic_thread_fence(std::memory_order_acquire); }
