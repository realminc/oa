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
//   explicit streaming: memcpyStream, only when the destination will not be
//             consumed soon and bypassing the cache is part of the contract
//   OA_MEMCPY_NT_PREFETCH: optional NTA prefetch distance for experiments
//             (default 0; max 8192)
//
// The size dispatch matters when the caller's size is dynamic. For a compile-
// time constant size, both memcpy and std::memcpy reduce to the same moves.
//
// ═══════════════════════════════════════════════════════════════════════════════

#pragma once

#define OA_TYPES_H_SKIP_REST
#include <oa/core/types.h>
#undef OA_TYPES_H_SKIP_REST
#include <atomic>
#include <cstring>

namespace oa {

// Explicit non-temporal copy implementation (defined in memory.cpp).
void* memcpyNt(void* inDst, const void* inSrc, oa::Usize inSize);

namespace detail {

template <oa::Usize size>
__attribute__((always_inline))
inline void copyBlock(oa::Byte* inDst, const oa::Byte* inSrc) {
	static_assert(size == 1 || size == 2 || size == 4 || size == 8
		|| size == 16 || size == 32);
	std::memcpy(inDst, inSrc, size);
}

} // namespace detail

// ═══════════════════════════════════════════════════════════════════════════════
// memcpy — Zero-overhead for all sizes
// ═══════════════════════════════════════════════════════════════════════════════

__attribute__((always_inline))
inline void* memcpy(void* __restrict__ inDst, const void* __restrict__ inSrc, oa::Usize inSize) {
	if (__builtin_expect(inSize == 0, 0)) return inDst;
	// Let the compiler emit its optimal single sequence when the call site knows
	// the size. The branches below are specifically for dynamic-size callers.
	if (__builtin_constant_p(inSize)) return std::memcpy(inDst, inSrc, inSize);

	oa::Byte* dst = static_cast<oa::Byte*>(inDst);
	const oa::Byte* src = static_cast<const oa::Byte*>(inSrc);
	using detail::copyBlock;

	// Fixed-size std::memcpy is a compiler primitive, not a libc call. Keeping
	// these accesses expressed as copies also makes unaligned data legal C++.
	if (__builtin_expect(inSize <= 16, 1)) {
		if (inSize >= 8) {
			copyBlock<8>(dst, src);
			copyBlock<8>(dst + inSize - 8, src + inSize - 8);
		} else if (inSize >= 4) {
			copyBlock<4>(dst, src);
			copyBlock<4>(dst + inSize - 4, src + inSize - 4);
		} else if (inSize >= 2) {
			copyBlock<2>(dst, src);
			copyBlock<2>(dst + inSize - 2, src + inSize - 2);
		} else {
			copyBlock<1>(dst, src);
		}
		return inDst;
	}

	if (inSize <= 32) {
		copyBlock<16>(dst, src);
		copyBlock<16>(dst + inSize - 16, src + inSize - 16);
		return inDst;
	}

	if (inSize <= 64) {
		copyBlock<32>(dst, src);
		copyBlock<32>(dst + inSize - 32, src + inSize - 32);
		return inDst;
	}

	if (inSize <= 128) {
		copyBlock<32>(dst, src);
		copyBlock<32>(dst + 32, src + 32);
		copyBlock<32>(dst + inSize - 64, src + inSize - 64);
		copyBlock<32>(dst + inSize - 32, src + inSize - 32);
		return inDst;
	}

	if (inSize <= 256) {
		copyBlock<32>(dst, src);
		copyBlock<32>(dst + 32, src + 32);
		copyBlock<32>(dst + 64, src + 64);
		copyBlock<32>(dst + 96, src + 96);
		copyBlock<32>(dst + inSize - 128, src + inSize - 128);
		copyBlock<32>(dst + inSize - 96, src + inSize - 96);
		copyBlock<32>(dst + inSize - 64, src + inSize - 64);
		copyBlock<32>(dst + inSize - 32, src + inSize - 32);
		return inDst;
	}

	return std::memcpy(inDst, inSrc, inSize);
}

// ═══════════════════════════════════════════════════════════════════════════════
// memcpyStream — Explicit non-temporal cache policy
// ═══════════════════════════════════════════════════════════════════════════════

inline void* memcpyStream(void* inDst, const void* inSrc, oa::Usize inSize) {
	if (inSize == 0 || inDst == inSrc) return inDst;
	return memcpyNt(inDst, inSrc, inSize);
}

// ═══════════════════════════════════════════════════════════════════════════════
// MEMSET / MEMZERO / MEMCMP (defined in memory.cpp)
// ═══════════════════════════════════════════════════════════════════════════════

inline void* memset(void* inDst, oa::I32 inValue, oa::Usize inSize) {
	return std::memset(inDst, inValue, inSize);
}

inline void* memzero(void* inDst, oa::Usize inSize) {
	return std::memset(inDst, 0, inSize);
}

inline oa::I32 memcmp(const void* inA, const void* inB, oa::Usize inSize) {
	return std::memcmp(inA, inB, inSize);
}

bool memEqual(const void* inA, const void* inB, oa::Usize inSize);

// ═══════════════════════════════════════════════════════════════════════════════
// ALIGNED ALLOCATION
// ═══════════════════════════════════════════════════════════════════════════════

void* alignedAlloc(oa::Usize inSize, oa::Usize inAlignment = 64);
void alignedFree(void* inPtr);

// ═══════════════════════════════════════════════════════════════════════════════
// PREFETCH / CACHE CONTROL
// ═══════════════════════════════════════════════════════════════════════════════

inline void prefetchL1(const void* inPtr)    { __builtin_prefetch(inPtr, 0, 3); }
inline void prefetchL2(const void* inPtr)    { __builtin_prefetch(inPtr, 0, 2); }
inline void prefetchWrite(void* inPtr)       { __builtin_prefetch(inPtr, 1, 3); }
inline void prefetchNta(const void* inPtr)   { __builtin_prefetch(inPtr, 0, 0); }
inline void memoryFence() { __sync_synchronize(); }
inline void storeFence()  { std::atomic_thread_fence(std::memory_order_release); }
inline void loadFence()   { std::atomic_thread_fence(std::memory_order_acquire); }

} // namespace oa
