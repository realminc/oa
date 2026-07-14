// ═══════════════════════════════════════════════════════════════════════════════
// OA - HIGH-PERFORMANCE MEMORY OPERATIONS
// ═══════════════════════════════════════════════════════════════════════════════
//
// Fully inlined memcpy with raw AVX2 assembly for zero-overhead copies.
//
// Strategy (benchmark-proven on Intel Ultra 9 275HX):
//   1-8B:    Register moves (inline)
//   9-16B:   Two overlapping 8B moves (inline)
//   17-32B:  Two overlapping 16B SSE moves (inline asm)
//   33-64B:  Two overlapping 32B AVX2 moves (inline asm)
//   65-128B: Four overlapping 32B AVX2 moves (inline asm)
//   129-256B: Eight overlapping 32B AVX2 moves (inline asm, 8 ymm regs)
//   257B–threshold: glibc memcpy (rep movsb / ifunc — hard to beat mid-band)
//   ≥threshold: non-temporal AVX2/512 streaming (default threshold 2MiB)
//   OA_MEMCPY_NT_MIN: decimal byte threshold at process start; 0 = disable auto NT
//   OA_MEMCPY_NT_PREFETCH: first NTA prefetch offset from current src in OaMemcpyNT (default 512; 0 = off; max 8192)
//   Prefetch sweep: bin/{preset}/Test/bench_memcpy_nt (Linux re-exec grid; optional --once --mb=N --compare)
//
// Why inline asm beats glibc at small sizes:
//   glibc's ifunc dispatch + internal branching costs ~2-4ns
//   Our inline asm: 0 function call, 0 dispatch, just register moves
//
// ═══════════════════════════════════════════════════════════════════════════════

#pragma once

#define OA_TYPES_H_SKIP_REST
#include <Oa/Core/Types.h>
#undef OA_TYPES_H_SKIP_REST
#include <atomic>
#include <cstring>

// Non-temporal copy for huge buffers (defined in memory.cpp)
void* OaMemcpyNT(void* InDst, const void* InSrc, OaUsize InSize);

// Minimum size for OaMemcpy / OaMemzero to take the non-temporal path (init from OA_MEMCPY_NT_MIN)
extern const OaUsize g_OaMemcpyNtMinBytes;

// ═══════════════════════════════════════════════════════════════════════════════
// OaMemcpy — Zero-overhead for all sizes
// ═══════════════════════════════════════════════════════════════════════════════

__attribute__((always_inline))
inline void* OaMemcpy(void* __restrict__ InDst, const void* __restrict__ InSrc, OaUsize InSize) {
	if (__builtin_expect(InSize == 0, 0)) return InDst;

	OaByte* Dst = static_cast<OaByte*>(InDst);
	const OaByte* Src = static_cast<const OaByte*>(InSrc);

	// ─── 1-16B: Register moves ─────────────────────────────────────────────
	if (__builtin_expect(InSize <= 16, 1)) {
		if (InSize >= 8) {
			*reinterpret_cast<OaU64*>(Dst) = *reinterpret_cast<const OaU64*>(Src);
			*reinterpret_cast<OaU64*>(Dst + InSize - 8) = *reinterpret_cast<const OaU64*>(Src + InSize - 8);
		} else if (InSize >= 4) {
			*reinterpret_cast<OaU32*>(Dst) = *reinterpret_cast<const OaU32*>(Src);
			*reinterpret_cast<OaU32*>(Dst + InSize - 4) = *reinterpret_cast<const OaU32*>(Src + InSize - 4);
		} else if (InSize >= 2) {
			*reinterpret_cast<OaU16*>(Dst) = *reinterpret_cast<const OaU16*>(Src);
			*reinterpret_cast<OaU16*>(Dst + InSize - 2) = *reinterpret_cast<const OaU16*>(Src + InSize - 2);
		} else {
			*Dst = *Src;
		}
		return InDst;
	}

#if defined(__AVX2__) && defined(__x86_64__)
	// ─── 17-32B: Two overlapping 16B SSE moves (inline asm) ────────────────
	if (InSize <= 32) {
		__asm__ __volatile__(
			"vmovdqu (%[src]), %%xmm0\n\t"
			"vmovdqu -16(%[src],%[sz],1), %%xmm1\n\t"
			"vmovdqu %%xmm0, (%[dst])\n\t"
			"vmovdqu %%xmm1, -16(%[dst],%[sz],1)"
			: : [dst] "r"(Dst), [src] "r"(Src), [sz] "r"(InSize)
			: "xmm0", "xmm1", "memory"
		);
		return InDst;
	}

	// ─── 33-64B: Two overlapping 32B AVX2 moves ───────────────────────────
	if (InSize <= 64) {
		__asm__ __volatile__(
			"vmovdqu (%[src]), %%ymm0\n\t"
			"vmovdqu -32(%[src],%[sz],1), %%ymm1\n\t"
			"vmovdqu %%ymm0, (%[dst])\n\t"
			"vmovdqu %%ymm1, -32(%[dst],%[sz],1)"
			: : [dst] "r"(Dst), [src] "r"(Src), [sz] "r"(InSize)
			: "ymm0", "ymm1", "memory"
		);
		return InDst;
	}

	// ─── 65-128B: Four overlapping 32B AVX2 moves ─────────────────────────
	if (InSize <= 128) {
		__asm__ __volatile__(
			"vmovdqu   (%[src]), %%ymm0\n\t"
			"vmovdqu 32(%[src]), %%ymm1\n\t"
			"vmovdqu -64(%[src],%[sz],1), %%ymm2\n\t"
			"vmovdqu -32(%[src],%[sz],1), %%ymm3\n\t"
			"vmovdqu %%ymm0,   (%[dst])\n\t"
			"vmovdqu %%ymm1, 32(%[dst])\n\t"
			"vmovdqu %%ymm2, -64(%[dst],%[sz],1)\n\t"
			"vmovdqu %%ymm3, -32(%[dst],%[sz],1)"
			: : [dst] "r"(Dst), [src] "r"(Src), [sz] "r"(InSize)
			: "ymm0", "ymm1", "ymm2", "ymm3", "memory"
		);
		return InDst;
	}

	// ─── 129-256B: Eight overlapping 32B AVX2 moves ───────────────────────
	if (InSize <= 256) {
		__asm__ __volatile__(
			"vmovdqu    (%[src]), %%ymm0\n\t"
			"vmovdqu  32(%[src]), %%ymm1\n\t"
			"vmovdqu  64(%[src]), %%ymm2\n\t"
			"vmovdqu  96(%[src]), %%ymm3\n\t"
			"vmovdqu -128(%[src],%[sz],1), %%ymm4\n\t"
			"vmovdqu  -96(%[src],%[sz],1), %%ymm5\n\t"
			"vmovdqu  -64(%[src],%[sz],1), %%ymm6\n\t"
			"vmovdqu  -32(%[src],%[sz],1), %%ymm7\n\t"
			"vmovdqu %%ymm0,    (%[dst])\n\t"
			"vmovdqu %%ymm1,  32(%[dst])\n\t"
			"vmovdqu %%ymm2,  64(%[dst])\n\t"
			"vmovdqu %%ymm3,  96(%[dst])\n\t"
			"vmovdqu %%ymm4, -128(%[dst],%[sz],1)\n\t"
			"vmovdqu %%ymm5,  -96(%[dst],%[sz],1)\n\t"
			"vmovdqu %%ymm6,  -64(%[dst],%[sz],1)\n\t"
			"vmovdqu %%ymm7,  -32(%[dst],%[sz],1)"
			: : [dst] "r"(Dst), [src] "r"(Src), [sz] "r"(InSize)
			: "ymm0", "ymm1", "ymm2", "ymm3", "ymm4", "ymm5", "ymm6", "ymm7", "memory"
		);
		return InDst;
	}
#else
	if (InSize <= 256) {
		return std::memcpy(InDst, InSrc, InSize);
	}
#endif

	// ─── ≥threshold: non-temporal streaming ────────────────────────────────
	if (__builtin_expect(InSize >= g_OaMemcpyNtMinBytes, 0)) {
		return OaMemcpyNT(InDst, InSrc, InSize);
	}

	// ─── Mid band: glibc memcpy ─────────────────────────────────────────
	return std::memcpy(InDst, InSrc, InSize);
}

// ═══════════════════════════════════════════════════════════════════════════════
// OaMemcpyStream — Always non-temporal
// ═══════════════════════════════════════════════════════════════════════════════

inline void* OaMemcpyStream(void* InDst, const void* InSrc, OaUsize InSize) {
	if (InSize == 0 || InDst == InSrc) return InDst;
	return OaMemcpyNT(InDst, InSrc, InSize);
}

// ═══════════════════════════════════════════════════════════════════════════════
// MEMSET / MEMZERO / MEMCMP (defined in memory.cpp)
// ═══════════════════════════════════════════════════════════════════════════════

void* OaMemset(void* InDst, OaI32 InValue, OaUsize InSize);
void* OaMemzero(void* InDst, OaUsize InSize);
OaI32 OaMemcmp(const void* InA, const void* InB, OaUsize InSize);
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
