// ═══════════════════════════════════════════════════════════════════════════════
// OA - MEMORY OPERATIONS (Compiled components)
// ═══════════════════════════════════════════════════════════════════════════════
//
// contains only what can't be inlined:
//   - Non-temporal streaming copy (AVX2/AVX-512)
//   - memEqual (AVX2/AVX-512)
//   - Aligned allocation
//
// The hot path (oa::memcpy) is fully inlined in memory.h
//
// ═══════════════════════════════════════════════════════════════════════════════

#include <oa/core/memory.h>
#include <oa/core/std/allocator.h>

#include <ctype.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>

#if defined(_WIN32)
	#include <malloc.h>
#endif

// SIMD intrinsics
#if defined(__x86_64__) || defined(_M_X64)
	#include <immintrin.h>
	#include <cpuid.h>
#endif

// ═══════════════════════════════════════════════════════════════════════════════
// RUNTIME CPU DETECTION
// ═══════════════════════════════════════════════════════════════════════════════

static bool hasAVX512F() {
#if defined(__x86_64__)
	oa::U32 eax, ebx, ecx, edx;
	if (__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) {
		return (ebx >> 16) & 1;
	}
#endif
	return false;
}

static const bool g_HasAVX512 = hasAVX512F();

static oa::Usize initOaMemcpyNtPrefetchBytes() {
	constexpr oa::Usize kDefault = 0;
	constexpr oa::Usize kMax = 8192;
	const char* env = ::getenv("OA_MEMCPY_NT_PREFETCH");
	if (!env || !*env) return kDefault;
	char* end = nullptr;
	errno = 0;
	unsigned long long val = ::strtoull(env, &end, 10);
	if (errno == ERANGE) return kMax;
	if (end == env) return kDefault;
	while (*end != '\0' && ::isspace(static_cast<unsigned char>(*end))) ++end;
	if (*end != '\0') return kDefault;
	if (val == 0ULL) return 0;
	if (val > static_cast<unsigned long long>(kMax)) return kMax;
	return static_cast<oa::Usize>(val);
}

static const oa::Usize g_OaMemcpyNtPrefetchBytes = initOaMemcpyNtPrefetchBytes();

// ═══════════════════════════════════════════════════════════════════════════════
// EXPLICIT NON-TEMPORAL STREAMING COPY
// ═══════════════════════════════════════════════════════════════════════════════
// Bypasses normal cache allocation. This is a semantic choice for one-way
// uploads (for example a mapped discrete-GPU BAR), not a universally faster
// memcpy selected from the byte count alone.

#if defined(__AVX512F__)
__attribute__((target("avx512f")))
static void* memcpyNT_AVX512(void* inDst, const void* inSrc, oa::Usize inSize) {
	oa::Byte* dst = static_cast<oa::Byte*>(inDst);
	const oa::Byte* src = static_cast<const oa::Byte*>(inSrc);

	// Align destination to 64B
	oa::Usize Align = (64 - (reinterpret_cast<oa::Usize>(dst) & 63)) & 63;
	if (Align > 0 && Align <= inSize) {
		oa::memcpy(dst, src, Align);
		dst += Align; src += Align; inSize -= Align;
	}

	while (inSize >= 256) {
		if (g_OaMemcpyNtPrefetchBytes > 0) {
			const char* PrefBase = reinterpret_cast<const char*>(src) + g_OaMemcpyNtPrefetchBytes;
			_mm_prefetch(PrefBase, _MM_HINT_NTA);
			_mm_prefetch(PrefBase + 64, _MM_HINT_NTA);
			_mm_prefetch(PrefBase + 128, _MM_HINT_NTA);
			_mm_prefetch(PrefBase + 192, _MM_HINT_NTA);
		}

		__m512i Z0 = _mm512_loadu_si512(src);
		__m512i Z1 = _mm512_loadu_si512(src + 64);
		__m512i Z2 = _mm512_loadu_si512(src + 128);
		__m512i Z3 = _mm512_loadu_si512(src + 192);
		_mm512_stream_si512(reinterpret_cast<__m512i*>(dst), Z0);
		_mm512_stream_si512(reinterpret_cast<__m512i*>(dst + 64), Z1);
		_mm512_stream_si512(reinterpret_cast<__m512i*>(dst + 128), Z2);
		_mm512_stream_si512(reinterpret_cast<__m512i*>(dst + 192), Z3);
		dst += 256; src += 256; inSize -= 256;
	}

	_mm_sfence();
	if (inSize > 0) oa::memcpy(dst, src, inSize);
	return inDst;
}
#endif

#if defined(__AVX2__)
static void* memcpyNT_AVX2(void* inDst, const void* inSrc, oa::Usize inSize) {
	oa::Byte* dst = static_cast<oa::Byte*>(inDst);
	const oa::Byte* src = static_cast<const oa::Byte*>(inSrc);

	// Align destination to 32B
	oa::Usize Align = (32 - (reinterpret_cast<oa::Usize>(dst) & 31)) & 31;
	if (Align > 0 && Align <= inSize) {
		oa::memcpy(dst, src, Align);
		dst += Align; src += Align; inSize -= Align;
	}

	while (inSize >= 128) {
		if (g_OaMemcpyNtPrefetchBytes > 0) {
			const char* PrefBase = reinterpret_cast<const char*>(src) + g_OaMemcpyNtPrefetchBytes;
			_mm_prefetch(PrefBase, _MM_HINT_NTA);
			_mm_prefetch(PrefBase + 64, _MM_HINT_NTA);
		}

		__m256i Y0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src));
		__m256i Y1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + 32));
		__m256i Y2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + 64));
		__m256i Y3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + 96));
		_mm256_stream_si256(reinterpret_cast<__m256i*>(dst), Y0);
		_mm256_stream_si256(reinterpret_cast<__m256i*>(dst + 32), Y1);
		_mm256_stream_si256(reinterpret_cast<__m256i*>(dst + 64), Y2);
		_mm256_stream_si256(reinterpret_cast<__m256i*>(dst + 96), Y3);
		dst += 128; src += 128; inSize -= 128;
	}

	_mm_sfence();
	if (inSize > 0) oa::memcpy(dst, src, inSize);
	return inDst;
}
#endif

void* oa::memcpyNt(void* inDst, const void* inSrc, oa::Usize inSize) {
	if (inSize == 0 || inDst == inSrc) return inDst;
#if defined(__AVX512F__)
	if (g_HasAVX512) return memcpyNT_AVX512(inDst, inSrc, inSize);
#endif
#if defined(__AVX2__)
	return memcpyNT_AVX2(inDst, inSrc, inSize);
#else
	return oa::memcpy(inDst, inSrc, inSize);
#endif
}

// ═══════════════════════════════════════════════════════════════════════════════
// MEMCMP / MEMEQUAL
// ═══════════════════════════════════════════════════════════════════════════════

bool oa::memEqual(const void* inA, const void* inB, oa::Usize inSize) {
	if (inA == inB || inSize == 0) return true;

#if defined(__AVX512F__)
	if (g_HasAVX512) {
		const oa::Byte* A = static_cast<const oa::Byte*>(inA);
		const oa::Byte* B = static_cast<const oa::Byte*>(inB);
		while (inSize >= 64) {
			__m512i Va = _mm512_loadu_si512(A);
			__m512i Vb = _mm512_loadu_si512(B);
			if (_mm512_cmpeq_epi8_mask(Va, Vb) != 0xFFFFFFFFFFFFFFFFULL) return false;
			A += 64; B += 64; inSize -= 64;
		}
		if (inSize > 0) return oa::memcmp(A, B, inSize) == 0;
		return true;
	}
#endif

#if defined(__AVX2__)
	{
		const oa::Byte* A = static_cast<const oa::Byte*>(inA);
		const oa::Byte* B = static_cast<const oa::Byte*>(inB);
		while (inSize >= 32) {
			__m256i Va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(A));
			__m256i Vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(B));
			if (_mm256_movemask_epi8(_mm256_cmpeq_epi8(Va, Vb)) != -1) return false;
			A += 32; B += 32; inSize -= 32;
		}
		if (inSize > 0) return oa::memcmp(A, B, inSize) == 0;
		return true;
	}
#else
	return oa::memcmp(inA, inB, inSize) == 0;
#endif
}

// ═══════════════════════════════════════════════════════════════════════════════
// ALIGNED ALLOCATION
// ═══════════════════════════════════════════════════════════════════════════════

oa::AllocationResult oa::tryAllocBytes(
	oa::Usize inBytes,
	oa::Usize inAlignment
) noexcept {
	const oa::Usize requestedAlignment = inAlignment == 0 ? 1 : inAlignment;
	if ((requestedAlignment & (requestedAlignment - 1)) != 0) {
		return {nullptr, oa::AllocationError::InvalidAlignment};
	}
	if (inBytes == 0) {
		return {nullptr, oa::AllocationError::None};
	}

	const oa::Usize effectiveAlignment =
		requestedAlignment < sizeof(void*) ? sizeof(void*) : requestedAlignment;
#if defined(_WIN32)
	void* const ptr = _aligned_malloc(inBytes, effectiveAlignment);
	return ptr != nullptr
		? oa::AllocationResult{ptr, oa::AllocationError::None}
		: oa::AllocationResult{nullptr, oa::AllocationError::OutOfMemory};
#else
	if (effectiveAlignment <= oa::defaultAllocationAlignment()) {
		void* const ptr = ::malloc(inBytes);
		return ptr != nullptr
			? oa::AllocationResult{ptr, oa::AllocationError::None}
			: oa::AllocationResult{nullptr, oa::AllocationError::OutOfMemory};
	}
	void* ptr = nullptr;
	const int result = ::posix_memalign(&ptr, effectiveAlignment, inBytes);
	if (result == 0) {
		return {ptr, oa::AllocationError::None};
	}
	return {
		nullptr,
		result == EINVAL
			? oa::AllocationError::InvalidAlignment
			: oa::AllocationError::OutOfMemory,
	};
#endif
}

oa::AllocationResult oa::tryAllocArray(
	oa::Usize inCount,
	oa::Usize inElementSize,
	oa::Usize inAlignment
) noexcept {
	if (inCount != 0 && inElementSize > static_cast<oa::Usize>(-1) / inCount) {
		return {nullptr, oa::AllocationError::SizeOverflow};
	}
	return oa::tryAllocBytes(inCount * inElementSize, inAlignment);
}

oa::AllocationResult oa::tryReallocBytes(
	void* inPtr,
	oa::Usize inBytes,
	oa::Usize inAlignment
) noexcept {
	const oa::Usize requestedAlignment = inAlignment == 0 ? 1 : inAlignment;
	if ((requestedAlignment & (requestedAlignment - 1)) != 0
		|| requestedAlignment > oa::defaultAllocationAlignment()) {
		return {nullptr, oa::AllocationError::InvalidAlignment};
	}
	if (inBytes == 0) {
		oa::freeBytes(inPtr, inAlignment);
		return {nullptr, oa::AllocationError::None};
	}

#if defined(_WIN32)
	const oa::Usize effectiveAlignment = requestedAlignment < sizeof(void*)
		? sizeof(void*)
		: requestedAlignment;
	void* const ptr = _aligned_realloc(inPtr, inBytes, effectiveAlignment);
#else
	void* const ptr = ::realloc(inPtr, inBytes);
#endif
	return ptr != nullptr
		? oa::AllocationResult{ptr, oa::AllocationError::None}
		: oa::AllocationResult{nullptr, oa::AllocationError::OutOfMemory};
}

[[noreturn]] void oa::allocationFailed(
	oa::AllocationError inError,
	oa::Usize inBytes,
	oa::Usize inAlignment
) noexcept {
	::fprintf(
		stderr,
		"OA allocation failure: %s (bytes=%zu, alignment=%zu)\n",
		oa::allocationErrorName(inError),
		inBytes,
		inAlignment);
	::abort();
}

void oa::freeBytes(void* inPtr, oa::Usize inAlignment) noexcept {
	(void)inAlignment;
	if (inPtr == nullptr) {
		return;
	}
#if defined(_WIN32)
	_aligned_free(inPtr);
#else
	::free(inPtr);
#endif
}

void* oa::alignedAlloc(oa::Usize inSize, oa::Usize inAlignment) {
	return oa::tryAllocBytes(inSize, inAlignment).data;
}

void oa::alignedFree(void* inPtr) {
	oa::freeBytes(inPtr);
}
