// ═══════════════════════════════════════════════════════════════════════════════
// OA - MEMORY OPERATIONS (Compiled components)
// ═══════════════════════════════════════════════════════════════════════════════
//
// contains only what can't be inlined:
//   - Qualified-window non-temporal streaming copy (AVX2/AVX-512)
//   - memEqual (AVX2/AVX-512)
//   - Aligned allocation
//
// The hot path (oa::memcpy) is fully inlined in memory.h
//
// ═══════════════════════════════════════════════════════════════════════════════

#include <oa/core/std/memory.h>
#include <oa/core/assert.h>
#include <oa/core/std/allocator.h>

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

static bool hasAVX512ByteOps() {
#if defined(__x86_64__)
	oa::U32 eax, ebx, ecx, edx;
	if (__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) {
		constexpr oa::U32 avx512fBit = 1U << 16U;
		constexpr oa::U32 avx512bwBit = 1U << 30U;
		return (ebx & (avx512fBit | avx512bwBit))
			== (avx512fBit | avx512bwBit);
	}
#endif
	return false;
}

[[maybe_unused]] static const bool g_HasAVX512ByteOps = hasAVX512ByteOps();

// ═══════════════════════════════════════════════════════════════════════════════
// EXPLICIT NON-TEMPORAL STREAMING COPY
// ═══════════════════════════════════════════════════════════════════════════════
// Bypasses normal cache allocation. This is a semantic choice for one-way
// uploads (for example a mapped discrete-GPU BAR), not a universally faster
// memcpy selected from the byte count alone.

#if defined(__AVX512F__) && defined(__AVX512BW__)
__attribute__((target("avx512f,avx512bw")))
static void* memcpyNT_AVX512(
	void* inDst,
	const void* inSrc,
	oa::Usize inSize
) noexcept {
	oa::Byte* dst = static_cast<oa::Byte*>(inDst);
	const oa::Byte* src = static_cast<const oa::Byte*>(inSrc);

	// Align destination to 64B
	oa::Usize Align = (64 - (reinterpret_cast<oa::Usize>(dst) & 63)) & 63;
	if (Align > 0 && Align <= inSize) {
		oa::memcpy(dst, src, Align);
		dst += Align; src += Align; inSize -= Align;
	}

	while (inSize >= 256) {
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
static void* memcpyNT_AVX2(
	void* inDst,
	const void* inSrc,
	oa::Usize inSize
) noexcept {
	oa::Byte* dst = static_cast<oa::Byte*>(inDst);
	const oa::Byte* src = static_cast<const oa::Byte*>(inSrc);

	// Align destination to 32B
	oa::Usize Align = (32 - (reinterpret_cast<oa::Usize>(dst) & 31)) & 31;
	if (Align > 0 && Align <= inSize) {
		oa::memcpy(dst, src, Align);
		dst += Align; src += Align; inSize -= Align;
	}

	while (inSize >= 128) {
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

void* oa::detail::memcpyStreamImpl(
	void* inDst,
	const void* inSrc,
	oa::Usize inSize
) noexcept {
	if (inSize == 0 || inDst == inSrc) return inDst;
#if defined(__AVX512F__) && defined(__AVX512BW__)
	if (g_HasAVX512ByteOps) return memcpyNT_AVX512(inDst, inSrc, inSize);
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

void oa::memzeroSecure(void* inDst, oa::Usize inSize) noexcept {
	if (inSize == 0) return;
	OA_REQUIRE(inDst != nullptr);
	volatile oa::Byte* const dst = static_cast<volatile oa::Byte*>(inDst);
	for (oa::Usize index = 0; index < inSize; ++index) {
		dst[index] = 0U;
	}
	__atomic_signal_fence(__ATOMIC_SEQ_CST);
}

bool oa::memEqualConstantTime(
	const void* inA,
	const void* inB,
	oa::Usize inSize
) noexcept {
	if (inSize == 0) return true;
	OA_REQUIRE(inA != nullptr);
	OA_REQUIRE(inB != nullptr);
	const oa::Byte* a = static_cast<const oa::Byte*>(inA);
	const oa::Byte* b = static_cast<const oa::Byte*>(inB);
	oa::Byte difference = 0U;

#if defined(__AVX512F__) && defined(__AVX512BW__)
	if (g_HasAVX512ByteOps) {
		oa::U64 equalMask = static_cast<oa::U64>(-1);
		while (inSize >= 64U) {
			const __m512i va = _mm512_loadu_si512(a);
			const __m512i vb = _mm512_loadu_si512(b);
			equalMask &= _mm512_cmpeq_epi8_mask(va, vb);
			a += 64U;
			b += 64U;
			inSize -= 64U;
		}
		difference |= static_cast<oa::Byte>(
			equalMask != static_cast<oa::U64>(-1));
	}
#elif defined(__AVX2__)
	__m256i vectorDifference = _mm256_setzero_si256();
	while (inSize >= 32U) {
		const __m256i va = _mm256_loadu_si256(
			reinterpret_cast<const __m256i*>(a));
		const __m256i vb = _mm256_loadu_si256(
			reinterpret_cast<const __m256i*>(b));
		vectorDifference = _mm256_or_si256(
			vectorDifference, _mm256_xor_si256(va, vb));
		a += 32U;
		b += 32U;
		inSize -= 32U;
	}
	const __m256i vectorEqual = _mm256_cmpeq_epi8(
		vectorDifference, _mm256_setzero_si256());
	difference |= static_cast<oa::Byte>(
		_mm256_movemask_epi8(vectorEqual) != -1);
#endif

	for (oa::Usize index = 0; index < inSize; ++index) {
		difference |= static_cast<oa::Byte>(a[index] ^ b[index]);
	}
	__atomic_signal_fence(__ATOMIC_SEQ_CST);
	return difference == 0U;
}

bool oa::memEqual(
	const void* inA,
	const void* inB,
	oa::Usize inSize
) noexcept {
	if (inSize == 0) return true;
	OA_REQUIRE(inA != nullptr);
	OA_REQUIRE(inB != nullptr);
	if (inA == inB) return true;

#if defined(__AVX512F__) && defined(__AVX512BW__)
	if (g_HasAVX512ByteOps) {
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
