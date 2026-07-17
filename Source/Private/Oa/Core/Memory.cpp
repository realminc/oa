// ═══════════════════════════════════════════════════════════════════════════════
// OA - MEMORY OPERATIONS (Compiled components)
// ═══════════════════════════════════════════════════════════════════════════════
//
// Contains only what can't be inlined:
//   - Non-temporal streaming copy (AVX2/AVX-512)
//   - MemEqual (AVX2/AVX-512)
//   - Aligned allocation
//
// The hot path (OaMemcpy) is fully inlined in memory.h
//
// ═══════════════════════════════════════════════════════════════════════════════

#include <Oa/Core/Memory.h>

#include <cctype>
#include <cerrno>
#include <cstdlib>

// SIMD intrinsics
#if defined(__x86_64__) || defined(_M_X64)
	#include <immintrin.h>
	#include <cpuid.h>
#endif

// ═══════════════════════════════════════════════════════════════════════════════
// RUNTIME CPU DETECTION
// ═══════════════════════════════════════════════════════════════════════════════

static bool HasAVX512F() {
#if defined(__x86_64__)
	OaU32 eax, ebx, ecx, edx;
	if (__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) {
		return (ebx >> 16) & 1;
	}
#endif
	return false;
}

static const bool g_HasAVX512 = HasAVX512F();

static OaUsize InitOaMemcpyNtPrefetchBytes() {
	constexpr OaUsize kDefault = 0;
	constexpr OaUsize kMax = 8192;
	const char* env = std::getenv("OA_MEMCPY_NT_PREFETCH");
	if (!env || !*env) return kDefault;
	char* end = nullptr;
	errno = 0;
	unsigned long long val = std::strtoull(env, &end, 10);
	if (errno == ERANGE) return kMax;
	if (end == env) return kDefault;
	while (*end != '\0' && std::isspace(static_cast<unsigned char>(*end))) ++end;
	if (*end != '\0') return kDefault;
	if (val == 0ULL) return 0;
	if (val > static_cast<unsigned long long>(kMax)) return kMax;
	return static_cast<OaUsize>(val);
}

static const OaUsize g_OaMemcpyNtPrefetchBytes = InitOaMemcpyNtPrefetchBytes();

// ═══════════════════════════════════════════════════════════════════════════════
// EXPLICIT NON-TEMPORAL STREAMING COPY
// ═══════════════════════════════════════════════════════════════════════════════
// Bypasses normal cache allocation. This is a semantic choice for one-way
// uploads (for example a mapped discrete-GPU BAR), not a universally faster
// memcpy selected from the byte count alone.

#if defined(__AVX512F__)
__attribute__((target("avx512f")))
static void* MemcpyNT_AVX512(void* InDst, const void* InSrc, OaUsize InSize) {
	OaByte* Dst = static_cast<OaByte*>(InDst);
	const OaByte* Src = static_cast<const OaByte*>(InSrc);

	// Align destination to 64B
	OaUsize Align = (64 - (reinterpret_cast<OaUsize>(Dst) & 63)) & 63;
	if (Align > 0 && Align <= InSize) {
		std::memcpy(Dst, Src, Align);
		Dst += Align; Src += Align; InSize -= Align;
	}

	while (InSize >= 256) {
		if (g_OaMemcpyNtPrefetchBytes > 0) {
			const char* PrefBase = reinterpret_cast<const char*>(Src) + g_OaMemcpyNtPrefetchBytes;
			_mm_prefetch(PrefBase, _MM_HINT_NTA);
			_mm_prefetch(PrefBase + 64, _MM_HINT_NTA);
			_mm_prefetch(PrefBase + 128, _MM_HINT_NTA);
			_mm_prefetch(PrefBase + 192, _MM_HINT_NTA);
		}

		__m512i Z0 = _mm512_loadu_si512(Src);
		__m512i Z1 = _mm512_loadu_si512(Src + 64);
		__m512i Z2 = _mm512_loadu_si512(Src + 128);
		__m512i Z3 = _mm512_loadu_si512(Src + 192);
		_mm512_stream_si512(reinterpret_cast<__m512i*>(Dst), Z0);
		_mm512_stream_si512(reinterpret_cast<__m512i*>(Dst + 64), Z1);
		_mm512_stream_si512(reinterpret_cast<__m512i*>(Dst + 128), Z2);
		_mm512_stream_si512(reinterpret_cast<__m512i*>(Dst + 192), Z3);
		Dst += 256; Src += 256; InSize -= 256;
	}

	_mm_sfence();
	if (InSize > 0) std::memcpy(Dst, Src, InSize);
	return InDst;
}
#endif

#if defined(__AVX2__)
static void* MemcpyNT_AVX2(void* InDst, const void* InSrc, OaUsize InSize) {
	OaByte* Dst = static_cast<OaByte*>(InDst);
	const OaByte* Src = static_cast<const OaByte*>(InSrc);

	// Align destination to 32B
	OaUsize Align = (32 - (reinterpret_cast<OaUsize>(Dst) & 31)) & 31;
	if (Align > 0 && Align <= InSize) {
		std::memcpy(Dst, Src, Align);
		Dst += Align; Src += Align; InSize -= Align;
	}

	while (InSize >= 128) {
		if (g_OaMemcpyNtPrefetchBytes > 0) {
			const char* PrefBase = reinterpret_cast<const char*>(Src) + g_OaMemcpyNtPrefetchBytes;
			_mm_prefetch(PrefBase, _MM_HINT_NTA);
			_mm_prefetch(PrefBase + 64, _MM_HINT_NTA);
		}

		__m256i Y0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(Src));
		__m256i Y1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(Src + 32));
		__m256i Y2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(Src + 64));
		__m256i Y3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(Src + 96));
		_mm256_stream_si256(reinterpret_cast<__m256i*>(Dst), Y0);
		_mm256_stream_si256(reinterpret_cast<__m256i*>(Dst + 32), Y1);
		_mm256_stream_si256(reinterpret_cast<__m256i*>(Dst + 64), Y2);
		_mm256_stream_si256(reinterpret_cast<__m256i*>(Dst + 96), Y3);
		Dst += 128; Src += 128; InSize -= 128;
	}

	_mm_sfence();
	if (InSize > 0) std::memcpy(Dst, Src, InSize);
	return InDst;
}
#endif

void* OaMemcpyNT(void* InDst, const void* InSrc, OaUsize InSize) {
	if (InSize == 0 || InDst == InSrc) return InDst;
#if defined(__AVX512F__)
	if (g_HasAVX512) return MemcpyNT_AVX512(InDst, InSrc, InSize);
#endif
#if defined(__AVX2__)
	return MemcpyNT_AVX2(InDst, InSrc, InSize);
#else
	return std::memcpy(InDst, InSrc, InSize);
#endif
}

// ═══════════════════════════════════════════════════════════════════════════════
// MEMCMP / MEMEQUAL
// ═══════════════════════════════════════════════════════════════════════════════

OaBool OaMemEqual(const void* InA, const void* InB, OaUsize InSize) {
	if (InA == InB || InSize == 0) return true;

#if defined(__AVX512F__)
	if (g_HasAVX512) {
		const OaByte* A = static_cast<const OaByte*>(InA);
		const OaByte* B = static_cast<const OaByte*>(InB);
		while (InSize >= 64) {
			__m512i Va = _mm512_loadu_si512(A);
			__m512i Vb = _mm512_loadu_si512(B);
			if (_mm512_cmpeq_epi8_mask(Va, Vb) != 0xFFFFFFFFFFFFFFFFULL) return false;
			A += 64; B += 64; InSize -= 64;
		}
		if (InSize > 0) return std::memcmp(A, B, InSize) == 0;
		return true;
	}
#endif

#if defined(__AVX2__)
	{
		const OaByte* A = static_cast<const OaByte*>(InA);
		const OaByte* B = static_cast<const OaByte*>(InB);
		while (InSize >= 32) {
			__m256i Va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(A));
			__m256i Vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(B));
			if (_mm256_movemask_epi8(_mm256_cmpeq_epi8(Va, Vb)) != -1) return false;
			A += 32; B += 32; InSize -= 32;
		}
		if (InSize > 0) return std::memcmp(A, B, InSize) == 0;
		return true;
	}
#else
	return std::memcmp(InA, InB, InSize) == 0;
#endif
}

// ═══════════════════════════════════════════════════════════════════════════════
// ALIGNED ALLOCATION
// ═══════════════════════════════════════════════════════════════════════════════

void* OaAlignedAlloc(OaUsize InSize, OaUsize InAlignment) {
#if defined(_WIN32)
	return _aligned_malloc(InSize, InAlignment);
#else
	void* Ptr = nullptr;
	if (posix_memalign(&Ptr, InAlignment, InSize) != 0) return nullptr;
	return Ptr;
#endif
}

void OaAlignedFree(void* InPtr) {
	if (InPtr) {
#if defined(_WIN32)
		_aligned_free(InPtr);
#else
		free(InPtr);
#endif
	}
}
