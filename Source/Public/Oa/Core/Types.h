// OA CORE - Foundation Types
//
// EMPYREALM FOUNDATION LIBRARY
//
// This is the FOUNDATION file - all other headers include this.
// Contains: scalars, strings, containers, smart pointers, numeric constants.
//
// No #pragma once: memory.h includes this with OA_TYPES_H_SKIP_REST so only the scalar
// block is parsed first (breaks types → std.h → vec.h → memory.h → full types cycle).
//

#ifndef OA_CORE_TYPES_H_SCALARS_DEFINED
#define OA_CORE_TYPES_H_SCALARS_DEFINED

#include <cstddef>
#include <cstdint>

using OaI8 = std::int8_t;
using OaI16 = std::int16_t;
using OaI32 = std::int32_t;
using OaI64 = std::int64_t;

using OaU8 = std::uint8_t;
using OaU16 = std::uint16_t;
using OaU32 = std::uint32_t;
using OaU64 = std::uint64_t;

using OaF32 = float;
using OaF64 = double;

using OaUsize = std::size_t;
using OaIsize = std::ptrdiff_t;

using OaBool = bool;
using OaByte = OaU8;
using OaChar = char;

// OaDeterminismMode is now auto-generated in Type.gen.h

#endif

#if !defined(OA_TYPES_H_SKIP_REST) && !defined(OA_CORE_TYPES_H_FULL_DEFINED)
#define OA_CORE_TYPES_H_FULL_DEFINED

// AUTO-GENERATED TYPES (must be included before functions that use them)
#include <Oa/Core/Type.gen.h>

// STANDARD LIBRARY INCLUDES
#include <cassert>
#include <limits>

// COMPILER & PLATFORM DETECTION
#if defined(_MSC_VER)
	#define OA_COMPILER_MSVC 1
	#define OA_FORCEINLINE __forceinline
	#define OA_NOINLINE __declspec(noinline)
	#define OA_RESTRICT __restrict
	#define OA_LIKELY(x) (x)
	#define OA_UNLIKELY(x) (x)
#elif defined(__clang__)
	#define OA_COMPILER_CLANG 1
	#define OA_FORCEINLINE __attribute__((always_inline)) inline
	#define OA_NOINLINE __attribute__((noinline))
	#define OA_RESTRICT __restrict__
	#define OA_LIKELY(x) __builtin_expect(!!(x), 1)
	#define OA_UNLIKELY(x) __builtin_expect(!!(x), 0)
#elif defined(__GNUC__)
	#define OA_COMPILER_GCC 1
	#define OA_FORCEINLINE __attribute__((always_inline)) inline
	#define OA_NOINLINE __attribute__((noinline))
	#define OA_RESTRICT __restrict__
	#define OA_LIKELY(x) __builtin_expect(!!(x), 1)
	#define OA_UNLIKELY(x) __builtin_expect(!!(x), 0)
#endif

// GPU compute is Vulkan + Slang. No CUDA/HIP/Metal qualifiers in C++ code.
// These macros are kept as no-ops for source compatibility during migration.
#define OA_GPU_CODE 0
#define OA_HOST
#define OA_DEVICE
#define OA_HOST_DEVICE

#if defined(__ANDROID__)
	#define OA_PLATFORM_ANDROID 1
#elif defined(__linux__)
	#define OA_PLATFORM_LINUX 1
#elif defined(_WIN32) || defined(_WIN64)
	#define OA_PLATFORM_WINDOWS 1
#elif defined(__APPLE__)
	#define OA_PLATFORM_APPLE 1
#endif

// Debug assertion — maps to assert() (no-op when NDEBUG is defined)
#define OA_ASSERT(expr) assert(expr)

// 128-bit integers (MSVC: pair of limbs; Clang/GCC: __int128 extension)
#ifdef OA_COMPILER_MSVC
class OaI128 {
public:
	OaI64 Lo;
	OaI64 Hi;
	OA_HOST_DEVICE constexpr OaI128() : Lo(0), Hi(0) {}
	OA_HOST_DEVICE constexpr OaI128(OaI64 InLo, OaI64 InHi) : Lo(InLo), Hi(InHi) {}
	OA_HOST_DEVICE constexpr explicit OaI128(OaI64 InVal)
		: Lo(InVal), Hi(InVal < 0 ? -1 : 0) {}
};
class OaU128 {
public:
	OaU64 Lo;
	OaU64 Hi;
	OA_HOST_DEVICE constexpr OaU128() : Lo(0), Hi(0) {}
	OA_HOST_DEVICE constexpr OaU128(OaU64 InLo, OaU64 InHi) : Lo(InLo), Hi(InHi) {}
	OA_HOST_DEVICE constexpr explicit OaU128(OaU64 InVal) : Lo(InVal), Hi(0) {}
};
#else
	// __int128 is a compiler extension; ISO C++ does not define it (-Wpedantic).
	#if defined(__clang__) || defined(__GNUC__)
		#pragma GCC diagnostic push
		#pragma GCC diagnostic ignored "-Wpedantic"
	#endif
using OaI128 = __int128;
using OaU128 = unsigned __int128;
	#if defined(__clang__) || defined(__GNUC__)
		#pragma GCC diagnostic pop
	#endif
#endif

// String and container aliases (OaStd*)

#include <Oa/Core/Std.h>

template<typename T, OaUsize N>
using OaArray = OaStdArray<T, N>;

using OaString     = OaStdString;
using OaStringView = OaStdStringView;

// Optional UE-style names for UTF-8 text at boundaries (same types as OaString / OaStringView).
using OaText     = OaString;
using OaTextView = OaStringView;

template<typename T>
using OaSpan = OaStdSpan<T>;

using OaPath = OaStdPath;

template<typename T>
using OaOpt = OaStdOptional<T>;
template<typename T>
using OaOption = OaStdOptional<T>;

template<typename... Ts>
using OaVariant = OaStdVariant<Ts...>;

template<typename K, typename V, typename Hash = std::hash<K>, typename KeyEq = std::equal_to<K>>
using OaHashMap = OaStdHashMap<K, V, Hash, KeyEq>;

template<typename K, typename Hash = std::hash<K>, typename KeyEq = std::equal_to<K>>
using OaHashSet = OaStdHashSet<K, Hash, KeyEq>;

// Smart pointers and OaFunc
template<typename T>
using OaUniquePtr = OaStdUniquePtr<T>;

template<typename T>
using OaSharedPtr = OaStdSharedPtr<T>;

template<typename T>
using OaWeakPtr = OaStdWeakPtr<T>;

template<typename T, typename... Args>
[[nodiscard]] OA_FORCEINLINE OaUniquePtr<T> OaMakeUniquePtr(Args&&... InArgs) {
	return OaStdMakeUnique<T>(OaStdForward<Args>(InArgs)...);
}

template<typename T, typename... Args>
[[nodiscard]] OA_FORCEINLINE OaSharedPtr<T> OaMakeSharedPtr(Args&&... InArgs) {
	return OaStdMakeShared<T>(OaStdForward<Args>(InArgs)...);
}

template<typename T>
using OaFunc = OaStdFn<T>;

// OaScalarType is now auto-generated in Type.gen.h
// Aliases — PyTorch "dtype" / short "stype" (same enum; prefer OaScalarType in public APIs).
using OaDtype = OaScalarType;
using OaStype = OaScalarType;

[[nodiscard]] OA_HOST_DEVICE constexpr OaUsize OaScalarSize(OaScalarType InType) noexcept {
	switch (InType) {
		case OaScalarType::Float64:
		case OaScalarType::Int64:
		case OaScalarType::UInt64:
		case OaScalarType::Complex64:  return 8;
		case OaScalarType::Float32:
		case OaScalarType::Int32:
		case OaScalarType::UInt32:     return 4;
		case OaScalarType::Float16:
		case OaScalarType::BFloat16:
		case OaScalarType::Int16:
		case OaScalarType::UInt16:     return 2;
		case OaScalarType::Int8:
		case OaScalarType::UInt8:
		case OaScalarType::Bool:       return 1;
		case OaScalarType::Complex128: return 16;
		default:                       return 0;
	}
}

[[nodiscard]] constexpr OaStringView OaScalarTypeName(OaScalarType InType) noexcept {
	switch (InType) {
		case OaScalarType::Float32:    return OaStringView("float32");
		case OaScalarType::Float16:    return OaStringView("float16");
		case OaScalarType::BFloat16:   return OaStringView("bfloat16");
		case OaScalarType::Float64:    return OaStringView("float64");
		case OaScalarType::Int8:       return OaStringView("int8");
		case OaScalarType::Int16:      return OaStringView("int16");
		case OaScalarType::Int32:      return OaStringView("int32");
		case OaScalarType::Int64:      return OaStringView("int64");
		case OaScalarType::UInt8:      return OaStringView("uint8");
		case OaScalarType::UInt16:     return OaStringView("uint16");
		case OaScalarType::UInt32:     return OaStringView("uint32");
		case OaScalarType::UInt64:     return OaStringView("uint64");
		case OaScalarType::Bool:       return OaStringView("bool");
		case OaScalarType::Complex64:  return OaStringView("complex64");
		case OaScalarType::Complex128: return OaStringView("complex128");
		default:                       return OaStringView("unknown");
	}
}

// Type trait to get OaScalarType from C++ type
template<typename T> struct OaScalarTypeOf;
template<> struct OaScalarTypeOf<OaF32>  { static constexpr OaScalarType Value = OaScalarType::Float32; };
template<> struct OaScalarTypeOf<OaF64>  { static constexpr OaScalarType Value = OaScalarType::Float64; };
template<> struct OaScalarTypeOf<OaI8>   { static constexpr OaScalarType Value = OaScalarType::Int8; };
template<> struct OaScalarTypeOf<OaI16>  { static constexpr OaScalarType Value = OaScalarType::Int16; };
template<> struct OaScalarTypeOf<OaI32>  { static constexpr OaScalarType Value = OaScalarType::Int32; };
template<> struct OaScalarTypeOf<OaI64>  { static constexpr OaScalarType Value = OaScalarType::Int64; };
template<> struct OaScalarTypeOf<OaU8>   { static constexpr OaScalarType Value = OaScalarType::UInt8; };
template<> struct OaScalarTypeOf<OaU16>  { static constexpr OaScalarType Value = OaScalarType::UInt16; };
template<> struct OaScalarTypeOf<OaU32>  { static constexpr OaScalarType Value = OaScalarType::UInt32; };
template<> struct OaScalarTypeOf<OaU64>  { static constexpr OaScalarType Value = OaScalarType::UInt64; };
template<> struct OaScalarTypeOf<bool>   { static constexpr OaScalarType Value = OaScalarType::Bool; };

template<typename T>
inline constexpr OaScalarType OaScalarTypeOf_v = OaScalarTypeOf<T>::Value;

// OaPrecision is now auto-generated in Type.gen.h

[[nodiscard]] constexpr OaStringView OaPrecisionName(OaPrecision InPrec) noexcept {
	switch (InPrec) {
		case OaPrecision::FP32: return OaStringView("fp32");
		case OaPrecision::TF32: return OaStringView("tf32");
		case OaPrecision::BF16: return OaStringView("bf16");
		case OaPrecision::FP16: return OaStringView("fp16");
		case OaPrecision::INT8: return OaStringView("int8");
		default:                return OaStringView("unknown");
	}
}

// OaFilter is now auto-generated in Type.gen.h

[[nodiscard]] constexpr OaScalarType OaPrecisionDtype(OaPrecision InPrec) noexcept {
	switch (InPrec) {
		case OaPrecision::BF16: return OaScalarType::BFloat16;
		case OaPrecision::FP16: return OaScalarType::Float16;
		default:                return OaScalarType::Float32;
	}
}

// BF16 <-> FP32 conversion (truncation, not rounding — matches storage.slang)
[[nodiscard]] OA_HOST_DEVICE constexpr OaU16 OaF32ToBf16(OaF32 InVal) noexcept {
	union { OaF32 f; OaU32 u; } bits;
	bits.f = InVal;
	return static_cast<OaU16>(bits.u >> 16);
}

[[nodiscard]] OA_HOST_DEVICE constexpr OaF32 OaBf16ToF32(OaU16 InVal) noexcept {
	union { OaU32 u; OaF32 f; } bits;
	bits.u = static_cast<OaU32>(InVal) << 16;
	return bits.f;
}

// Numeric limits and math constants

inline constexpr OaF32 OA_F32_MAX = std::numeric_limits<OaF32>::max();
inline constexpr OaF32 OA_F32_MIN = std::numeric_limits<OaF32>::lowest();
inline constexpr OaF32 OA_F32_EPS = std::numeric_limits<OaF32>::epsilon();
inline constexpr OaF32 OA_F32_INF = std::numeric_limits<OaF32>::infinity();

inline constexpr OaF64 OA_F64_MAX = std::numeric_limits<OaF64>::max();
inline constexpr OaF64 OA_F64_MIN = std::numeric_limits<OaF64>::lowest();
inline constexpr OaF64 OA_F64_EPS = std::numeric_limits<OaF64>::epsilon();
inline constexpr OaF64 OA_F64_INF = std::numeric_limits<OaF64>::infinity();

inline constexpr OaI32 OA_I32_MAX = std::numeric_limits<OaI32>::max();
inline constexpr OaI32 OA_I32_MIN = std::numeric_limits<OaI32>::min();
inline constexpr OaI64 OA_I64_MAX = std::numeric_limits<OaI64>::max();
inline constexpr OaI64 OA_I64_MIN = std::numeric_limits<OaI64>::min();
inline constexpr OaU64 OA_U64_MAX = std::numeric_limits<OaU64>::max();

// Mathematical constants
inline constexpr OaF64 OA_PI = 3.14159265358979323846;
inline constexpr OaF64 OA_E  = 2.71828182845904523536;
inline constexpr OaF64 OA_SQRT2 = 1.41421356237309504880;
inline constexpr OaF64 OA_LN2 = 0.69314718055994530942;

inline constexpr OaF32 OA_PI_F = static_cast<OaF32>(OA_PI);
inline constexpr OaF32 OA_E_F  = static_cast<OaF32>(OA_E);

// Memory alignment
inline constexpr OaUsize OA_CACHELINE_SIZE = 64;   // CPU cache line
inline constexpr OaUsize OA_SIMD_ALIGN = 32;       // AVX2/AVX-512
inline constexpr OaUsize OA_GPU_ALIGN = 256;       // GPU memory alignment

template<OaUsize Alignment>
[[nodiscard]] OA_HOST_DEVICE constexpr OaUsize OaAlignUp(OaUsize InSize) noexcept {
	static_assert((Alignment & (Alignment - 1)) == 0, "Alignment must be power of 2");
	return (InSize + Alignment - 1) & ~(Alignment - 1);
}

template<OaUsize Alignment>
[[nodiscard]] OA_HOST_DEVICE constexpr OaUsize OaAlignDown(OaUsize InSize) noexcept {
	static_assert((Alignment & (Alignment - 1)) == 0, "Alignment must be power of 2");
	return InSize & ~(Alignment - 1);
}

[[nodiscard]] OA_HOST_DEVICE constexpr OaUsize OaAlignUp(OaUsize InSize, OaUsize InAlign) noexcept {
	return (InSize + InAlign - 1) & ~(InAlign - 1);
}

[[nodiscard]] OA_HOST_DEVICE constexpr OaU32 OaDivCeil(OaU32 InA, OaU32 InB) noexcept {
	return (InA + InB - 1) / InB;
}

// Saturating / safe arithmetic and byte helpers

[[nodiscard]] OA_HOST_DEVICE constexpr bool OaSafeAdd(OaU64 InA, OaU64 InB, OaU64& OutResult) noexcept {
	if (InB > OA_U64_MAX - InA) return false;
	OutResult = InA + InB;
	return true;
}

[[nodiscard]] OA_HOST_DEVICE constexpr bool OaSafeMul(OaU64 InA, OaU64 InB, OaU64& OutResult) noexcept {
	if (InA == 0 || InB == 0) { OutResult = 0; return true; }
	if (InA > OA_U64_MAX / InB) return false;
	OutResult = InA * InB;
	return true;
}

[[nodiscard]] OA_HOST_DEVICE constexpr OaU64 OaSafeAddClamped(OaU64 InA, OaU64 InB) noexcept {
	if (InB > OA_U64_MAX - InA) return OA_U64_MAX;
	return InA + InB;
}

[[nodiscard]] OA_HOST_DEVICE constexpr OaU16 OaByteSwap16(OaU16 InVal) noexcept {
	return (InVal >> 8) | (InVal << 8);
}

[[nodiscard]] OA_HOST_DEVICE constexpr OaU32 OaByteSwap32(OaU32 InVal) noexcept {
	InVal = ((InVal << 8) & 0xFF00FF00U) | ((InVal >> 8) & 0x00FF00FFU);
	return (InVal << 16) | (InVal >> 16);
}

[[nodiscard]] OA_HOST_DEVICE constexpr OaU64 OaByteSwap64(OaU64 InVal) noexcept {
	InVal = ((InVal << 8) & 0xFF00FF00FF00FF00ULL) | ((InVal >> 8) & 0x00FF00FF00FF00FFULL);
	InVal = ((InVal << 16) & 0xFFFF0000FFFF0000ULL) | ((InVal >> 16) & 0x0000FFFF0000FFFFULL);
	return (InVal << 32) | (InVal >> 32);
}

#include <cmath>
[[nodiscard]] inline OaF32 OaSqrt(OaF32 InX) noexcept { return sqrtf(InX); }
[[nodiscard]] inline OaF64 OaSqrt(OaF64 InX) noexcept { return sqrt(InX); }

#endif
