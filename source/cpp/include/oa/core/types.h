// OA CORE - Foundation Types
//
// EMPYREALM FOUNDATION LIBRARY
//
// This is the FOUNDATION file - all other headers include this.
// contains: scalars, strings, containers, smart pointers, numeric constants.
//
// No #pragma once: memory.h includes this with OA_TYPES_H_SKIP_REST so only the scalar
// block is parsed first (breaks types → std.h → vec.h → memory.h → full types cycle).
//

// Fixed-width limit macros remain a C ABI boundary used by Vulkan-facing code.
// Use the C header directly instead of relying on a hosted C++ header to expose
// them transitively.
#include <stdint.h>

#ifndef OA_CORE_TYPES_H_SCALARS_DEFINED
#define OA_CORE_TYPES_H_SCALARS_DEFINED

namespace oa {

using I8 = __INT8_TYPE__;
using I16 = __INT16_TYPE__;
using I32 = __INT32_TYPE__;
using I64 = __INT64_TYPE__;

using U8 = __UINT8_TYPE__;
using U16 = __UINT16_TYPE__;
using U32 = __UINT32_TYPE__;
using U64 = __UINT64_TYPE__;

using F32 = float;
using F64 = double;

using Usize = __SIZE_TYPE__;
using Isize = __PTRDIFF_TYPE__;

using Byte = U8;
using Char = char;
using Bool = bool;

static_assert(sizeof(I8) == 1 and sizeof(U8) == 1);
static_assert(sizeof(I16) == 2 and sizeof(U16) == 2);
static_assert(sizeof(I32) == 4 and sizeof(U32) == 4);
static_assert(sizeof(I64) == 8 and sizeof(U64) == 8);

} // namespace oa

// oa::DeterminismMode is now auto-generated in type.gen.h

#endif

#if !defined(OA_TYPES_H_SKIP_REST) && !defined(OA_CORE_TYPES_H_FULL_DEFINED)
#define OA_CORE_TYPES_H_FULL_DEFINED

// AUTO-GENERATED TYPES (must be included before functions that use them)
#include <oa/core/type.gen.h>

#include <oa/core/assert.h>

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

// GPU compute is vulkan + slang. No CUDA/HIP/Metal qualifiers in C++ code.
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

// 128-bit integers (MSVC: pair of limbs; Clang/GCC: __int128 extension)
#ifdef OA_COMPILER_MSVC
namespace oa {

class I128 {
public:
	I64 lo;
	I64 hi;
	OA_HOST_DEVICE constexpr I128() : lo(0), hi(0) {}
	OA_HOST_DEVICE constexpr I128(I64 inLo, I64 inHi) : lo(inLo), hi(inHi) {}
	OA_HOST_DEVICE constexpr explicit I128(I64 inVal)
		: lo(inVal), hi(inVal < 0 ? -1 : 0) {}
};
class U128 {
public:
	U64 lo;
	U64 hi;
	OA_HOST_DEVICE constexpr U128() : lo(0), hi(0) {}
	OA_HOST_DEVICE constexpr U128(U64 inLo, U64 inHi) : lo(inLo), hi(inHi) {}
	OA_HOST_DEVICE constexpr explicit U128(U64 inVal) : lo(inVal), hi(0) {}
};

} // namespace oa
#else
	// __int128 is a compiler extension; ISO C++ does not define it (-Wpedantic).
	#if defined(__clang__) || defined(__GNUC__)
		#pragma GCC diagnostic push
		#pragma GCC diagnostic ignored "-Wpedantic"
	#endif
namespace oa {
using I128 = __int128;
using U128 = unsigned __int128;
} // namespace oa
	#if defined(__clang__) || defined(__GNUC__)
		#pragma GCC diagnostic pop
	#endif
#endif

// OA containers and utilities.

#include <oa/core/std.h>

namespace oa {

// ScalarType is auto-generated in type.gen.h.
[[nodiscard]] OA_HOST_DEVICE constexpr Usize scalarSize(ScalarType inType) noexcept {
	switch (inType) {
		case ScalarType::Float64:
		case ScalarType::Int64:
		case ScalarType::UInt64:
		case ScalarType::Complex64:  return 8;
		case ScalarType::Float32:
		case ScalarType::Int32:
		case ScalarType::UInt32:     return 4;
		case ScalarType::Float16:
		case ScalarType::BFloat16:
		case ScalarType::Int16:
		case ScalarType::UInt16:     return 2;
		case ScalarType::Int8:
		case ScalarType::UInt8:
		case ScalarType::Bool:       return 1;
		case ScalarType::Complex128: return 16;
		default:                       return 0;
	}
}

[[nodiscard]] constexpr StringView scalarTypeName(ScalarType inType) noexcept {
	switch (inType) {
		case ScalarType::Float32:    return StringView("float32");
		case ScalarType::Float16:    return StringView("float16");
		case ScalarType::BFloat16:   return StringView("bfloat16");
		case ScalarType::Float64:    return StringView("float64");
		case ScalarType::Int8:       return StringView("int8");
		case ScalarType::Int16:      return StringView("int16");
		case ScalarType::Int32:      return StringView("int32");
		case ScalarType::Int64:      return StringView("int64");
		case ScalarType::UInt8:      return StringView("uint8");
		case ScalarType::UInt16:     return StringView("uint16");
		case ScalarType::UInt32:     return StringView("uint32");
		case ScalarType::UInt64:     return StringView("uint64");
		case ScalarType::Bool:       return StringView("bool");
		case ScalarType::Complex64:  return StringView("complex64");
		case ScalarType::Complex128: return StringView("complex128");
		default:                     return StringView("unknown");
	}
}

// Type trait to get ScalarType from a C++ type.
template<typename T> struct ScalarTypeOf;
template<> struct ScalarTypeOf<F32>  { static constexpr ScalarType Value = ScalarType::Float32; };
template<> struct ScalarTypeOf<F64>  { static constexpr ScalarType Value = ScalarType::Float64; };
template<> struct ScalarTypeOf<I8>   { static constexpr ScalarType Value = ScalarType::Int8; };
template<> struct ScalarTypeOf<I16>  { static constexpr ScalarType Value = ScalarType::Int16; };
template<> struct ScalarTypeOf<I32>  { static constexpr ScalarType Value = ScalarType::Int32; };
template<> struct ScalarTypeOf<I64>  { static constexpr ScalarType Value = ScalarType::Int64; };
template<> struct ScalarTypeOf<U8>   { static constexpr ScalarType Value = ScalarType::UInt8; };
template<> struct ScalarTypeOf<U16>  { static constexpr ScalarType Value = ScalarType::UInt16; };
template<> struct ScalarTypeOf<U32>  { static constexpr ScalarType Value = ScalarType::UInt32; };
template<> struct ScalarTypeOf<U64>  { static constexpr ScalarType Value = ScalarType::UInt64; };
template<> struct ScalarTypeOf<bool> { static constexpr ScalarType Value = ScalarType::Bool; };

template<typename T>
inline constexpr ScalarType ScalarTypeOfV = ScalarTypeOf<T>::Value;

// oa::Precision is now auto-generated in type.gen.h

[[nodiscard]] constexpr StringView precisionName(Precision inPrec) noexcept {
	switch (inPrec) {
		case Precision::FP32: return StringView("fp32");
		case Precision::BF16: return StringView("bf16");
		case Precision::FP64: return StringView("fp64");
		default:              return StringView("unknown");
	}
}

// oa::Filter is now auto-generated in type.gen.h

[[nodiscard]] constexpr ScalarType precisionDtype(Precision inPrec) noexcept {
	switch (inPrec) {
		case Precision::BF16: return ScalarType::BFloat16;
		case Precision::FP64: return ScalarType::Float64;
		default:              return ScalarType::Float32;
	}
}

// BF16 <-> FP32 conversion (truncation, not rounding — matches storage.slang)
[[nodiscard]] OA_HOST_DEVICE constexpr U16 f32ToBf16(F32 inVal) noexcept {
	union { F32 f; U32 u; } bits;
	bits.f = inVal;
	return static_cast<U16>(bits.u >> 16);
}

[[nodiscard]] OA_HOST_DEVICE constexpr F32 bf16ToF32(U16 inVal) noexcept {
	union { U32 u; F32 f; } bits;
	bits.u = static_cast<U32>(inVal) << 16;
	return bits.f;
}

// Numeric limits and math constants

inline constexpr F32 F32Max = __FLT_MAX__;
inline constexpr F32 F32Min = -__FLT_MAX__;
inline constexpr F32 F32Epsilon = __FLT_EPSILON__;
inline constexpr F32 F32Infinity = __builtin_huge_valf();

inline constexpr F64 F64Max = __DBL_MAX__;
inline constexpr F64 F64Min = -__DBL_MAX__;
inline constexpr F64 F64Epsilon = __DBL_EPSILON__;
inline constexpr F64 F64Infinity = __builtin_huge_val();

inline constexpr I32 I32Max = __INT_MAX__;
inline constexpr I32 I32Min = -__INT_MAX__ - 1;
inline constexpr I64 I64Max = __LONG_LONG_MAX__;
inline constexpr I64 I64Min = -__LONG_LONG_MAX__ - 1;
inline constexpr U64 U64Max = ~static_cast<U64>(0);

// Mathematical constants
inline constexpr F64 Pi = 3.14159265358979323846;
inline constexpr F64 Euler = 2.71828182845904523536;
inline constexpr F64 Sqrt2 = 1.41421356237309504880;
inline constexpr F64 Ln2 = 0.69314718055994530942;

inline constexpr F32 PiF = static_cast<F32>(Pi);
inline constexpr F32 EulerF = static_cast<F32>(Euler);

// memory alignment
inline constexpr Usize CachelineSize = 64;
inline constexpr Usize SimdAlign = 32;
inline constexpr Usize GpuAlign = 256;

template<Usize alignment>
[[nodiscard]] OA_HOST_DEVICE constexpr Usize alignUp(Usize inSize) noexcept {
	static_assert((alignment & (alignment - 1)) == 0, "alignment must be power of 2");
	return (inSize + alignment - 1) & ~(alignment - 1);
}

template<Usize alignment>
[[nodiscard]] OA_HOST_DEVICE constexpr Usize alignDown(Usize inSize) noexcept {
	static_assert((alignment & (alignment - 1)) == 0, "alignment must be power of 2");
	return inSize & ~(alignment - 1);
}

[[nodiscard]] OA_HOST_DEVICE constexpr Usize alignUp(Usize inSize, Usize inAlign) noexcept {
	return (inSize + inAlign - 1) & ~(inAlign - 1);
}

[[nodiscard]] OA_HOST_DEVICE constexpr U32 divCeil(U32 inA, U32 inB) noexcept {
	return (inA + inB - 1) / inB;
}

// Saturating / safe arithmetic and byte helpers

[[nodiscard]] OA_HOST_DEVICE constexpr bool safeAdd(U64 inA, U64 inB, U64& outResult) noexcept {
	if (inB > U64Max - inA) {
		return false;
	}
	outResult = inA + inB;
	return true;
}

[[nodiscard]] OA_HOST_DEVICE constexpr bool safeMul(U64 inA, U64 inB, U64& outResult) noexcept {
	if (inA == 0 || inB == 0) {
		outResult = 0; return true;
	}
	if (inA > U64Max / inB) return false;
	outResult = inA * inB;
	return true;
}

[[nodiscard]] OA_HOST_DEVICE constexpr U64 safeAddClamped(U64 inA, U64 inB) noexcept {
	if (inB > U64Max - inA) return U64Max;
	return inA + inB;
}

[[nodiscard]] OA_HOST_DEVICE constexpr U16 byteSwap16(U16 inVal) noexcept {
	return static_cast<U16>((inVal >> 8) | (inVal << 8));
}

[[nodiscard]] OA_HOST_DEVICE constexpr U32 byteSwap32(U32 inVal) noexcept {
	inVal = ((inVal << 8) & 0xFF00FF00U) | ((inVal >> 8) & 0x00FF00FFU);
	return (inVal << 16) | (inVal >> 16);
}

[[nodiscard]] OA_HOST_DEVICE constexpr U64 byteSwap64(U64 inVal) noexcept {
	inVal = ((inVal << 8) & 0xFF00FF00FF00FF00ULL) | ((inVal >> 8) & 0x00FF00FF00FF00FFULL);
	inVal = ((inVal << 16) & 0xFFFF0000FFFF0000ULL) | ((inVal >> 16) & 0x0000FFFF0000FFFFULL);
	return (inVal << 32) | (inVal >> 32);
}

} // namespace oa

#endif
