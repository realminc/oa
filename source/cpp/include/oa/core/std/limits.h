#pragma once

// Compiler-defined numeric properties without <limits>. These values describe
// the active target ABI and therefore remain correct for Clang, clang-cl, and
// every platform OA supports through those toolchains.

namespace oa {

template<typename T>
struct Limits;

#define OA_DEFINE_SIGNED_LIMITS(Type, MaxValue) \
	template<> struct Limits<Type> { \
		[[nodiscard]] static constexpr Type min() noexcept { return static_cast<Type>(-MaxValue - 1); } \
		[[nodiscard]] static constexpr Type max() noexcept { return static_cast<Type>(MaxValue); } \
		[[nodiscard]] static constexpr Type lowest() noexcept { return min(); } \
		[[nodiscard]] static constexpr Type epsilon() noexcept { return 0; } \
		[[nodiscard]] static constexpr Type infinity() noexcept { return 0; } \
		[[nodiscard]] static constexpr Type quietNaN() noexcept { return 0; } \
		static constexpr bool isSigned = true; \
		static constexpr bool isInteger = true; \
		static constexpr bool hasNaN = false; \
		static constexpr int digits = static_cast<int>(sizeof(Type) * __CHAR_BIT__ - 1); \
		static constexpr int digits10 = digits * 301 / 1000; \
	}

#define OA_DEFINE_UNSIGNED_LIMITS(Type) \
	template<> struct Limits<Type> { \
		[[nodiscard]] static constexpr Type min() noexcept { return 0; } \
		[[nodiscard]] static constexpr Type max() noexcept { return static_cast<Type>(-1); } \
		[[nodiscard]] static constexpr Type lowest() noexcept { return 0; } \
		[[nodiscard]] static constexpr Type epsilon() noexcept { return 0; } \
		[[nodiscard]] static constexpr Type infinity() noexcept { return 0; } \
		[[nodiscard]] static constexpr Type quietNaN() noexcept { return 0; } \
		static constexpr bool isSigned = false; \
		static constexpr bool isInteger = true; \
		static constexpr bool hasNaN = false; \
		static constexpr int digits = static_cast<int>(sizeof(Type) * __CHAR_BIT__); \
		static constexpr int digits10 = digits * 301 / 1000; \
	}

template<>
struct Limits<bool> {
	[[nodiscard]] static constexpr bool min() noexcept { return false; }
	[[nodiscard]] static constexpr bool max() noexcept { return true; }
	[[nodiscard]] static constexpr bool lowest() noexcept { return false; }
	[[nodiscard]] static constexpr bool epsilon() noexcept { return false; }
	[[nodiscard]] static constexpr bool infinity() noexcept { return false; }
	[[nodiscard]] static constexpr bool quietNaN() noexcept { return false; }
	static constexpr bool isSigned = false;
	static constexpr bool isInteger = true;
	static constexpr bool hasNaN = false;
	static constexpr int digits = 1;
	static constexpr int digits10 = 0;
};

#ifdef __CHAR_UNSIGNED__
OA_DEFINE_UNSIGNED_LIMITS(char);
#else
OA_DEFINE_SIGNED_LIMITS(char, __SCHAR_MAX__);
#endif

OA_DEFINE_SIGNED_LIMITS(signed char, __SCHAR_MAX__);
OA_DEFINE_UNSIGNED_LIMITS(unsigned char);
OA_DEFINE_SIGNED_LIMITS(short, __SHRT_MAX__);
OA_DEFINE_UNSIGNED_LIMITS(unsigned short);
OA_DEFINE_SIGNED_LIMITS(int, __INT_MAX__);
OA_DEFINE_UNSIGNED_LIMITS(unsigned int);
OA_DEFINE_SIGNED_LIMITS(long, __LONG_MAX__);
OA_DEFINE_UNSIGNED_LIMITS(unsigned long);
OA_DEFINE_SIGNED_LIMITS(long long, __LONG_LONG_MAX__);
OA_DEFINE_UNSIGNED_LIMITS(unsigned long long);

#define OA_DEFINE_FLOAT_LIMITS(Type, Prefix, InfFn, NanFn) \
	template<> struct Limits<Type> { \
		[[nodiscard]] static constexpr Type min() noexcept { return Prefix##_MIN__; } \
		[[nodiscard]] static constexpr Type max() noexcept { return Prefix##_MAX__; } \
		[[nodiscard]] static constexpr Type lowest() noexcept { return -Prefix##_MAX__; } \
		[[nodiscard]] static constexpr Type epsilon() noexcept { return Prefix##_EPSILON__; } \
		[[nodiscard]] static constexpr Type infinity() noexcept { return InfFn(); } \
		[[nodiscard]] static constexpr Type quietNaN() noexcept { return NanFn(""); } \
		static constexpr bool isSigned = true; \
		static constexpr bool isInteger = false; \
		static constexpr bool hasNaN = true; \
		static constexpr int digits = Prefix##_MANT_DIG__; \
		static constexpr int digits10 = Prefix##_DIG__; \
	}

OA_DEFINE_FLOAT_LIMITS(float, __FLT, __builtin_huge_valf, __builtin_nanf);
OA_DEFINE_FLOAT_LIMITS(double, __DBL, __builtin_huge_val, __builtin_nan);
OA_DEFINE_FLOAT_LIMITS(long double, __LDBL, __builtin_huge_vall, __builtin_nanl);

#undef OA_DEFINE_FLOAT_LIMITS
#undef OA_DEFINE_UNSIGNED_LIMITS
#undef OA_DEFINE_SIGNED_LIMITS

} // namespace oa
