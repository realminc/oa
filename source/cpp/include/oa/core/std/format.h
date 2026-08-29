#pragma once

// Type-safe, locale-independent brace formatting without C varargs,
// iostreams, std::format, or a hosted C++ formatting dependency.

#define OA_TYPES_H_SKIP_REST
#include <oa/core/types.h>
#undef OA_TYPES_H_SKIP_REST

#include <oa/core/std/assert.h>
#include <oa/core/std/string.h>
#include <oa/core/std/typeTraits.h>
#include <oa/core/std/utility.h>

#if defined(_MSC_VER)
	#define OA_FORMAT_FORCEINLINE __forceinline
#else
	#define OA_FORMAT_FORCEINLINE __attribute__((always_inline)) inline
#endif

namespace oa {

[[nodiscard]] bool parseU64(StringView inText, U64& outValue) noexcept;
[[nodiscard]] bool parseI64(StringView inText, I64& outValue) noexcept;
[[nodiscard]] bool parseF64(StringView inText, F64& outValue) noexcept;
[[nodiscard]] bool formatF64(F64 inValue, String& outText, I32 inPrecision = 17) noexcept;

namespace detail {

inline constexpr oa::Usize MaxFormattedBytes = 16U * 1024U * 1024U;
inline constexpr oa::Usize MaxFormatWidth = 1U * 1024U * 1024U;

struct FormatSpec {
	char fill = ' ';
	char align = 0;
	char sign = 0;
	char type = 0;
	oa::Usize width = 0;
	oa::I32 precision = -1;
	bool alternate = false;
	bool zero = false;
};

// Small formatted records are assembled contiguously before constructing the
// owning String. This avoids routing every proven-external fragment through
// String's public self-alias validation while retaining a single exact-size
// allocation when the result exceeds SSO.
class FormatBuffer {
public:
	static constexpr oa::Usize InlineCap = 128U;

	[[nodiscard]] OA_FORMAT_FORCEINLINE oa::Usize size() const noexcept {
		return promoted_ ? overflow_.size() : size_;
	}

	OA_FORMAT_FORCEINLINE void reserve(oa::Usize inCapacity) {
		if (promoted_) {
			overflow_.reserve(inCapacity);
		} else if (inCapacity > InlineCap) {
			promote(inCapacity);
		}
	}

	OA_FORMAT_FORCEINLINE void append(oa::StringView inText) {
		if (inText.empty()) return;
		if (not promoted_ and inText.size() <= InlineCap - size_) {
			oa::memcpy(storage_ + size_, inText.data(), inText.size());
			size_ += inText.size();
			return;
		}
		if (not promoted_) promote(size_ + inText.size());
		overflow_.append(inText);
	}

	OA_FORMAT_FORCEINLINE void pushBack(char inCharacter) {
		if (not promoted_ and size_ < InlineCap) {
			storage_[size_++] = inCharacter;
			return;
		}
		if (not promoted_) promote(size_ + 1U);
		overflow_.pushBack(inCharacter);
	}

	[[nodiscard]] OA_FORMAT_FORCEINLINE oa::String finish() && {
		if (promoted_) return oa::move(overflow_);
		return oa::String(storage_, size_);
	}

private:
	char storage_[InlineCap];
	oa::Usize size_ = 0U;
	oa::String overflow_;
	bool promoted_ = false;

	OA_FORMAT_FORCEINLINE void promote(oa::Usize inCapacity) {
		overflow_.reserve(inCapacity);
		overflow_.append(oa::StringView(storage_, size_));
		promoted_ = true;
	}
};

[[nodiscard]] bool formatFloatValue(
	oa::F64 inValue,
	char inType,
	oa::I32 inPrecision,
	bool inAlternate,
	char* outText,
	oa::Usize inCapacity,
	oa::Usize& outSize
) noexcept;

inline void requireFormat(bool inCondition, const char* inMessage) {
	OA_REQUIRE_MSG(inCondition, inMessage);
}

template<typename Output>
inline void reserveAppend(Output& out, oa::Usize inBytes) {
	requireFormat(inBytes <= MaxFormattedBytes - out.size(),
		"formatted output exceeds the OA safety limit");
	out.reserve(out.size() + inBytes);
}

template<typename Output>
inline void appendView(Output& out, oa::StringView inText) {
	reserveAppend(out, inText.size());
	out.append(inText);
}

template<typename Output>
inline void appendRepeated(Output& out, char inCharacter, oa::Usize inCount) {
	reserveAppend(out, inCount);
	for (oa::Usize index = 0; index < inCount; ++index) out.pushBack(inCharacter);
}

[[nodiscard]] OA_FORMAT_FORCEINLINE FormatSpec parseFormatSpec(oa::StringView inText) {
	FormatSpec result;
	oa::Usize index = 0;
	if (inText.size() >= 2U
		and (inText[1] == '<' or inText[1] == '>' or inText[1] == '^')) {
		requireFormat(inText[0] != '{' and inText[0] != '}',
			"format fill character cannot be a brace");
		result.fill = inText[0];
		result.align = inText[1];
		index = 2U;
	} else if (index < inText.size()
		and (inText[index] == '<' or inText[index] == '>' or inText[index] == '^')) {
		result.align = inText[index++];
	}
	if (index < inText.size() and (inText[index] == '+' or inText[index] == ' ')) {
		result.sign = inText[index++];
	}
	if (index < inText.size() and inText[index] == '#') {
		result.alternate = true;
		++index;
	}
	if (index < inText.size() and inText[index] == '0') {
		result.zero = true;
		++index;
	}
	while (index < inText.size() and inText[index] >= '0' and inText[index] <= '9') {
		const oa::Usize digit = static_cast<oa::Usize>(inText[index++] - '0');
		requireFormat(result.width <= (MaxFormatWidth - digit) / 10U,
			"format width exceeds the OA safety limit");
		result.width = result.width * 10U + digit;
	}
	if (index < inText.size() and inText[index] == '.') {
		++index;
		requireFormat(index < inText.size() and inText[index] >= '0'
			and inText[index] <= '9', "format precision requires digits");
		oa::I32 precision = 0;
		while (index < inText.size() and inText[index] >= '0' and inText[index] <= '9') {
			const oa::I32 digit = static_cast<oa::I32>(inText[index++] - '0');
			requireFormat(precision <= (1024 - digit) / 10,
				"format precision exceeds the OA safety limit");
			precision = precision * 10 + digit;
		}
		result.precision = precision;
	}
	if (index < inText.size()) result.type = inText[index++];
	requireFormat(index == inText.size(), "unsupported characters in format specifier");
	return result;
}

template<typename Output>
inline void appendAligned(
	Output& out,
	oa::StringView inValue,
	const FormatSpec& inSpec,
	bool inNumeric
) {
	if (inValue.size() >= inSpec.width) {
		appendView(out, inValue);
		return;
	}
	const oa::Usize padding = inSpec.width - inValue.size();
	char align = inSpec.align;
	if (align == 0) align = inNumeric ? '>' : '<';
	if (inSpec.zero and inNumeric and inSpec.align == 0) {
		oa::Usize prefix = 0;
		if (not inValue.empty() and (inValue[0] == '-' or inValue[0] == '+' or inValue[0] == ' ')) {
			prefix = 1U;
		}
		if (inValue.size() >= prefix + 2U and inValue[prefix] == '0'
			and (inValue[prefix + 1U] == 'x' or inValue[prefix + 1U] == 'X'
				or inValue[prefix + 1U] == 'b' or inValue[prefix + 1U] == 'o')) {
			prefix += 2U;
		}
		appendView(out, inValue.subStr(0U, prefix));
		appendRepeated(out, '0', padding);
		appendView(out, inValue.subStr(prefix));
		return;
	}
	const oa::Usize left = align == '>' ? padding : (align == '^' ? padding / 2U : 0U);
	const oa::Usize right = padding - left;
	appendRepeated(out, inSpec.fill, left);
	appendView(out, inValue);
	appendRepeated(out, inSpec.fill, right);
}

template<typename Output>
inline void appendUnsignedDigits(
	Output& out,
	oa::U64 inValue,
	oa::U32 inBase,
	bool inUpper
) {
	char reversed[64];
	oa::Usize count = 0;
	do {
		const oa::U32 digit = static_cast<oa::U32>(inValue % inBase);
		reversed[count++] = digit < 10U
			? static_cast<char>('0' + digit)
			: static_cast<char>((inUpper ? 'A' : 'a') + digit - 10U);
		inValue /= inBase;
	} while (inValue != 0U);
	reserveAppend(out, count);
	while (count != 0U) out.pushBack(reversed[--count]);
}

template<typename Output, typename T>
inline void appendInteger(Output& out, T inValue, const FormatSpec& inSpec) {
	const char type = inSpec.type == 0 ? 'd' : inSpec.type;
	requireFormat(type == 'd' or type == 'x' or type == 'X' or type == 'b' or type == 'o',
		"integer format type must be d, x, X, b, or o");
	requireFormat(inSpec.precision < 0, "integer precision is not supported");
	const oa::U32 base = type == 'x' or type == 'X' ? 16U : (type == 'b' ? 2U : (type == 'o' ? 8U : 10U));
	using Value = oa::RemoveCvrefT<T>;
	constexpr bool isSigned = __is_signed(Value);
	const bool negative = isSigned and inValue < 0;
	const oa::U64 raw = static_cast<oa::U64>(inValue);
	const oa::U64 magnitude = negative ? oa::U64(0) - raw : raw;
	if (type == 'd' and inSpec.width == 0U and inSpec.sign == 0
		and not inSpec.alternate and not inSpec.zero) {
		char storage[32];
		char* cursor = storage + sizeof(storage);
		oa::U64 work = magnitude;
		do {
			*--cursor = static_cast<char>('0' + static_cast<char>(work % 10U));
			work /= 10U;
		} while (work != 0U);
		if (negative) *--cursor = '-';
		appendView(out, oa::StringView(cursor,
			static_cast<oa::Usize>(storage + sizeof(storage) - cursor)));
		return;
	}
	oa::String value;
	if (negative) value.pushBack('-');
	else if (inSpec.sign != 0) value.pushBack(inSpec.sign);
	if (inSpec.alternate and type != 'd') {
		if (type == 'o') {
			if (magnitude != 0U) value.pushBack('0');
		} else {
			value.pushBack('0');
			value.pushBack(type == 'X' ? 'X' : type);
		}
	}
	appendUnsignedDigits(value, magnitude, base, type == 'X');
	appendAligned(out, value.view(), inSpec, true);
}

template<typename Output>
inline void appendString(Output& out, oa::StringView inValue, const FormatSpec& inSpec) {
	requireFormat(inSpec.type == 0 or inSpec.type == 's',
		"string format type must be s");
	requireFormat(inSpec.sign == 0 and not inSpec.alternate and not inSpec.zero,
		"numeric flags are invalid for strings");
	if (inSpec.precision >= 0 and static_cast<oa::Usize>(inSpec.precision) < inValue.size()) {
		inValue = inValue.subStr(0U, static_cast<oa::Usize>(inSpec.precision));
	}
	appendAligned(out, inValue, inSpec, false);
}

template<typename Output, typename T>
inline void appendArgumentValue(Output& out, T&& inValue, const FormatSpec& spec) {
	using Value = oa::RemoveCvrefT<T>;
	if constexpr (oa::IsSameV<Value, oa::String>) {
		appendString(out, inValue.view(), spec);
	} else if constexpr (oa::IsSameV<Value, oa::StringView>) {
		appendString(out, inValue, spec);
	} else if constexpr (oa::IsSameV<Value, decltype(nullptr)>) {
		requireFormat(spec.type == 0 or spec.type == 'p',
			"pointer format type must be p");
		FormatSpec pointerSpec = spec;
		pointerSpec.type = 'x';
		pointerSpec.alternate = true;
		appendInteger(out, oa::Usize{0}, pointerSpec);
	} else if constexpr (oa::IsConvertibleV<T, const char*>) {
		const char* text = inValue;
		requireFormat(text != nullptr, "cannot format a null C string");
		appendString(out, oa::StringView(text), spec);
	} else if constexpr (oa::IsSameV<Value, bool>) {
		if (spec.type == 0 or spec.type == 's') {
			appendString(out, inValue ? oa::StringView("true") : oa::StringView("false"), spec);
		} else {
			appendInteger(out, static_cast<oa::U8>(inValue ? 1U : 0U), spec);
		}
	} else if constexpr (oa::IsSameV<Value, char>) {
		if (spec.type == 0 or spec.type == 'c') {
			requireFormat(spec.sign == 0 and not spec.alternate and not spec.zero
				and spec.precision < 0, "numeric flags are invalid for characters");
			const char character[1]{inValue};
			appendAligned(out, oa::StringView(character, 1U), spec, false);
		} else {
			appendInteger(out, static_cast<unsigned char>(inValue), spec);
		}
	} else if constexpr (oa::IsIntegralV<Value>) {
		appendInteger(out, inValue, spec);
	} else if constexpr (oa::IsEnumV<Value>) {
		appendInteger(out, static_cast<__underlying_type(Value)>(inValue), spec);
	} else if constexpr (oa::IsSameV<Value, float> or oa::IsSameV<Value, double>) {
		const char type = spec.type == 0 ? 'g' : spec.type;
		requireFormat(type == 'f' or type == 'F' or type == 'e' or type == 'E'
			or type == 'g' or type == 'G',
			"floating format type must be f, F, e, E, g, or G");
		requireFormat(spec.precision <= 24, "floating precision exceeds 24 digits");
		char value[64];
		oa::Usize valueSize = 0U;
		requireFormat(formatFloatValue(static_cast<oa::F64>(inValue), type,
			spec.precision, spec.alternate, value, sizeof(value), valueSize),
			"floating formatting failed");
		oa::StringView valueView(value, valueSize);
		char signedValue[65];
		if (not valueView.empty() and valueView[0] != '-' and spec.sign != 0) {
			signedValue[0] = spec.sign;
			oa::memcpy(signedValue + 1U, value, valueSize);
			valueView = oa::StringView(signedValue, valueSize + 1U);
		}
		FormatSpec alignedSpec = spec;
		if (not __builtin_isfinite(static_cast<oa::F64>(inValue))) {
			alignedSpec.zero = false;
		}
		appendAligned(out, valueView, alignedSpec, true);
	} else if constexpr (oa::IsPointerV<Value>) {
		requireFormat(spec.type == 0 or spec.type == 'p',
			"pointer format type must be p");
		FormatSpec pointerSpec = spec;
		pointerSpec.type = 'x';
		pointerSpec.alternate = true;
		appendInteger(out, reinterpret_cast<oa::Usize>(inValue), pointerSpec);
	} else if constexpr (requires { inValue.toString(); }) {
		const oa::String converted = inValue.toString();
		appendString(out, converted.view(), spec);
	} else {
		static_assert(sizeof(Value) == 0,
			"oa::format does not support this type; provide a toString() member");
	}
}

template<typename Output, typename T>
OA_FORMAT_FORCEINLINE void appendArgument(
	Output& out,
	T&& inValue,
	oa::StringView inSpecText
) {
	const FormatSpec spec = parseFormatSpec(inSpecText);
	appendArgumentValue(out, oa::forward<T>(inValue), spec);
}

template<typename Output>
OA_FORMAT_FORCEINLINE void appendFormatTail(Output& out, oa::StringView inFormat) {
	for (oa::Usize index = 0; index < inFormat.size(); ++index) {
		const char character = inFormat[index];
		if (character == '{' or character == '}') {
			requireFormat(index + 1U < inFormat.size() and inFormat[index + 1U] == character,
				character == '{' ? "format has too few arguments or an unmatched {"
					: "format contains an unmatched }");
			reserveAppend(out, 1U);
			out.pushBack(character);
			++index;
		} else {
			reserveAppend(out, 1U);
			out.pushBack(character);
		}
	}
}

template<typename Output>
OA_FORMAT_FORCEINLINE void appendFormat(Output& out, oa::StringView inFormat) {
	appendFormatTail(out, inFormat);
}

template<typename Output, typename T, typename... Rest>
OA_FORMAT_FORCEINLINE void appendFormat(
	Output& out,
	oa::StringView inFormat,
	T&& inValue,
	Rest&&... inRest
) {
	oa::Usize literalBegin = 0;
	for (oa::Usize index = 0; index < inFormat.size(); ++index) {
		const char character = inFormat[index];
		if (character == '}') {
			if (index + 1U < inFormat.size() and inFormat[index + 1U] == '}') {
				appendView(out, inFormat.subStr(literalBegin, index - literalBegin));
				reserveAppend(out, 1U);
				out.pushBack('}');
				++index;
				literalBegin = index + 1U;
				continue;
			}
			requireFormat(false, "format contains an unmatched }");
		}
		if (character != '{') continue;
		if (index + 1U < inFormat.size() and inFormat[index + 1U] == '{') {
			appendView(out, inFormat.subStr(literalBegin, index - literalBegin));
			reserveAppend(out, 1U);
			out.pushBack('{');
			++index;
			literalBegin = index + 1U;
			continue;
		}
		appendView(out, inFormat.subStr(literalBegin, index - literalBegin));
		const oa::Usize close = inFormat.find('}', index + 1U);
		requireFormat(close != oa::StringView::Npos, "format contains an unmatched {");
		const oa::StringView field = inFormat.subStr(index + 1U, close - index - 1U);
		requireFormat(field.empty() or field[0] == ':',
			"OA format fields are sequential: use {} or {:spec}");
		appendArgument(out, oa::forward<T>(inValue),
			field.empty() ? oa::StringView() : field.subStr(1U));
		appendFormat(out, inFormat.subStr(close + 1U), oa::forward<Rest>(inRest)...);
		return;
	}
	requireFormat(false, "format has more arguments than fields");
}

} // namespace detail

template<oa::Usize N, typename... Args>
[[nodiscard]] OA_FORMAT_FORCEINLINE oa::String format(const char (&inFormat)[N], Args&&... inArgs) {
	static_assert(N > 0U, "format string must be NUL terminated");
	constexpr oa::Usize reserveHint = (N - 1U) + sizeof...(Args) * 8U;
	if constexpr (reserveHint > oa::String::SsoCap
		and reserveHint <= oa::detail::FormatBuffer::InlineCap) {
		oa::detail::FormatBuffer out;
		oa::detail::appendFormat(out, oa::StringView(inFormat, N - 1U),
			oa::forward<Args>(inArgs)...);
		return oa::move(out).finish();
	} else {
		oa::String out;
		if constexpr (reserveHint > oa::String::SsoCap
			and reserveHint <= oa::detail::MaxFormattedBytes) out.reserve(reserveHint);
		oa::detail::appendFormat(out, oa::StringView(inFormat, N - 1U),
			oa::forward<Args>(inArgs)...);
		return out;
	}
}

[[nodiscard]] inline String toString(U32 inValue) { return oa::format("{}", inValue); }
[[nodiscard]] inline String toString(U64 inValue) { return oa::format("{}", inValue); }
[[nodiscard]] inline String toString(I64 inValue) { return oa::format("{}", inValue); }
[[nodiscard]] inline String toString(double inValue) { return oa::format("{}", inValue); }
[[nodiscard]] inline String toString(float inValue) { return oa::format("{}", inValue); }

} // namespace oa

#undef OA_FORMAT_FORCEINLINE
