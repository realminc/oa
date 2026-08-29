#include <oa/core/std/format.h>
#include <oa/core/std/limits.h>

#include <errno.h>
#include <locale.h>
#include <stdio.h>
#include <stdlib.h>

namespace {

#ifdef _WIN32
[[nodiscard]] _locale_t cNumericLocale() noexcept {
	static _locale_t locale = ::_create_locale(LC_NUMERIC, "C");
	return locale;
}
#else
[[nodiscard]] locale_t cNumericLocale() noexcept {
	static locale_t locale = ::newlocale(LC_NUMERIC_MASK, "C", nullptr);
	return locale;
}
#endif

} // namespace

bool oa::parseU64(oa::StringView inText, oa::U64& outValue) noexcept {
	if (inText.empty()) return false;
	oa::U64 value = 0;
	for (const char character : inText) {
		if (character < '0' or character > '9') return false;
		const oa::U64 digit = static_cast<oa::U64>(character - '0');
		if (value > (oa::Limits<oa::U64>::max() - digit) / 10U) return false;
		value = value * 10U + digit;
	}
	outValue = value;
	return true;
}

bool oa::parseI64(oa::StringView inText, oa::I64& outValue) noexcept {
	if (inText.empty() or inText[0] == '+') return false;
	const bool negative = inText[0] == '-';
	if (negative) {
		inText.removePrefix(1U);
		if (inText.empty()) return false;
	}
	oa::U64 magnitude = 0;
	if (not oa::parseU64(inText, magnitude)) return false;
	const oa::U64 negativeLimit = static_cast<oa::U64>(oa::Limits<oa::I64>::max()) + 1U;
	if (negative) {
		if (magnitude > negativeLimit) return false;
		outValue = magnitude == negativeLimit
			? oa::Limits<oa::I64>::min()
			: -static_cast<oa::I64>(magnitude);
		return true;
	}
	if (magnitude > static_cast<oa::U64>(oa::Limits<oa::I64>::max())) return false;
	outValue = static_cast<oa::I64>(magnitude);
	return true;
}

bool oa::parseF64(oa::StringView inText, oa::F64& outValue) noexcept {
	if (inText.empty()) return false;
	oa::String terminated(inText);
	char* end = nullptr;
	errno = 0;
#ifdef _WIN32
	_locale_t locale = cNumericLocale();
	if (locale == nullptr) return false;
	const double value = ::_strtod_l(terminated.cStr(), &end, locale);
#else
	locale_t locale = cNumericLocale();
	if (locale == nullptr) return false;
	const double value = ::strtod_l(terminated.cStr(), &end, locale);
#endif
	if (errno == ERANGE or end != terminated.cStr() + terminated.size()) return false;
	outValue = value;
	return true;
}

bool oa::formatF64(
	oa::F64 inValue,
	oa::String& outText,
	oa::I32 inPrecision
) noexcept {
	if (inPrecision < 1 or inPrecision > 24) return false;
	char buffer[64]{};
	oa::Usize size = 0U;
	if (not oa::detail::formatFloatValue(
		inValue, 'g', inPrecision, false, buffer, sizeof(buffer), size)) return false;
	outText = oa::String(buffer, size);
	return true;
}

bool oa::detail::formatFloatValue(
	oa::F64 inValue,
	char inType,
	oa::I32 inPrecision,
	bool inAlternate,
	char* outText,
	oa::Usize inCapacity,
	oa::Usize& outSize
) noexcept {
	if (inType != 'f' and inType != 'F' and inType != 'e' and inType != 'E'
		and inType != 'g' and inType != 'G') return false;
	if (inPrecision < 0) inPrecision = 6;
	if (inPrecision > 24) return false;
	char format[6]{'%'};
	oa::Usize offset = 1U;
	if (inAlternate) format[offset++] = '#';
	format[offset++] = '.';
	format[offset++] = '*';
	format[offset++] = inType;
	format[offset] = '\0';
	if (outText == nullptr or inCapacity == 0U) return false;
#ifdef _WIN32
	_locale_t locale = cNumericLocale();
	if (locale == nullptr) return false;
	const int written = ::_snprintf_l(
		outText, inCapacity, format, locale, inPrecision, inValue);
#else
	locale_t locale = cNumericLocale();
	if (locale == nullptr) return false;
	locale_t previous = ::uselocale(locale);
	const int written = ::snprintf(outText, inCapacity, format, inPrecision, inValue);
	if (previous != static_cast<locale_t>(0)) (void)::uselocale(previous);
#endif
	if (written <= 0 or static_cast<oa::Usize>(written) >= inCapacity) return false;
	outSize = static_cast<oa::Usize>(written);
	return true;
}
