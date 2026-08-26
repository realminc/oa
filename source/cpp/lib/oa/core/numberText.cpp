#include <oa/core/std/format.h>
#include <oa/core/std/limits.h>

#include <errno.h>
#include <locale.h>
#include <stdio.h>
#include <stdlib.h>

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
	_locale_t locale = ::_create_locale(LC_NUMERIC, "C");
	if (locale == nullptr) return false;
	const double value = ::_strtod_l(terminated.cStr(), &end, locale);
	::_free_locale(locale);
#else
	locale_t locale = ::newlocale(LC_NUMERIC_MASK, "C", nullptr);
	if (locale == nullptr) return false;
	const double value = ::strtod_l(terminated.cStr(), &end, locale);
	::freelocale(locale);
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
#ifdef _WIN32
	_locale_t locale = ::_create_locale(LC_NUMERIC, "C");
	if (locale == nullptr) return false;
	const int written = ::_snprintf_l(
		buffer, sizeof(buffer), "%.*g", locale, inPrecision, inValue);
	::_free_locale(locale);
#else
	locale_t locale = ::newlocale(LC_NUMERIC_MASK, "C", nullptr);
	if (locale == nullptr) return false;
	locale_t previous = ::uselocale(locale);
	const int written = ::snprintf(buffer, sizeof(buffer), "%.*g", inPrecision, inValue);
	if (previous != static_cast<locale_t>(0)) (void)::uselocale(previous);
	::freelocale(locale);
#endif
	if (written <= 0 or static_cast<oa::Usize>(written) >= sizeof(buffer)) return false;
	outText = oa::String(buffer, static_cast<oa::Usize>(written));
	return true;
}
