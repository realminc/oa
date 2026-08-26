#include <oa/core/envFlag.h>
#include <oa/core/log.h>

#include <stdlib.h>

namespace {

bool isFalsy(const char* v) {
	if (v == nullptr) return true;
	if (v[0] == '\0') return true;
	// Lowercase compare against {"0", "false", "no", "off"}
	auto eq = [](const char* a, const char* b) {
		auto lowerAscii = [](char inValue) {
			return inValue >= 'A' and inValue <= 'Z'
				? static_cast<char>(inValue + ('a' - 'A')) : inValue;
		};
		while (*a != '\0' && *b != '\0') {
			const char ca = lowerAscii(*a);
			const char cb = lowerAscii(*b);
			if (ca != cb) return false;
			++a; ++b;
		}
		return *a == '\0' && *b == '\0';
	};
	return eq(v, "0") || eq(v, "false") || eq(v, "no") || eq(v, "off");
}

} // namespace

bool oa::EnvFlag::isSet(const char* inName) {
	const char* v = ::getenv(inName);
	return !isFalsy(v);
}

oa::String oa::EnvFlag::getString(const char* inName, const char* inDefault) {
	const char* v = ::getenv(inName);
	if (v == nullptr || v[0] == '\0') {
		return oa::String(inDefault != nullptr ? inDefault : "");
	}
	return oa::String(v);
}

oa::I64 oa::EnvFlag::getInt(const char* inName, oa::I64 inDefault) {
	const char* v = ::getenv(inName);
	if (v == nullptr || v[0] == '\0') return inDefault;
	char* end = nullptr;
	long long parsed = ::strtoll(v, &end, 10);
	if (end == v || *end != '\0') return inDefault;  // not fully consumed
	return static_cast<oa::I64>(parsed);
}

bool oa::EnvFlag::setIfUnset(const char* inName, const char* inValue) {
	const char* existing = ::getenv(inName);
	if (existing != nullptr && existing[0] != '\0') {
		// User-supplied env wins.
		return false;
	}
#ifdef _WIN32
	_putenv_s(inName, inValue);
#else
	setenv(inName, inValue, /*overwrite=*/0);
#endif
	return true;
}

void oa::applyNumericMode(oa::NumericMode inMode) {
	if (inMode == oa::NumericMode::Fast) {
		return;
	}

	const char* modeStr =
		inMode == oa::NumericMode::Stable        ? "Stable" :
		inMode == oa::NumericMode::Deterministic ? "Deterministic" : "?";

	// Stable + Deterministic share the FP32 / no-CoopMat / no-DGC baseline.
	const bool setPrec = oa::EnvFlag::setIfUnset("OA_FORCE_PRECISION", "FP32");
	const bool setCm   = oa::EnvFlag::setIfUnset("OA_DISABLE_COOPMAT", "1");
	const bool setDgc  = oa::EnvFlag::setIfUnset("OA_DISABLE_DGC", "1");

	OaLogInfo(oa::LogComponent::Core,
		"oa::NumericMode=%s applied  OA_FORCE_PRECISION=%s "
		"OA_DISABLE_COOPMAT=%s OA_DISABLE_DGC=%s",
		modeStr,
		setPrec ? "FP32 (set)" : "(user-supplied, kept)",
		setCm   ? "1 (set)"    : "(user-supplied, kept)",
		setDgc  ? "1 (set)"    : "(user-supplied, kept)");

	if (inMode == oa::NumericMode::Deterministic) {
		const bool setPl = oa::EnvFlag::setIfUnset("OA_DISABLE_PERSISTENT_LOOP", "1");
		OaLogInfo(oa::LogComponent::Core,
			"oa::NumericMode=Deterministic added  OA_DISABLE_PERSISTENT_LOOP=%s",
			setPl ? "1 (set)" : "(user-supplied, kept)");
	}
}
