#include <Oa/Core/EnvFlag.h>
#include <Oa/Core/Log.h>
#include <Oa/Runtime/Engine.h>     // OaNumericMode definition

#include <cstdlib>
#include <cctype>
#include <cstring>

namespace {

bool IsFalsy(const char* v) {
	if (v == nullptr) return true;
	if (v[0] == '\0') return true;
	// Lowercase compare against {"0", "false", "no", "off"}
	auto eq = [](const char* a, const char* b) {
		while (*a != '\0' && *b != '\0') {
			char ca = static_cast<char>(std::tolower(static_cast<unsigned char>(*a)));
			char cb = static_cast<char>(std::tolower(static_cast<unsigned char>(*b)));
			if (ca != cb) return false;
			++a; ++b;
		}
		return *a == '\0' && *b == '\0';
	};
	return eq(v, "0") || eq(v, "false") || eq(v, "no") || eq(v, "off");
}

} // namespace

bool OaEnvFlag::IsSet(const char* InName) {
	const char* v = std::getenv(InName);
	return !IsFalsy(v);
}

OaString OaEnvFlag::GetString(const char* InName, const char* InDefault) {
	const char* v = std::getenv(InName);
	if (v == nullptr || v[0] == '\0') {
		return OaString(InDefault != nullptr ? InDefault : "");
	}
	return OaString(v);
}

OaI64 OaEnvFlag::GetInt(const char* InName, OaI64 InDefault) {
	const char* v = std::getenv(InName);
	if (v == nullptr || v[0] == '\0') return InDefault;
	char* end = nullptr;
	long long parsed = std::strtoll(v, &end, 10);
	if (end == v || *end != '\0') return InDefault;  // not fully consumed
	return static_cast<OaI64>(parsed);
}

bool OaEnvFlag::SetIfUnset(const char* InName, const char* InValue) {
	const char* existing = std::getenv(InName);
	if (existing != nullptr && existing[0] != '\0') {
		// User-supplied env wins.
		return false;
	}
#ifdef _WIN32
	_putenv_s(InName, InValue);
#else
	setenv(InName, InValue, /*overwrite=*/0);
#endif
	return true;
}

void OaApplyNumericMode(OaNumericMode InMode) {
	if (InMode == OaNumericMode::Fast) {
		return;
	}

	const char* modeStr =
		InMode == OaNumericMode::Stable        ? "Stable" :
		InMode == OaNumericMode::Deterministic ? "Deterministic" : "?";

	// Stable + Deterministic share the FP32 / no-CoopMat / no-DGC baseline.
	const bool setPrec = OaEnvFlag::SetIfUnset("OA_FORCE_PRECISION", "FP32");
	const bool setCm   = OaEnvFlag::SetIfUnset("OA_DISABLE_COOPMAT", "1");
	const bool setDgc  = OaEnvFlag::SetIfUnset("OA_DISABLE_DGC", "1");

	OA_LOG_INFO(OaLogComponent::Core,
		"OaNumericMode=%s applied  OA_FORCE_PRECISION=%s "
		"OA_DISABLE_COOPMAT=%s OA_DISABLE_DGC=%s",
		modeStr,
		setPrec ? "FP32 (set)" : "(user-supplied, kept)",
		setCm   ? "1 (set)"    : "(user-supplied, kept)",
		setDgc  ? "1 (set)"    : "(user-supplied, kept)");

	if (InMode == OaNumericMode::Deterministic) {
		const bool setGo = OaEnvFlag::SetIfUnset("OA_DISABLE_GRAPH_OPTIMIZE", "1");
		const bool setPl = OaEnvFlag::SetIfUnset("OA_DISABLE_PERSISTENT_LOOP", "1");
		OA_LOG_INFO(OaLogComponent::Core,
			"OaNumericMode=Deterministic added  "
			"OA_DISABLE_GRAPH_OPTIMIZE=%s OA_DISABLE_PERSISTENT_LOOP=%s",
			setGo ? "1 (set)" : "(user-supplied, kept)",
			setPl ? "1 (set)" : "(user-supplied, kept)");
	}
}
