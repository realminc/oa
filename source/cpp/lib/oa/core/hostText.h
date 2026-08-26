#pragma once

// Private hosted-library text bridge. Foundation string ownership remains in
// oa::String; only implementation and binding boundaries may depend on this.

#include <oa/core/std/string.h>

#include <string>

namespace oa::hostText {

[[nodiscard]] inline std::string copy(oa::StringView inText) {
	return std::string(inText.data(), inText.size());
}

[[nodiscard]] inline oa::String copy(const std::string& inText) {
	return oa::String(inText.data(), inText.size());
}

} // namespace oa::hostText
