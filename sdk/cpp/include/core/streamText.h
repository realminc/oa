#pragma once

// Hosted SDK serialization adapter. Core OA text types intentionally avoid
// iostream dependencies; SDK file emitters opt into this bridge explicitly.

#include <oa/core/std/string.h>

#include <ostream>
#include <string>

namespace oa::sdk {

[[nodiscard]] inline std::string toStdString(oa::StringView inText) {
	return std::string(inText.data(), inText.size());
}

[[nodiscard]] inline oa::String fromStdString(const std::string& inText) {
	return oa::String(inText.data(), inText.size());
}

} // namespace oa::sdk

#ifndef OA_HOSTED_TEXT_STREAM_OPERATORS
#define OA_HOSTED_TEXT_STREAM_OPERATORS

namespace oa {

inline std::ostream& operator<<(std::ostream& inOut, oa::StringView inValue) {
	if (!inValue.empty()) {
		inOut.write(inValue.data(), static_cast<std::streamsize>(inValue.size()));
	}
	return inOut;
}

inline std::ostream& operator<<(std::ostream& inOut, const oa::String& inValue) {
	return inOut << inValue.view();
}

} // namespace oa

#endif // OA_HOSTED_TEXT_STREAM_OPERATORS
