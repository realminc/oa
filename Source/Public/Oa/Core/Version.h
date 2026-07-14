#pragma once

// OA version — single source of truth is the top-level VERSION file, injected at build
// time via the OA_VERSION_STRING compile definition (see CMakeLists.txt). The fallback
// below only applies when a translation unit is parsed without that definition (e.g. an
// IDE/tooling index pass); a real build always defines it.
#ifndef OA_VERSION_STRING
#define OA_VERSION_STRING "0.0.0-unknown"
#endif

// Runtime accessor for the OA library version, e.g. "0.7.0-dev".
[[nodiscard]] inline const char* OaVersion() noexcept {
	return OA_VERSION_STRING;
}
