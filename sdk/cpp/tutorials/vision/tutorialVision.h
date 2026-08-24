#pragma once

#include <oa/core/paths.h>

#include <string>

inline std::string tutorialVideoPath(const char* inFilename) {
	const oa::Path path = oa::Paths::data("video") / inFilename;
	return path.lexicallyNormal().string().stdStr();
}

inline std::string tutorialVideoAssetPath(const char* inFilename) {
	const oa::Path path = oa::Paths::asset("video") / inFilename;
	return path.lexicallyNormal().string().stdStr();
}
