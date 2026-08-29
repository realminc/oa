#pragma once

#include <oa/core/paths.h>

inline oa::String tutorialVideoPath(const char* inFilename) {
	const oa::Path path = oa::Paths::data("video") / inFilename;
	return path.lexicallyNormal().string();
}

inline oa::String tutorialVideoAssetPath(const char* inFilename) {
	const oa::Path path = oa::Paths::asset("video") / inFilename;
	return path.lexicallyNormal().string();
}
