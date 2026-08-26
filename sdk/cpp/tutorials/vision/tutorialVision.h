#pragma once

#include <oa/core/paths.h>
#include <core/streamText.h>

#include <string>

inline std::string tutorialVideoPath(const char* inFilename) {
	const oa::Path path = oa::Paths::data("video") / inFilename;
	return oa::sdk::toStdString(path.lexicallyNormal().string());
}

inline std::string tutorialVideoAssetPath(const char* inFilename) {
	const oa::Path path = oa::Paths::asset("video") / inFilename;
	return oa::sdk::toStdString(path.lexicallyNormal().string());
}
