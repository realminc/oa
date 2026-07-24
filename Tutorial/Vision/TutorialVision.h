#pragma once

#include <cstdlib>
#include <filesystem>
#include <string>

inline std::filesystem::path TutorialVideoDataDir() {
	if (const char* env = std::getenv("OA_VIDEO_DATA"); env && *env) {
		return std::filesystem::path(env);
	}

	const auto oaDir = std::filesystem::path(__FILE__)
		.parent_path()
		.parent_path()
		.parent_path();
	return oaDir.parent_path() / "dataset" / "video";
}

inline std::string TutorialVideoPath(const char* InFilename) {
	return (TutorialVideoDataDir() / InFilename).lexically_normal().string();
}

inline std::filesystem::path TutorialVideoAssetDir() {
	const auto oaDir = std::filesystem::path(__FILE__)
		.parent_path()
		.parent_path()
		.parent_path();
	return oaDir / "Asset" / "Video";
}

inline std::string TutorialVideoAssetPath(const char* InFilename) {
	return (TutorialVideoAssetDir() / InFilename).lexically_normal().string();
}
