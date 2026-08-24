#pragma once

#include <cstdlib>
#include <filesystem>
#include <string>

inline std::filesystem::path tutorialRepositoryDir() {
	return std::filesystem::path(__FILE__)
		.parent_path()
		.parent_path()
		.parent_path()
		.parent_path()
		.parent_path();
}

inline std::filesystem::path tutorialVideoDataDir() {
	if (const char* env = std::getenv("OA_VIDEO_DATA"); env && *env) {
		return std::filesystem::path(env);
	}

	return tutorialRepositoryDir() / "dataset" / "video";
}

inline std::string tutorialVideoPath(const char* inFilename) {
	return (tutorialVideoDataDir() / inFilename).lexically_normal().string();
}

inline std::filesystem::path tutorialVideoAssetDir() {
	return tutorialRepositoryDir() / "asset" / "video";
}

inline std::string tutorialVideoAssetPath(const char* inFilename) {
	return (tutorialVideoAssetDir() / inFilename).lexically_normal().string();
}
