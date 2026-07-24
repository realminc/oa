// OA CORE - Named OA filesystem locations
//
// OaPaths resolves application locations. OaPath owns lexical path operations,
// while OaFilesystem performs host filesystem I/O.

#pragma once

#include <cstdlib>
#include <filesystem>

#include <Oa/Core/Types.h>

class OaPaths {
public:
	// Resolution: OA_ASSET_DIR > OA_REPO_ROOT/Asset > ./Asset.
	// These methods are inline so a consuming repository's OA_REPO_ROOT is used
	// at the call site rather than the path of the OA library build.
	[[nodiscard]] static inline OaPath Asset() {
		const char* env = std::getenv("OA_ASSET_DIR");
		if (env != nullptr && env[0] != '\0') {
			return OaPath(env);
		}
#ifdef OA_REPO_ROOT
		return OaPath(OA_REPO_ROOT) / "Asset";
#else
		return OaPath("Asset");
#endif
	}

	[[nodiscard]] static inline OaPath Asset(OaStringView InRelative) {
		return Asset() / OaPath(InRelative);
	}

	// Resolution: OA_VAR_DIR > OA_REPO_ROOT/var > ./var.
	[[nodiscard]] static inline OaPath Var() {
		const char* env = std::getenv("OA_VAR_DIR");
		if (env != nullptr && env[0] != '\0') {
			return OaPath(env);
		}
#ifdef OA_REPO_ROOT
		return OaPath(OA_REPO_ROOT) / "var";
#else
		return OaPath("var");
#endif
	}

	[[nodiscard]] static inline OaPath Var(OaStringView InRelative) {
		return Var() / OaPath(InRelative);
	}

	[[nodiscard]] static inline OaPath Current() {
		std::error_code ec;
		const auto path = std::filesystem::current_path(ec);
		return ec ? OaPath{} : OaPath(path);
	}

	[[nodiscard]] static inline OaPath Home() {
#ifdef _WIN32
		const char* home = std::getenv("USERPROFILE");
#else
		const char* home = std::getenv("HOME");
#endif
		return home != nullptr ? OaPath(home) : OaPath{};
	}

	[[nodiscard]] static inline OaPath Temp() {
		std::error_code ec;
		const auto path = std::filesystem::temp_directory_path(ec);
		return ec ? OaPath{} : OaPath(path);
	}
};
