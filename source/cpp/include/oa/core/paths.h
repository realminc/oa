// OA CORE - Named OA filesystem locations
//
// Paths resolves application locations. oa::Path owns lexical path operations,
// while oa::Filesystem performs host filesystem I/O.

#pragma once

#include <cstdlib>
#include <filesystem>

#include <oa/core/types.h>

namespace oa {

class Paths {
public:
	// Resolution: OA_ASSET_DIR > OA_REPO_ROOT/asset > ./asset.
	// These methods are inline so a consuming repository's OA_REPO_ROOT is used
	// at the call site rather than the path of the OA library build.
	[[nodiscard]] static inline oa::Path asset() {
		const char* env = std::getenv("OA_ASSET_DIR");
		if (env != nullptr && env[0] != '\0') {
			return oa::Path(env);
		}
#ifdef OA_REPO_ROOT
		return oa::Path(OA_REPO_ROOT) / "asset";
#else
		return oa::Path("asset");
#endif
	}

	[[nodiscard]] static inline oa::Path asset(oa::StringView inRelative) {
		return asset() / oa::Path(inRelative);
	}

	// Resolution: OA_VAR_DIR > OA_REPO_ROOT/var > ./var.
	[[nodiscard]] static inline oa::Path var() {
		const char* env = std::getenv("OA_VAR_DIR");
		if (env != nullptr && env[0] != '\0') {
			return oa::Path(env);
		}
#ifdef OA_REPO_ROOT
		return oa::Path(OA_REPO_ROOT) / "var";
#else
		return oa::Path("var");
#endif
	}

	[[nodiscard]] static inline oa::Path var(oa::StringView inRelative) {
		return var() / oa::Path(inRelative);
	}

	[[nodiscard]] static inline oa::Path current() {
		std::error_code ec;
		const auto path = std::filesystem::current_path(ec);
		return ec ? oa::Path{} : oa::Path(path);
	}

	[[nodiscard]] static inline oa::Path home() {
#ifdef _WIN32
		const char* home = std::getenv("USERPROFILE");
#else
		const char* home = std::getenv("HOME");
#endif
		return home != nullptr ? oa::Path(home) : oa::Path{};
	}

	[[nodiscard]] static inline oa::Path temp() {
		std::error_code ec;
		const auto path = std::filesystem::temp_directory_path(ec);
		return ec ? oa::Path{} : oa::Path(path);
	}
};

} // namespace oa
