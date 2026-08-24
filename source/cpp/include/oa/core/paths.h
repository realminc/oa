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
	// Resolution: OA_ASSET_DIR > nearest source-tree sdk/asset > ./asset.
	// The ancestor probe makes source-tree SDK examples convenient without
	// embedding the library build checkout in installed binaries or wheels.
	[[nodiscard]] static inline oa::Path asset() {
		const char* env = std::getenv("OA_ASSET_DIR");
		if (env != nullptr && env[0] != '\0') {
			return oa::Path(env);
		}

		const auto sourceRoot = findSourceRoot();
		if (!sourceRoot.empty()) {
			return oa::Path(sourceRoot / "sdk" / "asset");
		}
		return oa::Path("asset");
	}

	[[nodiscard]] static inline oa::Path asset(oa::StringView inRelative) {
		return asset() / oa::Path(inRelative);
	}

	// Resolution: OA_VAR_DIR > nearest source-tree var > ./var.
	[[nodiscard]] static inline oa::Path var() {
		const char* env = std::getenv("OA_VAR_DIR");
		if (env != nullptr && env[0] != '\0') {
			return oa::Path(env);
		}
		const auto sourceRoot = findSourceRoot();
		return sourceRoot.empty()
			? oa::Path("var")
			: oa::Path(sourceRoot / "var");
	}

	[[nodiscard]] static inline oa::Path var(oa::StringView inRelative) {
		return var() / oa::Path(inRelative);
	}

	// Optional datasets are never downloaded implicitly. Tools and applications
	// agree on this root; callers choose the pack below it explicitly.
	// Resolution: OA_DATA_DIR > resolved var/data.
	[[nodiscard]] static inline oa::Path data() {
		const char* env = std::getenv("OA_DATA_DIR");
		if (env != nullptr && env[0] != '\0') {
			return oa::Path(env);
		}
		return var("data");
	}

	[[nodiscard]] static inline oa::Path data(oa::StringView inRelative) {
		return data() / oa::Path(inRelative);
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

private:
	[[nodiscard]] static inline std::filesystem::path findSourceRoot() {
		std::error_code ec;
		auto current = std::filesystem::current_path(ec);
		if (ec) return {};
		while (!current.empty()) {
			const auto candidate = current / "sdk" / "asset";
			if (std::filesystem::is_directory(candidate, ec) && !ec) {
				return current;
			}
			ec.clear();
			const auto parent = current.parent_path();
			if (parent == current) break;
			current = parent;
		}
		return {};
	}
};

} // namespace oa
