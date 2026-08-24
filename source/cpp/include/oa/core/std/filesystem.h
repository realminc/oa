#pragma once

// StdFilesystem — PascalCase façade over `std::filesystem` using `oa::Path`.
//
// Thin delegates today (error_code internally; no throws from these entry points).
// `oa::Filesystem` is the app-facing API (`oa::Status` / logging policy). This module is
// the low-level boundary surface for parity tests and a future native VFS.
//
// Depends on: `path.h` (`oa::Path`).

#include <oa/core/std/path.h>

#include <cstdint>
#include <filesystem>
#include <system_error>

namespace oa {

struct StdFilesystem {
	StdFilesystem() = delete;

	[[nodiscard]] static bool exists(const oa::Path& inPath) {
		std::error_code ec;
		const bool ok = std::filesystem::exists(inPath.stdPath(), ec);
		return ok && !ec;
	}

	[[nodiscard]] static bool isRegularFile(const oa::Path& inPath) {
		std::error_code ec;
		const bool ok = std::filesystem::is_regular_file(inPath.stdPath(), ec);
		return ok && !ec;
	}

	[[nodiscard]] static bool isDirectory(const oa::Path& inPath) {
		std::error_code ec;
		const bool ok = std::filesystem::is_directory(inPath.stdPath(), ec);
		return ok && !ec;
	}

	[[nodiscard]] static bool isSymlink(const oa::Path& inPath) {
		std::error_code ec;
		const bool ok = std::filesystem::is_symlink(inPath.stdPath(), ec);
		return ok && !ec;
	}

	[[nodiscard]] static bool equivalent(const oa::Path& inLeft, const oa::Path& inRight) {
		std::error_code ec;
		const bool ok = std::filesystem::equivalent(inLeft.stdPath(), inRight.stdPath(), ec);
		return ok && !ec;
	}

	[[nodiscard]] static oa::Path currentPath() {
		std::error_code ec;
		const std::filesystem::path p = std::filesystem::current_path(ec);
		if (ec) {
			return oa::Path{};
		}
		return oa::Path(p);
	}

	[[nodiscard]] static oa::Path absolute(const oa::Path& inPath) {
		std::error_code ec;
		const std::filesystem::path p = std::filesystem::absolute(inPath.stdPath(), ec);
		if (ec) {
			return oa::Path{};
		}
		return oa::Path(p);
	}

	[[nodiscard]] static bool createDirectory(const oa::Path& inPath) {
		std::error_code ec;
		(void)std::filesystem::create_directory(inPath.stdPath(), ec);
		return !ec;
	}

	[[nodiscard]] static bool createDirectories(const oa::Path& inPath) {
		std::error_code ec;
		(void)std::filesystem::create_directories(inPath.stdPath(), ec);
		return !ec;
	}

	[[nodiscard]] static bool remove(const oa::Path& inPath) {
		std::error_code ec;
		const bool ok = std::filesystem::remove(inPath.stdPath(), ec);
		return ok && !ec;
	}

	[[nodiscard]] static bool removeAll(const oa::Path& inPath) {
		std::error_code ec;
		(void)std::filesystem::remove_all(inPath.stdPath(), ec);
		return !ec;
	}

	[[nodiscard]] static bool rename(const oa::Path& inFrom, const oa::Path& inTo) {
		std::error_code ec;
		std::filesystem::rename(inFrom.stdPath(), inTo.stdPath(), ec);
		return !ec;
	}

	[[nodiscard]] static bool copyFile(const oa::Path& inFrom, const oa::Path& inTo, bool inOverwriteExisting = false) {
		std::error_code ec;
		const auto opt = inOverwriteExisting ? std::filesystem::copy_options::overwrite_existing
						 : std::filesystem::copy_options::none;
		const bool ok = std::filesystem::copy_file(inFrom.stdPath(), inTo.stdPath(), opt, ec);
		return ok && !ec;
	}

	[[nodiscard]] static bool fileSize(const oa::Path& inPath, std::uintmax_t* outBytes) {
		if (!outBytes) {
			return false;
		}
		std::error_code ec;
		const std::uintmax_t sz = std::filesystem::file_size(inPath.stdPath(), ec);
		if (ec) {
			return false;
		}
		*outBytes = sz;
		return true;
	}
};

} // namespace oa
