#pragma once

// OaStdFilesystem — PascalCase façade over `std::filesystem` using `OaStdPath`.
//
// Thin delegates today (error_code internally; no throws from these entry points).
// **`OaFileIo`** remains the app-facing API (`OaStatus` / logging policy). This module is
// the OaStd-shaped surface for parity tests and future native VFS — see `docs/OaStd.md`.
//
// Depends on: `path.h` (`OaStdPath`).

#include <Oa/Core/Std/Path.h>

#include <cstdint>
#include <filesystem>
#include <system_error>

struct OaStdFilesystem {
	OaStdFilesystem() = delete;

	[[nodiscard]] static bool Exists(const OaStdPath& InPath) {
		std::error_code ec;
		const bool ok = std::filesystem::exists(InPath.StdPath(), ec);
		return ok && !ec;
	}

	[[nodiscard]] static bool IsRegularFile(const OaStdPath& InPath) {
		std::error_code ec;
		const bool ok = std::filesystem::is_regular_file(InPath.StdPath(), ec);
		return ok && !ec;
	}

	[[nodiscard]] static bool IsDirectory(const OaStdPath& InPath) {
		std::error_code ec;
		const bool ok = std::filesystem::is_directory(InPath.StdPath(), ec);
		return ok && !ec;
	}

	[[nodiscard]] static bool IsSymlink(const OaStdPath& InPath) {
		std::error_code ec;
		const bool ok = std::filesystem::is_symlink(InPath.StdPath(), ec);
		return ok && !ec;
	}

	[[nodiscard]] static bool Equivalent(const OaStdPath& InLeft, const OaStdPath& InRight) {
		std::error_code ec;
		const bool ok = std::filesystem::equivalent(InLeft.StdPath(), InRight.StdPath(), ec);
		return ok && !ec;
	}

	[[nodiscard]] static OaStdPath CurrentPath() {
		std::error_code ec;
		const std::filesystem::path p = std::filesystem::current_path(ec);
		if (ec) {
			return OaStdPath{};
		}
		return OaStdPath(p);
	}

	[[nodiscard]] static OaStdPath Absolute(const OaStdPath& InPath) {
		std::error_code ec;
		const std::filesystem::path p = std::filesystem::absolute(InPath.StdPath(), ec);
		if (ec) {
			return OaStdPath{};
		}
		return OaStdPath(p);
	}

	[[nodiscard]] static bool CreateDirectory(const OaStdPath& InPath) {
		std::error_code ec;
		(void)std::filesystem::create_directory(InPath.StdPath(), ec);
		return !ec;
	}

	[[nodiscard]] static bool CreateDirectories(const OaStdPath& InPath) {
		std::error_code ec;
		(void)std::filesystem::create_directories(InPath.StdPath(), ec);
		return !ec;
	}

	[[nodiscard]] static bool Remove(const OaStdPath& InPath) {
		std::error_code ec;
		const bool ok = std::filesystem::remove(InPath.StdPath(), ec);
		return ok && !ec;
	}

	[[nodiscard]] static bool RemoveAll(const OaStdPath& InPath) {
		std::error_code ec;
		(void)std::filesystem::remove_all(InPath.StdPath(), ec);
		return !ec;
	}

	[[nodiscard]] static bool Rename(const OaStdPath& From, const OaStdPath& To) {
		std::error_code ec;
		std::filesystem::rename(From.StdPath(), To.StdPath(), ec);
		return !ec;
	}

	[[nodiscard]] static bool CopyFile(const OaStdPath& From, const OaStdPath& To, bool OverwriteExisting = false) {
		std::error_code ec;
		const auto opt = OverwriteExisting ? std::filesystem::copy_options::overwrite_existing
						 : std::filesystem::copy_options::none;
		const bool ok = std::filesystem::copy_file(From.StdPath(), To.StdPath(), opt, ec);
		return ok && !ec;
	}

	[[nodiscard]] static bool FileSize(const OaStdPath& InPath, std::uintmax_t* OutBytes) {
		if (!OutBytes) {
			return false;
		}
		std::error_code ec;
		const std::uintmax_t sz = std::filesystem::file_size(InPath.StdPath(), ec);
		if (ec) {
			return false;
		}
		*OutBytes = sz;
		return true;
	}
};
