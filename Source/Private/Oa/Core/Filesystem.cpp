// ═══════════════════════════════════════════════════════════════════════════════
// OA CORE - Filesystem Implementation
// ═══════════════════════════════════════════════════════════════════════════════

#include <Oa/Core/Filesystem.h>
#include <Oa/Core/Std/Algo.h>

#include <chrono>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <system_error>

namespace {

OaStatus FilesystemError(const std::error_code& InError) {
	if (InError == std::errc::no_such_file_or_directory) {
		return OaStatus::Error(OaStatusCode::FileNotFound,
			OaString(InError.message()));
	}
	if (InError == std::errc::permission_denied) {
		return OaStatus::Error(OaStatusCode::PermissionError,
			OaString(InError.message()));
	}
	if (InError == std::errc::no_space_on_device) {
		return OaStatus::Error(OaStatusCode::DiskFull,
			OaString(InError.message()));
	}
	if (InError == std::errc::file_exists) {
		return OaStatus::Error(OaStatusCode::AlreadyExists,
			OaString(InError.message()));
	}
	return OaStatus::Error(OaStatusCode::Internal, OaString(InError.message()));
}

void SortPaths(OaVec<OaPath>& InOutPaths) {
	if (InOutPaths.Size() < 2U) {
		return;
	}
	OaStdSort(InOutPaths.Begin(), InOutPaths.End(),
		[](const OaPath& InA, const OaPath& InB) {
			return InA.GenericString() < InB.GenericString();
		});
}

} // namespace

// ─── Existence & Info ────────────────────────────────────────────────────────

bool OaFilesystem::Exists(const OaPath& InPath) {
	std::error_code ec;
	return std::filesystem::exists(InPath, ec);
}

bool OaFilesystem::IsFile(const OaPath& InPath) {
	std::error_code ec;
	return std::filesystem::is_regular_file(InPath, ec);
}

bool OaFilesystem::IsDirectory(const OaPath& InPath) {
	std::error_code ec;
	return std::filesystem::is_directory(InPath, ec);
}

OaResult<OaUsize> OaFilesystem::GetFileSize(const OaPath& InPath) {
	std::error_code ec;
	const auto size = std::filesystem::file_size(InPath, ec);
	if (ec) {
		return FilesystemError(ec);
	}
	if (size > static_cast<std::uintmax_t>(
			std::numeric_limits<OaUsize>::max())) {
		return OaStatus::Error(OaStatusCode::Internal,
			"File size exceeds addressable memory: " + InPath.String());
	}
	return static_cast<OaUsize>(size);
}

OaResult<OaI64> OaFilesystem::GetLastModified(const OaPath& InPath) {
	std::error_code ec;
	auto time = std::filesystem::last_write_time(InPath, ec);
	if (ec) {
		return FilesystemError(ec);
	}
	auto duration = time.time_since_epoch();
	return std::chrono::duration_cast<std::chrono::seconds>(duration).count();
}

// ─── Directory Operations ────────────────────────────────────────────────────

OaStatus OaFilesystem::CreateDirectory(const OaPath& InPath) {
	std::error_code ec;
	if (!std::filesystem::create_directory(InPath, ec) && ec) {
		return FilesystemError(ec);
	}
	return OaStatus::Ok();
}

OaStatus OaFilesystem::CreateDirectories(const OaPath& InPath) {
	std::error_code ec;
	if (!std::filesystem::create_directories(InPath, ec) && ec) {
		return FilesystemError(ec);
	}
	return OaStatus::Ok();
}

OaStatus OaFilesystem::RemoveFile(const OaPath& InPath) {
	std::error_code ec;
	if (!std::filesystem::remove(InPath, ec) && ec) {
		return FilesystemError(ec);
	}
	return OaStatus::Ok();
}

OaStatus OaFilesystem::RemoveDirectory(const OaPath& InPath, bool InRecursive) {
	std::error_code ec;
	if (InRecursive) {
		std::filesystem::remove_all(InPath, ec);
	} else {
		std::filesystem::remove(InPath, ec);
	}
	if (ec) {
		return FilesystemError(ec);
	}
	return OaStatus::Ok();
}

OaStatus OaFilesystem::Copy(const OaPath& InFrom, const OaPath& InTo) {
	std::error_code ec;
	std::filesystem::copy(InFrom, InTo, std::filesystem::copy_options::overwrite_existing, ec);
	if (ec) {
		return FilesystemError(ec);
	}
	return OaStatus::Ok();
}

OaStatus OaFilesystem::Move(const OaPath& InFrom, const OaPath& InTo) {
	std::error_code ec;
	std::filesystem::rename(InFrom, InTo, ec);
	if (ec) {
		return FilesystemError(ec);
	}
	return OaStatus::Ok();
}

// ─── Listing ─────────────────────────────────────────────────────────────────

OaResult<OaVec<OaPath>> OaFilesystem::ListFiles(const OaPath& InDir, OaStringView InExtension) {
	if (!IsDirectory(InDir)) {
		return OaStatus::NotFound("Directory does not exist: " + InDir.String());
	}

	OaVec<OaPath> files;
	std::error_code ec;
	std::filesystem::directory_iterator it(InDir.StdPath(), ec);
	const std::filesystem::directory_iterator end;
	if (ec) {
		return FilesystemError(ec);
	}
	while (it != end) {
		const auto& entry = *it;
		if (entry.is_regular_file(ec)) {
			const std::string extNative = entry.path().extension().string();
			const OaStringView extView(extNative.data(), extNative.size());
			if (InExtension.Empty() || InExtension.Equals(extView)) {
				files.PushBack(OaPath(entry.path()));
			}
		}
		if (ec) {
			return FilesystemError(ec);
		}
		it.increment(ec);
		if (ec) {
			return FilesystemError(ec);
		}
	}
	SortPaths(files);
	return files;
}

OaResult<OaVec<OaPath>> OaFilesystem::ListDirectories(const OaPath& InDir) {
	if (!IsDirectory(InDir)) {
		return OaStatus::NotFound("Directory does not exist: " + InDir.String());
	}

	OaVec<OaPath> dirs;
	std::error_code ec;
	std::filesystem::directory_iterator it(InDir.StdPath(), ec);
	const std::filesystem::directory_iterator end;
	if (ec) {
		return FilesystemError(ec);
	}
	while (it != end) {
		const auto& entry = *it;
		if (entry.is_directory(ec)) {
			dirs.PushBack(OaPath(entry.path()));
		}
		if (ec) {
			return FilesystemError(ec);
		}
		it.increment(ec);
		if (ec) {
			return FilesystemError(ec);
		}
	}
	SortPaths(dirs);
	return dirs;
}

OaResult<OaVec<OaPath>> OaFilesystem::ListAll(const OaPath& InDir, bool InRecursive) {
	if (!IsDirectory(InDir)) {
		return OaStatus::NotFound("Directory does not exist: " + InDir.String());
	}

	OaVec<OaPath> entries;
	std::error_code ec;

	if (InRecursive) {
		std::filesystem::recursive_directory_iterator it(InDir.StdPath(), ec);
		const std::filesystem::recursive_directory_iterator end;
		if (ec) {
			return FilesystemError(ec);
		}
		while (it != end) {
			entries.PushBack(OaPath(it->path()));
			it.increment(ec);
			if (ec) {
				return FilesystemError(ec);
			}
		}
	} else {
		std::filesystem::directory_iterator it(InDir.StdPath(), ec);
		const std::filesystem::directory_iterator end;
		if (ec) {
			return FilesystemError(ec);
		}
		while (it != end) {
			entries.PushBack(OaPath(it->path()));
			it.increment(ec);
			if (ec) {
				return FilesystemError(ec);
			}
		}
	}
	SortPaths(entries);
	return entries;
}

// ─── Text File Operations ────────────────────────────────────────────────────

OaResult<OaString> OaFilesystem::ReadText(const OaPath& InPath) {
	std::ifstream file(InPath.StdPath());
	if (!file) {
		return OaStatus::NotFound("Cannot open file: " + InPath.String());
	}

	std::ostringstream stream;
	stream << file.rdbuf();
	if (file.bad()) {
		return OaStatus::Error(OaStatusCode::Internal,
			"Failed to read file: " + InPath.String());
	}
	return OaString(stream.str());
}

OaStatus OaFilesystem::WriteText(const OaPath& InPath, OaStringView InContent) {
	const OaPath parent = InPath.ParentPath();
	if (!parent.Empty()) {
		OA_RETURN_IF_ERROR(CreateDirectories(parent));
	}

	std::ofstream file(InPath.StdPath());
	if (!file) {
		return OaStatus::Error(OaStatusCode::FileNotFound, "Cannot create file: " + InPath.String());
	}

	file << InContent;
	if (!file) {
		return OaStatus::Error(OaStatusCode::Internal,
			"Failed to write file: " + InPath.String());
	}
	return OaStatus::Ok();
}

OaStatus OaFilesystem::AppendText(const OaPath& InPath, OaStringView InContent) {
	const OaPath parent = InPath.ParentPath();
	if (!parent.Empty()) {
		OA_RETURN_IF_ERROR(CreateDirectories(parent));
	}

	std::ofstream file(InPath.StdPath(), std::ios::app);
	if (!file) {
		return OaStatus::Error(OaStatusCode::FileNotFound, "Cannot open file for append: " + InPath.String());
	}

	file << InContent;
	if (!file) {
		return OaStatus::Error(OaStatusCode::Internal,
			"Failed to append file: " + InPath.String());
	}
	return OaStatus::Ok();
}

OaResult<OaVec<OaString>> OaFilesystem::ReadLines(const OaPath& InPath) {
	std::ifstream file(InPath.StdPath());
	if (!file) {
		return OaStatus::NotFound("Cannot open file: " + InPath.String());
	}

	OaVec<OaString> lines;
	OaString line;
	char nextCh = '\0';
	while (file.get(nextCh)) {
		if (nextCh == '\n') {
			if (!line.Empty() && line.Back() == '\r') {
				line.PopBack();
			}
			lines.PushBack(std::move(line));
			line = OaString();
		} else {
			line.PushBack(nextCh);
		}
	}
	if (!line.Empty()) {
		lines.PushBack(std::move(line));
	}
	if (file.bad()) {
		return OaStatus::Error(OaStatusCode::Internal, "read error: " + InPath.String());
	}

	return lines;
}

// ─── Binary File Operations ─────────────────────────────────────────────────

OaResult<OaVec<OaU8>> OaFilesystem::ReadBinary(const OaPath& InPath) {
	std::ifstream file(InPath.StdPath(), std::ios::binary | std::ios::ate);
	if (!file) {
		return OaStatus::NotFound("Cannot open file: " + InPath.String());
	}

	const auto end = file.tellg();
	const std::streamoff sizeOffset = end;
	if (sizeOffset < 0 ||
		static_cast<OaU64>(sizeOffset) >
			static_cast<OaU64>(std::numeric_limits<std::streamsize>::max())) {
		return OaStatus::Error(OaStatusCode::Internal,
			"Invalid file size: " + InPath.String());
	}
	const auto size = static_cast<std::streamsize>(sizeOffset);
	file.seekg(0, std::ios::beg);
	if (!file) {
		return OaStatus::Error(OaStatusCode::Internal,
			"Failed to seek file: " + InPath.String());
	}

	OaVec<OaU8> data(static_cast<OaUsize>(size));
	if (size > 0 && !file.read(reinterpret_cast<char*>(data.Data()), size)) {
		return OaStatus::Error(OaStatusCode::Internal, "Failed to read file: " + InPath.String());
	}

	return data;
}

OaStatus OaFilesystem::WriteBinary(const OaPath& InPath, OaSpan<const OaU8> InData) {
	const OaPath parent = InPath.ParentPath();
	if (!parent.Empty()) {
		OA_RETURN_IF_ERROR(CreateDirectories(parent));
	}
	if (InData.size() > static_cast<OaUsize>(
			std::numeric_limits<std::streamsize>::max())) {
		return OaStatus::InvalidArgument(
			"Binary payload exceeds stream size: " + InPath.String());
	}

	std::ofstream file(InPath.StdPath(), std::ios::binary);
	if (!file) {
		return OaStatus::Error(OaStatusCode::FileNotFound, "Cannot create file: " + InPath.String());
	}

	if (!InData.empty() &&
		!file.write(reinterpret_cast<const char*>(InData.data()),
			static_cast<std::streamsize>(InData.size()))) {
		return OaStatus::Error(OaStatusCode::Internal, "Failed to write file: " + InPath.String());
	}

	return OaStatus::Ok();
}

// ─── Filesystem Resolution ──────────────────────────────────────────────────

OaResult<OaPath> OaFilesystem::Absolute(const OaPath& InPath) {
	std::error_code ec;
	const auto path = std::filesystem::absolute(InPath.StdPath(), ec);
	if (ec) {
		return FilesystemError(ec);
	}
	return OaPath(path);
}

// ─── Glob Pattern Matching ──────────────────────────────────────────────────

/// Simple glob match: * matches zero or more chars, ? matches exactly one char
static bool GlobMatch(OaStringView InPattern, OaStringView InName) {
	OaUsize p = 0;
	OaUsize n = 0;
	OaUsize starP = OaStringView::npos;
	OaUsize starN = 0;

	while (n < InName.size()) {
		if (p < InPattern.size() && (InPattern[p] == InName[n] || InPattern[p] == '?')) {
			++p;
			++n;
		} else if (p < InPattern.size() && InPattern[p] == '*') {
			starP = p++;
			starN = n;
		} else if (starP != OaStringView::npos) {
			p = starP + 1;
			n = ++starN;
		} else {
			return false;
		}
	}

	while (p < InPattern.size() && InPattern[p] == '*') {
		++p;
	}
	return p == InPattern.size();
}

OaResult<OaVec<OaPath>> OaFilesystem::Glob(const OaPath& InDir, OaStringView InPattern) {
	if (!IsDirectory(InDir)) {
		return OaStatus::NotFound("Directory does not exist: " + InDir.String());
	}

	OaVec<OaPath> matches;
	std::error_code ec;
	std::filesystem::directory_iterator it(InDir.StdPath(), ec);
	const std::filesystem::directory_iterator end;
	if (ec) {
		return FilesystemError(ec);
	}
	while (it != end) {
		const auto& entry = *it;
		const std::string nameNative = entry.path().filename().string();
		if (GlobMatch(InPattern, OaStringView(nameNative.data(), nameNative.size()))) {
			matches.PushBack(OaPath(entry.path()));
		}
		it.increment(ec);
		if (ec) {
			return FilesystemError(ec);
		}
	}
	SortPaths(matches);
	return matches;
}
