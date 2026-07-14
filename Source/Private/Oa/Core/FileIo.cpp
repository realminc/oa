// ═══════════════════════════════════════════════════════════════════════════════
// OA CORE - File I/O Implementation
// ═══════════════════════════════════════════════════════════════════════════════

#include <Oa/Core/FileIo.h>

#include <cstdlib>
#include <fstream>
#include <string>

// ─── Existence & Info ────────────────────────────────────────────────────────

bool OaFileIo::Exists(const OaPath& InPath) {
	std::error_code ec;
	return std::filesystem::exists(InPath, ec);
}

bool OaFileIo::IsFile(const OaPath& InPath) {
	std::error_code ec;
	return std::filesystem::is_regular_file(InPath, ec);
}

bool OaFileIo::IsDirectory(const OaPath& InPath) {
	std::error_code ec;
	return std::filesystem::is_directory(InPath, ec);
}

OaResult<OaUsize> OaFileIo::GetFileSize(const OaPath& InPath) {
	std::error_code ec;
	auto size = std::filesystem::file_size(InPath, ec);
	if (ec) {
		return OaStatus::NotFound(OaString(ec.message()));
	}
	return static_cast<OaUsize>(size);
}

OaResult<OaI64> OaFileIo::GetLastModified(const OaPath& InPath) {
	std::error_code ec;
	auto time = std::filesystem::last_write_time(InPath, ec);
	if (ec) {
		return OaStatus::NotFound(OaString(ec.message()));
	}
	auto duration = time.time_since_epoch();
	return std::chrono::duration_cast<std::chrono::seconds>(duration).count();
}

// ─── Directory Operations ────────────────────────────────────────────────────

OaStatus OaFileIo::CreateDirectory(const OaPath& InPath) {
	std::error_code ec;
	if (!std::filesystem::create_directory(InPath, ec) && ec) {
		return OaStatus::Error(OaStatusCode::Internal, OaString(ec.message()));
	}
	return OaStatus::Ok();
}

OaStatus OaFileIo::CreateDirectories(const OaPath& InPath) {
	std::error_code ec;
	if (!std::filesystem::create_directories(InPath, ec) && ec) {
		return OaStatus::Error(OaStatusCode::Internal, OaString(ec.message()));
	}
	return OaStatus::Ok();
}

OaStatus OaFileIo::RemoveFile(const OaPath& InPath) {
	std::error_code ec;
	if (!std::filesystem::remove(InPath, ec) && ec) {
		return OaStatus::Error(OaStatusCode::Internal, OaString(ec.message()));
	}
	return OaStatus::Ok();
}

OaStatus OaFileIo::RemoveDirectory(const OaPath& InPath, bool InRecursive) {
	std::error_code ec;
	if (InRecursive) {
		std::filesystem::remove_all(InPath, ec);
	} else {
		std::filesystem::remove(InPath, ec);
	}
	if (ec) {
		return OaStatus::Error(OaStatusCode::Internal, OaString(ec.message()));
	}
	return OaStatus::Ok();
}

OaStatus OaFileIo::Copy(const OaPath& InFrom, const OaPath& InTo) {
	std::error_code ec;
	std::filesystem::copy(InFrom, InTo, std::filesystem::copy_options::overwrite_existing, ec);
	if (ec) {
		return OaStatus::Error(OaStatusCode::Internal, OaString(ec.message()));
	}
	return OaStatus::Ok();
}

OaStatus OaFileIo::Move(const OaPath& InFrom, const OaPath& InTo) {
	std::error_code ec;
	std::filesystem::rename(InFrom, InTo, ec);
	if (ec) {
		return OaStatus::Error(OaStatusCode::Internal, OaString(ec.message()));
	}
	return OaStatus::Ok();
}

// ─── Listing ─────────────────────────────────────────────────────────────────

OaResult<OaVec<OaPath>> OaFileIo::ListFiles(const OaPath& InDir, OaStringView InExtension) {
	if (!IsDirectory(InDir)) {
		return OaStatus::NotFound("Directory does not exist: " + InDir.String());
	}

	OaVec<OaPath> files;
	std::error_code ec;

	for (const auto& entry : std::filesystem::directory_iterator(InDir, ec)) {
		if (entry.is_regular_file()) {
			const std::string extNative = entry.path().extension().string();
			const OaStringView extView(extNative.data(), extNative.size());
			if (InExtension.Empty() || InExtension.Equals(extView)) {
				files.PushBack(OaPath(entry.path()));
			}
		}
	}

	if (ec) {
		return OaStatus::Error(OaStatusCode::Internal, OaString(ec.message()));
	}

	return files;
}

OaResult<OaVec<OaPath>> OaFileIo::ListDirectories(const OaPath& InDir) {
	if (!IsDirectory(InDir)) {
		return OaStatus::NotFound("Directory does not exist: " + InDir.String());
	}

	OaVec<OaPath> dirs;
	std::error_code ec;

	for (const auto& entry : std::filesystem::directory_iterator(InDir, ec)) {
		if (entry.is_directory()) {
			dirs.PushBack(OaPath(entry.path()));
		}
	}

	if (ec) {
		return OaStatus::Error(OaStatusCode::Internal, OaString(ec.message()));
	}

	return dirs;
}

OaResult<OaVec<OaPath>> OaFileIo::ListAll(const OaPath& InDir, bool InRecursive) {
	if (!IsDirectory(InDir)) {
		return OaStatus::NotFound("Directory does not exist: " + InDir.String());
	}

	OaVec<OaPath> entries;
	std::error_code ec;

	if (InRecursive) {
		for (const auto& entry : std::filesystem::recursive_directory_iterator(InDir, ec)) {
			entries.PushBack(OaPath(entry.path()));
		}
	} else {
		for (const auto& entry : std::filesystem::directory_iterator(InDir, ec)) {
			entries.PushBack(OaPath(entry.path()));
		}
	}

	if (ec) {
		return OaStatus::Error(OaStatusCode::Internal, OaString(ec.message()));
	}

	return entries;
}

// ─── Text File Operations ────────────────────────────────────────────────────

OaResult<OaString> OaFileIo::ReadText(const OaPath& InPath) {
	std::ifstream file(InPath.StdPath());
	if (!file) {
		return OaStatus::NotFound("Cannot open file: " + InPath.String());
	}

	std::ostringstream stream;
	stream << file.rdbuf();
	return OaString(stream.str());
}

OaStatus OaFileIo::WriteText(const OaPath& InPath, OaStringView InContent) {
	if (InPath.has_parent_path()) {
		(void)CreateDirectories(InPath.parent_path());
	}

	std::ofstream file(InPath.StdPath());
	if (!file) {
		return OaStatus::Error(OaStatusCode::FileNotFound, "Cannot create file: " + InPath.String());
	}

	file << InContent;
	return OaStatus::Ok();
}

OaStatus OaFileIo::AppendText(const OaPath& InPath, OaStringView InContent) {
	std::ofstream file(InPath.StdPath(), std::ios::app);
	if (!file) {
		return OaStatus::Error(OaStatusCode::FileNotFound, "Cannot open file for append: " + InPath.String());
	}

	file << InContent;
	return OaStatus::Ok();
}

OaResult<OaVec<OaString>> OaFileIo::ReadLines(const OaPath& InPath) {
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

OaResult<OaVec<OaU8>> OaFileIo::ReadBinary(const OaPath& InPath) {
	std::ifstream file(InPath.StdPath(), std::ios::binary | std::ios::ate);
	if (!file) {
		return OaStatus::NotFound("Cannot open file: " + InPath.String());
	}

	auto size = file.tellg();
	file.seekg(0, std::ios::beg);

	OaVec<OaU8> data(static_cast<OaUsize>(size));
	if (!file.read(reinterpret_cast<char*>(data.Data()), size)) {
		return OaStatus::Error(OaStatusCode::Internal, "Failed to read file: " + InPath.String());
	}

	return data;
}

OaStatus OaFileIo::WriteBinary(const OaPath& InPath, OaSpan<const OaU8> InData) {
	if (InPath.has_parent_path()) {
		(void)CreateDirectories(InPath.parent_path());
	}

	std::ofstream file(InPath.StdPath(), std::ios::binary);
	if (!file) {
		return OaStatus::Error(OaStatusCode::FileNotFound, "Cannot create file: " + InPath.String());
	}

	if (!file.write(reinterpret_cast<const char*>(InData.data()),
		static_cast<std::streamsize>(InData.size()))) {
		return OaStatus::Error(OaStatusCode::Internal, "Failed to write file: " + InPath.String());
	}

	return OaStatus::Ok();
}

// ─── Path Utilities ─────────────────────────────────────────────────────────

OaPath OaFileIo::GetCurrentDirectory() {
	return OaPath(std::filesystem::current_path());
}

OaPath OaFileIo::GetHomeDirectory() {
#ifdef _WIN32
	const char* home = std::getenv("USERPROFILE");
#else
	const char* home = std::getenv("HOME");
#endif
	return (home != nullptr) ? OaPath(home) : OaPath{};
}

OaPath OaFileIo::GetTempDirectory() {
	return OaPath(std::filesystem::temp_directory_path());
}

OaPath OaFileIo::GetAbsolutePath(const OaPath& InPath) {
	return OaPath(std::filesystem::absolute(InPath));
}

OaPath OaFileIo::Join(const OaPath& InBase, const OaPath& InRelative) { return InBase / InRelative; }

OaString OaFileIo::GetExtension(const OaPath& InPath) {
	return InPath.Extension().String();
}

OaString OaFileIo::GetStem(const OaPath& InPath) {
	return InPath.Stem().String();
}

OaPath OaFileIo::GetParent(const OaPath& InPath) {
	return InPath.ParentPath();
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

OaResult<OaVec<OaPath>> OaFileIo::Glob(const OaPath& InDir, OaStringView InPattern) {
	if (!IsDirectory(InDir)) {
		return OaStatus::NotFound("Directory does not exist: " + InDir.String());
	}

	OaVec<OaPath> matches;
	std::error_code ec;

	for (const auto& entry : std::filesystem::directory_iterator(InDir, ec)) {
		const std::string nameNative = entry.path().filename().string();
		if (GlobMatch(InPattern, OaStringView(nameNative.data(), nameNative.size()))) {
			matches.PushBack(OaPath(entry.path()));
		}
	}

	return matches;
}
