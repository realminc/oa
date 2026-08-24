// ═══════════════════════════════════════════════════════════════════════════════
// OA CORE - Filesystem Implementation
// ═══════════════════════════════════════════════════════════════════════════════

#include <oa/core/filesystem.h>
#include <oa/core/std/algo.h>

#include <chrono>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <system_error>

namespace {

oa::Status filesystemError(const std::error_code& inError) {
	if (inError == std::errc::no_such_file_or_directory) {
		return oa::Status::error(oa::StatusCode::FileNotFound,
			oa::String(inError.message()));
	}
	if (inError == std::errc::permission_denied) {
		return oa::Status::error(oa::StatusCode::PermissionError,
			oa::String(inError.message()));
	}
	if (inError == std::errc::no_space_on_device) {
		return oa::Status::error(oa::StatusCode::DiskFull,
			oa::String(inError.message()));
	}
	if (inError == std::errc::file_exists) {
		return oa::Status::error(oa::StatusCode::AlreadyExists,
			oa::String(inError.message()));
	}
	return oa::Status::error(oa::StatusCode::Internal, oa::String(inError.message()));
}

void sortPaths(oa::Vec<oa::Path>& inOutPaths) {
	if (inOutPaths.size() < 2U) {
		return;
	}
	oa::sort(inOutPaths.begin(), inOutPaths.end(),
		[](const oa::Path& inA, const oa::Path& inB) {
			return inA.genericString() < inB.genericString();
		});
}

} // namespace

// ─── Existence & Info ────────────────────────────────────────────────────────

bool oa::Filesystem::exists(const oa::Path& inPath) {
	std::error_code ec;
	return std::filesystem::exists(inPath, ec);
}

bool oa::Filesystem::isFile(const oa::Path& inPath) {
	std::error_code ec;
	return std::filesystem::is_regular_file(inPath, ec);
}

bool oa::Filesystem::isDirectory(const oa::Path& inPath) {
	std::error_code ec;
	return std::filesystem::is_directory(inPath, ec);
}

oa::Result<oa::Usize> oa::Filesystem::getFileSize(const oa::Path& inPath) {
	std::error_code ec;
	const auto size = std::filesystem::file_size(inPath, ec);
	if (ec) {
		return filesystemError(ec);
	}
	if (size > static_cast<std::uintmax_t>(
			std::numeric_limits<oa::Usize>::max())) {
		return oa::Status::error(oa::StatusCode::Internal,
			"file size exceeds addressable memory: " + inPath.string());
	}
	return static_cast<oa::Usize>(size);
}

oa::Result<oa::I64> oa::Filesystem::getLastModified(const oa::Path& inPath) {
	std::error_code ec;
	auto time = std::filesystem::last_write_time(inPath, ec);
	if (ec) {
		return filesystemError(ec);
	}
	auto duration = time.time_since_epoch();
	return std::chrono::duration_cast<std::chrono::seconds>(duration).count();
}

// ─── directory operations ────────────────────────────────────────────────────

oa::Status oa::Filesystem::createDirectory(const oa::Path& inPath) {
	std::error_code ec;
	if (!std::filesystem::create_directory(inPath, ec) && ec) {
		return filesystemError(ec);
	}
	return oa::Status::ok();
}

oa::Status oa::Filesystem::createDirectories(const oa::Path& inPath) {
	std::error_code ec;
	if (!std::filesystem::create_directories(inPath, ec) && ec) {
		return filesystemError(ec);
	}
	return oa::Status::ok();
}

oa::Status oa::Filesystem::removeFile(const oa::Path& inPath) {
	std::error_code ec;
	if (!std::filesystem::remove(inPath, ec) && ec) {
		return filesystemError(ec);
	}
	return oa::Status::ok();
}

oa::Status oa::Filesystem::removeDirectory(const oa::Path& inPath, bool inRecursive) {
	std::error_code ec;
	if (inRecursive) {
		std::filesystem::remove_all(inPath, ec);
	} else {
		std::filesystem::remove(inPath, ec);
	}
	if (ec) {
		return filesystemError(ec);
	}
	return oa::Status::ok();
}

oa::Status oa::Filesystem::copy(const oa::Path& inFrom, const oa::Path& inTo) {
	std::error_code ec;
	std::filesystem::copy(inFrom, inTo, std::filesystem::copy_options::overwrite_existing, ec);
	if (ec) {
		return filesystemError(ec);
	}
	return oa::Status::ok();
}

oa::Status oa::Filesystem::move(const oa::Path& inFrom, const oa::Path& inTo) {
	std::error_code ec;
	std::filesystem::rename(inFrom, inTo, ec);
	if (ec) {
		return filesystemError(ec);
	}
	return oa::Status::ok();
}

// ─── Listing ─────────────────────────────────────────────────────────────────

oa::Result<oa::Vec<oa::Path>> oa::Filesystem::listFiles(const oa::Path& inDir, oa::StringView inExtension) {
	if (!isDirectory(inDir)) {
		return oa::Status::notFound("directory does not exist: " + inDir.string());
	}

	oa::Vec<oa::Path> files;
	std::error_code ec;
	std::filesystem::directory_iterator it(inDir.stdPath(), ec);
	const std::filesystem::directory_iterator end;
	if (ec) {
		return filesystemError(ec);
	}
	while (it != end) {
		const auto& entry = *it;
		if (entry.is_regular_file(ec)) {
			const std::string extNative = entry.path().extension().string();
			const oa::StringView extView(extNative.data(), extNative.size());
			if (inExtension.empty() || inExtension.equals(extView)) {
				files.pushBack(oa::Path(entry.path()));
			}
		}
		if (ec) {
			return filesystemError(ec);
		}
		it.increment(ec);
		if (ec) {
			return filesystemError(ec);
		}
	}
	sortPaths(files);
	return files;
}

oa::Result<oa::Vec<oa::Path>> oa::Filesystem::listDirectories(const oa::Path& inDir) {
	if (!isDirectory(inDir)) {
		return oa::Status::notFound("directory does not exist: " + inDir.string());
	}

	oa::Vec<oa::Path> dirs;
	std::error_code ec;
	std::filesystem::directory_iterator it(inDir.stdPath(), ec);
	const std::filesystem::directory_iterator end;
	if (ec) {
		return filesystemError(ec);
	}
	while (it != end) {
		const auto& entry = *it;
		if (entry.is_directory(ec)) {
			dirs.pushBack(oa::Path(entry.path()));
		}
		if (ec) {
			return filesystemError(ec);
		}
		it.increment(ec);
		if (ec) {
			return filesystemError(ec);
		}
	}
	sortPaths(dirs);
	return dirs;
}

oa::Result<oa::Vec<oa::Path>> oa::Filesystem::listAll(const oa::Path& inDir, bool inRecursive) {
	if (!isDirectory(inDir)) {
		return oa::Status::notFound("directory does not exist: " + inDir.string());
	}

	oa::Vec<oa::Path> entries;
	std::error_code ec;

	if (inRecursive) {
		std::filesystem::recursive_directory_iterator it(inDir.stdPath(), ec);
		const std::filesystem::recursive_directory_iterator end;
		if (ec) {
			return filesystemError(ec);
		}
		while (it != end) {
			entries.pushBack(oa::Path(it->path()));
			it.increment(ec);
			if (ec) {
				return filesystemError(ec);
			}
		}
	} else {
		std::filesystem::directory_iterator it(inDir.stdPath(), ec);
		const std::filesystem::directory_iterator end;
		if (ec) {
			return filesystemError(ec);
		}
		while (it != end) {
			entries.pushBack(oa::Path(it->path()));
			it.increment(ec);
			if (ec) {
				return filesystemError(ec);
			}
		}
	}
	sortPaths(entries);
	return entries;
}

// ─── text file operations ────────────────────────────────────────────────────

oa::Result<oa::String> oa::Filesystem::readText(const oa::Path& inPath) {
	std::ifstream file(inPath.stdPath());
	if (!file) {
		return oa::Status::notFound("Cannot open file: " + inPath.string());
	}

	std::ostringstream stream;
	stream << file.rdbuf();
	if (file.bad()) {
		return oa::Status::error(oa::StatusCode::Internal,
			"Failed to read file: " + inPath.string());
	}
	return oa::String(stream.str());
}

oa::Status oa::Filesystem::writeText(const oa::Path& inPath, oa::StringView inContent) {
	const oa::Path parent = inPath.parentPath();
	if (!parent.empty()) {
		OA_RETURN_IF_ERROR(createDirectories(parent));
	}

	std::ofstream file(inPath.stdPath());
	if (!file) {
		return oa::Status::error(oa::StatusCode::FileNotFound, "Cannot create file: " + inPath.string());
	}

	file << inContent;
	if (!file) {
		return oa::Status::error(oa::StatusCode::Internal,
			"Failed to write file: " + inPath.string());
	}
	return oa::Status::ok();
}

oa::Status oa::Filesystem::appendText(const oa::Path& inPath, oa::StringView inContent) {
	const oa::Path parent = inPath.parentPath();
	if (!parent.empty()) {
		OA_RETURN_IF_ERROR(createDirectories(parent));
	}

	std::ofstream file(inPath.stdPath(), std::ios::app);
	if (!file) {
		return oa::Status::error(oa::StatusCode::FileNotFound, "Cannot open file for append: " + inPath.string());
	}

	file << inContent;
	if (!file) {
		return oa::Status::error(oa::StatusCode::Internal,
			"Failed to append file: " + inPath.string());
	}
	return oa::Status::ok();
}

oa::Result<oa::Vec<oa::String>> oa::Filesystem::readLines(const oa::Path& inPath) {
	std::ifstream file(inPath.stdPath());
	if (!file) {
		return oa::Status::notFound("Cannot open file: " + inPath.string());
	}

	oa::Vec<oa::String> lines;
	oa::String line;
	char nextCh = '\0';
	while (file.get(nextCh)) {
		if (nextCh == '\n') {
			if (!line.empty() && line.back() == '\r') {
				line.popBack();
			}
			lines.pushBack(std::move(line));
			line = oa::String();
		} else {
			line.pushBack(nextCh);
		}
	}
	if (!line.empty()) {
		lines.pushBack(std::move(line));
	}
	if (file.bad()) {
		return oa::Status::error(oa::StatusCode::Internal, "read error: " + inPath.string());
	}

	return lines;
}

// ─── Binary file operations ─────────────────────────────────────────────────

oa::Result<oa::Vec<oa::U8>> oa::Filesystem::readBinary(const oa::Path& inPath) {
	std::ifstream file(inPath.stdPath(), std::ios::binary | std::ios::ate);
	if (!file) {
		return oa::Status::notFound("Cannot open file: " + inPath.string());
	}

	const auto end = file.tellg();
	const std::streamoff sizeOffset = end;
	if (sizeOffset < 0 ||
		static_cast<oa::U64>(sizeOffset) >
			static_cast<oa::U64>(std::numeric_limits<std::streamsize>::max())) {
		return oa::Status::error(oa::StatusCode::Internal,
			"Invalid file size: " + inPath.string());
	}
	const auto size = static_cast<std::streamsize>(sizeOffset);
	file.seekg(0, std::ios::beg);
	if (!file) {
		return oa::Status::error(oa::StatusCode::Internal,
			"Failed to seek file: " + inPath.string());
	}

	oa::Vec<oa::U8> data(static_cast<oa::Usize>(size));
	if (size > 0 && !file.read(reinterpret_cast<char*>(data.data()), size)) {
		return oa::Status::error(oa::StatusCode::Internal, "Failed to read file: " + inPath.string());
	}

	return data;
}

oa::Status oa::Filesystem::writeBinary(const oa::Path& inPath, oa::Span<const oa::U8> inData) {
	const oa::Path parent = inPath.parentPath();
	if (!parent.empty()) {
		OA_RETURN_IF_ERROR(createDirectories(parent));
	}
	if (inData.size() > static_cast<oa::Usize>(
			std::numeric_limits<std::streamsize>::max())) {
		return oa::Status::invalidArgument(
			"Binary payload exceeds stream size: " + inPath.string());
	}

	std::ofstream file(inPath.stdPath(), std::ios::binary);
	if (!file) {
		return oa::Status::error(oa::StatusCode::FileNotFound, "Cannot create file: " + inPath.string());
	}

	if (!inData.empty() &&
		!file.write(reinterpret_cast<const char*>(inData.data()),
			static_cast<std::streamsize>(inData.size()))) {
		return oa::Status::error(oa::StatusCode::Internal, "Failed to write file: " + inPath.string());
	}

	return oa::Status::ok();
}

// ─── Filesystem Resolution ──────────────────────────────────────────────────

oa::Result<oa::Path> oa::Filesystem::absolute(const oa::Path& inPath) {
	std::error_code ec;
	const auto path = std::filesystem::absolute(inPath.stdPath(), ec);
	if (ec) {
		return filesystemError(ec);
	}
	return oa::Path(path);
}

// ─── Glob Pattern Matching ──────────────────────────────────────────────────

/// Simple glob match: * matches zero or more chars, ? matches exactly one char
static bool globMatch(oa::StringView inPattern, oa::StringView inName) {
	oa::Usize p = 0;
	oa::Usize n = 0;
	oa::Usize starP = oa::StringView::Npos;
	oa::Usize starN = 0;

	while (n < inName.size()) {
		if (p < inPattern.size() && (inPattern[p] == inName[n] || inPattern[p] == '?')) {
			++p;
			++n;
		} else if (p < inPattern.size() && inPattern[p] == '*') {
			starP = p++;
			starN = n;
		} else if (starP != oa::StringView::Npos) {
			p = starP + 1;
			n = ++starN;
		} else {
			return false;
		}
	}

	while (p < inPattern.size() && inPattern[p] == '*') {
		++p;
	}
	return p == inPattern.size();
}

oa::Result<oa::Vec<oa::Path>> oa::Filesystem::glob(const oa::Path& inDir, oa::StringView inPattern) {
	if (!isDirectory(inDir)) {
		return oa::Status::notFound("directory does not exist: " + inDir.string());
	}

	oa::Vec<oa::Path> matches;
	std::error_code ec;
	std::filesystem::directory_iterator it(inDir.stdPath(), ec);
	const std::filesystem::directory_iterator end;
	if (ec) {
		return filesystemError(ec);
	}
	while (it != end) {
		const auto& entry = *it;
		const std::string nameNative = entry.path().filename().string();
		if (globMatch(inPattern, oa::StringView(nameNative.data(), nameNative.size()))) {
			matches.pushBack(oa::Path(entry.path()));
		}
		it.increment(ec);
		if (ec) {
			return filesystemError(ec);
		}
	}
	sortPaths(matches);
	return matches;
}
