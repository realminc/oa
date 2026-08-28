#include <oa/core/filesystem.h>
#include <oa/core/std/algo.h>

#include <errno.h>
#include <stdio.h>
#include <string.h>

#if defined(_WIN32)
#include <windows.h>
#else
#include <dirent.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace {

[[nodiscard]] oa::Status filesystemError(
	int inError, oa::StringView inAction, const oa::Path& inPath)
{
	oa::StatusCode code = oa::StatusCode::Internal;
	switch (inError) {
		case ENOENT: code = oa::StatusCode::FileNotFound; break;
		case EACCES:
		case EPERM: code = oa::StatusCode::PermissionError; break;
		case ENOSPC: code = oa::StatusCode::DiskFull; break;
		case EEXIST: code = oa::StatusCode::AlreadyExists; break;
		default: break;
	}
	oa::String message(inAction);
	message += ": ";
	message += inPath.string();
	if (const char* detail = ::strerror(inError); detail != nullptr) {
		message += " (";
		message += detail;
		message += ')';
	}
	return oa::Status::error(code, oa::move(message));
}

void sortPaths(oa::Vector<oa::Path>& inOutPaths) {
	if (inOutPaths.size() < 2U) return;
	oa::sort(inOutPaths.begin(), inOutPaths.end(),
		[](const oa::Path& inA, const oa::Path& inB) {
			return inA.genericString() < inB.genericString();
		});
}

[[nodiscard]] oa::Status writeBytes(
	const oa::Path& inPath, const void* inData, oa::Usize inBytes, const char* inMode)
{
	::FILE* file = ::fopen(inPath.cStr(), inMode);
	if (file == nullptr) return filesystemError(errno, "cannot open file", inPath);
	if (inBytes != 0U and ::fwrite(inData, 1U, inBytes, file) != inBytes) {
		const int error = errno != 0 ? errno : EIO;
		(void)::fclose(file);
		return filesystemError(error, "cannot write file", inPath);
	}
	if (::fclose(file) != 0) {
		return filesystemError(errno != 0 ? errno : EIO, "cannot close file", inPath);
	}
	return oa::Status::ok();
}

#if defined(_WIN32)

[[nodiscard]] bool queryPath(
	const oa::Path& inPath, WIN32_FILE_ATTRIBUTE_DATA& outData) noexcept
{
	return ::GetFileAttributesExA(
		inPath.cStr(), GetFileExInfoStandard, &outData) != FALSE;
}

[[nodiscard]] oa::Status createOneDirectory(const oa::Path& inPath) {
	if (::CreateDirectoryA(inPath.cStr(), nullptr) != FALSE) return oa::Status::ok();
	const DWORD error = ::GetLastError();
	if (error == ERROR_ALREADY_EXISTS and oa::Filesystem::isDirectory(inPath)) {
		return oa::Status::ok();
	}
	return filesystemError(static_cast<int>(error), "cannot create directory", inPath);
}

template<typename Visitor>
[[nodiscard]] oa::Status visitDirectory(const oa::Path& inDir, Visitor&& inVisitor) {
	const oa::String pattern = (inDir / "*").string();
	WIN32_FIND_DATAA data{};
	HANDLE handle = ::FindFirstFileA(pattern.cStr(), &data);
	if (handle == INVALID_HANDLE_VALUE) {
		return filesystemError(
			static_cast<int>(::GetLastError()), "cannot list directory", inDir);
	}
	oa::Status status = oa::Status::ok();
	do {
		const oa::StringView name(data.cFileName);
		if (name == "." or name == "..") continue;
		status = inVisitor(inDir / name,
			(data.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) != 0,
			(data.dwFileAttributes & FILE_ATTRIBUTE_REPARSE_POINT) != 0);
		if (status.isError()) break;
	} while (::FindNextFileA(handle, &data) != FALSE);
	const DWORD endError = ::GetLastError();
	(void)::FindClose(handle);
	if (status.isError()) return status;
	if (endError != ERROR_NO_MORE_FILES) {
		return filesystemError(static_cast<int>(endError), "cannot list directory", inDir);
	}
	return oa::Status::ok();
}

#else

[[nodiscard]] bool queryPath(
	const oa::Path& inPath, struct stat& outData, bool inFollow = true) noexcept
{
	return (inFollow ? ::stat(inPath.cStr(), &outData)
		: ::lstat(inPath.cStr(), &outData)) == 0;
}

[[nodiscard]] oa::Status createOneDirectory(const oa::Path& inPath) {
	if (::mkdir(inPath.cStr(), 0777) == 0) return oa::Status::ok();
	if (errno == EEXIST and oa::Filesystem::isDirectory(inPath)) {
		return oa::Status::ok();
	}
	return filesystemError(errno, "cannot create directory", inPath);
}

template<typename Visitor>
[[nodiscard]] oa::Status visitDirectory(const oa::Path& inDir, Visitor&& inVisitor) {
	DIR* directory = ::opendir(inDir.cStr());
	if (directory == nullptr) return filesystemError(errno, "cannot list directory", inDir);
	oa::Status status = oa::Status::ok();
	errno = 0;
	while (dirent* entry = ::readdir(directory)) {
		const oa::StringView name(entry->d_name);
		if (name == "." or name == "..") continue;
		const oa::Path path = inDir / name;
		struct stat info{};
		if (not queryPath(path, info, false)) {
			status = filesystemError(errno, "cannot inspect directory entry", path);
			break;
		}
		status = inVisitor(path, S_ISDIR(info.st_mode), S_ISLNK(info.st_mode));
		if (status.isError()) break;
		errno = 0;
	}
	const int iterationError = errno;
	(void)::closedir(directory);
	if (status.isError()) return status;
	if (iterationError != 0) {
		return filesystemError(iterationError, "cannot list directory", inDir);
	}
	return oa::Status::ok();
}

#endif

[[nodiscard]] oa::Status removeTree(const oa::Path& inPath);

[[nodiscard]] oa::Status removeLeaf(const oa::Path& inEntry) {
#if defined(_WIN32)
	if (::DeleteFileA(inEntry.cStr()) == FALSE) {
		return filesystemError(
			static_cast<int>(::GetLastError()), "cannot remove file", inEntry);
	}
#else
	if (::unlink(inEntry.cStr()) != 0) {
		return filesystemError(errno, "cannot remove file", inEntry);
	}
#endif
	return oa::Status::ok();
}

[[nodiscard]] oa::Status removeTree(const oa::Path& inPath) {
	const oa::Status contents = visitDirectory(inPath,
		[](const oa::Path& inEntry, bool inDirectory, bool inSymlink) {
			return inDirectory and not inSymlink
				? removeTree(inEntry) : removeLeaf(inEntry);
		});
	if (contents.isError()) return contents;
#if defined(_WIN32)
	if (::RemoveDirectoryA(inPath.cStr()) == FALSE) {
		return filesystemError(
			static_cast<int>(::GetLastError()), "cannot remove directory", inPath);
	}
#else
	if (::rmdir(inPath.cStr()) != 0) {
		return filesystemError(errno, "cannot remove directory", inPath);
	}
#endif
	return oa::Status::ok();
}

[[nodiscard]] bool globMatch(oa::StringView inPattern, oa::StringView inName) {
	oa::Usize pattern = 0;
	oa::Usize name = 0;
	oa::Usize starPattern = oa::StringView::Npos;
	oa::Usize starName = 0;
	while (name < inName.size()) {
		if (pattern < inPattern.size()
			and (inPattern[pattern] == inName[name] or inPattern[pattern] == '?')) {
			++pattern;
			++name;
		} else if (pattern < inPattern.size() and inPattern[pattern] == '*') {
			starPattern = pattern++;
			starName = name;
		} else if (starPattern != oa::StringView::Npos) {
			pattern = starPattern + 1U;
			name = ++starName;
		} else {
			return false;
		}
	}
	while (pattern < inPattern.size() and inPattern[pattern] == '*') ++pattern;
	return pattern == inPattern.size();
}

} // namespace

bool oa::Filesystem::exists(const oa::Path& inPath) {
#if defined(_WIN32)
	WIN32_FILE_ATTRIBUTE_DATA data{};
#else
	struct stat data{};
#endif
	return queryPath(inPath, data);
}

bool oa::Filesystem::isFile(const oa::Path& inPath) {
#if defined(_WIN32)
	WIN32_FILE_ATTRIBUTE_DATA data{};
	return queryPath(inPath, data)
		and (data.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) == 0;
#else
	struct stat data{};
	return queryPath(inPath, data) and S_ISREG(data.st_mode);
#endif
}

bool oa::Filesystem::isDirectory(const oa::Path& inPath) {
#if defined(_WIN32)
	WIN32_FILE_ATTRIBUTE_DATA data{};
	return queryPath(inPath, data)
		and (data.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) != 0;
#else
	struct stat data{};
	return queryPath(inPath, data) and S_ISDIR(data.st_mode);
#endif
}

oa::Result<oa::Usize> oa::Filesystem::getFileSize(const oa::Path& inPath) {
#if defined(_WIN32)
	WIN32_FILE_ATTRIBUTE_DATA data{};
	if (not queryPath(inPath, data)) {
		return filesystemError(static_cast<int>(::GetLastError()), "cannot inspect file", inPath);
	}
	const oa::U64 size = (static_cast<oa::U64>(data.nFileSizeHigh) << 32U)
		| data.nFileSizeLow;
#else
	struct stat data{};
	if (not queryPath(inPath, data)) return filesystemError(errno, "cannot inspect file", inPath);
	if (data.st_size < 0) {
		return oa::Status::error(oa::StatusCode::Internal,
			"filesystem returned a negative file size");
	}
	const oa::U64 size = static_cast<oa::U64>(data.st_size);
#endif
	if (size > oa::Limits<oa::Usize>::max()) {
		return oa::Status::error(oa::StatusCode::Internal,
			"file size exceeds addressable memory: " + inPath.string());
	}
	return static_cast<oa::Usize>(size);
}

oa::Result<oa::I64> oa::Filesystem::getLastModified(const oa::Path& inPath) {
#if defined(_WIN32)
	WIN32_FILE_ATTRIBUTE_DATA data{};
	if (not queryPath(inPath, data)) {
		return filesystemError(static_cast<int>(::GetLastError()), "cannot inspect file", inPath);
	}
	ULARGE_INTEGER ticks{};
	ticks.HighPart = data.ftLastWriteTime.dwHighDateTime;
	ticks.LowPart = data.ftLastWriteTime.dwLowDateTime;
	constexpr oa::U64 EpochDelta100ns = 116444736000000000ULL;
	return static_cast<oa::I64>((ticks.QuadPart - EpochDelta100ns) / 10000000ULL);
#else
	struct stat data{};
	if (not queryPath(inPath, data)) return filesystemError(errno, "cannot inspect file", inPath);
	return static_cast<oa::I64>(data.st_mtime);
#endif
}

oa::Status oa::Filesystem::createDirectory(const oa::Path& inPath) {
	return createOneDirectory(inPath);
}

oa::Status oa::Filesystem::createDirectories(const oa::Path& inPath) {
	if (inPath.empty() or isDirectory(inPath)) return oa::Status::ok();
	const oa::Path normalized = inPath.lexicallyNormal();
	const oa::String text = normalized.string();
	oa::Usize cursor = normalized.isAbsolute() ? 1U : 0U;
	while (cursor <= text.size()) {
		oa::Usize slash = text.find('/', cursor);
		if (slash == oa::String::Npos) slash = text.size();
		if (slash != 0U) {
			const oa::Path component(text.view().subStr(0U, slash));
			if (not component.empty() and component.string() != ".") {
				OA_RETURN_IF_ERROR(createOneDirectory(component));
			}
		}
		if (slash == text.size()) break;
		cursor = slash + 1U;
	}
	return oa::Status::ok();
}

oa::Status oa::Filesystem::removeFile(const oa::Path& inPath) {
#if defined(_WIN32)
	if (::DeleteFileA(inPath.cStr()) != FALSE or ::GetLastError() == ERROR_FILE_NOT_FOUND) {
		return oa::Status::ok();
	}
	return filesystemError(static_cast<int>(::GetLastError()), "cannot remove file", inPath);
#else
	if (::unlink(inPath.cStr()) == 0 or errno == ENOENT) return oa::Status::ok();
	return filesystemError(errno, "cannot remove file", inPath);
#endif
}

oa::Status oa::Filesystem::removeDirectory(const oa::Path& inPath, bool inRecursive) {
	if (not exists(inPath)) return oa::Status::ok();
	if (inRecursive) return removeTree(inPath);
#if defined(_WIN32)
	if (::RemoveDirectoryA(inPath.cStr()) != FALSE) return oa::Status::ok();
	return filesystemError(static_cast<int>(::GetLastError()), "cannot remove directory", inPath);
#else
	if (::rmdir(inPath.cStr()) == 0) return oa::Status::ok();
	return filesystemError(errno, "cannot remove directory", inPath);
#endif
}

oa::Status oa::Filesystem::copy(const oa::Path& inFrom, const oa::Path& inTo) {
	if (isDirectory(inFrom)) return createDirectories(inTo);
	auto data = readBinary(inFrom);
	if (data.isError()) return data.getStatus();
	return writeBinary(inTo,
		oa::Span<const oa::U8>(data->data(), data->size()));
}

oa::Status oa::Filesystem::move(const oa::Path& inFrom, const oa::Path& inTo) {
	const oa::Path parent = inTo.parentPath();
	if (not parent.empty()) OA_RETURN_IF_ERROR(createDirectories(parent));
#if defined(_WIN32)
	if (::MoveFileExA(inFrom.cStr(), inTo.cStr(), MOVEFILE_REPLACE_EXISTING) != FALSE) {
		return oa::Status::ok();
	}
	return filesystemError(static_cast<int>(::GetLastError()), "cannot move path", inFrom);
#else
	if (::rename(inFrom.cStr(), inTo.cStr()) == 0) return oa::Status::ok();
	return filesystemError(errno, "cannot move path", inFrom);
#endif
}

oa::Result<oa::Vector<oa::Path>> oa::Filesystem::listFiles(
	const oa::Path& inDir, oa::StringView inExtension)
{
	if (not isDirectory(inDir)) {
		return oa::Status::notFound("directory does not exist: " + inDir.string());
	}
	oa::Vector<oa::Path> files;
	const oa::Status status = visitDirectory(inDir,
		[&](const oa::Path& inPath, bool inDirectory, bool) {
			if (not inDirectory and (inExtension.empty()
				or inPath.extension().string() == inExtension)) files.pushBack(inPath);
			return oa::Status::ok();
		});
	if (status.isError()) return status;
	sortPaths(files);
	return files;
}

oa::Result<oa::Vector<oa::Path>> oa::Filesystem::listDirectories(const oa::Path& inDir) {
	if (not isDirectory(inDir)) {
		return oa::Status::notFound("directory does not exist: " + inDir.string());
	}
	oa::Vector<oa::Path> directories;
	const oa::Status status = visitDirectory(inDir,
		[&](const oa::Path& inPath, bool inDirectory, bool inSymlink) {
			if (inDirectory and not inSymlink) directories.pushBack(inPath);
			return oa::Status::ok();
		});
	if (status.isError()) return status;
	sortPaths(directories);
	return directories;
}

oa::Result<oa::Vector<oa::Path>> oa::Filesystem::listAll(
	const oa::Path& inDir, bool inRecursive)
{
	if (not isDirectory(inDir)) {
		return oa::Status::notFound("directory does not exist: " + inDir.string());
	}
	oa::Vector<oa::Path> entries;
	const auto appendDirectory = [&](auto&& self, const oa::Path& inCurrent) -> oa::Status {
		return visitDirectory(inCurrent,
			[&](const oa::Path& inPath, bool inDirectory, bool inSymlink) {
				entries.pushBack(inPath);
				if (inRecursive and inDirectory and not inSymlink) return self(self, inPath);
				return oa::Status::ok();
			});
	};
	const oa::Status status = appendDirectory(appendDirectory, inDir);
	if (status.isError()) return status;
	sortPaths(entries);
	return entries;
}

oa::Result<oa::String> oa::Filesystem::readText(const oa::Path& inPath) {
	auto binary = readBinary(inPath);
	if (binary.isError()) return binary.getStatus();
	return oa::String(reinterpret_cast<const char*>(binary->data()), binary->size());
}

oa::Status oa::Filesystem::writeText(
	const oa::Path& inPath, oa::StringView inContent)
{
	const oa::Path parent = inPath.parentPath();
	if (not parent.empty()) OA_RETURN_IF_ERROR(createDirectories(parent));
	return writeBytes(inPath, inContent.data(), inContent.size(), "wb");
}

oa::Status oa::Filesystem::appendText(
	const oa::Path& inPath, oa::StringView inContent)
{
	const oa::Path parent = inPath.parentPath();
	if (not parent.empty()) OA_RETURN_IF_ERROR(createDirectories(parent));
	return writeBytes(inPath, inContent.data(), inContent.size(), "ab");
}

oa::Result<oa::Vector<oa::String>> oa::Filesystem::readLines(const oa::Path& inPath) {
	auto text = readText(inPath);
	if (text.isError()) return text.getStatus();
	oa::Vector<oa::String> lines;
	oa::Usize begin = 0;
	for (oa::Usize index = 0; index < text->size(); ++index) {
		if ((*text)[index] != '\n') continue;
		oa::Usize end = index;
		if (end > begin and (*text)[end - 1U] == '\r') --end;
		lines.emplaceBack(text->view().subStr(begin, end - begin));
		begin = index + 1U;
	}
	if (begin < text->size()) lines.emplaceBack(text->view().subStr(begin));
	return lines;
}

oa::Result<oa::Vector<oa::U8>> oa::Filesystem::readBinary(const oa::Path& inPath) {
	auto sizeResult = getFileSize(inPath);
	if (sizeResult.isError()) return sizeResult.getStatus();
	::FILE* file = ::fopen(inPath.cStr(), "rb");
	if (file == nullptr) return filesystemError(errno, "cannot open file", inPath);
	oa::Vector<oa::U8> data(*sizeResult);
	if (not data.empty() and ::fread(data.data(), 1U, data.size(), file) != data.size()) {
		const int error = ::ferror(file) != 0 and errno != 0 ? errno : EIO;
		(void)::fclose(file);
		return filesystemError(error, "cannot read file", inPath);
	}
	if (::fclose(file) != 0) {
		return filesystemError(errno != 0 ? errno : EIO, "cannot close file", inPath);
	}
	return data;
}

oa::Status oa::Filesystem::writeBinary(
	const oa::Path& inPath, oa::Span<const oa::U8> inData)
{
	const oa::Path parent = inPath.parentPath();
	if (not parent.empty()) OA_RETURN_IF_ERROR(createDirectories(parent));
	return writeBytes(inPath, inData.data(), inData.size(), "wb");
}

oa::Result<oa::Path> oa::Filesystem::absolute(const oa::Path& inPath) {
#if defined(_WIN32)
	const DWORD needed = ::GetFullPathNameA(inPath.cStr(), 0U, nullptr, nullptr);
	if (needed == 0U) {
		return filesystemError(static_cast<int>(::GetLastError()), "cannot resolve path", inPath);
	}
	oa::Vector<char> buffer(static_cast<oa::Usize>(needed));
	if (::GetFullPathNameA(inPath.cStr(), needed, buffer.data(), nullptr) == 0U) {
		return filesystemError(static_cast<int>(::GetLastError()), "cannot resolve path", inPath);
	}
	return oa::Path(oa::StringView(buffer.data(), needed - 1U));
#else
	if (inPath.isAbsolute()) return inPath.lexicallyNormal();
	oa::Usize capacity = 256U;
	for (;;) {
		oa::Vector<char> buffer(capacity);
		if (::getcwd(buffer.data(), buffer.size()) != nullptr) {
			return (oa::Path(buffer.data()) / inPath).lexicallyNormal();
		}
		if (errno != ERANGE) return filesystemError(errno, "cannot resolve path", inPath);
		if (capacity > oa::Limits<oa::Usize>::max() / 2U) {
			return oa::Status::error(oa::StatusCode::ResourceExhausted,
				"current directory path is too large");
		}
		capacity *= 2U;
	}
#endif
}

oa::Result<oa::Vector<oa::Path>> oa::Filesystem::glob(
	const oa::Path& inDir, oa::StringView inPattern)
{
	if (not isDirectory(inDir)) {
		return oa::Status::notFound("directory does not exist: " + inDir.string());
	}
	oa::Vector<oa::Path> matches;
	const oa::Status status = visitDirectory(inDir,
		[&](const oa::Path& inPath, bool, bool) {
			if (globMatch(inPattern, inPath.filename().string())) matches.pushBack(inPath);
			return oa::Status::ok();
		});
	if (status.isError()) return status;
	sortPaths(matches);
	return matches;
}
