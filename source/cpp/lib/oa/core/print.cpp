#include <oa/core/std/print.h>

#include <stdio.h>

namespace {

class FileLock {
public:
	explicit FileLock(::FILE* inFile) noexcept : file_(inFile) {
#if defined(_WIN32)
		::_lock_file(file_);
#else
		::flockfile(file_);
#endif
	}

	~FileLock() {
#if defined(_WIN32)
		::_unlock_file(file_);
#else
		::funlockfile(file_);
#endif
	}

	FileLock(const FileLock&) = delete;
	FileLock& operator=(const FileLock&) = delete;

private:
	::FILE* file_;
};

[[nodiscard]] bool writeAll(::FILE* inFile, const char* inData, oa::Usize inSize) noexcept {
	oa::Usize written = 0;
	while (written < inSize) {
		const oa::Usize count = ::fwrite(inData + written, 1U, inSize - written, inFile);
		if (count == 0U) return false;
		written += count;
	}
	return true;
}

} // namespace

oa::Status oa::detail::writePrint(
	oa::PrintStream inStream,
	oa::StringView inText,
	bool inNewline
) noexcept {
	::FILE* file = nullptr;
	switch (inStream) {
		case oa::PrintStream::Out:
			file = stdout;
			break;
		case oa::PrintStream::Error:
			file = stderr;
			break;
		default:
			return oa::Status::invalidArgument("invalid OA print stream");
	}
	if (inText.size() != 0U and inText.data() == nullptr) {
		return oa::Status::invalidArgument("OA print text is null");
	}
	FileLock lock(file);
	if (not writeAll(file, inText.data(), inText.size())
		or (inNewline and not writeAll(file, "\n", 1U))) {
		return oa::Status::error(oa::StatusCode::Internal, "OA print write failed");
	}
	return oa::Status::ok();
}

oa::Status oa::detail::flushPrint(oa::PrintStream inStream) noexcept {
	::FILE* file = nullptr;
	switch (inStream) {
		case oa::PrintStream::Out:
			file = stdout;
			break;
		case oa::PrintStream::Error:
			file = stderr;
			break;
		default:
			return oa::Status::invalidArgument("invalid OA print stream");
	}
	FileLock lock(file);
	if (::fflush(file) != 0) {
		return oa::Status::error(oa::StatusCode::Internal, "OA print flush failed");
	}
	return oa::Status::ok();
}
