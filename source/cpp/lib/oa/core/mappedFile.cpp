#include <oa/core/mappedFile.h>

#include <cerrno>
#include <cstring>
#include <limits>

#ifdef OA_PLATFORM_LINUX
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

oa::MappedFile::~MappedFile() {
	close();
}

oa::MappedFile::MappedFile(oa::MappedFile&& inOther) noexcept {
	*this = oa::move(inOther);
}

oa::MappedFile& oa::MappedFile::operator=(oa::MappedFile&& inOther) noexcept {
	if (this == &inOther) return *this;
	close();

	path_ = oa::move(inOther.path_);
	data_ = inOther.data_;
	size_ = inOther.size_;
#ifdef OA_PLATFORM_LINUX
	fd_ = inOther.fd_;
	inOther.fd_ = -1;
#else
	owned_ = oa::move(inOther.owned_);
	data_ = owned_.empty() ? nullptr : owned_.data();
#endif
	inOther.data_ = nullptr;
	inOther.size_ = 0;
	return *this;
}

oa::Status oa::MappedFile::openReadOnly(const oa::Path& inPath) {
	close();
	path_ = inPath;

#ifdef OA_PLATFORM_LINUX
	fd_ = open(inPath.cStr(), O_RDONLY | O_CLOEXEC);
	if (fd_ < 0) {
		return oa::Status::error(oa::StatusCode::FileNotFound,
			oa::String("Cannot open mapped file: ") + inPath.cStr() + ": " + std::strerror(errno));
	}

	struct stat statBuf {};
	if (fstat(fd_, &statBuf) != 0) {
		const oa::String message = oa::String("Cannot stat mapped file: ") + std::strerror(errno);
		close();
		return oa::Status::error(oa::StatusCode::Internal, message);
	}
	if (statBuf.st_size <= 0) {
		close();
		return oa::Status::error(oa::StatusCode::FileCorrupt, "Cannot map an empty file");
	}
	if (static_cast<oa::U64>(statBuf.st_size) > std::numeric_limits<oa::Usize>::max()) {
		close();
		return oa::Status::error(oa::StatusCode::ResourceExhausted, "mapped file is too large for this process");
	}

	size_ = static_cast<oa::Usize>(statBuf.st_size);
	void* mapped = mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0);
	if (mapped == MAP_FAILED) {
		const oa::String message = oa::String("Cannot map file: ") + std::strerror(errno);
		data_ = nullptr;
		close();
		return oa::Status::error(oa::StatusCode::Internal, message);
	}
	data_ = static_cast<const oa::U8*>(mapped);
	(void)madvise(const_cast<oa::U8*>(data_), size_, MADV_RANDOM);
#else
	auto read = oa::Filesystem::readBinary(inPath);
	if (read.isError()) return read.getStatus();
	owned_ = oa::move(read).getValue();
	if (owned_.empty()) {
		return oa::Status::error(oa::StatusCode::FileCorrupt, "Cannot map an empty file");
	}
	data_ = owned_.data();
	size_ = owned_.size();
#endif
	return oa::Status::ok();
}

void oa::MappedFile::close() {
#ifdef OA_PLATFORM_LINUX
	if (data_ != nullptr) {
		(void)munmap(const_cast<oa::U8*>(data_), size_);
	}
	if (fd_ >= 0) {
		(void)::close(fd_);
		fd_ = -1;
	}
#else
	owned_.clear();
#endif
	data_ = nullptr;
	size_ = 0;
	path_ = {};
}

oa::Result<oa::Span<const oa::U8>> oa::MappedFile::slice(oa::U64 inOffset, oa::U64 inSize) const {
	if (!isOpen()) {
		return oa::Status::error(oa::StatusCode::FailedPrecondition, "mapped file is not open");
	}
	const oa::U64 size = static_cast<oa::U64>(size_);
	if (inOffset > size || inSize > size - inOffset) {
		return oa::Status::error(oa::StatusCode::OutOfRange, "mapped file slice is outside file bounds");
	}
	return oa::Span<const oa::U8>(data_ + static_cast<oa::Usize>(inOffset), static_cast<oa::Usize>(inSize));
}
