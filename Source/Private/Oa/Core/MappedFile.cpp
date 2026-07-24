#include <Oa/Core/MappedFile.h>

#include <cerrno>
#include <cstring>
#include <limits>

#ifdef OA_PLATFORM_LINUX
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

OaMappedFile::~OaMappedFile() {
	Close();
}

OaMappedFile::OaMappedFile(OaMappedFile&& InOther) noexcept {
	*this = OaStdMove(InOther);
}

OaMappedFile& OaMappedFile::operator=(OaMappedFile&& InOther) noexcept {
	if (this == &InOther) return *this;
	Close();

	Path_ = OaStdMove(InOther.Path_);
	Data_ = InOther.Data_;
	Size_ = InOther.Size_;
#ifdef OA_PLATFORM_LINUX
	Fd_ = InOther.Fd_;
	InOther.Fd_ = -1;
#else
	Owned_ = OaStdMove(InOther.Owned_);
	Data_ = Owned_.Empty() ? nullptr : Owned_.Data();
#endif
	InOther.Data_ = nullptr;
	InOther.Size_ = 0;
	return *this;
}

OaStatus OaMappedFile::OpenReadOnly(const OaPath& InPath) {
	Close();
	Path_ = InPath;

#ifdef OA_PLATFORM_LINUX
	Fd_ = open(InPath.CStr(), O_RDONLY | O_CLOEXEC);
	if (Fd_ < 0) {
		return OaStatus::Error(OaStatusCode::FileNotFound,
			OaString("Cannot open mapped file: ") + InPath.CStr() + ": " + std::strerror(errno));
	}

	struct stat statBuf {};
	if (fstat(Fd_, &statBuf) != 0) {
		const OaString message = OaString("Cannot stat mapped file: ") + std::strerror(errno);
		Close();
		return OaStatus::Error(OaStatusCode::Internal, message);
	}
	if (statBuf.st_size <= 0) {
		Close();
		return OaStatus::Error(OaStatusCode::FileCorrupt, "Cannot map an empty file");
	}
	if (static_cast<OaU64>(statBuf.st_size) > std::numeric_limits<OaUsize>::max()) {
		Close();
		return OaStatus::Error(OaStatusCode::ResourceExhausted, "Mapped file is too large for this process");
	}

	Size_ = static_cast<OaUsize>(statBuf.st_size);
	void* mapped = mmap(nullptr, Size_, PROT_READ, MAP_PRIVATE, Fd_, 0);
	if (mapped == MAP_FAILED) {
		const OaString message = OaString("Cannot map file: ") + std::strerror(errno);
		Data_ = nullptr;
		Close();
		return OaStatus::Error(OaStatusCode::Internal, message);
	}
	Data_ = static_cast<const OaU8*>(mapped);
	(void)madvise(const_cast<OaU8*>(Data_), Size_, MADV_RANDOM);
#else
	auto read = OaFilesystem::ReadBinary(InPath);
	if (read.IsError()) return read.GetStatus();
	Owned_ = OaStdMove(read).GetValue();
	if (Owned_.Empty()) {
		return OaStatus::Error(OaStatusCode::FileCorrupt, "Cannot map an empty file");
	}
	Data_ = Owned_.Data();
	Size_ = Owned_.Size();
#endif
	return OaStatus::Ok();
}

void OaMappedFile::Close() {
#ifdef OA_PLATFORM_LINUX
	if (Data_ != nullptr) {
		(void)munmap(const_cast<OaU8*>(Data_), Size_);
	}
	if (Fd_ >= 0) {
		(void)close(Fd_);
		Fd_ = -1;
	}
#else
	Owned_.Clear();
#endif
	Data_ = nullptr;
	Size_ = 0;
	Path_ = {};
}

OaResult<OaSpan<const OaU8>> OaMappedFile::Slice(OaU64 InOffset, OaU64 InSize) const {
	if (!IsOpen()) {
		return OaStatus::Error(OaStatusCode::FailedPrecondition, "Mapped file is not open");
	}
	const OaU64 size = static_cast<OaU64>(Size_);
	if (InOffset > size || InSize > size - InOffset) {
		return OaStatus::Error(OaStatusCode::OutOfRange, "Mapped file slice is outside file bounds");
	}
	return OaSpan<const OaU8>(Data_ + static_cast<OaUsize>(InOffset), static_cast<OaUsize>(InSize));
}
