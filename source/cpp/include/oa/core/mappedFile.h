#pragma once

#include <oa/core/filesystem.h>

namespace oa {

// Read-only whole-file mapping with RAII ownership.
//
// Linux uses mmap. Other platforms retain a read-only owned byte buffer until
// a native mapping implementation is added.
class MappedFile {
public:
	MappedFile() = default;
	~MappedFile();

	MappedFile(const MappedFile&) = delete;
	MappedFile& operator=(const MappedFile&) = delete;
	MappedFile(MappedFile&& inOther) noexcept;
	MappedFile& operator=(MappedFile&& inOther) noexcept;

	[[nodiscard]] oa::Status openReadOnly(const oa::Path& inPath);
	void close();

	[[nodiscard]] bool isOpen() const { return data_ != nullptr; }
	[[nodiscard]] const oa::Path& path() const { return path_; }
	[[nodiscard]] const oa::U8* data() const { return data_; }
	[[nodiscard]] oa::Usize size() const { return size_; }
	[[nodiscard]] oa::Span<const oa::U8> bytes() const { return {data_, size_}; }
	[[nodiscard]] oa::Result<oa::Span<const oa::U8>> slice(oa::U64 inOffset, oa::U64 inSize) const;

private:
	oa::Path path_;
	const oa::U8* data_ = nullptr;
	oa::Usize size_ = 0;

#ifdef OA_PLATFORM_LINUX
	int fd_ = -1;
#else
	oa::Vector<oa::U8> owned_;
#endif
};

} // namespace oa
