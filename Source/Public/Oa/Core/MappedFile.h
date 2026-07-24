#pragma once

#include <Oa/Core/Filesystem.h>

// Read-only whole-file mapping with RAII ownership.
//
// Linux uses mmap. Other platforms retain a read-only owned byte buffer until
// a native mapping implementation is added.
class OaMappedFile {
public:
	OaMappedFile() = default;
	~OaMappedFile();

	OaMappedFile(const OaMappedFile&) = delete;
	OaMappedFile& operator=(const OaMappedFile&) = delete;
	OaMappedFile(OaMappedFile&& InOther) noexcept;
	OaMappedFile& operator=(OaMappedFile&& InOther) noexcept;

	[[nodiscard]] OaStatus OpenReadOnly(const OaPath& InPath);
	void Close();

	[[nodiscard]] bool IsOpen() const { return Data_ != nullptr; }
	[[nodiscard]] const OaPath& Path() const { return Path_; }
	[[nodiscard]] const OaU8* Data() const { return Data_; }
	[[nodiscard]] OaUsize Size() const { return Size_; }
	[[nodiscard]] OaSpan<const OaU8> Bytes() const { return {Data_, Size_}; }
	[[nodiscard]] OaResult<OaSpan<const OaU8>> Slice(OaU64 InOffset, OaU64 InSize) const;

private:
	OaPath Path_;
	const OaU8* Data_ = nullptr;
	OaUsize Size_ = 0;

#ifdef OA_PLATFORM_LINUX
	int Fd_ = -1;
#else
	OaVec<OaU8> Owned_;
#endif
};
