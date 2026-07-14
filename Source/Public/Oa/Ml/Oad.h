// OA Dataset Archive (.oad) — packed train / validation / test byte corpora
// v1: fixed 64-byte LE header + contiguous raw UTF-8 (or any) payload. Mmap-friendly.
// Model checkpoints use oam.h (.oam) — keep .oad (data) and .oam (weights) separate.

#pragma once

#include <Oa/Core/Status.h>
#include <Oa/Core/Types.h>

#include <cstddef>

static constexpr OaU32 OA_OAD_MAGIC = 0x3144414F; // 'O','A','D','1' in LE file order
static constexpr OaI32 OA_OAD_HEADER_V1_BYTES = 64;
static constexpr OaU8 OA_OAD_VERSION_MAJOR = 1;
static constexpr OaU8 OA_OAD_VERSION_MINOR = 0;

enum class OaOadSplit : OaU8 {
	Train = 0,
	Val = 1,
	Test = 2,
};

// On-disk v1 header (64 bytes, little-endian integers). Packed for exact wire size.
#pragma pack(push, 1)
class OaOadHeaderV1 {
public:
	OaU32 Magic = 0;
	OaU8 VersionMajor = 0;
	OaU8 VersionMinor = 0;
	OaU16 Flags = 0;
	OaU32 Reserved0 = 0;
	OaU64 TrainOffset = 0;
	OaU64 TrainBytes = 0;
	OaU64 ValOffset = 0;
	OaU64 ValBytes = 0;
	OaU64 TestOffset = 0;
	OaU64 TestBytes = 0;
	OaU32 Reserved1 = 0;
};
#pragma pack(pop)

static_assert(sizeof(OaOadHeaderV1) == 64, "OaOadHeaderV1 must be 64 bytes");

[[nodiscard]] bool OaOadProbeMagic(const OaU8* InData, OaI64 InSize);

[[nodiscard]] OaStatus OaOadParseHeader(const OaU8* InData, OaI64 InSize, OaOadHeaderV1& OutHdr);

[[nodiscard]] OaSpan<const OaU8> OaOadSplitSpan(
	const OaU8* InBase, OaI64 InFileSize, const OaOadHeaderV1& InHdr, OaOadSplit InSplit
);

// Write .oad v1: Train required; Val/Test may be empty. Payload is contiguous after header.
[[nodiscard]] OaStatus OaOadWriteFile(
	const OaPath& InPath,
	OaSpan<const OaU8> InTrain,
	OaSpan<const OaU8> InVal,
	OaSpan<const OaU8> InTest
);

// Memory-map (Linux) or load entire file; exposes train/val/test spans into the mapping.
class OaOadFile {
public:
	OaOadFile() = default;
	~OaOadFile();
	OaOadFile(const OaOadFile&) = delete;
	OaOadFile& operator=(const OaOadFile&) = delete;
	OaOadFile(OaOadFile&& InOther) noexcept;
	OaOadFile& operator=(OaOadFile&& InOther) noexcept;

	void Close();

	// Returns true if path is a valid .oad v1 file and mapping/buffer is ready.
	[[nodiscard]] bool TryOpen(const OaString& InPath);

	[[nodiscard]] bool IsOpen() const { return Valid_; }
	[[nodiscard]] const OaOadHeaderV1& Header() const { return Hdr_; }
	[[nodiscard]] OaSpan<const OaU8> TrainSpan() const;
	[[nodiscard]] OaSpan<const OaU8> ValSpan() const;
	[[nodiscard]] OaSpan<const OaU8> TestSpan() const;

private:
	bool Valid_ = false;
	OaOadHeaderV1 Hdr_{};
	const OaU8* Bytes_ = nullptr;
	OaI64 FileSize_ = 0;
#ifdef OA_PLATFORM_LINUX
	int Fd_ = -1;
	OaU8* MapAddr_ = nullptr;
	OaI64 MapSize_ = 0;
#else
	OaVec<OaU8> Owned_;
#endif
};
