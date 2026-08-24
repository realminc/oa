// SDK dataset archive (.oad) — packed train/validation/test byte corpora.
// This format belongs to datasetctl and SDK data workflows, not the installed
// ML runtime. The v1 wire layout remains fixed and mmap-friendly.

#pragma once

#include <oa/core/status.h>
#include <oa/core/types.h>

#include <cstddef>

namespace oa {

inline constexpr oa::U32 kDatasetArchiveMagic = 0x3144414F;
inline constexpr oa::I32 kDatasetArchiveHeaderBytes = 64;
inline constexpr oa::U8 kDatasetArchiveVersionMajor = 1;
inline constexpr oa::U8 kDatasetArchiveVersionMinor = 0;

enum class DatasetSplit : oa::U8 {
	Train = 0,
	Val = 1,
	Test = 2,
};

// On-disk v1 header (64 bytes, little-endian integers). Packed for exact wire size.
#pragma pack(push, 1)
class DatasetArchiveHeader {
public:
	oa::U32 magic = 0;
	oa::U8 versionMajor = 0;
	oa::U8 versionMinor = 0;
	oa::U16 flags = 0;
	oa::U32 reserved0 = 0;
	oa::U64 trainOffset = 0;
	oa::U64 trainBytes = 0;
	oa::U64 valOffset = 0;
	oa::U64 valBytes = 0;
	oa::U64 testOffset = 0;
	oa::U64 testBytes = 0;
	oa::U32 reserved1 = 0;
};
#pragma pack(pop)

static_assert(sizeof(DatasetArchiveHeader) == 64);

[[nodiscard]] bool isDatasetArchive(const oa::U8* inData, oa::I64 inSize);

[[nodiscard]] oa::Status parseDatasetArchiveHeader(
	const oa::U8* inData,
	oa::I64 inSize,
	DatasetArchiveHeader& outHeader
);

[[nodiscard]] oa::Span<const oa::U8> datasetArchiveSplit(
	const oa::U8* inBase,
	oa::I64 inFileSize,
	const DatasetArchiveHeader& inHeader,
	DatasetSplit inSplit
);

// Write .oad v1: Train required; Val/Test may be empty. Payload is contiguous after header.
[[nodiscard]] oa::Status writeDatasetArchive(
	const oa::Path& inPath,
	oa::Span<const oa::U8> inTrain,
	oa::Span<const oa::U8> inVal,
	oa::Span<const oa::U8> inTest
);

// memory-map (Linux) or load entire file; exposes train/val/test spans into the mapping.
class DatasetArchive {
public:
	DatasetArchive() = default;
	~DatasetArchive();
	DatasetArchive(const DatasetArchive&) = delete;
	DatasetArchive& operator=(const DatasetArchive&) = delete;
	DatasetArchive(DatasetArchive&& inOther) noexcept;
	DatasetArchive& operator=(DatasetArchive&& inOther) noexcept;

	void close();

	// Returns true if path is a valid .oad v1 file and mapping/buffer is ready.
	[[nodiscard]] bool tryOpen(const oa::String& inPath);

	[[nodiscard]] bool isOpen() const { return valid_; }
	[[nodiscard]] const DatasetArchiveHeader& header() const { return header_; }
	[[nodiscard]] oa::Span<const oa::U8> trainSpan() const;
	[[nodiscard]] oa::Span<const oa::U8> valSpan() const;
	[[nodiscard]] oa::Span<const oa::U8> testSpan() const;

private:
	bool valid_ = false;
	DatasetArchiveHeader header_{};
	const oa::U8* bytes_ = nullptr;
	oa::I64 fileSize_ = 0;
#ifdef OA_PLATFORM_LINUX
	int fd_ = -1;
	oa::U8* mapAddr_ = nullptr;
	oa::I64 mapSize_ = 0;
#else
	oa::Vec<oa::U8> owned_;
#endif
};

} // namespace oa
