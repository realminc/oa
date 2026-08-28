// SDK .oad v1 dataset archive implementation.

#include <data/datasetArchive.h>
#include <oa/core/filesystem.h>
#include <oa/core/std/memory.h>
#include <cstring>
#include <fstream>

#ifdef OA_PLATFORM_LINUX
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace oa {

static bool validateDatasetArchiveHeader(const DatasetArchiveHeader& inHeader, oa::I64 inFileSize) {
	if (inHeader.magic != kDatasetArchiveMagic) return false;
	if (inHeader.versionMajor != kDatasetArchiveVersionMajor) return false;
	if (inFileSize < kDatasetArchiveHeaderBytes) return false;
	if (inHeader.trainOffset != static_cast<oa::U64>(kDatasetArchiveHeaderBytes)) return false;
	if (inHeader.trainBytes == 0) return false;
	const oa::I64 endTrain = static_cast<oa::I64>(inHeader.trainOffset + inHeader.trainBytes);
	if (endTrain > inFileSize || endTrain < 0) return false;
	if (inHeader.valBytes > 0) {
		if (inHeader.valOffset != inHeader.trainOffset + inHeader.trainBytes) return false;
		const oa::I64 endVal = static_cast<oa::I64>(inHeader.valOffset + inHeader.valBytes);
		if (endVal > inFileSize || endVal < 0) return false;
	} else {
		if (inHeader.valOffset != 0 && inHeader.valOffset != inHeader.trainOffset + inHeader.trainBytes) return false;
	}
	if (inHeader.testBytes > 0) {
		const oa::U64 expectOff =
			inHeader.valBytes > 0 ? (inHeader.valOffset + inHeader.valBytes) : (inHeader.trainOffset + inHeader.trainBytes);
		if (inHeader.testOffset != expectOff) return false;
		const oa::I64 endTest = static_cast<oa::I64>(inHeader.testOffset + inHeader.testBytes);
		if (endTest > inFileSize || endTest < 0) return false;
	} else {
		if (inHeader.testOffset != 0) {
			const oa::U64 expectOff =
				inHeader.valBytes > 0 ? (inHeader.valOffset + inHeader.valBytes) : (inHeader.trainOffset + inHeader.trainBytes);
			if (inHeader.testOffset != expectOff) return false;
		}
	}
	const oa::U64 payloadEnd = inHeader.testBytes > 0
		? (inHeader.testOffset + inHeader.testBytes)
		: (inHeader.valBytes > 0 ? (inHeader.valOffset + inHeader.valBytes)
							  : (inHeader.trainOffset + inHeader.trainBytes));
	if (payloadEnd != static_cast<oa::U64>(inFileSize)) return false;
	return true;
}

bool isDatasetArchive(const oa::U8* inData, oa::I64 inSize) {
	if (inSize < 4) return false;
	oa::U32 m = 0;
	oa::memcpy(&m, inData, sizeof(m));
	return m == kDatasetArchiveMagic;
}

oa::Status parseDatasetArchiveHeader(
	const oa::U8* inData,
	oa::I64 inSize,
	DatasetArchiveHeader& outHeader
) {
	if (inSize < kDatasetArchiveHeaderBytes) return oa::Status::invalidArgument("oad: file too small");
	oa::memcpy(&outHeader, inData, sizeof(outHeader));
	if (!validateDatasetArchiveHeader(outHeader, inSize)) return oa::Status::invalidArgument("oad: invalid header or layout");
	return oa::Status::ok();
}

oa::Span<const oa::U8> datasetArchiveSplit(
	const oa::U8* inBase, oa::I64 inFileSize, const DatasetArchiveHeader& inHeader, DatasetSplit inSplit
) {
	if (!validateDatasetArchiveHeader(inHeader, inFileSize)) return {};
	oa::U64 off = 0;
	oa::U64 len = 0;
	switch (inSplit) {
	case DatasetSplit::Train:
		off = inHeader.trainOffset;
		len = inHeader.trainBytes;
		break;
	case DatasetSplit::Val:
		off = inHeader.valOffset;
		len = inHeader.valBytes;
		break;
	case DatasetSplit::Test:
		off = inHeader.testOffset;
		len = inHeader.testBytes;
		break;
	}
	if (len == 0) return {};
	if (off + len > static_cast<oa::U64>(inFileSize)) return {};
	return oa::Span<const oa::U8>(inBase + off, static_cast<oa::Usize>(len));
}

oa::Status writeDatasetArchive(
	const oa::Path& inPath,
	oa::Span<const oa::U8> inTrain,
	oa::Span<const oa::U8> inVal,
	oa::Span<const oa::U8> inTest
) {
	if (inTrain.empty()) return oa::Status::invalidArgument("oad: train payload required");

	DatasetArchiveHeader hdr{};
	hdr.magic = kDatasetArchiveMagic;
	hdr.versionMajor = kDatasetArchiveVersionMajor;
	hdr.versionMinor = kDatasetArchiveVersionMinor;
	hdr.trainOffset = static_cast<oa::U64>(kDatasetArchiveHeaderBytes);
	hdr.trainBytes = static_cast<oa::U64>(inTrain.size());
	hdr.valOffset = hdr.trainOffset + hdr.trainBytes;
	hdr.valBytes = static_cast<oa::U64>(inVal.size());
	hdr.testOffset = hdr.valOffset + hdr.valBytes;
	hdr.testBytes = static_cast<oa::U64>(inTest.size());

	std::ofstream f(inPath.cStr(), std::ios::binary | std::ios::trunc);
	if (!f) return oa::Status::error(oa::StatusCode::Internal, "oad: cannot open output");
	f.write(reinterpret_cast<const char*>(&hdr), sizeof(hdr));
	if (!f) return oa::Status::error(oa::StatusCode::Internal, "oad: write header");
	f.write(reinterpret_cast<const char*>(inTrain.data()), static_cast<std::streamsize>(inTrain.size()));
	if (!f) return oa::Status::error(oa::StatusCode::Internal, "oad: write train");
	if (!inVal.empty()) {
		f.write(reinterpret_cast<const char*>(inVal.data()), static_cast<std::streamsize>(inVal.size()));
		if (!f) return oa::Status::error(oa::StatusCode::Internal, "oad: write val");
	}
	if (!inTest.empty()) {
		f.write(reinterpret_cast<const char*>(inTest.data()), static_cast<std::streamsize>(inTest.size()));
		if (!f) return oa::Status::error(oa::StatusCode::Internal, "oad: write test");
	}
	return oa::Status::ok();
}

DatasetArchive::~DatasetArchive() { close(); }

DatasetArchive::DatasetArchive(DatasetArchive&& inOther) noexcept {
	*this = std::move(inOther);
}

DatasetArchive& DatasetArchive::operator=(DatasetArchive&& inOther) noexcept {
	if (this == &inOther) return *this;
	close();
	valid_ = inOther.valid_;
	header_ = inOther.header_;
	bytes_ = inOther.bytes_;
	fileSize_ = inOther.fileSize_;
#ifdef OA_PLATFORM_LINUX
	fd_ = inOther.fd_;
	mapAddr_ = inOther.mapAddr_;
	mapSize_ = inOther.mapSize_;
	inOther.valid_ = false;
	inOther.bytes_ = nullptr;
	inOther.fileSize_ = 0;
	inOther.fd_ = -1;
	inOther.mapAddr_ = nullptr;
	inOther.mapSize_ = 0;
#else
	owned_ = std::move(inOther.owned_);
	inOther.valid_ = false;
	inOther.bytes_ = nullptr;
	inOther.fileSize_ = 0;
#endif
	return *this;
}

void DatasetArchive::close() {
#ifdef OA_PLATFORM_LINUX
	if (mapAddr_ != nullptr) {
		munmap(mapAddr_, static_cast<size_t>(mapSize_));
		mapAddr_ = nullptr;
	}
	if (fd_ >= 0) {
		::close(fd_);
		fd_ = -1;
	}
#else
	owned_.clear();
#endif
	valid_ = false;
	bytes_ = nullptr;
	fileSize_ = 0;
	header_ = {};
}

bool DatasetArchive::tryOpen(const oa::String& inPath) {
	close();
#ifdef OA_PLATFORM_LINUX
	fd_ = ::open(inPath.cStr(), O_RDONLY);
	if (fd_ < 0) return false;
	struct stat st;
	if (fstat(fd_, &st) != 0) {
		::close(fd_);
		fd_ = -1;
		return false;
	}
	fileSize_ = static_cast<oa::I64>(st.st_size);
	if (fileSize_ < kDatasetArchiveHeaderBytes) {
		::close(fd_);
		fd_ = -1;
		return false;
	}
	mapAddr_ = static_cast<oa::U8*>(mmap(nullptr, static_cast<size_t>(fileSize_), PROT_READ, MAP_PRIVATE, fd_, 0));
	if (mapAddr_ == MAP_FAILED) {
		mapAddr_ = nullptr;
		::close(fd_);
		fd_ = -1;
		return false;
	}
	mapSize_ = fileSize_;
	bytes_ = mapAddr_;
#else
	auto bin = oa::Filesystem::readBinary(oa::Path(inPath));
	if (!bin.isOk()) return false;
	owned_ = std::move(bin).getValue();
	fileSize_ = static_cast<oa::I64>(owned_.size());
	bytes_ = owned_.data();
#endif
	if (parseDatasetArchiveHeader(bytes_, fileSize_, header_).isError()) {
		close();
		return false;
	}
	valid_ = true;
	return true;
}

oa::Span<const oa::U8> DatasetArchive::trainSpan() const {
	if (!valid_) return {};
	return datasetArchiveSplit(bytes_, fileSize_, header_, DatasetSplit::Train);
}

oa::Span<const oa::U8> DatasetArchive::valSpan() const {
	if (!valid_) return {};
	return datasetArchiveSplit(bytes_, fileSize_, header_, DatasetSplit::Val);
}

oa::Span<const oa::U8> DatasetArchive::testSpan() const {
	if (!valid_) return {};
	return datasetArchiveSplit(bytes_, fileSize_, header_, DatasetSplit::Test);
}

} // namespace oa
