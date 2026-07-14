// .oad v1 — packed byte corpus (train / val / test)

#include <Oa/Ml/Oad.h>
#include <Oa/Core/FileIo.h>
#include <Oa/Core/Memory.h>
#include <cstring>
#include <fstream>

#ifdef OA_PLATFORM_LINUX
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

static bool OaOadValidateHeader(const OaOadHeaderV1& InHdr, OaI64 InFileSize) {
	if (InHdr.Magic != OA_OAD_MAGIC) return false;
	if (InHdr.VersionMajor != OA_OAD_VERSION_MAJOR) return false;
	if (InFileSize < OA_OAD_HEADER_V1_BYTES) return false;
	if (InHdr.TrainOffset != static_cast<OaU64>(OA_OAD_HEADER_V1_BYTES)) return false;
	if (InHdr.TrainBytes == 0) return false;
	const OaI64 endTrain = static_cast<OaI64>(InHdr.TrainOffset + InHdr.TrainBytes);
	if (endTrain > InFileSize || endTrain < 0) return false;
	if (InHdr.ValBytes > 0) {
		if (InHdr.ValOffset != InHdr.TrainOffset + InHdr.TrainBytes) return false;
		const OaI64 endVal = static_cast<OaI64>(InHdr.ValOffset + InHdr.ValBytes);
		if (endVal > InFileSize || endVal < 0) return false;
	} else {
		if (InHdr.ValOffset != 0 && InHdr.ValOffset != InHdr.TrainOffset + InHdr.TrainBytes) return false;
	}
	if (InHdr.TestBytes > 0) {
		const OaU64 expectOff =
			InHdr.ValBytes > 0 ? (InHdr.ValOffset + InHdr.ValBytes) : (InHdr.TrainOffset + InHdr.TrainBytes);
		if (InHdr.TestOffset != expectOff) return false;
		const OaI64 endTest = static_cast<OaI64>(InHdr.TestOffset + InHdr.TestBytes);
		if (endTest > InFileSize || endTest < 0) return false;
	} else {
		if (InHdr.TestOffset != 0) {
			const OaU64 expectOff =
				InHdr.ValBytes > 0 ? (InHdr.ValOffset + InHdr.ValBytes) : (InHdr.TrainOffset + InHdr.TrainBytes);
			if (InHdr.TestOffset != expectOff) return false;
		}
	}
	const OaU64 payloadEnd = InHdr.TestBytes > 0
		? (InHdr.TestOffset + InHdr.TestBytes)
		: (InHdr.ValBytes > 0 ? (InHdr.ValOffset + InHdr.ValBytes)
							  : (InHdr.TrainOffset + InHdr.TrainBytes));
	if (payloadEnd != static_cast<OaU64>(InFileSize)) return false;
	return true;
}

bool OaOadProbeMagic(const OaU8* InData, OaI64 InSize) {
	if (InSize < 4) return false;
	OaU32 m = 0;
	OaMemcpy(&m, InData, sizeof(m));
	return m == OA_OAD_MAGIC;
}

OaStatus OaOadParseHeader(const OaU8* InData, OaI64 InSize, OaOadHeaderV1& OutHdr) {
	if (InSize < OA_OAD_HEADER_V1_BYTES) return OaStatus::InvalidArgument("oad: file too small");
	OaMemcpy(&OutHdr, InData, sizeof(OutHdr));
	if (!OaOadValidateHeader(OutHdr, InSize)) return OaStatus::InvalidArgument("oad: invalid header or layout");
	return OaStatus::Ok();
}

OaSpan<const OaU8> OaOadSplitSpan(
	const OaU8* InBase, OaI64 InFileSize, const OaOadHeaderV1& InHdr, OaOadSplit InSplit
) {
	if (!OaOadValidateHeader(InHdr, InFileSize)) return {};
	OaU64 off = 0;
	OaU64 len = 0;
	switch (InSplit) {
	case OaOadSplit::Train:
		off = InHdr.TrainOffset;
		len = InHdr.TrainBytes;
		break;
	case OaOadSplit::Val:
		off = InHdr.ValOffset;
		len = InHdr.ValBytes;
		break;
	case OaOadSplit::Test:
		off = InHdr.TestOffset;
		len = InHdr.TestBytes;
		break;
	}
	if (len == 0) return {};
	if (off + len > static_cast<OaU64>(InFileSize)) return {};
	return OaSpan<const OaU8>(InBase + off, static_cast<OaUsize>(len));
}

OaStatus OaOadWriteFile(
	const OaPath& InPath,
	OaSpan<const OaU8> InTrain,
	OaSpan<const OaU8> InVal,
	OaSpan<const OaU8> InTest
) {
	if (InTrain.empty()) return OaStatus::InvalidArgument("oad: train payload required");

	OaOadHeaderV1 hdr{};
	hdr.Magic = OA_OAD_MAGIC;
	hdr.VersionMajor = OA_OAD_VERSION_MAJOR;
	hdr.VersionMinor = OA_OAD_VERSION_MINOR;
	hdr.TrainOffset = static_cast<OaU64>(OA_OAD_HEADER_V1_BYTES);
	hdr.TrainBytes = static_cast<OaU64>(InTrain.size());
	hdr.ValOffset = hdr.TrainOffset + hdr.TrainBytes;
	hdr.ValBytes = static_cast<OaU64>(InVal.size());
	hdr.TestOffset = hdr.ValOffset + hdr.ValBytes;
	hdr.TestBytes = static_cast<OaU64>(InTest.size());

	std::ofstream f(InPath.StdPath(), std::ios::binary | std::ios::trunc);
	if (!f) return OaStatus::Error(OaStatusCode::Internal, "oad: cannot open output");
	f.write(reinterpret_cast<const char*>(&hdr), sizeof(hdr));
	if (!f) return OaStatus::Error(OaStatusCode::Internal, "oad: write header");
	f.write(reinterpret_cast<const char*>(InTrain.data()), static_cast<std::streamsize>(InTrain.size()));
	if (!f) return OaStatus::Error(OaStatusCode::Internal, "oad: write train");
	if (!InVal.empty()) {
		f.write(reinterpret_cast<const char*>(InVal.data()), static_cast<std::streamsize>(InVal.size()));
		if (!f) return OaStatus::Error(OaStatusCode::Internal, "oad: write val");
	}
	if (!InTest.empty()) {
		f.write(reinterpret_cast<const char*>(InTest.data()), static_cast<std::streamsize>(InTest.size()));
		if (!f) return OaStatus::Error(OaStatusCode::Internal, "oad: write test");
	}
	return OaStatus::Ok();
}

OaOadFile::~OaOadFile() { Close(); }

OaOadFile::OaOadFile(OaOadFile&& InOther) noexcept {
	*this = std::move(InOther);
}

OaOadFile& OaOadFile::operator=(OaOadFile&& InOther) noexcept {
	if (this == &InOther) return *this;
	Close();
	Valid_ = InOther.Valid_;
	Hdr_ = InOther.Hdr_;
	Bytes_ = InOther.Bytes_;
	FileSize_ = InOther.FileSize_;
#ifdef OA_PLATFORM_LINUX
	Fd_ = InOther.Fd_;
	MapAddr_ = InOther.MapAddr_;
	MapSize_ = InOther.MapSize_;
	InOther.Valid_ = false;
	InOther.Bytes_ = nullptr;
	InOther.FileSize_ = 0;
	InOther.Fd_ = -1;
	InOther.MapAddr_ = nullptr;
	InOther.MapSize_ = 0;
#else
	Owned_ = std::move(InOther.Owned_);
	InOther.Valid_ = false;
	InOther.Bytes_ = nullptr;
	InOther.FileSize_ = 0;
#endif
	return *this;
}

void OaOadFile::Close() {
#ifdef OA_PLATFORM_LINUX
	if (MapAddr_ != nullptr) {
		munmap(MapAddr_, static_cast<size_t>(MapSize_));
		MapAddr_ = nullptr;
	}
	if (Fd_ >= 0) {
		close(Fd_);
		Fd_ = -1;
	}
#else
	Owned_.clear();
#endif
	Valid_ = false;
	Bytes_ = nullptr;
	FileSize_ = 0;
	Hdr_ = {};
}

bool OaOadFile::TryOpen(const OaString& InPath) {
	Close();
#ifdef OA_PLATFORM_LINUX
	Fd_ = open(InPath.c_str(), O_RDONLY);
	if (Fd_ < 0) return false;
	struct stat st;
	if (fstat(Fd_, &st) != 0) {
		close(Fd_);
		Fd_ = -1;
		return false;
	}
	FileSize_ = static_cast<OaI64>(st.st_size);
	if (FileSize_ < OA_OAD_HEADER_V1_BYTES) {
		close(Fd_);
		Fd_ = -1;
		return false;
	}
	MapAddr_ = static_cast<OaU8*>(mmap(nullptr, static_cast<size_t>(FileSize_), PROT_READ, MAP_PRIVATE, Fd_, 0));
	if (MapAddr_ == MAP_FAILED) {
		MapAddr_ = nullptr;
		close(Fd_);
		Fd_ = -1;
		return false;
	}
	MapSize_ = FileSize_;
	Bytes_ = MapAddr_;
#else
	auto bin = OaFileIo::ReadBinary(OaPath(InPath));
	if (!bin.IsOk()) return false;
	Owned_ = std::move(bin).GetValue();
	FileSize_ = static_cast<OaI64>(Owned_.size());
	Bytes_ = Owned_.data();
#endif
	if (OaOadParseHeader(Bytes_, FileSize_, Hdr_).IsError()) {
		Close();
		return false;
	}
	Valid_ = true;
	return true;
}

OaSpan<const OaU8> OaOadFile::TrainSpan() const {
	if (!Valid_) return {};
	return OaOadSplitSpan(Bytes_, FileSize_, Hdr_, OaOadSplit::Train);
}

OaSpan<const OaU8> OaOadFile::ValSpan() const {
	if (!Valid_) return {};
	return OaOadSplitSpan(Bytes_, FileSize_, Hdr_, OaOadSplit::Val);
}

OaSpan<const OaU8> OaOadFile::TestSpan() const {
	if (!Valid_) return {};
	return OaOadSplitSpan(Bytes_, FileSize_, Hdr_, OaOadSplit::Test);
}
