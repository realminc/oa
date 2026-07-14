// Dataset — Core data loading implementation
// OaByteDataset, OaMMapByteDataset, OaDataLoader, OaPinnedBuffer, OaByteDataPipeline

#include <Oa/Data/Dataset.h>
#include <Oa/Data/FnDataset.h>
#include <Oa/Core/FnMatrix.h>
#include <Oa/Core/Log.h>
#include <Oa/Core/Memory.h>
#include <fstream>
#include <filesystem>
#include <algorithm>
#include <random>
#include <numeric>
#include <cstdio>

#ifdef OA_PLATFORM_LINUX
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#endif

namespace fs = std::filesystem;

// OaByteDataset

OaByteDataset::OaByteDataset(const OaString& InPath, OaI64 InSeqLen)
	: SeqLen_(InSeqLen) {
	const fs::path FsRoot(InPath.CStr());
	if (fs::is_directory(FsRoot)) {
		for (const auto& entry : fs::recursive_directory_iterator(FsRoot)) {
			if (!entry.is_regular_file()) continue;
			std::ifstream file(entry.path(), std::ios::binary | std::ios::ate);
			if (!file.is_open()) continue;
			auto size = file.tellg();
			file.seekg(0);
			OaVec<OaU8> bytes(static_cast<OaUsize>(size));
			file.read(reinterpret_cast<char*>(bytes.Data()), size);
			TotalBytes_ += static_cast<OaI64>(size);
			Data_.PushBack(std::move(bytes));
		}
	} else {
		std::ifstream file(InPath.CStr(), std::ios::binary | std::ios::ate);
		if (file.is_open()) {
			auto size = file.tellg();
			file.seekg(0);
			OaVec<OaU8> bytes(static_cast<OaUsize>(size));
			file.read(reinterpret_cast<char*>(bytes.Data()), size);
			TotalBytes_ = static_cast<OaI64>(size);
			Data_.PushBack(std::move(bytes));
		}
	}

	NumSequences_ = 0;
	for (const auto& d : Data_) {
		if (static_cast<OaI64>(d.Size()) >= SeqLen_) {
			NumSequences_ += (static_cast<OaI64>(d.Size()) - SeqLen_) + 1;
		}
	}
}

OaByteDataset::OaByteDataset(const OaString& InPath, OaI64 InSeqLen, bool InSingleFile)
	: OaByteDataset(InPath, InSeqLen) {
	(void)InSingleFile;
}

OaMatrix OaByteDataset::GetItem(OaI64 InIndex) const {
	OaI64 remaining = InIndex;
	for (const auto& d : Data_) {
		OaI64 numSeqs = static_cast<OaI64>(d.Size()) - SeqLen_ + 1;
		if (numSeqs <= 0) continue;
		if (remaining < numSeqs) {
			OaSpan<const OaU8> slice(d.Data() + remaining, static_cast<OaUsize>(SeqLen_));
			return OaFnMatrix::FromBytes(slice, OaMatrixShape{SeqLen_});
		}
		remaining -= numSeqs;
	}
	return OaMatrix();
}

// OaDataLoader

OaDataLoader::OaDataLoader(OaDataset& InDataset, OaDataLoaderConfig InConfig)
	: Dataset_(InDataset), Config_(std::move(InConfig)) {
	if (Config_.BatchSize <= 0) {
		OA_LOG_ERROR(OaLogComponent::ML,
			"OaDataLoader: BatchSize must be positive; using 1");
		Config_.BatchSize = 1;
	}
	BuildIndices_();
}

void OaDataLoader::BuildIndices_() {
	Indices_.Clear();
	OaI64 total = std::max<OaI64>(0, Dataset_.Size());
	Indices_.Reserve(static_cast<OaUsize>(total));
	for (OaI64 i = 0; i < total; ++i) Indices_.PushBack(i);
	if (Config_.Shuffle) {
		OaU64 seed = Config_.Seed;
		if (seed == 0) seed = std::random_device{}();
		std::mt19937 rng(static_cast<OaU32>(seed));
		std::shuffle(Indices_.Begin(), Indices_.End(), rng);
	}
}

void OaDataLoader::ApplyTransforms_(OaDataset::Sample& InOutSample) const {
	for (const auto& t : Config_.Transforms) {
		if (t) t->Apply(InOutSample);
	}
}

OaOpt<OaDataLoader::Batch> OaDataLoader::NextBatch() {
	OaI64 start = CurrentBatch_ * Config_.BatchSize;
	if (start >= static_cast<OaI64>(Indices_.Size())) return std::nullopt;

	OaI64 end = std::min(start + Config_.BatchSize, static_cast<OaI64>(Indices_.Size()));
	OaI64 actualBatch = end - start;
	if (Config_.DropLast && actualBatch < Config_.BatchSize) {
		CurrentBatch_++;
		return std::nullopt;
	}

	OaVec<OaDataset::Sample> samples;
	samples.Reserve(static_cast<OaUsize>(actualBatch));
	for (OaI64 i = 0; i < actualBatch; ++i) {
		auto sample = Dataset_.GetSample(Indices_[static_cast<OaUsize>(start + i)]);
		ApplyTransforms_(sample);
		samples.PushBack(std::move(sample));
	}

	CurrentBatch_++;
	auto collated = OaFnDataset::Collate(
		OaSpan<const OaDataset::Sample>(samples.Data(), samples.Size()));
	if (collated.X.IsEmpty()) return std::nullopt;
	return Batch{std::move(collated.X), std::move(collated.Y)};
}

bool OaDataLoader::NextBatch(OaMatrix& OutX, OaMatrix& OutY) {
	auto batch = NextBatch();
	if (!batch) return false;
	OutX = std::move(batch->X);
	OutY = std::move(batch->Y);
	return true;
}

OaOpt<OaMatrix> OaDataLoader::Next() {
	auto batch = NextBatch();
	if (!batch) return std::nullopt;
	return std::move(batch->X);
}

void OaDataLoader::Reset() {
	CurrentBatch_ = 0;
	BuildIndices_();
}

OaI64 OaDataLoader::NumBatches() const {
	OaI64 total = static_cast<OaI64>(Indices_.Size());
	OaI64 batches = total / Config_.BatchSize;
	if (!Config_.DropLast && (total % Config_.BatchSize) > 0) batches++;
	return batches;
}

// Pipeline

OaByteDataPipeline OaByteDataPipeline::Create(const OaString& InPath, OaI64 InSeqLen, OaI32 InBatchSize) {
	OaByteDataPipeline pipe;
	pipe.Dataset = OaMakeUniquePtr<OaByteDataset>(InPath, InSeqLen);
	OaDataLoaderConfig cfg;
	cfg.BatchSize = InBatchSize;
	cfg.Shuffle = true;
	pipe.Loader = OaMakeUniquePtr<OaDataLoader>(*pipe.Dataset, cfg);
	return pipe;
}

// OaMMapByteDataset

#ifdef OA_PLATFORM_LINUX

OaMMapByteDataset::OaMMapByteDataset(const OaString& InPath, OaI64 InSeqLen)
	: SeqLen_(InSeqLen) {
	Fd_ = open(InPath.c_str(), O_RDONLY);
	if (Fd_ < 0) {
		fprintf(stderr, "[OaMMapByteDataset] Failed to open: %s\n", InPath.c_str());
		return;
	}

	struct stat st;
	if (fstat(Fd_, &st) != 0) {
		fprintf(stderr, "[OaMMapByteDataset] Failed to stat: %s\n", InPath.c_str());
		close(Fd_);
		Fd_ = -1;
		return;
	}

	FileSize_ = static_cast<OaI64>(st.st_size);
	if (FileSize_ < SeqLen_) {
		fprintf(stderr, "[OaMMapByteDataset] File too small (%lld bytes) for seq_len %lld\n",
			static_cast<long long>(FileSize_), static_cast<long long>(SeqLen_));
		close(Fd_);
		Fd_ = -1;
		return;
	}

	MappedData_ = static_cast<OaU8*>(mmap(nullptr, static_cast<size_t>(FileSize_),
		PROT_READ, MAP_PRIVATE | MAP_NORESERVE, Fd_, 0));

	if (MappedData_ == MAP_FAILED) {
		fprintf(stderr, "[OaMMapByteDataset] mmap failed for: %s\n", InPath.c_str());
		MappedData_ = nullptr;
		close(Fd_);
		Fd_ = -1;
		return;
	}

	madvise(MappedData_, static_cast<size_t>(FileSize_), MADV_RANDOM);
	NumSequences_ = FileSize_ - SeqLen_ + 1;
}

OaMMapByteDataset::~OaMMapByteDataset() {
	if (MappedData_ != nullptr) {
		munmap(MappedData_, static_cast<size_t>(FileSize_));
	}
	if (Fd_ >= 0) {
		close(Fd_);
	}
}

OaMMapByteDataset::OaMMapByteDataset(OaMMapByteDataset&& InOther) noexcept
	: MappedData_(InOther.MappedData_), FileSize_(InOther.FileSize_),
	  SeqLen_(InOther.SeqLen_), NumSequences_(InOther.NumSequences_), Fd_(InOther.Fd_) {
	InOther.MappedData_ = nullptr;
	InOther.Fd_ = -1;
	InOther.FileSize_ = 0;
	InOther.NumSequences_ = 0;
}

OaMMapByteDataset& OaMMapByteDataset::operator=(OaMMapByteDataset&& InOther) noexcept {
	if (this != &InOther) {
		if (MappedData_ != nullptr) munmap(MappedData_, static_cast<size_t>(FileSize_));
		if (Fd_ >= 0) close(Fd_);
		MappedData_ = InOther.MappedData_;
		FileSize_ = InOther.FileSize_;
		SeqLen_ = InOther.SeqLen_;
		NumSequences_ = InOther.NumSequences_;
		Fd_ = InOther.Fd_;
		InOther.MappedData_ = nullptr;
		InOther.Fd_ = -1;
		InOther.FileSize_ = 0;
		InOther.NumSequences_ = 0;
	}
	return *this;
}

OaMatrix OaMMapByteDataset::GetItem(OaI64 InIndex) const {
	if (MappedData_ == nullptr || InIndex < 0 || InIndex >= NumSequences_) {
		return OaMatrix();
	}
	OaSpan<const OaU8> slice(MappedData_ + InIndex, static_cast<OaUsize>(SeqLen_));
	return OaFnMatrix::FromBytes(slice, OaMatrixShape{SeqLen_});
}

#else
OaMMapByteDataset::OaMMapByteDataset(const OaString& InPath, OaI64 InSeqLen) : SeqLen_(InSeqLen) {
	fprintf(stderr, "[OaMMapByteDataset] mmap not available on this platform, use OaByteDataset\n");
}
OaMMapByteDataset::~OaMMapByteDataset() {}
OaMMapByteDataset::OaMMapByteDataset(OaMMapByteDataset&&) noexcept = default;
OaMMapByteDataset& OaMMapByteDataset::operator=(OaMMapByteDataset&&) noexcept = default;
OaMatrix OaMMapByteDataset::GetItem(OaI64) const { return OaMatrix(); }
#endif

// Pinned memory

#ifndef OA_USE_CUDA

void* OaAllocPinned(OaUsize InBytes) {
	if (InBytes == 0) return nullptr;
	return std::malloc(InBytes);
}

void OaFreePinned(void* InPtr) {
	std::free(InPtr);
}

#endif // !OA_USE_CUDA

OaPinnedBuffer::OaPinnedBuffer(OaUsize InBytes) : Size_(InBytes) {
	Data_ = OaAllocPinned(InBytes);
}

OaPinnedBuffer::~OaPinnedBuffer() {
	OaFreePinned(Data_);
}

OaPinnedBuffer::OaPinnedBuffer(OaPinnedBuffer&& InOther) noexcept
	: Data_(InOther.Data_), Size_(InOther.Size_) {
	InOther.Data_ = nullptr;
	InOther.Size_ = 0;
}

OaPinnedBuffer& OaPinnedBuffer::operator=(OaPinnedBuffer&& InOther) noexcept {
	if (this != &InOther) {
		OaFreePinned(Data_);
		Data_ = InOther.Data_;
		Size_ = InOther.Size_;
		InOther.Data_ = nullptr;
		InOther.Size_ = 0;
	}
	return *this;
}
