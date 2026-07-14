// OaDsMnist — Fashion-MNIST / MNIST Dataset Implementation

#include <Oa/Data/DsMnist.h>
#include <Oa/Core/Log.h>
#include <fstream>
#include <algorithm>

// ============================================================================
// OaDsMnist Implementation
// ============================================================================

OaDsMnist::OaDsMnist(const OaString& InDataDir, const OaString& InSplit, OaI32 InBatchSize, bool InShuffle)
	: DataDir_(InDataDir)
	, Split_(InSplit)
	, BatchSize_(InBatchSize)
	, Shuffle_(InShuffle)
	, Rng_(42)
{
	if (BatchSize_ <= 0) {
		OA_LOG_ERROR(OaLogComponent::ML, "MNIST batch size must be positive");
		return;
	}
	if (!LoadDataset()) {
		OA_LOG_ERROR(OaLogComponent::ML, "Failed to load MNIST dataset from: %s (split: %s)", InDataDir.c_str(), InSplit.c_str());
		return;
	}

	// Initialize indices for shuffling
	Indices_.Resize(NumSamples_);
	for (OaI32 i = 0; i < NumSamples_; ++i) Indices_[i] = i;

	if (Shuffle_) {
		std::shuffle(Indices_.begin(), Indices_.end(), Rng_);
	}

	// Pre-allocate batch buffers
	ImgBuf_.Resize(static_cast<OaI64>(BatchSize_) * 784);
	LblBuf_.Resize(BatchSize_);
}

bool OaDsMnist::LoadDataset() {
	// Construct file paths using the split name
	OaString imgPath = DataDir_ + "/" + Split_ + "-images-idx3-ubyte";
	OaString lblPath = DataDir_ + "/" + Split_ + "-labels-idx1-ubyte";

	std::ifstream imgF(imgPath.c_str(), std::ios::binary);
	std::ifstream lblF(lblPath.c_str(), std::ios::binary);

	if (!imgF || !lblF) {
		OA_LOG_ERROR(OaLogComponent::ML, "MNIST files not found: %s, %s", imgPath.c_str(), lblPath.c_str());
		return false;
	}

	// Verify magic numbers
	if (ReadBE32(imgF) != 0x00000803u) {
		OA_LOG_ERROR(OaLogComponent::ML, "Invalid MNIST image file magic number");
		return false;
	}
	if (ReadBE32(lblF) != 0x00000801u) {
		OA_LOG_ERROR(OaLogComponent::ML, "Invalid MNIST label file magic number");
		return false;
	}

	// Read dimensions
	OaU32 n = ReadBE32(imgF);
	OaU32 rows = ReadBE32(imgF);
	OaU32 cols = ReadBE32(imgF);
	OaU32 nLabels = ReadBE32(lblF);

	if (rows != 28 || cols != 28) {
		OA_LOG_ERROR(OaLogComponent::ML, "Invalid MNIST image dimensions: %ux%u", rows, cols);
		return false;
	}
	if (n != nLabels) {
		OA_LOG_ERROR(OaLogComponent::ML, "MNIST image/label count mismatch: %u vs %u", n, nLabels);
		return false;
	}

	NumSamples_ = static_cast<OaI32>(n);
	NumBatches_ = (NumSamples_ + BatchSize_ - 1) / BatchSize_;

	// Load images and labels
	Images_.Resize(static_cast<OaI64>(n) * 784);
	imgF.read(reinterpret_cast<char*>(Images_.Data()), Images_.Size());

	Labels_.Resize(static_cast<OaI64>(n));
	lblF.read(reinterpret_cast<char*>(Labels_.Data()), Labels_.Size());

	if (!imgF || !lblF) {
		OA_LOG_ERROR(OaLogComponent::ML, "Failed to read MNIST data");
		return false;
	}

	OA_LOG_INFO(OaLogComponent::ML, "Loaded MNIST: %d samples, %d batches (batch=%d)",
		NumSamples_, NumBatches_, BatchSize_);
	return true;
}

OaU32 OaDsMnist::ReadBE32(std::ifstream& InF) {
	OaU8 b[4];
	InF.read(reinterpret_cast<char*>(b), 4);
	return (OaU32(b[0]) << 24) | (OaU32(b[1]) << 16) | (OaU32(b[2]) << 8) | OaU32(b[3]);
}

OaMatrix OaDsMnist::GetItem(OaI64 InIndex) const {
	if (InIndex < 0 || InIndex >= NumSamples_) return {};
	OaSpan<const OaU8> imgSlice(Images_.Data() + InIndex * 784, 784);
	return OaFnMatrix::FromBytes(imgSlice, OaMatrixShape{784}, OaScalarType::UInt8);
}

OaDataset::Sample OaDsMnist::GetSample(OaI64 InIndex) const {
	if (InIndex < 0 || InIndex >= NumSamples_) return {};
	OaSpan<const OaU8> imgSlice(Images_.Data() + InIndex * 784, 784);
	OaSpan<const OaU8> lblSlice(Labels_.Data() + InIndex, 1);
	return Sample(
		OaFnMatrix::FromBytes(imgSlice, OaMatrixShape{784}, OaScalarType::UInt8),
		OaFnMatrix::FromBytes(lblSlice, OaMatrixShape{1}, OaScalarType::UInt8)
	);
}

bool OaDsMnist::NextBatch(OaMatrix& OutX, OaMatrix& OutY) {
	if (Cursor_ + BatchSize_ > NumSamples_) {
		return false;  // Epoch complete
	}

	// Copy batch from shuffled indices
	for (OaI32 i = 0; i < BatchSize_; ++i) {
		OaI32 idx = Indices_[Cursor_ + i];
		std::memcpy(ImgBuf_.Data() + OaI64(i) * 784,
			Images_.Data() + OaI64(idx) * 784, 784);
		LblBuf_[i] = Labels_[idx];
	}

	Cursor_ += BatchSize_;

	// Convert to OaMatrix (direct GPU upload)
	OutX = OaFnMatrix::FromBytes(
		OaSpan<const OaU8>(ImgBuf_.Data(), ImgBuf_.Size()),
		OaMatrixShape{BatchSize_, 784});
	OutY = OaFnMatrix::FromBytes(
		OaSpan<const OaU8>(LblBuf_.Data(), LblBuf_.Size()),
		OaMatrixShape{BatchSize_}, OaScalarType::UInt8);

	return true;
}

void OaDsMnist::Reset(bool InReshuffle) {
	Cursor_ = 0;
	if (InReshuffle && Shuffle_) {
		std::shuffle(Indices_.begin(), Indices_.end(), Rng_);
	}
}
