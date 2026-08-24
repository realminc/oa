// oa::DsMnist — Fashion-MNIST / MNIST dataset implementation.

#include <data/dsMnist.h>
#include <oa/core/log.h>
#include <fstream>
#include <algorithm>

namespace oa {

DsMnist::DsMnist(
	const oa::String& inDataDir,
	const oa::String& inSplit,
	oa::I32 inBatchSize,
	bool inShuffle
)
	: dataDir_(inDataDir)
	, split_(inSplit)
	, batchSize_(inBatchSize)
	, shuffle_(inShuffle)
	, rng_(42)
{
	if (batchSize_ <= 0) {
		OaLogError(oa::LogComponent::Data, "MNIST batch size must be positive");
		return;
	}
	if (!loadDataset()) {
		OaLogError(oa::LogComponent::Data,
			"Failed to load MNIST dataset from: %s (split: %s)",
			inDataDir.cStr(), inSplit.cStr());
		return;
	}

	// initialize indices for shuffling
	indices_.resize(numSamples_);
	for (oa::I32 i = 0; i < numSamples_; ++i) indices_[i] = i;

	if (shuffle_) {
		std::shuffle(indices_.begin(), indices_.end(), rng_);
	}

	// Pre-allocate batch buffers
	imgBuf_.resize(static_cast<oa::I64>(batchSize_) * 784);
	lblBuf_.resize(batchSize_);
}

bool DsMnist::loadDataset() {
	// Construct file paths using the split name
	oa::String imgPath = dataDir_ + "/" + split_ + "-images-idx3-ubyte";
	oa::String lblPath = dataDir_ + "/" + split_ + "-labels-idx1-ubyte";

	std::ifstream imgF(imgPath.cStr(), std::ios::binary);
	std::ifstream lblF(lblPath.cStr(), std::ios::binary);

	if (!imgF || !lblF) {
		OaLogError(oa::LogComponent::Data,
			"MNIST files not found: %s, %s", imgPath.cStr(), lblPath.cStr());
		return false;
	}

	// verify magic numbers
	if (readBE32(imgF) != 0x00000803u) {
		OaLogError(oa::LogComponent::Data, "Invalid MNIST image file magic number");
		return false;
	}
	if (readBE32(lblF) != 0x00000801u) {
		OaLogError(oa::LogComponent::Data, "Invalid MNIST label file magic number");
		return false;
	}

	// Read dimensions
	oa::U32 n = readBE32(imgF);
	oa::U32 rows = readBE32(imgF);
	oa::U32 cols = readBE32(imgF);
	oa::U32 nLabels = readBE32(lblF);

	if (rows != 28 || cols != 28) {
		OaLogError(oa::LogComponent::Data, "Invalid MNIST image dimensions: %ux%u", rows, cols);
		return false;
	}
	if (n != nLabels) {
		OaLogError(oa::LogComponent::Data, "MNIST image/label count mismatch: %u vs %u", n, nLabels);
		return false;
	}

	numSamples_ = static_cast<oa::I32>(n);
	numBatches_ = (numSamples_ + batchSize_ - 1) / batchSize_;

	// load images and labels
	images_.resize(static_cast<oa::I64>(n) * 784);
	imgF.read(reinterpret_cast<char*>(images_.data()), images_.size());

	labels_.resize(static_cast<oa::I64>(n));
	lblF.read(reinterpret_cast<char*>(labels_.data()), labels_.size());

	if (!imgF || !lblF) {
		OaLogError(oa::LogComponent::Data, "Failed to read MNIST data");
		return false;
	}

	OaLogInfo(oa::LogComponent::Data, "Loaded MNIST: %d samples, %d batches (batch=%d)",
		numSamples_, numBatches_, batchSize_);
	return true;
}

oa::U32 DsMnist::readBE32(std::ifstream& inF) {
	oa::U8 b[4];
	inF.read(reinterpret_cast<char*>(b), 4);
	return (oa::U32(b[0]) << 24) | (oa::U32(b[1]) << 16) | (oa::U32(b[2]) << 8) | oa::U32(b[3]);
}

oa::Matrix DsMnist::getItem(oa::I64 inIndex) const {
	if (inIndex < 0 || inIndex >= numSamples_) return {};
	oa::Span<const oa::U8> imgSlice(images_.data() + inIndex * 784, 784);
	return oa::FnMatrix::fromBytes(imgSlice, oa::MatrixShape{784}, oa::ScalarType::UInt8);
}

Dataset::Sample DsMnist::getSample(oa::I64 inIndex) const {
	if (inIndex < 0 || inIndex >= numSamples_) return {};
	oa::Span<const oa::U8> imgSlice(images_.data() + inIndex * 784, 784);
	oa::Span<const oa::U8> lblSlice(labels_.data() + inIndex, 1);
	return Sample(
		oa::FnMatrix::fromBytes(imgSlice, oa::MatrixShape{784}, oa::ScalarType::UInt8),
		oa::FnMatrix::fromBytes(lblSlice, oa::MatrixShape{1}, oa::ScalarType::UInt8)
	);
}

bool DsMnist::nextBatch(oa::Matrix& outX, oa::Matrix& outY) {
	if (cursor_ + batchSize_ > numSamples_) {
		return false;  // epoch complete
	}

	// Copy batch from shuffled indices
	for (oa::I32 i = 0; i < batchSize_; ++i) {
		oa::I32 idx = indices_[cursor_ + i];
		std::memcpy(imgBuf_.data() + oa::I64(i) * 784,
			images_.data() + oa::I64(idx) * 784, 784);
		lblBuf_[i] = labels_[idx];
	}

	cursor_ += batchSize_;

	// convert to oa::Matrix (direct GPU upload)
	outX = oa::FnMatrix::fromBytes(
		oa::Span<const oa::U8>(imgBuf_.data(), imgBuf_.size()),
		oa::MatrixShape{batchSize_, 784});
	outY = oa::FnMatrix::fromBytes(
		oa::Span<const oa::U8>(lblBuf_.data(), lblBuf_.size()),
		oa::MatrixShape{batchSize_}, oa::ScalarType::UInt8);

	return true;
}

void DsMnist::reset(bool inReshuffle) {
	cursor_ = 0;
	if (inReshuffle && shuffle_) {
		std::shuffle(indices_.begin(), indices_.end(), rng_);
	}
}

} // namespace oa
