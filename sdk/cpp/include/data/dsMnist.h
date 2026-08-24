// oa::DsMnist — Fashion-MNIST / MNIST Dataset (oa::Dataset subclass)
//
// Loads IDX-format images + labels. Inherits oa::Dataset so it works with
// the generic oa::DataLoader, oa::FnDataset::collate, and transform pipelines.
//
// usage:
//   oa::DsMnist ds("/data/fashion_mnist", "train");
//   oa::DataLoader loader(ds, {.batchSize = 64, .shuffle = true});
//   while (auto batch = loader.nextBatch()) { ... }
//
// Legacy convenience batching is also available directly on the class.

#pragma once

#include <oa/data/dataset.h>
#include <oa/core/types.h>
#include <oa/core/matrix.h>
#include <oa/core/fnMatrix.h>
#include <random>

namespace oa {

class DsMnist : public Dataset {
public:
	// load MNIST/Fashion-MNIST dataset from IDX files
	// inDataDir: directory containing *-images-idx3-ubyte and *-labels-idx1-ubyte
	// inSplit: dataset split ("train" or "t10k"), defaults to "train"
	// inBatchSize: batch size for training (used by legacy nextBatch)
	// inShuffle: enable Fisher-Yates shuffling on reset()
	DsMnist(const oa::String& inDataDir, const oa::String& inSplit = "train",
		oa::I32 inBatchSize = 64, bool inShuffle = true);

	// Dataset interface
	[[nodiscard]] oa::I64 size() const override { return numSamples_; }
	[[nodiscard]] Matrix getItem(oa::I64 inIndex) const override;
	[[nodiscard]] Sample getSample(oa::I64 inIndex) const override;

	// Convenience: legacy batching interface (keeps old tutorials working).
	// Internally this is the same as wrapping this loader in oa::DataLoader.
	bool nextBatch(Matrix& outX, Matrix& outY);
	void reset(bool inReshuffle = true);

	// query metadata
	[[nodiscard]] oa::I32 numSamples() const { return numSamples_; }
	[[nodiscard]] oa::I32 numBatches() const { return numBatches_; }
	[[nodiscard]] oa::I32 batchSize() const { return batchSize_; }
	[[nodiscard]] oa::I64 inputSize() const { return 784; }  // 28x28 uint8
	[[nodiscard]] oa::I64 outputSize() const { return 1; }   // uint8 label

private:
	bool loadDataset();
	static oa::U32 readBE32(std::ifstream& inF);

	oa::String dataDir_;
	oa::String split_;
	oa::I32    batchSize_;
	bool     shuffle_;

	oa::Vec<oa::U8> images_;   // [num_samples * 784] flattened images
	oa::Vec<oa::U8> labels_;   // [num_samples] class labels (0-9)
	oa::I32    numSamples_ = 0;
	oa::I32    numBatches_ = 0;

	oa::Vec<oa::I32> indices_;  // Shuffled sample indices
	oa::I32        cursor_ = 0;
	oa::Vec<oa::U8>  imgBuf_;  // Pre-allocated batch image buffer
	oa::Vec<oa::U8>  lblBuf_;  // Pre-allocated batch label buffer

	std::mt19937 rng_;
};

} // namespace oa
