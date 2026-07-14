// OaDsMnist — Fashion-MNIST / MNIST Dataset (OaDataset subclass)
//
// Loads IDX-format images + labels. Inherits OaDataset so it works with
// the generic OaDataLoader, OaFnDataset::Collate, and transform pipelines.
//
// Usage:
//   OaDsMnist ds("/data/fashion_mnist", "train");
//   OaDataLoader loader(ds, {.BatchSize = 64, .Shuffle = true});
//   while (auto batch = loader.NextBatch()) { ... }
//
// Legacy convenience batching is also available directly on the class.

#pragma once

#include <Oa/Data/Dataset.h>
#include <Oa/Core/Types.h>
#include <Oa/Core/Matrix.h>
#include <Oa/Core/FnMatrix.h>
#include <random>

class OaDsMnist : public OaDataset {
public:
	// Load MNIST/Fashion-MNIST dataset from IDX files
	// InDataDir: directory containing *-images-idx3-ubyte and *-labels-idx1-ubyte
	// InSplit: dataset split ("train" or "t10k"), defaults to "train"
	// InBatchSize: batch size for training (used by legacy NextBatch)
	// InShuffle: enable Fisher-Yates shuffling on Reset()
	OaDsMnist(const OaString& InDataDir, const OaString& InSplit = "train",
		OaI32 InBatchSize = 64, bool InShuffle = true);

	// OaDataset interface
	[[nodiscard]] OaI64 Size() const override { return NumSamples_; }
	[[nodiscard]] OaMatrix GetItem(OaI64 InIndex) const override;
	[[nodiscard]] Sample GetSample(OaI64 InIndex) const override;

	// Convenience: legacy batching interface (keeps old tutorials working).
	// Internally this is the same as wrapping this loader in OaDataLoader.
	bool NextBatch(OaMatrix& OutX, OaMatrix& OutY);
	void Reset(bool InReshuffle = true);

	// Query metadata
	[[nodiscard]] OaI32 NumSamples() const { return NumSamples_; }
	[[nodiscard]] OaI32 NumBatches() const { return NumBatches_; }
	[[nodiscard]] OaI32 BatchSize() const { return BatchSize_; }
	[[nodiscard]] OaI64 InputSize() const { return 784; }  // 28x28 uint8
	[[nodiscard]] OaI64 OutputSize() const { return 1; }   // uint8 label

private:
	bool LoadDataset();
	static OaU32 ReadBE32(std::ifstream& InF);

	OaString DataDir_;
	OaString Split_;
	OaI32    BatchSize_;
	bool     Shuffle_;

	OaVec<OaU8> Images_;   // [num_samples * 784] flattened images
	OaVec<OaU8> Labels_;   // [num_samples] class labels (0-9)
	OaI32    NumSamples_ = 0;
	OaI32    NumBatches_ = 0;

	OaVec<OaI32> Indices_;  // Shuffled sample indices
	OaI32        Cursor_ = 0;
	OaVec<OaU8>  ImgBuf_;  // Pre-allocated batch image buffer
	OaVec<OaU8>  LblBuf_;  // Pre-allocated batch label buffer

	std::mt19937 Rng_;
};
