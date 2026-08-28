// oa::Dataset — dataset values and deterministic batch iteration.

#pragma once

#include <oa/core/matrix.h>

namespace oa {

// Dataset (analogous to torch.utils.data.dataset)

class Dataset {
public:
	virtual ~Dataset() = default;

	/// Multi-tensor sample: input X, target Y (optional), and optional metadata
	struct Sample {
		Matrix x;
		Matrix y;  // Optional: empty if dataset has no labels

		Sample() = default;
		explicit Sample(Matrix inX) : x(oa::move(inX)) {}
		Sample(Matrix inX, Matrix inY) : x(oa::move(inX)), y(oa::move(inY)) {}

		[[nodiscard]] bool hasLabel() const { return !y.isEmpty(); }
	};

	/// Number of items
	[[nodiscard]] virtual oa::I64 size() const = 0;

	/// get single sample by index (unified path)
	[[nodiscard]] virtual Sample getSample(oa::I64 inIndex) const {
		return Sample(getItem(inIndex));
	}

	/// get single item by index (backward-compat for single-output datasets)
	[[nodiscard]] virtual Matrix getItem(oa::I64 inIndex) const = 0;

	/// operator[] for convenience
	[[nodiscard]] Matrix operator[](oa::I64 inIndex) const { return getItem(inIndex); }
};

// DataLoader (analogous to torch.utils.data.dataLoader)

class DataLoaderConfig {
public:
	oa::I32 batchSize = 32;
	bool shuffle = true;
	oa::U64 seed = 0;        // 0 = nondeterministic
	bool dropLast = false;  // Drop incomplete final batch
};

class DataLoader {
public:
	DataLoader(Dataset& inDataset, DataLoaderConfig inConfig = {});

	/// Batched output: X [B, ...] and optional Y [B, ...]
	struct Batch {
		Matrix x;
		Matrix y;

		[[nodiscard]] bool isValid() const { return !x.isEmpty(); }
		explicit operator bool() const { return isValid(); }
	};

	/// get next batch as multi-tensor. Returns nullopt when epoch ends.
	[[nodiscard]] oa::Optional<Batch> nextBatch();

	/// reset to beginning of epoch (re-shuffles if enabled)
	void reset();

	/// Number of batches per epoch
	[[nodiscard]] oa::I64 numBatches() const;

	/// current batch index
	[[nodiscard]] oa::I64 currentBatch() const { return currentBatch_; }

private:
	Dataset& dataset_;
	DataLoaderConfig config_;
	oa::Vector<oa::I64> indices_;
	oa::I64 currentBatch_ = 0;

	void buildIndices();
};

} // namespace oa
