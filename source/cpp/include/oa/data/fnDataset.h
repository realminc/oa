// oa::FnDataset — Unified dataset operations (shuffle, split, collate, normalize)
//
// Stateless functional API following the oa::FnMatrix / oa::FnImage pattern.
// Use oa::DataLoader for stateful batch iteration; use oa::FnDataset for one-shot
// dataset manipulation (splitting, shuffling, collation).
//
#pragma once

#include <oa/core/types.h>
#include <oa/core/matrix.h>
#include <oa/data/dataset.h>

namespace oa {

// index shuffling and splitting.
namespace FnDataset {

	/// Fisher-Yates shuffle on an index array.
	void shuffle(oa::Vec<oa::I64>& inOutIndices, oa::U64 inSeed = 0);

	/// Random split result for train / validation / test.
	struct SplitResult {
		oa::Vec<oa::I64> train;
		oa::Vec<oa::I64> val;
		oa::Vec<oa::I64> test;
	};

	/// Randomly split indices into train/val/test subsets.
	/// Ratios are clamped so that train+val+test <= totalSize.
	/// Any remainder goes to the training set.
	SplitResult randomSplit(
		oa::I64 inTotalSize,
		oa::F32 inTrainRatio = 0.8f,
		oa::F32 inValRatio   = 0.1f,
		oa::U64 inSeed       = 42
	);

	// Batch collation
	//
	/// Assemble individual samples into batched matrices.
	/// X is stacked along dim 0; Y is stacked along dim 0 (if present).
	struct Batch {
		Matrix x;
		Matrix y;
	};

	/// Collate a span of samples into a single batch.
	/// Returns an empty batch when samples are empty, inconsistent, or cannot be
	/// represented with a leading batch dimension.
	[[nodiscard]] Batch collate(oa::Span<const Dataset::Sample> inSamples);
} // namespace FnDataset

} // namespace oa
