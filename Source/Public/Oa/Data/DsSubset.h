// OaDsSubset — Indexed view over any OaDataset (zero-copy)
//
// Lightweight wrapper that references a parent dataset + a subset of indices.
// Used for train/val/test splits, subsampling, and index filtering.
// No data is copied — OaDsSubset::GetItem(i) forwards to Parent.GetItem(Indices[i]).
//
// Usage:
//   OaDsMnist full("/data/mnist", "train");
//   auto split = OaFnDataset::RandomSplit(full.Size(), 0.8f, 0.1f);
//   OaDsSubset trainDs(full, split.Train);
//   OaDsSubset valDs(full, split.Val);
//   OaDataLoader trainLoader(trainDs, {.BatchSize = 64, .Shuffle = true});

#pragma once

#include <Oa/Data/Dataset.h>

class OaDsSubset : public OaDataset {
public:
	/// Wrap a parent dataset with a subset of indices.
	OaDsSubset(OaDataset& InParent, OaSpan<const OaI64> InIndices);

	[[nodiscard]] OaI64 Size() const override { return static_cast<OaI64>(Indices_.Size()); }
	[[nodiscard]] OaMatrix GetItem(OaI64 InIndex) const override;
	[[nodiscard]] Sample GetSample(OaI64 InIndex) const override;

	/// Access to the underlying index list (for inspection / serialization)
	[[nodiscard]] OaSpan<const OaI64> Indices() const {
		return OaSpan<const OaI64>(Indices_.Data(), Indices_.Size());
	}

	/// Parent dataset reference
	[[nodiscard]] OaDataset& Parent() const { return Parent_; }

private:
	OaDataset& Parent_;
	OaVec<OaI64> Indices_;
};
