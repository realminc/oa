// oa::DsSubset — Indexed view over any oa::Dataset (zero-copy)
//
// Lightweight wrapper that references a parent dataset + a subset of indices.
// Used for train/val/test splits, subsampling, and index filtering.
// No data is copied — DsSubset::getItem(i) forwards to parent.getItem(indices[i]).
//
// usage:
//   // Any Dataset implementation can be wrapped without becoming framework API.
//   auto split = oa::FnDataset::randomSplit(full.size(), 0.8f, 0.1f);
//   oa::DsSubset trainDs(full, split.train);
//   oa::DsSubset valDs(full, split.val);
//   oa::DataLoader trainLoader(trainDs, {.batchSize = 64, .shuffle = true});

#pragma once

#include <oa/data/dataset.h>

namespace oa {

class DsSubset : public Dataset {
public:
	/// Wrap a parent dataset with a subset of indices.
	DsSubset(Dataset& inParent, oa::Span<const oa::I64> inIndices);

	[[nodiscard]] oa::I64 size() const override { return static_cast<oa::I64>(indices_.size()); }
	[[nodiscard]] Matrix getItem(oa::I64 inIndex) const override;
	[[nodiscard]] Sample getSample(oa::I64 inIndex) const override;

	/// access to the underlying index list (for inspection / serialization)
	[[nodiscard]] oa::Span<const oa::I64> indices() const {
		return oa::Span<const oa::I64>(indices_.data(), indices_.size());
	}

	/// parent dataset reference
	[[nodiscard]] Dataset& parent() const { return parent_; }

private:
	Dataset& parent_;
	oa::Vector<oa::I64> indices_;
};

} // namespace oa
