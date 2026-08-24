// oa::DsSubset — indexed view over any oa::Dataset.

#include <oa/data/dsSubset.h>

namespace oa {

DsSubset::DsSubset(Dataset& inParent, oa::Span<const oa::I64> inIndices)
	: parent_(inParent) {
	indices_.reserve(inIndices.size());
	for (oa::Usize i = 0; i < inIndices.size(); ++i) {
		indices_.pushBack(inIndices[i]);
	}
}

oa::Matrix DsSubset::getItem(oa::I64 inIndex) const {
	if (inIndex < 0 || inIndex >= static_cast<oa::I64>(indices_.size())) {
		return oa::Matrix();
	}
	return parent_.getItem(indices_[static_cast<oa::Usize>(inIndex)]);
}

Dataset::Sample DsSubset::getSample(oa::I64 inIndex) const {
	if (inIndex < 0 || inIndex >= static_cast<oa::I64>(indices_.size())) {
		return {};
	}
	return parent_.getSample(indices_[static_cast<oa::Usize>(inIndex)]);
}

} // namespace oa
