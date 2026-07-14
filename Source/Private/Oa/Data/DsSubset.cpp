// OaDsSubset — Indexed view over any OaDataset

#include <Oa/Data/DsSubset.h>

OaDsSubset::OaDsSubset(OaDataset& InParent, OaSpan<const OaI64> InIndices)
	: Parent_(InParent) {
	Indices_.Reserve(InIndices.Size());
	for (OaUsize i = 0; i < InIndices.Size(); ++i) {
		Indices_.PushBack(InIndices[i]);
	}
}

OaMatrix OaDsSubset::GetItem(OaI64 InIndex) const {
	if (InIndex < 0 || InIndex >= static_cast<OaI64>(Indices_.Size())) {
		return OaMatrix();
	}
	return Parent_.GetItem(Indices_[static_cast<OaUsize>(InIndex)]);
}

OaDataset::Sample OaDsSubset::GetSample(OaI64 InIndex) const {
	if (InIndex < 0 || InIndex >= static_cast<OaI64>(Indices_.Size())) {
		return {};
	}
	return Parent_.GetSample(Indices_[static_cast<OaUsize>(InIndex)]);
}
