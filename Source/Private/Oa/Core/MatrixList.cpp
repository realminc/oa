#include <Oa/Core/MatrixList.h>

void OaMatrixList::Add(OaStringView InName, OaMatrix&& InMat) {
	OaString key(InName);
	auto found = Map_.Find(key);
	if (found == Map_.End()) {
		Map_.Insert({key, std::move(InMat)});
		Order_.PushBack(key);
		return;
	}
	found->second = std::move(InMat);
}

void OaMatrixList::Remove(OaStringView InName) {
	OaString key(InName);
	if (Map_.Erase(key) == 0) {
		return;
	}
	for (OaUsize idx = 0; idx < Order_.Size(); ++idx) {
		if (Order_[idx] == key) {
			Order_[idx] = std::move(Order_.Back());
			Order_.PopBack();
			return;
		}
	}
}

void OaMatrixList::Clear() noexcept {
	Map_.Clear();
	Order_.Clear();
}

void OaMatrixList::ForEach(const OaFunc<void(OaStringView, const OaMatrix&)>& InFn) const {
	for (const OaString& name : Order_) {
		auto it = Map_.Find(name);
		if (it != Map_.End()) {
			InFn(OaStringView(name), it->second);
		}
	}
}

OaResult<OaMatrix> OaMatrixList::Get(OaStringView InName) const {
	OaString key(InName);
	auto it = Map_.Find(key);
	if (it == Map_.End()) {
		return OaResult<OaMatrix>(OaStatus::NotFound("OaMatrixList::Get"));
	}
	return OaResult<OaMatrix>(it->second);
}

bool OaMatrixList::Has(OaStringView InName) const {
	return Map_.Contains(OaString(InName));
}
