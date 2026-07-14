#pragma once

#include <Oa/Core/Matrix.h>
#include <Oa/Core/MatrixStorage.h>
#include <Oa/Core/Status.h>
#include <Oa/Core/Types.h>


class OaMatrixList {
public:
	// Named `OaMatrix` map. Save/Load use the same OAML v1 file format as `OaMatrixList`
	// (save converts via `OaMatrixRef::ToHostStorage`; load allocates on the current `OaMatrix` runtime).

	void Add(OaStringView InName, OaMatrix&& InMat);
	void Remove(OaStringView InName);
	void Clear() noexcept;
	void ForEach(const OaFunc<void(OaStringView, const OaMatrix&)>& InFn) const;

	[[nodiscard]] OaResult<OaMatrix> Get(OaStringView InName) const;
	[[nodiscard]] bool Has(OaStringView InName) const;
	[[nodiscard]] OaUsize Size() const noexcept { return Map_.Size(); }

	// SaveToFile / LoadFromFile removed — they returned Unimplemented and had
	// no callers. Re-add as part of the OAML file I/O pass when needed.

private:
	OaHashMap<OaString, OaMatrix> Map_{};
	OaVec<OaString> Order_{};
};
