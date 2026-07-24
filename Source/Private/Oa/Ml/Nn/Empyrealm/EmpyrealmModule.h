#pragma once

#include <Oa/Ml/Nn/Mamba3/Mamba3.h>

// OaEmpyrealmModule — Empyrealm SSM mixer (1:1 copy of Mamba3, renamed kernels).
//
// Inherits all parameter layout, init, and structure from OaMamba3Module.
// Overrides Forward, Preprocess, and Step to dispatch Empyrealm* kernels
// (EmpyrealmDt, EmpyrealmAdt, EmpyrealmSiso, EmpyrealmSisoStep) instead of
// the Mamba3* equivalents. The SPIR-V is identical today; this separation
// allows future architecture-specific divergence without touching the
// pristine Mamba3 reference.
class OaEmpyrealmModule : public OaMamba3Module {
public:
	using OaMamba3Module::OaMamba3Module;

	OaMatrix Forward(const OaMatrix& InInput) override;
	OaMatrix Step(const OaMatrix& InInput) override;

protected:
	PreprocOut Preprocess(const OaMatrix& InInput, OaI32 InBatch, OaI32 InSeqLen) override;
};
