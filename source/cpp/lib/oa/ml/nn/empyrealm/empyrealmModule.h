#pragma once

#include <oa/ml/nn/mamba3/mamba3.h>

namespace oa {

// EmpyrealmModule — empyrealm SSM mixer (1:1 copy of Mamba3, renamed kernels).
//
// Inherits all parameter layout, init, and structure from Mamba3Module.
// Overrides forward, Preprocess, and step to dispatch empyrealm* kernels
// (EmpyrealmDt, EmpyrealmAdt, EmpyrealmSiso, EmpyrealmSisoStep) instead of
// the Mamba3* equivalents. The SPIR-V is identical today; this separation
// allows future architecture-specific divergence without touching the
// pristine Mamba3 reference.
class EmpyrealmModule : public Mamba3Module {
public:
	using Mamba3Module::Mamba3Module;

	oa::Matrix forward(const oa::Matrix& inInput) override;
	oa::Matrix step(const oa::Matrix& inInput) override;

protected:
	PreprocOut preprocess(const oa::Matrix& inInput, oa::I32 inBatch, oa::I32 inSeqLen) override;
};

} // namespace oa
