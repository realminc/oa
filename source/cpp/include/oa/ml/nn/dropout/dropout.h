#pragma once

#include <oa/ml/module.h>

namespace oa {

/// Dropout module. training applies inverted dropout; evaluation is identity.
class Dropout : public oa::Module {
public:
	explicit Dropout(oa::F32 inProbability = 0.1F)
		: probability_(inProbability) {}

	oa::Matrix forward(const oa::Matrix& inInput) override;

private:
	oa::F32 probability_;
};

} // namespace oa
