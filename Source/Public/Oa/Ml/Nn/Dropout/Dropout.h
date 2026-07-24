#pragma once

#include <Oa/Ml/Module.h>

/// Dropout module. Training applies inverted dropout; evaluation is identity.
class OaDropout : public OaModule {
public:
	explicit OaDropout(OaF32 InProbability = 0.1F)
		: Probability_(InProbability) {}

	OaMatrix Forward(const OaMatrix& InInput) override;

private:
	OaF32 Probability_;
};
