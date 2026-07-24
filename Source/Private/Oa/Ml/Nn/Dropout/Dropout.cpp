// Activations — manual training-mode layers.

#include <Oa/Core/FnMatrix.h>
#include <Oa/Ml/Nn/Dropout/Dropout.h>

OaMatrix OaDropout::Forward(const OaMatrix& InInput) {
	if (!IsTraining() || Probability_ <= 0.0F) {
		// In eval mode, return input unchanged (identity passthrough)
		return InInput;
	}
	
	return OaFnMatrix::Dropout(InInput, Probability_);
}
