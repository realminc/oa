// Activations — manual training-mode layers.

#include <Oa/Ml/Nn.h>
#include <Oa/Core/FnMatrix.h>

OaMatrix OaDropout::Forward(const OaMatrix& InInput) {
	if (!IsTraining() || P_ <= 0.0f) {
		// In eval mode, return input unchanged (identity passthrough)
		return InInput;
	}
	
	return OaFnMatrix::Dropout(InInput, P_);
}
