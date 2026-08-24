// Activations — manual training-mode layers.

#include <oa/core/fnMatrix.h>
#include <oa/ml/nn/dropout/dropout.h>

oa::Matrix oa::Dropout::forward(const oa::Matrix& inInput) {
	if (!isTraining() || probability_ <= 0.0F) {
		// in eval mode, return input unchanged (identity passthrough)
		return inInput;
	}
	
	return oa::FnMatrix::dropout(inInput, probability_);
}
