#include <oa/ml/optim.h>
#include <oa/core/fnMatrix.h>

namespace {

bool isLowPrecisionParameter(const oa::Matrix& inData) {
	return inData.getDtype() == oa::ScalarType::BFloat16;
}

} // namespace

void oa::Optimizer::prepareMasters() {
	if (not mixedReady_) {
		master_.resize(params_.size());
		gradF32_.resize(params_.size());
		for (oa::Usize i = 0; i < params_.size(); ++i) {
			oa::Parameter* parameter = params_[i];
			if (parameter != nullptr and isLowPrecisionParameter(parameter->data)) {
				master_[i] = oa::FnMatrix::empty(
					parameter->data.getShape(), oa::ScalarType::Float32);
			}
		}
		mixedReady_ = true;
	}

	if (not masterSeeded_) {
		for (oa::Usize i = 0; i < master_.size(); ++i) {
			if (not master_[i].isEmpty()) {
				oa::FnMatrix::castInto(params_[i]->data, master_[i]);
			}
		}
		masterSeeded_ = true;
	}
}

bool oa::Optimizer::hasMaster(oa::Usize inIndex) const {
	return inIndex < master_.size() and not master_[inIndex].isEmpty();
}

oa::Matrix& oa::Optimizer::masterOrData(oa::Usize inIndex) {
	return hasMaster(inIndex) ? master_[inIndex] : params_[inIndex]->data;
}

oa::Matrix oa::Optimizer::masterGrad(
	oa::Usize inIndex,
	const oa::Matrix& inGrad
) {
	if (not hasMaster(inIndex)) return inGrad;
	gradF32_[inIndex] = oa::FnMatrix::cast(inGrad, oa::ScalarType::Float32);
	return gradF32_[inIndex];
}

void oa::Optimizer::writebackMaster(oa::Usize inIndex) {
	if (hasMaster(inIndex)) {
		oa::FnMatrix::castInto(master_[inIndex], params_[inIndex]->data);
	}
}
