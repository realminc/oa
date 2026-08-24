// ML activation and gated-composite gradient nodes.
#pragma once

#include <oa/core/fnMatrix.h>
#include <oa/ml/autograd.h>
#include <oa/ml/fnMatrix.h>

#include "autogradActivation.gen.h"

namespace oa {

class GradSoftmaxScaledMasked final : public oa::GradNode {
public:
	oa::F32 scale_ = 1.0F;

	void backward(const oa::Matrix& inDOut, oa::Vec<oa::Matrix>& outDIn) override {
		if (outDIn.size() > 0) {
			outDIn[0] = oa::FnMatrix::softmaxScaledMaskedBwd(
				saved(0), inDOut, scale_);
		}
	}
};

class GradSiluMul final : public oa::GradNode {
public:
	void backward(const oa::Matrix& inDOut, oa::Vec<oa::Matrix>& outDIn) override {
		if (outDIn.size() > 0) {
			outDIn[0] = oa::FnMatrix::siluMulBwd(saved(0), inDOut);
		}
	}
};

class GradGeglu final : public oa::GradNode {
public:
	void backward(const oa::Matrix& inDOut, oa::Vec<oa::Matrix>& outDIn) override {
		if (outDIn.size() > 0) {
			outDIn[0] = oa::FnMatrix::gegluBwd(saved(0), inDOut);
		}
	}
};

class GradSwiglu final : public oa::GradNode {
public:
	void backward(const oa::Matrix& inDOut, oa::Vec<oa::Matrix>& outDIn) override {
		const oa::Matrix& gate = saved(0);
		const oa::Matrix& up   = saved(1);
		const oa::Matrix& out  = saved(2);
		const bool needGate = outDIn.size() > 0;
		const bool needUp   = outDIn.size() > 1;
		if (needGate or needUp) {
			auto grads = oa::FnMatrix::swigluBwd(gate, up, out, inDOut);
			if (needGate) outDIn[0] = grads.dGate;
			if (needUp)   outDIn[1] = grads.dUp;
		}
	}
};

} // namespace oa
