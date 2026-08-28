// Core shape/view gradient nodes.
#pragma once

#include <oa/core/autograd.h>
#include <oa/core/fnMatrix.h>

namespace oa {

class GradCopy final : public oa::GradNode {
public:
	void backward(const oa::Matrix& inDOut, oa::Vector<oa::Matrix>& outDIn) override {
		if (outDIn.size() > 0) outDIn[0] = inDOut;
	}
};

class GradTranspose final : public oa::GradNode {
public:
	oa::I32 dim0_ = 0;
	oa::I32 dim1_ = 1;

	void backward(const oa::Matrix& inDOut, oa::Vector<oa::Matrix>& outDIn) override {
		if (outDIn.size() > 0) {
			outDIn[0] = oa::FnMatrix::transpose(inDOut, dim0_, dim1_);
		}
	}
};

class GradReshape final : public oa::GradNode {
public:
	oa::MatrixShape inputShape_{};

	void backward(const oa::Matrix& inDOut, oa::Vector<oa::Matrix>& outDIn) override {
		if (outDIn.size() > 0) {
			outDIn[0] = inDOut.reshape(inputShape_);
		}
	}
};

} // namespace oa
