// ML pooling gradient nodes.
#pragma once

#include <oa/core/fnMatrix.h>
#include <oa/ml/autograd.h>
#include <oa/ml/fnMatrix.h>

namespace oa {

class GradMaxPool2d final : public oa::GradNode {
public:
	oa::I32 kernelSize_ = 0;
	oa::I32 stride_ = 0;
	oa::I32 padding_ = 0;
	void backward(const oa::Matrix& inDOut, oa::Vec<oa::Matrix>& outDIn) override {
		if (outDIn.size() > 0) {
			outDIn[0] = oa::FnMatrix::maxPool2dBwd(
				saved(0), saved(2), inDOut,
				kernelSize_, stride_, padding_);
		}
	}
};

class GradAvgPool2d final : public oa::GradNode {
public:
	oa::I32 kernelSize_ = 0;
	oa::I32 stride_ = 0;
	oa::I32 padding_ = 0;

	void backward(const oa::Matrix& inDOut, oa::Vec<oa::Matrix>& outDIn) override {
		if (outDIn.size() > 0) {
			outDIn[0] = oa::FnMatrix::avgPool2dBwd(
				saved(0), inDOut, kernelSize_, stride_, padding_);
		}
	}
};

} // namespace oa
