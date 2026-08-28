// Core reduction gradient nodes.
#pragma once

#include <oa/core/autograd.h>
#include <oa/core/fnMatrix.h>

namespace oa {

class GradSum final : public oa::GradNode {
public:
	void backward(const oa::Matrix& inDOut, oa::Vector<oa::Matrix>& outDIn) override {
		if (outDIn.size() == 0) return;
		const oa::Matrix& input = saved(0);
		auto ones = oa::FnMatrix::ones(
			input.getShape(), input.getDtype());
		outDIn[0] = oa::FnMatrix::mul(ones, inDOut);
	}
};

class GradMean final : public oa::GradNode {
public:
	oa::I32 dim_ = -1;

	void backward(const oa::Matrix& inDOut, oa::Vector<oa::Matrix>& outDIn) override {
		if (outDIn.size() == 0) return;
		const oa::Matrix& input = saved(0);
		const oa::I64 count = dim_ >= 0 and dim_ < input.rank()
			? input.size(dim_) : input.numElements();
		auto scale = oa::FnMatrix::full(
			input.getShape(),
			1.0 / static_cast<oa::F64>(count),
			input.getDtype());
		outDIn[0] = oa::FnMatrix::mul(scale, inDOut);
	}
};

class GradSoftmax final : public oa::GradNode {
public:
	oa::I32 dim_ = -1;

	void backward(const oa::Matrix& inDOut, oa::Vector<oa::Matrix>& outDIn) override {
		if (outDIn.size() > 0) {
			outDIn[0] = oa::FnMatrix::softmaxBwd(
				saved(0), inDOut, dim_);
		}
	}
};

class GradLogSoftmax final : public oa::GradNode {
public:
	oa::I32 dim_ = -1;

	void backward(const oa::Matrix& inDOut, oa::Vector<oa::Matrix>& outDIn) override {
		if (outDIn.size() > 0) {
			outDIn[0] = oa::FnMatrix::logSoftmaxBwd(
				saved(0), inDOut, dim_);
		}
	}
};

class GradMax final : public oa::GradNode {
public:
	void backward(const oa::Matrix& inDOut, oa::Vector<oa::Matrix>& outDIn) override {
		if (outDIn.size() > 0) {
			outDIn[0] = oa::FnMatrix::maxBwd(
				saved(0), saved(1), inDOut);
		}
	}
};

} // namespace oa
