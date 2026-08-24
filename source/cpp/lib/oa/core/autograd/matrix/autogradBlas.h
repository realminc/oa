// Core BLAS gradient nodes.
#pragma once

#include <oa/core/autograd.h>
#include <oa/core/fnMatrix.h>

namespace oa {

class GradMatMulNt final : public oa::GradNode {
public:
	void backward(const oa::Matrix& inDOut, oa::Vec<oa::Matrix>& outDIn) override {
		const oa::Matrix& a = saved(0);
		const oa::Matrix& b = saved(1);
		const oa::MatrixShape expected{a.size(0), b.size(0)};
		oa::Matrix gradOutput = inDOut;
		if (gradOutput.getShape() != expected
			and gradOutput.numElements() == expected.numElements())
		{
			gradOutput = gradOutput.reshape(expected);
		}
		if (outDIn.size() > 0) {
			outDIn[0] = oa::FnMatrix::matMulNt(
				gradOutput, oa::FnMatrix::transpose(b, 0, 1));
		}
		if (outDIn.size() > 1) {
			outDIn[1] = oa::FnMatrix::matMulNt(
				oa::FnMatrix::transpose(gradOutput, 0, 1),
				oa::FnMatrix::transpose(a, 0, 1));
		}
	}
};

} // namespace oa
