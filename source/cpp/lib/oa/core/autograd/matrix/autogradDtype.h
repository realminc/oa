// Core dtype-conversion gradient nodes.
#pragma once

#include <oa/core/autograd.h>
#include <oa/core/fnMatrix.h>

namespace oa {

class GradCast final : public oa::GradNode {
public:
	oa::ScalarType srcDtype_ = oa::ScalarType::Float32;

	void backward(const oa::Matrix& inDOut, oa::Vector<oa::Matrix>& outDIn) override {
		if (outDIn.size() > 0) {
			outDIn[0] = oa::FnMatrix::cast(inDOut, srcDtype_);
		}
	}
};

} // namespace oa
