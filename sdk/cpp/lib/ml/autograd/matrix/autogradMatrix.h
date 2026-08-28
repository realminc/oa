// SDK autograd nodes for sample-domain fused losses.
//
// Pattern mirrors source/cpp/lib/oa/ml/autograd/matrix/autogradMatrix.h:
// each grad node subclasses oa::GradNode and dispatches the corresponding
// *Bwd kernel(s) in its backward() method.

#pragma once

#include <oa/ml/autograd.h>
#include <ml/fnLoss.h>

namespace oa {

// GradSmoothL1Mean — backward for fused SmoothL1 + Mean.
// saved_ = {a, b}. Dispatches SmoothL1MeanBwd kernel with the upstream scalar.
class GradSmoothL1Mean final : public oa::GradNode {
public:
	oa::U32 count_ = 0;
	void backward(const oa::Matrix& inDOut, oa::Vector<oa::Matrix>& outDIn) override {
		const oa::Matrix& a = saved(0);
		const oa::Matrix& b = saved(1);
		if (outDIn.size() > 0) {
			outDIn[0] = oa::FnLoss::smoothL1MeanBwd(a, b, inDOut);
		}
	}
};

// GradVelSmoothL1 — backward for fused velocity SmoothL1 + Mean.
// saved_ = {pred, target}. Dispatches VelSmoothL1Bwd kernel with upstream scalar.
class GradVelSmoothL1 final : public oa::GradNode {
public:
	oa::I32 batch_ = 0;
	oa::I32 seqLen_ = 0;
	oa::I32 featDim_ = 0;
	oa::U32 count_ = 0;
	void backward(const oa::Matrix& inDOut, oa::Vector<oa::Matrix>& outDIn) override {
		const oa::Matrix& pred = saved(0);
		const oa::Matrix& target = saved(1);
		if (outDIn.size() > 0) {
			outDIn[0] = oa::FnLoss::velSmoothL1Bwd(pred, target, inDOut);
		}
	}
};

} // namespace oa
