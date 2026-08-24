// Autograd loss nodes — gradient nodes for loss functions.

#pragma once

#include <oa/core/fnMatrix.h>
#include <oa/ml/autograd.h>
#include <oa/ml/fnLoss.h>

// oa::GradCrossEntropy — crossEntropy(logits, targets) → d_logits
namespace oa {

class GradCrossEntropy final : public oa::GradNode {
public:
	void backward(const oa::Matrix& inDLoss, oa::Vec<oa::Matrix>& outDIn) override {
		const oa::Matrix& logits  = saved(0);
		const oa::Matrix& targets = saved(1);
		if (outDIn.size() > 0) {
			oa::Matrix dLogits = oa::FnLoss::crossEntropyBwd(logits, targets);
			if (dLogits.getDtype() == inDLoss.getDtype()) {
				outDIn[0] = oa::FnMatrix::mul(dLogits, inDLoss);
			} else {
				// CrossEntropy has an FP32 scalar output even for BF16 logits. Preserve
				// the FP32 upstream while scaling, then restore the leaf-gradient dtype.
				const oa::Matrix scaled = oa::FnMatrix::mul(oa::FnMatrix::cast(dLogits, inDLoss.getDtype()), inDLoss);
				outDIn[0] = oa::FnMatrix::cast(scaled, dLogits.getDtype());
			}
		}
	}
};

class GradMaskedCrossEntropy final : public oa::GradNode {
public:
	oa::I32 validCount_ = 0;
	void backward(const oa::Matrix& /*inDLoss*/, oa::Vec<oa::Matrix>& outDIn) override {
		if (outDIn.size() > 0) {
			outDIn[0] = oa::FnLoss::maskedCrossEntropyBwd(
				saved(0), saved(1), saved(2), validCount_);
		}
	}
};

// oa::GradSmoothL1 — smoothL1(A, B) → d_A (B is target, no grad needed)
class GradSmoothL1 final : public oa::GradNode {
public:
	void backward(const oa::Matrix& inDLoss, oa::Vec<oa::Matrix>& outDIn) override {
		const oa::Matrix& a = saved(0);
		const oa::Matrix& b = saved(1);
		if (outDIn.size() > 0) {
			outDIn[0] = oa::FnMatrix::mul(oa::FnLoss::smoothL1Bwd(a, b), inDLoss);
		}
	}
};

// oa::GradMse — mse(A, B) → d_A
class GradMse final : public oa::GradNode {
public:
	void backward(const oa::Matrix& inDLoss, oa::Vec<oa::Matrix>& outDIn) override {
		const oa::Matrix& a = saved(0);
		const oa::Matrix& b = saved(1);
		if (outDIn.size() > 0) {
			outDIn[0] = oa::FnMatrix::mul(oa::FnLoss::mseBwd(a, b), inDLoss);
		}
	}
};

// oa::GradL1 — L1(A, B) → d_A
class GradL1 final : public oa::GradNode {
public:
	void backward(const oa::Matrix& inDLoss, oa::Vec<oa::Matrix>& outDIn) override {
		const oa::Matrix& a = saved(0);
		const oa::Matrix& b = saved(1);
		if (outDIn.size() > 0) {
			outDIn[0] = oa::FnMatrix::mul(oa::FnLoss::l1Bwd(a, b), inDLoss);
		}
	}
};

// oa::GradBce — bce(A, B) → d_A
class GradBce final : public oa::GradNode {
public:
	void backward(const oa::Matrix& inDLoss, oa::Vec<oa::Matrix>& outDIn) override {
		const oa::Matrix& a = saved(0);
		const oa::Matrix& b = saved(1);
		if (outDIn.size() > 0) {
			outDIn[0] = oa::FnMatrix::mul(oa::FnLoss::bceBwd(a, b), inDLoss);
		}
	}
};

} // namespace oa
