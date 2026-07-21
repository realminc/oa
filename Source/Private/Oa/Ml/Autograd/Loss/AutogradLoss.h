// OaAutogradFnLoss — Gradient nodes for loss functions.

#pragma once

#include <Oa/Ml/Autograd.h>
#include <Oa/Ml/FnLoss.h>

// OaGradCrossEntropy — CrossEntropy(logits, targets) → d_logits
class OaGradCrossEntropy final : public OaGradNode {
public:
	void Backward(const OaMatrix& InDLoss, OaVec<OaMatrix>& OutDIn) override {
		const OaMatrix& logits  = Saved_[0];
		const OaMatrix& targets = Saved_[1];
		if (OutDIn.Size() > 0) {
			OaMatrix dLogits = OaFnLoss::CrossEntropyBwd(logits, targets);
			if (dLogits.GetDtype() == InDLoss.GetDtype()) {
				OutDIn[0] = OaFnMatrix::Mul(dLogits, InDLoss);
			} else {
				// CrossEntropy has an FP32 scalar output even for BF16 logits. Preserve
				// the FP32 upstream while scaling, then restore the leaf-gradient dtype.
				const OaMatrix scaled = OaFnMatrix::Mul(OaFnMatrix::Cast(dLogits, InDLoss.GetDtype()), InDLoss);
				OutDIn[0] = OaFnMatrix::Cast(scaled, dLogits.GetDtype());
			}
		}
	}
};

class OaGradMaskedCrossEntropy final : public OaGradNode {
public:
	OaI32 ValidCount_ = 0;
	void Backward(const OaMatrix& /*InDLoss*/, OaVec<OaMatrix>& OutDIn) override {
		if (OutDIn.Size() > 0) {
			OutDIn[0] = OaFnLoss::MaskedCrossEntropyBwd(
				Saved_[0], Saved_[1], Saved_[2], ValidCount_);
		}
	}
};

// OaGradSmoothL1 — SmoothL1(A, B) → d_A (B is target, no grad needed)
class OaGradSmoothL1 final : public OaGradNode {
public:
	void Backward(const OaMatrix& InDLoss, OaVec<OaMatrix>& OutDIn) override {
		const OaMatrix& a = Saved_[0];
		const OaMatrix& b = Saved_[1];
		if (OutDIn.Size() > 0) {
			OutDIn[0] = OaFnMatrix::Mul(OaFnLoss::SmoothL1Bwd(a, b), InDLoss);
		}
	}
};

// OaGradMse — Mse(A, B) → d_A
class OaGradMse final : public OaGradNode {
public:
	void Backward(const OaMatrix& InDLoss, OaVec<OaMatrix>& OutDIn) override {
		const OaMatrix& a = Saved_[0];
		const OaMatrix& b = Saved_[1];
		if (OutDIn.Size() > 0) {
			OutDIn[0] = OaFnMatrix::Mul(OaFnLoss::MseBwd(a, b), InDLoss);
		}
	}
};

// OaGradL1 — L1(A, B) → d_A
class OaGradL1 final : public OaGradNode {
public:
	void Backward(const OaMatrix& InDLoss, OaVec<OaMatrix>& OutDIn) override {
		const OaMatrix& a = Saved_[0];
		const OaMatrix& b = Saved_[1];
		if (OutDIn.Size() > 0) {
			OutDIn[0] = OaFnMatrix::Mul(OaFnLoss::L1Bwd(a, b), InDLoss);
		}
	}
};

// OaGradBce — Bce(A, B) → d_A
class OaGradBce final : public OaGradNode {
public:
	void Backward(const OaMatrix& InDLoss, OaVec<OaMatrix>& OutDIn) override {
		const OaMatrix& a = Saved_[0];
		const OaMatrix& b = Saved_[1];
		if (OutDIn.Size() > 0) {
			OutDIn[0] = OaFnMatrix::Mul(OaFnLoss::BceBwd(a, b), InDLoss);
		}
	}
};
