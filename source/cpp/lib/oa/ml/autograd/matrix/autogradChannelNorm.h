#pragma once

#include <oa/ml/autograd.h>
#include <oa/ml/fnMatrix.h>

namespace oa {

// Backward node for channel-wise normalization of [B,C,T] storage.
class GradChannelNorm final : public GradNode {
public:
	I32 batch_ = 0;
	I32 channels_ = 0;
	I32 seqLen_ = 0;
	F32 eps_ = 1e-5F;

	void backward(const Matrix& inDOut, Vector<Matrix>& outDIn) override {
		const Matrix& x = saved(0);
		const Matrix& weight = saved(1);
		const bool needDX = outDIn.size() > 0;
		const bool needDW = outDIn.size() > 1;
		const bool needDB = outDIn.size() > 2;
		if (needDX or needDW or needDB) {
			auto grads = FnMatrix::channelNormBwd(
				x, weight, inDOut, batch_, channels_, seqLen_, eps_);
			if (needDX) { outDIn[0] = grads.dx; }
			if (needDW) { outDIn[1] = grads.dWeight; }
			if (needDB) { outDIn[2] = grads.dBias; }
		}
	}
};

// Backward node for fused channel normalization and ReLU.
class GradChannelNormRelu final : public GradNode {
public:
	I32 batch_ = 0;
	I32 channels_ = 0;
	I32 seqLen_ = 0;
	F32 eps_ = 1e-5F;

	void backward(const Matrix& inDOut, Vector<Matrix>& outDIn) override {
		const Matrix& x = saved(0);
		const Matrix& weight = saved(1);
		const Matrix& fwdOut = saved(2);
		const bool needDX = outDIn.size() > 0;
		const bool needDW = outDIn.size() > 1;
		const bool needDB = outDIn.size() > 2;
		if (needDX or needDW or needDB) {
			auto grads = FnMatrix::channelNormReluBwd(
				x, weight, fwdOut, inDOut,
				batch_, channels_, seqLen_, eps_);
			if (needDX) { outDIn[0] = grads.dx; }
			if (needDW) { outDIn[1] = grads.dWeight; }
			if (needDB) { outDIn[2] = grads.dBias; }
		}
	}
};

} // namespace oa
