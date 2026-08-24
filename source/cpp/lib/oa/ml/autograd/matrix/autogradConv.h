#pragma once

#include <oa/core/fnMatrix.h>
#include <oa/ml/autograd.h>
#include <oa/ml/fnMatrix.h>

// ─── Convolution ──────────────────────────────────────────────────────────

namespace oa {

// Backward node for the 1-D im2col transform used by Conv1d GEMM lowering.
class GradIm2Col1d final : public GradNode {
public:
	I32 n_ = 0;
	I32 inC_ = 0;
	I32 l_ = 0;
	I32 k_ = 0;
	I32 s_ = 1;
	I32 p_ = 0;
	I32 d_ = 1;
	I32 outL_ = 0;

	void backward(const Matrix& inDOut, Vec<Matrix>& outDIn) override {
		if (outDIn.size() > 0) {
			outDIn[0] = FnMatrix::col2Im1d(
				inDOut, n_, inC_, l_, k_, s_, p_, d_, outL_);
		}
	}
};

class GradConv2d final : public oa::GradNode {
public:
	oa::U32 stride_ = 1;
	oa::U32 padding_ = 0;
	oa::U32 groups_ = 1;
	void backward(const oa::Matrix& inDOut, oa::Vec<oa::Matrix>& outDIn) override {
		const oa::Matrix& x = saved(0);
		const oa::Matrix& weight = saved(1);
		const oa::MatrixShape input_shape = x.getShape();
		const bool needDX = outDIn.size() > 0;
		const bool needDW = outDIn.size() > 1;
		const bool needDB = outDIn.size() > 2;
		if (needDX) outDIn[0] = oa::FnMatrix::conv2dBwdData(inDOut, weight, stride_, padding_, input_shape, groups_);
		if (needDW or needDB) {
			auto gwb = oa::FnMatrix::conv2dBwdWeight(x, inDOut, weight, stride_, padding_, groups_);
			if (needDW) outDIn[1] = gwb.gradWeight;
			if (needDB) outDIn[2] = gwb.gradBias;
		}
	}
};
// oa::GradConv1d retired: oa::Conv1d::forward now uses oa::FnMatrix::conv1dGemm, which
// composes its own backward from Im2Col1d + MatMulNt + BiasAdd. The bare scalar
// Conv1d forward kernel it depended on was retired. Conv1dBwdData/Conv1dBwdWeight
// survive — they back oa::ConvTranspose1d (below) and are the adjoints reused there.

// oa::GradConvTranspose2d — 2D transposed convolution gradient node.
// forward: y = ConvTranspose2d(x, W, b) with W [inC, outC, K, K].
// backward reuses Conv2d kernels: dX = Conv2d(dOut, W), dW/dB = ConvTranspose2dBwdWeight.
class GradConvTranspose2d final : public oa::GradNode {
public:
	oa::U32 stride_ = 1;
	oa::U32 padding_ = 0;
	void backward(const oa::Matrix& inDOut, oa::Vec<oa::Matrix>& outDIn) override {
		const oa::Matrix& x = saved(0);
		const oa::Matrix& weight = saved(1);
		const oa::MatrixShape input_shape = x.getShape();
		const bool needDX = outDIn.size() > 0;
		const bool needDW = outDIn.size() > 1;
		const bool needDB = outDIn.size() > 2;
		if (needDX) outDIn[0] = oa::FnMatrix::convTranspose2dBwdData(inDOut, weight, stride_, padding_, input_shape);
		if (needDW or needDB) {
			auto gwb = oa::FnMatrix::convTranspose2dBwdWeight(x, inDOut, weight, stride_, padding_);
			if (needDW) outDIn[1] = gwb.gradWeight;
			if (needDB) outDIn[2] = gwb.gradBias;
		}
	}
};

// oa::ConvTranspose1d is the ADJOINT of Conv1d (learnable upsampling), so its forward is
// exactly Conv1d's backward-data and its backward reuses Conv1d's forward + weight-grad —
// no new kernels. weight is [inCh, outCh, K] (PyTorch ConvTranspose1d convention; equals
// the underlying conv's [cout, cin, K] with cout=inCh, cin=outCh). No bias (v1).
//   forward: y  = conv1dBwdData(x, W)                  [B,inCh,Lin] → [B,outCh,Lout]
//   dX = Conv1d(dOut, W)                (adjoint of bwd-data is the forward conv)
//   dW = conv1dBwdWeight(input=dOut, dOut=x, W).gradWeight
class GradConvTranspose1d final : public oa::GradNode {
public:
	oa::U32 stride_ = 1;
	oa::U32 padding_ = 0;
	oa::U32 dilation_ = 1;
	void backward(const oa::Matrix& inDOut, oa::Vec<oa::Matrix>& outDIn) override {
		const oa::Matrix& x      = saved(0);
		const oa::Matrix& weight = saved(1);
		const bool needDX = outDIn.size() > 0;
		const bool needDW = outDIn.size() > 1;
		if (needDX) {
			// Adjoint of ConvTranspose1d's bwd-data forward is a plain conv. Runs
			// under autograd-disabled backward, so Conv1dGemm here is pure im2col+GEMM
			// math (its sub-ops no-op their grad nodes) — the retired scalar Conv1d.
			oa::Matrix zeroBias = oa::FnMatrix::zeros(oa::MatrixShape{weight.size(0)}, weight.getDtype());
			outDIn[0] = oa::FnMatrix::conv1dGemm(inDOut, weight, zeroBias,
				static_cast<oa::I32>(stride_), static_cast<oa::I32>(padding_),
				static_cast<oa::I32>(dilation_)
			);
		}
		if (needDW) {
			auto gwb = oa::FnMatrix::conv1dBwdWeight(inDOut, x, weight, stride_, padding_, dilation_);
			outDIn[1] = gwb.gradWeight;
		}
	}
};

} // namespace oa
