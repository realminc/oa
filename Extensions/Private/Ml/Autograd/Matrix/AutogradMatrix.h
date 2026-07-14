// Extension autograd grad nodes — hand-written backward nodes for fused
// kernels that live in Extensions (not core oa).
//
// Pattern mirrors Source/Private/Oa/Ml/Autograd/Matrix/AutogradMatrix.h:
// each grad node subclasses OaGradNode and dispatches the corresponding
// *Bwd kernel(s) in its Backward() method.

#pragma once

#include <Oa/Ml/Autograd.h>
#include <Oa/Ml/FnMatrix.h>
#include <Ml/FnLoss.h>

// ─── Fused kernel gradient nodes ──────────────────────────────────────────

// OaGradChannelNorm — backward for fused ChannelNorm (channel-wise LayerNorm
// on [B,C,T]). Dispatches ChannelNormBwd kernel.
class OaGradChannelNorm final : public OaGradNode {
public:
	OaI32 Batch_ = 0;
	OaI32 Channels_ = 0;
	OaI32 SeqLen_ = 0;
	OaF32 Eps_ = 1e-5F;
	void Backward(const OaMatrix& InDOut, OaVec<OaMatrix>& OutDIn) override {
		const OaMatrix& x = Saved_[0];
		const OaMatrix& weight = Saved_[1];
		const bool needDX = OutDIn.Size() > 0;
		const bool needDW = OutDIn.Size() > 1;
		const bool needDB = OutDIn.Size() > 2;
		if (needDX or needDW or needDB) {
			auto grads = OaFnMatrix::ChannelNormBwd(x, weight, InDOut,
				Batch_, Channels_, SeqLen_, Eps_);
			if (needDX) { OutDIn[0] = grads.DX; }
			if (needDW) { OutDIn[1] = grads.DWeight; }
			if (needDB) { OutDIn[2] = grads.DBias; }
		}
	}
};

// OaGradChannelNormRelu — backward for fused ChannelNorm + ReLU. Dispatches
// the fused ChannelNormReluBwd kernel (masks gradient through ReLU in-shader).
// Saved_ = {x, weight, fwdOut} where fwdOut is the post-ReLU forward output.
class OaGradChannelNormRelu final : public OaGradNode {
public:
	OaI32 Batch_ = 0;
	OaI32 Channels_ = 0;
	OaI32 SeqLen_ = 0;
	OaF32 Eps_ = 1e-5F;
	void Backward(const OaMatrix& InDOut, OaVec<OaMatrix>& OutDIn) override {
		const OaMatrix& x = Saved_[0];
		const OaMatrix& weight = Saved_[1];
		const OaMatrix& fwdOut = Saved_[2];
		const bool needDX = OutDIn.Size() > 0;
		const bool needDW = OutDIn.Size() > 1;
		const bool needDB = OutDIn.Size() > 2;
		if (needDX or needDW or needDB) {
			auto grads = OaFnMatrix::ChannelNormReluBwd(x, weight, fwdOut, InDOut,
				Batch_, Channels_, SeqLen_, Eps_);
			if (needDX) { OutDIn[0] = grads.DX; }
			if (needDW) { OutDIn[1] = grads.DWeight; }
			if (needDB) { OutDIn[2] = grads.DBias; }
		}
	}
};

// OaGradConv1dRelu retired: the fused Conv1dRelu path is now OaFnMatrix::Conv1dReluGemm,
// which composes its own backward (Im2Col1d + MatMulNt + BiasAdd + Relu). The fused
// scalar Conv1dRelu kernel and its Conv1dReluBwd{Data,Weight} kernels were removed.

// OaGradSmoothL1Mean — backward for fused SmoothL1 + Mean.
// Saved_ = {a, b}. Dispatches SmoothL1MeanBwd kernel with the upstream scalar.
class OaGradSmoothL1Mean final : public OaGradNode {
public:
	OaU32 Count_ = 0;
	void Backward(const OaMatrix& InDOut, OaVec<OaMatrix>& OutDIn) override {
		const OaMatrix& a = Saved_[0];
		const OaMatrix& b = Saved_[1];
		if (OutDIn.Size() > 0) {
			OutDIn[0] = OaFnLoss::SmoothL1MeanBwd(a, b, InDOut);
		}
	}
};

// OaGradVelSmoothL1 — backward for fused velocity SmoothL1 + Mean.
// Saved_ = {pred, target}. Dispatches VelSmoothL1Bwd kernel with upstream scalar.
class OaGradVelSmoothL1 final : public OaGradNode {
public:
	OaI32 Batch_ = 0;
	OaI32 SeqLen_ = 0;
	OaI32 FeatDim_ = 0;
	OaU32 Count_ = 0;
	void Backward(const OaMatrix& InDOut, OaVec<OaMatrix>& OutDIn) override {
		const OaMatrix& pred = Saved_[0];
		const OaMatrix& target = Saved_[1];
		if (OutDIn.Size() > 0) {
			OutDIn[0] = OaFnLoss::VelSmoothL1Bwd(pred, target, InDOut);
		}
	}
};

// OaGradIm2Col1d — backward for Im2Col1d (the unfold that turns a 1-D conv into
// a GEMM). The forward output is the column matrix [N*OutL, InC*K]; the backward
// folds its gradient back to the input shape [N, InC, L] via the Col2Im1d kernel
// (gather-accumulate over overlapping receptive fields).
class OaGradIm2Col1d final : public OaGradNode {
public:
	OaI32 N_ = 0;
	OaI32 InC_ = 0;
	OaI32 L_ = 0;
	OaI32 K_ = 0;
	OaI32 S_ = 1;
	OaI32 P_ = 0;
	OaI32 D_ = 1;
	OaI32 OutL_ = 0;
	void Backward(const OaMatrix& InDOut, OaVec<OaMatrix>& OutDIn) override {
		if (OutDIn.Size() > 0) {
			OutDIn[0] = OaFnMatrix::Col2Im1d(InDOut, N_, InC_, L_, K_, S_, P_, D_, OutL_);
		}
	}
};
