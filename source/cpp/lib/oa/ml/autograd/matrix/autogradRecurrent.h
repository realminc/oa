// ML recurrent gradient nodes.
#pragma once

#include <oa/core/fnMatrix.h>
#include <oa/ml/autograd.h>
#include <oa/ml/fnMatrix.h>

namespace oa {

class GradGruCellPointwise final : public oa::GradNode {
public:
	oa::I32 hiddenSize_ = 0;
	oa::U32 timeOffset_ = 0;
	oa::U32 batchStride_ = 1;

	void backward(const oa::Matrix& inDOut, oa::Vec<oa::Matrix>& outDIn) override {
		const oa::Matrix& gatesI = saved(0);
		const oa::Matrix& gatesH = saved(1);
		const oa::Matrix& hidden = saved(2);
		const bool needDGatesI = outDIn.size() > 0;
		const bool needDGatesH = outDIn.size() > 1;
		const bool needDHidden = outDIn.size() > 2;
		if (needDGatesI or needDGatesH or needDHidden) {
			auto grads = oa::FnMatrix::gruCellPointwiseBwd(
				gatesI, gatesH, hidden, inDOut, hiddenSize_, timeOffset_,
				batchStride_);
			if (needDGatesI) outDIn[0] = grads.dGatesI;
			if (needDGatesH) outDIn[1] = grads.dGatesH;
			if (needDHidden) outDIn[2] = grads.dHidden;
		}
	}
};

class GradGruScan final : public oa::GradNode {
public:
	oa::I32 hiddenSize_ = 0;
	oa::I32 seqLen_ = 0;
	oa::I32 batch_ = 0;
	bool hasBias_ = false;

	void backward(const oa::Matrix& inDOut, oa::Vec<oa::Matrix>& outDIn) override {
		const oa::Matrix& gatesI = saved(0);
		const oa::Matrix& wHh = saved(1);
		const oa::Matrix& hprev3d = saved(3);
		const bool needDGatesI = outDIn.size() > 0;
		const bool needDW = outDIn.size() > 1;
		const bool needDBias = outDIn.size() > 2;
		if (not (needDGatesI or needDW or needDBias)) return;
		const oa::Matrix emptyBias;
		const oa::Matrix& biasHh = hasBias_ ? saved(2) : emptyBias;
		auto grads = oa::FnMatrix::gruScanBwd(
			inDOut, gatesI, hprev3d, wHh, hiddenSize_, seqLen_, batch_, biasHh);
		if (needDGatesI) outDIn[0] = grads.dGatesI;
		if (needDW or needDBias) {
			auto hprev2d = hprev3d.reshape(oa::MatrixShape{
				static_cast<oa::I64>(batch_) * seqLen_, hiddenSize_});
			auto gwb = oa::FnMatrix::linearWeightBiasBwd(hprev2d, grads.dGatesH);
			if (needDW) outDIn[1] = gwb.gradWeight;
			if (needDBias) outDIn[2] = gwb.gradBias;
		}
	}
};

class GradRnnCellPointwise final : public oa::GradNode {
public:
	oa::I32 hiddenSize_ = 0;
	oa::U32 timeOffset_ = 0;
	oa::U32 batchStride_ = 1;

	void backward(const oa::Matrix& inDOut, oa::Vec<oa::Matrix>& outDIn) override {
		const oa::Matrix& gatesI = saved(0);
		const oa::Matrix& gatesH = saved(1);
		const bool needDGatesI = outDIn.size() > 0;
		const bool needDGatesH = outDIn.size() > 1;
		if (needDGatesI or needDGatesH) {
			auto grads = oa::FnMatrix::rnnCellPointwiseBwd(
				gatesI, gatesH, inDOut, hiddenSize_, timeOffset_, batchStride_);
			if (needDGatesI) outDIn[0] = grads.dGatesI;
			if (needDGatesH) outDIn[1] = grads.dGatesH;
		}
	}
};

class GradRnnScan final : public oa::GradNode {
public:
	oa::I32 hiddenSize_ = 0;
	oa::I32 seqLen_ = 0;
	oa::I32 batch_ = 0;
	bool hasBias_ = false;

	void backward(const oa::Matrix& inDOut, oa::Vec<oa::Matrix>& outDIn) override {
		const oa::Matrix& gatesI = saved(0);
		const oa::Matrix& wHh = saved(1);
		const oa::Matrix& hprev3d = saved(3);
		const bool needDGatesI = outDIn.size() > 0;
		const bool needDW = outDIn.size() > 1;
		const bool needDBias = outDIn.size() > 2;
		if (not (needDGatesI or needDW or needDBias)) return;
		const oa::Matrix emptyBias;
		const oa::Matrix& biasHh = hasBias_ ? saved(2) : emptyBias;
		auto grads = oa::FnMatrix::rnnScanBwd(
			inDOut, gatesI, hprev3d, wHh, hiddenSize_, seqLen_, batch_, biasHh);
		if (needDGatesI) outDIn[0] = grads.dGatesI;
		if (needDW or needDBias) {
			auto hprev2d = hprev3d.reshape(oa::MatrixShape{
				static_cast<oa::I64>(batch_) * seqLen_, hiddenSize_});
			auto gwb = oa::FnMatrix::linearWeightBiasBwd(hprev2d, grads.dGatesH);
			if (needDW) outDIn[1] = gwb.gradWeight;
			if (needDBias) outDIn[2] = gwb.gradBias;
		}
	}
};

} // namespace oa
