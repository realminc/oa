// Core indexing and selection gradient nodes.
#pragma once

#include <oa/core/autograd.h>
#include <oa/core/fnMatrix.h>

namespace oa {

class GradRepeatInterleave final : public oa::GradNode {
public:
	oa::I32 repeats_ = 1;
	oa::I32 dim_ = 0;

	void backward(const oa::Matrix& inDOut, oa::Vec<oa::Matrix>& outDIn) override {
		if (outDIn.size() == 0) return;
		const oa::Matrix& input = saved(0);
		outDIn[0] = oa::FnMatrix::repeatInterleaveBwd(
			inDOut, input.getShape(), repeats_, dim_);
	}
};

class GradCausalMask final : public oa::GradNode {
public:
	void backward(const oa::Matrix& inDOut, oa::Vec<oa::Matrix>& outDIn) override {
		if (outDIn.size() > 0) {
			outDIn[0] = oa::FnMatrix::causalMaskBwd(inDOut);
		}
	}
};

class GradCompactRows final : public oa::GradNode {
public:
	oa::Matrix rowMap_;
	oa::Matrix count_;
	oa::Matrix dispatchArgs_;

	void backward(const oa::Matrix& inDOut, oa::Vec<oa::Matrix>& outDIn) override {
		if (outDIn.size() == 0) return;
		const oa::Matrix& input = saved(0);
		outDIn[0] = oa::FnMatrix::compactRowsBwd(
			inDOut, rowMap_, count_, dispatchArgs_, input.getShape());
	}
};

class GradScatterRows final : public oa::GradNode {
public:
	oa::Matrix rowMap_;
	oa::Matrix count_;
	oa::Matrix dispatchArgs_;

	void backward(const oa::Matrix& inDOut, oa::Vec<oa::Matrix>& outDIn) override {
		if (outDIn.size() < 2) return;
		outDIn[0] = oa::FnMatrix::copy(inDOut);
		outDIn[1] = dispatchArgs_.numElements() == 3
			? oa::FnMatrix::scatterRowsBwdSource(
				inDOut, rowMap_, count_, dispatchArgs_)
			: oa::FnMatrix::scatterRowsBwdSource(
				inDOut, rowMap_, count_);
	}
};

class GradGather final : public oa::GradNode {
public:
	void backward(const oa::Matrix& inDOut, oa::Vec<oa::Matrix>& outDIn) override {
		if (outDIn.size() == 0) return;
		const oa::Matrix& indices = saved(0);
		const oa::Matrix& weight = saved(1);
		outDIn[0] = oa::FnMatrix::gatherBwd(
			indices,
			inDOut,
			static_cast<oa::I32>(weight.size(0)),
			static_cast<oa::I32>(weight.size(1)));
	}
};

class GradGatherLastDim final : public oa::GradNode {
public:
	void backward(const oa::Matrix& inDOut, oa::Vec<oa::Matrix>& outDIn) override {
		if (outDIn.size() > 0) {
			outDIn[0] = oa::FnMatrix::gatherLastDimBwd(
				inDOut,
				saved(1),
				static_cast<oa::I32>(saved(0).size(1)));
		}
	}
};

class GradConcat final : public oa::GradNode {
public:
	oa::I32 dim_ = 0;
	oa::Vec<oa::I64> sizes_;

	void backward(const oa::Matrix& inDOut, oa::Vec<oa::Matrix>& outDIn) override {
		oa::I64 offset = 0;
		for (oa::I32 index = 0;
			index < static_cast<oa::I32>(sizes_.size());
			++index)
		{
			const oa::Usize slot = static_cast<oa::Usize>(index);
			if (index < static_cast<oa::I32>(outDIn.size())
				and sizes_[slot] > 0)
			{
				outDIn[slot] = oa::FnMatrix::slice(
					inDOut, dim_, offset, offset + sizes_[slot]);
			}
			offset += sizes_[slot];
		}
	}
};

class GradSlice final : public oa::GradNode {
public:
	oa::I32 dim_ = 0;
	oa::I64 start_ = 0;
	oa::I64 end_ = 0;

	void backward(const oa::Matrix& inDOut, oa::Vec<oa::Matrix>& outDIn) override {
		const oa::MatrixShape inputShape = saved(0).getShape();
		oa::MatrixShape sliceShape = inputShape;
		if (dim_ >= 0 and dim_ < sliceShape.rank) {
			sliceShape[dim_] = end_ - start_;
		}
		oa::Matrix sliceGradient = inDOut;
		if (sliceGradient.getShape() != sliceShape
			and sliceGradient.numElements() == sliceShape.numElements())
		{
			sliceGradient = sliceGradient.reshape(sliceShape);
		}
		if (outDIn.size() > 0) {
			outDIn[0] = oa::FnMatrix::sliceBwd(
				inputShape, dim_, start_, end_, sliceGradient);
		}
	}
};

} // namespace oa
