// ML normalization gradient nodes.
#pragma once

#include <oa/core/fnMatrix.h>
#include <oa/ml/autograd.h>
#include <oa/ml/fnMatrix.h>

namespace oa {

class GradLayerNorm final : public oa::GradNode {
public:
	oa::F32 eps_ = 1e-5F;
	void backward(const oa::Matrix& inDOut, oa::Vec<oa::Matrix>& outDIn) override {
		const oa::Matrix& x = saved(0);
		const oa::Matrix& weight = saved(1);
		const oa::Matrix& bias = saved(2);
		const oa::Matrix& out = saved(3);
		const bool needDX = outDIn.size() > 0;
		const bool needDW = outDIn.size() > 1;
		const bool needDB = outDIn.size() > 2;
		if (needDX or needDW or needDB) {
			auto grads = oa::FnMatrix::layerNormBwd(
				x, weight, bias, out, out, out, inDOut, eps_);
			if (needDX) outDIn[0] = grads.dX;
			if (needDW) outDIn[1] = grads.dWeight;
			if (needDB) outDIn[2] = grads.dBias;
		}
	}
};

class GradRmsNorm final : public oa::GradNode {
public:
	oa::F32 eps_ = 1e-5F;
	void backward(const oa::Matrix& inDOut, oa::Vec<oa::Matrix>& outDIn) override {
		const oa::Matrix& x = saved(0);
		const oa::Matrix& weight = saved(1);
		const bool needDX = outDIn.size() > 0;
		const bool needDW = outDIn.size() > 1;
		if (needDX or needDW) {
			auto grads = oa::FnMatrix::rmsNormBwd(x, weight, inDOut, eps_);
			if (needDX) outDIn[0] = grads.dX;
			if (needDW) outDIn[1] = grads.dWeight;
		}
	}
};

class GradRmsNormGated final : public oa::GradNode {
public:
	oa::F32 eps_ = 1e-5f;
	void backward(const oa::Matrix& inDOut, oa::Vec<oa::Matrix>& outDIn) override {
		const oa::Matrix& x = saved(0);
		const oa::Matrix& w = saved(1);
		const oa::Matrix& bias = saved(2);
		const oa::Matrix& z = saved(3);
		auto g = oa::FnMatrix::rmsNormGatedBwd(x, w, bias, z, inDOut, eps_);
		if (outDIn.size() > 0) outDIn[0] = g.dX;
		if (outDIn.size() > 1) outDIn[1] = g.dWeight;
		if (outDIn.size() > 2) outDIn[2] = g.dBias;
		if (outDIn.size() > 3) outDIn[3] = g.dZ;
	}
};

} // namespace oa
