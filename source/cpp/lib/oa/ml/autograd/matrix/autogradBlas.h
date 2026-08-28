// ML BLAS gradient nodes.
#pragma once

#include <oa/core/fnMatrix.h>
#include <oa/ml/autograd.h>
#include <oa/ml/fnMatrix.h>

namespace oa {

class GradLinear final : public oa::GradNode {
public:
	void backward(const oa::Matrix& inDOut, oa::Vector<oa::Matrix>& outDIn) override {
		const oa::Matrix& x      = saved(0);
		const oa::Matrix& weight = saved(1);
		const bool needDX = outDIn.size() > 0;
		const bool needDW = outDIn.size() > 1;
		const bool needDB = outDIn.size() > 2;
		if (needDX) outDIn[0] = oa::FnMatrix::linearDataBwd(inDOut, weight);
		if (needDW or needDB) {
			auto gwb = oa::FnMatrix::linearWeightBiasBwd(x, inDOut);
			if (needDW) outDIn[1] = gwb.gradWeight;
			if (needDB) outDIn[2] = gwb.gradBias;
		}
	}
};

class GradLinearRelu final : public oa::GradNode {
public:
	void backward(const oa::Matrix& inDOut, oa::Vector<oa::Matrix>& outDIn) override {
		const oa::Matrix& x      = saved(0);
		const oa::Matrix& weight = saved(1);
		const oa::Matrix& act    = saved(2);
		const bool needDX = outDIn.size() > 0;
		const bool needDW = outDIn.size() > 1;
		const bool needDB = outDIn.size() > 2;
		auto dZ = oa::FnMatrix::reluBwd(act, inDOut);
		if (needDX) outDIn[0] = oa::FnMatrix::linearDataBwd(dZ, weight);
		if (needDW or needDB) {
			auto gwb = oa::FnMatrix::linearWeightBiasBwd(x, dZ);
			if (needDW) outDIn[1] = gwb.gradWeight;
			if (needDB) outDIn[2] = gwb.gradBias;
		}
	}
};

// LinearGelu fused backward. Unlike reLU (whose mask is recoverable from the
// forward output), GELU'(pre) is a function of the pre-activation, which the
// fused forward kernel discards. Recompute it before applying GeluBwd.
class GradLinearGelu final : public oa::GradNode {
public:
	void backward(const oa::Matrix& inDOut, oa::Vector<oa::Matrix>& outDIn) override {
		const oa::Matrix& x      = saved(0);
		const oa::Matrix& weight = saved(1);
		const oa::Matrix& bias   = saved(2);
		const bool needDX = outDIn.size() > 0;
		const bool needDW = outDIn.size() > 1;
		const bool needDB = outDIn.size() > 2;
		auto pre = oa::FnMatrix::linear(x, weight, bias);
		auto dZ  = oa::FnMatrix::geluBwd(pre, inDOut);
		if (needDX) outDIn[0] = oa::FnMatrix::linearDataBwd(dZ, weight);
		if (needDW or needDB) {
			auto gwb = oa::FnMatrix::linearWeightBiasBwd(x, dZ);
			if (needDW) outDIn[1] = gwb.gradWeight;
			if (needDB) outDIn[2] = gwb.gradBias;
		}
	}
};

// SiLU'(pre), like GELU'(pre), requires the pre-activation discarded by the
// fused forward. Recompute it before applying SiluBwd.
class GradLinearSilu final : public oa::GradNode {
public:
	void backward(const oa::Matrix& inDOut, oa::Vector<oa::Matrix>& outDIn) override {
		const oa::Matrix& x      = saved(0);
		const oa::Matrix& weight = saved(1);
		const oa::Matrix& bias   = saved(2);
		const bool needDX = outDIn.size() > 0;
		const bool needDW = outDIn.size() > 1;
		const bool needDB = outDIn.size() > 2;
		auto pre = oa::FnMatrix::linear(x, weight, bias);
		auto dZ  = oa::FnMatrix::siluBwd(pre, inDOut);
		if (needDX) outDIn[0] = oa::FnMatrix::linearDataBwd(dZ, weight);
		if (needDW or needDB) {
			auto gwb = oa::FnMatrix::linearWeightBiasBwd(x, dZ);
			if (needDW) outDIn[1] = gwb.gradWeight;
			if (needDB) outDIn[2] = gwb.gradBias;
		}
	}
};

class GradBmm final : public oa::GradNode {
public:
	void backward(const oa::Matrix& inDOut, oa::Vector<oa::Matrix>& outDIn) override {
		const oa::Matrix& a = saved(0);
		const oa::Matrix& b = saved(1);
		if (outDIn.size() > 0) {
			outDIn[0] = oa::FnMatrix::bmm(
				inDOut, oa::FnMatrix::transpose(b, 1, 2));
		}
		if (outDIn.size() > 1) {
			outDIn[1] = oa::FnMatrix::bmm(
				oa::FnMatrix::transpose(a, 1, 2), inDOut);
		}
	}
};

class GradBmmNt final : public oa::GradNode {
public:
	void backward(const oa::Matrix& inDOut, oa::Vector<oa::Matrix>& outDIn) override {
		const oa::Matrix& a = saved(0);
		const oa::Matrix& b = saved(1);
		if (outDIn.size() > 0) {
			outDIn[0] = oa::FnMatrix::bmm(inDOut, b);
		}
		if (outDIn.size() > 1) {
			outDIn[1] = oa::FnMatrix::bmm(
				oa::FnMatrix::transpose(inDOut, 1, 2), a);
		}
	}
};

} // namespace oa
