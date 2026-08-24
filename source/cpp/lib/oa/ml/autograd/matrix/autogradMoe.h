#pragma once

#include <oa/core/fnMatrix.h>
#include <oa/ml/autograd.h>
#include <oa/ml/fnMatrix.h>

namespace oa {

class GradMoeRouteWeights final : public oa::GradNode {
public:
	void backward(const oa::Matrix& inDOut, oa::Vec<oa::Matrix>& outDIn) override {
		if (outDIn.size() == 0) return;
		outDIn[0] = oa::FnMatrix::moeRouteWeightsBwd(
			inDOut, saved(0), saved(1), saved(2));
	}
};
class GradGroupedGemmM final : public oa::GradNode {
public:
	void backward(const oa::Matrix& inDOut, oa::Vec<oa::Matrix>& outDIn) override {
		if (outDIn.size() < 2) return;
		auto result = oa::FnMatrix::groupedGemmMBwd(
			inDOut, saved(0), saved(1), saved(2));
		outDIn[0] = result.dInput;
		outDIn[1] = result.dWeight;
	}
};

class GradGroupedLinearM final : public oa::GradNode {
public:
	void backward(const oa::Matrix& inDOut, oa::Vec<oa::Matrix>& outDIn) override {
		if (outDIn.size() < 3) return;
		auto result = oa::FnMatrix::groupedLinearMBwd(
			inDOut, saved(0), saved(1), saved(3));
		outDIn[0] = result.dInput;
		outDIn[1] = result.dWeight;
		outDIn[2] = result.dBias;
	}
};

class GradMoeCombine final : public oa::GradNode {
public:
	void backward(const oa::Matrix& inDOut, oa::Vec<oa::Matrix>& outDIn) override {
		if (outDIn.size() < 2) return;
		auto result = oa::FnMatrix::moeCombineBwd(
			inDOut, saved(0), saved(1), saved(2), saved(3));
		outDIn[0] = result.dPacked;
		outDIn[1] = result.dRouteGate;
	}
};

class GradMoeGather final : public oa::GradNode {
public:
	void backward(const oa::Matrix& inDOut, oa::Vec<oa::Matrix>& outDIn) override {
		if (outDIn.size() == 0) return;
		const oa::Matrix& input = saved(0);
		outDIn[0] = oa::FnMatrix::moeGatherBwd(
			inDOut, saved(2), static_cast<oa::I32>(input.size(0)));
	}
};

class GradScatterAddRows final : public oa::GradNode {
public:
	void backward(const oa::Matrix& inDOut, oa::Vec<oa::Matrix>& outDIn) override {
		if (outDIn.size() > 0) outDIn[0] = oa::FnMatrix::gather(inDOut, saved(0));
	}
};

} // namespace oa
