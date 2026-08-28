// Core elementwise gradient nodes.
#pragma once

#include <oa/core/autograd.h>
#include <oa/core/fnMatrix.h>

#include "autogradElemwise.gen.h"

namespace oa {

class GradDiv final : public oa::GradNode {
public:
	void backward(const oa::Matrix& inDOut, oa::Vector<oa::Matrix>& outDIn) override {
		const oa::Matrix& a = saved(0);
		const oa::Matrix& b = saved(1);
		if (outDIn.size() > 0) outDIn[0] = oa::FnMatrix::div(inDOut, b);
		if (outDIn.size() > 1) {
			auto aDivB = oa::FnMatrix::div(a, b);
			auto dOutDivB = oa::FnMatrix::div(inDOut, b);
			outDIn[1] = oa::FnMatrix::scale(
				oa::FnMatrix::mul(aDivB, dOutDivB), -1.0F);
		}
	}
};

class GradNeg final : public oa::GradNode {
public:
	void backward(const oa::Matrix& inDOut, oa::Vector<oa::Matrix>& outDIn) override {
		if (outDIn.size() > 0) {
			outDIn[0] = oa::FnMatrix::scale(inDOut, -1.0F);
		}
	}
};

enum class BcastBinOp { Add, Sub, Mul, Div };

class GradBcastBinary final : public oa::GradNode {
public:
	BcastBinOp op_ = BcastBinOp::Mul;

	static oa::Matrix sumToShape(
		const oa::Matrix& inGradient, const oa::MatrixShape& inTarget)
	{
		const oa::MatrixShape gradientShape = inGradient.getShape();
		if (gradientShape == inTarget) return inGradient;
		oa::Matrix result = inGradient;
		oa::MatrixShape currentShape = gradientShape;
		const oa::I32 leading = gradientShape.rank - inTarget.rank;
		for (oa::I32 axis = 0; axis < gradientShape.rank; ++axis) {
			const oa::I32 targetAxis = axis - leading;
			const oa::Usize axisIndex = static_cast<oa::Usize>(axis);
			const oa::I64 targetDim =
				targetAxis >= 0
					? inTarget.dims[static_cast<oa::Usize>(targetAxis)]
					: 1;
			if (targetDim == 1 and currentShape.dims[axisIndex] > 1) {
				result = oa::FnMatrix::sum(result, axis);
				currentShape.dims[axisIndex] = 1;
			}
		}
		return result.getShape() == inTarget
			? result : result.reshape(inTarget);
	}

	void backward(const oa::Matrix& inDOut, oa::Vector<oa::Matrix>& outDIn) override {
		const oa::Matrix& a = saved(0);
		const oa::Matrix& b = saved(1);
		oa::Matrix gradA;
		oa::Matrix gradB;
		switch (op_) {
			case BcastBinOp::Add:
				gradA = inDOut;
				gradB = inDOut;
				break;
			case BcastBinOp::Sub:
				gradA = inDOut;
				gradB = oa::FnMatrix::scale(inDOut, -1.0F);
				break;
			case BcastBinOp::Mul:
				gradA = oa::FnMatrix::mul(inDOut, b);
				gradB = oa::FnMatrix::mul(inDOut, a);
				break;
			case BcastBinOp::Div:
				gradA = oa::FnMatrix::div(inDOut, b);
				gradB = oa::FnMatrix::scale(
					oa::FnMatrix::div(
						oa::FnMatrix::mul(inDOut, a),
						oa::FnMatrix::mul(b, b)),
					-1.0F);
				break;
		}
		if (outDIn.size() > 0) {
			outDIn[0] = sumToShape(gradA, a.getShape());
		}
		if (outDIn.size() > 1) {
			outDIn[1] = sumToShape(gradB, b.getShape());
		}
	}
};

class GradAbs final : public oa::GradNode {
public:
	void backward(const oa::Matrix& inDOut, oa::Vector<oa::Matrix>& outDIn) override {
		if (outDIn.size() == 0) return;
		const oa::Matrix& input = saved(0);
		auto positive = oa::FnMatrix::equal(
			oa::FnMatrix::clampMin(input, 0.0F), 0.0F);
		auto negative = oa::FnMatrix::equal(
			oa::FnMatrix::clampMax(input, 0.0F), 0.0F);
		auto sign = oa::FnMatrix::sub(positive, negative);
		outDIn[0] = oa::FnMatrix::mul(inDOut, sign);
	}
};

class GradClampMax final : public oa::GradNode {
public:
	explicit GradClampMax(oa::F32 inMax) noexcept : max_(inMax) {}

	void backward(const oa::Matrix& inDOut, oa::Vector<oa::Matrix>& outDIn) override {
		if (outDIn.size() == 0) return;
		const oa::Matrix& input = saved(0);
		auto difference = oa::FnMatrix::subScalar(input, max_);
		auto mask = oa::FnMatrix::equal(
			oa::FnMatrix::clampMax(difference, 0.0F), 0.0F);
		outDIn[0] = oa::FnMatrix::mul(inDOut, mask);
	}

private:
	oa::F32 max_;
};

class GradClampMin final : public oa::GradNode {
public:
	explicit GradClampMin(oa::F32 inMin) noexcept : min_(inMin) {}

	void backward(const oa::Matrix& inDOut, oa::Vector<oa::Matrix>& outDIn) override {
		if (outDIn.size() == 0) return;
		const oa::Matrix& input = saved(0);
		auto negativeInput = oa::FnMatrix::scale(input, -1.0F);
		auto difference = oa::FnMatrix::addScalar(negativeInput, min_);
		auto mask = oa::FnMatrix::equal(
			oa::FnMatrix::clampMax(difference, 0.0F), 0.0F);
		outDIn[0] = oa::FnMatrix::mul(inDOut, mask);
	}

private:
	oa::F32 min_;
};

} // namespace oa
