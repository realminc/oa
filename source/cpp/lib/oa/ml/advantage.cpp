#include <oa/ml/advantage.h>

#include <oa/core/bufferAccess.h>
#include <oa/core/fnMatrix.h>
#include <oa/core/log.h>
#include <oa/core/op.h>
#include <oa/ml/autograd.h>
#include <oa/runtime/executionSession.h>
#include <oa/core/std/limits.h>
#include <oa/core/std/scalarMath.h>

namespace {

bool isRolloutF32(const oa::Matrix& inMatrix, const oa::MatrixShape& inShape) {
	return !inMatrix.isEmpty()
		&& inMatrix.getDtype() == oa::ScalarType::Float32
		&& inMatrix.getShape() == inShape;
}

oa::Status validateGaeInputs(
	const oa::Matrix& inReward,
	const oa::Matrix& inValue,
	const oa::Matrix& inNextValue,
	const oa::Matrix& inTerminated,
	const oa::Matrix& inTruncated,
	const oa::GaeConfig& inConfig) {
	const oa::MatrixShape shape = inReward.getShape();
	const bool configValid = oa::isFinite(inConfig.gamma)
		&& oa::isFinite(inConfig.lambda)
		&& inConfig.gamma >= 0.0F && inConfig.gamma <= 1.0F
		&& inConfig.lambda >= 0.0F && inConfig.lambda <= 1.0F;
	const bool shapeValid = shape.rank == 2 && shape[0] > 0 && shape[1] > 0
		&& shape[0] <= static_cast<oa::I64>(oa::Limits<oa::U32>::max())
		&& shape[1] <= static_cast<oa::I64>(oa::Limits<oa::U32>::max());
	const bool valuesValid = isRolloutF32(inReward, shape)
		&& isRolloutF32(inValue, shape)
		&& isRolloutF32(inNextValue, shape);
	const bool masksValid = inTerminated.getShape() == shape
		&& inTruncated.getShape() == shape
		&& inTerminated.getDtype() == oa::ScalarType::UInt8
		&& inTruncated.getDtype() == oa::ScalarType::UInt8;
	if (!configValid || !shapeValid || !valuesValid || !masksValid) {
		return oa::Status::invalidArgument("oa::FnAdvantage::gae expects matching non-empty [T,E] FP32 reward/value/next-value matrices, UInt8 boundary masks, and gamma/lambda in [0,1]");
	}
	return oa::Status::ok();
}

} // namespace

oa::Matrix oa::FnAdvantage::normalize(const oa::Matrix& inAdvantage,	oa::F32 inEpsilon) {
	if (inAdvantage.isEmpty()
		|| inAdvantage.getDtype() != oa::ScalarType::Float32
		|| !oa::isFinite(inEpsilon) || inEpsilon <= 0.0F) {
		OaLogError(oa::LogComponent::Ml,
			"oa::FnAdvantage::normalize expects non-empty FP32 input and positive finite epsilon");
		return {};
	}
	auto& context = oa::ExecutionSession::getActive();
	oa::OpLoweringScope lowering(context);
	const oa::Matrix mean = oa::FnMatrix::mean(inAdvantage);
	const oa::Matrix centered = oa::FnMatrix::sub(inAdvantage, mean);
	// Do not express x^2 as pow(x, 2): shader pow is undefined for negative
	// bases on several vulkan backends, and centered advantages are commonly
	// negative. Multiplication is both cheaper and well-defined.
	const oa::Matrix variance = oa::FnMatrix::mean(oa::FnMatrix::mul(centered, centered));
	const oa::Matrix denominator = oa::FnMatrix::sqrt(oa::FnMatrix::addScalar(variance, inEpsilon));
	oa::Matrix result = oa::FnMatrix::div(centered, denominator);
	auto semantic = lowering.commitWithId(
		oa::detail::opRegistry::FnAdvantage::normalize,
		{&inAdvantage}, {&result},
		{oa::OpAttribute::fromFloat("epsilon", inEpsilon)});
	if (not semantic.isOk()) return {};
	if (auto grad = result.getGradFn()) {
		if (not oa::FnAutograd::attachSemantic(
			grad, semantic.getValue()).isOk())
		{
			return {};
		}
	}
	return result;
}

oa::GaeResult oa::FnAdvantage::gae(
	const oa::Matrix& inReward,
	const oa::Matrix& inValue,
	const oa::Matrix& inNextValue,
	const oa::Matrix& inTerminated,
	const oa::Matrix& inTruncated,
	const oa::GaeConfig& inConfig) {
	const oa::Status validation = validateGaeInputs(
		inReward, inValue, inNextValue, inTerminated, inTruncated, inConfig);
	if (validation.isError()) {
		OaLogError(oa::LogComponent::Ml, "{}", validation.getMessage().cStr());
		return {};
	}
	auto& context = oa::ExecutionSession::getActive();
	oa::OpLoweringScope lowering(context);
	const oa::MatrixShape shape = inReward.getShape();
	oa::GaeResult result{
		.advantage = oa::FnMatrix::empty(shape, oa::ScalarType::Float32),
		.ret = oa::FnMatrix::empty(shape, oa::ScalarType::Float32),
	};
	const oa::Status status = gaeInto(
		inReward, inValue, inNextValue, inTerminated, inTruncated,
		result.advantage, result.ret, inConfig);
	if (status.isError()) {
		OaLogError(oa::LogComponent::Ml, "{}", status.getMessage().cStr());
		return {};
	}
	if (not lowering.commit(
		oa::detail::opRegistry::FnAdvantage::gae,
		{&inReward, &inValue, &inNextValue, &inTerminated, &inTruncated},
		{&result.advantage, &result.ret},
		{
			oa::OpAttribute::fromFloat("gamma", inConfig.gamma),
			oa::OpAttribute::fromFloat("lambda", inConfig.lambda),
		}).isOk())
	{
		return {};
	}
	return result;
}

oa::Status oa::FnAdvantage::gaeInto(
	const oa::Matrix& inReward,
	const oa::Matrix& inValue,
	const oa::Matrix& inNextValue,
	const oa::Matrix& inTerminated,
	const oa::Matrix& inTruncated,
	oa::Matrix& outAdvantage,
	oa::Matrix& outReturn,
	const oa::GaeConfig& inConfig
) {
	const oa::Status validation = validateGaeInputs(inReward, inValue, inNextValue, inTerminated, inTruncated, inConfig);
	if (validation.isError()) { return validation; }
	const oa::MatrixShape shape = inReward.getShape();
	if (!isRolloutF32(outAdvantage, shape)	|| !isRolloutF32(outReturn, shape)) {
		return oa::Status::invalidArgument("oa::FnAdvantage::gaeInto expects matching FP32 [T,E] output matrices");
	}
	auto& context = oa::ExecutionSession::getActive();
	oa::OpLoweringScope lowering(context);
	struct Push {
		oa::U32 time;
		oa::U32 environments;
		oa::F32 gamma;
		oa::F32 lambda;
	} push {
		.time = static_cast<oa::U32>(shape[0]),
		.environments = static_cast<oa::U32>(shape[1]),
		.gamma = inConfig.gamma,
		.lambda = inConfig.lambda,
	};
	oa::BufferAccess access[] = {
		oa::BufferAccess::Read, oa::BufferAccess::Read, oa::BufferAccess::Read,
		oa::BufferAccess::Read, oa::BufferAccess::Read,
		oa::BufferAccess::Write, oa::BufferAccess::Write,
	};
	context.add(
		"RlGae",
		{&inReward, &inValue, &inNextValue, &inTerminated, &inTruncated,
		 &outAdvantage, &outReturn},
		access,
		&push,
		sizeof(push),
		(push.environments + 63U) / 64U);
	return lowering.commit(
		oa::detail::opRegistry::FnAdvantage::gaeInto,
		{&inReward, &inValue, &inNextValue, &inTerminated, &inTruncated},
		{&outAdvantage, &outReturn},
		{
			oa::OpAttribute::fromFloat("gamma", inConfig.gamma),
			oa::OpAttribute::fromFloat("lambda", inConfig.lambda),
		});
}
