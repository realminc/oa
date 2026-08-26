#include <oa/ml/fnFlow.h>

#include <oa/core/bufferAccess.h>
#include <oa/core/assert.h>
#include <oa/core/fnMatrix.h>
#include <oa/core/op.h>
#include <oa/ml/autograd.h>
#include <oa/runtime/executionSession.h>

namespace {

struct MaskedMsePush {
	oa::U32 count = 0;
	oa::U32 rank = 0;
	oa::U32 predictionStrides[OA_MAX_TENSOR_DIMS]{};
	oa::U32 maskDims[OA_MAX_TENSOR_DIMS]{};
	oa::U32 maskStrides[OA_MAX_TENSOR_DIMS]{};
};

MaskedMsePush makeMaskedMsePush(
	const oa::Matrix& inPrediction,
	const oa::Matrix& inMask) {
	MaskedMsePush push;
	push.count = static_cast<oa::U32>(inPrediction.numElements());
	push.rank = static_cast<oa::U32>(inPrediction.rank());

	oa::U32 stride = 1;
	for (oa::I32 dim = inPrediction.rank() - 1; dim >= 0; --dim) {
		push.predictionStrides[dim] = stride;
		stride *= static_cast<oa::U32>(inPrediction.size(dim));
	}
	const oa::I32 rankOffset = inPrediction.rank() - inMask.rank();
	for (oa::I32 dim = 0; dim < inPrediction.rank(); ++dim) {
		push.maskDims[dim] = dim < rankOffset
			? 1U : static_cast<oa::U32>(inMask.size(dim - rankOffset));
	}
	stride = 1;
	for (oa::I32 dim = inPrediction.rank() - 1; dim >= 0; --dim) {
		push.maskStrides[dim] = stride;
		stride *= push.maskDims[dim];
	}
	return push;
}

oa::Matrix maskedMseBackward(
	const oa::Matrix& inPrediction,
	const oa::Matrix& inTarget,
	const oa::Matrix& inMask,
	const oa::Matrix& inDenominator,
	const oa::Matrix& inUpstream) {
	auto gradient = oa::FnMatrix::empty(
		inPrediction.getShape(), inPrediction.getDtype());
	auto push = makeMaskedMsePush(inPrediction, inMask);
	oa::BufferAccess access[] = {
		oa::BufferAccess::Read,
		oa::BufferAccess::Read,
		oa::BufferAccess::Read,
		oa::BufferAccess::Read,
		oa::BufferAccess::Read,
		oa::BufferAccess::Write,
	};
	oa::ExecutionSession::getActive().add( "MaskedMseBwd",
		{&inPrediction, &inTarget, &inMask, &inDenominator,
			&inUpstream, &gradient},
		access, &push, sizeof(push), (push.count + 255U) / 256U);
	return gradient;
}

class GradFlowMaskedMse final : public oa::GradNode {
public:
	void backward(
		const oa::Matrix& inUpstream,
		oa::Vec<oa::Matrix>& outInputGrads) override {
		if (outInputGrads.empty()) return;
		outInputGrads[0] = maskedMseBackward(
			saved(0), saved(1), saved(2), saved(3), inUpstream);
	}
};

oa::Matrix broadcastTime(const oa::Matrix& inTime, const oa::Matrix& inState) {
	OA_REQUIRE_MSG(!inTime.isEmpty() && !inState.isEmpty(),
		"Flow matching requires non-empty state and time matrices");

	oa::Matrix time = inTime;
	if (inTime.rank() == 1 && inState.rank() > 1 &&
		(inTime.numElements() == 1 || inTime.size(0) == inState.size(0))) {
		oa::MatrixShape shape;
		shape.rank = inState.rank();
		for (oa::I32 dim = 0; dim < shape.rank; ++dim) shape[dim] = 1;
		if (inTime.numElements() != 1) shape[0] = inState.size(0);
		time = inTime.reshape(shape);
	}

	auto broadcast = time.getShape().broadcast(inState.getShape());
	OA_REQUIRE_MSG(broadcast.isOk() && *broadcast == inState.getShape(),
		"Flow time must be scalar, [B], or broadcastable to the state");
	return time;
}

} // namespace

oa::FlowMatchBatch oa::FnFlow::linearMatch(
	const oa::Matrix& inClean,
	const oa::Matrix& inNoise,
	const oa::Matrix& inTime) {
	OA_REQUIRE_MSG(inClean.getShape() == inNoise.getShape()
		&& inClean.getDtype() == inNoise.getDtype(),
		"Flow clean/noise matrices require identical shape and dtype");
	auto& context = oa::ExecutionSession::getActive();
	oa::OpLoweringScope lowering(context);
	auto time = broadcastTime(inTime, inClean);
	if (time.getDtype() != inClean.getDtype()) {
		time = oa::FnMatrix::cast(time, inClean.getDtype());
	}
	auto velocity = inNoise - inClean;
	auto result = oa::FlowMatchBatch{
		.state = inClean + (velocity * time),
		.velocity = velocity,
	};
	auto semantic = lowering.commitWithId(
		oa::detail::opRegistry::FnFlow::linearMatch,
		{&inClean, &inNoise, &inTime},
		{&result.state, &result.velocity});
	if (not semantic.isOk()) return {};
	if (auto grad = result.state.getGradFn()) {
		if (not oa::FnAutograd::attachSemantic(
			grad, semantic.getValue(), 0U).isOk())
		{
			return {};
		}
	}
	if (auto grad = result.velocity.getGradFn()) {
		if (not oa::FnAutograd::attachSemantic(
			grad, semantic.getValue(), 1U).isOk())
		{
			return {};
		}
	}
	return result;
}

oa::Matrix oa::FnFlow::eulerStep(
	const oa::Matrix& inState,
	const oa::Matrix& inVelocity,
	oa::F32 inDeltaTime) {
	OA_REQUIRE_MSG(inState.getShape() == inVelocity.getShape()
		&& inState.getDtype() == inVelocity.getDtype(),
		"Flow Euler state/velocity require identical shape and dtype");
	auto& context = oa::ExecutionSession::getActive();
	oa::OpLoweringScope lowering(context);
	auto result = inState + (inVelocity * inDeltaTime);
	auto semantic = lowering.commitWithId(
		oa::detail::opRegistry::FnFlow::eulerStep,
		{&inState, &inVelocity}, {&result},
		{oa::OpAttribute::fromFloat("deltaTime", inDeltaTime)});
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

oa::Matrix oa::FnFlow::maskedMse(
	const oa::Matrix& inPrediction,
	const oa::Matrix& inTarget,
	const oa::Matrix& inMask) {
	OA_REQUIRE_MSG(!inPrediction.isEmpty() && !inTarget.isEmpty()
		&& !inMask.isEmpty(),
		"Flow masked MSE requires non-empty matrices");
	OA_REQUIRE_MSG(inPrediction.getShape() == inTarget.getShape()
		&& inPrediction.getDtype() == inTarget.getDtype(),
		"Flow masked MSE prediction/target require identical shape and dtype");
	auto broadcast = inMask.getShape().broadcast(inPrediction.getShape());
	OA_REQUIRE_MSG(broadcast.isOk() && *broadcast == inPrediction.getShape(),
		"Flow masked MSE mask must be broadcastable to prediction");

	auto& context = oa::ExecutionSession::getActive();
	oa::OpLoweringScope lowering(context);
	auto mask = inMask;
	if (mask.getDtype() != inPrediction.getDtype()) {
		mask = oa::FnMatrix::cast(mask, inPrediction.getDtype());
	}
	auto loss = oa::FnMatrix::empty(
		oa::MatrixShape{1}, inPrediction.getDtype());
	auto denominator = oa::FnMatrix::empty(
		oa::MatrixShape{1}, inPrediction.getDtype());
	auto push = makeMaskedMsePush(inPrediction, mask);
	oa::BufferAccess access[] = {
		oa::BufferAccess::Read,
		oa::BufferAccess::Read,
		oa::BufferAccess::Read,
		oa::BufferAccess::Write,
		oa::BufferAccess::Write,
	};
	context.add( "MaskedMse",
		{&inPrediction, &inTarget, &mask, &loss, &denominator},
		access, &push, sizeof(push), 1);
	auto semantic = lowering.commitWithId(
		oa::detail::opRegistry::FnFlow::maskedMse,
		{&inPrediction, &inTarget, &inMask}, {&loss});
	if (not semantic.isOk()) return {};

	if (oa::FnAutograd::isEnabled() && inPrediction.requiresGrad()) {
		auto gradFn = oa::makeShared<GradFlowMaskedMse>();
		gradFn->saveForBackward(
			inPrediction, inTarget, mask, denominator);
		gradFn->setGraphInputs(oa::Vec<oa::Matrix>{inPrediction});
		gradFn->sequenceNr_ = oa::FnAutograd::nextSeq();
		gradFn->outputShape_ = loss.getShape();
		if (not oa::FnAutograd::attachSemantic(
			gradFn, semantic.getValue()).isOk())
		{
			return {};
		}
		loss.mutAutograd().gradFn = gradFn;
	}
	return loss;
}
