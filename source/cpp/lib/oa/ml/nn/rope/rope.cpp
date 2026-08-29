// oa::Rope — Rotary position Embedding (RoPE)
//
// Dispatches RopeApply/RopeApplyBwd compute shaders.

#include <oa/core/fnMatrix.h>
#include <oa/core/validation.h>
#include <oa/ml/autograd/matrix/autogradRope.h>
#include <oa/ml/nn/rope/rope.h>
#include <oa/runtime/executionSession.h>

#include <oa/core/std/assert.h>

static oa::U32 divCeil(oa::U32 inA, oa::U32 inB) { return (inA + inB - 1) / inB; }

oa::Rope::Rope(oa::I32 inNumHeads, oa::I32 inHeadDim, oa::F32 inThetaBase)
	: numHeads_(inNumHeads), headDim_(inHeadDim), thetaBase_(inThetaBase) {}

oa::Matrix oa::Rope::forward(const oa::Matrix& inInput) {
	oa::I64 T = inInput.size(0);
	oa::I64 D = inInput.size(1);
	oa::I64 expectedD = static_cast<oa::I64>(numHeads_) * headDim_;
	OA_REQUIRE_MSG(D == expectedD, "oa::Rope: input dim must match num_heads * head_dim");
	(void)expectedD;

	auto& ctx = oa::ExecutionSession::getActive();
	oa::Matrix out = oa::FnMatrix::empty(inInput.getShape(), inInput.getDtype());

	oa::U32 halfDim = static_cast<oa::U32>(headDim_ / 2);
	oa::U32 total = static_cast<oa::U32>(T) * static_cast<oa::U32>(numHeads_) * halfDim;

	struct Push {
		oa::U32 T;
		oa::U32 num_heads;
		oa::U32 head_dim;
		oa::F32 theta_base;
		oa::U32 pos_offset;
	} push{
		static_cast<oa::U32>(T),
		static_cast<oa::U32>(numHeads_),
		static_cast<oa::U32>(headDim_),
		thetaBase_,
		0u
	};

	oa::BufferAccess access[] = {oa::BufferAccess::Read, oa::BufferAccess::Write};
	ctx.add( "RopeApply", {&inInput, &out}, access, &push, sizeof(push), divCeil(total, 256));

	if (oa::FnAutograd::isEnabled() and inInput.requiresGrad()) {
		auto gradFn = oa::makeShared<oa::GradRoPE>();
		gradFn->saveForBackward(inInput);
		gradFn->setGraphInputs(oa::Vector<oa::Matrix>{inInput});
		gradFn->sequenceNr_ = oa::FnAutograd::nextSeq();
		gradFn->numHeads_ = numHeads_;
		gradFn->headDim_ = headDim_;
		gradFn->thetaBase_ = thetaBase_;
		gradFn->posOffset_ = 0;
		gradFn->outputShape_ = out.getShape();
		out.mutAutograd().gradFn = gradFn;
	}

	return out;
}
