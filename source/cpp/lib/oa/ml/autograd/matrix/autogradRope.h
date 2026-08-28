#pragma once

#include <oa/core/fnMatrix.h>
#include <oa/ml/autograd.h>
#include <oa/runtime/executionSession.h>

// ─── RoPE ───────────────────────────────────────────────────────────────────

namespace oa {

class GradRoPE final : public oa::GradNode {
public:
	oa::I32 numHeads_ = 0;
	oa::I32 headDim_ = 0;
	oa::F32 thetaBase_ = 10000.0f;
	oa::U32 posOffset_ = 0;
	void backward(const oa::Matrix& inDOut, oa::Vector<oa::Matrix>& outDIn) override {
		if (outDIn.size() == 0) return;
		const oa::Matrix& x = saved(0);
		oa::I64 T = x.size(0);
		oa::U32 halfDim = static_cast<oa::U32>(headDim_ / 2);
		oa::U32 total = static_cast<oa::U32>(T) * static_cast<oa::U32>(numHeads_) * halfDim;
		oa::Matrix gradInput = oa::FnMatrix::empty(x.getShape(), x.getDtype());
		auto& ctx = oa::ExecutionSession::getActive();
		struct Push { oa::U32 T, num_heads, head_dim; oa::F32 theta_base; oa::U32 pos_offset; }
			push{ static_cast<oa::U32>(T), static_cast<oa::U32>(numHeads_), static_cast<oa::U32>(headDim_), thetaBase_, posOffset_ };
		oa::BufferAccess access[] = {oa::BufferAccess::Read, oa::BufferAccess::Write};
		ctx.add( "RopeApplyBwd", {&inDOut, &gradInput}, access, &push, sizeof(push), (total + 255u) / 256u);
		outDIn[0] = gradInput;
	}
};

} // namespace oa
