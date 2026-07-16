// OaFnOptim — Optimizer Operations
// Stateless parameter update functions. Records into OaContext (clean api lvl1).

#include <Oa/Ml/FnOptim.h>
#include <Oa/Ml/MuonRef.h>
#include <Oa/Core/FnMatrix.h>

#include <cstring>
#include <Oa/Core/Matrix.h>
#include <Oa/Core/BufferAccess.h>
#include <Oa/Runtime/Context.h>
#include <Oa/Core/Log.h>
#include <Oa/Core/EnvFlag.h>

#include <algorithm>
#include <cmath>

namespace OaFnOptim {

namespace {

constexpr OaF32 kNsA = 3.4445f;
constexpr OaF32 kNsB = -4.7750f;
constexpr OaF32 kNsC = 2.0315f;

void MuonNewtonSchulz5Gpu(
	OaMatrix& InOutZ,
	OaU32 InOperRows,
	OaU32 InOperCols,
	OaI32 InNS5Steps)
{
	for (OaI32 step = 0; step < InNS5Steps; ++step) {
		OaMatrix zT = OaFnMatrix::Transpose(InOutZ, 0, 1);
		OaMatrix a = OaFnMatrix::MatMulNt(InOutZ, zT);
		OaMatrix aa = OaFnMatrix::MatMulNt(a, a);
		OaMatrix b = OaFnMatrix::Add(
			OaFnMatrix::Scale(a, kNsB),
			OaFnMatrix::Scale(aa, kNsC));
		OaMatrix bz = OaFnMatrix::MatMulNt(b, InOutZ);
		InOutZ = OaFnMatrix::Add(OaFnMatrix::Scale(InOutZ, kNsA), bz);
		(void)InOperRows;
		(void)InOperCols;
	}
}

void MuonMatrixStepGpu(
	OaMatrix& InOutParam,
	OaMatrix& InOutMomentum,
	const OaMatrix& InGrad,
	OaU32 InRows,
	OaU32 InCols,
	OaF32 InLr,
	OaF32 InBeta,
	OaF32 InWeightDecay,
	OaF32 InEps,
	OaI32 InNS5Iterations)
{
	auto& ctx = OaContext::GetDefault();
	const OaU32 count = InRows * InCols;
	const OaU32 groups = (count + 255) / 256;

	OaMatrix update = OaFnMatrix::Zeros(InOutParam.GetShape());
	{
		struct MuonNesterovPush {
			OaU32 Count;
			OaF32 Beta;
		};
		MuonNesterovPush push{count, InBeta};
		OaBufferAccess access[] = {
			OaBufferAccess::Read,
			OaBufferAccess::ReadWrite,
			OaBufferAccess::Write
		};
		ctx.Add("MuonNesterov", {&InGrad, &InOutMomentum, &update}, access, &push, sizeof(push), groups);
	}

	const bool transposed = InRows > InCols;
	const OaU32 operRows = transposed ? InCols : InRows;
	const OaU32 operCols = transposed ? InRows : InCols;

	OaMatrix z = transposed ? OaFnMatrix::Transpose(update, 0, 1) : update;

	OaMatrix normScalar = OaFnMatrix::Zeros(OaMatrixShape{1});
	{
		struct MuonNormalizePush {
			OaU32 Rows;
			OaU32 Cols;
			OaF32 Eps;
		};
		MuonNormalizePush push{operRows, operCols, InEps};
		OaBufferAccess access[] = {
			OaBufferAccess::Read,
			OaBufferAccess::ReadWrite,
			OaBufferAccess::Write
		};
		ctx.Add("MuonNormalize", {&z, &z, &normScalar}, access, &push, sizeof(push), 1);
	}

	if (InNS5Iterations > 0) {
		MuonNewtonSchulz5Gpu(z, operRows, operCols, InNS5Iterations);
	}

	OaMatrix ortho = transposed ? OaFnMatrix::Transpose(z, 0, 1) : z;

	const OaF32 moonshotScale = OaMuonRef::MoonshotScale(InRows, InCols);
	{
		struct MuonApplyPush {
			OaU32 Count;
			OaF32 Lr;
			OaF32 WeightDecay;
			OaF32 MoonshotScale;
		};
		MuonApplyPush push{count, InLr, InWeightDecay, moonshotScale};
		OaBufferAccess access[] = {
			OaBufferAccess::ReadWrite,
			OaBufferAccess::Read
		};
		ctx.Add("MuonApply", {&InOutParam, &ortho}, access, &push, sizeof(push), groups);
	}
}

} // namespace

void AdamWAdvanceGraphState(OaMatrix& InOutState) {
	auto& ctx = OaContext::GetDefault();
	OaBufferAccess access[] = {OaBufferAccess::ReadWrite};
	ctx.Add("AdamwGraphAdvance", {&InOutState}, access, nullptr, 0, 1);
}

void AdamWStep(
	OaMatrix& InOutParam,
	OaMatrix& InOutM,
	OaMatrix& InOutV,
	const OaMatrix& InGrad,
	OaF32 InLr,
	OaF32 InBeta1,
	OaF32 InBeta2,
	OaF32 InEps,
	OaF32 InWeightDecay,
	OaI32 InStep
) {
	auto& ctx = OaContext::GetDefault();

	const OaU32 count = static_cast<OaU32>(InOutParam.NumElements());
	struct AdamWPush {
		OaU32 Count;
		OaF32 Lr;
		OaF32 Beta1;
		OaF32 Beta2;
		OaF32 Eps;
		OaF32 WeightDecay;
		OaU32 Step;
	};
	AdamWPush push{count, InLr, InBeta1, InBeta2, InEps, InWeightDecay,
		static_cast<OaU32>(InStep)};

	OaBufferAccess access[] = {
		OaBufferAccess::ReadWrite,
		OaBufferAccess::Read,
		OaBufferAccess::ReadWrite,
		OaBufferAccess::ReadWrite
	};
	const OaU32 groups = (count + 255) / 256;
	ctx.Add("Adamw", {&InOutParam, &InGrad, &InOutM, &InOutV}, access, &push, sizeof(push), groups);
}

void AdamWStepGraph(
	OaMatrix& InOutParam,
	OaMatrix& InOutM,
	OaMatrix& InOutV,
	const OaMatrix& InGrad,
	const OaMatrix& InState
) {
	auto& ctx = OaContext::GetDefault();
	const OaU32 count = static_cast<OaU32>(InOutParam.NumElements());
	struct Push { OaU32 Count; } push{count};
	OaBufferAccess access[] = {
		OaBufferAccess::ReadWrite,
		OaBufferAccess::Read,
		OaBufferAccess::ReadWrite,
		OaBufferAccess::ReadWrite,
		OaBufferAccess::Read,
	};
	ctx.Add("AdamwGraph", {&InOutParam, &InGrad, &InOutM, &InOutV, &InState},
		access, &push, sizeof(push), (count + 255) / 256);
}

void AdamWStepMany(
	OaSpan<const OaAdamWParamSet> InParams,
	OaF32 InLr,
	OaF32 InBeta1,
	OaF32 InBeta2,
	OaF32 InEps,
	OaF32 InWeightDecay,
	OaI32 InStep
) {
	if (InParams.size() != 4) {
		for (const auto& p : InParams) {
			AdamWStep(*p.Param, *p.M, *p.V, *p.Grad,
				InLr, InBeta1, InBeta2, InEps, InWeightDecay, InStep);
		}
		return;
	}

	auto& ctx = OaContext::GetDefault();

	const OaU32 count0 = static_cast<OaU32>(InParams[0].Param->NumElements());
	const OaU32 count1 = static_cast<OaU32>(InParams[1].Param->NumElements());
	const OaU32 count2 = static_cast<OaU32>(InParams[2].Param->NumElements());
	const OaU32 count3 = static_cast<OaU32>(InParams[3].Param->NumElements());
	const OaU32 maxCount = std::max(std::max(count0, count1), std::max(count2, count3));

	struct AdamWMany4Push {
		OaU32 Count0;
		OaU32 Count1;
		OaU32 Count2;
		OaU32 Count3;
		OaF32 Lr;
		OaF32 Beta1;
		OaF32 Beta2;
		OaF32 Eps;
		OaF32 WeightDecay;
		OaU32 Step;
	};
	AdamWMany4Push push{
		count0, count1, count2, count3,
		InLr, InBeta1, InBeta2, InEps, InWeightDecay,
		static_cast<OaU32>(InStep)
	};

	OaBufferAccess access[] = {
		OaBufferAccess::ReadWrite, OaBufferAccess::Read, OaBufferAccess::ReadWrite, OaBufferAccess::ReadWrite,
		OaBufferAccess::ReadWrite, OaBufferAccess::Read, OaBufferAccess::ReadWrite, OaBufferAccess::ReadWrite,
		OaBufferAccess::ReadWrite, OaBufferAccess::Read, OaBufferAccess::ReadWrite, OaBufferAccess::ReadWrite,
		OaBufferAccess::ReadWrite, OaBufferAccess::Read, OaBufferAccess::ReadWrite, OaBufferAccess::ReadWrite,
	};
	const OaU32 groups = (maxCount + 255) / 256;
	ctx.Add("AdamwMany4", {
		InParams[0].Param, InParams[0].Grad, InParams[0].M, InParams[0].V,
		InParams[1].Param, InParams[1].Grad, InParams[1].M, InParams[1].V,
		InParams[2].Param, InParams[2].Grad, InParams[2].M, InParams[2].V,
		InParams[3].Param, InParams[3].Grad, InParams[3].M, InParams[3].V,
	}, access, &push, sizeof(push), groups);
}

void AdamWStepManyGraph(
	OaSpan<const OaAdamWParamSet> InParams,
	const OaMatrix& InState
) {
	if (InParams.size() != 4) {
		for (const auto& p : InParams) {
			AdamWStepGraph(*p.Param, *p.M, *p.V, *p.Grad, InState);
		}
		return;
	}

	auto& ctx = OaContext::GetDefault();
	const OaU32 count0 = static_cast<OaU32>(InParams[0].Param->NumElements());
	const OaU32 count1 = static_cast<OaU32>(InParams[1].Param->NumElements());
	const OaU32 count2 = static_cast<OaU32>(InParams[2].Param->NumElements());
	const OaU32 count3 = static_cast<OaU32>(InParams[3].Param->NumElements());
	const OaU32 maxCount = std::max(std::max(count0, count1), std::max(count2, count3));
	struct Push {
		OaU32 Count0, Count1, Count2, Count3;
	} push{count0, count1, count2, count3};
	OaBufferAccess access[] = {
		OaBufferAccess::ReadWrite, OaBufferAccess::Read, OaBufferAccess::ReadWrite, OaBufferAccess::ReadWrite,
		OaBufferAccess::ReadWrite, OaBufferAccess::Read, OaBufferAccess::ReadWrite, OaBufferAccess::ReadWrite,
		OaBufferAccess::ReadWrite, OaBufferAccess::Read, OaBufferAccess::ReadWrite, OaBufferAccess::ReadWrite,
		OaBufferAccess::ReadWrite, OaBufferAccess::Read, OaBufferAccess::ReadWrite, OaBufferAccess::ReadWrite,
		OaBufferAccess::Read,
	};
	ctx.Add("AdamwMany4Graph", {
		InParams[0].Param, InParams[0].Grad, InParams[0].M, InParams[0].V,
		InParams[1].Param, InParams[1].Grad, InParams[1].M, InParams[1].V,
		InParams[2].Param, InParams[2].Grad, InParams[2].M, InParams[2].V,
		InParams[3].Param, InParams[3].Grad, InParams[3].M, InParams[3].V,
		&InState,
	}, access, &push, sizeof(push), (maxCount + 255) / 256);
}

void SgdStep(
	OaMatrix& InOutParam,
	OaMatrix& InOutMomentum,
	const OaMatrix& InGrad,
	OaF32 InLr,
	OaF32 InMomentum,
	OaF32 InWeightDecay
) {
	auto& ctx = OaContext::GetDefault();
	const OaU32 count = static_cast<OaU32>(InOutParam.NumElements());
	const OaU32 groups = (count + 255) / 256;

	if (InMomentum > 0.0F && InOutMomentum.HasStorage()) {
		struct Push { OaU32 count; OaF32 lr; OaF32 momentum_coef; OaF32 weight_decay; }
			push{count, InLr, InMomentum, InWeightDecay};
		OaBufferAccess access[] = {
			OaBufferAccess::ReadWrite,
			OaBufferAccess::Read,
			OaBufferAccess::ReadWrite
		};
		ctx.Add("SgdMomentum", {&InOutParam, &InGrad, &InOutMomentum}, access, &push, sizeof(push), groups);
		return;
	}

	struct Push { OaU32 count; OaF32 lr; OaF32 weight_decay; }
		push{count, InLr, InWeightDecay};
	OaBufferAccess access[] = {
		OaBufferAccess::ReadWrite,
		OaBufferAccess::Read
	};
	ctx.Add("Sgd", {&InOutParam, &InGrad}, access, &push, sizeof(push), groups);
}

void MuonStep(
	OaMatrix& InOutParam,
	OaMatrix& InOutMomentum,
	const OaMatrix& InGrad,
	OaF32 InLr,
	OaF32 InBeta,
	OaF32 InWeightDecay,
	OaF32 InEps,
	OaI32 InNS5Iterations
) {
	auto& ctx = OaContext::GetDefault();
	const OaU32 count = static_cast<OaU32>(InOutParam.NumElements());
	const OaU32 groups = (count + 255) / 256;

	const bool is2DMatrix = (InOutParam.Shape_.Rank == 2);
	const OaU32 rows = is2DMatrix ? static_cast<OaU32>(InOutParam.Shape_[0]) : 0U;
	const OaU32 cols = is2DMatrix ? static_cast<OaU32>(InOutParam.Shape_[1]) : 0U;

	if (is2DMatrix && InNS5Iterations > 0) {
		MuonMatrixStepGpu(
			InOutParam, InOutMomentum, InGrad,
			rows, cols, InLr, InBeta, InWeightDecay, InEps, InNS5Iterations);
		return;
	}

	// 1D / vector fallback: fused GPU kernel (Nesterov momentum + decoupled WD apply).
	struct MuonVectorPush {
		OaU32 Count;
		OaF32 Lr;
		OaF32 Beta;
		OaF32 WeightDecay;
	};
	MuonVectorPush push{count, InLr, InBeta, InWeightDecay};
	OaBufferAccess access[] = {
		OaBufferAccess::ReadWrite,
		OaBufferAccess::Read,
		OaBufferAccess::ReadWrite
	};
	ctx.Add("MuonVector", {&InOutParam, &InGrad, &InOutMomentum}, access, &push, sizeof(push), groups);
}

void ClipGradNorm(
	OaSpan<OaMatrix*> InGrads,
	OaF32 InMaxNorm,
	OaMatrix& InOutParams,
	OaMatrix& InOutPartials
) {
	auto& ctx = OaContext::GetDefault();

	// Collect non-empty grad buffers (skip params with no grad yet).
	OaMatrix* grads[16];
	OaI32 counts[16];
	OaI32 n = 0;
	for (OaUsize i = 0; i < InGrads.size() && n < 16; ++i) {
		OaMatrix* g = InGrads[i];
		if (g && !g->IsEmpty()) {
			grads[n]  = g;
			counts[n] = static_cast<OaI32>(g->NumElements());
			++n;
		}
	}
	if (n == 0) return;

	// Upload params buffer: [n_tensors, count0..count15, max_norm_bits] as uint32.
	// max_norm is carried in slot 17 (float bit-pattern) rather than as a trailing
	// push payload: the runtime packs only (2 + n) prepended buffer indices, so a
	// payload after the shader's fixed grad_idx[16] field lands at the wrong offset
	// for n < 16 (it read zero → scale 0 → all grads zeroed → training frozen).
	OaI32 paramData[18] = {};
	paramData[0] = n;
	for (OaI32 i = 0; i < n; ++i) paramData[i + 1] = counts[i];
	std::memcpy(&paramData[17], &InMaxNorm, sizeof(OaF32));  // float bits → slot 17
	InOutParams = OaFnMatrix::FromInt32(
		OaSpan<const OaI32>(paramData, 18),
		OaMatrixShape{18}, OaScalarType::Int32);

	// Build VkBuffer arrays — ctx.Add OaVkBuffer-span overload.
	OaVkBuffer vkBufs[18];
	OaBufferAccess accessBufs[18];
	vkBufs[0] = InOutParams.GetVkBuffer();
	vkBufs[1] = InOutPartials.GetVkBuffer();
	for (OaI32 i = 0; i < n; ++i) vkBufs[2 + i] = grads[i]->GetVkBuffer();
	const OaI32 total = 2 + n;

	// ── Pass 1: ClipGradNormReduce ──────────────────────────────────────────
	accessBufs[0] = OaBufferAccess::Read;    // params
	accessBufs[1] = OaBufferAccess::Write;   // partials
	for (OaI32 i = 0; i < n; ++i) accessBufs[2 + i] = OaBufferAccess::Read;

	ctx.Add("ClipGradNormReduce",
		OaSpan<OaVkBuffer>(vkBufs, static_cast<OaUsize>(total)),
		OaSpan<OaBufferAccess>(accessBufs, static_cast<OaUsize>(total)),
		nullptr, 0,
		static_cast<OaU32>(n));

	// ── Pass 2: ClipGradNormScale ────────────────────────────────────────────
	accessBufs[0] = OaBufferAccess::Read;    // params
	accessBufs[1] = OaBufferAccess::Read;    // partials
	for (OaI32 i = 0; i < n; ++i) accessBufs[2 + i] = OaBufferAccess::ReadWrite;

	ctx.Add("ClipGradNormScale",
		OaSpan<OaVkBuffer>(vkBufs, static_cast<OaUsize>(total)),
		OaSpan<OaBufferAccess>(accessBufs, static_cast<OaUsize>(total)),
		nullptr, 0,   // max_norm now carried in params[17], not a push payload
		static_cast<OaU32>(n));
}

} // namespace OaFnOptim
