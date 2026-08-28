// oa::FnOptim — Optimizer operations
// Stateless parameter update functions. Records into oa::ExecutionSession (clean api lvl1).

#include <oa/ml/fnOptim.h>
#include <oa/core/fnMatrix.h>

#include <oa/core/matrix.h>
#include <oa/core/matrixAccess.h>
#include <oa/core/bufferAccess.h>
#include <oa/runtime/executionSession.h>
#include <oa/runtime/allocator.h>
#include <oa/core/log.h>
#include <oa/core/envFlag.h>
#include <oa/core/std/memory.h>
#include <oa/core/std/algo.h>
#include <oa/core/std/scalarMath.h>

namespace oa {

namespace FnOptim {

namespace {

constexpr oa::F32 kNsA = 3.4445f;
constexpr oa::F32 kNsB = -4.7750f;
constexpr oa::F32 kNsC = 2.0315f;

oa::Matrix muonLinearCombination(
	const oa::Matrix& inA,
	oa::F32 inScaleA,
	const oa::Matrix& inB,
	oa::F32 inScaleB
) {
	auto& ctx = oa::ExecutionSession::getActive();
	oa::Matrix out = oa::FnMatrix::empty(inA.getShape(), inA.getDtype());
	const oa::U32 count = static_cast<oa::U32>(inA.numElements());
	const oa::U32 groups = (count + 255) / 256;
	struct Push {
		oa::U32 count;
		oa::F32 scaleA;
		oa::F32 scaleB;
	} push{count, inScaleA, inScaleB};
	oa::BufferAccess access[] = {
		oa::BufferAccess::Read,
		oa::BufferAccess::Read,
		oa::BufferAccess::Write,
	};
	ctx.add(
		"MuonLinearCombination", {&inA, &inB, &out}, access,
		&push, sizeof(push), groups);
	return out;
}

oa::Matrix muonMatMulAxpby(
	const oa::Matrix& inA,
	const oa::Matrix& inB,
	const oa::Matrix& inResidual,
	oa::F32 inAlpha,
	oa::F32 inBeta
) {
	auto& ctx = oa::ExecutionSession::getActive();
	const oa::U32 m = static_cast<oa::U32>(inA.size(0));
	const oa::U32 n = static_cast<oa::U32>(inB.size(1));
	const oa::U32 k = static_cast<oa::U32>(inA.size(1));
	oa::Matrix out = oa::FnMatrix::empty(oa::MatrixShape{m, n}, inA.getDtype());
	struct Push {
		oa::U32 m;
		oa::U32 n;
		oa::U32 k;
		oa::F32 alpha;
		oa::F32 beta;
	} push{m, n, k, inAlpha, inBeta};
	oa::BufferAccess access[] = {
		oa::BufferAccess::Read,
		oa::BufferAccess::Read,
		oa::BufferAccess::Read,
		oa::BufferAccess::Write,
	};
	ctx.add(
		"MuonMatMulAxpby", {&inA, &inB, &inResidual, &out}, access,
		&push, sizeof(push), (n + 15) / 16, (m + 15) / 16);
	return out;
}

void muonNewtonSchulz5(
	oa::Matrix& inOutZ,
	oa::I32 inNS5Steps
) {
	const bool useFusedSmallRoute = inOutZ.size(0) <= 64;
	for (oa::I32 step = 0; step < inNS5Steps; ++step) {
		// OA MatMulNt(A, B) computes A @ B^T with B stored as [N,K].
		// A = Z @ Z^T therefore passes Z twice. BZ = B @ Z passes Z^T
		// as the stored right operand. Keeping these layouts explicit is
		// essential for non-square Muon matrices.
		oa::Matrix a = oa::FnMatrix::matMulNt(inOutZ, inOutZ);
		oa::Matrix b;
		if (useFusedSmallRoute) {
			// A = Z Z^T is symmetric, so A A^T and A A are equivalent.
			b = muonMatMulAxpby(a, a, a, kNsC, kNsB);
		} else {
			oa::Matrix aa = oa::FnMatrix::matMulNt(a, a);
			b = muonLinearCombination(a, kNsB, aa, kNsC);
		}
		if (useFusedSmallRoute) {
			inOutZ = muonMatMulAxpby(b, inOutZ, inOutZ, 1.0F, kNsA);
		} else {
			oa::Matrix zT = oa::FnMatrix::transpose(inOutZ, 0, 1);
			oa::Matrix bz = oa::FnMatrix::matMulNt(b, zT);
			inOutZ = muonLinearCombination(inOutZ, kNsA, bz, 1.0F);
		}
	}
}

void muonMatrixStep(
	oa::Matrix& inOutParam,
	oa::Matrix& inOutMomentum,
	const oa::Matrix& inGrad,
	oa::U32 inRows,
	oa::U32 inCols,
	oa::F32 inLr,
	oa::F32 inBeta,
	oa::F32 inWeightDecay,
	oa::F32 inEps,
	oa::I32 inNS5Iterations)
{
	auto& ctx = oa::ExecutionSession::getActive();
	const oa::U32 count = inRows * inCols;
	const oa::U32 groups = (count + 255) / 256;

	oa::Matrix update = oa::FnMatrix::zeros(inOutParam.getShape());
	{
		struct MuonNesterovPush {
			oa::U32 Count;
			oa::F32 Beta;
		};
		MuonNesterovPush push{count, inBeta};
		oa::BufferAccess access[] = {
			oa::BufferAccess::Read,
			oa::BufferAccess::ReadWrite,
			oa::BufferAccess::Write
		};
		ctx.add( "MuonNesterov", {&inGrad, &inOutMomentum, &update}, access, &push, sizeof(push), groups);
	}

	const bool transposed = inRows > inCols;
	const oa::U32 operRows = transposed ? inCols : inRows;
	const oa::U32 operCols = transposed ? inRows : inCols;

	oa::Matrix z = transposed ? oa::FnMatrix::transpose(update, 0, 1) : update;

	oa::Matrix normScalar = oa::FnMatrix::zeros(oa::MatrixShape{1});
	{
		struct MuonNormalizePush {
			oa::U32 rows;
			oa::U32 cols;
			oa::F32 eps;
		};
		MuonNormalizePush push{operRows, operCols, inEps};
		oa::BufferAccess access[] = {
			oa::BufferAccess::Read,
			oa::BufferAccess::ReadWrite,
			oa::BufferAccess::Write
		};
		ctx.add( "MuonNormalize", {&z, &z, &normScalar}, access, &push, sizeof(push), 1);
	}

	if (inNS5Iterations > 0) {
		muonNewtonSchulz5(z, inNS5Iterations);
	}

	oa::Matrix ortho = transposed ? oa::FnMatrix::transpose(z, 0, 1) : z;

	const oa::U32 maxDimension = oa::max(inRows, inCols);
	const oa::F32 moonshotScale = 0.2F
		* oa::sqrt(static_cast<oa::F32>(maxDimension));
	{
		struct MuonApplyPush {
			oa::U32 Count;
			oa::F32 lr;
			oa::F32 weightDecay;
			oa::F32 MoonshotScale;
		};
		MuonApplyPush push{count, inLr, inWeightDecay, moonshotScale};
		oa::BufferAccess access[] = {
			oa::BufferAccess::ReadWrite,
			oa::BufferAccess::Read
		};
		ctx.add( "MuonApply", {&inOutParam, &ortho}, access, &push, sizeof(push), groups);
	}
}

} // namespace

void adamWAdvanceGraphState(oa::Matrix& inOutState) {
	auto& ctx = oa::ExecutionSession::getActive();
	oa::BufferAccess access[] = {oa::BufferAccess::ReadWrite};
	ctx.add( "AdamwGraphAdvance", {&inOutState}, access, nullptr, 0, 1);
}

void adamWStep(
	oa::Matrix& inOutParam,
	oa::Matrix& inOutM,
	oa::Matrix& inOutV,
	const oa::Matrix& inGrad,
	oa::F32 inLr,
	oa::F32 inBeta1,
	oa::F32 inBeta2,
	oa::F32 inEps,
	oa::F32 inWeightDecay,
	oa::I32 inStep
) {
	auto& ctx = oa::ExecutionSession::getActive();

	const oa::U32 count = static_cast<oa::U32>(inOutParam.numElements());
	struct AdamWPush {
		oa::U32 Count;
		oa::F32 lr;
		oa::F32 beta1;
		oa::F32 beta2;
		oa::F32 eps;
		oa::F32 weightDecay;
		oa::U32 step;
	};
	AdamWPush push{count, inLr, inBeta1, inBeta2, inEps, inWeightDecay,
		static_cast<oa::U32>(inStep)};

	oa::BufferAccess access[] = {
		oa::BufferAccess::ReadWrite,
		oa::BufferAccess::Read,
		oa::BufferAccess::ReadWrite,
		oa::BufferAccess::ReadWrite
	};
	const oa::U32 groups = (count + 255) / 256;
	ctx.add( "Adamw", {&inOutParam, &inGrad, &inOutM, &inOutV}, access, &push, sizeof(push), groups);
}

void adamWStepGraph(
	oa::Matrix& inOutParam,
	oa::Matrix& inOutM,
	oa::Matrix& inOutV,
	const oa::Matrix& inGrad,
	const oa::Matrix& inState
) {
	auto& ctx = oa::ExecutionSession::getActive();
	const oa::U32 count = static_cast<oa::U32>(inOutParam.numElements());
	struct Push { oa::U32 Count; } push{count};
	oa::BufferAccess access[] = {
		oa::BufferAccess::ReadWrite,
		oa::BufferAccess::Read,
		oa::BufferAccess::ReadWrite,
		oa::BufferAccess::ReadWrite,
		oa::BufferAccess::Read,
	};
	ctx.add( "AdamwGraph", {&inOutParam, &inGrad, &inOutM, &inOutV, &inState},
		access, &push, sizeof(push), (count + 255) / 256);
}

void adamWStepMany(
	oa::Span<const AdamWParamSet> inParams,
	oa::F32 inLr,
	oa::F32 inBeta1,
	oa::F32 inBeta2,
	oa::F32 inEps,
	oa::F32 inWeightDecay,
	oa::I32 inStep
) {
	if (inParams.size() != 4) {
		for (const auto& p : inParams) {
			adamWStep(*p.param, *p.m, *p.v, *p.grad,
				inLr, inBeta1, inBeta2, inEps, inWeightDecay, inStep);
		}
		return;
	}

	auto& ctx = oa::ExecutionSession::getActive();

	const oa::U32 count0 = static_cast<oa::U32>(inParams[0].param->numElements());
	const oa::U32 count1 = static_cast<oa::U32>(inParams[1].param->numElements());
	const oa::U32 count2 = static_cast<oa::U32>(inParams[2].param->numElements());
	const oa::U32 count3 = static_cast<oa::U32>(inParams[3].param->numElements());
	const oa::U32 maxCount = oa::max(oa::max(count0, count1), oa::max(count2, count3));

	struct AdamWMany4Push {
		oa::U32 count0;
		oa::U32 count1;
		oa::U32 count2;
		oa::U32 count3;
		oa::F32 lr;
		oa::F32 beta1;
		oa::F32 beta2;
		oa::F32 eps;
		oa::F32 weightDecay;
		oa::U32 step;
	};
	AdamWMany4Push push{
		count0, count1, count2, count3,
		inLr, inBeta1, inBeta2, inEps, inWeightDecay,
		static_cast<oa::U32>(inStep)
	};

	oa::BufferAccess access[] = {
		oa::BufferAccess::ReadWrite, oa::BufferAccess::Read, oa::BufferAccess::ReadWrite, oa::BufferAccess::ReadWrite,
		oa::BufferAccess::ReadWrite, oa::BufferAccess::Read, oa::BufferAccess::ReadWrite, oa::BufferAccess::ReadWrite,
		oa::BufferAccess::ReadWrite, oa::BufferAccess::Read, oa::BufferAccess::ReadWrite, oa::BufferAccess::ReadWrite,
		oa::BufferAccess::ReadWrite, oa::BufferAccess::Read, oa::BufferAccess::ReadWrite, oa::BufferAccess::ReadWrite,
	};
	const oa::U32 groups = (maxCount + 255) / 256;
	ctx.add( "AdamwMany4", {
		inParams[0].param, inParams[0].grad, inParams[0].m, inParams[0].v,
		inParams[1].param, inParams[1].grad, inParams[1].m, inParams[1].v,
		inParams[2].param, inParams[2].grad, inParams[2].m, inParams[2].v,
		inParams[3].param, inParams[3].grad, inParams[3].m, inParams[3].v,
	}, access, &push, sizeof(push), groups);
}

void adamWStepManyGraph(
	oa::Span<const AdamWParamSet> inParams,
	const oa::Matrix& inState
) {
	if (inParams.size() != 4) {
		for (const auto& p : inParams) {
			adamWStepGraph(*p.param, *p.m, *p.v, *p.grad, inState);
		}
		return;
	}

	auto& ctx = oa::ExecutionSession::getActive();
	const oa::U32 count0 = static_cast<oa::U32>(inParams[0].param->numElements());
	const oa::U32 count1 = static_cast<oa::U32>(inParams[1].param->numElements());
	const oa::U32 count2 = static_cast<oa::U32>(inParams[2].param->numElements());
	const oa::U32 count3 = static_cast<oa::U32>(inParams[3].param->numElements());
	const oa::U32 maxCount = oa::max(oa::max(count0, count1), oa::max(count2, count3));
	struct Push {
		oa::U32 count0, count1, count2, count3;
	} push{count0, count1, count2, count3};
	oa::BufferAccess access[] = {
		oa::BufferAccess::ReadWrite, oa::BufferAccess::Read, oa::BufferAccess::ReadWrite, oa::BufferAccess::ReadWrite,
		oa::BufferAccess::ReadWrite, oa::BufferAccess::Read, oa::BufferAccess::ReadWrite, oa::BufferAccess::ReadWrite,
		oa::BufferAccess::ReadWrite, oa::BufferAccess::Read, oa::BufferAccess::ReadWrite, oa::BufferAccess::ReadWrite,
		oa::BufferAccess::ReadWrite, oa::BufferAccess::Read, oa::BufferAccess::ReadWrite, oa::BufferAccess::ReadWrite,
		oa::BufferAccess::Read,
	};
	ctx.add( "AdamwMany4Graph", {
		inParams[0].param, inParams[0].grad, inParams[0].m, inParams[0].v,
		inParams[1].param, inParams[1].grad, inParams[1].m, inParams[1].v,
		inParams[2].param, inParams[2].grad, inParams[2].m, inParams[2].v,
		inParams[3].param, inParams[3].grad, inParams[3].m, inParams[3].v,
		&inState,
	}, access, &push, sizeof(push), (maxCount + 255) / 256);
}

void sgdStep(
	oa::Matrix& inOutParam,
	oa::Matrix& inOutMomentum,
	const oa::Matrix& inGrad,
	oa::F32 inLr,
	oa::F32 inMomentum,
	oa::F32 inWeightDecay
) {
	auto& ctx = oa::ExecutionSession::getActive();
	const oa::U32 count = static_cast<oa::U32>(inOutParam.numElements());
	const oa::U32 groups = (count + 255) / 256;

	if (inMomentum > 0.0F && inOutMomentum.hasStorage()) {
		struct Push { oa::U32 count; oa::F32 lr; oa::F32 momentum_coef; oa::F32 weight_decay; }
			push{count, inLr, inMomentum, inWeightDecay};
		oa::BufferAccess access[] = {
			oa::BufferAccess::ReadWrite,
			oa::BufferAccess::Read,
			oa::BufferAccess::ReadWrite
		};
		ctx.add( "SgdMomentum", {&inOutParam, &inGrad, &inOutMomentum}, access, &push, sizeof(push), groups);
		return;
	}

	struct Push { oa::U32 count; oa::F32 lr; oa::F32 weight_decay; }
		push{count, inLr, inWeightDecay};
	oa::BufferAccess access[] = {
		oa::BufferAccess::ReadWrite,
		oa::BufferAccess::Read
	};
	ctx.add( "Sgd", {&inOutParam, &inGrad}, access, &push, sizeof(push), groups);
}

void muonStep(
	oa::Matrix& inOutParam,
	oa::Matrix& inOutMomentum,
	const oa::Matrix& inGrad,
	oa::F32 inLr,
	oa::F32 inBeta,
	oa::F32 inWeightDecay,
	oa::F32 inEps,
	oa::I32 inNS5Iterations
) {
	auto& ctx = oa::ExecutionSession::getActive();
	const oa::U32 count = static_cast<oa::U32>(inOutParam.numElements());
	const oa::U32 groups = (count + 255) / 256;

	const bool is2DMatrix = (inOutParam.getShape().rank == 2);
	const oa::U32 rows = is2DMatrix ? static_cast<oa::U32>(inOutParam.getShape()[0]) : 0U;
	const oa::U32 cols = is2DMatrix ? static_cast<oa::U32>(inOutParam.getShape()[1]) : 0U;

	if (is2DMatrix && inNS5Iterations > 0) {
		muonMatrixStep(
			inOutParam, inOutMomentum, inGrad,
			rows, cols, inLr, inBeta, inWeightDecay, inEps, inNS5Iterations);
		return;
	}

	// Non-matrix parameters use the optimizer's fused GPU momentum update.
	struct MuonVectorPush {
		oa::U32 Count;
		oa::F32 lr;
		oa::F32 Beta;
		oa::F32 weightDecay;
	};
	MuonVectorPush push{count, inLr, inBeta, inWeightDecay};
	oa::BufferAccess access[] = {
		oa::BufferAccess::ReadWrite,
		oa::BufferAccess::Read,
		oa::BufferAccess::ReadWrite
	};
	ctx.add( "MuonVector", {&inOutParam, &inGrad, &inOutMomentum}, access, &push, sizeof(push), groups);
}

void clipGradNorm(
	oa::Span<oa::Matrix*> inGrads,
	oa::F32 inMaxNorm,
	oa::Matrix& inOutParams,
	oa::Matrix& inOutPartials
) {
	auto& ctx = oa::ExecutionSession::getActive();

	// Collect non-empty grad buffers (skip params with no grad yet).
	oa::Matrix* grads[16];
	oa::I32 counts[16];
	oa::I32 n = 0;
	for (oa::Usize i = 0; i < inGrads.size() && n < 16; ++i) {
		oa::Matrix* g = inGrads[i];
		if (g && !g->isEmpty()) {
			grads[n]  = g;
			counts[n] = static_cast<oa::I32>(g->numElements());
			++n;
		}
	}
	if (n == 0) return;

	// upload params buffer: [n_tensors, count0..count15, max_norm_bits] as uint32.
	// max_norm is carried in slot 17 (float bit-pattern) rather than as a trailing
	// push payload: the runtime packs only (2 + n) prepended buffer indices, so a
	// payload after the shader's fixed grad_idx[16] field lands at the wrong offset
	// for n < 16 (it read zero → scale 0 → all grads zeroed → training frozen).
	oa::I32 paramData[18] = {};
	paramData[0] = n;
	for (oa::I32 i = 0; i < n; ++i) paramData[i + 1] = counts[i];
	oa::memcpy(&paramData[17], &inMaxNorm, sizeof(oa::F32));  // float bits → slot 17
	inOutParams = oa::FnMatrix::fromInt32(
		oa::Span<const oa::I32>(paramData, 18),
		oa::MatrixShape{18}, oa::ScalarType::Int32);

	// Build VkBuffer arrays — ctx.add oavk::Buffer-span overload.
	oavk::Buffer vkBufs[18];
	oa::BufferAccess accessBufs[18];
	vkBufs[0] = oa::MatrixAccess::descriptor(inOutParams);
	vkBufs[1] = oa::MatrixAccess::descriptor(inOutPartials);
	for (oa::I32 i = 0; i < n; ++i) {
		vkBufs[2 + i] = oa::MatrixAccess::descriptor(*grads[i]);
	}
	const oa::I32 total = 2 + n;

	// ── Pass 1: ClipGradNormReduce ──────────────────────────────────────────
	accessBufs[0] = oa::BufferAccess::Read;    // params
	accessBufs[1] = oa::BufferAccess::Write;   // partials
	for (oa::I32 i = 0; i < n; ++i) accessBufs[2 + i] = oa::BufferAccess::Read;

	ctx.add( "ClipGradNormReduce",
		oa::Span<oavk::Buffer>(vkBufs, static_cast<oa::Usize>(total)),
		oa::Span<oa::BufferAccess>(accessBufs, static_cast<oa::Usize>(total)),
		nullptr, 0,
		static_cast<oa::U32>(n));

	// ── Pass 2: ClipGradNormScale ────────────────────────────────────────────
	accessBufs[0] = oa::BufferAccess::Read;    // params
	accessBufs[1] = oa::BufferAccess::Read;    // partials
	for (oa::I32 i = 0; i < n; ++i) accessBufs[2 + i] = oa::BufferAccess::ReadWrite;

	ctx.add( "ClipGradNormScale",
		oa::Span<oavk::Buffer>(vkBufs, static_cast<oa::Usize>(total)),
		oa::Span<oa::BufferAccess>(accessBufs, static_cast<oa::Usize>(total)),
		nullptr, 0,   // max_norm now carried in params[17], not a push payload
		static_cast<oa::U32>(n));
}

} // namespace FnOptim

} // namespace oa
