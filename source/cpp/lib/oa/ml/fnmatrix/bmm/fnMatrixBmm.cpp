// oa::FnMatrix::bmm — batched matrix multiply A[N,M,K] @ B[N,K,P] = out[N,M,P].
// GPU kernel "Bmm" (ops/Bmm.slang) + autograd (oa::GradBmm). The enabler for
// differentiable forward kinematics and any per-row small-matrix algebra.

#include <oa/ml/fnMatrix.h>
#include <oa/core/envFlag.h>
#include <oa/core/fnMatrix.h>
#include <oa/core/op.h>
#include <oa/runtime/executionSession.h>
#include "../../autograd/autogradAttach.gen.h"

namespace {
oa::U32 divCeil(oa::U32 inA, oa::U32 inB) { return (inA + inB - 1) / inB; }
}

oa::Matrix oa::FnMatrix::bmm(const oa::Matrix& inA, const oa::Matrix& inB) {
	auto& ctx = oa::ExecutionSession::getActive();
	OA_ASSERT(inA.rank() == 3 && inB.rank() == 3 && "Bmm expects rank-3 [N,M,K] @ [N,K,P]");
	const oa::I64 N = inA.size(0), M = inA.size(1), K = inA.size(2), P = inB.size(2);
	OA_ASSERT(inB.size(0) == N && inB.size(1) == K && "Bmm batch/inner dim mismatch");

	oa::OpLoweringScope lowering(ctx);
	oa::Matrix out = oa::FnMatrix::empty(oa::MatrixShape{N, M, P}, inA.getDtype());
	struct { oa::U32 Batch; oa::U32 M; oa::U32 K; oa::U32 P; } push{
		static_cast<oa::U32>(N), static_cast<oa::U32>(M), static_cast<oa::U32>(K), static_cast<oa::U32>(P)
	};
	oa::BufferAccess access[] = {oa::BufferAccess::Read, oa::BufferAccess::Read, oa::BufferAccess::Write};
	const bool useTiled = inA.getDtype() == oa::ScalarType::Float32
		and N > 0 and N <= 65535 and M >= 8 and K >= 8 and P >= 8
		and not oa::EnvFlag::isSet("OA_DISABLE_TILED_BMM");
	if (useTiled) {
		ctx.add( "BmmTiled16", {&inA, &inB, &out},
			access, &push, sizeof(push),
			divCeil(static_cast<oa::U32>(P), 16),
			divCeil(static_cast<oa::U32>(M), 16),
			static_cast<oa::U32>(N));
	} else {
		ctx.add( "Bmm", {&inA, &inB, &out},
			access, &push, sizeof(push),
			divCeil(static_cast<oa::U32>(N * M * P), 256));
	}

	const auto semantic = lowering.commitWithId(
		oa::detail::opRegistry::FnMatrix::bmm,
		{&inA, &inB}, {&out});
	if (not semantic.isOk()) return {};
	if (not oa::detail::generatedAutogradAttach::FnMatrix::bmm(
		out, inA, inB, semantic.getValue()).isOk())
	{
		return {};
	}
	return out;
}

oa::Matrix oa::FnMatrix::bmmNt(const oa::Matrix& inA, const oa::Matrix& inB) {
	auto& ctx = oa::ExecutionSession::getActive();
	OA_ASSERT(inA.rank() == 3 && inB.rank() == 3 &&
		"BmmNt expects rank-3 [N,M,K] @ [N,P,K]^T");
	const oa::I64 N = inA.size(0), M = inA.size(1);
	const oa::I64 K = inA.size(2), P = inB.size(1);
	OA_ASSERT(inB.size(0) == N && inB.size(2) == K &&
		"BmmNt batch/inner dim mismatch");
	OA_ASSERT(inA.getDtype() == inB.getDtype() && "BmmNt dtype mismatch");

	oa::OpLoweringScope lowering(ctx);
	oa::Matrix out = oa::FnMatrix::empty(oa::MatrixShape{N, M, P}, inA.getDtype());
	struct { oa::U32 Batch; oa::U32 M; oa::U32 K; oa::U32 P; } push{
		static_cast<oa::U32>(N), static_cast<oa::U32>(M),
		static_cast<oa::U32>(K), static_cast<oa::U32>(P)
	};
	oa::BufferAccess access[] = {
		oa::BufferAccess::Read, oa::BufferAccess::Read, oa::BufferAccess::Write};
	const bool useTiled = inA.getDtype() == oa::ScalarType::Float32
		and N > 0 and N <= 65535 and M >= 8 and K >= 8 and P >= 8
		and not oa::EnvFlag::isSet("OA_DISABLE_TILED_BMM_NT");
	if (useTiled) {
		ctx.add( "BmmNtTiled16", {&inA, &inB, &out},
			access, &push, sizeof(push),
			divCeil(static_cast<oa::U32>(P), 16),
			divCeil(static_cast<oa::U32>(M), 16),
			static_cast<oa::U32>(N));
	} else {
		ctx.add( "BmmNt", {&inA, &inB, &out},
			access, &push, sizeof(push),
			divCeil(static_cast<oa::U32>(N * M * P), 256));
	}

	const auto semantic = lowering.commitWithId(
		oa::detail::opRegistry::FnMatrix::bmmNt,
		{&inA, &inB}, {&out});
	if (not semantic.isOk()) return {};
	if (not oa::detail::generatedAutogradAttach::FnMatrix::bmmNt(
		out, inA, inB, semantic.getValue()).isOk())
	{
		return {};
	}
	return out;
}
