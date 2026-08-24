// GPU-native sparse-MoE systems primitives.

#include <oa/core/fnMatrix.h>
#include <oa/ml/fnMatrix.h>
#include <oa/core/bufferAccess.h>
#include <oa/core/log.h>
#include <oa/core/op.h>
#include <oa/runtime/executionSession.h>
#include "../../autograd/autogradAttach.gen.h"

static oa::U32 divCeil(oa::U32 inA, oa::U32 inB) { return (inA + inB - 1) / inB; }

oa::Matrix oa::FnMatrix::moeRouteWeights(const oa::Matrix& inProbs,	const oa::Matrix& inExpertIndices) {
	if (inProbs.rank() != 2 or inExpertIndices.rank() != 2 or
		(inProbs.getDtype() != oa::ScalarType::Float32 and
		 inProbs.getDtype() != oa::ScalarType::BFloat16) or
		inExpertIndices.getDtype() != oa::ScalarType::Int32 or
		inProbs.size(0) != inExpertIndices.size(0) or
		inExpertIndices.size(1) <= 0 or inExpertIndices.size(1) > inProbs.size(1)) {
		OaLogError(oa::LogComponent::Ml,
			"MoeRouteWeights: expected Probs[T,E] and Int32 indices[T,K], 1 <= K <= E");
		return {};
	}
	const oa::U32 T = static_cast<oa::U32>(inProbs.size(0));
	const oa::U32 E = static_cast<oa::U32>(inProbs.size(1));
	const oa::U32 K = static_cast<oa::U32>(inExpertIndices.size(1));
	auto& ctx = oa::ExecutionSession::getActive();
	oa::OpLoweringScope lowering(ctx);
	auto out = oa::FnMatrix::empty(inExpertIndices.getShape(), inProbs.getDtype());
	struct { oa::U32 T, E, K; } push{T, E, K};
	oa::BufferAccess access[] = {
		oa::BufferAccess::Read, oa::BufferAccess::Read, oa::BufferAccess::Write};
	ctx.add( "MoeRouteWeights", {&inProbs, &inExpertIndices, &out}, access,
		&push, sizeof(push), divCeil(T, 256));
	const auto semantic = lowering.commitWithId(
		oa::detail::opRegistry::FnMatrix::moeRouteWeights,
		{&inProbs, &inExpertIndices}, {&out});
	if (not semantic.isOk()) return {};
	if (not oa::detail::generatedAutogradAttach::FnMatrix::moeRouteWeights(
		out, inProbs, inExpertIndices, semantic.getValue()).isOk())
	{
		return {};
	}
	return out;
}

oa::Matrix oa::FnMatrix::moeRouteWeightsBwd(const oa::Matrix& inDOut,	const oa::Matrix& inProbs, const oa::Matrix& inExpertIndices,	const oa::Matrix& inRouteWeights) {
	if (inProbs.rank() != 2 or inDOut.getShape() != inExpertIndices.getShape() or
		inDOut.getShape() != inRouteWeights.getShape() or
		inDOut.getDtype() != inProbs.getDtype() or
		inRouteWeights.getDtype() != inProbs.getDtype() or
		inExpertIndices.getDtype() != oa::ScalarType::Int32 or
		inProbs.size(0) != inExpertIndices.size(0)) {
		OaLogError(oa::LogComponent::Ml, "MoeRouteWeightsBwd: incompatible tensors");
		return {};
	}
	const oa::U32 T = static_cast<oa::U32>(inProbs.size(0));
	const oa::U32 E = static_cast<oa::U32>(inProbs.size(1));
	const oa::U32 K = static_cast<oa::U32>(inExpertIndices.size(1));
	auto& ctx = oa::ExecutionSession::getActive();
	oa::OpLoweringScope lowering(ctx);
	auto out = oa::FnMatrix::empty(inProbs.getShape(), inProbs.getDtype());
	struct { oa::U32 T, E, K; } push{T, E, K};
	oa::BufferAccess access[] = {oa::BufferAccess::Read, oa::BufferAccess::Read,
		oa::BufferAccess::Read, oa::BufferAccess::Read, oa::BufferAccess::Write};
	ctx.add( "MoeRouteWeightsBwd",
		{&inDOut, &inProbs, &inExpertIndices, &inRouteWeights, &out}, access,
		&push, sizeof(push), divCeil(T * E, 256));
	if (not lowering.commit(
		oa::detail::opRegistry::FnMatrix::moeRouteWeightsBwd,
		{&inDOut, &inProbs, &inExpertIndices, &inRouteWeights}, {&out}).isOk())
	{
		return {};
	}
	return out;
}

oa::Matrix oa::FnMatrix::groupedGemmM(const oa::Matrix& inX, const oa::Matrix& inWeight,
	const oa::Matrix& inOffsets) {
	if (inX.rank() != 2 or inWeight.rank() != 3 or inOffsets.rank() != 1 or
		inOffsets.getDtype() != oa::ScalarType::UInt32 or inX.getDtype() != inWeight.getDtype() or
		inX.size(1) != inWeight.size(2) or inOffsets.size(0) != inWeight.size(0) + 1) {
		OaLogError(oa::LogComponent::Ml,	"GroupedGemmM: expected X[R,K], W[E,N,K], UInt32 offsets[E+1]");
		return {};
	}
	const oa::U32 R = static_cast<oa::U32>(inX.size(0));
	const oa::U32 K = static_cast<oa::U32>(inX.size(1));
	const oa::U32 E = static_cast<oa::U32>(inWeight.size(0));
	const oa::U32 N = static_cast<oa::U32>(inWeight.size(1));
	auto& ctx = oa::ExecutionSession::getActive();
	oa::OpLoweringScope lowering(ctx);
	auto out = oa::FnMatrix::empty(oa::MatrixShape{R, N}, inX.getDtype());
	struct { oa::U32 R, N, K, E; } push{R, N, K, E};
	oa::BufferAccess access[] = {oa::BufferAccess::Read, oa::BufferAccess::Read,
		oa::BufferAccess::Read, oa::BufferAccess::Write};
	ctx.add( "GroupedGemmM", {&inX, &inWeight, &inOffsets, &out}, access,
		&push, sizeof(push), divCeil(R, 32), divCeil(N, 32), 1);
	const auto semantic = lowering.commitWithId(
		oa::detail::opRegistry::FnMatrix::groupedGemmM,
		{&inX, &inWeight, &inOffsets}, {&out});
	if (not semantic.isOk()) return {};
	if (not oa::detail::generatedAutogradAttach::FnMatrix::groupedGemmM(
		out, inX, inWeight, inOffsets, semantic.getValue()).isOk())
	{
		return {};
	}
	return out;
}

oa::Matrix oa::FnMatrix::groupedLinearM(const oa::Matrix& inX, const oa::Matrix& inWeight, const oa::Matrix& inBias, const oa::Matrix& inOffsets) {
	if (inX.rank() != 2 or inWeight.rank() != 3 or inBias.rank() != 2 or
		inOffsets.rank() != 1 or inOffsets.getDtype() != oa::ScalarType::UInt32 or
		inX.getDtype() != inWeight.getDtype() or inX.getDtype() != inBias.getDtype() or
		inX.size(1) != inWeight.size(2) or inBias.size(0) != inWeight.size(0) or
		inBias.size(1) != inWeight.size(1) or inOffsets.size(0) != inWeight.size(0) + 1) {
		OaLogError(oa::LogComponent::Ml,	"GroupedLinearM: expected X[R,K], W[E,N,K], Bias[E,N], UInt32 offsets[E+1]");
		return {};
	}
	const oa::U32 R = static_cast<oa::U32>(inX.size(0));
	const oa::U32 K = static_cast<oa::U32>(inX.size(1));
	const oa::U32 E = static_cast<oa::U32>(inWeight.size(0));
	const oa::U32 N = static_cast<oa::U32>(inWeight.size(1));
	auto& ctx = oa::ExecutionSession::getActive();
	oa::OpLoweringScope lowering(ctx);
	auto out = oa::FnMatrix::empty(oa::MatrixShape{R, N}, inX.getDtype());
	struct { oa::U32 R, N, K, E; } push{R, N, K, E};
	oa::BufferAccess access[] = {oa::BufferAccess::Read, oa::BufferAccess::Read,
		oa::BufferAccess::Read, oa::BufferAccess::Read, oa::BufferAccess::Write};
	ctx.add( "GroupedLinearM", {&inX, &inWeight, &inBias, &inOffsets, &out}, access,
		&push, sizeof(push), divCeil(R, 32), divCeil(N, 32), 1);
	const auto semantic = lowering.commitWithId(
		oa::detail::opRegistry::FnMatrix::groupedLinearM,
		{&inX, &inWeight, &inBias, &inOffsets}, {&out});
	if (not semantic.isOk()) return {};
	if (not oa::detail::generatedAutogradAttach::FnMatrix::groupedLinearM(
		out, inX, inWeight, inBias, inOffsets, semantic.getValue()).isOk())
	{
		return {};
	}
	return out;
}

oa::GroupedGemmMBwdResult oa::FnMatrix::groupedGemmMBwd(
	const oa::Matrix& inDOut, const oa::Matrix& inX, const oa::Matrix& inWeight,
	const oa::Matrix& inOffsets) {
	if (inDOut.rank() != 2 or inX.rank() != 2 or inWeight.rank() != 3 or
		inOffsets.rank() != 1 or inOffsets.getDtype() != oa::ScalarType::UInt32 or
		inDOut.getDtype() != inX.getDtype() or inX.getDtype() != inWeight.getDtype() or
		inDOut.size(0) != inX.size(0) or inDOut.size(1) != inWeight.size(1) or
		inX.size(1) != inWeight.size(2) or inOffsets.size(0) != inWeight.size(0) + 1) {
		OaLogError(oa::LogComponent::Ml, "GroupedGemmMBwd: incompatible tensors");
		return {};
	}
	const oa::U32 R = static_cast<oa::U32>(inX.size(0));
	const oa::U32 K = static_cast<oa::U32>(inX.size(1));
	const oa::U32 E = static_cast<oa::U32>(inWeight.size(0));
	const oa::U32 N = static_cast<oa::U32>(inWeight.size(1));
	oa::GroupedGemmMBwdResult result;
	auto& ctx = oa::ExecutionSession::getActive();
	oa::OpLoweringScope lowering(ctx);
	result.dInput = oa::FnMatrix::empty(inX.getShape(), inX.getDtype());
	result.dWeight = oa::FnMatrix::empty(inWeight.getShape(), inWeight.getDtype());
	struct { oa::U32 R, N, K, E; } push{R, N, K, E};
	{
		oa::BufferAccess access[] = {oa::BufferAccess::Read, oa::BufferAccess::Read,
			oa::BufferAccess::Read, oa::BufferAccess::Write};
		ctx.add( "GroupedGemmMDataBwd", {&inDOut, &inWeight, &inOffsets, &result.dInput},
			access, &push, sizeof(push), divCeil(R, 32), divCeil(K, 32), 1);
	}
	{
		oa::BufferAccess access[] = {oa::BufferAccess::Read, oa::BufferAccess::Read,
			oa::BufferAccess::Read, oa::BufferAccess::Write};
		ctx.add( "GroupedGemmMWeightBwd", {&inDOut, &inX, &inOffsets, &result.dWeight},
			access, &push, sizeof(push), divCeil(N, 32), divCeil(K, 32), E);
	}
	if (not lowering.commit(
		oa::detail::opRegistry::FnMatrix::groupedGemmMBwd,
		{&inDOut, &inX, &inWeight, &inOffsets},
		{&result.dInput, &result.dWeight}).isOk())
	{
		return {};
	}
	return result;
}

oa::GroupedLinearMBwdResult oa::FnMatrix::groupedLinearMBwd(
	const oa::Matrix& inDOut, const oa::Matrix& inX, const oa::Matrix& inWeight,
	const oa::Matrix& inOffsets) {
	if (inDOut.rank() != 2 or inX.rank() != 2 or inWeight.rank() != 3 or
		inOffsets.rank() != 1 or inOffsets.getDtype() != oa::ScalarType::UInt32 or
		inDOut.getDtype() != inX.getDtype() or inX.getDtype() != inWeight.getDtype() or
		inDOut.size(0) != inX.size(0) or inDOut.size(1) != inWeight.size(1) or
		inX.size(1) != inWeight.size(2) or inOffsets.size(0) != inWeight.size(0) + 1) {
		OaLogError(oa::LogComponent::Ml, "GroupedLinearMBwd: incompatible tensors");
		return {};
	}
	const oa::U32 R = static_cast<oa::U32>(inX.size(0));
	const oa::U32 K = static_cast<oa::U32>(inX.size(1));
	const oa::U32 E = static_cast<oa::U32>(inWeight.size(0));
	const oa::U32 N = static_cast<oa::U32>(inWeight.size(1));
	oa::GroupedLinearMBwdResult result;
	auto& ctx = oa::ExecutionSession::getActive();
	oa::OpLoweringScope lowering(ctx);
	result.dInput = oa::FnMatrix::empty(inX.getShape(), inX.getDtype());
	result.dWeight = oa::FnMatrix::empty(inWeight.getShape(), inWeight.getDtype());
	result.dBias = oa::FnMatrix::empty(oa::MatrixShape{E, N}, inDOut.getDtype());
	struct { oa::U32 R, N, K, E; } push{R, N, K, E};
	{
		oa::BufferAccess access[] = {oa::BufferAccess::Read, oa::BufferAccess::Read,
			oa::BufferAccess::Read, oa::BufferAccess::Write};
		ctx.add( "GroupedGemmMDataBwd",
			{&inDOut, &inWeight, &inOffsets, &result.dInput}, access,
			&push, sizeof(push), divCeil(R, 32), divCeil(K, 32), 1);
	}
	{
		oa::BufferAccess access[] = {oa::BufferAccess::Read, oa::BufferAccess::Read,
			oa::BufferAccess::Read, oa::BufferAccess::Write, oa::BufferAccess::Write};
		ctx.add( "GroupedLinearMWeightBiasBwd",
			{&inDOut, &inX, &inOffsets, &result.dWeight, &result.dBias}, access,
			&push, sizeof(push), divCeil(N, 32), divCeil(K, 32), E);
	}
	if (not lowering.commit(
		oa::detail::opRegistry::FnMatrix::groupedLinearMBwd,
		{&inDOut, &inX, &inWeight, &inOffsets},
		{&result.dInput, &result.dWeight, &result.dBias}).isOk())
	{
		return {};
	}
	return result;
}

oa::Matrix oa::FnMatrix::groupedLinearMBiasBwd(const oa::Matrix& inDOut,
	const oa::Matrix& inOffsets, oa::I32 inNumExperts) {
	if (inDOut.rank() != 2 or inOffsets.rank() != 1 or
		inOffsets.getDtype() != oa::ScalarType::UInt32 or inNumExperts <= 0 or
		inOffsets.size(0) != inNumExperts + 1) return {};
	const oa::U32 R = static_cast<oa::U32>(inDOut.size(0));
	const oa::U32 N = static_cast<oa::U32>(inDOut.size(1));
	const oa::U32 E = static_cast<oa::U32>(inNumExperts);
	auto& ctx = oa::ExecutionSession::getActive();
	oa::OpLoweringScope lowering(ctx);
	auto out = oa::FnMatrix::empty(oa::MatrixShape{E, N}, inDOut.getDtype());
	struct { oa::U32 R, N, E; } push{R, N, E};
	oa::BufferAccess access[] = {
		oa::BufferAccess::Read, oa::BufferAccess::Read, oa::BufferAccess::Write};
	ctx.add( "GroupedLinearMBiasBwd", {&inDOut, &inOffsets, &out}, access,
		&push, sizeof(push), divCeil(N, 16), divCeil(E, 16), 1);
	if (not lowering.commit(
		oa::detail::opRegistry::FnMatrix::groupedLinearMBiasBwd,
		{&inDOut, &inOffsets}, {&out},
		{oa::OpAttribute::fromSignedInteger(
			"expertCount", inNumExperts)}).isOk())
	{
		return {};
	}
	return out;
}

oa::Matrix oa::FnMatrix::moeCombine(const oa::Matrix& inPacked,
	const oa::Matrix& inRouteGate, const oa::Matrix& inInverse,
	const oa::Matrix& inPackedSlot) {
	if (inPacked.rank() != 2 or inRouteGate.rank() != 2 or
		inPacked.getDtype() != inRouteGate.getDtype() or
		inInverse.rank() != 1 or inPackedSlot.rank() != 1 or
		inInverse.getDtype() != oa::ScalarType::UInt32 or
		inPackedSlot.getDtype() != oa::ScalarType::UInt32 or
		inPacked.size(0) != inRouteGate.numElements() or
		inInverse.size(0) != inPacked.size(0) or
		inPackedSlot.size(0) != inPacked.size(0)) {
		OaLogError(oa::LogComponent::Ml,
			"MoeCombine: expected Packed[R,D], gate[T,K], UInt32 maps[R]");
		return {};
	}
	const oa::U32 T = static_cast<oa::U32>(inRouteGate.size(0));
	const oa::U32 K = static_cast<oa::U32>(inRouteGate.size(1));
	const oa::U32 D = static_cast<oa::U32>(inPacked.size(1));
	const oa::U32 R = static_cast<oa::U32>(inPacked.size(0));
	auto& ctx = oa::ExecutionSession::getActive();
	oa::OpLoweringScope lowering(ctx);
	auto out = oa::FnMatrix::empty(oa::MatrixShape{T, D}, inPacked.getDtype());
	struct { oa::U32 T, K, D, R; } push{T, K, D, R};
	oa::BufferAccess access[] = {oa::BufferAccess::Read, oa::BufferAccess::Read,
		oa::BufferAccess::Read, oa::BufferAccess::Write};
	ctx.add( "MoeCombine", {&inPacked, &inRouteGate, &inInverse, &out},
		access, &push, sizeof(push), divCeil(T * D, 256));
	const auto semantic = lowering.commitWithId(
		oa::detail::opRegistry::FnMatrix::moeCombine,
		{&inPacked, &inRouteGate, &inInverse, &inPackedSlot}, {&out});
	if (not semantic.isOk()) return {};
	if (not oa::detail::generatedAutogradAttach::FnMatrix::moeCombine(
		out, inPacked, inRouteGate, inInverse, inPackedSlot,
		semantic.getValue()).isOk())
	{
		return {};
	}
	return out;
}

oa::MoeCombineBwdResult oa::FnMatrix::moeCombineBwd(
	const oa::Matrix& inDOut, const oa::Matrix& inPacked,
	const oa::Matrix& inRouteGate, const oa::Matrix& inInverse,
	const oa::Matrix& inPackedSlot) {
	oa::MoeCombineBwdResult result;
	if (inDOut.rank() != 2 or inPacked.rank() != 2 or
		inRouteGate.rank() != 2 or inDOut.getDtype() != inPacked.getDtype() or
		inPacked.getDtype() != inRouteGate.getDtype() or
		inInverse.rank() != 1 or inPackedSlot.rank() != 1 or
		inInverse.getDtype() != oa::ScalarType::UInt32 or
		inPackedSlot.getDtype() != oa::ScalarType::UInt32 or
		inDOut.size(0) != inRouteGate.size(0) or
		inDOut.size(1) != inPacked.size(1) or
		inPacked.size(0) != inRouteGate.numElements() or
		inInverse.size(0) != inPacked.size(0) or
		inPackedSlot.size(0) != inPacked.size(0)) {
		OaLogError(oa::LogComponent::Ml,
			"MoeCombineBwd: incompatible dOut, Packed, gate, or UInt32 maps");
		return result;
	}
	result.dPacked = oa::FnMatrix::empty(inPacked.getShape(), inPacked.getDtype());
	result.dRouteGate = oa::FnMatrix::empty(
		inRouteGate.getShape(), inRouteGate.getDtype());
	auto& ctx = oa::ExecutionSession::getActive();
	oa::OpLoweringScope lowering(ctx);
	const oa::U32 T = static_cast<oa::U32>(inRouteGate.size(0));
	const oa::U32 K = static_cast<oa::U32>(inRouteGate.size(1));
	const oa::U32 D = static_cast<oa::U32>(inPacked.size(1));
	const oa::U32 R = static_cast<oa::U32>(inPacked.size(0));
	struct { oa::U32 T, K, D, R; } push{T, K, D, R};
	oa::BufferAccess access[] = {oa::BufferAccess::Read, oa::BufferAccess::Read,
		oa::BufferAccess::Read, oa::BufferAccess::Read, oa::BufferAccess::Read,
		oa::BufferAccess::Write, oa::BufferAccess::Write};
	ctx.add( "MoeCombineBwd", {&inDOut, &inPacked, &inRouteGate, &inInverse,
		&inPackedSlot, &result.dPacked, &result.dRouteGate}, access,
		&push, sizeof(push), divCeil(R * D, 256));
	if (not lowering.commit(
		oa::detail::opRegistry::FnMatrix::moeCombineBwd,
		{&inDOut, &inPacked, &inRouteGate, &inInverse, &inPackedSlot},
		{&result.dPacked, &result.dRouteGate}).isOk())
	{
		return {};
	}
	return result;
}

oa::Matrix oa::FnMatrix::scatterAddRows(const oa::Matrix& inSource,
	const oa::Matrix& inIndices, oa::I32 inOutRows) {
	if (inSource.rank() != 2 or inIndices.rank() != 1 or
		(inSource.getDtype() != oa::ScalarType::Float32 and
		 inSource.getDtype() != oa::ScalarType::BFloat16) or
		inIndices.getDtype() != oa::ScalarType::UInt32 or
		inIndices.size(0) != inSource.size(0) or inOutRows <= 0) {
		OaLogError(oa::LogComponent::Ml,
			"ScatterAddRows: expected source[R,D], UInt32 indices[R], outRows > 0");
		return {};
	}
	const oa::U32 R = static_cast<oa::U32>(inSource.size(0));
	const oa::U32 D = static_cast<oa::U32>(inSource.size(1));
	const oa::U32 T = static_cast<oa::U32>(inOutRows);
	const oa::U32 dtype = inSource.getDtype() == oa::ScalarType::BFloat16 ? 1U : 0U;
	auto& ctx = oa::ExecutionSession::getActive();
	oa::OpLoweringScope lowering(ctx);
	auto out = oa::FnMatrix::zeros(oa::MatrixShape{inOutRows, D}, inSource.getDtype());
	struct { oa::U32 R, D, T, dtype; } push{R, D, T, dtype};
	oa::BufferAccess access[] = {
		oa::BufferAccess::Read, oa::BufferAccess::Read, oa::BufferAccess::ReadWrite};
	ctx.add( "ScatterAddRows", {&inSource, &inIndices, &out}, access,
		&push, sizeof(push), divCeil(R * D, 256));
	const auto semantic = lowering.commitWithId(
		oa::detail::opRegistry::FnMatrix::scatterAddRows,
		{&inSource, &inIndices}, {&out},
		{oa::OpAttribute::fromSignedInteger("outputRows", inOutRows)});
	if (not semantic.isOk()) return {};
	if (not oa::detail::generatedAutogradAttach::FnMatrix::scatterAddRows(
		out, inSource, inIndices, semantic.getValue()).isOk())
	{
		return {};
	}
	return out;
}

oa::Matrix oa::FnMatrix::moeGather(const oa::Matrix& inSelf,const oa::Matrix& inIndices, const oa::Matrix& inInverse) {
	if (inSelf.rank() != 2 or inIndices.rank() != 1 or
		inInverse.rank() != 1 or inIndices.getDtype() != oa::ScalarType::UInt32 or
		inInverse.getDtype() != oa::ScalarType::UInt32 or
		inInverse.numElements() != inIndices.numElements() or
		inSelf.size(0) <= 0 or
		inIndices.numElements() % inSelf.size(0) != 0) {
		OaLogError(oa::LogComponent::Ml,	"MoeGather: expected Self[T,D] and UInt32 indices/inverse[R], R %% T == 0");
		return {};
	}
	const oa::U32 R = static_cast<oa::U32>(inIndices.numElements());
	const oa::U32 D = static_cast<oa::U32>(inSelf.size(1));
	auto& ctx = oa::ExecutionSession::getActive();
	oa::OpLoweringScope lowering(ctx);
	auto out = oa::FnMatrix::empty(oa::MatrixShape{R, D}, inSelf.getDtype());
	struct { oa::U32 numIndices, RowSize, indexDtype; } push{R, D, 1U};
	oa::BufferAccess access[] = {oa::BufferAccess::Read, oa::BufferAccess::Read, oa::BufferAccess::Write};
	ctx.add( "Gather", {&inSelf, &inIndices, &out}, access,
		&push, sizeof(push), divCeil(R * D, 256));
	const auto semantic = lowering.commitWithId(
		oa::detail::opRegistry::FnMatrix::moeGather,
		{&inSelf, &inIndices, &inInverse}, {&out});
	if (not semantic.isOk()) return {};
	if (not oa::detail::generatedAutogradAttach::FnMatrix::moeGather(
		out, inSelf, inIndices, inInverse, semantic.getValue()).isOk())
	{
		return {};
	}
	return out;
}

oa::Matrix oa::FnMatrix::moeGatherBwd(const oa::Matrix& inSource,
	const oa::Matrix& inInverse, oa::I32 inOutRows) {
	if (inSource.rank() != 2 or inInverse.rank() != 1 or
		(inSource.getDtype() != oa::ScalarType::Float32 and
		 inSource.getDtype() != oa::ScalarType::BFloat16) or
		inInverse.getDtype() != oa::ScalarType::UInt32 or
		inInverse.numElements() != inSource.size(0) or inOutRows <= 0 or
		inSource.size(0) % inOutRows != 0) {
		OaLogError(oa::LogComponent::Ml,
			"MoeGatherBwd: expected source[R,D], UInt32 inverse[R], R %% T == 0");
		return {};
	}
	const oa::U32 R = static_cast<oa::U32>(inSource.size(0));
	const oa::U32 D = static_cast<oa::U32>(inSource.size(1));
	const oa::U32 T = static_cast<oa::U32>(inOutRows);
	const oa::U32 K = R / T;
	const oa::U32 dtype = inSource.getDtype() == oa::ScalarType::BFloat16 ? 1U : 0U;
	auto& ctx = oa::ExecutionSession::getActive();
	oa::OpLoweringScope lowering(ctx);
	auto out = oa::FnMatrix::empty(oa::MatrixShape{inOutRows, D}, inSource.getDtype());
	struct { oa::U32 R, D, T, K, dtype; } push{R, D, T, K, dtype};
	oa::BufferAccess access[] = {
		oa::BufferAccess::Read, oa::BufferAccess::Read, oa::BufferAccess::Write};
	ctx.add( "MoeGatherBwd", {&inSource, &inInverse, &out}, access,
		&push, sizeof(push), divCeil(T * D, 256));
	if (not lowering.commit(
		oa::detail::opRegistry::FnMatrix::moeGatherBwd,
		{&inSource, &inInverse}, {&out},
		{oa::OpAttribute::fromSignedInteger("outputRows", inOutRows)}).isOk())
	{
		return {};
	}
	return out;
}
