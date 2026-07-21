// GPU-native sparse-MoE systems primitives.

#include <Oa/Core/FnMatrix.h>
#include <Oa/Ml/FnMatrix.h>
#include <Oa/Ml/Autograd.h>
#include <Oa/Ml/Autograd/Matrix/AutogradMatrix.h>
#include <Oa/Core/BufferAccess.h>
#include <Oa/Core/Log.h>
#include <Oa/Runtime/Context.h>

static OaU32 DivCeil(OaU32 InA, OaU32 InB) { return (InA + InB - 1) / InB; }

OaMatrix OaFnMatrix::MoeRouteWeights(const OaMatrix& InProbs,	const OaMatrix& InExpertIndices) {
	if (InProbs.Rank() != 2 or InExpertIndices.Rank() != 2 or
		(InProbs.GetDtype() != OaScalarType::Float32 and
		 InProbs.GetDtype() != OaScalarType::BFloat16) or
		InExpertIndices.GetDtype() != OaScalarType::Int32 or
		InProbs.Size(0) != InExpertIndices.Size(0) or
		InExpertIndices.Size(1) <= 0 or InExpertIndices.Size(1) > InProbs.Size(1)) {
		OA_LOG_ERROR(OaLogComponent::ML,
			"MoeRouteWeights: expected Probs[T,E] and Int32 Indices[T,K], 1 <= K <= E");
		return {};
	}
	const OaU32 T = static_cast<OaU32>(InProbs.Size(0));
	const OaU32 E = static_cast<OaU32>(InProbs.Size(1));
	const OaU32 K = static_cast<OaU32>(InExpertIndices.Size(1));
	auto out = OaFnMatrix::Empty(InExpertIndices.GetShape(), InProbs.GetDtype());
	struct { OaU32 T, E, K; } push{T, E, K};
	OaBufferAccess access[] = {
		OaBufferAccess::Read, OaBufferAccess::Read, OaBufferAccess::Write};
	auto& ctx = OaContext::GetDefault();
	ctx.Add("MoeRouteWeights", {&InProbs, &InExpertIndices, &out}, access,
		&push, sizeof(push), DivCeil(T, 256));
	if (OaFnAutograd::IsEnabled() and InProbs.RequiresGrad()) {
		auto gradFn = OaMakeSharedPtr<OaGradMoeRouteWeights>();
		gradFn->Saved_ = OaVec<OaMatrix>{InProbs, InExpertIndices, out};
		gradFn->SetGraphInputs(OaVec<OaMatrix>{InProbs, InExpertIndices});
		gradFn->SequenceNr_ = OaFnAutograd::NextSeq();
		gradFn->OutputShape_ = out.GetShape();
		out.MutAutograd().GradFn = gradFn;
	}
	return out;
}

OaMatrix OaFnMatrix::MoeRouteWeightsBwd(const OaMatrix& InDOut,	const OaMatrix& InProbs, const OaMatrix& InExpertIndices,	const OaMatrix& InRouteWeights) {
	if (InProbs.Rank() != 2 or InDOut.GetShape() != InExpertIndices.GetShape() or
		InDOut.GetShape() != InRouteWeights.GetShape() or
		InDOut.GetDtype() != InProbs.GetDtype() or
		InRouteWeights.GetDtype() != InProbs.GetDtype() or
		InExpertIndices.GetDtype() != OaScalarType::Int32 or
		InProbs.Size(0) != InExpertIndices.Size(0)) {
		OA_LOG_ERROR(OaLogComponent::ML, "MoeRouteWeightsBwd: incompatible tensors");
		return {};
	}
	const OaU32 T = static_cast<OaU32>(InProbs.Size(0));
	const OaU32 E = static_cast<OaU32>(InProbs.Size(1));
	const OaU32 K = static_cast<OaU32>(InExpertIndices.Size(1));
	auto out = OaFnMatrix::Empty(InProbs.GetShape(), InProbs.GetDtype());
	struct { OaU32 T, E, K; } push{T, E, K};
	OaBufferAccess access[] = {OaBufferAccess::Read, OaBufferAccess::Read,
		OaBufferAccess::Read, OaBufferAccess::Read, OaBufferAccess::Write};
	auto& ctx = OaContext::GetDefault();
	ctx.Add("MoeRouteWeightsBwd",
		{&InDOut, &InProbs, &InExpertIndices, &InRouteWeights, &out}, access,
		&push, sizeof(push), DivCeil(T * E, 256));
	return out;
}

OaMatrix OaFnMatrix::GroupedGemmM(const OaMatrix& InX, const OaMatrix& InWeight,
	const OaMatrix& InOffsets) {
	if (InX.Rank() != 2 or InWeight.Rank() != 3 or InOffsets.Rank() != 1 or
		InOffsets.GetDtype() != OaScalarType::UInt32 or InX.GetDtype() != InWeight.GetDtype() or
		InX.Size(1) != InWeight.Size(2) or InOffsets.Size(0) != InWeight.Size(0) + 1) {
		OA_LOG_ERROR(OaLogComponent::ML,	"GroupedGemmM: expected X[R,K], W[E,N,K], UInt32 offsets[E+1]");
		return {};
	}
	const OaU32 R = static_cast<OaU32>(InX.Size(0));
	const OaU32 K = static_cast<OaU32>(InX.Size(1));
	const OaU32 E = static_cast<OaU32>(InWeight.Size(0));
	const OaU32 N = static_cast<OaU32>(InWeight.Size(1));
	auto out = OaFnMatrix::Empty(OaMatrixShape{R, N}, InX.GetDtype());
	struct { OaU32 R, N, K, E; } push{R, N, K, E};
	OaBufferAccess access[] = {OaBufferAccess::Read, OaBufferAccess::Read,
		OaBufferAccess::Read, OaBufferAccess::Write};
	auto& ctx = OaContext::GetDefault();
	ctx.Add("GroupedGemmM", {&InX, &InWeight, &InOffsets, &out}, access,
		&push, sizeof(push), DivCeil(R, 32), DivCeil(N, 32), 1);
	if (OaFnAutograd::IsEnabled() and (InX.RequiresGrad() or InWeight.RequiresGrad())) {
		auto gradFn = OaMakeSharedPtr<OaGradGroupedGemmM>();
		gradFn->Saved_ = OaVec<OaMatrix>{InX, InWeight, InOffsets};
		gradFn->SetGraphInputs(OaVec<OaMatrix>{InX, InWeight, InOffsets});
		gradFn->SequenceNr_ = OaFnAutograd::NextSeq();
		gradFn->OutputShape_ = out.GetShape();
		out.MutAutograd().GradFn = gradFn;
	}
	return out;
}

OaMatrix OaFnMatrix::GroupedLinearM(const OaMatrix& InX, const OaMatrix& InWeight, const OaMatrix& InBias, const OaMatrix& InOffsets) {
	if (InX.Rank() != 2 or InWeight.Rank() != 3 or InBias.Rank() != 2 or
		InOffsets.Rank() != 1 or InOffsets.GetDtype() != OaScalarType::UInt32 or
		InX.GetDtype() != InWeight.GetDtype() or InX.GetDtype() != InBias.GetDtype() or
		InX.Size(1) != InWeight.Size(2) or InBias.Size(0) != InWeight.Size(0) or
		InBias.Size(1) != InWeight.Size(1) or InOffsets.Size(0) != InWeight.Size(0) + 1) {
		OA_LOG_ERROR(OaLogComponent::ML,	"GroupedLinearM: expected X[R,K], W[E,N,K], Bias[E,N], UInt32 offsets[E+1]");
		return {};
	}
	const OaU32 R = static_cast<OaU32>(InX.Size(0));
	const OaU32 K = static_cast<OaU32>(InX.Size(1));
	const OaU32 E = static_cast<OaU32>(InWeight.Size(0));
	const OaU32 N = static_cast<OaU32>(InWeight.Size(1));
	auto out = OaFnMatrix::Empty(OaMatrixShape{R, N}, InX.GetDtype());
	struct { OaU32 R, N, K, E; } push{R, N, K, E};
	OaBufferAccess access[] = {OaBufferAccess::Read, OaBufferAccess::Read,
		OaBufferAccess::Read, OaBufferAccess::Read, OaBufferAccess::Write};
	auto& ctx = OaContext::GetDefault();
	ctx.Add("GroupedLinearM", {&InX, &InWeight, &InBias, &InOffsets, &out}, access,
		&push, sizeof(push), DivCeil(R, 32), DivCeil(N, 32), 1);
	if (OaFnAutograd::IsEnabled() and
		(InX.RequiresGrad() or InWeight.RequiresGrad() or InBias.RequiresGrad())) {
		auto gradFn = OaMakeSharedPtr<OaGradGroupedLinearM>();
		gradFn->Saved_ = OaVec<OaMatrix>{InX, InWeight, InBias, InOffsets};
		gradFn->SetGraphInputs(OaVec<OaMatrix>{InX, InWeight, InBias, InOffsets});
		gradFn->SequenceNr_ = OaFnAutograd::NextSeq();
		gradFn->OutputShape_ = out.GetShape();
		out.MutAutograd().GradFn = gradFn;
	}
	return out;
}

OaFnMatrix::OaGroupedGemmMBwdResult OaFnMatrix::GroupedGemmMBwd(
	const OaMatrix& InDOut, const OaMatrix& InX, const OaMatrix& InWeight,
	const OaMatrix& InOffsets) {
	if (InDOut.Rank() != 2 or InX.Rank() != 2 or InWeight.Rank() != 3 or
		InOffsets.Rank() != 1 or InOffsets.GetDtype() != OaScalarType::UInt32 or
		InDOut.GetDtype() != InX.GetDtype() or InX.GetDtype() != InWeight.GetDtype() or
		InDOut.Size(0) != InX.Size(0) or InDOut.Size(1) != InWeight.Size(1) or
		InX.Size(1) != InWeight.Size(2) or InOffsets.Size(0) != InWeight.Size(0) + 1) {
		OA_LOG_ERROR(OaLogComponent::ML, "GroupedGemmMBwd: incompatible tensors");
		return {};
	}
	const OaU32 R = static_cast<OaU32>(InX.Size(0));
	const OaU32 K = static_cast<OaU32>(InX.Size(1));
	const OaU32 E = static_cast<OaU32>(InWeight.Size(0));
	const OaU32 N = static_cast<OaU32>(InWeight.Size(1));
	OaGroupedGemmMBwdResult result;
	result.DInput = OaFnMatrix::Empty(InX.GetShape(), InX.GetDtype());
	result.DWeight = OaFnMatrix::Empty(InWeight.GetShape(), InWeight.GetDtype());
	struct { OaU32 R, N, K, E; } push{R, N, K, E};
	auto& ctx = OaContext::GetDefault();
	{
		OaBufferAccess access[] = {OaBufferAccess::Read, OaBufferAccess::Read,
			OaBufferAccess::Read, OaBufferAccess::Write};
		ctx.Add("GroupedGemmMDataBwd", {&InDOut, &InWeight, &InOffsets, &result.DInput},
			access, &push, sizeof(push), DivCeil(R, 32), DivCeil(K, 32), 1);
	}
	{
		OaBufferAccess access[] = {OaBufferAccess::Read, OaBufferAccess::Read,
			OaBufferAccess::Read, OaBufferAccess::Write};
		ctx.Add("GroupedGemmMWeightBwd", {&InDOut, &InX, &InOffsets, &result.DWeight},
			access, &push, sizeof(push), DivCeil(N, 32), DivCeil(K, 32), E);
	}
	return result;
}

OaFnMatrix::OaGroupedLinearMBwdResult OaFnMatrix::GroupedLinearMBwd(
	const OaMatrix& InDOut, const OaMatrix& InX, const OaMatrix& InWeight,
	const OaMatrix& InOffsets) {
	if (InDOut.Rank() != 2 or InX.Rank() != 2 or InWeight.Rank() != 3 or
		InOffsets.Rank() != 1 or InOffsets.GetDtype() != OaScalarType::UInt32 or
		InDOut.GetDtype() != InX.GetDtype() or InX.GetDtype() != InWeight.GetDtype() or
		InDOut.Size(0) != InX.Size(0) or InDOut.Size(1) != InWeight.Size(1) or
		InX.Size(1) != InWeight.Size(2) or InOffsets.Size(0) != InWeight.Size(0) + 1) {
		OA_LOG_ERROR(OaLogComponent::ML, "GroupedLinearMBwd: incompatible tensors");
		return {};
	}
	const OaU32 R = static_cast<OaU32>(InX.Size(0));
	const OaU32 K = static_cast<OaU32>(InX.Size(1));
	const OaU32 E = static_cast<OaU32>(InWeight.Size(0));
	const OaU32 N = static_cast<OaU32>(InWeight.Size(1));
	OaGroupedLinearMBwdResult result;
	result.DInput = OaFnMatrix::Empty(InX.GetShape(), InX.GetDtype());
	result.DWeight = OaFnMatrix::Empty(InWeight.GetShape(), InWeight.GetDtype());
	result.DBias = OaFnMatrix::Empty(OaMatrixShape{E, N}, InDOut.GetDtype());
	struct { OaU32 R, N, K, E; } push{R, N, K, E};
	auto& ctx = OaContext::GetDefault();
	{
		OaBufferAccess access[] = {OaBufferAccess::Read, OaBufferAccess::Read,
			OaBufferAccess::Read, OaBufferAccess::Write};
		ctx.Add("GroupedGemmMDataBwd",
			{&InDOut, &InWeight, &InOffsets, &result.DInput}, access,
			&push, sizeof(push), DivCeil(R, 32), DivCeil(K, 32), 1);
	}
	{
		OaBufferAccess access[] = {OaBufferAccess::Read, OaBufferAccess::Read,
			OaBufferAccess::Read, OaBufferAccess::Write, OaBufferAccess::Write};
		ctx.Add("GroupedLinearMWeightBiasBwd",
			{&InDOut, &InX, &InOffsets, &result.DWeight, &result.DBias}, access,
			&push, sizeof(push), DivCeil(N, 32), DivCeil(K, 32), E);
	}
	return result;
}

OaMatrix OaFnMatrix::GroupedLinearMBiasBwd(const OaMatrix& InDOut,
	const OaMatrix& InOffsets, OaI32 InNumExperts) {
	if (InDOut.Rank() != 2 or InOffsets.Rank() != 1 or
		InOffsets.GetDtype() != OaScalarType::UInt32 or InNumExperts <= 0 or
		InOffsets.Size(0) != InNumExperts + 1) return {};
	const OaU32 R = static_cast<OaU32>(InDOut.Size(0));
	const OaU32 N = static_cast<OaU32>(InDOut.Size(1));
	const OaU32 E = static_cast<OaU32>(InNumExperts);
	auto out = OaFnMatrix::Empty(OaMatrixShape{E, N}, InDOut.GetDtype());
	struct { OaU32 R, N, E; } push{R, N, E};
	OaBufferAccess access[] = {
		OaBufferAccess::Read, OaBufferAccess::Read, OaBufferAccess::Write};
	auto& ctx = OaContext::GetDefault();
	ctx.Add("GroupedLinearMBiasBwd", {&InDOut, &InOffsets, &out}, access,
		&push, sizeof(push), DivCeil(N, 16), DivCeil(E, 16), 1);
	return out;
}

OaMatrix OaFnMatrix::MoeCombine(const OaMatrix& InPacked,
	const OaMatrix& InRouteGate, const OaMatrix& InInverse,
	const OaMatrix& InPackedSlot) {
	if (InPacked.Rank() != 2 or InRouteGate.Rank() != 2 or
		InPacked.GetDtype() != InRouteGate.GetDtype() or
		InInverse.Rank() != 1 or InPackedSlot.Rank() != 1 or
		InInverse.GetDtype() != OaScalarType::UInt32 or
		InPackedSlot.GetDtype() != OaScalarType::UInt32 or
		InPacked.Size(0) != InRouteGate.NumElements() or
		InInverse.Size(0) != InPacked.Size(0) or
		InPackedSlot.Size(0) != InPacked.Size(0)) {
		OA_LOG_ERROR(OaLogComponent::ML,
			"MoeCombine: expected Packed[R,D], Gate[T,K], UInt32 maps[R]");
		return {};
	}
	const OaU32 T = static_cast<OaU32>(InRouteGate.Size(0));
	const OaU32 K = static_cast<OaU32>(InRouteGate.Size(1));
	const OaU32 D = static_cast<OaU32>(InPacked.Size(1));
	const OaU32 R = static_cast<OaU32>(InPacked.Size(0));
	auto out = OaFnMatrix::Empty(OaMatrixShape{T, D}, InPacked.GetDtype());
	struct { OaU32 T, K, D, R; } push{T, K, D, R};
	OaBufferAccess access[] = {OaBufferAccess::Read, OaBufferAccess::Read,
		OaBufferAccess::Read, OaBufferAccess::Write};
	auto& ctx = OaContext::GetDefault();
	ctx.Add("MoeCombine", {&InPacked, &InRouteGate, &InInverse, &out},
		access, &push, sizeof(push), DivCeil(T * D, 256));
	if (OaFnAutograd::IsEnabled() and
		(InPacked.RequiresGrad() or InRouteGate.RequiresGrad())) {
		auto gradFn = OaMakeSharedPtr<OaGradMoeCombine>();
		gradFn->Saved_ = OaVec<OaMatrix>{InPacked, InRouteGate, InInverse, InPackedSlot};
		gradFn->SetGraphInputs(
			OaVec<OaMatrix>{InPacked, InRouteGate, InInverse, InPackedSlot});
		gradFn->SequenceNr_ = OaFnAutograd::NextSeq();
		gradFn->OutputShape_ = out.GetShape();
		out.MutAutograd().GradFn = gradFn;
	}
	return out;
}

OaFnMatrix::OaMoeCombineBwdResult OaFnMatrix::MoeCombineBwd(
	const OaMatrix& InDOut, const OaMatrix& InPacked,
	const OaMatrix& InRouteGate, const OaMatrix& InInverse,
	const OaMatrix& InPackedSlot) {
	OaMoeCombineBwdResult result;
	if (InDOut.Rank() != 2 or InPacked.Rank() != 2 or
		InRouteGate.Rank() != 2 or InDOut.GetDtype() != InPacked.GetDtype() or
		InPacked.GetDtype() != InRouteGate.GetDtype() or
		InInverse.Rank() != 1 or InPackedSlot.Rank() != 1 or
		InInverse.GetDtype() != OaScalarType::UInt32 or
		InPackedSlot.GetDtype() != OaScalarType::UInt32 or
		InDOut.Size(0) != InRouteGate.Size(0) or
		InDOut.Size(1) != InPacked.Size(1) or
		InPacked.Size(0) != InRouteGate.NumElements() or
		InInverse.Size(0) != InPacked.Size(0) or
		InPackedSlot.Size(0) != InPacked.Size(0)) {
		OA_LOG_ERROR(OaLogComponent::ML,
			"MoeCombineBwd: incompatible dOut, Packed, Gate, or UInt32 maps");
		return result;
	}
	result.DPacked = OaFnMatrix::Empty(InPacked.GetShape(), InPacked.GetDtype());
	result.DRouteGate = OaFnMatrix::Empty(
		InRouteGate.GetShape(), InRouteGate.GetDtype());
	const OaU32 T = static_cast<OaU32>(InRouteGate.Size(0));
	const OaU32 K = static_cast<OaU32>(InRouteGate.Size(1));
	const OaU32 D = static_cast<OaU32>(InPacked.Size(1));
	const OaU32 R = static_cast<OaU32>(InPacked.Size(0));
	struct { OaU32 T, K, D, R; } push{T, K, D, R};
	OaBufferAccess access[] = {OaBufferAccess::Read, OaBufferAccess::Read,
		OaBufferAccess::Read, OaBufferAccess::Read, OaBufferAccess::Read,
		OaBufferAccess::Write, OaBufferAccess::Write};
	auto& ctx = OaContext::GetDefault();
	ctx.Add("MoeCombineBwd", {&InDOut, &InPacked, &InRouteGate, &InInverse,
		&InPackedSlot, &result.DPacked, &result.DRouteGate}, access,
		&push, sizeof(push), DivCeil(R * D, 256));
	return result;
}

OaMatrix OaFnMatrix::ScatterAddRows(const OaMatrix& InSource,
	const OaMatrix& InIndices, OaI32 InOutRows) {
	if (InSource.Rank() != 2 or InIndices.Rank() != 1 or
		(InSource.GetDtype() != OaScalarType::Float32 and
		 InSource.GetDtype() != OaScalarType::BFloat16) or
		InIndices.GetDtype() != OaScalarType::UInt32 or
		InIndices.Size(0) != InSource.Size(0) or InOutRows <= 0) {
		OA_LOG_ERROR(OaLogComponent::ML,
			"ScatterAddRows: expected Source[R,D], UInt32 Indices[R], OutRows > 0");
		return {};
	}
	const OaU32 R = static_cast<OaU32>(InSource.Size(0));
	const OaU32 D = static_cast<OaU32>(InSource.Size(1));
	const OaU32 T = static_cast<OaU32>(InOutRows);
	const OaU32 dtype = InSource.GetDtype() == OaScalarType::BFloat16 ? 1U : 0U;
	auto out = OaFnMatrix::Zeros(OaMatrixShape{InOutRows, D}, InSource.GetDtype());
	struct { OaU32 R, D, T, Dtype; } push{R, D, T, dtype};
	OaBufferAccess access[] = {
		OaBufferAccess::Read, OaBufferAccess::Read, OaBufferAccess::ReadWrite};
	auto& ctx = OaContext::GetDefault();
	ctx.Add("ScatterAddRows", {&InSource, &InIndices, &out}, access,
		&push, sizeof(push), DivCeil(R * D, 256));
	if (OaFnAutograd::IsEnabled() and InSource.RequiresGrad()) {
		auto gradFn = OaMakeSharedPtr<OaGradScatterAddRows>();
		gradFn->Saved_ = OaVec<OaMatrix>{InIndices};
		gradFn->SetGraphInputs(OaVec<OaMatrix>{InSource, InIndices});
		gradFn->SequenceNr_ = OaFnAutograd::NextSeq();
		gradFn->OutputShape_ = out.GetShape();
		out.MutAutograd().GradFn = gradFn;
	}
	return out;
}

OaMatrix OaFnMatrix::MoeGather(const OaMatrix& InSelf,const OaMatrix& InIndices, const OaMatrix& InInverse) {
	if (InSelf.Rank() != 2 or InIndices.Rank() != 1 or
		InInverse.Rank() != 1 or InIndices.GetDtype() != OaScalarType::UInt32 or
		InInverse.GetDtype() != OaScalarType::UInt32 or
		InInverse.NumElements() != InIndices.NumElements() or
		InSelf.Size(0) <= 0 or
		InIndices.NumElements() % InSelf.Size(0) != 0) {
		OA_LOG_ERROR(OaLogComponent::ML,	"MoeGather: expected Self[T,D] and UInt32 Indices/Inverse[R], R %% T == 0");
		return {};
	}
	const OaU32 R = static_cast<OaU32>(InIndices.NumElements());
	const OaU32 D = static_cast<OaU32>(InSelf.Size(1));
	auto out = OaFnMatrix::Empty(OaMatrixShape{R, D}, InSelf.GetDtype());
	struct { OaU32 NumIndices, RowSize, IndexDtype; } push{R, D, 1U};
	OaBufferAccess access[] = {OaBufferAccess::Read, OaBufferAccess::Read, OaBufferAccess::Write};
	auto& ctx = OaContext::GetDefault();
	ctx.Add("Gather", {&InSelf, &InIndices, &out}, access,
		&push, sizeof(push), DivCeil(R * D, 256));
	if (OaFnAutograd::IsEnabled() and InSelf.RequiresGrad()) {
		auto gradFn = OaMakeSharedPtr<OaGradMoeGather>();
		gradFn->Saved_ = OaVec<OaMatrix>{InSelf, InIndices, InInverse};
		gradFn->SetGraphInputs(OaVec<OaMatrix>{InSelf, InIndices, InInverse});
		gradFn->SequenceNr_ = OaFnAutograd::NextSeq();
		gradFn->OutputShape_ = out.GetShape();
		out.MutAutograd().GradFn = gradFn;
	}
	return out;
}

OaMatrix OaFnMatrix::MoeGatherBwd(const OaMatrix& InSource,
	const OaMatrix& InInverse, OaI32 InOutRows) {
	if (InSource.Rank() != 2 or InInverse.Rank() != 1 or
		(InSource.GetDtype() != OaScalarType::Float32 and
		 InSource.GetDtype() != OaScalarType::BFloat16) or
		InInverse.GetDtype() != OaScalarType::UInt32 or
		InInverse.NumElements() != InSource.Size(0) or InOutRows <= 0 or
		InSource.Size(0) % InOutRows != 0) {
		OA_LOG_ERROR(OaLogComponent::ML,
			"MoeGatherBwd: expected Source[R,D], UInt32 Inverse[R], R %% T == 0");
		return {};
	}
	const OaU32 R = static_cast<OaU32>(InSource.Size(0));
	const OaU32 D = static_cast<OaU32>(InSource.Size(1));
	const OaU32 T = static_cast<OaU32>(InOutRows);
	const OaU32 K = R / T;
	const OaU32 dtype = InSource.GetDtype() == OaScalarType::BFloat16 ? 1U : 0U;
	auto out = OaFnMatrix::Empty(OaMatrixShape{InOutRows, D}, InSource.GetDtype());
	struct { OaU32 R, D, T, K, Dtype; } push{R, D, T, K, dtype};
	OaBufferAccess access[] = {
		OaBufferAccess::Read, OaBufferAccess::Read, OaBufferAccess::Write};
	auto& ctx = OaContext::GetDefault();
	ctx.Add("MoeGatherBwd", {&InSource, &InInverse, &out}, access,
		&push, sizeof(push), DivCeil(T * D, 256));
	return out;
}
