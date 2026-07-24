#include <Oa/Runtime/Dnn.h>

#include <algorithm>

namespace {

void HashMix(OaU64& InOut, OaU64 InValue) {
	InOut ^= InValue;
	InOut *= 0x100000001b3ULL;
}

bool Consumes(const OaDnnOpDesc& InConsumer, OaDnnMatrixId InValue) {
	return std::find(InConsumer.Inputs.Begin(), InConsumer.Inputs.End(), InValue)	!= InConsumer.Inputs.End();
}

bool SingleEdge(const OaDnnOpDesc& InA, const OaDnnOpDesc& InB) {
	return InA.Outputs.Size() == 1U and Consumes(InB, InA.Outputs[0]);
}

OaDnnPartition Portable(OaU32 InOp) {
	OaDnnPartition p;
	p.Ops.PushBack(InOp);
	return p;
}

} // Namespace

OaStatus OaDnnGraph::AddMatrix(const OaDnnMatrixDesc& InMatrix) {
	if (InMatrix.Id == OaInvalidDnnMatrixId or InMatrix.Shape.Rank <= 0
		or InMatrix.Shape.Rank > 4 or InMatrix.Shape.NumElements() <= 0
	) {
		return OaStatus::InvalidArgument("OaDnn matrix has an invalid id or shape");
	}
	if (FindMatrix(InMatrix.Id) != nullptr) {
		return OaStatus::InvalidArgument("OaDnn matrix ids must be unique");
	}
	Matrices_.PushBack(InMatrix);
	return OaStatus::Ok();
}

OaStatus OaDnnGraph::AddOp(const OaDnnOpDesc& InOp) {
	if (InOp.Inputs.Empty() or InOp.Outputs.Empty()) {
		return OaStatus::InvalidArgument("OaDnn operation needs inputs and outputs");
	}
	Ops_.PushBack(InOp);
	return OaStatus::Ok();
}

const OaDnnMatrixDesc* OaDnnGraph::FindMatrix(OaDnnMatrixId InId) const {
	for (const auto& matrix : Matrices_) {
		if (matrix.Id == InId) return &matrix;
	}
	return nullptr;
}

OaStatus OaDnnGraph::Validate() const {
	if (Matrices_.Empty() or Ops_.Empty()) {
		return OaStatus::InvalidArgument("OaDnn graph cannot be empty");
	}
	OaVec<OaDnnMatrixId> produced;
	for (OaU32 opIdx = 0; opIdx < Ops_.Size(); ++opIdx) {
		const auto& op = Ops_[opIdx];
		for (auto id : op.Inputs) {
			const auto* matrix = FindMatrix(id);
			if (matrix == nullptr) return OaStatus::InvalidArgument("OaDnn op has a dangling input");
			const bool hasProducer = std::find(produced.Begin(), produced.End(), id) != produced.End();
			if (not matrix->External and not hasProducer) {
				return OaStatus::InvalidArgument("OaDnn op consumes a value before its producer");
			}
		}
		for (auto id : op.Outputs) {
			const auto* matrix = FindMatrix(id);
			if (matrix == nullptr) return OaStatus::InvalidArgument("OaDnn op has a dangling output");
			if (std::find(produced.Begin(), produced.End(), id) != produced.End()) {
				return OaStatus::InvalidArgument("OaDnn graph violates single-assignment output semantics");
			}
			produced.PushBack(id);
		}
	}
	return OaStatus::Ok();
}

OaResult<OaDnnPlan> OaDnnPlanner::Plan(const OaDnnGraph& InGraph, const OaDnnPolicy& InPolicy) {
	const auto status = InGraph.Validate();
	if (not status.IsOk()) {
		return status;
	}

	OaDnnPlan plan;
	OaU64 hash = 0xcbf29ce484222325ULL;
	for (const auto& matrix : InGraph.Matrices()) {
		HashMix(hash, matrix.Id); HashMix(hash, static_cast<OaU8>(matrix.Dtype));
		HashMix(hash, matrix.Shape.Rank);
		for (OaI32 d = 0; d < matrix.Shape.Rank; ++d) HashMix(hash, matrix.Shape[d]);
		HashMix(hash, matrix.External); HashMix(hash, matrix.Virtual);
	}

	const auto ops = InGraph.Ops();
	for (OaU32 i = 0; i < ops.Size();) {
		const auto& op = ops[i];
		HashMix(hash, static_cast<OaU8>(op.Type));

		// Three projections sharing one activation are one packed-QKV region.
		if (i + 2U < ops.Size() and op.Type == OaDnnOpType::Matmul
			and ops[i + 1U].Type == OaDnnOpType::Matmul
			and ops[i + 2U].Type == OaDnnOpType::Matmul
			and not op.Inputs.Empty() and not ops[i + 1U].Inputs.Empty()
			and not ops[i + 2U].Inputs.Empty()
			and op.Inputs[0] == ops[i + 1U].Inputs[0]
			and op.Inputs[0] == ops[i + 2U].Inputs[0]) {
			OaDnnPartition p; p.Engine = OaDnnEngineType::PackedQkv;
			p.Ops = {i, i + 1U, i + 2U};
			if (op.Training) p.SavedForBackward.PushBack(op.Inputs[0]);
			plan.Partitions.PushBack(std::move(p)); i += 3U; continue;
		}

		// gate matmul, up matmul, SiLU(gate), multiply(silu, up)
		if (i + 3U < ops.Size() and op.Type == OaDnnOpType::Matmul
			and ops[i + 1U].Type == OaDnnOpType::Matmul
			and ops[i + 2U].Type == OaDnnOpType::Silu
			and ops[i + 3U].Type == OaDnnOpType::Multiply
			and not op.Inputs.Empty() and not ops[i + 1U].Inputs.Empty()
			and op.Inputs[0] == ops[i + 1U].Inputs[0]
			and SingleEdge(op, ops[i + 2U])
			and SingleEdge(ops[i + 2U], ops[i + 3U])
			and SingleEdge(ops[i + 1U], ops[i + 3U])) {
			OaDnnPartition p; p.Engine = OaDnnEngineType::GatedFfn;
			p.Ops = {i, i + 1U, i + 2U, i + 3U};
			if (op.Training) p.SavedForBackward.PushBack(op.Inputs[0]);
			plan.Partitions.PushBack(std::move(p)); i += 4U; continue;
		}

		if (op.Type == OaDnnOpType::Matmul and i + 1U < ops.Size()
			and ops[i + 1U].Type == OaDnnOpType::BiasAdd and SingleEdge(op, ops[i + 1U])) {
			OaDnnPartition p; p.Engine = OaDnnEngineType::BlasLtEpilogue;
			p.Ops = {i, i + 1U};
			if (i + 2U < ops.Size()
				and (ops[i + 2U].Type == OaDnnOpType::Relu
					or ops[i + 2U].Type == OaDnnOpType::Gelu
					or ops[i + 2U].Type == OaDnnOpType::Silu)
				and SingleEdge(ops[i + 1U], ops[i + 2U])) {
				p.Ops.PushBack(i + 2U);
				if (op.Training and not InPolicy.AllowRecompute) {
					p.SavedForBackward.PushBack(ops[i + 1U].Outputs[0]);
				}
			}
			const OaU32 consumed = static_cast<OaU32>(p.Ops.Size());
			plan.Partitions.PushBack(std::move(p)); i += consumed; continue;
		}

		if (op.Type == OaDnnOpType::FlashAttentionCausal) {
			auto p = Portable(i); p.Engine = OaDnnEngineType::FlashAttention;
			if (op.Training) {
				for (auto id : op.Outputs) p.SavedForBackward.PushBack(id);
			}
			plan.Partitions.PushBack(std::move(p)); ++i; continue;
		}
		if (op.Type == OaDnnOpType::GroupedGemm) {
			auto p = Portable(i); p.Engine = OaDnnEngineType::GroupedMoe;
			if (op.Training and not op.Inputs.Empty()) p.SavedForBackward.PushBack(op.Inputs[0]);
			plan.Partitions.PushBack(std::move(p)); ++i; continue;
		}
		if (op.Type == OaDnnOpType::Add and i + 1U < ops.Size()
			and ops[i + 1U].Type == OaDnnOpType::RmsNorm and SingleEdge(op, ops[i + 1U])) {
			OaDnnPartition p; p.Engine = OaDnnEngineType::ResidualNorm;
			p.Ops = {i, i + 1U};
			plan.Partitions.PushBack(std::move(p)); i += 2U; continue;
		}

		plan.Partitions.PushBack(Portable(i)); ++i;
	}
	plan.GraphHash = hash;
	(void)InPolicy.MaxWorkspaceBytes;
	(void)InPolicy.RequireDeterministic;
	return plan;
}
