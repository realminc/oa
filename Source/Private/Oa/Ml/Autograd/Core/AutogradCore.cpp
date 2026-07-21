// OaAutogradCore — thread-local tape state + OaGradientTape::Backward reverse walk.

#include <Oa/Ml/Autograd.h>

#include <Oa/Core/FnMatrix.h>
#include <Oa/Core/Log.h>
#include <Oa/Core/Validation.h>
#include <Oa/Runtime/Context.h>
#include <Oa/Runtime/SemanticGraph.h>

#include <algorithm>
#include <unordered_map>
#include <unordered_set>

namespace {

thread_local bool   g_IsEnabled = false;
thread_local OaU64  g_NextSeq     = 0;

} // anonymous

namespace OaFnAutograd {

bool IsEnabled() noexcept                { return g_IsEnabled; }
void SetEnabled(bool InEnabled) noexcept { g_IsEnabled = InEnabled; }
OaU64 NextSeq() noexcept                 { return ++g_NextSeq; }

OaStatus AttachSemantic(
	const OaSharedPtr<OaGradNode>& InNode,
	OaSemanticOperationId InForwardOperation,
	OaU32 InOutputIndex)
{
	if (not InNode or InForwardOperation == OaInvalidSemanticOperationId) {
		return OaStatus::Error(OaStatusCode::InvalidArgument,
			"semantic autograd attachment requires a node and forward operation");
	}
	auto* context = OaContext::GetDefaultPtr();
	auto* graph = context ? context->SemanticGraph() : nullptr;
	if (not graph) {
		return OaStatus::Error(OaStatusCode::FailedPrecondition,
			"semantic autograd attachment requires an active graph");
	}
	const auto attached = graph->AttachAutograd(
		InForwardOperation, InOutputIndex, InNode->SequenceNr_);
	if (not attached.IsOk()) return attached;
	InNode->ForwardSemanticOperation_ = InForwardOperation;
	InNode->ForwardSemanticOutput_ = InOutputIndex;
	return OaStatus::Ok();
}

OaStatus CompleteSemantic(
	const OaGradNode& InNode,
	OaSemanticOperationId InBackwardFirstOperation,
	OaU32 InBackwardOperationCount)
{
	if (InNode.ForwardSemanticOperation_ == OaInvalidSemanticOperationId) {
		return OaStatus::Ok();
	}
	auto* context = OaContext::GetDefaultPtr();
	auto* graph = context ? context->SemanticGraph() : nullptr;
	if (not graph) {
		return OaStatus::Error(OaStatusCode::FailedPrecondition,
			"semantic backward completion requires an active graph");
	}
	const auto completed = graph->CompleteAutograd(
		InNode.ForwardSemanticOperation_, InNode.SequenceNr_,
		InBackwardFirstOperation, InBackwardOperationCount);
	// Execute() closes and clears the active semantic recording. A tape may
	// intentionally materialize its forward values before expanding backward
	// (Max is the canonical example), in which case the submitted forward
	// attachment no longer belongs to the new recording. Numerical backward
	// remains valid, but there is no live forward graph to amend.
	if (completed.GetCode() == OaStatusCode::NotFound
		and graph->Autograd().Empty())
	{
		return OaStatus::Ok();
	}
	return completed;
}

} // namespace OaFnAutograd

namespace {

void TopoCollect_(const OaSharedPtr<OaGradNode>& InNode,
                  OaVec<OaSharedPtr<OaGradNode>>& OutNodes,
                  std::unordered_set<OaGradNode*>& InVisited) {
	if (not InNode) return;
	auto* raw = InNode.get();
	if (InVisited.find(raw) != InVisited.end()) return;
	InVisited.insert(raw);
	OutNodes.PushBack(InNode);
	for (const auto& input : InNode->GraphInputs()) {
		if (auto childFn = input.GetGradFn()) {
			TopoCollect_(childFn, OutNodes, InVisited);
		}
	}
}

} // anonymous

void OaGradientTape::Backward(const OaMatrix& InRoot) {
	OA_ASSERT(InRoot.NumElements() == 1 and "backward root must be a scalar");
	auto rootFn = InRoot.GetGradFn();
	if (not rootFn) return;

	OaVec<OaSharedPtr<OaGradNode>> topo;
	std::unordered_set<OaGradNode*> visited;
	TopoCollect_(rootFn, topo, visited);
	std::sort(topo.begin(), topo.end(),
		[](const OaSharedPtr<OaGradNode>& a, const OaSharedPtr<OaGradNode>& b) {
			return a->SequenceNr_ < b->SequenceNr_;
		});

	std::unordered_map<OaGradNode*, OaMatrix> gradMap;
	gradMap.emplace(rootFn.get(), OaFnMatrix::Full(InRoot.GetShape(), 1.0, InRoot.GetDtype()));

	OaGradNo noGradGuard;
	for (OaI64 i = static_cast<OaI64>(topo.Size()) - 1; i >= 0; --i) {
		auto* fn = topo[static_cast<OaUsize>(i)].get();
		auto it = gradMap.find(fn);
		if (it == gradMap.end()) continue;
		OaMatrix upstream = it->second;
		auto* semanticGraph = OaContext::GetDefault().SemanticGraph();
		const auto backwardFirst = semanticGraph
			? semanticGraph->OperationCount() : 0U;

		// Robustness: normalize the upstream d (dout for this node) to the exact shape
		// the node was attached with. View/reshape chains in forward (very common in
		// Mamba3 preprocess, post-siso output path, LM wrapper, and everywhere) can
		// deliver a d with viewed shape (different rank/leading dims but same numel)
		// via shared Autograd_ meta. Without this, generic and custom Backwards receive
		// shape-mismatched InUpstream, leading to bad bcast dispatches ("bad optional
		// access") or zero/ garbage grads on params.
		if (fn->OutputShape_.NumElements() > 0 && upstream.NumElements() == fn->OutputShape_.NumElements() && upstream.GetShape() != fn->OutputShape_) {
			upstream = upstream.Reshape(fn->OutputShape_);
		}
		auto& inputs = fn->MutGraphInputs();
		OaVec<OaMatrix> inGrads(inputs.Size());
		fn->Backward(upstream, inGrads);

		for (OaUsize j = 0; j < inGrads.Size(); ++j) {
			auto& input = inputs[j];
			if (not input.RequiresGrad()) continue;
			if (inGrads[j].IsEmpty()) continue;

			// Robustness for every edge: the d produced for this graph input (which
			// may have been a reshape/view of a param or activation at forward time)
			// is normalized to the *recorded input desc's shape* before accumulate or
			// feeding as upstream to the previous gradfn. This ensures leaf params
			// (e.g. Mamba B_bias passed as 2D) get contribs shaped for their registered
			// grad buf, and slice/reshape producers get correctly shaped dIn for their
			// bwd (e.g. SliceBwd).
			OaMatrix d = inGrads[j];
			if (d.NumElements() == input.NumElements() && d.GetShape() != input.GetShape()) {
				d = d.Reshape(input.GetShape());
			}

			if (input.IsLeaf()) {
				input.AccumulateGrad(d);
				continue;
			}
			auto childFn = input.GetGradFn();
			if (not childFn) continue;
			auto* childRaw = childFn.get();
			auto existing = gradMap.find(childRaw);
			if (existing == gradMap.end()) {
				gradMap.emplace(childRaw, d);
			} else {
				// Fan-out accumulation: two edges can reach the same gradfn via
				// different *views* of the same tensor (same numel, different recorded
				// shape — common once broadcast ops deliver their previously-dropped
				// contribution). Add requires matching shapes, so normalize the new
				// contribution to the shape already accumulated before summing.
				OaMatrix dAcc = d;
				if (dAcc.NumElements() == existing->second.NumElements() && dAcc.GetShape() != existing->second.GetShape()) {
					dAcc = dAcc.Reshape(existing->second.GetShape());
				}
				existing->second = OaFnMatrix::Add(existing->second, dAcc);
			}
		}

		if (semanticGraph) {
			const auto backwardEnd = semanticGraph->OperationCount();
			const auto completed = OaFnAutograd::CompleteSemantic(
				*fn, backwardFirst, backwardEnd - backwardFirst);
			if (not completed.IsOk()) {
				OA_LOG_ERROR(OaLogComponent::ML,
					"semantic backward expansion failed: %s",
					completed.GetMessage().c_str());
				return;
			}
		}

		// Release saved tensors and graph inputs now that this node is fully
		// processed. This breaks the reference chain from grad nodes back to
		// intermediate forward tensors, allowing them to be freed during the
		// walk instead of only after the entire tape is destroyed.
		fn->ClearTensors();
	}
	OaFnMatrix::FlushDeferredAccum();
}
