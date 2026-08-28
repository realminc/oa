// Autograd core — thread-local tape state + oa::GradientTape::backward reverse walk.

#include <oa/ml/autograd.h>

#include <oa/core/fnMatrix.h>
#include <oa/core/log.h>
#include <oa/core/std/algo.h>
#include <oa/core/std/hashMap.h>
#include <oa/core/validation.h>
#include <oa/core/fnmatrix/fnMatrixInternal.h>
#include <oa/runtime/executionSession.h>
#include <oa/runtime/semanticGraph.h>

namespace {

thread_local bool   g_IsEnabled = false;
thread_local oa::U64  g_NextSeq     = 0;

} // anonymous

namespace oa {

namespace FnAutograd {

bool isEnabled() noexcept                { return g_IsEnabled; }
void setEnabled(bool inEnabled) noexcept { g_IsEnabled = inEnabled; }
oa::U64 nextSeq() noexcept                 { return ++g_NextSeq; }

oa::Status attachSemantic(
	const oa::SharedPtr<oa::GradNode>& inNode,
	oa::U32 inForwardOp,
	oa::U32 inOutputIndex)
{
	if (not inNode) {
		return oa::Status::error(oa::StatusCode::InvalidArgument,
			"semantic autograd attachment requires a node");
	}
	// A composed lowering suppresses its nested semantic operations while
	// retaining their ordinary autograd chain. The outer operation owns the
	// executable nodes; the nested gradient node therefore has no independent
	// forward semantic operation to attach.
	if (inForwardOp == oa::invalidSemanticOpId) {
		return oa::Status::ok();
	}
	auto* context = oa::ExecutionSession::getActivePtr();
	auto* graph = context
		? context->semanticGraph() : nullptr;
	if (not graph) {
		return oa::Status::error(oa::StatusCode::FailedPrecondition,
			"semantic autograd attachment requires an active graph");
	}
	const auto attached = graph->attachAutograd(
		inForwardOp, inOutputIndex, inNode->sequenceNr_);
	if (not attached.isOk()) return attached;
	inNode->forwardSemanticOp_ = inForwardOp;
	inNode->forwardSemanticOutput_ = inOutputIndex;
	inNode->forwardSemanticGeneration_ = graph->generation();
	return oa::Status::ok();
}

oa::Status completeSemantic(
	const oa::GradNode& inNode,
	oa::U32 inBackwardFirstOp,
	oa::U32 inBackwardOpCount)
{
	if (inNode.forwardSemanticOp_ == oa::invalidSemanticOpId) {
		return oa::Status::ok();
	}
	auto* context = oa::ExecutionSession::getActivePtr();
	auto* graph = context
		? context->semanticGraph() : nullptr;
	if (not graph) {
		return oa::Status::error(oa::StatusCode::FailedPrecondition,
			"semantic backward completion requires an active graph");
	}
	// Submission or clear() retires a semantic recording, but the numerical tape
	// may intentionally retain its values. Operation ids restart at zero, so
	// only an exact recording generation may complete the old attachment.
	if (inNode.forwardSemanticGeneration_ != graph->generation()) {
		return oa::Status::ok();
	}
	const auto completed = graph->completeAutograd(
		inNode.forwardSemanticOp_, inNode.sequenceNr_,
		inBackwardFirstOp, inBackwardOpCount);
	// Submission closes and clears the active semantic recording. A tape may
	// intentionally materialize its forward values before expanding backward
	// (Max is the canonical example), in which case the submitted forward
	// attachment no longer belongs to the new recording. Numerical backward
	// remains valid, but there is no live forward graph to amend.
	if (completed.getCode() == oa::StatusCode::NotFound
		and graph->autograd().empty())
	{
		return oa::Status::ok();
	}
	return completed;
}

} // namespace FnAutograd

} // namespace oa

namespace {

void topoCollect_(const oa::SharedPtr<oa::GradNode>& inNode,
                  oa::Vector<oa::SharedPtr<oa::GradNode>>& outNodes,
                  oa::HashSet<oa::GradNode*>& inVisited) {
	if (not inNode) return;
	auto* raw = inNode.get();
	if (inVisited.find(raw) != inVisited.end()) return;
	inVisited.insert(raw);
	outNodes.pushBack(inNode);
	for (const auto& input : inNode->graphInputs()) {
		if (auto childFn = input.getGradFn()) {
			topoCollect_(childFn, outNodes, inVisited);
		}
	}
}

} // anonymous

oa::Status oa::GradientTape::tryBackward(const oa::Matrix& inRoot) {
	if (inRoot.numElements() != 1) {
		return oa::Status::error(oa::StatusCode::InvalidArgument,
			"backward root must be a scalar");
	}
	auto rootFn = inRoot.getGradFn();
	if (not rootFn) return oa::Status::ok();

	oa::Vector<oa::SharedPtr<oa::GradNode>> topo;
	oa::HashSet<oa::GradNode*> visited;
	topoCollect_(rootFn, topo, visited);
	oa::sort(topo.begin(), topo.end(),
		[](const oa::SharedPtr<oa::GradNode>& a, const oa::SharedPtr<oa::GradNode>& b) {
			return a->sequenceNr_ < b->sequenceNr_;
		});
	// Preflight the entire tape before authoring any backward dispatch. A late
	// mutation failure must not leave a partially recorded gradient graph.
	for (const auto& node : topo) {
		const auto status = node->validateSavedVersions();
		if (not status.isOk()) return status;
	}

	oa::HashMap<oa::GradNode*, oa::Matrix> gradMap;
	gradMap.emplace(rootFn.get(), oa::FnMatrix::full(inRoot.getShape(), 1.0, inRoot.getDtype()));

	oa::GradNo noGradGuard;
	for (oa::I64 i = static_cast<oa::I64>(topo.size()) - 1; i >= 0; --i) {
		auto* fn = topo[static_cast<oa::Usize>(i)].get();
		auto it = gradMap.find(fn);
		if (it == gradMap.end()) continue;
		oa::Matrix upstream = it->second;
		auto* semanticGraph =
			oa::ExecutionSession::getActive().semanticGraph();
		const auto backwardFirst = semanticGraph
			? semanticGraph->operationCount() : 0U;

		// Robustness: normalize the upstream d (dout for this node) to the exact shape
		// the node was attached with. view/reshape chains in forward (very common in
		// Mamba3 preprocess, post-siso output path, LM wrapper, and everywhere) can
		// deliver a d with viewed shape (different rank/leading dims but same numel)
		// via shared autograd_ meta. Without this, generic and custom Backwards receive
		// shape-mismatched inUpstream, leading to bad bcast dispatches ("bad optional
		// access") or zero/ garbage grads on params.
		if (fn->outputShape_.numElements() > 0 && upstream.numElements() == fn->outputShape_.numElements() && upstream.getShape() != fn->outputShape_) {
			upstream = upstream.reshape(fn->outputShape_);
		}
		auto& inputs = fn->mutGraphInputs();
		oa::Vector<oa::Matrix> inGrads(inputs.size());
		fn->backward(upstream, inGrads);

		for (oa::Usize j = 0; j < inGrads.size(); ++j) {
			auto& input = inputs[j];
			if (not input.requiresGrad()) continue;
			if (inGrads[j].isEmpty()) continue;

			// Robustness for every edge: the d produced for this graph input (which
			// may have been a reshape/view of a param or activation at forward time)
			// is normalized to the *recorded input desc's shape* before accumulate or
			// feeding as upstream to the previous gradfn. This ensures leaf params
			// (e.g. Mamba B_bias passed as 2D) get contribs shaped for their registered
			// grad buf, and slice/reshape producers get correctly shaped dIn for their
			// bwd (e.g. SliceBwd).
			oa::Matrix d = inGrads[j];
			if (d.numElements() == input.numElements() && d.getShape() != input.getShape()) {
				d = d.reshape(input.getShape());
			}

			if (input.isLeaf()) {
				input.accumulateGrad(d);
				continue;
			}
			auto childFn = input.getGradFn();
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
				oa::Matrix dAcc = d;
				if (dAcc.numElements() == existing->second.numElements() && dAcc.getShape() != existing->second.getShape()) {
					dAcc = dAcc.reshape(existing->second.getShape());
				}
				existing->second = oa::FnMatrix::add(existing->second, dAcc);
			}
		}

		if (semanticGraph) {
			const auto backwardEnd = semanticGraph->operationCount();
			const auto completed = oa::FnAutograd::completeSemantic(
				*fn, backwardFirst, backwardEnd - backwardFirst);
			if (not completed.isOk()) {
				return completed;
			}
		}

		// release saved tensors and graph inputs now that this node is fully
		// processed. This breaks the reference chain from grad nodes back to
		// intermediate forward tensors, allowing them to be freed during the
		// walk instead of only after the entire tape is destroyed.
		fn->clearTensors();
	}
	oa::FnMatrix::flushDeferredAccum();
	return oa::Status::ok();
}

void oa::GradientTape::backward(const oa::Matrix& inRoot) {
	const auto status = tryBackward(inRoot);
	if (not status.isOk()) {
		OaLogError(oa::LogComponent::Ml,
			"autograd backward failed: %s", status.getMessage().cStr());
	}
}
