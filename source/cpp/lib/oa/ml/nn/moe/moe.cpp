// oa::Moe — Mixture of Experts Implementation
//
// Routing is assembled from differentiable matrix operations and the sparse
// executor uses GPU-native planning, grouped expert projections, and a direct
// weighted combine. The dense executor remains a mathematical oracle over the
// same stacked expert parameters; both paths are end-to-end gradchecked.
//
// routing (top-k) is a non-differentiable decision: topK picks exact expert IDs
// and TopKMask turns them into a constant 0/1 mask. Gradient therefore flows only
// through selected gate magnitudes and selected experts — standard MoE semantics.
//
// The production path packs only selected token/expert pairs. Evaluating every
// expert for every token is opt-in and exists only for correctness comparison.
//
// stage 0 (anti-collapse, all inert at defaults so the oracle stays byte-identical):
//   • telemetry — routeStats() exposes per-expert load fraction/entropy/dead count;
//   • aux-loss-free balancing — a gradient-free per-expert selection bias nudged by
//     updateRoutingBias(); zero bias skips the Add entirely;
//   • differentiable switch aux loss + router z-loss — opt-in, added via auxLoss();
//   • shared always-on experts — DeepSeekMoE specialization anchor, default 0.

#include <oa/ml/nn/moe/moe.h>
#include <oa/ml/nn.h>
#include <oa/ml/module.h>
#include <oa/core/fnMatrix.h>
#include <oa/ml/fnMatrix.h>
#include <oa/core/std/format.h>
#include <oa/runtime/executionSession.h>
#include <cmath>
#include <stdexcept>

oa::Matrix oa::Moe::denseExpertDelta(const oa::Matrix& inNormed, const oa::Matrix& inGate) const {
	const oa::I64 T = inNormed.size(0);
	oa::Vec<oa::Matrix> expertOutputs;
	expertOutputs.reserve(numExperts_);
	for (oa::I32 e = 0; e < numExperts_; ++e) {
		auto gateW = oa::FnMatrix::reshape(
			oa::FnMatrix::slice(expertGateUpWeight(), 0, e, e + 1),
			oa::MatrixShape{2 * dFF_, dModel_});
		auto gateB = oa::FnMatrix::reshape(
			oa::FnMatrix::slice(expertGateUpBias(), 0, e, e + 1),
			oa::MatrixShape{2 * dFF_});
		auto downW = oa::FnMatrix::reshape(
			oa::FnMatrix::slice(expertDownWeight(), 0, e, e + 1),
			oa::MatrixShape{dModel_, dFF_});
		auto downB = oa::FnMatrix::reshape(
			oa::FnMatrix::slice(expertDownBias(), 0, e, e + 1),
			oa::MatrixShape{dModel_});
		// Raw oa::FnMatrix::linear is a dispatch primitive; oa::Linear normally owns
		// its autograd attachment. Build the dense oracle from differentiable
		// MatMulNt + broadcast Add so sliced stacked parameters receive gradients.
		auto gateUp = oa::FnMatrix::add(oa::FnMatrix::matMulNt(inNormed, gateW), gateB);
		auto hidden = oa::FnMatrix::siluMul(gateUp, static_cast<oa::U32>(dFF_));
		auto expertOut = oa::FnMatrix::add(oa::FnMatrix::matMulNt(hidden, downW), downB);
		expertOutputs.pushBack(oa::FnMatrix::reshape(expertOut,	oa::MatrixShape{T, 1, dModel_}));
	}
	auto allExperts = oa::FnMatrix::concat(oa::Span<oa::Matrix>(expertOutputs), 1);
	auto gate3d = oa::FnMatrix::reshape(inGate, oa::MatrixShape{T, numExperts_, 1});
	auto weighted = oa::FnMatrix::mul(allExperts, gate3d);
	return oa::FnMatrix::reshape(oa::FnMatrix::sum(weighted, 1),	oa::MatrixShape{T, dModel_});
}

oa::Matrix oa::Moe::sparseExpertDelta(const oa::Matrix& inNormed, const oa::Matrix& inRouteGate, const oa::Matrix& inTopKIndices) const {
	auto plan = oa::FnMatrix::moeExpertPlan(inTopKIndices, numExperts_);

	// Gather duplicated routed tokens into stable expert-major layout. Selected
	// route weights already have token-major [T,K] layout for direct combine.
	auto packedX = oa::FnMatrix::moeGather(inNormed, plan.packedToken, plan.inverse);

	auto gateUp = oa::FnMatrix::groupedLinearM(packedX, expertGateUpWeight(), expertGateUpBias(), plan.offsets);
	auto hidden = oa::FnMatrix::siluMul(gateUp, static_cast<oa::U32>(dFF_));
	auto packedOut = oa::FnMatrix::groupedLinearM(hidden, expertDownWeight(), expertDownBias(), plan.offsets);
	return oa::FnMatrix::moeCombine(packedOut, inRouteGate, plan.inverse, plan.packedSlot);
}

oa::Moe::Moe(oa::I32 inDModel, oa::I32 inDFF, oa::I32 inNumExperts, oa::I32 inExpertsPerToken, oa::F32 inRmsEps, oa::I32 inNumSharedExperts) {
	init(inDModel, inDFF, inNumExperts, inExpertsPerToken, inRmsEps, inNumSharedExperts);
}

void oa::Moe::init(oa::I32 inDModel, oa::I32 inDFF, oa::I32 inNumExperts, oa::I32 inExpertsPerToken,
	oa::F32 inRmsEps, oa::I32 inNumSharedExperts) {
	if (inDModel <= 0 or inDFF <= 0 or inNumExperts <= 0) {
		throw std::invalid_argument("oa::Moe dimensions and expert count must be positive");
	}
	dModel_ = inDModel;
	dFF_ = inDFF;
	numExperts_ = inNumExperts;
	// Clamp top-k to [1, numExperts]. K == numExperts degenerates to a dense
	// soft-moE (mask all ones), which is correct.
	expertsPerToken_ = inExpertsPerToken;
	if (expertsPerToken_ < 1) expertsPerToken_ = 1;
	if (expertsPerToken_ > numExperts_) expertsPerToken_ = numExperts_;
	rmsEps_ = inRmsEps;
	numSharedExperts_ = inNumSharedExperts < 0 ? 0 : inNumSharedExperts;

	auto wd = oa::FnMatrix::weightDtype();
	norm_ = oa::makeShared<oa::RmsNorm>(dModel_, rmsEps_);
	registerModule("norm", norm_);

	// Router: projects input to expert logits [D → numExperts].
	router_ = oa::makeShared<oa::Linear>(dModel_, numExperts_);
	auto& routerParams = router_->parameters();
	// grad is the single source of truth on each param's Data (oa::Parameter::grad());
	// setRequiresGrad allocates it — no manual snapshot re-sync needed.
	routerParams[0].data = oa::FnMatrix::randXavier(oa::MatrixShape{numExperts_, dModel_}, wd);
	routerParams[0].data.setRequiresGrad(true);
	routerParams[1].data.setRequiresGrad(true);
	registerModule("router", router_);

	// Stacked routed-expert parameters are the optimizer/checkpoint source of
	// truth and are consumed directly by grouped sparse kernels. initialize with
	// the same per-linear fan-in bounds as oa::Linear without constructing temporary
	// expert modules or a per-forward packing graph.
	const oa::F32 gateBound = std::sqrt(1.0F / static_cast<oa::F32>(dModel_));
	const oa::F32 downBound = std::sqrt(1.0F / static_cast<oa::F32>(dFF_));
	auto gateW = oa::FnMatrix::empty(
		oa::MatrixShape{numExperts_, 2 * dFF_, dModel_}, wd);
	gateW = oa::FnMatrix::philoxUniform(gateW, -gateBound, gateBound, 0);
	auto downW = oa::FnMatrix::empty(
		oa::MatrixShape{numExperts_, dModel_, dFF_}, wd);
	downW = oa::FnMatrix::philoxUniform(downW, -downBound, downBound, 0);
	registerParameter("expert_gate_up_weight", gateW);
	registerParameter("expert_gate_up_bias",
		oa::FnMatrix::zeros(oa::MatrixShape{numExperts_, 2 * dFF_}, wd));
	registerParameter("expert_down_weight", downW);
	registerParameter("expert_down_bias",
		oa::FnMatrix::zeros(oa::MatrixShape{numExperts_, dModel_}, wd));

	// Shared always-on experts (DeepSeekMoE). Applied unconditionally, no gate —
	// they absorb common knowledge so the routed experts are free to specialize.
	// Reuse the canonical SwiGLU layer instead of exposing a second expert-only
	// public module with the same computation.
	sharedExperts_.reserve(numSharedExperts_);
	for (oa::I32 s = 0; s < numSharedExperts_; ++s) {
		auto expert = oa::makeShared<oa::Swiglu>(dModel_, dFF_, true);
		registerModule(oa::String("shared_expert.") + oa::toString(static_cast<oa::I64>(s)), expert);
		sharedExperts_.pushBack(expert);
	}

	// Aux-loss-free balancing bias starts at zero (⇒ forward identical to classic).
	oa::Vec<oa::F32> zeroBias(numExperts_, 0.0F);
	routingBias_ = oa::FnMatrix::fromBytes(
		oa::Span<const oa::U8>(reinterpret_cast<const oa::U8*>(zeroBias.data()),
			zeroBias.size() * sizeof(oa::F32)),
		oa::MatrixShape{1, numExperts_}, oa::ScalarType::Float32);
	registerBuffer("routing_bias", routingBias_);

	// persistent 0 scalar so auxLoss() is always valid even when both coefficients
	// are zero (no per-forward allocation on the disabled path).
	zeroAuxLoss_ = oa::FnMatrix::zeros(oa::MatrixShape{1}, oa::ScalarType::Float32);
	lastAuxLoss_ = zeroAuxLoss_;
}

oa::Matrix oa::Moe::forward(const oa::Matrix& inX) {
	// input: [T, D]
	const oa::I64 T = inX.getShape()[0];
	const oa::I32 E = numExperts_;
	const oa::I32 K = expertsPerToken_;

	// 1. Shared pre-norm, then router logits → probabilities. [T, E]
	oa::Matrix normed = norm_->forward(inX);
	oa::Matrix logits = router_->forward(normed);       // autograd (Linear)
	oa::Matrix probs = oa::FnMatrix::softmax(logits, 1);  // autograd (Softmax) — gate affinity
	lastGateProbs_ = probs;

	// 2. Selection logits carry the aux-loss-free balancing bias — for the top-k
	//    DECISION only, never the gate magnitude. The host policy scalar decides
	//    whether balancing is enabled; the bias tensor is never read on the CPU.
	oa::Matrix selLogits = logits;
	if (balanceGamma_ > 0.0F) {
		selLogits = oa::FnMatrix::add(logits, routingBias_);  // [T,E] + [1,E] broadcast
	}

	// 3. top-k selection. topK is a routing decision (no autograd node). Its
	//    deterministic indices are the membership source of truth, including ties.
	oa::TopKResult topk = oa::FnMatrix::topK(selLogits, K, 1);
	oa::Matrix mask = oa::FnMatrix::topKMask(topk.indices, E);
	lastSelectionMask_ = mask;

	// 4. normalize unbiased probabilities over the exact selected routes. The
	//    sparse path emits [T,K] directly; the dense oracle deliberately retains
	//    the generic [T,E] construction as an independent reference.
	oa::Matrix out;
	if (sparseExecution_) {
		lastGate_ = oa::FnMatrix::moeRouteWeights(probs, topk.indices);
		out = sparseExpertDelta(normed, lastGate_, topk.indices);
	} else {
		auto gateUnnorm = oa::FnMatrix::mul(probs, mask);
		auto denom = oa::FnMatrix::sum(gateUnnorm, 1);
		lastGate_ = oa::FnMatrix::div(gateUnnorm, denom);
		out = denseExpertDelta(normed, lastGate_);
	}

	// 6. Shared always-on experts (no gate): each adds its full delta.
	for (oa::I32 s = 0; s < numSharedExperts_; ++s) {
		out = oa::FnMatrix::add(out, sharedExperts_[s]->forward(normed));
	}

	// 7. Optional differentiable balancing losses, recorded on the tape so the
	//    caller can add(taskLoss, auxLoss()) before backward. Disabled ⇒ untouched
	//    (lastAuxLoss_ stays the persistent 0 scalar).
	if (auxAlpha_ > 0.0f or zBeta_ > 0.0f) {
		oa::Matrix acc;
		if (auxAlpha_ > 0.0f) {
			// Switch/GShard: α·E·Σ_e f_e·P_e. f = hard dispatch fraction (constant),
			// P = mean router prob (differentiable → pushes probability mass toward
			// under-used experts).
			oa::Matrix fe = oa::FnMatrix::mean(mask, 0);   // [1,E] constant
			oa::Matrix pe = oa::FnMatrix::mean(probs, 0);  // [1,E] grad→router
			oa::Matrix sw = oa::FnMatrix::sum(oa::FnMatrix::mul(fe, pe), 1);       // [1,1]
			// Divide by K so a perfectly balanced router has loss alpha rather
			// than alpha*K for top-k routing.
			acc = oa::FnMatrix::scale(sw,
				auxAlpha_ * static_cast<oa::F32>(E) / static_cast<oa::F32>(K));
		}
		if (zBeta_ > 0.0f) {
			// Router z-loss: β·mean_t LSE(logits)². LSE_i = logits_i,0 − logsoftmax_i,0
			// (identity for any column) keeps logits from drifting large.
			oa::Matrix lsm = oa::FnMatrix::logSoftmax(logits, 1);        // [T,E]
			oa::Matrix lse = oa::FnMatrix::sub(oa::FnMatrix::slice(logits, 1, 0, 1),
				oa::FnMatrix::slice(lsm, 1, 0, 1));                    // [T,1]
			oa::Matrix z = oa::FnMatrix::scale(oa::FnMatrix::mean(oa::FnMatrix::mul(lse, lse), 0), zBeta_);
			acc = (acc.numElements() == 0) ? z : oa::FnMatrix::add(acc, z);
		}
		lastAuxLoss_ = oa::FnMatrix::reshape(acc, oa::MatrixShape{1});
	} else {
		lastAuxLoss_ = zeroAuxLoss_;
	}

	(void)T;
	return oa::FnMatrix::add(inX, out);
}

// ── Telemetry / balancing helpers (host-side; require a prior execute+Sync) ────

oa::Vec<oa::F32> oa::Moe::lastLoadFraction() const {
	oa::Vec<oa::F32> frac;
	frac.reserve(numExperts_);
	for (oa::I32 e = 0; e < numExperts_; ++e) frac.pushBack(0.0f);
	if (lastSelectionMask_.numElements() == 0) return frac;

	const oa::I32 E = numExperts_;
	auto load = oa::FnMatrix::cast(
		oa::FnMatrix::reshape(oa::FnMatrix::sum(lastSelectionMask_, 0), oa::MatrixShape{E}),
		oa::ScalarType::Float32);
	auto& ctx = oa::ExecutionSession::getActive();
	(void)ctx.submitAndWait();
	oa::Vec<oa::F32> loadHost(static_cast<oa::Usize>(E));
	if (not oa::FnMatrix::copyToHost(load, loadHost.data(),
		static_cast<oa::U64>(load.byteSize())).isOk()) return frac;
	oa::F64 total = 0.0;
	for (oa::I32 e = 0; e < E; ++e) {
		frac[e] = loadHost[static_cast<oa::Usize>(e)];
		total += loadHost[static_cast<oa::Usize>(e)];
	}
	if (total > 0.0)
		for (oa::I32 e = 0; e < E; ++e) frac[e] = static_cast<oa::F32>(frac[e] / total);
	return frac;
}

oa::MoeRouteStats oa::Moe::routeStats() const {
	oa::MoeRouteStats s;
	s.loadFraction = lastLoadFraction();
	const oa::I32 E = numExperts_;

	// Mean router probability per expert (from the stored softmax).
	s.meanProb.reserve(E);
	for (oa::I32 e = 0; e < E; ++e) s.meanProb.pushBack(0.0f);
	if (lastGateProbs_.numElements() != 0) {
		auto mean = oa::FnMatrix::cast(
			oa::FnMatrix::reshape(oa::FnMatrix::mean(lastGateProbs_, 0), oa::MatrixShape{E}),
			oa::ScalarType::Float32);
		auto& ctx = oa::ExecutionSession::getActive();
		(void)ctx.submitAndWait();
		oa::Vec<oa::F32> meanHost(static_cast<oa::Usize>(E));
		if (oa::FnMatrix::copyToHost(mean, meanHost.data(),
			static_cast<oa::U64>(mean.byteSize())).isOk()) {
			for (oa::I32 e = 0; e < E; ++e)
				s.meanProb[e] = meanHost[static_cast<oa::Usize>(e)];
		}
	}

	// normalized load entropy (1 = balanced) + max-load ratio + dead count.
	oa::F64 ent = 0.0, maxLoad = 0.0;
	for (oa::I32 e = 0; e < E; ++e) {
		const oa::F64 f = s.loadFraction[e];
		if (f > 0.0) ent -= f * std::log(f);
		else ++s.deadExperts;
		if (f > maxLoad) maxLoad = f;
	}
	s.entropy = (E > 1) ? static_cast<oa::F32>(ent / std::log(static_cast<oa::F64>(E))) : 1.0f;
	s.maxLoadRatio = static_cast<oa::F32>(maxLoad * static_cast<oa::F64>(E));
	return s;
}

void oa::Moe::updateRoutingBias() {
	if (balanceGamma_ == 0.0f or lastSelectionMask_.isEmpty()) return;
	oa::FnMatrix::moeRoutingBiasUpdate(
		lastSelectionMask_, routingBias_, expertsPerToken_, balanceGamma_);
}

oa::F32 oa::Moe::routingBias(oa::I32 inExpert) const {
	if (inExpert < 0 or inExpert >= numExperts_ or routingBias_.isEmpty()) return 0.0F;
	auto value = oa::FnMatrix::cast(
		oa::FnMatrix::slice(routingBias_, 1, inExpert, inExpert + 1),
		oa::ScalarType::Float32);
	auto& ctx = oa::ExecutionSession::getActive();
	(void)ctx.submitAndWait();
	oa::F32 host = 0.0F;
	(void)oa::FnMatrix::copyToHost(value, &host, sizeof(host));
	return host;
}
