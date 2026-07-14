// OaMoE — Mixture of Experts Implementation
//
// Routing is assembled from differentiable matrix operations and the sparse
// executor uses GPU-native planning, grouped expert projections, and a direct
// weighted combine. The dense executor remains a mathematical oracle over the
// same stacked expert parameters; both paths are end-to-end gradchecked.
//
// Routing (top-k) is a non-differentiable decision: TopK picks exact expert IDs
// and TopKMask turns them into a constant 0/1 mask. Gradient therefore flows only
// through selected gate magnitudes and selected experts — standard MoE semantics.
//
// The production path packs only selected token/expert pairs. Evaluating every
// expert for every token is opt-in and exists only for correctness comparison.
//
// Stage 0 (anti-collapse, all inert at defaults so the oracle stays byte-identical):
//   • telemetry — RouteStats() exposes per-expert load fraction/entropy/dead count;
//   • aux-loss-free balancing — a gradient-free per-expert selection bias nudged by
//     UpdateRoutingBias(); zero bias skips the Add entirely;
//   • differentiable switch aux loss + router z-loss — opt-in, added via AuxLoss();
//   • shared always-on experts — DeepSeekMoE specialization anchor, default 0.

#include <Oa/Ml/Nn/Moe/Moe.h>
#include <Oa/Ml/Nn.h>
#include <Oa/Ml/Module.h>
#include <Oa/Core/FnMatrix.h>
#include <Oa/Ml/FnMatrix.h>
#include <Oa/Core/Std/Format.h>
#include <Oa/Runtime/Context.h>
#include <cmath>
#include <stdexcept>

OaMoeExpert::OaMoeExpert(OaI32 InDModel, OaI32 InDFF) {
	Init(InDModel, InDFF);
}

void OaMoeExpert::Init(OaI32 InDModel, OaI32 InDFF) {
	DFF_ = InDFF;
	auto wd = OaFnMatrix::GetWeightDtype();
	GateUp_ = OaMakeSharedPtr<OaLinear>(InDModel, 2 * InDFF);
	Down_ = OaMakeSharedPtr<OaLinear>(InDFF, InDModel);

	// Preserve the initialization scale of two independent D->DFF projections
	// while storing them contiguously for one D->2*DFF GEMM.
	OaMatrix gateUpParts[] = {
		OaFnMatrix::RandXavier(OaMatrixShape{InDFF, InDModel}, wd),
		OaFnMatrix::RandXavier(OaMatrixShape{InDFF, InDModel}, wd),
	};
	GateUp_->Parameters()[0].Data = OaFnMatrix::Concat(OaSpan<OaMatrix>(gateUpParts), 0);
	Down_->Parameters()[0].Data = OaFnMatrix::RandXavier(OaMatrixShape{InDModel, InDFF}, wd);
	RegisterModule("gate_up", GateUp_);
	RegisterModule("down", Down_);
	for (auto* parameter : AllParameterPtrs()) parameter->Data.SetRequiresGrad(true);
}

OaMatrix OaMoeExpert::Forward(const OaMatrix& InX) {
	return Down_->Forward(OaFnMatrix::SiluMul(
		GateUp_->Forward(InX), static_cast<OaU32>(DFF_)));
}

const OaMatrix& OaMoeExpert::GateUpWeight() const { return GateUp_->Parameters()[0].Data; }
const OaMatrix& OaMoeExpert::GateUpBias() const { return GateUp_->Parameters()[1].Data; }
const OaMatrix& OaMoeExpert::DownWeight() const { return Down_->Parameters()[0].Data; }
const OaMatrix& OaMoeExpert::DownBias() const { return Down_->Parameters()[1].Data; }

OaMatrix OaMoE::DenseExpertDelta(const OaMatrix& InNormed, const OaMatrix& InGate) const {
	const OaI64 T = InNormed.Size(0);
	OaVec<OaMatrix> expertOutputs;
	expertOutputs.Reserve(NumExperts_);
	for (OaI32 e = 0; e < NumExperts_; ++e) {
		auto gateW = OaFnMatrix::Reshape(
			OaFnMatrix::Slice(ExpertGateUpWeight(), 0, e, e + 1),
			OaMatrixShape{2 * DFF_, DModel_});
		auto gateB = OaFnMatrix::Reshape(
			OaFnMatrix::Slice(ExpertGateUpBias(), 0, e, e + 1),
			OaMatrixShape{2 * DFF_});
		auto downW = OaFnMatrix::Reshape(
			OaFnMatrix::Slice(ExpertDownWeight(), 0, e, e + 1),
			OaMatrixShape{DModel_, DFF_});
		auto downB = OaFnMatrix::Reshape(
			OaFnMatrix::Slice(ExpertDownBias(), 0, e, e + 1),
			OaMatrixShape{DModel_});
		// Raw OaFnMatrix::Linear is a dispatch primitive; OaLinear normally owns
		// its autograd attachment. Build the dense oracle from differentiable
		// MatMulNt + broadcast Add so sliced stacked parameters receive gradients.
		auto gateUp = OaFnMatrix::Add(OaFnMatrix::MatMulNt(InNormed, gateW), gateB);
		auto hidden = OaFnMatrix::SiluMul(gateUp, static_cast<OaU32>(DFF_));
		auto expertOut = OaFnMatrix::Add(OaFnMatrix::MatMulNt(hidden, downW), downB);
		expertOutputs.PushBack(OaFnMatrix::Reshape(expertOut,
			OaMatrixShape{T, 1, DModel_}));
	}
	auto allExperts = OaFnMatrix::Concat(OaSpan<OaMatrix>(expertOutputs), 1);
	auto gate3d = OaFnMatrix::Reshape(InGate, OaMatrixShape{T, NumExperts_, 1});
	auto weighted = OaFnMatrix::Mul(allExperts, gate3d);
	return OaFnMatrix::Reshape(OaFnMatrix::Sum(weighted, 1),
		OaMatrixShape{T, DModel_});
}

OaMatrix OaMoE::SparseExpertDelta(const OaMatrix& InNormed,
	const OaMatrix& InRouteGate, const OaMatrix& InTopKIndices) const {
	auto plan = OaFnMatrix::MoeExpertPlan(InTopKIndices, NumExperts_);

	// Gather duplicated routed tokens into stable expert-major layout. Selected
	// route weights already have token-major [T,K] layout for direct combine.
	auto packedX = OaFnMatrix::Gather(InNormed, plan.PackedToken);

	auto gateUp = OaFnMatrix::GroupedLinearM(
		packedX, ExpertGateUpWeight(), ExpertGateUpBias(), plan.Offsets);
	auto hidden = OaFnMatrix::SiluMul(gateUp, static_cast<OaU32>(DFF_));
	auto packedOut = OaFnMatrix::GroupedLinearM(
		hidden, ExpertDownWeight(), ExpertDownBias(), plan.Offsets);
	return OaFnMatrix::MoeCombine(
		packedOut, InRouteGate, plan.Inverse, plan.PackedSlot);
}

OaMoE::OaMoE(OaI32 InDModel, OaI32 InDFF, OaI32 InNumExperts, OaI32 InExpertsPerToken,
	OaF32 InRmsEps, OaI32 InNumSharedExperts) {
	Init(InDModel, InDFF, InNumExperts, InExpertsPerToken, InRmsEps, InNumSharedExperts);
}

void OaMoE::Init(OaI32 InDModel, OaI32 InDFF, OaI32 InNumExperts, OaI32 InExpertsPerToken,
	OaF32 InRmsEps, OaI32 InNumSharedExperts) {
	if (InDModel <= 0 or InDFF <= 0 or InNumExperts <= 0) {
		throw std::invalid_argument("OaMoE dimensions and expert count must be positive");
	}
	DModel_ = InDModel;
	DFF_ = InDFF;
	NumExperts_ = InNumExperts;
	// Clamp top-k to [1, NumExperts]. K == NumExperts degenerates to a dense
	// soft-MoE (mask all ones), which is correct.
	ExpertsPerToken_ = InExpertsPerToken;
	if (ExpertsPerToken_ < 1) ExpertsPerToken_ = 1;
	if (ExpertsPerToken_ > NumExperts_) ExpertsPerToken_ = NumExperts_;
	RmsEps_ = InRmsEps;
	NumSharedExperts_ = InNumSharedExperts < 0 ? 0 : InNumSharedExperts;

	auto wd = OaFnMatrix::GetWeightDtype();
	Norm_ = OaMakeSharedPtr<OaRmsNorm>(DModel_, RmsEps_);
	RegisterModule("norm", Norm_);

	// Router: projects input to expert logits [D → NumExperts].
	Router_ = OaMakeSharedPtr<OaLinear>(DModel_, NumExperts_);
	auto& routerParams = Router_->Parameters();
	// Grad is the single source of truth on each param's Data (OaParameter::Grad());
	// SetRequiresGrad allocates it — no manual snapshot re-sync needed.
	routerParams[0].Data = OaFnMatrix::RandXavier(OaMatrixShape{NumExperts_, DModel_}, wd);
	routerParams[0].Data.SetRequiresGrad(true);
	routerParams[1].Data.SetRequiresGrad(true);
	RegisterModule("router", Router_);

	// Stacked routed-expert parameters are the optimizer/checkpoint source of
	// truth and are consumed directly by grouped sparse kernels. Initialize with
	// the same per-linear fan-in bounds as OaLinear without constructing temporary
	// expert modules or a per-forward packing graph.
	const OaF32 gateBound = std::sqrt(1.0F / static_cast<OaF32>(DModel_));
	const OaF32 downBound = std::sqrt(1.0F / static_cast<OaF32>(DFF_));
	auto gateW = OaFnMatrix::Empty(
		OaMatrixShape{NumExperts_, 2 * DFF_, DModel_}, wd);
	gateW = OaFnMatrix::PhiloxUniform(gateW, -gateBound, gateBound, 0);
	auto downW = OaFnMatrix::Empty(
		OaMatrixShape{NumExperts_, DModel_, DFF_}, wd);
	downW = OaFnMatrix::PhiloxUniform(downW, -downBound, downBound, 0);
	RegisterParameter("expert_gate_up_weight", gateW);
	RegisterParameter("expert_gate_up_bias",
		OaFnMatrix::Zeros(OaMatrixShape{NumExperts_, 2 * DFF_}, wd));
	RegisterParameter("expert_down_weight", downW);
	RegisterParameter("expert_down_bias",
		OaFnMatrix::Zeros(OaMatrixShape{NumExperts_, DModel_}, wd));

	// Shared always-on experts (DeepSeekMoE). Applied unconditionally, no gate —
	// they absorb common knowledge so the routed experts are free to specialize.
	SharedExperts_.Reserve(NumSharedExperts_);
	for (OaI32 s = 0; s < NumSharedExperts_; ++s) {
		auto expert = OaMakeSharedPtr<OaMoeExpert>(DModel_, DFF_);
		RegisterModule(OaStdString("shared_expert.") + OaToString(static_cast<OaI64>(s)), expert);
		SharedExperts_.PushBack(expert);
	}

	// Aux-loss-free balancing bias starts at zero (⇒ forward identical to classic).
	OaVec<OaF32> zeroBias(NumExperts_, 0.0F);
	RoutingBias_ = OaFnMatrix::FromBytes(
		OaSpan<const OaU8>(reinterpret_cast<const OaU8*>(zeroBias.Data()),
			zeroBias.Size() * sizeof(OaF32)),
		OaMatrixShape{1, NumExperts_}, OaScalarType::Float32);
	RegisterBuffer("routing_bias", RoutingBias_);

	// Persistent 0 scalar so AuxLoss() is always valid even when both coefficients
	// are zero (no per-forward allocation on the disabled path).
	ZeroAuxLoss_ = OaFnMatrix::Zeros(OaMatrixShape{1}, OaScalarType::Float32);
	LastAuxLoss_ = ZeroAuxLoss_;
}

OaMatrix OaMoE::Forward(const OaMatrix& InX) {
	// Input: [T, D]
	const OaI64 T = InX.GetShape()[0];
	const OaI32 E = NumExperts_;
	const OaI32 K = ExpertsPerToken_;

	// 1. Shared pre-norm, then router logits → probabilities. [T, E]
	OaMatrix normed = Norm_->Forward(InX);
	OaMatrix logits = Router_->Forward(normed);       // autograd (Linear)
	OaMatrix probs = OaFnMatrix::Softmax(logits, 1);  // autograd (Softmax) — gate affinity
	LastGateProbs_ = probs;

	// 2. Selection logits carry the aux-loss-free balancing bias — for the top-k
	//    DECISION only, never the gate magnitude. The host policy scalar decides
	//    whether balancing is enabled; the bias tensor is never read on the CPU.
	OaMatrix selLogits = logits;
	if (BalanceGamma_ > 0.0F) {
		selLogits = OaFnMatrix::Add(logits, RoutingBias_);  // [T,E] + [1,E] broadcast
	}

	// 3. Top-k selection. TopK is a routing decision (no autograd node). Its
	//    deterministic indices are the membership source of truth, including ties.
	OaTopKResult topk = OaFnMatrix::TopK(selLogits, K, 1);
	OaMatrix mask = OaFnMatrix::TopKMask(topk.Indices, E);
	LastSelectionMask_ = mask;

	// 4. Normalize unbiased probabilities over the exact selected routes. The
	//    sparse path emits [T,K] directly; the dense oracle deliberately retains
	//    the generic [T,E] construction as an independent reference.
	OaMatrix out;
	if (SparseExecution_) {
		LastGate_ = OaFnMatrix::MoeRouteWeights(probs, topk.Indices);
		out = SparseExpertDelta(normed, LastGate_, topk.Indices);
	} else {
		auto gateUnnorm = OaFnMatrix::Mul(probs, mask);
		auto denom = OaFnMatrix::Sum(gateUnnorm, 1);
		LastGate_ = OaFnMatrix::Div(gateUnnorm, denom);
		out = DenseExpertDelta(normed, LastGate_);
	}

	// 6. Shared always-on experts (no gate): each adds its full delta.
	for (OaI32 s = 0; s < NumSharedExperts_; ++s) {
		out = OaFnMatrix::Add(out, SharedExperts_[s]->Forward(normed));
	}

	// 7. Optional differentiable balancing losses, recorded on the tape so the
	//    caller can Add(taskLoss, AuxLoss()) before Backward. Disabled ⇒ untouched
	//    (LastAuxLoss_ stays the persistent 0 scalar).
	if (AuxAlpha_ > 0.0f or ZBeta_ > 0.0f) {
		OaMatrix acc;
		if (AuxAlpha_ > 0.0f) {
			// Switch/GShard: α·E·Σ_e f_e·P_e. f = hard dispatch fraction (constant),
			// P = mean router prob (differentiable → pushes probability mass toward
			// under-used experts).
			OaMatrix fe = OaFnMatrix::Mean(mask, 0);   // [1,E] constant
			OaMatrix pe = OaFnMatrix::Mean(probs, 0);  // [1,E] grad→router
			OaMatrix sw = OaFnMatrix::Sum(OaFnMatrix::Mul(fe, pe), 1);       // [1,1]
			// Divide by K so a perfectly balanced router has loss alpha rather
			// than alpha*K for top-k routing.
			acc = OaFnMatrix::Scale(sw,
				AuxAlpha_ * static_cast<OaF32>(E) / static_cast<OaF32>(K));
		}
		if (ZBeta_ > 0.0f) {
			// Router z-loss: β·mean_t LSE(logits)². LSE_i = logits_i,0 − logsoftmax_i,0
			// (identity for any column) keeps logits from drifting large.
			OaMatrix lsm = OaFnMatrix::LogSoftmax(logits, 1);        // [T,E]
			OaMatrix lse = OaFnMatrix::Sub(OaFnMatrix::Slice(logits, 1, 0, 1),
				OaFnMatrix::Slice(lsm, 1, 0, 1));                    // [T,1]
			OaMatrix z = OaFnMatrix::Scale(OaFnMatrix::Mean(OaFnMatrix::Mul(lse, lse), 0), ZBeta_);
			acc = (acc.NumElements() == 0) ? z : OaFnMatrix::Add(acc, z);
		}
		LastAuxLoss_ = OaFnMatrix::Reshape(acc, OaMatrixShape{1});
	} else {
		LastAuxLoss_ = ZeroAuxLoss_;
	}

	(void)T;
	return OaFnMatrix::Add(InX, out);
}

// ── Telemetry / balancing helpers (host-side; require a prior Execute+Sync) ────

OaVec<OaF32> OaMoE::LastLoadFraction() const {
	OaVec<OaF32> frac;
	frac.Reserve(NumExperts_);
	for (OaI32 e = 0; e < NumExperts_; ++e) frac.PushBack(0.0f);
	if (LastSelectionMask_.NumElements() == 0) return frac;

	const OaI32 E = NumExperts_;
	auto load = OaFnMatrix::Cast(
		OaFnMatrix::Reshape(OaFnMatrix::Sum(LastSelectionMask_, 0), OaMatrixShape{E}),
		OaScalarType::Float32);
	auto& ctx = OaContext::GetDefault();
	(void)ctx.Execute(); (void)ctx.Sync();
	const OaF32* loadHost = load.DataAs<const OaF32>();
	OaF64 total = 0.0;
	for (OaI32 e = 0; e < E; ++e) { frac[e] = loadHost[e]; total += loadHost[e]; }
	if (total > 0.0)
		for (OaI32 e = 0; e < E; ++e) frac[e] = static_cast<OaF32>(frac[e] / total);
	return frac;
}

OaMoeRouteStats OaMoE::RouteStats() const {
	OaMoeRouteStats s;
	s.LoadFraction = LastLoadFraction();
	const OaI32 E = NumExperts_;

	// Mean router probability per expert (from the stored softmax).
	s.MeanProb.Reserve(E);
	for (OaI32 e = 0; e < E; ++e) s.MeanProb.PushBack(0.0f);
	if (LastGateProbs_.NumElements() != 0) {
		auto mean = OaFnMatrix::Cast(
			OaFnMatrix::Reshape(OaFnMatrix::Mean(LastGateProbs_, 0), OaMatrixShape{E}),
			OaScalarType::Float32);
		auto& ctx = OaContext::GetDefault();
		(void)ctx.Execute(); (void)ctx.Sync();
		const OaF32* meanHost = mean.DataAs<const OaF32>();
		for (OaI32 e = 0; e < E; ++e) s.MeanProb[e] = meanHost[e];
	}

	// Normalized load entropy (1 = balanced) + max-load ratio + dead count.
	OaF64 ent = 0.0, maxLoad = 0.0;
	for (OaI32 e = 0; e < E; ++e) {
		const OaF64 f = s.LoadFraction[e];
		if (f > 0.0) ent -= f * std::log(f);
		else ++s.DeadExperts;
		if (f > maxLoad) maxLoad = f;
	}
	s.Entropy = (E > 1) ? static_cast<OaF32>(ent / std::log(static_cast<OaF64>(E))) : 1.0f;
	s.MaxLoadRatio = static_cast<OaF32>(maxLoad * static_cast<OaF64>(E));
	return s;
}

void OaMoE::UpdateRoutingBias() {
	if (BalanceGamma_ == 0.0f or LastSelectionMask_.IsEmpty()) return;
	OaFnMatrix::MoeRoutingBiasUpdate(
		LastSelectionMask_, RoutingBias_, ExpertsPerToken_, BalanceGamma_);
}

OaF32 OaMoE::RoutingBias(OaI32 InExpert) const {
	if (InExpert < 0 or InExpert >= NumExperts_ or RoutingBias_.IsEmpty()) return 0.0F;
	auto value = OaFnMatrix::Cast(
		OaFnMatrix::Slice(RoutingBias_, 1, InExpert, InExpert + 1),
		OaScalarType::Float32);
	auto& ctx = OaContext::GetDefault();
	(void)ctx.Execute();
	(void)ctx.Sync();
	return value.DataAs<const OaF32>()[0];
}
