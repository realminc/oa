#include <Oa/Ml/Optim.h>
#include <Oa/Ml/FnOptim.h>
#include <Oa/Ml/Oam.h>
#include <Oa/Core/FnMatrix.h>
#include <Oa/Core/Log.h>
#include <Oa/Runtime/Context.h>
#include <Oa/Runtime/Engine.h>

#include <cstring>

static OaMatrix GetParamGrad(OaParameter* InP) {
	return InP->Grad();  // live grad (single source of truth: Data's autograd meta)
}

static void EnsureMomentumBuffers(OaVec<OaParameter*>& InParams, OaVec<OaMatrix>& InOutMomentum) {
	if (not InOutMomentum.Empty()) return;
	for (auto* p : InParams) {
		InOutMomentum.PushBack(OaFnMatrix::Zeros(p->Data.GetShape()));
	}
}

void OaMuon::Step() {
	++Step_;

	// Lazy-initialize momentum buffers on first step
	EnsureMomentumBuffers(Params_, Momentum_);
	// fp32 master weights for any low-precision (bf16) params — the Muon NS5/apply
	// runs on the fp32 master (grad upcast) and the bf16 weight is re-derived.
	PrepareMasters();

	// Apply Muon update to each parameter using OaFnOptim stateless function
	for (OaUsize i = 0; i < Params_.Size(); ++i) {
		auto* p = Params_[i];
		OaMatrix grad = GetParamGrad(p);
		if (!grad.HasStorage()) continue;

		OaMatrix gradUse = MasterGrad(i, grad);
		OaFnOptim::MuonStep(
			MasterOrData(i),   // InOutParam: fp32 master (bf16) or Data (fp32)
			Momentum_[i],      // InOutMomentum (mutated)
			gradUse,           // InGrad: fp32 upcast (bf16) or grad (fp32)
			Lr_, Beta_, WeightDecay_, Eps_, NS5Iterations_
		);
		WritebackMaster(i);
	}
}

void OaMuon::ZeroGrad() {
	// Zero the single source of truth GPU-side (self-guarded Fill kernel, never a host
	// memset). See OaSGD::ZeroGrad.
	for (auto* p : Params_) { p->Data.ZeroGrad(); }
}

// ─── Persistence ──────────────────────────────────────────────────────────

void OaMuon::SaveTo(OamModel& OutOam) const {
	// Header (hyperparams + step count). NumParams is total flat element count.
	OutOam.Optimizer = OamOptimizerHeader{};
	std::strncpy(OutOam.Optimizer.Type, "Muon", sizeof(OutOam.Optimizer.Type) - 1);
	OutOam.Optimizer.Lr = Lr_;
	OutOam.Optimizer.Beta1 = Beta_;  // Repurposed: Muon has no Beta1/Beta2, store Beta in Beta1
	OutOam.Optimizer.Beta2 = 0.0f;   // Not used by Muon
	OutOam.Optimizer.Eps = Eps_;
	OutOam.Optimizer.WeightDecay = WeightDecay_;
	OutOam.Optimizer.Step = static_cast<OaI64>(Step_);

	// Momentum_ may not be allocated yet (no Step() called). Then nothing to save.
	if (Momentum_.Empty()) {
		OutOam.Optimizer.NumParams = 0;
		OutOam.AdamM.Clear();
		OutOam.AdamV.Clear();
		return;
	}

	// Drain pending GPU writes to Momentum_ so the memcpy sees the latest state.
	(void)OaContext::GetDefault().Execute();

	OaI64 total = 0;
	for (const auto& m : Momentum_) total += m.NumElements();
	OutOam.Optimizer.NumParams = static_cast<OaU64>(total);
	OutOam.AdamM.Resize(total);
	OutOam.AdamV.Clear();  // Muon has no second moment, so AdamV is unused

	OaI64 off = 0;
	for (OaUsize i = 0; i < Momentum_.Size(); ++i) {
		OaI64 n = Momentum_[i].NumElements();
		const auto bytes = static_cast<OaU64>(n) * sizeof(OaF32);
		(void)OaFnMatrix::CopyToHost(
			Momentum_[i], OutOam.AdamM.Data() + off, bytes);
		off += n;
	}
}

void OaMuon::LoadFrom(const OamModel& InOam) {
	// Hyperparams + step always restored (cheap, always present in checkpoint).
	Lr_          = InOam.Optimizer.Lr;
	Beta_        = InOam.Optimizer.Beta1;  // Muon Beta stored in Beta1 slot
	Eps_         = InOam.Optimizer.Eps;
	WeightDecay_ = InOam.Optimizer.WeightDecay;
	Step_        = static_cast<OaU64>(InOam.Optimizer.Step);
	ResetMasterSeed();  // re-seed fp32 masters from the reloaded (bf16) weights

	if (not InOam.HasOptimizer()) {
		OA_LOG_INFO(OaLogComponent::Core,
			"OaMuon::LoadFrom: checkpoint has no optimizer section, keeping zero-init state");
		return;
	}

	// Allocate momentum buffers if first use, then verify sizes line up.
	EnsureMomentumBuffers(Params_, Momentum_);

	OaI64 expected = 0;
	for (const auto& m : Momentum_) expected += m.NumElements();
	if (InOam.AdamM.Size() != static_cast<OaUsize>(expected)) {
		OA_LOG_ERROR(OaLogComponent::Core,
			"OaMuon::LoadFrom: Momentum size mismatch — model expects %lld elements, "
			"checkpoint has %zu (model architecture differs from saved model)",
			static_cast<long long>(expected), InOam.AdamM.Size());
		return;
	}

	// Drain pending GPU writes to Momentum_ before memcpy.
	(void)OaContext::GetDefault().Execute();

	OaI64 off = 0;
	for (OaUsize i = 0; i < Momentum_.Size(); ++i) {
		OaI64 n = Momentum_[i].NumElements();
		const auto bytes = static_cast<OaU64>(n) * sizeof(OaF32);
		if (auto* runtime = OaContext::GetDefault().GetEngine()) {
			(void)runtime->UploadBuffer(Momentum_[i].GetVkBuffer(), 0,
				InOam.AdamM.Data() + off, bytes);
		}
		off += n;
	}
}
