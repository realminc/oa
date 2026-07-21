#include <Oa/Ml/Optim.h>
#include <Oa/Ml/Oam.h>
#include <Oa/Runtime/Context.h>
#include <Oa/Core/BufferAccess.h>
#include <Oa/Core/FnMatrix.h>
#include <Oa/Core/Validation.h>

#include <cstring>

static OaU32 DivCeil(OaU32 InA, OaU32 InB) { return (InA + InB - 1) / InB; }

static OaMatrix GetParamGrad(OaParameter* InP) {
	return InP->Grad();  // live grad (single source of truth: Data's autograd meta)
}

void OaSGD::Step() {
	++Step_;
	auto& ctx = OaContext::GetDefault();
	// fp32 master weights for any low-precision (bf16) params — the "Sgd" kernel
	// then runs on the fp32 master (grad upcast) and the bf16 weight is re-derived.
	PrepareMasters();

	for (OaUsize i = 0; i < Params_.Size(); ++i) {
		auto* p = Params_[i];
		OaMatrix grad = GetParamGrad(p);
		if (!grad.HasStorage()) continue;

		OaMatrix gradUse = MasterGrad(i, grad);
		OaMatrix& weight = MasterOrData(i);
		OaU32 n = static_cast<OaU32>(weight.NumElements());

		struct { OaU32 Count; OaF32 Lr; OaF32 WeightDecay; }
			push{n, Lr_, WeightDecay_};
		OaBufferAccess access[] = {OaBufferAccess::ReadWrite, OaBufferAccess::Read};
		ctx.Add("Sgd", {&weight, &gradUse}, access, &push, sizeof(push), DivCeil(n, 256));
		WritebackMaster(i);
	}
}

void OaSGD::ZeroGrad() {
	// Zero the single source of truth (Data's autograd grad) GPU-side. OaMatrix::ZeroGrad
	// self-guards (no-op if the param has no grad buffer) and records a Fill kernel — never
	// a host memset, which was a silent no-op on GPU buffers that let AccumulateGrad grow
	// the grad without bound across steps → divergence.
	for (auto* p : Params_) { p->Data.ZeroGrad(); }
}

// SGD has no per-param momentum buffer in the current impl (the kernel ignores
// Momentum_). Persistence is header-only: Lr, WeightDecay, Step, plus Momentum
// stashed in Beta1 (repurposed) so future SgdMomentum work can recover it.

OaStatus OaSGD::SaveTo(OamModel& OutOam) const {
	OutOam.OptimizerPresent = true;
	OutOam.Optimizer = OamOptimizerHeader{};
	std::strncpy(OutOam.Optimizer.Type, "SGD", sizeof(OutOam.Optimizer.Type) - 1);
	OutOam.Optimizer.Lr          = Lr_;
	OutOam.Optimizer.Beta1       = Momentum_;  // repurposed: SGD has no Beta1
	OutOam.Optimizer.WeightDecay = WeightDecay_;
	OutOam.Optimizer.Step        = static_cast<OaI64>(Step_);
	OutOam.Optimizer.NumParams   = 0;
	OutOam.AdamM.Clear();
	OutOam.AdamV.Clear();
	return OaStatus::Ok();
}

OaStatus OaSGD::ValidateLoad(const OamModel& InOam) const {
	if (not InOam.HasOptimizer()
		or std::strncmp(InOam.Optimizer.Type, "SGD", sizeof(InOam.Optimizer.Type)) != 0)
	{
		return OaStatus::Error(OaStatusCode::FailedPrecondition,
			"SGD checkpoint optimizer state is missing or has the wrong type");
	}
	if (not InOam.AdamM.Empty() or not InOam.AdamV.Empty() or not InOam.MuonM.Empty()) {
		return OaStatus::Error(OaStatusCode::CheckpointCorrupt,
			"SGD checkpoint contains unexpected optimizer payloads");
	}
	return OaStatus::Ok();
}

OaStatus OaSGD::LoadFrom(const OamModel& InOam) {
	OA_RETURN_IF_ERROR(ValidateLoad(InOam));
	Lr_          = InOam.Optimizer.Lr;
	Momentum_    = InOam.Optimizer.Beta1;
	WeightDecay_ = InOam.Optimizer.WeightDecay;
	Step_        = static_cast<OaU64>(InOam.Optimizer.Step);
	ResetMasterSeed();  // re-seed fp32 masters from the reloaded (bf16) weights
	return OaStatus::Ok();
}
