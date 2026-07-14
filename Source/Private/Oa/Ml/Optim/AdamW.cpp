#include <Oa/Ml/Optim.h>
#include <Oa/Ml/FnOptim.h>
#include <Oa/Ml/Oam.h>
#include <Oa/Core/FnMatrix.h>
#include <Oa/Core/Log.h>
#include <Oa/Runtime/Context.h>

#include <cstring>

static OaMatrix GetParamGrad(OaParameter* InP) {
	return InP->Grad();  // live grad (single source of truth: Data's autograd meta)
}

static void EnsureMomentBuffers(OaVec<OaParameter*>& InParams,
	OaVec<OaMatrix>& InOutM, OaVec<OaMatrix>& InOutV)
{
	if (not InOutM.Empty()) return;
	for (auto* p : InParams) {
		InOutM.PushBack(OaFnMatrix::Zeros(p->Data.GetShape()));
		InOutV.PushBack(OaFnMatrix::Zeros(p->Data.GetShape()));
	}
}

void OaAdamW::Step() {
	++Step_;

	// Lazy-initialize moment buffers on first step
	EnsureMomentBuffers(Params_, M_, V_);
	// Lazy-allocate + seed fp32 master weights for any low-precision (bf16) params.
	PrepareMasters();

	// Any low-precision param routes through the fp32-master per-param path. The
	// fused 4-param fast path assumes fp32 m/v under a single DTYPE, which breaks
	// once a weight is bf16 (it would read the fp32 moments as bf16) — so skip it.
	bool anyMaster = false;
	for (OaUsize i = 0; i < Params_.Size(); ++i) {
		if (HasMaster(i)) { anyMaster = true; break; }
	}

	// Grad() returns a handle sharing the live grad buffer. The param set needs
	// stable addresses, so hold the four handles in locals; ctx.Add (inside
	// AdamWStepMany) snapshots their VkBuf_ at record time and keeps the buffer
	// owners alive, so these locals can safely die when Step() returns.
	if (not anyMaster and Params_.Size() == 4) {
		OaMatrix grads4[4] = {
			Params_[0]->Grad(), Params_[1]->Grad(),
			Params_[2]->Grad(), Params_[3]->Grad(),
		};
		if (grads4[0].Data() and grads4[1].Data()
			and grads4[2].Data() and grads4[3].Data()) {
			OaFnOptim::OaAdamWParamSet sets[4] = {
				{&Params_[0]->Data, &M_[0], &V_[0], &grads4[0]},
				{&Params_[1]->Data, &M_[1], &V_[1], &grads4[1]},
				{&Params_[2]->Data, &M_[2], &V_[2], &grads4[2]},
				{&Params_[3]->Data, &M_[3], &V_[3], &grads4[3]},
			};
			OaFnOptim::AdamWStepMany(
				OaSpan<const OaFnOptim::OaAdamWParamSet>(sets, 4),
				Lr_, Beta1_, Beta2_, Eps_, WeightDecay_, static_cast<OaI32>(Step_));
			return;
		}
	}

	// Apply AdamW update to each parameter using OaFnOptim stateless function.
	// For bf16 params the base helpers redirect the math onto the fp32 master
	// (grad upcast) and re-derive the bf16 weight afterwards; fp32 params update
	// in place (MasterOrData → Data, MasterGrad → grad, WritebackMaster → no-op).
	for (OaUsize i = 0; i < Params_.Size(); ++i) {
		auto* p = Params_[i];
		OaMatrix grad = GetParamGrad(p);
		if (!grad.Data()) continue;

		OaMatrix gradUse = MasterGrad(i, grad);
		OaFnOptim::AdamWStep(
			MasterOrData(i),  // InOutParam: fp32 master (bf16) or Data (fp32)
			M_[i],            // InOutM (mutated, fp32)
			V_[i],            // InOutV (mutated, fp32)
			gradUse,          // InGrad: fp32 upcast (bf16) or grad (fp32)
			Lr_, Beta1_, Beta2_, Eps_, WeightDecay_, static_cast<OaI32>(Step_)
		);
		WritebackMaster(i);   // fp32 master → bf16 weight (no-op for fp32 params)
	}
}

void OaAdamW::ZeroGrad() {
	// Zero the single source of truth (each param's autograd grad) GPU-side. Grad()
	// returns a handle sharing the live buffer, so MultiFill on these copies fills the
	// real GPU buffers. Never a host memset (silent no-op on GPU → grads accumulate
	// across steps → divergence).
	//
	// Fast path: batch up to 4 grads into a single MultiMatrixFill dispatch.
	OaVec<OaMatrix> grads;
	grads.Reserve(4);
	for (auto* p : Params_) {
		OaMatrix g = p->Grad();
		if (g.Data()) grads.PushBack(g);
	}
	for (OaUsize i = 0; i < grads.Size(); i += 4) {
		OaUsize end = std::min(i + 4, grads.Size());
		OaFnMatrix::MultiFill(OaSpan<OaMatrix>(grads.Data() + i, end - i), 0.0F);
	}
}

// ─── Persistence ──────────────────────────────────────────────────────────

void OaAdamW::SaveTo(OamModel& OutOam) const {
	// Header (hyperparams + step count). NumParams is total flat element count.
	OutOam.Optimizer = OamOptimizerHeader{};
	std::strncpy(OutOam.Optimizer.Type, "AdamW", sizeof(OutOam.Optimizer.Type) - 1);
	OutOam.Optimizer.Lr = Lr_;
	OutOam.Optimizer.Beta1 = Beta1_;
	OutOam.Optimizer.Beta2 = Beta2_;
	OutOam.Optimizer.Eps = Eps_;
	OutOam.Optimizer.WeightDecay = WeightDecay_;
	OutOam.Optimizer.Step = static_cast<OaI64>(Step_);

	// M_/V_ may not be allocated yet (no Step() called). Then nothing to save.
	if (M_.Empty()) {
		OutOam.Optimizer.NumParams = 0;
		OutOam.AdamM.Clear();
		OutOam.AdamV.Clear();
		return;
	}

	// Drain pending GPU writes to M_/V_ so the memcpy sees the latest state.
	(void)OaContext::GetDefault().Execute();

	OaI64 total = 0;
	for (const auto& m : M_) total += m.NumElements();
	OutOam.Optimizer.NumParams = static_cast<OaU64>(total);
	OutOam.AdamM.Resize(total);
	OutOam.AdamV.Resize(total);

	OaI64 off = 0;
	for (OaUsize i = 0; i < M_.Size(); ++i) {
		OaI64 n = M_[i].NumElements();
		std::memcpy(OutOam.AdamM.Data() + off, M_[i].Data(),
			static_cast<size_t>(n) * sizeof(OaF32));
		std::memcpy(OutOam.AdamV.Data() + off, V_[i].Data(),
			static_cast<size_t>(n) * sizeof(OaF32));
		off += n;
	}
}

void OaAdamW::LoadFrom(const OamModel& InOam) {
	// Hyperparams + step always restored (cheap, always present in checkpoint).
	Lr_          = InOam.Optimizer.Lr;
	Beta1_       = InOam.Optimizer.Beta1;
	Beta2_       = InOam.Optimizer.Beta2;
	Eps_         = InOam.Optimizer.Eps;
	WeightDecay_ = InOam.Optimizer.WeightDecay;
	Step_        = static_cast<OaU64>(InOam.Optimizer.Step);
	ResetMasterSeed();  // re-seed fp32 masters from the reloaded (bf16) weights

	if (not InOam.HasOptimizer()) {
		OA_LOG_INFO(OaLogComponent::Core,
			"OaAdamW::LoadFrom: checkpoint has no AdamM/AdamV section, keeping zero-init state");
		return;
	}

	// Allocate moment buffers if first use, then verify sizes line up.
	EnsureMomentBuffers(Params_, M_, V_);

	OaI64 expected = 0;
	for (const auto& m : M_) expected += m.NumElements();
	if (InOam.AdamM.Size() != static_cast<OaUsize>(expected)) {
		OA_LOG_ERROR(OaLogComponent::Core,
			"OaAdamW::LoadFrom: AdamM size mismatch — model expects %lld elements, "
			"checkpoint has %zu (model architecture differs from saved model)",
			static_cast<long long>(expected), InOam.AdamM.Size());
		return;
	}
	if (InOam.AdamV.Size() != InOam.AdamM.Size()) {
		OA_LOG_ERROR(OaLogComponent::Core,
			"OaAdamW::LoadFrom: AdamV size (%zu) != AdamM size (%zu)",
			InOam.AdamV.Size(), InOam.AdamM.Size());
		return;
	}

	// Drain pending GPU writes to M_/V_ before memcpy.
	(void)OaContext::GetDefault().Execute();

	OaI64 off = 0;
	for (OaUsize i = 0; i < M_.Size(); ++i) {
		OaI64 n = M_[i].NumElements();
		std::memcpy(M_[i].Data(), InOam.AdamM.Data() + off,
			static_cast<size_t>(n) * sizeof(OaF32));
		std::memcpy(V_[i].Data(), InOam.AdamV.Data() + off,
			static_cast<size_t>(n) * sizeof(OaF32));
		off += n;
	}
}
