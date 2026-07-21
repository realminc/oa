#include <Oa/Ml/Optim.h>
#include <Oa/Ml/FnOptim.h>
#include <Oa/Ml/Oam.h>
#include <Oa/Core/FnMatrix.h>
#include <Oa/Core/Log.h>
#include <Oa/Runtime/Context.h>
#include <Oa/Runtime/Engine.h>
#include <Oa/Runtime/ExecutionMemory.h>

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

void OaAdamW::SetLr(OaF32 InLr) {
	Lr_ = InLr;
	if (not GraphState_.HasStorage() or GraphState_.Data() == nullptr) return;

	// Replay reads LR from the mutable state buffer rather than baked push
	// constants. OaItTraining waits for the replay before callbacks run, so this
	// host update cannot race the preceding optimizer dispatch.
	auto* state = static_cast<OaU32*>(GraphState_.Data());
	std::memcpy(state + 1, &Lr_, sizeof(Lr_));
	if (auto* runtime = OaContext::GetDefault().GetEngine()) {
		(void)runtime->Allocator.FlushHostBuffer(
			GraphState_.GetVkBuffer(), sizeof(OaU32), sizeof(OaF32));
	}
}

void OaAdamW::Step() {
	++Step_;

	// Lazy-initialize moment buffers on first step
	EnsureMomentBuffers(Params_, M_, V_);
	// Lazy-allocate + seed fp32 master weights for any low-precision (bf16) params.
	PrepareMasters();

	if (not GraphState_.HasStorage()) {
		// Replay metadata is intentionally host-visible: callbacks update LR and
		// the completed step between submissions. Model matrices remain governed
		// by the engine's device-local placement policy.
		GraphState_ = OaFnMatrix::Empty(
			OaMatrixShape{6}, OaScalarType::UInt32, OaMemoryPlacement::HostUpload);
	}
	auto& ctx = OaContext::GetDefault();
	const OaBool replayState = OaExecutionMemory::IsStableFrameActive(ctx)
		and GraphState_.HasStorage();
	if (replayState) {
		auto* state = static_cast<OaU32*>(GraphState_.Data());
		// The recorded graph advances this word on the GPU before any parameter
		// update. Seeding the previous completed step preserves eager semantics on
		// the first execution and lets subsequent command-buffer replays advance
		// without a host rewrite.
		state[0] = static_cast<OaU32>(Step_ - 1U);
		const OaF32 scalars[] = {Lr_, Beta1_, Beta2_, Eps_, WeightDecay_};
		std::memcpy(state + 1, scalars, sizeof(scalars));
		if (auto* runtime = ctx.GetEngine()) {
			(void)runtime->Allocator.FlushHostBuffer(
				GraphState_.GetVkBuffer(), 0, static_cast<OaU64>(GraphState_.ByteSize()));
		}
		OaFnOptim::AdamWAdvanceGraphState(GraphState_);
	}

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
		if (grads4[0].HasStorage() and grads4[1].HasStorage()
			and grads4[2].HasStorage() and grads4[3].HasStorage()) {
			OaFnOptim::OaAdamWParamSet sets[4] = {
				{&Params_[0]->Data, &M_[0], &V_[0], &grads4[0]},
				{&Params_[1]->Data, &M_[1], &V_[1], &grads4[1]},
				{&Params_[2]->Data, &M_[2], &V_[2], &grads4[2]},
				{&Params_[3]->Data, &M_[3], &V_[3], &grads4[3]},
			};
			if (replayState) {
				OaFnOptim::AdamWStepManyGraph(
					OaSpan<const OaFnOptim::OaAdamWParamSet>(sets, 4), GraphState_);
			} else {
				OaFnOptim::AdamWStepMany(
					OaSpan<const OaFnOptim::OaAdamWParamSet>(sets, 4),
					Lr_, Beta1_, Beta2_, Eps_, WeightDecay_, static_cast<OaI32>(Step_));
			}
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
		if (!grad.HasStorage()) continue;

		OaMatrix gradUse = MasterGrad(i, grad);
		if (replayState) {
			OaFnOptim::AdamWStepGraph(
				MasterOrData(i), M_[i], V_[i], gradUse, GraphState_);
		} else {
			OaFnOptim::AdamWStep(
				MasterOrData(i), M_[i], V_[i], gradUse,
				Lr_, Beta1_, Beta2_, Eps_, WeightDecay_, static_cast<OaI32>(Step_));
		}
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
		if (g.HasStorage()) grads.PushBack(g);
	}
	for (OaUsize i = 0; i < grads.Size(); i += 4) {
		OaUsize end = std::min(i + 4, grads.Size());
		OaFnMatrix::MultiFill(OaSpan<OaMatrix>(grads.Data() + i, end - i), 0.0F);
	}
}

// ─── Persistence ──────────────────────────────────────────────────────────

OaStatus OaAdamW::SaveTo(OamModel& OutOam) const {
	// Header (hyperparams + step count). NumParams is total flat element count.
	OutOam.OptimizerPresent = true;
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
		return OaStatus::Ok();
	}

	// Drain pending GPU writes to M_/V_ so the memcpy sees the latest state.
	OA_RETURN_IF_ERROR(OaContext::GetDefault().Execute());

	OaI64 total = 0;
	for (const auto& m : M_) total += m.NumElements();
	OutOam.Optimizer.NumParams = static_cast<OaU64>(total);
	OutOam.AdamM.Resize(total);
	OutOam.AdamV.Resize(total);

	OaI64 off = 0;
	for (OaUsize i = 0; i < M_.Size(); ++i) {
		OaI64 n = M_[i].NumElements();
		const auto bytes = static_cast<OaU64>(n) * sizeof(OaF32);
		OA_RETURN_IF_ERROR(OaFnMatrix::CopyToHost(
			M_[i], OutOam.AdamM.Data() + off, bytes));
		OA_RETURN_IF_ERROR(OaFnMatrix::CopyToHost(
			V_[i], OutOam.AdamV.Data() + off, bytes));
		off += n;
	}
	return OaStatus::Ok();
}

OaStatus OaAdamW::ValidateLoad(const OamModel& InOam) const {
	if (not InOam.HasOptimizer()
		or std::strncmp(InOam.Optimizer.Type, "AdamW", sizeof(InOam.Optimizer.Type)) != 0)
	{
		return OaStatus::Error(OaStatusCode::FailedPrecondition,
			"AdamW checkpoint optimizer state is missing or has the wrong type");
	}
	OaU64 expected = 0;
	for (const auto* parameter : Params_) {
		expected += static_cast<OaU64>(parameter->Data.NumElements());
	}
	if (InOam.Optimizer.Step == 0 and InOam.AdamM.Empty()
		and InOam.AdamV.Empty()) return OaStatus::Ok();
	if (InOam.AdamM.Size() != expected or InOam.AdamV.Size() != expected) {
		return OaStatus::Error(OaStatusCode::ShapeMismatch,
			"AdamW checkpoint moment size does not match the model");
	}
	return OaStatus::Ok();
}

OaStatus OaAdamW::LoadFrom(const OamModel& InOam) {
	OA_RETURN_IF_ERROR(ValidateLoad(InOam));
	// Hyperparams + step always restored (cheap, always present in checkpoint).
	Lr_          = InOam.Optimizer.Lr;
	Beta1_       = InOam.Optimizer.Beta1;
	Beta2_       = InOam.Optimizer.Beta2;
	Eps_         = InOam.Optimizer.Eps;
	WeightDecay_ = InOam.Optimizer.WeightDecay;
	Step_        = static_cast<OaU64>(InOam.Optimizer.Step);
	ResetMasterSeed();  // re-seed fp32 masters from the reloaded (bf16) weights

	// Allocate moment buffers if first use, then verify sizes line up.
	EnsureMomentBuffers(Params_, M_, V_);
	if (InOam.AdamM.Empty()) return OaStatus::Ok();

	// Drain pending GPU writes to M_/V_ before memcpy.
	OA_RETURN_IF_ERROR(OaContext::GetDefault().Execute());

	OaI64 off = 0;
	for (OaUsize i = 0; i < M_.Size(); ++i) {
		OaI64 n = M_[i].NumElements();
		const auto bytes = static_cast<OaU64>(n) * sizeof(OaF32);
		if (auto* runtime = OaContext::GetDefault().GetEngine()) {
			OA_RETURN_IF_ERROR(runtime->UploadBuffer(M_[i].GetVkBuffer(), 0,
				InOam.AdamM.Data() + off, bytes));
			OA_RETURN_IF_ERROR(runtime->UploadBuffer(V_[i].GetVkBuffer(), 0,
				InOam.AdamV.Data() + off, bytes));
		} else {
			return OaStatus::Error(OaStatusCode::FailedPrecondition,
				"AdamW restore requires an active OA engine");
		}
		off += n;
	}
	return OaStatus::Ok();
}
