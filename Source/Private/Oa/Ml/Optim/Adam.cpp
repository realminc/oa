#include <Oa/Ml/Optim.h>
#include <Oa/Ml/Oam.h>
#include <Oa/Runtime/Context.h>
#include <Oa/Runtime/Engine.h>
#include <Oa/Core/BufferAccess.h>
#include <Oa/Core/FnMatrix.h>
#include <Oa/Core/Validation.h>
#include <Oa/Core/Log.h>

#include <cstring>

static OaU32 DivCeil(OaU32 InA, OaU32 InB) { return (InA + InB - 1) / InB; }

static OaMatrix GetParamGrad(OaParameter* InP) {
	return InP->Grad();  // live grad (single source of truth: Data's autograd meta)
}

void OaAdam::Step() {
	++Step_;

	if (M_.Empty()) {
		for (auto* p : Params_) {
			M_.PushBack(OaFnMatrix::Zeros(p->Data.GetShape()));
			V_.PushBack(OaFnMatrix::Zeros(p->Data.GetShape()));
		}
	}

	auto& ctx = OaContext::GetDefault();
	for (OaUsize i = 0; i < Params_.Size(); ++i) {
		auto* p = Params_[i];
		OaMatrix grad = GetParamGrad(p);
		if (!grad.HasStorage()) continue;
		OaU32 n = static_cast<OaU32>(p->Data.NumElements());

		struct { OaU32 Count; OaF32 Lr; OaF32 Beta1; OaF32 Beta2; OaF32 Eps; OaU32 Step; }
			push{n, Lr_, Beta1_, Beta2_, Eps_, static_cast<OaU32>(Step_)};
		OaBufferAccess access[] = {
			OaBufferAccess::ReadWrite, OaBufferAccess::Read,
			OaBufferAccess::ReadWrite, OaBufferAccess::ReadWrite
		};
		ctx.Add("Adam", {&p->Data, &grad, &M_[i], &V_[i]}, access, &push, sizeof(push), DivCeil(n, 256));
	}
}

void OaAdam::ZeroGrad() {
	// Zero the single source of truth GPU-side (self-guarded Fill kernel, never a host
	// memset). See OaSGD::ZeroGrad.
	for (auto* p : Params_) { p->Data.ZeroGrad(); }
}

// Adam shares the OamModel optimizer schema with AdamW. WeightDecay is unused;
// stored as 0 in the header for round-trip consistency.

OaStatus OaAdam::SaveTo(OamModel& OutOam) const {
	OutOam.OptimizerPresent = true;
	OutOam.Optimizer = OamOptimizerHeader{};
	std::strncpy(OutOam.Optimizer.Type, "Adam", sizeof(OutOam.Optimizer.Type) - 1);
	OutOam.Optimizer.Lr    = Lr_;
	OutOam.Optimizer.Beta1 = Beta1_;
	OutOam.Optimizer.Beta2 = Beta2_;
	OutOam.Optimizer.Eps   = Eps_;
	OutOam.Optimizer.Step  = static_cast<OaI64>(Step_);

	if (M_.Empty()) {
		OutOam.Optimizer.NumParams = 0;
		OutOam.AdamM.Clear();
		OutOam.AdamV.Clear();
		return OaStatus::Ok();
	}

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

OaStatus OaAdam::ValidateLoad(const OamModel& InOam) const {
	if (not InOam.HasOptimizer()
		or std::strncmp(InOam.Optimizer.Type, "Adam", sizeof(InOam.Optimizer.Type)) != 0)
	{
		return OaStatus::Error(OaStatusCode::FailedPrecondition,
			"Adam checkpoint optimizer state is missing or has the wrong type");
	}
	OaU64 expected = 0;
	for (const auto* parameter : Params_) {
		expected += static_cast<OaU64>(parameter->Data.NumElements());
	}
	if (InOam.Optimizer.Step == 0 and InOam.AdamM.Empty()
		and InOam.AdamV.Empty()) return OaStatus::Ok();
	if (InOam.AdamM.Size() != expected or InOam.AdamV.Size() != expected) {
		return OaStatus::Error(OaStatusCode::ShapeMismatch,
			"Adam checkpoint moment size does not match the model");
	}
	return OaStatus::Ok();
}

OaStatus OaAdam::LoadFrom(const OamModel& InOam) {
	OA_RETURN_IF_ERROR(ValidateLoad(InOam));
	Lr_    = InOam.Optimizer.Lr;
	Beta1_ = InOam.Optimizer.Beta1;
	Beta2_ = InOam.Optimizer.Beta2;
	Eps_   = InOam.Optimizer.Eps;
	Step_  = static_cast<OaU64>(InOam.Optimizer.Step);

	if (M_.Empty()) {
		for (auto* p : Params_) {
			M_.PushBack(OaFnMatrix::Zeros(p->Data.GetShape()));
			V_.PushBack(OaFnMatrix::Zeros(p->Data.GetShape()));
		}
	}
	if (InOam.AdamM.Empty()) return OaStatus::Ok();

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
				"Adam restore requires an active OA engine");
		}
		off += n;
	}
	return OaStatus::Ok();
}
