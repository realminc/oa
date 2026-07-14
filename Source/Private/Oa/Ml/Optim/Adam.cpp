#include <Oa/Ml/Optim.h>
#include <Oa/Ml/Oam.h>
#include <Oa/Runtime/Context.h>
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
		if (!grad.Data()) continue;
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

void OaAdam::SaveTo(OamModel& OutOam) const {
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
		return;
	}

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

void OaAdam::LoadFrom(const OamModel& InOam) {
	Lr_    = InOam.Optimizer.Lr;
	Beta1_ = InOam.Optimizer.Beta1;
	Beta2_ = InOam.Optimizer.Beta2;
	Eps_   = InOam.Optimizer.Eps;
	Step_  = static_cast<OaU64>(InOam.Optimizer.Step);

	if (not InOam.HasOptimizer()) {
		OA_LOG_INFO(OaLogComponent::Core,
			"OaAdam::LoadFrom: checkpoint has no AdamM/AdamV section, keeping zero-init state");
		return;
	}

	if (M_.Empty()) {
		for (auto* p : Params_) {
			M_.PushBack(OaFnMatrix::Zeros(p->Data.GetShape()));
			V_.PushBack(OaFnMatrix::Zeros(p->Data.GetShape()));
		}
	}

	OaI64 expected = 0;
	for (const auto& m : M_) expected += m.NumElements();
	if (InOam.AdamM.Size() != static_cast<OaUsize>(expected)) {
		OA_LOG_ERROR(OaLogComponent::Core,
			"OaAdam::LoadFrom: AdamM size mismatch — model expects %lld, checkpoint has %zu",
			static_cast<long long>(expected), InOam.AdamM.Size());
		return;
	}

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
