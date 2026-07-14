#include <Oa/Ml/OptimUtil.h>
#include <Oa/Ml/MuonRef.h>
#include <Oa/Core/FnMatrix.h>

// ═════════════════════════════════════════════════════════════════════════════
// OaOptimizer — shared mixed-precision master-weight machinery
// ═════════════════════════════════════════════════════════════════════════════
// Every optimizer (AdamW / SGD / Muon / …) inherits these. A low-precision
// (bf16/fp16) parameter gets an fp32 master accumulator; the optimizer math runs
// in fp32 on the master (grad upcast), and the bf16 weight the forward reads is
// re-derived as round(master). No optimizer needs a precision-aware kernel.

static bool OaIsLowPrecParam(const OaMatrix& InData) {
	const OaScalarType dt = InData.GetDtype();
	return dt == OaScalarType::BFloat16 || dt == OaScalarType::Float16;
}

void OaOptimizer::PrepareMasters() {
	if (not MixedReady_) {
		Master_.Resize(Params_.Size());
		GradF32_.Resize(Params_.Size());
		for (OaUsize i = 0; i < Params_.Size(); ++i) {
			OaParameter* p = Params_[i];
			if (p != nullptr and OaIsLowPrecParam(p->Data)) {
				Master_[i] = OaFnMatrix::Empty(p->Data.GetShape(), OaScalarType::Float32);
			}
		}
		MixedReady_ = true;
	}
	if (not MasterSeeded_) {
		for (OaUsize i = 0; i < Master_.Size(); ++i) {
			if (not Master_[i].IsEmpty()) {
				OaFnMatrix::CastInto(Params_[i]->Data, Master_[i]);  // bf16 → fp32 seed
			}
		}
		MasterSeeded_ = true;
	}
}

bool OaOptimizer::HasMaster(OaUsize InIdx) const {
	return InIdx < Master_.Size() and not Master_[InIdx].IsEmpty();
}

OaMatrix& OaOptimizer::MasterOrData(OaUsize InIdx) {
	return HasMaster(InIdx) ? Master_[InIdx] : Params_[InIdx]->Data;
}

OaMatrix OaOptimizer::MasterGrad(OaUsize InIdx, const OaMatrix& InGrad) {
	if (not HasMaster(InIdx)) return InGrad;
	GradF32_[InIdx] = OaFnMatrix::Cast(InGrad, OaScalarType::Float32);  // bf16 → fp32
	return GradF32_[InIdx];
}

void OaOptimizer::WritebackMaster(OaUsize InIdx) {
	if (HasMaster(InIdx)) {
		OaFnMatrix::CastInto(Master_[InIdx], Params_[InIdx]->Data);  // fp32 → bf16
	}
}

OaUniquePtr<OaOptimizerComposite> MakeMuonAdamWOptimizer(
	OaModule& InModel,
	const OaMuonAdamWConfig& InCfg)
{
	const auto named = InModel.AllNamedParameterPtrs();
	OaMuonRef::OaOfficialMuonSplit split = OaMuonRef::SplitOfficialRouting(
		OaSpan<const OaNamedParameter>(named.Data(), named.Size()));

	auto composite = OaMakeUniquePtr<OaOptimizerComposite>();
	composite->Add(OaMakeUniquePtr<OaMuon>(
		split.Muon,
		InCfg.MuonLr,
		InCfg.MuonBeta,
		InCfg.MuonWeightDecay,
		InCfg.MuonEps,
		InCfg.MuonNs5Iters));
	composite->Add(OaMakeUniquePtr<OaAdamW>(
		split.AdamW,
		InCfg.AdamWLr,
		InCfg.AdamWBeta1,
		InCfg.AdamWBeta2,
		InCfg.AdamWEps,
		InCfg.AdamWWeightDecay));
	composite->InitDualLr(InCfg.MuonLr, InCfg.AdamWLr);
	return composite;
}