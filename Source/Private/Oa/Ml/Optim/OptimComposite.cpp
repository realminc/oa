#include <Oa/Ml/Optim.h>
#include <Oa/Ml/Oam.h>
#include <Oa/Core/Log.h>

#include <cstring>

void OaOptimizerComposite::Add(OaUniquePtr<OaOptimizer> InOptimizer) {
	if (!InOptimizer) return;
	if (Children_.Empty()) {
		Lr_ = InOptimizer->Lr();
		Step_ = InOptimizer->GetStep();
	}
	Children_.PushBack(std::move(InOptimizer));
}

void OaOptimizerComposite::InitDualLr(OaF32 InMuonLr, OaF32 InAdamWLr) {
	BaseMuonLr_ = InMuonLr;
	BaseAdamWLr_ = InAdamWLr;
	for (auto& child : Children_) {
		if (dynamic_cast<OaMuon*>(child.get())) {
			child->SetLr(InMuonLr);
		} else if (dynamic_cast<OaAdamW*>(child.get())) {
			child->SetLr(InAdamWLr);
		}
	}
}

void OaOptimizerComposite::SetLr(OaF32 InLr) {
	Lr_ = InLr;
	if (BaseAdamWLr_ <= 0.0f) {
		BaseAdamWLr_ = InLr;
	}
	const OaF32 scale = InLr / BaseAdamWLr_;
	for (auto& child : Children_) {
		if (dynamic_cast<OaMuon*>(child.get())) {
			child->SetLr(BaseMuonLr_ * scale);
		} else {
			child->SetLr(InLr);
		}
	}
}

void OaOptimizerComposite::Step() {
	++Step_;
	for (auto& child : Children_) {
		child->Step();
	}
}

void OaOptimizerComposite::NotifyProgramReplay(OaU64 InCount) {
	Step_ += InCount;
	for (auto& child : Children_) {
		child->NotifyProgramReplay(InCount);
	}
}

void OaOptimizerComposite::ZeroGrad() {
	for (auto& child : Children_) {
		child->ZeroGrad();
	}
}

void OaOptimizerComposite::SaveTo(OamModel& OutOam) const {
	OaAdamW* adam = nullptr;
	OaMuon* muon = nullptr;
	for (const auto& child : Children_) {
		if (!adam) adam = dynamic_cast<OaAdamW*>(child.get());
		if (!muon) muon = dynamic_cast<OaMuon*>(child.get());
	}

	if (!adam || !muon) {
		if (Children_.Size() == 1 && Children_[0]) {
			Children_[0]->SaveTo(OutOam);
		}
		return;
	}

	OamModel adamPart;
	OamModel muonPart;
	adam->SaveTo(adamPart);
	muon->SaveTo(muonPart);

	OutOam.Optimizer = adamPart.Optimizer;
	std::strncpy(OutOam.Optimizer.Type, "MuonAdamW", sizeof(OutOam.Optimizer.Type) - 1);
	OutOam.Optimizer.Beta1 = muonPart.Optimizer.Beta1;
	OutOam.Optimizer.Beta2 = muonPart.Optimizer.Lr;
	OutOam.Optimizer.Step = static_cast<OaI64>(Step_);

	OutOam.AdamM = std::move(adamPart.AdamM);
	OutOam.AdamV = std::move(adamPart.AdamV);
	OutOam.MuonM = std::move(muonPart.AdamM);
	OamSetMuonNumParams(OutOam.Optimizer, OutOam.MuonM.Size());
}

void OaOptimizerComposite::LoadFrom(const OamModel& InOam) {
	if (!OamIsMuonAdamWType(InOam.Optimizer)) {
		OA_LOG_WARN(OaLogComponent::ML,
			"OaOptimizerComposite::LoadFrom: expected MuonAdamW checkpoint, got '%s'",
			InOam.Optimizer.Type);
		if (Children_.Size() == 1 && Children_[0]) {
			Children_[0]->LoadFrom(InOam);
		}
		return;
	}

	Lr_ = InOam.Optimizer.Lr;
	Step_ = static_cast<OaU64>(InOam.Optimizer.Step);
	BaseAdamWLr_ = InOam.Optimizer.Lr;
	BaseMuonLr_ = InOam.Optimizer.Beta2;

	OaAdamW* adam = nullptr;
	OaMuon* muon = nullptr;
	for (auto& child : Children_) {
		if (!adam) adam = dynamic_cast<OaAdamW*>(child.get());
		if (!muon) muon = dynamic_cast<OaMuon*>(child.get());
	}
	if (!adam || !muon) {
		OA_LOG_ERROR(OaLogComponent::ML,
			"OaOptimizerComposite::LoadFrom: need Muon+AdamW children");
		return;
	}

	adam->LoadFrom(InOam);

	OamModel muonPart;
	muonPart.Optimizer = InOam.Optimizer;
	std::strncpy(muonPart.Optimizer.Type, "Muon", sizeof(muonPart.Optimizer.Type) - 1);
	muonPart.Optimizer.Lr = InOam.Optimizer.Beta2;
	muonPart.Optimizer.Beta1 = InOam.Optimizer.Beta1;
	muonPart.Optimizer.NumParams = OamGetMuonNumParams(InOam.Optimizer);
	muonPart.AdamM = InOam.MuonM;
	muon->LoadFrom(muonPart);

	InitDualLr(BaseMuonLr_, BaseAdamWLr_);
}
