#include <Oa/Ml/LrScheduler.h>
#include <Oa/Ml/Optim.h>

#include <cmath>

static constexpr OaF32 kPi = 3.14159265358979323846f;

OaF32 OaCosineScheduler::GetLr(OaU64 InStep) const {
	if (InStep >= TotalSteps_) return MinLr_;
	OaF32 progress = static_cast<OaF32>(InStep) / static_cast<OaF32>(TotalSteps_);
	return MinLr_ + 0.5f * (MaxLr_ - MinLr_) * (1.0f + std::cos(progress * kPi));
}

OaF32 OaWarmupScheduler::GetLr(OaU64 InStep) const {
	if (InStep < WarmupSteps_) {
		return TargetLr_ * static_cast<OaF32>(InStep + 1) / static_cast<OaF32>(WarmupSteps_);
	}
	if (After_) return After_->GetLr(InStep - WarmupSteps_);
	return TargetLr_;
}

// 1cycle: ramp up to MaxLr, then cosine anneal down
OaF32 OaOneCycleScheduler::GetLr(OaU64 InStep) const {
	OaF32 initialLr = MaxLr_ / DivFactor_;
	OaF32 finalLr = initialLr / FinalDivFactor_;
	OaF32 step = static_cast<OaF32>(InStep);
	OaF32 total = static_cast<OaF32>(TotalSteps_);
	OaF32 upSteps = PctStart_ * total;
	OaF32 downSteps = total - upSteps;

	if (step <= upSteps && upSteps > 0.0f) {
		OaF32 pct = step / upSteps;
		return finalLr + (MaxLr_ - finalLr) / 2.0f * (1.0f - std::cos(kPi * pct));
	}
	if (downSteps > 0.0f) {
		OaF32 pct = (step - upSteps) / downSteps;
		if (pct > 1.0f) pct = 1.0f;
		return finalLr + (MaxLr_ - finalLr) / 2.0f * (1.0f + std::cos(kPi * pct));
	}
	return finalLr;
}

// CyclicLR: triangular oscillation between BaseLr and MaxLr
OaF32 OaCyclicScheduler::GetLr(OaU64 InStep) const {
	OaF32 step = static_cast<OaF32>(InStep);
	OaF32 sizeUp = static_cast<OaF32>(StepSizeUp_);
	OaF32 cycle = std::floor(step / (2.0f * sizeUp));
	OaF32 x = std::abs(step / sizeUp - 2.0f * cycle - 1.0f);

	OaF32 scale = 1.0f;
	switch (Mode_) {
	case OaCyclicMode::Triangular:
		scale = 1.0f;
		break;
	case OaCyclicMode::Triangular2:
		scale = 1.0f / std::pow(2.0f, cycle);
		break;
	case OaCyclicMode::ExpRange:
		scale = std::pow(Gamma_, step);
		break;
	}

	OaF32 ramp = (1.0f - x > 0.0f) ? 1.0f - x : 0.0f;
	return BaseLr_ + (MaxLr_ - BaseLr_) * ramp * scale;
}

// SGDR: cosine annealing with warm restarts. Period T0, multiplied by TMult each restart.
OaF32 OaCosineWarmRestartsScheduler::GetLr(OaU64 InStep) const {
	OaU64 tCur = InStep;
	OaU64 ti = T0_;

	if (TMult_ == 1) {
		tCur = InStep % T0_;
	} else {
		OaU64 cumulative = 0;
		OaU64 period = T0_;
		while (cumulative + period <= InStep) {
			cumulative += period;
			period *= TMult_;
		}
		tCur = InStep - cumulative;
		ti = period;
	}

	OaF32 progress = static_cast<OaF32>(tCur) / static_cast<OaF32>(ti);
	return EtaMin_ + 0.5f * (MaxLr_ - EtaMin_) * (1.0f + std::cos(kPi * progress));
}

// ReduceOnPlateau: stateful metric-driven scheduler
void OaReduceOnPlateauScheduler::Step(OaF32 InMetric) {
	bool improved = false;
	if (Mode_ == OaPlateauMode::Min) {
		improved = InMetric < Best_ - Threshold_;
	} else {
		improved = InMetric > Best_ + Threshold_;
	}

	if (improved) {
		Best_ = InMetric;
		NumBadEpochs_ = 0;
	} else {
		NumBadEpochs_++;
	}

	if (NumBadEpochs_ > Patience_) {
		OaF32 newLr = CurrentLr_ * Factor_;
		CurrentLr_ = (newLr > MinLr_) ? newLr : MinLr_;
		NumBadEpochs_ = 0;
	}
}

OaF32 OaReduceOnPlateauScheduler::GetLr(OaU64) const {
	return CurrentLr_;
}

// Sequential: delegate to the scheduler that covers the given step
OaF32 OaSequentialScheduler::GetLr(OaU64 InStep) const {
	if (Schedulers_.Empty()) return 0.0f;

	OaU64 offset = 0;
	for (OaUsize i = 0; i < Milestones_.Size() && i < Schedulers_.Size(); ++i) {
		if (InStep < Milestones_[i]) {
			return Schedulers_[i]->GetLr(InStep - offset);
		}
		offset = Milestones_[i];
	}

	return Schedulers_.Back()->GetLr(InStep - offset);
}

// LinearWarmupCosine: compose warmup + cosine
OaLinearWarmupCosineScheduler::OaLinearWarmupCosineScheduler(
	OaI32 InWarmupSteps, OaI32 InTotalSteps, OaF32 InMaxLr, OaF32 InMinLr) {
	const OaU64 w = static_cast<OaU64>(std::max(0, InWarmupSteps));
	const OaI32 cosSpan = std::max(1, InTotalSteps - InWarmupSteps);
	auto cos = OaMakeSharedPtr<OaCosineScheduler>(InMaxLr, InMinLr, static_cast<OaU64>(cosSpan));
	Inner_ = OaMakeSharedPtr<OaWarmupScheduler>(InMaxLr, w, std::move(cos));
}

OaF32 OaLinearWarmupCosineScheduler::GetLr(OaU64 InStep) const {
	if (InStep == 0) {
		return Inner_->GetLr(0);
	}
	return Inner_->GetLr(InStep - 1);
}
