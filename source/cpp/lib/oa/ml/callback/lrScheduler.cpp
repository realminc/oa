#include <oa/ml/lrScheduler.h>
#include <oa/ml/optim.h>
#include <oa/core/std/algo.h>
#include <oa/core/std/scalarMath.h>
#include <oa/core/std/utility.h>

static constexpr oa::F32 kPi = 3.14159265358979323846f;

oa::F32 oa::CosineScheduler::getLr(oa::U64 inStep) const {
	if (inStep >= totalSteps_) return minLr_;
	oa::F32 progress = static_cast<oa::F32>(inStep) / static_cast<oa::F32>(totalSteps_);
	return minLr_ + 0.5f * (maxLr_ - minLr_) * (1.0f + oa::cos(progress * kPi));
}

oa::F32 oa::WarmupScheduler::getLr(oa::U64 inStep) const {
	if (inStep < warmupSteps_) {
		return targetLr_ * static_cast<oa::F32>(inStep + 1) / static_cast<oa::F32>(warmupSteps_);
	}
	if (after_) return after_->getLr(inStep - warmupSteps_);
	return targetLr_;
}

// 1cycle: ramp up to MaxLr, then cosine anneal down
oa::F32 oa::OneCycleScheduler::getLr(oa::U64 inStep) const {
	oa::F32 initialLr = maxLr_ / divFactor_;
	oa::F32 finalLr = initialLr / finalDivFactor_;
	oa::F32 step = static_cast<oa::F32>(inStep);
	oa::F32 total = static_cast<oa::F32>(totalSteps_);
	oa::F32 upSteps = pctStart_ * total;
	oa::F32 downSteps = total - upSteps;

	if (step <= upSteps && upSteps > 0.0f) {
		oa::F32 pct = step / upSteps;
		return finalLr + (maxLr_ - finalLr) / 2.0f * (1.0f - oa::cos(kPi * pct));
	}
	if (downSteps > 0.0f) {
		oa::F32 pct = (step - upSteps) / downSteps;
		if (pct > 1.0f) pct = 1.0f;
		return finalLr + (maxLr_ - finalLr) / 2.0f * (1.0f + oa::cos(kPi * pct));
	}
	return finalLr;
}

// CyclicLR: triangular oscillation between BaseLr and MaxLr
oa::F32 oa::CyclicScheduler::getLr(oa::U64 inStep) const {
	oa::F32 step = static_cast<oa::F32>(inStep);
	oa::F32 sizeUp = static_cast<oa::F32>(stepSizeUp_);
	oa::F32 cycle = oa::floor(step / (2.0f * sizeUp));
	oa::F32 x = oa::abs(step / sizeUp - 2.0f * cycle - 1.0f);

	oa::F32 scale = 1.0f;
	switch (mode_) {
	case oa::CyclicMode::Triangular:
		scale = 1.0f;
		break;
	case oa::CyclicMode::Triangular2:
		scale = 1.0f / oa::pow(2.0f, cycle);
		break;
	case oa::CyclicMode::ExpRange:
		scale = oa::pow(gamma_, step);
		break;
	}

	oa::F32 ramp = (1.0f - x > 0.0f) ? 1.0f - x : 0.0f;
	return baseLr_ + (maxLr_ - baseLr_) * ramp * scale;
}

// SGDR: cosine annealing with warm restarts. Period T0, multiplied by TMult each restart.
oa::F32 oa::CosineWarmRestartsScheduler::getLr(oa::U64 inStep) const {
	oa::U64 tCur = inStep;
	oa::U64 ti = t0_;

	if (tMult_ == 1) {
		tCur = inStep % t0_;
	} else {
		oa::U64 cumulative = 0;
		oa::U64 period = t0_;
		while (cumulative + period <= inStep) {
			cumulative += period;
			period *= tMult_;
		}
		tCur = inStep - cumulative;
		ti = period;
	}

	oa::F32 progress = static_cast<oa::F32>(tCur) / static_cast<oa::F32>(ti);
	return etaMin_ + 0.5f * (maxLr_ - etaMin_) * (1.0f + oa::cos(kPi * progress));
}

// ReduceOnPlateau: stateful metric-driven scheduler
void oa::ReduceOnPlateauScheduler::step(oa::F32 inMetric) {
	bool improved = false;
	if (mode_ == oa::PlateauMode::Min) {
		improved = inMetric < best_ - threshold_;
	} else {
		improved = inMetric > best_ + threshold_;
	}

	if (improved) {
		best_ = inMetric;
		numBadEpochs_ = 0;
	} else {
		numBadEpochs_++;
	}

	if (numBadEpochs_ > patience_) {
		oa::F32 newLr = currentLr_ * factor_;
		currentLr_ = (newLr > minLr_) ? newLr : minLr_;
		numBadEpochs_ = 0;
	}
}

oa::F32 oa::ReduceOnPlateauScheduler::getLr(oa::U64) const {
	return currentLr_;
}

// Sequential: delegate to the scheduler that covers the given step
oa::F32 oa::SequentialScheduler::getLr(oa::U64 inStep) const {
	if (schedulers_.empty()) return 0.0f;

	oa::U64 offset = 0;
	for (oa::Usize i = 0; i < milestones_.size() && i < schedulers_.size(); ++i) {
		if (inStep < milestones_[i]) {
			return schedulers_[i]->getLr(inStep - offset);
		}
		offset = milestones_[i];
	}

	return schedulers_.back()->getLr(inStep - offset);
}

// LinearWarmupCosine: compose warmup + cosine
oa::LinearWarmupCosineScheduler::LinearWarmupCosineScheduler(
	oa::I32 inWarmupSteps, oa::I32 inTotalSteps, oa::F32 inMaxLr, oa::F32 inMinLr) {
	const oa::U64 w = static_cast<oa::U64>(oa::max(0, inWarmupSteps));
	const oa::I32 cosSpan = oa::max(1, inTotalSteps - inWarmupSteps);
	auto cos = oa::makeShared<oa::CosineScheduler>(inMaxLr, inMinLr, static_cast<oa::U64>(cosSpan));
	inner_ = oa::makeShared<oa::WarmupScheduler>(inMaxLr, w, oa::move(cos));
}

oa::F32 oa::LinearWarmupCosineScheduler::getLr(oa::U64 inStep) const {
	if (inStep == 0) {
		return inner_->getLr(0);
	}
	return inner_->getLr(inStep - 1);
}
