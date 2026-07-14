#include <Oa/Ml/Metric.h>
#include <Oa/Core/FnMatrix.h>

#include <algorithm>
#include <cstdio>
#include <cstring>

// ─── OaMetricValueFormatter ────────────────────────────────────────────────

OaI32 OaMetricValueFormatter::Format(char* OutBuffer, OaI32 InBufferSize, OaF64 InValue) {
	if (OutBuffer == nullptr or InBufferSize <= 0) return 0;
	if (not std::isfinite(InValue)) {
		return std::snprintf(OutBuffer, static_cast<size_t>(InBufferSize), "%g", InValue);
	}

	const OaF64 magnitude = std::abs(InValue);
	OaI32 requiredPrecision = DefaultPrecision_;
	if (magnitude > 0.0 and magnitude < 1.0) {
		const OaI32 leadingZeros = std::max<OaI32>(0,
			-static_cast<OaI32>(std::floor(std::log10(magnitude))) - 1);
		requiredPrecision = std::max(requiredPrecision, leadingZeros + 3);
	}
	if (requiredPrecision > MaxPrecision_ or magnitude >= 1.0e9) {
		HavePrevious_ = true;
		PreviousValue_ = InValue;
		return std::snprintf(OutBuffer, static_cast<size_t>(InBufferSize), "%.3g", InValue);
	}
	Precision_ = std::max(Precision_, requiredPrecision);

	if (HavePrevious_ and InValue != PreviousValue_) {
		while (Precision_ < MaxPrecision_) {
			const OaF64 scale = std::pow(10.0, static_cast<OaF64>(Precision_));
			const OaF64 previousRounded = std::round(PreviousValue_ * scale) / scale;
			const OaF64 currentRounded = std::round(InValue * scale) / scale;
			if (previousRounded != currentRounded) break;
			++Precision_;
		}
	}
	HavePrevious_ = true;
	PreviousValue_ = InValue;

	char fixed[96]{};
	std::snprintf(fixed, sizeof(fixed), "%.*f", Precision_, InValue);
	char* end = fixed + std::strlen(fixed);
	while (end > fixed and end[-1] == '0') --end;
	if (end > fixed and end[-1] == '.') --end;
	*end = '\0';
	return std::snprintf(OutBuffer, static_cast<size_t>(InBufferSize), "%s", fixed);
}

// ─── OaMetricLoss ──────────────────────────────────────────────────────────

void OaMetricLoss::Update(const OaMatrix& InPreds, const OaMatrix& InLabels) {
	// For loss, preds is actually the loss tensor, labels is unused
	(void)InLabels;
	OaF32 lossValue = OaFnMatrix::Scalar(InPreds);
	Update(lossValue);
}

void OaMetricLoss::Update(OaF32 InLossValue) {
	// Freeze the loss name on the first completed sample. OaFnLoss::LastName()
	// is process-global and may change later during evaluation or checkpoint IO.
	if (Name_.empty()) {
		const char* name = OaFnLoss::LastName();
		Name_ = name ? name : "loss";
	}
	Sum_ += static_cast<OaF64>(InLossValue);
	Last_ = static_cast<OaF64>(InLossValue);
	Count_++;
}

void OaMetricLoss::Reset() {
	Sum_ = 0.0;
	Last_ = 0.0;
	Count_ = 0;
}

OaF64 OaMetricLoss::Result() const {
	if (Count_ == 0) return 0.0;
	return Mode_ == Mode::Last ? Last_ : Sum_ / static_cast<OaF64>(Count_);
}

OaI32 OaMetricLoss::Render(char* OutBuffer, OaI32 InBufferSize, bool InFirst) const {
	(void)InFirst; // Separator is handled by OaCbProgressBar
	if (InBufferSize < 32) return 0;
	const char* name = Name();
	if (Count_ == 0) return snprintf(OutBuffer, InBufferSize, "%s: n/a", name);
	// Progress bars are space-constrained and the metric's aggregation policy is
	// already part of its configuration. Summaries spell out initial/final/mean.
	char value[96]{};
	Formatter_.Format(value, sizeof(value), Result());
	return snprintf(OutBuffer, InBufferSize, "%s: %s", name, value);
}

// ─── OaMetricAccuracy ───────────────────────────────────────────────────────

void OaMetricAccuracy::Update(const OaMatrix& InPreds, const OaMatrix& InLabels) {
	auto count = OaFnMatrix::CategoricalAccuracyCount(InPreds, InLabels);
	if (count.IsEmpty()) return;
	auto& ctx = OaContext::GetDefault();
	if (not ctx.Execute().IsOk() or not ctx.Sync().IsOk()) return;
	Correct_ += static_cast<OaI64>(count.DataAs<const OaU32>()[0]);
	Total_ += InLabels.NumElements();
}

void OaMetricAccuracy::Reset() {
	Correct_ = 0;
	Total_ = 0;
}

OaF64 OaMetricAccuracy::Result() const {
	return Total_ > 0 ? static_cast<OaF64>(Correct_) / static_cast<OaF64>(Total_) : 0.0;
}

OaI32 OaMetricAccuracy::Render(char* OutBuffer, OaI32 InBufferSize, bool InFirst) const {
	(void)InFirst; // Separator is handled by OaCbProgressBar
	if (InBufferSize < 32) return 0;
	if (Total_ == 0) return snprintf(OutBuffer, InBufferSize, "accuracy: n/a");
	return snprintf(OutBuffer, InBufferSize, "accuracy: %.4f", Result());
}

// ─── OaFnMetric ───────────────────────────────────────────────────────────

namespace OaFnMetric {

OaF32 ScalarLoss(const OaMatrix& InLossTensor) {
	return OaFnMatrix::Scalar(InLossTensor);
}

OaF32 Accuracy(const OaMatrix& InPreds, const OaMatrix& InLabels) {
	// One-shot accuracy calculation
	OaMetricAccuracy acc;
	acc.Update(InPreds, InLabels);
	return static_cast<OaF32>(acc.Result());
}

} // namespace OaFnMetric
