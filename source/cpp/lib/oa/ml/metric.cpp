#include <oa/ml/metric.h>
#include <oa/core/fnMatrix.h>
#include <oa/runtime/executionSession.h>

#include <algorithm>
#include <cstdio>
#include <cstring>

// ─── oa::MetricValueFormatter ────────────────────────────────────────────────

oa::I32 oa::MetricValueFormatter::format(char* outBuffer, oa::I32 inBufferSize, oa::F64 inValue) {
	if (outBuffer == nullptr or inBufferSize <= 0) return 0;
	if (not std::isfinite(inValue)) {
		return std::snprintf(outBuffer, static_cast<size_t>(inBufferSize), "%g", inValue);
	}

	const oa::F64 magnitude = std::abs(inValue);
	oa::I32 requiredPrecision = defaultPrecision_;
	if (magnitude > 0.0 and magnitude < 1.0) {
		const oa::I32 leadingZeros = std::max<oa::I32>(0,
			-static_cast<oa::I32>(std::floor(std::log10(magnitude))) - 1);
		requiredPrecision = std::max(requiredPrecision, leadingZeros + 3);
	}
	if (requiredPrecision > maxPrecision_ or magnitude >= 1.0e9) {
		havePrevious_ = true;
		previousValue_ = inValue;
		return std::snprintf(outBuffer, static_cast<size_t>(inBufferSize), "%.3g", inValue);
	}
	precision_ = std::max(precision_, requiredPrecision);

	if (havePrevious_ and inValue != previousValue_) {
		while (precision_ < maxPrecision_) {
			const oa::F64 scale = std::pow(10.0, static_cast<oa::F64>(precision_));
			const oa::F64 previousRounded = std::round(previousValue_ * scale) / scale;
			const oa::F64 currentRounded = std::round(inValue * scale) / scale;
			if (previousRounded != currentRounded) break;
			++precision_;
		}
	}
	havePrevious_ = true;
	previousValue_ = inValue;

	char fixed[96]{};
	std::snprintf(fixed, sizeof(fixed), "%.*f", precision_, inValue);
	char* end = fixed + std::strlen(fixed);
	while (end > fixed and end[-1] == '0') --end;
	if (end > fixed and end[-1] == '.') --end;
	*end = '\0';
	return std::snprintf(outBuffer, static_cast<size_t>(inBufferSize), "%s", fixed);
}

// ─── oa::MetricLoss ──────────────────────────────────────────────────────────

void oa::MetricLoss::update(const oa::Matrix& inPreds, const oa::Matrix& inLabels) {
	// For loss, preds is actually the loss tensor, labels is unused
	(void)inLabels;
	oa::F32 lossValue = oa::FnMatrix::scalar(inPreds);
	update(lossValue);
}

void oa::MetricLoss::update(oa::F32 inLossValue) {
	// freeze the loss name on the first completed sample. oa::FnLoss::lastName()
	// is process-global and may change later during evaluation or checkpoint IO.
	if (name_.empty()) {
		const char* name = oa::FnLoss::lastName();
		name_ = name ? name : "loss";
	}
	sum_ += static_cast<oa::F64>(inLossValue);
	last_ = static_cast<oa::F64>(inLossValue);
	count_++;
}

void oa::MetricLoss::reset() {
	sum_ = 0.0;
	last_ = 0.0;
	count_ = 0;
}

oa::F64 oa::MetricLoss::result() const {
	if (count_ == 0) return 0.0;
	return mode_ == Mode::Last ? last_ : sum_ / static_cast<oa::F64>(count_);
}

oa::I32 oa::MetricLoss::render(char* outBuffer, oa::I32 inBufferSize, bool inFirst) const {
	(void)inFirst; // separator is handled by oa::CbProgressBar
	if (inBufferSize < 32) return 0;
	const char* metricName = name();
	if (count_ == 0) {
		return snprintf(outBuffer, inBufferSize, "%s: n/a", metricName);
	}
	// Progress bars are space-constrained and the metric's aggregation policy is
	// already part of its configuration. Summaries spell out initial/final/mean.
	char value[96]{};
	formatter_.format(value, sizeof(value), result());
	return snprintf(outBuffer, inBufferSize, "%s: %s", metricName, value);
}

// ─── oa::MetricAccuracy ───────────────────────────────────────────────────────

void oa::MetricAccuracy::update(const oa::Matrix& inPreds, const oa::Matrix& inLabels) {
	auto count = oa::FnMatrix::categoricalAccuracyCount(inPreds, inLabels);
	if (count.isEmpty()) return;
	auto& ctx = oa::ExecutionSession::getActive();
	if (not ctx.submitAndWait().isOk()) return;
	oa::U32 correct = 0;
	if (not oa::FnMatrix::copyToHost(count, &correct, sizeof(correct)).isOk()) return;
	correct_ += static_cast<oa::I64>(correct);
	total_ += inLabels.numElements();
}

void oa::MetricAccuracy::reset() {
	correct_ = 0;
	total_ = 0;
}

oa::F64 oa::MetricAccuracy::result() const {
	return total_ > 0 ? static_cast<oa::F64>(correct_) / static_cast<oa::F64>(total_) : 0.0;
}

oa::I32 oa::MetricAccuracy::render(char* outBuffer, oa::I32 inBufferSize, bool inFirst) const {
	(void)inFirst; // separator is handled by oa::CbProgressBar
	if (inBufferSize < 32) return 0;
	if (total_ == 0) return snprintf(outBuffer, inBufferSize, "accuracy: n/a");
	return snprintf(outBuffer, inBufferSize, "accuracy: %.4f", result());
}

// ─── oa::FnMetric ───────────────────────────────────────────────────────────

namespace oa {

namespace FnMetric {

oa::F32 scalarLoss(const oa::Matrix& inLossTensor) {
	return oa::FnMatrix::scalar(inLossTensor);
}

oa::F32 accuracy(const oa::Matrix& inPreds, const oa::Matrix& inLabels) {
	// One-shot accuracy calculation
	oa::MetricAccuracy acc;
	acc.update(inPreds, inLabels);
	return static_cast<oa::F32>(acc.result());
}

} // namespace FnMetric

} // namespace oa
