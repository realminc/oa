#pragma once

#include <oa/core/types.h>
#include <oa/core/matrix.h>
#include <oa/ml/fnLoss.h>

namespace oa {

// ─── Metric ──────────────────────────────────────────────────────────────────
//
// base class for all metrics. Follows the keras pattern:
//   - update(): add a new batch of predictions/labels
//   - reset(): clear accumulated state
//   - result(): get the current metric value

class Metric {
public:
	virtual ~Metric() = default;

	virtual void update(const Matrix& inPreds, const Matrix& inLabels) = 0;

	// Consume one completed training step.
	virtual void updateStep(oa::F32 inLoss, bool inHasLoss,
		oa::F64 inElapsedSeconds, oa::I64 inStepCount) {
		(void)inLoss; (void)inHasLoss; (void)inElapsedSeconds; (void)inStepCount;
	}

	virtual void reset() = 0;

	[[nodiscard]] virtual oa::F64 result() const = 0;
	[[nodiscard]] virtual const char* name() const = 0;

	// Render the metric to a buffer for progress bar display.
	// inFirst: if true, don't prepend " · " separator.
	virtual oa::I32 render(char* outBuffer, oa::I32 inBufferSize, bool inFirst) const = 0;

	[[nodiscard]] virtual bool isLossMetric() const { return false; }
};

// Compact fixed-point formatting for live metrics.
class MetricValueFormatter {
public:
	MetricValueFormatter(oa::I32 inDefaultPrecision = 4, oa::I32 inMaxPrecision = 8)
		: defaultPrecision_(inDefaultPrecision)
		, maxPrecision_(inMaxPrecision)
		, precision_(inDefaultPrecision)
	{}

	oa::I32 format(char* outBuffer, oa::I32 inBufferSize, oa::F64 inValue);
	[[nodiscard]] oa::I32 precision() const { return precision_; }

private:
	oa::I32 defaultPrecision_ = 4;
	oa::I32 maxPrecision_     = 8;
	oa::I32 precision_        = 4;
	bool  havePrevious_     = false;
	oa::F64 previousValue_    = 0.0;
};

// ─── MetricLoss ──────────────────────────────────────────────────────────────

class MetricLoss : public Metric {
public:
	enum class Mode { Mean, Last };

	MetricLoss() = default;
	explicit MetricLoss(oa::String inName, Mode inMode = Mode::Mean)
		: name_(oa::move(inName)), mode_(inMode) {}

	void update(const Matrix& inPreds, const Matrix& inLabels) override;
	void update(oa::F32 inLossValue);
	void updateStep(oa::F32 inLoss, bool inHasLoss, oa::F64, oa::I64) override {
		if (inHasLoss) update(inLoss);
	}
	void reset() override;
	[[nodiscard]] oa::F64 result() const override;
	[[nodiscard]] const char* name() const override {
		if (!name_.empty()) return name_.cStr();
		const char* last = FnLoss::lastName();
		return last ? last : "loss";
	}
	[[nodiscard]] oa::I64 count() const { return count_; }
	[[nodiscard]] oa::F64 mean()  const { return count_ > 0 ? sum_ / static_cast<oa::F64>(count_) : 0.0; }
	[[nodiscard]] oa::F64 last()  const { return count_ > 0 ? last_ : 0.0; }
	oa::I32 render(char* outBuffer, oa::I32 inBufferSize, bool inFirst) const override;
	[[nodiscard]] bool isLossMetric() const override { return true; }

private:
	oa::String name_;
	Mode     mode_  = Mode::Mean;
	oa::F64    sum_   = 0.0;
	oa::F64    last_  = 0.0;
	oa::I64    count_ = 0;
	mutable MetricValueFormatter formatter_;
};

// ─── MetricAccuracy ──────────────────────────────────────────────────────────

class MetricAccuracy : public Metric {
public:
	MetricAccuracy() = default;

	void update(const Matrix& inPreds, const Matrix& inLabels) override;
	void reset() override;
	[[nodiscard]] oa::F64 result() const override;
	[[nodiscard]] oa::I64 count() const { return total_; }
	[[nodiscard]] const char* name() const override { return "accuracy"; }
	oa::I32 render(char* outBuffer, oa::I32 inBufferSize, bool inFirst) const override;

private:
	oa::I64 correct_ = 0;
	oa::I64 total_   = 0;
};

// ─── FnMetric ────────────────────────────────────────────────────────────────
// oa::FnMetric — stateless metric functions (Maya MFn convention).

namespace FnMetric {

	[[nodiscard]] oa::F32 scalarLoss(const Matrix& inLossTensor);
	[[nodiscard]] oa::F32 accuracy(const Matrix& inPreds, const Matrix& inLabels);
} // namespace FnMetric

} // namespace oa
