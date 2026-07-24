#pragma once

#include <Oa/Core/Types.h>
#include <Oa/Core/Matrix.h>
#include <Oa/Ml/FnLoss.h>

#include <cmath>
#include <vector>

// ─── OaMetric ─────────────────────────────────────────────────────────────
//
// Base class for all metrics. Follows the TF/Keras pattern:
//   - Update(): add a new batch of predictions/labels
//   - Reset(): clear accumulated state
//   - Result(): get the current metric value
//
// Two usage styles (C++ class vs C-style functional):
//
//   // C++ class style (stateful object)
//   OaMetricAccuracy acc;
//   acc.Update(preds, labels);
//   acc.Update(preds2, labels2);
//   printf("Accuracy: %.2f%%\n", acc.Result() * 100.0);
//
//   // C-style functional (stateless, manual accumulation)
//   OaF32 acc = OaFnMetric::Accuracy(preds, labels);
//   OaF32 acc2 = OaFnMetric::Accuracy(preds2, labels2);
//   printf("Accuracy: %.2f%%\n", (acc + acc2) / 2.0 * 100.0);

class OaMetric {
public:
	virtual ~OaMetric() = default;

	// Update the metric with new data. Exact signature varies by metric type.
	virtual void Update(const OaMatrix& InPreds, const OaMatrix& InLabels) = 0;

	// Consume one completed training step. This keeps iterator integration
	// type-safe: callbacks do not inspect Name() strings or downcast metrics.
	virtual void UpdateStep(OaF32 InLoss, bool InHasLoss,
		OaF64 InElapsedSeconds, OaI64 InStepCount) {
		(void)InLoss; (void)InHasLoss; (void)InElapsedSeconds; (void)InStepCount;
	}

	// Reset accumulated state (e.g., at epoch boundaries).
	virtual void Reset() = 0;

	// Get the current metric value.
	[[nodiscard]] virtual OaF64 Result() const = 0;

	// Get the metric name for logging.
	[[nodiscard]] virtual const char* Name() const = 0;

	// Render the metric to a buffer for progress bar display.
	// Returns the number of characters written (excluding null terminator).
	// InFirst: if true, don't prepend " · " separator.
	virtual OaI32 Render(char* OutBuffer, OaI32 InBufferSize, bool InFirst) const = 0;

	// True if this metric tracks a loss value.
	[[nodiscard]] virtual bool IsLossMetric() const { return false; }
};

// Compact fixed-point formatting for live metrics. Starts at four decimal
// places, preserves useful digits for small values (for example 0.0000495),
// and increases precision only when the rounded display stalls while the
// underlying value still changes. Formatting cost is negligible beside I/O.
class OaMetricValueFormatter {
public:
	OaMetricValueFormatter(OaI32 InDefaultPrecision = 4, OaI32 InMaxPrecision = 8)
		: DefaultPrecision_(InDefaultPrecision)
		, MaxPrecision_(InMaxPrecision)
		, Precision_(InDefaultPrecision)
	{}

	OaI32 Format(char* OutBuffer, OaI32 InBufferSize, OaF64 InValue);
	[[nodiscard]] OaI32 Precision() const { return Precision_; }

private:
	OaI32 DefaultPrecision_ = 4;
	OaI32 MaxPrecision_ = 8;
	OaI32 Precision_ = 4;
	bool HavePrevious_ = false;
	OaF64 PreviousValue_ = 0.0;
};

// ─── OaMetricLoss ──────────────────────────────────────────────────────────
//
// Tracks scalar loss values. Can be used as a running average or per-batch loss.
// Example:
//   OaMetricLoss loss;
//   loss.Update(lossTensor);  // scalar loss matrix
//   printf("Mean loss: %.4f\n", loss.Result());

class OaMetricLoss : public OaMetric {
public:
	enum class Mode { Mean, Last };

	OaMetricLoss() = default;
	explicit OaMetricLoss(OaString InName, Mode InMode = Mode::Mean)
		: Name_(std::move(InName))
		, Mode_(InMode)
	{}

	void Update(const OaMatrix& InPreds, const OaMatrix& InLabels) override;
	void Update(OaF32 InLossValue);  // Convenience overload for scalar values
	void UpdateStep(OaF32 InLoss, bool InHasLoss, OaF64, OaI64) override {
		if (InHasLoss) Update(InLoss);
	}
	void Reset() override;
	[[nodiscard]] OaF64 Result() const override;
	[[nodiscard]] const char* Name() const override {
		if (!Name_.empty()) { return Name_.c_str(); }
		const char* last = OaFnLoss::LastName();
		return last ? last : "loss";
	}
	[[nodiscard]] OaI64 Count() const { return Count_; }
	[[nodiscard]] OaF64 Mean() const { return Count_ > 0 ? Sum_ / static_cast<OaF64>(Count_) : 0.0; }
	[[nodiscard]] OaF64 Last() const { return Count_ > 0 ? Last_ : 0.0; }
	OaI32 Render(char* OutBuffer, OaI32 InBufferSize, bool InFirst) const override;
	[[nodiscard]] bool IsLossMetric() const override { return true; }

private:
	OaString Name_;
	Mode Mode_ = Mode::Mean;
	OaF64 Sum_ = 0.0;
	OaF64 Last_ = 0.0;
	OaI64 Count_ = 0;
	mutable OaMetricValueFormatter Formatter_;
};

// ─── OaMetricAccuracy ───────────────────────────────────────────────────────
//
// Classification accuracy: (correct predictions) / (total predictions).
// Expects:
//   - InPreds: shape [batch, num_classes] or [batch] (already argmax'd)
//   - InLabels: shape [batch] with class indices
// Example:
//   OaMetricAccuracy acc;
//   acc.Update(logits, labels);
//   printf("Accuracy: %.2f%%\n", acc.Result() * 100.0);

class OaMetricAccuracy : public OaMetric {
public:
	OaMetricAccuracy() = default;

	void Update(const OaMatrix& InPreds, const OaMatrix& InLabels) override;
	void Reset() override;
	[[nodiscard]] OaF64 Result() const override;
	[[nodiscard]] OaI64 Count() const { return Total_; }
	[[nodiscard]] const char* Name() const override { return "accuracy"; }
	OaI32 Render(char* OutBuffer, OaI32 InBufferSize, bool InFirst) const override;

private:
	OaI64 Correct_ = 0;
	OaI64 Total_ = 0;
};

// ─── OaFnMetric ───────────────────────────────────────────────────────────
//
// C-style functional API for metrics. Stateless functions that compute metrics
// on-the-fly. Useful for:
//   - One-off metric calculations
//   - Custom accumulation logic
//   - Interop with C codebases
//
// Usage:
//   OaF32 acc = OaFnMetric::Accuracy(preds, labels);
//   OaF32 loss = OaFnMetric::ScalarLoss(lossTensor);

namespace OaFnMetric {

// Loss metrics
[[nodiscard]] OaF32 ScalarLoss(const OaMatrix& InLossTensor);

// Classification metrics
[[nodiscard]] OaF32 Accuracy(const OaMatrix& InPreds, const OaMatrix& InLabels);

} // namespace OaFnMetric
