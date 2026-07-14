// OaCallback — base callback interface for iterators
//
// Keras-style hooks for OaIterator subclasses (OaItTraining, OaItBatch, etc.).
// Subclass and attach via AddCallback. All hooks have default no-op
// implementations so you only override what you need.
//
// Design follows TensorFlow/Keras callback pattern:
//   - OnTrainBegin/End: lifecycle hooks
//   - OnEpochBegin/End: epoch boundaries (when StepsPerEpoch > 0)
//   - OnStepEnd: after each iteration step
//
// Callbacks are non-owning — caller controls lifetime (typically stack-allocated
// above the iterator). Registration order = invocation order.

#pragma once

#include <Oa/Core/Types.h>

class OaIterator;

// Base callback interface — all methods are optional (no-op by default)
class OaCallback {
public:
	virtual ~OaCallback() = default;

	// Fired once before the first iteration. Use for resource allocation,
	// header prints, opening files, etc.
	virtual void OnBegin(OaIterator& InIter) { (void)InIter; }

	// Fired after each iteration step. The iterator has just advanced.
	virtual void OnStep(OaIterator& InIter) { (void)InIter; }

	// Fired once after the last iteration or explicit Finish(). Use for
	// summary prints, file flush, cleanup, etc.
	virtual void OnEnd(OaIterator& InIter) { (void)InIter; }

	OaCallback(const OaCallback&)            = delete;
	OaCallback& operator=(const OaCallback&) = delete;
	OaCallback(OaCallback&&) noexcept        = default;
	OaCallback& operator=(OaCallback&&) noexcept = default;

protected:
	OaCallback() = default;
};
