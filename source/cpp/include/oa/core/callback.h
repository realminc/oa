// Callback — base callback interface for iterators
//
// keras-style hooks for Iterator subclasses (ItTraining, ItBatch, etc.).
// Subclass and attach via addCallback. All hooks have default no-op
// implementations so you only override what you need.

#pragma once

#include <oa/core/types.h>

namespace oa {

class Iterator;

// base callback interface — all methods are optional (no-op by default)
class Callback {
public:
	virtual ~Callback() = default;

	// Fired once before the first iteration.
	virtual void onBegin(Iterator& inIter) { (void)inIter; }

	// Fired after each iteration step. The iterator has just advanced.
	virtual void onStep(Iterator& inIter) { (void)inIter; }

	// Fired once after the last iteration or explicit finish().
	virtual void onEnd(Iterator& inIter) { (void)inIter; }

	Callback(const Callback&)            = delete;
	Callback& operator=(const Callback&) = delete;
	Callback(Callback&&) noexcept        = default;
	Callback& operator=(Callback&&) noexcept = default;

protected:
	Callback() = default;
};

// Legacy aliases — remove once call sites are migrated.

} // namespace oa
