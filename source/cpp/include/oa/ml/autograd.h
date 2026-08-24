// oa::GradNo / oa::GradientTape — Public ML reverse-mode automatic-differentiation contract.
//
// The foundational node and recording contract lives in oa/core/autograd.h so
// core operations do not depend on ml. ml adds tape lifetime and traversal.
#pragma once

#include <oa/core/autograd.h>

namespace oa {

// oa::GradNo — RAII scope guard that disables autograd for its lifetime.
class GradNo {
	bool prev_;

public:
	GradNo() noexcept : prev_(FnAutograd::isEnabled()) {
		FnAutograd::setEnabled(false);
	}
	~GradNo() noexcept { FnAutograd::setEnabled(prev_); }
	GradNo(const GradNo&) = delete;
	GradNo& operator=(const GradNo&) = delete;
};

// oa::GradientTape — RAII scope guard that enables autograd for its lifetime.
// call backward() to execute the recorded backward graph.
class GradientTape {
	bool prev_;
	bool active_ = true;

public:
	GradientTape() noexcept : prev_(FnAutograd::isEnabled()) {
		FnAutograd::setEnabled(true);
	}
	~GradientTape() noexcept { close(); }
	GradientTape(const GradientTape&) = delete;
	GradientTape& operator=(const GradientTape&) = delete;

	void close() noexcept {
		if (active_) {
			FnAutograd::setEnabled(prev_);
			active_ = false;
		}
	}

	// Records backward operations into the active graph. Submission and waiting
	// remain the caller/training-session responsibility.
	[[nodiscard]] oa::Status tryBackward(const Matrix& inRoot);
	void backward(const Matrix& inRoot);
};

} // namespace oa
