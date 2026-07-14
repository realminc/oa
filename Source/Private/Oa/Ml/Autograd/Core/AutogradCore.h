// OaAutogradCore — Base autograd infrastructure.
//
// Public surface:
//   - OaGradNode        : abstract tape node; one concrete subclass per differentiable op.
//   - OaFnAutograd::*       : namespace-level thread-local tape state (IsEnabled, NextSeq, ...).
//   - OaGradNo            : RAII guard that disables tape attachment (PyTorch's torch.no_grad).
//   - OaGradientTape      : RAII tape scope + Backward(loss) walk (TF's GradientTape).

#pragma once

#include <Oa/Core/Matrix.h>
#include <Oa/Core/Types.h>

// ─── OaGradNode — abstract tape node ──────────────────────────────────────

class OaGradNode {
public:
	virtual ~OaGradNode() = default;

	// Records the *Bwd kernels for this op into OaContext::GetDefault().
	//   InUpstream     — dL/dOutput (one tensor; multi-output ops are not v1).
	//   OutInputGrads  — pre-sized vector of dL/dInput[j] slots. Bodies leave
	//                    slots empty when the corresponding input does not
	//                    require grad.
	virtual void Backward(const OaMatrix& InUpstream, OaVec<OaMatrix>& OutInputGrads) = 0;

	[[nodiscard]] OaSpan<const OaMatrix> GraphInputs() const { return Inputs_.Span(); }
	[[nodiscard]] OaVec<OaMatrix>&       MutGraphInputs()    { return Inputs_; }
	void SetGraphInputs(OaVec<OaMatrix> InInputs)            { Inputs_ = std::move(InInputs); }

	// Clear saved tensors and inputs to release GPU resources
	void ClearTensors() {
		Saved_.Clear();
		Inputs_.Clear();
	}

	OaVec<OaMatrix> Saved_;            // ≈ PyTorch save_for_backward
	OaU64           SequenceNr_ = 0;   // monotonic creation order; reverse-walk key
	OaMatrixShape         OutputShape_{};    // shape of the tensor this GradFn was attached to (for robust dout normalization across views/reshapes)

protected:
	OaVec<OaMatrix> Inputs_;           // owning refs keep upstream nodes + buffers alive
};

// ─── OaAutograd — thread-local tape state ───────────────────────────────────

namespace OaFnAutograd {

[[nodiscard]] bool  IsEnabled() noexcept;
void                SetEnabled(bool InEnabled) noexcept;
[[nodiscard]] OaU64 NextSeq() noexcept;

} // namespace OaFnAutograd

// ─── OaGradNo — RAII: disable tape attach (≈ torch.no_grad) ─────────────────

class OaGradNo {
	bool Prev_;
public:
	OaGradNo() noexcept : Prev_(OaFnAutograd::IsEnabled()) { OaFnAutograd::SetEnabled(false); }
	~OaGradNo() noexcept { OaFnAutograd::SetEnabled(Prev_); }
	OaGradNo(const OaGradNo&) = delete;
	OaGradNo& operator=(const OaGradNo&) = delete;
};

// ─── OaGradientTape — RAII: enable tape attach + Backward(loss) ─────────────

class OaGradientTape {
	bool Prev_;
public:
	OaGradientTape() noexcept : Prev_(OaFnAutograd::IsEnabled()) { OaFnAutograd::SetEnabled(true); }
	~OaGradientTape() noexcept { OaFnAutograd::SetEnabled(Prev_); }
	OaGradientTape(const OaGradientTape&) = delete;
	OaGradientTape& operator=(const OaGradientTape&) = delete;

	// Reverse-mode walk from a scalar root. Records *Bwd + AccumulateGrad into
	// the active OaContext. Does NOT call Execute/Sync — OaItTraining::Next()
	// owns the cadence.
	void Backward(const OaMatrix& InRoot);
};
