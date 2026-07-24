// OaAutograd — Public reverse-mode automatic-differentiation contract.
//
// Concrete OaGrad* operation nodes are private implementation details. Public
// code records differentiation through OaMatrix, OaFn*, OaGradNo, and
// OaGradientTape without depending on the generated node catalog.
#pragma once

#include <Oa/Core/Matrix.h>
#include <Oa/Core/Types.h>
#include <Oa/Runtime/SemanticGraphFwd.h>

class OaGradNode {
public:
	virtual ~OaGradNode() = default;

	// Records this operation's backward kernels into the active graph. The tape
	// owns traversal and execution cadence; nodes only describe their adjoints.
	virtual void Backward(
		const OaMatrix& InUpstream,
		OaVec<OaMatrix>& OutInputGrads) = 0;

	[[nodiscard]] OaSpan<const OaMatrix> GraphInputs() const {
		return Inputs_.Span();
	}
	[[nodiscard]] OaVec<OaMatrix>& MutGraphInputs() { return Inputs_; }
	void SetGraphInputs(OaVec<OaMatrix> InInputs) {
		Inputs_ = std::move(InInputs);
	}

	void ClearTensors() {
		Saved_.Clear();
		Inputs_.Clear();
	}

	OaVec<OaMatrix> Saved_;
	OaU64 SequenceNr_ = 0;
	OaSemanticOperationId ForwardSemanticOperation_ =
		OaInvalidSemanticOperationId;
	OaU32 ForwardSemanticOutput_ = 0;
	OaMatrixShape OutputShape_{};

protected:
	OaVec<OaMatrix> Inputs_;
};

namespace OaFnAutograd {

[[nodiscard]] bool IsEnabled() noexcept;
void SetEnabled(bool InEnabled) noexcept;
[[nodiscard]] OaU64 NextSeq() noexcept;
[[nodiscard]] OaStatus AttachSemantic(
	const OaSharedPtr<OaGradNode>& InNode,
	OaSemanticOperationId InForwardOperation,
	OaU32 InOutputIndex = 0);
[[nodiscard]] OaStatus CompleteSemantic(
	const OaGradNode& InNode,
	OaSemanticOperationId InBackwardFirstOperation,
	OaU32 InBackwardOperationCount);

} // namespace OaFnAutograd

class OaGradNo {
	bool Prev_;

public:
	OaGradNo() noexcept : Prev_(OaFnAutograd::IsEnabled()) {
		OaFnAutograd::SetEnabled(false);
	}
	~OaGradNo() noexcept { OaFnAutograd::SetEnabled(Prev_); }
	OaGradNo(const OaGradNo&) = delete;
	OaGradNo& operator=(const OaGradNo&) = delete;
};

class OaGradientTape {
	bool Prev_;
	bool Active_ = true;

public:
	OaGradientTape() noexcept : Prev_(OaFnAutograd::IsEnabled()) {
		OaFnAutograd::SetEnabled(true);
	}
	~OaGradientTape() noexcept { Close(); }
	OaGradientTape(const OaGradientTape&) = delete;
	OaGradientTape& operator=(const OaGradientTape&) = delete;

	void Close() noexcept {
		if (Active_) {
			OaFnAutograd::SetEnabled(Prev_);
			Active_ = false;
		}
	}

	// Records backward operations into the active graph. Submission and waiting
	// remain the caller/training-session responsibility.
	void Backward(const OaMatrix& InRoot);
};
