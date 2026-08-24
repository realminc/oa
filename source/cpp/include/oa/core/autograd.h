// oa::GradNode — foundational reverse-mode recording contract.
//
// Core operations may participate in differentiation without depending on the
// ml module. ml owns the gradient tape and higher-level training behavior;
// concrete core gradient nodes remain private implementation details.
#pragma once

#include <oa/core/matrix.h>
#include <oa/core/status.h>

#include <initializer_list>
#include <utility>

namespace oa {

class GradientTape;

class GradNode {
public:
	virtual ~GradNode() = default;

	// Records this operation's backward kernels into the active graph. The tape
	// owns traversal and execution cadence; nodes only describe their adjoints.
	virtual void backward(
		const Matrix& inUpstream,
		oa::Vec<Matrix>& outInputGrads
	) = 0;

	[[nodiscard]] oa::Span<const Matrix> graphInputs() const {
		return inputs_.span();
	}
	[[nodiscard]] oa::Vec<Matrix>& mutGraphInputs() { return inputs_; }
	void setGraphInputs(oa::Vec<Matrix> inInputs) {
		inputs_ = std::move(inInputs);
	}

	/// Retains matrices required by backward and snapshots their storage
	/// mutation versions. Backward fails before recording any kernels if a
	/// retained matrix was modified in place.
	void saveForBackward(oa::Vec<Matrix> inMatrices);
	void saveForBackward(std::initializer_list<Matrix> inMatrices);

	void clearTensors() {
		savedMatrices_.clear();
		inputs_.clear();
	}

	oa::U64     sequenceNr_            = 0;
	oa::U32     forwardSemanticOp_     = UINT32_MAX;
	oa::U32     forwardSemanticOutput_ = 0;
	// Operation ids are local to one semantic recording. The generation keeps
	// a retained tape node from binding a reused id after submission or clear().
	oa::U64     forwardSemanticGeneration_ = 0;
	MatrixShape outputShape_{};

protected:
	[[nodiscard]] Matrix& saved(oa::Usize inIndex) {
		return savedMatrices_[inIndex].matrix;
	}
	[[nodiscard]] const Matrix& saved(oa::Usize inIndex) const {
		return savedMatrices_[inIndex].matrix;
	}

	oa::Vec<Matrix> inputs_;

private:
	friend class GradientTape;

	struct SavedMatrix {
		Matrix matrix;
		oa::U64 mutationVersion = 0U;
	};

	[[nodiscard]] oa::Status validateSavedVersions() const;

	oa::Vec<SavedMatrix> savedMatrices_;
};

// oa::FnAutograd — autograd tape control.
namespace FnAutograd {

	[[nodiscard]] bool isEnabled() noexcept;
	void setEnabled(bool inEnabled) noexcept;
	[[nodiscard]] oa::U64 nextSeq() noexcept;
	[[nodiscard]] oa::Status attachSemantic(
		const oa::SharedPtr<GradNode>& inNode,
		oa::U32 inForwardOp,
		oa::U32 inOutputIndex = 0
	);
	[[nodiscard]] oa::Status completeSemantic(
		const GradNode& inNode,
		oa::U32 inBackwardFirstOp,
		oa::U32 inBackwardOpCount
	);
} // namespace FnAutograd

} // namespace oa
