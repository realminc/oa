// AutogradMeta + oa::Matrix autograd accessors.
//
// The grad accumulator is a heap-stored oa::Matrix (in oa::UniquePtr) so that
// AutogradMeta can be defined in matrix.h before oa::Matrix is complete.
// The destructor and accessor bodies live here, where oa::Matrix is complete.
//
#include <oa/core/autograd.h>
#include <oa/core/matrix.h>
#include <oa/core/fnMatrix.h>
#include <oa/core/fnmatrix/fnMatrixInternal.h>

#include <oa/core/std/utility.h>

namespace {

// Deferred leaf-gradient accumulation batching.
struct DeferredAccum {
	oa::Matrix  grad;
	oa::Matrix  contribution;
};
thread_local oa::Vec<DeferredAccum> g_DeferredAccum;

} // anonymous

// ─── AutogradMeta ─────────────────────────────────────────────────────────

oa::AutogradMeta::AutogradMeta() = default;
oa::AutogradMeta::~AutogradMeta() = default;

// ─── oa::GradNode saved values ───────────────────────────────────────────────

void oa::GradNode::saveForBackward(oa::Vec<oa::Matrix> inMatrices) {
	savedMatrices_.clear();
	savedMatrices_.reserve(inMatrices.size());
	for (auto& matrix : inMatrices) {
		const oa::U64 mutationVersion = matrix.observeStorageMutationVersion();
		savedMatrices_.pushBack({oa::move(matrix), mutationVersion});
	}
}

oa::Status oa::GradNode::validateSavedVersions() const {
	for (const auto& savedMatrix : savedMatrices_) {
		if (savedMatrix.matrix.currentStorageMutationVersion()
			== savedMatrix.mutationVersion) {
			continue;
		}
		return oa::Status::error(oa::StatusCode::FailedPrecondition,
			"a value saved for backward was modified in place");
	}
	return oa::Status::ok();
}

// ─── oa::Matrix autograd accessors ────────────────────────────────────────────

void oa::Matrix::setRequiresGrad(bool inValue) {
	if (not inValue) {
		if (autograd_) autograd_->requiresGrad_ = false;
		return;
	}
	if (not autograd_) autograd_ = oa::makeShared<oa::AutogradMeta>();
	autograd_->requiresGrad_ = true;
	if (not autograd_->grad) {
		// Tier-1: allocate once, persist across pool resets. zeros() records a
		// Fill kernel into the active context; the buffer is owned by the
		// resulting oa::Matrix's vkBuf_ (engine allocator, not the train pool).
		autograd_->grad = oa::makeUnique<oa::Matrix>(oa::FnMatrix::zeros(shape_, dtype_));
	}
}

bool oa::Matrix::requiresGrad() const noexcept {
	return autograd_ and (autograd_->requiresGrad_ or autograd_->gradFn);
}

bool oa::Matrix::isLeaf() const noexcept {
	return not autograd_ or not autograd_->gradFn;
}

oa::SharedPtr<oa::GradNode> oa::Matrix::getGradFn() const noexcept {
	return autograd_ ? autograd_->gradFn : nullptr;
}

oa::AutogradMeta& oa::Matrix::mutAutograd() {
	if (not autograd_) autograd_ = oa::makeShared<oa::AutogradMeta>();
	return *autograd_;
}

void oa::Matrix::detachForGradAttach(bool inRequiresGrad) {
	// Replace any shared/aliased autograd_ (e.g. from a view that copies the
	// source's shared_ptr) with a fresh, independent meta carrying only the
	// requires-grad flag. This lets a differentiable wrapper attach its OWN
	// gradfn to a view of its input without clobbering the source's gradfn or
	// corrupting the source's leaf-ness (the view-clobber bug).
	autograd_ = oa::makeShared<oa::AutogradMeta>();
	autograd_->requiresGrad_ = inRequiresGrad;
}

oa::Matrix oa::Matrix::gradMatrix() const {
	if (autograd_ and autograd_->grad) return *autograd_->grad;
	return oa::Matrix{};
}

oa::Matrix& oa::Matrix::mutGradMatrix() {
	// Mutable lvalue ref to the single source of truth. Allocates the persistent
	// grad on first use (mirrors setRequiresGrad's Tier-1 alloc) so callers can
	// Fill / AddInPlace / assign into it directly. The returned reference is stable
	// (autograd_->grad is allocated once and never reseated), so &mutGradMatrix() is
	// a valid pointer to pass to optimizer dispatches.
	if (not autograd_ or not autograd_->grad) {
		setRequiresGrad(true);
	}
	return *autograd_->grad;
}

void oa::Matrix::accumulateGrad(const oa::Matrix& inContribution) {
	// accumulateGrad is a leaf sink: grad += contribution.
	if (not autograd_ or not autograd_->grad) {
		// Lazily allocate so accidental accumulateGrad on a fresh leaf doesn't
		// silently drop the contribution. Mirrors PyTorch's accumulateGrad sink.
		setRequiresGrad(true);
	}
	// Keep owning matrix handles in the deferred list. A gradient sink may be
	// reached through an input copy whose autograd metadata is released when
	// its node clears saved tensors, before the tape performs the final flush.
	g_DeferredAccum.pushBack({*autograd_->grad, inContribution});
}

void oa::FnMatrix::flushDeferredAccum() {
	// One AddInPlace per leaf — MultiMatrixAdd batched path does not reliably
	// update grad buffers when N>1 (see TutorialNlpTransformerAg MatMul test).
	for (const auto& acc : g_DeferredAccum) {
		oa::Matrix dst = acc.grad;
		oa::Matrix contrib = acc.contribution;
		// A contribution can arrive shaped as a *view* of the leaf (same element
		// count, different shape/rank) — e.g. oa::GradSum expands grad to the input's
		// view shape, or a reshaped operand fans back. The leaf's persistent grad
		// buffer owns the canonical shape; row-major reshape preserves element order,
		// so normalize before accumulating. Without this AddInPlace's broadcast()
		// fails on incompatible ranks (e.g. [N] vs [B,L,H,P]) → "bad optional access".
		if (contrib.numElements() == dst.numElements() and
			contrib.getShape() != dst.getShape()) {
			contrib = contrib.reshape(dst.getShape());
		}
		oa::FnMatrix::addInPlace(dst, contrib);
	}
	g_DeferredAccum.clear();
}

void oa::Matrix::zeroGrad() {
	if (autograd_ and autograd_->grad) {
		oa::FnMatrix::fillInPlace(*autograd_->grad, 0.0F);
	}
}
