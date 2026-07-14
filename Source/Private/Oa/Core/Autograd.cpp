// OaAutogradMeta + OaMatrix autograd accessors.
//
// The grad accumulator is a heap-stored OaMatrix (in OaUniquePtr) so that
// OaAutogradMeta can be defined in Core/Matrix.h before OaMatrix is complete.
// The destructor and accessor bodies live here, where OaMatrix is complete.
//
#include <Oa/Core/Matrix.h>
#include <Oa/Core/FnMatrix.h>

#include <algorithm>

namespace {

// Deferred leaf-gradient accumulation batching.
struct DeferredAccum {
	OaMatrix* Grad;
	OaMatrix  Contribution;
};
thread_local OaVec<DeferredAccum> g_DeferredAccum;

} // anonymous

// ─── OaAutogradMeta ─────────────────────────────────────────────────────────

OaAutogradMeta::OaAutogradMeta() = default;
OaAutogradMeta::~OaAutogradMeta() = default;

// ─── OaMatrix autograd accessors ────────────────────────────────────────────

void OaMatrix::SetRequiresGrad(bool InValue) {
	if (not InValue) {
		if (Autograd_) Autograd_->RequiresGrad = false;
		return;
	}
	if (not Autograd_) Autograd_ = OaMakeSharedPtr<OaAutogradMeta>();
	Autograd_->RequiresGrad = true;
	if (not Autograd_->Grad) {
		// Tier-1: allocate once, persist across pool resets. Zeros() records a
		// Fill kernel into the active context; the buffer is owned by the
		// resulting OaMatrix's VkBuf_ (engine allocator, not the train pool).
		Autograd_->Grad = OaMakeUniquePtr<OaMatrix>(OaFnMatrix::Zeros(Shape_, Dtype_));
	}
}

bool OaMatrix::RequiresGrad() const noexcept {
	return Autograd_ and (Autograd_->RequiresGrad or Autograd_->GradFn);
}

bool OaMatrix::IsLeaf() const noexcept {
	return not Autograd_ or not Autograd_->GradFn;
}

OaSharedPtr<OaGradNode> OaMatrix::GetGradFn() const noexcept {
	return Autograd_ ? Autograd_->GradFn : nullptr;
}

OaAutogradMeta& OaMatrix::MutAutograd() {
	if (not Autograd_) Autograd_ = OaMakeSharedPtr<OaAutogradMeta>();
	return *Autograd_;
}

void OaMatrix::DetachForGradAttach(bool InRequiresGrad) {
	// Replace any shared/aliased Autograd_ (e.g. from a view that copies the
	// source's shared_ptr) with a fresh, independent meta carrying only the
	// requires-grad flag. This lets a differentiable wrapper attach its OWN
	// gradfn to a view of its input without clobbering the source's gradfn or
	// corrupting the source's leaf-ness (the view-clobber bug).
	Autograd_ = OaMakeSharedPtr<OaAutogradMeta>();
	Autograd_->RequiresGrad = InRequiresGrad;
}

OaMatrix OaMatrix::GradMatrix() const {
	if (Autograd_ and Autograd_->Grad) return *Autograd_->Grad;
	return OaMatrix{};
}

OaMatrix& OaMatrix::MutGradMatrix() {
	// Mutable lvalue ref to the single source of truth. Allocates the persistent
	// grad on first use (mirrors SetRequiresGrad's Tier-1 alloc) so callers can
	// Fill / AddInPlace / assign into it directly. The returned reference is stable
	// (Autograd_->Grad is allocated once and never reseated), so &MutGradMatrix() is
	// a valid pointer to pass to optimizer dispatches.
	if (not Autograd_ or not Autograd_->Grad) {
		SetRequiresGrad(true);
	}
	return *Autograd_->Grad;
}

void OaMatrix::AccumulateGrad(const OaMatrix& InContribution) {
	// AccumulateGrad is a leaf sink: grad += contribution.
	if (not Autograd_ or not Autograd_->Grad) {
		// Lazily allocate so accidental AccumulateGrad on a fresh leaf doesn't
		// silently drop the contribution. Mirrors PyTorch's AccumulateGrad sink.
		SetRequiresGrad(true);
	}
	// Batch leaf accumulations into a deferred list; flushed as a single
	// MultiMatrixAdd dispatch at the end of OaGradientTape::Backward.
	g_DeferredAccum.PushBack({Autograd_->Grad.get(), InContribution});
}

void OaFnMatrix::FlushDeferredAccum() {
	// One AddInPlace per leaf — MultiMatrixAdd batched path does not reliably
	// update grad buffers when N>1 (see TutorialNlpTransformerAg MatMul test).
	for (const auto& acc : g_DeferredAccum) {
		OaMatrix dst = *acc.Grad;
		OaMatrix contrib = acc.Contribution;
		// A contribution can arrive shaped as a *view* of the leaf (same element
		// count, different shape/rank) — e.g. OaGradSum expands grad to the input's
		// view shape, or a reshaped operand fans back. The leaf's persistent grad
		// buffer owns the canonical shape; row-major reshape preserves element order,
		// so normalize before accumulating. Without this AddInPlace's Broadcast()
		// fails on incompatible ranks (e.g. [N] vs [B,L,H,P]) → "bad optional access".
		if (contrib.NumElements() == dst.NumElements() and
			contrib.GetShape() != dst.GetShape()) {
			contrib = contrib.Reshape(dst.GetShape());
		}
		OaFnMatrix::AddInPlace(dst, contrib);
	}
	g_DeferredAccum.Clear();
}

void OaMatrix::ZeroGrad() {
	if (Autograd_ and Autograd_->Grad) {
		OaFnMatrix::Fill(*Autograd_->Grad, 0.0F);
	}
}
