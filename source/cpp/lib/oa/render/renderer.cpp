#include "rendererInternal.h"

#include <oa/core/log.h>
#include <oa/runtime/engine.h>
#include <oa/runtime/engine/borrowedServiceRetirement.h>

oa::Status oa::Renderer::Impl::completeRetirement(void* inPayload) {
	auto* impl = static_cast<Impl*>(inPayload);
	return impl == nullptr ? oa::Status::ok() : impl->completeRetired();
}

void oa::Renderer::Impl::releaseRetirement(void* inPayload) {
	oa::UniquePtr<Impl> impl(static_cast<Impl*>(inPayload));
	if (impl && !impl->closed) {
		OaLogError(
			oa::LogComponent::Render,
			"oa::Renderer retirement released without successful completion; preserving live resources");
		(void)impl.release();
	}
}

oa::Renderer::~Renderer() {
	if (!impl_ || impl_->closed) {
		impl_.reset();
		return;
	}
	oa::Engine* engine = impl_->engine;
	const bool hasSubmission = impl_->prepareNonWaitingRetirement();
	if (!hasSubmission && engine != nullptr && engine->isReady()) {
		const oa::Status cleanup = impl_->cleanupWithoutSubmission();
		if (cleanup.isOk()) {
			OaLogError(
				oa::LogComponent::Render,
				"oa::Renderer destroyed without Close; performing non-waiting cleanup");
			impl_.reset();
			return;
		}
	}
	if (engine != nullptr && engine->isReady()) {
		OaLogError(
			oa::LogComponent::Render,
			"oa::Renderer abandoned; exact frame retirement transferred to oa::Engine");
		oa::BorrowedServiceRetirement::retire(
			*engine,
			impl_.release(),
			&Impl::completeRetirement,
			&Impl::releaseRetirement);
		return;
	}
	OaLogError(
		oa::LogComponent::Render,
		"oa::Renderer outlived oa::Engine; resources cannot be retired safely");
	(void)impl_.release();
}

oa::Status oa::Renderer::beginFrame(
	const oa::MeshData& inMesh,
	const oa::CameraState& inCamera) {
	if (!impl_ || impl_->closed) {
		return oa::Status::error(
			oa::StatusCode::FailedPrecondition,
			"oa::Renderer session is closed");
	}
	return impl_->beginMeshFrame(inMesh, inCamera);
}

oa::Status oa::Renderer::beginFrame(oa::F32 inDeltaMs, oa::F32 inContentScale) {
	if (!impl_ || impl_->closed) {
		return oa::Status::error(
			oa::StatusCode::FailedPrecondition,
			"oa::Renderer session is closed");
	}
	return impl_->beginUiFrame(inDeltaMs, inContentScale);
}

oa::Ui* oa::Renderer::ui() noexcept {
	return impl_ && !impl_->closed ? impl_->ui() : nullptr;
}

const oa::Ui* oa::Renderer::ui() const noexcept {
	return impl_ && !impl_->closed ? impl_->ui() : nullptr;
}

oa::Result<oa::RenderFrame> oa::Renderer::submitFrame(
	oa::Span<const oa::Event> inDependencies) {
	if (!impl_ || impl_->closed) {
		return oa::Status::error(
			oa::StatusCode::FailedPrecondition,
			"oa::Renderer session is closed");
	}
	return impl_->submitFrame(inDependencies);
}

oa::Status oa::Renderer::cancelFrame() {
	if (!impl_ || impl_->closed) {
		return oa::Status::error(
			oa::StatusCode::FailedPrecondition,
			"oa::Renderer session is closed");
	}
	return impl_->cancelFrame();
}

oa::Result<oa::RenderReadback> oa::Renderer::consumeReadback(
	const oa::RenderFrame& inFrame) {
	if (!impl_ || impl_->closed) {
		return oa::Status::error(
			oa::StatusCode::FailedPrecondition,
			"oa::Renderer session is closed");
	}
	return impl_->consumeReadback(inFrame);
}

oa::Status oa::Renderer::markConsumed(
	const oa::RenderFrame& inFrame,
	const oa::Event& inConsumer) {
	if (!impl_ || impl_->closed) {
		return oa::Status::error(
			oa::StatusCode::FailedPrecondition,
			"oa::Renderer session is closed");
	}
	return impl_->markConsumed(inFrame, inConsumer);
}

oa::Status oa::Renderer::abandonFrame(const oa::RenderFrame& inFrame) {
	if (!impl_ || impl_->closed) {
		return oa::Status::error(
			oa::StatusCode::FailedPrecondition,
			"oa::Renderer session is closed");
	}
	return impl_->abandonFrame(inFrame);
}

oa::Status oa::Renderer::collect() {
	if (!impl_ || impl_->closed) {
		return oa::Status::error(
			oa::StatusCode::FailedPrecondition,
			"oa::Renderer session is closed");
	}
	return impl_->collect();
}

oa::Status oa::Renderer::resize(oa::U32 inWidth, oa::U32 inHeight) {
	if (!impl_ || impl_->closed) {
		return oa::Status::error(
			oa::StatusCode::FailedPrecondition,
			"oa::Renderer session is closed");
	}
	return impl_->resize(inWidth, inHeight);
}

oa::Status oa::Renderer::close() {
	return !impl_ || impl_->closed ? oa::Status::ok() : impl_->close();
}
