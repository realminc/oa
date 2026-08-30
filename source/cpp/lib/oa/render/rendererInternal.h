#pragma once

#include <oa/render/renderer.h>

class oa::Renderer::Impl {
public:
	virtual ~Impl() = default;

	oa::Engine* engine = nullptr;
	bool closed = false;

	[[nodiscard]] virtual oa::Status beginMeshFrame(
		const oa::MeshData&,
		const oa::CameraState&) {
		return oa::Status::error(
			oa::StatusCode::FailedPrecondition,
			"oa::Renderer was created for UI composition, not mesh rendering");
	}
	[[nodiscard]] virtual oa::Status beginSceneFrame(
		const oa::Scene&,
		const oa::CameraState&) {
		return oa::Status::error(
			oa::StatusCode::FailedPrecondition,
			"oa::Renderer was created for UI composition, not scene rendering");
	}
	[[nodiscard]] virtual oa::Status beginUiFrame(oa::F32, oa::F32) {
		return oa::Status::error(
			oa::StatusCode::FailedPrecondition,
			"oa::Renderer was created for mesh rendering, not UI composition");
	}
	[[nodiscard]] virtual oa::Ui* ui() noexcept { return nullptr; }
	[[nodiscard]] virtual const oa::Ui* ui() const noexcept { return nullptr; }

	[[nodiscard]] virtual oa::Result<oa::RenderFrame> submitFrame(
		oa::Span<const oa::Event> inDependencies) = 0;
	[[nodiscard]] virtual oa::Status cancelFrame() = 0;
	[[nodiscard]] virtual oa::Result<oa::RenderReadback> consumeReadback(
		const oa::RenderFrame& inFrame) = 0;
	[[nodiscard]] virtual oa::Status markConsumed(
		const oa::RenderFrame& inFrame,
		const oa::Event& inConsumer) = 0;
	[[nodiscard]] virtual oa::Status abandonFrame(
		const oa::RenderFrame& inFrame) = 0;
	[[nodiscard]] virtual oa::Status collect() = 0;
	[[nodiscard]] virtual oa::Status resize(oa::U32 inWidth, oa::U32 inHeight) = 0;
	[[nodiscard]] virtual oa::Status close() = 0;

	[[nodiscard]] virtual bool prepareNonWaitingRetirement() noexcept = 0;
	[[nodiscard]] virtual oa::Status cleanupWithoutSubmission() = 0;
	[[nodiscard]] virtual oa::Status completeRetired() = 0;

	[[nodiscard]] static oa::Status completeRetirement(void* inPayload);
	static void releaseRetirement(void* inPayload);
};
