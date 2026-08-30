// oa::Renderer — engine-borrowing render-to-texture session.
//
// One session renders either 3D mesh snapshots or an oa::Ui command stream into
// a bounded target ring. Submitted frames expose an image-backed oa::Texture and
// its exact producer event. Presentation, readback, encoding, and ML remain
// independent consumers of that value.

#pragma once

#include <oa/core/vlm.h>
#include <oa/core/status.h>
#include <oa/core/types.h>
#include <oa/render/fnCamera.h>
#include <oa/render/fnMesh.h>
#include <oa/render/scene.h>
#include <oa/runtime/texture.h>
#include <oa/runtime/event.h>

namespace oa {
class Engine;
class Ui;
class UiRenderConfig;
class Viewer;

// ─── 3D mesh renderer ────────────────────────────────────────────────────────

enum class RendererMode : oa::U8 {
	Rasterization,
	RayTracing,
};

class RendererConfig {
public:
	// Rendering algorithms are exact requests. Unsupported or unavailable
	// modes fail create() and never fall back to a different algorithm.
	RendererMode mode_ = RendererMode::Rasterization;
	oa::U32 width_ = 256U;
	oa::U32 height_ = 192U;
	oa::U32 targetSlotCount_ = 3U;
	oa::U32 maxVertexCount_ = 4096U;
	oa::U32 maxIndexCount_ = 12288U;
	// Raster sample count for 3D color/depth attachments. One disables
	// multisampling; unsupported counts fail create() instead of falling back.
	oa::U32 sampleCount_ = 1U;
	oa::vlm::Vec4 clearColor_{0.02F, 0.03F, 0.05F, 1.0F};
	oa::vlm::Vec3 lightDirection_{0.35F, 0.82F, 0.45F};
	oa::F32 ambientLight_ = 0.24F;
};

// generation-safe non-owning handle for one submitted renderer target.
// Color() remains valid until the exact frame is consumed or abandoned.
class RenderFrame {
public:

	[[nodiscard]] oa::U32 width() const noexcept { return width_; }
	[[nodiscard]] oa::U32 height() const noexcept { return height_; }
	[[nodiscard]] const oa::Texture& color() const noexcept { return color_; }
	[[nodiscard]] const oa::Event& producer() const noexcept { return producer_; }

private:
	friend class Renderer;
	friend class ::oa::Viewer;

	enum class SourceKind : oa::U8 {
		Unknown,
		Mesh,
		Ui,
	};

	// Opaque generation metadata used to reject stale or forged handles.
	oa::U32 slot_ = 0U;
	oa::U64 slotGeneration_ = 0U;
	oa::U64 targetGeneration_ = 0U;
	oa::U32 width_ = 0U;
	oa::U32 height_ = 0U;
	oa::Texture color_;
	oa::Event producer_;
	SourceKind sourceKind_ = SourceKind::Unknown;
};

class RenderReadback {
public:
	oa::U32 width_ = 0U;
	oa::U32 height_ = 0U;
	oa::Vector<oa::U8> colorRgba8_;
	oa::Vector<oa::F32> depth32_;
};

class Renderer {
public:
	Renderer(const Renderer&) = delete;
	Renderer& operator=(const Renderer&) = delete;
	Renderer(Renderer&&) = delete;
	Renderer& operator=(Renderer&&) = delete;
	~Renderer();

	[[nodiscard]] static oa::Result<oa::UniquePtr<Renderer>> create(oa::Engine& inEngine,	const RendererConfig& inConfig = {});
	[[nodiscard]] static oa::Result<oa::UniquePtr<Renderer>> create(oa::Engine& inEngine,	const ::oa::UiRenderConfig& inConfig);

	// beginFrame copies the complete semantic mesh snapshot before returning.
	// No caller-owned mesh memory is retained.
	[[nodiscard]] oa::Status beginFrame(const MeshData& inMesh,	const CameraState& inCamera);
	// Scene compilation validates stable identities, hierarchy, transforms, and
	// configured geometry capacity before recording. The semantic scene remains
	// caller-owned and no caller memory is retained.
	[[nodiscard]] oa::Status beginFrame(const Scene& inScene, const CameraState& inCamera);
	// The UI overload starts immediate command collection. ui() is non-null only
	// for a renderer created with oa::UiRenderConfig.
	[[nodiscard]] oa::Status beginFrame(oa::F32 inDeltaMs, oa::F32 inContentScale = 1.0F);
	[[nodiscard]] ::oa::Ui* ui() noexcept;
	[[nodiscard]] const ::oa::Ui* ui() const noexcept;
	[[nodiscard]] oa::Result<RenderFrame> submitFrame(oa::Span<const oa::Event> inDependencies = {});
	[[nodiscard]] oa::Status cancelFrame();

	// Normal submission keeps targets device-resident and records no host copy.
	// consumeReadback is the explicit synchronous sink. markConsumed registers
	// the exact graphics completion of an external image consumer. abandonFrame
	// never waits; Collect recycles completed retired slots.
	[[nodiscard]] oa::Result<RenderReadback> consumeReadback(const RenderFrame& inFrame);
	// Explicit synchronous file sink. The frame is consumed before encoding;
	// an encoder/filesystem failure does not resurrect its target lease.
	[[nodiscard]] oa::Status saveTo(
		const RenderFrame& inFrame,
		oa::StringView inPath,
		oa::U32 inQuality = 90U
	);
	[[nodiscard]] oa::Status markConsumed(const RenderFrame& inFrame, const oa::Event& inConsumer);
	[[nodiscard]] oa::Status abandonFrame(const RenderFrame& inFrame);
	[[nodiscard]] oa::Status collect();

	// resize is non-waiting and requires all old-generation frames to have
	// reached a reusable state.
	[[nodiscard]] oa::Status resize(oa::U32 inWidth, oa::U32 inHeight);

	// Mandatory explicit shutdown boundary. It waits only exact outstanding
	// producer/consumer events and reports failures.
	[[nodiscard]] oa::Status close();

private:
	Renderer() = default;
	class Impl;
	class MeshImpl;
	class UiImpl;
	oa::UniquePtr<Impl> impl_;
};

} // namespace oa
