// Viewport — Passive camera/target/view description for rendering.
//
// architecture naming resolution:
//   - Viewport (Ui/): passive camera + target + viewport description (data)
//   - Viewer (Ui/): windowed interactive application (controller)
//
// This is NOT a window or interactive element. It describes what to render
// and where, without owning swapchains or handling input.
//
// usage (2D image viewing):
//   oa::Camera camera(imageWidth, imageHeight);
//   camera.fitToWindow(windowWidth, windowHeight);
//
//   Viewport view;
//   view.setCamera(&camera);
//   view.setTarget(&renderTarget);
//   OA_CHECK_OK(view.setViewport({0, 0, windowWidth, windowHeight}));
//
// usage (3D render target):
//   oa::Camera camera({0, 2, 5}, {0, 0, 0});  // perspective, positioned
//
//   Viewport view;
//   view.setCamera(&camera);
//   view.setTarget(&renderTarget);
//
#pragma once

#include <oa/core/status.h>
#include <oa/core/types.h>
#include <oa/render/camera.h>
#include <oa/runtime/texture.h>

#include <cmath>

// forward declarations.
namespace oa { struct Swapchain; }    // runtime/swapchain.h

// ViewportMode — what kind of content is being viewed.
namespace oa {

enum class ViewportMode : oa::U8 {
	Image2D,      // 2D image/video with orthographic camera
	Scene3D,      // 3D scene with perspective camera
	Matrix,       // Matrix-as-heatmap visualization
	Video,        // Video with timeline (2D + time control)
	CameraStream, // Live camera feed
};

// ViewportDesc — viewport rectangle and depth range.

struct ViewportDesc {
	oa::I32 x = 0;
	oa::I32 y = 0;
	oa::I32 width = 1280;
	oa::I32 height = 720;
	oa::F32 minDepth = 0.0f;
	oa::F32 maxDepth = 1.0f;

	[[nodiscard]] oa::Result<oa::F32> getAspectRatio() const {
		if (not isValid()) {
			return oa::Status::invalidArgument(
				"ViewportDesc aspect ratio requires a valid viewport");
		}
		return static_cast<oa::F32>(width) / static_cast<oa::F32>(height);
	}

	[[nodiscard]] bool isValid() const noexcept {
		return width > 0 and height > 0
			and std::isfinite(minDepth) and std::isfinite(maxDepth)
			and minDepth >= 0.0F and minDepth <= maxDepth
			and maxDepth <= 1.0F;
	}
};

// ScissorDesc — Scissor rectangle for clipping.
struct ScissorDesc {
	oa::I32 x = 0;
	oa::I32 y = 0;
	oa::I32 width = 1280;
	oa::I32 height = 720;

	[[nodiscard]] constexpr bool isValid() const noexcept {
		return width > 0 and height > 0;
	}

	[[nodiscard]] bool contains(oa::I32 inX, oa::I32 inY) const noexcept {
		return isValid()
			and static_cast<oa::I64>(inX) >= static_cast<oa::I64>(x)
			and static_cast<oa::I64>(inX) < static_cast<oa::I64>(x) + width
			and static_cast<oa::I64>(inY) >= static_cast<oa::I64>(y)
			and static_cast<oa::I64>(inY) < static_cast<oa::I64>(y) + height;
	}
};

// ═════════════════════════════════════════════════════════════════════════════
// RenderLayer — layer/filter flags for selective rendering
// ═════════════════════════════════════════════════════════════════════════════

enum class RenderLayer : oa::U32 {
	None = 0,
	Default = 1 << 0,
	Ui = 1 << 1,
	Overlay = 1 << 2,
	Background = 1 << 3,
	Debug = 1 << 4,
	All = 0xFFFFFFFF,
};

inline RenderLayer operator|(RenderLayer inA, RenderLayer inB) noexcept {
	return static_cast<RenderLayer>(static_cast<oa::U32>(inA) | static_cast<oa::U32>(inB));
}

inline RenderLayer operator&(RenderLayer inA, RenderLayer inB) noexcept {
	return static_cast<RenderLayer>(static_cast<oa::U32>(inA) & static_cast<oa::U32>(inB));
}

// Viewport — Passive view description for rendering
//
// Declares: camera, target, viewport, and view-specific settings.
// Does NOT: handle input, own windows, manage swapchains, or animate.

class Viewport {
public:
	Viewport() = default;
	explicit Viewport(ViewportMode inMode) : mode_(inMode) {}


	// Mode and configuration.
	[[nodiscard]] oa::Status setMode(ViewportMode inMode);
	[[nodiscard]] ViewportMode getMode() const noexcept { return mode_; }

	// camera (required).
	void setCamera(const oa::Camera* inCamera) noexcept { camera_ = inCamera; }
	[[nodiscard]] const oa::Camera* getCamera() const noexcept { return camera_; }

	// ═══════════════════════════════════════════════════════════════════════
	// Render target (required)
	// ═══════════════════════════════════════════════════════════════════════

	void setTarget(const oa::Texture* inTarget) noexcept { target_ = inTarget; }
	[[nodiscard]] const oa::Texture* getTarget() const noexcept { return target_; }

	// ═══════════════════════════════════════════════════════════════════════
	// viewport and scissor
	// ═══════════════════════════════════════════════════════════════════════

	[[nodiscard]] oa::Status setViewport(const ViewportDesc& inViewport);
	[[nodiscard]] const ViewportDesc& getViewport() const noexcept { return viewport_; }

	[[nodiscard]] oa::Status setScissor(const ScissorDesc& inScissor);
	void disableScissor() noexcept { useScissor_ = false; }
	[[nodiscard]] bool isScissorEnabled() const noexcept { return useScissor_; }
	[[nodiscard]] const ScissorDesc& getScissor() const noexcept { return scissor_; }

	// ═══════════════════════════════════════════════════════════════════════
	// layer filtering
	// ═══════════════════════════════════════════════════════════════════════

	void setLayerMask(RenderLayer inMask) noexcept { layerMask_ = inMask; }
	[[nodiscard]] RenderLayer getLayerMask() const noexcept { return layerMask_; }

	// Check if a layer is visible
	[[nodiscard]] bool isLayerVisible(RenderLayer inLayer) const noexcept {
		return (static_cast<oa::U32>(layerMask_) & static_cast<oa::U32>(inLayer)) != 0;
	}

	// ═══════════════════════════════════════════════════════════════════════
	// clear values
	// ═══════════════════════════════════════════════════════════════════════

	[[nodiscard]] oa::Status setClearColor(const oa::vlm::Vec4& inColor);
	[[nodiscard]] const oa::vlm::Vec4& getClearColor() const noexcept { return clearColor_; }

	[[nodiscard]] oa::Status setClearDepth(oa::F32 inDepth);
	[[nodiscard]] oa::F32 getClearDepth() const noexcept { return clearDepth_; }

	// Validation.

	[[nodiscard]] bool isValid() const noexcept {
		return static_cast<oa::U8>(mode_)
				<= static_cast<oa::U8>(ViewportMode::CameraStream)
			and camera_ != nullptr and target_ != nullptr and target_->isValid()
			and viewport_.isValid()
			and (not useScissor_ or scissor_.isValid())
			and std::isfinite(clearColor_.x) and std::isfinite(clearColor_.y)
			and std::isfinite(clearColor_.z) and std::isfinite(clearColor_.w)
			and std::isfinite(clearDepth_)
			and clearDepth_ >= 0.0F and clearDepth_ <= 1.0F;
	}

	// Convenience helpers.

	// Setup for 2D image viewing (orthographic camera)
	[[nodiscard]] oa::Status setupImage2D(
		oa::F32 inImageWidth,
		oa::F32 inImageHeight,
		const oa::Texture* inTarget,
		const ViewportDesc& inViewport
	);

	// Setup for aspect-fit image display
	[[nodiscard]] oa::Status setupImageAspectFit(
		oa::F32 inImageWidth,
		oa::F32 inImageHeight,
		oa::F32 inWindowWidth,
		oa::F32 inWindowHeight,
		const oa::Texture* inTarget
	);

private:
	ViewportMode mode_ = ViewportMode::Image2D;
	const oa::Camera* camera_ = nullptr;
	const oa::Texture* target_ = nullptr;
	ViewportDesc viewport_;
	ScissorDesc scissor_;
	bool useScissor_ = false;
	RenderLayer layerMask_ = RenderLayer::Default;
	oa::vlm::Vec4 clearColor_ = {0.01f, 0.01f, 0.01f, 1.0f};
	oa::F32 clearDepth_ = 1.0f;

	// Internal camera storage for convenience setups
	oa::Camera internalCamera_;
};

}  // namespace oa
