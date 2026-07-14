// OaViewport — Passive camera/target/view description for rendering.
//
// Per CoreAnimationRenderArchitecture.md §3.5 naming resolution:
//   - OaViewport (Render/): passive camera + target + viewport description (data)
//   - OaViewer (Ui/): windowed interactive application (controller)
//
// This is NOT a window or interactive element. It describes what to render
// and where, without owning swapchains or handling input.
//
// Usage (2D image viewing):
//   OaCamera camera(imageWidth, imageHeight);
//   camera.FitToWindow(windowWidth, windowHeight);
//
//   OaViewport view;
//   view.SetCamera(&camera);
//   view.SetTarget(&renderTarget);
//   view.SetViewport({0, 0, windowWidth, windowHeight});
//
// Usage (3D scene):
//   OaCamera camera({0, 2, 5}, {0, 0, 0});  // perspective, positioned
//   OaScene scene = ...;
//
//   OaViewport view;
//   view.SetCamera(&camera);
//   view.SetScene(&scene);
//   view.SetTarget(&renderTarget);
//
#pragma once

#include <Oa/Core/Types.h>
#include <Oa/Ui/Camera.h>
#include <Oa/Render/Scene.h>

// Forward declarations.

struct OaTexture;      // Runtime/Texture.h
struct OaSwapchain;    // Runtime/Swapchain.h

// OaViewportMode — What kind of content is being viewed.

enum class OaViewportMode : OaU8 {
	Image2D,      // 2D image/video with orthographic camera
	Scene3D,      // 3D scene with perspective camera
	Matrix,       // Matrix-as-heatmap visualization
	Video,        // Video with timeline (2D + time control)
	CameraStream, // Live camera feed
};

// OaViewportDesc — Viewport rectangle and depth range.

struct OaViewportDesc {
	OaI32 X = 0;
	OaI32 Y = 0;
	OaI32 Width = 1280;
	OaI32 Height = 720;
	OaF32 MinDepth = 0.0f;
	OaF32 MaxDepth = 1.0f;

	[[nodiscard]] OaF32 GetAspectRatio() const noexcept {
		return static_cast<OaF32>(Width) / static_cast<OaF32>(Height);
	}

	[[nodiscard]] bool IsValid() const noexcept {
		return Width > 0 && Height > 0;
	}
};

// ═════════════════════════════════════════════════════════════════════════════
// OaScissorDesc — Scissor rectangle for clipping
// ═════════════════════════════════════════════════════════════════════════════

struct OaScissorDesc {
	OaI32 X = 0;
	OaI32 Y = 0;
	OaI32 Width = 1280;
	OaI32 Height = 720;

	[[nodiscard]] bool Contains(OaI32 InX, OaI32 InY) const noexcept {
		return InX >= X && InX < X + Width && InY >= Y && InY < Y + Height;
	}
};

// ═════════════════════════════════════════════════════════════════════════════
// OaRenderLayer — Layer/filter flags for selective rendering
// ═════════════════════════════════════════════════════════════════════════════

enum class OaRenderLayer : OaU32 {
	None = 0,
	Default = 1 << 0,
	UI = 1 << 1,
	Overlay = 1 << 2,
	Background = 1 << 3,
	Debug = 1 << 4,
	All = 0xFFFFFFFF,
};

inline OaRenderLayer operator|(OaRenderLayer InA, OaRenderLayer InB) noexcept {
	return static_cast<OaRenderLayer>(static_cast<OaU32>(InA) | static_cast<OaU32>(InB));
}

inline OaRenderLayer operator&(OaRenderLayer InA, OaRenderLayer InB) noexcept {
	return static_cast<OaRenderLayer>(static_cast<OaU32>(InA) & static_cast<OaU32>(InB));
}

// ═════════════════════════════════════════════════════════════════════════════
// OaViewport — Passive view description for rendering
//
// Declares: camera, target, viewport, and view-specific settings.
// Does NOT: handle input, own windows, manage swapchains, or animate.
// ═════════════════════════════════════════════════════════════════════════════

class OaViewport {
public:
	OaViewport() = default;
	explicit OaViewport(OaViewportMode InMode) : Mode_(InMode) {}

	// ═══════════════════════════════════════════════════════════════════════
	// Mode and configuration
	// ═══════════════════════════════════════════════════════════════════════

	void SetMode(OaViewportMode InMode) noexcept { Mode_ = InMode; }
	[[nodiscard]] OaViewportMode GetMode() const noexcept { return Mode_; }

	// ═══════════════════════════════════════════════════════════════════════
	// Camera (required)
	// ═══════════════════════════════════════════════════════════════════════

	void SetCamera(const OaCamera* InCamera) noexcept { Camera_ = InCamera; }
	[[nodiscard]] const OaCamera* GetCamera() const noexcept { return Camera_; }

	// ═══════════════════════════════════════════════════════════════════════
	// Scene (3D mode only, optional)
	// ═══════════════════════════════════════════════════════════════════════

	void SetScene(const OaScene* InScene) noexcept { Scene_ = InScene; }
	[[nodiscard]] const OaScene* GetScene() const noexcept { return Scene_; }

	// ═══════════════════════════════════════════════════════════════════════
	// Render target (required)
	// ═══════════════════════════════════════════════════════════════════════

	void SetTarget(const OaTexture* InTarget) noexcept { Target_ = InTarget; }
	[[nodiscard]] const OaTexture* GetTarget() const noexcept { return Target_; }

	// ═══════════════════════════════════════════════════════════════════════
	// Viewport and scissor
	// ═══════════════════════════════════════════════════════════════════════

	void SetViewport(const OaViewportDesc& InViewport) noexcept { Viewport_ = InViewport; }
	[[nodiscard]] const OaViewportDesc& GetViewport() const noexcept { return Viewport_; }

	void SetScissor(const OaScissorDesc& InScissor) noexcept { Scissor_ = InScissor; UseScissor_ = true; }
	void DisableScissor() noexcept { UseScissor_ = false; }
	[[nodiscard]] bool IsScissorEnabled() const noexcept { return UseScissor_; }
	[[nodiscard]] const OaScissorDesc& GetScissor() const noexcept { return Scissor_; }

	// ═══════════════════════════════════════════════════════════════════════
	// Layer filtering
	// ═══════════════════════════════════════════════════════════════════════

	void SetLayerMask(OaRenderLayer InMask) noexcept { LayerMask_ = InMask; }
	[[nodiscard]] OaRenderLayer GetLayerMask() const noexcept { return LayerMask_; }

	// Check if a layer is visible
	[[nodiscard]] bool IsLayerVisible(OaRenderLayer InLayer) const noexcept {
		return (static_cast<OaU32>(LayerMask_) & static_cast<OaU32>(InLayer)) != 0;
	}

	// ═══════════════════════════════════════════════════════════════════════
	// Clear values
	// ═══════════════════════════════════════════════════════════════════════

	void SetClearColor(const VlmVec4& InColor) noexcept { ClearColor_ = InColor; }
	[[nodiscard]] const VlmVec4& GetClearColor() const noexcept { return ClearColor_; }

	void SetClearDepth(OaF32 InDepth) noexcept { ClearDepth_ = InDepth; }
	[[nodiscard]] OaF32 GetClearDepth() const noexcept { return ClearDepth_; }

	// Validation.

	[[nodiscard]] bool IsValid() const noexcept {
		return Camera_ != nullptr && Target_ != nullptr && Viewport_.IsValid();
	}

	// Convenience helpers.

	// Setup for 2D image viewing (orthographic camera)
	void SetupImage2D(
		OaF32 InImageWidth,
		OaF32 InImageHeight,
		const OaTexture* InTarget,
		const OaViewportDesc& InViewport
	) noexcept;

	// Setup for aspect-fit image display
	void SetupImageAspectFit(
		OaF32 InImageWidth,
		OaF32 InImageHeight,
		OaF32 InWindowWidth,
		OaF32 InWindowHeight,
		const OaTexture* InTarget
	) noexcept;

private:
	OaViewportMode Mode_ = OaViewportMode::Image2D;
	const OaCamera* Camera_ = nullptr;
	const OaScene* Scene_ = nullptr;
	const OaTexture* Target_ = nullptr;
	OaViewportDesc Viewport_;
	OaScissorDesc Scissor_;
	bool UseScissor_ = false;
	OaRenderLayer LayerMask_ = OaRenderLayer::Default;
	VlmVec4 ClearColor_ = {0.01f, 0.01f, 0.01f, 1.0f};
	OaF32 ClearDepth_ = 1.0f;

	// Internal camera storage for convenience setups
	OaCamera InternalCamera_;
};
