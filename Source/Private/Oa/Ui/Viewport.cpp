// OaViewport — Implementation
//
// Passive view description for rendering.

#include <Oa/Ui/Viewport.h>
#include <algorithm>

void OaViewport::SetupImage2D(
	OaF32 InImageWidth,
	OaF32 InImageHeight,
	const OaTexture* InTarget,
	const OaViewportDesc& InViewport
) noexcept {
	Mode_ = OaViewportMode::Image2D;

	// Setup orthographic camera for 2D viewing
	InternalCamera_ = OaCamera(InImageWidth, InImageHeight);
	InternalCamera_.SetOrthographic(InImageWidth, InImageHeight);

	Camera_ = &InternalCamera_;
	Target_ = InTarget;
	Viewport_ = InViewport;

	// Default scissor to viewport
	Scissor_ = {
		static_cast<OaI32>(InViewport.X),
		static_cast<OaI32>(InViewport.Y),
		static_cast<OaI32>(InViewport.Width),
		static_cast<OaI32>(InViewport.Height)
	};
	UseScissor_ = false;  // Disabled by default
}

void OaViewport::SetupImageAspectFit(
	OaF32 InImageWidth,
	OaF32 InImageHeight,
	OaF32 InWindowWidth,
	OaF32 InWindowHeight,
	const OaTexture* InTarget
) noexcept {
	Mode_ = OaViewportMode::Image2D;

	// Calculate aspect-fit dimensions
	OaF32 imageAspect = InImageWidth / InImageHeight;
	OaF32 windowAspect = InWindowWidth / InWindowHeight;

	OaF32 displayWidth, displayHeight;
	if (imageAspect > windowAspect) {
		// Image is wider relative to window - fit to width
		displayWidth = InWindowWidth;
		displayHeight = InWindowWidth / imageAspect;
	} else {
		// Image is taller relative to window - fit to height
		displayHeight = InWindowHeight;
		displayWidth = InWindowHeight * imageAspect;
	}

	// Center in window
	OaF32 offsetX = (InWindowWidth - displayWidth) * 0.5f;
	OaF32 offsetY = (InWindowHeight - displayHeight) * 0.5f;

	// Setup viewport
	OaViewportDesc viewport;
	viewport.X = static_cast<OaI32>(offsetX);
	viewport.Y = static_cast<OaI32>(offsetY);
	viewport.Width = static_cast<OaI32>(displayWidth);
	viewport.Height = static_cast<OaI32>(displayHeight);

	SetupImage2D(InImageWidth, InImageHeight, InTarget, viewport);

	// Update internal camera to match the aspect-fit zoom
	InternalCamera_.SetZoom(1.0f);
}
