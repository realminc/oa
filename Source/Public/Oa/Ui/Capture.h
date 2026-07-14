// OaUi compatibility aliases plus screenshot API.
// Camera ownership lives in <Oa/Vision/CameraCapture.h>.

#pragma once

#include <Oa/Core/Types.h>
#include <Oa/Core/Status.h>
#include <Oa/Vision/CameraCapture.h>

using OaCaptureConfig = OaCameraCaptureConfig;
using OaCapture = OaCameraCapture;


// ─── OaScreenshot ─────────────────────────────────────────────────────────────

struct OaScreenshotResult {
	OaString Path;
	OaI32    Width  = 0;
	OaI32    Height = 0;
};

// Write InRgba (CPU RGBA8) to InPath as PNG via stb_image_write.
// If InPath is empty, auto-generates "screenshot_YYYYMMDD_HHMMSS.png".
[[nodiscard]] OaResult<OaScreenshotResult> OaScreenshot(
	OaStringView       InPath,
	OaSpan<const OaU8> InRgba,
	OaI32              InW,
	OaI32              InH);
