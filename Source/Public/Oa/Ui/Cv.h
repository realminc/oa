// OaUi — legacy CPU reference CV compositing.
//
// OaCvFrame remains a diagnostic/reference path for edges, masks and saved
// images. It reads back and uploads a complete RGBA frame and must not be used
// for realtime playback. Use <Oa/Ui/DetectionOverlay.h> for completion-safe,
// GPU-resident boxes and SDF labels over image/video views.
//
// OA does not depend on OpenCV.
//
// Usage:
//   OaCvFrame frame;
//   frame.Base = camera_buf;
//   frame.W = 1280;  frame.H = 720;
//   frame.AddEdges({.Low = 50.F, .High = 150.F});
//   frame.AddBboxes(detections, {.ShowLabels = true});
//   oui.CvFrame(frame, {20, 20, 640, 360});

#pragma once

#include <Oa/Core/Types.h>
#include <Oa/Core/Status.h>
#include <Oa/Core/Color.h>

class OaUi;
class OaEngine;
class OaContext;
struct OaTexture;
class OaVkBuffer;
struct OaPixelRect;


// ─── Overlay config structs ───────────────────────────────────────────────────

struct OaCvEdgesConfig {
	OaF32    Low   = 50.0F;
	OaF32    High  = 150.0F;
	OaColor  Color = OaColor::Cyan();
	OaF32    Alpha = 0.8F;
};

struct OaCvBlobsConfig {
	OaF32  MinArea       = 100.0F;
	OaF32  MaxArea       = 1e6F;
	OaColor Color        = OaColor::Success();
	bool   ShowContours  = true;
	bool   ShowCentroids = true;
	OaF32  Alpha         = 0.7F;
};

struct OaCvBbox {
	OaF32      X, Y, W, H;
	OaF32      Score     = 1.0F;
	OaI32      ClassId   = 0;
	OaString   Label;
};

struct OaCvBboxesConfig {
	OaColor Color      = OaColor::Error();
	OaF32   LineWidth  = 1.5F;
	OaF32   Alpha      = 1.0F;
	OaI32   LabelScale = 1;
	bool    ShowLabels = true;
	bool    ShowScores = true;
};

struct OaCvMask {
	OaVkBuffer* Buffer = nullptr;  // OaU8 device buffer, same W×H as base
};

struct OaCvMasksConfig {
	OaColor Color = OaColor::Accent();
	OaF32   Alpha = 0.5F;
};

struct OaCvFlowConfig {
	OaF32 Scale  = 1.0F;
	OaF32 Step   = 16.0F;  // grid step in pixels
	OaColor Color = OaColor::Yellow();
	OaF32   Alpha = 0.8F;
};

// Stats overlay drawn as text in a corner.
enum class OaCvStatsCorner : OaU8 {
	TopLeft     = 0,
	TopRight    = 1,
	BottomLeft  = 2,
	BottomRight = 3,
};

struct OaCvStatsConfig {
	OaCvStatsCorner Corner = OaCvStatsCorner::TopLeft;
	bool ShowFps   = true;
	bool ShowShape = true;
	bool ShowMin   = true;
	bool ShowMax   = true;
	bool ShowMean  = true;
};


// ─── OaCvOverlay ──────────────────────────────────────────────────────────────

enum class OaCvOverlayKind : OaU8 {
	Edges = 0,
	Blobs = 1,
	Bboxes = 2,
	Masks  = 3,
	Flow   = 4,
	Stats  = 5,
};

struct OaCvOverlayEdges  { OaCvEdgesConfig  Config; };
struct OaCvOverlayBlobs  { OaCvBlobsConfig  Config; };
struct OaCvOverlayBboxes { OaCvBboxesConfig Config; OaVec<OaCvBbox> Boxes; };
struct OaCvOverlayMasks  { OaCvMasksConfig  Config; OaVec<OaCvMask> Masks; };
struct OaCvOverlayFlow   { OaCvFlowConfig   Config; OaVkBuffer* FlowX = nullptr; OaVkBuffer* FlowY = nullptr; };
struct OaCvOverlayStats  { OaCvStatsConfig  Config; };

using OaCvOverlay = OaVariant<
	OaCvOverlayEdges,
	OaCvOverlayBlobs,
	OaCvOverlayBboxes,
	OaCvOverlayMasks,
	OaCvOverlayFlow,
	OaCvOverlayStats
>;


// ─── OaCvFrame ────────────────────────────────────────────────────────────────

struct OaCvFrame {
	OaVkBuffer*         Base    = nullptr;  // RGBA8 device buffer
	OaI32               W       = 0;
	OaI32               H       = 0;
	OaVec<OaCvOverlay>  Overlays;

	void AddEdges (const OaCvEdgesConfig&  InCfg = {});
	void AddBlobs (const OaCvBlobsConfig&  InCfg = {});
	void AddBboxes(OaVec<OaCvBbox>         InBoxes, const OaCvBboxesConfig& InCfg = {});
	void AddMasks (OaVec<OaCvMask>         InMasks, const OaCvMasksConfig&  InCfg = {});
	void AddFlow  (OaVkBuffer* InFlowX, OaVkBuffer* InFlowY, const OaCvFlowConfig& InCfg = {});
	void AddStats (const OaCvStatsConfig&  InCfg = {});

	void ClearOverlays() noexcept { Overlays.clear(); }

	// CPU reference/diagnostic composite -> upload RGBA8 OaTexture.
	// The context overload completes pending work before reading Base. The engine
	// overload uses a matching thread-default context when available. Realtime
	// consumers must use OaDetectionOverlay or another resident path.
	[[nodiscard]] OaResult<OaTexture> Render(OaContext& InContext) const;
	[[nodiscard]] OaResult<OaTexture> Render(
		OaContext& InContext,
		OaSpan<const OaU8> InBaseRgba) const;
	[[nodiscard]] OaResult<OaTexture> Render(OaEngine& InRt) const;
	[[nodiscard]] OaResult<OaTexture> Render(
		OaEngine& InRt,
		OaSpan<const OaU8> InBaseRgba) const;
};
