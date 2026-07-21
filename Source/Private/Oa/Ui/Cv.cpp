// OaUi — OaCv implementation (Phase 1: CPU compositing, no OpenCV)
//
// OaCvFrame::Render() composites all overlays into a CPU RGBA8 buffer and
// uploads via OaTexture::FromPixels. This is the display path for the
// TutorialBboxOverlay and camera loop.  Phase 2 will replace with Slang.

// Engine first — VK_NO_PROTOTYPES before any vulkan.h pull-in.
#include <Oa/Runtime/Engine.h>
#include <Oa/Runtime/Context.h>

#include <Oa/Ui/Cv.h>
#include <Oa/Ui/Image.h>
#include <Oa/Core/Color.h>
#include <Oa/Core/Status.h>
#include <algorithm>
#include <cstring>
#include <cstdio>
#include <cmath>

namespace {

OaContext& ContextForEngine(OaEngine& InEngine) {
	OaContext* active = OaContext::GetDefaultPtr();
	return active != nullptr and active->GetEngine() == &InEngine
		? *active : InEngine.GetContext();
}

} // namespace

// ─── OaCvFrame method bodies ──────────────────────────────────────────────────

void OaCvFrame::AddEdges(const OaCvEdgesConfig& InCfg) {
	Overlays.PushBack(OaCvOverlayEdges{InCfg});
}

void OaCvFrame::AddBlobs(const OaCvBlobsConfig& InCfg) {
	Overlays.PushBack(OaCvOverlayBlobs{InCfg});
}

void OaCvFrame::AddBboxes(OaVec<OaCvBbox> InBoxes, const OaCvBboxesConfig& InCfg) {
	OaCvOverlayBboxes ov;
	ov.Config = InCfg;
	ov.Boxes  = std::move(InBoxes);
	Overlays.PushBack(std::move(ov));
}

void OaCvFrame::AddMasks(OaVec<OaCvMask> InMasks, const OaCvMasksConfig& InCfg) {
	OaCvOverlayMasks ov;
	ov.Config = InCfg;
	ov.Masks  = std::move(InMasks);
	Overlays.PushBack(std::move(ov));
}

void OaCvFrame::AddFlow(OaVkBuffer* InFlowX, OaVkBuffer* InFlowY, const OaCvFlowConfig& InCfg) {
	Overlays.PushBack(OaCvOverlayFlow{InCfg, InFlowX, InFlowY});
}

void OaCvFrame::AddStats(const OaCvStatsConfig& InCfg) {
	Overlays.PushBack(OaCvOverlayStats{InCfg});
}

// ─── CPU drawing helpers ─────────────────────────────────────────────────────

namespace {

// Unpack OaColor (0xRRGGBBAA) into r,g,b,a bytes
static void UnpackColor(OaColor InCol, OaU8& R, OaU8& G, OaU8& B, OaU8& A) {
	OaU32 v = InCol.ToU32();
	R = static_cast<OaU8>((v >> 24) & 0xFFu);
	G = static_cast<OaU8>((v >> 16) & 0xFFu);
	B = static_cast<OaU8>((v >>  8) & 0xFFu);
	A = static_cast<OaU8>( v        & 0xFFu);
}

// Alpha-blend src over dst in place
static void BlendPixel(OaU8* Dst, OaU8 R, OaU8 G, OaU8 B, OaU8 A) {
	OaF32 a = static_cast<OaF32>(A) / 255.0f;
	Dst[0] = static_cast<OaU8>(Dst[0] * (1.0f - a) + R * a);
	Dst[1] = static_cast<OaU8>(Dst[1] * (1.0f - a) + G * a);
	Dst[2] = static_cast<OaU8>(Dst[2] * (1.0f - a) + B * a);
	Dst[3] = 255;
}

static void DrawHLine(OaVec<OaU8>& Buf, OaI32 W, OaI32 H,
	OaI32 X0, OaI32 X1, OaI32 Y,
	OaU8 R, OaU8 G, OaU8 B, OaU8 A)
{
	if (Y < 0 || Y >= H) return;
	OaI32 x0 = std::max(0, X0);
	OaI32 x1 = std::min(W - 1, X1);
	for (OaI32 x = x0; x <= x1; ++x) {
		BlendPixel(Buf.Data() + (Y * W + x) * 4, R, G, B, A);
	}
}

static void DrawVLine(OaVec<OaU8>& Buf, OaI32 W, OaI32 H,
	OaI32 X, OaI32 Y0, OaI32 Y1,
	OaU8 R, OaU8 G, OaU8 B, OaU8 A)
{
	if (X < 0 || X >= W) return;
	OaI32 y0 = std::max(0, Y0);
	OaI32 y1 = std::min(H - 1, Y1);
	for (OaI32 y = y0; y <= y1; ++y) {
		BlendPixel(Buf.Data() + (y * W + X) * 4, R, G, B, A);
	}
}

// Draw axis-aligned rect outline with integer thickness
static void DrawRect(OaVec<OaU8>& Buf, OaI32 W, OaI32 H,
	OaI32 X, OaI32 Y, OaI32 RW, OaI32 RH, OaI32 T,
	OaU8 R, OaU8 G, OaU8 B, OaU8 A)
{
	for (OaI32 t = 0; t < T; ++t) {
		DrawHLine(Buf, W, H, X,      X + RW - 1, Y + t,      R, G, B, A);
		DrawHLine(Buf, W, H, X,      X + RW - 1, Y + RH - 1 - t, R, G, B, A);
		DrawVLine(Buf, W, H, X + t,      Y,      Y + RH - 1, R, G, B, A);
		DrawVLine(Buf, W, H, X + RW - 1 - t, Y,  Y + RH - 1, R, G, B, A);
	}
}

static void FillRect(OaVec<OaU8>& Buf, OaI32 W, OaI32 H,
	OaI32 X, OaI32 Y, OaI32 RW, OaI32 RH,
	OaU8 R, OaU8 G, OaU8 B, OaU8 A)
{
	for (OaI32 y = std::max(0, Y); y < std::min(H, Y + RH); ++y) {
		DrawHLine(Buf, W, H, X, X + RW - 1, y, R, G, B, A);
	}
}

// Draw a compact 5x7 label font.
static const OaU8 kFont5x7[][7] = {
	// 0
	{0b01110, 0b10001, 0b10011, 0b10101, 0b11001, 0b10001, 0b01110},
	// 1
	{0b00100, 0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110},
	// 2
	{0b01110, 0b10001, 0b00001, 0b00010, 0b00100, 0b01000, 0b11111},
	// 3
	{0b11111, 0b00010, 0b00100, 0b00010, 0b00001, 0b10001, 0b01110},
	// 4
	{0b00010, 0b00110, 0b01010, 0b10010, 0b11111, 0b00010, 0b00010},
	// 5
	{0b11111, 0b10000, 0b11110, 0b00001, 0b00001, 0b10001, 0b01110},
	// 6
	{0b00110, 0b01000, 0b10000, 0b11110, 0b10001, 0b10001, 0b01110},
	// 7
	{0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b01000, 0b01000},
	// 8
	{0b01110, 0b10001, 0b10001, 0b01110, 0b10001, 0b10001, 0b01110},
	// 9
	{0b01110, 0b10001, 0b10001, 0b01111, 0b00001, 0b00010, 0b01100},
	// : (10)
	{0b00000, 0b00100, 0b00000, 0b00000, 0b00100, 0b00000, 0b00000},
	// . (11)
	{0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00100},
	// ' ' (12)
	{0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000},
	// % (13)
	{0b11000, 0b11001, 0b00010, 0b00100, 0b01000, 0b10011, 0b00011},
	{0b01110, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001}, // A
	{0b11110, 0b10001, 0b10001, 0b11110, 0b10001, 0b10001, 0b11110}, // B
	{0b01110, 0b10001, 0b10000, 0b10000, 0b10000, 0b10001, 0b01110}, // C
	{0b11110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b11110}, // D
	{0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b11111}, // E
	{0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b10000}, // F
	{0b01110, 0b10001, 0b10000, 0b10111, 0b10001, 0b10001, 0b01110}, // G
	{0b10001, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001}, // H
	{0b01110, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110}, // I
	{0b00001, 0b00001, 0b00001, 0b00001, 0b10001, 0b10001, 0b01110}, // J
	{0b10001, 0b10010, 0b10100, 0b11000, 0b10100, 0b10010, 0b10001}, // K
	{0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b11111}, // L
	{0b10001, 0b11011, 0b10101, 0b10101, 0b10001, 0b10001, 0b10001}, // M
	{0b10001, 0b11001, 0b10101, 0b10011, 0b10001, 0b10001, 0b10001}, // N
	{0b01110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110}, // O
	{0b11110, 0b10001, 0b10001, 0b11110, 0b10000, 0b10000, 0b10000}, // P
	{0b01110, 0b10001, 0b10001, 0b10001, 0b10101, 0b10010, 0b01101}, // Q
	{0b11110, 0b10001, 0b10001, 0b11110, 0b10100, 0b10010, 0b10001}, // R
	{0b01111, 0b10000, 0b10000, 0b01110, 0b00001, 0b00001, 0b11110}, // S
	{0b11111, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100}, // T
	{0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110}, // U
	{0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01010, 0b00100}, // V
	{0b10001, 0b10001, 0b10001, 0b10101, 0b10101, 0b10101, 0b01010}, // W
	{0b10001, 0b10001, 0b01010, 0b00100, 0b01010, 0b10001, 0b10001}, // X
	{0b10001, 0b10001, 0b01010, 0b00100, 0b00100, 0b00100, 0b00100}, // Y
	{0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b10000, 0b11111}, // Z
};

static OaI32 CharIndex(char C) {
	if (C >= '0' && C <= '9') return C - '0';
	if (C == ':') return 10;
	if (C == '.') return 11;
	if (C == ' ') return 12;
	if (C == '%') return 13;
	if (C >= 'a' && C <= 'z') C = static_cast<char>(C - 'a' + 'A');
	if (C >= 'A' && C <= 'Z') return 14 + C - 'A';
	// fallback: space
	return 12;
}

static void DrawChar(OaVec<OaU8>& Buf, OaI32 W, OaI32 H,
	OaI32 X, OaI32 Y, char C, OaI32 Scale, OaU8 R, OaU8 G, OaU8 B)
{
	OaI32 idx = CharIndex(C);
	for (OaI32 row = 0; row < 7; ++row) {
		OaU8 bits = kFont5x7[idx][row];
		for (OaI32 col = 0; col < 5; ++col) {
			if (bits & (0b10000 >> col)) {
				for (OaI32 sy = 0; sy < Scale; ++sy) {
					for (OaI32 sx = 0; sx < Scale; ++sx) {
						const OaI32 px = X + col * Scale + sx;
						const OaI32 py = Y + row * Scale + sy;
						if (px >= 0 && px < W && py >= 0 && py < H) {
							OaU8* p = Buf.Data() + (py * W + px) * 4;
							p[0] = R; p[1] = G; p[2] = B; p[3] = 255;
						}
					}
				}
			}
		}
	}
}

static void DrawString(OaVec<OaU8>& Buf, OaI32 W, OaI32 H,
	OaI32 X, OaI32 Y, const char* Str, OaI32 Scale, OaU8 R, OaU8 G, OaU8 B)
{
	OaI32 cx = X;
	for (const char* p = Str; *p; ++p) {
		DrawChar(Buf, W, H, cx, Y, *p, Scale, R, G, B);
		cx += 6 * Scale;
	}
}

} // anonymous namespace

// ─── OaCvFrame::Render ────────────────────────────────────────────────────────
// Composites all overlays onto a copy of the base pixels (or a black canvas if
// Base is nullptr) and uploads to GPU via OaTexture::FromPixels.

static OaResult<OaTexture> RenderFrame(
	const OaCvFrame& InFrame,
	OaEngine& InRt,
	OaSpan<const OaU8> InBaseRgba)
{
	const OaI32 W = InFrame.W;
	const OaI32 H = InFrame.H;
	if (W <= 0 || H <= 0) return OaStatus::Error("OaCvFrame: invalid W/H");

	OaVec<OaU8> pixels;
	const OaU64 nBytes = static_cast<OaU64>(W) * static_cast<OaU64>(H) * 4U;
	pixels.Resize(static_cast<OaI64>(nBytes), 0);

	if (InBaseRgba.Size() >= nBytes) {
		std::memcpy(pixels.Data(), InBaseRgba.Data(), nBytes);
	} else if (InFrame.Base != nullptr) {
		if (InFrame.Base->Buffer == nullptr or InFrame.Base->Size < nBytes) {
			return OaStatus::InvalidArgument(
				"OaCvFrame::Render: Base is null or smaller than W*H*4");
		}
		// Read back the base RGBA8 buffer before composing host-side overlays.
		if (auto status = InRt.ReadbackBuffer(
			*InFrame.Base, 0U, pixels.Data(), nBytes);
			not status.IsOk()) return status;
	} else {
		// Black canvas fallback — keep alpha=255 so overlays show on it.
		for (OaI64 i = 3; i < pixels.Size(); i += 4) pixels[i] = 255;
	}

	// Apply overlays in order
	for (const auto& ov : InFrame.Overlays) {
		const_cast<OaCvOverlay&>(ov).Visit([&](const auto& InOverlay) {
			using T = std::decay_t<decltype(InOverlay)>;

			if constexpr (std::is_same_v<T, OaCvOverlayBboxes>) {
				OaU8 r, g, b, a;
				UnpackColor(InOverlay.Config.Color, r, g, b, a);
				a = static_cast<OaU8>(InOverlay.Config.Alpha * 255.0f);
				OaI32 thickness = static_cast<OaI32>(std::max(1.0f, InOverlay.Config.LineWidth));

				for (const auto& box : InOverlay.Boxes) {
					OaI32 bx = static_cast<OaI32>(box.X);
					OaI32 by = static_cast<OaI32>(box.Y);
					OaI32 bw = static_cast<OaI32>(box.W);
					OaI32 bh = static_cast<OaI32>(box.H);
					DrawRect(pixels, W, H, bx, by, bw, bh, thickness, r, g, b, a);

					if (InOverlay.Config.ShowLabels || InOverlay.Config.ShowScores) {
						char label[64];
						if (InOverlay.Config.ShowLabels && InOverlay.Config.ShowScores) {
							std::snprintf(label, sizeof(label), "%s %.0f%%",
								box.Label.empty() ? "obj" : box.Label.c_str(),
								box.Score * 100.0f);
						} else if (InOverlay.Config.ShowLabels) {
							std::snprintf(label, sizeof(label), "%s",
								box.Label.empty() ? "obj" : box.Label.c_str());
						} else {
							std::snprintf(label, sizeof(label), "%.0f%%", box.Score * 100.0f);
						}
						const OaI32 labelScale = std::max(1, InOverlay.Config.LabelScale);
						const OaI32 labelWidth =
							static_cast<OaI32>(std::strlen(label)) * 6 * labelScale + 4;
						const OaI32 labelHeight = 7 * labelScale + 4;
						OaI32 lx = bx + 2;
						OaI32 ly = by - labelHeight;
						if (ly < 0) ly = by + 2;
						FillRect(pixels, W, H, lx - 2, ly - 2,
							labelWidth, labelHeight, 0, 0, 0, 192);
						DrawString(pixels, W, H, lx, ly, label,
							labelScale, r, g, b);
					}
				}
			}
			// Edges, Blobs, Masks, Flow, Stats — Phase 2 (stub for now)
		});
	}

	return OaTexture::FromPixels(InRt,
		OaSpan<const OaU8>(pixels.Data(), pixels.Size()), W, H);
}

static OaStatus CompleteDeviceBaseIfNeeded(
	const OaCvFrame& InFrame,
	OaContext& InContext,
	OaSpan<const OaU8> InBaseRgba)
{
	if (InFrame.W <= 0 or InFrame.H <= 0) return OaStatus::Ok();
	const OaU64 bytes = static_cast<OaU64>(InFrame.W)
		* static_cast<OaU64>(InFrame.H) * 4U;
	if (InBaseRgba.Size() >= bytes or InFrame.Base == nullptr) {
		return OaStatus::Ok();
	}

	const OaVkBuffer& base = *InFrame.Base;
	OaEngine& engine = InContext.Engine();
	if (base.Buffer == nullptr or base.Size < bytes) {
		return OaStatus::InvalidArgument(
			"OaCvFrame::Render: Base is null or smaller than W*H*4");
	}
	if (base.Allocation == nullptr or base.AliasIdentity != nullptr
		or base.IsImported() or base.NodeIndex != 0U
		or base.AllocatorIdentity != engine.Allocator.Allocator) {
		return OaStatus::InvalidArgument(
			"OaCvFrame::Render: Base must be a non-aliased buffer owned by the context engine");
	}
	OA_RETURN_IF_ERROR(InContext.Execute());
	return InContext.Sync();
}

OaResult<OaTexture> OaCvFrame::Render(OaEngine& InRt) const {
	return Render(ContextForEngine(InRt));
}

OaResult<OaTexture> OaCvFrame::Render(
	OaEngine& InRt,
	OaSpan<const OaU8> InBaseRgba) const {
	return Render(ContextForEngine(InRt), InBaseRgba);
}

OaResult<OaTexture> OaCvFrame::Render(OaContext& InContext) const {
	return Render(InContext, {});
}

OaResult<OaTexture> OaCvFrame::Render(
	OaContext& InContext,
	OaSpan<const OaU8> InBaseRgba) const
{
	OA_RETURN_IF_ERROR(CompleteDeviceBaseIfNeeded(
		*this, InContext, InBaseRgba));
	return RenderFrame(*this, InContext.Engine(), InBaseRgba);
}
