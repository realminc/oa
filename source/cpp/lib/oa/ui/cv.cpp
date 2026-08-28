// oa::Ui — diagnostic CPU bounding-box compositing, no OpenCV.
//
// oa::CvFrame::render() composites all overlays into a CPU RGBA8 buffer and
// uploads via oa::FnTexture::FromPixels. This is the display path for the
// saved-image references. Realtime overlays use oa::DetectionOverlay.

// Engine first — VK_NO_PROTOTYPES before any vulkan.h pull-in.
#include <oa/runtime/engine.h>
#include <oa/runtime/engine/resourceAccess.h>
#include <oa/runtime/engine/allocatorAccess.h>
#include <oa/runtime/executionSession.h>

#include <oa/ui/cv.h>
#include <oa/ui/image.h>
#include <oa/core/color.h>
#include <oa/core/std/memory.h>
#include <oa/core/status.h>
#include <oa/core/std/algo.h>
#include <oa/core/std/cString.h>
#include <oa/core/std/limits.h>
#include <oa/core/std/scalarMath.h>
#include <oa/core/std/utility.h>
#include <stdio.h>

// ─── oa::CvFrame method bodies ──────────────────────────────────────────────────

void oa::CvFrame::addBboxes(oa::Vector<oa::CvBbox> inBoxes, const oa::CvBboxesConfig& inCfg) {
	oa::CvOverlayBboxes ov;
	ov.config = inCfg;
	ov.boxes  = oa::move(inBoxes);
	overlays.pushBack(oa::move(ov));
}

// ─── CPU drawing helpers ─────────────────────────────────────────────────────

namespace {

// Unpack oa::Color (0xRRGGBBAA) into r,g,b,a bytes
static void unpackColor(oa::Color inCol, oa::U8& R, oa::U8& G, oa::U8& B, oa::U8& A) {
	oa::U32 v = inCol.toU32();
	R = static_cast<oa::U8>((v >> 24) & 0xFFu);
	G = static_cast<oa::U8>((v >> 16) & 0xFFu);
	B = static_cast<oa::U8>((v >>  8) & 0xFFu);
	A = static_cast<oa::U8>( v        & 0xFFu);
}

// alpha-blend src over dst in place
static void blendPixel(oa::U8* dst, oa::U8 R, oa::U8 G, oa::U8 B, oa::U8 A) {
	oa::F32 a = static_cast<oa::F32>(A) / 255.0f;
	dst[0] = static_cast<oa::U8>(dst[0] * (1.0f - a) + R * a);
	dst[1] = static_cast<oa::U8>(dst[1] * (1.0f - a) + G * a);
	dst[2] = static_cast<oa::U8>(dst[2] * (1.0f - a) + B * a);
	dst[3] = 255;
}

static void drawHLine(oa::Vector<oa::U8>& buf, oa::I32 W, oa::I32 H,
	oa::I64 X0, oa::I64 X1, oa::I64 Y,
	oa::U8 R, oa::U8 G, oa::U8 B, oa::U8 A)
{
	if (Y < 0 || Y >= H) return;
	const oa::I64 x0 = oa::max<oa::I64>(0, X0);
	const oa::I64 x1 = oa::min<oa::I64>(static_cast<oa::I64>(W) - 1, X1);
	for (oa::I64 x = x0; x <= x1; ++x) {
		const oa::Usize offset =
			(static_cast<oa::Usize>(Y) * static_cast<oa::Usize>(W)
				+ static_cast<oa::Usize>(x)) * 4U;
		blendPixel(buf.data() + offset, R, G, B, A);
	}
}

static void drawVLine(oa::Vector<oa::U8>& buf, oa::I32 W, oa::I32 H,
	oa::I64 X, oa::I64 Y0, oa::I64 Y1,
	oa::U8 R, oa::U8 G, oa::U8 B, oa::U8 A)
{
	if (X < 0 || X >= W) return;
	const oa::I64 y0 = oa::max<oa::I64>(0, Y0);
	const oa::I64 y1 = oa::min<oa::I64>(static_cast<oa::I64>(H) - 1, Y1);
	for (oa::I64 y = y0; y <= y1; ++y) {
		const oa::Usize offset =
			(static_cast<oa::Usize>(y) * static_cast<oa::Usize>(W)
				+ static_cast<oa::Usize>(X)) * 4U;
		blendPixel(buf.data() + offset, R, G, B, A);
	}
}

// draw axis-aligned rect outline with integer thickness
static void drawRect(oa::Vector<oa::U8>& buf, oa::I32 W, oa::I32 H,
	oa::I64 X, oa::I64 Y, oa::I64 RW, oa::I64 RH, oa::I32 T,
	oa::U8 R, oa::U8 G, oa::U8 B, oa::U8 A)
{
	for (oa::I64 t = 0; t < T; ++t) {
		drawHLine(buf, W, H, X,      X + RW - 1, Y + t,      R, G, B, A);
		drawHLine(buf, W, H, X,      X + RW - 1, Y + RH - 1 - t, R, G, B, A);
		drawVLine(buf, W, H, X + t,      Y,      Y + RH - 1, R, G, B, A);
		drawVLine(buf, W, H, X + RW - 1 - t, Y,  Y + RH - 1, R, G, B, A);
	}
}

static void fillRect(oa::Vector<oa::U8>& buf, oa::I32 W, oa::I32 H,
	oa::I64 X, oa::I64 Y, oa::I64 RW, oa::I64 RH,
	oa::U8 R, oa::U8 G, oa::U8 B, oa::U8 A)
{
	const oa::I64 y0 = oa::max<oa::I64>(0, Y);
	const oa::I64 y1 = oa::min<oa::I64>(H, Y + RH);
	for (oa::I64 y = y0; y < y1; ++y) {
		drawHLine(buf, W, H, X, X + RW - 1, y, R, G, B, A);
	}
}

// draw a compact 5x7 label font.
static const oa::U8 kFont5x7[][7] = {
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

static oa::I32 charIndex(char C) {
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

static void drawChar(oa::Vector<oa::U8>& buf, oa::I32 W, oa::I32 H,
	oa::I64 X, oa::I64 Y, char C, oa::I32 scale, oa::U8 R, oa::U8 G, oa::U8 B)
{
	const oa::I32 idx = charIndex(C);
	for (oa::I32 row = 0; row < 7; ++row) {
		const oa::U8 bits = kFont5x7[idx][row];
		for (oa::I32 col = 0; col < 5; ++col) {
			if (bits & (0b10000 >> col)) {
				for (oa::I32 sy = 0; sy < scale; ++sy) {
					for (oa::I32 sx = 0; sx < scale; ++sx) {
						const oa::I64 px = X + col * scale + sx;
						const oa::I64 py = Y + row * scale + sy;
						if (px >= 0 && px < W && py >= 0 && py < H) {
							const oa::Usize offset =
								(static_cast<oa::Usize>(py) * static_cast<oa::Usize>(W)
									+ static_cast<oa::Usize>(px)) * 4U;
							oa::U8* p = buf.data() + offset;
							p[0] = R; p[1] = G; p[2] = B; p[3] = 255;
						}
					}
				}
			}
		}
	}
}

static void drawString(oa::Vector<oa::U8>& buf, oa::I32 W, oa::I32 H,
	oa::I64 X, oa::I64 Y, const char* str, oa::I32 scale, oa::U8 R, oa::U8 G, oa::U8 B)
{
	oa::I64 cx = X;
	for (const char* p = str; *p; ++p) {
		drawChar(buf, W, H, cx, Y, *p, scale, R, G, B);
		cx += 6 * scale;
	}
}

} // anonymous namespace

// ─── oa::CvFrame::Render ────────────────────────────────────────────────────────
// Composites all overlays onto a copy of the base pixels (or a black canvas if
// base is nullptr) and uploads to GPU via oa::FnTexture::FromPixels.

static oa::Result<oa::Texture> renderFrame(
	const oa::CvFrame& inFrame,
	oa::Engine& inRt,
	oa::Span<const oa::U8> inBaseRgba)
{
	const oa::I32 W = inFrame.w;
	const oa::I32 H = inFrame.h;
	if (W <= 0 || H <= 0) {
		return oa::Status::invalidArgument("oa::CvFrame: invalid W/H");
	}

	oa::Vector<oa::U8> pixels;
	const oa::U64 nBytes = static_cast<oa::U64>(W) * static_cast<oa::U64>(H) * 4U;
	if (nBytes > static_cast<oa::U64>(oa::Limits<oa::I64>::max())) {
		return oa::Status::error(oa::StatusCode::ResourceExhausted,
			"oa::CvFrame: image is too large for host composition");
	}
	pixels.resize(static_cast<oa::Usize>(nBytes), 0);

	if (not inBaseRgba.empty() and inBaseRgba.size() < nBytes) {
		return oa::Status::invalidArgument(
			"oa::CvFrame::render: host base is smaller than W*H*4");
	}
	if (inBaseRgba.size() >= nBytes) {
		oa::memcpy(pixels.data(), inBaseRgba.data(), nBytes);
	} else if (inFrame.base != nullptr) {
		if (inFrame.base->buffer == nullptr or inFrame.base->size < nBytes) {
			return oa::Status::invalidArgument(
				"oa::CvFrame::render: base is null or smaller than W*H*4");
		}
		// Read back the base RGBA8 buffer before composing host-side overlays.
		if (auto status = oa::EngineResourceAccess::readbackBuffer(inRt,
			*inFrame.base, 0U, pixels.data(), nBytes);
			not status.isOk()) return status;
	} else {
		// Black canvas fallback — keep alpha=255 so overlays show on it.
		for (oa::Usize i = 3U; i < pixels.size(); i += 4U) pixels[i] = 255;
	}

	// apply overlays in order
	for (const auto& ov : inFrame.overlays) {
		const auto& inOverlay = ov;
		if (not oa::isFinite(inOverlay.config.alpha)
			or inOverlay.config.alpha < 0.0F or inOverlay.config.alpha > 1.0F
			or not oa::isFinite(inOverlay.config.lineWidth)
			or inOverlay.config.lineWidth <= 0.0F
			or inOverlay.config.lineWidth > 1024.0F
			or inOverlay.config.labelScale <= 0
			or inOverlay.config.labelScale > 16
			or not oa::isFinite(inOverlay.config.color.r)
			or not oa::isFinite(inOverlay.config.color.g)
			or not oa::isFinite(inOverlay.config.color.b)
			or not oa::isFinite(inOverlay.config.color.a)) {
			return oa::Status::invalidArgument(
				"oa::CvFrame: invalid bounding-box overlay config");
		}
		oa::U8 r, g, b, a;
		unpackColor(inOverlay.config.color, r, g, b, a);
		a = static_cast<oa::U8>(inOverlay.config.alpha * 255.0F);
		const oa::I32 thickness = static_cast<oa::I32>(oa::max(
			1.0F, inOverlay.config.lineWidth));

		for (const auto& box : inOverlay.boxes) {
			const oa::F64 right = static_cast<oa::F64>(box.x) + box.w;
			const oa::F64 bottom = static_cast<oa::F64>(box.y) + box.h;
			constexpr oa::F64 minCoordinate =
				static_cast<oa::F64>(oa::Limits<oa::I32>::min());
			constexpr oa::F64 maxCoordinate =
				static_cast<oa::F64>(oa::Limits<oa::I32>::max());
			if (not oa::isFinite(box.x) or not oa::isFinite(box.y)
				or not oa::isFinite(box.w) or not oa::isFinite(box.h)
				or not oa::isFinite(box.score) or box.w <= 0.0F or box.h <= 0.0F
				or box.x < minCoordinate or box.y < minCoordinate
				or box.w > maxCoordinate or box.h > maxCoordinate
				or right > maxCoordinate or bottom > maxCoordinate) {
				return oa::Status::invalidArgument(
					"oa::CvFrame: invalid bounding box");
			}
			const oa::I64 bx = static_cast<oa::I32>(box.x);
			const oa::I64 by = static_cast<oa::I32>(box.y);
			const oa::I64 bw = static_cast<oa::I32>(box.w);
			const oa::I64 bh = static_cast<oa::I32>(box.h);
			if (bw <= 0 or bh <= 0) {
				return oa::Status::invalidArgument(
					"oa::CvFrame: bounding box has no whole-pixel extent");
			}
			drawRect(pixels, W, H, bx, by, bw, bh, thickness, r, g, b, a);

			if (inOverlay.config.showLabels || inOverlay.config.showScores) {
				char label[64];
				if (inOverlay.config.showLabels && inOverlay.config.showScores) {
					::snprintf(label, sizeof(label), "%s %.0f%%",
						box.label.empty() ? "obj" : box.label.cStr(),
						box.score * 100.0F);
				} else if (inOverlay.config.showLabels) {
					::snprintf(label, sizeof(label), "%s",
						box.label.empty() ? "obj" : box.label.cStr());
				} else {
					::snprintf(label, sizeof(label), "%.0f%%", box.score * 100.0F);
				}
				const oa::I32 labelScale = oa::max(1, inOverlay.config.labelScale);
				const oa::I64 labelWidth =
					static_cast<oa::I64>(oa::strlen(label)) * 6 * labelScale + 4;
				const oa::I64 labelHeight = 7 * labelScale + 4;
				const oa::I64 lx = bx + 2;
				oa::I64 ly = by - labelHeight;
				if (ly < 0) ly = by + 2;
				fillRect(pixels, W, H, lx - 2, ly - 2,
					labelWidth, labelHeight, 0, 0, 0, 192);
				drawString(pixels, W, H, lx, ly, label,
					labelScale, r, g, b);
			}
		}
	}

	return oa::FnTexture::fromPixels(inRt,
		oa::Span<const oa::U8>(pixels.data(), pixels.size()), W, H);
}

static oa::Status completeDeviceBaseIfNeeded(
	const oa::CvFrame& inFrame,
	oa::ExecutionSession& inContext,
	oa::Span<const oa::U8> inBaseRgba)
{
	if (inFrame.w <= 0 or inFrame.h <= 0) return oa::Status::ok();
	const oa::U64 bytes = static_cast<oa::U64>(inFrame.w)
		* static_cast<oa::U64>(inFrame.h) * 4U;
	if (inBaseRgba.size() >= bytes or inFrame.base == nullptr) {
		return oa::Status::ok();
	}

	const oavk::Buffer& base = *inFrame.base;
	oa::Engine& engine = inContext.engine();
	if (base.buffer == nullptr or base.size < bytes) {
		return oa::Status::invalidArgument(
			"oa::CvFrame::render: base is null or smaller than W*H*4");
	}
	if (base.allocation == nullptr or base.aliasIdentity != nullptr
		or base.allocatorIdentity != oa::EngineAllocatorAccess::get(engine).allocator) {
		return oa::Status::invalidArgument(
			"oa::CvFrame::render: base must be a non-aliased buffer owned by the context engine");
	}
	return inContext.submitAndWait();
}

oa::Result<oa::Texture> oa::CvFrame::render(oa::Engine& inRt) const {
	return render(inRt, {});
}

oa::Result<oa::Texture> oa::CvFrame::render(
	oa::Engine& inRt,
	oa::Span<const oa::U8> inBaseRgba) const {
	auto& context = oa::ExecutionSession::forEngine(inRt);
	OA_RETURN_IF_ERROR(completeDeviceBaseIfNeeded(
		*this, context, inBaseRgba));
	return renderFrame(*this, inRt, inBaseRgba);
}
