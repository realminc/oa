// OaUi — plot style enums (OaColor, OaMarker, OaColormapKind).
//
// The original Ui/Plot.h declared an OaFigure / OaAxes / namespace OaPlot
// API that was never wired up — zero callers, no implementation. The
// matplotlib-style plotting now lives at Source/Public/Oa/Plot/ (OaPlot::
// Figure / Axes; UnifiedExecutionArchitecture.md §3.5, Step 3e). This header keeps only the
// colour/marker/colormap enums because they're still part of the Realm
// Design System vocabulary and referenced by Oa/Ui/Cv.h + Oa/Ui/Rl.h.

#pragma once

#include <Oa/Core/Types.h>


// ─── OaPlotColor ────────────────────────────────────────────────────────────────
// 0xRRGGBBAA packed values.  Realm Design System palette from theme.ts.
// DEPRECATED: Use OaColor from Oa/Core/Color.h instead.

enum class OaPlotColor : OaU32 {
	// Auto = pick next from cycle
	Auto        = 0x00000000U,

	// Realm semantic
	Accent      = 0x6366F1FFU,   // #6366f1 indigo
	AccentHover = 0x818CF8FFU,   // #818cf8
	Success     = 0x30D158FFU,   // #30d158 green
	Warning     = 0xF59E0BFFU,   // #f59e0b amber
	Error       = 0xFF453AFFU,   // #ff453a red

	// Backgrounds
	BgBase      = 0x0A0A0AFFU,
	BgWidget    = 0x1A1A1AFFU,
	BgHover     = 0x222222FFU,
	BgActive    = 0x303030FFU,

	// Text
	TextPrimary   = 0xF5F5F5FFU,
	TextSecondary = 0xD4D4D4FFU,
	TextMuted     = 0x909090FFU,

	// Named plot colors (matplotlib-like C0..C7)
	C0     = Accent,              // indigo
	C1     = 0xFF6B35FFU,         // orange
	C2     = Success,             // green
	C3     = Error,               // red
	C4     = 0xA855F7FFU,         // purple
	C5     = 0x22D3EEFFU,         // cyan
	C6     = 0xEC4899FFU,         // pink
	C7     = Warning,             // amber/yellow

	// Aliases
	Blue   = Accent,
	Orange = C1,
	Green  = Success,
	Red    = Error,
	Purple = C4,
	Cyan   = C5,
	Pink   = C6,
	Yellow = Warning,
};


// ─── OaMarker ─────────────────────────────────────────────────────────────────

enum class OaMarker : OaU8 {
	None    = 0,
	Circle  = 1,
	Square  = 2,
	Diamond = 3,
	Cross   = 4,
	Plus    = 5,
};


// ─── OaColormapKind ───────────────────────────────────────────────────────────

enum class OaColormapKind : OaU8 {
	Plasma   = 0,
	Viridis  = 1,
	Coolwarm = 2,
	Grays    = 3,
	Realm    = 4,
};
