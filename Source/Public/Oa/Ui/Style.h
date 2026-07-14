// OaUi — Style system: colors, theme, push/pop stack.
//
// All colors are linear RGBA floats [0,1].  Realm Design System dark theme is
// the default (struct defaults = RealmDark).  Call OaUiStyle::RealmLight() for
// the inverted palette.  Both sets are derived from ui/src/styles/theme.ts.
//
// Usage:
//   oui.PushStyle({.Accent = OaColor{1.F, 0.F, 0.F, 1.F}});
//   oui.Button("Special");
//   oui.PopStyle();

#pragma once

#include <Oa/Core/Color.h>

// Backward compatibility alias
using OaUiColor = OaColor;


// ─── OaUiStyle ─────────────────────────────────────────────────────────────────
// Struct defaults = Realm Design System dark theme (ui/src/styles/theme.ts).

struct OaUiStyle {
	// Geometry
	OaF32 CornerRadius  = 6.0F;
	OaF32 BorderWidth   = 1.0F;
	OaF32 ShadowBlur    = 8.0F;
	OaF32 ShadowOffset  = 2.0F;
	OaF32 FontSize      = 14.0F;
	OaF32 ItemSpacing   = 4.0F;
	OaF32 Padding       = 8.0F;

	// Backgrounds — #0a0a0a / #1a1a1a / #222222 / #303030
	OaColor Background    = {0.039F, 0.039F, 0.039F, 1.0F};
	OaColor Surface       = {0.102F, 0.102F, 0.102F, 1.0F};
	OaColor SurfaceHover  = {0.133F, 0.133F, 0.133F, 1.0F};
	OaColor SurfaceActive = {0.188F, 0.188F, 0.188F, 1.0F};

	// Borders — rgba(255,255,255, 0.06 / 0.10 / 0.15)
	OaColor BorderSubtle = {1.0F, 1.0F, 1.0F, 0.06F};
	OaColor Border       = {1.0F, 1.0F, 1.0F, 0.10F};
	OaColor BorderStrong = {1.0F, 1.0F, 1.0F, 0.15F};

	// Text — #f5f5f5 / #d4d4d4 / #909090 / #666666
	OaColor Text          = {0.961F, 0.961F, 0.961F, 1.0F};
	OaColor TextSecondary = {0.831F, 0.831F, 0.831F, 1.0F};
	OaColor TextMuted     = {0.565F, 0.565F, 0.565F, 1.0F};
	OaColor TextDisabled  = {0.400F, 0.400F, 0.400F, 1.0F};

	// Accent — #6366f1 indigo / #818cf8 hover
	OaColor Accent       = {0.388F, 0.400F, 0.945F, 1.0F};
	OaColor AccentHover  = {0.506F, 0.549F, 0.973F, 1.0F};
	OaColor AccentActive = {0.306F, 0.333F, 0.902F, 1.0F};

	// Semantic — #30d158 green / #f59e0b amber / #ff453a red
	OaColor Success = {0.188F, 0.820F, 0.345F, 1.0F};
	OaColor Warning = {0.961F, 0.620F, 0.043F, 1.0F};
	OaColor Error   = {1.000F, 0.271F, 0.227F, 1.0F};

	// ── Preset factories ──────────────────────────────────────────────────────

	// Struct defaults are already RealmDark; this exists for explicitness.
	[[nodiscard]] static OaUiStyle RealmDark();

	// Realm Design System light variant.
	[[nodiscard]] static OaUiStyle RealmLight();
};
