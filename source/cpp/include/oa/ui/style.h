// Ui — style system: colors, theme, push/pop stack.
//
// UI colors are display-referred sRGB RGBA floats [0,1]. They are packed into
// the RGBA8 UNORM compose target and presented without a second transfer
// function when the surface exposes the preferred UNORM format. RealmDark is
// the struct default; EditorDark/EditorLight carry the compact OA editor look.
//
// usage:
//   ui.pushStyle({.accent = oa::Color{1.f, 0.f, 0.f, 1.f}});
//   ui.button("Special");
//   ui.popStyle();

#pragma once

#include <oa/core/color.h>
#include <oa/core/status.h>

// backward compatibility alias


// ─── UiStyle ─────────────────────────────────────────────────────────────────
// Struct defaults = Realm Design System dark theme (ui/src/styles/theme.ts).

namespace oa {

struct UiStyle {
	// Geometry
	oa::F32 cornerRadius  = 6.0F;
	oa::F32 borderWidth   = 1.0F;
	oa::F32 shadowBlur    = 8.0F;
	oa::F32 shadowOffset  = 2.0F;
	oa::F32 fontSize      = 14.0F;
	oa::F32 itemSpacing   = 4.0F;
	oa::F32 padding       = 8.0F;
	// Compact control geometry carried forward from the OA editor theme.
	oa::F32 framePaddingX = 6.0F;
	oa::F32 framePaddingY = 4.0F;

	// Backgrounds — #0a0a0a / #1a1a1a / #222222 / #303030
	oa::Color background    = {0.039F, 0.039F, 0.039F, 1.0F};
	oa::Color surface       = {0.102F, 0.102F, 0.102F, 1.0F};
	oa::Color surfaceHover  = {0.133F, 0.133F, 0.133F, 1.0F};
	oa::Color surfaceActive = {0.188F, 0.188F, 0.188F, 1.0F};

	// Borders — rgba(255,255,255, 0.06 / 0.10 / 0.15)
	oa::Color borderSubtle = {1.0F, 1.0F, 1.0F, 0.06F};
	oa::Color border       = {1.0F, 1.0F, 1.0F, 0.10F};
	oa::Color borderStrong = {1.0F, 1.0F, 1.0F, 0.15F};

	// text — #f5f5f5 / #d4d4d4 / #909090 / #666666
	oa::Color text          = {0.961F, 0.961F, 0.961F, 1.0F};
	oa::Color textSecondary = {0.831F, 0.831F, 0.831F, 1.0F};
	oa::Color textMuted     = {0.565F, 0.565F, 0.565F, 1.0F};
	oa::Color textDisabled  = {0.400F, 0.400F, 0.400F, 1.0F};

	// accent — #6366f1 indigo / #818cf8 hover
	oa::Color accent       = {0.388F, 0.400F, 0.945F, 1.0F};
	oa::Color accentHover  = {0.506F, 0.549F, 0.973F, 1.0F};
	oa::Color accentActive = {0.306F, 0.333F, 0.902F, 1.0F};

	// Semantic — #30d158 green / #f59e0b amber / #ff453a red
	oa::Color success = {0.188F, 0.820F, 0.345F, 1.0F};
	oa::Color warning = {0.961F, 0.620F, 0.043F, 1.0F};
	oa::Color error   = {1.000F, 0.271F, 0.227F, 1.0F};

	// ── Preset factories ──────────────────────────────────────────────────────

	// Every geometry value must be finite; sizes are non-negative and fontSize
	// is positive. Every color component must be finite and in [0,1].
	[[nodiscard]] oa::Status validate() const;

	// Struct defaults are already RealmDark; this exists for explicitness.
	[[nodiscard]] static UiStyle realmDark();

	// Realm Design System light variant.
	[[nodiscard]] static UiStyle realmLight();

	// Compact OA editor presets adapted from the prior editor's Dark Modern and
	// Light Modern palettes. These are the default Viewer application themes.
	[[nodiscard]] static UiStyle editorDark();
	[[nodiscard]] static UiStyle editorLight();
};

}  // namespace oa
