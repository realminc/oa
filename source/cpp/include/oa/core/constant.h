// OA branding and application constants.
//
// Central location for Oa Library branding, ASCII banners, and application metadata.

#pragma once

#include <oa/core/types.h>

namespace oa {

// ─── ASCII BANNERS ─────────────────────────────────────────────────────────────

// main Realm banner (centered for ~80 char terminal)
inline constexpr const char* RealmBanner = R"(
                  ██████╗ ███████╗ █████╗ ██╗     ███╗   ███╗
                  ██╔══██╗██╔════╝██╔══██╗██║     ████╗ ████║
                  ██████╔╝█████╗  ███████║██║     ██╔████╔██║
                  ██╔══██╗██╔══╝  ██╔══██║██║     ██║╚██╔╝██║
                  ██║  ██║███████╗██║  ██║███████╗██║ ╚═╝ ██║
                  ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚══════╝╚═╝     ╚═╝
)";

// Compact banner for CLI tools
inline constexpr const char* CompactBanner = R"(
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                oa::Library                                     ║
║                  The Library. Every Vendor. Vulkan Native.                    ║
╚═══════════════════════════════════════════════════════════════════════════════╝
)";

// ─── APP TITLES ────────────────────────────────────────────────────────────

inline constexpr const char* ViewportTitle = "OaViewport";

// ─── BRANDING HELPERS ─────────────────────────────────────────────────────────

// Prepend OA_TITLE_VIEWPORT to a custom title (no automatic separator)
// User can add separator in their custom title if desired
// Returns OA_TITLE_VIEWPORT for empty/null input or if already starts with it
inline oa::String brandViewport(const oa::String& inTitle) {
	if (inTitle.empty()) {
		return ViewportTitle;
	}
	// Don't double-prepend if title already starts with OA_TITLE_VIEWPORT
	if (inTitle.find(ViewportTitle) == 0) {
		return inTitle;
	}
	return String(ViewportTitle) + inTitle;
}

inline oa::String brandViewport(const char* inTitle) {
	if (inTitle == nullptr or inTitle[0] == '\0') {
		return ViewportTitle;
	}
	// Don't double-prepend if title already starts with OA_TITLE_VIEWPORT
	if (strncmp(inTitle, ViewportTitle, strlen(ViewportTitle)) == 0) {
		return inTitle;
	}
	return String(ViewportTitle) + inTitle;
}

} // namespace oa
