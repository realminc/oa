// OaConstant — Branding and application constants
//
// Central location for Oa Library branding, ASCII banners, and application metadata.

#pragma once

#include <Oa/Core/Types.h>

// ─── ASCII BANNERS ─────────────────────────────────────────────────────────────

// Main Realm banner (centered for ~80 char terminal)
constexpr const char* OA_BANNER_REALM = R"(
                  ██████╗ ███████╗ █████╗ ██╗     ███╗   ███╗
                  ██╔══██╗██╔════╝██╔══██╗██║     ████╗ ████║
                  ██████╔╝█████╗  ███████║██║     ██╔████╔██║
                  ██╔══██╗██╔══╝  ██╔══██║██║     ██║╚██╔╝██║
                  ██║  ██║███████╗██║  ██║███████╗██║ ╚═╝ ██║
                  ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚══════╝╚═╝     ╚═╝
)";

// Compact banner for CLI tools
constexpr const char* OA_BANNER_COMPACT = R"(
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                Oa Library                                     ║
║                  The Library. Every Vendor. Vulkan Native.                    ║
╚═══════════════════════════════════════════════════════════════════════════════╝
)";

// ─── APP TITLES ────────────────────────────────────────────────────────────

constexpr const char* OA_TITLE_VIEWPORT = "OaViewport";

// ─── BRANDING HELPERS ─────────────────────────────────────────────────────────

// Prepend OA_TITLE_VIEWPORT to a custom title (no automatic separator)
// User can add separator in their custom title if desired
// Returns OA_TITLE_VIEWPORT for empty/null input or if already starts with it
inline OaString BrandViewport(const OaString& InTitle) {
	if (InTitle.empty()) {
		return OA_TITLE_VIEWPORT;
	}
	// Don't double-prepend if title already starts with OA_TITLE_VIEWPORT
	if (InTitle.find(OA_TITLE_VIEWPORT) == 0) {
		return InTitle;
	}
	return OaString(OA_TITLE_VIEWPORT) + InTitle;
}

inline OaString BrandViewport(const char* InTitle) {
	if (InTitle == nullptr || InTitle[0] == '\0') {
		return OA_TITLE_VIEWPORT;
	}
	// Don't double-prepend if title already starts with OA_TITLE_VIEWPORT
	if (strncmp(InTitle, OA_TITLE_VIEWPORT, strlen(OA_TITLE_VIEWPORT)) == 0) {
		return InTitle;
	}
	return OaString(OA_TITLE_VIEWPORT) + InTitle;
}
