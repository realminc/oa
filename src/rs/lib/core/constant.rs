//! OA branding constants and ASCII banners.
//!
//! Port provenance: `oa/core/constant.h`.
//!
//! Central location for OA/Realm Library branding, ASCII banners, and
//! application metadata. Used by CLI tools, viewer applications, and tutorials.

// ─── ASCII banners ────────────────────────────────────────────────────────────

/// Main Realm banner (centered for ~80-character terminal).
pub const REALM_BANNER: &str = r"
                  ██████╗ ███████╗ █████╗ ██╗     ███╗   ███╗
                  ██╔══██╗██╔════╝██╔══██╗██║     ████╗ ████║
                  ██████╔╝█████╗  ███████║██║     ██╔████╔██║
                  ██╔══██╗██╔══╝  ██╔══██║██║     ██║╚██╔╝██║
                  ██║  ██║███████╗██║  ██║███████╗██║ ╚═╝ ██║
                  ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚══════╝╚═╝     ╚═╝
";

/// Compact single-box banner for CLI tools.
pub const COMPACT_BANNER: &str = r"
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                oa::Library                                    ║
║                  The Library. Every Vendor. Vulkan Native.                    ║
╚═══════════════════════════════════════════════════════════════════════════════╝
";

// ─── Application titles ───────────────────────────────────────────────────────

/// Window / application title for the OA viewport.
pub const VIEWPORT_TITLE: &str = "OaViewport";

// ─── Branding helpers ─────────────────────────────────────────────────────────

/// Prepend [`VIEWPORT_TITLE`] to a custom window title.
///
/// - Empty `title` → returns `"OaViewport"`.
/// - Title already starts with `"OaViewport"` → returned unchanged (no double-prepend).
/// - Otherwise → `"OaViewport" + title`.
///
/// Port provenance: `oa::brandViewport` in `oa/core/constant.h`.
pub fn brand_viewport(title: &str) -> String {
	if title.is_empty() {
		return VIEWPORT_TITLE.to_owned();
	}
	if title.starts_with(VIEWPORT_TITLE) {
		return title.to_owned();
	}
	format!("{VIEWPORT_TITLE}{title}")
}
