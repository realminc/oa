//! Tests for `oa::constant` — banners and branding helpers.

use oa::{COMPACT_BANNER, REALM_BANNER, VIEWPORT_TITLE, brand_viewport};

#[test]
fn realm_banner_is_non_empty() {
	assert!(!REALM_BANNER.is_empty());
	// Banner uses Unicode block characters; check for the box-drawing chars
	// that appear in every variant of the Realm logo.
	assert!(REALM_BANNER.contains('█') || REALM_BANNER.contains('╗'));
}

#[test]
fn compact_banner_is_non_empty() {
	assert!(!COMPACT_BANNER.is_empty());
	assert!(COMPACT_BANNER.contains("oa::Library"));
}

#[test]
fn viewport_title_value() {
	assert_eq!(VIEWPORT_TITLE, "OaViewport");
}

#[test]
fn brand_viewport_empty_returns_title() {
	assert_eq!(brand_viewport(""), "OaViewport");
}

#[test]
fn brand_viewport_prepends() {
	assert_eq!(brand_viewport(" — My App"), "OaViewport — My App");
}

#[test]
fn brand_viewport_no_double_prepend() {
	assert_eq!(
		brand_viewport("OaViewport — Already"),
		"OaViewport — Already"
	);
}
