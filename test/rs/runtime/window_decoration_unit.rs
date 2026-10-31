use super::*;
#[test]
fn native_hit_regions_preserve_donor_controls_and_resize_precedence() {
	use HitTestResult::*;
	assert_eq!(hit_test(1, 1, 1280, 720, true, false), ResizeTopLeft);
	assert_eq!(
		hit_test(1279, 719, 1280, 720, true, false),
		ResizeBottomRight
	);
	assert_eq!(hit_test(600, 2, 1280, 720, true, false), ResizeTop);
	assert_eq!(hit_test(600, 20, 1280, 720, true, false), Draggable);
	for x in [23, 1143, 1183, 1220, 1257] {
		assert_eq!(hit_test(x, 20, 1280, 720, true, false), Normal);
	}
	assert_eq!(hit_test(600, 100, 1280, 720, true, false), Normal);
	assert_eq!(hit_test(1, 1, 1280, 720, true, true), Draggable);
	assert_eq!(hit_test(-1, 20, 1280, 720, true, false), Normal);
}
