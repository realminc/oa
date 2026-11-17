use super::*;

#[test]
fn video_transport_leaves_canvas_gestures_available_at_each_scale() {
	for scale in [1.0, 1.5, 2.0] {
		let viewport = PixelRect::new(0, 0, (1280.0 * scale) as u32, (720.0 * scale) as u32);
		assert!(!owns_pointer(viewport, scale, 360.0 * scale));
		let bar = geometry(viewport, scale);
		assert!(owns_pointer(viewport, scale, bar.y as f32));
		assert!(owns_pointer(viewport, scale, viewport.h as f32 - 1.0));
		assert!(!owns_pointer(viewport, scale, viewport.h as f32));
	}
}
