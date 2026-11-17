use super::CAMERA_PIXEL_FORMAT;

#[test]
fn camera_format_preserves_rgba_byte_order() {
	use sdl3::sys::pixels::{SDL_GetPixelFormatDetails, SDL_GetRGBA};
	let expected = [231_u8, 47, 13, 199];
	let mut actual = [0_u8; 4];
	// SAFETY: SDL owns the immutable format description. Each channel output
	// points to one distinct initialized byte for the duration of the call.
	unsafe {
		let details = SDL_GetPixelFormatDetails(CAMERA_PIXEL_FORMAT);
		assert!(!details.is_null());
		SDL_GetRGBA(
			u32::from_ne_bytes(expected),
			details,
			std::ptr::null(),
			&mut actual[0],
			&mut actual[1],
			&mut actual[2],
			&mut actual[3],
		);
	}
	assert_eq!(actual, expected);
}
