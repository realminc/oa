use super::*;

#[test]
fn mapped_formats_preserve_colors_alpha_and_padded_rows() {
	for (format, input, alpha) in [
		(VideoFormat::RGBA, [231, 47, 13, 99], 99),
		(VideoFormat::RGBx, [231, 47, 13, 99], 255),
		(VideoFormat::BGRA, [13, 47, 231, 99], 99),
		(VideoFormat::BGRx, [13, 47, 231, 99], 255),
	] {
		let source = [input.as_slice(), &[9, 9, 9, 9], input.as_slice()].concat();
		assert_eq!(
			pack_rgba(&source, 8, 1, 2, format).unwrap(),
			[231, 47, 13, alpha, 231, 47, 13, alpha]
		);
	}
}
#[test]
fn malformed_mapped_frames_fail_before_pixel_access() {
	for (bytes, stride, width, height) in [
		(3, 4, 1, 1),
		(4, -4, 1, 1),
		(4, 3, 1, 1),
		(8, 8, 1, 2),
		(0, 4, 0, 1),
		(0, 4, 1, 0),
		(4, 4, u32::MAX, 1),
	] {
		assert!(pack_rgba(&vec![0; bytes], stride, width, height, VideoFormat::RGBA).is_err());
	}
	assert!(pack_rgba(&[0; 4], 4, 1, 1, VideoFormat::RGB).is_err());
}
