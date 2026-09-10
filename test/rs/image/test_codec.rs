use std::time::{SystemTime, UNIX_EPOCH};

use oa::{ErrorKind, Image, ImageFormat, ImageLayout, Matrix, image::ImageCodec};

#[test]
fn codec_capabilities_and_host_rgba_sink_are_checked() -> oa::Result<()> {
	assert!(!oa::image::can_decode(ImageCodec::Auto));
	assert!(!oa::image::can_encode(ImageCodec::Auto));
	for codec in [
		ImageCodec::Jpeg,
		ImageCodec::Png,
		ImageCodec::Webp,
		ImageCodec::Bmp,
		ImageCodec::Tga,
	] {
		assert!(oa::image::can_decode(codec));
		assert!(oa::image::can_encode(codec));
	}

	let path = temporary_path("png");
	oa::image::save_rgba_file(&[255, 0, 0, 255, 0, 255, 0, 128], 2, 1, &path, 90)?;
	let decoded = image_codec::open(&path)
		.expect("saved PNG must decode")
		.to_rgba8();
	assert_eq!(decoded.dimensions(), (2, 1));
	assert_eq!(decoded.into_raw(), [255, 0, 0, 255, 0, 255, 0, 128]);
	std::fs::remove_file(path).expect("temporary PNG removal failed");

	let invalid_path = temporary_path("gif");
	let error = oa::image::save_rgba_file(&[0; 4], 1, 1, invalid_path, 90)
		.expect_err("unsupported extension was accepted");
	assert_eq!(error.kind(), ErrorKind::InvalidArgument);
	assert!(oa::image::save_rgba_file(&[0; 3], 1, 1, temporary_path("png"), 90).is_err());
	Ok(())
}

test_vk!(
	codec_memory_and_file_roundtrips_preserve_contracts,
	engine,
	{
		let values = [
			1.0, 0.0, 0.25, 0.75, // red
			0.0, 1.0, 0.5, 0.25, // green
			0.0, 0.0, 0.75, 1.0, // blue
		];
		let image = Image::new(
			Matrix::from_f32(&engine, [1, 3, 2, 2], &values)?,
			ImageLayout::Nchw,
			ImageFormat::Rgb,
		)?;

		for codec in [
			ImageCodec::Png,
			ImageCodec::Bmp,
			ImageCodec::Tga,
			ImageCodec::Jpeg,
			ImageCodec::Webp,
		] {
			let encoded = oa::image::encode(&image, codec, 95)?;
			assert!(!encoded.is_empty());
			let decoded = oa::image::decode_memory(&engine, &encoded, ImageFormat::Rgb)?;
			assert_eq!(decoded.layout(), ImageLayout::Nchw);
			assert_eq!(decoded.format(), ImageFormat::Rgb);
			assert_eq!(decoded.as_matrix().shape(), [1, 3, 2, 2]);
			let actual = decoded.as_matrix().read_f32()?;
			if matches!(codec, ImageCodec::Jpeg | ImageCodec::Webp) {
				assert!(
					actual
						.iter()
						.all(|value| value.is_finite() && (0.0..=1.0).contains(value))
				);
			} else {
				for (actual, expected) in actual.iter().zip(values) {
					assert!(
						(actual - expected).abs() <= 1.0 / 255.0 + 1e-6,
						"{codec:?}: expected {expected}, found {actual}"
					);
				}
			}
		}

		let path = temporary_path("png");
		oa::image::save_file(&path, &image, 90)?;
		let decoded = oa::image::decode_file(&engine, &path, ImageFormat::Rgba)?;
		assert_eq!(decoded.format(), ImageFormat::Rgba);
		assert_eq!(decoded.as_matrix().shape(), [1, 4, 2, 2]);
		assert_eq!(decoded.as_matrix().read_f32()?[12..], [1.0, 1.0, 1.0, 1.0]);
		std::fs::remove_file(path).expect("temporary image removal failed");
		Ok(())
	}
);

fn temporary_path(extension: &str) -> std::path::PathBuf {
	let nonce = SystemTime::now()
		.duration_since(UNIX_EPOCH)
		.expect("system time before Unix epoch")
		.as_nanos();
	std::env::temp_dir().join(format!(
		"oa-image-codec-{}-{nonce}.{extension}",
		std::process::id()
	))
}
