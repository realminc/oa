use std::{
	any::TypeId,
	time::{SystemTime, UNIX_EPOCH},
};

use oa::{ErrorKind, Texture};

#[test]
fn root_texture_export_is_the_render_value_identity() {
	assert_eq!(TypeId::of::<Texture>(), TypeId::of::<oa::render::Texture>());
}

test_vk!(
	packed_texture_upload_readback_and_file_sink_are_exact,
	engine,
	{
		let rgba = [
			255, 0, 0, 255, 0, 255, 0, 128, // row zero
			0, 0, 255, 64, 255, 255, 255, 0, // row one
		];
		let texture = oa::render::texture_from_rgba8(&engine, &rgba, 2, 2)?;
		assert_eq!((texture.width(), texture.height()), (2, 2));
		assert_eq!(texture.dtype(), oa::DType::U8);
		assert_eq!(texture.read_rgba8()?, rgba);
		assert_eq!(texture.clone().read_rgba8()?, rgba);

		let path = temporary_path();
		oa::render::save_texture_file(&texture, &path, 90)?;
		let decoded = image_codec::open(&path)
			.expect("saved texture PNG must decode")
			.to_rgba8();
		assert_eq!(decoded.into_raw(), rgba);
		std::fs::remove_file(path).expect("temporary texture image removal failed");
		Ok(())
	}
);

test_vk!(packed_texture_rejects_invalid_host_ranges, engine, {
	for error in [
		oa::render::texture_from_rgba8(&engine, &[], 0, 1)
			.err()
			.expect("zero width was accepted"),
		oa::render::texture_from_rgba8(&engine, &[0; 3], 1, 1)
			.err()
			.expect("short RGBA input was accepted"),
	] {
		assert_eq!(error.kind(), ErrorKind::InvalidArgument);
	}
	Ok(())
});

fn temporary_path() -> std::path::PathBuf {
	let nonce = SystemTime::now()
		.duration_since(UNIX_EPOCH)
		.expect("system time before Unix epoch")
		.as_nanos();
	std::env::temp_dir().join(format!(
		"oa-render-texture-{}-{nonce}.png",
		std::process::id()
	))
}
