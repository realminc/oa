// OA_DOC_BEGIN: vision-image
use anyhow::Context;
use oa::{Engine, ImageFormat, Path, image};

fn main() -> anyhow::Result<()> {
	let engine = Engine::new().context("create OA engine")?;

	let path = Path::asset_rel("image/visionTestPattern320x180.jpg");

	let img = image::decode_file(&engine, &path, ImageFormat::Rgb)
		.with_context(|| format!("decode {}", path.display()))?;
	let small = image::resize(&img, 160, 90, image::InterpolationMode::Bilinear)?;
	let adjusted = image::brightness_contrast(&small, 0.05, 1.1)?;

	let matrix = adjusted.as_matrix();
	let values = matrix.read_f32()?;

	assert_eq!(matrix.shape(), [1, 3, 90, 160]);
	assert_eq!(values.len(), 3 * 90 * 160);

	let min = values.iter().copied().fold(f32::INFINITY, f32::min);
	let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
	println!("{:?}  min={min:.4}  max={max:.4}", matrix.shape());
	Ok(())
}
// OA_DOC_END: vision-image
