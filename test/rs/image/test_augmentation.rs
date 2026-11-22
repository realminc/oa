//! Deterministic image augmentation contracts.

use oa::{Image, ImageFormat, ImageLayout, Matrix, audio::NoiseDistribution};

fn gradient(engine: &oa::Engine, width: usize, height: usize) -> oa::Result<Image> {
	let values = (0..width * height)
		.map(|value| value as f32 / (width * height) as f32)
		.collect::<Vec<_>>();
	Image::new(
		Matrix::new_from_f32_on(engine, [1, 1, height, width], &values)?,
		ImageLayout::Nchw,
		ImageFormat::Gray,
	)
}

test_vk!(
	augmentation_noise_like_is_deterministic_and_bounded,
	engine,
	{
		let image = gradient(&engine, 8, 8)?;
		let first = oa::image::noise_like(&image, NoiseDistribution::Gaussian, 0.1, 42)?;
		let repeat = oa::image::noise_like(&image, NoiseDistribution::Gaussian, 0.1, 42)?;
		let other = oa::image::noise_like(&image, NoiseDistribution::Gaussian, 0.1, 43)?;
		assert_eq!(
			first.as_matrix().read_f32()?,
			repeat.as_matrix().read_f32()?
		);
		assert_ne!(first.as_matrix().read_f32()?, other.as_matrix().read_f32()?);
		let uniform = oa::image::noise_like(&image, NoiseDistribution::Uniform, 0.5, 42)?;
		assert!(
			uniform
				.as_matrix()
				.read_f32()?
				.iter()
				.all(|value| value.abs() <= 0.5 + 1e-6)
		);
		assert!(oa::image::noise_like(&image, NoiseDistribution::Gaussian, -1.0, 42).is_err());
		assert!(oa::image::noise_like(&image, NoiseDistribution::Gaussian, f32::NAN, 42).is_err());
		Ok(())
	}
);

test_vk!(
	augmentation_recipes_are_deterministic_and_validated,
	engine,
	{
		let image = gradient(&engine, 8, 8)?;

		let flip_a = oa::image::random_flip(&image, 1.0, 0.0, 7)?;
		let flip_b = oa::image::random_flip(&image, 1.0, 0.0, 7)?;
		assert_eq!(
			flip_a.as_matrix().read_f32()?,
			flip_b.as_matrix().read_f32()?
		);
		// p_horizontal = 1 forces a horizontal mirror for every seed.
		let source = image.as_matrix().read_f32()?;
		let flipped = flip_a.as_matrix().read_f32()?;
		for row in 0..8 {
			for column in 0..8 {
				assert_eq!(flipped[row * 8 + column], source[row * 8 + (7 - column)]);
			}
		}
		assert!(oa::image::random_flip(&image, 1.5, 0.0, 7).is_err());
		assert!(oa::image::random_flip(&image, 0.0, f32::NAN, 7).is_err());

		let crop_a = oa::image::random_crop(&image, 4, 4, 9)?;
		let crop_b = oa::image::random_crop(&image, 4, 4, 9)?;
		assert_eq!(
			crop_a.as_matrix().read_f32()?,
			crop_b.as_matrix().read_f32()?
		);
		assert_eq!(crop_a.as_matrix().shape(), [1, 1, 4, 4]);
		// The crop window must be a valid 4x4 contiguous window of the source.
		let cropped = crop_a.as_matrix().read_f32()?;
		let (mut offset_x, mut offset_y) = (usize::MAX, usize::MAX);
		for y in 0..=4 {
			for x in 0..=4 {
				if (0..16).all(|index| cropped[index] == source[(y + index / 4) * 8 + (x + index % 4)]) {
					(offset_x, offset_y) = (x, y);
				}
			}
		}
		assert!(offset_x <= 4 && offset_y <= 4);
		assert!(oa::image::random_crop(&image, 0, 4, 9).is_err());
		assert!(oa::image::random_crop(&image, 9, 4, 9).is_err());

		let rotate_a = oa::image::random_rotate(&image, 5)?;
		let rotate_b = oa::image::random_rotate(&image, 5)?;
		assert_eq!(
			rotate_a.as_matrix().read_f32()?,
			rotate_b.as_matrix().read_f32()?
		);
		// Every quarter-turn is a permutation of the source pixels.
		let mut sorted_a = rotate_a.as_matrix().read_f32()?;
		let mut sorted_s = source.clone();
		sorted_a.sort_by(f32::total_cmp);
		sorted_s.sort_by(f32::total_cmp);
		assert_eq!(sorted_a, sorted_s);
		Ok(())
	}
);
