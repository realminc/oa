use oa::{
	ErrorKind, Image, ImageFormat, ImageLayout, Matrix, OpAttribute, OpValueKind,
	image::{BorderMode, InterpolationMode},
};

#[test]
fn resize_api_defaults_to_bilinear_without_hiding_the_call_mode() {
	assert_eq!(InterpolationMode::default(), InterpolationMode::Bilinear);
	let _: fn(&Image, u32, u32, InterpolationMode) -> oa::Result<Image> = oa::image::resize;
}

#[test]
fn geometric_api_exposes_the_complete_donor_family() {
	let _: fn(&Image, u32, u32, u32, u32) -> oa::Result<Image> = oa::image::crop;
	let _: fn(&Image, bool, bool) -> oa::Result<Image> = oa::image::flip;
	let _: fn(&Image, u32) -> oa::Result<Image> = oa::image::rotate;
	let _: fn(&Image, u32, u32) -> oa::Result<Image> = oa::image::center_crop;
	let _: fn(&Image, &Matrix, InterpolationMode, BorderMode, f32) -> oa::Result<Image> =
		oa::image::remap;
	assert_eq!(BorderMode::default(), BorderMode::Constant);
}

test_vk!(
	crop_flip_rotate_and_center_crop_match_exact_oracles,
	engine,
	{
		let values = (0..12).map(|value| value as f32).collect::<Vec<_>>();
		let image = Image::new(
			Matrix::from_f32(&engine, [1, 1, 3, 4], &values)?,
			ImageLayout::Nchw,
			ImageFormat::Gray,
		)?;
		let crop = oa::image::crop(&image, 1, 1, 9, 2)?;
		assert_eq!(crop.as_matrix().shape(), [1, 1, 2, 3]);
		assert_eq!(
			crop.as_matrix().read_f32()?,
			[5.0, 6.0, 7.0, 9.0, 10.0, 11.0]
		);
		assert_eq!(
			oa::image::flip(&image, true, true)?
				.as_matrix()
				.read_f32()?,
			[11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0]
		);
		let rotate_90 = oa::image::rotate(&image, 90)?;
		assert_eq!(rotate_90.as_matrix().shape(), [1, 1, 4, 3]);
		assert_eq!(
			rotate_90.as_matrix().read_f32()?,
			[8.0, 4.0, 0.0, 9.0, 5.0, 1.0, 10.0, 6.0, 2.0, 11.0, 7.0, 3.0]
		);
		assert_eq!(
			oa::image::rotate(&image, 360)?.as_matrix().read_f32()?,
			values
		);
		assert_eq!(
			oa::image::center_crop(&image, 2, 1)?
				.as_matrix()
				.read_f32()?,
			[5.0, 6.0]
		);
		Ok(())
	}
);

test_vk!(
	pad_matches_all_five_border_mode_oracles_on_odd_padding,
	engine,
	{
		let source = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
		let image = Image::new(
			Matrix::from_f32(&engine, [1, 2, 3], &source)?,
			ImageLayout::Chw,
			ImageFormat::Gray,
		)?;
		for mode in [
			BorderMode::Constant,
			BorderMode::Replicate,
			BorderMode::Reflect,
			BorderMode::Reflect101,
			BorderMode::Wrap,
		] {
			let output = oa::image::pad(&image, 2, 1, 1, 2, mode, -7.0)?;
			let expected = pad_reference(&source, 2, 3, 2, 1, 1, 2, mode, -7.0);
			assert_eq!(output.as_matrix().shape(), [1, 5, 6]);
			assert_eq!(output.as_matrix().read_f32()?, expected, "{mode:?}");
		}
		Ok(())
	}
);

test_vk!(
	remap_affine_and_perspective_match_coordinate_oracles,
	engine,
	{
		let source = (0..9).map(|value| value as f32).collect::<Vec<_>>();
		let image = Image::new(
			Matrix::from_f32(&engine, [1, 1, 3, 3], &source)?,
			ImageLayout::Nchw,
			ImageFormat::Gray,
		)?;
		let map = Matrix::from_f32(
			&engine,
			[1, 2, 2, 2],
			&[0.0, 1.0, 2.0, 1.0, 0.0, 1.0, 2.0, 0.0],
		)?;
		assert_eq!(
			oa::image::remap(
				&image,
				&map,
				InterpolationMode::Nearest,
				BorderMode::Constant,
				-1.0,
			)?
			.as_matrix()
			.read_f32()?,
			[0.0, 4.0, 8.0, 1.0]
		);
		let affine = Matrix::from_f32(&engine, [2, 3], &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0])?;
		assert_eq!(
			oa::image::warp_affine(
				&image,
				&affine,
				3,
				3,
				InterpolationMode::Bilinear,
				BorderMode::Constant,
				-1.0,
			)?
			.as_matrix()
			.read_f32()?,
			source
		);
		let perspective = Matrix::from_f32(
			&engine,
			[3, 3],
			&[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
		)?;
		let (plan, warped) = engine.capture(|| {
			oa::image::warp_perspective(
				&image,
				&perspective,
				3,
				3,
				InterpolationMode::Nearest,
				BorderMode::Replicate,
				0.0,
			)
		})?;
		assert_eq!(
			plan
				.semantic_graph()
				.values()
				.iter()
				.map(|value| value.kind())
				.collect::<Vec<_>>(),
			[OpValueKind::Image, OpValueKind::Matrix, OpValueKind::Image]
		);
		assert_eq!(
			plan.semantic_graph().operations()[0].name(),
			"oa::image::warp_perspective"
		);
		engine.submit(&plan)?.wait()?;
		assert_eq!(warped.as_matrix().read_f32()?, source);
		Ok(())
	}
);

test_vk!(
	geometric_operations_reject_invalid_structural_contracts,
	engine,
	{
		let image = Image::new(
			Matrix::from_f32(&engine, [1, 1, 2, 2], &[0.0; 4])?,
			ImageLayout::Nchw,
			ImageFormat::Gray,
		)?;
		let bad_map = Matrix::from_f32(&engine, [1, 3, 2, 2], &[0.0; 12])?;
		let bad_affine = Matrix::from_f32(&engine, [3, 2], &[0.0; 6])?;
		for error in [
			oa::image::crop(&image, 2, 0, 1, 1)
				.err()
				.expect("invalid crop was accepted"),
			oa::image::center_crop(&image, 3, 1)
				.err()
				.expect("invalid center crop was accepted"),
			oa::image::rotate(&image, 45)
				.err()
				.expect("invalid rotation was accepted"),
			oa::image::pad(&image, 0, 0, 0, 0, BorderMode::Constant, f32::NAN)
				.err()
				.expect("nonfinite border value was accepted"),
			oa::image::remap(
				&image,
				&bad_map,
				InterpolationMode::Nearest,
				BorderMode::Constant,
				0.0,
			)
			.err()
			.expect("invalid remap was accepted"),
			oa::image::warp_affine(
				&image,
				&bad_affine,
				2,
				2,
				InterpolationMode::Nearest,
				BorderMode::Constant,
				0.0,
			)
			.err()
			.expect("invalid affine transform was accepted"),
		] {
			assert_eq!(error.kind(), ErrorKind::InvalidArgument);
		}
		Ok(())
	}
);

#[allow(clippy::too_many_arguments)]
fn pad_reference(
	input: &[f32],
	height: usize,
	width: usize,
	left: usize,
	right: usize,
	top: usize,
	bottom: usize,
	mode: BorderMode,
	constant: f32,
) -> Vec<f32> {
	let output_width = width + left + right;
	let output_height = height + top + bottom;
	let mut output = Vec::with_capacity(output_width * output_height);
	for y in 0..output_height {
		for x in 0..output_width {
			let source_y = border_coordinate(y as isize - top as isize, height, mode);
			let source_x = border_coordinate(x as isize - left as isize, width, mode);
			output.push(match (source_y, source_x) {
				(Some(source_y), Some(source_x)) => input[source_y * width + source_x],
				_ => constant,
			});
		}
	}
	output
}

fn border_coordinate(coordinate: isize, extent: usize, mode: BorderMode) -> Option<usize> {
	if (0..extent as isize).contains(&coordinate) {
		return Some(coordinate as usize);
	}
	match mode {
		BorderMode::Constant => None,
		BorderMode::Replicate => Some(coordinate.clamp(0, extent as isize - 1) as usize),
		BorderMode::Reflect if extent == 1 => Some(0),
		BorderMode::Reflect => {
			let period = 2 * extent as isize;
			let folded = coordinate.rem_euclid(period);
			Some(if folded < extent as isize {
				folded
			} else {
				period - 1 - folded
			} as usize)
		}
		BorderMode::Reflect101 if extent == 1 => Some(0),
		BorderMode::Reflect101 => {
			let period = 2 * (extent as isize - 1);
			let folded = coordinate.rem_euclid(period);
			Some(if folded < extent as isize {
				folded
			} else {
				period - folded
			} as usize)
		}
		BorderMode::Wrap => Some(coordinate.rem_euclid(extent as isize) as usize),
	}
}

test_vk!(
	resize_matches_independent_nearest_and_bilinear_oracles,
	engine,
	{
		let shape = [2_usize, 3, 3, 4];
		let values = (0..shape.iter().product::<usize>())
			.map(|index| ((index * 17 % 31) as f32 - 15.0) * 0.125)
			.collect::<Vec<_>>();
		let image = Image::new(
			Matrix::from_f32(&engine, shape, &values)?,
			ImageLayout::Nchw,
			ImageFormat::Rgb,
		)?;

		for (target_width, target_height) in [(7, 5), (2, 2), (4, 3)] {
			for mode in [InterpolationMode::Nearest, InterpolationMode::Bilinear] {
				let resized = oa::image::resize(&image, target_width, target_height, mode)?;
				assert_eq!(
					resized.as_matrix().shape(),
					[2, 3, target_height as usize, target_width as usize]
				);
				assert_eq!(resized.layout(), ImageLayout::Nchw);
				assert_eq!(resized.format(), ImageFormat::Rgb);
				assert_eq!(
					resized
						.as_matrix()
						.try_read_f32()
						.expect_err("recorded resize output was prematurely readable")
						.kind(),
					ErrorKind::NotReady
				);
				let expected = resize_reference(
					&values,
					6,
					3,
					4,
					target_height as usize,
					target_width as usize,
					mode,
				);
				for (index, (actual, expected)) in resized
					.as_matrix()
					.read_f32()?
					.iter()
					.zip(expected)
					.enumerate()
				{
					assert!(
						(actual - expected).abs() <= 1e-5,
						"{mode:?} element {index}: expected {expected}, found {actual}"
					);
				}
			}
		}
		assert_eq!(image.as_matrix().read_f32()?, values);
		Ok(())
	}
);

test_vk!(
	resize_preserves_chw_metadata_and_semantic_graph_identity,
	engine,
	{
		let values = (0..12).map(|value| value as f32 * 0.25).collect::<Vec<_>>();
		let image = Image::new(
			Matrix::from_f32(&engine, [1, 3, 4], &values)?,
			ImageLayout::Chw,
			ImageFormat::Gray,
		)?;
		let (plan, resized) =
			engine.capture(|| oa::image::resize(&image, 5, 7, InterpolationMode::Nearest))?;
		assert_eq!(resized.as_matrix().shape(), [1, 7, 5]);
		assert_eq!(resized.layout(), ImageLayout::Chw);
		assert_eq!(resized.format(), ImageFormat::Gray);

		let graph = plan.semantic_graph();
		assert_eq!(graph.operations().len(), 1);
		assert_eq!(graph.values().len(), 2);
		assert_eq!(graph.operations()[0].name(), "oa::image::resize");
		assert_eq!(
			graph.operations()[0].contract_hash(),
			oa::core::operation::image::RESIZE.hash()
		);
		assert!(
			graph
				.values()
				.iter()
				.all(|value| value.kind() == OpValueKind::Image)
		);
		assert_eq!(graph.operations()[0].attributes().len(), 3);
		assert!(graph.operations()[0].aliases().is_empty());
		assert_eq!(
			graph.operations()[0].attributes(),
			[
				OpAttribute::UnsignedInteger {
					name: "target_width".into(),
					value: 5,
				},
				OpAttribute::UnsignedInteger {
					name: "target_height".into(),
					value: 7,
				},
				OpAttribute::Enum {
					name: "interpolation".into(),
					value: "nearest".into(),
				},
			]
		);
		assert_eq!(plan.semantic_lowering().direct_op_count(), 1);
		engine.submit(&plan)?.wait()?;
		assert_eq!(resized.as_matrix().read_f32()?.len(), 35);
		Ok(())
	}
);

test_vk!(
	resize_rejects_unadmitted_layout_dtype_and_empty_extents,
	engine,
	{
		let chw_i32 = Image::new(
			Matrix::from_slice(&engine, [1, 2, 2], &[0_i32; 4])?,
			ImageLayout::Chw,
			ImageFormat::Gray,
		)?;
		let nhwc = Image::new(
			Matrix::from_f32(&engine, [1, 2, 2, 1], &[0.0; 4])?,
			ImageLayout::Nhwc,
			ImageFormat::Gray,
		)?;
		let empty = Image::new(
			Matrix::from_f32(&engine, [1, 1, 0, 2], &[])?,
			ImageLayout::Nchw,
			ImageFormat::Gray,
		)?;
		let valid = Image::new(
			Matrix::from_f32(&engine, [1, 2, 2], &[0.0; 4])?,
			ImageLayout::Chw,
			ImageFormat::Gray,
		)?;

		for error in [
			oa::image::resize(&chw_i32, 2, 2, InterpolationMode::Nearest)
				.err()
				.expect("integer input was accepted"),
			oa::image::resize(&nhwc, 2, 2, InterpolationMode::Nearest)
				.err()
				.expect("NHWC input was accepted"),
			oa::image::resize(&empty, 2, 2, InterpolationMode::Nearest)
				.err()
				.expect("empty input was accepted"),
			oa::image::resize(&valid, 0, 2, InterpolationMode::Nearest)
				.err()
				.expect("zero target width was accepted"),
			oa::image::resize(&valid, 2, 0, InterpolationMode::Bilinear)
				.err()
				.expect("zero target height was accepted"),
		] {
			assert_eq!(error.kind(), ErrorKind::InvalidArgument);
		}
		Ok(())
	}
);

fn resize_reference(
	input: &[f32],
	planes: usize,
	input_height: usize,
	input_width: usize,
	output_height: usize,
	output_width: usize,
	mode: InterpolationMode,
) -> Vec<f32> {
	let mut output = vec![0.0; planes * output_height * output_width];
	for plane in 0..planes {
		for output_y in 0..output_height {
			for output_x in 0..output_width {
				let value = match mode {
					InterpolationMode::Nearest => {
						let input_x = output_x * input_width / output_width;
						let input_y = output_y * input_height / output_height;
						input[(plane * input_height + input_y) * input_width + input_x]
					}
					InterpolationMode::Bilinear => bilinear_reference(
						input,
						plane,
						input_height,
						input_width,
						output_y,
						output_x,
						output_height,
						output_width,
					),
				};
				output[(plane * output_height + output_y) * output_width + output_x] = value;
			}
		}
	}
	output
}

#[allow(clippy::too_many_arguments)]
fn bilinear_reference(
	input: &[f32],
	plane: usize,
	input_height: usize,
	input_width: usize,
	output_y: usize,
	output_x: usize,
	output_height: usize,
	output_width: usize,
) -> f32 {
	let input_x = (((output_x as f32 + 0.5) * input_width as f32 / output_width as f32) - 0.5)
		.clamp(0.0, (input_width - 1) as f32);
	let input_y = (((output_y as f32 + 0.5) * input_height as f32 / output_height as f32) - 0.5)
		.clamp(0.0, (input_height - 1) as f32);
	let x0 = (input_x.floor() as usize).min(input_width - 1);
	let y0 = (input_y.floor() as usize).min(input_height - 1);
	let x1 = (x0 + 1).min(input_width - 1);
	let y1 = (y0 + 1).min(input_height - 1);
	let x_weight = input_x - x0 as f32;
	let y_weight = input_y - y0 as f32;
	let offset = plane * input_height * input_width;
	let top_left = input[offset + y0 * input_width + x0];
	let top_right = input[offset + y0 * input_width + x1];
	let bottom_left = input[offset + y1 * input_width + x0];
	let bottom_right = input[offset + y1 * input_width + x1];
	let top = top_left + (top_right - top_left) * x_weight;
	let bottom = bottom_left + (bottom_right - bottom_left) * x_weight;
	top + (bottom - top) * y_weight
}
