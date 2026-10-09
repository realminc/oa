use oa::{DType, ErrorKind, Image, ImageFormat, ImageLayout, Matrix};

#[test]
fn image_metadata_has_exact_rank_and_channel_contracts() {
	assert_eq!(ImageLayout::Nchw.rank(), 4);
	assert_eq!(ImageLayout::Nhwc.rank(), 4);
	assert_eq!(ImageLayout::Chw.rank(), 3);
	assert_eq!(ImageLayout::Hwc.rank(), 3);
	assert_eq!(ImageLayout::Hw.rank(), 2);

	assert_eq!(ImageFormat::Gray.channels(), 1);
	assert_eq!(ImageFormat::GrayAlpha.channels(), 2);
	assert_eq!(ImageFormat::Rgb.channels(), 3);
	assert_eq!(ImageFormat::Rgba.channels(), 4);
	assert_eq!(ImageFormat::Bgr.channels(), 3);
	assert_eq!(ImageFormat::Bgra.channels(), 4);
}

test_vk!(
	image_layouts_preserve_dense_storage_and_semantics,
	engine,
	{
		let cases = [
			(
				vec![2, 3, 4, 5],
				ImageLayout::Nchw,
				ImageFormat::Rgb,
				(2, 3, 4, 5),
			),
			(
				vec![2, 4, 5, 4],
				ImageLayout::Nhwc,
				ImageFormat::Rgba,
				(2, 4, 4, 5),
			),
			(
				vec![3, 4, 5],
				ImageLayout::Chw,
				ImageFormat::Bgr,
				(1, 3, 4, 5),
			),
			(
				vec![4, 5, 2],
				ImageLayout::Hwc,
				ImageFormat::GrayAlpha,
				(1, 2, 4, 5),
			),
			(vec![4, 5], ImageLayout::Hw, ImageFormat::Gray, (1, 1, 4, 5)),
			(
				vec![1, 3, 0, 5],
				ImageLayout::Nchw,
				ImageFormat::Rgb,
				(1, 3, 0, 5),
			),
		];

		for (shape, layout, format, expected) in cases {
			let values = vec![0.25_f32; shape.iter().product()];
			let matrix = Matrix::from_slice(&engine, shape.clone(), &values)?;
			let image = Image::new(matrix, layout, format)?;
			assert_eq!(image.as_matrix().shape(), shape);
			assert_eq!(image.batch_size(), expected.0);
			assert_eq!(image.channels(), expected.1);
			assert_eq!(image.height(), expected.2);
			assert_eq!(image.width(), expected.3);
			assert_eq!(image.layout(), layout);
			assert_eq!(image.format(), format);
			assert_eq!(image.dtype(), DType::F32);
			assert_eq!(image.clone().into_matrix().read_f32()?, values);
		}
		Ok(())
	}
);

test_vk!(image_rejects_inconsistent_metadata, engine, {
	let rank_three = Matrix::from_slice(&engine, [3, 4, 5], &[0_u32; 60])?;
	let error = Image::new(rank_three, ImageLayout::Nchw, ImageFormat::Rgb)
		.err()
		.expect("rank mismatch must fail");
	assert_eq!(error.kind(), ErrorKind::InvalidArgument);

	let four_channels = Matrix::from_slice(&engine, [4, 4, 5], &[0_u32; 80])?;
	let error = Image::new(four_channels, ImageLayout::Chw, ImageFormat::Rgb)
		.err()
		.expect("channel mismatch must fail");
	assert_eq!(error.kind(), ErrorKind::InvalidArgument);

	let implicit_gray = Matrix::from_slice(&engine, [4, 5], &[0_u32; 20])?;
	let error = Image::new(implicit_gray, ImageLayout::Hw, ImageFormat::Rgba)
		.err()
		.expect("Hw must reject non-gray formats");
	assert_eq!(error.kind(), ErrorKind::InvalidArgument);
	Ok(())
});
