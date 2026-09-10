use std::{any::TypeId, time::Duration};

use oa::{
	ErrorKind, Image, ImageFormat, ImageLayout, Matrix, VideoFrame,
	video::{VideoColorInfo, VideoColorMatrix, VideoColorRange, VideoFrameTiming},
};

#[test]
fn root_frame_export_is_the_video_contract_identity() {
	assert_eq!(
		TypeId::of::<oa::VideoFrame>(),
		TypeId::of::<oa::video::VideoFrame>()
	);
}

#[test]
fn timing_and_color_metadata_are_explicit() -> oa::Result<()> {
	let timing = VideoFrameTiming::new(
		Duration::from_micros(33_366),
		Some(Duration::from_micros(16_683)),
	)?;
	assert_eq!(
		timing.presentation_timestamp(),
		Duration::from_micros(33_366)
	);
	assert_eq!(timing.duration(), Some(Duration::from_micros(16_683)));

	let unknown_duration = VideoFrameTiming::from_microseconds(100, 0);
	assert_eq!(
		unknown_duration.presentation_timestamp(),
		Duration::from_micros(100)
	);
	assert_eq!(unknown_duration.duration(), None);

	let error = VideoFrameTiming::new(Duration::ZERO, Some(Duration::ZERO))
		.expect_err("a known zero duration must fail");
	assert_eq!(error.kind(), ErrorKind::InvalidArgument);

	let color = VideoColorInfo::new(VideoColorMatrix::Bt709, VideoColorRange::Limited);
	assert_eq!(color.matrix(), VideoColorMatrix::Bt709);
	assert_eq!(color.range(), VideoColorRange::Limited);
	assert_eq!(
		VideoColorInfo::unspecified().matrix(),
		VideoColorMatrix::Unspecified
	);
	Ok(())
}

test_vk!(
	packed_video_frame_retains_image_storage_and_metadata,
	engine,
	{
		let matrix = Matrix::from_f32(&engine, [2, 4, 4], &[0.25; 32])?;
		let image = Image::new(matrix, ImageLayout::Hwc, ImageFormat::Rgba)?;
		let timing = VideoFrameTiming::from_microseconds(1_000_000, 33_333);
		let color = VideoColorInfo::new(VideoColorMatrix::Bt709, VideoColorRange::Full);
		let frame = VideoFrame::from_image(image, timing, color)?;

		assert_eq!(frame.width(), 4);
		assert_eq!(frame.height(), 2);
		assert_eq!(frame.timing(), timing);
		assert_eq!(frame.color_info(), color);
		let retained = frame.as_image().expect("packed image backing");
		assert_eq!(retained.layout(), ImageLayout::Hwc);
		assert_eq!(retained.format(), ImageFormat::Rgba);
		assert_eq!(retained.as_matrix().read_f32()?, vec![0.25; 32]);
		assert_eq!(
			frame
				.clone()
				.into_image()
				.expect("packed image backing")
				.as_matrix()
				.shape(),
			[2, 4, 4]
		);
		Ok(())
	}
);

test_vk!(video_frame_rejects_batches_and_empty_extents, engine, {
	let batch = Matrix::from_f32(&engine, [2, 3, 1, 1], &[0.0; 6])?;
	let batch = Image::new(batch, ImageLayout::Nchw, ImageFormat::Rgb)?;
	let Err(error) = VideoFrame::from_image(
		batch,
		VideoFrameTiming::from_microseconds(0, 1),
		VideoColorInfo::unspecified(),
	) else {
		panic!("batched frame must fail");
	};
	assert_eq!(error.kind(), ErrorKind::InvalidArgument);

	let empty = Matrix::from_f32(&engine, [0, 1, 3], &[])?;
	let empty = Image::new(empty, ImageLayout::Hwc, ImageFormat::Rgb)?;
	let Err(error) = VideoFrame::from_image(
		empty,
		VideoFrameTiming::from_microseconds(0, 1),
		VideoColorInfo::unspecified(),
	) else {
		panic!("empty frame must fail");
	};
	assert_eq!(error.kind(), ErrorKind::InvalidArgument);
	Ok(())
});
