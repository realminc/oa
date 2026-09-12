use std::any::TypeId;

use oa::{Engine, VideoDecoder, VideoDemuxer, video::VideoPixelFormat};

fn ffmpeg_yuv420(path: &std::path::Path) -> Vec<u8> {
	let output = std::process::Command::new("ffmpeg")
		.args(["-v", "error", "-i"])
		.arg(path)
		.args([
			"-map", "0:v:0", "-pix_fmt", "yuv420p", "-f", "rawvideo", "-",
		])
		.output()
		.expect("FFmpeg must be available for this ignored oracle test");
	assert!(
		output.status.success(),
		"FFmpeg oracle failed: {}",
		String::from_utf8_lossy(&output.stderr)
	);
	output.stdout
}

fn yuv420_frame_size(width: u32, height: u32) -> usize {
	usize::try_from(width)
		.ok()
		.and_then(|width| {
			usize::try_from(height)
				.ok()
				.and_then(|height| width.checked_mul(height))
		})
		.and_then(|luma| luma.checked_mul(3))
		.and_then(|samples| samples.checked_div(2))
		.expect("fixture geometry must fit usize")
}

#[test]
fn root_decoder_export_is_the_video_session_identity() {
	assert_eq!(
		TypeId::of::<oa::VideoDecoder>(),
		TypeId::of::<oa::video::VideoDecoder>()
	);
}

#[test]
#[ignore = "requires Vulkan H.264 hardware and the OA donor fixture"]
fn native_decoder_retains_planes_until_consumer_completion() -> oa::Result<()> {
	let engine = Engine::new()?;
	let fixture = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
		.with_file_name("oa")
		.join("sdk/asset/video/clip/shibuya_720p_30fps_h264_high_8bit_420.mp4");
	let mut demuxer = VideoDemuxer::open(&fixture)?;
	let info = demuxer.info();
	let mut decoder = VideoDecoder::create(&engine, info)?;
	let mut foreign_decoder = VideoDecoder::create(&engine, info)?;
	let mut decoded = 0_usize;
	let mut decoded_frames = Vec::new();
	let before_decode = engine.checkpoint()?;
	let mut checked_event_order = false;

	while let Some(packet) = demuxer.read_next_packet()? {
		let Some(frame) = decoder.decode_native(&packet)? else {
			continue;
		};
		assert_eq!(frame.width(), info.width() as usize);
		assert_eq!(frame.height(), info.height() as usize);
		assert!(matches!(
			frame.native_format(),
			Some(VideoPixelFormat::Nv12 | VideoPixelFormat::Yuv420Planar8)
		));
		assert!(frame.as_image().is_none());
		assert!(frame.as_texture().is_none());
		assert!(frame.as_yuv420p().is_none());
		assert!(
			frame
				.ready_event()
				.expect("native frame carries decode readiness")
				.is_complete()?
		);
		let pixels = decoder.read_native_yuv420p(&frame)?;
		if !checked_event_order {
			assert!(foreign_decoder.read_native_yuv420p(&frame).is_err());
			assert_eq!(decoder.read_native_yuv420p(&frame)?, pixels);
			assert!(frame.mark_consumed(&before_decode).is_err());
			checked_event_order = true;
		}
		decoded_frames.push(pixels);
		let consumer = engine.checkpoint()?;
		frame.mark_consumed(&consumer)?;
		drop(frame);
		decoded += 1;
	}
	for frame in decoder.flush()? {
		decoded_frames.push(decoder.read_native_yuv420p(&frame)?);
		let consumer = engine.checkpoint()?;
		frame.mark_consumed(&consumer)?;
		drop(frame);
		decoded += 1;
	}
	assert_eq!(decoded, info.sample_count() as usize);
	assert!(checked_event_order);
	let frame_size = yuv420_frame_size(info.width(), info.height());
	let reference = ffmpeg_yuv420(&fixture);
	assert_eq!(reference.len(), frame_size * decoded_frames.len());
	for (actual, expected) in decoded_frames
		.iter()
		.zip(reference.chunks_exact(frame_size))
	{
		assert_eq!(actual, expected);
	}
	decoder.close();
	foreign_decoder.close();
	Ok(())
}

#[test]
#[ignore = "requires Vulkan H.265 hardware and the OA donor fixture"]
fn native_h265_decoder_retains_planes_without_host_materialization() -> oa::Result<()> {
	let engine = Engine::new()?;
	let fixture = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
		.with_file_name("oa")
		.join("sdk/asset/video/clip/shibuya_720p_30fps_h265_main_8bit_420.mp4");
	let mut demuxer = VideoDemuxer::open(&fixture)?;
	let info = demuxer.info();
	let mut decoder = VideoDecoder::create(&engine, info)?;
	let mut decoded = 0_usize;
	let mut decoded_frames = Vec::new();
	let mut retained = None;

	while let Some(packet) = demuxer.read_next_packet()? {
		let Some(frame) = decoder.decode_native(&packet)? else {
			continue;
		};
		assert_eq!(frame.width(), info.width() as usize);
		assert_eq!(frame.height(), info.height() as usize);
		assert!(matches!(
			frame.native_format(),
			Some(VideoPixelFormat::Nv12 | VideoPixelFormat::Yuv420Planar8)
		));
		assert!(frame.as_yuv420p().is_none());
		decoded_frames.push(decoder.read_native_yuv420p(&frame)?);
		drop(frame);
		decoded += 1;
	}
	for frame in decoder.flush()? {
		assert!(frame.native_format().is_some());
		decoded_frames.push(decoder.read_native_yuv420p(&frame)?);
		retained = Some(frame);
		decoded += 1;
	}
	assert_eq!(decoded, info.sample_count() as usize);
	let frame_size = yuv420_frame_size(info.width(), info.height());
	let reference = ffmpeg_yuv420(&fixture);
	assert_eq!(reference.len(), frame_size * decoded_frames.len());
	for (actual, expected) in decoded_frames
		.iter()
		.zip(reference.chunks_exact(frame_size))
	{
		assert_eq!(actual, expected);
	}
	decoder.close();
	drop(decoder);
	drop(engine);
	let retained = retained.expect("the reordered stream must retain a flush tail");
	assert_eq!(retained.width(), info.width() as usize);
	assert!(
		retained
			.ready_event()
			.expect("native readiness")
			.is_complete()?
	);
	Ok(())
}

#[test]
#[ignore = "requires Vulkan H.265 hardware, OA donor fixture, and FFmpeg"]
fn public_decoder_matches_complete_h265_ffmpeg_stream() -> oa::Result<()> {
	let engine = Engine::new()?;
	let fixture = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
		.with_file_name("oa")
		.join("sdk/asset/video/clip/shibuya_720p_30fps_h265_main_8bit_420.mp4");
	let mut demuxer = VideoDemuxer::open(&fixture)?;
	let info = demuxer.info();
	let mut decoder = VideoDecoder::create(&engine, info)?;
	let mut decoded_frames = Vec::new();
	while let Some(packet) = demuxer.read_next_packet()? {
		let Some(frame) = decoder.decode(&packet)? else {
			continue;
		};
		assert_eq!(frame.width(), info.width() as usize);
		assert_eq!(frame.height(), info.height() as usize);
		decoded_frames.push(frame);
	}
	decoded_frames.extend(decoder.flush()?);
	assert_eq!(decoded_frames.len(), info.sample_count() as usize);
	let reference = std::process::Command::new("ffmpeg")
		.args(["-v", "error", "-i"])
		.arg(&fixture)
		.args([
			"-map", "0:v:0", "-pix_fmt", "yuv420p", "-f", "rawvideo", "-",
		])
		.output()
		.expect("FFmpeg must be available for this ignored oracle test");
	assert!(
		reference.status.success(),
		"FFmpeg oracle failed: {}",
		String::from_utf8_lossy(&reference.stderr)
	);
	let frame_size = usize::try_from(info.width())
		.ok()
		.and_then(|width| {
			usize::try_from(info.height())
				.ok()
				.and_then(|height| width.checked_mul(height))
		})
		.and_then(|luma| luma.checked_mul(3))
		.and_then(|samples| samples.checked_div(2))
		.expect("fixture geometry must fit usize");
	let expected_size = frame_size
		.checked_mul(info.sample_count() as usize)
		.expect("fixture stream size must fit usize");
	assert_eq!(reference.stdout.len(), expected_size);
	for (frame, expected) in decoded_frames
		.iter()
		.zip(reference.stdout.chunks_exact(frame_size))
	{
		assert_eq!(
			frame
				.as_yuv420p()
				.expect("public hardware decode returns planar YUV420"),
			expected,
			"public Vulkan H.265 output differs from FFmpeg at {:?}",
			frame.timing().presentation_timestamp()
		);
	}

	assert!(decoder.flush()?.is_empty());
	demuxer.seek(0)?;
	let first_packet = demuxer
		.read_next_packet()?
		.expect("seek must restore the first random-access packet");
	assert!(decoder.decode(&first_packet)?.is_none());
	let mut first_after_flush = decoder.flush()?;
	assert_eq!(first_after_flush.len(), 1);
	let first_after_flush = first_after_flush.pop().expect("one flushed keyframe");
	assert_eq!(
		first_after_flush
			.as_yuv420p()
			.expect("flushed decode remains planar YUV420"),
		&reference.stdout[..frame_size]
	);
	decoder.close();
	assert!(!decoder.is_open());
	assert!(decoder.info().is_none());
	Ok(())
}

#[test]
#[ignore = "requires Vulkan AV1 hardware, OA donor fixture, and FFmpeg"]
fn public_decoder_matches_complete_av1_ffmpeg_stream() -> oa::Result<()> {
	let engine = Engine::new()?;
	let fixture = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
		.with_file_name("oa")
		.join("sdk/asset/video/clip/shibuya_720p_30fps_av1_main_8bit_420.mp4");
	let mut demuxer = VideoDemuxer::open(&fixture)?;
	let info = demuxer.info();
	let mut decoder = VideoDecoder::create(&engine, info)?;
	let mut decoded_frames = Vec::new();
	while let Some(packet) = demuxer.read_next_packet()? {
		if let Some(frame) = decoder.decode(&packet)? {
			decoded_frames.push(frame);
		}
	}
	decoded_frames.extend(decoder.flush()?);
	let reference = ffmpeg_yuv420(&fixture);
	let frame_size = yuv420_frame_size(info.width(), info.height());
	assert_eq!(reference.len(), frame_size * decoded_frames.len());
	for (frame, expected) in decoded_frames
		.iter()
		.zip(reference.chunks_exact(frame_size))
	{
		assert_eq!(frame.width(), info.width() as usize);
		assert_eq!(frame.height(), info.height() as usize);
		assert_eq!(
			frame
				.as_yuv420p()
				.expect("public AV1 decode returns planar YUV420"),
			expected,
			"public Vulkan AV1 output differs from FFmpeg at {:?}",
			frame.timing().presentation_timestamp()
		);
	}
	assert_eq!(decoded_frames.len(), info.sample_count() as usize);
	Ok(())
}

#[test]
#[ignore = "requires Vulkan AV1 hardware, OA donor fixture, and FFmpeg"]
fn native_av1_decoder_retains_and_reuses_reference_slots() -> oa::Result<()> {
	let engine = Engine::new()?;
	let fixture = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
		.with_file_name("oa")
		.join("sdk/asset/video/clip/shibuya_720p_30fps_av1_main_8bit_420.mp4");
	let mut demuxer = VideoDemuxer::open(&fixture)?;
	let info = demuxer.info();
	let mut decoder = VideoDecoder::create(&engine, info)?;
	let mut decoded_frames = Vec::new();
	let mut retained = None;
	while let Some(packet) = demuxer.read_next_packet()? {
		let Some(frame) = decoder.decode_native(&packet)? else {
			continue;
		};
		assert!(matches!(
			frame.native_format(),
			Some(VideoPixelFormat::Nv12 | VideoPixelFormat::Yuv420Planar8)
		));
		decoded_frames.push(decoder.read_native_yuv420p(&frame)?);
		if retained.is_none() {
			retained = Some(frame.clone());
		}
	}
	for frame in decoder.flush()? {
		decoded_frames.push(decoder.read_native_yuv420p(&frame)?);
	}
	let reference = ffmpeg_yuv420(&fixture);
	let frame_size = yuv420_frame_size(info.width(), info.height());
	assert_eq!(decoded_frames.len(), info.sample_count() as usize);
	assert_eq!(reference.len(), frame_size * decoded_frames.len());
	for (actual, expected) in decoded_frames
		.iter()
		.zip(reference.chunks_exact(frame_size))
	{
		assert_eq!(actual, expected);
	}
	let retained = retained.expect("AV1 stream must publish a native frame");
	assert!(
		retained
			.ready_event()
			.expect("native AV1 frame carries readiness")
			.is_complete()?
	);
	Ok(())
}

#[test]
#[ignore = "requires Vulkan VP9 hardware, OA donor fixture, and FFmpeg"]
fn public_decoder_matches_complete_vp9_ffmpeg_stream() -> oa::Result<()> {
	let engine = Engine::new()?;
	let fixture = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
		.with_file_name("oa")
		.join("sdk/asset/video/clip/shibuya_720p_30fps_vp9_profile0_8bit_420.mp4");
	let mut demuxer = VideoDemuxer::open(&fixture)?;
	let info = demuxer.info();
	let mut decoder = VideoDecoder::create(&engine, info)?;
	let mut decoded_frames = Vec::new();
	while let Some(packet) = demuxer.read_next_packet()? {
		if let Some(frame) = decoder.decode(&packet)? {
			decoded_frames.push(frame);
		}
	}
	decoded_frames.extend(decoder.flush()?);
	let reference = ffmpeg_yuv420(&fixture);
	let frame_size = yuv420_frame_size(info.width(), info.height());
	assert_eq!(decoded_frames.len(), info.sample_count() as usize);
	assert_eq!(reference.len(), frame_size * decoded_frames.len());
	for (frame, expected) in decoded_frames
		.iter()
		.zip(reference.chunks_exact(frame_size))
	{
		assert_eq!(
			frame
				.as_yuv420p()
				.expect("public VP9 decode returns planar YUV420"),
			expected,
			"public Vulkan VP9 output differs from FFmpeg at {:?}",
			frame.timing().presentation_timestamp()
		);
	}
	Ok(())
}

#[test]
#[ignore = "requires Vulkan VP9 hardware, OA donor fixture, and FFmpeg"]
fn native_vp9_decoder_retains_and_reuses_reference_slots() -> oa::Result<()> {
	let engine = Engine::new()?;
	let fixture = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
		.with_file_name("oa")
		.join("sdk/asset/video/clip/shibuya_720p_30fps_vp9_profile0_8bit_420.mp4");
	let mut demuxer = VideoDemuxer::open(&fixture)?;
	let info = demuxer.info();
	let mut decoder = VideoDecoder::create(&engine, info)?;
	let mut decoded_frames = Vec::new();
	let mut retained = None;
	while let Some(packet) = demuxer.read_next_packet()? {
		let Some(frame) = decoder.decode_native(&packet)? else {
			continue;
		};
		assert!(matches!(
			frame.native_format(),
			Some(VideoPixelFormat::Nv12 | VideoPixelFormat::Yuv420Planar8)
		));
		decoded_frames.push(decoder.read_native_yuv420p(&frame)?);
		if retained.is_none() {
			retained = Some(frame.clone());
		}
	}
	for frame in decoder.flush()? {
		decoded_frames.push(decoder.read_native_yuv420p(&frame)?);
	}
	let reference = ffmpeg_yuv420(&fixture);
	let frame_size = yuv420_frame_size(info.width(), info.height());
	assert_eq!(decoded_frames.len(), info.sample_count() as usize);
	assert_eq!(reference.len(), frame_size * decoded_frames.len());
	for (actual, expected) in decoded_frames
		.iter()
		.zip(reference.chunks_exact(frame_size))
	{
		assert_eq!(actual, expected);
	}
	let retained = retained.expect("VP9 stream must publish a native frame");
	assert!(
		retained
			.ready_event()
			.expect("native VP9 frame carries readiness")
			.is_complete()?
	);
	Ok(())
}

#[test]
#[ignore = "requires Vulkan H.264 hardware, OA donor fixture, and FFmpeg"]
fn public_decoder_matches_complete_h264_ffmpeg_stream() -> oa::Result<()> {
	let engine = Engine::new()?;
	let fixture = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
		.with_file_name("oa")
		.join("sdk/asset/video/clip/shibuya_720p_30fps_h264_high_8bit_420.mp4");
	let mut demuxer = VideoDemuxer::open(&fixture)?;
	let info = demuxer.info();
	let mut decoder = VideoDecoder::create(&engine, info)?;
	let mut decoded_frames = Vec::new();
	while let Some(packet) = demuxer.read_next_packet()? {
		if let Some(frame) = decoder.decode(&packet)? {
			assert_eq!(frame.width(), info.width() as usize);
			assert_eq!(frame.height(), info.height() as usize);
			decoded_frames.push(frame);
		}
	}
	decoded_frames.extend(decoder.flush()?);
	assert_eq!(decoded_frames.len(), info.sample_count() as usize);
	let reference = std::process::Command::new("ffmpeg")
		.args(["-v", "error", "-i"])
		.arg(&fixture)
		.args([
			"-map", "0:v:0", "-pix_fmt", "yuv420p", "-f", "rawvideo", "-",
		])
		.output()
		.expect("FFmpeg must be available for this ignored oracle test");
	assert!(
		reference.status.success(),
		"FFmpeg oracle failed: {}",
		String::from_utf8_lossy(&reference.stderr)
	);
	let frame_size = usize::try_from(info.width())
		.ok()
		.and_then(|width| {
			usize::try_from(info.height())
				.ok()
				.and_then(|height| width.checked_mul(height))
		})
		.and_then(|luma| luma.checked_mul(3))
		.and_then(|samples| samples.checked_div(2))
		.expect("fixture geometry must fit usize");
	assert_eq!(
		reference.stdout.len(),
		frame_size * info.sample_count() as usize
	);
	for (frame, expected) in decoded_frames
		.iter()
		.zip(reference.stdout.chunks_exact(frame_size))
	{
		assert_eq!(
			frame
				.as_yuv420p()
				.expect("public hardware decode returns planar YUV420"),
			expected,
			"public Vulkan H.264 output differs from FFmpeg at {:?}",
			frame.timing().presentation_timestamp()
		);
	}

	assert!(decoder.flush()?.is_empty());
	demuxer.seek(0)?;
	let first_packet = demuxer
		.read_next_packet()?
		.expect("seek must restore the first random-access packet");
	assert!(decoder.decode(&first_packet)?.is_none());
	let first_after_flush = decoder.flush()?.pop().expect("one flushed keyframe");
	assert_eq!(
		first_after_flush
			.as_yuv420p()
			.expect("flushed decode remains planar YUV420"),
		&reference.stdout[..frame_size]
	);
	decoder.close();
	assert!(!decoder.is_open());
	assert!(decoder.info().is_none());
	Ok(())
}
