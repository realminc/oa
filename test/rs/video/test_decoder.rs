use std::any::TypeId;

use oa::{
	Runtime, VideoDecoder, VideoDemuxer,
	video::{
		Av1ObuType, Av1ReferenceState, VideoPixelFormat, parse_av1_frame_header, parse_av1_obus,
		parse_av1_sequence_header,
	},
};

fn ffmpeg_yuv420(path: &std::path::Path) -> Vec<u8> {
	ffmpeg_yuv(path, "yuv420p")
}

fn ffmpeg_yuv420p10(path: &std::path::Path) -> Vec<u8> {
	ffmpeg_yuv(path, "yuv420p10le")
}

fn ffmpeg_yuv(path: &std::path::Path, pixel_format: &str) -> Vec<u8> {
	let output = std::process::Command::new("ffmpeg")
		.args(["-v", "error", "-i"])
		.arg(path)
		.args([
			"-map",
			"0:v:0",
			"-pix_fmt",
			pixel_format,
			"-f",
			"rawvideo",
			"-",
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

fn assert_av1_pixels(actual: &[u8], expected: &[u8], frame_index: usize) {
	assert_eq!(
		actual.len(),
		expected.len(),
		"AV1 frame {frame_index} extent"
	);
	if let Some(index) = actual.iter().zip(expected).position(|(a, b)| a != b) {
		let mismatches = actual.iter().zip(expected).filter(|(a, b)| a != b).count();
		panic!(
			"AV1 frame {frame_index}: {mismatches}/{} differing bytes; first at {index}: actual={} expected={}",
			actual.len(),
			actual[index],
			expected[index]
		);
	}
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
	let runtime = Runtime::new()?;
	let fixture = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
		.with_file_name("oa")
		.join("sdk/asset/video/clip/shibuya_720p_30fps_h264_high_8bit_420.mp4");
	let mut demuxer = VideoDemuxer::open(&fixture)?;
	let info = demuxer.info();
	let mut decoder = VideoDecoder::new(&runtime, info)?;
	let mut foreign_decoder = VideoDecoder::new(&runtime, info)?;
	let mut decoded = 0_usize;
	let mut decoded_frames = Vec::new();
	let before_decode = runtime.checkpoint()?;
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
		let consumer = runtime.checkpoint()?;
		frame.mark_consumed(&consumer)?;
		drop(frame);
		decoded += 1;
	}
	for frame in decoder.flush()? {
		decoded_frames.push(decoder.read_native_yuv420p(&frame)?);
		let consumer = runtime.checkpoint()?;
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
	let runtime = Runtime::new()?;
	let fixture = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
		.with_file_name("oa")
		.join("sdk/asset/video/clip/shibuya_720p_30fps_h265_main_8bit_420.mp4");
	let mut demuxer = VideoDemuxer::open(&fixture)?;
	let info = demuxer.info();
	let mut decoder = VideoDecoder::new(&runtime, info)?;
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
	drop(runtime);
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
	let runtime = Runtime::new()?;
	let fixture = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
		.with_file_name("oa")
		.join("sdk/asset/video/clip/shibuya_720p_30fps_h265_main_8bit_420.mp4");
	let mut demuxer = VideoDemuxer::open(&fixture)?;
	let info = demuxer.info();
	let mut decoder = VideoDecoder::new(&runtime, info)?;
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
	let runtime = Runtime::new()?;
	let fixture = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
		.with_file_name("oa")
		.join("sdk/asset/video/clip/shibuya_720p_30fps_av1_main_8bit_420.mp4");
	let mut demuxer = VideoDemuxer::open(&fixture)?;
	let info = demuxer.info();
	let mut decoder = VideoDecoder::new(&runtime, info)?;
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
	for (index, (frame, expected)) in decoded_frames
		.iter()
		.zip(reference.chunks_exact(frame_size))
		.enumerate()
	{
		assert_eq!(frame.width(), info.width() as usize);
		assert_eq!(frame.height(), info.height() as usize);
		assert_av1_pixels(
			frame
				.as_yuv420p()
				.expect("public AV1 decode returns planar YUV420"),
			expected,
			index,
		);
	}
	assert_eq!(decoded_frames.len(), info.sample_count() as usize);
	Ok(())
}

#[test]
#[ignore = "probe: does a first AV1 pass warm the AVP runtime for the second?"]
fn av1_second_pass_probe() -> oa::Result<()> {
	let runtime = Runtime::new()?;
	let fixture = std::env::var_os("OA_AV1_FIXTURE")
		.map(std::path::PathBuf::from)
		.unwrap_or_else(|| {
			std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
				.with_file_name("oa")
				.join("sdk/asset/video/clip/shibuya_720p_30fps_av1_main_8bit_420.mp4")
		});
	let reference = ffmpeg_yuv420(&fixture);
	for pass in 0..2 {
		let mut demuxer = VideoDemuxer::open(&fixture)?;
		let info = demuxer.info();
		if pass == 0 {
			let first = demuxer.read_next_packet()?.expect("first packet");
			let mut probe_sequence = None;
			for obu in parse_av1_obus(first.data()).unwrap_or_default() {
				eprintln!(
					"obu {:?} offset={} size={}",
					obu.type_(),
					obu.header_offset(),
					obu.payload().len()
				);
				if obu.type_() == Av1ObuType::SequenceHeader
					&& let Ok(seq) = parse_av1_sequence_header(obu.payload())
				{
					eprintln!("seq: {seq:?}");
					probe_sequence = Some(seq);
				}
				if (obu.type_() == Av1ObuType::Frame || obu.type_() == Av1ObuType::FrameHeader)
					&& let Some(seq) = &probe_sequence
					&& let Ok(header) =
						parse_av1_frame_header(obu.payload(), seq, &Av1ReferenceState::default())
				{
					eprintln!("frame-header: {header:?}");
				}
			}
		}
		demuxer.seek(0)?;
		let mut decoder = VideoDecoder::new(&runtime, info)?;
		let mut decoded_frames: Vec<Vec<u8>> = Vec::new();
		if std::env::var_os("OA_AV1_NATIVE").is_some() {
			while let Some(packet) = demuxer.read_next_packet()? {
				if let Some(frame) = decoder.decode_native(&packet)? {
					decoded_frames.push(decoder.read_native_yuv420p(&frame)?);
				}
			}
			for frame in decoder.flush()? {
				decoded_frames.push(decoder.read_native_yuv420p(&frame)?);
			}
		} else {
			while let Some(packet) = demuxer.read_next_packet()? {
				if let Some(frame) = decoder.decode(&packet)? {
					decoded_frames.push(frame.as_yuv420p().expect("planar").to_vec());
				}
			}
			for frame in decoder.flush()? {
				decoded_frames.push(frame.as_yuv420p().expect("planar").to_vec());
			}
		}
		let frame_size = yuv420_frame_size(info.width(), info.height());
		let (frame_size, reference) = if decoded_frames
			.first()
			.is_some_and(|frame| frame.len() == frame_size * 2)
		{
			let reference10 = ffmpeg_yuv420p10(&fixture);
			(frame_size * 2, reference10)
		} else {
			(frame_size, reference.clone())
		};
		eprintln!(
			"pass {pass}: decoded {} frames, oracle {} frames, samples {}",
			decoded_frames.len(),
			reference.len() / frame_size,
			info.sample_count()
		);
		for (index, (frame, expected)) in decoded_frames
			.iter()
			.zip(reference.chunks_exact(frame_size))
			.enumerate()
		{
			let actual = frame.as_slice();
			if pass == 0 && index == 0 {
				let unique = actual
					.iter()
					.collect::<std::collections::BTreeSet<_>>()
					.len();
				eprintln!(
					"frame0: {unique} unique byte values; actual[..16]={:?} expected[..16]={:?}",
					&actual[..16.min(actual.len())],
					&expected[..16.min(expected.len())]
				);
				if let Ok(path) = std::env::var("OA_AV1_DUMP") {
					std::fs::write(path, actual).ok();
				}
			}
			let differing = actual
				.iter()
				.zip(expected.iter())
				.filter(|(a, b)| a != b)
				.count();
			if differing > 0 {
				eprintln!(
					"pass {pass} frame {index}: {differing}/{} differing",
					expected.len()
				);
			}
		}
		decoder.close();
	}
	Ok(())
}

#[test]
#[ignore = "requires four Vulkan decode codecs and FFmpeg; OA_VIDEO_FIXTURE optionally selects one clip"]
fn video_fixture_probe() -> oa::Result<()> {
	let runtime = Runtime::new()?;
	let fixtures = if let Some(path) = std::env::var_os("OA_VIDEO_FIXTURE") {
		vec![std::path::PathBuf::from(path)]
	} else {
		let directory = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("sdk/asset/video/clip");
		let mut paths = std::fs::read_dir(directory)
			.expect("video fixtures")
			.map(|entry| entry.expect("fixture entry").path())
			.filter(|path| path.extension().is_some_and(|ext| ext == "mp4"))
			.collect::<Vec<_>>();
		paths.sort();
		assert_eq!(
			paths.len(),
			9,
			"every qualified decode profile must be probed"
		);
		paths
	};
	for fixture in fixtures {
		assert_video_fixture(&runtime, &fixture)?;
	}
	Ok(())
}

fn assert_video_fixture(runtime: &Runtime, fixture: &std::path::Path) -> oa::Result<()> {
	let reference = ffmpeg_yuv420(fixture);
	for pass in 0..2 {
		let mut demuxer = VideoDemuxer::open(fixture)?;
		let info = demuxer.info();
		demuxer.seek(0)?;
		let mut decoder = VideoDecoder::new(runtime, info)?;
		let mut decoded_frames: Vec<Vec<u8>> = Vec::new();
		let mut decode_error = None;
		while let Some(packet) = demuxer.read_next_packet()? {
			match decoder.decode(&packet) {
				Ok(Some(frame)) => {
					let bytes = frame
						.as_yuv420p()
						.map(|s| s.to_vec())
						.or_else(|| {
							frame
								.as_yuv420p10()
								.map(|s| s.iter().flat_map(|v| v.to_le_bytes()).collect())
						})
						.expect("planar");
					decoded_frames.push(bytes);
				}
				Ok(None) => {}
				Err(error) => {
					decode_error = Some(error);
					break;
				}
			}
		}
		if decode_error.is_none() {
			match decoder.flush() {
				Ok(frames) => {
					for frame in frames {
						let bytes = frame
							.as_yuv420p()
							.map(|s| s.to_vec())
							.or_else(|| {
								frame
									.as_yuv420p10()
									.map(|s| s.iter().flat_map(|v| v.to_le_bytes()).collect())
							})
							.expect("planar");
						decoded_frames.push(bytes);
					}
				}
				Err(error) => decode_error = Some(error),
			}
		}
		if let Some(error) = decode_error {
			decoder.close();
			return Err(error);
		}
		let frame_size = yuv420_frame_size(info.width(), info.height());
		let (frame_size, reference) = if decoded_frames
			.first()
			.is_some_and(|frame| frame.len() == frame_size * 2)
		{
			let reference10 = ffmpeg_yuv420p10(fixture);
			(frame_size * 2, reference10)
		} else {
			(frame_size, reference.clone())
		};
		assert_eq!(reference.len() % frame_size, 0, "oracle frame alignment");
		assert_eq!(
			decoded_frames.len(),
			reference.len() / frame_size,
			"{} pass {pass}: decoded frame count",
			fixture.display()
		);
		assert_eq!(
			decoded_frames.len(),
			info.sample_count() as usize,
			"container sample count"
		);
		eprintln!(
			"pass {pass}: decoded {} frames, oracle {} frames, samples {}",
			decoded_frames.len(),
			reference.len() / frame_size,
			info.sample_count()
		);
		for (index, (frame, expected)) in decoded_frames
			.iter()
			.zip(reference.chunks_exact(frame_size))
			.enumerate()
		{
			let differing = frame
				.iter()
				.zip(expected.iter())
				.filter(|(a, b)| a != b)
				.count();
			assert_eq!(
				differing,
				0,
				"{} pass {pass} frame {index}: {differing}/{} differing bytes",
				fixture.display(),
				expected.len()
			);
		}
		decoder.close();
	}
	Ok(())
}

#[test]
#[ignore = "requires Vulkan AV1 hardware, OA donor fixture, and FFmpeg"]
fn native_av1_decoder_retains_and_reuses_reference_slots() -> oa::Result<()> {
	let runtime = Runtime::new()?;
	let fixture = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
		.with_file_name("oa")
		.join("sdk/asset/video/clip/shibuya_720p_30fps_av1_main_8bit_420.mp4");
	let mut demuxer = VideoDemuxer::open(&fixture)?;
	let info = demuxer.info();
	let mut decoder = VideoDecoder::new(&runtime, info)?;
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
	for (index, (actual, expected)) in decoded_frames
		.iter()
		.zip(reference.chunks_exact(frame_size))
		.enumerate()
	{
		assert_av1_pixels(actual, expected, index);
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
	let runtime = Runtime::new()?;
	let fixture = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
		.with_file_name("oa")
		.join("sdk/asset/video/clip/shibuya_720p_30fps_vp9_profile0_8bit_420.mp4");
	let mut demuxer = VideoDemuxer::open(&fixture)?;
	let info = demuxer.info();
	let mut decoder = VideoDecoder::new(&runtime, info)?;
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
	let runtime = Runtime::new()?;
	let fixture = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
		.with_file_name("oa")
		.join("sdk/asset/video/clip/shibuya_720p_30fps_vp9_profile0_8bit_420.mp4");
	let mut demuxer = VideoDemuxer::open(&fixture)?;
	let info = demuxer.info();
	let mut decoder = VideoDecoder::new(&runtime, info)?;
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
	let runtime = Runtime::new()?;
	let fixture = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
		.with_file_name("oa")
		.join("sdk/asset/video/clip/shibuya_720p_30fps_h264_high_8bit_420.mp4");
	let mut demuxer = VideoDemuxer::open(&fixture)?;
	let info = demuxer.info();
	let mut decoder = VideoDecoder::new(&runtime, info)?;
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

#[test]
#[ignore = "requires Vulkan H.264 hardware and the OA donor fixture"]
fn decoder_rejects_work_inside_capture_and_can_retry_the_same_packet() -> oa::Result<()> {
	let runtime = Runtime::new()?;
	let fixture = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
		.join("sdk/asset/video/clip/shibuya_720p_30fps_h264_baseline_8bit_420.mp4");
	let mut demuxer = VideoDemuxer::open(fixture)?;
	let info = demuxer.info();
	let mut decoder = VideoDecoder::new(&runtime, info)?;
	let first = demuxer.read_next_packet()?.expect("first access unit");

	let error = runtime
		.capture(|| decoder.decode_native(&first))
		.err()
		.expect("decode inside capture must fail");
	assert_eq!(error.kind(), oa::ErrorKind::FailedPrecondition);
	assert!(error.to_string().contains("outside capture"));

	let mut frame = decoder.decode_native(&first)?;
	while frame.is_none() {
		let packet = demuxer
			.read_next_packet()?
			.expect("stream produces a frame");
		frame = decoder.decode_native(&packet)?;
	}
	let frame = frame.expect("native output");
	let error = runtime
		.capture(|| decoder.read_native_yuv420p(&frame))
		.unwrap_err();
	assert_eq!(error.kind(), oa::ErrorKind::FailedPrecondition);
	assert!(error.to_string().contains("outside capture"));
	assert_eq!(
		decoder.read_native_yuv420p(&frame)?.len(),
		yuv420_frame_size(info.width(), info.height())
	);
	Ok(())
}

#[test]
#[ignore = "requires Vulkan H.264 hardware, the OA donor fixture, and FFmpeg"]
fn retained_native_planes_replay_across_queue_families() -> oa::Result<()> {
	retained_native_planes_replay("h264_baseline_8bit", "yuv420p")
}

#[test]
#[ignore = "requires Vulkan Main10/Profile2 hardware, the OA donor fixtures, and FFmpeg"]
fn retained_p010_planes_replay_across_queue_families() -> oa::Result<()> {
	for profile in ["h265_main10_10bit", "vp9_profile2_10bit", "av1_main_10bit"] {
		retained_native_planes_replay(profile, "p010le")?;
	}
	Ok(())
}

fn retained_native_planes_replay(profile: &str, pixel_format: &str) -> oa::Result<()> {
	use oa::video::VideoMatrixConfig;

	let runtime = Runtime::new()?;
	let fixture = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(format!(
		"sdk/asset/video/clip/shibuya_720p_30fps_{profile}_420.mp4"
	));
	let mut demuxer = VideoDemuxer::open(&fixture)?;
	let info = demuxer.info();
	let mut decoder = VideoDecoder::new(&runtime, info)?;
	let frame = loop {
		let packet = demuxer
			.read_next_packet()?
			.expect("stream produces a frame");
		if let Some(frame) = decoder.decode_native(&packet)? {
			break frame;
		}
	};
	let config = VideoMatrixConfig {
		size: Some((37, 21)),
		..VideoMatrixConfig::default()
	};
	let independent = ffmpeg_yuv(&fixture, pixel_format);
	let frame_size = yuv420_frame_size(info.width(), info.height());
	let reference_frame = if pixel_format == "p010le" {
		assert_eq!(
			frame.native_format(),
			Some(VideoPixelFormat::P010),
			"{profile}"
		);
		let luma_bytes = frame.width() * frame.height() * 2;
		oa::VideoFrame::upload_planes(
			VideoPixelFormat::P010,
			vec![
				independent[..luma_bytes].to_vec(),
				independent[luma_bytes..2 * frame_size].to_vec(),
			],
			frame.width(),
			frame.height(),
			frame.timing(),
			frame.color_info(),
		)?
	} else {
		oa::VideoFrame::new_from_yuv420p(
			independent[..frame_size].to_vec(),
			frame.width(),
			frame.height(),
			frame.timing(),
			frame.color_info(),
		)?
		.upload()?
	};
	let reference = reference_frame.to_matrix(&config)?.read_f32()?;

	let eager = decoder.stage_native_planes(&frame)?;
	assert_eq!(
		eager.to_matrix(&config)?.read_f32()?,
		reference,
		"{profile}"
	);

	let (plan, staged) =
		runtime.capture(|| decoder.stage_native_planes(&frame)?.to_matrix(&config))?;
	let report: serde_json::Value = serde_json::from_str(&plan.debug_report_json("native planes"))
		.expect("valid executable graph report");
	let transfers = plan.diagnostics().ownership_transfer_count();
	assert!(matches!(transfers, 0 | 2));
	assert_eq!(plan.diagnostics().queue_run_count(), transfers + 1);
	assert_eq!(plan.diagnostics().cross_queue_edge_count(), transfers);
	assert_eq!(report["stats"]["video_commands"], transfers + 1);
	assert_eq!(report["stats"]["image_transfers"], transfers);
	assert_eq!(report["stats"]["dispatches"], 1);
	plan.validate_serialized_descriptor(&plan.serialize_descriptor()?)?;

	decoder.close();
	drop(frame);
	let first = runtime.submit(&plan)?;
	let second = runtime.submit(&plan)?;
	second.wait()?;
	assert!(first.is_complete()?);
	assert_eq!(staged.read_f32()?, reference, "{profile}");
	assert_eq!(plan.diagnostics().command_cache_hit_count(), 1);
	Ok(())
}

#[test]
#[ignore = "requires Vulkan Video hardware, OA fixtures, and FFmpeg"]
fn recorded_avc_hevc_av1_packets_replay_through_compute() -> oa::Result<()> {
	let runtime = Runtime::new()?;
	for profile in [
		"h264_baseline_8bit",
		"h264_main_8bit",
		"h264_high_8bit",
		"h265_main_8bit",
		"h265_main10_10bit",
		"av1_main_8bit",
		"av1_main_10bit",
	] {
		recorded_video_replay(&runtime, profile).unwrap_or_else(|error| panic!("{profile}: {error:?}"));
	}
	Ok(())
}

#[test]
#[ignore = "requires VP9 Vulkan Video capture support, OA fixtures, and FFmpeg"]
fn recorded_vp9_packets_replay_through_compute() -> oa::Result<()> {
	let runtime = Runtime::new()?;
	for profile in ["vp9_profile0_8bit", "vp9_profile2_10bit"] {
		recorded_video_replay(&runtime, profile).unwrap_or_else(|error| panic!("{profile}: {error:?}"));
	}
	Ok(())
}

#[test]
#[ignore = "requires an Intel ANV driver older than Mesa 26.3"]
fn recorded_vp9_rejects_host_probability_updates() -> oa::Result<()> {
	let runtime = Runtime::new()?;
	let info = runtime.device_info();
	// The rejection only applies to ANV builds older than Mesa 26.3, where
	// the driver still rewrites mapped VP9 probability tables on the host
	// during command recording. Other drivers, and ANV 26.3+ with the
	// GPU-ordered update, skip this gate entirely.
	let affected = info.driver_name().contains("Intel open-source Mesa")
		&& info.driver_version() < (26 << 22) | (3 << 12);
	if !affected {
		return Ok(());
	}
	let fixture = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
		.join("sdk/asset/video/clip/shibuya_720p_30fps_vp9_profile0_8bit_420.mp4");
	let mut demux = VideoDemuxer::open(&fixture)?;
	let mut decoder = VideoDecoder::new(&runtime, demux.info())?;
	let packet = demux.read_next_packet()?.expect("fixture key picture");
	let capture = runtime.capture(|| decoder.record_to_planes(std::slice::from_ref(&packet)));
	assert_eq!(
		capture
			.err()
			.expect("affected driver must reject capture")
			.kind(),
		oa::ErrorKind::MissingCapability
	);
	assert!(decoder.decode_native(&packet)?.is_some());
	Ok(())
}

fn recorded_video_replay(runtime: &Runtime, profile: &str) -> oa::Result<()> {
	let fixture = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(format!(
		"sdk/asset/video/clip/shibuya_720p_30fps_{profile}_420.mp4"
	));
	let mut demux = VideoDemuxer::open(&fixture)?;
	let info = demux.info();
	let mut decoder = VideoDecoder::new(runtime, info)?;
	let mut packets = Vec::new();
	while let Some(packet) = demux.read_next_packet()? {
		packets.push(packet);
	}
	assert_eq!(packets.len(), 60, "{profile}");
	assert_eq!(
		decoder.record_to_planes(&packets).err().unwrap().kind(),
		oa::ErrorKind::FailedPrecondition,
	);
	let config = oa::video::VideoMatrixConfig {
		size: Some((37, 21)),
		..Default::default()
	};
	let (plan, outputs) = runtime.capture(|| {
		assert_eq!(
			decoder
				.record_to_planes(&packets[1..])
				.err()
				.unwrap()
				.kind(),
			oa::ErrorKind::FailedPrecondition,
		);
		let frames = decoder.record_to_planes(&packets)?;
		frames
			.into_iter()
			.map(|frame| {
				Ok((
					frame.timing(),
					frame.color_info(),
					frame.to_matrix(&config)?,
				))
			})
			.collect::<oa::Result<Vec<_>>>()
	})?;
	assert_eq!(outputs.len(), packets.len(), "{profile}");
	let mut timestamps: Vec<_> = packets
		.iter()
		.map(|packet| packet.presentation_timestamp())
		.collect();
	timestamps.sort_unstable();
	for ((timing, _, _), pts) in outputs.iter().zip(timestamps) {
		let timebase = info.time_base();
		let nanos = u128::from(pts) * u128::from(timebase.numerator()) * 1_000_000_000
			/ u128::from(timebase.denominator());
		assert_eq!(timing.presentation_timestamp().as_nanos(), nanos);
	}
	assert_eq!(
		outputs[0].2.try_read_f32().unwrap_err().kind(),
		oa::ErrorKind::NotReady
	);
	let report: serde_json::Value = serde_json::from_str(&plan.debug_report_json(profile)).unwrap();
	let picture_commands = report["nodes"]
		.as_array()
		.unwrap()
		.iter()
		.filter(|node| node["kind"] == "video_decode")
		.count();
	assert!(picture_commands > 0, "{profile}");
	if profile.starts_with("h26") {
		assert_eq!(picture_commands, packets.len(), "{profile}");
	}
	assert_eq!(report["stats"]["dispatches"], packets.len());
	plan.validate_serialized_descriptor(&plan.serialize_descriptor()?)?;

	// A separate live decode still starts at the first access unit.
	decoder.decode_native(&packets[0])?;
	decoder.close();
	drop(packets);

	let first = runtime.submit(&plan)?;
	let second = runtime.submit(&plan)?;
	second.wait()?;
	assert!(first.is_complete()?);
	first.wait()?;
	assert_eq!(plan.diagnostics().command_cache_hit_count(), 0);
	assert_eq!(plan.diagnostics().command_recording_count(), 2);

	let luma_bytes = info.width() as usize * info.height() as usize;
	let frame_size = yuv420_frame_size(info.width(), info.height());
	let (format, independent) = if profile.ends_with("_10bit") {
		(VideoPixelFormat::P010, ffmpeg_yuv(&fixture, "p010le"))
	} else {
		(VideoPixelFormat::Yuv420Planar8, ffmpeg_yuv420(&fixture))
	};
	let frame_size = if format == VideoPixelFormat::P010 {
		frame_size * 2
	} else {
		frame_size
	};
	assert_eq!(independent.len(), frame_size * outputs.len());
	for ((timing, color, output), bytes) in outputs
		.into_iter()
		.zip(independent.chunks_exact(frame_size))
	{
		let planes = if format == VideoPixelFormat::P010 {
			vec![
				bytes[..2 * luma_bytes].to_vec(),
				bytes[2 * luma_bytes..].to_vec(),
			]
		} else {
			let uv = bytes[luma_bytes..].split_at(luma_bytes / 4);
			vec![bytes[..luma_bytes].to_vec(), uv.0.to_vec(), uv.1.to_vec()]
		};
		let reference = oa::VideoFrame::upload_planes(
			format,
			planes,
			info.width() as usize,
			info.height() as usize,
			timing,
			color,
		)?;
		let actual = output.read_f32()?;
		let expected = reference.to_matrix(&config)?.read_f32()?;
		assert_eq!(actual.len(), expected.len());
		if let Some(index) = actual.iter().zip(&expected).position(|(a, b)| a != b) {
			panic!(
				"{profile} at {timing:?}, sample {index}: {} != {}",
				actual[index], expected[index]
			);
		}
	}
	Ok(())
}
