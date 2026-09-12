use std::{
	any::TypeId,
	fs,
	path::PathBuf,
	process::Command,
	time::{SystemTime, UNIX_EPOCH},
};

use oa::{
	ErrorKind,
	audio::EncodedAudioPacket,
	video::{
		EncodedVideoPacket, VideoCodec, VideoDemuxer, VideoMuxer, VideoMuxerAudioConfig,
		VideoMuxerConfig, emit_nal_annex_b, extract_pps, extract_pps_h265, extract_sps,
		extract_sps_h265, extract_vps_h265, parse_nal_annex_b,
	},
};

#[test]
fn root_muxer_export_is_the_video_session_identity() {
	assert_eq!(
		TypeId::of::<oa::VideoMuxer>(),
		TypeId::of::<oa::video::VideoMuxer>()
	);
}

#[test]
fn h264_muxer_streams_and_finalizes_a_demuxable_mp4() -> oa::Result<()> {
	let path = temporary_mp4("h264_roundtrip");
	let config = VideoMuxerConfig::new(VideoCodec::H264, 320, 240)?;
	let mut muxer = VideoMuxer::create(&path, config)?;
	let sps = [0x67, 0x42, 0x00, 0x1e, 0xf4, 0x05, 0x01, 0x7f, 0xca, 0x80];
	let pps = [0x68, 0xce, 0x38, 0x80];
	muxer.set_h264_codec_config(&sps, &pps)?;
	muxer.write_packet(&EncodedVideoPacket::new(
		vec![0, 0, 0, 1, 0x65, 0x80],
		0,
		true,
	)?)?;
	muxer.write_packet(&EncodedVideoPacket::new(
		vec![0, 0, 1, 0x41, 0x80],
		33_333,
		false,
	)?)?;
	assert_eq!(muxer.packet_count(), 2);
	muxer.finalize()?;
	assert!(muxer.is_finalized());
	assert_eq!(
		muxer
			.finalize()
			.expect_err("a finalized muxer finalized twice")
			.kind(),
		ErrorKind::FailedPrecondition
	);

	let mut demuxer = VideoDemuxer::open(&path)?;
	let info = demuxer.info();
	assert_eq!(info.codec(), VideoCodec::H264);
	assert_eq!((info.width(), info.height()), (320, 240));
	assert_eq!(info.sample_count(), 2);
	let first = demuxer
		.read_next_packet()?
		.expect("muxed file must retain its first packet");
	assert!(first.is_keyframe());
	assert_eq!(extract_sps(first.data()).as_deref(), Some(sps.as_slice()));
	assert_eq!(extract_pps(first.data()).as_deref(), Some(pps.as_slice()));
	assert!(
		parse_nal_annex_b(first.data())
			.iter()
			.any(|unit| unit.nal_unit_type() == 5)
	);
	drop(demuxer);
	fs::remove_file(path).expect("test MP4 cleanup failed");
	Ok(())
}

#[test]
fn close_abandons_without_implicit_finalization() -> oa::Result<()> {
	let path = temporary_mp4("abandon");
	let mut muxer = VideoMuxer::create(&path, VideoMuxerConfig::new(VideoCodec::H264, 16, 16)?)?;
	muxer.close();
	muxer.close();
	let error = match VideoDemuxer::open(&path) {
		Ok(_) => panic!("an abandoned stream was presented as a complete MP4"),
		Err(error) => error,
	};
	assert_eq!(error.kind(), ErrorKind::DataLoss);
	fs::remove_file(path).expect("test MP4 cleanup failed");
	Ok(())
}

#[test]
fn optional_pcm_packets_create_a_second_track() -> oa::Result<()> {
	let path = temporary_mp4("pcm_track");
	let mut config = VideoMuxerConfig::new(VideoCodec::H264, 16, 16)?;
	config.audio = Some(VideoMuxerAudioConfig {
		sample_rate: 48_000,
		channel_count: 2,
		priming_frames: 1,
	});
	let mut muxer = VideoMuxer::create(&path, config)?;
	muxer.set_h264_codec_config(
		&[0x67, 0x42, 0x00, 0x1e, 0xf4, 0x05, 0x01, 0x7f, 0xca, 0x80],
		&[0x68, 0xce, 0x38, 0x80],
	)?;
	muxer.write_packet(&EncodedVideoPacket::new(
		vec![0, 0, 1, 0x65, 0x80],
		0,
		true,
	)?)?;
	muxer.write_audio_packet(&EncodedAudioPacket {
		bitstream: vec![0, 0, 1, 0, 2, 0, 3, 0],
		presentation_frame: 0,
		duration_frames: 2,
	})?;
	muxer.finalize()?;
	assert_eq!(VideoDemuxer::open(&path)?.info().track_count(), 2);
	fs::remove_file(path).expect("test MP4 cleanup failed");
	Ok(())
}

#[test]
fn signed_composition_offsets_preserve_decode_and_display_order() -> oa::Result<()> {
	let path = temporary_mp4("composition_order");
	let mut muxer = VideoMuxer::create(&path, VideoMuxerConfig::new(VideoCodec::H264, 16, 16)?)?;
	muxer.set_h264_codec_config(
		&[0x67, 0x42, 0x00, 0x1e, 0xf4, 0x05, 0x01, 0x7f, 0xca, 0x80],
		&[0x68, 0xce, 0x38, 0x80],
	)?;
	for (presentation, decode, keyframe) in [
		(0, 0, true),
		(99_999, 33_333, false),
		(33_333, 66_666, false),
		(66_666, 99_999, false),
	] {
		muxer.write_packet(&EncodedVideoPacket::with_timestamps(
			vec![0, 0, 1, if keyframe { 0x65 } else { 0x41 }, 0x80],
			presentation,
			decode,
			keyframe,
		)?)?;
	}
	muxer.finalize()?;
	let mut demuxer = VideoDemuxer::open(&path)?;
	let mut timing = Vec::new();
	while let Some(packet) = demuxer.read_next_packet()? {
		timing.push((packet.presentation_timestamp(), packet.decode_timestamp()));
	}
	assert_eq!(
		timing,
		[(0, 0), (9_000, 3_000), (3_000, 6_000), (6_000, 9_000)]
	);
	fs::remove_file(path).expect("test MP4 cleanup failed");
	Ok(())
}

#[test]
#[ignore = "requires OA donor video fixtures and FFmpeg"]
fn muxes_h264_and_h265_donor_packets() -> oa::Result<()> {
	for (name, codec) in [
		(
			"shibuya_720p_30fps_h264_high_8bit_420.mp4",
			VideoCodec::H264,
		),
		(
			"shibuya_720p_30fps_h265_main_8bit_420.mp4",
			VideoCodec::H265,
		),
	] {
		let mut source = VideoDemuxer::open(fixture(name))?;
		let info = source.info();
		let first = source
			.read_next_packet()?
			.expect("donor fixture must have a packet");
		let path = temporary_mp4(match codec {
			VideoCodec::H264 => "h264_donor",
			VideoCodec::H265 => "h265_donor",
			_ => unreachable!("test only covers AVC and HEVC"),
		});
		let mut muxer = VideoMuxer::create(
			&path,
			VideoMuxerConfig::new(codec, info.width(), info.height())?,
		)?;
		match codec {
			VideoCodec::H264 => muxer.set_h264_codec_config(
				&extract_sps(first.data()).expect("donor AVC packet must contain SPS"),
				&extract_pps(first.data()).expect("donor AVC packet must contain PPS"),
			)?,
			VideoCodec::H265 => muxer.set_h265_codec_config(
				&extract_vps_h265(first.data()).expect("donor HEVC packet must contain VPS"),
				&extract_sps_h265(first.data()).expect("donor HEVC packet must contain SPS"),
				&extract_pps_h265(first.data()).expect("donor HEVC packet must contain PPS"),
			)?,
			_ => unreachable!("test only covers AVC and HEVC"),
		}
		let mut packets = vec![first];
		for _ in 1..8 {
			let Some(packet) = source.read_next_packet()? else {
				break;
			};
			packets.push(packet);
		}
		for packet in &packets {
			let bitstream = strip_parameter_sets(packet.data(), codec)?;
			let time_base = info.time_base();
			muxer.write_packet(&EncodedVideoPacket::with_timestamps(
				bitstream,
				to_micros(packet.presentation_timestamp(), time_base),
				to_micros(packet.decode_timestamp(), time_base),
				packet.is_keyframe(),
			)?)?;
		}
		muxer.finalize()?;
		let output = VideoDemuxer::open(&path)?;
		assert_eq!(output.info().codec(), codec);
		assert_eq!(
			output.info().sample_count(),
			u32::try_from(packets.len()).expect("small test packet count")
		);
		drop(output);
		let status = Command::new("ffmpeg")
			.args(["-v", "error", "-i"])
			.arg(&path)
			.args(["-frames:v", "8", "-f", "null", "-"])
			.status()
			.expect("FFmpeg must be installed for this ignored qualification");
		assert!(
			status.success(),
			"FFmpeg rejected the muxed {codec:?} stream"
		);
		fs::remove_file(path).expect("test MP4 cleanup failed");
	}
	Ok(())
}

fn strip_parameter_sets(bytes: &[u8], codec: VideoCodec) -> oa::Result<Vec<u8>> {
	let units = parse_nal_annex_b(bytes);
	let retained = units
		.into_iter()
		.filter(|unit| match codec {
			VideoCodec::H264 => !matches!(unit.nal_unit_type(), 7 | 8),
			VideoCodec::H265 => !matches!((unit.payload()[0] >> 1) & 0x3f, 32..=34),
			_ => false,
		})
		.collect::<Vec<_>>();
	emit_nal_annex_b(&retained)
}

fn temporary_mp4(label: &str) -> PathBuf {
	let nonce = SystemTime::now()
		.duration_since(UNIX_EPOCH)
		.expect("system clock precedes Unix epoch")
		.as_nanos();
	std::env::temp_dir().join(format!("oars_{label}_{}_{nonce}.mp4", std::process::id()))
}

fn fixture(name: &str) -> PathBuf {
	PathBuf::from(env!("CARGO_MANIFEST_DIR"))
		.with_file_name("oa")
		.join("sdk/asset/video/clip")
		.join(name)
}

fn to_micros(timestamp: u64, time_base: oa::video::VideoTimeBase) -> u64 {
	timestamp
		.checked_mul(u64::from(time_base.numerator()))
		.and_then(|value| value.checked_mul(1_000_000))
		.expect("donor timestamp must fit microseconds")
		/ u64::from(time_base.denominator())
}
