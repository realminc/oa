use std::path::PathBuf;

use oa::{
	ErrorKind,
	video::{
		H264SliceType, VideoDemuxer, parse_h264_pps, parse_h264_slice_header, parse_h264_sps,
		parse_nal_annex_b,
	},
};

#[test]
fn h264_parameter_parsers_reject_wrong_or_truncated_nals() {
	assert_eq!(
		parse_h264_sps(&[])
			.expect_err("empty SPS was accepted")
			.kind(),
		ErrorKind::DataLoss
	);
	assert_eq!(
		parse_h264_sps(&[0x68])
			.expect_err("PPS header was accepted as SPS")
			.kind(),
		ErrorKind::InvalidArgument
	);
	assert_eq!(
		parse_h264_pps(
			&[0x68],
			&parse_h264_sps(&[0x67, 0x42, 0x00, 0x1e, 0xf4, 0x05, 0x01, 0x7f, 0xca, 0x80])
				.expect("test SPS must parse"),
		)
		.expect_err("truncated PPS was accepted")
		.kind(),
		ErrorKind::DataLoss
	);
}

#[test]
fn h264_slice_parser_rejects_non_slice_nals() {
	let sps = parse_h264_sps(&[0x67, 0x42, 0x00, 0x1e, 0xf4, 0x05, 0x01, 0x7f, 0xca, 0x80])
		.expect("test SPS must parse");
	let pps = parse_h264_pps(&[0x68, 0xce, 0x3c, 0x80], &sps).expect("test PPS must parse");
	assert_eq!(
		parse_h264_slice_header(&[0x67], &sps, &pps)
			.expect_err("SPS was accepted as a slice")
			.kind(),
		ErrorKind::InvalidArgument
	);
}

#[test]
#[ignore = "requires OA donor H.264 fixture"]
fn h264_parameter_sets_match_donor_stream_geometry() -> oa::Result<()> {
	let mut demuxer = VideoDemuxer::open(fixture("shibuya_720p_30fps_h264_high_8bit_420.mp4"))?;
	let packet = demuxer
		.read_next_packet()?
		.expect("H.264 fixture must contain a first packet");
	let nals = parse_nal_annex_b(packet.data());
	let sps_nal = nals
		.iter()
		.find(|nal| nal.payload()[0] & 0x1f == 7)
		.expect("demuxed keyframe must include SPS");
	let pps_nal = nals
		.iter()
		.find(|nal| nal.payload()[0] & 0x1f == 8)
		.expect("demuxed keyframe must include PPS");
	let sps = parse_h264_sps(sps_nal.payload())?;
	let pps = parse_h264_pps(pps_nal.payload(), &sps)?;
	assert_eq!(sps.profile_idc, 100);
	assert_eq!(sps.chroma_format_idc, 1);
	assert_eq!(sps.bit_depth_luma_minus_8, 0);
	assert_eq!(sps.bit_depth_chroma_minus_8, 0);
	assert_eq!(sps.scaling_lists, None);
	let vui = sps.vui.as_ref().expect("donor SPS must retain VUI");
	let aspect = vui.aspect_ratio.expect("aspect ratio");
	assert_eq!((aspect.idc, aspect.sar_width, aspect.sar_height), (1, 0, 0));
	assert_eq!(vui.overscan_appropriate, None);
	let signal = vui.video_signal.expect("video signal");
	assert_eq!(signal.video_format, 5);
	assert!(!signal.full_range);
	let colour = signal.colour_description.expect("colour description");
	assert_eq!(
		(
			colour.colour_primaries,
			colour.transfer_characteristics,
			colour.matrix_coefficients,
		),
		(2, 2, 5)
	);
	assert_eq!(vui.chroma_location, None);
	let timing = vui.timing.expect("timing");
	assert_eq!((timing.num_units_in_tick, timing.time_scale), (1, 60));
	assert!(!timing.fixed_frame_rate);
	assert_eq!(vui.nal_hrd, None);
	assert_eq!(vui.vcl_hrd, None);
	assert_eq!(vui.low_delay_hrd, None);
	assert!(!vui.picture_structure_present);
	let restriction = vui.bitstream_restriction.expect("bitstream restriction");
	assert!(restriction.motion_vectors_over_picture_boundaries);
	assert_eq!(restriction.max_num_reorder_frames, 2);
	assert_eq!(restriction.max_dec_frame_buffering, 4);
	assert!(sps.frame_mbs_only);
	assert_eq!(sps.coded_width()?, 1280);
	assert_eq!(sps.coded_height()?, 720);
	assert_eq!(pps.sequence_parameter_set_id, sps.id);
	assert_eq!(pps.scaling_lists, None);
	let slice_nal = nals
		.iter()
		.find(|nal| matches!(nal.payload()[0] & 0x1f, 1 | 5))
		.expect("demuxed keyframe must include a coded slice");
	let slice = parse_h264_slice_header(slice_nal.payload(), &sps, &pps)?;
	assert_eq!(slice.first_macroblock_in_slice, 0);
	assert_eq!(slice.slice_type, H264SliceType::I);
	assert_eq!(slice.picture_parameter_set_id, pps.id);
	assert_eq!(slice.frame_number, 0);
	assert_eq!(slice.picture_order_count_lsb, Some(0));
	assert!(slice.is_idr);
	assert!(slice.is_reference);
	Ok(())
}

fn fixture(name: &str) -> PathBuf {
	PathBuf::from(env!("CARGO_MANIFEST_DIR"))
		.with_file_name("oa")
		.join("sdk/asset/video/clip")
		.join(name)
}
