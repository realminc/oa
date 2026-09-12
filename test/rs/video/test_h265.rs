use std::path::PathBuf;

use oa::{
	ErrorKind,
	video::{
		H265SliceType, VideoDemuxer, parse_h265_pps, parse_h265_slice_header, parse_h265_sps,
		parse_h265_vps, parse_nal_annex_b,
	},
};

#[test]
fn h265_parameter_parsers_reject_wrong_or_truncated_nals() {
	assert_eq!(
		parse_h265_vps(&[])
			.expect_err("empty VPS was accepted")
			.kind(),
		ErrorKind::DataLoss
	);
	assert_eq!(
		parse_h265_sps(&[0x40, 1])
			.expect_err("VPS header was accepted as SPS")
			.kind(),
		ErrorKind::InvalidArgument
	);
	assert_eq!(
		parse_h265_pps(&[0x44, 1])
			.expect_err("truncated PPS was accepted")
			.kind(),
		ErrorKind::DataLoss
	);
}

#[test]
#[ignore = "requires OA donor H.265 fixture"]
fn h265_parameter_sets_match_donor_stream_geometry() -> oa::Result<()> {
	let mut demuxer = VideoDemuxer::open(fixture("shibuya_720p_30fps_h265_main_8bit_420.mp4"))?;
	let packet = demuxer
		.read_next_packet()?
		.expect("H.265 fixture must contain a first packet");
	let nals = parse_nal_annex_b(packet.data());
	let find = |nal_type| {
		nals.iter()
			.find(|nal| (nal.payload()[0] >> 1) & 0x3f == nal_type)
			.map(|nal| nal.payload())
			.expect("demuxed keyframe must include the parameter set")
	};
	let vps = parse_h265_vps(find(32))?;
	let sps = parse_h265_sps(find(33))?;
	let pps = parse_h265_pps(find(34))?;
	assert_eq!(vps.profile_tier_level.profile_idc, 1);
	assert_eq!(vps.profile_tier_level.level_idc, 93);
	assert_eq!(
		vps.decoded_picture_buffer
			.max_decoded_picture_buffering_minus_1[0],
		4
	);
	assert_eq!(vps.decoded_picture_buffer.max_num_reorder_pictures[0], 2);
	assert_eq!(vps.decoded_picture_buffer.max_latency_increase_plus_1[0], 5);
	assert!(vps.timing.is_none());
	assert!(vps.hrd_parameters.is_empty());
	assert_eq!(sps.video_parameter_set_id, vps.id);
	assert_eq!(sps.profile_tier_level, vps.profile_tier_level);
	assert_eq!(sps.chroma_format_idc, 1);
	assert_eq!(sps.bit_depth_luma_minus_8, 0);
	assert_eq!(sps.bit_depth_chroma_minus_8, 0);
	assert_eq!((sps.width, sps.height), (1280, 720));
	assert!(!sps.scaling_list_enabled);
	assert_eq!(sps.scaling_lists, None);
	assert!(sps.short_term_reference_picture_sets.is_empty());
	assert!(sps.long_term_reference_pictures.is_empty());
	assert!(!sps.pcm_enabled);
	assert!(sps.pcm.is_none());
	let vui = sps.vui.as_ref().expect("donor SPS must retain VUI");
	assert_eq!(vui.aspect_ratio.expect("donor aspect ratio").idc, 1);
	let signal = vui.video_signal.expect("donor video signal");
	assert_eq!(signal.video_format, 5);
	assert!(!signal.full_range);
	let colour = signal.colour_description.expect("donor colour description");
	assert_eq!(colour.colour_primaries, 2);
	assert_eq!(colour.transfer_characteristics, 2);
	assert_eq!(colour.matrix_coefficients, 5);
	let timing = vui.timing.expect("donor VUI timing");
	assert_eq!((timing.num_units_in_tick, timing.time_scale), (1, 30));
	assert_eq!(timing.num_ticks_poc_diff_one_minus_1, None);
	assert!(vui.hrd.is_none());
	assert_eq!(pps.scaling_lists, None);
	assert!(pps.column_width_minus_1.is_empty());
	assert!(pps.row_height_minus_1.is_empty());
	assert_eq!(pps.sequence_parameter_set_id, sps.id);
	let slice_nal = nals
		.iter()
		.find(|nal| ((nal.payload()[0] >> 1) & 0x3f) < 32)
		.expect("demuxed keyframe must include a coded slice");
	let slice = parse_h265_slice_header(slice_nal.payload(), &sps, &pps)?;
	assert_eq!(slice.picture_parameter_set_id, pps.id);
	assert_eq!(slice.sequence_parameter_set_id, sps.id);
	assert_eq!(slice.slice_type, H265SliceType::I);
	assert_eq!(slice.slice_segment_address, 0);
	assert_eq!(slice.picture_order_count_lsb, None);
	assert!(slice.first_slice_segment_in_picture);
	assert!(slice.is_irap);
	assert!(slice.is_idr);
	assert!(slice.is_reference);
	let second_packet = demuxer
		.read_next_packet()?
		.expect("H.265 fixture must contain a second packet");
	let second_nals = parse_nal_annex_b(second_packet.data());
	let second_slice_nal = second_nals
		.iter()
		.find(|nal| ((nal.payload()[0] >> 1) & 0x3f) < 32)
		.expect("second donor packet must include a coded slice");
	let second_slice = parse_h265_slice_header(second_slice_nal.payload(), &sps, &pps)?;
	assert_eq!(second_slice.slice_type, H265SliceType::P);
	assert_eq!(second_slice.picture_order_count_lsb, Some(5));
	assert!(!second_slice.short_term_reference_picture_set_sps);
	assert_eq!(second_slice.short_term_current_before_delta_pocs, [-5]);
	assert!(second_slice.short_term_current_after_delta_pocs.is_empty());
	assert!(second_slice.short_term_following_delta_pocs.is_empty());
	let mut wrong_pps = pps.clone();
	wrong_pps.id += 1;
	assert_eq!(
		parse_h265_slice_header(slice_nal.payload(), &sps, &wrong_pps)
			.expect_err("slice accepted the wrong PPS")
			.kind(),
		ErrorKind::InvalidArgument
	);
	Ok(())
}

fn fixture(name: &str) -> PathBuf {
	PathBuf::from(env!("CARGO_MANIFEST_DIR"))
		.with_file_name("oa")
		.join("sdk/asset/video/clip")
		.join(name)
}
