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
	assert_eq!(sps.video_parameter_set_id, vps.id);
	assert_eq!(sps.chroma_format_idc, 1);
	assert_eq!(sps.bit_depth_luma_minus_8, 0);
	assert_eq!(sps.bit_depth_chroma_minus_8, 0);
	assert_eq!((sps.width, sps.height), (1280, 720));
	assert!(!sps.scaling_list_enabled);
	assert_eq!(sps.scaling_lists, None);
	assert!(sps.short_term_reference_picture_sets.is_empty());
	assert!(sps.long_term_reference_pictures.is_empty());
	assert_eq!(pps.scaling_lists, None);
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
