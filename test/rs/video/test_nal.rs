use oa::{
	ErrorKind,
	video::{
		NalUnit, emit_nal_annex_b, extract_pps, extract_pps_h265, extract_sps, extract_sps_h265,
		extract_vps_h265, parse_nal_annex_b,
	},
};

#[test]
fn h264_annex_b_split_emit_and_parameter_extraction_match_donor() -> oa::Result<()> {
	let sps = [0x67, 0x42, 0xc0, 0x1e];
	let pps = [0x68, 0xce, 0x38, 0x80];
	let stream = [
		0xaa, 0x00, 0x00, 0x00, 0x01, 0x67, 0x42, 0xc0, 0x1e, 0x00, 0x00, 0x01, 0x68, 0xce, 0x38, 0x80,
	];
	let units = parse_nal_annex_b(&stream);
	assert_eq!(units.len(), 2);
	assert_eq!(units[0].nal_unit_type(), 7);
	assert_eq!(units[0].reference_idc(), 3);
	assert_eq!(units[0].payload(), sps);
	assert_eq!(units[0].start_code_offset(), Some(1));
	assert_eq!(units[0].start_code_len(), 4);
	assert_eq!(units[1].nal_unit_type(), 8);
	assert_eq!(units[1].payload(), pps);
	assert_eq!(units[1].start_code_offset(), Some(9));
	assert_eq!(units[1].start_code_len(), 3);
	assert_eq!(extract_sps(&stream), Some(sps.to_vec()));
	assert_eq!(extract_pps(&stream), Some(pps.to_vec()));

	let emitted = emit_nal_annex_b(&units)?;
	assert_eq!(
		emitted,
		[
			0, 0, 0, 1, 0x67, 0x42, 0xc0, 0x1e, 0, 0, 0, 1, 0x68, 0xce, 0x38, 0x80,
		]
	);
	Ok(())
}

#[test]
fn h265_parameter_extraction_uses_two_byte_header_types() {
	let vps = [0x40, 0x01, 0xaa];
	let sps = [0x42, 0x01, 0xbb];
	let pps = [0x44, 0x01, 0xcc];
	let stream = [
		0, 0, 0, 1, 0x40, 0x01, 0xaa, 0, 0, 1, 0x42, 0x01, 0xbb, 0, 0, 0, 1, 0x44, 0x01, 0xcc, 0, 0,
	];
	assert_eq!(extract_vps_h265(&stream), Some(vps.to_vec()));
	assert_eq!(extract_sps_h265(&stream), Some(sps.to_vec()));
	assert_eq!(extract_pps_h265(&stream), Some(pps.to_vec()));
}

#[test]
fn nal_boundaries_are_borrowed_and_malformed_inputs_fail_closed() {
	assert!(parse_nal_annex_b(&[]).is_empty());
	assert!(parse_nal_annex_b(b"not annex-b").is_empty());
	assert!(parse_nal_annex_b(&[0, 0, 1]).is_empty());
	assert_eq!(extract_sps(&[0, 0, 1, 0x68]), None);
	assert_eq!(extract_vps_h265(&[0, 0, 1]), None);

	let error = NalUnit::from_h264_payload(&[]).expect_err("empty payload must fail");
	assert_eq!(error.kind(), ErrorKind::InvalidArgument);

	let stream = [0, 0, 1, 0x65, 0xaa];
	let unit = parse_nal_annex_b(&stream)[0];
	assert_eq!(unit.payload().as_ptr(), stream[3..].as_ptr());
	assert_eq!(unit.start_code_offset(), Some(0));
	assert_eq!(unit.start_code_len(), 3);
}
