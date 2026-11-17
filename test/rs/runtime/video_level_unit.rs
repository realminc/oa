use super::*;

#[test]
fn h264_capability_levels_are_bitstream_values_not_enum_ordinals() {
	for (native, expected) in [
		(vk::StdVideoH264LevelIdc::_1_0, 10),
		(vk::StdVideoH264LevelIdc::_3_1, 31),
		(vk::StdVideoH264LevelIdc::_4_2, 42),
		(vk::StdVideoH264LevelIdc::_6_2, 62),
	] {
		assert_eq!(h264_level_idc(native).unwrap(), expected);
		assert_eq!(std_h264_level(expected).unwrap(), native);
	}
	assert!(h264_level_idc(vk::StdVideoH264LevelIdc::INVALID).is_err());
	assert!(h264_level_idc(vk::StdVideoH264LevelIdc(-1)).is_err());
	assert!(std_h264_level(99).is_err());
}
