// SPDX-License-Identifier: MIT
// Ash VP9 layout oracles retained from the ABI donor; see NOTICE.md.
#![cfg(target_pointer_width = "64")]

use super::native::*;
use super::*;
use core::mem::{offset_of, size_of};

#[test]
fn vp9_extension_layout_and_structure_tags() {
	assert_eq!(size_of::<VideoDecodeVP9ProfileInfoKHR<'_>>(), 24);
	assert_eq!(size_of::<VideoDecodeVP9CapabilitiesKHR<'_>>(), 24);
	assert_eq!(size_of::<VideoDecodeVP9PictureInfoKHR<'_>>(), 48);
	assert_eq!(
		offset_of!(VideoDecodeVP9PictureInfoKHR<'_>, p_std_picture_info),
		16
	);
	assert_eq!(
		offset_of!(
			VideoDecodeVP9PictureInfoKHR<'_>,
			reference_name_slot_indices
		),
		24
	);
	assert_eq!(
		offset_of!(VideoDecodeVP9PictureInfoKHR<'_>, tiles_offset),
		44
	);
	assert_eq!(
		VideoDecodeVP9ProfileInfoKHR::default().s_type.as_raw(),
		1_000_514_003
	);
	assert_eq!(
		VideoDecodeVP9CapabilitiesKHR::default().s_type.as_raw(),
		1_000_514_001
	);
	assert_eq!(
		VideoDecodeVP9PictureInfoKHR::default().s_type.as_raw(),
		1_000_514_002
	);
}

#[test]
fn bindgen_test_layout_StdVideoVP9ColorConfigFlags() {
	assert_eq!(
		::core::mem::size_of::<StdVideoVP9ColorConfigFlags>(),
		4usize,
		concat!("Size of: ", stringify!(StdVideoVP9ColorConfigFlags))
	);
	assert_eq!(
		::core::mem::align_of::<StdVideoVP9ColorConfigFlags>(),
		4usize,
		concat!("Alignment of ", stringify!(StdVideoVP9ColorConfigFlags))
	);
}

#[test]
fn bindgen_test_layout_StdVideoVP9ColorConfig() {
	const UNINIT: ::core::mem::MaybeUninit<StdVideoVP9ColorConfig> =
		::core::mem::MaybeUninit::uninit();
	let ptr = UNINIT.as_ptr();
	assert_eq!(
		::core::mem::size_of::<StdVideoVP9ColorConfig>(),
		12usize,
		concat!("Size of: ", stringify!(StdVideoVP9ColorConfig))
	);
	assert_eq!(
		::core::mem::align_of::<StdVideoVP9ColorConfig>(),
		4usize,
		concat!("Alignment of ", stringify!(StdVideoVP9ColorConfig))
	);
	assert_eq!(
		unsafe { ::core::ptr::addr_of!((*ptr).flags) as usize - ptr as usize },
		0usize,
		concat!(
			"Offset of field: ",
			stringify!(StdVideoVP9ColorConfig),
			"::",
			stringify!(flags)
		)
	);
	assert_eq!(
		unsafe { ::core::ptr::addr_of!((*ptr).BitDepth) as usize - ptr as usize },
		4usize,
		concat!(
			"Offset of field: ",
			stringify!(StdVideoVP9ColorConfig),
			"::",
			stringify!(BitDepth)
		)
	);
	assert_eq!(
		unsafe { ::core::ptr::addr_of!((*ptr).subsampling_x) as usize - ptr as usize },
		5usize,
		concat!(
			"Offset of field: ",
			stringify!(StdVideoVP9ColorConfig),
			"::",
			stringify!(subsampling_x)
		)
	);
	assert_eq!(
		unsafe { ::core::ptr::addr_of!((*ptr).subsampling_y) as usize - ptr as usize },
		6usize,
		concat!(
			"Offset of field: ",
			stringify!(StdVideoVP9ColorConfig),
			"::",
			stringify!(subsampling_y)
		)
	);
	assert_eq!(
		unsafe { ::core::ptr::addr_of!((*ptr).reserved1) as usize - ptr as usize },
		7usize,
		concat!(
			"Offset of field: ",
			stringify!(StdVideoVP9ColorConfig),
			"::",
			stringify!(reserved1)
		)
	);
	assert_eq!(
		unsafe { ::core::ptr::addr_of!((*ptr).color_space) as usize - ptr as usize },
		8usize,
		concat!(
			"Offset of field: ",
			stringify!(StdVideoVP9ColorConfig),
			"::",
			stringify!(color_space)
		)
	);
}

#[test]
fn bindgen_test_layout_StdVideoVP9LoopFilterFlags() {
	assert_eq!(
		::core::mem::size_of::<StdVideoVP9LoopFilterFlags>(),
		4usize,
		concat!("Size of: ", stringify!(StdVideoVP9LoopFilterFlags))
	);
	assert_eq!(
		::core::mem::align_of::<StdVideoVP9LoopFilterFlags>(),
		4usize,
		concat!("Alignment of ", stringify!(StdVideoVP9LoopFilterFlags))
	);
}

#[test]
fn bindgen_test_layout_StdVideoVP9LoopFilter() {
	const UNINIT: ::core::mem::MaybeUninit<StdVideoVP9LoopFilter> =
		::core::mem::MaybeUninit::uninit();
	let ptr = UNINIT.as_ptr();
	assert_eq!(
		::core::mem::size_of::<StdVideoVP9LoopFilter>(),
		16usize,
		concat!("Size of: ", stringify!(StdVideoVP9LoopFilter))
	);
	assert_eq!(
		::core::mem::align_of::<StdVideoVP9LoopFilter>(),
		4usize,
		concat!("Alignment of ", stringify!(StdVideoVP9LoopFilter))
	);
	assert_eq!(
		unsafe { ::core::ptr::addr_of!((*ptr).flags) as usize - ptr as usize },
		0usize,
		concat!(
			"Offset of field: ",
			stringify!(StdVideoVP9LoopFilter),
			"::",
			stringify!(flags)
		)
	);
	assert_eq!(
		unsafe { ::core::ptr::addr_of!((*ptr).loop_filter_level) as usize - ptr as usize },
		4usize,
		concat!(
			"Offset of field: ",
			stringify!(StdVideoVP9LoopFilter),
			"::",
			stringify!(loop_filter_level)
		)
	);
	assert_eq!(
		unsafe { ::core::ptr::addr_of!((*ptr).loop_filter_sharpness) as usize - ptr as usize },
		5usize,
		concat!(
			"Offset of field: ",
			stringify!(StdVideoVP9LoopFilter),
			"::",
			stringify!(loop_filter_sharpness)
		)
	);
	assert_eq!(
		unsafe { ::core::ptr::addr_of!((*ptr).update_ref_delta) as usize - ptr as usize },
		6usize,
		concat!(
			"Offset of field: ",
			stringify!(StdVideoVP9LoopFilter),
			"::",
			stringify!(update_ref_delta)
		)
	);
	assert_eq!(
		unsafe { ::core::ptr::addr_of!((*ptr).loop_filter_ref_deltas) as usize - ptr as usize },
		7usize,
		concat!(
			"Offset of field: ",
			stringify!(StdVideoVP9LoopFilter),
			"::",
			stringify!(loop_filter_ref_deltas)
		)
	);
	assert_eq!(
		unsafe { ::core::ptr::addr_of!((*ptr).update_mode_delta) as usize - ptr as usize },
		11usize,
		concat!(
			"Offset of field: ",
			stringify!(StdVideoVP9LoopFilter),
			"::",
			stringify!(update_mode_delta)
		)
	);
	assert_eq!(
		unsafe { ::core::ptr::addr_of!((*ptr).loop_filter_mode_deltas) as usize - ptr as usize },
		12usize,
		concat!(
			"Offset of field: ",
			stringify!(StdVideoVP9LoopFilter),
			"::",
			stringify!(loop_filter_mode_deltas)
		)
	);
}

#[test]
fn bindgen_test_layout_StdVideoVP9SegmentationFlags() {
	assert_eq!(
		::core::mem::size_of::<StdVideoVP9SegmentationFlags>(),
		4usize,
		concat!("Size of: ", stringify!(StdVideoVP9SegmentationFlags))
	);
	assert_eq!(
		::core::mem::align_of::<StdVideoVP9SegmentationFlags>(),
		4usize,
		concat!("Alignment of ", stringify!(StdVideoVP9SegmentationFlags))
	);
}

#[test]
fn bindgen_test_layout_StdVideoVP9Segmentation() {
	const UNINIT: ::core::mem::MaybeUninit<StdVideoVP9Segmentation> =
		::core::mem::MaybeUninit::uninit();
	let ptr = UNINIT.as_ptr();
	assert_eq!(
		::core::mem::size_of::<StdVideoVP9Segmentation>(),
		88usize,
		concat!("Size of: ", stringify!(StdVideoVP9Segmentation))
	);
	assert_eq!(
		::core::mem::align_of::<StdVideoVP9Segmentation>(),
		4usize,
		concat!("Alignment of ", stringify!(StdVideoVP9Segmentation))
	);
	assert_eq!(
		unsafe { ::core::ptr::addr_of!((*ptr).flags) as usize - ptr as usize },
		0usize,
		concat!(
			"Offset of field: ",
			stringify!(StdVideoVP9Segmentation),
			"::",
			stringify!(flags)
		)
	);
	assert_eq!(
		unsafe { ::core::ptr::addr_of!((*ptr).segmentation_tree_probs) as usize - ptr as usize },
		4usize,
		concat!(
			"Offset of field: ",
			stringify!(StdVideoVP9Segmentation),
			"::",
			stringify!(segmentation_tree_probs)
		)
	);
	assert_eq!(
		unsafe { ::core::ptr::addr_of!((*ptr).segmentation_pred_prob) as usize - ptr as usize },
		11usize,
		concat!(
			"Offset of field: ",
			stringify!(StdVideoVP9Segmentation),
			"::",
			stringify!(segmentation_pred_prob)
		)
	);
	assert_eq!(
		unsafe { ::core::ptr::addr_of!((*ptr).FeatureEnabled) as usize - ptr as usize },
		14usize,
		concat!(
			"Offset of field: ",
			stringify!(StdVideoVP9Segmentation),
			"::",
			stringify!(FeatureEnabled)
		)
	);
	assert_eq!(
		unsafe { ::core::ptr::addr_of!((*ptr).FeatureData) as usize - ptr as usize },
		22usize,
		concat!(
			"Offset of field: ",
			stringify!(StdVideoVP9Segmentation),
			"::",
			stringify!(FeatureData)
		)
	);
}

#[test]
fn bindgen_test_layout_StdVideoDecodeVP9PictureInfoFlags() {
	assert_eq!(
		::core::mem::size_of::<StdVideoDecodeVP9PictureInfoFlags>(),
		4usize,
		concat!("Size of: ", stringify!(StdVideoDecodeVP9PictureInfoFlags))
	);
	assert_eq!(
		::core::mem::align_of::<StdVideoDecodeVP9PictureInfoFlags>(),
		4usize,
		concat!(
			"Alignment of ",
			stringify!(StdVideoDecodeVP9PictureInfoFlags)
		)
	);
}

#[test]
fn bindgen_test_layout_StdVideoDecodeVP9PictureInfo() {
	const UNINIT: ::core::mem::MaybeUninit<StdVideoDecodeVP9PictureInfo> =
		::core::mem::MaybeUninit::uninit();
	let ptr = UNINIT.as_ptr();
	assert_eq!(
		::core::mem::size_of::<StdVideoDecodeVP9PictureInfo>(),
		56usize,
		concat!("Size of: ", stringify!(StdVideoDecodeVP9PictureInfo))
	);
	assert_eq!(
		::core::mem::align_of::<StdVideoDecodeVP9PictureInfo>(),
		8usize,
		concat!("Alignment of ", stringify!(StdVideoDecodeVP9PictureInfo))
	);
	assert_eq!(
		unsafe { ::core::ptr::addr_of!((*ptr).flags) as usize - ptr as usize },
		0usize,
		concat!(
			"Offset of field: ",
			stringify!(StdVideoDecodeVP9PictureInfo),
			"::",
			stringify!(flags)
		)
	);
	assert_eq!(
		unsafe { ::core::ptr::addr_of!((*ptr).profile) as usize - ptr as usize },
		4usize,
		concat!(
			"Offset of field: ",
			stringify!(StdVideoDecodeVP9PictureInfo),
			"::",
			stringify!(profile)
		)
	);
	assert_eq!(
		unsafe { ::core::ptr::addr_of!((*ptr).frame_type) as usize - ptr as usize },
		8usize,
		concat!(
			"Offset of field: ",
			stringify!(StdVideoDecodeVP9PictureInfo),
			"::",
			stringify!(frame_type)
		)
	);
	assert_eq!(
		unsafe { ::core::ptr::addr_of!((*ptr).frame_context_idx) as usize - ptr as usize },
		12usize,
		concat!(
			"Offset of field: ",
			stringify!(StdVideoDecodeVP9PictureInfo),
			"::",
			stringify!(frame_context_idx)
		)
	);
	assert_eq!(
		unsafe { ::core::ptr::addr_of!((*ptr).reset_frame_context) as usize - ptr as usize },
		13usize,
		concat!(
			"Offset of field: ",
			stringify!(StdVideoDecodeVP9PictureInfo),
			"::",
			stringify!(reset_frame_context)
		)
	);
	assert_eq!(
		unsafe { ::core::ptr::addr_of!((*ptr).refresh_frame_flags) as usize - ptr as usize },
		14usize,
		concat!(
			"Offset of field: ",
			stringify!(StdVideoDecodeVP9PictureInfo),
			"::",
			stringify!(refresh_frame_flags)
		)
	);
	assert_eq!(
		unsafe { ::core::ptr::addr_of!((*ptr).ref_frame_sign_bias_mask) as usize - ptr as usize },
		15usize,
		concat!(
			"Offset of field: ",
			stringify!(StdVideoDecodeVP9PictureInfo),
			"::",
			stringify!(ref_frame_sign_bias_mask)
		)
	);
	assert_eq!(
		unsafe { ::core::ptr::addr_of!((*ptr).interpolation_filter) as usize - ptr as usize },
		16usize,
		concat!(
			"Offset of field: ",
			stringify!(StdVideoDecodeVP9PictureInfo),
			"::",
			stringify!(interpolation_filter)
		)
	);
	assert_eq!(
		unsafe { ::core::ptr::addr_of!((*ptr).base_q_idx) as usize - ptr as usize },
		20usize,
		concat!(
			"Offset of field: ",
			stringify!(StdVideoDecodeVP9PictureInfo),
			"::",
			stringify!(base_q_idx)
		)
	);
	assert_eq!(
		unsafe { ::core::ptr::addr_of!((*ptr).delta_q_y_dc) as usize - ptr as usize },
		21usize,
		concat!(
			"Offset of field: ",
			stringify!(StdVideoDecodeVP9PictureInfo),
			"::",
			stringify!(delta_q_y_dc)
		)
	);
	assert_eq!(
		unsafe { ::core::ptr::addr_of!((*ptr).delta_q_uv_dc) as usize - ptr as usize },
		22usize,
		concat!(
			"Offset of field: ",
			stringify!(StdVideoDecodeVP9PictureInfo),
			"::",
			stringify!(delta_q_uv_dc)
		)
	);
	assert_eq!(
		unsafe { ::core::ptr::addr_of!((*ptr).delta_q_uv_ac) as usize - ptr as usize },
		23usize,
		concat!(
			"Offset of field: ",
			stringify!(StdVideoDecodeVP9PictureInfo),
			"::",
			stringify!(delta_q_uv_ac)
		)
	);
	assert_eq!(
		unsafe { ::core::ptr::addr_of!((*ptr).tile_cols_log2) as usize - ptr as usize },
		24usize,
		concat!(
			"Offset of field: ",
			stringify!(StdVideoDecodeVP9PictureInfo),
			"::",
			stringify!(tile_cols_log2)
		)
	);
	assert_eq!(
		unsafe { ::core::ptr::addr_of!((*ptr).tile_rows_log2) as usize - ptr as usize },
		25usize,
		concat!(
			"Offset of field: ",
			stringify!(StdVideoDecodeVP9PictureInfo),
			"::",
			stringify!(tile_rows_log2)
		)
	);
	assert_eq!(
		unsafe { ::core::ptr::addr_of!((*ptr).reserved1) as usize - ptr as usize },
		26usize,
		concat!(
			"Offset of field: ",
			stringify!(StdVideoDecodeVP9PictureInfo),
			"::",
			stringify!(reserved1)
		)
	);
	assert_eq!(
		unsafe { ::core::ptr::addr_of!((*ptr).pColorConfig) as usize - ptr as usize },
		32usize,
		concat!(
			"Offset of field: ",
			stringify!(StdVideoDecodeVP9PictureInfo),
			"::",
			stringify!(pColorConfig)
		)
	);
	assert_eq!(
		unsafe { ::core::ptr::addr_of!((*ptr).pLoopFilter) as usize - ptr as usize },
		40usize,
		concat!(
			"Offset of field: ",
			stringify!(StdVideoDecodeVP9PictureInfo),
			"::",
			stringify!(pLoopFilter)
		)
	);
	assert_eq!(
		unsafe { ::core::ptr::addr_of!((*ptr).pSegmentation) as usize - ptr as usize },
		48usize,
		concat!(
			"Offset of field: ",
			stringify!(StdVideoDecodeVP9PictureInfo),
			"::",
			stringify!(pSegmentation)
		)
	);
}
