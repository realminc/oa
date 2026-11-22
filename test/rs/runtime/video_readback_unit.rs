use super::*;

#[test]
fn wide_readback_preserves_all_significant_bits_and_chroma_order() -> Result<()> {
	let extent = video::VideoExtent {
		width: 3,
		height: 1,
	};
	for (format, bits, interleaved) in [
		(
			vk::Format::G10X6_B10X6R10X6_2PLANE_420_UNORM_3PACK16,
			10,
			true,
		),
		(
			vk::Format::G10X6_B10X6_R10X6_3PLANE_420_UNORM_3PACK16,
			10,
			false,
		),
		(
			vk::Format::G12X4_B12X4R12X4_2PLANE_420_UNORM_3PACK16,
			12,
			true,
		),
		(
			vk::Format::G12X4_B12X4_R12X4_3PLANE_420_UNORM_3PACK16,
			12,
			false,
		),
	] {
		let peak = (1_u16 << bits) - 1;
		let samples = if interleaved {
			[0, 1, peak, 0, 2, 4, 3, 5]
		} else {
			[0, 1, peak, 0, 2, 3, 4, 5]
		};
		let bytes = samples
			.into_iter()
			.flat_map(|v| (v << (16 - bits)).to_le_bytes())
			.collect();
		let video::VideoHostPlanes::U16(actual) = normalize_yuv420(format, extent, bytes)? else {
			panic!("expected wide planes")
		};
		assert_eq!(actual.as_slice(), [0, 1, peak, 2, 3, 4, 5]);
		let (size, regions) = yuv420_copy_layout(format, extent, 3, [0, 0])?;
		assert_eq!(size, 16);
		assert_eq!(regions[1].buffer_offset, 8);
		assert_eq!(regions[1].image_extent.width, 2);
		assert_eq!(regions[1].image_subresource.base_array_layer, 3);
		assert!(normalize_yuv420(format, extent, vec![0; size - 1]).is_err());
	}
	Ok(())
}

#[test]
fn eight_bit_readback_reuses_luma_and_removes_plane_padding() -> Result<()> {
	let extent = video::VideoExtent {
		width: 3,
		height: 1,
	};
	for (format, raw) in [
		(
			vk::Format::G8_B8R8_2PLANE_420_UNORM,
			vec![1, 2, 3, 99, 4, 6, 5, 7],
		),
		(
			vk::Format::G8_B8_R8_3PLANE_420_UNORM,
			vec![1, 2, 3, 99, 4, 5, 99, 99, 6, 7, 99, 99],
		),
	] {
		let source = raw.as_ptr();
		let video::VideoHostPlanes::U8(actual) = normalize_yuv420(format, extent, raw)? else {
			panic!("expected byte planes")
		};
		assert_eq!(actual.as_ptr(), source);
		assert_eq!(actual.as_slice(), [1, 2, 3, 4, 5, 6, 7]);
		assert!(normalize_yuv420(format, extent, vec![]).is_err());
		assert!(
			yuv420_copy_layout(
				format,
				video::VideoExtent {
					width: u32::MAX,
					height: u32::MAX
				},
				0,
				[0, 0],
			)
			.is_err()
		);
	}
	Ok(())
}
