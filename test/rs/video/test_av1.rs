use oa::video::{
	Av1ChromaSamplePosition, Av1CodingToolChoice, Av1ObuType, Av1Profile, Av1ReferenceState,
	VideoChromaSubsampling, VideoComponentBitDepth, VideoDemuxer, inspect_av1_access_unit,
	parse_av1_frame_header, parse_av1_obus, parse_av1_sequence_header, parse_av1_tile_group,
};

struct BitWriter {
	bytes: Vec<u8>,
	bit_count: usize,
}

impl BitWriter {
	fn new() -> Self {
		Self {
			bytes: Vec::new(),
			bit_count: 0,
		}
	}

	fn bits(&mut self, value: u32, count: usize) {
		for shift in (0..count).rev() {
			if self.bit_count & 7 == 0 {
				self.bytes.push(0);
			}
			let bit = ((value >> shift) & 1) as u8;
			let byte_index = self.bit_count >> 3;
			self.bytes[byte_index] |= bit << (7 - (self.bit_count & 7));
			self.bit_count += 1;
		}
	}

	fn finish(mut self) -> Vec<u8> {
		self.bits(1, 1);
		self.bytes
	}
}

#[test]
fn av1_obus_are_borrowed_bounded_and_counted() -> oa::Result<()> {
	let bytes = [
		0x0a, 2, 0xaa, 0xbb, // sequence header
		0x1a, 1, 0xcc, // frame header
		0x22, 2, 0xdd, 0xee, // tile group
	];
	let obus = parse_av1_obus(&bytes)?;
	assert_eq!(obus.len(), 3);
	assert_eq!(obus[0].type_(), Av1ObuType::SequenceHeader);
	assert_eq!(obus[0].header_offset(), 0);
	assert_eq!(obus[0].header_size(), 2);
	assert_eq!(obus[0].payload(), &bytes[2..4]);
	assert_eq!(obus[1].type_(), Av1ObuType::FrameHeader);
	assert_eq!(obus[2].type_(), Av1ObuType::TileGroup);

	let info = inspect_av1_access_unit(&bytes)?;
	assert_eq!(info.sequence_headers, 1);
	assert_eq!(info.frame_headers, 1);
	assert_eq!(info.tile_groups, 1);
	assert_eq!(info.picture_count(), 1);
	assert_eq!(info.ivf_timestamp, None);
	Ok(())
}

#[test]
fn av1_ivf_and_malformed_sizes_fail_or_unwrap_exactly() -> oa::Result<()> {
	let obu = [0x32, 1, 0x80];
	let parsed = parse_av1_obus(&obu)?;
	let tiles = parse_av1_tile_group(parsed[0], &oa::video::Av1FrameHeader::default())?;
	assert_eq!(tiles.tile_offsets, [2]);
	assert_eq!(tiles.tile_sizes, [1]);
	let mut ivf = vec![0_u8; 44];
	ivf[..4].copy_from_slice(b"DKIF");
	ivf[6..8].copy_from_slice(&32_u16.to_le_bytes());
	ivf[8..12].copy_from_slice(b"AV01");
	ivf[32..36].copy_from_slice(&(obu.len() as u32).to_le_bytes());
	ivf[36..44].copy_from_slice(&37_u64.to_le_bytes());
	ivf.extend_from_slice(&obu);
	let info = inspect_av1_access_unit(&ivf)?;
	assert_eq!(info.frames, 1);
	assert_eq!(info.picture_count(), 1);
	assert_eq!(info.ivf_timestamp, Some(37));

	assert!(parse_av1_obus(&[]).is_err());
	assert!(parse_av1_obus(&[0x80]).is_err());
	assert!(parse_av1_obus(&[0x0e]).is_err());
	assert!(parse_av1_obus(&[0x0a, 4, 1]).is_err());
	assert!(parse_av1_obus(&[0x0a, 0x80]).is_err());
	assert!(
		parse_av1_tile_group(
			parse_av1_obus(&[0x0a, 1, 0x80])?[0],
			&oa::video::Av1FrameHeader::default(),
		)
		.is_err()
	);
	Ok(())
}

#[test]
fn av1_reduced_sequence_header_preserves_profile_tools_color_and_extent() -> oa::Result<()> {
	let mut bits = BitWriter::new();
	bits.bits(0, 3); // Main profile.
	bits.bits(1, 1); // still_picture.
	bits.bits(1, 1); // reduced_still_picture_header.
	bits.bits(8, 5); // Implicit operating point zero, level 4.0.
	bits.bits(0, 1); // Main tier.
	bits.bits(10, 4); // 11 width bits.
	bits.bits(9, 4); // 10 height bits.
	bits.bits(1279, 11);
	bits.bits(719, 10);
	bits.bits(0, 1); // use_128x128_superblock.
	bits.bits(1, 1); // enable_filter_intra.
	bits.bits(1, 1); // enable_intra_edge_filter.
	bits.bits(0, 1); // enable_superres.
	bits.bits(1, 1); // enable_cdef.
	bits.bits(1, 1); // enable_restoration.
	bits.bits(0, 1); // high_bitdepth.
	bits.bits(0, 1); // mono_chrome.
	bits.bits(1, 1); // color_description_present_flag.
	bits.bits(1, 8); // BT.709 primaries.
	bits.bits(1, 8); // BT.709 transfer.
	bits.bits(1, 8); // BT.709 matrix.
	bits.bits(0, 1); // studio range.
	bits.bits(2, 2); // colocated chroma samples.
	bits.bits(0, 1); // separate_uv_delta_q.
	bits.bits(1, 1); // film_grain_params_present.

	let mut sequence = parse_av1_sequence_header(&bits.finish())?;
	assert_eq!(sequence.profile, Av1Profile::Main);
	assert!(sequence.still_picture);
	assert!(sequence.reduced_still_picture_header);
	assert_eq!(sequence.coded_width(), 1280);
	assert_eq!(sequence.coded_height(), 720);
	assert_eq!(sequence.operating_points.len(), 1);
	assert_eq!(sequence.operating_points[0].idc, 0);
	assert_eq!(sequence.operating_points[0].level, 8);
	assert_eq!(sequence.order_hint_bits, 0);
	assert_eq!(
		sequence.screen_content_tools,
		Av1CodingToolChoice::SelectPerFrame
	);
	assert_eq!(sequence.color.bit_depth, VideoComponentBitDepth::Eight);
	assert_eq!(
		sequence.color.chroma_subsampling,
		VideoChromaSubsampling::Yuv420
	);
	assert_eq!(
		sequence.color.chroma_sample_position,
		Av1ChromaSamplePosition::Colocated
	);
	assert!(sequence.film_grain_params_present);

	sequence.reduced_still_picture_header = false;
	let mut references = Av1ReferenceState::new([
		Some(1),
		Some(2),
		Some(3),
		Some(4),
		Some(5),
		Some(6),
		Some(7),
		Some(8),
	]);
	let shown = parse_av1_frame_header(&[0xd0], &sequence, &references)?;
	assert!(shown.show_existing_frame);
	assert_eq!(shown.frame_to_show_map_idx, 5);
	assert_eq!(shown.header_size, 1);
	references.refresh(&shown);
	assert_eq!(references.order_hint(5), Some(6));
	assert!(parse_av1_frame_header(&[], &sequence, &references).is_err());
	Ok(())
}

#[test]
fn av1_sequence_header_rejects_reserved_profiles_and_truncation() {
	assert!(parse_av1_sequence_header(&[]).is_err());
	assert!(parse_av1_sequence_header(&[0x60]).is_err());
	assert!(parse_av1_sequence_header(&[0x10]).is_err());
}

#[test]
#[ignore = "requires OA donor AV1 fixture"]
fn av1_donor_packets_have_bounded_picture_inventory() -> oa::Result<()> {
	let fixture = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
		.with_file_name("oa")
		.join("sdk/asset/video/clip/shibuya_720p_30fps_av1_main_8bit_420.mp4");
	let mut demuxer = VideoDemuxer::open(fixture)?;
	let expected_packets = demuxer.info().sample_count();
	let mut packets = 0_u32;
	let mut pictures = 0_u32;
	let mut sequence_headers = 0_u32;
	let mut parsed_sequence = None;
	let mut references = Av1ReferenceState::default();
	let mut parsed_frames = 0_u32;
	let mut decoded_frames = 0_u32;
	let mut parsed_tiles = 0_u32;
	while let Some(packet) = demuxer.read_next_packet()? {
		let info = inspect_av1_access_unit(packet.data())?;
		assert!(info.picture_count() > 0);
		let mut separate_header = None;
		for obu in parse_av1_obus(packet.data())? {
			match obu.type_() {
				Av1ObuType::SequenceHeader => {
					parsed_sequence = Some(parse_av1_sequence_header(obu.payload())?);
				}
				Av1ObuType::Frame | Av1ObuType::FrameHeader => {
					let sequence = parsed_sequence
						.as_ref()
						.expect("AV1 picture must follow a sequence header");
					let frame = parse_av1_frame_header(obu.payload(), sequence, &references)?;
					assert!(frame.header_size <= obu.payload().len());
					if obu.type_() == Av1ObuType::Frame && !frame.show_existing_frame {
						decoded_frames += 1;
						assert!(frame.header_size < obu.payload().len());
						let tiles = parse_av1_tile_group(obu, &frame)?;
						assert_eq!(tiles.first_tile, 0);
						assert_eq!(tiles.tile_offsets.len(), tiles.tile_sizes.len());
						for (&offset, &size) in tiles.tile_offsets.iter().zip(&tiles.tile_sizes) {
							assert!(offset as usize + size as usize <= packet.data().len());
						}
						parsed_tiles += tiles.tile_offsets.len() as u32;
					} else if obu.type_() == Av1ObuType::FrameHeader && !frame.show_existing_frame {
						decoded_frames += 1;
						separate_header = Some(frame.clone());
					}
					references.refresh(&frame);
					parsed_frames += 1;
				}
				Av1ObuType::TileGroup => {
					let frame = separate_header
						.as_ref()
						.expect("AV1 tile group must follow a separate frame header");
					let tiles = parse_av1_tile_group(obu, frame)?;
					assert!(
						usize::from(tiles.first_tile)
							< usize::from(frame.tile_columns) * usize::from(frame.tile_rows)
					);
					assert_eq!(tiles.tile_offsets.len(), tiles.tile_sizes.len());
					for (&offset, &size) in tiles.tile_offsets.iter().zip(&tiles.tile_sizes) {
						assert!(offset as usize + size as usize <= packet.data().len());
					}
					parsed_tiles += tiles.tile_offsets.len() as u32;
				}
				_ => {}
			}
		}
		packets += 1;
		pictures += info.picture_count();
		sequence_headers += info.sequence_headers;
	}
	assert_eq!(packets, expected_packets);
	assert!(pictures > packets);
	assert!(sequence_headers > 0);
	assert_eq!(parsed_frames, pictures);
	assert!(parsed_tiles >= decoded_frames);
	let sequence = parsed_sequence.expect("donor AV1 fixture must carry a sequence header");
	assert_eq!(sequence.profile, Av1Profile::Main);
	assert_eq!(sequence.color.bit_depth, VideoComponentBitDepth::Eight);
	assert_eq!(
		sequence.color.chroma_subsampling,
		VideoChromaSubsampling::Yuv420
	);
	assert_eq!(sequence.coded_width(), 1280);
	assert_eq!(sequence.coded_height(), 720);
	Ok(())
}
