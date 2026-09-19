//! Bounded AV1 access-unit and open-bitstream-unit structure.

use crate::{Error, Result};

use super::{Av1Profile, VideoChromaSubsampling, VideoComponentBitDepth};

mod frame;

pub use frame::{
	Av1FrameHeader, Av1FrameType, Av1InterpolationFilter, Av1ReferenceState, Av1RestorationType,
	Av1TileGroup, Av1TransformMode, parse_av1_frame_header, parse_av1_tile_group,
};

/// AV1 sequence-level timing metadata.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Av1TimingInfo {
	pub num_units_in_display_tick: u32,
	pub time_scale: u32,
	pub equal_picture_interval: bool,
	pub num_ticks_per_picture_minus_1: Option<u32>,
}

/// Optional AV1 color-description triplet.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Av1ColorDescription {
	pub color_primaries: u8,
	pub transfer_characteristics: u8,
	pub matrix_coefficients: u8,
}

/// AV1 chroma sample position for 4:2:0 streams.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Av1ChromaSamplePosition {
	Unknown,
	Vertical,
	Colocated,
	Reserved,
}

impl Av1ChromaSamplePosition {
	const fn from_raw(value: u32) -> Self {
		match value {
			0 => Self::Unknown,
			1 => Self::Vertical,
			2 => Self::Colocated,
			_ => Self::Reserved,
		}
	}
}

/// Parsed AV1 color configuration required by decoder backends.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Av1ColorConfig {
	pub bit_depth: VideoComponentBitDepth,
	pub monochrome: bool,
	pub color_description: Option<Av1ColorDescription>,
	pub full_range: bool,
	pub chroma_subsampling: VideoChromaSubsampling,
	pub chroma_sample_position: Av1ChromaSamplePosition,
	pub separate_uv_delta_q: bool,
}

/// One AV1 operating point declared by a sequence header.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Av1OperatingPoint {
	pub idc: u16,
	pub level: u8,
	pub high_tier: bool,
	pub decoder_model_present: bool,
	pub initial_display_delay_minus_1: Option<u8>,
}

/// How a sequence constrains one optional frame coding tool.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Av1CodingToolChoice {
	Disabled,
	Enabled,
	SelectPerFrame,
}

/// Parsed AV1 sequence header required by decode-session setup.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Av1SequenceHeader {
	pub profile: Av1Profile,
	pub still_picture: bool,
	pub reduced_still_picture_header: bool,
	pub timing: Option<Av1TimingInfo>,
	pub decoder_model_info_present: bool,
	pub initial_display_delay_present: bool,
	pub operating_points: Vec<Av1OperatingPoint>,
	pub frame_width_bits_minus_1: u8,
	pub frame_height_bits_minus_1: u8,
	pub max_frame_width_minus_1: u16,
	pub max_frame_height_minus_1: u16,
	pub frame_id_numbers_present: bool,
	pub delta_frame_id_length_minus_2: u8,
	pub additional_frame_id_length_minus_1: u8,
	pub use_128x128_superblock: bool,
	pub enable_filter_intra: bool,
	pub enable_intra_edge_filter: bool,
	pub enable_inter_intra_compound: bool,
	pub enable_masked_compound: bool,
	pub enable_warped_motion: bool,
	pub enable_dual_filter: bool,
	pub enable_order_hint: bool,
	pub enable_joint_compound: bool,
	pub enable_reference_frame_motion_vectors: bool,
	pub screen_content_tools: Av1CodingToolChoice,
	pub integer_motion_vectors: Av1CodingToolChoice,
	pub order_hint_bits: u8,
	pub enable_superres: bool,
	pub enable_cdef: bool,
	pub enable_restoration: bool,
	pub color: Av1ColorConfig,
	pub film_grain_params_present: bool,
}

impl Av1SequenceHeader {
	/// Return the maximum coded width in luma samples.
	pub const fn coded_width(&self) -> u32 {
		self.max_frame_width_minus_1 as u32 + 1
	}

	/// Return the maximum coded height in luma samples.
	pub const fn coded_height(&self) -> u32 {
		self.max_frame_height_minus_1 as u32 + 1
	}
}

/// AV1 open-bitstream-unit identity.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Av1ObuType {
	SequenceHeader,
	TemporalDelimiter,
	FrameHeader,
	TileGroup,
	Metadata,
	Frame,
	RedundantFrameHeader,
	TileList,
	Padding,
	Reserved(u8),
}

impl Av1ObuType {
	const fn from_raw(value: u8) -> Self {
		match value {
			1 => Self::SequenceHeader,
			2 => Self::TemporalDelimiter,
			3 => Self::FrameHeader,
			4 => Self::TileGroup,
			5 => Self::Metadata,
			6 => Self::Frame,
			7 => Self::RedundantFrameHeader,
			8 => Self::TileList,
			15 => Self::Padding,
			other => Self::Reserved(other),
		}
	}
}

/// One borrowed AV1 OBU and its exact access-unit byte range.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Av1Obu<'a> {
	type_: Av1ObuType,
	header_offset: usize,
	header_size: usize,
	payload: &'a [u8],
}

impl<'a> Av1Obu<'a> {
	/// Return the OBU type.
	pub const fn type_(&self) -> Av1ObuType {
		self.type_
	}

	/// Return the OBU header byte offset in the parsed payload.
	pub const fn header_offset(&self) -> usize {
		self.header_offset
	}

	/// Return the header, optional extension, and size-field byte count.
	pub const fn header_size(&self) -> usize {
		self.header_size
	}

	/// Borrow the OBU payload without copying.
	pub const fn payload(&self) -> &'a [u8] {
		self.payload
	}
}

/// Structural counts for one raw or IVF-wrapped AV1 access unit.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Av1AccessUnitInfo {
	pub sequence_headers: u32,
	pub frames: u32,
	pub frame_headers: u32,
	pub tile_groups: u32,
	pub ivf_timestamp: Option<u64>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Av1Picture {
	pub sequence: Av1SequenceHeader,
	pub frame: Av1FrameHeader,
	pub frame_header_offset: u32,
	pub tiles: Option<Av1TileGroup>,
}

#[derive(Clone, Debug, Default)]
pub(crate) struct Av1Parser {
	sequence: Option<Av1SequenceHeader>,
	references: Av1ReferenceState,
}

impl Av1Parser {
	pub fn parse_access_unit(&mut self, bytes: &[u8]) -> Result<Vec<Av1Picture>> {
		let mut next = self.clone();
		let pictures = next.parse_access_unit_in_place(bytes)?;
		*self = next;
		Ok(pictures)
	}

	fn parse_access_unit_in_place(&mut self, bytes: &[u8]) -> Result<Vec<Av1Picture>> {
		let mut pictures = Vec::new();
		let mut pending = None;
		for obu in parse_av1_obus(bytes)? {
			match obu.type_() {
				Av1ObuType::SequenceHeader => {
					finish_pending_picture(&mut pending, &mut pictures)?;
					self.sequence = Some(parse_av1_sequence_header(obu.payload())?);
				}
				Av1ObuType::Frame | Av1ObuType::FrameHeader => {
					finish_pending_picture(&mut pending, &mut pictures)?;
					let sequence = self.sequence.as_ref().ok_or_else(|| {
						Error::failed_precondition("AV1 picture arrived before a sequence header")
					})?;
					let frame = parse_av1_frame_header(obu.payload(), sequence, &self.references)?;
					let picture = Av1Picture {
						sequence: sequence.clone(),
						frame_header_offset: u32::try_from(obu.header_offset())
							.map_err(|_| Error::out_of_range("AV1 frame-header offset exceeds u32"))?,
						tiles: None,
						frame,
					};
					self.references.refresh(&picture.frame);
					if picture.frame.show_existing_frame {
						pictures
							.try_reserve(1)
							.map_err(|_| Error::resource_exhausted("AV1 picture allocation failed"))?;
						pictures.push(picture);
					} else if obu.type_() == Av1ObuType::Frame {
						let mut picture = picture;
						picture.tiles = Some(parse_av1_tile_group(obu, &picture.frame)?);
						validate_complete_picture(&picture)?;
						pictures
							.try_reserve(1)
							.map_err(|_| Error::resource_exhausted("AV1 picture allocation failed"))?;
						pictures.push(picture);
					} else {
						pending = Some(picture);
					}
				}
				Av1ObuType::TileGroup => {
					let picture = pending
						.as_mut()
						.ok_or_else(|| Error::data_loss("AV1 tile group has no preceding frame header"))?;
					let group = parse_av1_tile_group(obu, &picture.frame)?;
					let tiles = picture.tiles.get_or_insert_with(|| Av1TileGroup {
						first_tile: 0,
						tile_offsets: Vec::new(),
						tile_sizes: Vec::new(),
					});
					if usize::from(group.first_tile) != tiles.tile_offsets.len() {
						return Err(Error::data_loss(
							"AV1 tile groups are not contiguous in tile order",
						));
					}
					tiles
						.tile_offsets
						.try_reserve(group.tile_offsets.len())
						.map_err(|_| Error::resource_exhausted("AV1 tile allocation failed"))?;
					tiles
						.tile_sizes
						.try_reserve(group.tile_sizes.len())
						.map_err(|_| Error::resource_exhausted("AV1 tile allocation failed"))?;
					tiles.tile_offsets.extend(group.tile_offsets);
					tiles.tile_sizes.extend(group.tile_sizes);
				}
				_ => {}
			}
		}
		finish_pending_picture(&mut pending, &mut pictures)?;
		Ok(pictures)
	}
}

fn finish_pending_picture(
	pending: &mut Option<Av1Picture>,
	pictures: &mut Vec<Av1Picture>,
) -> Result<()> {
	let Some(picture) = pending.take() else {
		return Ok(());
	};
	validate_complete_picture(&picture)?;
	pictures
		.try_reserve(1)
		.map_err(|_| Error::resource_exhausted("AV1 picture allocation failed"))?;
	pictures.push(picture);
	Ok(())
}

fn validate_complete_picture(picture: &Av1Picture) -> Result<()> {
	if picture.frame.show_existing_frame {
		if picture.tiles.is_some() {
			return Err(Error::data_loss(
				"AV1 show-existing picture unexpectedly carries tiles",
			));
		}
		return Ok(());
	}
	let expected = usize::from(picture.frame.tile_columns)
		.checked_mul(usize::from(picture.frame.tile_rows))
		.ok_or_else(|| Error::out_of_range("AV1 tile-grid size overflowed"))?;
	let tiles = picture
		.tiles
		.as_ref()
		.ok_or_else(|| Error::data_loss("coded AV1 picture carries no tile group"))?;
	if tiles.first_tile != 0
		|| tiles.tile_offsets.len() != expected
		|| tiles.tile_sizes.len() != expected
	{
		return Err(Error::data_loss(
			"coded AV1 picture does not contain one complete ordered tile set",
		));
	}
	Ok(())
}

impl Av1AccessUnitInfo {
	/// Return the number of coded or show-existing picture headers.
	pub const fn picture_count(self) -> u32 {
		self.frames + self.frame_headers
	}
}

/// Parse one AV1 sequence-header OBU payload.
///
/// Pass [`Av1Obu::payload`] from an OBU whose type is
/// [`Av1ObuType::SequenceHeader`]. The returned representation is independent
/// of Vulkan's standard-video C structures and retains the stream metadata
/// needed to create exact backend profiles and session parameters.
///
/// # Errors
///
/// Returns an error when the payload is empty or truncated, declares a
/// reserved profile, carries an invalid frame-ID length, or encodes a chroma
/// layout outside the AV1 profile rules.
pub fn parse_av1_sequence_header(payload: &[u8]) -> Result<Av1SequenceHeader> {
	if payload.is_empty() {
		return Err(Error::invalid_argument(
			"AV1 sequence-header payload is empty",
		));
	}
	let mut reader = Av1BitReader::new(payload)?;
	let profile = match reader.bits(3, "sequence profile")? {
		0 => Av1Profile::Main,
		1 => Av1Profile::High,
		2 => Av1Profile::Professional,
		_ => {
			return Err(Error::data_loss(
				"AV1 sequence header uses a reserved profile",
			));
		}
	};
	let still_picture = reader.bit("still-picture flag")?;
	let reduced_still_picture_header = reader.bit("reduced-still-picture flag")?;

	let mut timing = None;
	let mut decoder_model_info_present = false;
	let mut buffer_delay_length = 0_u32;
	let mut initial_display_delay_present = false;
	let operating_point_count;
	if reduced_still_picture_header {
		operating_point_count = 1;
	} else {
		if reader.bit("sequence timing-present flag")? {
			let num_units_in_display_tick = reader.bits(32, "display-tick numerator")?;
			let time_scale = reader.bits(32, "display-tick time scale")?;
			let equal_picture_interval = reader.bit("equal-picture-interval flag")?;
			let num_ticks_per_picture_minus_1 = equal_picture_interval
				.then(|| reader.uvlc("ticks-per-picture interval"))
				.transpose()?;
			timing = Some(Av1TimingInfo {
				num_units_in_display_tick,
				time_scale,
				equal_picture_interval,
				num_ticks_per_picture_minus_1,
			});
			decoder_model_info_present = reader.bit("decoder-model-present flag")?;
			if decoder_model_info_present {
				buffer_delay_length = reader
					.bits(5, "decoder-model buffer-delay length")?
					.checked_add(1)
					.ok_or_else(|| Error::data_loss("AV1 buffer-delay length overflowed"))?;
				reader.skip(32, "decoder-model decoding tick")?;
				reader.skip(5, "decoder-model removal-time length")?;
				reader.skip(5, "decoder-model presentation-time length")?;
			}
		}
		initial_display_delay_present = reader.bit("initial-display-delay-present flag")?;
		operating_point_count = reader
			.bits(5, "operating-point count")?
			.checked_add(1)
			.ok_or_else(|| Error::data_loss("AV1 operating-point count overflowed"))?;
	}

	let operating_point_capacity = usize::try_from(operating_point_count)
		.map_err(|_| Error::out_of_range("AV1 operating-point count exceeds usize"))?;
	let mut operating_points = Vec::new();
	operating_points
		.try_reserve_exact(operating_point_capacity)
		.map_err(|_| Error::resource_exhausted("AV1 operating-point allocation failed"))?;
	for index in 0..operating_point_count {
		// In a reduced still-picture header the operating-point IDC is inferred
		// as zero rather than present in the payload (AV1 section 5.5.1).
		let idc = if reduced_still_picture_header {
			0
		} else {
			u16::try_from(reader.bits(12, "operating-point IDC")?)
				.map_err(|_| Error::internal("AV1 operating-point IDC exceeds u16"))?
		};
		let level = u8::try_from(reader.bits(5, "operating-point level")?)
			.map_err(|_| Error::internal("AV1 operating-point level exceeds u8"))?;
		let high_tier = if level > 7 {
			reader.bit("operating-point tier")?
		} else {
			false
		};
		let decoder_model_present = if decoder_model_info_present {
			reader.bit("operating-point decoder-model flag")?
		} else {
			false
		};
		if decoder_model_present {
			reader.skip(buffer_delay_length, "decoder buffer delay")?;
			reader.skip(buffer_delay_length, "encoder buffer delay")?;
			reader.skip(1, "low-delay mode flag")?;
		}
		let initial_display_delay_minus_1 = if initial_display_delay_present {
			reader
				.bit("operating-point initial-display-delay flag")?
				.then(|| reader.bits(4, "operating-point initial display delay"))
				.transpose()?
				.map(|value| {
					u8::try_from(value).expect("a four-bit AV1 initial display delay always fits in u8")
				})
		} else {
			None
		};
		operating_points.push(Av1OperatingPoint {
			idc,
			level,
			high_tier,
			decoder_model_present,
			initial_display_delay_minus_1,
		});
		if reduced_still_picture_header && index != 0 {
			return Err(Error::internal(
				"reduced AV1 sequence header created multiple operating points",
			));
		}
	}

	let frame_width_bits_minus_1 = u8::try_from(reader.bits(4, "frame-width bit count")?)
		.map_err(|_| Error::internal("AV1 frame-width bit count exceeds u8"))?;
	let frame_height_bits_minus_1 = u8::try_from(reader.bits(4, "frame-height bit count")?)
		.map_err(|_| Error::internal("AV1 frame-height bit count exceeds u8"))?;
	let max_frame_width_minus_1 = u16::try_from(reader.bits(
		u32::from(frame_width_bits_minus_1) + 1,
		"maximum frame width",
	)?)
	.map_err(|_| Error::internal("AV1 maximum frame width exceeds u16"))?;
	let max_frame_height_minus_1 = u16::try_from(reader.bits(
		u32::from(frame_height_bits_minus_1) + 1,
		"maximum frame height",
	)?)
	.map_err(|_| Error::internal("AV1 maximum frame height exceeds u16"))?;

	let frame_id_numbers_present = if reduced_still_picture_header {
		false
	} else {
		reader.bit("frame-ID-numbers-present flag")?
	};
	let (delta_frame_id_length_minus_2, additional_frame_id_length_minus_1) =
		if frame_id_numbers_present {
			let delta = u8::try_from(reader.bits(4, "delta frame-ID length")?)
				.map_err(|_| Error::internal("AV1 delta frame-ID length exceeds u8"))?;
			let additional = u8::try_from(reader.bits(3, "additional frame-ID length")?)
				.map_err(|_| Error::internal("AV1 additional frame-ID length exceeds u8"))?;
			if u16::from(delta) + u16::from(additional) + 6 > 16 {
				return Err(Error::data_loss("AV1 frame-ID length exceeds 16 bits"));
			}
			(delta, additional)
		} else {
			(0, 0)
		};

	let use_128x128_superblock = reader.bit("128x128-superblock flag")?;
	let enable_filter_intra = reader.bit("filter-intra flag")?;
	let enable_intra_edge_filter = reader.bit("intra-edge-filter flag")?;
	let (
		enable_inter_intra_compound,
		enable_masked_compound,
		enable_warped_motion,
		enable_dual_filter,
		enable_order_hint,
		enable_joint_compound,
		enable_reference_frame_motion_vectors,
		screen_content_tools,
		integer_motion_vectors,
		order_hint_bits,
	) = if reduced_still_picture_header {
		(
			false,
			false,
			false,
			false,
			false,
			false,
			false,
			Av1CodingToolChoice::SelectPerFrame,
			Av1CodingToolChoice::SelectPerFrame,
			0,
		)
	} else {
		let enable_inter_intra_compound = reader.bit("inter-intra-compound flag")?;
		let enable_masked_compound = reader.bit("masked-compound flag")?;
		let enable_warped_motion = reader.bit("warped-motion flag")?;
		let enable_dual_filter = reader.bit("dual-filter flag")?;
		let enable_order_hint = reader.bit("order-hint flag")?;
		let (enable_joint_compound, enable_reference_frame_motion_vectors) = if enable_order_hint {
			(
				reader.bit("joint-compound flag")?,
				reader.bit("reference-frame-motion-vector flag")?,
			)
		} else {
			(false, false)
		};
		let screen_content_tools = read_tool_choice(&mut reader, "screen-content tools")?;
		let integer_motion_vectors = if screen_content_tools == Av1CodingToolChoice::Disabled {
			Av1CodingToolChoice::SelectPerFrame
		} else {
			read_tool_choice(&mut reader, "integer motion vectors")?
		};
		let order_hint_bits = if enable_order_hint {
			u8::try_from(reader.bits(3, "order-hint bit count")? + 1)
				.map_err(|_| Error::internal("AV1 order-hint bit count exceeds u8"))?
		} else {
			0
		};
		(
			enable_inter_intra_compound,
			enable_masked_compound,
			enable_warped_motion,
			enable_dual_filter,
			enable_order_hint,
			enable_joint_compound,
			enable_reference_frame_motion_vectors,
			screen_content_tools,
			integer_motion_vectors,
			order_hint_bits,
		)
	};

	let enable_superres = reader.bit("super-resolution flag")?;
	let enable_cdef = reader.bit("CDEF flag")?;
	let enable_restoration = reader.bit("restoration flag")?;
	let color = parse_color_config(&mut reader, profile)?;
	let film_grain_params_present = reader.bit("film-grain-present flag")?;

	Ok(Av1SequenceHeader {
		profile,
		still_picture,
		reduced_still_picture_header,
		timing,
		decoder_model_info_present,
		initial_display_delay_present,
		operating_points,
		frame_width_bits_minus_1,
		frame_height_bits_minus_1,
		max_frame_width_minus_1,
		max_frame_height_minus_1,
		frame_id_numbers_present,
		delta_frame_id_length_minus_2,
		additional_frame_id_length_minus_1,
		use_128x128_superblock,
		enable_filter_intra,
		enable_intra_edge_filter,
		enable_inter_intra_compound,
		enable_masked_compound,
		enable_warped_motion,
		enable_dual_filter,
		enable_order_hint,
		enable_joint_compound,
		enable_reference_frame_motion_vectors,
		screen_content_tools,
		integer_motion_vectors,
		order_hint_bits,
		enable_superres,
		enable_cdef,
		enable_restoration,
		color,
		film_grain_params_present,
	})
}

fn read_tool_choice(reader: &mut Av1BitReader<'_>, name: &str) -> Result<Av1CodingToolChoice> {
	if reader.bit(&format!("{name} selection flag"))? {
		Ok(Av1CodingToolChoice::SelectPerFrame)
	} else if reader.bit(&format!("{name} value"))? {
		Ok(Av1CodingToolChoice::Enabled)
	} else {
		Ok(Av1CodingToolChoice::Disabled)
	}
}

fn parse_color_config(
	reader: &mut Av1BitReader<'_>,
	profile: Av1Profile,
) -> Result<Av1ColorConfig> {
	let high_bit_depth = reader.bit("high-bit-depth flag")?;
	let bit_depth = if profile == Av1Profile::Professional && high_bit_depth {
		if reader.bit("twelve-bit flag")? {
			VideoComponentBitDepth::Twelve
		} else {
			VideoComponentBitDepth::Ten
		}
	} else if high_bit_depth {
		VideoComponentBitDepth::Ten
	} else {
		VideoComponentBitDepth::Eight
	};
	let monochrome = if profile == Av1Profile::High {
		false
	} else {
		reader.bit("monochrome flag")?
	};
	let color_description = if reader.bit("color-description-present flag")? {
		Some(Av1ColorDescription {
			color_primaries: u8::try_from(reader.bits(8, "color primaries")?)
				.map_err(|_| Error::internal("AV1 color primaries exceed u8"))?,
			transfer_characteristics: u8::try_from(reader.bits(8, "transfer characteristics")?)
				.map_err(|_| Error::internal("AV1 transfer characteristics exceed u8"))?,
			matrix_coefficients: u8::try_from(reader.bits(8, "matrix coefficients")?)
				.map_err(|_| Error::internal("AV1 matrix coefficients exceed u8"))?,
		})
	} else {
		None
	};

	if monochrome {
		return Ok(Av1ColorConfig {
			bit_depth,
			monochrome,
			color_description,
			full_range: reader.bit("monochrome color range")?,
			chroma_subsampling: VideoChromaSubsampling::Monochrome,
			chroma_sample_position: Av1ChromaSamplePosition::Unknown,
			separate_uv_delta_q: false,
		});
	}

	let identity_rgb = color_description.is_some_and(|color| {
		color.color_primaries == 1
			&& color.transfer_characteristics == 13
			&& color.matrix_coefficients == 0
	});
	let (full_range, subsampling_x, subsampling_y) = if identity_rgb {
		(true, false, false)
	} else {
		let full_range = reader.bit("color range")?;
		let (subsampling_x, subsampling_y) = match (profile, bit_depth) {
			(Av1Profile::Main, _) => (true, true),
			(Av1Profile::High, _) => (false, false),
			(Av1Profile::Professional, VideoComponentBitDepth::Twelve) => {
				let x = reader.bit("chroma-subsampling-X flag")?;
				let y = x && reader.bit("chroma-subsampling-Y flag")?;
				(x, y)
			}
			(Av1Profile::Professional, _) => (true, false),
		};
		(full_range, subsampling_x, subsampling_y)
	};
	let chroma_subsampling = match (subsampling_x, subsampling_y) {
		(false, false) => VideoChromaSubsampling::Yuv444,
		(true, false) => VideoChromaSubsampling::Yuv422,
		(true, true) => VideoChromaSubsampling::Yuv420,
		(false, true) => {
			return Err(Error::data_loss(
				"AV1 sequence header declares invalid chroma subsampling",
			));
		}
	};
	let chroma_sample_position = if subsampling_x && subsampling_y {
		Av1ChromaSamplePosition::from_raw(reader.bits(2, "chroma sample position")?)
	} else {
		Av1ChromaSamplePosition::Unknown
	};
	let separate_uv_delta_q = reader.bit("separate-UV-delta-Q flag")?;
	Ok(Av1ColorConfig {
		bit_depth,
		monochrome,
		color_description,
		full_range,
		chroma_subsampling,
		chroma_sample_position,
		separate_uv_delta_q,
	})
}

struct Av1BitReader<'a> {
	data: &'a [u8],
	bit_offset: usize,
	bit_len: usize,
}

impl<'a> Av1BitReader<'a> {
	fn new(data: &'a [u8]) -> Result<Self> {
		let bit_len = data
			.len()
			.checked_mul(8)
			.ok_or_else(|| Error::out_of_range("AV1 payload bit length overflowed"))?;
		Ok(Self {
			data,
			bit_offset: 0,
			bit_len,
		})
	}

	fn bit(&mut self, field: &str) -> Result<bool> {
		Ok(self.bits(1, field)? != 0)
	}

	fn bits(&mut self, count: u32, field: &str) -> Result<u32> {
		if count > 32 {
			return Err(Error::internal("AV1 bit reader request exceeds 32 bits"));
		}
		let count =
			usize::try_from(count).map_err(|_| Error::out_of_range("AV1 bit count exceeds usize"))?;
		let end = self
			.bit_offset
			.checked_add(count)
			.ok_or_else(|| Error::out_of_range("AV1 bit-reader offset overflowed"))?;
		if end > self.bit_len {
			return Err(Error::data_loss(format!("truncated AV1 {field}")));
		}
		let mut value = 0_u32;
		while self.bit_offset < end {
			let byte = self.data[self.bit_offset >> 3];
			value = (value << 1) | u32::from((byte >> (7 - (self.bit_offset & 7))) & 1);
			self.bit_offset += 1;
		}
		Ok(value)
	}

	fn skip(&mut self, mut count: u32, field: &str) -> Result<()> {
		while count > 0 {
			let chunk = count.min(32);
			self.bits(chunk, field)?;
			count -= chunk;
		}
		Ok(())
	}

	fn uvlc(&mut self, field: &str) -> Result<u32> {
		let mut leading_zeros = 0_u32;
		while leading_zeros < 32 {
			if self.bit(field)? {
				let suffix = if leading_zeros == 0 {
					0
				} else {
					self.bits(leading_zeros, field)?
				};
				return Ok(((1_u32 << leading_zeros) - 1) + suffix);
			}
			leading_zeros += 1;
		}
		Ok(u32::MAX)
	}

	fn byte_align(&mut self) -> Result<()> {
		let aligned = self
			.bit_offset
			.checked_add(7)
			.ok_or_else(|| Error::out_of_range("AV1 byte-alignment offset overflowed"))?
			& !7;
		if aligned > self.bit_len {
			return Err(Error::data_loss("truncated AV1 byte alignment"));
		}
		self.bit_offset = aligned;
		Ok(())
	}

	const fn byte_offset(&self) -> usize {
		self.bit_offset.div_ceil(8)
	}
}

/// Split one raw AV1 access-unit payload into borrowed OBUs.
///
/// # Errors
///
/// Returns an error for forbidden/reserved header bits, truncated extensions
/// or LEB128 sizes, payload overruns, empty input, and arithmetic overflow.
pub fn parse_av1_obus(bytes: &[u8]) -> Result<Vec<Av1Obu<'_>>> {
	let mut offset = 0_usize;
	let mut obus = Vec::new();
	while offset < bytes.len() {
		let header_offset = offset;
		let header = bytes[offset];
		offset += 1;
		if header & 0x81 != 0 {
			return Err(Error::data_loss(
				"invalid AV1 OBU forbidden or reserved bit",
			));
		}
		let type_ = Av1ObuType::from_raw((header >> 3) & 0x0f);
		if header & 0x04 != 0 {
			offset = offset
				.checked_add(1)
				.ok_or_else(|| Error::out_of_range("AV1 OBU extension offset overflowed"))?;
			if offset > bytes.len() {
				return Err(Error::data_loss("truncated AV1 OBU extension header"));
			}
		}
		let payload_size = if header & 0x02 != 0 {
			read_leb128(bytes, &mut offset)?
		} else {
			u64::try_from(bytes.len() - offset)
				.map_err(|_| Error::out_of_range("AV1 OBU payload size exceeds u64"))?
		};
		let payload_size = usize::try_from(payload_size)
			.map_err(|_| Error::out_of_range("AV1 OBU payload size exceeds usize"))?;
		let payload_end = offset
			.checked_add(payload_size)
			.ok_or_else(|| Error::out_of_range("AV1 OBU payload range overflowed"))?;
		let payload = bytes
			.get(offset..payload_end)
			.ok_or_else(|| Error::data_loss("AV1 OBU payload exceeds access unit"))?;
		obus
			.try_reserve(1)
			.map_err(|_| Error::resource_exhausted("AV1 OBU inventory allocation failed"))?;
		obus.push(Av1Obu {
			type_,
			header_offset,
			header_size: offset - header_offset,
			payload,
		});
		offset = payload_end;
	}
	if obus.is_empty() {
		return Err(Error::invalid_argument("AV1 access unit contains no OBUs"));
	}
	Ok(obus)
}

/// Inspect one raw access unit or the first frame of an IVF byte stream.
///
/// # Errors
///
/// Returns an error for malformed IVF framing, invalid OBU structure, count
/// overflow, or an empty frame payload.
pub fn inspect_av1_access_unit(bytes: &[u8]) -> Result<Av1AccessUnitInfo> {
	let (payload, ivf_timestamp) = av1_frame_payload(bytes)?;
	let mut info = Av1AccessUnitInfo {
		ivf_timestamp,
		..Av1AccessUnitInfo::default()
	};
	for obu in parse_av1_obus(payload)? {
		let count = match obu.type_ {
			Av1ObuType::SequenceHeader => &mut info.sequence_headers,
			Av1ObuType::Frame => &mut info.frames,
			Av1ObuType::FrameHeader => &mut info.frame_headers,
			Av1ObuType::TileGroup => &mut info.tile_groups,
			_ => continue,
		};
		*count = count
			.checked_add(1)
			.ok_or_else(|| Error::resource_exhausted("AV1 OBU count overflowed"))?;
	}
	Ok(info)
}

fn av1_frame_payload(bytes: &[u8]) -> Result<(&[u8], Option<u64>)> {
	if bytes.len() >= 44 && bytes.get(..4) == Some(b"DKIF") && bytes.get(8..12) == Some(b"AV01") {
		let header_size = usize::from(u16::from_le_bytes([bytes[6], bytes[7]]));
		if header_size < 32 {
			return Err(Error::data_loss("AV1 IVF header is shorter than 32 bytes"));
		}
		let frame_header_end = header_size
			.checked_add(12)
			.ok_or_else(|| Error::out_of_range("AV1 IVF frame-header range overflowed"))?;
		let frame_header = bytes
			.get(header_size..frame_header_end)
			.ok_or_else(|| Error::data_loss("truncated AV1 IVF frame header"))?;
		let frame_size = usize::try_from(u32::from_le_bytes([
			frame_header[0],
			frame_header[1],
			frame_header[2],
			frame_header[3],
		]))
		.map_err(|_| Error::out_of_range("AV1 IVF frame size exceeds usize"))?;
		if frame_size == 0 {
			return Err(Error::data_loss("AV1 IVF frame payload is empty"));
		}
		let timestamp = u64::from_le_bytes(
			frame_header[4..12]
				.try_into()
				.map_err(|_| Error::internal("AV1 IVF timestamp slice has wrong length"))?,
		);
		let payload_end = frame_header_end
			.checked_add(frame_size)
			.ok_or_else(|| Error::out_of_range("AV1 IVF frame payload range overflowed"))?;
		let payload = bytes
			.get(frame_header_end..payload_end)
			.ok_or_else(|| Error::data_loss("AV1 IVF frame payload exceeds input"))?;
		return Ok((payload, Some(timestamp)));
	}
	if bytes.is_empty() {
		return Err(Error::invalid_argument("AV1 access unit is empty"));
	}
	Ok((bytes, None))
}

fn read_leb128(bytes: &[u8], offset: &mut usize) -> Result<u64> {
	let mut value = 0_u64;
	for shift in (0..56).step_by(7) {
		let byte = *bytes
			.get(*offset)
			.ok_or_else(|| Error::data_loss("truncated AV1 OBU LEB128 size"))?;
		*offset = offset
			.checked_add(1)
			.ok_or_else(|| Error::out_of_range("AV1 OBU size-field offset overflowed"))?;
		value |= u64::from(byte & 0x7f) << shift;
		if byte & 0x80 == 0 {
			return Ok(value);
		}
	}
	Err(Error::data_loss("AV1 OBU LEB128 size exceeds eight bytes"))
}
