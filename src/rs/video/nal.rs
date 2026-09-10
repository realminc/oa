//! CPU-only Annex-B NAL operations ported from `oa::FnVideo`.

use crate::{Error, Result};

const H264_SPS: u8 = 7;
const H264_PPS: u8 = 8;
const H265_VPS: u8 = 32;
const H265_SPS: u8 = 33;
const H265_PPS: u8 = 34;
const ANNEX_B_START_CODE: [u8; 4] = [0, 0, 0, 1];

/// Borrowed H.264 NAL payload and decoded header fields.
///
/// The payload starts at the NAL header byte and aliases the input Annex-B
/// buffer. It does not include the three- or four-byte start code.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NalUnit<'a> {
	nal_unit_type: u8,
	reference_idc: u8,
	payload: &'a [u8],
	start_code_offset: Option<usize>,
	start_code_len: usize,
}

impl<'a> NalUnit<'a> {
	/// Interpret a non-empty payload using the H.264 one-byte NAL header.
	///
	/// # Errors
	///
	/// Returns an error when `payload` is empty.
	pub fn from_h264_payload(payload: &'a [u8]) -> Result<Self> {
		let Some(&header) = payload.first() else {
			return Err(Error::invalid_argument("H.264 NAL payload is empty"));
		};
		Ok(Self {
			nal_unit_type: header & 0x1f,
			reference_idc: (header >> 5) & 0x03,
			payload,
			start_code_offset: None,
			start_code_len: 0,
		})
	}

	/// Return the H.264 `nal_unit_type` header field.
	pub const fn nal_unit_type(self) -> u8 {
		self.nal_unit_type
	}

	/// Return the H.264 `nal_ref_idc` header field.
	pub const fn reference_idc(self) -> u8 {
		self.reference_idc
	}

	/// Return the borrowed payload including its NAL header byte.
	pub const fn payload(self) -> &'a [u8] {
		self.payload
	}

	/// Return the source Annex-B start-code byte offset, when parsed from a stream.
	pub const fn start_code_offset(self) -> Option<usize> {
		self.start_code_offset
	}

	/// Return the source Annex-B start-code width, or zero for a raw payload.
	pub const fn start_code_len(self) -> usize {
		self.start_code_len
	}
}

/// Split an Annex-B byte stream into borrowed H.264 NAL payloads.
///
/// Both `00 00 01` and `00 00 00 01` start codes are recognized. Bytes before
/// the first start code and empty units are ignored, matching the donor
/// `oa::FnVideo::parseNalAnnexB` contract.
pub fn parse_nal_annex_b(bytes: &[u8]) -> Vec<NalUnit<'_>> {
	let mut units = Vec::new();
	let Some((mut current, mut prefix_len)) = find_start_code(bytes, 0) else {
		return units;
	};

	loop {
		let payload_start = current + prefix_len;
		let next = find_start_code(bytes, payload_start);
		let payload_end = next.map_or(bytes.len(), |(offset, _)| offset);
		if let Ok(mut unit) = NalUnit::from_h264_payload(&bytes[payload_start..payload_end]) {
			unit.start_code_offset = Some(current);
			unit.start_code_len = prefix_len;
			units.push(unit);
		}
		let Some((next_offset, next_prefix_len)) = next else {
			break;
		};
		current = next_offset;
		prefix_len = next_prefix_len;
	}

	units
}

/// Emit borrowed NAL payloads as canonical four-byte-start-code Annex-B.
///
/// # Errors
///
/// Returns an error when the output size overflows or host allocation fails.
pub fn emit_nal_annex_b(units: &[NalUnit<'_>]) -> Result<Vec<u8>> {
	let total = units.iter().try_fold(0_usize, |total, unit| {
		total
			.checked_add(ANNEX_B_START_CODE.len())
			.and_then(|total| total.checked_add(unit.payload.len()))
			.ok_or_else(|| Error::invalid_argument("Annex-B output size overflows usize"))
	})?;
	let mut output = Vec::new();
	output
		.try_reserve_exact(total)
		.map_err(|_| Error::resource_exhausted("Annex-B output allocation failed"))?;
	for unit in units {
		output.extend_from_slice(&ANNEX_B_START_CODE);
		output.extend_from_slice(unit.payload);
	}
	Ok(output)
}

/// Copy the first H.264 sequence parameter set payload.
pub fn extract_sps(bytes: &[u8]) -> Option<Vec<u8>> {
	extract_first_h264(bytes, H264_SPS)
}

/// Copy the first H.264 picture parameter set payload.
pub fn extract_pps(bytes: &[u8]) -> Option<Vec<u8>> {
	extract_first_h264(bytes, H264_PPS)
}

/// Copy the first H.265 video parameter set payload.
pub fn extract_vps_h265(bytes: &[u8]) -> Option<Vec<u8>> {
	extract_first_h265(bytes, H265_VPS)
}

/// Copy the first H.265 sequence parameter set payload.
pub fn extract_sps_h265(bytes: &[u8]) -> Option<Vec<u8>> {
	extract_first_h265(bytes, H265_SPS)
}

/// Copy the first H.265 picture parameter set payload.
pub fn extract_pps_h265(bytes: &[u8]) -> Option<Vec<u8>> {
	extract_first_h265(bytes, H265_PPS)
}

fn find_start_code(bytes: &[u8], offset: usize) -> Option<(usize, usize)> {
	let mut index = offset;
	while index + 2 < bytes.len() {
		if bytes[index] == 0 && bytes[index + 1] == 0 {
			if bytes[index + 2] == 1 {
				return Some((index, 3));
			}
			if index + 3 < bytes.len() && bytes[index + 2] == 0 && bytes[index + 3] == 1 {
				return Some((index, 4));
			}
		}
		index += 1;
	}
	None
}

fn extract_first_h264(bytes: &[u8], nal_unit_type: u8) -> Option<Vec<u8>> {
	parse_nal_annex_b(bytes)
		.into_iter()
		.find(|unit| unit.nal_unit_type == nal_unit_type)
		.map(|unit| unit.payload.to_vec())
}

fn extract_first_h265(bytes: &[u8], nal_unit_type: u8) -> Option<Vec<u8>> {
	let (mut current, mut prefix_len) = find_start_code(bytes, 0)?;
	loop {
		let payload_start = current + prefix_len;
		let next = find_start_code(bytes, payload_start);
		let mut payload_end = next.map_or(bytes.len(), |(offset, _)| offset);
		while payload_end > payload_start && bytes[payload_end - 1] == 0 {
			payload_end -= 1;
		}
		let payload = &bytes[payload_start..payload_end];
		if payload.len() >= 2 && ((payload[0] >> 1) & 0x3f) == nal_unit_type {
			return Some(payload.to_vec());
		}
		let (next_offset, next_prefix_len) = next?;
		current = next_offset;
		prefix_len = next_prefix_len;
	}
}
