use crate::{Error, Result};

pub(super) fn remove_emulation_prevention(data: &[u8]) -> Vec<u8> {
	let mut output = Vec::with_capacity(data.len());
	let mut zeros = 0;
	for &byte in data {
		if zeros >= 2 && byte == 3 {
			zeros = 0;
			continue;
		}
		output.push(byte);
		zeros = if byte == 0 { (zeros + 1).min(2) } else { 0 };
	}
	output
}

pub(super) struct BitReader<'a> {
	data: &'a [u8],
	bit: usize,
}

impl<'a> BitReader<'a> {
	pub(super) const fn new(data: &'a [u8]) -> Self {
		Self { data, bit: 0 }
	}

	pub(super) const fn position(&self) -> usize {
		self.bit
	}

	pub(super) fn read_bits(&mut self, count: usize) -> Result<u32> {
		if count > 32 {
			return Err(Error::data_loss("codec bit field exceeds u32"));
		}
		let start = self.bit;
		self.skip_bits(count)?;
		let mut value = 0_u32;
		for bit in start..self.bit {
			value = (value << 1) | u32::from((self.data[bit / 8] >> (7 - bit % 8)) & 1);
		}
		Ok(value)
	}

	pub(super) fn skip_bits(&mut self, count: usize) -> Result<()> {
		self.bit = self
			.bit
			.checked_add(count)
			.filter(|end| end.div_ceil(8) <= self.data.len())
			.ok_or_else(|| Error::data_loss("truncated codec bitstream"))?;
		Ok(())
	}

	pub(super) fn read_ue(&mut self) -> Result<u32> {
		let mut leading_zeros = 0_usize;
		while self.read_bits(1)? == 0 {
			leading_zeros += 1;
			if leading_zeros > 31 {
				return Err(Error::data_loss("Exp-Golomb value exceeds u32"));
			}
		}
		let suffix = self.read_bits(leading_zeros)?;
		Ok(((1_u32 << leading_zeros) - 1) + suffix)
	}

	pub(super) fn read_se(&mut self) -> Result<i32> {
		let code = self.read_ue()?;
		let magnitude = i32::try_from(code.div_ceil(2))
			.map_err(|_| Error::data_loss("signed Exp-Golomb value exceeds i32"))?;
		Ok(if code.is_multiple_of(2) {
			-magnitude
		} else {
			magnitude
		})
	}

	pub(super) fn more_rbsp_data(&self) -> bool {
		let Some((byte_index, byte)) = self
			.data
			.iter()
			.enumerate()
			.rev()
			.find(|(_, byte)| **byte != 0)
		else {
			return false;
		};
		let stop_in_byte = (u8::BITS - 1 - byte.trailing_zeros()) as usize;
		(self.bit / 8, self.bit % 8) < (byte_index, stop_in_byte)
	}
}
