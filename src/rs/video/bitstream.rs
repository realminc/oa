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
		zeros = if byte == 0 { zeros + 1 } else { 0 };
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
		if count > 32
			|| self
				.bit
				.checked_add(count)
				.is_none_or(|end| end > self.data.len() * 8)
		{
			return Err(Error::data_loss("truncated codec bitstream"));
		}
		let mut value = 0_u32;
		for _ in 0..count {
			value = (value << 1) | u32::from((self.data[self.bit / 8] >> (7 - self.bit % 8)) & 1);
			self.bit += 1;
		}
		Ok(value)
	}

	pub(super) fn skip_bits(&mut self, mut count: usize) -> Result<()> {
		while count != 0 {
			let chunk = count.min(32);
			self.read_bits(chunk)?;
			count -= chunk;
		}
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
		let stop_bit = byte_index * 8 + (u8::BITS - 1 - byte.trailing_zeros()) as usize;
		self.bit < stop_bit
	}
}

#[cfg(test)]
mod tests {
	use super::BitReader;

	#[test]
	fn rbsp_data_ends_at_the_last_set_bit() -> crate::Result<()> {
		let mut reader = BitReader::new(&[0b1010_0000]);
		assert!(reader.more_rbsp_data());
		reader.read_bits(2)?;
		assert!(!reader.more_rbsp_data());

		let mut final_byte_stop = BitReader::new(&[0b1100_1011]);
		final_byte_stop.read_bits(7)?;
		assert!(!final_byte_stop.more_rbsp_data());
		Ok(())
	}
}
