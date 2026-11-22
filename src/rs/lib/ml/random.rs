struct SplitMix64 {
	state: u64,
}

impl SplitMix64 {
	const fn new(seed: u64) -> Self {
		Self { state: seed }
	}

	fn next_f32(&mut self) -> f32 {
		self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
		let mut value = self.state;
		value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
		value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
		value ^= value >> 31;
		let mantissa = (value >> 40) as u32;
		mantissa as f32 / 16_777_216.0
	}
}

pub(super) fn symmetric_uniform(length: usize, limit: f32, seed: u64) -> Vec<f32> {
	let mut random = SplitMix64::new(seed);
	(0..length)
		.map(|_| (random.next_f32() * 2.0 - 1.0) * limit)
		.collect()
}

pub(super) fn unit_uniform(length: usize, seed: u64) -> Vec<f32> {
	let mut random = SplitMix64::new(seed);
	(0..length).map(|_| random.next_f32()).collect()
}
