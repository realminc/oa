//! Audio - planar Float32 [channels, samples] matrix composition plus sample rate and channel layout

use crate::Result;

/// Audio represents audio data with sample rate and channel layout
pub struct Audio {
	// TODO: Add planar Float32 matrix, sample rate, channel layout
}

impl Audio {
	/// Create audio from samples
	pub fn new(_samples: Vec<f32>, _sample_rate: u32) -> Result<Self> {
		todo!("Audio::new")
	}
}
