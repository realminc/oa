//! Deterministic on-device image augmentation composed from existing OA
//! operations.
//!
//! Mirrors `audio::augmentation`: every helper composes ordinary semantic
//! operations on the device, so an augmentation chain can be captured into
//! one execution plan like any other OA graph. Recipe decisions — the flip
//! coin, crop offsets, and rotation selection — are derived from `seed` on
//! the host so the pixel work stays a pure device-side composition that
//! replays exactly across runs and languages.

use crate::audio::NoiseDistribution;
use crate::image::{self, Image};
use crate::{Error, Result, matrix};

/// Generate deterministic Philox noise matching `input`'s shape and layout.
///
/// The same `(seed, distribution, amplitude)` triple always produces the same
/// noise tensor. The result carries `input`'s layout and format so it can be
/// mixed back into the image or an augmentation chain.
///
/// # Errors
///
/// Returns an error when `amplitude` is negative or non-finite, or noise
/// generation/allocation fails.
pub fn noise_like(
	input: &Image,
	distribution: NoiseDistribution,
	amplitude: f32,
	seed: u64,
) -> Result<Image> {
	if !(amplitude.is_finite() && amplitude >= 0.0) {
		return Err(Error::invalid_argument(
			"image::noise_like amplitude must be finite and nonnegative",
		));
	}
	let noise = match distribution {
		NoiseDistribution::Uniform => {
			matrix::philox_uniform(input.as_matrix(), -amplitude, amplitude, seed)?
		}
		NoiseDistribution::Gaussian => matrix::philox_normal(input.as_matrix(), 0.0, amplitude, seed)?,
	};
	Image::new(noise, input.layout(), input.format())
}

/// Randomly mirror `input` horizontally and/or vertically.
///
/// `p_horizontal` and `p_vertical` are independent probabilities in `[0, 1]`.
/// Both coin draws are derived from `seed`, so the same recipe flips the same
/// way on every run.
///
/// # Errors
///
/// Returns an error when either probability is outside `[0, 1]` or
/// non-finite, or the flip fails.
pub fn random_flip(input: &Image, p_horizontal: f32, p_vertical: f32, seed: u64) -> Result<Image> {
	if !probability(p_horizontal) || !probability(p_vertical) {
		return Err(Error::invalid_argument(
			"image::random_flip probabilities must be finite and within [0, 1]",
		));
	}
	let mut state = seed;
	let horizontal = next_f32(&mut state) < p_horizontal;
	let vertical = next_f32(&mut state) < p_vertical;
	if !horizontal && !vertical {
		return Ok(input.clone());
	}
	image::flip(input, horizontal, vertical)
}

/// Crop a `width`×`height` window at a seed-derived uniform offset.
///
/// The offset is drawn uniformly over the valid top-left range from `seed`,
/// keeping the crop itself a single recorded device operation.
///
/// # Errors
///
/// Returns an error when the window is zero-sized or exceeds `input`, or the
/// crop fails.
pub fn random_crop(input: &Image, width: u32, height: u32, seed: u64) -> Result<Image> {
	if width == 0 || height == 0 {
		return Err(Error::invalid_argument(
			"image::random_crop window must be nonzero",
		));
	}
	let (input_width, input_height) = (input.width(), input.height());
	if width as usize > input_width || height as usize > input_height {
		return Err(Error::invalid_argument(
			"image::random_crop window exceeds the input extent",
		));
	}
	let mut state = seed;
	let x = next_bounded(&mut state, (input_width - width as usize) as u64 + 1) as u32;
	let y = next_bounded(&mut state, (input_height - height as usize) as u64 + 1) as u32;
	image::crop(input, x, y, width, height)
}

/// Rotate `input` by a seed-derived quarter-turn: 0°, 90°, 180°, or 270°.
///
/// # Errors
///
/// Returns the rotation error when dispatch fails.
pub fn random_rotate(input: &Image, seed: u64) -> Result<Image> {
	let mut state = seed;
	let degrees = (next_bounded(&mut state, 4) * 90) as u32;
	image::rotate(input, degrees)
}

fn probability(value: f32) -> bool {
	value.is_finite() && (0.0..=1.0).contains(&value)
}

/// SplitMix64 recipe generator: host-side, deterministic, no device readback.
fn next_u64(state: &mut u64) -> u64 {
	*state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
	let mut z = *state;
	z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
	z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
	z ^ (z >> 31)
}

fn next_bounded(state: &mut u64, bound: u64) -> u64 {
	next_u64(state) % bound
}

fn next_f32(state: &mut u64) -> f32 {
	(next_u64(state) >> 40) as f32 / (1 << 24) as f32
}
