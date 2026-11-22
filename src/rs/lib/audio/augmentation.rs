//! Deterministic on-device augmentation composed from existing OA operations.
//!
//! These helpers keep every intermediate on the device: noise generation,
//! level measurement, scaling, and mixing all record ordinary semantic
//! operations, so an augmentation chain can be captured into one execution
//! plan like any other OA graph.

use crate::{Audio, Error, Result, matrix};

use super::audio::AudioChannelLayout;

/// Noise distribution used by [`noise_like`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum NoiseDistribution {
	/// Uniform samples in `[-amplitude, amplitude]`.
	Uniform,
	/// Zero-mean Gaussian samples with `amplitude` standard deviation.
	#[default]
	Gaussian,
}

/// Generate deterministic Philox noise matching `input`'s shape, rate, and layout.
///
/// The same `(seed, distribution, amplitude)` triple always produces the same
/// samples, which keeps augmentation recipes replayable across runs and
/// languages.
///
/// # Errors
///
/// Returns an error when `amplitude` is negative or non-finite, or noise
/// generation/allocation fails.
pub fn noise_like(
	input: &Audio,
	distribution: NoiseDistribution,
	amplitude: f32,
	seed: u64,
) -> Result<Audio> {
	if !(amplitude.is_finite() && amplitude >= 0.0) {
		return Err(Error::invalid_argument(
			"audio::noise_like amplitude must be finite and nonnegative",
		));
	}
	let matrix = match distribution {
		NoiseDistribution::Uniform => {
			matrix::philox_uniform(input.as_matrix(), -amplitude, amplitude, seed)?
		}
		NoiseDistribution::Gaussian => matrix::philox_normal(input.as_matrix(), 0.0, amplitude, seed)?,
	};
	Audio::new(matrix, input.sample_rate(), input.layout())
}

/// Mix `noise` into `signal` at `snr_db` decibels signal-to-noise ratio.
///
/// `noise` is scaled so its RMS sits `snr_db` below `signal`'s RMS before the
/// two are summed. A `1e-8` floor on the noise RMS keeps zero-valued noise
/// inputs finite. Both inputs must share channel count, sample count, sample
/// rate, and layout.
///
/// # Errors
///
/// Returns an error when `snr_db` is non-finite, the inputs' audio contract
/// differs, or an intermediate operation fails.
pub fn mix_snr(signal: &Audio, noise: &Audio, snr_db: f32) -> Result<Audio> {
	if !snr_db.is_finite() {
		return Err(Error::invalid_argument(
			"audio::mix_snr snr_db must be finite",
		));
	}
	if signal.channels() != noise.channels()
		|| signal.samples() != noise.samples()
		|| signal.sample_rate() != noise.sample_rate()
		|| signal.layout() != noise.layout()
	{
		return Err(Error::invalid_argument(
			"audio::mix_snr inputs must share channels, samples, rate, and layout",
		));
	}
	let signal_matrix = signal.as_matrix();
	let noise_matrix = noise.as_matrix();
	let signal_power = matrix::mean(&matrix::mul(signal_matrix, signal_matrix)?, -1)?;
	let noise_power = matrix::mean(&matrix::mul(noise_matrix, noise_matrix)?, -1)?;
	// RMS floor keeps zero-valued noise finite instead of producing inf/NaN.
	let gain = matrix::mul(
		&matrix::div(
			&matrix::sqrt(&signal_power)?,
			&matrix::sqrt(&matrix::add_scalar(&noise_power, 1.0e-16)?)?,
		)?,
		&matrix::full([1], 10.0_f32.powf(-snr_db / 20.0))?,
	)?;
	let mixed = matrix::add(signal_matrix, &matrix::mul(noise_matrix, &gain)?)?;
	Audio::new(
		mixed,
		signal.sample_rate(),
		AudioChannelLayout::for_channel_count(signal.channels()),
	)
}
