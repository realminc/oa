//! Audio analysis cache. Donor: `oa::Viewer::openAudio` analysis section.

use super::config::ViewerConfig;
use crate::{Engine, Error, Matrix};

/// Pre-computed offline audio analysis matrices.
pub struct AudioAnalysis {
	pub envelope: Option<Matrix>,
	pub spectrum: Option<Matrix>,
	pub mel: Option<Matrix>,
}

/// Analyze an audio file and return cached matrices.
///
/// Uses the viewer's execution owner; analysis failures remain visible.
/// Donor: `oa/ui/viewer.cpp::openAudio` and `validateAudioAnalysisConfig`.
pub fn analyze(
	engine: &Engine,
	path: &str,
	config: &ViewerConfig,
) -> crate::Result<Option<AudioAnalysis>> {
	if !(1..=65_536).contains(&config.audio_waveform_bins)
		|| !(2..=65_536).contains(&config.audio_analysis_frames)
		|| !config.audio_fft_size.is_power_of_two()
		|| !(16..=1_024).contains(&config.audio_fft_size)
		|| config.audio_hop_size == 0
		|| !(1..=4_096).contains(&config.audio_mel_bins)
	{
		return Err(Error::invalid_argument(
			"invalid Viewer audio analysis configuration",
		));
	}
	let decoded = crate::audio::decode_file(engine, path)?;
	let samples = u32::try_from(decoded.samples())
		.map_err(|_| Error::out_of_range("Viewer audio sample count exceeds u32"))?;
	if samples == 0 {
		return Err(Error::invalid_argument("Viewer audio source is empty"));
	}
	let hop = config
		.audio_hop_size
		.max(samples.div_ceil(config.audio_analysis_frames - 1));
	let envelope = crate::audio::waveform_envelope(&decoded, config.audio_waveform_bins)?;
	let stft = crate::audio::stft(
		&decoded,
		crate::audio::StftConfig {
			fft_size: config.audio_fft_size,
			hop_size: hop,
			win_size: config.audio_fft_size,
			..Default::default()
		},
	)?;
	// Donor channel mean, then log-magnitude and frequency-major snapshot.
	let channel_scale = 1.0 / decoded.channels() as f32;
	let spectrum = crate::matrix::scale(&crate::matrix::sum(&stft, 0)?, channel_scale)?.squeeze(0)?;
	let spectrum = crate::matrix::log(&crate::matrix::add_scalar(&spectrum, 1.0e-6)?)?;
	let spectrum = crate::matrix::transpose(&spectrum, 0, 1)?;
	let mel = crate::audio::mel_spectrogram(
		&decoded,
		crate::audio::MelConfig {
			fft_size: config.audio_fft_size,
			hop_size: hop,
			num_mels: config.audio_mel_bins,
			log_scale: true,
			..Default::default()
		},
	)?;
	let mel = crate::matrix::scale(&crate::matrix::sum(&mel, 0)?, channel_scale)?.squeeze(0)?;
	Ok(Some(AudioAnalysis {
		envelope: Some(envelope),
		spectrum: Some(spectrum),
		mel: Some(mel),
	}))
}
