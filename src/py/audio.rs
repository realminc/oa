use pyo3::{exceptions::PyValueError, prelude::*, types::PyBytes};

use crate::{error::python_error, matrix::PythonMatrix, runtime::PythonEngine};

mod sessions;

fn layout_from_token(token: &str) -> PyResult<oa::audio::AudioChannelLayout> {
	match token {
		"mono" => Ok(oa::audio::AudioChannelLayout::Mono),
		"stereo" => Ok(oa::audio::AudioChannelLayout::Stereo),
		"stereo_2_1" => Ok(oa::audio::AudioChannelLayout::Stereo21),
		"surround_5_1" => Ok(oa::audio::AudioChannelLayout::Surround51),
		"surround_7_1" => Ok(oa::audio::AudioChannelLayout::Surround71),
		"unknown" => Ok(oa::audio::AudioChannelLayout::Unknown),
		_ => Err(PyValueError::new_err(format!(
			"unknown audio channel layout: {token}"
		))),
	}
}

fn layout_token(layout: oa::audio::AudioChannelLayout) -> &'static str {
	match layout {
		oa::audio::AudioChannelLayout::Mono => "mono",
		oa::audio::AudioChannelLayout::Stereo => "stereo",
		oa::audio::AudioChannelLayout::Stereo21 => "stereo_2_1",
		oa::audio::AudioChannelLayout::Surround51 => "surround_5_1",
		oa::audio::AudioChannelLayout::Surround71 => "surround_7_1",
		oa::audio::AudioChannelLayout::Unknown => "unknown",
	}
}

#[pyclass(name = "Audio", unsendable)]
pub(crate) struct PythonAudio {
	inner: oa::Audio,
}

impl PythonAudio {
	fn wrap(inner: oa::Audio) -> Self {
		Self { inner }
	}
}

#[pymethods]
impl PythonAudio {
	#[staticmethod]
	#[pyo3(signature = (engine, samples, channels, sample_rate, layout="unknown"))]
	fn from_planar_f32(
		engine: &PythonEngine,
		samples: Vec<f32>,
		channels: usize,
		sample_rate: u32,
		layout: &str,
	) -> PyResult<Self> {
		oa::Audio::from_planar_f32(
			&engine.inner,
			&samples,
			channels,
			sample_rate,
			layout_from_token(layout)?,
		)
		.map(Self::wrap)
		.map_err(python_error)
	}

	#[getter]
	fn channels(&self) -> usize {
		self.inner.channels()
	}

	#[getter]
	fn samples(&self) -> usize {
		self.inner.samples()
	}

	#[getter]
	fn sample_rate(&self) -> u32 {
		self.inner.sample_rate()
	}

	#[getter]
	fn layout(&self) -> &'static str {
		layout_token(self.inner.layout())
	}

	#[getter]
	fn duration_seconds(&self) -> f64 {
		self.inner.duration_seconds()
	}

	fn as_matrix(&self) -> PythonMatrix {
		PythonMatrix::wrap(self.inner.as_matrix().clone())
	}

	fn __repr__(&self) -> String {
		format!(
			"Audio(channels={}, samples={}, sample_rate={}, layout='{}')",
			self.inner.channels(),
			self.inner.samples(),
			self.inner.sample_rate(),
			layout_token(self.inner.layout()),
		)
	}
}

macro_rules! unary_audio {
	($binding:ident, $operation:path) => {
		#[pyfunction]
		pub(crate) fn $binding(input: &PythonAudio) -> PyResult<PythonAudio> {
			$operation(&input.inner)
				.map(PythonAudio::wrap)
				.map_err(python_error)
		}
	};
}

unary_audio!(audio_to_mono, oa::audio::to_mono);

#[pyfunction]
pub(crate) fn audio_decode_file(engine: &PythonEngine, path: &str) -> PyResult<PythonAudio> {
	oa::audio::decode_file(&engine.inner, path)
		.map(PythonAudio::wrap)
		.map_err(python_error)
}

#[pyfunction]
pub(crate) fn audio_decode_memory(
	engine: &PythonEngine,
	encoded: Vec<u8>,
) -> PyResult<PythonAudio> {
	oa::audio::decode_memory(&engine.inner, &encoded)
		.map(PythonAudio::wrap)
		.map_err(python_error)
}

#[pyfunction]
pub(crate) fn audio_encode_wav_f32(py: Python<'_>, input: &PythonAudio) -> PyResult<Py<PyBytes>> {
	let encoded = oa::audio::encode_wav_f32(&input.inner).map_err(python_error)?;
	Ok(PyBytes::new(py, &encoded).unbind())
}

#[pyfunction]
pub(crate) fn audio_save_wav_f32(path: &str, input: &PythonAudio) -> PyResult<()> {
	oa::audio::save_wav_f32(path, &input.inner).map_err(python_error)
}

#[pyfunction]
#[pyo3(signature = (input, target_db=-3.0, mode="peak"))]
pub(crate) fn audio_normalize(
	input: &PythonAudio,
	target_db: f32,
	mode: &str,
) -> PyResult<PythonAudio> {
	let mode = match mode {
		"peak" => oa::audio::NormalizeAudioMode::Peak,
		"rms" => oa::audio::NormalizeAudioMode::Rms,
		_ => {
			return Err(PyValueError::new_err(
				"normalize mode must be 'peak' or 'rms'",
			));
		}
	};
	oa::audio::normalize(
		&input.inner,
		oa::audio::NormalizeAudioConfig { target_db, mode },
	)
	.map(PythonAudio::wrap)
	.map_err(python_error)
}

#[pyfunction]
#[pyo3(signature = (input, out_rate, filter_half_width=64))]
pub(crate) fn audio_resample(
	input: &PythonAudio,
	out_rate: u32,
	filter_half_width: u32,
) -> PyResult<PythonAudio> {
	oa::audio::resample(
		&input.inner,
		oa::audio::ResampleConfig {
			out_rate,
			filter_half_width,
		},
	)
	.map(PythonAudio::wrap)
	.map_err(python_error)
}

macro_rules! one_f32_audio {
	($binding:ident, $operation:path, $argument:ident) => {
		#[pyfunction]
		pub(crate) fn $binding(input: &PythonAudio, $argument: f32) -> PyResult<PythonAudio> {
			$operation(&input.inner, $argument)
				.map(PythonAudio::wrap)
				.map_err(python_error)
		}
	};
}

one_f32_audio!(audio_gain, oa::audio::gain, gain_db);
one_f32_audio!(audio_pre_emphasis, oa::audio::pre_emphasis, alpha);

#[pyfunction]
pub(crate) fn audio_clip(input: &PythonAudio, minimum: f32, maximum: f32) -> PyResult<PythonAudio> {
	oa::audio::clip(&input.inner, minimum, maximum)
		.map(PythonAudio::wrap)
		.map_err(python_error)
}

#[pyfunction]
pub(crate) fn audio_saturate(
	input: &PythonAudio,
	drive_db: f32,
	mix: f32,
) -> PyResult<PythonAudio> {
	oa::audio::saturate(&input.inner, drive_db, mix)
		.map(PythonAudio::wrap)
		.map_err(python_error)
}

#[pyfunction]
pub(crate) fn audio_reverb(
	input: &PythonAudio,
	decay_seconds: f32,
	wet: f32,
) -> PyResult<PythonAudio> {
	oa::audio::reverb(&input.inner, decay_seconds, wet)
		.map(PythonAudio::wrap)
		.map_err(python_error)
}

#[pyfunction]
pub(crate) fn audio_fade(
	input: &PythonAudio,
	fade_in_samples: u64,
	fade_out_samples: u64,
) -> PyResult<PythonAudio> {
	oa::audio::fade(&input.inner, fade_in_samples, fade_out_samples)
		.map(PythonAudio::wrap)
		.map_err(python_error)
}

#[pyfunction]
pub(crate) fn audio_mix(
	a: &PythonAudio,
	b: &PythonAudio,
	gain_a: f32,
	gain_b: f32,
) -> PyResult<PythonAudio> {
	oa::audio::mix(&a.inner, &b.inner, gain_a, gain_b)
		.map(PythonAudio::wrap)
		.map_err(python_error)
}

#[pyfunction]
pub(crate) fn audio_amplitude_to_db(input: &PythonAudio, floor_db: f32) -> PyResult<PythonMatrix> {
	oa::audio::amplitude_to_db(&input.inner, floor_db)
		.map(PythonMatrix::wrap)
		.map_err(python_error)
}

#[pyfunction]
pub(crate) fn audio_waveform_envelope(input: &PythonAudio, bins: u32) -> PyResult<PythonMatrix> {
	oa::audio::waveform_envelope(&input.inner, bins)
		.map(PythonMatrix::wrap)
		.map_err(python_error)
}

#[pyfunction]
#[pyo3(signature = (input, fft_size=1024, hop_size=256, win_size=0, window="hann", center=true))]
pub(crate) fn audio_stft(
	input: &PythonAudio,
	fft_size: u32,
	hop_size: u32,
	win_size: u32,
	window: &str,
	center: bool,
) -> PyResult<PythonMatrix> {
	let window = match window {
		"hann" => oa::audio::StftWindow::Hann,
		"hamming" => oa::audio::StftWindow::Hamming,
		"blackman" => oa::audio::StftWindow::Blackman,
		"rectangular" => oa::audio::StftWindow::Rectangular,
		_ => return Err(PyValueError::new_err("unknown STFT window")),
	};
	oa::audio::stft(
		&input.inner,
		oa::audio::StftConfig {
			fft_size,
			hop_size,
			win_size,
			window,
			center,
		},
	)
	.map(PythonMatrix::wrap)
	.map_err(python_error)
}

#[pyfunction]
#[pyo3(signature = (input, fft_size=1024, hop_size=256, num_mels=80, f_min=0.0, f_max=0.0, log_scale=true, normalize=false))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn audio_mel_spectrogram(
	input: &PythonAudio,
	fft_size: u32,
	hop_size: u32,
	num_mels: u32,
	f_min: f32,
	f_max: f32,
	log_scale: bool,
	normalize: bool,
) -> PyResult<PythonMatrix> {
	let config = oa::audio::MelConfig {
		fft_size,
		hop_size,
		num_mels,
		f_min,
		f_max,
		log_scale,
		normalize,
	};
	oa::audio::mel_spectrogram(&input.inner, config)
		.map(PythonMatrix::wrap)
		.map_err(python_error)
}

#[pyfunction]
#[pyo3(signature = (input, num_coeffs=13, num_mels=80, fft_size=1024, hop_size=256, f_min=0.0, f_max=0.0, log_scale=true, normalize=false))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn audio_mfcc(
	input: &PythonAudio,
	num_coeffs: u32,
	num_mels: u32,
	fft_size: u32,
	hop_size: u32,
	f_min: f32,
	f_max: f32,
	log_scale: bool,
	normalize: bool,
) -> PyResult<PythonMatrix> {
	let config = oa::audio::MfccConfig {
		num_coeffs,
		mel: oa::audio::MelConfig {
			fft_size,
			hop_size,
			num_mels,
			f_min,
			f_max,
			log_scale,
			normalize,
		},
	};
	oa::audio::mfcc(&input.inner, config)
		.map(PythonMatrix::wrap)
		.map_err(python_error)
}

#[pyfunction]
pub(crate) fn audio_encode_interleaved_wav_f32(
	py: Python<'_>,
	samples: Vec<f32>,
	sample_rate: u32,
	channels: usize,
) -> PyResult<Py<PyBytes>> {
	let encoded =
		oa::audio::encode_interleaved_wav_f32(&samples, sample_rate, channels).map_err(python_error)?;
	Ok(PyBytes::new(py, &encoded).unbind())
}

#[pyfunction]
pub(crate) fn audio_biquad(
	input: &PythonAudio,
	b0: f32,
	b1: f32,
	b2: f32,
	a1: f32,
	a2: f32,
) -> PyResult<PythonAudio> {
	let coefficients = oa::audio::BiquadCoefficients { b0, b1, b2, a1, a2 };
	oa::audio::biquad(&input.inner, coefficients)
		.map(PythonAudio::wrap)
		.map_err(python_error)
}

#[pyfunction]
pub(crate) fn audio_sos_filter(
	input: &PythonAudio,
	sections: Vec<[f32; 5]>,
) -> PyResult<PythonAudio> {
	let coefficients = sections
		.into_iter()
		.map(|s| oa::audio::BiquadCoefficients {
			b0: s[0],
			b1: s[1],
			b2: s[2],
			a1: s[3],
			a2: s[4],
		})
		.collect::<Vec<_>>();
	oa::audio::sos_filter(&input.inner, &coefficients)
		.map(PythonAudio::wrap)
		.map_err(python_error)
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
	sessions::register(module)?;
	module.add_class::<PythonAudio>()?;
	macro_rules! add_functions {
		($($function:ident),+ $(,)?) => { $(module.add_function(wrap_pyfunction!($function, module)?)?;)+ };
	}
	add_functions!(
		audio_decode_file,
		audio_decode_memory,
		audio_encode_wav_f32,
		audio_encode_interleaved_wav_f32,
		audio_save_wav_f32,
		audio_normalize,
		audio_resample,
		audio_gain,
		audio_clip,
		audio_saturate,
		audio_reverb,
		audio_pre_emphasis,
		audio_to_mono,
		audio_fade,
		audio_mix,
		audio_amplitude_to_db,
		audio_waveform_envelope,
		audio_stft,
		audio_mel_spectrogram,
		audio_mfcc,
		audio_biquad,
		audio_sos_filter,
	);
	Ok(())
}
