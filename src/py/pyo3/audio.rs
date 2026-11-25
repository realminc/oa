use pyo3::{exceptions::PyValueError, prelude::*, types::PyBytes};

use crate::{error::python_error, matrix::PythonMatrix, runtime::PythonRuntime};

#[path = "audio/audio.gen.rs"]
mod generated;
pub(crate) mod sessions;

fn normalize_mode(token: &str) -> PyResult<oa::audio::NormalizeAudioMode> {
	match token {
		"peak" => Ok(oa::audio::NormalizeAudioMode::Peak),
		"rms" => Ok(oa::audio::NormalizeAudioMode::Rms),
		_ => Err(PyValueError::new_err(
			"normalize mode must be 'peak' or 'rms'",
		)),
	}
}

fn noise_distribution(token: &str) -> PyResult<oa::audio::NoiseDistribution> {
	match token {
		"uniform" => Ok(oa::audio::NoiseDistribution::Uniform),
		"gaussian" => Ok(oa::audio::NoiseDistribution::Gaussian),
		_ => Err(PyValueError::new_err(
			"noise distribution must be 'uniform' or 'gaussian'",
		)),
	}
}

fn stft_window(token: &str) -> PyResult<oa::audio::StftWindow> {
	match token {
		"hann" => Ok(oa::audio::StftWindow::Hann),
		"hamming" => Ok(oa::audio::StftWindow::Hamming),
		"blackman" => Ok(oa::audio::StftWindow::Blackman),
		"rectangular" => Ok(oa::audio::StftWindow::Rectangular),
		_ => Err(PyValueError::new_err("unknown STFT window")),
	}
}

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
	#[pyo3(signature = (samples, channels, sample_rate, layout="unknown", runtime=None))]
	fn new_from_planar_f32(
		samples: Vec<f32>,
		channels: usize,
		sample_rate: u32,
		layout: &str,
		runtime: Option<&PythonRuntime>,
	) -> PyResult<Self> {
		let runtime = PythonRuntime::resolve(runtime)?;
		oa::Audio::new_from_planar_f32(
			&runtime,
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

#[pyfunction]
pub(crate) fn audio_noise_like(
	input: &PythonAudio,
	distribution: &str,
	amplitude: f32,
	seed: u64,
) -> PyResult<PythonAudio> {
	oa::audio::noise_like(
		&input.inner,
		noise_distribution(distribution)?,
		amplitude,
		seed,
	)
	.map(PythonAudio::wrap)
	.map_err(python_error)
}

#[pyfunction]
pub(crate) fn audio_mix_snr(
	signal: &PythonAudio,
	noise: &PythonAudio,
	snr_db: f32,
) -> PyResult<PythonAudio> {
	oa::audio::mix_snr(&signal.inner, &noise.inner, snr_db)
		.map(PythonAudio::wrap)
		.map_err(python_error)
}

#[pyfunction]
pub(crate) fn audio_pad_to(input: &PythonAudio, samples: usize) -> PyResult<PythonAudio> {
	oa::audio::pad_to(&input.inner, samples)
		.map(PythonAudio::wrap)
		.map_err(python_error)
}

#[pyfunction]
pub(crate) fn audio_crop_to(input: &PythonAudio, samples: usize) -> PyResult<PythonAudio> {
	oa::audio::crop_to(&input.inner, samples)
		.map(PythonAudio::wrap)
		.map_err(python_error)
}

#[pyfunction]
pub(crate) fn audio_stack(inputs: Vec<PyRef<'_, PythonAudio>>) -> PyResult<PythonAudio> {
	let audios: Vec<oa::Audio> = inputs.iter().map(|input| input.inner.clone()).collect();
	oa::audio::stack(&audios)
		.map(PythonAudio::wrap)
		.map_err(python_error)
}

#[pyfunction]
pub(crate) fn audio_length_mask(
	reference: &PythonAudio,
	lengths: Vec<usize>,
) -> PyResult<PythonMatrix> {
	oa::audio::length_mask(&reference.inner, &lengths)
		.map(PythonMatrix::wrap)
		.map_err(python_error)
}

#[pyfunction]
pub(crate) fn audio_decode_file(
	path: &str,
	runtime: Option<&PythonRuntime>,
) -> PyResult<PythonAudio> {
	let runtime = PythonRuntime::resolve(runtime)?;
	oa::audio::decode_file_on(&runtime, path)
		.map(PythonAudio::wrap)
		.map_err(python_error)
}

#[pyfunction]
pub(crate) fn audio_decode_memory(
	encoded: Vec<u8>,
	runtime: Option<&PythonRuntime>,
) -> PyResult<PythonAudio> {
	let runtime = PythonRuntime::resolve(runtime)?;
	oa::audio::decode_memory_on(&runtime, &encoded)
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

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
	sessions::register(module)?;
	module.add_class::<PythonAudio>()?;
	macro_rules! add_functions {
		($($function:ident),+ $(,)?) => { $(module.add_function(wrap_pyfunction!($function, module)?)?;)+ };
	}
	generated::register(module)?;
	add_functions!(
		audio_crop_to,
		audio_length_mask,
		audio_mix_snr,
		audio_noise_like,
		audio_pad_to,
		audio_stack,
		audio_decode_file,
		audio_decode_memory,
		audio_encode_wav_f32,
		audio_encode_interleaved_wav_f32,
		audio_save_wav_f32,
	);
	Ok(())
}
