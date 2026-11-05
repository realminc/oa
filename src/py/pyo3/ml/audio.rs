use pyo3::prelude::*;

use oa::sdk::ml::audio;

use crate::{error::python_error, matrix::PythonMatrix, runtime::PythonEngine};

use super::nn::PythonSequential;

/// Ordered speech-commands dataset backed by `<dir>/<label>/*.wav` files.
#[pyclass(name = "SpeechCommandsDataset", unsendable)]
pub(crate) struct PythonSpeechCommandsDataset {
	pub(crate) inner: audio::SpeechCommandsDataset,
}

#[pymethods]
impl PythonSpeechCommandsDataset {
	/// Load and decode every clip under `<dir>/<label>/`.
	#[new]
	pub fn new(engine: &PythonEngine, dir: &str) -> PyResult<Self> {
		audio::SpeechCommandsDataset::load(&engine.inner, std::path::Path::new(dir))
			.map(|inner| Self { inner })
			.map_err(python_error)
	}

	/// Number of loaded clips.
	pub fn __len__(&self) -> usize {
		self.inner.clips.len()
	}

	/// One mel feature per selected clip as `[batch, 1, mels, frames]` NCHW.
	pub fn mel_features(&self, engine: &PythonEngine, indices: Vec<usize>) -> PyResult<PythonMatrix> {
		self
			.inner
			.mel_features(&engine.inner, &indices)
			.map(PythonMatrix::wrap)
			.map_err(python_error)
	}

	/// U32 class targets for the selected clips.
	pub fn labels(&self, engine: &PythonEngine, indices: Vec<usize>) -> PyResult<PythonMatrix> {
		self
			.inner
			.labels(&engine.inner, &indices)
			.map(PythonMatrix::wrap)
			.map_err(python_error)
	}

	pub fn __repr__(&self) -> String {
		format!("SpeechCommandsDataset(clips={})", self.inner.clips.len())
	}
}

/// Build the TF-tutorial-mirrored command classifier as a `Sequential`.
#[pyfunction]
pub fn ml_audio_commands_cnn(
	engine: &PythonEngine,
	mel_bands: usize,
	mel_frames: usize,
	num_classes: usize,
) -> PyResult<PythonSequential> {
	audio::commands_cnn(&engine.inner, mel_bands, mel_frames, num_classes)
		.map(PythonSequential::wrap)
		.map_err(python_error)
}

/// Fraction of correct predictions over `[N, classes]` logits and `[N]` targets.
#[pyfunction]
pub fn ml_audio_accuracy(logits: &PythonMatrix, targets: &PythonMatrix) -> PyResult<f32> {
	audio::accuracy(&logits.inner, &targets.inner).map_err(python_error)
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
	module.add_class::<PythonSpeechCommandsDataset>()?;
	module.add_function(wrap_pyfunction!(ml_audio_commands_cnn, module)?)?;
	module.add_function(wrap_pyfunction!(ml_audio_accuracy, module)?)?;

	module.add("AUDIO_COMMAND_LABELS", audio::COMMAND_LABELS.to_vec())?;
	module.add("AUDIO_CLIP_SAMPLE_RATE", audio::CLIP_SAMPLE_RATE)?;
	module.add("AUDIO_SAMPLES_PER_CLIP", audio::SAMPLES_PER_CLIP)?;
	module.add("AUDIO_MEL_FFT_SIZE", audio::MEL_FFT_SIZE)?;
	module.add("AUDIO_MEL_HOP_SIZE", audio::MEL_HOP_SIZE)?;
	module.add("AUDIO_MEL_BANDS", audio::MEL_BANDS)?;
	module.add("AUDIO_RNG_SEED", audio::RNG_SEED)?;
	module.add("AUDIO_TRAINING_EPOCHS", audio::TRAINING_EPOCHS)?;
	module.add("AUDIO_BATCH_SIZE", audio::BATCH_SIZE)?;
	module.add("AUDIO_MEL_CHUNK_CLIPS", audio::MEL_CHUNK_CLIPS)?;
	module.add("COMMANDS_FINAL_LOSS", audio::COMMANDS_FINAL_LOSS)?;
	module.add("COMMANDS_TEST_ACCURACY", audio::COMMANDS_TEST_ACCURACY)?;
	Ok(())
}
