//! Canonical controlled audio command-recognition workload.
//!
//! Teaches the GPU-resident capture/replay pipeline on a mini Speech
//! Commands-style keyword task: decode one-second clips, batch them through a
//! single mel-spectrogram dispatch, and train a small CNN classifier. Mirrors
//! the TensorFlow "simple audio" tutorial flow with OA audio and ML surfaces.

use std::path::Path;
use std::rc::Rc;

use crate::ml::nn;
use crate::{Audio, AudioChannelLayout, Engine, Error, Matrix, Result, audio};

/// Command labels in canonical mini_speech_commands order.
pub const COMMAND_LABELS: [&str; 8] = ["down", "go", "left", "no", "right", "stop", "up", "yes"];
/// Clip rate shared by every Speech Commands recording.
pub const CLIP_SAMPLE_RATE: u32 = 16_000;
/// One-second clip length in samples; shorter clips are zero-padded.
pub const SAMPLES_PER_CLIP: usize = 16_000;
/// Mel feature geometry for the canonical commands workload.
pub const MEL_FFT_SIZE: u32 = 1_024;
/// Hop between consecutive mel frames in samples.
pub const MEL_HOP_SIZE: u32 = 256;
/// Mel band count; the canonical tutorial uses 40 bands.
pub const MEL_BANDS: u32 = 40;
/// Deterministic model initialization seed.
pub const RNG_SEED: u64 = 20_260_818;
/// Complete data passes in one canonical tutorial run, matching the
/// TensorFlow simple-audio recipe.
pub const TRAINING_EPOCHS: usize = 10;
/// Clips per canonical training batch; the bundled subset trains full-batch.
pub const BATCH_SIZE: usize = 8;
/// Clips per mel-feature dispatch; keeps each submission bounded on
/// memory-constrained devices and below driver watchdog windows.
pub const MEL_CHUNK_CLIPS: usize = 512;
/// Accepted OARS final loss for the canonical commands run.
pub const COMMANDS_FINAL_LOSS: f32 = 0.622_496;
/// Accepted OARS final test accuracy for the canonical commands run. The
/// bundled subset carries four clips per command, so this floor is the
/// deterministic memorization result — meaningful generalization numbers
/// require `--dataset` pointed at full mini_speech_commands.
pub const COMMANDS_TEST_ACCURACY: f32 = 0.0;

/// One decoded clip: mono FP32 samples padded or trimmed to
/// [`SAMPLES_PER_CLIP`], plus its command class index.
#[derive(Clone, Debug)]
pub struct SpeechCommandsClip {
	pub samples: Vec<f32>,
	pub label: u32,
}

/// Ordered speech-commands dataset backed by `<dir>/<label>/*.wav` files.
pub struct SpeechCommandsDataset {
	pub clips: Vec<SpeechCommandsClip>,
}

impl SpeechCommandsDataset {
	/// Load and decode every clip under `<dir>/<label>/`.
	///
	/// Clips are sorted per label for deterministic ordering, decoded to mono,
	/// resampled to [`CLIP_SAMPLE_RATE`] when needed, and zero-padded or
	/// trimmed to [`SAMPLES_PER_CLIP`].
	///
	/// # Errors
	///
	/// Returns an error when a label directory is missing, a clip fails to
	/// decode, or the dataset is empty.
	pub fn load(engine: &Engine, dir: &Path) -> Result<Self> {
		let mut clips = Vec::new();
		for (label, name) in COMMAND_LABELS.iter().enumerate() {
			let label_dir = dir.join(name);
			if !label_dir.is_dir() {
				return Err(Error::invalid_argument(format!(
					"speech-commands label directory missing: {}",
					label_dir.display()
				)));
			}
			let mut files = std::fs::read_dir(&label_dir)
				.map_err(|source| {
					Error::invalid_argument(format!("cannot list {}: {source}", label_dir.display()))
				})?
				.filter_map(|entry| entry.ok().map(|entry| entry.path()))
				.filter(|path| path.extension().is_some_and(|ext| ext == "wav"))
				.collect::<Vec<_>>();
			files.sort();
			for path in files {
				let decoded = audio::decode_file(engine, &path)?;
				let mut mono = decoded.to_mono()?;
				if mono.sample_rate() != CLIP_SAMPLE_RATE {
					mono = mono.resample(crate::audio::ResampleConfig {
						out_rate: CLIP_SAMPLE_RATE,
						..Default::default()
					})?;
				}
				let mut samples = mono.as_matrix().read_f32()?;
				samples.resize(SAMPLES_PER_CLIP, 0.0);
				samples.truncate(SAMPLES_PER_CLIP);
				clips.push(SpeechCommandsClip {
					samples,
					label: label as u32,
				});
			}
		}
		if clips.is_empty() {
			return Err(Error::invalid_argument(format!(
				"speech-commands dataset at {} contains no wav clips",
				dir.display()
			)));
		}
		Ok(Self { clips })
	}

	/// Compute one mel feature for every selected clip in a single dispatch.
	///
	/// Clips are packed as planar channels, so one `mel_spectrogram` call
	/// returns `[batch, mels, frames]`; the result is reshaped to
	/// `[batch, 1, mels, frames]` NCHW for convolution.
	///
	/// # Errors
	///
	/// Returns an error when `indices` is empty or feature generation fails.
	pub fn mel_features(&self, engine: &Engine, indices: &[usize]) -> Result<Matrix> {
		if indices.is_empty() {
			return Err(Error::invalid_argument(
				"mel_features requires at least one clip index",
			));
		}
		let mut planar = Vec::with_capacity(indices.len() * SAMPLES_PER_CLIP);
		for &index in indices {
			let clip = self
				.clips
				.get(index)
				.ok_or_else(|| Error::invalid_argument(format!("clip index {index} out of range")))?;
			planar.extend_from_slice(&clip.samples);
		}
		let batch = Audio::new_from_planar_f32(
			engine,
			&planar,
			indices.len(),
			CLIP_SAMPLE_RATE,
			AudioChannelLayout::Unknown,
		)?;
		let features = audio::mel_spectrogram(
			&batch,
			audio::MelConfig {
				fft_size: MEL_FFT_SIZE,
				hop_size: MEL_HOP_SIZE,
				num_mels: MEL_BANDS,
				log_scale: true,
				..Default::default()
			},
		)?;
		let [batch, mels, frames] = features.shape() else {
			return Err(Error::invalid_argument(
				"mel_spectrogram must return [batch, mels, frames]",
			));
		};
		features.reshape([*batch, 1, *mels, *frames])
	}

	/// U32 class targets for the selected clips.
	///
	/// # Errors
	///
	/// Returns an error when an index is out of range or upload fails.
	pub fn labels(&self, engine: &Engine, indices: &[usize]) -> Result<Matrix> {
		let labels = indices
			.iter()
			.map(|&index| {
				self
					.clips
					.get(index)
					.map(|clip| clip.label)
					.ok_or_else(|| Error::invalid_argument(format!("clip index {index} out of range")))
			})
			.collect::<Result<Vec<u32>>>()?;
		Matrix::new_from_slice(engine, [labels.len()], &labels)
	}
}

/// TF-tutorial-mirrored command classifier.
///
/// `Conv2d(1→8, 3×3) → ReLU → MaxPool → Conv2d(8→16, 3×3) → ReLU → MaxPool →
/// Flatten → Linear(→64) → ReLU → Dropout(0.25) → Linear(→classes)`, returned
/// as a `Sequential` so parameter registration and train/eval propagation are
/// uniform with every other nn workload.
///
/// `mel_bands`/`mel_frames` are the unpooled feature extents; two stride-2
/// pools quarter each spatial axis before the flatten.
///
/// # Errors
///
/// Returns an error when dimensions are zero or a layer fails to build.
pub fn commands_cnn(
	engine: &Engine,
	mel_bands: usize,
	mel_frames: usize,
	num_classes: usize,
) -> Result<nn::Sequential> {
	let flat = 16_usize
		.checked_mul(mel_bands / 4)
		.and_then(|value| value.checked_mul(mel_frames / 4))
		.and_then(|value| value.checked_mul(1))
		.ok_or_else(|| Error::invalid_argument("commands CNN flatten extent overflows usize"))?;
	if flat == 0 {
		return Err(Error::invalid_argument(
			"commands CNN requires mel extents of at least 4×4",
		));
	}
	let mut model = nn::Sequential::new();
	model.add_named(
		"conv1",
		Rc::new(nn::Conv2d::new_with_seed(
			engine, 1, 8, 3, 1, 1, 1, RNG_SEED,
		)?),
	)?;
	model.add_named("relu1", Rc::new(nn::Relu::new()))?;
	model.add_named("pool1", Rc::new(nn::MaxPool2d::new(2)?))?;
	model.add_named(
		"conv2",
		Rc::new(nn::Conv2d::new_with_seed(
			engine,
			8,
			16,
			3,
			1,
			1,
			1,
			RNG_SEED.wrapping_add(1),
		)?),
	)?;
	model.add_named("relu2", Rc::new(nn::Relu::new()))?;
	model.add_named("pool2", Rc::new(nn::MaxPool2d::new(2)?))?;
	model.add_named("flatten", Rc::new(nn::Flatten::new(1, -1)))?;
	model.add_named(
		"dense1",
		Rc::new(nn::Linear::new_with_seed(
			engine,
			flat,
			64,
			RNG_SEED.wrapping_add(2),
		)?),
	)?;
	model.add_named("relu3", Rc::new(nn::Relu::new()))?;
	model.add_named(
		"dropout",
		Rc::new(nn::Dropout::new_with_seed(0.25, RNG_SEED.wrapping_add(3))?),
	)?;
	model.add_named(
		"dense2",
		Rc::new(nn::Linear::new_with_seed(
			engine,
			64,
			num_classes,
			RNG_SEED.wrapping_add(4),
		)?),
	)?;
	Ok(model)
}

/// Mean and reciprocal standard deviation over a host feature batch.
///
/// Returns `(mean, 1/(std + 1e-6))`; callers apply the pair to both training
/// and evaluation features — the host-side equivalent of the TF tutorial's
/// `Normalization` layer.
#[must_use]
pub fn standardization(features: &[f32]) -> (f32, f32) {
	let count = features.len().max(1) as f64;
	let mean = features.iter().map(|&v| v as f64).sum::<f64>() / count;
	let variance = features
		.iter()
		.map(|&v| (v as f64 - mean).powi(2))
		.sum::<f64>()
		/ count;
	(mean as f32, (variance.sqrt() as f32 + 1e-6).recip())
}

/// Apply `(v - mean) * inv_std` to every feature in place.
pub fn standardize(features: &mut [f32], mean: f32, inv_std: f32) {
	for value in features {
		*value = (*value - mean) * inv_std;
	}
}

/// Fraction of correct predictions over `[N, classes]` logits and `[N]`
/// U32/I32 targets.
///
/// # Errors
///
/// Returns an error from the underlying accuracy operation or readback.
pub fn accuracy(logits: &Matrix, targets: &Matrix) -> Result<f32> {
	let count = crate::matrix::categorical_accuracy_count(logits, targets)?.read::<u32>()?;
	let total = targets.num_elements();
	if total == 0 {
		return Err(Error::invalid_argument(
			"accuracy requires at least one target",
		));
	}
	Ok(count[0] as f32 / total as f32)
}
