//! Thread-local gradient-node registry for audio operations.
//!
//! Mirrors `crate::matrix::autograd`: audio operations record saved state as
//! [`GradNodeAudio`] values into whichever observer the active gradient tape
//! installed, so the audio domain stays independent of the ML tape owner.

use std::{cell::RefCell, marker::PhantomData, rc::Rc};

use crate::{Matrix, Result, matrix};

use super::StftWindow;

/// Saved state for one recorded audio operation's backward pass.
pub(crate) enum GradNodeAudio {
	/// `stft_complex`: backward synthesizes the time-domain gradient through
	/// `istft`, the exact Hermitian adjoint of the complex analysis map.
	StftComplex {
		input: Matrix,
		output_id: u64,
		fft_size: u32,
		hop_size: u32,
		win_size: u32,
		window: StftWindow,
		center: bool,
		sample_rate: u32,
	},
	/// `istft`: backward re-analyzes the time-domain gradient through
	/// `stft_complex`, the exact adjoint of the synthesis map.
	Istft {
		input: Matrix,
		output_id: u64,
		fft_size: u32,
		hop_size: u32,
		win_size: u32,
		window: StftWindow,
		center: bool,
		sample_rate: u32,
	},
}

type Recorder = Rc<dyn Fn(GradNodeAudio) -> Result<()>>;

thread_local! {
	static RECORDERS: RefCell<Vec<Recorder>> = const { RefCell::new(Vec::new()) };
}

/// Selection of one thread-local observer, removed exactly when dropped.
pub(crate) struct RecordingSelection {
	recorder: Recorder,
	_not_send_sync: PhantomData<Rc<()>>,
}

pub(crate) fn select(record: impl Fn(GradNodeAudio) -> Result<()> + 'static) -> RecordingSelection {
	let recorder: Recorder = Rc::new(record);
	RECORDERS.with(|recorders| recorders.borrow_mut().push(recorder.clone()));
	RecordingSelection {
		recorder,
		_not_send_sync: PhantomData,
	}
}

pub(crate) fn record(node: GradNodeAudio) -> Result<()> {
	let recorder = RECORDERS.with(|recorders| recorders.borrow().last().cloned());
	if let Some(recorder) = recorder {
		return recorder(node);
	}
	Ok(())
}

/// Host-side squared-window overlap-add norm, matching the accumulation the
/// `istft_frames` kernel performs per padded sample.
fn window_norm(
	channels: usize,
	frames: usize,
	padded: usize,
	fft_size: u32,
	hop_size: u32,
	win_size: u32,
	window: StftWindow,
) -> Vec<f32> {
	let kind = match window {
		StftWindow::Hann => 0_u32,
		StftWindow::Hamming => 1,
		StftWindow::Blackman => 2,
		StftWindow::Rectangular => 3,
	};
	let weights: Vec<f32> = (0..fft_size)
		.map(|index| {
			if index >= win_size {
				0.0
			} else if kind == 3 {
				1.0
			} else {
				let phase = 2.0 * std::f32::consts::PI * index as f32 / win_size as f32;
				match kind {
					1 => 0.54 - 0.46 * phase.cos(),
					2 => 0.42 - 0.5 * phase.cos() + 0.08 * (2.0 * phase).cos(),
					_ => 0.5 * (1.0 - phase.cos()),
				}
			}
		})
		.collect();
	let mut norm = vec![0.0_f32; channels * padded];
	for channel in 0..channels {
		for frame in 0..frames {
			let base = channel * padded + frame * hop_size as usize;
			for (index, weight) in weights.iter().enumerate() {
				norm[base + index] += weight * weight;
			}
		}
	}
	norm
}

/// Compute the input gradient for one recorded audio node.
///
/// Both transforms are linear maps; their adjoints reuse the same synthesis
/// and analysis kernels. `stft` backward is the unnormalized windowed
/// overlap-add (`frames`), recovered as `istft(g) * norm` because the forward
/// `istft` divides the frames output by the window-squared norm. `istft`
/// backward removes that normalization (`d / norm`, uncropped into the padded
/// domain) and then applies the uncentered analysis map, which is frame-exact
/// for the padded length `(frames - 1) * hop + fft`. The norm is deterministic
/// given the transform configuration, so it is derived on the host and
/// uploaded like the mel filterbank.
pub(crate) fn backward(node: &GradNodeAudio, output_gradient: &Matrix) -> Result<(u64, Matrix)> {
	match node {
		GradNodeAudio::StftComplex {
			input,
			output_id: _,
			fft_size,
			hop_size,
			win_size,
			window,
			center,
			sample_rate,
		} => {
			let frames = output_gradient.shape()[1];
			let channels = output_gradient.shape()[0];
			let padded = (frames - 1) * *hop_size as usize + *fft_size as usize;
			let offset = usize::from(*center) * (*fft_size as usize) / 2;
			let samples = input.shape()[1];
			let reconstructed = super::istft(
				output_gradient,
				super::IstftConfig {
					fft_size: *fft_size,
					hop_size: *hop_size,
					win_size: *win_size,
					window: *window,
					center: *center,
					sample_rate: *sample_rate,
				},
			)?;
			let norm_values = window_norm(
				channels,
				frames,
				padded,
				*fft_size,
				*hop_size,
				if *win_size == 0 { *fft_size } else { *win_size },
				*window,
			);
			let norm = Matrix::new_from_slice_handle(
				output_gradient.runtime(),
				vec![channels, padded],
				&norm_values,
			)?;
			let norm_cropped = matrix::slice(
				&norm,
				1,
				i64::try_from(offset).expect("crop offset fits i64"),
				i64::try_from(offset + samples).expect("crop end fits i64"),
			)?;
			let gradient = matrix::mul(reconstructed.as_matrix(), &norm_cropped)?;
			Ok((input.value_id(), gradient))
		}
		GradNodeAudio::Istft {
			input,
			output_id: _,
			fft_size,
			hop_size,
			win_size,
			window,
			center,
			sample_rate,
		} => {
			let frames = input.shape()[1];
			let channels = input.shape()[0];
			let padded = (frames - 1) * *hop_size as usize + *fft_size as usize;
			let offset = usize::from(*center) * (*fft_size as usize) / 2;
			let out_samples = output_gradient.shape()[1];
			let runtime = output_gradient.runtime();
			let norm_values = window_norm(
				channels,
				frames,
				padded,
				*fft_size,
				*hop_size,
				if *win_size == 0 { *fft_size } else { *win_size },
				*window,
			);
			let norm = Matrix::new_from_slice_handle(runtime, vec![channels, padded], &norm_values)?;
			// The forward normalize emits `scratch / norm` only where
			// `norm > 1e-12`; the adjoint applies the same mask so window-edge
			// samples (hann w[0] = 0) produce exact-zero input gradients.
			let divided = |gradient: &Matrix| -> Result<Matrix> {
				let safe = matrix::clamp_min(&norm, 1.0e-12)?;
				matrix::mul(
					&matrix::div(gradient, &safe)?,
					&matrix::greater_equal(&norm, 1.0e-12)?,
				)
			};
			let gradient_padded = if offset == 0 && out_samples == padded {
				divided(output_gradient)?
			} else {
				let zeros = |width: usize| -> Result<Matrix> {
					Matrix::allocate(
						runtime,
						vec![channels, width],
						channels * width,
						crate::DType::F32,
					)
				};
				divided(&matrix::concat(
					&[
						zeros(offset)?,
						output_gradient.clone(),
						zeros(padded - offset - out_samples)?,
					],
					1,
				)?)?
			};
			let audio = crate::Audio::new(
				gradient_padded,
				*sample_rate,
				crate::audio::AudioChannelLayout::for_channel_count(channels),
			)?;
			let spectrum = super::stft_complex(
				&audio,
				super::StftConfig {
					fft_size: *fft_size,
					hop_size: *hop_size,
					win_size: *win_size,
					window: *window,
					center: false,
				},
			)?;
			Ok((input.value_id(), spectrum))
		}
	}
}

impl Drop for RecordingSelection {
	fn drop(&mut self) {
		RECORDERS.with(|recorders| {
			recorders
				.borrow_mut()
				.retain(|candidate| !Rc::ptr_eq(candidate, &self.recorder));
		});
	}
}
