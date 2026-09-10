//! Stateless GPU feature transforms from Audio to Matrix.

use crate::{
	Audio, DType, Error, Matrix, OpAttribute, Result,
	runtime::{
		AudioSemanticDispatch, AudioSemanticOutput, BufferBinding, ComputeDispatch, KernelId,
		PushConstant,
	},
};

use super::signal::{float_attribute, u32_extent, unsigned_attribute};

/// Window function applied before each FFT frame.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum StftWindow {
	#[default]
	Hann,
	Hamming,
	Blackman,
	Rectangular,
}

/// Short-time Fourier transform configuration.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StftConfig {
	pub fft_size: u32,
	pub hop_size: u32,
	/// Zero selects `fft_size`.
	pub win_size: u32,
	pub window: StftWindow,
	pub center: bool,
}

impl Default for StftConfig {
	fn default() -> Self {
		Self {
			fft_size: 1_024,
			hop_size: 256,
			win_size: 0,
			window: StftWindow::Hann,
			center: true,
		}
	}
}

/// Mel-spectrogram configuration.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MelConfig {
	pub fft_size: u32,
	pub hop_size: u32,
	pub num_mels: u32,
	pub f_min: f32,
	/// Zero selects the Nyquist frequency.
	pub f_max: f32,
	pub log_scale: bool,
	pub normalize: bool,
}

impl Default for MelConfig {
	fn default() -> Self {
		Self {
			fft_size: 1_024,
			hop_size: 256,
			num_mels: 80,
			f_min: 0.0,
			f_max: 0.0,
			log_scale: true,
			normalize: false,
		}
	}
}

/// Mel-frequency cepstral coefficient configuration.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MfccConfig {
	pub num_coeffs: u32,
	pub mel: MelConfig,
}

impl Default for MfccConfig {
	fn default() -> Self {
		Self {
			num_coeffs: 13,
			mel: MelConfig::default(),
		}
	}
}

/// Compute magnitude STFT features in `[channels, frames, frequency_bins]` layout.
pub fn stft(input: &Audio, config: StftConfig) -> Result<Matrix> {
	let shape = validate_stft(input, config)?;
	let output = allocate_stft(input, shape)?;
	let buffers = [
		BufferBinding::read(input.storage()),
		BufferBinding::write(output.storage()),
	];
	let push = stft_push(config, shape);
	let attributes = stft_attributes(config);
	let outputs = [AudioSemanticOutput::Matrix(&output)];
	input.engine_handle().record_audio_semantic(
		ComputeDispatch {
			kernel: KernelId::AudioStftF32,
			buffers: &buffers,
			push_constants: &push,
			workgroups: [shape.frames, 1, shape.channels],
		},
		AudioSemanticDispatch {
			contract: crate::core::operation::audio::STFT,
			inputs: &[input],
			outputs: &outputs,
			attributes: &attributes,
		},
	)?;
	Ok(output)
}

/// Compute HTK triangular-filter mel features in `[channels, mels, frames]` layout.
pub fn mel_spectrogram(input: &Audio, config: MelConfig) -> Result<Matrix> {
	validate_mel(input, config)?;
	let stft_config = StftConfig {
		fft_size: config.fft_size,
		hop_size: config.hop_size,
		win_size: config.fft_size,
		..StftConfig::default()
	};
	let shape = validate_stft(input, stft_config)?;
	let spectrum = allocate_stft(input, shape)?;
	let filterbank = build_mel_filterbank(input, config, shape.freq_bins)?;
	let values_per_channel = config
		.num_mels
		.checked_mul(shape.frames)
		.ok_or_else(|| Error::out_of_range("audio::mel_spectrogram channel extent exceeds u32"))?;
	let output_elements = shape
		.channels
		.checked_mul(values_per_channel)
		.ok_or_else(|| Error::out_of_range("audio::mel_spectrogram output exceeds u32"))?;

	let spectrum_buffers = [
		BufferBinding::read(input.storage()),
		BufferBinding::write(spectrum.storage()),
	];
	let spectrum_push = stft_push(stft_config, shape);
	let mel_push = mel_push(config, shape);
	let attributes = mel_attributes(config);

	let output = if config.normalize {
		let raw = Matrix::allocate(
			input.engine_handle(),
			vec![
				shape.channels as usize,
				config.num_mels as usize,
				shape.frames as usize,
			],
			output_elements as usize,
			DType::F32,
		)?;
		let output = Matrix::allocate(
			input.engine_handle(),
			raw.shape().to_vec(),
			raw.num_elements(),
			DType::F32,
		)?;
		let mel_buffers = [
			BufferBinding::read(spectrum.storage()),
			BufferBinding::write(raw.storage()),
			BufferBinding::read(filterbank.storage()),
		];
		let normalize_buffers = [
			BufferBinding::read(raw.storage()),
			BufferBinding::write(output.storage()),
		];
		let normalize_push = [
			PushConstant::U32(shape.channels),
			PushConstant::U32(values_per_channel),
		];
		let dispatches = [
			stft_dispatch(&spectrum_buffers, &spectrum_push, shape),
			mel_dispatch(&mel_buffers, &mel_push, config, shape),
			ComputeDispatch {
				kernel: KernelId::AudioMelNormalizeF32,
				buffers: &normalize_buffers,
				push_constants: &normalize_push,
				workgroups: [shape.channels, 1, 1],
			},
		];
		record_matrix_split(
			input,
			&output,
			&dispatches,
			crate::core::operation::audio::MEL_SPECTROGRAM,
			&attributes,
		)?;
		output
	} else {
		let output = Matrix::allocate(
			input.engine_handle(),
			vec![
				shape.channels as usize,
				config.num_mels as usize,
				shape.frames as usize,
			],
			output_elements as usize,
			DType::F32,
		)?;
		let mel_buffers = [
			BufferBinding::read(spectrum.storage()),
			BufferBinding::write(output.storage()),
			BufferBinding::read(filterbank.storage()),
		];
		let dispatches = [
			stft_dispatch(&spectrum_buffers, &spectrum_push, shape),
			mel_dispatch(&mel_buffers, &mel_push, config, shape),
		];
		record_matrix_split(
			input,
			&output,
			&dispatches,
			crate::core::operation::audio::MEL_SPECTROGRAM,
			&attributes,
		)?;
		output
	};
	Ok(output)
}

/// Compute orthonormal DCT-II coefficients in `[channels, coefficients, frames]` layout.
pub fn mfcc(input: &Audio, config: MfccConfig) -> Result<Matrix> {
	if config.num_coeffs == 0 || config.num_coeffs > config.mel.num_mels {
		return Err(Error::invalid_argument(
			"audio::mfcc num_coeffs must be in [1, num_mels]",
		));
	}
	validate_mel(input, config.mel)?;
	let stft_config = StftConfig {
		fft_size: config.mel.fft_size,
		hop_size: config.mel.hop_size,
		win_size: config.mel.fft_size,
		..StftConfig::default()
	};
	let shape = validate_stft(input, stft_config)?;
	let spectrum = allocate_stft(input, shape)?;
	let filterbank = build_mel_filterbank(input, config.mel, shape.freq_bins)?;
	let mel_elements = shape
		.channels
		.checked_mul(config.mel.num_mels)
		.and_then(|value| value.checked_mul(shape.frames))
		.ok_or_else(|| Error::out_of_range("audio::mfcc mel workspace exceeds u32"))?;
	let mel = Matrix::allocate(
		input.engine_handle(),
		vec![
			shape.channels as usize,
			config.mel.num_mels as usize,
			shape.frames as usize,
		],
		mel_elements as usize,
		DType::F32,
	)?;
	let output_elements = shape
		.channels
		.checked_mul(config.num_coeffs)
		.and_then(|value| value.checked_mul(shape.frames))
		.ok_or_else(|| Error::out_of_range("audio::mfcc output exceeds u32"))?;
	let output = Matrix::allocate(
		input.engine_handle(),
		vec![
			shape.channels as usize,
			config.num_coeffs as usize,
			shape.frames as usize,
		],
		output_elements as usize,
		DType::F32,
	)?;
	let spectrum_buffers = [
		BufferBinding::read(input.storage()),
		BufferBinding::write(spectrum.storage()),
	];
	let spectrum_push = stft_push(stft_config, shape);
	let mel_buffers = [
		BufferBinding::read(spectrum.storage()),
		BufferBinding::write(mel.storage()),
		BufferBinding::read(filterbank.storage()),
	];
	let mut mel_config = config.mel;
	mel_config.log_scale = true;
	mel_config.normalize = false;
	let mel_push = mel_push(mel_config, shape);
	let dct_buffers = [
		BufferBinding::read(mel.storage()),
		BufferBinding::write(output.storage()),
	];
	let dct_push = [
		PushConstant::U32(shape.channels),
		PushConstant::U32(shape.frames),
		PushConstant::U32(config.mel.num_mels),
		PushConstant::U32(config.num_coeffs),
	];
	let dispatches = [
		stft_dispatch(&spectrum_buffers, &spectrum_push, shape),
		mel_dispatch(&mel_buffers, &mel_push, mel_config, shape),
		ComputeDispatch {
			kernel: KernelId::AudioMfccDctF32,
			buffers: &dct_buffers,
			push_constants: &dct_push,
			workgroups: [config.num_coeffs.div_ceil(32), shape.frames, shape.channels],
		},
	];
	let mut attributes = vec![unsigned_attribute(
		"num_coeffs",
		u64::from(config.num_coeffs),
	)];
	attributes.extend(mel_attributes(config.mel));
	record_matrix_split(
		input,
		&output,
		&dispatches,
		crate::core::operation::audio::MFCC,
		&attributes,
	)?;
	Ok(output)
}

#[derive(Clone, Copy)]
struct StftShape {
	channels: u32,
	samples: u32,
	frames: u32,
	freq_bins: u32,
}

fn validate_stft(input: &Audio, config: StftConfig) -> Result<StftShape> {
	if !config.fft_size.is_power_of_two() || !(16..=1_024).contains(&config.fft_size) {
		return Err(Error::invalid_argument(
			"audio::stft fft_size must be a power of two in [16, 1024]",
		));
	}
	if config.hop_size == 0 {
		return Err(Error::invalid_argument(
			"audio::stft hop_size must be nonzero",
		));
	}
	let win_size = if config.win_size == 0 {
		config.fft_size
	} else {
		config.win_size
	};
	if win_size > config.fft_size {
		return Err(Error::invalid_argument(
			"audio::stft win_size must be zero or no larger than fft_size",
		));
	}
	let channels = u32_extent(input.channels(), "audio::stft channels")?;
	let samples = u32_extent(input.samples(), "audio::stft samples")?;
	let frames = if config.center {
		1 + samples / config.hop_size
	} else if samples < config.fft_size {
		1
	} else {
		1 + (samples - config.fft_size) / config.hop_size
	};
	let freq_bins = config.fft_size / 2 + 1;
	channels
		.checked_mul(frames)
		.and_then(|value| value.checked_mul(freq_bins))
		.ok_or_else(|| Error::out_of_range("audio::stft output exceeds u32"))?;
	Ok(StftShape {
		channels,
		samples,
		frames,
		freq_bins,
	})
}

fn validate_mel(input: &Audio, config: MelConfig) -> Result<()> {
	let nyquist = input.sample_rate() as f32 * 0.5;
	let f_max = if config.f_max > 0.0 {
		config.f_max
	} else {
		nyquist
	};
	if !config.fft_size.is_power_of_two()
		|| !(16..=1_024).contains(&config.fft_size)
		|| config.hop_size == 0
		|| !(1..=4_096).contains(&config.num_mels)
		|| !config.f_min.is_finite()
		|| !f_max.is_finite()
		|| config.f_min < 0.0
		|| config.f_min >= f_max
		|| f_max > nyquist
	{
		return Err(Error::invalid_argument(
			"audio mel configuration is invalid",
		));
	}
	Ok(())
}

fn allocate_stft(input: &Audio, shape: StftShape) -> Result<Matrix> {
	let elements = shape.channels * shape.frames * shape.freq_bins;
	Matrix::allocate(
		input.engine_handle(),
		vec![
			shape.channels as usize,
			shape.frames as usize,
			shape.freq_bins as usize,
		],
		elements as usize,
		DType::F32,
	)
}

fn stft_push(config: StftConfig, shape: StftShape) -> [PushConstant; 9] {
	let win_size = if config.win_size == 0 {
		config.fft_size
	} else {
		config.win_size
	};
	let window: u32 = match config.window {
		StftWindow::Hann => 0,
		StftWindow::Hamming => 1,
		StftWindow::Blackman => 2,
		StftWindow::Rectangular => 3,
	};
	[
		PushConstant::U32(config.fft_size),
		PushConstant::U32(config.hop_size),
		PushConstant::U32(win_size),
		PushConstant::U32(window),
		PushConstant::U32(shape.channels),
		PushConstant::U32(shape.samples),
		PushConstant::U32(shape.frames),
		PushConstant::U32(shape.freq_bins),
		PushConstant::U32(u32::from(config.center)),
	]
}

fn stft_attributes(config: StftConfig) -> [OpAttribute; 5] {
	let window: u32 = match config.window {
		StftWindow::Hann => 0,
		StftWindow::Hamming => 1,
		StftWindow::Blackman => 2,
		StftWindow::Rectangular => 3,
	};
	[
		unsigned_attribute("fft_size", u64::from(config.fft_size)),
		unsigned_attribute("hop_size", u64::from(config.hop_size)),
		unsigned_attribute("win_size", u64::from(config.win_size)),
		unsigned_attribute("window", u64::from(window)),
		OpAttribute::Boolean {
			name: "center".into(),
			value: config.center,
		},
	]
}

fn mel_push(config: MelConfig, shape: StftShape) -> [PushConstant; 5] {
	[
		PushConstant::U32(shape.channels),
		PushConstant::U32(shape.frames),
		PushConstant::U32(shape.freq_bins),
		PushConstant::U32(config.num_mels),
		PushConstant::U32(u32::from(config.log_scale)),
	]
}

fn mel_attributes(config: MelConfig) -> Vec<OpAttribute> {
	vec![
		unsigned_attribute("fft_size", u64::from(config.fft_size)),
		unsigned_attribute("hop_size", u64::from(config.hop_size)),
		unsigned_attribute("num_mels", u64::from(config.num_mels)),
		float_attribute("f_min", config.f_min),
		float_attribute("f_max", config.f_max),
		OpAttribute::Boolean {
			name: "log_scale".into(),
			value: config.log_scale,
		},
		OpAttribute::Boolean {
			name: "normalize".into(),
			value: config.normalize,
		},
	]
}

fn build_mel_filterbank(input: &Audio, config: MelConfig, freq_bins: u32) -> Result<Matrix> {
	let sample_rate = input.sample_rate() as f32;
	let f_max = if config.f_max > 0.0 {
		config.f_max
	} else {
		sample_rate * 0.5
	};
	let hz_to_mel = |frequency: f32| 2_595.0 * (1.0 + frequency / 700.0).log10();
	let mel_to_hz = |mel: f32| 700.0 * (10.0_f32.powf(mel / 2_595.0) - 1.0);
	let mel_min = hz_to_mel(config.f_min);
	let mel_max = hz_to_mel(f_max);
	let point_count = config.num_mels as usize + 2;
	let mut points = Vec::new();
	points
		.try_reserve_exact(point_count)
		.map_err(|_| Error::resource_exhausted("mel point allocation failed"))?;
	for index in 0..point_count {
		let ratio = index as f32 / (config.num_mels + 1) as f32;
		points.push(mel_to_hz(mel_min + ratio * (mel_max - mel_min)));
	}
	let element_count = (config.num_mels as usize)
		.checked_mul(freq_bins as usize)
		.ok_or_else(|| Error::out_of_range("mel filterbank exceeds usize"))?;
	let mut weights = vec![0.0_f32; element_count];
	let bin_hz = sample_rate / config.fft_size as f32;
	for mel in 0..config.num_mels as usize {
		for bin in 0..freq_bins as usize {
			let frequency = bin as f32 * bin_hz;
			let (low, center, high) = (points[mel], points[mel + 1], points[mel + 2]);
			weights[mel * freq_bins as usize + bin] =
				if frequency >= low && frequency <= center && center > low {
					(frequency - low) / (center - low)
				} else if frequency > center && frequency <= high && high > center {
					(high - frequency) / (high - center)
				} else {
					0.0
				};
		}
	}
	Matrix::from_slice_handle(
		input.engine_handle(),
		vec![config.num_mels as usize, freq_bins as usize],
		&weights,
	)
}

fn stft_dispatch<'a>(
	buffers: &'a [BufferBinding<'a>],
	push: &'a [PushConstant],
	shape: StftShape,
) -> ComputeDispatch<'a> {
	ComputeDispatch {
		kernel: KernelId::AudioStftF32,
		buffers,
		push_constants: push,
		workgroups: [shape.frames, 1, shape.channels],
	}
}

fn mel_dispatch<'a>(
	buffers: &'a [BufferBinding<'a>],
	push: &'a [PushConstant],
	config: MelConfig,
	shape: StftShape,
) -> ComputeDispatch<'a> {
	ComputeDispatch {
		kernel: KernelId::AudioMelFilterbankF32,
		buffers,
		push_constants: push,
		workgroups: [config.num_mels.div_ceil(32), shape.frames, shape.channels],
	}
}

fn record_matrix_split(
	input: &Audio,
	output: &Matrix,
	dispatches: &[ComputeDispatch<'_>],
	contract: crate::OperationContract,
	attributes: &[OpAttribute],
) -> Result<()> {
	let outputs = [AudioSemanticOutput::Matrix(output)];
	input.engine_handle().record_audio_split_semantic(
		dispatches,
		AudioSemanticDispatch {
			contract,
			inputs: &[input],
			outputs: &outputs,
			attributes,
		},
	)
}
