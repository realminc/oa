//! Stateless GPU audio-signal operations.

use crate::{
	Audio, Error, Matrix, OpAttribute, Result,
	runtime::{
		AudioSemanticDispatch, AudioSemanticOutput, BufferBinding, ComputeDispatch, KernelId,
		PushConstant,
	},
};

use super::{AudioChannelLayout, lowering};

/// Global level estimator used by [`normalize`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum NormalizeAudioMode {
	/// Normalize the largest absolute sample.
	#[default]
	Peak,
	/// Normalize the root-mean-square level over all channels and samples.
	Rms,
}

/// Parameters for [`normalize`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NormalizeAudioConfig {
	pub target_db: f32,
	pub mode: NormalizeAudioMode,
}

impl Default for NormalizeAudioConfig {
	fn default() -> Self {
		Self {
			target_db: -3.0,
			mode: NormalizeAudioMode::Peak,
		}
	}
}

/// Parameters for [`resample`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResampleConfig {
	pub out_rate: u32,
	pub filter_half_width: u32,
}

impl Default for ResampleConfig {
	fn default() -> Self {
		Self {
			out_rate: 16_000,
			filter_half_width: 64,
		}
	}
}

/// Real, `a0`-normalized coefficients for one stable biquad section.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BiquadCoefficients {
	pub b0: f32,
	pub b1: f32,
	pub b2: f32,
	pub a1: f32,
	pub a2: f32,
}

impl Default for BiquadCoefficients {
	fn default() -> Self {
		Self {
			b0: 1.0,
			b1: 0.0,
			b2: 0.0,
			a1: 0.0,
			a2: 0.0,
		}
	}
}

/// Normalize global peak or RMS level to the configured decibel target.
pub fn normalize(input: &Audio, config: NormalizeAudioConfig) -> Result<Audio> {
	if !config.target_db.is_finite() || !(-300.0..=100.0).contains(&config.target_db) {
		return Err(Error::invalid_argument(
			"audio::normalize target_db must be finite and in [-300, 100]",
		));
	}
	let count = element_count_u32(input, "audio::normalize")?;
	let level = Matrix::allocate(input.engine_handle(), vec![1], 1, crate::DType::F32)?;
	let output_matrix = allocate_like(input)?;
	let output = wrap_like(output_matrix, input)?;
	let level_buffers = [
		BufferBinding::read(input.storage()),
		BufferBinding::write(level.storage()),
	];
	let mode = match config.mode {
		NormalizeAudioMode::Peak => 0,
		NormalizeAudioMode::Rms => 1,
	};
	let level_push = [PushConstant::U32(count), PushConstant::U32(mode)];
	let apply_buffers = [
		BufferBinding::read(input.storage()),
		BufferBinding::read(level.storage()),
		BufferBinding::write(output.storage()),
	];
	let apply_push = [
		PushConstant::U32(count),
		PushConstant::F32(10.0_f32.powf(config.target_db / 20.0)),
	];
	let dispatches = [
		ComputeDispatch {
			kernel: KernelId::AudioNormalizeLevelF32,
			buffers: &level_buffers,
			push_constants: &level_push,
			workgroups: [1, 1, 1],
		},
		ComputeDispatch {
			kernel: KernelId::AudioNormalizeApplyF32,
			buffers: &apply_buffers,
			push_constants: &apply_push,
			workgroups: KernelId::AudioNormalizeApplyF32.linear_workgroups(count),
		},
	];
	let attributes = [
		float_attribute("target_db", config.target_db),
		unsigned_attribute("mode", u64::from(mode)),
	];
	record_audio_split(
		input,
		&output,
		&dispatches,
		crate::core::operation::audio::NORMALIZE,
		&attributes,
	)?;
	Ok(output)
}

/// Resample planar audio using the donor windowed-sinc gather kernel.
pub fn resample(input: &Audio, config: ResampleConfig) -> Result<Audio> {
	if config.out_rate == 0 || !(1..=1_024).contains(&config.filter_half_width) {
		return Err(Error::invalid_argument(
			"audio::resample requires a nonzero rate and filter_half_width in [1, 1024]",
		));
	}
	let divisor = gcd(input.sample_rate(), config.out_rate);
	let in_rate = input.sample_rate() / divisor;
	let out_rate = config.out_rate / divisor;
	let out_samples = input
		.samples()
		.checked_mul(out_rate as usize)
		.and_then(|value| value.checked_div(in_rate as usize))
		.ok_or_else(|| Error::out_of_range("audio::resample output length exceeds usize"))?;
	if out_samples == 0 {
		return Err(Error::invalid_argument(
			"audio::resample output would be empty",
		));
	}
	let channels = u32_extent(input.channels(), "audio::resample channels")?;
	let in_samples = u32_extent(input.samples(), "audio::resample input samples")?;
	let out_samples_u32 = u32_extent(out_samples, "audio::resample output samples")?;
	let count = channels
		.checked_mul(out_samples_u32)
		.ok_or_else(|| Error::out_of_range("audio::resample dispatch exceeds u32"))?;
	let matrix = Matrix::allocate(
		input.engine_handle(),
		vec![input.channels(), out_samples],
		count as usize,
		crate::DType::F32,
	)?;
	let output = Audio::new(matrix, config.out_rate, input.layout())?;
	let push = [
		PushConstant::U32(count),
		PushConstant::U32(in_rate),
		PushConstant::U32(out_rate),
		PushConstant::U32(config.filter_half_width),
		PushConstant::U32(in_samples),
		PushConstant::U32(out_samples_u32),
	];
	let attributes = [
		unsigned_attribute("out_rate", u64::from(config.out_rate)),
		unsigned_attribute("filter_half_width", u64::from(config.filter_half_width)),
	];
	record_audio_direct(
		input,
		&output,
		KernelId::AudioResampleF32,
		&push,
		crate::core::operation::audio::RESAMPLE,
		&attributes,
	)?;
	Ok(output)
}

/// Apply scalar gain in decibels.
pub fn gain(input: &Audio, gain_db: f32) -> Result<Audio> {
	if !gain_db.is_finite() || !(-300.0..=100.0).contains(&gain_db) {
		return Err(Error::invalid_argument(
			"audio::gain gain_db must be finite and in [-300, 100]",
		));
	}
	direct_like(
		input,
		KernelId::AudioGainF32,
		&[PushConstant::F32(gain_db)],
		crate::core::operation::audio::GAIN,
		&[float_attribute("gain_db", gain_db)],
	)
}

/// Clamp every sample to an inclusive finite interval.
pub fn clip(input: &Audio, minimum: f32, maximum: f32) -> Result<Audio> {
	if !minimum.is_finite() || !maximum.is_finite() || minimum > maximum {
		return Err(Error::invalid_argument(
			"audio::clip requires finite ordered bounds",
		));
	}
	direct_like(
		input,
		KernelId::AudioClipF32,
		&[PushConstant::F32(minimum), PushConstant::F32(maximum)],
		crate::core::operation::audio::CLIP,
		&[
			float_attribute("minimum", minimum),
			float_attribute("maximum", maximum),
		],
	)
}

/// Apply tanh soft clipping with a dry/wet mix.
pub fn saturate(input: &Audio, drive_db: f32, mix: f32) -> Result<Audio> {
	if !drive_db.is_finite()
		|| !(-60.0..=60.0).contains(&drive_db)
		|| !mix.is_finite()
		|| !(0.0..=1.0).contains(&mix)
	{
		return Err(Error::invalid_argument(
			"audio::saturate requires drive_db in [-60, 60] and mix in [0, 1]",
		));
	}
	direct_like(
		input,
		KernelId::AudioSaturateF32,
		&[PushConstant::F32(drive_db), PushConstant::F32(mix)],
		crate::core::operation::audio::SATURATE,
		&[
			float_attribute("drive_db", drive_db),
			float_attribute("mix", mix),
		],
	)
}

/// Apply one stable zero-state biquad independently to every channel.
pub fn biquad(input: &Audio, coefficients: BiquadCoefficients) -> Result<Audio> {
	lowering::biquad(input, &[coefficients], false)
}

/// Render a finite zero-state Schroeder reverberation tail.
pub fn reverb(input: &Audio, decay_seconds: f32, wet: f32) -> Result<Audio> {
	lowering::reverb(input, decay_seconds, wet)
}

/// Apply one to 64 stable biquad sections as a zero-state cascade.
pub fn sos_filter(input: &Audio, sections: &[BiquadCoefficients]) -> Result<Audio> {
	lowering::biquad(input, sections, true)
}

/// Apply `y[n] = x[n] - alpha*x[n-1]` independently per channel.
pub fn pre_emphasis(input: &Audio, alpha: f32) -> Result<Audio> {
	if !alpha.is_finite() {
		return Err(Error::invalid_argument(
			"audio::pre_emphasis alpha must be finite",
		));
	}
	let samples = u32_extent(input.samples(), "audio::pre_emphasis samples")?;
	direct_like(
		input,
		KernelId::AudioPreEmphasisF32,
		&[PushConstant::U32(samples), PushConstant::F32(alpha)],
		crate::core::operation::audio::PRE_EMPHASIS,
		&[float_attribute("alpha", alpha)],
	)
}

/// Average all channels into one mono channel.
pub fn to_mono(input: &Audio) -> Result<Audio> {
	let channels = u32_extent(input.channels(), "audio::to_mono channels")?;
	let samples = u32_extent(input.samples(), "audio::to_mono samples")?;
	let matrix = Matrix::allocate(
		input.engine_handle(),
		vec![1, input.samples()],
		input.samples(),
		crate::DType::F32,
	)?;
	let output = Audio::new(matrix, input.sample_rate(), AudioChannelLayout::Mono)?;
	let buffers = [
		BufferBinding::read(input.storage()),
		BufferBinding::write(output.storage()),
	];
	let push = [PushConstant::U32(channels), PushConstant::U32(samples)];
	let outputs = [AudioSemanticOutput::Audio(&output)];
	input.engine_handle().record_audio_semantic(
		ComputeDispatch {
			kernel: KernelId::AudioToMonoF32,
			buffers: &buffers,
			push_constants: &push,
			workgroups: KernelId::AudioToMonoF32.linear_workgroups(samples),
		},
		AudioSemanticDispatch {
			contract: crate::core::operation::audio::TO_MONO,
			inputs: &[input],
			outputs: &outputs,
			attributes: &[],
		},
	)?;
	Ok(output)
}

/// Apply linear fade-in and fade-out envelopes measured in samples.
pub fn fade(input: &Audio, fade_in_samples: u64, fade_out_samples: u64) -> Result<Audio> {
	let samples = u32_extent(input.samples(), "audio::fade samples")?;
	let fade_in = u32::try_from(fade_in_samples.min(u64::from(samples)))
		.map_err(|_| Error::out_of_range("audio::fade fade-in exceeds u32"))?;
	let fade_out = u32::try_from(fade_out_samples.min(u64::from(samples)))
		.map_err(|_| Error::out_of_range("audio::fade fade-out exceeds u32"))?;
	direct_like(
		input,
		KernelId::AudioFadeF32,
		&[
			PushConstant::U32(samples),
			PushConstant::U32(fade_in),
			PushConstant::U32(fade_out),
		],
		crate::core::operation::audio::FADE,
		&[
			unsigned_attribute("fade_in_samples", fade_in_samples),
			unsigned_attribute("fade_out_samples", fade_out_samples),
		],
	)
}

/// Compute a weighted sum of matching Audio values.
pub fn mix(a: &Audio, b: &Audio, gain_a: f32, gain_b: f32) -> Result<Audio> {
	if a.sample_rate() != b.sample_rate()
		|| a.layout() != b.layout()
		|| a.as_matrix().shape() != b.as_matrix().shape()
		|| !a.engine_handle().same_as(b.engine_handle())
		|| !gain_a.is_finite()
		|| !gain_b.is_finite()
	{
		return Err(Error::invalid_argument(
			"audio::mix requires matching ownership, rate, layout, and shape plus finite gains",
		));
	}
	let count = element_count_u32(a, "audio::mix")?;
	let output = wrap_like(allocate_like(a)?, a)?;
	let buffers = [
		BufferBinding::read(a.storage()),
		BufferBinding::read(b.storage()),
		BufferBinding::write(output.storage()),
	];
	let push = [
		PushConstant::U32(count),
		PushConstant::F32(gain_a),
		PushConstant::F32(gain_b),
	];
	let attributes = [
		float_attribute("gain_a", gain_a),
		float_attribute("gain_b", gain_b),
	];
	let outputs = [AudioSemanticOutput::Audio(&output)];
	a.engine_handle().record_audio_semantic(
		ComputeDispatch {
			kernel: KernelId::AudioMixF32,
			buffers: &buffers,
			push_constants: &push,
			workgroups: KernelId::AudioMixF32.linear_workgroups(count),
		},
		AudioSemanticDispatch {
			contract: crate::core::operation::audio::MIX,
			inputs: &[a, b],
			outputs: &outputs,
			attributes: &attributes,
		},
	)?;
	Ok(output)
}

/// Convert absolute amplitude to finite decibels with a silence floor.
pub fn amplitude_to_db(input: &Audio, floor_db: f32) -> Result<Matrix> {
	if !floor_db.is_finite() || !(-300.0..=0.0).contains(&floor_db) {
		return Err(Error::invalid_argument(
			"audio::amplitude_to_db floor_db must be finite and in [-300, 0]",
		));
	}
	let count = element_count_u32(input, "audio::amplitude_to_db")?;
	let output = allocate_like(input)?;
	let buffers = [
		BufferBinding::read(input.storage()),
		BufferBinding::write(output.storage()),
	];
	let push = [PushConstant::U32(count), PushConstant::F32(floor_db)];
	let outputs = [AudioSemanticOutput::Matrix(&output)];
	let attributes = [float_attribute("floor_db", floor_db)];
	input.engine_handle().record_audio_semantic(
		ComputeDispatch {
			kernel: KernelId::AudioAmplitudeToDbF32,
			buffers: &buffers,
			push_constants: &push,
			workgroups: KernelId::AudioAmplitudeToDbF32.linear_workgroups(count),
		},
		AudioSemanticDispatch {
			contract: crate::core::operation::audio::AMPLITUDE_TO_DB,
			inputs: &[input],
			outputs: &outputs,
			attributes: &attributes,
		},
	)?;
	Ok(output)
}

/// Reduce multichannel audio to `[bins, 2]` minimum/maximum pairs.
pub fn waveform_envelope(input: &Audio, bins: u32) -> Result<Matrix> {
	if bins == 0 || bins > 65_536 {
		return Err(Error::invalid_argument(
			"audio::waveform_envelope bins must be in [1, 65536]",
		));
	}
	let channels = u32_extent(input.channels(), "audio::waveform_envelope channels")?;
	let samples = u32_extent(input.samples(), "audio::waveform_envelope samples")?;
	let elements = usize::try_from(bins)
		.ok()
		.and_then(|value| value.checked_mul(2))
		.ok_or_else(|| Error::out_of_range("audio::waveform_envelope output exceeds usize"))?;
	let output = Matrix::allocate(
		input.engine_handle(),
		vec![bins as usize, 2],
		elements,
		crate::DType::F32,
	)?;
	let buffers = [
		BufferBinding::read(input.storage()),
		BufferBinding::write(output.storage()),
	];
	let push = [
		PushConstant::U32(channels),
		PushConstant::U32(samples),
		PushConstant::U32(bins),
	];
	let outputs = [AudioSemanticOutput::Matrix(&output)];
	let attributes = [unsigned_attribute("bins", u64::from(bins))];
	input.engine_handle().record_audio_semantic(
		ComputeDispatch {
			kernel: KernelId::AudioWaveformEnvelopeF32,
			buffers: &buffers,
			push_constants: &push,
			workgroups: KernelId::AudioWaveformEnvelopeF32.linear_workgroups(bins),
		},
		AudioSemanticDispatch {
			contract: crate::core::operation::audio::WAVEFORM_ENVELOPE,
			inputs: &[input],
			outputs: &outputs,
			attributes: &attributes,
		},
	)?;
	Ok(output)
}

fn direct_like(
	input: &Audio,
	kernel: KernelId,
	extra_push: &[PushConstant],
	contract: crate::OperationContract,
	attributes: &[OpAttribute],
) -> Result<Audio> {
	let count = element_count_u32(input, contract.name())?;
	let output = wrap_like(allocate_like(input)?, input)?;
	let mut push = Vec::with_capacity(extra_push.len() + 1);
	push.push(PushConstant::U32(count));
	push.extend_from_slice(extra_push);
	record_audio_direct(input, &output, kernel, &push, contract, attributes)?;
	Ok(output)
}

fn record_audio_direct(
	input: &Audio,
	output: &Audio,
	kernel: KernelId,
	push: &[PushConstant],
	contract: crate::OperationContract,
	attributes: &[OpAttribute],
) -> Result<()> {
	let buffers = [
		BufferBinding::read(input.storage()),
		BufferBinding::write(output.storage()),
	];
	let outputs = [AudioSemanticOutput::Audio(output)];
	input.engine_handle().record_audio_semantic(
		ComputeDispatch {
			kernel,
			buffers: &buffers,
			push_constants: push,
			workgroups: kernel.linear_workgroups(element_count_u32(input, contract.name())?),
		},
		AudioSemanticDispatch {
			contract,
			inputs: &[input],
			outputs: &outputs,
			attributes,
		},
	)
}

fn record_audio_split(
	input: &Audio,
	output: &Audio,
	dispatches: &[ComputeDispatch<'_>],
	contract: crate::OperationContract,
	attributes: &[OpAttribute],
) -> Result<()> {
	let outputs = [AudioSemanticOutput::Audio(output)];
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

fn allocate_like(input: &Audio) -> Result<Matrix> {
	Matrix::allocate(
		input.engine_handle(),
		input.as_matrix().shape().to_vec(),
		input.as_matrix().num_elements(),
		crate::DType::F32,
	)
}

fn wrap_like(matrix: Matrix, input: &Audio) -> Result<Audio> {
	Audio::new(matrix, input.sample_rate(), input.layout())
}

pub(super) fn element_count_u32(input: &Audio, operation: &str) -> Result<u32> {
	u32::try_from(input.as_matrix().num_elements())
		.map_err(|_| Error::out_of_range(format!("{operation} dispatch exceeds u32")))
}

pub(super) fn u32_extent(value: usize, label: &str) -> Result<u32> {
	u32::try_from(value).map_err(|_| Error::out_of_range(format!("{label} exceeds u32")))
}

pub(super) fn float_attribute(name: &str, value: f32) -> OpAttribute {
	OpAttribute::Float {
		name: name.into(),
		value: f64::from(value),
	}
}

pub(super) fn unsigned_attribute(name: &str, value: u64) -> OpAttribute {
	OpAttribute::UnsignedInteger {
		name: name.into(),
		value,
	}
}

fn gcd(mut left: u32, mut right: u32) -> u32 {
	while right != 0 {
		let remainder = left % right;
		left = right;
		right = remainder;
	}
	left
}

pub(super) fn stable_biquad(coefficients: BiquadCoefficients) -> bool {
	if ![
		coefficients.b0,
		coefficients.b1,
		coefficients.b2,
		coefficients.a1,
		coefficients.a2,
	]
	.into_iter()
	.all(f32::is_finite)
	{
		return false;
	}
	let a1 = f64::from(coefficients.a1);
	let a2 = f64::from(coefficients.a2);
	1.0 + a1 + a2 > 0.0 && 1.0 - a1 + a2 > 0.0 && 1.0 - a2 > 0.0
}
