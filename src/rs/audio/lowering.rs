//! Multi-dispatch Audio lowerings kept behind the public operation facade.

use crate::{
	Audio, DType, Error, Matrix, Result,
	runtime::{
		AudioSemanticDispatch, AudioSemanticOutput, BufferBinding, ComputeDispatch, KernelId,
		PushConstant,
	},
};

use super::signal::{
	BiquadCoefficients, float_attribute, stable_biquad, u32_extent, unsigned_attribute,
};

const BIQUAD_BLOCK_SIZE: u32 = 256;

pub(super) fn biquad(
	input: &Audio,
	sections: &[BiquadCoefficients],
	sos_contract: bool,
) -> Result<Audio> {
	if sections.is_empty() || sections.len() > 64 || !sections.iter().copied().all(stable_biquad) {
		return Err(Error::invalid_argument(
			"audio biquad filtering requires one to 64 finite stable a0-normalized sections",
		));
	}
	if sos_contract {
		return sos_filter(input, sections);
	}

	let channels = u32_extent(input.channels(), "audio::biquad channels")?;
	let samples = u32_extent(input.samples(), "audio::biquad samples")?;
	let blocks = samples.div_ceil(BIQUAD_BLOCK_SIZE);
	let tasks = channels
		.checked_mul(blocks)
		.ok_or_else(|| Error::out_of_range("audio::biquad workspace exceeds u32"))?;
	let summary_elements = usize::try_from(tasks)
		.ok()
		.and_then(|value| value.checked_mul(6))
		.ok_or_else(|| Error::out_of_range("audio::biquad summary exceeds usize"))?;
	let state_elements = usize::try_from(tasks)
		.ok()
		.and_then(|value| value.checked_mul(2))
		.ok_or_else(|| Error::out_of_range("audio::biquad state exceeds usize"))?;
	let summaries = Matrix::allocate(
		input.engine_handle(),
		vec![input.channels(), blocks as usize, 6],
		summary_elements,
		DType::F32,
	)?;
	let states = Matrix::allocate(
		input.engine_handle(),
		vec![input.channels(), blocks as usize, 2],
		state_elements,
		DType::F32,
	)?;
	let output_matrix = Matrix::allocate(
		input.engine_handle(),
		input.as_matrix().shape().to_vec(),
		input.as_matrix().num_elements(),
		DType::F32,
	)?;
	let output = Audio::new(output_matrix, input.sample_rate(), input.layout())?;
	let coefficients = sections[0];
	let filter_push = [
		PushConstant::U32(channels),
		PushConstant::U32(samples),
		PushConstant::U32(blocks),
		PushConstant::U32(BIQUAD_BLOCK_SIZE),
		PushConstant::F32(coefficients.b0),
		PushConstant::F32(coefficients.b1),
		PushConstant::F32(coefficients.b2),
		PushConstant::F32(coefficients.a1),
		PushConstant::F32(coefficients.a2),
	];
	let summary_buffers = [
		BufferBinding::read(input.storage()),
		BufferBinding::write(summaries.storage()),
	];
	let scan_buffers = [
		BufferBinding::read(summaries.storage()),
		BufferBinding::write(states.storage()),
	];
	let scan_push = [PushConstant::U32(channels), PushConstant::U32(blocks)];
	let apply_buffers = [
		BufferBinding::read(input.storage()),
		BufferBinding::read(states.storage()),
		BufferBinding::write(output.storage()),
	];
	let dispatches = [
		ComputeDispatch {
			kernel: KernelId::AudioBiquadSummaryF32,
			buffers: &summary_buffers,
			push_constants: &filter_push,
			workgroups: KernelId::AudioBiquadSummaryF32.linear_workgroups(tasks),
		},
		ComputeDispatch {
			kernel: KernelId::AudioBiquadScanF32,
			buffers: &scan_buffers,
			push_constants: &scan_push,
			workgroups: KernelId::AudioBiquadScanF32.linear_workgroups(channels),
		},
		ComputeDispatch {
			kernel: KernelId::AudioBiquadApplyF32,
			buffers: &apply_buffers,
			push_constants: &filter_push,
			workgroups: KernelId::AudioBiquadApplyF32.linear_workgroups(tasks),
		},
	];
	let attributes = [
		float_attribute("b0", coefficients.b0),
		float_attribute("b1", coefficients.b1),
		float_attribute("b2", coefficients.b2),
		float_attribute("a1", coefficients.a1),
		float_attribute("a2", coefficients.a2),
	];
	let outputs = [AudioSemanticOutput::Audio(&output)];
	input.engine_handle().record_audio_split_semantic(
		&dispatches,
		AudioSemanticDispatch {
			contract: crate::core::operation::audio::BIQUAD,
			inputs: &[input],
			outputs: &outputs,
			attributes: &attributes,
		},
	)?;
	Ok(output)
}

fn sos_filter(input: &Audio, sections: &[BiquadCoefficients]) -> Result<Audio> {
	let channels = u32_extent(input.channels(), "audio::sos_filter channels")?;
	let samples = u32_extent(input.samples(), "audio::sos_filter samples")?;
	let section_count = u32::try_from(sections.len())
		.map_err(|_| Error::out_of_range("audio::sos_filter section count exceeds u32"))?;
	let mut packed = Vec::new();
	packed
		.try_reserve_exact(sections.len() * 5)
		.map_err(|_| Error::resource_exhausted("audio::sos_filter coefficient allocation failed"))?;
	for section in sections {
		packed.extend_from_slice(&[section.b0, section.b1, section.b2, section.a1, section.a2]);
	}
	let coefficient_matrix =
		Matrix::from_slice_handle(input.engine_handle(), vec![sections.len(), 5], &packed)?;
	let output_matrix = Matrix::allocate(
		input.engine_handle(),
		input.as_matrix().shape().to_vec(),
		input.as_matrix().num_elements(),
		DType::F32,
	)?;
	let output = Audio::new(output_matrix, input.sample_rate(), input.layout())?;
	let buffers = [
		BufferBinding::read(input.storage()),
		BufferBinding::read(coefficient_matrix.storage()),
		BufferBinding::write(output.storage()),
	];
	let push = [
		PushConstant::U32(channels),
		PushConstant::U32(samples),
		PushConstant::U32(section_count),
	];
	let mut hash = 0xcbf2_9ce4_8422_2325_u64;
	for value in packed {
		for byte in value.to_bits().to_le_bytes() {
			hash = (hash ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3);
		}
	}
	let attributes = [
		unsigned_attribute("section_count", u64::from(section_count)),
		unsigned_attribute("coefficient_hash", hash),
	];
	let outputs = [AudioSemanticOutput::Audio(&output)];
	input.engine_handle().record_audio_semantic(
		ComputeDispatch {
			kernel: KernelId::AudioSosFilterF32,
			buffers: &buffers,
			push_constants: &push,
			workgroups: KernelId::AudioSosFilterF32.linear_workgroups(channels),
		},
		AudioSemanticDispatch {
			contract: crate::core::operation::audio::SOS_FILTER,
			inputs: &[input],
			outputs: &outputs,
			attributes: &attributes,
		},
	)?;
	Ok(output)
}

pub(super) fn reverb(input: &Audio, decay_seconds: f32, wet: f32) -> Result<Audio> {
	if !decay_seconds.is_finite()
		|| !(0.1..=10.0).contains(&decay_seconds)
		|| !wet.is_finite()
		|| !(0.0..=1.0).contains(&wet)
	{
		return Err(Error::invalid_argument(
			"audio::reverb requires decay_seconds in [0.1, 10] and wet in [0, 1]",
		));
	}
	let channels = u32_extent(input.channels(), "audio::reverb channels")?;
	let in_samples = u32_extent(input.samples(), "audio::reverb input samples")?;
	let tail = (f64::from(input.sample_rate()) * f64::from(decay_seconds)).ceil();
	if !(1.0..=f64::from(u32::MAX)).contains(&tail) {
		return Err(Error::out_of_range("audio::reverb tail exceeds u32"));
	}
	let tail = tail as u32;
	let out_samples = in_samples
		.checked_add(tail)
		.ok_or_else(|| Error::out_of_range("audio::reverb output samples exceed u32"))?;
	let count = channels
		.checked_mul(out_samples)
		.ok_or_else(|| Error::out_of_range("audio::reverb output dispatch exceeds u32"))?;
	let output_shape = [input.channels(), out_samples as usize];
	let allocate_output = || {
		Matrix::allocate(
			input.engine_handle(),
			output_shape.to_vec(),
			count as usize,
			DType::F32,
		)
	};
	let combs = [
		allocate_output()?,
		allocate_output()?,
		allocate_output()?,
		allocate_output()?,
	];
	let summed = allocate_output()?;
	let diffuse_a = allocate_output()?;
	let diffuse_b = allocate_output()?;
	let output = Audio::new(allocate_output()?, input.sample_rate(), input.layout())?;

	let delay_seconds = [0.0297_f64, 0.0371, 0.0411, 0.0437];
	let mut delays = [0_u32; 4];
	let mut feedback = [0_f32; 4];
	let mut denominator = 0.0_f64;
	for index in 0..4 {
		delays[index] = (delay_seconds[index] * f64::from(input.sample_rate()))
			.round()
			.max(1.0) as u32;
		feedback[index] = 0.001_f64
			.powf((f64::from(delays[index]) / f64::from(input.sample_rate())) / f64::from(decay_seconds))
			as f32;
		denominator += 1.0 / (1.0 - f64::from(feedback[index]));
	}
	let allpass_delays = [
		(0.0050_f64 * f64::from(input.sample_rate()))
			.round()
			.max(1.0) as u32,
		(0.0017_f64 * f64::from(input.sample_rate()))
			.round()
			.max(1.0) as u32,
	];
	for delay in delays.into_iter().chain(allpass_delays) {
		channels
			.checked_mul(delay)
			.ok_or_else(|| Error::out_of_range("audio::reverb recurrence dispatch exceeds u32"))?;
	}

	let comb_buffers_0 = [
		BufferBinding::read(input.storage()),
		BufferBinding::write(combs[0].storage()),
	];
	let comb_buffers_1 = [
		BufferBinding::read(input.storage()),
		BufferBinding::write(combs[1].storage()),
	];
	let comb_buffers_2 = [
		BufferBinding::read(input.storage()),
		BufferBinding::write(combs[2].storage()),
	];
	let comb_buffers_3 = [
		BufferBinding::read(input.storage()),
		BufferBinding::write(combs[3].storage()),
	];
	let comb_push_0 = comb_push(channels, in_samples, out_samples, delays[0], feedback[0]);
	let comb_push_1 = comb_push(channels, in_samples, out_samples, delays[1], feedback[1]);
	let comb_push_2 = comb_push(channels, in_samples, out_samples, delays[2], feedback[2]);
	let comb_push_3 = comb_push(channels, in_samples, out_samples, delays[3], feedback[3]);
	let sum_buffers = [
		BufferBinding::read(combs[0].storage()),
		BufferBinding::read(combs[1].storage()),
		BufferBinding::read(combs[2].storage()),
		BufferBinding::read(combs[3].storage()),
		BufferBinding::write(summed.storage()),
	];
	let sum_push = [
		PushConstant::U32(count),
		PushConstant::F32((1.0 / denominator) as f32),
	];
	let allpass_a_buffers = [
		BufferBinding::read(summed.storage()),
		BufferBinding::write(diffuse_a.storage()),
	];
	let allpass_b_buffers = [
		BufferBinding::read(diffuse_a.storage()),
		BufferBinding::write(diffuse_b.storage()),
	];
	let allpass_a_push = allpass_push(channels, out_samples, allpass_delays[0]);
	let allpass_b_push = allpass_push(channels, out_samples, allpass_delays[1]);
	let mix_buffers = [
		BufferBinding::read(input.storage()),
		BufferBinding::read(diffuse_b.storage()),
		BufferBinding::write(output.storage()),
	];
	let mix_push = [
		PushConstant::U32(channels),
		PushConstant::U32(in_samples),
		PushConstant::U32(out_samples),
		PushConstant::F32(wet),
	];
	let dispatches = [
		comb_dispatch(&comb_buffers_0, &comb_push_0, channels * delays[0]),
		comb_dispatch(&comb_buffers_1, &comb_push_1, channels * delays[1]),
		comb_dispatch(&comb_buffers_2, &comb_push_2, channels * delays[2]),
		comb_dispatch(&comb_buffers_3, &comb_push_3, channels * delays[3]),
		ComputeDispatch {
			kernel: KernelId::AudioReverbSumF32,
			buffers: &sum_buffers,
			push_constants: &sum_push,
			workgroups: KernelId::AudioReverbSumF32.linear_workgroups(count),
		},
		allpass_dispatch(
			&allpass_a_buffers,
			&allpass_a_push,
			channels * allpass_delays[0],
		),
		allpass_dispatch(
			&allpass_b_buffers,
			&allpass_b_push,
			channels * allpass_delays[1],
		),
		ComputeDispatch {
			kernel: KernelId::AudioReverbMixF32,
			buffers: &mix_buffers,
			push_constants: &mix_push,
			workgroups: KernelId::AudioReverbMixF32.linear_workgroups(count),
		},
	];
	let attributes = [
		float_attribute("decay_seconds", decay_seconds),
		float_attribute("wet", wet),
	];
	let outputs = [AudioSemanticOutput::Audio(&output)];
	input.engine_handle().record_audio_split_semantic(
		&dispatches,
		AudioSemanticDispatch {
			contract: crate::core::operation::audio::REVERB,
			inputs: &[input],
			outputs: &outputs,
			attributes: &attributes,
		},
	)?;
	Ok(output)
}

fn comb_push(
	channels: u32,
	in_samples: u32,
	out_samples: u32,
	delay: u32,
	feedback: f32,
) -> [PushConstant; 5] {
	[
		PushConstant::U32(channels),
		PushConstant::U32(in_samples),
		PushConstant::U32(out_samples),
		PushConstant::U32(delay),
		PushConstant::F32(feedback),
	]
}

fn allpass_push(channels: u32, samples: u32, delay: u32) -> [PushConstant; 4] {
	[
		PushConstant::U32(channels),
		PushConstant::U32(samples),
		PushConstant::U32(delay),
		PushConstant::F32(0.7),
	]
}

fn comb_dispatch<'a>(
	buffers: &'a [BufferBinding<'a>],
	push: &'a [PushConstant],
	tasks: u32,
) -> ComputeDispatch<'a> {
	ComputeDispatch {
		kernel: KernelId::AudioReverbCombF32,
		buffers,
		push_constants: push,
		workgroups: KernelId::AudioReverbCombF32.linear_workgroups(tasks),
	}
}

fn allpass_dispatch<'a>(
	buffers: &'a [BufferBinding<'a>],
	push: &'a [PushConstant],
	tasks: u32,
) -> ComputeDispatch<'a> {
	ComputeDispatch {
		kernel: KernelId::AudioReverbAllpassF32,
		buffers,
		push_constants: push,
		workgroups: KernelId::AudioReverbAllpassF32.linear_workgroups(tasks),
	}
}
