//! Real-time device capture into a bounded lock-free FP32 ring.

use std::{
	sync::{
		Arc, Mutex, OnceLock,
		atomic::{AtomicBool, AtomicU64, Ordering},
	},
	thread,
	time::Instant,
};

use cpal::{
	Data, FromSample, Sample, SampleFormat, SizedSample, Stream,
	traits::{DeviceTrait, HostTrait, StreamTrait},
};
use rtrb::{Consumer, Producer, PushError, RingBuffer};

use super::player::{cpal_error, ring_capacity, select_config};
use crate::{Engine, Error, Result, runtime::EngineHandle};

static MONOTONIC_EPOCH: OnceLock<Instant> = OnceLock::new();

/// Configuration for one default-device capture session.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AudioCaptureConfig {
	pub sample_rate: u32,
	pub channel_count: u32,
	pub ring_milliseconds: u32,
}

impl Default for AudioCaptureConfig {
	fn default() -> Self {
		Self {
			sample_rate: 48_000,
			channel_count: 2,
			ring_milliseconds: 500,
		}
	}
}

/// One contiguous run of captured interleaved FP32 frames.
#[derive(Clone, Debug, PartialEq)]
pub struct AudioCaptureChunk {
	pub interleaved: Vec<f32>,
	pub sample_rate: u32,
	pub channel_count: u32,
	pub frame_count: u64,
	pub first_frame_index: u64,
	/// Process-monotonic presentation timestamp in microseconds.
	pub presentation_timestamp_us: u64,
}

/// Stateful default-device audio capture session.
pub struct AudioCapture {
	_engine: EngineHandle,
	service: Option<CaptureService>,
	shared: Arc<CaptureShared>,
	consumer: Consumer<CaptureFrame>,
	pending: Option<CaptureFrame>,
	config: AudioCaptureConfig,
}

struct CaptureService {
	stream: Stream,
}

struct CaptureShared {
	accepting: AtomicBool,
	started: AtomicBool,
	captured_frames: AtomicU64,
	dropped_frames: AtomicU64,
	epoch_us: AtomicU64,
	device_errors: AtomicU64,
}

#[derive(Clone, Copy, Default)]
struct CaptureFrame {
	samples: [f32; 8],
	index: u64,
}

impl AudioCapture {
	/// Open the default input device. The returned session is initially stopped.
	pub fn open(engine: &Engine, config: AudioCaptureConfig) -> Result<Self> {
		if config.sample_rate == 0
			|| !(1..=8).contains(&config.channel_count)
			|| config.ring_milliseconds < 20
		{
			return Err(Error::invalid_argument(
				"AudioCapture requires a sample rate, 1..8 channels, and at least 20 ms of ring storage",
			));
		}
		let capacity = ring_capacity(config.sample_rate, config.ring_milliseconds, "AudioCapture")?;
		// Initialize process timing before the real-time callback can run.
		let _ = MONOTONIC_EPOCH.get_or_init(Instant::now);
		let (mut producer, consumer) = RingBuffer::<CaptureFrame>::new(capacity);
		let shared = Arc::new(CaptureShared {
			accepting: AtomicBool::new(false),
			started: AtomicBool::new(false),
			captured_frames: AtomicU64::new(0),
			dropped_frames: AtomicU64::new(0),
			epoch_us: AtomicU64::new(0),
			device_errors: AtomicU64::new(0),
		});

		let host = cpal::default_host();
		let device = host
			.default_input_device()
			.ok_or_else(|| Error::no_suitable_device("AudioCapture found no default input device"))?;
		let ranges = device
			.supported_input_configs()
			.map_err(|error| cpal_error("input configuration query", error))?;
		let stream_config = select_config(ranges, config.sample_rate, config.channel_count, "input")?;
		let sample_format = stream_config.sample_format();
		let callback_shared = Arc::clone(&shared);
		let error_shared = Arc::clone(&shared);
		let stream = device
			.build_input_stream_raw(
				stream_config.config(),
				sample_format,
				move |input, _| {
					capture_callback(
						&callback_shared,
						&mut producer,
						input,
						config.sample_rate,
						config.channel_count,
					);
				},
				move |_| {
					error_shared.device_errors.fetch_add(1, Ordering::Relaxed);
				},
				None,
			)
			.map_err(|error| cpal_error("capture stream creation", error))?;
		Ok(Self {
			_engine: engine.handle(),
			service: Some(CaptureService { stream }),
			shared,
			consumer,
			pending: None,
			config,
		})
	}

	/// Start capture and reset frame indices, timestamps, and drop counters.
	pub fn start(&mut self) -> Result<()> {
		let service = self
			.service
			.as_ref()
			.ok_or_else(|| Error::failed_precondition("AudioCapture is closed"))?;
		if self.shared.started.load(Ordering::Acquire) {
			return Ok(());
		}
		while self.consumer.pop().is_ok() {}
		self.pending = None;
		self.shared.captured_frames.store(0, Ordering::Relaxed);
		self.shared.dropped_frames.store(0, Ordering::Relaxed);
		self.shared.device_errors.store(0, Ordering::Relaxed);
		self.shared.epoch_us.store(0, Ordering::Release);
		self.shared.accepting.store(true, Ordering::Release);
		if let Err(error) = service.stream.play() {
			self.shared.accepting.store(false, Ordering::Release);
			return Err(cpal_error("capture stream start", error));
		}
		self.shared.started.store(true, Ordering::Release);
		Ok(())
	}

	/// Stop callback delivery without releasing the device.
	pub fn stop(&mut self) -> Result<()> {
		self.shared.accepting.store(false, Ordering::Release);
		if !self.shared.started.swap(false, Ordering::AcqRel) {
			return Ok(());
		}
		let Some(service) = self.service.as_ref() else {
			return Ok(());
		};
		service
			.stream
			.pause()
			.map_err(|error| cpal_error("capture stream stop", error))
	}

	/// Return the next contiguous frame run without blocking.
	///
	/// A physical capture gap caused by a full ring ends the current chunk. The
	/// next call begins at the first frame after that gap.
	pub fn poll(&mut self, max_frames: u32) -> Option<AudioCaptureChunk> {
		if self.service.is_none() || max_frames == 0 {
			return None;
		}
		let first = self.pending.take().or_else(|| self.consumer.pop().ok())?;
		let mut interleaved =
			Vec::with_capacity(max_frames as usize * self.config.channel_count as usize);
		interleaved.extend_from_slice(&first.samples[..self.config.channel_count as usize]);
		let mut expected = first.index.saturating_add(1);
		let mut frame_count = 1_u64;
		while frame_count < u64::from(max_frames) {
			let Ok(frame) = self.consumer.pop() else {
				break;
			};
			if frame.index != expected {
				self.pending = Some(frame);
				break;
			}
			interleaved.extend_from_slice(&frame.samples[..self.config.channel_count as usize]);
			expected = expected.saturating_add(1);
			frame_count += 1;
		}
		let epoch = self.shared.epoch_us.load(Ordering::Acquire);
		Some(AudioCaptureChunk {
			interleaved,
			sample_rate: self.config.sample_rate,
			channel_count: self.config.channel_count,
			frame_count,
			first_frame_index: first.index,
			presentation_timestamp_us: epoch
				.saturating_add(first.index.saturating_mul(1_000_000) / u64::from(self.config.sample_rate)),
		})
	}

	/// Stop capture and release the device at an explicit result-bearing boundary.
	pub fn close(&mut self) -> Result<()> {
		let Some(service) = self.service.take() else {
			return Ok(());
		};
		self.shared.accepting.store(false, Ordering::Release);
		let was_started = self.shared.started.swap(false, Ordering::AcqRel);
		service.close(was_started)
	}

	pub fn is_open(&self) -> bool {
		self.service.is_some()
	}

	pub fn is_started(&self) -> bool {
		self.is_open() && self.shared.started.load(Ordering::Acquire)
	}

	pub fn dropped_frame_count(&self) -> u64 {
		self.shared.dropped_frames.load(Ordering::Relaxed)
	}

	pub fn device_error_count(&self) -> u64 {
		self.shared.device_errors.load(Ordering::Relaxed)
	}
}

impl Drop for AudioCapture {
	fn drop(&mut self) {
		let Some(service) = self.service.take() else {
			return;
		};
		self.shared.accepting.store(false, Ordering::Release);
		let was_started = self.shared.started.swap(false, Ordering::AcqRel);
		retire_capture(service, was_started);
	}
}

impl CaptureService {
	fn close(self, was_started: bool) -> Result<()> {
		if was_started {
			self
				.stream
				.pause()
				.map_err(|error| cpal_error("capture stream stop", error))?;
		}
		Ok(())
	}
}

fn retire_capture(service: CaptureService, was_started: bool) {
	let retirement = Arc::new(Mutex::new(Some(service)));
	let worker = Arc::clone(&retirement);
	let spawned = thread::Builder::new()
		.name("oa-audio-capture-retirement".into())
		.spawn(move || {
			if let Ok(mut slot) = worker.lock()
				&& let Some(service) = slot.take()
			{
				let _ = service.close(was_started);
			}
		});
	if spawned.is_err() {
		// A worker-spawn failure must not turn Drop into an implicit device stop.
		std::mem::forget(retirement);
	}
}

fn capture_callback(
	shared: &CaptureShared,
	producer: &mut Producer<CaptureFrame>,
	input: &Data,
	sample_rate: u32,
	channels: u32,
) {
	if !shared.accepting.load(Ordering::Acquire) {
		return;
	}
	match input.sample_format() {
		SampleFormat::I8 => capture_input::<i8>(shared, producer, input, sample_rate, channels),
		SampleFormat::I16 => capture_input::<i16>(shared, producer, input, sample_rate, channels),
		SampleFormat::I24 => capture_input::<cpal::I24>(shared, producer, input, sample_rate, channels),
		SampleFormat::I32 => capture_input::<i32>(shared, producer, input, sample_rate, channels),
		SampleFormat::I64 => capture_input::<i64>(shared, producer, input, sample_rate, channels),
		SampleFormat::U8 => capture_input::<u8>(shared, producer, input, sample_rate, channels),
		SampleFormat::U16 => capture_input::<u16>(shared, producer, input, sample_rate, channels),
		SampleFormat::U24 => capture_input::<cpal::U24>(shared, producer, input, sample_rate, channels),
		SampleFormat::U32 => capture_input::<u32>(shared, producer, input, sample_rate, channels),
		SampleFormat::U64 => capture_input::<u64>(shared, producer, input, sample_rate, channels),
		SampleFormat::F32 => capture_input::<f32>(shared, producer, input, sample_rate, channels),
		SampleFormat::F64 => capture_input::<f64>(shared, producer, input, sample_rate, channels),
		_ => {}
	}
}

fn capture_input<T>(
	shared: &CaptureShared,
	producer: &mut Producer<CaptureFrame>,
	input: &Data,
	sample_rate: u32,
	channels: u32,
) where
	T: SizedSample + Copy,
	f32: FromSample<T>,
{
	let Some(samples) = input.as_slice::<T>() else {
		return;
	};
	let channels = channels as usize;
	let frame_count = samples.len() / channels;
	if frame_count == 0 {
		return;
	}
	let first_index = shared
		.captured_frames
		.fetch_add(frame_count as u64, Ordering::Relaxed);
	if shared.epoch_us.load(Ordering::Relaxed) == 0 {
		let duration = (frame_count as u64).saturating_mul(1_000_000) / u64::from(sample_rate);
		let epoch = monotonic_microseconds().saturating_sub(duration).max(1);
		let _ = shared
			.epoch_us
			.compare_exchange(0, epoch, Ordering::Release, Ordering::Relaxed);
	}
	for frame_offset in 0..frame_count {
		let mut frame = CaptureFrame {
			index: first_index.saturating_add(frame_offset as u64),
			..CaptureFrame::default()
		};
		let begin = frame_offset * channels;
		for channel in 0..channels {
			frame.samples[channel] = <f32 as Sample>::from_sample(samples[begin + channel]);
		}
		if let Err(PushError::Full(_)) = producer.push(frame) {
			shared
				.dropped_frames
				.fetch_add((frame_count - frame_offset) as u64, Ordering::Relaxed);
			break;
		}
	}
}

fn monotonic_microseconds() -> u64 {
	let Some(epoch) = MONOTONIC_EPOCH.get() else {
		return 0;
	};
	u64::try_from(epoch.elapsed().as_micros()).unwrap_or(u64::MAX)
}
