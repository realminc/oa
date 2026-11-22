//! Real-time audio playback, transport, and output clock.

pub use crate::audio::types::AudioPlayerConfig;
use std::{
	path::{Path, PathBuf},
	sync::{
		Arc, Condvar, Mutex, OnceLock,
		atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering},
	},
	thread::{self, JoinHandle},
	time::{Duration, Instant},
};

use crate::{Engine, Error, Result, runtime::EngineHandle};
use cpal::{
	Data, FromSample, OutputCallbackInfo, SampleFormat, SizedSample, Stream, SupportedStreamConfig,
	SupportedStreamConfigRange,
	traits::{DeviceTrait, StreamTrait},
};
use rtrb::{Consumer, Producer, PushError, RingBuffer};

use super::decoder::{StreamingDecoder, frames_to_us, us_to_frames};

impl AudioPlayerConfig {
	pub fn new(uri: impl Into<PathBuf>) -> Self {
		Self {
			uri: uri.into(),
			loop_playback: false,
			ring_milliseconds: 500,
		}
	}
}

/// Incremental decoded PCM playback through the default output device.
pub struct AudioPlayer {
	_engine: EngineHandle,
	service: Option<PlayerService>,
	shared: Arc<PlayerShared>,
	sample_rate: u32,
	channels: u32,
	duration_us: u64,
	track_id: u32,
}

struct PlayerService {
	stream: Stream,
	decoder_thread: Option<JoinHandle<()>>,
}

struct PlayerShared {
	playing: AtomicBool,
	eos: AtomicBool,
	stop: AtomicBool,
	loop_playback: AtomicBool,
	muted: AtomicBool,
	volume: AtomicU32,
	peak: AtomicU32,
	end_frame: AtomicU64,
	output: Mutex<OutputClock>,
	generation: AtomicU64,
	position_frame: AtomicU64,
	underrun_frames: AtomicU64,
	device_errors: OutputErrors,
	wake: Condvar,
	control: Mutex<PlayerControl>,
}

#[derive(Default)]
struct OutputErrors {
	count: AtomicU64,
	failure: OnceLock<cpal::Error>,
}

impl OutputErrors {
	fn record(&self, error: cpal::Error) {
		self.count.fetch_add(1, Ordering::Relaxed);
		// CPAL keeps the stream running after these notifications. A failed
		// backend recovery arrives separately as a terminal device error.
		if !matches!(
			error.kind(),
			cpal::ErrorKind::Xrun | cpal::ErrorKind::DeviceChanged | cpal::ErrorKind::RealtimeDenied
		) {
			let _ = self.failure.set(error);
		}
	}
}

#[derive(Default)]
struct PlayerControl {
	next_seek_serial: u64,
	applied_seek_serial: u64,
	seek_request: Option<(u64, u64)>,
	seek_error: Option<String>,
	decode_error: Option<Error>,
}

#[derive(Default)]
struct OutputClock {
	origin_frame: u64,
	end_frame: u64,
	ends_at: Option<Instant>,
}

impl OutputClock {
	fn audible_frame(&self, now: Instant, sample_rate: u32) -> u64 {
		let remaining = self
			.ends_at
			.map_or(0, |end| end.saturating_duration_since(now).as_nanos());
		let queued = (remaining * u128::from(sample_rate)).div_ceil(1_000_000_000);
		self
			.end_frame
			.saturating_sub(u64::try_from(queued).unwrap_or(u64::MAX))
			.max(self.origin_frame)
	}
}

#[derive(Clone, Copy, Default)]
struct PlaybackFrame {
	samples: [f32; 8],
	generation: u64,
}

impl AudioPlayer {
	/// Open the source and default output device with decoding initially paused.
	///
	/// # Errors
	///
	/// Returns an error for an empty URI, a ring outside 40–5000 ms, or when
	/// the selected audio track and output device cannot be opened.
	pub fn open(engine: &Engine, config: AudioPlayerConfig) -> Result<Self> {
		validate_config(&config)?;
		let (decoder, duration_us) = StreamingDecoder::open(&config.uri, false)?
			.ok_or_else(|| Error::invalid_argument("AudioPlayer source has no audio track"))?;
		Self::open_decoder(engine, config, decoder, duration_us)
	}

	pub(crate) fn open_embedded(engine: &Engine, config: AudioPlayerConfig) -> Result<Option<Self>> {
		validate_config(&config)?;
		StreamingDecoder::open(&config.uri, true)?
			.map(|(decoder, duration)| Self::open_decoder(engine, config, decoder, duration))
			.transpose()
	}

	fn open_decoder(
		engine: &Engine,
		config: AudioPlayerConfig,
		decoder: StreamingDecoder,
		duration_us: u64,
	) -> Result<Self> {
		let sample_rate = decoder.sample_rate;
		let track_id = decoder.track_id;
		if !(1..=384_000).contains(&sample_rate) {
			return Err(Error::missing_capability(
				"AudioPlayer sample rate exceeds the supported PCM range",
			));
		}
		let channels = u32::try_from(decoder.channels)
			.map_err(|_| Error::out_of_range("AudioPlayer channel count exceeds u32"))?;
		if !(1..=8).contains(&channels) {
			return Err(Error::invalid_argument(
				"AudioPlayer supports streams with one to eight channels",
			));
		}
		let capacity = ring_capacity(sample_rate, config.ring_milliseconds, "AudioPlayer")?;
		let (sender, mut receiver) = RingBuffer::<PlaybackFrame>::new(capacity);
		let shared = Arc::new(PlayerShared {
			playing: AtomicBool::new(false),
			eos: AtomicBool::new(false),
			stop: AtomicBool::new(false),
			loop_playback: AtomicBool::new(config.loop_playback),
			muted: AtomicBool::new(false),
			volume: AtomicU32::new(1.0_f32.to_bits()),
			peak: AtomicU32::new(0),
			end_frame: AtomicU64::new(u64::MAX),
			output: Mutex::new(OutputClock::default()),
			generation: AtomicU64::new(0),
			position_frame: AtomicU64::new(0),
			underrun_frames: AtomicU64::new(0),
			device_errors: OutputErrors::default(),
			wake: Condvar::new(),
			control: Mutex::new(PlayerControl::default()),
		});

		let host = cpal::default_host();
		let (device, stream_config) = super::device::select_device(
			engine,
			&host,
			super::device::Direction::Output,
			sample_rate,
			channels,
		)?;
		let sample_format = stream_config.sample_format();
		let callback_shared = Arc::clone(&shared);
		let error_shared = Arc::clone(&shared);
		let stream = device
			.build_output_stream_raw(
				stream_config.config(),
				sample_format,
				move |output, info| {
					playback_callback(
						&callback_shared,
						&mut receiver,
						output,
						info,
						channels,
						sample_rate,
					);
				},
				move |error| error_shared.device_errors.record(error),
				None,
			)
			.map_err(|error| cpal_error("playback stream creation", error))?;
		stream
			.play()
			.map_err(|error| cpal_error("playback stream start", error))?;

		let thread_shared = Arc::clone(&shared);
		let decoder_thread = thread::Builder::new()
			.name("oa-audio-decode".into())
			.spawn(move || decode_loop(decoder, sender, capacity, thread_shared))
			.map_err(|source| Error::backend_failure("host", "AudioPlayer decoder thread", source))?;
		Ok(Self {
			_engine: engine.handle(),
			service: Some(PlayerService {
				stream,
				decoder_thread: Some(decoder_thread),
			}),
			shared,
			sample_rate,
			channels,
			duration_us,
			track_id,
		})
	}

	/// Open a URI using the default playback configuration.
	pub fn new_from_uri(engine: &Engine, uri: impl AsRef<Path>) -> Result<Self> {
		Self::open(engine, AudioPlayerConfig::new(uri.as_ref()))
	}

	/// Start or resume playback, rewinding after end-of-stream.
	pub fn play(&mut self) -> Result<()> {
		self.require_open("play")?;
		self.check_playback()?;
		if self.is_eos() {
			self.seek(0)?;
		}
		self.shared.eos.store(false, Ordering::Release);
		self.shared.playing.store(true, Ordering::Release);
		self.shared.wake.notify_all();
		Ok(())
	}

	/// Pause playback without disturbing decoder position.
	pub fn pause(&self) {
		self.shared.playing.store(false, Ordering::Release);
		self.shared.peak.store(0, Ordering::Relaxed);
	}

	/// Seek to a timestamp in microseconds and wait for decoder acknowledgement.
	pub fn seek(&mut self, timestamp_us: u64) -> Result<()> {
		self.require_open("seek")?;
		let resume = self.shared.playing.swap(false, Ordering::AcqRel);
		let serial = {
			let mut control = self
				.shared
				.control
				.lock()
				.map_err(|_| Error::internal("AudioPlayer seek control was poisoned"))?;
			control.next_seek_serial = control
				.next_seek_serial
				.checked_add(1)
				.ok_or_else(|| Error::resource_exhausted("AudioPlayer seek serial exhausted"))?;
			let serial = control.next_seek_serial;
			control.seek_request = Some((serial, timestamp_us));
			control.seek_error = None;
			self.shared.generation.store(serial, Ordering::Release);
			serial
		};
		self.shared.wake.notify_all();
		let control = self
			.shared
			.control
			.lock()
			.map_err(|_| Error::internal("AudioPlayer seek control was poisoned"))?;
		let (control, timeout) = self
			.shared
			.wake
			.wait_timeout_while(control, Duration::from_secs(2), |control| {
				control.applied_seek_serial < serial
			})
			.map_err(|_| Error::internal("AudioPlayer seek control was poisoned"))?;
		if timeout.timed_out() || control.applied_seek_serial < serial {
			return Err(Error::backend_failure(
				"audio decoder",
				"AudioPlayer seek",
				std::io::Error::new(
					std::io::ErrorKind::TimedOut,
					"seek acknowledgement timed out",
				),
			));
		}
		if let Some(message) = &control.seek_error {
			return Err(Error::backend_failure(
				"audio decoder",
				"AudioPlayer seek",
				std::io::Error::other(message.clone()),
			));
		}
		drop(control);
		if resume {
			self.shared.playing.store(true, Ordering::Release);
			self.shared.wake.notify_all();
		}
		Ok(())
	}

	pub fn set_loop(&self, enabled: bool) {
		self.shared.loop_playback.store(enabled, Ordering::Release);
	}

	pub fn set_muted(&self, muted: bool) -> Result<()> {
		self.require_open("set_muted")?;
		self.shared.muted.store(muted, Ordering::Release);
		Ok(())
	}

	pub(crate) fn set_volume(&self, volume: f32) -> Result<()> {
		self.require_open("set_volume")?;
		if !(0.0..=1.0).contains(&volume) {
			return Err(Error::invalid_argument(
				"AudioPlayer volume must be between 0 and 1",
			));
		}
		self
			.shared
			.volume
			.store(volume.to_bits(), Ordering::Relaxed);
		Ok(())
	}

	pub(crate) fn volume(&self) -> f32 {
		f32::from_bits(self.shared.volume.load(Ordering::Relaxed))
	}

	pub(crate) fn output_peak(&self) -> f32 {
		if self.is_playing() {
			f32::from_bits(self.shared.peak.load(Ordering::Relaxed))
		} else {
			0.0
		}
	}

	pub(crate) fn audible_position_us(&self) -> u64 {
		self.shared.output.lock().map_or_else(
			|_| self.position_us(),
			|clock| {
				frames_to_us(
					clock.audible_frame(Instant::now(), self.sample_rate),
					self.sample_rate,
				)
			},
		)
	}

	pub(crate) fn check_playback(&self) -> Result<()> {
		if let Some(error) = self.shared.device_errors.failure.get() {
			return Err(Error::unavailable(format!(
				"AudioPlayer output device failed: {error}"
			)));
		}
		let control = self
			.shared
			.control
			.lock()
			.map_err(|_| Error::internal("AudioPlayer playback control was poisoned"))?;
		if let Some(error) = &control.decode_error {
			return Err(Error::data_loss(format!("AudioPlayer stopped: {error}")));
		}
		Ok(())
	}

	pub(crate) fn track_id(&self) -> u32 {
		self.track_id
	}

	pub(crate) fn set_playback_end_us(&self, end: u64) {
		self
			.shared
			.end_frame
			.store(us_to_frames(end, self.sample_rate), Ordering::Release);
	}

	/// Stop decoding and release the device at an explicit result-bearing boundary.
	pub fn close(&mut self) -> Result<()> {
		let Some(service) = self.service.take() else {
			return Ok(());
		};
		self.shared.stop.store(true, Ordering::Release);
		self.shared.playing.store(false, Ordering::Release);
		self.shared.wake.notify_all();
		service.close()
	}

	pub const fn is_open(&self) -> bool {
		self.service.is_some()
	}

	pub fn is_playing(&self) -> bool {
		self.is_open() && self.shared.playing.load(Ordering::Acquire)
	}

	pub fn is_eos(&self) -> bool {
		!self.is_open() || self.shared.eos.load(Ordering::Acquire)
	}

	pub fn is_muted(&self) -> bool {
		self.is_open() && self.shared.muted.load(Ordering::Acquire)
	}

	pub const fn sample_rate(&self) -> u32 {
		self.sample_rate
	}

	pub const fn channel_count(&self) -> u32 {
		self.channels
	}

	pub const fn duration_us(&self) -> u64 {
		self.duration_us
	}

	pub fn position_us(&self) -> u64 {
		frames_to_us(
			self.shared.position_frame.load(Ordering::Acquire),
			self.sample_rate,
		)
	}

	pub fn underrun_frame_count(&self) -> u64 {
		self.shared.underrun_frames.load(Ordering::Relaxed)
	}

	pub fn device_error_count(&self) -> u64 {
		self.shared.device_errors.count.load(Ordering::Relaxed)
	}

	fn require_open(&self, operation: &str) -> Result<()> {
		if self.is_open() {
			Ok(())
		} else {
			Err(Error::failed_precondition(format!(
				"AudioPlayer::{operation} called on a closed session"
			)))
		}
	}
}

impl Drop for AudioPlayer {
	fn drop(&mut self) {
		let Some(service) = self.service.take() else {
			return;
		};
		self.shared.stop.store(true, Ordering::Release);
		self.shared.playing.store(false, Ordering::Release);
		self.shared.wake.notify_all();
		retire_player(service);
	}
}

impl PlayerService {
	fn close(mut self) -> Result<()> {
		// CPAL stream destruction stops and joins the callback worker. Requiring
		// hardware pause first rejects valid ALSA desktop-server bridges.
		drop(self.stream);
		if let Some(handle) = self.decoder_thread.take() {
			handle
				.join()
				.map_err(|_| Error::internal("AudioPlayer decoder thread panicked"))?;
		}
		Ok(())
	}
}

fn retire_player(service: PlayerService) {
	let retirement = Arc::new(Mutex::new(Some(service)));
	let worker = Arc::clone(&retirement);
	let spawned = thread::Builder::new()
		.name("oa-audio-player-retirement".into())
		.spawn(move || {
			if let Ok(mut slot) = worker.lock()
				&& let Some(service) = slot.take()
			{
				let _ = service.close();
			}
		});
	if spawned.is_err() {
		// A worker-spawn failure must not turn Drop into an implicit stop/join.
		std::mem::forget(retirement);
	}
}

fn playback_callback(
	shared: &PlayerShared,
	receiver: &mut Consumer<PlaybackFrame>,
	output: &mut Data,
	info: &OutputCallbackInfo,
	channels: u32,
	sample_rate: u32,
) {
	// CPAL's timestamps share the device's clock epoch. Carry over their
	// difference to the host monotonic clock and account for queued samples.
	let timestamp = info.timestamp();
	let start = Instant::now().checked_add(timestamp.playback.duration_since(timestamp.callback));
	let buffer = OutputBuffer {
		channels,
		sample_rate,
		start,
	};
	match output.sample_format() {
		SampleFormat::I8 => render_output::<i8>(shared, receiver, output, buffer),
		SampleFormat::I16 => render_output::<i16>(shared, receiver, output, buffer),
		SampleFormat::I24 => render_output::<cpal::I24>(shared, receiver, output, buffer),
		SampleFormat::I32 => render_output::<i32>(shared, receiver, output, buffer),
		SampleFormat::I64 => render_output::<i64>(shared, receiver, output, buffer),
		SampleFormat::U8 => render_output::<u8>(shared, receiver, output, buffer),
		SampleFormat::U16 => render_output::<u16>(shared, receiver, output, buffer),
		SampleFormat::U24 => render_output::<cpal::U24>(shared, receiver, output, buffer),
		SampleFormat::U32 => render_output::<u32>(shared, receiver, output, buffer),
		SampleFormat::U64 => render_output::<u64>(shared, receiver, output, buffer),
		SampleFormat::F32 => render_output::<f32>(shared, receiver, output, buffer),
		SampleFormat::F64 => render_output::<f64>(shared, receiver, output, buffer),
		_ => output.bytes_mut().fill(0),
	}
}

struct OutputBuffer {
	channels: u32,
	sample_rate: u32,
	start: Option<Instant>,
}

fn render_output<T>(
	shared: &PlayerShared,
	receiver: &mut Consumer<PlaybackFrame>,
	output: &mut Data,
	buffer: OutputBuffer,
) where
	T: SizedSample + FromSample<f32>,
{
	let Some(samples) = output.as_slice_mut::<T>() else {
		return;
	};
	let silence = T::from_sample(0.0);
	let Ok(mut clock) = shared.output.try_lock() else {
		samples.fill(silence);
		return;
	};
	let generation = shared.generation.load(Ordering::Acquire);
	// The producer never writes a newer epoch ahead of an older one.
	if receiver
		.peek()
		.is_ok_and(|frame| frame.generation < generation)
		&& let Ok(stale) = receiver.read_chunk(receiver.slots())
	{
		let (head, tail) = stale.as_slices();
		let count = head
			.iter()
			.chain(tail)
			.take_while(|frame| frame.generation < generation)
			.count();
		stale.commit(count);
		shared.wake.notify_one();
	}
	if !shared.playing.load(Ordering::Acquire) {
		samples.fill(silence);
		return;
	}
	let channels = buffer.channels as usize;
	let requested = samples.len() / channels;
	let gain = if shared.muted.load(Ordering::Relaxed) {
		0.0
	} else {
		f32::from_bits(shared.volume.load(Ordering::Relaxed))
	};
	let end = shared.end_frame.load(Ordering::Acquire);
	let to_play = (end.saturating_sub(clock.end_frame)).min(requested as u64) as usize;
	let mut peak = 0.0_f32;
	let mut written = 0;
	let available = receiver.slots().min(to_play);
	if available != 0
		&& let Ok(chunk) = receiver.read_chunk(available)
	{
		let (head, tail) = chunk.as_slices();
		let pcm = head
			.iter()
			.chain(tail)
			.take_while(|frame| frame.generation == generation);
		for (destination, frame) in samples.chunks_exact_mut(channels).zip(pcm) {
			for (sample, value) in destination.iter_mut().zip(frame.samples) {
				let value = (value * gain).clamp(-1.0, 1.0);
				peak = peak.max(value.abs());
				*sample = T::from_sample(value);
			}
			written += 1;
		}
		chunk.commit(written);
		shared.wake.notify_one();
	}
	samples[written * channels..].fill(silence);
	if written != 0 {
		clock.end_frame = clock.end_frame.saturating_add(written as u64);
		clock.ends_at = buffer.start.and_then(|start| {
			start.checked_add(Duration::from_micros(frames_to_us(
				written as u64,
				buffer.sample_rate,
			)))
		});
		shared
			.position_frame
			.store(clock.end_frame, Ordering::Release);
	}
	if clock.end_frame >= end {
		shared.eos.store(true, Ordering::Release);
		shared.playing.store(false, Ordering::Release);
	}
	shared.peak.store(peak.to_bits(), Ordering::Relaxed);
	shared
		.underrun_frames
		.fetch_add((to_play - written) as u64, Ordering::Relaxed);
}

fn decode_loop(
	mut decoder: StreamingDecoder,
	mut sender: Producer<PlaybackFrame>,
	capacity: usize,
	shared: Arc<PlayerShared>,
) {
	let mut generation = 0;
	let channels = decoder.channels;
	while !shared.stop.load(Ordering::Acquire) {
		if let Some(serial) = apply_seek_request(&mut decoder, &shared) {
			generation = serial;
			continue;
		}
		if shared.eos.load(Ordering::Acquire) {
			wait_decoder(&shared, Duration::from_millis(100));
			continue;
		}
		match decoder.read_frames(|| {
			shared.stop.load(Ordering::Acquire) || shared.generation.load(Ordering::Acquire) != generation
		}) {
			Ok(Some((gap, samples))) => {
				let frames = (0..gap)
					.map(|_| &[][..])
					.chain(samples.chunks_exact(channels));
				for samples in frames {
					let mut frame = PlaybackFrame {
						generation,
						..PlaybackFrame::default()
					};
					frame.samples[..samples.len()].copy_from_slice(samples);
					if !push_frame(&mut sender, &shared, frame) {
						break;
					}
				}
			}
			Ok(None) => {
				while sender.slots() < capacity
					&& !shared.stop.load(Ordering::Acquire)
					&& shared.generation.load(Ordering::Acquire) == generation
					&& !shared.eos.load(Ordering::Acquire)
				{
					wait_decoder(&shared, Duration::from_millis(16));
				}
				// Serialize the drain transition with a new seek. Only the worker
				// changes the PCM epoch, including after an in-flight packet read.
				let Ok(control) = shared.control.lock() else {
					break;
				};
				if control.next_seek_serial != generation {
					continue;
				}
				if shared.loop_playback.load(Ordering::Acquire) && shared.playing.load(Ordering::Acquire) {
					drop(control);
					match decoder.seek(0) {
						Ok(()) => {
							reset_output_clock(&shared, 0);
							continue;
						}
						Err(error) => fail_decode(&shared, generation, error),
					}
				} else {
					shared.eos.store(true, Ordering::Release);
					shared.playing.store(false, Ordering::Release);
				}
			}
			Err(error) => fail_decode(&shared, generation, error),
		}
	}
}

fn push_frame(
	sender: &mut Producer<PlaybackFrame>,
	shared: &PlayerShared,
	frame: PlaybackFrame,
) -> bool {
	while !shared.stop.load(Ordering::Relaxed)
		&& shared.generation.load(Ordering::Acquire) == frame.generation
		&& !shared.eos.load(Ordering::Acquire)
	{
		match sender.push(frame) {
			Ok(()) => return true,
			Err(PushError::Full(_)) => wait_decoder(shared, Duration::from_millis(16)),
		}
	}
	false
}

fn reset_output_clock(shared: &PlayerShared, frame: u64) {
	if let Ok(mut clock) = shared.output.lock() {
		*clock = OutputClock {
			origin_frame: frame,
			end_frame: frame,
			ends_at: None,
		};
		shared.position_frame.store(frame, Ordering::Release);
	}
	shared.peak.store(0, Ordering::Relaxed);
}

fn fail_decode(shared: &PlayerShared, generation: u64, error: Error) {
	if let Ok(mut control) = shared.control.lock()
		&& control.next_seek_serial == generation
	{
		control.decode_error = Some(error);
		shared.eos.store(true, Ordering::Release);
		shared.playing.store(false, Ordering::Release);
	}
}

fn apply_seek_request(decoder: &mut StreamingDecoder, shared: &PlayerShared) -> Option<u64> {
	let (serial, timestamp_us) = shared.control.lock().ok()?.seek_request.take()?;
	let result = decoder.seek(timestamp_us);
	if let Ok(mut control) = shared.control.lock()
		&& control.next_seek_serial == serial
	{
		if result.is_ok() {
			control.decode_error = None;
			reset_output_clock(shared, us_to_frames(timestamp_us, decoder.sample_rate));
			shared.eos.store(false, Ordering::Release);
		} else {
			shared.eos.store(true, Ordering::Release);
		}
		control.seek_error = result.err().map(|error| error.to_string());
		control.applied_seek_serial = serial;
		shared.wake.notify_all();
	}
	Some(serial)
}

fn wait_decoder(shared: &PlayerShared, duration: Duration) {
	if let Ok(control) = shared.control.lock() {
		let _ = shared.wake.wait_timeout(control, duration);
	}
}

fn validate_config(config: &AudioPlayerConfig) -> Result<()> {
	if config.uri.as_os_str().is_empty() || !(40..=5_000).contains(&config.ring_milliseconds) {
		return Err(Error::invalid_argument(
			"AudioPlayer requires a URI and 40–5000 ms of ring storage",
		));
	}
	Ok(())
}

pub(super) fn ring_capacity(sample_rate: u32, milliseconds: u32, owner: &str) -> Result<usize> {
	u64::from(sample_rate)
		.checked_mul(u64::from(milliseconds))
		.and_then(|value| value.checked_div(1_000))
		.and_then(|value| usize::try_from(value.max(1)).ok())
		.ok_or_else(|| Error::out_of_range(format!("{owner} ring capacity exceeds usize")))
}

pub(super) fn select_config(
	ranges: impl Iterator<Item = SupportedStreamConfigRange>,
	sample_rate: u32,
	channels: u32,
	direction: &str,
) -> Result<SupportedStreamConfig> {
	let channels = u16::try_from(channels)
		.map_err(|_| Error::out_of_range("audio device channel count exceeds u16"))?;
	let mut fallback = None;
	for range in ranges {
		if range.channels() != channels
			|| sample_rate < range.min_sample_rate()
			|| sample_rate > range.max_sample_rate()
			|| !is_pcm_format(range.sample_format())
		{
			continue;
		}
		let configured = range.with_sample_rate(sample_rate);
		if configured.sample_format() == SampleFormat::F32 {
			return Ok(configured);
		}
		fallback.get_or_insert(configured);
	}
	fallback.ok_or_else(|| {
		Error::missing_capability(format!(
			"default {direction} device has no {channels}-channel {sample_rate} Hz PCM configuration"
		))
	})
}

fn is_pcm_format(format: SampleFormat) -> bool {
	matches!(
		format,
		SampleFormat::I8
			| SampleFormat::I16
			| SampleFormat::I24
			| SampleFormat::I32
			| SampleFormat::I64
			| SampleFormat::U8
			| SampleFormat::U16
			| SampleFormat::U24
			| SampleFormat::U32
			| SampleFormat::U64
			| SampleFormat::F32
			| SampleFormat::F64
	)
}

pub(super) fn cpal_error<E>(operation: &'static str, error: E) -> Error
where
	E: std::error::Error + Send + Sync + 'static,
{
	Error::backend_failure("CPAL", operation, error)
}

#[cfg(test)]
#[path = "../../../../test/rs/audio/playback_clock_unit.rs"]
mod tests;
