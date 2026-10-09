//! Incremental decode and real-time playback session.

use std::{
	fs::File,
	path::{Path, PathBuf},
	sync::{
		Arc, Condvar, Mutex,
		atomic::{AtomicBool, AtomicU64, Ordering},
	},
	thread::{self, JoinHandle},
	time::Duration,
};

use cpal::{
	Data, FromSample, SampleFormat, SizedSample, Stream, SupportedStreamConfig,
	SupportedStreamConfigRange,
	traits::{DeviceTrait, HostTrait, StreamTrait},
};
use rtrb::{Consumer, Producer, PushError, RingBuffer};
use symphonia::core::{
	codecs::audio::{AudioDecoder as SymphoniaDecoder, AudioDecoderOptions},
	formats::{FormatOptions, FormatReader, SeekMode, SeekTo, TrackType, probe::Hint},
	io::MediaSourceStream,
	meta::MetadataOptions,
	units::Time,
};

use crate::{Engine, Error, Result, runtime::EngineHandle};

/// Configuration for one incremental playback session.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AudioPlayerConfig {
	pub uri: PathBuf,
	pub loop_playback: bool,
	pub ring_milliseconds: u32,
}

impl AudioPlayerConfig {
	pub fn new(uri: impl Into<PathBuf>) -> Self {
		Self {
			uri: uri.into(),
			loop_playback: false,
			ring_milliseconds: 500,
		}
	}
}

/// Incremental WAV, FLAC, or MP3 playback through the default output device.
pub struct AudioPlayer {
	_engine: EngineHandle,
	service: Option<PlayerService>,
	shared: Arc<PlayerShared>,
	sample_rate: u32,
	channels: u32,
	duration_us: u64,
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
	generation: AtomicU64,
	position_frame: AtomicU64,
	underrun_frames: AtomicU64,
	device_errors: AtomicU64,
	wake: Condvar,
	control: Mutex<PlayerControl>,
}

#[derive(Default)]
struct PlayerControl {
	next_seek_serial: u64,
	applied_seek_serial: u64,
	seek_request: Option<(u64, u64)>,
	seek_error: Option<String>,
}

#[derive(Clone, Copy, Default)]
struct PlaybackFrame {
	samples: [f32; 8],
	generation: u64,
}

struct StreamingDecoder {
	format: Box<dyn FormatReader>,
	decoder: Box<dyn SymphoniaDecoder>,
	track_id: u32,
	channels: usize,
	sample_rate: u32,
	discard_frames: usize,
}

impl AudioPlayer {
	/// Open the source and default output device with decoding initially paused.
	pub fn open(engine: &Engine, config: AudioPlayerConfig) -> Result<Self> {
		if config.uri.as_os_str().is_empty() || config.ring_milliseconds < 40 {
			return Err(Error::invalid_argument(
				"AudioPlayer requires a URI and at least 40 ms of ring storage",
			));
		}
		let (decoder, duration_us) = StreamingDecoder::open(&config.uri)?;
		let sample_rate = decoder.sample_rate;
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
			generation: AtomicU64::new(0),
			position_frame: AtomicU64::new(0),
			underrun_frames: AtomicU64::new(0),
			device_errors: AtomicU64::new(0),
			wake: Condvar::new(),
			control: Mutex::new(PlayerControl::default()),
		});

		let host = cpal::default_host();
		let device = host
			.default_output_device()
			.ok_or_else(|| Error::no_suitable_device("AudioPlayer found no default output device"))?;
		let ranges = device
			.supported_output_configs()
			.map_err(|error| cpal_error("output configuration query", error))?;
		let stream_config = select_config(ranges, sample_rate, channels, "output")?;
		let sample_format = stream_config.sample_format();
		let callback_shared = Arc::clone(&shared);
		let error_shared = Arc::clone(&shared);
		let stream = device
			.build_output_stream_raw(
				stream_config.config(),
				sample_format,
				move |output, _| {
					playback_callback(&callback_shared, &mut receiver, output, channels);
				},
				move |_| {
					error_shared.device_errors.fetch_add(1, Ordering::Relaxed);
				},
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
		})
	}

	/// Open a URI using the default playback configuration.
	pub fn open_uri(engine: &Engine, uri: impl AsRef<Path>) -> Result<Self> {
		Self::open(engine, AudioPlayerConfig::new(uri.as_ref()))
	}

	/// Start or resume playback, rewinding after end-of-stream.
	pub fn play(&mut self) -> Result<()> {
		self.require_open("play")?;
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
			control.next_seek_serial = control.next_seek_serial.wrapping_add(1);
			let serial = control.next_seek_serial;
			control.seek_request = Some((serial, timestamp_us));
			control.seek_error = None;
			serial
		};
		// Invalidate queued PCM immediately so a decoder blocked on a full or
		// draining ring can observe the seek before it applies the new position.
		self.shared.generation.fetch_add(1, Ordering::AcqRel);
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
				"Symphonia",
				"AudioPlayer seek",
				std::io::Error::new(
					std::io::ErrorKind::TimedOut,
					"seek acknowledgement timed out",
				),
			));
		}
		if let Some(message) = &control.seek_error {
			return Err(Error::backend_failure(
				"Symphonia",
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
		self
			.shared
			.position_frame
			.load(Ordering::Relaxed)
			.saturating_mul(1_000_000)
			/ u64::from(self.sample_rate)
	}

	pub fn underrun_frame_count(&self) -> u64 {
		self.shared.underrun_frames.load(Ordering::Relaxed)
	}

	pub fn device_error_count(&self) -> u64 {
		self.shared.device_errors.load(Ordering::Relaxed)
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
		let pause_result = self
			.stream
			.pause()
			.map_err(|error| cpal_error("playback stream stop", error));
		if let Some(handle) = self.decoder_thread.take() {
			handle
				.join()
				.map_err(|_| Error::internal("AudioPlayer decoder thread panicked"))?;
		}
		pause_result
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
	channels: u32,
) {
	match output.sample_format() {
		SampleFormat::I8 => render_output::<i8>(shared, receiver, output, channels),
		SampleFormat::I16 => render_output::<i16>(shared, receiver, output, channels),
		SampleFormat::I24 => render_output::<cpal::I24>(shared, receiver, output, channels),
		SampleFormat::I32 => render_output::<i32>(shared, receiver, output, channels),
		SampleFormat::I64 => render_output::<i64>(shared, receiver, output, channels),
		SampleFormat::U8 => render_output::<u8>(shared, receiver, output, channels),
		SampleFormat::U16 => render_output::<u16>(shared, receiver, output, channels),
		SampleFormat::U24 => render_output::<cpal::U24>(shared, receiver, output, channels),
		SampleFormat::U32 => render_output::<u32>(shared, receiver, output, channels),
		SampleFormat::U64 => render_output::<u64>(shared, receiver, output, channels),
		SampleFormat::F32 => render_output::<f32>(shared, receiver, output, channels),
		SampleFormat::F64 => render_output::<f64>(shared, receiver, output, channels),
		_ => output.bytes_mut().fill(0),
	}
}

fn render_output<T>(
	shared: &PlayerShared,
	receiver: &mut Consumer<PlaybackFrame>,
	output: &mut Data,
	channels: u32,
) where
	T: SizedSample + FromSample<f32>,
{
	let Some(samples) = output.as_slice_mut::<T>() else {
		return;
	};
	samples.fill(T::from_sample(0.0));
	let channels = channels as usize;
	let generation = shared.generation.load(Ordering::Acquire);
	let playing = shared.playing.load(Ordering::Acquire);
	while receiver
		.peek()
		.is_ok_and(|frame| frame.generation != generation)
	{
		let _ = receiver.pop();
	}
	if !playing {
		shared.wake.notify_one();
		return;
	}
	let requested = samples.len() / channels;
	let muted = shared.muted.load(Ordering::Acquire);
	let mut written = 0_usize;
	while written < requested {
		let Ok(frame) = receiver.pop() else {
			break;
		};
		if frame.generation != generation {
			continue;
		}
		if !muted {
			let offset = written * channels;
			for channel in 0..channels {
				samples[offset + channel] = T::from_sample(frame.samples[channel]);
			}
		}
		written += 1;
	}
	shared
		.position_frame
		.fetch_add(written as u64, Ordering::Relaxed);
	shared
		.underrun_frames
		.fetch_add((requested - written) as u64, Ordering::Relaxed);
	shared.wake.notify_one();
}

fn decode_loop(
	mut decoder: StreamingDecoder,
	mut sender: Producer<PlaybackFrame>,
	capacity: usize,
	shared: Arc<PlayerShared>,
) {
	while !shared.stop.load(Ordering::Acquire) {
		if apply_seek_request(&mut decoder, &shared) {
			continue;
		}
		if !shared.playing.load(Ordering::Acquire) {
			wait_decoder(&shared, Duration::from_millis(10));
			continue;
		}
		match decoder.read_frames() {
			Ok(Some(samples)) => {
				let generation = shared.generation.load(Ordering::Acquire);
				for samples in samples.chunks_exact(decoder.channels) {
					let mut frame = PlaybackFrame {
						generation,
						..PlaybackFrame::default()
					};
					frame.samples[..decoder.channels].copy_from_slice(samples);
					loop {
						if shared.stop.load(Ordering::Acquire)
							|| shared.generation.load(Ordering::Acquire) != generation
						{
							break;
						}
						match sender.push(frame) {
							Ok(()) => break,
							Err(PushError::Full(returned)) => {
								frame = returned;
								wait_decoder(&shared, Duration::from_millis(2));
							}
						}
					}
				}
			}
			Ok(None) => {
				if shared.loop_playback.load(Ordering::Acquire) && decoder.seek(0).is_ok() {
					shared.generation.fetch_add(1, Ordering::AcqRel);
					shared.position_frame.store(0, Ordering::Relaxed);
					continue;
				}
				let generation = shared.generation.load(Ordering::Acquire);
				while sender.slots() < capacity
					&& !shared.stop.load(Ordering::Acquire)
					&& shared.generation.load(Ordering::Acquire) == generation
				{
					wait_decoder(&shared, Duration::from_millis(2));
				}
				if shared.generation.load(Ordering::Acquire) != generation {
					continue;
				}
				shared.eos.store(true, Ordering::Release);
				shared.playing.store(false, Ordering::Release);
			}
			Err(_) => {
				shared.eos.store(true, Ordering::Release);
				shared.playing.store(false, Ordering::Release);
			}
		}
	}
}

fn apply_seek_request(decoder: &mut StreamingDecoder, shared: &PlayerShared) -> bool {
	let request = match shared.control.lock() {
		Ok(mut control) => control.seek_request.take(),
		Err(_) => {
			shared.stop.store(true, Ordering::Release);
			return false;
		}
	};
	let Some((serial, timestamp_us)) = request else {
		return false;
	};
	let result = decoder.seek(timestamp_us);
	if result.is_ok() {
		shared.position_frame.store(
			timestamp_us.saturating_mul(u64::from(decoder.sample_rate)) / 1_000_000,
			Ordering::Relaxed,
		);
		shared.eos.store(false, Ordering::Release);
	}
	if let Ok(mut control) = shared.control.lock() {
		control.seek_error = result.err().map(|error| error.to_string());
		control.applied_seek_serial = serial;
		shared.wake.notify_all();
	}
	true
}

fn wait_decoder(shared: &PlayerShared, duration: Duration) {
	if let Ok(control) = shared.control.lock() {
		let _ = shared.wake.wait_timeout(control, duration);
	}
}

impl StreamingDecoder {
	fn open(path: &Path) -> Result<(Self, u64)> {
		let file = File::open(path).map_err(|source| Error::io("AudioPlayer source open", source))?;
		let stream = MediaSourceStream::new(Box::new(file), Default::default());
		let mut hint = Hint::new();
		if let Some(extension) = path.extension().and_then(|value| value.to_str()) {
			hint.with_extension(extension);
		}
		let format = symphonia::default::get_probe()
			.probe(
				&hint,
				stream,
				FormatOptions::default(),
				MetadataOptions::default(),
			)
			.map_err(|source| Error::backend_failure("Symphonia", "AudioPlayer probe", source))?;
		let track = format
			.default_track(TrackType::Audio)
			.ok_or_else(|| Error::invalid_argument("AudioPlayer source has no decodable audio track"))?;
		let parameters = track
			.codec_params
			.as_ref()
			.and_then(|parameters| parameters.audio())
			.ok_or_else(|| Error::invalid_argument("AudioPlayer track has no audio parameters"))?;
		let channels = parameters
			.channels
			.as_ref()
			.ok_or_else(|| Error::invalid_argument("AudioPlayer track has no channel declaration"))?
			.count();
		let sample_rate = parameters
			.sample_rate
			.ok_or_else(|| Error::invalid_argument("AudioPlayer track has no sample rate"))?;
		let duration_us = track
			.num_frames
			.and_then(|frames| frames.checked_mul(1_000_000))
			.map_or(0, |value| value / u64::from(sample_rate));
		let track_id = track.id;
		let decoder = symphonia::default::get_codecs()
			.make_audio_decoder(parameters, &AudioDecoderOptions::default())
			.map_err(|source| Error::backend_failure("Symphonia", "AudioPlayer decoder", source))?;
		Ok((
			Self {
				format,
				decoder,
				track_id,
				channels,
				sample_rate,
				discard_frames: 0,
			},
			duration_us,
		))
	}

	fn read_frames(&mut self) -> Result<Option<Vec<f32>>> {
		loop {
			let Some(packet) = self
				.format
				.next_packet()
				.map_err(|source| Error::backend_failure("Symphonia", "AudioPlayer packet read", source))?
			else {
				return Ok(None);
			};
			if packet.track_id != self.track_id {
				continue;
			}
			let decoded = self
				.decoder
				.decode(&packet)
				.map_err(|source| Error::backend_failure("Symphonia", "AudioPlayer decode", source))?;
			if decoded.spec().channels().count() != self.channels
				|| decoded.spec().rate() != self.sample_rate
			{
				return Err(Error::failed_precondition(
					"AudioPlayer stream parameters changed during playback",
				));
			}
			let mut samples = vec![0.0; decoded.samples_interleaved()];
			decoded.copy_to_slice_interleaved(&mut samples);
			if self.discard_frames > 0 {
				let frames = samples.len() / self.channels;
				let discard = frames.min(self.discard_frames);
				self.discard_frames -= discard;
				samples.drain(..discard * self.channels);
			}
			if !samples.is_empty() {
				return Ok(Some(samples));
			}
		}
	}

	fn seek(&mut self, timestamp_us: u64) -> Result<()> {
		let seeked = self
			.format
			.seek(
				SeekMode::Accurate,
				SeekTo::Time {
					time: Time::from_micros_u64(timestamp_us),
					track_id: Some(self.track_id),
				},
			)
			.map_err(|source| Error::backend_failure("Symphonia", "AudioPlayer seek", source))?;
		self.decoder.reset();
		let actual_frame = self
			.format
			.tracks()
			.iter()
			.find(|track| track.id == seeked.track_id)
			.and_then(|track| track.time_base)
			.and_then(|base| base.calc_time(seeked.actual_ts))
			.map(|time| {
				let (seconds, nanos) = time.parts();
				(seconds.max(0) as u64)
					.saturating_mul(u64::from(self.sample_rate))
					.saturating_add(u64::from(nanos) * u64::from(self.sample_rate) / 1_000_000_000)
			})
			.unwrap_or(0);
		let requested_frame = timestamp_us.saturating_mul(u64::from(self.sample_rate)) / 1_000_000;
		self.discard_frames =
			usize::try_from(requested_frame.saturating_sub(actual_frame)).unwrap_or(usize::MAX);
		Ok(())
	}
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
