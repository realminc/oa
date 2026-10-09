use pyo3::{prelude::*, types::PyBytes};

use crate::{error::python_error, runtime::PythonEngine};

use super::PythonAudio;

// ── AudioCaptureConfig ────────────────────────────────────────────────────────

/// Configuration for one default-device capture session.
#[pyclass(name = "AudioCaptureConfig")]
#[derive(Clone)]
pub(crate) struct PythonAudioCaptureConfig {
	pub(crate) inner: oa::audio::AudioCaptureConfig,
}

#[pymethods]
impl PythonAudioCaptureConfig {
	/// Construct with OA defaults (48 000 Hz, 2 channels, 500 ms ring).
	#[new]
	#[pyo3(signature = (sample_rate = 48_000, channel_count = 2, ring_milliseconds = 500))]
	pub fn new(sample_rate: u32, channel_count: u32, ring_milliseconds: u32) -> Self {
		Self {
			inner: oa::audio::AudioCaptureConfig {
				sample_rate,
				channel_count,
				ring_milliseconds,
			},
		}
	}

	#[getter]
	pub fn sample_rate(&self) -> u32 {
		self.inner.sample_rate
	}

	#[getter]
	pub fn channel_count(&self) -> u32 {
		self.inner.channel_count
	}

	#[getter]
	pub fn ring_milliseconds(&self) -> u32 {
		self.inner.ring_milliseconds
	}

	pub fn __repr__(&self) -> String {
		format!(
			"AudioCaptureConfig(sample_rate={}, channel_count={}, ring_ms={})",
			self.inner.sample_rate, self.inner.channel_count, self.inner.ring_milliseconds,
		)
	}
}

// ── AudioCaptureChunk ─────────────────────────────────────────────────────────

/// One contiguous run of captured interleaved FP32 frames.
#[pyclass(name = "AudioCaptureChunk")]
#[derive(Clone)]
pub(crate) struct PythonAudioCaptureChunk {
	inner: oa::audio::AudioCaptureChunk,
}

#[pymethods]
impl PythonAudioCaptureChunk {
	/// Interleaved FP32 sample data.
	#[getter]
	pub fn interleaved(&self) -> Vec<f32> {
		self.inner.interleaved.clone()
	}

	#[getter]
	pub fn sample_rate(&self) -> u32 {
		self.inner.sample_rate
	}

	#[getter]
	pub fn channel_count(&self) -> u32 {
		self.inner.channel_count
	}

	#[getter]
	pub fn frame_count(&self) -> u64 {
		self.inner.frame_count
	}

	#[getter]
	pub fn first_frame_index(&self) -> u64 {
		self.inner.first_frame_index
	}

	/// Process-monotonic presentation timestamp in microseconds.
	#[getter]
	pub fn presentation_timestamp_us(&self) -> u64 {
		self.inner.presentation_timestamp_us
	}

	pub fn __repr__(&self) -> String {
		format!(
			"AudioCaptureChunk(frames={}, channels={}, sample_rate={})",
			self.inner.frame_count, self.inner.channel_count, self.inner.sample_rate,
		)
	}
}

// ── AudioCapture ──────────────────────────────────────────────────────────────

/// Stateful default-device audio capture session.
#[pyclass(name = "AudioCapture", unsendable)]
pub(crate) struct PythonAudioCapture {
	inner: oa::audio::AudioCapture,
}

#[pymethods]
impl PythonAudioCapture {
	/// Open the default input device. The session is initially stopped.
	#[new]
	#[pyo3(signature = (engine, config = None))]
	pub fn new(engine: &PythonEngine, config: Option<&PythonAudioCaptureConfig>) -> PyResult<Self> {
		let cfg = config.map(|c| c.inner).unwrap_or_default();
		oa::audio::AudioCapture::open(&engine.inner, cfg)
			.map(|inner| Self { inner })
			.map_err(python_error)
	}

	/// Start capture and reset all counters and frame indices.
	pub fn start(&mut self) -> PyResult<()> {
		self.inner.start().map_err(python_error)
	}

	/// Stop callback delivery without releasing the device.
	pub fn stop(&mut self) -> PyResult<()> {
		self.inner.stop().map_err(python_error)
	}

	/// Return the next contiguous frame run without blocking, or None.
	#[pyo3(signature = (max_frames = 4096))]
	pub fn poll(&mut self, max_frames: u32) -> Option<PythonAudioCaptureChunk> {
		self
			.inner
			.poll(max_frames)
			.map(|inner| PythonAudioCaptureChunk { inner })
	}

	/// Stop capture and release the device.
	pub fn close(&mut self) -> PyResult<()> {
		self.inner.close().map_err(python_error)
	}

	#[getter]
	pub fn is_open(&self) -> bool {
		self.inner.is_open()
	}

	#[getter]
	pub fn is_started(&self) -> bool {
		self.inner.is_started()
	}

	pub fn dropped_frame_count(&self) -> u64 {
		self.inner.dropped_frame_count()
	}

	pub fn device_error_count(&self) -> u64 {
		self.inner.device_error_count()
	}

	pub fn __repr__(&self) -> String {
		format!(
			"AudioCapture(open={}, started={})",
			self.inner.is_open(),
			self.inner.is_started(),
		)
	}
}

// ── AudioPlayerConfig ─────────────────────────────────────────────────────────

/// Configuration for one incremental playback session.
#[pyclass(name = "AudioPlayerConfig")]
#[derive(Clone)]
pub(crate) struct PythonAudioPlayerConfig {
	pub(crate) inner: oa::audio::AudioPlayerConfig,
}

#[pymethods]
impl PythonAudioPlayerConfig {
	/// Construct a config for a given file URI with optional loop and ring options.
	#[new]
	#[pyo3(signature = (uri, loop_playback = false, ring_milliseconds = 500))]
	pub fn new(uri: &str, loop_playback: bool, ring_milliseconds: u32) -> Self {
		let mut inner = oa::audio::AudioPlayerConfig::new(uri);
		inner.loop_playback = loop_playback;
		inner.ring_milliseconds = ring_milliseconds;
		Self { inner }
	}

	#[getter]
	pub fn uri(&self) -> String {
		self.inner.uri.to_string_lossy().into_owned()
	}

	#[getter]
	pub fn loop_playback(&self) -> bool {
		self.inner.loop_playback
	}

	#[getter]
	pub fn ring_milliseconds(&self) -> u32 {
		self.inner.ring_milliseconds
	}

	pub fn __repr__(&self) -> String {
		format!(
			"AudioPlayerConfig(uri={:?}, loop={}, ring_ms={})",
			self.inner.uri, self.inner.loop_playback, self.inner.ring_milliseconds,
		)
	}
}

// ── AudioPlayer ───────────────────────────────────────────────────────────────

/// Incremental WAV, FLAC, or MP3 playback through the default output device.
#[pyclass(name = "AudioPlayer", unsendable)]
pub(crate) struct PythonAudioPlayer {
	inner: oa::audio::AudioPlayer,
}

#[pymethods]
impl PythonAudioPlayer {
	/// Open the source and default output device with decoding initially paused.
	#[new]
	pub fn new(engine: &PythonEngine, config: &PythonAudioPlayerConfig) -> PyResult<Self> {
		oa::audio::AudioPlayer::open(&engine.inner, config.inner.clone())
			.map(|inner| Self { inner })
			.map_err(python_error)
	}

	/// Open a file URI with default playback configuration.
	#[staticmethod]
	pub fn open_uri(engine: &PythonEngine, uri: &str) -> PyResult<Self> {
		oa::audio::AudioPlayer::open_uri(&engine.inner, uri)
			.map(|inner| Self { inner })
			.map_err(python_error)
	}

	/// Start or resume playback, rewinding after end-of-stream.
	pub fn play(&mut self) -> PyResult<()> {
		self.inner.play().map_err(python_error)
	}

	/// Pause playback without disturbing decoder position.
	pub fn pause(&self) {
		self.inner.pause();
	}

	/// Seek to a timestamp in microseconds and wait for decoder acknowledgement.
	pub fn seek(&mut self, timestamp_us: u64) -> PyResult<()> {
		self.inner.seek(timestamp_us).map_err(python_error)
	}

	pub fn set_loop(&self, enabled: bool) {
		self.inner.set_loop(enabled);
	}

	pub fn set_muted(&self, muted: bool) -> PyResult<()> {
		self.inner.set_muted(muted).map_err(python_error)
	}

	/// Stop decoding and release the device.
	pub fn close(&mut self) -> PyResult<()> {
		self.inner.close().map_err(python_error)
	}

	#[getter]
	pub fn is_open(&self) -> bool {
		self.inner.is_open()
	}

	#[getter]
	pub fn is_playing(&self) -> bool {
		self.inner.is_playing()
	}

	#[getter]
	pub fn is_eos(&self) -> bool {
		self.inner.is_eos()
	}

	#[getter]
	pub fn is_muted(&self) -> bool {
		self.inner.is_muted()
	}

	#[getter]
	pub fn sample_rate(&self) -> u32 {
		self.inner.sample_rate()
	}

	#[getter]
	pub fn channel_count(&self) -> u32 {
		self.inner.channel_count()
	}

	/// Total duration in microseconds (0 if unknown).
	#[getter]
	pub fn duration_us(&self) -> u64 {
		self.inner.duration_us()
	}

	/// Current playback position in microseconds.
	#[getter]
	pub fn position_us(&self) -> u64 {
		self.inner.position_us()
	}

	pub fn underrun_frame_count(&self) -> u64 {
		self.inner.underrun_frame_count()
	}

	pub fn device_error_count(&self) -> u64 {
		self.inner.device_error_count()
	}

	pub fn __repr__(&self) -> String {
		format!(
			"AudioPlayer(open={}, playing={}, position_us={})",
			self.inner.is_open(),
			self.inner.is_playing(),
			self.inner.position_us(),
		)
	}
}

// ── AudioEncodeProfile ────────────────────────────────────────────────────────

/// Configuration for one streaming audio encoder.
#[pyclass(name = "AudioEncodeProfile")]
#[derive(Clone)]
pub(crate) struct PythonAudioEncodeProfile {
	pub(crate) inner: oa::audio::AudioEncodeProfile,
}

#[pymethods]
impl PythonAudioEncodeProfile {
	/// Construct with OA defaults (PCM-S16, 48 000 Hz, 2 ch, 1024 frames/pkt).
	#[new]
	#[pyo3(signature = (sample_rate = 48_000, channel_count = 2, frames_per_packet = 1_024))]
	pub fn new(sample_rate: u32, channel_count: u32, frames_per_packet: u32) -> Self {
		Self {
			inner: oa::audio::AudioEncodeProfile {
				codec: oa::audio::AudioCodec::PcmS16,
				sample_rate,
				channel_count,
				frames_per_packet,
			},
		}
	}

	#[getter]
	pub fn sample_rate(&self) -> u32 {
		self.inner.sample_rate
	}

	#[getter]
	pub fn channel_count(&self) -> u32 {
		self.inner.channel_count
	}

	#[getter]
	pub fn frames_per_packet(&self) -> u32 {
		self.inner.frames_per_packet
	}

	pub fn __repr__(&self) -> String {
		format!(
			"AudioEncodeProfile(sample_rate={}, channel_count={}, frames_per_packet={})",
			self.inner.sample_rate, self.inner.channel_count, self.inner.frames_per_packet,
		)
	}
}

// ── EncodedAudioPacket ────────────────────────────────────────────────────────

/// One encoded elementary-stream packet with frame-domain timing.
#[pyclass(name = "EncodedAudioPacket")]
#[derive(Clone)]
pub(crate) struct PythonEncodedAudioPacket {
	inner: oa::audio::EncodedAudioPacket,
}

#[pymethods]
impl PythonEncodedAudioPacket {
	/// Encoded packet bytes.
	pub fn bitstream<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
		PyBytes::new(py, &self.inner.bitstream)
	}

	/// Zero-based input frame corresponding to the packet's first sample.
	#[getter]
	pub fn presentation_frame(&self) -> i64 {
		self.inner.presentation_frame
	}

	/// Number of interleaved frames represented by the packet.
	#[getter]
	pub fn duration_frames(&self) -> u32 {
		self.inner.duration_frames
	}

	pub fn __repr__(&self) -> String {
		format!(
			"EncodedAudioPacket(presentation_frame={}, duration_frames={}, bytes={})",
			self.inner.presentation_frame,
			self.inner.duration_frames,
			self.inner.bitstream.len(),
		)
	}
}

// ── AudioEncoder ──────────────────────────────────────────────────────────────

/// Stateful deterministic PCM-S16 audio encoder.
#[pyclass(name = "AudioEncoder", unsendable)]
pub(crate) struct PythonAudioEncoder {
	inner: oa::audio::AudioEncoder,
}

#[pymethods]
impl PythonAudioEncoder {
	/// Create one open encoder with a checked profile.
	#[new]
	#[pyo3(signature = (profile = None))]
	pub fn new(profile: Option<&PythonAudioEncodeProfile>) -> PyResult<Self> {
		let prof = profile.map(|p| p.inner).unwrap_or_default();
		oa::audio::AudioEncoder::create(prof)
			.map(|inner| Self { inner })
			.map_err(python_error)
	}

	/// Append interleaved FP32 frames and return all full packets now ready.
	pub fn encode(&mut self, interleaved: Vec<f32>) -> PyResult<Vec<PythonEncodedAudioPacket>> {
		self
			.inner
			.encode(&interleaved)
			.map(|packets| {
				packets
					.into_iter()
					.map(|inner| PythonEncodedAudioPacket { inner })
					.collect()
			})
			.map_err(python_error)
	}

	/// Emit the final partial packet, if any.
	pub fn flush(&mut self) -> PyResult<Vec<PythonEncodedAudioPacket>> {
		self
			.inner
			.flush()
			.map(|packets| {
				packets
					.into_iter()
					.map(|inner| PythonEncodedAudioPacket { inner })
					.collect()
			})
			.map_err(python_error)
	}

	/// Discard any unflushed partial packet and close the encoder.
	pub fn close(&mut self) {
		self.inner.close();
	}

	#[getter]
	pub fn is_open(&self) -> bool {
		self.inner.is_open()
	}

	pub fn __repr__(&self) -> String {
		format!("AudioEncoder(open={})", self.inner.is_open())
	}
}

// ── AudioPlayer factory from Audio ────────────────────────────────────────────

/// Construct an AudioPlayer from a decoded Audio value for in-memory playback.
#[pyfunction]
pub(crate) fn audio_player_from_audio(
	engine: &PythonEngine,
	audio: &PythonAudio,
) -> PyResult<PythonAudioPlayer> {
	// Re-encode in-memory as WAV and open via the streaming decoder.
	let wav = oa::audio::encode_wav_f32(&audio.inner).map_err(python_error)?;
	// Write to a temporary file and open with the player.
	let tmp = std::env::temp_dir().join(format!(
		"oa_audio_{}.wav",
		std::time::SystemTime::now()
			.duration_since(std::time::UNIX_EPOCH)
			.map(|d| d.as_nanos())
			.unwrap_or(0)
	));
	std::fs::write(&tmp, &wav).map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
	let result = oa::audio::AudioPlayer::open_uri(&engine.inner, &tmp)
		.map(|inner| PythonAudioPlayer { inner })
		.map_err(python_error);
	let _ = std::fs::remove_file(&tmp);
	result
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
	module.add_class::<PythonAudioCaptureConfig>()?;
	module.add_class::<PythonAudioCaptureChunk>()?;
	module.add_class::<PythonAudioCapture>()?;
	module.add_class::<PythonAudioPlayerConfig>()?;
	module.add_class::<PythonAudioPlayer>()?;
	module.add_class::<PythonAudioEncodeProfile>()?;
	module.add_class::<PythonEncodedAudioPacket>()?;
	module.add_class::<PythonAudioEncoder>()?;
	module.add_function(wrap_pyfunction!(audio_player_from_audio, module)?)?;
	Ok(())
}
