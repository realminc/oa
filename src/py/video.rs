use pyo3::{prelude::*, types::PyBytes};

use crate::{error::python_error, image::PythonImage, runtime::PythonEngine};

// ── VideoColorMatrix ──────────────────────────────────────────────────────────

/// Matrix coefficients associated with a frame's source color conversion.
#[pyclass(name = "VideoColorMatrix", eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum PythonVideoColorMatrix {
	Unspecified = 0,
	Bt601 = 1,
	Bt709 = 2,
	Bt2020 = 3,
}

impl From<oa::video::VideoColorMatrix> for PythonVideoColorMatrix {
	fn from(v: oa::video::VideoColorMatrix) -> Self {
		match v {
			oa::video::VideoColorMatrix::Unspecified => Self::Unspecified,
			oa::video::VideoColorMatrix::Bt601 => Self::Bt601,
			oa::video::VideoColorMatrix::Bt709 => Self::Bt709,
			oa::video::VideoColorMatrix::Bt2020 => Self::Bt2020,
		}
	}
}

impl From<PythonVideoColorMatrix> for oa::video::VideoColorMatrix {
	fn from(v: PythonVideoColorMatrix) -> Self {
		match v {
			PythonVideoColorMatrix::Unspecified => Self::Unspecified,
			PythonVideoColorMatrix::Bt601 => Self::Bt601,
			PythonVideoColorMatrix::Bt709 => Self::Bt709,
			PythonVideoColorMatrix::Bt2020 => Self::Bt2020,
		}
	}
}

// ── VideoColorRange ───────────────────────────────────────────────────────────

/// Encoded or converted component range associated with a video frame.
#[pyclass(name = "VideoColorRange", eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum PythonVideoColorRange {
	Unspecified = 0,
	Limited = 1,
	Full = 2,
}

impl From<oa::video::VideoColorRange> for PythonVideoColorRange {
	fn from(v: oa::video::VideoColorRange) -> Self {
		match v {
			oa::video::VideoColorRange::Unspecified => Self::Unspecified,
			oa::video::VideoColorRange::Limited => Self::Limited,
			oa::video::VideoColorRange::Full => Self::Full,
		}
	}
}

impl From<PythonVideoColorRange> for oa::video::VideoColorRange {
	fn from(v: PythonVideoColorRange) -> Self {
		match v {
			PythonVideoColorRange::Unspecified => Self::Unspecified,
			PythonVideoColorRange::Limited => Self::Limited,
			PythonVideoColorRange::Full => Self::Full,
		}
	}
}

// ── VideoColorInfo ────────────────────────────────────────────────────────────

/// Source color metadata retained across frame transformations.
#[pyclass(name = "VideoColorInfo")]
#[derive(Clone)]
pub(crate) struct PythonVideoColorInfo {
	inner: oa::video::VideoColorInfo,
}

#[pymethods]
impl PythonVideoColorInfo {
	/// Construct explicit source color metadata.
	#[new]
	#[pyo3(signature = (matrix=PythonVideoColorMatrix::Unspecified, range=PythonVideoColorRange::Unspecified))]
	pub fn new(matrix: PythonVideoColorMatrix, range: PythonVideoColorRange) -> Self {
		Self {
			inner: oa::video::VideoColorInfo::new(matrix.into(), range.into()),
		}
	}

	/// Construct metadata for a producer that did not specify color properties.
	#[staticmethod]
	pub fn unspecified() -> Self {
		Self {
			inner: oa::video::VideoColorInfo::unspecified(),
		}
	}

	#[getter]
	pub fn matrix(&self) -> PythonVideoColorMatrix {
		self.inner.matrix().into()
	}

	#[getter]
	pub fn range(&self) -> PythonVideoColorRange {
		self.inner.range().into()
	}

	pub fn __repr__(&self) -> String {
		format!(
			"VideoColorInfo(matrix={:?}, range={:?})",
			self.inner.matrix(),
			self.inner.range()
		)
	}
}

// ── VideoFrameTiming ──────────────────────────────────────────────────────────

/// Presentation timing for one decoded, captured, or generated frame.
#[pyclass(name = "VideoFrameTiming")]
#[derive(Clone)]
pub(crate) struct PythonVideoFrameTiming {
	inner: oa::video::VideoFrameTiming,
}

#[pymethods]
impl PythonVideoFrameTiming {
	/// Construct timing from microsecond values.
	///
	/// A zero `duration_us` means duration is unknown.
	#[new]
	#[pyo3(signature = (presentation_us, duration_us=0))]
	pub fn new(presentation_us: u64, duration_us: u64) -> Self {
		Self {
			inner: oa::video::VideoFrameTiming::from_microseconds(presentation_us, duration_us),
		}
	}

	/// Return the presentation timestamp in microseconds.
	#[getter]
	pub fn presentation_us(&self) -> u64 {
		self.inner.presentation_timestamp().as_micros() as u64
	}

	/// Return the known frame duration in microseconds, if set.
	#[getter]
	pub fn duration_us(&self) -> Option<u64> {
		self.inner.duration().map(|d| d.as_micros() as u64)
	}

	pub fn __repr__(&self) -> String {
		format!(
			"VideoFrameTiming(presentation_us={}, duration_us={:?})",
			self.inner.presentation_timestamp().as_micros(),
			self.inner.duration().map(|d| d.as_micros()),
		)
	}
}

// ── VideoCodec ────────────────────────────────────────────────────────────────

/// Compressed video codec carried by a stream.
#[pyclass(name = "VideoCodec", eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum PythonVideoCodec {
	H264 = 0,
	H265 = 1,
	Av1 = 2,
	Vp9 = 3,
}

impl From<oa::video::VideoCodec> for PythonVideoCodec {
	fn from(v: oa::video::VideoCodec) -> Self {
		match v {
			oa::video::VideoCodec::H264 => Self::H264,
			oa::video::VideoCodec::H265 => Self::H265,
			oa::video::VideoCodec::Av1 => Self::Av1,
			oa::video::VideoCodec::Vp9 => Self::Vp9,
		}
	}
}

impl From<PythonVideoCodec> for oa::video::VideoCodec {
	fn from(v: PythonVideoCodec) -> Self {
		match v {
			PythonVideoCodec::H264 => Self::H264,
			PythonVideoCodec::H265 => Self::H265,
			PythonVideoCodec::Av1 => Self::Av1,
			PythonVideoCodec::Vp9 => Self::Vp9,
		}
	}
}

// ── VideoContainerKind ────────────────────────────────────────────────────────

/// Container kind recognized by the current demux boundary.
#[pyclass(name = "VideoContainerKind", eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum PythonVideoContainerKind {
	Mp4 = 0,
}

impl From<oa::video::VideoContainerKind> for PythonVideoContainerKind {
	fn from(v: oa::video::VideoContainerKind) -> Self {
		match v {
			oa::video::VideoContainerKind::Mp4 => Self::Mp4,
		}
	}
}

// ── VideoContainerInfo ────────────────────────────────────────────────────────

/// Immutable metadata for the selected video track.
#[pyclass(name = "VideoContainerInfo")]
#[derive(Clone)]
pub(crate) struct PythonVideoContainerInfo {
	inner: oa::video::VideoContainerInfo,
}

impl PythonVideoContainerInfo {
	pub(crate) fn wrap(inner: oa::video::VideoContainerInfo) -> Self {
		Self { inner }
	}
}

#[pymethods]
impl PythonVideoContainerInfo {
	#[getter]
	pub fn kind(&self) -> PythonVideoContainerKind {
		self.inner.kind().into()
	}

	#[getter]
	pub fn codec(&self) -> PythonVideoCodec {
		self.inner.codec().into()
	}

	#[getter]
	pub fn width(&self) -> u32 {
		self.inner.width()
	}

	#[getter]
	pub fn height(&self) -> u32 {
		self.inner.height()
	}

	#[getter]
	pub fn duration(&self) -> u64 {
		self.inner.duration()
	}

	#[getter]
	pub fn sample_count(&self) -> u32 {
		self.inner.sample_count()
	}

	#[getter]
	pub fn track_id(&self) -> u32 {
		self.inner.track_id()
	}

	#[getter]
	pub fn track_count(&self) -> u32 {
		self.inner.track_count()
	}

	#[getter]
	pub fn time_base_numerator(&self) -> u32 {
		self.inner.time_base().numerator()
	}

	#[getter]
	pub fn time_base_denominator(&self) -> u32 {
		self.inner.time_base().denominator()
	}

	pub fn frame_rate(&self) -> f64 {
		self.inner.frame_rate()
	}

	pub fn __repr__(&self) -> String {
		format!(
			"VideoContainerInfo(codec={:?}, {}x{}, samples={}, fps={:.3})",
			self.inner.codec(),
			self.inner.width(),
			self.inner.height(),
			self.inner.sample_count(),
			self.inner.frame_rate(),
		)
	}
}

// ── VideoPacket ───────────────────────────────────────────────────────────────

/// One owned compressed video access unit.
#[pyclass(name = "VideoPacket", unsendable)]
pub(crate) struct PythonVideoPacket {
	inner: oa::video::VideoPacket,
}

impl PythonVideoPacket {
	pub(crate) fn wrap(inner: oa::video::VideoPacket) -> Self {
		Self { inner }
	}
}

#[pymethods]
impl PythonVideoPacket {
	/// Return the access-unit bytes.
	pub fn data<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
		PyBytes::new(py, self.inner.data())
	}

	#[getter]
	pub fn presentation_timestamp(&self) -> u64 {
		self.inner.presentation_timestamp()
	}

	#[getter]
	pub fn decode_timestamp(&self) -> u64 {
		self.inner.decode_timestamp()
	}

	#[getter]
	pub fn duration(&self) -> u32 {
		self.inner.duration()
	}

	#[getter]
	pub fn is_keyframe(&self) -> bool {
		self.inner.is_keyframe()
	}

	#[getter]
	pub fn track_id(&self) -> u32 {
		self.inner.track_id()
	}

	pub fn __repr__(&self) -> String {
		format!(
			"VideoPacket(pts={}, dts={}, keyframe={}, len={})",
			self.inner.presentation_timestamp(),
			self.inner.decode_timestamp(),
			self.inner.is_keyframe(),
			self.inner.data().len(),
		)
	}
}

// ── VideoDemuxer ──────────────────────────────────────────────────────────────

/// Stateful, seekable packet source for one unfragmented MP4 video track.
#[pyclass(name = "VideoDemuxer", unsendable)]
pub(crate) struct PythonVideoDemuxer {
	inner: oa::VideoDemuxer,
}

#[pymethods]
impl PythonVideoDemuxer {
	/// Open an MP4 file and validate its video track metadata.
	#[new]
	pub fn new(path: &str) -> PyResult<Self> {
		oa::VideoDemuxer::open(path)
			.map(|inner| Self { inner })
			.map_err(python_error)
	}

	/// Return the container metadata for the selected video track.
	pub fn info(&self) -> PythonVideoContainerInfo {
		PythonVideoContainerInfo::wrap(self.inner.info())
	}

	/// Read the next access unit from the current position.
	///
	/// Returns `None` at end of stream.
	pub fn read_next_packet(&mut self) -> PyResult<Option<PythonVideoPacket>> {
		self
			.inner
			.read_next_packet()
			.map(|opt| opt.map(PythonVideoPacket::wrap))
			.map_err(python_error)
	}

	/// Seek to the sample nearest to `presentation_timestamp` (time-base ticks).
	pub fn seek(&mut self, presentation_timestamp: u64) -> PyResult<()> {
		self
			.inner
			.seek(presentation_timestamp)
			.map_err(python_error)
	}

	/// Close the demuxer and release the file handle.
	pub fn close(&mut self) {
		self.inner.close();
	}

	pub fn __repr__(&self) -> String {
		let info = self.inner.info();
		format!(
			"VideoDemuxer({:?}, {}x{}, samples={})",
			info.codec(),
			info.width(),
			info.height(),
			info.sample_count(),
		)
	}
}

// ── VideoDecoder ──────────────────────────────────────────────────────────────

/// Stateful hardware decoder for one demuxed video stream.
#[pyclass(name = "VideoDecoder", unsendable)]
pub(crate) struct PythonVideoDecoder {
	inner: oa::VideoDecoder,
}

#[pymethods]
impl PythonVideoDecoder {
	/// Create a hardware decoder for the given container info.
	#[new]
	pub fn new(engine: &PythonEngine, info: &PythonVideoContainerInfo) -> PyResult<Self> {
		oa::VideoDecoder::create(&engine.inner, info.inner)
			.map(|inner| Self { inner })
			.map_err(python_error)
	}

	/// Decode one access unit and return the resulting frame, if any.
	pub fn decode(&mut self, packet: &PythonVideoPacket) -> PyResult<Option<PythonVideoFrame>> {
		self
			.inner
			.decode(&packet.inner)
			.map(|opt| opt.map(PythonVideoFrame::wrap))
			.map_err(python_error)
	}

	/// Flush any buffered frames after the last packet has been submitted.
	pub fn flush(&mut self) -> PyResult<Vec<PythonVideoFrame>> {
		self
			.inner
			.flush()
			.map(|frames| frames.into_iter().map(PythonVideoFrame::wrap).collect())
			.map_err(python_error)
	}

	/// Return the current container info, if already decoded.
	pub fn info(&self) -> Option<PythonVideoContainerInfo> {
		self.inner.info().map(PythonVideoContainerInfo::wrap)
	}

	/// Close the decoder and release GPU resources.
	pub fn close(&mut self) {
		self.inner.close();
	}

	pub fn __repr__(&self) -> &'static str {
		"VideoDecoder"
	}
}

// ── VideoPlayerConfig ─────────────────────────────────────────────────────────

/// Playback policy for one local video source.
#[pyclass(name = "VideoPlayerConfig")]
#[derive(Clone)]
pub(crate) struct PythonVideoPlayerConfig {
	inner: oa::video::VideoPlayerConfig,
}

#[pymethods]
impl PythonVideoPlayerConfig {
	#[new]
	#[pyo3(signature = (loop_playback=true, start_playing=true, frame_rate_override=None, presentation_cache_frames=32))]
	pub fn new(
		loop_playback: bool,
		start_playing: bool,
		frame_rate_override: Option<f64>,
		presentation_cache_frames: usize,
	) -> Self {
		let inner = oa::video::VideoPlayerConfig {
			loop_playback,
			start_playing,
			frame_rate_override,
			presentation_cache_frames,
			..Default::default()
		};
		Self { inner }
	}

	#[getter]
	pub fn loop_playback(&self) -> bool {
		self.inner.loop_playback
	}

	#[getter]
	pub fn start_playing(&self) -> bool {
		self.inner.start_playing
	}

	#[getter]
	pub fn frame_rate_override(&self) -> Option<f64> {
		self.inner.frame_rate_override
	}

	#[getter]
	pub fn presentation_cache_frames(&self) -> usize {
		self.inner.presentation_cache_frames
	}

	pub fn __repr__(&self) -> String {
		format!(
			"VideoPlayerConfig(loop={}, start_playing={}, fps_override={:?})",
			self.inner.loop_playback, self.inner.start_playing, self.inner.frame_rate_override,
		)
	}
}

// ── VideoPlayerStats ──────────────────────────────────────────────────────────

/// Counters and timing summary for one playback session.
#[pyclass(name = "VideoPlayerStats")]
#[derive(Clone)]
pub(crate) struct PythonVideoPlayerStats {
	inner: oa::video::VideoPlayerStats,
}

#[pymethods]
impl PythonVideoPlayerStats {
	#[getter]
	pub fn presented_frames(&self) -> u64 {
		self.inner.presented_frames
	}

	#[getter]
	pub fn decoded_packets(&self) -> u64 {
		self.inner.decoded_packets
	}

	#[getter]
	pub fn seek_resets(&self) -> u64 {
		self.inner.seek_resets
	}

	#[getter]
	pub fn loop_restarts(&self) -> u64 {
		self.inner.loop_restarts
	}

	#[getter]
	pub fn presentation_cache_hits(&self) -> u64 {
		self.inner.presentation_cache_hits
	}

	#[getter]
	pub fn presentation_cache_misses(&self) -> u64 {
		self.inner.presentation_cache_misses
	}

	#[getter]
	pub fn presentation_cache_resident(&self) -> usize {
		self.inner.presentation_cache_resident
	}

	pub fn __repr__(&self) -> String {
		format!(
			"VideoPlayerStats(presented={}, decoded_packets={})",
			self.inner.presented_frames, self.inner.decoded_packets,
		)
	}
}

// ── VideoPlayer ───────────────────────────────────────────────────────────────

/// Composed demux, decode, and presentation-order playback session.
#[pyclass(name = "VideoPlayer", unsendable)]
pub(crate) struct PythonVideoPlayer {
	inner: oa::VideoPlayer,
}

#[pymethods]
impl PythonVideoPlayer {
	/// Open a local MP4 file for playback.
	#[new]
	#[pyo3(signature = (engine, path, config=None))]
	pub fn new(
		engine: &PythonEngine,
		path: &str,
		config: Option<&PythonVideoPlayerConfig>,
	) -> PyResult<Self> {
		let cfg = config.map_or_else(oa::video::VideoPlayerConfig::default, |c| c.inner);
		oa::VideoPlayer::open(&engine.inner, path, cfg)
			.map(|inner| Self { inner })
			.map_err(python_error)
	}

	/// Advance to the next display-order frame.
	///
	/// Returns `True` while frames remain; `False` at end of stream.
	pub fn advance(&mut self) -> PyResult<bool> {
		self.inner.advance().map_err(python_error)
	}

	/// Step `delta` frames forward (positive) or backward (negative).
	pub fn step_frames(&mut self, delta: i32) -> PyResult<()> {
		self.inner.step_frames(delta).map_err(python_error)
	}

	/// Advance playback by `elapsed_us` microseconds.
	///
	/// Returns the number of frames advanced.
	pub fn tick(&mut self, elapsed_us: u64) -> PyResult<u32> {
		self
			.inner
			.tick(std::time::Duration::from_micros(elapsed_us))
			.map_err(python_error)
	}

	/// Seek to the sample nearest to `timestamp` (presentation microseconds).
	pub fn seek(&mut self, timestamp: u64) -> PyResult<()> {
		self.inner.seek(timestamp).map_err(python_error)
	}

	/// Seek to a specific frame index.
	pub fn seek_frame(&mut self, index: u64) -> PyResult<()> {
		self.inner.seek_frame(index).map_err(python_error)
	}

	/// Reset to the beginning of the stream.
	pub fn reset(&mut self) -> PyResult<()> {
		self.inner.reset().map_err(python_error)
	}

	/// Begin playback.
	pub fn play(&mut self) -> PyResult<()> {
		self.inner.play().map_err(python_error)
	}

	/// Pause playback.
	pub fn pause(&mut self) -> PyResult<()> {
		self.inner.pause().map_err(python_error)
	}

	/// Toggle between playing and paused states.
	pub fn toggle_play(&mut self) -> PyResult<()> {
		self.inner.toggle_play().map_err(python_error)
	}

	/// Set looping behavior.
	pub fn set_looping(&mut self, looping: bool) -> PyResult<()> {
		self.inner.set_looping(looping).map_err(python_error)
	}

	/// Return the current display-order frame.
	pub fn current_frame(&self) -> PyResult<PythonVideoFrame> {
		self
			.inner
			.current_frame()
			.map(|f| PythonVideoFrame::wrap(f.clone()))
			.map_err(python_error)
	}

	/// Return the current frame index.
	pub fn current_frame_index(&self) -> PyResult<u64> {
		self.inner.current_frame_index().map_err(python_error)
	}

	/// Return the container info, if the player has opened the file.
	pub fn info(&self) -> Option<PythonVideoContainerInfo> {
		self.inner.info().map(PythonVideoContainerInfo::wrap)
	}

	/// Return current playback statistics.
	pub fn stats(&self) -> Option<PythonVideoPlayerStats> {
		self
			.inner
			.stats()
			.map(|s| PythonVideoPlayerStats { inner: s })
	}

	#[getter]
	pub fn is_playing(&self) -> bool {
		self.inner.is_playing()
	}

	#[getter]
	pub fn is_done(&self) -> bool {
		self.inner.is_done()
	}

	/// Close the player and release resources.
	pub fn close(&mut self) {
		self.inner.close();
	}

	pub fn __repr__(&self) -> &'static str {
		"VideoPlayer"
	}
}

// ── VideoMuxerAudioConfig ─────────────────────────────────────────────────────

/// Optional native PCM-S16 track configuration for VideoMuxer.
#[pyclass(name = "VideoMuxerAudioConfig")]
#[derive(Clone)]
pub(crate) struct PythonVideoMuxerAudioConfig {
	inner: oa::video::VideoMuxerAudioConfig,
}

#[pymethods]
impl PythonVideoMuxerAudioConfig {
	#[new]
	#[pyo3(signature = (sample_rate=48000, channel_count=2, priming_frames=0))]
	pub fn new(sample_rate: u32, channel_count: u32, priming_frames: u32) -> Self {
		Self {
			inner: oa::video::VideoMuxerAudioConfig {
				sample_rate,
				channel_count,
				priming_frames,
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
	pub fn priming_frames(&self) -> u32 {
		self.inner.priming_frames
	}

	pub fn __repr__(&self) -> String {
		format!(
			"VideoMuxerAudioConfig(sample_rate={}, channels={})",
			self.inner.sample_rate, self.inner.channel_count,
		)
	}
}

// ── VideoMuxerConfig ──────────────────────────────────────────────────────────

/// Configuration for one streaming MP4 muxer.
#[pyclass(name = "VideoMuxerConfig")]
#[derive(Clone)]
pub(crate) struct PythonVideoMuxerConfig {
	inner: oa::video::VideoMuxerConfig,
}

#[pymethods]
impl PythonVideoMuxerConfig {
	/// Construct the default 30-fps, 90-kHz video-only profile.
	#[new]
	#[pyo3(signature = (codec, width, height, frame_rate=30, audio=None))]
	pub fn new(
		codec: PythonVideoCodec,
		width: u32,
		height: u32,
		frame_rate: u32,
		audio: Option<&PythonVideoMuxerAudioConfig>,
	) -> PyResult<Self> {
		let mut inner =
			oa::video::VideoMuxerConfig::new(codec.into(), width, height).map_err(python_error)?;
		inner.frame_rate = frame_rate;
		inner.audio = audio.map(|a| a.inner);
		Ok(Self { inner })
	}

	#[getter]
	pub fn codec(&self) -> PythonVideoCodec {
		self.inner.codec.into()
	}

	#[getter]
	pub fn width(&self) -> u32 {
		self.inner.width
	}

	#[getter]
	pub fn height(&self) -> u32 {
		self.inner.height
	}

	#[getter]
	pub fn frame_rate(&self) -> u32 {
		self.inner.frame_rate
	}

	pub fn __repr__(&self) -> String {
		format!(
			"VideoMuxerConfig(codec={:?}, {}x{}, fps={})",
			self.inner.codec, self.inner.width, self.inner.height, self.inner.frame_rate,
		)
	}
}

// ── EncodedVideoPacket ────────────────────────────────────────────────────────

/// One owned Annex-B video access unit with microsecond presentation timing.
#[pyclass(name = "EncodedVideoPacket", unsendable)]
pub(crate) struct PythonEncodedVideoPacket {
	inner: oa::video::EncodedVideoPacket,
}

impl PythonEncodedVideoPacket {
	pub(crate) fn wrap(inner: oa::video::EncodedVideoPacket) -> Self {
		Self { inner }
	}
}

#[pymethods]
impl PythonEncodedVideoPacket {
	/// Construct an access unit from Annex-B bytes and a presentation timestamp.
	#[new]
	#[pyo3(signature = (bitstream, presentation_timestamp_micros, keyframe=false))]
	pub fn new(
		bitstream: Vec<u8>,
		presentation_timestamp_micros: u64,
		keyframe: bool,
	) -> PyResult<Self> {
		oa::video::EncodedVideoPacket::new(bitstream, presentation_timestamp_micros, keyframe)
			.map(Self::wrap)
			.map_err(python_error)
	}

	/// Construct an access unit with distinct presentation and decode timestamps.
	#[staticmethod]
	pub fn with_timestamps(
		bitstream: Vec<u8>,
		presentation_timestamp_micros: u64,
		decode_timestamp_micros: u64,
		keyframe: bool,
	) -> PyResult<Self> {
		oa::video::EncodedVideoPacket::with_timestamps(
			bitstream,
			presentation_timestamp_micros,
			decode_timestamp_micros,
			keyframe,
		)
		.map(Self::wrap)
		.map_err(python_error)
	}

	/// Return the encoded Annex-B access unit.
	pub fn bitstream<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
		PyBytes::new(py, self.inner.bitstream())
	}

	#[getter]
	pub fn presentation_timestamp_micros(&self) -> u64 {
		self.inner.presentation_timestamp_micros()
	}

	#[getter]
	pub fn decode_timestamp_micros(&self) -> u64 {
		self.inner.decode_timestamp_micros()
	}

	#[getter]
	pub fn is_keyframe(&self) -> bool {
		self.inner.is_keyframe()
	}

	pub fn __repr__(&self) -> String {
		format!(
			"EncodedVideoPacket(pts={}us, keyframe={}, len={})",
			self.inner.presentation_timestamp_micros(),
			self.inner.is_keyframe(),
			self.inner.bitstream().len(),
		)
	}
}

// ── VideoMuxer ────────────────────────────────────────────────────────────────

/// Stateful streaming MP4 muxer for H.264/H.265 and optional PCM-S16 audio.
#[pyclass(name = "VideoMuxer", unsendable)]
pub(crate) struct PythonVideoMuxer {
	inner: oa::VideoMuxer,
}

#[pymethods]
impl PythonVideoMuxer {
	/// Create a new streaming MP4 file and write its fixed header.
	#[new]
	pub fn new(path: &str, config: &PythonVideoMuxerConfig) -> PyResult<Self> {
		oa::VideoMuxer::create(path, config.inner)
			.map(|inner| Self { inner })
			.map_err(python_error)
	}

	/// Set H.264 SPS/PPS codec config (call before writing the first packet).
	pub fn set_h264_codec_config(&mut self, sps: Vec<u8>, pps: Vec<u8>) -> PyResult<()> {
		self
			.inner
			.set_h264_codec_config(&sps, &pps)
			.map_err(python_error)
	}

	/// Set H.265 VPS/SPS/PPS codec config (call before writing the first packet).
	pub fn set_h265_codec_config(
		&mut self,
		vps: Vec<u8>,
		sps: Vec<u8>,
		pps: Vec<u8>,
	) -> PyResult<()> {
		self
			.inner
			.set_h265_codec_config(&vps, &sps, &pps)
			.map_err(python_error)
	}

	/// Write one encoded video packet.
	pub fn write_packet(&mut self, packet: &PythonEncodedVideoPacket) -> PyResult<()> {
		self.inner.write_packet(&packet.inner).map_err(python_error)
	}

	/// Append MP4 sample tables and close the file.
	pub fn finalize(&mut self) -> PyResult<()> {
		self.inner.finalize().map_err(python_error)
	}

	pub fn __repr__(&self) -> &'static str {
		"VideoMuxer"
	}
}

// ── VideoFrame ────────────────────────────────────────────────────────────────

#[pyclass(name = "VideoFrame", unsendable)]
pub(crate) struct PythonVideoFrame {
	inner: oa::VideoFrame,
}

impl PythonVideoFrame {
	pub(crate) fn wrap(inner: oa::VideoFrame) -> Self {
		Self { inner }
	}
}

#[pymethods]
impl PythonVideoFrame {
	/// Wrap an Image as a single-frame VideoFrame with zero-duration timing.
	#[staticmethod]
	#[pyo3(signature = (image, presentation_us=0))]
	fn from_image(image: &PythonImage, presentation_us: u64) -> PyResult<Self> {
		let timing = oa::video::VideoFrameTiming::from_microseconds(presentation_us, 0);
		let color = oa::video::VideoColorInfo::unspecified();
		oa::VideoFrame::from_image(image.inner.clone(), timing, color)
			.map(Self::wrap)
			.map_err(python_error)
	}

	/// Borrow the packed Image backing, if present.
	fn as_image(&self) -> PyResult<Option<PythonImage>> {
		Ok(
			self
				.inner
				.as_image()
				.map(|img| PythonImage::wrap(img.clone())),
		)
	}

	fn width(&self) -> usize {
		self.inner.width()
	}

	fn height(&self) -> usize {
		self.inner.height()
	}

	/// Return the presentation timestamp in microseconds.
	fn presentation_us(&self) -> u64 {
		self.inner.timing().presentation_timestamp().as_micros() as u64
	}

	/// Return the color metadata.
	fn color(&self) -> PythonVideoColorInfo {
		PythonVideoColorInfo {
			inner: self.inner.color_info(),
		}
	}

	/// Return the timing metadata.
	fn timing(&self) -> PythonVideoFrameTiming {
		PythonVideoFrameTiming {
			inner: self.inner.timing(),
		}
	}

	fn __repr__(&self) -> String {
		format!(
			"VideoFrame(width={}, height={})",
			self.inner.width(),
			self.inner.height(),
		)
	}
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
	module.add_class::<PythonVideoColorMatrix>()?;
	module.add_class::<PythonVideoColorRange>()?;
	module.add_class::<PythonVideoColorInfo>()?;
	module.add_class::<PythonVideoFrameTiming>()?;
	module.add_class::<PythonVideoCodec>()?;
	module.add_class::<PythonVideoContainerKind>()?;
	module.add_class::<PythonVideoContainerInfo>()?;
	module.add_class::<PythonVideoPacket>()?;
	module.add_class::<PythonVideoDemuxer>()?;
	module.add_class::<PythonVideoDecoder>()?;
	module.add_class::<PythonVideoPlayerConfig>()?;
	module.add_class::<PythonVideoPlayerStats>()?;
	module.add_class::<PythonVideoPlayer>()?;
	module.add_class::<PythonVideoMuxerAudioConfig>()?;
	module.add_class::<PythonVideoMuxerConfig>()?;
	module.add_class::<PythonEncodedVideoPacket>()?;
	module.add_class::<PythonVideoMuxer>()?;
	module.add_class::<PythonVideoFrame>()?;
	Ok(())
}
