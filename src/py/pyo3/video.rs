use pyo3::{prelude::*, types::PyBytes};

use crate::{error::python_error, image::PythonImage, runtime::PythonRuntime};

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
			inner: oa::video::VideoFrameTiming::new_from_microseconds(presentation_us, duration_us),
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
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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
	Matroska = 1,
	WebM = 2,
	MpegTs = 3,
}

impl From<oa::video::VideoContainerKind> for PythonVideoContainerKind {
	fn from(v: oa::video::VideoContainerKind) -> Self {
		match v {
			oa::video::VideoContainerKind::Mp4 => Self::Mp4,
			oa::video::VideoContainerKind::Matroska => Self::Matroska,
			oa::video::VideoContainerKind::WebM => Self::WebM,
			oa::video::VideoContainerKind::MpegTs => Self::MpegTs,
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
	pub fn new(info: &PythonVideoContainerInfo, runtime: Option<&PythonRuntime>) -> PyResult<Self> {
		let runtime = PythonRuntime::resolve(runtime)?;
		oa::VideoDecoder::new(&runtime, info.inner)
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
	#[pyo3(signature = (path, config=None, runtime=None))]
	pub fn new(
		path: &str,
		config: Option<&PythonVideoPlayerConfig>,
		runtime: Option<&PythonRuntime>,
	) -> PyResult<Self> {
		let runtime = PythonRuntime::resolve(runtime)?;
		let cfg = config.map_or_else(oa::video::VideoPlayerConfig::default, |c| c.inner);
		oa::VideoPlayer::open(&runtime, path, cfg)
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
	pub fn new_with_timestamps(
		bitstream: Vec<u8>,
		presentation_timestamp_micros: u64,
		decode_timestamp_micros: u64,
		keyframe: bool,
	) -> PyResult<Self> {
		oa::video::EncodedVideoPacket::new_with_timestamps(
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
		oa::VideoMuxer::new(path, config.inner)
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
	fn new_from_image(image: &PythonImage, presentation_us: u64) -> PyResult<Self> {
		let timing = oa::video::VideoFrameTiming::new_from_microseconds(presentation_us, 0);
		let color = oa::video::VideoColorInfo::unspecified();
		oa::VideoFrame::new_from_image(image.inner.clone(), timing, color)
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

// ── Profile and capability enums ────────────────────────────────────────────

macro_rules! video_enum {
	($name:literal, $outer:ident, $inner:ident, { $($variant:ident),+ $(,)? }) => {
		#[pyclass(name = $name, eq, eq_int)]
		#[derive(Clone, Copy, Debug, PartialEq, Eq)]
		pub(crate) enum $outer {
			$($variant,)+
		}

		impl From<oa::video::$inner> for $outer {
			fn from(v: oa::video::$inner) -> Self {
				match v {
					$(oa::video::$inner::$variant => Self::$variant,)+
				}
			}
		}

		impl From<$outer> for oa::video::$inner {
			fn from(v: $outer) -> Self {
				match v {
					$($outer::$variant => Self::$variant,)+
				}
			}
		}
	};
}

video_enum!("VideoChromaSubsampling", PythonVideoChromaSubsampling, VideoChromaSubsampling, {
	Monochrome, Yuv420, Yuv422, Yuv444,
});
video_enum!("VideoComponentBitDepth", PythonVideoComponentBitDepth, VideoComponentBitDepth, {
	Eight, Ten, Twelve,
});
video_enum!("H264Profile", PythonH264Profile, H264Profile, {
	Baseline, Main, High, High10, High444Predictive,
});
video_enum!("H264PictureLayout", PythonH264PictureLayout, H264PictureLayout, {
	Progressive, InterlacedInterleavedLines, InterlacedSeparatePlanes,
});
video_enum!("H265Profile", PythonH265Profile, H265Profile, {
	Main, Main10, MainStillPicture, FormatRangeExtensions, ScreenContentCodingExtensions,
});
video_enum!("Av1Profile", PythonAv1Profile, Av1Profile, {
	Main, High, Professional,
});
video_enum!("Vp9Profile", PythonVp9Profile, Vp9Profile, {
	Profile0, Profile1, Profile2, Profile3,
});
video_enum!("VideoRateControl", PythonVideoRateControl, VideoRateControl, {
	ConstantQp, Cbr, Vbr,
});
video_enum!("VideoPixelFormat", PythonVideoPixelFormat, VideoPixelFormat, {
	Yuv420Planar8, Nv12, Yuv420Planar10, P010, P012,
});

// ── VideoExtent ─────────────────────────────────────────────────────────────

/// Two-dimensional extent reported or requested without Vulkan types.
#[pyclass(name = "VideoExtent")]
#[derive(Clone)]
pub(crate) struct PythonVideoExtent {
	inner: oa::video::VideoExtent,
}

impl PythonVideoExtent {
	fn wrap(inner: oa::video::VideoExtent) -> Self {
		Self { inner }
	}
}

#[pymethods]
impl PythonVideoExtent {
	#[new]
	pub fn new(width: u32, height: u32) -> Self {
		Self::wrap(oa::video::VideoExtent { width, height })
	}

	#[getter]
	pub fn width(&self) -> u32 {
		self.inner.width
	}

	#[getter]
	pub fn height(&self) -> u32 {
		self.inner.height
	}

	pub fn __repr__(&self) -> String {
		format!("VideoExtent({}x{})", self.inner.width, self.inner.height)
	}
}

// ── VideoDecodeProfile ──────────────────────────────────────────────────────

/// One complete decode profile presented to the selected device.
#[pyclass(name = "VideoDecodeProfile")]
#[derive(Clone)]
pub(crate) struct PythonVideoDecodeProfile {
	inner: oa::video::VideoDecodeProfile,
}

#[pymethods]
impl PythonVideoDecodeProfile {
	/// Construct the common progressive H.264 4:2:0 8-bit profile.
	#[staticmethod]
	pub fn h264_420_8bit(profile: PythonH264Profile) -> Self {
		Self {
			inner: oa::video::VideoDecodeProfile::h264_420_8bit(profile.into()),
		}
	}

	/// Construct the common H.265 4:2:0 profile at one component precision.
	#[staticmethod]
	pub fn h265_420(profile: PythonH265Profile, bit_depth: PythonVideoComponentBitDepth) -> Self {
		Self {
			inner: oa::video::VideoDecodeProfile::h265_420(profile.into(), bit_depth.into()),
		}
	}

	/// Construct the common AV1 4:2:0 profile at one component precision.
	#[staticmethod]
	pub fn av1_420(
		profile: PythonAv1Profile,
		bit_depth: PythonVideoComponentBitDepth,
		film_grain_support: bool,
	) -> Self {
		Self {
			inner: oa::video::VideoDecodeProfile::av1_420(
				profile.into(),
				bit_depth.into(),
				film_grain_support,
			),
		}
	}

	/// Construct the common VP9 4:2:0 profile at one component precision.
	#[staticmethod]
	pub fn vp9_420(profile: PythonVp9Profile, bit_depth: PythonVideoComponentBitDepth) -> Self {
		Self {
			inner: oa::video::VideoDecodeProfile::vp9_420(profile.into(), bit_depth.into()),
		}
	}

	/// Construct an exact H.264 profile with explicit layout and precision.
	#[staticmethod]
	pub fn h264(
		profile: PythonH264Profile,
		picture_layout: PythonH264PictureLayout,
		chroma_subsampling: PythonVideoChromaSubsampling,
		luma_bit_depth: PythonVideoComponentBitDepth,
		chroma_bit_depth: PythonVideoComponentBitDepth,
	) -> Self {
		Self {
			inner: oa::video::VideoDecodeProfile::H264 {
				profile: profile.into(),
				picture_layout: picture_layout.into(),
				chroma_subsampling: chroma_subsampling.into(),
				luma_bit_depth: luma_bit_depth.into(),
				chroma_bit_depth: chroma_bit_depth.into(),
			},
		}
	}

	/// Construct an exact H.265 profile with explicit precision.
	#[staticmethod]
	pub fn h265(
		profile: PythonH265Profile,
		chroma_subsampling: PythonVideoChromaSubsampling,
		luma_bit_depth: PythonVideoComponentBitDepth,
		chroma_bit_depth: PythonVideoComponentBitDepth,
	) -> Self {
		Self {
			inner: oa::video::VideoDecodeProfile::H265 {
				profile: profile.into(),
				chroma_subsampling: chroma_subsampling.into(),
				luma_bit_depth: luma_bit_depth.into(),
				chroma_bit_depth: chroma_bit_depth.into(),
			},
		}
	}

	/// Construct an exact AV1 profile with explicit precision.
	#[staticmethod]
	pub fn av1(
		profile: PythonAv1Profile,
		film_grain_support: bool,
		chroma_subsampling: PythonVideoChromaSubsampling,
		luma_bit_depth: PythonVideoComponentBitDepth,
		chroma_bit_depth: PythonVideoComponentBitDepth,
	) -> Self {
		Self {
			inner: oa::video::VideoDecodeProfile::Av1 {
				profile: profile.into(),
				film_grain_support,
				chroma_subsampling: chroma_subsampling.into(),
				luma_bit_depth: luma_bit_depth.into(),
				chroma_bit_depth: chroma_bit_depth.into(),
			},
		}
	}

	/// Construct an exact VP9 profile with explicit precision.
	#[staticmethod]
	pub fn vp9(
		profile: PythonVp9Profile,
		chroma_subsampling: PythonVideoChromaSubsampling,
		luma_bit_depth: PythonVideoComponentBitDepth,
		chroma_bit_depth: PythonVideoComponentBitDepth,
	) -> Self {
		Self {
			inner: oa::video::VideoDecodeProfile::Vp9 {
				profile: profile.into(),
				chroma_subsampling: chroma_subsampling.into(),
				luma_bit_depth: luma_bit_depth.into(),
				chroma_bit_depth: chroma_bit_depth.into(),
			},
		}
	}

	/// Elementary codec selected by this profile.
	pub fn codec(&self) -> PythonVideoCodec {
		match self.inner {
			oa::video::VideoDecodeProfile::H264 { .. } => PythonVideoCodec::H264,
			oa::video::VideoDecodeProfile::H265 { .. } => PythonVideoCodec::H265,
			oa::video::VideoDecodeProfile::Av1 { .. } => PythonVideoCodec::Av1,
			oa::video::VideoDecodeProfile::Vp9 { .. } => PythonVideoCodec::Vp9,
		}
	}

	pub fn __repr__(&self) -> String {
		format!("VideoDecodeProfile(codec={:?})", self.codec())
	}
}

// ── VideoEncodeProfile ──────────────────────────────────────────────────────

/// One exact encode profile presented to the selected device.
#[pyclass(name = "VideoEncodeProfile")]
#[derive(Clone)]
pub(crate) struct PythonVideoEncodeProfile {
	inner: oa::video::VideoEncodeProfile,
}

#[pymethods]
impl PythonVideoEncodeProfile {
	/// Construct the donor H.264 High 4:2:0 8-bit encode profile.
	#[staticmethod]
	pub fn h264_high_420_8bit() -> Self {
		Self {
			inner: oa::video::VideoEncodeProfile::h264_high_420_8bit(),
		}
	}

	/// Construct the donor H.265 Main 4:2:0 8-bit encode profile.
	#[staticmethod]
	pub fn h265_main_420_8bit() -> Self {
		Self {
			inner: oa::video::VideoEncodeProfile::h265_main_420_8bit(),
		}
	}

	/// Construct an exact H.264 profile with explicit precision.
	#[staticmethod]
	pub fn h264(
		profile: PythonH264Profile,
		chroma_subsampling: PythonVideoChromaSubsampling,
		luma_bit_depth: PythonVideoComponentBitDepth,
		chroma_bit_depth: PythonVideoComponentBitDepth,
	) -> Self {
		Self {
			inner: oa::video::VideoEncodeProfile::H264 {
				profile: profile.into(),
				chroma_subsampling: chroma_subsampling.into(),
				luma_bit_depth: luma_bit_depth.into(),
				chroma_bit_depth: chroma_bit_depth.into(),
			},
		}
	}

	/// Construct an exact H.265 profile with explicit precision.
	#[staticmethod]
	pub fn h265(
		profile: PythonH265Profile,
		chroma_subsampling: PythonVideoChromaSubsampling,
		luma_bit_depth: PythonVideoComponentBitDepth,
		chroma_bit_depth: PythonVideoComponentBitDepth,
	) -> Self {
		Self {
			inner: oa::video::VideoEncodeProfile::H265 {
				profile: profile.into(),
				chroma_subsampling: chroma_subsampling.into(),
				luma_bit_depth: luma_bit_depth.into(),
				chroma_bit_depth: chroma_bit_depth.into(),
			},
		}
	}

	/// Elementary codec selected by this profile.
	pub fn codec(&self) -> PythonVideoCodec {
		match self.inner {
			oa::video::VideoEncodeProfile::H264 { .. } => PythonVideoCodec::H264,
			oa::video::VideoEncodeProfile::H265 { .. } => PythonVideoCodec::H265,
		}
	}

	pub fn __repr__(&self) -> String {
		format!("VideoEncodeProfile(codec={:?})", self.codec())
	}
}

// ── Capability and format reports ───────────────────────────────────────────

/// One image representation usable for video output or reconstruction.
#[pyclass(name = "VideoImageFormat")]
#[derive(Clone)]
pub(crate) struct PythonVideoImageFormat {
	inner: oa::video::VideoImageFormat,
}

impl PythonVideoImageFormat {
	fn wrap(inner: oa::video::VideoImageFormat) -> Self {
		Self { inner }
	}
}

#[pymethods]
impl PythonVideoImageFormat {
	#[getter]
	pub fn pixel_format(&self) -> PythonVideoPixelFormat {
		self.inner.pixel_format().into()
	}

	#[getter]
	pub fn optimal_tiling(&self) -> bool {
		self.inner.optimal_tiling()
	}

	#[getter]
	pub fn sampled(&self) -> bool {
		self.inner.sampled()
	}

	#[getter]
	pub fn storage(&self) -> bool {
		self.inner.storage()
	}

	#[getter]
	pub fn transfer_source(&self) -> bool {
		self.inner.transfer_source()
	}

	#[getter]
	pub fn transfer_destination(&self) -> bool {
		self.inner.transfer_destination()
	}

	pub fn __repr__(&self) -> String {
		format!("VideoImageFormat(pixel_format={:?})", self.pixel_format())
	}
}

/// Codec-specific maximum level reported for a decode profile.
#[pyclass(name = "VideoDecodeLevel")]
#[derive(Clone)]
pub(crate) struct PythonVideoDecodeLevel {
	inner: oa::video::VideoDecodeLevel,
}

#[pymethods]
impl PythonVideoDecodeLevel {
	#[getter]
	pub fn codec(&self) -> PythonVideoCodec {
		match self.inner {
			oa::video::VideoDecodeLevel::H264(_) => PythonVideoCodec::H264,
			oa::video::VideoDecodeLevel::H265(_) => PythonVideoCodec::H265,
			oa::video::VideoDecodeLevel::Av1(_) => PythonVideoCodec::Av1,
			oa::video::VideoDecodeLevel::Vp9(_) => PythonVideoCodec::Vp9,
		}
	}

	/// Numeric bitstream level_idc or codec-equivalent level value.
	#[getter]
	pub fn level(&self) -> u32 {
		match self.inner {
			oa::video::VideoDecodeLevel::H264(v)
			| oa::video::VideoDecodeLevel::H265(v)
			| oa::video::VideoDecodeLevel::Av1(v)
			| oa::video::VideoDecodeLevel::Vp9(v) => v,
		}
	}

	pub fn __repr__(&self) -> String {
		format!(
			"VideoDecodeLevel(codec={:?}, level={})",
			self.codec(),
			self.level()
		)
	}
}

/// Device-level video queue and codec admission summary.
#[pyclass(name = "VideoDeviceCapabilities")]
#[derive(Clone)]
pub(crate) struct PythonVideoDeviceCapabilities {
	inner: oa::video::VideoDeviceCapabilities,
}

#[pymethods]
impl PythonVideoDeviceCapabilities {
	#[getter]
	pub fn decode_queue_family(&self) -> Option<u32> {
		self.inner.decode_queue_family()
	}

	#[getter]
	pub fn encode_queue_family(&self) -> Option<u32> {
		self.inner.encode_queue_family()
	}

	#[getter]
	pub fn supports_decode_result_status_queries(&self) -> bool {
		self.inner.supports_decode_result_status_queries()
	}

	#[getter]
	pub fn supports_encode_result_status_queries(&self) -> bool {
		self.inner.supports_encode_result_status_queries()
	}

	#[getter]
	pub fn supports_h264_decode(&self) -> bool {
		self.inner.supports_h264_decode()
	}

	#[getter]
	pub fn supports_h265_decode(&self) -> bool {
		self.inner.supports_h265_decode()
	}

	#[getter]
	pub fn supports_av1_decode(&self) -> bool {
		self.inner.supports_av1_decode()
	}

	#[getter]
	pub fn supports_vp9_decode(&self) -> bool {
		self.inner.supports_vp9_decode()
	}

	#[getter]
	pub fn supports_h264_encode(&self) -> bool {
		self.inner.supports_h264_encode()
	}

	#[getter]
	pub fn supports_h265_encode(&self) -> bool {
		self.inner.supports_h265_encode()
	}

	#[getter]
	pub fn supports_av1_encode(&self) -> bool {
		self.inner.supports_av1_encode()
	}

	#[getter]
	pub fn video_queues_enabled(&self) -> bool {
		self.inner.video_queues_enabled()
	}

	#[getter]
	pub fn decoder_sessions_available(&self) -> bool {
		self.inner.decoder_sessions_available()
	}

	#[getter]
	pub fn encoder_sessions_available(&self) -> bool {
		self.inner.encoder_sessions_available()
	}
}

/// Exact limits returned for one decode profile query.
#[pyclass(name = "VideoDecodeCapabilities")]
#[derive(Clone)]
pub(crate) struct PythonVideoDecodeCapabilities {
	inner: oa::video::VideoDecodeCapabilities,
}

#[pymethods]
impl PythonVideoDecodeCapabilities {
	#[getter]
	pub fn profile(&self) -> PythonVideoDecodeProfile {
		PythonVideoDecodeProfile {
			inner: self.inner.profile(),
		}
	}

	#[getter]
	pub fn min_coded_extent(&self) -> PythonVideoExtent {
		PythonVideoExtent::wrap(self.inner.min_coded_extent())
	}

	#[getter]
	pub fn max_coded_extent(&self) -> PythonVideoExtent {
		PythonVideoExtent::wrap(self.inner.max_coded_extent())
	}

	#[getter]
	pub fn picture_access_granularity(&self) -> PythonVideoExtent {
		PythonVideoExtent::wrap(self.inner.picture_access_granularity())
	}

	#[getter]
	pub fn min_bitstream_offset_alignment(&self) -> u64 {
		self.inner.min_bitstream_offset_alignment()
	}

	#[getter]
	pub fn min_bitstream_size_alignment(&self) -> u64 {
		self.inner.min_bitstream_size_alignment()
	}

	#[getter]
	pub fn max_dpb_slots(&self) -> u32 {
		self.inner.max_dpb_slots()
	}

	#[getter]
	pub fn max_active_reference_pictures(&self) -> u32 {
		self.inner.max_active_reference_pictures()
	}

	#[getter]
	pub fn level(&self) -> PythonVideoDecodeLevel {
		PythonVideoDecodeLevel {
			inner: self.inner.level(),
		}
	}

	#[getter]
	pub fn field_offset_granularity(&self) -> Option<(i32, i32)> {
		self.inner.field_offset_granularity()
	}

	#[getter]
	pub fn dpb_and_output_coincide(&self) -> bool {
		self.inner.dpb_and_output_coincide()
	}

	#[getter]
	pub fn dpb_and_output_distinct(&self) -> bool {
		self.inner.dpb_and_output_distinct()
	}

	#[getter]
	pub fn protected_content(&self) -> bool {
		self.inner.protected_content()
	}

	#[getter]
	pub fn separate_reference_images(&self) -> bool {
		self.inner.separate_reference_images()
	}
}

/// Codec-specific limits returned with an exact encode capability query.
#[pyclass(name = "VideoEncodeCodecCapabilities")]
#[derive(Clone)]
pub(crate) struct PythonVideoEncodeCodecCapabilities {
	inner: oa::video::VideoEncodeCodecCapabilities,
}

#[pymethods]
impl PythonVideoEncodeCodecCapabilities {
	#[getter]
	pub fn codec(&self) -> PythonVideoCodec {
		match self.inner {
			oa::video::VideoEncodeCodecCapabilities::H264 { .. } => PythonVideoCodec::H264,
			oa::video::VideoEncodeCodecCapabilities::H265 { .. } => PythonVideoCodec::H265,
		}
	}

	#[getter]
	pub fn max_level(&self) -> u32 {
		match self.inner {
			oa::video::VideoEncodeCodecCapabilities::H264 { max_level, .. }
			| oa::video::VideoEncodeCodecCapabilities::H265 { max_level, .. } => max_level,
		}
	}

	#[getter]
	pub fn max_slice_count(&self) -> Option<u32> {
		match self.inner {
			oa::video::VideoEncodeCodecCapabilities::H264 {
				max_slice_count, ..
			} => Some(max_slice_count),
			oa::video::VideoEncodeCodecCapabilities::H265 { .. } => None,
		}
	}

	#[getter]
	pub fn max_slice_segment_count(&self) -> Option<u32> {
		match self.inner {
			oa::video::VideoEncodeCodecCapabilities::H265 {
				max_slice_segment_count,
				..
			} => Some(max_slice_segment_count),
			oa::video::VideoEncodeCodecCapabilities::H264 { .. } => None,
		}
	}

	#[getter]
	pub fn max_tiles(&self) -> Option<PythonVideoExtent> {
		match self.inner {
			oa::video::VideoEncodeCodecCapabilities::H265 { max_tiles, .. } => {
				Some(PythonVideoExtent::wrap(max_tiles))
			}
			oa::video::VideoEncodeCodecCapabilities::H264 { .. } => None,
		}
	}

	#[getter]
	pub fn ctb_size_16(&self) -> Option<bool> {
		match self.inner {
			oa::video::VideoEncodeCodecCapabilities::H265 { ctb_size_16, .. } => Some(ctb_size_16),
			oa::video::VideoEncodeCodecCapabilities::H264 { .. } => None,
		}
	}

	#[getter]
	pub fn ctb_size_32(&self) -> Option<bool> {
		match self.inner {
			oa::video::VideoEncodeCodecCapabilities::H265 { ctb_size_32, .. } => Some(ctb_size_32),
			oa::video::VideoEncodeCodecCapabilities::H264 { .. } => None,
		}
	}

	#[getter]
	pub fn ctb_size_64(&self) -> Option<bool> {
		match self.inner {
			oa::video::VideoEncodeCodecCapabilities::H265 { ctb_size_64, .. } => Some(ctb_size_64),
			oa::video::VideoEncodeCodecCapabilities::H264 { .. } => None,
		}
	}

	#[getter]
	pub fn transform_size_4(&self) -> Option<bool> {
		match self.inner {
			oa::video::VideoEncodeCodecCapabilities::H265 {
				transform_size_4, ..
			} => Some(transform_size_4),
			oa::video::VideoEncodeCodecCapabilities::H264 { .. } => None,
		}
	}

	#[getter]
	pub fn transform_size_8(&self) -> Option<bool> {
		match self.inner {
			oa::video::VideoEncodeCodecCapabilities::H265 {
				transform_size_8, ..
			} => Some(transform_size_8),
			oa::video::VideoEncodeCodecCapabilities::H264 { .. } => None,
		}
	}

	#[getter]
	pub fn transform_size_16(&self) -> Option<bool> {
		match self.inner {
			oa::video::VideoEncodeCodecCapabilities::H265 {
				transform_size_16, ..
			} => Some(transform_size_16),
			oa::video::VideoEncodeCodecCapabilities::H264 { .. } => None,
		}
	}

	#[getter]
	pub fn transform_size_32(&self) -> Option<bool> {
		match self.inner {
			oa::video::VideoEncodeCodecCapabilities::H265 {
				transform_size_32, ..
			} => Some(transform_size_32),
			oa::video::VideoEncodeCodecCapabilities::H264 { .. } => None,
		}
	}

	#[getter]
	pub fn max_p_l0_references(&self) -> u32 {
		match self.inner {
			oa::video::VideoEncodeCodecCapabilities::H264 {
				max_p_l0_references,
				..
			}
			| oa::video::VideoEncodeCodecCapabilities::H265 {
				max_p_l0_references,
				..
			} => max_p_l0_references,
		}
	}

	#[getter]
	pub fn max_b_l0_references(&self) -> u32 {
		match self.inner {
			oa::video::VideoEncodeCodecCapabilities::H264 {
				max_b_l0_references,
				..
			}
			| oa::video::VideoEncodeCodecCapabilities::H265 {
				max_b_l0_references,
				..
			} => max_b_l0_references,
		}
	}

	#[getter]
	pub fn max_l1_references(&self) -> u32 {
		match self.inner {
			oa::video::VideoEncodeCodecCapabilities::H264 {
				max_l1_references, ..
			}
			| oa::video::VideoEncodeCodecCapabilities::H265 {
				max_l1_references, ..
			} => max_l1_references,
		}
	}

	#[getter]
	pub fn max_temporal_layers(&self) -> Option<u32> {
		match self.inner {
			oa::video::VideoEncodeCodecCapabilities::H264 {
				max_temporal_layers,
				..
			} => Some(max_temporal_layers),
			oa::video::VideoEncodeCodecCapabilities::H265 { .. } => None,
		}
	}

	#[getter]
	pub fn max_sub_layers(&self) -> Option<u32> {
		match self.inner {
			oa::video::VideoEncodeCodecCapabilities::H265 { max_sub_layers, .. } => Some(max_sub_layers),
			oa::video::VideoEncodeCodecCapabilities::H264 { .. } => None,
		}
	}

	#[getter]
	pub fn min_qp(&self) -> i32 {
		match self.inner {
			oa::video::VideoEncodeCodecCapabilities::H264 { min_qp, .. }
			| oa::video::VideoEncodeCodecCapabilities::H265 { min_qp, .. } => min_qp,
		}
	}

	#[getter]
	pub fn max_qp(&self) -> i32 {
		match self.inner {
			oa::video::VideoEncodeCodecCapabilities::H264 { max_qp, .. }
			| oa::video::VideoEncodeCodecCapabilities::H265 { max_qp, .. } => max_qp,
		}
	}
}

/// Exact limits returned for one encode profile query.
#[pyclass(name = "VideoEncodeCapabilities")]
#[derive(Clone)]
pub(crate) struct PythonVideoEncodeCapabilities {
	inner: oa::video::VideoEncodeCapabilities,
}

#[pymethods]
impl PythonVideoEncodeCapabilities {
	#[getter]
	pub fn profile(&self) -> PythonVideoEncodeProfile {
		PythonVideoEncodeProfile {
			inner: self.inner.profile(),
		}
	}

	#[getter]
	pub fn min_coded_extent(&self) -> PythonVideoExtent {
		PythonVideoExtent::wrap(self.inner.min_coded_extent())
	}

	#[getter]
	pub fn max_coded_extent(&self) -> PythonVideoExtent {
		PythonVideoExtent::wrap(self.inner.max_coded_extent())
	}

	#[getter]
	pub fn picture_access_granularity(&self) -> PythonVideoExtent {
		PythonVideoExtent::wrap(self.inner.picture_access_granularity())
	}

	#[getter]
	pub fn input_picture_granularity(&self) -> PythonVideoExtent {
		PythonVideoExtent::wrap(self.inner.input_picture_granularity())
	}

	#[getter]
	pub fn min_bitstream_offset_alignment(&self) -> u64 {
		self.inner.min_bitstream_offset_alignment()
	}

	#[getter]
	pub fn min_bitstream_size_alignment(&self) -> u64 {
		self.inner.min_bitstream_size_alignment()
	}

	#[getter]
	pub fn max_dpb_slots(&self) -> u32 {
		self.inner.max_dpb_slots()
	}

	#[getter]
	pub fn max_active_reference_pictures(&self) -> u32 {
		self.inner.max_active_reference_pictures()
	}

	#[getter]
	pub fn max_rate_control_layers(&self) -> u32 {
		self.inner.max_rate_control_layers()
	}

	#[getter]
	pub fn max_bitrate(&self) -> u64 {
		self.inner.max_bitrate()
	}

	#[getter]
	pub fn max_quality_levels(&self) -> u32 {
		self.inner.max_quality_levels()
	}

	#[getter]
	pub fn supports_constant_qp(&self) -> bool {
		self.inner.supports_constant_qp()
	}

	#[getter]
	pub fn supports_cbr(&self) -> bool {
		self.inner.supports_cbr()
	}

	#[getter]
	pub fn supports_vbr(&self) -> bool {
		self.inner.supports_vbr()
	}

	#[getter]
	pub fn supports_feedback_offset(&self) -> bool {
		self.inner.supports_feedback_offset()
	}

	#[getter]
	pub fn supports_feedback_bytes_written(&self) -> bool {
		self.inner.supports_feedback_bytes_written()
	}

	#[getter]
	pub fn supports_feedback_overrides(&self) -> bool {
		self.inner.supports_feedback_overrides()
	}

	#[getter]
	pub fn protected_content(&self) -> bool {
		self.inner.protected_content()
	}

	#[getter]
	pub fn separate_reference_images(&self) -> bool {
		self.inner.separate_reference_images()
	}

	#[getter]
	pub fn codec(&self) -> PythonVideoEncodeCodecCapabilities {
		PythonVideoEncodeCodecCapabilities {
			inner: self.inner.codec(),
		}
	}
}

/// Decode-output and decoded-picture-buffer formats for one exact profile.
#[pyclass(name = "VideoDecodeFormats")]
#[derive(Clone)]
pub(crate) struct PythonVideoDecodeFormats {
	inner: oa::video::VideoDecodeFormats,
}

#[pymethods]
impl PythonVideoDecodeFormats {
	#[getter]
	pub fn profile(&self) -> PythonVideoDecodeProfile {
		PythonVideoDecodeProfile {
			inner: self.inner.profile(),
		}
	}

	#[getter]
	pub fn output(&self) -> Vec<PythonVideoImageFormat> {
		self
			.inner
			.output()
			.iter()
			.copied()
			.map(PythonVideoImageFormat::wrap)
			.collect()
	}

	#[getter]
	pub fn dpb(&self) -> Vec<PythonVideoImageFormat> {
		self
			.inner
			.dpb()
			.iter()
			.copied()
			.map(PythonVideoImageFormat::wrap)
			.collect()
	}

	#[getter]
	pub fn unrecognized_output_formats(&self) -> usize {
		self.inner.unrecognized_output_formats()
	}

	#[getter]
	pub fn unrecognized_dpb_formats(&self) -> usize {
		self.inner.unrecognized_dpb_formats()
	}
}

/// Encode-input and reconstructed-picture formats for one exact profile.
#[pyclass(name = "VideoEncodeFormats")]
#[derive(Clone)]
pub(crate) struct PythonVideoEncodeFormats {
	inner: oa::video::VideoEncodeFormats,
}

#[pymethods]
impl PythonVideoEncodeFormats {
	#[getter]
	pub fn profile(&self) -> PythonVideoEncodeProfile {
		PythonVideoEncodeProfile {
			inner: self.inner.profile(),
		}
	}

	#[getter]
	pub fn input(&self) -> Vec<PythonVideoImageFormat> {
		self
			.inner
			.input()
			.iter()
			.copied()
			.map(PythonVideoImageFormat::wrap)
			.collect()
	}

	#[getter]
	pub fn dpb(&self) -> Vec<PythonVideoImageFormat> {
		self
			.inner
			.dpb()
			.iter()
			.copied()
			.map(PythonVideoImageFormat::wrap)
			.collect()
	}

	#[getter]
	pub fn unrecognized_input_formats(&self) -> usize {
		self.inner.unrecognized_input_formats()
	}

	#[getter]
	pub fn unrecognized_dpb_formats(&self) -> usize {
		self.inner.unrecognized_dpb_formats()
	}
}

// ── VideoEncodeConfig / VideoEncodePlan ─────────────────────────────────────

/// Validated, hardware-independent settings for encode session creation.
#[pyclass(name = "VideoEncodeConfig")]
#[derive(Clone)]
pub(crate) struct PythonVideoEncodeConfig {
	inner: oa::video::VideoEncodeConfig,
}

#[pymethods]
impl PythonVideoEncodeConfig {
	/// Construct the donor's default encode settings for a visible extent.
	///
	/// Defaults: 30 fps, GOP 30, constant QP 26, bitrate 4 Mbit/s, quality
	/// level 0, async depth 3, no B-frames, default DPB ring.
	#[allow(clippy::too_many_arguments)]
	#[new]
	#[pyo3(signature = (
		profile, width, height,
		frame_rate = 30, gop_size = 30,
		rate_control = PythonVideoRateControl::ConstantQp,
		bitrate = 4_000_000, max_bitrate = 0, constant_qp = 26,
		quality_level = 0, async_depth = 3, max_b_frames = 0, max_dpb_slots = 0,
	))]
	pub fn new(
		profile: &PythonVideoEncodeProfile,
		width: u32,
		height: u32,
		frame_rate: u32,
		gop_size: u32,
		rate_control: PythonVideoRateControl,
		bitrate: u32,
		max_bitrate: u32,
		constant_qp: u32,
		quality_level: u32,
		async_depth: u32,
		max_b_frames: u32,
		max_dpb_slots: u32,
	) -> Self {
		Self {
			inner: oa::video::VideoEncodeConfig {
				profile: profile.inner,
				width,
				height,
				frame_rate,
				gop_size,
				rate_control: rate_control.into(),
				bitrate,
				max_bitrate,
				constant_qp,
				quality_level,
				async_depth,
				max_b_frames,
				max_dpb_slots,
			},
		}
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

	#[getter]
	pub fn gop_size(&self) -> u32 {
		self.inner.gop_size
	}

	/// Validate this request against one exact queried capability row.
	pub fn validate_for_device(
		&self,
		caps: &PythonVideoEncodeCapabilities,
	) -> PyResult<PythonVideoEncodePlan> {
		self
			.inner
			.validate_for_device(caps.inner)
			.map(|inner| PythonVideoEncodePlan { inner })
			.map_err(python_error)
	}

	pub fn __repr__(&self) -> String {
		format!(
			"VideoEncodeConfig({}x{}@{})",
			self.inner.width, self.inner.height, self.inner.frame_rate,
		)
	}
}

/// Exact preflight result for an admitted encode session.
#[pyclass(name = "VideoEncodePlan")]
#[derive(Clone)]
pub(crate) struct PythonVideoEncodePlan {
	inner: oa::video::VideoEncodePlan,
}

#[pymethods]
impl PythonVideoEncodePlan {
	#[getter]
	pub fn coded_extent(&self) -> PythonVideoExtent {
		PythonVideoExtent::wrap(self.inner.coded_extent())
	}

	#[getter]
	pub fn peak_bitrate(&self) -> u32 {
		self.inner.peak_bitrate()
	}

	/// Numeric H.264 level IDC selected for the coded extent, if applicable.
	#[getter]
	pub fn h264_level(&self) -> Option<u32> {
		self.inner.h264_level()
	}
}

// ── VideoEncoder ────────────────────────────────────────────────────────────

/// Stateful hardware encoder session for H.264 High or H.265 Main/Main10.
#[pyclass(name = "VideoEncoder", unsendable)]
pub(crate) struct PythonVideoEncoder {
	inner: oa::video::VideoEncoder,
}

#[pymethods]
impl PythonVideoEncoder {
	/// Create a session admitted by exact capability preflight.
	#[new]
	pub fn new(config: &PythonVideoEncodeConfig, runtime: Option<&PythonRuntime>) -> PyResult<Self> {
		let runtime = PythonRuntime::resolve(runtime)?;
		oa::video::VideoEncoder::new(&runtime, config.inner)
			.map(|inner| Self { inner })
			.map_err(python_error)
	}

	/// Encode one packed RGBA8 frame and return its access unit.
	pub fn encode(
		&mut self,
		rgba: &[u8],
		presentation_timestamp_micros: u64,
	) -> PyResult<PythonEncodedVideoPacket> {
		self
			.inner
			.encode(rgba, presentation_timestamp_micros)
			.map(PythonEncodedVideoPacket::wrap)
			.map_err(python_error)
	}

	/// Encode one packed RGBA64 frame (little-endian u16 samples) and return
	/// its access unit. Requires a wide-component profile such as H.265 Main10.
	pub fn encode_rgba16(
		&mut self,
		rgba: Vec<u16>,
		presentation_timestamp_micros: u64,
	) -> PyResult<PythonEncodedVideoPacket> {
		self
			.inner
			.encode_rgba16(&rgba, presentation_timestamp_micros)
			.map(PythonEncodedVideoPacket::wrap)
			.map_err(python_error)
	}

	/// Wait for every outstanding encode job and drain all remaining packets.
	pub fn flush(&mut self) -> PyResult<Vec<PythonEncodedVideoPacket>> {
		self
			.inner
			.flush()
			.map(|packets| {
				packets
					.into_iter()
					.map(PythonEncodedVideoPacket::wrap)
					.collect()
			})
			.map_err(python_error)
	}

	/// Close the encoder and release its session resources.
	pub fn close(&mut self) {
		self.inner.close();
	}

	/// Return whether the encoder still accepts frames.
	#[getter]
	pub fn is_open(&self) -> bool {
		self.inner.is_open()
	}

	/// Coded (aligned) extent, while the session is open.
	#[getter]
	pub fn coded_extent(&self) -> Option<PythonVideoExtent> {
		self.inner.coded_extent().map(PythonVideoExtent::wrap)
	}

	/// Visible input extent, while the session is open.
	#[getter]
	pub fn visible_extent(&self) -> Option<PythonVideoExtent> {
		self.inner.visible_extent().map(PythonVideoExtent::wrap)
	}

	pub fn __repr__(&self) -> String {
		format!("VideoEncoder(open={})", self.inner.is_open())
	}
}

// ── VideoRecorder ───────────────────────────────────────────────────────────

/// Settings for one container-bound hardware recorder session.
#[pyclass(name = "VideoRecorderConfig")]
#[derive(Clone)]
pub(crate) struct PythonVideoRecorderConfig {
	inner: oa::video::VideoRecorderConfig,
}

#[pymethods]
impl PythonVideoRecorderConfig {
	/// Construct recorder settings for one output path and encode profile.
	///
	/// `color_space=None` selects BT.709; `audio` accepts an
	/// `AudioEncodeProfile` for an interleaved PCM-S16 track.
	#[new]
	#[pyo3(signature = (output_path, encode, color_space = None, full_range = false, audio = None))]
	pub fn new(
		output_path: &str,
		encode: &PythonVideoEncodeConfig,
		color_space: Option<PythonVideoColorMatrix>,
		full_range: bool,
		audio: Option<&crate::audio::sessions::PythonAudioEncodeProfile>,
	) -> Self {
		Self {
			inner: oa::video::VideoRecorderConfig {
				output_path: output_path.to_owned(),
				encode: encode.inner,
				color_space: color_space
					.map(Into::into)
					.unwrap_or(oa::video::VideoColorMatrix::Unspecified),
				full_range,
				audio: audio.map(|profile| profile.inner),
			},
		}
	}
}

/// Stateful hardware video recorder writing one MP4 container.
#[pyclass(name = "VideoRecorder", unsendable)]
pub(crate) struct PythonVideoRecorder {
	inner: oa::video::VideoRecorder,
}

#[pymethods]
impl PythonVideoRecorder {
	/// Open the output container and create the admitted encode session.
	#[new]
	pub fn new(
		config: &PythonVideoRecorderConfig,
		runtime: Option<&PythonRuntime>,
	) -> PyResult<Self> {
		let runtime = PythonRuntime::resolve(runtime)?;
		oa::video::VideoRecorder::new(&runtime, config.inner.clone())
			.map(|inner| Self { inner })
			.map_err(python_error)
	}

	/// Convert and encode one packed RGBA8 frame at `pts_micros`.
	pub fn write_rgba(&mut self, rgba: &[u8], pts_micros: u64) -> PyResult<()> {
		self
			.inner
			.write_rgba(rgba, pts_micros)
			.map_err(python_error)
	}

	/// Encode one chunk of interleaved FP32 audio samples.
	pub fn write_audio(
		&mut self,
		interleaved: Vec<f32>,
		sample_rate: u32,
		channel_count: u32,
		pts_micros: u64,
	) -> PyResult<()> {
		self
			.inner
			.write_audio(&interleaved, sample_rate, channel_count, pts_micros)
			.map_err(python_error)
	}

	/// Append one capture chunk through `write_audio`.
	pub fn write_audio_chunk(
		&mut self,
		chunk: &crate::audio::sessions::PythonAudioCaptureChunk,
	) -> PyResult<()> {
		self
			.inner
			.write_audio_chunk(&chunk.inner)
			.map_err(python_error)
	}

	/// Drain the encoder, flush pending audio, and finalize the container.
	pub fn finalize(&mut self) -> PyResult<()> {
		self.inner.finalize().map_err(python_error)
	}

	/// Close the recorder without finalizing remaining state.
	pub fn close(&mut self) {
		self.inner.close();
	}

	/// Return whether the recorder still accepts frames.
	#[getter]
	pub fn is_open(&self) -> bool {
		self.inner.is_open()
	}

	/// Number of video frames written so far.
	#[getter]
	pub fn frame_count(&self) -> u64 {
		self.inner.frame_count()
	}

	pub fn __repr__(&self) -> String {
		format!(
			"VideoRecorder(open={}, frames={})",
			self.inner.is_open(),
			self.inner.frame_count(),
		)
	}
}

// ── Capability queries ──────────────────────────────────────────────────────

/// Query the selected device's video queue and codec admission summary.
#[pyfunction]
pub fn query_device_capabilities(
	runtime: Option<&PythonRuntime>,
) -> PyResult<PythonVideoDeviceCapabilities> {
	let runtime = PythonRuntime::resolve(runtime)?;
	oa::video::query_device_capabilities(&runtime)
		.map(|inner| PythonVideoDeviceCapabilities { inner })
		.map_err(python_error)
}

/// Query the selected device's exact limits for one decoder profile.
#[pyfunction]
pub fn query_decode_capabilities(
	profile: &PythonVideoDecodeProfile,
	runtime: Option<&PythonRuntime>,
) -> PyResult<PythonVideoDecodeCapabilities> {
	let runtime = PythonRuntime::resolve(runtime)?;
	oa::video::query_decode_capabilities(&runtime, profile.inner)
		.map(|inner| PythonVideoDecodeCapabilities { inner })
		.map_err(python_error)
}

/// Enumerate decoded-output and DPB image formats for one exact profile.
#[pyfunction]
pub fn query_decode_formats(
	profile: &PythonVideoDecodeProfile,
	runtime: Option<&PythonRuntime>,
) -> PyResult<PythonVideoDecodeFormats> {
	let runtime = PythonRuntime::resolve(runtime)?;
	oa::video::query_decode_formats(&runtime, profile.inner)
		.map(|inner| PythonVideoDecodeFormats { inner })
		.map_err(python_error)
}

/// Query exact H.264 or H.265 encoder limits without creating a session.
#[pyfunction]
pub fn query_encode_capabilities(
	profile: &PythonVideoEncodeProfile,
	runtime: Option<&PythonRuntime>,
) -> PyResult<PythonVideoEncodeCapabilities> {
	let runtime = PythonRuntime::resolve(runtime)?;
	oa::video::query_encode_capabilities(&runtime, profile.inner)
		.map(|inner| PythonVideoEncodeCapabilities { inner })
		.map_err(python_error)
}

/// Enumerate encode-input and DPB formats for one exact profile.
#[pyfunction]
pub fn query_encode_formats(
	profile: &PythonVideoEncodeProfile,
	runtime: Option<&PythonRuntime>,
) -> PyResult<PythonVideoEncodeFormats> {
	let runtime = PythonRuntime::resolve(runtime)?;
	oa::video::query_encode_formats(&runtime, profile.inner)
		.map(|inner| PythonVideoEncodeFormats { inner })
		.map_err(python_error)
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
	module.add_class::<PythonVideoChromaSubsampling>()?;
	module.add_class::<PythonVideoComponentBitDepth>()?;
	module.add_class::<PythonH264Profile>()?;
	module.add_class::<PythonH264PictureLayout>()?;
	module.add_class::<PythonH265Profile>()?;
	module.add_class::<PythonAv1Profile>()?;
	module.add_class::<PythonVp9Profile>()?;
	module.add_class::<PythonVideoRateControl>()?;
	module.add_class::<PythonVideoPixelFormat>()?;
	module.add_class::<PythonVideoExtent>()?;
	module.add_class::<PythonVideoDecodeProfile>()?;
	module.add_class::<PythonVideoEncodeProfile>()?;
	module.add_class::<PythonVideoImageFormat>()?;
	module.add_class::<PythonVideoDecodeLevel>()?;
	module.add_class::<PythonVideoDeviceCapabilities>()?;
	module.add_class::<PythonVideoDecodeCapabilities>()?;
	module.add_class::<PythonVideoEncodeCodecCapabilities>()?;
	module.add_class::<PythonVideoEncodeCapabilities>()?;
	module.add_class::<PythonVideoDecodeFormats>()?;
	module.add_class::<PythonVideoEncodeFormats>()?;
	module.add_class::<PythonVideoEncodeConfig>()?;
	module.add_class::<PythonVideoEncodePlan>()?;
	module.add_class::<PythonVideoEncoder>()?;
	module.add_class::<PythonVideoRecorderConfig>()?;
	module.add_class::<PythonVideoRecorder>()?;
	module.add_function(wrap_pyfunction!(query_device_capabilities, module)?)?;
	module.add_function(wrap_pyfunction!(query_decode_capabilities, module)?)?;
	module.add_function(wrap_pyfunction!(query_decode_formats, module)?)?;
	module.add_function(wrap_pyfunction!(query_encode_capabilities, module)?)?;
	module.add_function(wrap_pyfunction!(query_encode_formats, module)?)?;
	Ok(())
}
