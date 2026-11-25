"""Tests for the oa.video Python binding surface.

Coverage strategy
─────────────────
* Enum types (VideoColorMatrix, VideoColorRange, VideoCodec, VideoContainerKind):
  verify all members exist and are distinct.
* VideoColorInfo: constructor, staticmethod, getters, repr.
* VideoFrameTiming: constructor, getters (presentation_us, duration_us), repr.
* VideoFrame: from_image smoke test, color/timing accessors.
* VideoPacket / VideoContainerInfo / VideoDemuxer: require a real MP4 file —
  skipped on hosts where no video fixture is available.
* VideoMuxerAudioConfig, VideoMuxerConfig, EncodedVideoPacket:
  pure data objects — exercised without a GPU.
* VideoMuxer write round-trip is skipped (requires a real codec config).
* VideoDecoder, VideoPlayer: GPU-gated.
* Root + module identity aliases for all new types.
"""

import os
import tempfile
import unittest

import oa

# ── helpers ───────────────────────────────────────────────────────────────────


def _has_gpu() -> bool:
	try:
		oa.Runtime()
		return True
	except Exception:
		return False


HAS_GPU = _has_gpu()
require_gpu = unittest.skipUnless(HAS_GPU, "requires a hardware Vulkan compute device")


# ── VideoColorMatrix ──────────────────────────────────────────────────────────


class VideoColorMatrixTest(unittest.TestCase):
	MEMBERS = ["Unspecified", "Bt601", "Bt709", "Bt2020"]

	def test_all_members_exist(self) -> None:
		for name in self.MEMBERS:
			with self.subTest(name=name):
				self.assertTrue(hasattr(oa.VideoColorMatrix, name))

	def test_members_are_distinct(self) -> None:
		m = oa.VideoColorMatrix
		self.assertNotEqual(m.Unspecified, m.Bt601)
		self.assertNotEqual(m.Bt601, m.Bt709)
		self.assertNotEqual(m.Bt709, m.Bt2020)

	def test_equal_to_themselves(self) -> None:
		m = oa.VideoColorMatrix
		self.assertEqual(m.Bt709, m.Bt709)

	def test_identity_alias(self) -> None:
		self.assertIs(oa.VideoColorMatrix, oa.video.VideoColorMatrix)


# ── VideoColorRange ───────────────────────────────────────────────────────────


class VideoColorRangeTest(unittest.TestCase):
	MEMBERS = ["Unspecified", "Limited", "Full"]

	def test_all_members_exist(self) -> None:
		for name in self.MEMBERS:
			with self.subTest(name=name):
				self.assertTrue(hasattr(oa.VideoColorRange, name))

	def test_members_are_distinct(self) -> None:
		r = oa.VideoColorRange
		self.assertNotEqual(r.Unspecified, r.Limited)
		self.assertNotEqual(r.Limited, r.Full)

	def test_identity_alias(self) -> None:
		self.assertIs(oa.VideoColorRange, oa.video.VideoColorRange)


# ── VideoColorInfo ────────────────────────────────────────────────────────────


class VideoColorInfoTest(unittest.TestCase):
	def test_constructor_round_trips_fields(self) -> None:
		info = oa.VideoColorInfo(
			matrix=oa.VideoColorMatrix.Bt709,
			range=oa.VideoColorRange.Full,
		)
		self.assertEqual(info.matrix, oa.VideoColorMatrix.Bt709)
		self.assertEqual(info.range, oa.VideoColorRange.Full)
		self.assertIn("VideoColorInfo", repr(info))

	def test_unspecified_factory(self) -> None:
		info = oa.VideoColorInfo.unspecified()
		self.assertEqual(info.matrix, oa.VideoColorMatrix.Unspecified)
		self.assertEqual(info.range, oa.VideoColorRange.Unspecified)

	def test_default_constructor(self) -> None:
		info = oa.VideoColorInfo()
		self.assertEqual(info.matrix, oa.VideoColorMatrix.Unspecified)

	def test_identity_alias(self) -> None:
		self.assertIs(oa.VideoColorInfo, oa.video.VideoColorInfo)


# ── VideoFrameTiming ──────────────────────────────────────────────────────────


class VideoFrameTimingTest(unittest.TestCase):
	def test_presentation_us_round_trips(self) -> None:
		timing = oa.VideoFrameTiming(presentation_us=1_000_000)
		self.assertEqual(timing.presentation_us, 1_000_000)

	def test_duration_us_is_none_when_zero(self) -> None:
		timing = oa.VideoFrameTiming(presentation_us=0, duration_us=0)
		self.assertIsNone(timing.duration_us)

	def test_duration_us_set_when_nonzero(self) -> None:
		timing = oa.VideoFrameTiming(presentation_us=500, duration_us=33_333)
		self.assertEqual(timing.duration_us, 33_333)

	def test_repr_includes_class_name(self) -> None:
		self.assertIn("VideoFrameTiming", repr(oa.VideoFrameTiming(presentation_us=0)))

	def test_identity_alias(self) -> None:
		self.assertIs(oa.VideoFrameTiming, oa.video.VideoFrameTiming)


# ── VideoCodec ────────────────────────────────────────────────────────────────


class VideoCodecTest(unittest.TestCase):
	MEMBERS = ["H264", "H265", "Av1", "Vp9"]

	def test_all_members_exist(self) -> None:
		for name in self.MEMBERS:
			with self.subTest(codec=name):
				self.assertTrue(hasattr(oa.VideoCodec, name))

	def test_members_are_distinct(self) -> None:
		c = oa.VideoCodec
		self.assertNotEqual(c.H264, c.H265)
		self.assertNotEqual(c.H265, c.Av1)
		self.assertNotEqual(c.Av1, c.Vp9)

	def test_identity_alias(self) -> None:
		self.assertIs(oa.VideoCodec, oa.video.VideoCodec)


# ── VideoContainerKind ────────────────────────────────────────────────────────


class VideoContainerKindTest(unittest.TestCase):
	def test_mp4_member_exists(self) -> None:
		self.assertTrue(hasattr(oa.VideoContainerKind, "Mp4"))

	def test_equal_to_itself(self) -> None:
		self.assertEqual(oa.VideoContainerKind.Mp4, oa.VideoContainerKind.Mp4)

	def test_identity_alias(self) -> None:
		self.assertIs(oa.VideoContainerKind, oa.video.VideoContainerKind)


# ── VideoFrame (from_image) ───────────────────────────────────────────────────


@require_gpu
class VideoFrameFromImageTest(unittest.TestCase):
	@classmethod
	def setUpClass(cls) -> None:
		cls.runtime = oa.Runtime()

	def test_from_image_produces_correct_dimensions(self) -> None:
		m = oa.Matrix.new_from_f32( [1, 3, 4, 4], [0.5] * 48, runtime=self.runtime)
		image = oa.Image.new_from_matrix(m, "nchw", "rgb")
		frame = oa.VideoFrame.new_from_image(image, presentation_us=1_000)
		self.assertEqual(frame.width(), 4)
		self.assertEqual(frame.height(), 4)
		self.assertEqual(frame.presentation_us(), 1_000)
		self.assertIn("VideoFrame", repr(frame))

	def test_color_returns_video_color_info(self) -> None:
		m = oa.Matrix.new_from_f32( [1, 3, 4, 4], [0.5] * 48, runtime=self.runtime)
		image = oa.Image.new_from_matrix(m, "nchw", "rgb")
		frame = oa.VideoFrame.new_from_image(image)
		color = frame.color()
		self.assertIsInstance(color, oa.VideoColorInfo)
		self.assertEqual(color.matrix, oa.VideoColorMatrix.Unspecified)

	def test_timing_returns_video_frame_timing(self) -> None:
		m = oa.Matrix.new_from_f32( [1, 3, 4, 4], [0.5] * 48, runtime=self.runtime)
		image = oa.Image.new_from_matrix(m, "nchw", "rgb")
		frame = oa.VideoFrame.new_from_image(image, presentation_us=2_500)
		timing = frame.timing()
		self.assertIsInstance(timing, oa.VideoFrameTiming)
		self.assertEqual(timing.presentation_us, 2_500)

	def test_as_image_returns_backing_image(self) -> None:
		m = oa.Matrix.new_from_f32( [1, 3, 4, 4], [0.5] * 48, runtime=self.runtime)
		image = oa.Image.new_from_matrix(m, "nchw", "rgb")
		frame = oa.VideoFrame.new_from_image(image)
		result = frame.as_image()
		self.assertIsNotNone(result)

	def test_identity_alias(self) -> None:
		self.assertIs(oa.VideoFrame, oa.video.VideoFrame)


# ── EncodedVideoPacket ────────────────────────────────────────────────────────


class EncodedVideoPacketTest(unittest.TestCase):
	def test_round_trips_fields(self) -> None:
		data = b"\x00\x00\x00\x01\x65" + b"\x00" * 10  # minimal fake NAL unit
		pkt = oa.EncodedVideoPacket(
			bitstream=list(data),
			presentation_timestamp_micros=5_000,
			keyframe=True,
		)
		self.assertEqual(bytes(pkt.bitstream()), data)
		self.assertEqual(pkt.presentation_timestamp_micros, 5_000)
		self.assertEqual(pkt.decode_timestamp_micros, 5_000)
		self.assertTrue(pkt.is_keyframe)
		self.assertIn("EncodedVideoPacket", repr(pkt))

	def test_with_timestamps_allows_distinct_pts_dts(self) -> None:
		data = b"\x00\x00\x00\x01\x41" + b"\x00" * 10
		pkt = oa.EncodedVideoPacket.new_with_timestamps(
			bitstream=list(data),
			presentation_timestamp_micros=10_000,
			decode_timestamp_micros=8_000,
			keyframe=False,
		)
		self.assertEqual(pkt.presentation_timestamp_micros, 10_000)
		self.assertEqual(pkt.decode_timestamp_micros, 8_000)
		self.assertFalse(pkt.is_keyframe)

	def test_rejects_empty_bitstream(self) -> None:
		with self.assertRaises(Exception):
			oa.EncodedVideoPacket(bitstream=[], presentation_timestamp_micros=0)

	def test_identity_alias(self) -> None:
		self.assertIs(oa.EncodedVideoPacket, oa.video.EncodedVideoPacket)


# ── VideoMuxerAudioConfig ─────────────────────────────────────────────────────


class VideoMuxerAudioConfigTest(unittest.TestCase):
	def test_round_trips_fields(self) -> None:
		cfg = oa.VideoMuxerAudioConfig(
			sample_rate=44_100, channel_count=1, priming_frames=128
		)
		self.assertEqual(cfg.sample_rate, 44_100)
		self.assertEqual(cfg.channel_count, 1)
		self.assertEqual(cfg.priming_frames, 128)
		self.assertIn("VideoMuxerAudioConfig", repr(cfg))

	def test_defaults(self) -> None:
		cfg = oa.VideoMuxerAudioConfig()
		self.assertEqual(cfg.sample_rate, 48_000)
		self.assertEqual(cfg.channel_count, 2)
		self.assertEqual(cfg.priming_frames, 0)

	def test_identity_alias(self) -> None:
		self.assertIs(oa.VideoMuxerAudioConfig, oa.video.VideoMuxerAudioConfig)


# ── VideoMuxerConfig ──────────────────────────────────────────────────────────


class VideoMuxerConfigTest(unittest.TestCase):
	def test_round_trips_required_fields(self) -> None:
		cfg = oa.VideoMuxerConfig(
			codec=oa.VideoCodec.H264, width=1280, height=720, frame_rate=60
		)
		self.assertEqual(cfg.codec, oa.VideoCodec.H264)
		self.assertEqual(cfg.width, 1280)
		self.assertEqual(cfg.height, 720)
		self.assertEqual(cfg.frame_rate, 60)
		self.assertIn("VideoMuxerConfig", repr(cfg))

	def test_with_h265_codec(self) -> None:
		cfg = oa.VideoMuxerConfig(codec=oa.VideoCodec.H265, width=640, height=480)
		self.assertEqual(cfg.codec, oa.VideoCodec.H265)

	def test_rejects_zero_extent(self) -> None:
		with self.assertRaises(Exception):
			oa.VideoMuxerConfig(codec=oa.VideoCodec.H264, width=0, height=480)

	def test_identity_alias(self) -> None:
		self.assertIs(oa.VideoMuxerConfig, oa.video.VideoMuxerConfig)


# ── VideoMuxer (construction only, finalize via temp file) ────────────────────


class VideoMuxerConstructionTest(unittest.TestCase):
	def test_constructs_and_creates_file(self) -> None:
		with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
			path = f.name
		try:
			cfg = oa.VideoMuxerConfig(
				codec=oa.VideoCodec.H264, width=640, height=480, frame_rate=30
			)
			muxer = oa.VideoMuxer(path, cfg)
			# File is created and the fixed MP4 header is written at construction.
			self.assertGreater(os.path.getsize(path), 0)
			self.assertEqual(repr(muxer), "VideoMuxer")
		finally:
			try:
				os.unlink(path)
			except OSError:
				pass

	def test_identity_alias(self) -> None:
		self.assertIs(oa.VideoMuxer, oa.video.VideoMuxer)


# ── VideoPlayerConfig ─────────────────────────────────────────────────────────


class VideoPlayerConfigTest(unittest.TestCase):
	def test_round_trips_fields(self) -> None:
		cfg = oa.VideoPlayerConfig(
			loop_playback=False,
			start_playing=False,
			frame_rate_override=24.0,
			presentation_cache_frames=16,
		)
		self.assertFalse(cfg.loop_playback)
		self.assertFalse(cfg.start_playing)
		self.assertAlmostEqual(cfg.frame_rate_override, 24.0)
		self.assertEqual(cfg.presentation_cache_frames, 16)
		self.assertIn("VideoPlayerConfig", repr(cfg))

	def test_defaults(self) -> None:
		cfg = oa.VideoPlayerConfig()
		self.assertTrue(cfg.loop_playback)
		self.assertTrue(cfg.start_playing)
		self.assertIsNone(cfg.frame_rate_override)

	def test_identity_alias(self) -> None:
		self.assertIs(oa.VideoPlayerConfig, oa.video.VideoPlayerConfig)


# ── VideoPlayerStats (type presence) ─────────────────────────────────────────


class VideoPlayerStatsPresenceTest(unittest.TestCase):
	def test_type_is_accessible(self) -> None:
		self.assertTrue(hasattr(oa, "VideoPlayerStats"))

	def test_identity_alias(self) -> None:
		self.assertIs(oa.VideoPlayerStats, oa.video.VideoPlayerStats)

	def test_expected_attribute_names(self) -> None:
		# Verify the getters exist on the class without instantiating.
		for attr in ("presented_frames", "decoded_packets", "seek_resets", "loop_restarts",
					  "presentation_cache_hits", "presentation_cache_misses",
					  "presentation_cache_resident"):
			self.assertTrue(
				hasattr(oa.VideoPlayerStats, attr),
				f"VideoPlayerStats.{attr} is missing",
			)


# ── VideoDemuxer, VideoDecoder, VideoPlayer (presence only) ──────────────────


class VideoSessionTypePresenceTest(unittest.TestCase):
	def test_video_demuxer_is_accessible(self) -> None:
		self.assertIs(oa.VideoDemuxer, oa.video.VideoDemuxer)

	def test_video_decoder_is_accessible(self) -> None:
		self.assertIs(oa.VideoDecoder, oa.video.VideoDecoder)

	def test_video_player_is_accessible(self) -> None:
		self.assertIs(oa.VideoPlayer, oa.video.VideoPlayer)

	def test_video_container_info_is_accessible(self) -> None:
		self.assertIs(oa.VideoContainerInfo, oa.video.VideoContainerInfo)

	def test_video_packet_is_accessible(self) -> None:
		self.assertIs(oa.VideoPacket, oa.video.VideoPacket)


# ── Encode surface presence ─────────────────────────────────────────────────


class VideoEncodeTypePresenceTest(unittest.TestCase):
	TYPES = [
		"VideoEncoder",
		"VideoEncodeConfig",
		"VideoEncodePlan",
		"VideoEncodeProfile",
		"VideoRateControl",
		"VideoRecorder",
		"VideoRecorderConfig",
		"VideoExtent",
		"VideoImageFormat",
		"VideoPixelFormat",
		"VideoDeviceCapabilities",
		"VideoDecodeCapabilities",
		"VideoDecodeFormats",
		"VideoDecodeLevel",
		"VideoDecodeProfile",
		"VideoEncodeCapabilities",
		"VideoEncodeCodecCapabilities",
		"VideoEncodeFormats",
		"H264Profile",
		"H264PictureLayout",
		"H265Profile",
		"Av1Profile",
		"Vp9Profile",
		"VideoChromaSubsampling",
		"VideoComponentBitDepth",
	]

	FUNCTIONS = [
		"query_device_capabilities",
		"query_decode_capabilities",
		"query_decode_formats",
		"query_encode_capabilities",
		"query_encode_formats",
	]

	def test_types_are_accessible(self) -> None:
		for name in self.TYPES:
			with self.subTest(name=name):
				self.assertIs(getattr(oa, name), getattr(oa.video, name))

	def test_query_functions_exist(self) -> None:
		for name in self.FUNCTIONS:
			with self.subTest(name=name):
				self.assertTrue(callable(getattr(oa.video, name)))


# ── Encode enums ────────────────────────────────────────────────────────────


class VideoEncodeEnumTest(unittest.TestCase):
	def test_rate_control_members(self) -> None:
		m = oa.VideoRateControl
		self.assertNotEqual(m.ConstantQp, m.Cbr)
		self.assertNotEqual(m.Cbr, m.Vbr)

	def test_bit_depth_members(self) -> None:
		m = oa.VideoComponentBitDepth
		self.assertNotEqual(m.Eight, m.Ten)
		self.assertNotEqual(m.Ten, m.Twelve)

	def test_chroma_members(self) -> None:
		m = oa.VideoChromaSubsampling
		self.assertNotEqual(m.Yuv420, m.Yuv422)
		self.assertNotEqual(m.Yuv422, m.Yuv444)

	def test_codec_profile_members(self) -> None:
		self.assertNotEqual(oa.H264Profile.High, oa.H264Profile.Baseline)
		self.assertNotEqual(oa.H265Profile.Main, oa.H265Profile.Main10)
		self.assertNotEqual(oa.Av1Profile.Main, oa.Av1Profile.Professional)
		self.assertNotEqual(oa.Vp9Profile.Profile0, oa.Vp9Profile.Profile3)


# ── Profile and config construction ─────────────────────────────────────────


class VideoProfileConfigTest(unittest.TestCase):
	def test_encode_profile_presets(self) -> None:
		p8 = oa.VideoEncodeProfile.h264_high_420_8bit()
		self.assertEqual(p8.codec(), oa.VideoCodec.H264)
		p265 = oa.VideoEncodeProfile.h265_main_420_8bit()
		self.assertEqual(p265.codec(), oa.VideoCodec.H265)

	def test_encode_profile_explicit(self) -> None:
		p = oa.VideoEncodeProfile.h265(
			oa.H265Profile.Main10,
			oa.VideoChromaSubsampling.Yuv420,
			oa.VideoComponentBitDepth.Ten,
			oa.VideoComponentBitDepth.Ten,
		)
		self.assertEqual(p.codec(), oa.VideoCodec.H265)

	def test_decode_profile_presets(self) -> None:
		self.assertEqual(
			oa.VideoDecodeProfile.h264_420_8bit(oa.H264Profile.High).codec(),
			oa.VideoCodec.H264,
		)
		self.assertEqual(
			oa.VideoDecodeProfile.h265_420(
				oa.H265Profile.Main10, oa.VideoComponentBitDepth.Ten
			).codec(),
			oa.VideoCodec.H265,
		)
		self.assertEqual(
			oa.VideoDecodeProfile.av1_420(
				oa.Av1Profile.Main, oa.VideoComponentBitDepth.Ten, False
			).codec(),
			oa.VideoCodec.Av1,
		)
		self.assertEqual(
			oa.VideoDecodeProfile.vp9_420(
				oa.Vp9Profile.Profile0, oa.VideoComponentBitDepth.Eight
			).codec(),
			oa.VideoCodec.Vp9,
		)

	def test_encode_config_defaults(self) -> None:
		cfg = oa.VideoEncodeConfig(
			oa.VideoEncodeProfile.h265_main_420_8bit(), 1280, 720
		)
		self.assertEqual(cfg.width, 1280)
		self.assertEqual(cfg.height, 720)
		self.assertEqual(cfg.frame_rate, 30)
		self.assertEqual(cfg.gop_size, 30)

	def test_extent_round_trip(self) -> None:
		ext = oa.VideoExtent(1280, 720)
		self.assertEqual((ext.width, ext.height), (1280, 720))


# ── Encoder session (GPU) ───────────────────────────────────────────────────


@require_gpu
class VideoEncoderSessionTest(unittest.TestCase):
	@classmethod
	def setUpClass(cls) -> None:
		cls.runtime = oa.Runtime()

	def setUp(self) -> None:
		caps = oa.video.query_device_capabilities(runtime=self.runtime)
		if not caps.supports_h265_encode or not caps.encoder_sessions_available:
			self.skipTest("device does not admit H.265 encode sessions")

	def test_capability_and_encoder_round_trip(self) -> None:
		profile = oa.VideoEncodeProfile.h265_main_420_8bit()
		caps = oa.video.query_encode_capabilities( profile, runtime=self.runtime)
		config = oa.VideoEncodeConfig(profile, 640, 360, gop_size=4)
		config.validate_for_device(caps)
		encoder = oa.video.VideoEncoder( config, runtime=self.runtime)
		try:
			self.assertTrue(encoder.is_open)
			frame = bytes(640 * 360 * 4)
			packet = encoder.encode(frame, 0)
			self.assertGreater(len(packet.bitstream()), 0)
			self.assertIsInstance(encoder.flush(), list)
		finally:
			encoder.close()
		self.assertFalse(encoder.is_open)


if __name__ == "__main__":
	unittest.main()
