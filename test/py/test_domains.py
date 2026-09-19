import unittest

import oa


class DomainBindingTest(unittest.TestCase):
	@classmethod
	def setUpClass(cls) -> None:
		cls.engine = oa.Engine()

	def test_root_types_are_identity_aliases(self) -> None:
		self.assertIs(oa.runtime.Engine, oa.Engine)
		self.assertIs(oa.core.Matrix, oa.Matrix)
		self.assertIs(oa.audio.Audio, oa.Audio)
		self.assertIs(oa.image.Image, oa.Image)

	def test_native_types_report_their_owning_module(self) -> None:
		for value_type in (oa.Engine, oa.Event, oa.Matrix, oa.Image, oa.Audio):
			with self.subTest(value_type=value_type.__name__):
				self.assertEqual(value_type.__module__, "oa._native")

	def test_audio_value_and_dsp_round_trip(self) -> None:
		audio = oa.Audio.from_planar_f32(
			self.engine, [0.25, -0.5, 0.75, -1.0], 1, 16_000, "mono"
		)
		amplified = oa.audio.gain(audio, 0.0)
		self.assertEqual(amplified.channels, 1)
		self.assertEqual(amplified.samples, 4)
		self.assertEqual(amplified.layout, "mono")
		self.assertEqual(amplified.as_matrix().read_f32(), [0.25, -0.5, 0.75, -1.0])
		self.assertIsInstance(oa.audio.encode_wav_f32(amplified), bytes)

	def test_image_value_and_geometry(self) -> None:
		pixels = oa.Matrix.from_f32(self.engine, [3, 2, 2], [float(i) for i in range(12)])
		image = oa.Image.from_matrix(pixels, "chw", "rgb")
		flipped = oa.image.flip(image, True, False)
		self.assertEqual((image.width, image.height, image.channels), (2, 2, 3))
		self.assertEqual((flipped.layout, flipped.format, flipped.dtype), ("chw", "rgb", "f32"))
		self.assertEqual(
			flipped.as_matrix().read_f32(),
			[1.0, 0.0, 3.0, 2.0, 5.0, 4.0, 7.0, 6.0, 9.0, 8.0, 11.0, 10.0],
		)

	def test_ml_module_uses_same_matrix_type(self) -> None:
		values = oa.Matrix.from_f32(self.engine, [3], [-1.0, 0.0, 2.0])
		output = oa.ml.matrix.relu(values)
		self.assertIsInstance(output, oa.Matrix)
		self.assertEqual(output.read_f32(), [0.0, 0.0, 2.0])


if __name__ == "__main__":
	unittest.main()
