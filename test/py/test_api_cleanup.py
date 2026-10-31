"""Generated API parity and independent oracles for the cleanup checkpoint."""

import ast
import inspect
import unittest
from pathlib import Path

import oa


class SignatureParityTest(unittest.TestCase):
	def test_every_generated_value_method_matches_its_free_signature(self):
		stub = Path(oa.__file__).parent / "_gen/native.pyi"
		for node in ast.parse(stub.read_text()).body:
			if not isinstance(node, ast.FunctionDef):
				continue
			domain, _, operation = node.name.partition("_")
			if domain not in {"matrix", "audio", "image"}:
				continue
			receiver = node.args.args[0].annotation if node.args.args else None
			if not (isinstance(receiver, ast.Name) and receiver.id == domain.title()):
				continue
			with self.subTest(operation=node.name):
				free = getattr(getattr(oa, domain), operation)
				method = getattr(getattr(oa, domain.title()), operation)
				free_parameters = list(inspect.signature(free).parameters.values())
				method_parameters = list(inspect.signature(method).parameters.values())
				self.assertEqual(free_parameters[1:], method_parameters[1:])
				self.assertEqual([p.name for p in free_parameters], [arg.arg for arg in node.args.args])
				defaults = [ast.literal_eval(default) for default in node.args.defaults]
				if defaults:
					self.assertEqual([p.default for p in free_parameters[-len(defaults):]], defaults)

	def test_path_transformation_and_constructor_spellings(self):
		self.assertEqual(str(oa.Path("sample.wav").with_extension("flac")), "sample.flac")
		self.assertFalse(hasattr(oa.Path, "new_with_extension"))
		for value, old in ((oa.Matrix, "from_f32"), (oa.Audio, "from_planar_f32")):
			self.assertTrue(hasattr(value, "new_" + old))
			self.assertFalse(hasattr(value, old))


class GeneratedOperationOracleTest(unittest.TestCase):
	@classmethod
	def setUpClass(cls):
		cls.engine = oa.Engine()

	def test_matrix_named_keyword_methods_and_operators(self):
		left = oa.Matrix.new_from_f32(self.engine, [3], [1.0, -2.0, 3.0])
		right = oa.Matrix.new_from_f32(self.engine, [3], [4.0, 5.0, -6.0])
		for result in (oa.matrix.add(left=left, right=right), left.add(right=right), left + right):
			self.assertEqual(result.read_f32(), [5.0, 3.0, -3.0])
		for result in (oa.matrix.pow(input=left, exponent=2.0), left.pow(exponent=2.0)):
			self.assertEqual(result.read_f32(), [1.0, 4.0, 9.0])
		self.assertEqual((-left).read_f32(), [-1.0, 2.0, -3.0])
		self.assertEqual(abs(left).read_f32(), [1.0, 2.0, 3.0])

	def test_nt_matmul_preserves_the_weight_layout(self):
		left = oa.Matrix.new_from_f32(self.engine, [1, 3], [1.0, 2.0, 3.0])
		right = oa.Matrix.new_from_f32(self.engine, [2, 3], [1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
		for result in (oa.matrix.mat_mul_nt(left, right), left.mat_mul_nt(right=right), left @ right):
			self.assertEqual(result.shape, [1, 2])
			self.assertEqual(result.read_f32(), [4.0, 2.0])

	def test_flattened_audio_configuration(self):
		audio = oa.Audio.new_from_planar_f32(self.engine, [0.25, -0.5, 0.75, -1.0], 1, 8000, "mono")
		for result in (oa.audio.normalize(audio, target_db=0.0), audio.normalize(target_db=0.0)):
			self.assertEqual(result.as_matrix().read_f32(), [0.25, -0.5, 0.75, -1.0])
		for result in (
			audio.biquad(b0=1.0, b1=0.0, b2=0.0, a1=0.0, a2=0.0),
			audio.sos_filter(sections=[[1.0, 0.0, 0.0, 0.0, 0.0]]),
		):
			self.assertEqual(result.as_matrix().read_f32(), [0.25, -0.5, 0.75, -1.0])
		for call in (lambda: oa.audio.stft(audio, window="unknown"), lambda: audio.stft(window="unknown")):
			with self.assertRaisesRegex(ValueError, "unknown STFT window"):
				call()

	def test_flattened_image_configuration_and_defaults(self):
		pixels = oa.Matrix.new_from_f32(self.engine, [3, 1, 1], [0.5, 0.75, 1.0])
		image = oa.Image.new_from_matrix(pixels, "chw", "rgb")
		for result in (
			oa.image.normalize(image, mean=[0.25, 0.25, 0.5], std=[0.5, 0.5, 0.5]),
			image.normalize(mean=[0.25, 0.25, 0.5], std=[0.5, 0.5, 0.5]),
		):
			self.assertEqual(result.as_matrix().read_f32(), [0.5, 1.0, 1.0])
		self.assertEqual(image.invert().as_matrix().read_f32(), [0.5, 0.25, 0.0])
		resized = image.resize(target_width=2, target_height=2, interpolation="nearest")
		self.assertEqual((resized.width, resized.height), (2, 2))
		for call in (lambda: oa.image.gaussian_blur(image, sigma=-1.0), lambda: image.gaussian_blur(sigma=-1.0)):
			with self.assertRaises(ValueError):
				call()
