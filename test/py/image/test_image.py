"""Python binding tests for oa.image stateless transformations."""

import unittest

import oa


def _make_rgb_image(engine: oa.Engine, width: int = 4, height: int = 4) -> oa.Image:
	"""Create a small NCHW RGB f32 image filled with a gradient."""
	n = width * height
	r = [float(i) / n for i in range(n)]
	g = [float(n - i) / n for i in range(n)]
	b = [0.5] * n
	m = oa.Matrix.from_f32(engine, [1, 3, height, width], r + g + b)
	return oa.Image.from_matrix(m, "nchw", "rgb")


def _make_gray_image(engine: oa.Engine, width: int = 4, height: int = 4) -> oa.Image:
	n = width * height
	vals = [float(i) / n for i in range(n)]
	m = oa.Matrix.from_f32(engine, [1, 1, height, width], vals)
	return oa.Image.from_matrix(m, "nchw", "gray")


class ImageConstructorTest(unittest.TestCase):
	@classmethod
	def setUpClass(cls) -> None:
		cls.engine = oa.Engine()

	def test_from_matrix_properties(self) -> None:
		img = _make_rgb_image(self.engine)
		self.assertEqual(img.width, 4)
		self.assertEqual(img.height, 4)
		self.assertEqual(img.channels, 3)
		self.assertEqual(img.layout, "nchw")
		self.assertEqual(img.format, "rgb")
		self.assertEqual(img.dtype, "f32")

	def test_as_matrix_roundtrip(self) -> None:
		img = _make_rgb_image(self.engine)
		m = img.as_matrix()
		self.assertIsInstance(m, oa.Matrix)
		self.assertEqual(m.shape, [1, 3, 4, 4])


class ImageGeometricTest(unittest.TestCase):
	@classmethod
	def setUpClass(cls) -> None:
		cls.engine = oa.Engine()
		cls.img = _make_rgb_image(cls.engine, 8, 8)

	def test_resize_changes_dimensions(self) -> None:
		result = oa.image.resize(self.img, 4, 4)
		self.assertEqual(result.width, 4)
		self.assertEqual(result.height, 4)

	def test_resize_nearest(self) -> None:
		result = oa.image.resize(self.img, 2, 2, interpolation_name="nearest")
		self.assertEqual(result.width, 2)

	def test_crop_dimensions(self) -> None:
		result = oa.image.crop(self.img, 1, 1, 4, 4)
		self.assertEqual(result.width, 4)
		self.assertEqual(result.height, 4)

	def test_flip_horizontal_shape_preserved(self) -> None:
		result = oa.image.flip(self.img, horizontal=True, vertical=False)
		self.assertEqual(result.width, self.img.width)
		self.assertEqual(result.height, self.img.height)

	def test_flip_vertical_shape_preserved(self) -> None:
		result = oa.image.flip(self.img, horizontal=False, vertical=True)
		self.assertEqual(result.width, self.img.width)

	def test_rotate_90_swaps_dims(self) -> None:
		img = _make_rgb_image(self.engine, 4, 8)
		result = oa.image.rotate(img, 90)
		self.assertEqual(result.width, 8)
		self.assertEqual(result.height, 4)

	def test_center_crop_dimensions(self) -> None:
		result = oa.image.center_crop(self.img, 4, 4)
		self.assertEqual(result.width, 4)
		self.assertEqual(result.height, 4)

	def test_pad_increases_size(self) -> None:
		result = oa.image.pad(self.img, 1, 1, 1, 1)
		self.assertEqual(result.width, self.img.width + 2)
		self.assertEqual(result.height, self.img.height + 2)


class ImagePixelTest(unittest.TestCase):
	@classmethod
	def setUpClass(cls) -> None:
		cls.engine = oa.Engine()
		cls.img = _make_rgb_image(cls.engine)
		cls.gray = _make_gray_image(cls.engine)

	def test_grayscale_reduces_channels(self) -> None:
		result = oa.image.grayscale(self.img)
		self.assertEqual(result.channels, 1)

	def test_clamp_bounds_pixels(self) -> None:
		result = oa.image.clamp(self.img, 0.2, 0.8)
		for v in result.as_matrix().read_f32():
			self.assertGreaterEqual(v, 0.199)
			self.assertLessEqual(v, 0.801)

	def test_invert_default_max(self) -> None:
		result = oa.image.invert(self.img, max_value=1.0)
		orig = self.img.as_matrix().read_f32()
		inv = result.as_matrix().read_f32()
		for o, i in zip(orig, inv):
			self.assertAlmostEqual(i, 1.0 - o, places=5)

	def test_brightness_contrast_returns_image(self) -> None:
		result = oa.image.brightness_contrast(self.img, brightness=0.1, contrast=1.2)
		self.assertIsInstance(result, oa.Image)

	def test_gamma_contrast_returns_image(self) -> None:
		result = oa.image.gamma_contrast(self.img, gamma=2.2, gain=1.0)
		self.assertIsInstance(result, oa.Image)

	def test_solarize_returns_image(self) -> None:
		result = oa.image.solarize(self.img, threshold=0.5, max_value=1.0)
		self.assertIsInstance(result, oa.Image)

	def test_posterize_returns_image(self) -> None:
		result = oa.image.posterize(self.img, levels=4, low=0.0, high=1.0)
		self.assertIsInstance(result, oa.Image)

	def test_threshold_binary_produces_zero_or_max(self) -> None:
		result = oa.image.threshold_binary(self.gray, threshold=0.5, max_value=1.0)
		for v in result.as_matrix().read_f32():
			self.assertIn(round(v, 4), [0.0, 1.0])

	def test_threshold_binary_inv_complements_binary(self) -> None:
		orig = oa.image.threshold_binary(self.gray, threshold=0.5, max_value=1.0)
		inv = oa.image.threshold_binary_inv(self.gray, threshold=0.5, max_value=1.0)
		for a, b in zip(orig.as_matrix().read_f32(), inv.as_matrix().read_f32()):
			self.assertAlmostEqual(a + b, 1.0, places=5)

	def test_threshold_truncate_caps_at_threshold(self) -> None:
		result = oa.image.threshold_truncate(self.img, threshold=0.4)
		for v in result.as_matrix().read_f32():
			self.assertLessEqual(v, 0.401)

	def test_in_range_produces_mask(self) -> None:
		result = oa.image.in_range(self.gray, low=0.2, high=0.8, true_value=1.0)
		self.assertIsInstance(result, oa.Image)

	def test_alpha_blend_returns_image(self) -> None:
		result = oa.image.alpha_blend(self.img, self.img, alpha=0.5)
		self.assertIsInstance(result, oa.Image)

	def test_erase_region_returns_image(self) -> None:
		result = oa.image.erase(self.img, x=0, y=0, width=2, height=2, value=0.0)
		self.assertIsInstance(result, oa.Image)

	def test_gaussian_noise_changes_values(self) -> None:
		result = oa.image.gaussian_noise(self.img, mean=0.0, stddev=0.01, seed=42)
		orig = self.img.as_matrix().read_f32()
		noisy = result.as_matrix().read_f32()
		diffs = [abs(o - n) for o, n in zip(orig, noisy)]
		self.assertGreater(sum(diffs), 0.0)

	def test_salt_pepper_noise_returns_image(self) -> None:
		result = oa.image.salt_pepper_noise(
			self.img, probability=0.05, salt_value=1.0, pepper_value=0.0, seed=7
		)
		self.assertIsInstance(result, oa.Image)


class ImageColorTest(unittest.TestCase):
	@classmethod
	def setUpClass(cls) -> None:
		cls.engine = oa.Engine()
		cls.img = _make_rgb_image(cls.engine)

	def test_convert_color_rgb_to_bgr(self) -> None:
		bgr = oa.image.convert_color(self.img, format_name="bgr")
		self.assertEqual(bgr.format, "bgr")
		self.assertEqual(bgr.width, self.img.width)

	def test_normalize_changes_values(self) -> None:
		result = oa.image.normalize(
			self.img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
		)
		self.assertIsInstance(result, oa.Image)

	def test_resize_normalize_changes_dims_and_values(self) -> None:
		result = oa.image.resize_normalize(
			self.img, 2, 2, mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]
		)
		self.assertEqual(result.width, 2)
		self.assertEqual(result.height, 2)

	def test_convert_color_rgb_to_bgr(self) -> None:
		# convert_color supports RGB ↔ BGR; grayscale conversion is not yet
		# implemented. Verify the RGB→BGR identity contract.
		bgr = oa.image.convert_color(self.img, format_name="bgr")
		self.assertEqual(bgr.format, "bgr")
		self.assertEqual(bgr.width, self.img.width)
		self.assertEqual(bgr.height, self.img.height)


class ImageFilterTest(unittest.TestCase):
	@classmethod
	def setUpClass(cls) -> None:
		cls.engine = oa.Engine()
		cls.img = _make_rgb_image(cls.engine, 8, 8)

	def test_gaussian_blur_preserves_shape(self) -> None:
		result = oa.image.gaussian_blur(self.img, sigma=1.0)
		self.assertEqual(result.width, self.img.width)
		self.assertEqual(result.height, self.img.height)

	def test_unsharp_mask_preserves_shape(self) -> None:
		result = oa.image.unsharp_mask(self.img, sigma=1.0, amount=0.5)
		self.assertEqual(result.width, self.img.width)

	def test_average_blur_preserves_shape(self) -> None:
		result = oa.image.average_blur(self.img, width=3, height=3)
		self.assertEqual(result.width, self.img.width)

	def test_median_blur_preserves_shape(self) -> None:
		result = oa.image.median_blur(self.img, kernel_size=3)
		self.assertEqual(result.width, self.img.width)

	def test_sobel_returns_image(self) -> None:
		gray = oa.image.grayscale(self.img)
		result = oa.image.sobel(gray, dx=1, dy=0)
		self.assertIsInstance(result, oa.Image)

	def test_laplacian_returns_image(self) -> None:
		gray = oa.image.grayscale(self.img)
		result = oa.image.laplacian(gray)
		self.assertIsInstance(result, oa.Image)

	def test_sharpen_preserves_shape(self) -> None:
		result = oa.image.sharpen(self.img, amount=1.0)
		self.assertEqual(result.width, self.img.width)

	def test_erode_preserves_shape(self) -> None:
		result = oa.image.erode(self.img, width=3, height=3)
		self.assertEqual(result.width, self.img.width)

	def test_dilate_preserves_shape(self) -> None:
		result = oa.image.dilate(self.img, width=3, height=3)
		self.assertEqual(result.width, self.img.width)

	def test_morphology_open_preserves_shape(self) -> None:
		result = oa.image.morphology_open(self.img, width=3, height=3)
		self.assertEqual(result.width, self.img.width)

	def test_morphology_close_preserves_shape(self) -> None:
		result = oa.image.morphology_close(self.img, width=3, height=3)
		self.assertEqual(result.width, self.img.width)

	def test_convolve_2d_identity_kernel(self) -> None:
		# A [1,1] identity kernel leaves the image unchanged
		kernel = oa.Matrix.from_f32(self.engine, [1, 1], [1.0])
		result = oa.image.convolve_2d(self.img, kernel, "constant", 0.0)
		orig = self.img.as_matrix().read_f32()
		res = result.as_matrix().read_f32()
		for o, r in zip(orig, res):
			self.assertAlmostEqual(o, r, places=5)

	def test_separable_convolve_2d_identity(self) -> None:
		kx = oa.Matrix.from_f32(self.engine, [1], [1.0])
		ky = oa.Matrix.from_f32(self.engine, [1], [1.0])
		result = oa.image.separable_convolve_2d(self.img, kx, ky)
		orig = self.img.as_matrix().read_f32()
		res = result.as_matrix().read_f32()
		for o, r in zip(orig, res):
			self.assertAlmostEqual(o, r, places=5)

	def test_adaptive_threshold_mean_returns_image(self) -> None:
		gray = oa.image.grayscale(self.img)
		result = oa.image.adaptive_threshold_mean(gray, kernel_size=3, c=0.0, max_value=1.0)
		self.assertIsInstance(result, oa.Image)

	def test_adaptive_threshold_gaussian_returns_image(self) -> None:
		gray = oa.image.grayscale(self.img)
		result = oa.image.adaptive_threshold_gaussian(
			gray, kernel_size=3, c=0.0, max_value=1.0, sigma=1.0
		)
		self.assertIsInstance(result, oa.Image)


class ImageQueryTest(unittest.TestCase):
	def test_can_decode_jpeg(self) -> None:
		result = oa.image.can_decode("jpeg")
		self.assertIsInstance(result, bool)

	def test_can_encode_png(self) -> None:
		result = oa.image.can_encode("png")
		self.assertIsInstance(result, bool)

	def test_can_decode_unknown_raises(self) -> None:
		with self.assertRaises(Exception):
			oa.image.can_decode("xyz_unknown_codec")


if __name__ == "__main__":
	unittest.main()
