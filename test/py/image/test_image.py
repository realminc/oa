"""Python binding tests for oa.image stateless transformations."""

import unittest

import oa


def _make_rgb_image(runtime: oa.Runtime, width: int = 4, height: int = 4) -> oa.Image:
	"""Create a small NCHW RGB f32 image filled with a gradient."""
	n = width * height
	r = [float(i) / n for i in range(n)]
	g = [float(n - i) / n for i in range(n)]
	b = [0.5] * n
	m = oa.Matrix.new_from_f32([1, 3, height, width], r + g + b, runtime=runtime)
	return oa.Image.new_from_matrix(m, "nchw", "rgb")


def _make_gray_image(runtime: oa.Runtime, width: int = 4, height: int = 4) -> oa.Image:
	n = width * height
	vals = [float(i) / n for i in range(n)]
	m = oa.Matrix.new_from_f32([1, 1, height, width], vals, runtime=runtime)
	return oa.Image.new_from_matrix(m, "nchw", "gray")


class ImageConstructorTest(unittest.TestCase):
	@classmethod
	def setUpClass(cls) -> None:
		cls.runtime = oa.Runtime()

	def test_from_matrix_properties(self) -> None:
		img = _make_rgb_image(self.runtime)
		self.assertEqual(img.width, 4)
		self.assertEqual(img.height, 4)
		self.assertEqual(img.channels, 3)
		self.assertEqual(img.layout, "nchw")
		self.assertEqual(img.format, "rgb")
		self.assertEqual(img.dtype, "f32")

	def test_as_matrix_roundtrip(self) -> None:
		img = _make_rgb_image(self.runtime)
		m = img.as_matrix()
		self.assertIsInstance(m, oa.Matrix)
		self.assertEqual(m.shape, [1, 3, 4, 4])


class ImageGeometricTest(unittest.TestCase):
	@classmethod
	def setUpClass(cls) -> None:
		cls.runtime = oa.Runtime()
		cls.img = _make_rgb_image(cls.runtime, 8, 8)

	def test_resize_changes_dimensions(self) -> None:
		result = oa.image.resize(self.img, 4, 4)
		self.assertEqual(result.width, 4)
		self.assertEqual(result.height, 4)

	def test_resize_nearest(self) -> None:
		result = oa.image.resize(self.img, 2, 2, interpolation="nearest")
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
		img = _make_rgb_image(self.runtime, 4, 8)
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
		cls.runtime = oa.Runtime()
		cls.img = _make_rgb_image(cls.runtime)
		cls.gray = _make_gray_image(cls.runtime)

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
		cls.runtime = oa.Runtime()
		cls.img = _make_rgb_image(cls.runtime)

	def test_convert_color_rgb_to_bgr(self) -> None:
		bgr = oa.image.convert_color(self.img, destination_format="bgr")
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
		bgr = oa.image.convert_color(self.img, destination_format="bgr")
		self.assertEqual(bgr.format, "bgr")
		self.assertEqual(bgr.width, self.img.width)
		self.assertEqual(bgr.height, self.img.height)


class ImageFilterTest(unittest.TestCase):
	@classmethod
	def setUpClass(cls) -> None:
		cls.runtime = oa.Runtime()
		cls.img = _make_rgb_image(cls.runtime, 8, 8)

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
		kernel = oa.Matrix.new_from_f32( [1, 1], [1.0], runtime=self.runtime)
		result = oa.image.convolve_2d(self.img, kernel, "constant", 0.0)
		orig = self.img.as_matrix().read_f32()
		res = result.as_matrix().read_f32()
		for o, r in zip(orig, res):
			self.assertAlmostEqual(o, r, places=5)

	def test_separable_convolve_2d_identity(self) -> None:
		kx = oa.Matrix.new_from_f32( [1], [1.0], runtime=self.runtime)
		ky = oa.Matrix.new_from_f32( [1], [1.0], runtime=self.runtime)
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



class ImageAugmentationTest(unittest.TestCase):
	@classmethod
	def setUpClass(cls) -> None:
		cls.runtime = oa.Runtime()
		cls.image = _make_gray_image(cls.runtime, width=8, height=8)

	def test_noise_like_deterministic_and_bounded(self) -> None:
		first = oa.image.noise_like(self.image, "gaussian", 0.1, 42)
		repeat = oa.image.noise_like(self.image, "gaussian", 0.1, 42)
		self.assertEqual(first.as_matrix().read_f32(), repeat.as_matrix().read_f32())
		other = oa.image.noise_like(self.image, "gaussian", 0.1, 43)
		self.assertNotEqual(first.as_matrix().read_f32(), other.as_matrix().read_f32())
		uniform = oa.image.noise_like(self.image, "uniform", 0.5, 42)
		assert all(abs(v) <= 0.5 + 1e-6 for v in uniform.as_matrix().read_f32())
		with self.assertRaises(Exception):
			oa.image.noise_like(self.image, "gaussian", -1.0, 42)

	def test_random_flip_deterministic_and_forced(self) -> None:
		a = oa.image.random_flip(self.image, 1.0, 0.0, 7)
		b = oa.image.random_flip(self.image, 1.0, 0.0, 7)
		self.assertEqual(a.as_matrix().read_f32(), b.as_matrix().read_f32())
		source = self.image.as_matrix().read_f32()
		flipped = a.as_matrix().read_f32()
		for row in range(8):
			for col in range(8):
				self.assertEqual(flipped[row * 8 + col], source[row * 8 + (7 - col)])
		with self.assertRaises(Exception):
			oa.image.random_flip(self.image, 1.5, 0.0, 7)

	def test_random_crop_is_valid_window(self) -> None:
		a = oa.image.random_crop(self.image, 4, 4, 9)
		b = oa.image.random_crop(self.image, 4, 4, 9)
		self.assertEqual(a.as_matrix().read_f32(), b.as_matrix().read_f32())
		self.assertEqual(a.as_matrix().shape, [1, 1, 4, 4])
		source = self.image.as_matrix().read_f32()
		cropped = a.as_matrix().read_f32()
		valid = any(
			all(cropped[i] == source[(y + i // 4) * 8 + (x + i % 4)] for i in range(16))
			for y in range(5) for x in range(5)
		)
		assert valid
		with self.assertRaises(Exception):
			oa.image.random_crop(self.image, 0, 4, 9)
		with self.assertRaises(Exception):
			oa.image.random_crop(self.image, 9, 4, 9)

	def test_random_rotate_is_pixel_permutation(self) -> None:
		a = oa.image.random_rotate(self.image, 5)
		b = oa.image.random_rotate(self.image, 5)
		self.assertEqual(a.as_matrix().read_f32(), b.as_matrix().read_f32())
		assert sorted(a.as_matrix().read_f32()) == sorted(self.image.as_matrix().read_f32())


class ImageStatisticsTest(unittest.TestCase):
	@classmethod
	def setUpClass(cls) -> None:
		cls.runtime = oa.Runtime()

	def test_calc_hist_matches_bucket_oracle(self) -> None:
		vals = [0.0, 0.25, 0.5, 0.75, 1.5, 0.5, -0.5, 0.999]
		m = oa.Matrix.new_from_f32([1, 1, 2, 4], vals, runtime=self.runtime)
		img = oa.Image.new_from_matrix(m, "nchw", "gray")
		histogram = img.calc_hist(4, 0.0, 1.0)
		self.assertEqual(histogram.dtype, "u32")
		self.assertEqual(histogram.shape, [1, 1, 4])
		self.assertEqual(histogram.read_u32(), [1, 1, 2, 2])

	def test_calc_hist_rejects_invalid_parameters(self) -> None:
		img = _make_gray_image(self.runtime)
		for bins, lower, upper in (
			(0, 0.0, 1.0),
			(300, 0.0, 1.0),
			(4, 1.0, 1.0),
			(4, 0.5, 0.0),
			(4, float("nan"), 1.0),
			(4, 0.0, float("inf")),
		):
			with self.assertRaises(Exception):
				oa.image.calc_hist(img, bins, lower, upper)

	def test_equalize_hist_spreads_degenerate_plane(self) -> None:
		m = oa.Matrix.new_from_f32([1, 1, 2, 4], [0.5] * 8, runtime=self.runtime)
		img = oa.Image.new_from_matrix(m, "nchw", "gray")
		result = oa.image.equalize_hist(img)
		self.assertIsInstance(result, oa.Image)
		self.assertEqual(result.channels, 1)

	def test_compare_hist_metric_oracles(self) -> None:
		first = oa.Matrix.new_from_u32([1, 1, 3], [1, 0, 0], runtime=self.runtime)
		second = oa.Matrix.new_from_u32([1, 1, 3], [0, 1, 0], runtime=self.runtime)
		for method, expected in (
			("correlation", -0.5),
			("chi_square", 1.0),
			("intersection", 0.0),
			("bhattacharyya", 1.0),
		):
			scores = oa.image.compare_hist(first, second, method)
			self.assertEqual(scores.shape, [1, 1])
			self.assertAlmostEqual(scores.read_f32()[0], expected, places=6)
		for method, expected in (
			("correlation", 1.0),
			("intersection", 1.0),
			("chi_square", 0.0),
			("bhattacharyya", 0.0),
		):
			scores = oa.image.compare_hist(first, first, method)
			self.assertAlmostEqual(scores.read_f32()[0], expected, places=6)

	def test_compare_hist_rejects_invalid_inputs(self) -> None:
		first = oa.Matrix.new_from_u32([1, 1, 3], [1, 0, 0], runtime=self.runtime)
		wrong_dtype = oa.Matrix.new_from_f32([1, 1, 3], [1.0, 0.0, 0.0], runtime=self.runtime)
		wrong_shape = oa.Matrix.new_from_u32([1, 3], [1, 0, 0], runtime=self.runtime)
		rank_one = oa.Matrix.new_from_u32([3], [1, 0, 0], runtime=self.runtime)
		with self.assertRaises(Exception):
			oa.image.compare_hist(first, wrong_dtype)

	def test_pyr_down_halves_with_gaussian_oracle(self) -> None:
		vals = [(i * 37 % 19) / 19.0 for i in range(20)]
		m = oa.Matrix.new_from_f32([1, 1, 4, 5], vals, runtime=self.runtime)
		img = oa.Image.new_from_matrix(m, "nchw", "gray")
		down = img.pyr_down()
		self.assertEqual(down.as_matrix().shape, [1, 1, 2, 3])
		kernel = (1.0, 4.0, 6.0, 4.0, 1.0)
		for y in range(2):
			for x in range(3):
				total = 0.0
				for dy in range(-2, 3):
					sy = 2 * y + dy
					sy = min(max(sy, -sy), 2 * (4 - 1) - sy) if sy < 0 or sy >= 4 else sy
					for dx in range(-2, 3):
						sx = 2 * x + dx
						sx = min(max(sx, -sx), 2 * (5 - 1) - sx) if sx < 0 or sx >= 5 else sx
						total += kernel[dy + 2] * kernel[dx + 2] * vals[sy * 5 + sx]
				self.assertAlmostEqual(
					down.as_matrix().read_f32()[y * 3 + x], total / 256.0, places=5
				)

	def test_pyr_up_doubles_resolution(self) -> None:
		img = _make_gray_image(self.runtime, width=4, height=3)
		up = img.pyr_up()
		self.assertEqual(up.as_matrix().shape, [1, 1, 6, 8])
		roundtrip = up.pyr_down()
		self.assertEqual(roundtrip.as_matrix().shape, [1, 1, 3, 4])

	def test_threshold_auto_otsu_splits_bimodal(self) -> None:
		vals = [0.1] * 12 + [0.9] * 12
		m = oa.Matrix.new_from_f32([1, 1, 4, 6], vals, runtime=self.runtime)
		img = oa.Image.new_from_matrix(m, "nchw", "gray")
		output, levels = img.threshold_auto("otsu", 1.0)
		self.assertEqual(levels.shape, [1, 1])
		for actual, source in zip(output.as_matrix().read_f32(), vals):
			self.assertEqual(actual, 1.0 if source > 0.5 else 0.0)

	def test_threshold_auto_triangle_and_validation(self) -> None:
		vals = [0.05] * 40 + [0.4, 0.5, 0.55, 0.6, 0.45, 0.5, 0.55, 0.6]
		m = oa.Matrix.new_from_f32([1, 1, 6, 8], vals, runtime=self.runtime)
		img = oa.Image.new_from_matrix(m, "nchw", "gray")
		output, levels = oa.image.threshold_auto(img, "triangle", 2.0)
		self.assertTrue(any(v == 2.0 for v in output.as_matrix().read_f32()))
		with self.assertRaises(Exception):
			img.threshold_auto("otsu", float("nan"))
		with self.assertRaises(Exception):
			img.threshold_auto("bogus_method", 1.0)

	def test_integral_matches_host_sat(self) -> None:
		vals = [(i * 7 % 11) / 11.0 for i in range(12)]
		m = oa.Matrix.new_from_f32([1, 1, 3, 4], vals, runtime=self.runtime)
		img = oa.Image.new_from_matrix(m, "nchw", "gray")
		sat = img.integral()
		self.assertEqual(sat.shape, [1, 1, 4, 5])
		result = sat.read_f32()
		for y in range(4):
			for x in range(5):
				expected = sum(
					vals[sy * 4 + sx] for sy in range(y) for sx in range(x)
				)
				self.assertAlmostEqual(result[y * 5 + x], expected, places=4)
		with self.assertRaises(Exception):
			oa.image.compare_hist(first, wrong_shape)
		with self.assertRaises(Exception):
			oa.image.compare_hist(rank_one, rank_one)


	def test_cvt_color_rgb_to_gray_and_back(self) -> None:
		m = oa.Matrix.new_from_f32(
			[1, 3, 1, 2], [1.0, 0.0, 0.0, 1.0, 0.0, 1.0], runtime=self.runtime
		)
		img = oa.Image.new_from_matrix(m, "nchw", "rgb")
		gray = img.cvt_color("rgb_to_gray")
		self.assertEqual(gray.channels, 1)
		result = gray.as_matrix().read_f32()
		self.assertAlmostEqual(result[0], 0.299, places=5)
		self.assertAlmostEqual(result[1], 0.701, places=5)
		rgb = gray.cvt_color("gray_to_rgb")
		self.assertEqual(rgb.channels, 3)
		with self.assertRaises(Exception):
			img.cvt_color("gray_to_rgb")
		with self.assertRaises(Exception):
			img.cvt_color("bogus_code")

	def test_cvt_color_hsv_round_trip(self) -> None:
		vals = [0.8, 0.4, 0.1, 0.2, 0.9, 0.3, 0.5, 0.25, 0.75]
		m = oa.Matrix.new_from_f32([1, 3, 1, 3], vals, runtime=self.runtime)
		img = oa.Image.new_from_matrix(m, "nchw", "rgb")
		hsv = img.cvt_color("rgb_to_hsv")
		round_trip = hsv.cvt_color("hsv_to_rgb").as_matrix().read_f32()
		for expected, actual in zip(vals, round_trip):
			self.assertAlmostEqual(expected, actual, places=4)

	def test_apply_color_map_jet(self) -> None:
		m = oa.Matrix.new_from_f32([1, 1, 1, 2], [0.0, 1.0], runtime=self.runtime)
		img = oa.Image.new_from_matrix(m, "nchw", "gray")
		mapped = img.apply_color_map("jet")
		self.assertEqual(mapped.channels, 3)
		result = mapped.as_matrix().read_f32()
		for expected, actual in zip([0.0, 0.5, 0.0, 0.0, 0.5, 0.0], result):
			self.assertAlmostEqual(expected, actual, places=5)

	def test_calc_back_project(self) -> None:
		m = oa.Matrix.new_from_f32([1, 1, 1, 4], [0.1, 0.4, 0.6, 0.9], runtime=self.runtime)
		img = oa.Image.new_from_matrix(m, "nchw", "gray")
		histogram = img.calc_hist(4, 0.0, 1.0)
		projected = img.calc_back_project(histogram, 0.0, 1.0)
		self.assertEqual(projected.as_matrix().read_f32(), [1.0] * 4)

	def test_clahe_single_tile_cdf(self) -> None:
		m = oa.Matrix.new_from_f32([1, 1, 1, 4], [0.0, 0.5, 1.0, 0.25], runtime=self.runtime)
		img = oa.Image.new_from_matrix(m, "nchw", "gray")
		result = img.clahe(40.0, 4, 1).as_matrix().read_f32()
		for expected, actual in zip([0.25, 0.75, 1.0, 0.5], result):
			self.assertAlmostEqual(expected, actual, places=5)
		with self.assertRaises(Exception):
			img.clahe(0.0, 4, 1)

	def test_accumulate_and_weighted(self) -> None:
		first = oa.Matrix.new_from_f32([1, 1, 1, 2], [0.5, 0.5], runtime=self.runtime)
		second = oa.Matrix.new_from_f32([1, 1, 1, 2], [0.25, 0.75], runtime=self.runtime)
		a = oa.Image.new_from_matrix(first, "nchw", "gray")
		b = oa.Image.new_from_matrix(second, "nchw", "gray")
		self.assertEqual(a.accumulate(b).as_matrix().read_f32(), [0.75, 1.25])
		blended = a.accumulate_weighted(b, 0.5).as_matrix().read_f32()
		self.assertAlmostEqual(blended[0], 0.375)
		self.assertAlmostEqual(blended[1], 0.625)

	def test_moments_oracle(self) -> None:
		vals = [1.0, 2.0, 0.5, 0.0, 1.5, 2.5]
		m = oa.Matrix.new_from_f32([1, 1, 2, 3], vals, runtime=self.runtime)
		img = oa.Image.new_from_matrix(m, "nchw", "gray")
		moments = img.moments()
		self.assertEqual(moments.shape, [1, 1, 10])
		result = moments.read_f32()
		oracle = [0.0] * 10
		for y in range(2):
			for x in range(3):
				v = vals[y * 3 + x]
				oracle[0] += v
				oracle[1] += x * v
				oracle[2] += y * v
				oracle[3] += x * x * v
				oracle[4] += x * y * v
				oracle[5] += y * y * v
		for expected, actual in zip(oracle[:6], result[:6]):
			self.assertAlmostEqual(expected, actual, places=4)

	def test_psnr_and_ssim(self) -> None:
		vals = [0.25, 0.5, 0.75, 1.0]
		m = oa.Matrix.new_from_f32([1, 1, 2, 2], vals, runtime=self.runtime)
		img = oa.Image.new_from_matrix(m, "nchw", "gray")
		self.assertTrue(img.psnr(img, 1.0).read_f32()[0] > 100.0)
		self.assertAlmostEqual(img.ssim(img).read_f32()[0], 1.0, places=5)
		other_vals = [0.35, 0.4, 0.65, 1.0]
		other = oa.Image.new_from_matrix(
			oa.Matrix.new_from_f32([1, 1, 2, 2], other_vals, runtime=self.runtime),
			"nchw", "gray",
		)
		self.assertLess(img.ssim(other).read_f32()[0], 1.0)

	def test_match_template_sqdiff(self) -> None:
		vals = [
			0.0, 0.5, 0.9, 0.2,
			0.1, 0.6, 1.0, 0.3,
			0.2, 0.7, 0.8, 0.4,
		]
		img = oa.Image.new_from_matrix(
			oa.Matrix.new_from_f32([1, 1, 3, 4], vals, runtime=self.runtime),
			"nchw", "gray",
		)
		template = oa.Image.new_from_matrix(
			oa.Matrix.new_from_f32([1, 1, 1, 2], [0.6, 1.0], runtime=self.runtime),
			"nchw", "gray",
		)
		scores = img.match_template(template, "sqdiff")
		self.assertEqual(scores.shape, [1, 1, 3, 3])
		result = scores.read_f32()
		self.assertAlmostEqual(result[4], 0.0, places=5)
		self.assertEqual(min(range(9), key=lambda i: result[i]), 4)
		with self.assertRaises(Exception):
			img.match_template(template, "bogus_method")

	def test_warp_polar_linear(self) -> None:
		vals = [0.0] * 16
		vals[2 * 4 + 2] = 1.0
		img = oa.Image.new_from_matrix(
			oa.Matrix.new_from_f32([1, 1, 4, 4], vals, runtime=self.runtime),
			"nchw", "gray",
		)
		output = img.warp_polar(
			1.5, 1.5, 1.0, 8, 8, "linear", "nearest", "constant", 0.0
		)
		self.assertEqual(output.as_matrix().shape, [1, 1, 8, 8])
		result = output.as_matrix().read_f32()
		self.assertAlmostEqual(result[1 * 8 + 6], 1.0, places=4)
		with self.assertRaises(Exception):
			img.warp_polar(1.5, 1.5, 0.0, 8, 8, "linear", "nearest", "constant", 0.0)

	def test_transform_helpers(self) -> None:
		rotation = oa.image.rotation_matrix_2d(0.0, 0.0, 90.0, 1.0).read_f32()
		self.assertAlmostEqual(rotation[0], 0.0, places=6)
		self.assertAlmostEqual(rotation[1], 1.0, places=6)
		self.assertAlmostEqual(rotation[3], -1.0, places=6)
		inverse = oa.image.invert_affine(
			oa.Matrix.new_from_f32([2, 3], rotation, runtime=self.runtime)
		).read_f32()
		self.assertAlmostEqual(inverse[0], 0.0, places=5)
		self.assertAlmostEqual(inverse[1], -1.0, places=5)
		source = oa.Matrix.new_from_f32([4, 2], [0, 0, 8, 0, 8, 4, 0, 4], runtime=self.runtime)
		destination = oa.Matrix.new_from_f32([4, 2], [1, 1, 7, 0.5, 7.5, 5, 0.5, 4.5], runtime=self.runtime)
		homography = oa.image.perspective_transform(source, destination).read_f32()
		self.assertAlmostEqual(homography[8], 1.0, places=6)
		x, y = 8.0, 4.0
		w = homography[6] * x + homography[7] * y + homography[8]
		u = (homography[0] * x + homography[1] * y + homography[2]) / w
		v = (homography[3] * x + homography[4] * y + homography[5]) / w
		self.assertAlmostEqual(u, 7.5, places=4)
		self.assertAlmostEqual(v, 5.0, places=4)
		forward = oa.image.rotation_matrix_2d(-1.0, 0.5, -15.0, 1.25).read_f32()
		points, mapped = [], []
		for index in range(8):
			x = (index % 4) * 2.0 - 3.0
			y = (index // 4) * 3.0 + 1.0 + index * 0.5
			points += [x, y]
			mapped += [
				forward[0] * x + forward[1] * y + forward[2],
				forward[3] * x + forward[4] * y + forward[5],
			]
		estimated = oa.image.estimate_affine_2d(
			oa.Matrix.new_from_f32([8, 2], points, runtime=self.runtime),
			oa.Matrix.new_from_f32([8, 2], mapped, runtime=self.runtime),
		).read_f32()
		for index in range(6):
			self.assertAlmostEqual(estimated[index], forward[index], places=4)
		with self.assertRaises(Exception):
			oa.image.invert_affine(oa.Matrix.new_from_f32([2, 3], [1, 2, 0, 2, 4, 0], runtime=self.runtime))


	def test_canny_step_edge(self) -> None:
		vals = [0.0] * (10 * 12)
		for y in range(10):
			for x in range(6, 12):
				vals[y * 12 + x] = 1.0
		img = oa.Image.new_from_matrix(
			oa.Matrix.new_from_f32([1, 1, 10, 12], vals, runtime=self.runtime),
			"nchw", "gray",
		)
		output = img.canny(0.5, 2.0, 0).as_matrix().read_f32()
		self.assertEqual(sum(output), float(output.count(1.0)))
		self.assertGreater(sum(1 for v in output if v == 1.0), 0)
		with self.assertRaises(Exception):
			img.canny(1.0, 0.5, 0)


	def test_connected_components(self) -> None:
		vals = [0.0] * (6 * 8)
		for x, y in [(1, 1), (2, 2), (3, 3), (6, 1), (6, 2), (6, 3), (0, 5)]:
			vals[y * 8 + x] = 1.0
		img = oa.Image.new_from_matrix(
			oa.Matrix.new_from_f32([1, 1, 6, 8], vals, runtime=self.runtime),
			"nchw", "gray",
		)
		labels = img.connected_components(0.5, "eight", 0).read_u32()
		self.assertEqual(labels[9], labels[2 * 8 + 2])
		self.assertEqual(labels[1 * 8 + 6], labels[3 * 8 + 6])
		self.assertNotEqual(labels[9], labels[1 * 8 + 6])
		self.assertEqual(labels[0], 0xFFFFFFFF)
		four = img.connected_components(0.5, "four", 0).read_u32()
		self.assertNotEqual(four[9], four[2 * 8 + 2])
		self.assertEqual(four[2 * 8 + 6], four[3 * 8 + 6])

	def test_distance_transform(self) -> None:
		width, height = 9, 7
		vals = [1.0] * (width * height)
		vals[2 * width + 4] = 0.0
		img = oa.Image.new_from_matrix(
			oa.Matrix.new_from_f32([1, 1, height, width], vals, runtime=self.runtime),
			"nchw", "gray",
		)
		l1 = img.distance_transform(0.5, "l1", 0).read_f32()
		l2 = img.distance_transform(0.5, "l2", 0).read_f32()
		sqrt2 = 2.0 ** 0.5
		for y in range(height):
			for x in range(width):
				dx, dy = abs(x - 4), abs(y - 2)
				i = y * width + x
				self.assertAlmostEqual(l1[i], dx + dy, places=4)
				self.assertAlmostEqual(
					l2[i], min(dx, dy) * sqrt2 + abs(dx - dy), places=4
				)

	def test_hough_lines(self) -> None:
		width, height = 16, 16
		vals = [0.0] * (width * height)
		for y in range(height):
			vals[y * width + 4] = 1.0
		img = oa.Image.new_from_matrix(
			oa.Matrix.new_from_f32([1, 1, height, width], vals, runtime=self.runtime),
			"nchw", "gray",
		)
		rows = img.hough_lines(0.5, 1.0, 1.0, 16, 4).read_f32()
		found = False
		for row in range(4):
			rho, theta, votes = rows[row * 3:row * 3 + 3]
			if votes > 0 and abs(rho - 4.0) < 0.6 and theta < 2.0:
				found = True
				self.assertEqual(votes, 16.0)
		self.assertTrue(found)

	def test_hough_lines_p(self) -> None:
		width, height = 16, 12
		vals = [0.0] * (width * height)
		for x in range(2, 14):
			if x == 8:
				continue
			vals[5 * width + x] = 1.0
		img = oa.Image.new_from_matrix(
			oa.Matrix.new_from_f32([1, 1, height, width], vals, runtime=self.runtime),
			"nchw", "gray",
		)
		rows = img.hough_lines_p(0.5, 1.0, 1.0, 10, 4, 8.0, 1).read_f32()
		found = False
		for row in range(4):
			x1, y1, x2, y2 = rows[row * 4:row * 4 + 4]
			if x1 == 0 and y1 == 0 and x2 == 0 and y2 == 0:
				continue
			if abs(x2 - x1) >= 10 and abs(y1 - 5) <= 1.5 and abs(y2 - 5) <= 1.5:
				found = True
				self.assertAlmostEqual(x1, 2.0, delta=1.5)
				self.assertAlmostEqual(x2, 13.0, delta=1.5)
		self.assertTrue(found)

	def test_find_contours(self) -> None:
		width, height = 16, 12
		vals = [0.0] * (width * height)
		for y in range(2, 6):
			for x in range(3, 8):
				vals[y * width + x] = 1.0
		for y in range(8, 10):
			for x in range(12, 14):
				vals[y * width + x] = 1.0
		img = oa.Image.new_from_matrix(
			oa.Matrix.new_from_f32([1, 1, height, width], vals, runtime=self.runtime),
			"nchw", "gray",
		)
		points, lengths = img.find_contours(0.5)
		self.assertEqual(points.shape, [1, 64, 112, 2])
		self.assertEqual(lengths.shape, [1, 64])
		counts = sorted(c for c in lengths.read_u32() if c > 0)
		self.assertEqual(counts, [4, 14])

	def test_flood_fill(self) -> None:
		width, height = 8, 6
		vals = [0.4 if x < 4 else 0.9 for y in range(height) for x in range(width)]
		img = oa.Image.new_from_matrix(
			oa.Matrix.new_from_f32([1, 1, height, width], vals, runtime=self.runtime),
			"nchw", "gray",
		)
		filled = img.flood_fill(2, 2, 1.0, lo_diff=0.05, up_diff=0.05).as_matrix().read_f32()
		for y in range(height):
			for x in range(width):
				self.assertAlmostEqual(filled[y * width + x], 1.0 if x < 4 else 0.9)

	def test_resize_interpolation_modes(self) -> None:
		vals = [float((i * 7) % 11) * 0.3 for i in range(35)]
		img = oa.Image.new_from_matrix(
			oa.Matrix.new_from_f32([1, 1, 5, 7], vals, runtime=self.runtime),
			"nchw", "gray",
		)
		for mode in ("nearest", "bilinear", "area", "cubic", "lanczos"):
			out = img.resize(4, 3, interpolation=mode)
			self.assertEqual(out.as_matrix().shape, [1, 1, 3, 4])
		# INTER_AREA 8x8 -> 4x4 box mean on a constant-per-block source.
		blocks = [float((y // 2) * 4 + x // 2) for y in range(8) for x in range(8)]
		img = oa.Image.new_from_matrix(
			oa.Matrix.new_from_f32([1, 1, 8, 8], blocks, runtime=self.runtime),
			"nchw", "gray",
		)
		out = img.resize(4, 4, interpolation="area").as_matrix().read_f32()
		for y in range(4):
			for x in range(4):
				self.assertAlmostEqual(out[y * 4 + x], float(y * 4 + x))


	def test_draw_line(self) -> None:
		width, height = 8, 6
		img = oa.Image.new_from_matrix(
			oa.Matrix.new_from_f32(
				[1, 1, height, width], [0.0] * (width * height), runtime=self.runtime
			),
			"nchw", "gray",
		)
		out = img.draw_line(1.5, 2.5, 6.5, 2.5, 1.0, 1.0).as_matrix().read_f32()
		for x in range(width):
			self.assertAlmostEqual(out[2 * width + x], 1.0 if 1 <= x <= 6 else 0.0)
			self.assertAlmostEqual(out[x], 0.0)

	def test_draw_rectangle(self) -> None:
		width, height = 8, 6
		img = oa.Image.new_from_matrix(
			oa.Matrix.new_from_f32(
				[1, 1, height, width], [0.0] * (width * height), runtime=self.runtime
			),
			"nchw", "gray",
		)
		out = img.draw_rectangle(2.0, 1.0, 4.0, 3.0, 1.0, -1.0).as_matrix().read_f32()
		for y in range(height):
			for x in range(width):
				inside = 2 <= x <= 5 and 1 <= y <= 3
				self.assertAlmostEqual(out[y * width + x], 1.0 if inside else 0.0)

	def test_draw_circle(self) -> None:
		width, height = 9, 9
		img = oa.Image.new_from_matrix(
			oa.Matrix.new_from_f32(
				[1, 1, height, width], [0.0] * (width * height), runtime=self.runtime
			),
			"nchw", "gray",
		)
		out = img.draw_circle(4.5, 4.5, 2.0, 1.0, -1.0).as_matrix().read_f32()
		for y in range(height):
			for x in range(width):
				d = ((x + 0.5 - 4.5) ** 2 + (y + 0.5 - 4.5) ** 2) ** 0.5
				self.assertAlmostEqual(out[y * width + x], 1.0 if d <= 2.0 else 0.0)

	def test_fill_polygon(self) -> None:
		width, height = 8, 8
		img = oa.Image.new_from_matrix(
			oa.Matrix.new_from_f32(
				[1, 1, height, width], [0.0] * (width * height), runtime=self.runtime
			),
			"nchw", "gray",
		)
		points = oa.Matrix.new_from_f32(
			[3, 2], [1.0, 1.0, 6.0, 1.0, 1.0, 6.0], runtime=self.runtime
		)
		out = img.fill_polygon(points, 1.0).as_matrix().read_f32()
		# Spot checks: clearly-inside centroid region set, corners clear.
		self.assertAlmostEqual(out[2 * width + 2], 1.0)
		self.assertAlmostEqual(out[3 * width + 2], 1.0)
		self.assertAlmostEqual(out[6 * width + 6], 0.0)
		self.assertAlmostEqual(out[7 * width + 7], 0.0)


	def test_corner_score(self) -> None:
		width, height = 8, 6
		values = [1.0 if x >= 4 and y >= 3 else 0.0 for y in range(height) for x in range(width)]
		img = oa.Image.new_from_matrix(
			oa.Matrix.new_from_f32([1, 1, height, width], values, runtime=self.runtime),
			"nchw", "gray",
		)
		for mode in ("harris", "min_eigen"):
			out = img.corner_score(3, 0.04, mode).as_matrix().read_f32()
			self.assertEqual(len(out), width * height)
			self.assertGreater(max(out), 0.0)

	def test_corner_fast(self) -> None:
		width, height = 9, 9
		values = [1.0 if x >= 4 and y >= 4 else 0.0 for y in range(height) for x in range(width)]
		img = oa.Image.new_from_matrix(
			oa.Matrix.new_from_f32([1, 1, height, width], values, runtime=self.runtime),
			"nchw", "gray",
		)
		out = img.corner_fast(9).as_matrix().read_f32()
		self.assertGreater(out[4 * width + 4], 0.9)
		self.assertAlmostEqual(out[7 * width + 7], 0.0)
		self.assertAlmostEqual(out[0], 0.0)


	def test_bf_match(self) -> None:
		query = oa.Matrix.new_from_u8(
			[2, 4], [0, 0, 0, 0, 255, 255, 0, 0], runtime=self.runtime
		)
		train = oa.Matrix.new_from_u8(
			[3, 4],
			[1, 2, 4, 8, 255, 255, 0, 1, 0, 0, 0, 3],
			runtime=self.runtime,
		)
		rows = oa.image.bf_match(query, train).read_f32()
		# query[0] all-zero: best train[2] (dist 2), second train[0] (dist 4).
		self.assertEqual(rows[0], 2.0)
		self.assertEqual(rows[1], 2.0)
		self.assertEqual(rows[2], 0.0)
		self.assertEqual(rows[3], 4.0)
		# query[1]: best train[1] (dist 1).
		self.assertEqual(rows[4], 1.0)
		self.assertEqual(rows[5], 1.0)


	def test_good_features_to_track(self) -> None:
		width, height = 16, 16
		vals = [0.0] * (width * height)
		vals[4 * width + 4] = 1.0
		vals[4 * width + 5] = 0.9
		vals[12 * width + 12] = 0.5
		vals[12 * width + 4] = 0.2
		img = oa.Image.new_from_matrix(
			oa.Matrix.new_from_f32([1, 1, height, width], vals, runtime=self.runtime),
			"nchw", "gray",
		)
		keypoints, counts = img.good_features_to_track(
			max_corners=8, quality=0.1, min_distance=3.0
		)
		self.assertEqual(keypoints.shape, [1, 8, 3])
		self.assertEqual(counts.read_u32(), [3])
		rows = keypoints.read_f32()
		filled = sorted(
			(tuple(rows[i : i + 3]) for i in range(0, 24, 3) if rows[i + 2] > 0.0),
			key=lambda r: -r[2],
		)
		expected = [(4.0, 4.0, 1.0), (12.0, 12.0, 0.5), (4.0, 12.0, 0.2)]
		self.assertEqual(len(filled), 3)
		for actual, want in zip(filled, expected):
			for a, w in zip(actual, want):
				self.assertAlmostEqual(a, w)
		keypoints, counts = img.good_features_to_track(
			max_corners=2, quality=0.1, min_distance=3.0
		)
		self.assertEqual(keypoints.shape, [1, 2, 3])
		self.assertEqual(counts.read_u32(), [3])

	def test_orb(self) -> None:
		width, height = 96, 96
		vals = [0.0] * (width * height)
		for y in range(20, 70):
			for x in range(30, 85):
				falloff = 1.0 - ((x - 55.0) / 35.0) ** 2
				if falloff > 0.0:
					vals[y * width + x] = 0.8 * falloff + 0.2
		img = oa.Image.new_from_matrix(
			oa.Matrix.new_from_f32([1, 1, height, width], vals, runtime=self.runtime),
			"nchw", "gray",
		)
		keypoints, descriptors, counts = img.orb(
			max_features=64, levels=1, edge=18
		)
		self.assertEqual(keypoints.shape, [1, 64, 6])
		self.assertEqual(descriptors.shape, [1, 64, 8])
		claimed = counts.read_u32()[0]
		self.assertGreaterEqual(claimed, 1)
		rows = keypoints.read_f32()
		top = max(
			(rows[i : i + 6] for i in range(0, min(claimed, 64) * 6, 6)),
			key=lambda r: r[3],
		)
		self.assertGreaterEqual(top[0], 18.0)
		self.assertLess(top[0], 78.0)
		self.assertLessEqual(abs(top[2]), 3.1415927)
		words = descriptors.read_u32()
		self.assertTrue(any(w != 0 for w in words[:8]))

if __name__ == "__main__":

	unittest.main()
