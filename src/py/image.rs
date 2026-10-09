use pyo3::{exceptions::PyValueError, prelude::*, types::PyBytes};

use crate::{error::python_error, matrix::PythonMatrix, runtime::PythonEngine};

fn layout(token: &str) -> PyResult<oa::ImageLayout> {
	match token {
		"nchw" => Ok(oa::ImageLayout::Nchw),
		"nhwc" => Ok(oa::ImageLayout::Nhwc),
		"chw" => Ok(oa::ImageLayout::Chw),
		"hwc" => Ok(oa::ImageLayout::Hwc),
		"hw" => Ok(oa::ImageLayout::Hw),
		_ => Err(PyValueError::new_err(format!(
			"unknown image layout: {token}"
		))),
	}
}

fn layout_token(value: oa::ImageLayout) -> &'static str {
	match value {
		oa::ImageLayout::Nchw => "nchw",
		oa::ImageLayout::Nhwc => "nhwc",
		oa::ImageLayout::Chw => "chw",
		oa::ImageLayout::Hwc => "hwc",
		oa::ImageLayout::Hw => "hw",
	}
}

fn format(token: &str) -> PyResult<oa::ImageFormat> {
	match token {
		"gray" => Ok(oa::ImageFormat::Gray),
		"gray_alpha" => Ok(oa::ImageFormat::GrayAlpha),
		"rgb" => Ok(oa::ImageFormat::Rgb),
		"rgba" => Ok(oa::ImageFormat::Rgba),
		"bgr" => Ok(oa::ImageFormat::Bgr),
		"bgra" => Ok(oa::ImageFormat::Bgra),
		_ => Err(PyValueError::new_err(format!(
			"unknown image format: {token}"
		))),
	}
}

fn format_token(value: oa::ImageFormat) -> &'static str {
	match value {
		oa::ImageFormat::Gray => "gray",
		oa::ImageFormat::GrayAlpha => "gray_alpha",
		oa::ImageFormat::Rgb => "rgb",
		oa::ImageFormat::Rgba => "rgba",
		oa::ImageFormat::Bgr => "bgr",
		oa::ImageFormat::Bgra => "bgra",
	}
}

fn codec(token: &str) -> PyResult<oa::image::ImageCodec> {
	match token {
		"auto" => Ok(oa::image::ImageCodec::Auto),
		"jpeg" | "jpg" => Ok(oa::image::ImageCodec::Jpeg),
		"png" => Ok(oa::image::ImageCodec::Png),
		"webp" => Ok(oa::image::ImageCodec::Webp),
		"bmp" => Ok(oa::image::ImageCodec::Bmp),
		"tga" => Ok(oa::image::ImageCodec::Tga),
		_ => Err(PyValueError::new_err(format!(
			"unknown image codec: {token}"
		))),
	}
}

fn interpolation(token: &str) -> PyResult<oa::image::InterpolationMode> {
	match token {
		"nearest" => Ok(oa::image::InterpolationMode::Nearest),
		"bilinear" => Ok(oa::image::InterpolationMode::Bilinear),
		_ => Err(PyValueError::new_err(format!(
			"unknown interpolation mode: {token}"
		))),
	}
}

fn border(token: &str) -> PyResult<oa::image::BorderMode> {
	match token {
		"constant" => Ok(oa::image::BorderMode::Constant),
		"replicate" => Ok(oa::image::BorderMode::Replicate),
		"reflect" => Ok(oa::image::BorderMode::Reflect),
		"reflect_101" => Ok(oa::image::BorderMode::Reflect101),
		"wrap" => Ok(oa::image::BorderMode::Wrap),
		_ => Err(PyValueError::new_err(format!(
			"unknown border mode: {token}"
		))),
	}
}

#[pyclass(name = "Image", unsendable)]
pub(crate) struct PythonImage {
	pub(crate) inner: oa::Image,
}

impl PythonImage {
	pub(crate) fn wrap(inner: oa::Image) -> Self {
		Self { inner }
	}
}

#[pymethods]
impl PythonImage {
	#[staticmethod]
	fn from_matrix(matrix: &PythonMatrix, layout_name: &str, format_name: &str) -> PyResult<Self> {
		oa::Image::new(
			matrix.inner.clone(),
			layout(layout_name)?,
			format(format_name)?,
		)
		.map(Self::wrap)
		.map_err(python_error)
	}

	#[getter]
	fn width(&self) -> usize {
		self.inner.width()
	}
	#[getter]
	fn height(&self) -> usize {
		self.inner.height()
	}
	#[getter]
	fn channels(&self) -> usize {
		self.inner.channels()
	}
	#[getter]
	fn batch_size(&self) -> usize {
		self.inner.batch_size()
	}
	#[getter]
	fn layout(&self) -> &'static str {
		layout_token(self.inner.layout())
	}
	#[getter]
	fn format(&self) -> &'static str {
		format_token(self.inner.format())
	}
	#[getter]
	fn dtype(&self) -> &'static str {
		self.inner.dtype().token()
	}

	fn as_matrix(&self) -> PythonMatrix {
		PythonMatrix::wrap(self.inner.as_matrix().clone())
	}

	fn __repr__(&self) -> String {
		format!(
			"Image(width={}, height={}, channels={}, layout='{}', format='{}', dtype='{}')",
			self.inner.width(),
			self.inner.height(),
			self.inner.channels(),
			layout_token(self.inner.layout()),
			format_token(self.inner.format()),
			self.inner.dtype().token(),
		)
	}
}

macro_rules! unary_image {
	($binding:ident, $operation:path) => {
		#[pyfunction]
		fn $binding(input: &PythonImage) -> PyResult<PythonImage> {
			$operation(&input.inner)
				.map(PythonImage::wrap)
				.map_err(python_error)
		}
	};
}

#[pyfunction]
fn image_decode_file(
	engine: &PythonEngine,
	path: &str,
	format_name: &str,
) -> PyResult<PythonImage> {
	oa::image::decode_file(&engine.inner, path, format(format_name)?)
		.map(PythonImage::wrap)
		.map_err(python_error)
}

#[pyfunction]
fn image_decode_memory(
	engine: &PythonEngine,
	encoded: Vec<u8>,
	format_name: &str,
) -> PyResult<PythonImage> {
	oa::image::decode_memory(&engine.inner, &encoded, format(format_name)?)
		.map(PythonImage::wrap)
		.map_err(python_error)
}

#[pyfunction]
#[pyo3(signature = (input, codec_name, quality=90))]
fn image_encode(
	py: Python<'_>,
	input: &PythonImage,
	codec_name: &str,
	quality: u32,
) -> PyResult<Py<PyBytes>> {
	let encoded =
		oa::image::encode(&input.inner, codec(codec_name)?, quality).map_err(python_error)?;
	Ok(PyBytes::new(py, &encoded).unbind())
}

#[pyfunction]
#[pyo3(signature = (path, input, quality=90))]
fn image_save_file(path: &str, input: &PythonImage, quality: u32) -> PyResult<()> {
	oa::image::save_file(path, &input.inner, quality).map_err(python_error)
}

#[pyfunction]
#[pyo3(signature = (input, width, height, interpolation_name="bilinear"))]
fn image_resize(
	input: &PythonImage,
	width: u32,
	height: u32,
	interpolation_name: &str,
) -> PyResult<PythonImage> {
	oa::image::resize(
		&input.inner,
		width,
		height,
		interpolation(interpolation_name)?,
	)
	.map(PythonImage::wrap)
	.map_err(python_error)
}

#[pyfunction]
fn image_normalize(input: &PythonImage, mean: [f32; 3], std: [f32; 3]) -> PyResult<PythonImage> {
	oa::image::normalize(&input.inner, oa::image::NormalizationParams { mean, std })
		.map(PythonImage::wrap)
		.map_err(python_error)
}

#[pyfunction]
fn image_convert_color(input: &PythonImage, format_name: &str) -> PyResult<PythonImage> {
	oa::image::convert_color(&input.inner, format(format_name)?)
		.map(PythonImage::wrap)
		.map_err(python_error)
}

#[pyfunction]
fn image_crop(
	input: &PythonImage,
	x: u32,
	y: u32,
	width: u32,
	height: u32,
) -> PyResult<PythonImage> {
	oa::image::crop(&input.inner, x, y, width, height)
		.map(PythonImage::wrap)
		.map_err(python_error)
}

#[pyfunction]
fn image_flip(input: &PythonImage, horizontal: bool, vertical: bool) -> PyResult<PythonImage> {
	oa::image::flip(&input.inner, horizontal, vertical)
		.map(PythonImage::wrap)
		.map_err(python_error)
}

#[pyfunction]
fn image_rotate(input: &PythonImage, degrees: u32) -> PyResult<PythonImage> {
	oa::image::rotate(&input.inner, degrees)
		.map(PythonImage::wrap)
		.map_err(python_error)
}

#[pyfunction]
fn image_center_crop(input: &PythonImage, width: u32, height: u32) -> PyResult<PythonImage> {
	oa::image::center_crop(&input.inner, width, height)
		.map(PythonImage::wrap)
		.map_err(python_error)
}

#[pyfunction]
#[pyo3(signature = (input, left, right, top, bottom, border_name="constant", border_value=0.0))]
#[allow(clippy::too_many_arguments)]
fn image_pad(
	input: &PythonImage,
	left: u32,
	right: u32,
	top: u32,
	bottom: u32,
	border_name: &str,
	border_value: f32,
) -> PyResult<PythonImage> {
	oa::image::pad(
		&input.inner,
		left,
		right,
		top,
		bottom,
		border(border_name)?,
		border_value,
	)
	.map(PythonImage::wrap)
	.map_err(python_error)
}

unary_image!(image_grayscale, oa::image::grayscale);

#[pyfunction]
fn image_clamp(input: &PythonImage, low: f32, high: f32) -> PyResult<PythonImage> {
	oa::image::clamp(&input.inner, low, high)
		.map(PythonImage::wrap)
		.map_err(python_error)
}

#[pyfunction]
#[pyo3(signature = (input, max_value=1.0))]
fn image_invert(input: &PythonImage, max_value: f32) -> PyResult<PythonImage> {
	oa::image::invert(&input.inner, max_value)
		.map(PythonImage::wrap)
		.map_err(python_error)
}

#[pyfunction]
fn image_brightness_contrast(
	input: &PythonImage,
	brightness: f32,
	contrast: f32,
) -> PyResult<PythonImage> {
	oa::image::brightness_contrast(&input.inner, brightness, contrast)
		.map(PythonImage::wrap)
		.map_err(python_error)
}

#[pyfunction]
fn image_gamma_contrast(input: &PythonImage, gamma: f32, gain: f32) -> PyResult<PythonImage> {
	oa::image::gamma_contrast(&input.inner, gamma, gain)
		.map(PythonImage::wrap)
		.map_err(python_error)
}

#[pyfunction]
fn image_solarize(input: &PythonImage, threshold: f32, max_value: f32) -> PyResult<PythonImage> {
	oa::image::solarize(&input.inner, threshold, max_value)
		.map(PythonImage::wrap)
		.map_err(python_error)
}

#[pyfunction]
fn image_posterize(input: &PythonImage, levels: u32, low: f32, high: f32) -> PyResult<PythonImage> {
	oa::image::posterize(&input.inner, levels, low, high)
		.map(PythonImage::wrap)
		.map_err(python_error)
}

#[pyfunction]
fn image_threshold_binary(
	input: &PythonImage,
	threshold: f32,
	max_value: f32,
) -> PyResult<PythonImage> {
	oa::image::threshold_binary(&input.inner, threshold, max_value)
		.map(PythonImage::wrap)
		.map_err(python_error)
}

#[pyfunction]
fn image_threshold_binary_inv(
	input: &PythonImage,
	threshold: f32,
	max_value: f32,
) -> PyResult<PythonImage> {
	oa::image::threshold_binary_inv(&input.inner, threshold, max_value)
		.map(PythonImage::wrap)
		.map_err(python_error)
}

#[pyfunction]
fn image_threshold_truncate(input: &PythonImage, threshold: f32) -> PyResult<PythonImage> {
	oa::image::threshold_truncate(&input.inner, threshold)
		.map(PythonImage::wrap)
		.map_err(python_error)
}

#[pyfunction]
fn image_threshold_to_zero(input: &PythonImage, threshold: f32) -> PyResult<PythonImage> {
	oa::image::threshold_to_zero(&input.inner, threshold)
		.map(PythonImage::wrap)
		.map_err(python_error)
}

#[pyfunction]
fn image_threshold_to_zero_inv(input: &PythonImage, threshold: f32) -> PyResult<PythonImage> {
	oa::image::threshold_to_zero_inv(&input.inner, threshold)
		.map(PythonImage::wrap)
		.map_err(python_error)
}

#[pyfunction]
fn image_in_range(
	input: &PythonImage,
	low: f32,
	high: f32,
	true_value: f32,
) -> PyResult<PythonImage> {
	oa::image::in_range(&input.inner, low, high, true_value)
		.map(PythonImage::wrap)
		.map_err(python_error)
}

#[pyfunction]
fn image_alpha_blend(a: &PythonImage, b: &PythonImage, alpha: f32) -> PyResult<PythonImage> {
	oa::image::alpha_blend(&a.inner, &b.inner, alpha)
		.map(PythonImage::wrap)
		.map_err(python_error)
}

#[pyfunction]
fn image_composite(a: &PythonImage, b: &PythonImage, mask: &PythonImage) -> PyResult<PythonImage> {
	oa::image::composite(&a.inner, &b.inner, &mask.inner)
		.map(PythonImage::wrap)
		.map_err(python_error)
}

#[pyfunction]
fn image_erase(
	input: &PythonImage,
	x: u32,
	y: u32,
	width: u32,
	height: u32,
	value: f32,
) -> PyResult<PythonImage> {
	oa::image::erase(&input.inner, x, y, width, height, value)
		.map(PythonImage::wrap)
		.map_err(python_error)
}

#[pyfunction]
fn image_color_twist(input: &PythonImage, transform: &PythonMatrix) -> PyResult<PythonImage> {
	oa::image::color_twist(&input.inner, &transform.inner)
		.map(PythonImage::wrap)
		.map_err(python_error)
}

#[pyfunction]
fn image_channel_reorder(
	input: &PythonImage,
	order: [u32; 4],
	format_name: &str,
) -> PyResult<PythonImage> {
	oa::image::channel_reorder(&input.inner, order, format(format_name)?)
		.map(PythonImage::wrap)
		.map_err(python_error)
}

#[pyfunction]
fn image_gaussian_noise(
	input: &PythonImage,
	mean: f32,
	stddev: f32,
	seed: u64,
) -> PyResult<PythonImage> {
	oa::image::gaussian_noise(&input.inner, mean, stddev, seed)
		.map(PythonImage::wrap)
		.map_err(python_error)
}

#[pyfunction]
fn image_salt_pepper_noise(
	input: &PythonImage,
	probability: f32,
	salt_value: f32,
	pepper_value: f32,
	seed: u64,
) -> PyResult<PythonImage> {
	oa::image::salt_pepper_noise(&input.inner, probability, salt_value, pepper_value, seed)
		.map(PythonImage::wrap)
		.map_err(python_error)
}

// ── Geometric ────────────────────────────────────────────────────────────────

#[pyfunction]
#[pyo3(signature = (input, map, interpolation_name="bilinear", border_name="constant", border_value=0.0))]
fn image_remap(
	input: &PythonImage,
	map: &PythonMatrix,
	interpolation_name: &str,
	border_name: &str,
	border_value: f32,
) -> PyResult<PythonImage> {
	oa::image::remap(
		&input.inner,
		&map.inner,
		interpolation(interpolation_name)?,
		border(border_name)?,
		border_value,
	)
	.map(PythonImage::wrap)
	.map_err(python_error)
}

#[pyfunction]
#[pyo3(signature = (input, transform, width, height, interpolation_name="bilinear", border_name="constant", border_value=0.0))]
#[allow(clippy::too_many_arguments)]
fn image_warp_affine(
	input: &PythonImage,
	transform: &PythonMatrix,
	width: u32,
	height: u32,
	interpolation_name: &str,
	border_name: &str,
	border_value: f32,
) -> PyResult<PythonImage> {
	oa::image::warp_affine(
		&input.inner,
		&transform.inner,
		width,
		height,
		interpolation(interpolation_name)?,
		border(border_name)?,
		border_value,
	)
	.map(PythonImage::wrap)
	.map_err(python_error)
}

#[pyfunction]
#[pyo3(signature = (input, transform, width, height, interpolation_name="bilinear", border_name="constant", border_value=0.0))]
#[allow(clippy::too_many_arguments)]
fn image_warp_perspective(
	input: &PythonImage,
	transform: &PythonMatrix,
	width: u32,
	height: u32,
	interpolation_name: &str,
	border_name: &str,
	border_value: f32,
) -> PyResult<PythonImage> {
	oa::image::warp_perspective(
		&input.inner,
		&transform.inner,
		width,
		height,
		interpolation(interpolation_name)?,
		border(border_name)?,
		border_value,
	)
	.map(PythonImage::wrap)
	.map_err(python_error)
}

// ── Color ─────────────────────────────────────────────────────────────────────

#[pyfunction]
fn image_resize_normalize(
	input: &PythonImage,
	width: u32,
	height: u32,
	mean: [f32; 3],
	std: [f32; 3],
) -> PyResult<PythonImage> {
	oa::image::resize_normalize(
		&input.inner,
		width,
		height,
		oa::image::NormalizationParams { mean, std },
	)
	.map(PythonImage::wrap)
	.map_err(python_error)
}

#[pyfunction]
fn image_segmentation_overlay(
	input: &PythonImage,
	mask: &PythonMatrix,
	palette: &PythonMatrix,
	alpha: f32,
) -> PyResult<PythonImage> {
	oa::image::segmentation_overlay(&input.inner, &mask.inner, &palette.inner, alpha)
		.map(PythonImage::wrap)
		.map_err(python_error)
}

// ── Codec helpers ─────────────────────────────────────────────────────────────

#[pyfunction]
#[pyo3(signature = (path, rgba, width, height, quality=90))]
fn image_save_rgba_file(
	path: &str,
	rgba: Vec<u8>,
	width: u32,
	height: u32,
	quality: u32,
) -> PyResult<()> {
	oa::image::save_rgba_file(&rgba, width, height, path, quality).map_err(python_error)
}

#[pyfunction]
fn image_can_decode(codec_name: &str) -> PyResult<bool> {
	Ok(oa::image::can_decode(codec(codec_name)?))
}

#[pyfunction]
fn image_can_encode(codec_name: &str) -> PyResult<bool> {
	Ok(oa::image::can_encode(codec(codec_name)?))
}

// ── Filter ────────────────────────────────────────────────────────────────────

#[pyfunction]
#[pyo3(signature = (input, sigma, kernel_size=0))]
fn image_gaussian_blur(input: &PythonImage, sigma: f32, kernel_size: u32) -> PyResult<PythonImage> {
	oa::image::gaussian_blur(&input.inner, sigma, kernel_size)
		.map(PythonImage::wrap)
		.map_err(python_error)
}

#[pyfunction]
#[pyo3(signature = (input, sigma, amount, kernel_size=0))]
fn image_unsharp_mask(
	input: &PythonImage,
	sigma: f32,
	amount: f32,
	kernel_size: u32,
) -> PyResult<PythonImage> {
	oa::image::unsharp_mask(&input.inner, sigma, amount, kernel_size)
		.map(PythonImage::wrap)
		.map_err(python_error)
}

#[pyfunction]
#[pyo3(signature = (input, width, height, border_name="constant"))]
fn image_average_blur(
	input: &PythonImage,
	width: u32,
	height: u32,
	border_name: &str,
) -> PyResult<PythonImage> {
	oa::image::average_blur(&input.inner, width, height, border(border_name)?)
		.map(PythonImage::wrap)
		.map_err(python_error)
}

#[pyfunction]
#[pyo3(signature = (input, kernel_size, border_name="constant"))]
fn image_median_blur(
	input: &PythonImage,
	kernel_size: u32,
	border_name: &str,
) -> PyResult<PythonImage> {
	oa::image::median_blur(&input.inner, kernel_size, border(border_name)?)
		.map(PythonImage::wrap)
		.map_err(python_error)
}

#[pyfunction]
#[pyo3(signature = (input, kernel_size, sigma_color, sigma_space, border_name="constant"))]
fn image_bilateral_filter(
	input: &PythonImage,
	kernel_size: u32,
	sigma_color: f32,
	sigma_space: f32,
	border_name: &str,
) -> PyResult<PythonImage> {
	oa::image::bilateral_filter(
		&input.inner,
		kernel_size,
		sigma_color,
		sigma_space,
		border(border_name)?,
	)
	.map(PythonImage::wrap)
	.map_err(python_error)
}

#[pyfunction]
#[pyo3(signature = (input, dx, dy, border_name="constant"))]
fn image_sobel(input: &PythonImage, dx: u32, dy: u32, border_name: &str) -> PyResult<PythonImage> {
	oa::image::sobel(&input.inner, dx, dy, border(border_name)?)
		.map(PythonImage::wrap)
		.map_err(python_error)
}

#[pyfunction]
#[pyo3(signature = (input, dx, dy, border_name="constant"))]
fn image_scharr(input: &PythonImage, dx: u32, dy: u32, border_name: &str) -> PyResult<PythonImage> {
	oa::image::scharr(&input.inner, dx, dy, border(border_name)?)
		.map(PythonImage::wrap)
		.map_err(python_error)
}

#[pyfunction]
#[pyo3(signature = (input, border_name="constant"))]
fn image_laplacian(input: &PythonImage, border_name: &str) -> PyResult<PythonImage> {
	oa::image::laplacian(&input.inner, border(border_name)?)
		.map(PythonImage::wrap)
		.map_err(python_error)
}

#[pyfunction]
#[pyo3(signature = (input, amount, border_name="constant"))]
fn image_sharpen(input: &PythonImage, amount: f32, border_name: &str) -> PyResult<PythonImage> {
	oa::image::sharpen(&input.inner, amount, border(border_name)?)
		.map(PythonImage::wrap)
		.map_err(python_error)
}

#[pyfunction]
#[pyo3(signature = (input, width, height, border_name="constant", border_value=0.0))]
fn image_erode(
	input: &PythonImage,
	width: u32,
	height: u32,
	border_name: &str,
	border_value: f32,
) -> PyResult<PythonImage> {
	oa::image::erode(
		&input.inner,
		width,
		height,
		border(border_name)?,
		border_value,
	)
	.map(PythonImage::wrap)
	.map_err(python_error)
}

#[pyfunction]
#[pyo3(signature = (input, width, height, border_name="constant", border_value=0.0))]
fn image_dilate(
	input: &PythonImage,
	width: u32,
	height: u32,
	border_name: &str,
	border_value: f32,
) -> PyResult<PythonImage> {
	oa::image::dilate(
		&input.inner,
		width,
		height,
		border(border_name)?,
		border_value,
	)
	.map(PythonImage::wrap)
	.map_err(python_error)
}

#[pyfunction]
#[pyo3(signature = (input, width, height, border_name="constant", border_value=0.0))]
fn image_morphology_open(
	input: &PythonImage,
	width: u32,
	height: u32,
	border_name: &str,
	border_value: f32,
) -> PyResult<PythonImage> {
	oa::image::morphology_open(
		&input.inner,
		width,
		height,
		border(border_name)?,
		border_value,
	)
	.map(PythonImage::wrap)
	.map_err(python_error)
}

#[pyfunction]
#[pyo3(signature = (input, width, height, border_name="constant", border_value=0.0))]
fn image_morphology_close(
	input: &PythonImage,
	width: u32,
	height: u32,
	border_name: &str,
	border_value: f32,
) -> PyResult<PythonImage> {
	oa::image::morphology_close(
		&input.inner,
		width,
		height,
		border(border_name)?,
		border_value,
	)
	.map(PythonImage::wrap)
	.map_err(python_error)
}

#[pyfunction]
#[pyo3(signature = (input, width, height, border_name="constant", border_value=0.0))]
fn image_morphology_gradient(
	input: &PythonImage,
	width: u32,
	height: u32,
	border_name: &str,
	border_value: f32,
) -> PyResult<PythonImage> {
	oa::image::morphology_gradient(
		&input.inner,
		width,
		height,
		border(border_name)?,
		border_value,
	)
	.map(PythonImage::wrap)
	.map_err(python_error)
}

#[pyfunction]
#[pyo3(signature = (input, width, height, border_name="constant", border_value=0.0))]
fn image_morphology_top_hat(
	input: &PythonImage,
	width: u32,
	height: u32,
	border_name: &str,
	border_value: f32,
) -> PyResult<PythonImage> {
	oa::image::morphology_top_hat(
		&input.inner,
		width,
		height,
		border(border_name)?,
		border_value,
	)
	.map(PythonImage::wrap)
	.map_err(python_error)
}

#[pyfunction]
#[pyo3(signature = (input, width, height, border_name="constant", border_value=0.0))]
fn image_morphology_black_hat(
	input: &PythonImage,
	width: u32,
	height: u32,
	border_name: &str,
	border_value: f32,
) -> PyResult<PythonImage> {
	oa::image::morphology_black_hat(
		&input.inner,
		width,
		height,
		border(border_name)?,
		border_value,
	)
	.map(PythonImage::wrap)
	.map_err(python_error)
}

#[pyfunction]
fn image_convolve_2d(
	input: &PythonImage,
	kernel: &PythonMatrix,
	border_name: &str,
	border_value: f32,
) -> PyResult<PythonImage> {
	oa::image::convolve_2d(
		&input.inner,
		&kernel.inner,
		border(border_name)?,
		border_value,
	)
	.map(PythonImage::wrap)
	.map_err(python_error)
}

#[pyfunction]
#[pyo3(signature = (input, kernel_x, kernel_y, border_name="constant", border_value=0.0))]
fn image_separable_convolve_2d(
	input: &PythonImage,
	kernel_x: &PythonMatrix,
	kernel_y: &PythonMatrix,
	border_name: &str,
	border_value: f32,
) -> PyResult<PythonImage> {
	oa::image::separable_convolve_2d(
		&input.inner,
		&kernel_x.inner,
		&kernel_y.inner,
		border(border_name)?,
		border_value,
	)
	.map(PythonImage::wrap)
	.map_err(python_error)
}

#[pyfunction]
#[pyo3(signature = (input, kernel_size, c, max_value, border_name="constant"))]
fn image_adaptive_threshold_mean(
	input: &PythonImage,
	kernel_size: u32,
	c: f32,
	max_value: f32,
	border_name: &str,
) -> PyResult<PythonImage> {
	oa::image::adaptive_threshold_mean(
		&input.inner,
		kernel_size,
		c,
		max_value,
		border(border_name)?,
	)
	.map(PythonImage::wrap)
	.map_err(python_error)
}

#[pyfunction]
#[pyo3(signature = (input, kernel_size, c, max_value, sigma, border_name="constant"))]
fn image_adaptive_threshold_gaussian(
	input: &PythonImage,
	kernel_size: u32,
	c: f32,
	max_value: f32,
	sigma: f32,
	border_name: &str,
) -> PyResult<PythonImage> {
	oa::image::adaptive_threshold_gaussian(
		&input.inner,
		kernel_size,
		c,
		max_value,
		sigma,
		border(border_name)?,
	)
	.map(PythonImage::wrap)
	.map_err(python_error)
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
	module.add_class::<PythonImage>()?;
	macro_rules! add_functions {
		($($function:ident),+ $(,)?) => { $(module.add_function(wrap_pyfunction!($function, module)?)?;)+ };
	}
	add_functions!(
		// codec
		image_decode_file,
		image_decode_memory,
		image_encode,
		image_save_file,
		image_save_rgba_file,
		image_can_decode,
		image_can_encode,
		// color
		image_resize,
		image_normalize,
		image_convert_color,
		image_resize_normalize,
		image_segmentation_overlay,
		// geometric
		image_crop,
		image_flip,
		image_rotate,
		image_center_crop,
		image_pad,
		image_remap,
		image_warp_affine,
		image_warp_perspective,
		// pixel
		image_grayscale,
		image_clamp,
		image_invert,
		image_brightness_contrast,
		image_gamma_contrast,
		image_solarize,
		image_posterize,
		image_threshold_binary,
		image_threshold_binary_inv,
		image_threshold_truncate,
		image_threshold_to_zero,
		image_threshold_to_zero_inv,
		image_in_range,
		image_alpha_blend,
		image_composite,
		image_erase,
		image_color_twist,
		image_channel_reorder,
		image_gaussian_noise,
		image_salt_pepper_noise,
		// filter
		image_gaussian_blur,
		image_unsharp_mask,
		image_average_blur,
		image_median_blur,
		image_bilateral_filter,
		image_sobel,
		image_scharr,
		image_laplacian,
		image_sharpen,
		image_erode,
		image_dilate,
		image_morphology_open,
		image_morphology_close,
		image_morphology_gradient,
		image_morphology_top_hat,
		image_morphology_black_hat,
		image_convolve_2d,
		image_separable_convolve_2d,
		image_adaptive_threshold_mean,
		image_adaptive_threshold_gaussian,
	);
	Ok(())
}
