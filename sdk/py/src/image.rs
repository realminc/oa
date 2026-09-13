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
	inner: oa::Image,
}

impl PythonImage {
	fn wrap(inner: oa::Image) -> Self {
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
#[pyo3(signature = (input, sigma, kernel_size=0))]
fn image_gaussian_blur(input: &PythonImage, sigma: f32, kernel_size: u32) -> PyResult<PythonImage> {
	oa::image::gaussian_blur(&input.inner, sigma, kernel_size)
		.map(PythonImage::wrap)
		.map_err(python_error)
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
	module.add_class::<PythonImage>()?;
	macro_rules! add_functions {
		($($function:ident),+ $(,)?) => { $(module.add_function(wrap_pyfunction!($function, module)?)?;)+ };
	}
	add_functions!(
		image_decode_file,
		image_decode_memory,
		image_encode,
		image_save_file,
		image_resize,
		image_normalize,
		image_convert_color,
		image_crop,
		image_flip,
		image_rotate,
		image_center_crop,
		image_pad,
		image_grayscale,
		image_clamp,
		image_invert,
		image_brightness_contrast,
		image_gaussian_blur,
	);
	Ok(())
}
