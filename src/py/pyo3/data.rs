//! Python ownership adapters for native columnar values. Immutable snapshots
//! retain Rust buffers; Python list conversion and device packing are explicit.

use std::sync::Arc;

use oa::data::Dataset;
use pyo3::{exceptions::PyValueError, prelude::*};

use crate::{error::python_error, matrix::PythonMatrix, runtime::PythonEngine};

#[pyclass(name = "UnixTimeUnit", eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum PythonUnixTimeUnit {
	Seconds,
	Milliseconds,
	Microseconds,
	Nanoseconds,
}

impl From<PythonUnixTimeUnit> for oa::UnixTimeUnit {
	fn from(value: PythonUnixTimeUnit) -> Self {
		match value {
			PythonUnixTimeUnit::Seconds => Self::Seconds,
			PythonUnixTimeUnit::Milliseconds => Self::Milliseconds,
			PythonUnixTimeUnit::Microseconds => Self::Microseconds,
			PythonUnixTimeUnit::Nanoseconds => Self::Nanoseconds,
		}
	}
}

/// Ordered host columns. Edits copy shared snapshots only when necessary.
#[pyclass(name = "DataFrame")]
#[derive(Default)]
pub(crate) struct PythonDataFrame {
	inner: Arc<oa::DataFrame>,
}

#[pymethods]
impl PythonDataFrame {
	#[new]
	fn new() -> Self {
		Self::default()
	}

	/// Copy Python integer values once into exact native I64 storage.
	fn set_i64(&mut self, name: String, values: Vec<i64>) -> PyResult<()> {
		Arc::make_mut(&mut self.inner)
			.set_column(name, oa::Series::I64(values))
			.map_err(python_error)
	}

	/// Copy Python numbers once into native F32 storage.
	fn set_f32(&mut self, name: String, values: Vec<f32>) -> PyResult<()> {
		Arc::make_mut(&mut self.inner)
			.set_column(name, oa::Series::F32(values))
			.map_err(python_error)
	}

	fn set_f64(&mut self, name: String, values: Vec<f64>) -> PyResult<()> {
		Arc::make_mut(&mut self.inner)
			.set_column(name, oa::Series::F64(values))
			.map_err(python_error)
	}

	fn __len__(&self) -> usize {
		self.inner.len()
	}

	fn column_names(&self) -> Vec<String> {
		self.inner.column_names().map(str::to_owned).collect()
	}

	fn column(&self, name: &str, py: Python<'_>) -> PyResult<Py<PyAny>> {
		column_list(py, self.inner.column(name).map_err(python_error)?)
	}

	/// Retain a zero-copy immutable snapshot of a checked row interval.
	fn slice(&self, start: usize, end: usize) -> PyResult<PythonDataFrameView> {
		self.inner.slice(start..end).map_err(python_error)?;
		Ok(PythonDataFrameView {
			frame: Arc::clone(&self.inner),
			start,
			end,
		})
	}
}

/// Immutable snapshot: source deletion or edits cannot invalidate its buffers.
#[pyclass(name = "DataFrameView", frozen)]
#[derive(Clone)]
pub(crate) struct PythonDataFrameView {
	frame: Arc<oa::DataFrame>,
	start: usize,
	end: usize,
}

#[pymethods]
impl PythonDataFrameView {
	fn __len__(&self) -> usize {
		self.end - self.start
	}

	fn column(&self, name: &str, py: Python<'_>) -> PyResult<Py<PyAny>> {
		let view = self
			.frame
			.slice(self.start..self.end)
			.map_err(python_error)?;
		column_list(py, view.column(name).map_err(python_error)?)
	}

	fn slice(&self, start: usize, end: usize) -> PyResult<Self> {
		if start > end || end > self.__len__() {
			return Err(PyValueError::new_err("frame view range outside source"));
		}
		Ok(Self {
			frame: Arc::clone(&self.frame),
			start: self.start + start,
			end: self.start + end,
		})
	}
}

fn column_list(py: Python<'_>, column: oa::SeriesView<'_>) -> PyResult<Py<PyAny>> {
	Ok(match column {
		oa::SeriesView::I64(values) => values.into_pyobject(py)?.into_any().unbind(),
		oa::SeriesView::F32(values) => values.into_pyobject(py)?.into_any().unbind(),
		oa::SeriesView::F64(values) => values.into_pyobject(py)?.into_any().unbind(),
	})
}

/// Host-only windows retaining their immutable frame snapshot.
#[pyclass(name = "OhlcvWindows", frozen)]
#[derive(Clone)]
pub(crate) struct PythonOhlcvWindows {
	view: PythonDataFrameView,
	unit: PythonUnixTimeUnit,
	config: oa::OhlcvWindowConfig,
	count: usize,
}

impl PythonOhlcvWindows {
	fn native(&self) -> oa::Result<oa::OhlcvWindows<'_>> {
		let source = oa::OhlcvView::new_from_frame_view(
			self.view.frame.slice(self.view.start..self.view.end)?,
			self.unit.into(),
		)?;
		oa::OhlcvWindows::new(source, self.config)
	}

	fn bounds(&self, index: i64) -> PyResult<(usize, usize, usize)> {
		let index =
			usize::try_from(index).map_err(|_| PyValueError::new_err("negative OHLCV window index"))?;
		if index >= self.count {
			return Err(PyValueError::new_err("OHLCV window index outside dataset"));
		}
		// The native constructor already checked the source span and count.
		let start = self.view.start + index * self.config.stride;
		let middle = start + self.config.lookback;
		Ok((start, middle, middle + self.config.horizon))
	}

	fn native_window(&self, index: i64) -> PyResult<oa::OhlcvWindows<'_>> {
		let (start, _, end) = self.bounds(index)?;
		let source = oa::OhlcvView::new_from_frame_view(
			self.view.frame.slice(start..end).map_err(python_error)?,
			self.unit.into(),
		)
		.map_err(python_error)?;
		oa::OhlcvWindows::new(source, self.config).map_err(python_error)
	}

	fn build(
		view: PythonDataFrameView,
		unit: PythonUnixTimeUnit,
		lookback: usize,
		horizon: usize,
		stride: usize,
	) -> PyResult<Self> {
		let mut value = Self {
			view,
			unit,
			config: oa::OhlcvWindowConfig {
				lookback,
				horizon,
				stride,
			},
			count: 0,
		};
		value.count = usize::try_from(value.native().map_err(python_error)?.len())
			.map_err(|_| PyValueError::new_err("window count exceeds usize"))?;
		Ok(value)
	}
}

#[pymethods]
impl PythonOhlcvWindows {
	#[new]
	#[pyo3(signature = (frame, time_unit, lookback, horizon, stride=1))]
	fn new(
		frame: &PythonDataFrame,
		time_unit: PythonUnixTimeUnit,
		lookback: usize,
		horizon: usize,
		stride: usize,
	) -> PyResult<Self> {
		Self::build(
			PythonDataFrameView {
				frame: Arc::clone(&frame.inner),
				start: 0,
				end: frame.inner.len(),
			},
			time_unit,
			lookback,
			horizon,
			stride,
		)
	}

	/// Create windows inside an already separated chronological row partition.
	#[staticmethod]
	#[pyo3(signature = (view, time_unit, lookback, horizon, stride=1))]
	fn new_from_view(
		view: &PythonDataFrameView,
		time_unit: PythonUnixTimeUnit,
		lookback: usize,
		horizon: usize,
		stride: usize,
	) -> PyResult<Self> {
		Self::build(view.clone(), time_unit, lookback, horizon, stride)
	}

	fn __len__(&self) -> usize {
		self.count
	}

	#[getter]
	fn time_unit(&self) -> PythonUnixTimeUnit {
		self.unit
	}

	fn window(&self, index: i64) -> PyResult<(PythonDataFrameView, PythonDataFrameView)> {
		let (start, middle, end) = self.bounds(index)?;
		Ok((
			PythonDataFrameView {
				frame: Arc::clone(&self.view.frame),
				start,
				end: middle,
			},
			PythonDataFrameView {
				frame: Arc::clone(&self.view.frame),
				start: middle,
				end,
			},
		))
	}
}

/// Device adapter retaining the existing Engine, never a second runtime owner.
#[pyclass(name = "DsOhlcv", unsendable)]
pub(crate) struct PythonDsOhlcv {
	engine: Py<PythonEngine>,
	windows: PythonOhlcvWindows,
}

#[pymethods]
impl PythonDsOhlcv {
	#[new]
	fn new(engine: Py<PythonEngine>, windows: &PythonOhlcvWindows) -> Self {
		Self {
			engine,
			windows: windows.clone(),
		}
	}

	fn __len__(&self) -> usize {
		self.windows.count
	}

	fn get_item(&self, py: Python<'_>, index: i64) -> PyResult<PythonMatrix> {
		let engine = self.engine.borrow(py);
		let windows = self.windows.native_window(index)?;
		oa::DsOhlcv::new(&engine.inner, windows)
			.get_item(0)
			.map(PythonMatrix::wrap)
			.map_err(python_error)
	}

	fn get_sample(
		&self,
		py: Python<'_>,
		index: i64,
	) -> PyResult<(PythonMatrix, Option<PythonMatrix>)> {
		let engine = self.engine.borrow(py);
		let windows = self.windows.native_window(index)?;
		let sample = oa::DsOhlcv::new(&engine.inner, windows)
			.get_sample(0)
			.map_err(python_error)?;
		Ok((
			PythonMatrix::wrap(sample.x),
			sample.y.map(PythonMatrix::wrap),
		))
	}
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
	module.add_class::<PythonDataFrame>()?;
	module.add_class::<PythonDataFrameView>()?;
	module.add_class::<PythonUnixTimeUnit>()?;
	module.add_class::<PythonOhlcvWindows>()?;
	module.add_class::<PythonDsOhlcv>()?;
	Ok(())
}
