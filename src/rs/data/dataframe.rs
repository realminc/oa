//! Host columnar values adapted from `trade/include/oa/core/dataframe.h` and
//! `trade/lib/oa/core/dataframe.cpp`. Rust ownership adaptation; checked lengths
//! replace donor truncation. Slices borrow storage rather than copying it.

use crate::{Error, Result};
use std::ops::Range;

/// Contiguous typed host column. Moving it into a frame does not copy its data.
#[derive(Clone, Debug, PartialEq)]
pub enum Series {
	/// Exact signed integers, including Unix timestamps.
	I64(Vec<i64>),
	/// Single-precision numerical values.
	F32(Vec<f32>),
	/// Double-precision numerical values.
	F64(Vec<f64>),
}

/// Borrowed contiguous column; independent of an Engine.
#[derive(Clone, Copy, Debug)]
pub enum SeriesView<'a> {
	/// Signed integers.
	I64(&'a [i64]),
	/// Single-precision values.
	F32(&'a [f32]),
	/// Double-precision values.
	F64(&'a [f64]),
}

/// Exclusively borrowed column storage for in-place edits without resizing.
#[derive(Debug)]
pub enum SeriesViewMut<'a> {
	/// Signed integer storage.
	I64(&'a mut [i64]),
	/// Single-precision storage.
	F32(&'a mut [f32]),
	/// Double-precision storage.
	F64(&'a mut [f64]),
}

impl Series {
	/// Borrow the entire column without allocation.
	pub fn view(&self) -> SeriesView<'_> {
		match self {
			Self::I64(v) => SeriesView::I64(v),
			Self::F32(v) => SeriesView::F32(v),
			Self::F64(v) => SeriesView::F64(v),
		}
	}

	/// Exclusively borrow values for in-place edits without resizing the column.
	pub fn view_mut(&mut self) -> SeriesViewMut<'_> {
		match self {
			Self::I64(v) => SeriesViewMut::I64(v),
			Self::F32(v) => SeriesViewMut::F32(v),
			Self::F64(v) => SeriesViewMut::F64(v),
		}
	}

	/// Number of rows.
	pub fn len(&self) -> usize {
		self.view().len()
	}

	/// Whether this column has no rows.
	pub fn is_empty(&self) -> bool {
		self.len() == 0
	}
}

impl<'a> SeriesView<'a> {
	/// Number of rows.
	pub fn len(self) -> usize {
		match self {
			Self::I64(v) => v.len(),
			Self::F32(v) => v.len(),
			Self::F64(v) => v.len(),
		}
	}

	/// Whether the view is empty.
	pub fn is_empty(self) -> bool {
		self.len() == 0
	}

	/// Borrow a checked row range. No allocation or copy.
	///
	/// # Errors
	/// Returns an error for reversed or out-of-bounds ranges.
	pub fn slice(self, range: Range<usize>) -> Result<Self> {
		if range.start > range.end || range.end > self.len() {
			return Err(Error::out_of_range(
				"series row range is outside the column",
			));
		}
		Ok(match self {
			Self::I64(v) => Self::I64(&v[range]),
			Self::F32(v) => Self::F32(&v[range]),
			Self::F64(v) => Self::F64(&v[range]),
		})
	}

	pub(super) fn float_at(self, row: usize) -> Option<f64> {
		match self {
			Self::I64(_) => None,
			Self::F32(v) => v.get(row).map(|&v| f64::from(v)),
			Self::F64(v) => v.get(row).copied(),
		}
	}
}

/// Ordered named host columns with one checked row count.
///
/// ```
/// let mut frame = oa::DataFrame::new();
/// frame.set_column("timestamp", oa::Series::I64(vec![0, 1]))?;
/// frame.set_column("close", oa::Series::F64(vec![10.0, 11.0]))?;
/// let last = frame.slice(1..2)?;
/// assert_eq!(last.len(), 1);
/// # Ok::<(), oa::Error>(())
/// ```
///
/// Unlike the donor, missing columns and unequal lengths are errors. Whole-column replacement
/// is transactional; exclusive mutable slices cannot invalidate lengths.
#[derive(Clone, Debug, Default)]
pub struct DataFrame {
	columns: Vec<(String, Series)>,
	rows: usize,
}

impl DataFrame {
	/// Create an empty frame.
	pub fn new() -> Self {
		Self::default()
	}

	/// Insert or replace a column, moving its buffer without copying.
	///
	/// # Errors
	/// Rejects empty names and lengths differing from existing columns.
	pub fn set_column(&mut self, name: impl Into<String>, series: Series) -> Result<()> {
		let name = name.into();
		if name.is_empty() || (!self.columns.is_empty() && series.len() != self.rows) {
			return Err(Error::invalid_argument(
				"frame columns require a name and equal row counts",
			));
		}
		self.rows = series.len();
		if let Some((_, column)) = self.columns.iter_mut().find(|(n, _)| *n == name) {
			*column = series;
		} else {
			self.columns.push((name, series));
		}
		Ok(())
	}

	/// Borrow a named column.
	///
	/// # Errors
	/// Returns an error when the column does not exist.
	pub fn column(&self, name: &str) -> Result<SeriesView<'_>> {
		self
			.columns
			.iter()
			.find(|(n, _)| n == name)
			.map(|(_, c)| c.view())
			.ok_or_else(|| Error::invalid_argument(format!("missing frame column: {name}")))
	}

	/// Exclusively borrow column values for in-place edits. Length cannot change.
	/// Existing immutable frame/OHLCV views prevent this borrow in safe Rust.
	///
	/// # Errors
	/// Returns an error when the column does not exist.
	pub fn column_mut(&mut self, name: &str) -> Result<SeriesViewMut<'_>> {
		self
			.columns
			.iter_mut()
			.find(|(n, _)| n == name)
			.map(|(_, c)| c.view_mut())
			.ok_or_else(|| Error::invalid_argument(format!("missing frame column: {name}")))
	}

	/// Column names in insertion order.
	pub fn column_names(&self) -> impl ExactSizeIterator<Item = &str> {
		self.columns.iter().map(|(n, _)| n.as_str())
	}

	/// Number of rows.
	pub fn len(&self) -> usize {
		self.rows
	}

	/// Whether the frame has no rows.
	pub fn is_empty(&self) -> bool {
		self.rows == 0
	}
}

/// Borrowed row interval over a frame. Columns retain their original buffers.
#[derive(Clone, Copy, Debug)]
pub struct DataFrameView<'a> {
	frame: &'a DataFrame,
	start: usize,
	end: usize,
}

impl DataFrame {
	/// Borrow a checked row interval without allocating columns or metadata.
	///
	/// # Errors
	/// Rejects reversed or out-of-bounds ranges.
	pub fn slice(&self, range: Range<usize>) -> Result<DataFrameView<'_>> {
		if range.start > range.end || range.end > self.rows {
			return Err(Error::out_of_range("frame row range is outside the source"));
		}
		Ok(DataFrameView {
			frame: self,
			start: range.start,
			end: range.end,
		})
	}
}

impl<'a> DataFrameView<'a> {
	/// Number of rows in this view.
	pub fn len(self) -> usize {
		self.end - self.start
	}

	/// Whether this view is empty.
	pub fn is_empty(self) -> bool {
		self.start == self.end
	}

	/// Borrow a column restricted to the view's rows.
	///
	/// # Errors
	/// Returns an error when the column does not exist.
	pub fn column(self, name: &str) -> Result<SeriesView<'a>> {
		self.frame.column(name)?.slice(self.start..self.end)
	}
}
