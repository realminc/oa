//! OHLCV views adapted from `trade/include/oa/core/database.h` (OaDataOhlcv)
//! and `trade/lib/oa/core/dataframe.cpp` (FromOhlcvData). Explicit timestamp
//! units, full validation and borrowed windows replace donor copies/f64 time.

use super::{DataFrame, DataFrameView, Dataset, Sample, SeriesView};
use crate::{Engine, Error, Matrix, Result};
use std::ops::Range;

/// Unit of exact signed Unix timestamps; negative values precede the epoch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UnixTimeUnit {
	/// Seconds since 1970-01-01T00:00:00Z.
	Seconds,
	/// Milliseconds since the epoch.
	Milliseconds,
	/// Microseconds since the epoch.
	Microseconds,
	/// Nanoseconds since the epoch.
	Nanoseconds,
}

/// Validated borrowed candles. Prices and volume accept F32 or F64 columns;
/// timestamps are I64. Source columns remain immutable while this view exists.
#[derive(Clone, Copy, Debug)]
pub struct OhlcvView<'a> {
	timestamps: &'a [i64],
	unit: UnixTimeUnit,
	columns: [SeriesView<'a>; 5],
}

impl<'a> OhlcvView<'a> {
	/// Borrow timestamp/open/high/low/close/volume columns by their standard names.
	///
	/// # Errors
	/// Rejects missing/wrongly typed columns, non-finite values, negative volume,
	/// inconsistent candle bounds, or timestamps not strictly increasing.
	pub fn new_from_frame(frame: &'a DataFrame, unit: UnixTimeUnit) -> Result<Self> {
		Self::new_from_frame_view(frame.slice(0..frame.len())?, unit)
	}

	/// Validate and borrow candles inside one existing frame row partition.
	///
	/// # Errors
	/// Uses the same column, timestamp and candle validation as `from_frame`.
	pub fn new_from_frame_view(frame: DataFrameView<'a>, unit: UnixTimeUnit) -> Result<Self> {
		let SeriesView::I64(timestamps) = frame.column("timestamp")? else {
			return Err(Error::invalid_argument(
				"OHLCV timestamps must be exact I64 integers",
			));
		};
		let columns = [
			frame.column("open")?,
			frame.column("high")?,
			frame.column("low")?,
			frame.column("close")?,
			frame.column("volume")?,
		];
		if columns
			.iter()
			.any(|column| matches!(column, SeriesView::I64(_)))
		{
			return Err(Error::invalid_argument(
				"OHLCV prices and volume require F32 or F64 columns",
			));
		}
		if timestamps.windows(2).any(|pair| pair[0] >= pair[1]) {
			return Err(Error::invalid_argument(
				"OHLCV timestamps must be strictly increasing",
			));
		}
		for row in 0..timestamps.len() {
			let values = columns.map(|column| column.float_at(row).expect("frame lengths validated"));
			let [open, high, low, close, volume] = values;
			if values.iter().any(|v| !v.is_finite())
				|| volume < 0.0
				|| low > open
				|| low > close
				|| high < open
				|| high < close
				|| low > high
			{
				return Err(Error::invalid_argument(format!(
					"invalid OHLCV candle at row {row}"
				)));
			}
		}
		Ok(Self {
			timestamps,
			unit,
			columns,
		})
	}

	/// Exact timestamp storage.
	pub fn timestamps(self) -> &'a [i64] {
		self.timestamps
	}

	/// Timestamp unit, preserved by every view.
	pub fn time_unit(self) -> UnixTimeUnit {
		self.unit
	}

	/// Columns in open/high/low/close/volume order.
	pub fn columns(self) -> [SeriesView<'a>; 5] {
		self.columns
	}

	/// Number of candles.
	pub fn len(self) -> usize {
		self.timestamps.len()
	}

	/// Whether there are no candles.
	pub fn is_empty(self) -> bool {
		self.timestamps.is_empty()
	}

	/// Borrow a row interval without rescanning or copying validated candles.
	///
	/// # Errors
	/// Rejects reversed or out-of-bounds intervals.
	pub fn slice(self, range: Range<usize>) -> Result<Self> {
		if range.start > range.end || range.end > self.len() {
			return Err(Error::out_of_range("OHLCV row range is outside the source"));
		}
		let columns = [
			self.columns[0].slice(range.clone())?,
			self.columns[1].slice(range.clone())?,
			self.columns[2].slice(range.clone())?,
			self.columns[3].slice(range.clone())?,
			self.columns[4].slice(range.clone())?,
		];
		Ok(Self {
			timestamps: &self.timestamps[range],
			unit: self.unit,
			columns,
		})
	}

	/// Partition raw rows before creating windows, avoiding cross-split leakage.
	///
	/// # Errors
	/// Rejects boundaries outside the source or in descending order.
	pub fn chronological_split(self, train_end: usize, validation_end: usize) -> Result<[Self; 3]> {
		if train_end > validation_end || validation_end > self.len() {
			return Err(Error::out_of_range(
				"OHLCV split boundaries must be ordered and within rows",
			));
		}
		Ok([
			self.slice(0..train_end)?,
			self.slice(train_end..validation_end)?,
			self.slice(validation_end..self.len())?,
		])
	}
}

/// Window shape; targets are the immediately following future close prices.
#[derive(Clone, Copy, Debug)]
pub struct OhlcvWindowConfig {
	/// Number of history candles per input.
	pub lookback: usize,
	/// Number of future closes per label.
	pub horizon: usize,
	/// Advance between consecutive window starts.
	pub stride: usize,
}

/// Host-only borrowed sliding windows. Construct separately within each split.
#[derive(Clone, Copy, Debug)]
pub struct OhlcvWindows<'a> {
	source: OhlcvView<'a>,
	config: OhlcvWindowConfig,
	count: i64,
}

impl<'a> OhlcvWindows<'a> {
	/// Number of complete windows.
	pub fn len(&self) -> i64 {
		self.count
	}

	/// Whether no complete window fits.
	pub fn is_empty(&self) -> bool {
		self.count == 0
	}

	/// Construct checked windows over one chronological partition.
	///
	/// # Errors
	/// Rejects zero dimensions/stride or shape/count overflow.
	pub fn new(source: OhlcvView<'a>, config: OhlcvWindowConfig) -> Result<Self> {
		let span = config
			.lookback
			.checked_add(config.horizon)
			.ok_or_else(|| Error::invalid_argument("OHLCV window extent overflow"))?;
		if config.lookback == 0
			|| config.horizon == 0
			|| config.stride == 0
			|| config.lookback.checked_mul(5).is_none()
		{
			return Err(Error::invalid_argument(
				"OHLCV windows require positive checked extents and stride",
			));
		}
		let count = if source.len() < span {
			0
		} else {
			(source.len() - span) / config.stride + 1
		};
		let count = i64::try_from(count)
			.map_err(|_| Error::invalid_argument("OHLCV window count exceeds i64"))?;
		Ok(Self {
			source,
			config,
			count,
		})
	}

	/// Borrow input and future-label candle intervals without allocation.
	///
	/// # Errors
	/// Rejects negative/out-of-range indices.
	pub fn window(&self, index: i64) -> Result<(OhlcvView<'a>, OhlcvView<'a>)> {
		if index < 0 || index >= self.count {
			return Err(Error::out_of_range("OHLCV window index outside dataset"));
		}
		let start = index as usize * self.config.stride;
		let middle = start + self.config.lookback;
		Ok((
			self.source.slice(start..middle)?,
			self.source.slice(middle..middle + self.config.horizon)?,
		))
	}
}

/// Engine-borrowing adapter for the existing Dataset/DataLoader contract.
///
/// Host windows borrow columns. `get_sample` packs row-major FP32 [lookback, 5]
/// input and `horizon` future-close labels and uploads each once. Timestamps
/// remain metadata, never float features. No normalization or readback occurs.
/// Host windowing is zero-copy; packing and device upload are explicit copies.
pub struct DsOhlcv<'a> {
	engine: &'a Engine,
	windows: OhlcvWindows<'a>,
}

impl<'a> DsOhlcv<'a> {
	/// Attach an existing checked host window set to a borrowed Engine.
	pub fn new(engine: &'a Engine, windows: OhlcvWindows<'a>) -> Self {
		Self { engine, windows }
	}

	/// Borrow the host windows without triggering device work.
	pub fn windows(&self) -> &OhlcvWindows<'a> {
		&self.windows
	}
}

impl Dataset for DsOhlcv<'_> {
	fn len(&self) -> i64 {
		self.windows.len()
	}

	fn get_item(&self, index: i64) -> Result<Matrix> {
		let (input, _) = self.windows.window(index)?;
		let mut values = Vec::new();
		values
			.try_reserve_exact(input.len() * 5)
			.map_err(|_| Error::resource_exhausted("OHLCV feature packing allocation failed"))?;
		for row in 0..input.len() {
			for column in input.columns {
				let value = column.float_at(row).expect("validated float column") as f32;
				if !value.is_finite() {
					return Err(Error::invalid_argument("OHLCV value exceeds FP32 range"));
				}
				values.push(value);
			}
		}
		Matrix::new_from_slice(self.engine, vec![input.len(), 5], &values)
	}

	fn get_sample(&self, index: i64) -> Result<Sample> {
		let (_, future) = self.windows.window(index)?;
		let mut labels = Vec::new();
		labels
			.try_reserve_exact(future.len())
			.map_err(|_| Error::resource_exhausted("OHLCV label packing allocation failed"))?;
		for row in 0..future.len() {
			let value = future.columns[3]
				.float_at(row)
				.expect("validated close column") as f32;
			if !value.is_finite() {
				return Err(Error::invalid_argument("OHLCV close exceeds FP32 range"));
			}
			labels.push(value);
		}
		let x = self.get_item(index)?;
		let y = Matrix::new_from_slice(self.engine, vec![future.len()], &labels)?;
		Ok(Sample::new_with_label(x, y))
	}
}
