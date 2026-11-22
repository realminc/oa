//! Exact completed-step CSV training log.
//!
//! Port provenance: `oa/ml/callbacks.h` and `oa/ml/callbacks.cpp`.

use std::{
	fs::File,
	io::Write,
	path::{Path, PathBuf},
};

use crate::{Error, Result};

use super::super::{TrainingCallback, TrainingCallbackContext, TrainingControl};

const HEADER: &str = concat!(
	"epoch,step,batch_size,sequence_length,sequence_unit,source_unit,",
	"loss,gpu_ms,wall_ms_per_step,wall_samples_per_second,",
	"gpu_samples_per_second,wall_units_per_second,gpu_units_per_second,",
	"wall_source_units_per_second,gpu_source_units_per_second\n",
);
const FLUSH_THRESHOLD: usize = 64 * 1024;

/// Buffered one-row-per-completed-step CSV logger.
///
/// File creation, writes, and close-time flushing happen only at explicit
/// callback boundaries. Abandoning the callback discards any unflushed host
/// buffer; `Drop` does not hide an I/O failure.
pub struct CsvLogger {
	path: PathBuf,
	file: Option<File>,
	buffer: Vec<u8>,
}

impl CsvLogger {
	/// Construct a logger that truncates `path` when training begins.
	pub fn new(path: impl Into<PathBuf>) -> Self {
		Self {
			path: path.into(),
			file: None,
			buffer: Vec::with_capacity(FLUSH_THRESHOLD),
		}
	}

	/// Return the configured output path.
	pub fn path(&self) -> &Path {
		&self.path
	}

	fn begin(&mut self) -> Result<()> {
		self.file = None;
		self.buffer.clear();
		let file = File::create(&self.path).map_err(|error| Error::io("create training CSV", error))?;
		self.file = Some(file);
		self.buffer.extend_from_slice(HEADER.as_bytes());
		Ok(())
	}

	fn append_step(&mut self, context: &TrainingCallbackContext<'_>) -> Result<()> {
		if self.file.is_none() {
			return Err(Error::callback(
				"training CSV received a step before train begin",
			));
		}
		let snapshot = context.snapshot();
		let config = context.config();
		let loss = snapshot.last_loss().unwrap_or(f32::NAN);
		let gpu_ms = snapshot
			.last_gpu_time()
			.map_or(f64::NAN, |duration| duration.as_secs_f64() * 1000.0);
		let sequence_unit = csv_field(&config.sequence_unit);
		let source_unit = csv_field(&config.source_unit);
		let row = format!(
			"{},{},{},{},{},{},{loss},{gpu_ms},{},{},{},{},{},{},{}\n",
			snapshot.epoch(),
			snapshot.step_count(),
			config.batch_size,
			config.sequence_length,
			sequence_unit,
			source_unit,
			snapshot.wall_ms_per_step(),
			snapshot.wall_samples_per_second(),
			snapshot.gpu_samples_per_second(),
			snapshot.wall_units_per_second(),
			snapshot.gpu_units_per_second(),
			snapshot.wall_source_units_per_second(),
			snapshot.gpu_source_units_per_second(),
		);
		self.buffer.extend_from_slice(row.as_bytes());
		if self.buffer.len() >= FLUSH_THRESHOLD {
			self.flush()?;
		}
		Ok(())
	}

	fn flush(&mut self) -> Result<()> {
		if self.buffer.is_empty() {
			return Ok(());
		}
		let file = self
			.file
			.as_mut()
			.ok_or_else(|| Error::callback("training CSV is not open"))?;
		file
			.write_all(&self.buffer)
			.map_err(|error| Error::io("write training CSV", error))?;
		file
			.flush()
			.map_err(|error| Error::io("flush training CSV", error))?;
		self.buffer.clear();
		Ok(())
	}

	fn finish(&mut self) -> Result<()> {
		self.flush()?;
		self.file = None;
		Ok(())
	}
}

impl TrainingCallback for CsvLogger {
	fn on_train_begin(
		&mut self,
		_context: &mut TrainingCallbackContext<'_>,
	) -> Result<TrainingControl> {
		self.begin()?;
		Ok(TrainingControl::Continue)
	}

	fn on_step_end(&mut self, context: &mut TrainingCallbackContext<'_>) -> Result<TrainingControl> {
		self.append_step(context)?;
		Ok(TrainingControl::Continue)
	}

	fn on_epoch_end(
		&mut self,
		_context: &mut TrainingCallbackContext<'_>,
	) -> Result<TrainingControl> {
		self.flush()?;
		Ok(TrainingControl::Continue)
	}

	fn on_train_end(
		&mut self,
		_context: &mut TrainingCallbackContext<'_>,
	) -> Result<TrainingControl> {
		self.finish()?;
		Ok(TrainingControl::Continue)
	}
}

/// OA C++ compatibility spelling for [`CsvLogger`].
pub type CbCsvLogger = CsvLogger;

fn csv_field(value: &str) -> String {
	if !value
		.bytes()
		.any(|byte| matches!(byte, b',' | b'"' | b'\n' | b'\r'))
	{
		return value.to_owned();
	}
	format!("\"{}\"", value.replace('"', "\"\""))
}
