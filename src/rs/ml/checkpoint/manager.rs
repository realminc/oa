//! Native `.oam` checkpoint path, metric, and rotation policy.

use std::{
	fs,
	path::{Path, PathBuf},
};

use crate::{Engine, Error, Result};

use super::{
	CheckpointProgress, load_checkpoint, load_checkpoint_at_step, save_checkpoint_with_progress,
};
use crate::ml::model_file::ModelFile;
use crate::ml::{CheckpointOptimizer, Module};

/// Filesystem and metric policy for [`CheckpointManager`].
#[derive(Clone, Debug)]
pub struct CheckpointManagerConfig {
	/// Root directory beneath which the model directory is created.
	pub directory: PathBuf,
	/// Stable filename component and master checkpoint stem.
	pub model_name: String,
	/// Optional incremental-checkpoint directory/filename component.
	pub context: String,
	/// Maximum retained incremental checkpoints; zero disables rotation.
	pub max_keep: usize,
	/// Whether an improved metric updates the master checkpoint.
	pub save_best: bool,
	/// Metric name written into progress metadata and filenames.
	pub metric_name: String,
	/// Whether a lower metric is considered better.
	pub lower_is_better: bool,
}

impl Default for CheckpointManagerConfig {
	fn default() -> Self {
		Self {
			directory: PathBuf::from("var/model/dev"),
			model_name: "Module".to_owned(),
			context: String::new(),
			max_keep: 5,
			save_best: true,
			metric_name: "loss".to_owned(),
			lower_is_better: true,
		}
	}
}

#[derive(Clone, Debug)]
struct SavedCheckpoint {
	path: PathBuf,
	step: u64,
}

/// Stateful checkpoint selection and bounded incremental-retention policy.
pub struct CheckpointManager<'engine> {
	engine: &'engine Engine,
	config: CheckpointManagerConfig,
	best_metric: f64,
	saved: Vec<SavedCheckpoint>,
}

impl<'engine> CheckpointManager<'engine> {
	/// Construct a manager around the one execution owner.
	///
	/// # Errors
	///
	/// Returns an error when a filename component or metric name is invalid.
	pub fn new(engine: &'engine Engine, config: CheckpointManagerConfig) -> Result<Self> {
		validate_component(&config.model_name, "model name")?;
		if !config.context.is_empty() {
			validate_component(&config.context, "checkpoint context")?;
		}
		validate_component(&config.metric_name, "checkpoint metric name")?;
		let best_metric = if config.lower_is_better {
			f64::INFINITY
		} else {
			f64::NEG_INFINITY
		};
		Ok(Self {
			engine,
			config,
			best_metric,
			saved: Vec::new(),
		})
	}

	/// Return the directory containing master and incremental checkpoints.
	pub fn model_directory(&self) -> PathBuf {
		self.config.directory.join(&self.config.model_name)
	}

	/// Return the incremental checkpoint directory.
	pub fn incremental_directory(&self) -> PathBuf {
		let name = if self.config.context.is_empty() {
			"checkpoint".to_owned()
		} else {
			format!("checkpoint_{}", self.config.context)
		};
		self.model_directory().join(name)
	}

	/// Return the master/best checkpoint path.
	pub fn master_path(&self) -> PathBuf {
		self
			.model_directory()
			.join(format!("{}.oam", self.config.model_name))
	}

	/// Return whether `metric` improves the currently retained best value.
	pub fn is_better(&self, metric: f64) -> bool {
		if self.config.lower_is_better {
			metric < self.best_metric
		} else {
			metric > self.best_metric
		}
	}

	/// Return the best metric admitted in this manager session.
	pub const fn best_metric(&self) -> f64 {
		self.best_metric
	}

	/// Return the configured metric name.
	pub fn metric_name(&self) -> &str {
		&self.config.metric_name
	}

	/// Save an incremental checkpoint and update the master file on improvement.
	///
	/// When `force` is false, a non-improving metric performs no write. When it is
	/// true, the incremental file is still written but the master changes only on
	/// improvement.
	///
	/// # Errors
	///
	/// Returns an error for non-finite metrics, optimizer/progress mismatch,
	/// serialization, directory creation, rotation, or filesystem failure.
	pub fn maybe_save(
		&mut self,
		model: &dyn Module,
		optimizer: &dyn CheckpointOptimizer,
		step: u64,
		metric: f64,
		force: bool,
	) -> Result<bool> {
		validate_metric_step(optimizer, step, metric)?;
		let improved = self.is_better(metric);
		if !improved && !force {
			return Ok(false);
		}
		self.save_incremental(model, optimizer, step, metric, None)?;
		if improved {
			if self.config.save_best {
				fs::create_dir_all(self.model_directory())
					.map_err(|source| Error::io("create checkpoint model directory", source))?;
				save_checkpoint_with_progress(
					&self.master_path(),
					model,
					optimizer,
					Some(self.progress(step, metric, &self.config.metric_name)),
				)?;
			}
			self.best_metric = metric;
		}
		Ok(improved)
	}

	/// Save a resumable incremental checkpoint without changing best selection.
	///
	/// # Errors
	///
	/// Returns an error for invalid metric/progress state, serialization,
	/// directory creation, rotation, or filesystem failure.
	pub fn save_incremental(
		&mut self,
		model: &dyn Module,
		optimizer: &dyn CheckpointOptimizer,
		step: u64,
		metric: f64,
		metric_name: Option<&str>,
	) -> Result<PathBuf> {
		validate_metric_step(optimizer, step, metric)?;
		let metric_name = metric_name.unwrap_or(&self.config.metric_name);
		validate_component(metric_name, "checkpoint metric name")?;
		let directory = self.incremental_directory();
		fs::create_dir_all(&directory)
			.map_err(|source| Error::io("create incremental checkpoint directory", source))?;
		let filename = self.filename(step, metric, metric_name);
		let path = directory.join(filename);
		save_checkpoint_with_progress(
			&path,
			model,
			optimizer,
			Some(self.progress(step, metric, metric_name)),
		)?;
		self.saved.retain(|saved| saved.path != path);
		self.saved.push(SavedCheckpoint {
			path: path.clone(),
			step,
		});
		self.rotate()?;
		Ok(path)
	}

	/// Restore the master/best checkpoint into existing owners.
	///
	/// # Errors
	///
	/// Returns the native load, validation, allocation, or filesystem error.
	pub fn load_best_into(
		&self,
		model: &dyn Module,
		optimizer: &mut dyn CheckpointOptimizer,
	) -> Result<()> {
		load_checkpoint(self.engine, self.master_path(), model, optimizer)
	}

	/// Restore the highest-step incremental checkpoint into existing owners.
	///
	/// The manager prefers files recorded in this session and otherwise scans its
	/// incremental directory using the same strict filename grammar it writes.
	/// The returned value is the exact completed step restored into the optimizer.
	///
	/// # Errors
	///
	/// Returns an error when no checkpoint exists or loading fails.
	pub fn load_latest_into(
		&self,
		model: &dyn Module,
		optimizer: &mut dyn CheckpointOptimizer,
	) -> Result<u64> {
		let path = self.latest_path()?.ok_or_else(|| {
			Error::failed_precondition(format!(
				"no checkpoints in {}",
				self.incremental_directory().display()
			))
		})?;
		let step = parse_step(&path)
			.ok_or_else(|| Error::failed_precondition("latest checkpoint has no valid step segment"))?;
		load_checkpoint_at_step(self.engine, &path, model, optimizer, step)?;
		Ok(step)
	}

	/// Restore the latest incremental state and recover historical best-metric
	/// selection from the master checkpoint when it exists.
	///
	/// This joins model, optimizer, completed step, and best-checkpoint policy at
	/// one restart boundary. A master file written for another metric or ordering
	/// policy is rejected instead of silently resetting selection history.
	///
	/// # Errors
	///
	/// Returns the same errors as [`Self::load_latest_into`], plus malformed or
	/// incompatible master progress metadata.
	pub fn resume_latest_into(
		&mut self,
		model: &dyn Module,
		optimizer: &mut dyn CheckpointOptimizer,
	) -> Result<u64> {
		let master = self.master_path();
		let restored_best = if master.is_file() {
			let file = ModelFile::load(&master)?;
			if file.progress.metric_name != self.config.metric_name
				|| file.progress.lower_is_better != self.config.lower_is_better
				|| !file.progress.best_metric.is_finite()
			{
				return Err(Error::invalid_argument(
					"master checkpoint progress does not match resume policy",
				));
			}
			Some(f64::from(file.progress.best_metric))
		} else {
			None
		};
		let step = self.load_latest_into(model, optimizer)?;
		if let Some(best) = restored_best {
			self.best_metric = best;
		}
		Ok(step)
	}

	fn progress(&self, step: u64, metric: f64, metric_name: &str) -> CheckpointProgress {
		CheckpointProgress {
			step,
			metric,
			metric_name: metric_name.to_owned(),
			lower_is_better: self.config.lower_is_better,
		}
	}

	fn filename(&self, step: u64, metric: f64, metric_name: &str) -> String {
		if self.config.context.is_empty() {
			format!(
				"{}_step{step}_{metric_name}{metric:.2}.oam",
				self.config.model_name
			)
		} else {
			format!(
				"{}_{}_step{step}_{metric_name}{metric:.2}.oam",
				self.config.model_name, self.config.context
			)
		}
	}

	fn rotate(&mut self) -> Result<()> {
		if self.config.max_keep == 0 || self.saved.len() <= self.config.max_keep {
			return Ok(());
		}
		self.saved.sort_unstable_by(|left, right| {
			right
				.step
				.cmp(&left.step)
				.then_with(|| right.path.cmp(&left.path))
		});
		while self.saved.len() > self.config.max_keep {
			let removed = self.saved.pop().expect("retention excess exists");
			if let Err(source) = fs::remove_file(&removed.path) {
				self.saved.push(removed);
				return Err(Error::io("remove rotated checkpoint", source));
			}
		}
		Ok(())
	}

	fn latest_path(&self) -> Result<Option<PathBuf>> {
		if let Some(saved) = self.saved.iter().max_by_key(|saved| saved.step) {
			return Ok(Some(saved.path.clone()));
		}
		let directory = self.incremental_directory();
		let entries = match fs::read_dir(&directory) {
			Ok(entries) => entries,
			Err(source) if source.kind() == std::io::ErrorKind::NotFound => return Ok(None),
			Err(source) => return Err(Error::io("scan checkpoint directory", source)),
		};
		let mut latest: Option<(u64, PathBuf)> = None;
		for entry in entries {
			let entry = entry.map_err(|source| Error::io("read checkpoint directory", source))?;
			let path = entry.path();
			if path.extension().and_then(|value| value.to_str()) != Some("oam") {
				continue;
			}
			let Some(step) = parse_step(&path) else {
				continue;
			};
			if latest.as_ref().is_none_or(|(current, _)| step > *current) {
				latest = Some((step, path));
			}
		}
		Ok(latest.map(|(_, path)| path))
	}
}

fn validate_component(value: &str, name: &'static str) -> Result<()> {
	if value.is_empty()
		|| value == "."
		|| value == ".."
		|| !value
			.bytes()
			.all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-' | b'.'))
	{
		return Err(Error::invalid_argument(format!(
			"{name} must be a nonempty portable filename component"
		)));
	}
	Ok(())
}

fn validate_metric_step(optimizer: &dyn CheckpointOptimizer, step: u64, metric: f64) -> Result<()> {
	if !metric.is_finite() || !(metric as f32).is_finite() {
		return Err(Error::invalid_argument("checkpoint metric must be finite"));
	}
	if optimizer.step_count() != step {
		return Err(Error::invalid_argument(
			"checkpoint step does not match optimizer state",
		));
	}
	Ok(())
}

fn parse_step(path: &Path) -> Option<u64> {
	let stem = path.file_stem()?.to_str()?;
	let marker = stem.rfind("_step")?;
	let digits = stem[marker + 5..].split('_').next()?;
	(!digits.is_empty()).then(|| digits.parse().ok()).flatten()
}

#[cfg(test)]
mod tests {
	use super::{parse_step, validate_component};
	use std::path::Path;

	#[test]
	fn parses_only_the_written_step_segment() {
		assert_eq!(parse_step(Path::new("model_step42_loss0.50.oam")), Some(42));
		assert_eq!(parse_step(Path::new("model_step_loss0.50.oam")), None);
		assert_eq!(parse_step(Path::new("model_loss0.50.oam")), None);
	}

	#[test]
	fn rejects_path_components_in_manager_names() {
		assert!(validate_component("model-v1", "model name").is_ok());
		assert!(validate_component("../escape", "model name").is_err());
		assert!(validate_component("", "model name").is_err());
	}
}
