//! Donor-backed learning-rate schedules.
//!
//! Port provenance: `oa/ml/lrScheduler.h` and
//! `oa/ml/callback/lrScheduler.cpp`.

use std::f32::consts::PI;

use crate::{Error, Result};

/// Stateless step-to-learning-rate policy.
pub trait LrScheduler {
	/// Return the learning rate for `step`.
	fn learning_rate(&self, step: u64) -> f32;
}

/// Cosine annealing from a maximum rate to a minimum rate.
pub struct CosineScheduler {
	max_learning_rate: f32,
	min_learning_rate: f32,
	total_steps: u64,
}

impl CosineScheduler {
	/// Construct a cosine schedule.
	///
	/// # Errors
	///
	/// Returns an error for non-finite or negative rates, an inverted range, or
	/// a zero step span.
	pub fn new(max_learning_rate: f32, min_learning_rate: f32, total_steps: u64) -> Result<Self> {
		validate_rate(max_learning_rate, "maximum")?;
		validate_rate(min_learning_rate, "minimum")?;
		if min_learning_rate > max_learning_rate {
			return Err(Error::invalid_argument(
				"cosine minimum learning rate exceeds its maximum",
			));
		}
		if total_steps == 0 {
			return Err(Error::invalid_argument(
				"cosine schedule requires at least one step",
			));
		}
		Ok(Self {
			max_learning_rate,
			min_learning_rate,
			total_steps,
		})
	}
}

impl LrScheduler for CosineScheduler {
	fn learning_rate(&self, step: u64) -> f32 {
		if step >= self.total_steps {
			return self.min_learning_rate;
		}
		let progress = step as f32 / self.total_steps as f32;
		self.min_learning_rate
			+ 0.5
				* (self.max_learning_rate - self.min_learning_rate)
				* (1.0 + (progress * PI).cos())
	}
}

/// Linear warmup followed by an optional delegated schedule.
pub struct WarmupScheduler {
	target_learning_rate: f32,
	warmup_steps: u64,
	after: Option<Box<dyn LrScheduler>>,
}

impl WarmupScheduler {
	/// Construct a warmup that remains at its target after the warmup span.
	///
	/// # Errors
	///
	/// Returns an error when the target rate is non-finite or negative.
	pub fn new(target_learning_rate: f32, warmup_steps: u64) -> Result<Self> {
		Self::with_optional_after(target_learning_rate, warmup_steps, None)
	}

	/// Construct a warmup followed by `after` with a step origin of zero.
	///
	/// # Errors
	///
	/// Returns an error when the target rate is non-finite or negative.
	pub fn with_after(
		target_learning_rate: f32,
		warmup_steps: u64,
		after: Box<dyn LrScheduler>,
	) -> Result<Self> {
		Self::with_optional_after(target_learning_rate, warmup_steps, Some(after))
	}

	fn with_optional_after(
		target_learning_rate: f32,
		warmup_steps: u64,
		after: Option<Box<dyn LrScheduler>>,
	) -> Result<Self> {
		validate_rate(target_learning_rate, "warmup target")?;
		Ok(Self {
			target_learning_rate,
			warmup_steps,
			after,
		})
	}
}

impl LrScheduler for WarmupScheduler {
	fn learning_rate(&self, step: u64) -> f32 {
		if step < self.warmup_steps {
			return self.target_learning_rate * (step + 1) as f32 / self.warmup_steps as f32;
		}
		self.after
			.as_ref()
			.map_or(self.target_learning_rate, |after| {
				after.learning_rate(step - self.warmup_steps)
			})
	}
}

/// Smith one-cycle ramp and cosine decay policy.
pub struct OneCycleScheduler {
	max_learning_rate: f32,
	total_steps: u64,
	percent_start: f32,
	division_factor: f32,
	final_division_factor: f32,
}

impl OneCycleScheduler {
	/// Construct a one-cycle schedule with OA's 0.3/25/10,000 defaults.
	///
	/// # Errors
	///
	/// Returns an error for an invalid maximum rate or empty span.
	pub fn with_defaults(max_learning_rate: f32, total_steps: u64) -> Result<Self> {
		Self::new(max_learning_rate, total_steps, 0.3, 25.0, 10_000.0)
	}

	/// Construct the donor-compatible one-cycle schedule.
	///
	/// # Errors
	///
	/// Returns an error for an invalid rate, empty span, percentage outside
	/// `[0, 1]`, or non-positive division factor.
	pub fn new(
		max_learning_rate: f32,
		total_steps: u64,
		percent_start: f32,
		division_factor: f32,
		final_division_factor: f32,
	) -> Result<Self> {
		validate_rate(max_learning_rate, "one-cycle maximum")?;
		if total_steps == 0 {
			return Err(Error::invalid_argument(
				"one-cycle schedule requires at least one step",
			));
		}
		if !percent_start.is_finite() || !(0.0..=1.0).contains(&percent_start) {
			return Err(Error::invalid_argument(
				"one-cycle start percentage must be finite and in [0, 1]",
			));
		}
		for (name, value) in [
			("division", division_factor),
			("final division", final_division_factor),
		] {
			if !value.is_finite() || value <= 0.0 {
				return Err(Error::invalid_argument(format!(
					"one-cycle {name} factor must be finite and positive"
				)));
			}
		}
		Ok(Self {
			max_learning_rate,
			total_steps,
			percent_start,
			division_factor,
			final_division_factor,
		})
	}
}

impl LrScheduler for OneCycleScheduler {
	fn learning_rate(&self, step: u64) -> f32 {
		let initial = self.max_learning_rate / self.division_factor;
		let final_rate = initial / self.final_division_factor;
		let step = step as f32;
		let total = self.total_steps as f32;
		let up_steps = self.percent_start * total;
		let down_steps = total - up_steps;
		if step <= up_steps && up_steps > 0.0 {
			let progress = step / up_steps;
			return final_rate
				+ 0.5 * (self.max_learning_rate - final_rate) * (1.0 - (PI * progress).cos());
		}
		if down_steps > 0.0 {
			let progress = ((step - up_steps) / down_steps).min(1.0);
			return final_rate
				+ 0.5 * (self.max_learning_rate - final_rate) * (1.0 + (PI * progress).cos());
		}
		final_rate
	}
}

/// Amplitude policy for [`CyclicScheduler`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CyclicMode {
	/// Constant triangular amplitude.
	#[default]
	Triangular,
	/// Halve amplitude after each cycle.
	Triangular2,
	/// Multiply amplitude by `gamma.powf(step)`.
	ExpRange,
}

/// Triangular cyclic learning-rate policy.
pub struct CyclicScheduler {
	base_learning_rate: f32,
	max_learning_rate: f32,
	step_size_up: u64,
	mode: CyclicMode,
	gamma: f32,
}

impl CyclicScheduler {
	/// Construct OA's constant-amplitude triangular schedule.
	///
	/// # Errors
	///
	/// Returns an error for invalid rates, an inverted range, or a zero
	/// half-cycle span.
	pub fn triangular(
		base_learning_rate: f32,
		max_learning_rate: f32,
		step_size_up: u64,
	) -> Result<Self> {
		Self::new(
			base_learning_rate,
			max_learning_rate,
			step_size_up,
			CyclicMode::Triangular,
			1.0,
		)
	}

	/// Construct a cyclic schedule.
	///
	/// # Errors
	///
	/// Returns an error for invalid rates, an inverted range, a zero half-cycle,
	/// or a non-finite/non-positive exponential factor.
	pub fn new(
		base_learning_rate: f32,
		max_learning_rate: f32,
		step_size_up: u64,
		mode: CyclicMode,
		gamma: f32,
	) -> Result<Self> {
		validate_rate(base_learning_rate, "cyclic base")?;
		validate_rate(max_learning_rate, "cyclic maximum")?;
		if base_learning_rate > max_learning_rate {
			return Err(Error::invalid_argument(
				"cyclic base learning rate exceeds its maximum",
			));
		}
		if step_size_up == 0 {
			return Err(Error::invalid_argument(
				"cyclic schedule requires a nonzero rising span",
			));
		}
		if !gamma.is_finite() || gamma <= 0.0 {
			return Err(Error::invalid_argument(
				"cyclic gamma must be finite and positive",
			));
		}
		Ok(Self {
			base_learning_rate,
			max_learning_rate,
			step_size_up,
			mode,
			gamma,
		})
	}
}

impl LrScheduler for CyclicScheduler {
	fn learning_rate(&self, step: u64) -> f32 {
		let step = step as f32;
		let size = self.step_size_up as f32;
		let cycle = (step / (2.0 * size)).floor();
		let x = (step / size - 2.0 * cycle - 1.0).abs();
		let scale = match self.mode {
			CyclicMode::Triangular => 1.0,
			CyclicMode::Triangular2 => 1.0 / 2.0_f32.powf(cycle),
			CyclicMode::ExpRange => self.gamma.powf(step),
		};
		let ramp = (1.0 - x).max(0.0);
		self.base_learning_rate + (self.max_learning_rate - self.base_learning_rate) * ramp * scale
	}
}

/// Cosine annealing with periodic warm restarts.
pub struct CosineWarmRestartsScheduler {
	max_learning_rate: f32,
	initial_period: u64,
	period_multiplier: u64,
	min_learning_rate: f32,
}

impl CosineWarmRestartsScheduler {
	/// Construct fixed-period warm restarts with a zero minimum rate.
	///
	/// # Errors
	///
	/// Returns an error for an invalid maximum rate or zero period.
	pub fn fixed(max_learning_rate: f32, initial_period: u64) -> Result<Self> {
		Self::new(max_learning_rate, initial_period, 1, 0.0)
	}

	/// Construct a cosine-warm-restarts schedule.
	///
	/// # Errors
	///
	/// Returns an error for invalid rates, an inverted range, or zero period and
	/// multiplier values.
	pub fn new(
		max_learning_rate: f32,
		initial_period: u64,
		period_multiplier: u64,
		min_learning_rate: f32,
	) -> Result<Self> {
		validate_rate(max_learning_rate, "restart maximum")?;
		validate_rate(min_learning_rate, "restart minimum")?;
		if min_learning_rate > max_learning_rate {
			return Err(Error::invalid_argument(
				"restart minimum learning rate exceeds its maximum",
			));
		}
		if initial_period == 0 || period_multiplier == 0 {
			return Err(Error::invalid_argument(
				"restart period and multiplier must be nonzero",
			));
		}
		Ok(Self {
			max_learning_rate,
			initial_period,
			period_multiplier,
			min_learning_rate,
		})
	}
}

impl LrScheduler for CosineWarmRestartsScheduler {
	fn learning_rate(&self, step: u64) -> f32 {
		let (current, period) = if self.period_multiplier == 1 {
			(step % self.initial_period, self.initial_period)
		} else {
			let mut start = 0_u64;
			let mut period = self.initial_period;
			loop {
				let Some(end) = start.checked_add(period) else {
					break (step - start, period);
				};
				if step < end {
					break (step - start, period);
				}
				start = end;
				let Some(next) = period.checked_mul(self.period_multiplier) else {
					break (step - start, u64::MAX);
				};
				period = next;
			}
		};
		let progress = current as f32 / period as f32;
		self.min_learning_rate
			+ 0.5
				* (self.max_learning_rate - self.min_learning_rate)
				* (1.0 + (PI * progress).cos())
	}
}

/// Improvement direction for [`ReduceOnPlateauScheduler`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PlateauMode {
	/// A lower monitored value is better.
	#[default]
	Min,
	/// A higher monitored value is better.
	Max,
}

/// Stateful schedule that lowers its rate after a monitored plateau.
pub struct ReduceOnPlateauScheduler {
	current_learning_rate: f32,
	factor: f32,
	patience: u64,
	threshold: f32,
	min_learning_rate: f32,
	mode: PlateauMode,
	best: f32,
	bad_epochs: u64,
}

impl ReduceOnPlateauScheduler {
	/// Construct OA's default minimize-mode plateau policy.
	///
	/// This uses factor `0.1`, patience `10`, threshold `1e-4`, and a zero
	/// minimum rate.
	///
	/// # Errors
	///
	/// Returns an error when the initial rate is invalid.
	pub fn with_defaults(initial_learning_rate: f32) -> Result<Self> {
		Self::new(
			initial_learning_rate,
			0.1,
			10,
			1.0e-4,
			0.0,
			PlateauMode::Min,
		)
	}

	/// Construct a reduce-on-plateau schedule.
	///
	/// # Errors
	///
	/// Returns an error for invalid rates, a factor outside `(0, 1]`, or a
	/// non-finite/negative threshold.
	pub fn new(
		initial_learning_rate: f32,
		factor: f32,
		patience: u64,
		threshold: f32,
		min_learning_rate: f32,
		mode: PlateauMode,
	) -> Result<Self> {
		validate_rate(initial_learning_rate, "plateau initial")?;
		validate_rate(min_learning_rate, "plateau minimum")?;
		if min_learning_rate > initial_learning_rate {
			return Err(Error::invalid_argument(
				"plateau minimum learning rate exceeds its initial rate",
			));
		}
		if !factor.is_finite() || factor <= 0.0 || factor > 1.0 {
			return Err(Error::invalid_argument(
				"plateau factor must be finite and in (0, 1]",
			));
		}
		if !threshold.is_finite() || threshold < 0.0 {
			return Err(Error::invalid_argument(
				"plateau threshold must be finite and non-negative",
			));
		}
		Ok(Self {
			current_learning_rate: initial_learning_rate,
			factor,
			patience,
			threshold,
			min_learning_rate,
			mode,
			best: match mode {
				PlateauMode::Min => f32::INFINITY,
				PlateauMode::Max => f32::NEG_INFINITY,
			},
			bad_epochs: 0,
		})
	}

	/// Observe one completed metric value and update the schedule state.
	///
	/// # Errors
	///
	/// Returns an error when `metric` is not finite.
	pub fn step(&mut self, metric: f32) -> Result<()> {
		if !metric.is_finite() {
			return Err(Error::invalid_argument("plateau metric must be finite"));
		}
		let improved = match self.mode {
			PlateauMode::Min => metric < self.best - self.threshold,
			PlateauMode::Max => metric > self.best + self.threshold,
		};
		if improved {
			self.best = metric;
			self.bad_epochs = 0;
		} else {
			self.bad_epochs = self.bad_epochs.saturating_add(1);
			if self.bad_epochs > self.patience {
				self.current_learning_rate =
					(self.current_learning_rate * self.factor).max(self.min_learning_rate);
				self.bad_epochs = 0;
			}
		}
		Ok(())
	}
}

impl LrScheduler for ReduceOnPlateauScheduler {
	fn learning_rate(&self, _step: u64) -> f32 {
		self.current_learning_rate
	}
}

/// Consecutive schedules selected by absolute step milestones.
pub struct SequentialScheduler {
	schedulers: Vec<Box<dyn LrScheduler>>,
	milestones: Vec<u64>,
}

impl SequentialScheduler {
	/// Construct a sequence whose `i`th milestone starts scheduler `i + 1`.
	///
	/// # Errors
	///
	/// Returns an error for an empty schedule list, a milestone count other than
	/// `schedulers.len() - 1`, a zero first milestone, or non-increasing values.
	pub fn new(schedulers: Vec<Box<dyn LrScheduler>>, milestones: Vec<u64>) -> Result<Self> {
		if schedulers.is_empty() || milestones.len() + 1 != schedulers.len() {
			return Err(Error::invalid_argument(
				"sequential schedule requires one fewer milestone than schedulers",
			));
		}
		if milestones.first().is_some_and(|first| *first == 0)
			|| milestones.windows(2).any(|pair| pair[0] >= pair[1])
		{
			return Err(Error::invalid_argument(
				"sequential schedule milestones must be positive and strictly increasing",
			));
		}
		Ok(Self {
			schedulers,
			milestones,
		})
	}
}

impl LrScheduler for SequentialScheduler {
	fn learning_rate(&self, step: u64) -> f32 {
		let index = self
			.milestones
			.partition_point(|milestone| *milestone <= step);
		let origin = if index == 0 {
			0
		} else {
			self.milestones[index - 1]
		};
		self.schedulers[index].learning_rate(step - origin)
	}
}

/// Linear warmup followed by cosine annealing.
pub struct LinearWarmupCosineScheduler {
	inner: WarmupScheduler,
}

impl LinearWarmupCosineScheduler {
	/// Construct the donor-compatible composed schedule.
	///
	/// # Errors
	///
	/// Returns an error for invalid rates, an inverted range, or a total span not
	/// greater than the warmup span.
	pub fn new(
		warmup_steps: u64,
		total_steps: u64,
		max_learning_rate: f32,
		min_learning_rate: f32,
	) -> Result<Self> {
		if total_steps <= warmup_steps {
			return Err(Error::invalid_argument(
				"warmup-cosine total steps must exceed warmup steps",
			));
		}
		let cosine = CosineScheduler::new(
			max_learning_rate,
			min_learning_rate,
			total_steps - warmup_steps,
		)?;
		Ok(Self {
			inner: WarmupScheduler::with_after(max_learning_rate, warmup_steps, Box::new(cosine))?,
		})
	}
}

impl LrScheduler for LinearWarmupCosineScheduler {
	fn learning_rate(&self, step: u64) -> f32 {
		self.inner.learning_rate(step.saturating_sub(1))
	}
}

fn validate_rate(rate: f32, name: &str) -> Result<()> {
	if !rate.is_finite() || rate < 0.0 {
		return Err(Error::invalid_argument(format!(
			"{name} learning rate must be finite and non-negative"
		)));
	}
	Ok(())
}
