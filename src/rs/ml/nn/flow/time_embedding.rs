use crate::{DType, Engine, Error, Matrix, Result, matrix};

use super::super::super::{Module, ModuleRegistry};

/// GPU sinusoidal embedding for normalized continuous flow time.
pub struct FlowTimeEmbedding {
	embedding_dim: usize,
	max_period: f32,
	time_scale: f32,
	frequencies: Matrix,
	registry: ModuleRegistry,
}

impl FlowTimeEmbedding {
	/// Construct the donor sinusoidal time embedding.
	///
	/// # Errors
	///
	/// Returns an error unless `embedding_dim` is positive and even,
	/// `max_period` is finite and greater than one, and `time_scale` is finite
	/// and positive.
	pub fn new(
		engine: &Engine,
		embedding_dim: usize,
		max_period: f32,
		time_scale: f32,
	) -> Result<Self> {
		if embedding_dim == 0
			|| !embedding_dim.is_multiple_of(2)
			|| !max_period.is_finite()
			|| max_period <= 1.0
			|| !time_scale.is_finite()
			|| time_scale <= 0.0
		{
			return Err(Error::invalid_argument(
				"FlowTimeEmbedding requires a positive even dimension, finite max period greater than one, and finite positive scale",
			));
		}
		let half = embedding_dim / 2;
		let log_period = max_period.ln();
		let frequencies = (0..half)
			.map(|index| time_scale * (-log_period * index as f32 / half as f32).exp())
			.collect::<Vec<_>>();
		let frequencies = Matrix::from_f32(engine, [1, half], &frequencies)?;
		let mut registry = ModuleRegistry::new();
		registry.register_buffer("frequencies", frequencies.clone(), false)?;
		Ok(Self {
			embedding_dim,
			max_period,
			time_scale,
			frequencies,
			registry,
		})
	}

	/// Embed F32 time values shaped `[B]` or `[B, 1]` as `[B, D]`.
	///
	/// # Errors
	///
	/// Returns an error for any other shape or dtype, another engine, or a
	/// failed Matrix operation.
	pub fn forward(&self, time: &Matrix) -> Result<Matrix> {
		let batch = match time.shape() {
			[batch] => *batch,
			[batch, 1] => *batch,
			_ => {
				return Err(Error::invalid_argument(
					"FlowTimeEmbedding expects F32 [B] or [B, 1]",
				));
			}
		};
		if time.dtype() != DType::F32
			|| !time
				.engine_handle()
				.same_as(self.frequencies.engine_handle())
		{
			return Err(Error::invalid_argument(
				"FlowTimeEmbedding expects same-engine F32 [B] or [B, 1]",
			));
		}
		let time = matrix::reshape(time, [batch, 1])?;
		let phase = matrix::mul(&time, &self.frequencies)?;
		matrix::concat(&[matrix::sin(&phase)?, matrix::cos(&phase)?], 1)
	}

	/// Return the output embedding width.
	pub const fn embedding_dim(&self) -> usize {
		self.embedding_dim
	}

	/// Return the maximum sinusoidal period.
	pub const fn max_period(&self) -> f32 {
		self.max_period
	}

	/// Return the multiplicative input-time scale.
	pub const fn time_scale(&self) -> f32 {
		self.time_scale
	}
}

impl Module for FlowTimeEmbedding {
	fn forward(&self, input: &Matrix) -> Result<Matrix> {
		FlowTimeEmbedding::forward(self, input)
	}

	fn registry(&self) -> &ModuleRegistry {
		&self.registry
	}
}
