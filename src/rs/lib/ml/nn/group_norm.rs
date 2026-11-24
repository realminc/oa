//! Group- and instance-normalization modules composed over reduce ops.

use crate::{DType, Engine, Error, Matrix, Result, matrix};

use super::super::{Module, ModuleRegistry, Parameter, autograd};

/// Group normalization: normalize each `channels / num_groups` channel group
/// per sample, then apply a learned per-channel affine.
///
/// Composed contract over `reshape` + `mean` + `mul`/`sub`/`div`/`sqrt`, so
/// gradients flow through the elementwise and reduce adjoints into `weight`
/// and `bias` parameters without a fused kernel. The variance uses the
/// biased `mean((x - mu)^2)` form matching `nn.GroupNorm`.
pub struct GroupNorm {
	num_groups: usize,
	num_channels: usize,
	epsilon: f32,
	weight: Parameter,
	bias: Parameter,
	registry: ModuleRegistry,
}

impl GroupNorm {
	/// Construct with unit weight and zero bias.
	///
	/// # Errors
	///
	/// Returns an error when `num_channels` is zero, `num_groups` does not
	/// divide it, `epsilon` is not finite and positive, or allocation fails.
	pub fn new_on(
		engine: &Engine,
		num_groups: usize,
		num_channels: usize,
		epsilon: f32,
	) -> Result<Self> {
		if num_channels == 0 || num_groups == 0 || !num_channels.is_multiple_of(num_groups) {
			return Err(Error::invalid_argument(format!(
				"group norm requires 0 < num_groups | num_channels; found {num_groups}/{num_channels}"
			)));
		}
		if !epsilon.is_finite() || epsilon <= 0.0 {
			return Err(Error::invalid_argument(
				"group norm epsilon must be finite and positive",
			));
		}
		let weight = Parameter::new(
			"weight",
			Matrix::new_from_f32_on(engine, [num_channels], &vec![1.0_f32; num_channels])?,
		)?;
		let bias = Parameter::new(
			"bias",
			Matrix::new_from_f32_on(engine, [num_channels], &vec![0.0_f32; num_channels])?,
		)?;
		let mut registry = ModuleRegistry::new();
		registry.register_parameter("weight", weight.clone())?;
		registry.register_parameter("bias", bias.clone())?;
		Ok(Self {
			num_groups,
			num_channels,
			epsilon,
			weight,
			bias,
			registry,
		})
	}

	/// Construct on the current engine with `epsilon = 1e-5`.
	///
	/// # Errors
	///
	/// Returns an error under the same contract as [`Self::new_on`].
	pub fn new(num_groups: usize, num_channels: usize) -> Result<Self> {
		Self::new_on(&Engine::current()?, num_groups, num_channels, 1e-5)
	}

	/// Normalize `input` shaped `[N, C, *spatial]` and apply the affine.
	///
	/// # Errors
	///
	/// Returns an error when the input is not F32 rank >= 2 with
	/// `num_channels` channels, or the composed ops fail.
	pub fn forward(&self, input: &Matrix) -> Result<Matrix> {
		if input.dtype() != DType::F32
			|| input.shape().len() < 2
			|| input.shape()[1] != self.num_channels
		{
			return Err(Error::invalid_argument(format!(
				"group norm requires F32 [N, {}, *]; found {:?} {}",
				self.num_channels,
				input.shape(),
				input.dtype().token(),
			)));
		}
		let batch = input.shape()[0];
		let spatial: usize = input.shape()[2..].iter().product();
		let inner = (self.num_channels / self.num_groups)
			.checked_mul(spatial)
			.ok_or_else(|| Error::invalid_argument("group norm inner size overflows usize"))?;
		let rows = input.reshape([batch * self.num_groups, inner])?;
		let mean = matrix::mean(&rows, 1)?;
		let centered = matrix::sub(&rows, &mean)?;
		let variance = matrix::mean(&matrix::mul(&centered, &centered)?, 1)?;
		let inv_std = matrix::sqrt(&matrix::add_scalar(&variance, self.epsilon)?)?;
		let normalized = matrix::div(&centered, &inv_std)?;
		autograd::record_parameter_leaf(&self.weight)?;
		autograd::record_parameter_leaf(&self.bias)?;
		let mut channel_shape = vec![1_usize; input.shape().len()];
		channel_shape[1] = self.num_channels;
		let restored = normalized.reshape(input.shape().to_vec())?;
		let scaled = matrix::mul(
			&restored,
			&self.weight.data().reshape(channel_shape.clone())?,
		)?;
		matrix::add(&scaled, &self.bias.data().reshape(channel_shape)?)
	}
}

impl Module for GroupNorm {
	fn forward(&self, input: &Matrix) -> Result<Matrix> {
		GroupNorm::forward(self, input)
	}

	fn registry(&self) -> &ModuleRegistry {
		&self.registry
	}
}

/// Instance normalization: [`GroupNorm`] with `num_groups == num_channels`.
///
/// Each channel normalizes per sample over its own spatial extent, matching
/// `nn.InstanceNormNd` semantics without running statistics.
pub struct InstanceNorm {
	inner: GroupNorm,
}

impl InstanceNorm {
	/// Construct per-channel normalization with `epsilon = 1e-5`.
	///
	/// # Errors
	///
	/// Returns an error under the same contract as [`GroupNorm::new_on`].
	pub fn new_on(engine: &Engine, num_channels: usize, epsilon: f32) -> Result<Self> {
		Ok(Self {
			inner: GroupNorm::new_on(engine, num_channels, num_channels, epsilon)?,
		})
	}

	/// Construct on the current engine.
	///
	/// # Errors
	///
	/// Returns an error under the same contract as [`Self::new_on`].
	pub fn new(num_channels: usize) -> Result<Self> {
		Self::new_on(&Engine::current()?, num_channels, 1e-5)
	}
}

impl Module for InstanceNorm {
	fn forward(&self, input: &Matrix) -> Result<Matrix> {
		self.inner.forward(input)
	}

	fn registry(&self) -> &ModuleRegistry {
		GroupNorm::registry(&self.inner)
	}
}
