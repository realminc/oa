//! Dropout variants: channel-masked `Dropout1d/2d/3d`, `AlphaDropout`, and
//! channel-masked `FeatureAlphaDropout`, composed over `matrix::dropout`.

use crate::{Error, Matrix, Result, matrix};

use super::super::{Module, ModuleRegistry};

/// Generate a broadcastable keep mask at channel granularity.
///
/// `matrix::dropout` applies the keep mask scaled by `1/(1-p)`; multiplying
/// the input by that scaled mask is the inverted-dropout contract. The mask
/// is generated on a ones matrix at `[N, C]` granularity, then reshaped to
/// broadcast over the trailing spatial axes.
fn channel_mask(
	input: &Matrix,
	spatial_rank: usize,
	probability: f32,
	seed: u64,
) -> Result<Matrix> {
	let shape = input.shape();
	let mask_shape = shape[..2].to_vec();
	let ones = Matrix::new_from_slice_handle(
		input.engine_handle(),
		mask_shape.clone(),
		&vec![1.0_f32; mask_shape.iter().product()],
	)?;
	let mask = matrix::dropout(&ones, probability, seed)?;
	mask.reshape(
		[shape[0], shape[1]]
			.into_iter()
			.chain(vec![1; spatial_rank])
			.collect::<Vec<_>>(),
	)
}

fn validate_spatial(name: &str, input: &Matrix, spatial_rank: usize) -> Result<()> {
	if input.shape().len() != spatial_rank + 2 {
		return Err(Error::invalid_argument(format!(
			"{name} requires a rank-{} [N, C, ...] input; found {:?}",
			spatial_rank + 2,
			input.shape()
		)));
	}
	Ok(())
}

macro_rules! channel_dropout {
	($name:ident, $spatial:literal, $doc:literal) => {
		/// $doc
		pub struct $name {
			probability: f32,
			seed: u64,
			registry: ModuleRegistry,
		}

		impl $name {
			/// Construct with the thread-local random seed source.
			///
			/// # Errors
			///
			/// Returns an error unless `probability` is finite and in `[0, 1)`.
			pub fn new(probability: f32) -> Result<Self> {
				Self::new_with_seed(probability, 0)
			}

			/// Construct with an explicit Philox seed.
			///
			/// # Errors
			///
			/// Returns an error unless `probability` is finite and in `[0, 1)`.
			pub fn new_with_seed(probability: f32, seed: u64) -> Result<Self> {
				if !probability.is_finite() || !(0.0..1.0).contains(&probability) {
					return Err(Error::invalid_argument(
						"dropout probability must be finite and in [0, 1)",
					));
				}
				Ok(Self {
					probability,
					seed,
					registry: ModuleRegistry::new(),
				})
			}

			/// Return the probability of dropping each channel during training.
			pub const fn probability(&self) -> f32 {
				self.probability
			}

			/// Apply channel dropout in training mode or return an identity
			/// handle in eval mode.
			///
			/// # Errors
			///
			/// Returns an error for a wrongly ranked input or a mask dispatch
			/// failure.
			pub fn forward(&self, input: &Matrix) -> Result<Matrix> {
				if !self.is_training() || self.probability == 0.0 {
					return Ok(input.clone());
				}
				validate_spatial(stringify!($name), input, $spatial)?;
				let mask = channel_mask(input, $spatial, self.probability, self.seed)?;
				matrix::mul(input, &mask)
			}
		}

		impl Module for $name {
			fn forward(&self, input: &Matrix) -> Result<Matrix> {
				$name::forward(self, input)
			}

			fn registry(&self) -> &ModuleRegistry {
				&self.registry
			}
		}
	};
}

channel_dropout!(
	Dropout1d,
	1,
	"Randomly zero entire channels of a `[N, C, L]` input in training mode."
);
channel_dropout!(
	Dropout2d,
	2,
	"Randomly zero entire channels of a `[N, C, H, W]` input in training mode."
);
channel_dropout!(
	Dropout3d,
	3,
	"Randomly zero entire channels of a `[N, C, D, H, W]` input in training mode."
);

/// Donor `alpha_prime` for SELU-preserving dropout (`-alpha * lambda`).
const ALPHA_PRIME: f32 = -1.758_099_3_f32;

/// Apply the alpha-dropout affine map `a * (x * d + alpha' * (1 - d)) + b`
/// given an elementwise or channel-broadcast keep mask scaled by `1 - p`.
fn alpha_dropout_apply(input: &Matrix, raw_mask: &Matrix, probability: f32) -> Result<Matrix> {
	let scale_a = (1.0_f32 + ALPHA_PRIME * ALPHA_PRIME * probability * (1.0 - probability))
		.sqrt()
		.recip();
	let scale_b = -scale_a * ALPHA_PRIME * probability;
	let retained = matrix::mul(input, raw_mask)?;
	let dropped = matrix::scale(
		&matrix::add_scalar(&matrix::neg(raw_mask)?, 1.0)?,
		ALPHA_PRIME,
	)?;
	matrix::add_scalar(
		&matrix::scale(&matrix::add(&retained, &dropped)?, scale_a)?,
		scale_b,
	)
}

/// Recover the raw `{0, 1}` keep mask from `matrix::dropout`'s scaled mask.
fn raw_mask(keep_shape: &[usize], input: &Matrix, probability: f32, seed: u64) -> Result<Matrix> {
	let ones = Matrix::new_from_slice_handle(
		input.engine_handle(),
		keep_shape.to_vec(),
		&vec![1.0_f32; keep_shape.iter().product()],
	)?;
	matrix::scale(
		&matrix::dropout(&ones, probability, seed)?,
		1.0 - probability,
	)
}

/// Elementwise SELU-preserving dropout (`nn.AlphaDropout`).
///
/// Applies `a * (x * d + alpha' * (1 - d)) + b` elementwise, keeping the
/// input mean and variance compatible with SELU activations.
pub struct AlphaDropout {
	probability: f32,
	seed: u64,
	registry: ModuleRegistry,
}

impl AlphaDropout {
	/// Construct with the thread-local random seed source.
	///
	/// # Errors
	///
	/// Returns an error unless `probability` is finite and in `[0, 1)`.
	pub fn new(probability: f32) -> Result<Self> {
		Self::new_with_seed(probability, 0)
	}

	/// Construct with an explicit Philox seed.
	///
	/// # Errors
	///
	/// Returns an error unless `probability` is finite and in `[0, 1)`.
	pub fn new_with_seed(probability: f32, seed: u64) -> Result<Self> {
		if !probability.is_finite() || !(0.0..1.0).contains(&probability) {
			return Err(Error::invalid_argument(
				"alpha dropout probability must be finite and in [0, 1)",
			));
		}
		Ok(Self {
			probability,
			seed,
			registry: ModuleRegistry::new(),
		})
	}

	/// Return the probability of dropping each element during training.
	pub const fn probability(&self) -> f32 {
		self.probability
	}

	/// Apply alpha dropout in training mode or return an identity handle in
	/// eval mode.
	///
	/// # Errors
	///
	/// Returns an error from the underlying Matrix operations.
	pub fn forward(&self, input: &Matrix) -> Result<Matrix> {
		if !self.is_training() || self.probability == 0.0 {
			return Ok(input.clone());
		}
		let mask = raw_mask(input.shape(), input, self.probability, self.seed)?;
		alpha_dropout_apply(input, &mask, self.probability)
	}
}

impl Module for AlphaDropout {
	fn forward(&self, input: &Matrix) -> Result<Matrix> {
		AlphaDropout::forward(self, input)
	}

	fn registry(&self) -> &ModuleRegistry {
		&self.registry
	}
}

/// Channel-granular SELU-preserving dropout (`nn.FeatureAlphaDropout`).
///
/// Shares one `{0, 1}` keep mask per `[N, C]` channel across the trailing
/// axes, then applies the alpha-dropout affine map.
pub struct FeatureAlphaDropout {
	probability: f32,
	seed: u64,
	registry: ModuleRegistry,
}

impl FeatureAlphaDropout {
	/// Construct with the thread-local random seed source.
	///
	/// # Errors
	///
	/// Returns an error unless `probability` is finite and in `[0, 1)`.
	pub fn new(probability: f32) -> Result<Self> {
		Self::new_with_seed(probability, 0)
	}

	/// Construct with an explicit Philox seed.
	///
	/// # Errors
	///
	/// Returns an error unless `probability` is finite and in `[0, 1)`.
	pub fn new_with_seed(probability: f32, seed: u64) -> Result<Self> {
		if !probability.is_finite() || !(0.0..1.0).contains(&probability) {
			return Err(Error::invalid_argument(
				"feature alpha dropout probability must be finite and in [0, 1)",
			));
		}
		Ok(Self {
			probability,
			seed,
			registry: ModuleRegistry::new(),
		})
	}

	/// Return the probability of dropping each channel during training.
	pub const fn probability(&self) -> f32 {
		self.probability
	}

	/// Apply channel-granular alpha dropout in training mode or return an
	/// identity handle in eval mode.
	///
	/// # Errors
	///
	/// Returns an error for a rank-below-two input or a mask dispatch failure.
	pub fn forward(&self, input: &Matrix) -> Result<Matrix> {
		if !self.is_training() || self.probability == 0.0 {
			return Ok(input.clone());
		}
		let shape = input.shape();
		if shape.len() < 2 {
			return Err(Error::invalid_argument(
				"FeatureAlphaDropout requires an input of rank at least two",
			));
		}
		let mask = raw_mask(&shape[..2], input, self.probability, self.seed)?.reshape(
			[shape[0], shape[1]]
				.into_iter()
				.chain(vec![1; shape.len() - 2])
				.collect::<Vec<_>>(),
		)?;
		alpha_dropout_apply(input, &mask, self.probability)
	}
}

impl Module for FeatureAlphaDropout {
	fn forward(&self, input: &Matrix) -> Result<Matrix> {
		FeatureAlphaDropout::forward(self, input)
	}

	fn registry(&self) -> &ModuleRegistry {
		&self.registry
	}
}
