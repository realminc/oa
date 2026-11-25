//! Channel shuffle (`nn.ChannelShuffle`): interleave channel groups.

use crate::{Error, Matrix, Result, matrix};

use super::super::{Module, ModuleRegistry};

/// Shuffle channels between `groups` equal-sized groups.
///
/// Input `[N, C, *]` is flattened to `[N, C, S]`, transposed to rows of
/// channels, gathered by the interleave index `c * groups + g`, and mapped
/// back — matching the donor channel order with a differentiable adjoint.
pub struct ChannelShuffle {
	groups: usize,
	registry: ModuleRegistry,
}

impl ChannelShuffle {
	/// Construct a channel shuffle with `groups` groups.
	///
	/// # Errors
	///
	/// Returns an error when `groups` is zero.
	pub fn new(groups: usize) -> Result<Self> {
		if groups == 0 {
			return Err(Error::invalid_argument(
				"channel shuffle requires at least one group",
			));
		}
		Ok(Self {
			groups,
			registry: ModuleRegistry::new(),
		})
	}

	/// Return the configured group count.
	pub const fn groups(&self) -> usize {
		self.groups
	}

	/// Interleave the channels of a `[N, C, *]` input.
	///
	/// # Errors
	///
	/// Returns an error for a rank-below-two input, a channel count not
	/// divisible by `groups`, or a composed dispatch failure.
	pub fn forward(&self, input: &Matrix) -> Result<Matrix> {
		let operation = "ChannelShuffle";
		let shape = input.shape();
		if shape.len() < 2 {
			return Err(Error::invalid_argument(format!(
				"{operation} requires an input of rank at least two"
			)));
		}
		let (batch, channels) = (shape[0], shape[1]);
		if !channels.is_multiple_of(self.groups) {
			return Err(Error::invalid_argument(format!(
				"{operation} channel count {channels} is not divisible by {}",
				self.groups
			)));
		}
		let per_group = channels / self.groups;
		let spatial: usize = shape[2..].iter().product();
		let flattened = input.reshape([batch, channels, spatial])?;
		let rows = matrix::transpose(
			&matrix::transpose(&flattened, 1, 2)?.reshape([batch * spatial, channels])?,
			0,
			1,
		)?;
		let indices: Vec<i32> = (0..per_group)
			.flat_map(|within| (0..self.groups).map(move |group| (group * per_group + within) as i32))
			.collect();
		let index = Matrix::new_from_slice_handle(input.runtime(), vec![channels], &indices)?;
		let shuffled = matrix::transpose(&matrix::gather(&rows, &index)?, 0, 1)?
			.reshape([batch, spatial, channels])?;
		matrix::transpose(&shuffled, 1, 2)?.reshape(shape.to_vec())
	}
}

impl Module for ChannelShuffle {
	fn forward(&self, input: &Matrix) -> Result<Matrix> {
		ChannelShuffle::forward(self, input)
	}

	fn registry(&self) -> &ModuleRegistry {
		&self.registry
	}
}
