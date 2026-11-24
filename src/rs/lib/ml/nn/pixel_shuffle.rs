//! Pixel-shuffle (`nn.PixelShuffle`) over a last-axis gather remap.

use crate::{Error, Matrix, Result, matrix};

use super::super::{Module, ModuleRegistry};

/// Rearrange `[N, C*r^2, H, W]` to `[N, C, H*r, W*r]`.
///
/// The composed contract flattens each sample and remaps elements with a
/// single [`matrix::gather_last_dim`] using a host-built I32 index plane —
/// the same channel ordering as `nn.PixelShuffle`
/// (`out[.., c, h*r+i, w*r+j] = in[.., c*r^2 + i*r + j, h, w]`). The index
/// plane is rebuilt per forward, matching the explicit host-observation
/// boundary used elsewhere for data-shaping metadata. Parameterless;
/// gradients flow through the gather adjoint.
pub struct PixelShuffle {
	upscale_factor: usize,
	registry: ModuleRegistry,
}

impl PixelShuffle {
	/// Construct with the integer `upscale_factor` (`r` in `nn.PixelShuffle`).
	///
	/// # Errors
	///
	/// Returns an error when `upscale_factor` is less than two.
	pub fn new(upscale_factor: usize) -> Result<Self> {
		if upscale_factor < 2 {
			return Err(Error::invalid_argument(
				"pixel shuffle requires an upscale factor of at least 2",
			));
		}
		Ok(Self {
			upscale_factor,
			registry: ModuleRegistry::new(),
		})
	}

	/// Return the configured upscale factor.
	pub const fn upscale_factor(&self) -> usize {
		self.upscale_factor
	}

	/// Rearrange `[N, C*r^2, H, W]` into `[N, C, H*r, W*r]`.
	///
	/// # Errors
	///
	/// Returns an error unless `input` is rank four F32 with a channel count
	/// divisible by `r^2`, or the gather dispatch fails.
	pub fn forward(&self, input: &Matrix) -> Result<Matrix> {
		let r = self.upscale_factor;
		let &[batch, packed, height, width] = input.shape() else {
			return Err(Error::invalid_argument(format!(
				"pixel shuffle requires [N, C*r^2, H, W]; found {:?}",
				input.shape(),
			)));
		};
		let squared = r
			.checked_mul(r)
			.ok_or_else(|| Error::invalid_argument("pixel shuffle factor overflows usize"))?;
		if packed == 0 || !packed.is_multiple_of(squared) {
			return Err(Error::invalid_argument(format!(
				"pixel shuffle channels {packed} must be a nonzero multiple of r^2 = {squared}"
			)));
		}
		let channels = packed / squared;
		let out_height = height * r;
		let out_width = width * r;
		// out[.., c, h_out, w_out] = in[.., c*r^2 + (h_out%r)*r + (w_out%r), h_out/r, w_out/r]
		let mut indices = Vec::with_capacity(batch * channels * out_height * out_width);
		for _ in 0..batch {
			for c in 0..channels {
				for h_out in 0..out_height {
					for w_out in 0..out_width {
						let channel = c * squared + (h_out % r) * r + (w_out % r);
						indices.push((channel * height + h_out / r) as i32 * width as i32 + (w_out / r) as i32);
					}
				}
			}
		}
		let flat = input.reshape([batch, packed * height * width])?;
		let index_matrix =
			Matrix::new_from_slice([batch, channels * out_height * out_width], &indices)?;
		let gathered = matrix::gather_last_dim(&flat, &index_matrix)?;
		gathered.reshape([batch, channels, out_height, out_width])
	}
}

impl Module for PixelShuffle {
	fn forward(&self, input: &Matrix) -> Result<Matrix> {
		PixelShuffle::forward(self, input)
	}

	fn registry(&self) -> &ModuleRegistry {
		&self.registry
	}
}

/// Rearrange `[N, C, H, W]` to `[N, C*r^2, H/r, W/r]` — the exact inverse of
/// [`PixelShuffle`].
///
/// Uses the same composed contract as `PixelShuffle`: flatten each sample and
/// remap elements with one [`matrix::gather_last_dim`] over a host-built I32
/// index plane (`out[.., c*r^2 + i*r + j, h, w] = in[.., c, h*r + i, w*r +
/// j]`). Parameterless; gradients flow through the gather adjoint.
pub struct PixelUnshuffle {
	downscale_factor: usize,
	registry: ModuleRegistry,
}

impl PixelUnshuffle {
	/// Construct with the integer `downscale_factor` (`r` in
	/// `nn.PixelUnshuffle`).
	///
	/// # Errors
	///
	/// Returns an error when `downscale_factor` is less than two.
	pub fn new(downscale_factor: usize) -> Result<Self> {
		if downscale_factor < 2 {
			return Err(Error::invalid_argument(
				"pixel unshuffle requires a downscale factor of at least 2",
			));
		}
		Ok(Self {
			downscale_factor,
			registry: ModuleRegistry::new(),
		})
	}

	/// Return the configured downscale factor.
	pub const fn downscale_factor(&self) -> usize {
		self.downscale_factor
	}

	/// Rearrange `[N, C, H, W]` into `[N, C*r^2, H/r, W/r]`.
	///
	/// # Errors
	///
	/// Returns an error unless `input` is rank four F32 with spatial extents
	/// divisible by `r`, or the gather dispatch fails.
	pub fn forward(&self, input: &Matrix) -> Result<Matrix> {
		let r = self.downscale_factor;
		let &[batch, channels, height, width] = input.shape() else {
			return Err(Error::invalid_argument(format!(
				"pixel unshuffle requires [N, C, H, W]; found {:?}",
				input.shape(),
			)));
		};
		if height == 0 || !height.is_multiple_of(r) || width == 0 || !width.is_multiple_of(r) {
			return Err(Error::invalid_argument(format!(
				"pixel unshuffle spatial extents {height}x{width} must be nonzero multiples of r = {r}"
			)));
		}
		let squared = r
			.checked_mul(r)
			.ok_or_else(|| Error::invalid_argument("pixel unshuffle factor overflows usize"))?;
		let out_height = height / r;
		let out_width = width / r;
		// out[.., c*r^2 + i*r + j, h, w] = in[.., c, h*r + i, w*r + j]
		let mut indices = Vec::with_capacity(batch * channels * squared * out_height * out_width);
		for _ in 0..batch {
			for c in 0..channels {
				for i in 0..r {
					for j in 0..r {
						for h in 0..out_height {
							for w in 0..out_width {
								indices.push((c * height + h * r + i) as i32 * width as i32 + (w * r + j) as i32);
							}
						}
					}
				}
			}
		}
		let flat = input.reshape([batch, channels * height * width])?;
		let index_matrix = Matrix::new_from_slice(
			[batch, channels * squared * out_height * out_width],
			&indices,
		)?;
		let gathered = matrix::gather_last_dim(&flat, &index_matrix)?;
		gathered.reshape([batch, channels * squared, out_height, out_width])
	}
}

impl Module for PixelUnshuffle {
	fn forward(&self, input: &Matrix) -> Result<Matrix> {
		PixelUnshuffle::forward(self, input)
	}

	fn registry(&self) -> &ModuleRegistry {
		&self.registry
	}
}
