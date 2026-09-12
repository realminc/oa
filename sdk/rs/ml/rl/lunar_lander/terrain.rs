//! Deterministic bounded height-field terrain for Lunar Lander 3D.

use crate::core::vlm::DVec3;
use crate::{Error, Result};

use super::{LunarEpisodeManifest, LunarRandomPurpose};

const TERRAIN_HASH_X: u64 = 0x9e37_79b1_85eb_ca87;
const TERRAIN_HASH_Z: u64 = 0xc2b2_ae3d_27d4_eb4f;

/// Checked rectangular terrain-generation contract.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LunarTerrainConfig {
	/// Number of height-field cells along the world X axis.
	pub cells_x: u32,
	/// Number of height-field cells along the world Z axis.
	pub cells_z: u32,
	/// Width and depth of every cell in world units.
	pub cell_size: f64,
	/// Maximum permitted absolute vertex height.
	pub max_abs_height: f64,
	/// Maximum permitted magnitude of either triangle's height gradient.
	pub max_slope: f64,
	/// Half extent of the square, level landing pad around the origin.
	pub pad_half_extent: f64,
	/// Width over which generated terrain rises away from the guarded pad.
	pub pad_transition_width: f64,
}

impl Default for LunarTerrainConfig {
	fn default() -> Self {
		Self {
			cells_x: 32,
			cells_z: 32,
			cell_size: 1.0,
			max_abs_height: 1.5,
			max_slope: 0.35,
			pad_half_extent: 3.0,
			pad_transition_width: 3.0,
		}
	}
}

impl LunarTerrainConfig {
	/// Validate terrain dimensions and geometric bounds.
	///
	/// # Errors
	///
	/// Returns an error for an empty or excessively large grid, non-finite or
	/// negative bounds, a non-positive cell size, or a pad that does not fit.
	pub fn validate(self) -> Result<()> {
		if self.cells_x == 0 || self.cells_z == 0 {
			return Err(Error::invalid_argument(
				"lunar terrain requires at least one cell on each axis",
			));
		}
		if self.cells_x > 1_048_575
			|| self.cells_z > 1_048_575
			|| u64::from(self.cells_x + 1) * u64::from(self.cells_z + 1) > 1_048_576
		{
			return Err(Error::invalid_argument(
				"lunar terrain exceeds the host-oracle vertex bound",
			));
		}
		if !self.cell_size.is_finite() || self.cell_size <= 0.0 {
			return Err(Error::invalid_argument(
				"lunar terrain cell size must be finite and positive",
			));
		}
		if !self.max_abs_height.is_finite()
			|| self.max_abs_height < 0.0
			|| !self.max_slope.is_finite()
			|| self.max_slope < 0.0
			|| !self.pad_half_extent.is_finite()
			|| self.pad_half_extent < 0.0
			|| !self.pad_transition_width.is_finite()
			|| self.pad_transition_width < 0.0
		{
			return Err(Error::invalid_argument(
				"lunar terrain bounds must be finite and non-negative",
			));
		}
		let available =
			(f64::from(self.cells_x).min(f64::from(self.cells_z)) * self.cell_size) * 0.5;
		if self.pad_half_extent + self.cell_size + self.pad_transition_width > available {
			return Err(Error::invalid_argument(
				"lunar terrain pad and guarded transition do not fit the tile",
			));
		}
		Ok(())
	}

	fn vertex_count(self) -> usize {
		(self.cells_x as usize + 1) * (self.cells_z as usize + 1)
	}

	fn index(self, x: u32, z: u32) -> usize {
		z as usize * (self.cells_x as usize + 1) + x as usize
	}
}

/// Triangle selected by the fixed `v00 -> v11` cell diagonal.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum LunarTerrainTriangle {
	/// Triangle containing vertices `v00`, `v10`, and `v11`.
	#[default]
	LowerRight,
	/// Triangle containing vertices `v00`, `v11`, and `v01`.
	UpperLeft,
}

/// Exact height and normal returned for one terrain coordinate.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LunarTerrainSample {
	/// Whether the requested coordinate lies inside the terrain tile.
	pub in_bounds: bool,
	/// Interpolated world-space height.
	pub height: f64,
	/// Unit surface normal of the selected triangle.
	pub normal: DVec3,
	/// Cell index along the X axis.
	pub cell_x: u32,
	/// Cell index along the Z axis.
	pub cell_z: u32,
	/// X coordinate within the selected cell in `[0, 1]`.
	pub local_x: f64,
	/// Z coordinate within the selected cell in `[0, 1]`.
	pub local_z: f64,
	/// Triangle selected by the cell's fixed diagonal.
	pub triangle: LunarTerrainTriangle,
}

impl Default for LunarTerrainSample {
	fn default() -> Self {
		Self {
			in_bounds: false,
			height: 0.0,
			normal: DVec3 {
				x: 0.0,
				y: 1.0,
				z: 0.0,
			},
			cell_x: 0,
			cell_z: 0,
			local_x: 0.0,
			local_z: 0.0,
			triangle: LunarTerrainTriangle::LowerRight,
		}
	}
}

/// Immutable checked Lunar Lander height field.
#[derive(Clone, Debug, PartialEq)]
pub struct LunarTerrain {
	config: LunarTerrainConfig,
	heights: Vec<f64>,
}

impl LunarTerrain {
	/// Build a level terrain tile with the requested geometry.
	///
	/// # Errors
	///
	/// Returns an error when `config` is invalid.
	pub fn flat(config: LunarTerrainConfig) -> Result<Self> {
		config.validate()?;
		Self::from_heights(config, vec![0.0; config.vertex_count()])
	}

	/// Build the deterministic terrain selected by an episode manifest.
	///
	/// Generation matches the C++ donor's seeded noise, four smoothing passes,
	/// guarded landing pad, and final height-and-slope normalization.
	///
	/// # Errors
	///
	/// Returns an error when `config` or `manifest` is invalid.
	pub fn seeded(config: LunarTerrainConfig, manifest: LunarEpisodeManifest) -> Result<Self> {
		config.validate()?;
		manifest.validate()?;
		let mut heights = vec![0.0; config.vertex_count()];
		for z in 0..=config.cells_z {
			for x in 0..=config.cells_x {
				heights[config.index(x, z)] =
					signed_unit(manifest.seed_for(LunarRandomPurpose::Terrain), x, z);
			}
		}
		let mut smoothed = vec![0.0; heights.len()];
		for _ in 0..4 {
			for z in 0..=config.cells_z {
				for x in 0..=config.cells_x {
					let mut sum = 4.0 * heights[config.index(x, z)];
					let mut weight = 4.0;
					for (nx, nz, valid) in [
						(x.wrapping_sub(1), z, x > 0),
						(x + 1, z, x < config.cells_x),
						(x, z.wrapping_sub(1), z > 0),
						(x, z + 1, z < config.cells_z),
					] {
						if valid {
							sum += heights[config.index(nx, nz)];
							weight += 1.0;
						}
					}
					smoothed[config.index(x, z)] = sum / weight;
				}
			}
			std::mem::swap(&mut heights, &mut smoothed);
		}
		let min_x = -f64::from(config.cells_x) * config.cell_size * 0.5;
		let min_z = -f64::from(config.cells_z) * config.cell_size * 0.5;
		let guarded_pad = config.pad_half_extent + config.cell_size;
		for z in 0..=config.cells_z {
			for x in 0..=config.cells_x {
				let px = min_x + f64::from(x) * config.cell_size;
				let pz = min_z + f64::from(z) * config.cell_size;
				let radius = px.abs().max(pz.abs());
				let weight = if radius <= guarded_pad {
					0.0
				} else if config.pad_transition_width == 0.0 {
					1.0
				} else {
					smooth_step((radius - guarded_pad) / config.pad_transition_width)
				};
				heights[config.index(x, z)] *= weight;
			}
		}
		let max_height = heights
			.iter()
			.fold(0.0_f64, |value, height| value.max(height.abs()));
		let mut max_slope = 0.0_f64;
		for z in 0..config.cells_z {
			for x in 0..config.cells_x {
				let h = [
					heights[config.index(x, z)],
					heights[config.index(x + 1, z)],
					heights[config.index(x, z + 1)],
					heights[config.index(x + 1, z + 1)],
				];
				max_slope = max_slope
					.max(triangle_slope(
						h,
						config.cell_size,
						LunarTerrainTriangle::LowerRight,
					))
					.max(triangle_slope(
						h,
						config.cell_size,
						LunarTerrainTriangle::UpperLeft,
					));
			}
		}
		let mut scale = 1.0_f64;
		if max_height > 0.0 {
			scale = scale.min(config.max_abs_height / max_height);
		}
		if max_slope > 0.0 {
			scale = scale.min(config.max_slope / max_slope);
		}
		for height in &mut heights {
			*height *= scale;
		}
		Self::from_heights(config, heights)
	}

	/// Build a terrain tile from row-major vertex heights.
	///
	/// # Errors
	///
	/// Returns an error when the configuration is invalid, the height count
	/// differs from `(cells_x + 1) * (cells_z + 1)`, a height is non-finite or
	/// out of range, or either triangle exceeds the configured slope bound.
	pub fn from_heights(config: LunarTerrainConfig, heights: Vec<f64>) -> Result<Self> {
		config.validate()?;
		if heights.len() != config.vertex_count() {
			return Err(Error::invalid_argument(
				"lunar terrain height count does not match its grid",
			));
		}
		if heights
			.iter()
			.any(|height| !height.is_finite() || height.abs() > config.max_abs_height + 1.0e-12)
		{
			return Err(Error::invalid_argument(
				"lunar terrain contains an invalid height",
			));
		}
		for z in 0..config.cells_z {
			for x in 0..config.cells_x {
				let h = [
					heights[config.index(x, z)],
					heights[config.index(x + 1, z)],
					heights[config.index(x, z + 1)],
					heights[config.index(x + 1, z + 1)],
				];
				if triangle_slope(h, config.cell_size, LunarTerrainTriangle::LowerRight)
					> config.max_slope + 1.0e-12
					|| triangle_slope(h, config.cell_size, LunarTerrainTriangle::UpperLeft)
						> config.max_slope + 1.0e-12
				{
					return Err(Error::invalid_argument(
						"lunar terrain exceeds its triangle-slope bound",
					));
				}
			}
		}
		Ok(Self { config, heights })
	}

	/// Return the checked terrain configuration.
	pub const fn config(&self) -> LunarTerrainConfig {
		self.config
	}

	/// Return row-major vertex heights without copying them.
	pub fn heights(&self) -> &[f64] {
		&self.heights
	}

	/// Return the inclusive minimum world-space X coordinate.
	pub fn min_x(&self) -> f64 {
		-f64::from(self.config.cells_x) * self.config.cell_size * 0.5
	}
	/// Return the inclusive maximum world-space X coordinate.
	pub fn max_x(&self) -> f64 {
		-self.min_x()
	}
	/// Return the inclusive minimum world-space Z coordinate.
	pub fn min_z(&self) -> f64 {
		-f64::from(self.config.cells_z) * self.config.cell_size * 0.5
	}
	/// Return the inclusive maximum world-space Z coordinate.
	pub fn max_z(&self) -> f64 {
		-self.min_z()
	}
	/// Return whether a finite world-space coordinate lies inside the tile.
	pub fn contains(&self, x: f64, z: f64) -> bool {
		x.is_finite()
			&& z.is_finite()
			&& x >= self.min_x()
			&& x <= self.max_x()
			&& z >= self.min_z()
			&& z <= self.max_z()
	}
	/// Return whether a coordinate lies on the square landing pad.
	pub fn is_on_pad(&self, x: f64, z: f64) -> bool {
		self.contains(x, z) && x.abs().max(z.abs()) <= self.config.pad_half_extent
	}
	/// Return a grid vertex height, or `None` when either index is out of range.
	pub fn vertex_height(&self, x: u32, z: u32) -> Option<f64> {
		(x <= self.config.cells_x && z <= self.config.cells_z)
			.then_some(self.heights[self.config.index(x, z)])
	}

	/// Interpolate terrain height and return the selected triangle's normal.
	///
	/// Coordinates outside the tile return a sample with `in_bounds == false`.
	pub fn query(&self, x: f64, z: f64) -> LunarTerrainSample {
		if !self.contains(x, z) {
			return LunarTerrainSample::default();
		}
		let gx = (x - self.min_x()) / self.config.cell_size;
		let gz = (z - self.min_z()) / self.config.cell_size;
		let (cell_x, local_x) = if gx == f64::from(self.config.cells_x) {
			(self.config.cells_x - 1, 1.0)
		} else {
			let c = gx.floor() as u32;
			(c, gx - f64::from(c))
		};
		let (cell_z, local_z) = if gz == f64::from(self.config.cells_z) {
			(self.config.cells_z - 1, 1.0)
		} else {
			let c = gz.floor() as u32;
			(c, gz - f64::from(c))
		};
		let h00 = self.heights[self.config.index(cell_x, cell_z)];
		let h10 = self.heights[self.config.index(cell_x + 1, cell_z)];
		let h01 = self.heights[self.config.index(cell_x, cell_z + 1)];
		let h11 = self.heights[self.config.index(cell_x + 1, cell_z + 1)];
		let (triangle, height, dx, dz) = if local_z <= local_x {
			(
				LunarTerrainTriangle::LowerRight,
				h00 + local_x * (h10 - h00) + local_z * (h11 - h10),
				(h10 - h00) / self.config.cell_size,
				(h11 - h10) / self.config.cell_size,
			)
		} else {
			(
				LunarTerrainTriangle::UpperLeft,
				h00 + local_x * (h11 - h01) + local_z * (h01 - h00),
				(h11 - h01) / self.config.cell_size,
				(h01 - h00) / self.config.cell_size,
			)
		};
		let normal = DVec3 {
			x: -dx,
			y: 1.0,
			z: -dz,
		}
		.try_normalized()
		.unwrap_or(DVec3 {
			x: 0.0,
			y: 1.0,
			z: 0.0,
		});
		LunarTerrainSample {
			in_bounds: true,
			height,
			normal,
			cell_x,
			cell_z,
			local_x,
			local_z,
			triangle,
		}
	}
}

fn terrain_mix(mut value: u64) -> u64 {
	value ^= value >> 30;
	value = value.wrapping_mul(0xbf58_476d_1ce4_e5b9);
	value ^= value >> 27;
	value = value.wrapping_mul(0x94d0_49bb_1331_11eb);
	value ^ (value >> 31)
}
fn signed_unit(seed: u64, x: u32, z: u32) -> f64 {
	let bits = terrain_mix(
		seed ^ u64::from(x).wrapping_mul(TERRAIN_HASH_X)
			^ u64::from(z).wrapping_mul(TERRAIN_HASH_Z),
	);
	((bits >> 11) as f64) * (1.0 / 9_007_199_254_740_992.0) * 2.0 - 1.0
}
fn smooth_step(value: f64) -> f64 {
	let value = value.clamp(0.0, 1.0);
	value * value * (3.0 - 2.0 * value)
}
fn triangle_slope(h: [f64; 4], size: f64, triangle: LunarTerrainTriangle) -> f64 {
	let (dx, dz) = match triangle {
		LunarTerrainTriangle::LowerRight => ((h[1] - h[0]) / size, (h[3] - h[1]) / size),
		LunarTerrainTriangle::UpperLeft => ((h[3] - h[2]) / size, (h[2] - h[0]) / size),
	};
	dx.hypot(dz)
}
