//! Locked skeletal-joint transform value.
//!
//! The matrix contract follows Maya's joint transform semantics while
//! retaining OA composition: scale/shear, scale orientation (rotateAxis),
//! animated rotation, joint orientation, optional parent-scale compensation,
//! translation.
//!
//! Donor: `oa/animation/joint.h` and `oa/animation/fnJoint.{h,cpp}`. The
//! donor's `Joint` methods already forward every behavior to `FnJoint`; the
//! Rust port keeps the methods as the single implementation and drops the
//! separate operation namespace.

use crate::{
	Error, Result,
	core::Transform,
	vlm::{AffineDecomposition, Mat4, Quat, RotationOrder, Vec3},
};

/// Per-axis degrees-of-freedom flags.
///
/// Donor: `oa::JointDegreesOfFreedom`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct JointDegreesOfFreedom {
	pub x: bool,
	pub y: bool,
	pub z: bool,
}

impl Default for JointDegreesOfFreedom {
	fn default() -> Self {
		Self {
			x: true,
			y: true,
			z: true,
		}
	}
}

/// Optional per-axis rotation limits in degrees.
///
/// An axis is clamped only when the matching enabled flag is set; enabling
/// both bounds on an axis requires `minimum <= maximum` on it. Limits are
/// disabled by default even though [`JointDegreesOfFreedom::default`] is
/// all-enabled.
///
/// Donor: `oa::JointRotationLimits`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct JointRotationLimits {
	pub minimum_degrees: Vec3,
	pub maximum_degrees: Vec3,
	pub minimum_enabled: JointDegreesOfFreedom,
	pub maximum_enabled: JointDegreesOfFreedom,
}

impl Default for JointRotationLimits {
	fn default() -> Self {
		Self {
			minimum_degrees: Vec3::default(),
			maximum_degrees: Vec3::default(),
			minimum_enabled: JointDegreesOfFreedom {
				x: false,
				y: false,
				z: false,
			},
			maximum_enabled: JointDegreesOfFreedom {
				x: false,
				y: false,
				z: false,
			},
		}
	}
}

/// Locked skeletal-joint transform value.
///
/// Donor: `oa::Joint`. Setters that the donor guards with contract aborts
/// return `Result` here instead.
#[derive(Clone, Debug, PartialEq)]
pub struct Joint {
	transform: Transform,
	orientation: Quat,
	scale_orientation: Quat,
	segment_scale_compensate: bool,
	rotation_order: RotationOrder,
	degrees_of_freedom: JointDegreesOfFreedom,
	rotation_limits: JointRotationLimits,
	preferred_angles: Vec3,
	stiffness: Vec3,
}

impl Default for Joint {
	fn default() -> Self {
		Self {
			transform: Transform::default(),
			orientation: Quat::identity(),
			scale_orientation: Quat::identity(),
			segment_scale_compensate: true,
			rotation_order: RotationOrder::Xyz,
			degrees_of_freedom: JointDegreesOfFreedom::default(),
			rotation_limits: JointRotationLimits::default(),
			preferred_angles: Vec3::default(),
			stiffness: Vec3::default(),
		}
	}
}

impl From<Transform> for Joint {
	fn from(transform: Transform) -> Self {
		Self {
			transform,
			..Self::default()
		}
	}
}

fn valid_quaternion(value: Quat) -> bool {
	value.is_finite() && value.norm_squared().is_finite() && value.norm_squared() > 0.0
}

fn valid_stiffness(value: Vec3) -> bool {
	value.is_finite() && value.x >= 0.0 && value.y >= 0.0 && value.z >= 0.0
}

fn valid_limits(limits: &JointRotationLimits) -> bool {
	if !limits.minimum_degrees.is_finite() || !limits.maximum_degrees.is_finite() {
		return false;
	}
	let x_valid = !(limits.minimum_enabled.x && limits.maximum_enabled.x)
		|| limits.minimum_degrees.x <= limits.maximum_degrees.x;
	let y_valid = !(limits.minimum_enabled.y && limits.maximum_enabled.y)
		|| limits.minimum_degrees.y <= limits.maximum_degrees.y;
	let z_valid = !(limits.minimum_enabled.z && limits.maximum_enabled.z)
		|| limits.minimum_degrees.z <= limits.maximum_degrees.z;
	x_valid && y_valid && z_valid
}

fn clamp_axis(
	value: f32,
	minimum: f32,
	maximum: f32,
	minimum_enabled: bool,
	maximum_enabled: bool,
) -> f32 {
	if minimum_enabled && value < minimum {
		return minimum;
	}
	if maximum_enabled && value > maximum {
		return maximum;
	}
	value
}

impl Joint {
	/// Validate the complete authoring state.
	///
	/// # Errors
	///
	/// Returns an error for a non-finite transform or orientation, an
	/// inverted enabled limit pair, non-finite preferred angles, or
	/// non-finite/negative stiffness.
	pub fn validate(&self) -> Result<()> {
		self.transform.validate()?;
		if !valid_quaternion(self.orientation) || !valid_quaternion(self.scale_orientation) {
			return Err(Error::invalid_argument(
				"Joint orientations must be finite non-zero quaternions",
			));
		}
		if !valid_limits(&self.rotation_limits) {
			return Err(Error::invalid_argument("Joint rotation limits are invalid"));
		}
		if !self.preferred_angles.is_finite() {
			return Err(Error::invalid_argument(
				"Joint preferred angles must be finite",
			));
		}
		if !valid_stiffness(self.stiffness) {
			return Err(Error::invalid_argument(
				"Joint stiffness must be finite and non-negative",
			));
		}
		Ok(())
	}

	/// Animated local transform.
	pub fn transform(&self) -> &Transform {
		&self.transform
	}

	/// Mutable access to the animated local transform.
	pub fn transform_mut(&mut self) -> &mut Transform {
		&mut self.transform
	}

	/// Joint orientation (Maya jointOrient); normalized on assignment.
	///
	/// # Errors
	///
	/// Returns an error for a non-finite or zero quaternion.
	pub fn set_orientation(&mut self, orientation: Quat) -> Result<()> {
		if !valid_quaternion(orientation) {
			return Err(Error::invalid_argument(
				"Joint orientation must be a finite non-zero quaternion",
			));
		}
		self.orientation = orientation.try_normalized().unwrap_or(orientation);
		Ok(())
	}

	/// Stored joint orientation.
	pub fn orientation(&self) -> Quat {
		self.orientation
	}

	/// Scale orientation (Maya rotateAxis); normalized on assignment.
	///
	/// # Errors
	///
	/// Returns an error for a non-finite or zero quaternion.
	pub fn set_scale_orientation(&mut self, orientation: Quat) -> Result<()> {
		if !valid_quaternion(orientation) {
			return Err(Error::invalid_argument(
				"Joint scale orientation must be a finite non-zero quaternion",
			));
		}
		self.scale_orientation = orientation.try_normalized().unwrap_or(orientation);
		Ok(())
	}

	/// Stored scale orientation.
	pub fn scale_orientation(&self) -> Quat {
		self.scale_orientation
	}

	/// Enable or disable parent-scale compensation.
	pub fn set_segment_scale_compensate(&mut self, enabled: bool) {
		self.segment_scale_compensate = enabled;
	}

	/// Whether parent-scale compensation applies.
	pub fn segment_scale_compensate(&self) -> bool {
		self.segment_scale_compensate
	}

	/// Euler decomposition order retained as metadata.
	pub fn set_rotation_order(&mut self, order: RotationOrder) {
		self.rotation_order = order;
	}

	/// Stored rotation order.
	pub fn rotation_order(&self) -> RotationOrder {
		self.rotation_order
	}

	/// Degrees-of-freedom flags.
	pub fn set_degrees_of_freedom(&mut self, degrees_of_freedom: JointDegreesOfFreedom) {
		self.degrees_of_freedom = degrees_of_freedom;
	}

	/// Stored degrees-of-freedom flags.
	pub fn degrees_of_freedom(&self) -> JointDegreesOfFreedom {
		self.degrees_of_freedom
	}

	/// Rotation limits in degrees.
	///
	/// # Errors
	///
	/// Returns an error for non-finite bounds or an axis whose enabled
	/// minimum exceeds its enabled maximum.
	pub fn set_rotation_limits(&mut self, limits: JointRotationLimits) -> Result<()> {
		if !valid_limits(&limits) {
			return Err(Error::invalid_argument("Joint rotation limits are invalid"));
		}
		self.rotation_limits = limits;
		Ok(())
	}

	/// Stored rotation limits.
	pub fn rotation_limits(&self) -> JointRotationLimits {
		self.rotation_limits
	}

	/// Preferred rest angles in degrees.
	///
	/// # Errors
	///
	/// Returns an error for non-finite angles.
	pub fn set_preferred_angles(&mut self, degrees: Vec3) -> Result<()> {
		if !degrees.is_finite() {
			return Err(Error::invalid_argument(
				"Joint preferred angles must be finite",
			));
		}
		self.preferred_angles = degrees;
		Ok(())
	}

	/// Stored preferred angles in degrees.
	pub fn preferred_angles(&self) -> Vec3 {
		self.preferred_angles
	}

	/// Per-axis stiffness in `[0, ∞)`.
	///
	/// # Errors
	///
	/// Returns an error for non-finite or negative stiffness.
	pub fn set_stiffness(&mut self, stiffness: Vec3) -> Result<()> {
		if !valid_stiffness(stiffness) {
			return Err(Error::invalid_argument(
				"Joint stiffness must be finite and non-negative",
			));
		}
		self.stiffness = stiffness;
		Ok(())
	}

	/// Stored stiffness.
	pub fn stiffness(&self) -> Vec3 {
		self.stiffness
	}

	/// Animated rotation expressed in joint orientation space:
	/// `orientation * transform.rotation`, normalized.
	pub fn oriented_rotation(&self) -> Quat {
		(self.orientation * self.transform.rotation())
			.try_normalized()
			.unwrap_or_else(Quat::identity)
	}

	/// Clamp rotation angles in degrees to the enabled limits.
	///
	/// # Errors
	///
	/// Returns an error for non-finite input.
	pub fn clamp_rotation_degrees(&self, degrees: Vec3) -> Result<Vec3> {
		if !degrees.is_finite() {
			return Err(Error::invalid_argument(
				"Joint rotation angles must be finite",
			));
		}
		let limits = &self.rotation_limits;
		Ok(Vec3 {
			x: clamp_axis(
				degrees.x,
				limits.minimum_degrees.x,
				limits.maximum_degrees.x,
				limits.minimum_enabled.x,
				limits.maximum_enabled.x,
			),
			y: clamp_axis(
				degrees.y,
				limits.minimum_degrees.y,
				limits.maximum_degrees.y,
				limits.minimum_enabled.y,
				limits.maximum_enabled.y,
			),
			z: clamp_axis(
				degrees.z,
				limits.minimum_degrees.z,
				limits.maximum_degrees.z,
				limits.minimum_enabled.z,
				limits.maximum_enabled.z,
			),
		})
	}

	/// Compose the joint matrix in Maya factor order:
	/// `S*Shear * SO * R * O * P^-1 * T` under the row-vector convention.
	///
	/// When segment-scale compensation is enabled, `parent_scale` must be
	/// finite and non-zero on every axis.
	///
	/// # Errors
	///
	/// Returns an error when the joint state is invalid, `parent_scale` is
	/// non-finite, or compensation needs a zero parent-scale axis.
	pub fn matrix(&self, parent_scale: Vec3) -> Result<Mat4> {
		self.validate()?;
		if !parent_scale.is_finite() {
			return Err(Error::invalid_argument("Joint parent scale must be finite"));
		}
		let mut inverse_parent_scale = Vec3 {
			x: 1.0,
			y: 1.0,
			z: 1.0,
		};
		if self.segment_scale_compensate {
			if parent_scale.x == 0.0 || parent_scale.y == 0.0 || parent_scale.z == 0.0 {
				return Err(Error::invalid_argument(
					"Joint segment-scale compensation requires non-zero parent scale",
				));
			}
			inverse_parent_scale = Vec3 {
				x: 1.0 / parent_scale.x,
				y: 1.0 / parent_scale.y,
				z: 1.0 / parent_scale.z,
			};
		}
		let scale_and_shear = AffineDecomposition {
			scale: self.transform.scale(),
			shear: self.transform.shear(),
			..AffineDecomposition::default()
		};
		let compose = |quat: Quat| {
			Mat4::try_from_quaternion(quat)
				.ok_or_else(|| Error::invalid_argument("Joint quaternions must be normalizable"))
		};
		Ok(
			Mat4::compose_affine(scale_and_shear)
				.ok_or_else(|| Error::invalid_argument("Joint scale/shear must be finite"))?
				* compose(self.scale_orientation)?
				* compose(self.transform.rotation())?
				* compose(self.orientation)?
				* Mat4::scaling(inverse_parent_scale)
				* Mat4::translation(self.transform.position()),
		)
	}
}
