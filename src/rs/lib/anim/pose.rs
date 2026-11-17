//! Validated local-transform sample for one skeleton.
//!
//! Donor: `oa/animation/skelPose.h`.

use crate::core::Transform;

/// One validated local-transform sample for a skeleton.
///
/// Construction is validated through [`super::create_pose`]; every joint
/// transform is finite with a normalizable rotation.
///
/// Donor: `oa::SkelPose`.
#[derive(Clone, Debug, PartialEq)]
pub struct SkelPose {
	pub(crate) skeleton_id: u32,
	pub(crate) local_transforms: Vec<Transform>,
}

impl SkelPose {
	/// Skeleton identity carried through sampling.
	pub fn skeleton_id(&self) -> u32 {
		self.skeleton_id
	}

	/// Number of local joint transforms stored.
	pub fn joint_count(&self) -> u32 {
		self.local_transforms.len() as u32
	}

	/// All local joint transforms.
	pub fn local_transforms(&self) -> &[Transform] {
		&self.local_transforms
	}

	/// The local transform of `joint`, or `None` when out of range.
	///
	/// The donor's `getLocalTransform` aborts on an out-of-range index; the
	/// Rust port returns `Option` like slice indexing.
	pub fn local_transform(&self, joint: u32) -> Option<&Transform> {
		self.local_transforms.get(joint as usize)
	}
}
