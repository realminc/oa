#pragma once

// OaSkeleton — a UE-Skeleton-asset-style joint definition for the 3D anim stack.
// Data derived from Epic MetaHuman/CitySample + Autodesk HumanIK — see NOTICE.md.
//
// The skeleton is the *identity* layer that OaPoseClip lacks: it names every
// joint, fixes the parent hierarchy and joint order (⇒ fixed D_pose), and
// carries each joint's rest transform (an OaJoint: local offset + rest
// orientation) so the IO layer can convert between USD SkelAnimation (joints +
// bind/rest + trans/quat) and the model's packed canonical channels (root
// trans+6D, per-joint 6D, contacts).
//
// It also carries per-limb physics priors — bone length (cm, = |rest offset to
// parent|) and a mass approximation (kg) — so motion/physics work can reason
// about segment inertia and joint distances without a separate rig.
//
// Identity + rest data come from oa3d's rtg definition
// (oa3d/oa/maya/resources/rtgdefinition/unreal.py): `RTG_DEFINITION_UNREAL` for
// the HumanIK ids and joint set, and `RTG_POSES_SKEL_MANNY` (tPose) for the real
// UE5 mannequin rest translate/orient — so a clip retargeted through HumanIK ⇄
// MetaHuman lands on exactly these joints. Masses remain ~80 kg Dempster-ish
// approximations.
//
// Builders: OaSkMetaHuman() (the clean UE body), OaSkHumanIk() (the HumanIK
// characterization rig used as the retarget hub), and OaSkHumanMl3d() (the
// HumanML3D/SMPL 22-joint body for the text-to-motion datasets).

#include <Core/Joint.h>
#include <Oa/Core/FileIo.h>
#include <Oa/Core/Types.h>

// One joint in the skeleton. Joints are stored in a flat array in hierarchy
// order (every parent precedes its children), so ParentIndex < own index, and
// the root joint is index 0 with ParentIndex == -1.
struct OaSkelJoint {
	OaString Name;                  // UE bone name, e.g. "thigh_l"
	OaI32    ParentIndex = -1;      // index into OaSkeleton::Joints; -1 for root
	OaI32    HumanIkId   = 0;       // HumanIK slot id from the rtg definition
	OaF32    Mass        = 0.0f;    // segment mass approximation, kg
	OaJoint  Rest;                  // rest transform: Translate = local offset (cm),
	                                // JointOrient = rest orientation, Rotate = identity

	// Channel spec (from the rig's LockSkeleton): which DOFs are actually animated.
	// Locked channels are not stored/predicted — they stay at rest.
	bool     HasTranslate = false;  // live translate channel (root + pelvis only)
	OaU8     RotDof       = 3;      // 3 = full rotation (6D), 1 = hinge (rotateZ), 0 = none

	// Bone length (cm) = |local offset to parent|. Root / zero-offset → 0.
	[[nodiscard]] OaF32 Length() const noexcept {
		return Vlm::Length(Rest.Translate);
	}

	// Canonical channels this joint contributes: translate (3 if live) + rotation
	// (6 for full 6D, 1 for a hinge angle, 0 if fully locked).
	[[nodiscard]] OaI32 ChannelCount() const noexcept {
		const OaI32 t = HasTranslate ? 3 : 0;
		const OaI32 r = (RotDof == 3) ? 6 : (RotDof == 1 ? 1 : 0);
		return t + r;
	}
};

struct OaSkeleton {
	static constexpr OaU32 FormatVersion = 2;   // v2: rest carries orientation

	OaString             Name       = "metahuman_body";
	OaU32                SkeletonId = 0;   // matches OaPoseClip::SkeletonId
	OaVec<OaSkelJoint>   Joints;
	// Indices into Joints of the foot joints whose contacts are packed (in
	// channel order). Two for a biped: { foot_l, foot_r }.
	OaVec<OaI32>         ContactJoints;

	[[nodiscard]] OaI32 JointCount() const noexcept {
		return static_cast<OaI32>(Joints.Size());
	}

	// Channel budget for OaPoseClip::PoseDim under the compact canonical layout:
	// each joint contributes only its live channels (see OaSkelJoint::ChannelCount),
	// in joint order, followed by one soft-contact channel per ContactJoints entry.
	//   root/pelvis : translate(3) + 6D(6) = 9 each
	//   regular     : 6D(6)
	//   hinge       : rotateZ angle(1)
	//   contacts    : C trailing
	[[nodiscard]] OaI32 PoseDim() const noexcept {
		OaI32 d = 0;
		for (const OaSkelJoint& j : Joints) {
			d += j.ChannelCount();
		}
		return d + static_cast<OaI32>(ContactJoints.Size());
	}

	// Canonical channel offset where joint InJoint's block begins (sum of the
	// channel counts of all earlier joints). Contacts trail all joint blocks.
	[[nodiscard]] OaI32 ChannelOffset(OaI32 InJoint) const noexcept {
		OaI32 off = 0;
		for (OaI32 k = 0; k < InJoint && k < JointCount(); ++k) {
			off += Joints[static_cast<OaUsize>(k)].ChannelCount();
		}
		return off;
	}

	// Offset of the first contact channel (= total joint channels).
	[[nodiscard]] OaI32 ContactOffset() const noexcept {
		return ChannelOffset(JointCount());
	}

	// Find a joint index by UE bone name; -1 if absent.
	[[nodiscard]] OaI32 IndexOf(OaStringView InName) const noexcept;

	// Bind-pose (rest) world position of a joint via forward kinematics over the
	// per-joint rest OaJoints (orientation-aware). cm, Z-up.
	[[nodiscard]] VlmVec3 RestWorld(OaI32 InJoint) const noexcept;

	// Bind-pose world *orientation* of a joint (accumulated rest orientations).
	[[nodiscard]] VlmQuat RestWorldRotation(OaI32 InJoint) const noexcept;

	[[nodiscard]] bool IsValid() const noexcept;

	// `.skel` JSON IO (human-readable / editable; the built-in builders below are
	// the authoritative defaults, so the model path never depends on parsing).
	[[nodiscard]] static OaResult<OaSkeleton> ReadSkel(const OaPath& InPath);
	[[nodiscard]] OaStatus WriteSkel(const OaPath& InPath) const;
};

// The canonical clean MetaHuman base skeleton (SkeletonId 0): 64 joints =
// LIST_SKEL_MH minus individual toes (full body + neck/head + both arms with all
// fingers + legs to ball). Rest seeded from manny's tPose; per-joint channel spec
// from the rig's LockSkeleton. Corrective/twist/IK joints are never part of it.
[[nodiscard]] const OaSkeleton& OaSkMetaHuman();

// The HumanIK characterization skeleton (the named HumanIK slots → UE nodes),
// used as the hub rig for retargeting (SkeletonId 1).
[[nodiscard]] const OaSkeleton& OaSkHumanIk();

// The HumanML3D / SMPL 22-joint body (SkeletonId 2) — the skeleton the HumanML3D
// text-to-motion datasets and AnimationGPT's CMP are authored on (t2m kinematic
// chain). Joint order matches the 263-dim feature vector and `new_joints` arrays.
// Rest offsets are placeholders (SkHumanMl3d.inc); the USD preview supplies
// per-frame world joint positions directly.
[[nodiscard]] const OaSkeleton& OaSkHumanMl3d();

// Back-compat alias for the MetaHuman body (pre-refactor name).
[[nodiscard]] inline const OaSkeleton& OaMetaHumanBodySkeleton() {
	return OaSkMetaHuman();
}
