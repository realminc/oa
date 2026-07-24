#pragma once

// OaUsd — minimal in-tree `.usda` UsdSkel reader/writer (read + write).
//
// Exactly the prims the squashed-USD dataset format uses, nothing more
// (OaDsGen3dAnimCtl.md): SkelRoot -> Skeleton (joints + bind/rest) -> SkelAnimation
// (per-frame translations + quat rotations as timeSamples). NOT a USD
// composition engine — no layers/references/variants/Crate/Python.
//
// OaUsdSkelClip is the on-disk representation (joint paths, bind/rest, raw
// trans+quat per frame). It is deliberately ML-free: the lossy conversion to
// the model's packed canonical channels (quat→6D, channel-pack, derive
// contacts) lives in PoseClip packing (PosePack.h), not here, so all USD
// parsing complexity is quarantined in this one file (§4.0).

#include <Oa/Core/Filesystem.h>
#include <Oa/Core/Vlm.h>
#include <Oa/Core/Types.h>

// One UsdSkel SkelAnimation clip as stored on disk.
//
// Translations/Rotations are flat, frame-major: element [f*JointCount + j] is
// joint j at frame f. Rotations are unit quaternions (USD quatf order is
// (w,x,y,z); stored here in VlmQuat's (x,y,z,w) fields). Bind/RestTransforms are
// one matrix per joint (USD matrix4d, row-major, translation in row 3).
struct OaUsdSkelClip {
	OaVec<OaString> JointPaths;       // full UsdSkel paths, e.g. "root/pelvis/spine_01"
	OaVec<VlmMat4>   BindTransforms;   // per joint, world bind pose
	OaVec<VlmMat4>   RestTransforms;   // per joint, local rest pose
	OaU32           FrameCount = 0;
	OaF32           Fps        = 30.0f;
	// Stage up-axis as a Vec3 component index: 2 = Z (our writer's default), 1 = Y
	// (Maya/UE exports). Used by the packer to pick the contact-floor axis.
	OaI32           UpAxis     = 2;
	OaVec<VlmVec3>   Translations;     // FrameCount * JointCount
	OaVec<VlmQuat>   Rotations;        // FrameCount * JointCount

	[[nodiscard]] OaI32 JointCount() const noexcept {
		return static_cast<OaI32>(JointPaths.Size());
	}
	[[nodiscard]] bool IsValid() const noexcept;
};

// One named SkelAnimation prim within a multi-clip dataset stage. `Split` is the
// dataset partition tag ("train" / "val" / "test"), round-tripped via the prim's
// `customData.oa` dictionary. `Name` is the SkelAnimation prim name (source stem).
struct OaUsdNamedClip {
	OaString      Name;
	OaString      Split = "train";
	OaUsdSkelClip Clip;
};

namespace OaUsd {

// Emit `clip` as a valid `.usda` SkelRoot/Skeleton/SkelAnimation stage (Z-up,
// cm). Mirrors Tools/Gen3dAnim/walk_to_usda.py. `defaultPrim` names the
// SkelRoot prim.
[[nodiscard]] OaStatus WriteUsda(const OaPath& InPath,
                                 const OaUsdSkelClip& InClip,
                                 OaStringView InDefaultPrim = "rig");

// Parse a `.usda` produced by WriteUsda / a stripped UE export. Recovers joint
// paths, bind/rest transforms (if present), and the per-frame trans+quat
// timeSamples. Tolerant of the subset only — composition arcs are unsupported.
[[nodiscard]] OaResult<OaUsdSkelClip> ReadUsda(const OaPath& InPath);

// Combined-dataset form: one SkelRoot + one Skeleton (joints/bind/rest taken from
// the first clip) with N `def SkelAnimation` prims, one per clip. The on-disk
// dataset for Gen3dAnim — a single human-readable, usdview-renderable `.usda`.
[[nodiscard]] OaStatus WriteUsdaMulti(const OaPath& InPath,
                                      OaSpan<const OaUsdNamedClip> InClips,
                                      OaStringView InDefaultPrim = "rig");

// Read every SkelAnimation prim out of a WriteUsdaMulti stage. The shared
// Skeleton bind/rest transforms are applied to each returned clip. Also reads a
// single-anim stage as a one-element vector.
[[nodiscard]] OaResult<OaVec<OaUsdNamedClip>> ReadUsdaMulti(const OaPath& InPath);

} // namespace OaUsd
