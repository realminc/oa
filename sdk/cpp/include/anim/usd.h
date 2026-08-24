#pragma once

// Usd — minimal in-tree `.usda` UsdSkel reader/writer (read + write).
//
// Exactly the prims the squashed-USD dataset format uses, nothing more
// (OaDsGen3dAnimCtl.md): SkelRoot -> skeleton (joints + bind/rest) -> SkelAnimation
// (per-frame translations + quat rotations as timeSamples). NOT a USD
// composition engine — no layers/references/variants/Crate/Python.
//
// UsdSkelClip is the on-disk representation (joint paths, bind/rest, raw
// trans+quat per frame). It is deliberately ML-free: the lossy conversion to
// the model's packed canonical channels (quat→6D, channel-pack, derive
// contacts) lives in PoseClip packing (PosePack.h), not here, so all USD
// parsing complexity is quarantined in this one file (§4.0).

#include <oa/core/filesystem.h>
#include <oa/core/vlm.h>
#include <oa/core/types.h>

// One UsdSkel SkelAnimation clip as stored on disk.
//
// translations/rotations are flat, frame-major: element [f*jointCount + j] is
// joint j at frame f. rotations are unit quaternions (USD quatf order is
// (w,x,y,z); stored here in oa::vlm::Quat's (x,y,z,w) fields). Bind/restTransforms are
// one matrix per joint (USD matrix4d, row-major, translation in row 3).
namespace oa {

struct UsdSkelClip {
	oa::Vec<oa::String> jointPaths;       // full UsdSkel paths, e.g. "root/pelvis/spine_01"
	oa::Vec<oa::vlm::Mat4>   bindTransforms;   // per joint, world bind pose
	oa::Vec<oa::vlm::Mat4>   restTransforms;   // per joint, local rest pose
	oa::U32           frameCount = 0;
	oa::F32           fps        = 30.0f;
	// stage up-axis as a Vec3 component index: 2 = Z (our writer's default), 1 = Y
	// (Maya/UE exports). Used by the packer to pick the contact-floor axis.
	oa::I32           upAxis     = 2;
	oa::Vec<oa::vlm::Vec3>   translations;     // frameCount * jointCount
	oa::Vec<oa::vlm::Quat>   rotations;        // frameCount * jointCount

	[[nodiscard]] oa::I32 jointCount() const noexcept {
		return static_cast<oa::I32>(jointPaths.size());
	}
	[[nodiscard]] bool isValid() const noexcept;
};

// One named SkelAnimation prim within a multi-clip dataset stage. `split` is the
// dataset partition tag ("train" / "val" / "test"), round-tripped via the prim's
// `customData.oa` dictionary. `Name` is the SkelAnimation prim name (source stem).
struct UsdNamedClip {
	oa::String      name;
	oa::String      split = "train";
	UsdSkelClip clip;
};

namespace Usd {

// Emit `clip` as a valid `.usda` SkelRoot/Skeleton/SkelAnimation stage (Z-up,
// cm). Mirrors Tools/Gen3dAnim/walk_to_usda.py. `defaultPrim` names the
// SkelRoot prim.
[[nodiscard]] oa::Status writeUsda(const oa::Path& inPath,
                                 const UsdSkelClip& inClip,
                                 oa::StringView inDefaultPrim = "rig");

// parse a `.usda` produced by WriteUsda / a stripped UE export. Recovers joint
// paths, bind/rest transforms (if present), and the per-frame trans+quat
// timeSamples. Tolerant of the subset only — composition arcs are unsupported.
[[nodiscard]] oa::Result<UsdSkelClip> readUsda(const oa::Path& inPath);

// Combined-dataset form: one SkelRoot + one skeleton (joints/bind/rest taken from
// the first clip) with N `def SkelAnimation` prims, one per clip. The on-disk
// dataset for Gen3dAnim — a single human-readable, usdview-renderable `.usda`.
[[nodiscard]] oa::Status writeUsdaMulti(const oa::Path& inPath,
                                      oa::Span<const UsdNamedClip> inClips,
                                      oa::StringView inDefaultPrim = "rig");

// Read every SkelAnimation prim out of a WriteUsdaMulti stage. The shared
// Skeleton bind/rest transforms are applied to each returned clip. Also reads a
// single-anim stage as a one-element vector.
[[nodiscard]] oa::Result<oa::Vec<UsdNamedClip>> readUsdaMulti(const oa::Path& inPath);

} // namespace Usd

}  // namespace oa
