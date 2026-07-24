#pragma once

// OaFbxWriter -- export-only ASCII FBX 7500 writer for DCC review. Writes a
// Z-up / cm skeleton hierarchy + animation curves shaped like a real FBX SDK /
// Unreal export, so the Autodesk importer (Maya, FBX Review) accepts it. There
// is deliberately NO FBX reader -- ingest is USD-only.
//
// Input is an OaUsdSkelClip, the same struct OaUsd reads/writes, so the FBX
// path is purely "USD clip -> viewable FBX"; the model never round-trips through
// FBX. The header / SceneInfo / PropertyTemplates / Takes ceremony is
// load-bearing; the format rejects incomplete files.

#include <Oa/Core/Filesystem.h>
#include <Oa/Core/Types.h>

struct OaUsdSkelClip;

namespace OaFbx {

// Emit `clip` as an ASCII FBX 7500 file. Bone hierarchy is recovered from the
// clip's slash-delimited joint paths; rest pose comes from frame 0; rotation
// curves are written for joints whose orientation changes, translation curves
// for joints whose position changes, typically just the root.
[[nodiscard]] OaStatus WriteFbx(const OaPath& InPath, const OaUsdSkelClip& InClip);

} // namespace OaFbx
