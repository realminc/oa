// oa::DsHumanMl3d — HumanML3D-format motion dataset (oa::Dataset subclass)
//
// A small class hierarchy for the standard HumanML3D layout used by T2M-GPT /
// MoMask / MotionGPT:
//
//   oa::DsHumanMl3d                 generic loader for ANY HumanML3D-format corpus
//        ↑                        (reads Mean/Std + new_joint_vecs + texts + split)
//   ├─ oa::DsCombatMotionProcessed  AnimationGPT CMP set; 263-dim / 22-joint SMPL.
//   └─ oa::DsKitMl                  KIT-ML corpus;        251-dim / 21-joint.
//
// On-disk layout (identical for every corpus):
//
//   <dir>/Mean.npy, Std.npy          per-feature normalization stats [featDim]
//   <dir>/new_joint_vecs/<id>.npy    motion features [frames, featDim] (model input)
//   <dir>/texts/<id>.txt             captions ("text#pos-tags#start#end", all lines kept)
//   <dir>/text_feats/manifest.json   frozen text-encoder identity and feature contract
//   <dir>/text_feats/<id>.npy        optional aligned caption features [Captions, dim]
//   <dir>/<split>.txt                one clip id per line (train|val|test|train_val)
//
// features are loaded RAW and standardized in place: (x - Mean) / std.

#pragma once

#include <oa/data/dataset.h>
#include <oa/core/types.h>
#include <oa/core/matrix.h>

namespace oa {

// ── base: generic HumanML3D-format motion loader ─────────────────────────────
struct HumanMl3dCaption {
	oa::String text;
	oa::F32 startSec = 0.0F;
	oa::F32 endSec = 0.0F;
	bool hasRange = false;
};

// Reference inverse for the root-motion + root-relative-position channels in a
// HumanML3D feature stream. inFeatures must be de-standardized first. The
// returned row-major array is [inFrames, J, 3] in the dataset's canonical
// metric coordinate system.
[[nodiscard]] oa::Vec<oa::F32> humanMl3dRecoverWorldJoints(
	oa::Span<const oa::F32> inFeatures,
	oa::I32 inFrames,
	oa::I32 inFeatDim = 263);

// Mean Euclidean joint-position error between two recovered world-joint arrays.
// Inputs are [frames, joints, 3] in dataset units. The return value is in cm.
[[nodiscard]] oa::F64 humanMl3dMpjpeCm(
	oa::Span<const oa::F32> inPredWorld,
	oa::Span<const oa::F32> inTargetWorld);

/// Geometry/contact diagnostics for denormalized HumanML3D feature streams.
struct HumanMl3dMotionMetrics {
	oa::F64 mpjpeCm = 0.0;
	oa::F64 velocityErrorCmPerFrame = 0.0;
	oa::F64 contactAccuracy = 0.0;
	oa::F64 footSkateCmPerFrame = 0.0;
	oa::Bool ok = false;
};

[[nodiscard]] HumanMl3dMotionMetrics humanMl3dEvaluateMotion(
	oa::Span<const oa::F32> inPredFeatures,
	oa::Span<const oa::F32> inTargetFeatures,
	oa::I32 inFrames,
	oa::I32 inFeatDim = 263,
	oa::F32 inContactThreshold = 0.5F);

class DsHumanMl3d : public Dataset {
public:
	// inDataDir: dataset root. inSplit: split file stem.
	// inMaxClips: 0 = all. inFeatDim: 263 (HumanML3D / SMPL-22) / 251 (KIT-21).
	DsHumanMl3d(const oa::String& inDataDir, const oa::String& inSplit = "train",
		oa::I32 inMaxClips = 0, oa::I32 inFeatDim = 263);

	virtual ~DsHumanMl3d() = default;

	[[nodiscard]] bool ok() const { return ok_; }

	// corpus identity (overridden by named subclasses for logging/eval).
	[[nodiscard]] virtual const char* name() const { return "HumanML3D"; }

	// Dataset — one standardized clip [frames, featDim] per index.
	[[nodiscard]] oa::I64 size() const override { return numClips_; }
	[[nodiscard]] Matrix getItem(oa::I64 inIndex) const override;

	// ── Host accessors (for synchronous window batching) ─────────────────────
	[[nodiscard]] oa::I32 featDim() const { return featDim_; }
	[[nodiscard]] oa::I32 numJoints() const { return numJoints_; }
	[[nodiscard]] oa::I32 numClips() const { return static_cast<oa::I32>(numClips_); }
	[[nodiscard]] oa::I64 totalFrames() const { return totalFrames_; }
	[[nodiscard]] oa::I32 clipFrames(oa::I64 inIndex) const;
	// Row-major [clipFrames(i), featDim] standardized features for clip i.
	[[nodiscard]] const oa::F32* clipData(oa::I64 inIndex) const;
	[[nodiscard]] const oa::String& clipId(oa::I64 inIndex) const {
		return ids_[static_cast<oa::Usize>(inIndex)];
	}
	// Every caption record, including HumanML3D partial-clip time ranges.
	[[nodiscard]] const oa::Vec<HumanMl3dCaption>& clipCaptions(oa::I64 inIndex) const {
		return captions_[static_cast<oa::Usize>(inIndex)];
	}
	// Optional frozen text-encoder rows from text_feats/<id>.npy.
	[[nodiscard]] oa::I32 textFeatureDim() const { return textFeatureDim_; }
	[[nodiscard]] bool hasTextFeatures() const { return textFeatureDim_ > 0; }
	[[nodiscard]] const oa::String& textFeatureFormat() const { return textFeatureFormat_; }
	[[nodiscard]] const oa::String& textFeatureModel() const { return textFeatureModel_; }
	[[nodiscard]] oa::I32 clipTextFeatureCount(oa::I64 inIndex) const {
		const oa::Usize index = static_cast<oa::Usize>(inIndex);
		return static_cast<oa::I32>(textFeatureOffsets_[index + 1] - textFeatureOffsets_[index]);
	}
	[[nodiscard]] const oa::F32* clipTextFeatureData(oa::I64 inIndex) const {
		const oa::Usize index = static_cast<oa::Usize>(inIndex);
		return textFeatures_.data() + textFeatureOffsets_[index] * textFeatureDim_;
	}

	[[nodiscard]] oa::Span<const oa::F32> mean() const { return oa::Span<const oa::F32>(mean_.data(), mean_.size()); }
	[[nodiscard]] oa::Span<const oa::F32> stdDev() const { return oa::Span<const oa::F32>(std_.data(), std_.size()); }

	// De-standardize inOutFeat [inFrames, featDim] in place: x*std + mean.
	void denormalize(oa::F32* inOutFeat, oa::I64 inFrames) const;

protected:
	// Subclasses with a fixed corpus contract call this with their featDim.
	static oa::I32 jointsForFeatDim(oa::I32 inFeatDim);

	bool load(const oa::String& inDataDir, const oa::String& inSplit, oa::I32 inMaxClips);

	oa::I32 featDim_   = 263;
	oa::I32 numJoints_ = 22;
	bool  ok_        = false;

	oa::Vec<oa::F32>    mean_, std_;       // [featDim]
	oa::Vec<oa::F32>    feat_;             // all clips concatenated, [totalFrames * featDim]
	oa::Vec<oa::I64>    offsets_;          // row offset per clip, size numClips+1
	oa::Vec<oa::String> ids_;
	oa::Vec<oa::String> texts_;
	oa::Vec<oa::Vec<HumanMl3dCaption>> captions_;
	oa::Vec<oa::F32> textFeatures_;        // concatenated [all caption rows, textFeatureDim]
	oa::Vec<oa::I64> textFeatureOffsets_;  // caption-row offset per clip, size numClips+1
	oa::I32 textFeatureDim_ = 0;
	oa::I32 textFeatureManifestDim_ = 0;
	oa::String textFeatureFormat_;
	oa::String textFeatureModel_;
	oa::I64 numClips_    = 0;
	oa::I64 totalFrames_ = 0;
};

// ── Derived: AnimationGPT combatMotionProcessed (CMP) ────────────────────────
class DsCombatMotionProcessed : public DsHumanMl3d {
public:
	DsCombatMotionProcessed(const oa::String& inDataDir, const oa::String& inSplit = "train",
		oa::I32 inMaxClips = 0)
		: DsHumanMl3d(inDataDir, inSplit, inMaxClips, 263) {}

	[[nodiscard]] const char* name() const override { return "CombatMotionProcessed"; }
};

// ── Derived: KIT-ML (21-joint, 251-dim) ─────────────────────────────────────
class DsKitMl : public DsHumanMl3d {
public:
	DsKitMl(const oa::String& inDataDir, const oa::String& inSplit = "train",
		oa::I32 inMaxClips = 0)
		: DsHumanMl3d(inDataDir, inSplit, inMaxClips, 251) {}

	[[nodiscard]] const char* name() const override { return "KIT-ML"; }
};

} // namespace oa
