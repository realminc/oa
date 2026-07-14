// OaDsHumanMl3d — HumanML3D-format motion dataset (OaDataset subclass)
//
// A small class hierarchy for the standard HumanML3D layout used by T2M-GPT /
// MoMask / MotionGPT:
//
//   OaDsHumanMl3d                 generic loader for ANY HumanML3D-format corpus
//        ↑                        (reads Mean/Std + new_joint_vecs + texts + split)
//   ├─ OaDsCombatMotionProcessed  AnimationGPT CMP set; 263-dim / 22-joint SMPL.
//   │  alias OaDsCmp
//   └─ OaDsKitMl                  KIT-ML corpus;        251-dim / 21-joint.
//
// On-disk layout (identical for every corpus):
//
//   <dir>/Mean.npy, Std.npy          per-feature normalization stats [FeatDim]
//   <dir>/new_joint_vecs/<id>.npy    motion features [Frames, FeatDim] (model input)
//   <dir>/texts/<id>.txt             captions ("text#pos-tags#start#end", all lines kept)
//   <dir>/text_feats/manifest.json   frozen text-encoder identity and feature contract
//   <dir>/text_feats/<id>.npy        optional aligned caption features [Captions, Dim]
//   <dir>/<split>.txt                one clip id per line (train|val|test|train_val)
//
// Features are loaded RAW and standardized in place: (x - Mean) / Std. The
// companion new_joints/<id>.npy (raw [Frames, J, 3] positions) is NOT loaded —
// generation recovers positions from the feature vector instead.
//
// Usage:
//   OaDsCombatMotionProcessed ds("/data/Cmp", "train");        // CMP  263/22
//   OaDsHumanMl3d             ds("/data/HumanML3D", "train");  // H3D  263/22
//   OaDsKitMl                 ds("/data/KIT-ML", "train");     // KIT  251/21
//   if (!ds.Ok()) { ... }
//   const float* clip = ds.ClipData(i);   // [ds.ClipFrames(i), ds.FeatDim()], normalized

#pragma once

#include <Oa/Data/Dataset.h>
#include <Oa/Core/Types.h>
#include <Oa/Core/Matrix.h>

// ── Base: generic HumanML3D-format motion loader ─────────────────────────────
struct OaHumanMl3dCaption {
	OaString Text;
	OaF32 StartSec = 0.0F;
	OaF32 EndSec = 0.0F;
	bool HasRange = false;
};

// Reference inverse for the root-motion + root-relative-position channels in a
// HumanML3D feature stream. InFeatures must be de-standardized first. The
// returned row-major array is [InFrames, J, 3] in the dataset's canonical
// metric coordinate system. This is the authoritative ALM preview/evaluation
// path; DCC-rig rotation solving and retargeting are separate downstream work.
[[nodiscard]] OaVec<OaF32> OaHumanMl3dRecoverWorldJoints(
	OaSpan<const OaF32> InFeatures,
	OaI32 InFrames,
	OaI32 InFeatDim = 263);

// Mean Euclidean joint-position error between two recovered world-joint arrays.
// Inputs are [frames, joints, 3] in dataset units. The return value is in cm.
[[nodiscard]] OaF64 OaHumanMl3dMpjpeCm(
	OaSpan<const OaF32> InPredWorld,
	OaSpan<const OaF32> InTargetWorld);

class OaDsHumanMl3d : public OaDataset {
public:
	// InDataDir: dataset root (see layout above). InSplit: split file stem.
	// InMaxClips: 0 = all. InFeatDim: 263 (HumanML3D / SMPL-22) / 251 (KIT-21).
	OaDsHumanMl3d(const OaString& InDataDir, const OaString& InSplit = "train",
		OaI32 InMaxClips = 0, OaI32 InFeatDim = 263);

	virtual ~OaDsHumanMl3d() = default;

	[[nodiscard]] bool Ok() const { return Ok_; }

	// Corpus identity (overridden by named subclasses for logging/eval).
	[[nodiscard]] virtual const char* Name() const { return "HumanML3D"; }

	// OaDataset — one standardized clip [Frames, FeatDim] per index.
	[[nodiscard]] OaI64 Size() const override { return NumClips_; }
	[[nodiscard]] OaMatrix GetItem(OaI64 InIndex) const override;

	// ── Host accessors (for synchronous window batching) ─────────────────────
	[[nodiscard]] OaI32 FeatDim() const { return FeatDim_; }
	[[nodiscard]] OaI32 NumJoints() const { return NumJoints_; }
	[[nodiscard]] OaI32 NumClips() const { return static_cast<OaI32>(NumClips_); }
	[[nodiscard]] OaI64 TotalFrames() const { return TotalFrames_; }
	[[nodiscard]] OaI32 ClipFrames(OaI64 InIndex) const;
	// Row-major [ClipFrames(i), FeatDim] standardized features for clip i.
	[[nodiscard]] const OaF32* ClipData(OaI64 InIndex) const;
	[[nodiscard]] const OaString& ClipId(OaI64 InIndex) const { return Ids_[InIndex]; }
	[[nodiscard]] const OaString& ClipText(OaI64 InIndex) const { return Texts_[InIndex]; }
	// Every caption record, including HumanML3D partial-clip time ranges. ClipText
	// remains a compatibility alias for the first caption.
	[[nodiscard]] const OaVec<OaHumanMl3dCaption>& ClipCaptions(OaI64 InIndex) const {
		return Captions_[InIndex];
	}
	// Optional frozen text-encoder rows from text_feats/<id>.npy. Rows are aligned
	// with ClipCaptions() and loaded as host FP32 for synchronous batch assembly.
	[[nodiscard]] OaI32 TextFeatureDim() const { return TextFeatureDim_; }
	[[nodiscard]] bool HasTextFeatures() const { return TextFeatureDim_ > 0; }
	[[nodiscard]] const OaString& TextFeatureFormat() const { return TextFeatureFormat_; }
	[[nodiscard]] const OaString& TextFeatureModel() const { return TextFeatureModel_; }
	[[nodiscard]] OaI32 ClipTextFeatureCount(OaI64 InIndex) const {
		return static_cast<OaI32>(TextFeatureOffsets_[InIndex + 1] - TextFeatureOffsets_[InIndex]);
	}
	[[nodiscard]] const OaF32* ClipTextFeatureData(OaI64 InIndex) const {
		return TextFeatures_.Data() + TextFeatureOffsets_[InIndex] * TextFeatureDim_;
	}

	[[nodiscard]] OaSpan<const OaF32> Mean() const { return OaSpan<const OaF32>(Mean_.Data(), Mean_.Size()); }
	[[nodiscard]] OaSpan<const OaF32> Std() const { return OaSpan<const OaF32>(Std_.Data(), Std_.Size()); }

	// De-standardize InOutFeat [InFrames, FeatDim] in place: x*Std + Mean.
	void Denormalize(OaF32* InOutFeat, OaI64 InFrames) const;

protected:
	// Subclasses with a fixed corpus contract call this with their FeatDim.
	static OaI32 JointsForFeatDim(OaI32 InFeatDim);

	bool Load(const OaString& InDataDir, const OaString& InSplit, OaI32 InMaxClips);

	OaI32 FeatDim_   = 263;
	OaI32 NumJoints_ = 22;
	bool  Ok_        = false;

	OaVec<OaF32>    Mean_, Std_;       // [FeatDim]
	OaVec<OaF32>    Feat_;             // all clips concatenated, [TotalFrames * FeatDim]
	OaVec<OaI64>    Offsets_;          // row offset per clip, size NumClips+1
	OaVec<OaString> Ids_;
	OaVec<OaString> Texts_;
	OaVec<OaVec<OaHumanMl3dCaption>> Captions_;
	OaVec<OaF32> TextFeatures_;        // concatenated [all caption rows, TextFeatureDim]
	OaVec<OaI64> TextFeatureOffsets_;  // caption-row offset per clip, size NumClips+1
	OaI32 TextFeatureDim_ = 0;
	OaI32 TextFeatureManifestDim_ = 0;
	OaString TextFeatureFormat_;
	OaString TextFeatureModel_;
	OaI64 NumClips_    = 0;
	OaI64 TotalFrames_ = 0;
};

// ── Derived: AnimationGPT CombatMotionProcessed (CMP) ────────────────────────
//   263-dim HumanML3D features, 22-joint SMPL skeleton. A named dataset so call
//   sites don't pass magic dims; the corpus contract (and any future CMP-only
//   post-processing / prompt vocab) lives here, not at every use site.
class OaDsCombatMotionProcessed : public OaDsHumanMl3d {
public:
	// InDataDir: CMP root (layout above). InSplit: split stem. InMaxClips: 0 = all.
	OaDsCombatMotionProcessed(const OaString& InDataDir, const OaString& InSplit = "train",
		OaI32 InMaxClips = 0)
		: OaDsHumanMl3d(InDataDir, InSplit, InMaxClips, 263) {}

	[[nodiscard]] const char* Name() const override { return "CombatMotionProcessed"; }
};

// Short alias.
using OaDsCmp = OaDsCombatMotionProcessed;

// ── Derived: KIT-ML (21-joint, 251-dim) ─────────────────────────────────────
//   The KIT-ML corpus in HumanML3D 251-dim format (21-joint skeleton). Same
//   on-disk layout as the base; only the feature contract differs. NOTE: KIT
//   ships as raw .c3d/.xml — it must be run through the HumanML3D preprocessing
//   pipeline (which emits Mean.npy / new_joint_vecs / texts / split files) before
//   this loader can read it, exactly like the pytorch reference.
class OaDsKitMl : public OaDsHumanMl3d {
public:
	// InDataDir: KIT-ML root AFTER HumanML3D preprocessing (layout above).
	// InSplit: split stem. InMaxClips: 0 = all.
	OaDsKitMl(const OaString& InDataDir, const OaString& InSplit = "train",
		OaI32 InMaxClips = 0)
		: OaDsHumanMl3d(InDataDir, InSplit, InMaxClips, 251) {}

	[[nodiscard]] const char* Name() const override { return "KIT-ML"; }
};
