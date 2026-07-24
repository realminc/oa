#pragma once

// OaDsGen3dAnim — 3D animation pose dataset (OaDataset subclass).
//
// Loads a combined ".usd" dataset: one SkelRoot/Skeleton with N SkelAnimation
// prims (built by `dsgen3danimctl pack`). Each clip is PosePacked to the model's canonical
// channels and the per-channel train-split normalization stats (mean/std) are
// derived at load time. Per-clip train/val/test split comes from the prim's
// customData.oa.split tag.
//
// As an OaDataset subclass, it works with OaDataLoader for shuffling,
// batching, and parallel loading. The split filtering is handled via
// OaDataLoader indices or a split-specific view wrapper.

#include <Oa/Data/Dataset.h>
#include <Oa/Core/Filesystem.h>
#include <Oa/Core/Types.h>
#include <Anim/PoseClip.h>

#include <random>

enum class OaDsSplit : OaU8 { Train = 0, Val = 1, Test = 2 };

struct OaDsClipMeta {
	OaString  Name;                 // file stem, e.g. "MTN_N_Idle_E"
	OaString  Content;              // animation content, e.g. "N_Idle_E"
	OaU8      Character = 255;      // 0=Manny 1=Quinn, 255=unknown
	OaU8      BodyType  = 255;      // 0=N(normal) 1=O(over) 2=U(under), 255=unknown
	OaU8      Category  = 255;      // coarse motion class, 255=unknown
	OaDsSplit Split     = OaDsSplit::Train;
	OaU32     Frames    = 0;
};

class OaDsGen3dAnim : public OaDataset {
public:
	OaDsGen3dAnim() = default;

	/// Load a .oads archive. Call BuildIndices() before using with OaDataLoader.
	explicit OaDsGen3dAnim(const OaPath& InPath);

	/// Backward-compat: load/reload from path.
	[[nodiscard]] OaStatus Load(const OaPath& InPath);

	// OaDataset interface
	[[nodiscard]] OaI64 Size() const override;
	[[nodiscard]] OaMatrix GetItem(OaI64 InIndex) const override;
	[[nodiscard]] Sample GetSample(OaI64 InIndex) const override;

	/// Build sliding-window indices for the given context length.
	/// Call after Load() and before using with OaDataLoader or NextBatch().
	void BuildIndices(OaI32 InContextLen, OaI64 InSeed = 1234);

	/// View root translation xyz as per-frame displacement instead of absolute
	/// clip-space position. The archive stays unchanged; batches/seeds/stats are
	/// transformed on load so training and generation use the same model space.
	void SetRootTranslationDelta(bool InEnabled);
	[[nodiscard]] bool RootTranslationDelta() const noexcept { return RootTranslationDelta_; }

	/// Backward-compat alias for BuildIndices().
	void PrepareWindows(OaI32 InContextLen, OaI64 InSeed = 1234) { BuildIndices(InContextLen, InSeed); }

	/// Legacy/Convenience: number of windows for a specific split.
	[[nodiscard]] OaUsize WindowCount(OaDsSplit InSplit) const noexcept;

	/// Restrict an already-built split window list to a single clip name.
	/// Normalization stats are not recomputed; this is intended for real-clip
	/// overfit/debug runs that still want full train-set channel scaling.
	[[nodiscard]] bool RestrictSplitToClip(OaDsSplit InSplit, const OaString& InClipName);

	/// Legacy/Convenience: get next batch for a specific split.
	/// Returns standardized X [B, Ctx, PoseDim] and Y [B, Ctx, PoseDim] (shifted +1).
	void NextBatch(OaDsSplit InSplit, OaI32 InBatch, OaMatrix& OutX, OaMatrix& OutY);

	/// Metadata
	[[nodiscard]] bool    Ok()      const noexcept { return !Clips_.Empty() && PoseDim_ > 0; }
	[[nodiscard]] OaI32   PoseDim() const noexcept { return PoseDim_; }
	[[nodiscard]] OaF32   Fps()     const noexcept { return Fps_; }
	[[nodiscard]] OaUsize ClipCount() const noexcept { return Clips_.Size(); }
	[[nodiscard]] const OaVec<OaF32>& Mean() const noexcept { return Mean_; }
	[[nodiscard]] const OaVec<OaF32>& Std()  const noexcept { return Std_; }
	[[nodiscard]] const OaVec<OaDsClipMeta>& Metas() const noexcept { return Metas_; }

	/// Count clips in a specific split.
	[[nodiscard]] OaUsize SplitClipCount(OaDsSplit InSplit) const noexcept;

	/// Find clip by name (file stem), or -1 if not found.
	[[nodiscard]] OaI32 FindClipByName(const OaString& InName) const;

	/// Copy raw (un-normalized) frames from a clip. For seeding generation.
	[[nodiscard]] bool SeedRaw(OaI32 InClipIdx, OaI32 InContext, OaVec<OaF32>& OutRaw) const;
	[[nodiscard]] bool ClipRaw(OaI32 InClipIdx, OaVec<OaF32>& OutRaw, OaU32& OutFrames) const;
	[[nodiscard]] bool SeedModelRaw(OaI32 InClipIdx, OaI32 InContext, OaVec<OaF32>& OutRaw) const;
	[[nodiscard]] bool ClipModelRaw(OaI32 InClipIdx, OaVec<OaF32>& OutRaw, OaU32& OutFrames) const;

private:
	struct Window { OaI32 Clip; OaI32 Start; };

	OaPath Path_;
	OaI32 PoseDim_    = 0;
	OaF32 Fps_        = 30.0f;
	OaI32 ContextLen_ = 0;

	OaVec<OaPoseClip>   Clips_;
	OaVec<OaDsClipMeta> Metas_;
	OaVec<OaF32>        Mean_;
	OaVec<OaF32>        Std_;

	// Window indices for OaDataLoader
	OaVec<Window> Windows_;
	OaVec<OaI64>  Indices_;  // Shuffled window indices
	OaVec<OaUsize> SplitWindows_[3];
	OaUsize SplitCursor_[3] = {0, 0, 0};
	
	// Per-split RNGs for deterministic, independent sampling
	std::mt19937_64 TrainRng_{1234};
	std::mt19937_64 ValRng_{5678};
	std::mt19937_64 TestRng_{9012};

	bool LoadInternal_();
	bool LoadUsd_();        // load a combined .usd (multi-SkelAnimation) dataset
	void RecomputeStats_();
	[[nodiscard]] OaF32 ModelFeature_(const OaPoseClip& InClip, OaI32 InFrame, OaI32 InChannel) const;

	bool RootTranslationDelta_ = false;
};

// Coarse motion category classification.
[[nodiscard]] OaU8        OaDsCategoryOf(const OaString& InContent);
[[nodiscard]] const char* OaDsCategoryName(OaU8 InCategory);
