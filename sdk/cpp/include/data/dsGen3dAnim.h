#pragma once

// DsGen3dAnim — 3D animation pose dataset (OaDataset subclass).
//
// Loads a combined ".usd" dataset: one SkelRoot/Skeleton with N SkelAnimation
// prims (built by `dsgen3danimctl pack`). Each clip is PosePacked to the model's canonical
// channels and the per-channel train-split normalization stats (mean/std) are
// derived at load time. Per-clip train/val/test split comes from the prim's
// customData.oa.split tag.
//
// as an OaDataset subclass, it works with OaDataLoader for shuffling,
// batching, and parallel loading. The split filtering is handled via
// OaDataLoader indices or a split-specific view wrapper.

#include <oa/data/dataset.h>
#include <oa/core/filesystem.h>
#include <oa/core/std/random.h>
#include <oa/core/types.h>
#include <anim/poseClip.h>

namespace oa {

enum class DsSplit : oa::U8 { Train = 0, Val = 1, Test = 2 };

struct DsClipMeta {
	oa::String  name;                 // file stem, e.g. "MTN_N_Idle_E"
	oa::String  content;              // animation content, e.g. "N_Idle_E"
	oa::U8      character = 255;      // 0=Manny 1=Quinn, 255=unknown
	oa::U8      bodyType  = 255;      // 0=N(normal) 1=O(over) 2=U(under), 255=unknown
	oa::U8      category  = 255;      // coarse motion class, 255=unknown
	DsSplit split     = DsSplit::Train;
	oa::U32     frames    = 0;
};

class DsGen3dAnim : public oa::Dataset {
public:
	DsGen3dAnim() = default;

	/// load a .oads archive. call buildIndices() before using with OaDataLoader.
	explicit DsGen3dAnim(const oa::Path& inPath);

	/// backward-compat: load/reload from path.
	[[nodiscard]] oa::Status load(const oa::Path& inPath);

	// OaDataset interface
	[[nodiscard]] oa::I64 size() const override;
	[[nodiscard]] oa::Matrix getItem(oa::I64 inIndex) const override;
	[[nodiscard]] Sample getSample(oa::I64 inIndex) const override;

	/// Build sliding-window indices for the given context length.
	/// call after load() and before using with OaDataLoader or nextBatch().
	void buildIndices(oa::I32 inContextLen, oa::I64 inSeed = 1234);

	/// view root translation xyz as per-frame displacement instead of absolute
	/// clip-space position. The archive stays unchanged; batches/seeds/stats are
	/// transformed on load so training and generation use the same model space.
	void setRootTranslationDelta(bool inEnabled);
	[[nodiscard]] bool rootTranslationDelta() const noexcept { return rootTranslationDelta_; }

	/// Legacy/Convenience: number of windows for a specific split.
	[[nodiscard]] oa::Usize windowCount(DsSplit inSplit) const noexcept;

	/// Restrict an already-built split window list to a single clip name.
	/// Normalization stats are not recomputed; this is intended for real-clip
	/// overfit/debug runs that still want full train-set channel scaling.
	[[nodiscard]] bool restrictSplitToClip(DsSplit inSplit, const oa::String& inClipName);

	/// Legacy/Convenience: get next batch for a specific split.
	/// Returns standardized X [B, ctx, poseDim] and Y [B, ctx, poseDim] (shifted +1).
	void nextBatch(DsSplit inSplit, oa::I32 inBatch, oa::Matrix& outX, oa::Matrix& outY);

	/// Metadata
	[[nodiscard]] bool    ok()      const noexcept { return !clips_.empty() && poseDim_ > 0; }
	[[nodiscard]] oa::I32   poseDim() const noexcept { return poseDim_; }
	[[nodiscard]] oa::F32   fps()     const noexcept { return fps_; }
	[[nodiscard]] oa::Usize clipCount() const noexcept { return clips_.size(); }
	[[nodiscard]] const oa::Vector<oa::F32>& mean() const noexcept { return mean_; }
	[[nodiscard]] const oa::Vector<oa::F32>& std()  const noexcept { return std_; }
	[[nodiscard]] const oa::Vector<DsClipMeta>& metas() const noexcept { return metas_; }

	/// Count clips in a specific split.
	[[nodiscard]] oa::Usize splitClipCount(DsSplit inSplit) const noexcept;

	/// find clip by name (file stem), or -1 if not found.
	[[nodiscard]] oa::I32 findClipByName(const oa::String& inName) const;

	/// Copy raw (un-normalized) frames from a clip. For seeding generation.
	[[nodiscard]] bool seedRaw(oa::I32 inClipIdx, oa::I32 inContext, oa::Vector<oa::F32>& outRaw) const;
	[[nodiscard]] bool clipRaw(oa::I32 inClipIdx, oa::Vector<oa::F32>& outRaw, oa::U32& outFrames) const;
	[[nodiscard]] bool seedModelRaw(oa::I32 inClipIdx, oa::I32 inContext, oa::Vector<oa::F32>& outRaw) const;
	[[nodiscard]] bool clipModelRaw(oa::I32 inClipIdx, oa::Vector<oa::F32>& outRaw, oa::U32& outFrames) const;

private:
	struct Window { oa::I32 clip; oa::I32 start; };

	oa::Path path_;
	oa::I32 poseDim_    = 0;
	oa::F32 fps_        = 30.0f;
	oa::I32 contextLen_ = 0;

	oa::Vector<PoseClip>   clips_;
	oa::Vector<DsClipMeta> metas_;
	oa::Vector<oa::F32>        mean_;
	oa::Vector<oa::F32>        std_;

	// Window indices for OaDataLoader
	oa::Vector<Window> windows_;
	oa::Vector<oa::I64>  indices_;  // Shuffled window indices
	oa::Vector<oa::Usize> splitWindows_[3];
	oa::Usize splitCursor_[3] = {0, 0, 0};
	
	// Per-split RNGs for deterministic, independent sampling
	oa::Random trainRng_{1234};
	oa::Random valRng_{5678};
	oa::Random testRng_{9012};

	bool loadInternal_();
	bool loadUsd_();        // load a combined .usd (multi-SkelAnimation) dataset
	void recomputeStats_();
	[[nodiscard]] oa::F32 modelFeature_(const PoseClip& inClip, oa::I32 inFrame, oa::I32 inChannel) const;

	bool rootTranslationDelta_ = false;
};

// Coarse motion category classification.
[[nodiscard]] oa::U8        dsCategoryOf(const oa::String& inContent);
[[nodiscard]] const char* dsCategoryName(oa::U8 inCategory);

}  // namespace oa
