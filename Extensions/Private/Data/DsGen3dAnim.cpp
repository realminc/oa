// OaDsGen3dAnim — 3D animation pose dataset implementation.

#include <Data/DsGen3dAnim.h>

#include <Oa/Core/Log.h>
#include <Oa/Core/FnMatrix.h>

#include <Anim/Usd.h>
#include <Anim/PosePack.h>
#include <Rig/Skeleton.h>

#include <cctype>
#include <cmath>
#include <cstring>
#include <utility>

namespace {

constexpr OaF32 kStdFloor = 1e-4f;

const char* const kCategoryNames[] = {
	"idle", "walk", "backpack", "briefcase", "cup", "phone", "purse", "react", "other"
};

OaUsize SplitIndex(OaDsSplit InSplit) {
	const OaUsize idx = static_cast<OaUsize>(InSplit);
	return idx < 3 ? idx : 0;
}

} // namespace

OaU8 OaDsCategoryOf(const OaString& InContent) {
	OaString c = InContent;
	for (OaUsize i = 0; i < c.Size(); ++i) {
		c[i] = static_cast<char>(std::tolower(static_cast<unsigned char>(c[i])));
	}
	auto has = [&](const char* k) { return c.find(k) != OaString::npos; };
	if (has("walk"))      { return 1; }
	if (has("backpack"))  { return 2; }
	if (has("briefcase")) { return 3; }
	if (has("cup"))       { return 4; }
	if (has("phone") || has("cellphone")) { return 5; }
	if (has("purse"))     { return 6; }
	if (has("react"))     { return 7; }
	if (has("idle"))      { return 0; }
	return 8;
}

const char* OaDsCategoryName(OaU8 InCategory) {
	const OaUsize n = sizeof(kCategoryNames) / sizeof(kCategoryNames[0]);
	return InCategory < n ? kCategoryNames[InCategory] : "?";
}

// ─────────────────────────────────────────────────────────────────────────────

OaDsGen3dAnim::OaDsGen3dAnim(const OaPath& InPath) : Path_(InPath) {
	(void)LoadInternal_();
}

OaStatus OaDsGen3dAnim::Load(const OaPath& InPath) {
	Path_ = InPath;
	Clips_.Clear();
	Metas_.Clear();
	Mean_.Clear();
	Std_.Clear();
	Windows_.Clear();
	Indices_.Clear();
	for (auto& splitWindows : SplitWindows_) { splitWindows.Clear(); }
	for (auto& cursor : SplitCursor_) { cursor = 0; }
	ContextLen_ = 0;
	if (!LoadInternal_()) {
		return OaStatus::Error("OaDsGen3dAnim::Load failed");
	}
	if (RootTranslationDelta_) {
		RecomputeStats_();
	}

	// Conditioning diagnostic: channels whose std falls below kStdFloor are
	// near-constant in this dataset and are neutralized (scale=1 → normalized ≈0)
	// rather than amplified. A high count means most of the pose is static in this
	// clip (expected for a single locomotion clip — fixed fingers, locked joints).
	if (!Std_.Empty()) {
		// Dead channels were set to the neutralizing scale of exactly 1.0 above; all
		// other entries are the genuine per-channel std (active signal).
		OaUsize dead = 0;
		OaF32 mn = 0.0f, mx = 0.0f;
		bool haveActive = false;
		for (OaUsize i = 0; i < Std_.Size(); ++i) {
			const OaF32 s = Std_[i];
			if (s == 1.0f) { ++dead; continue; }
			if (!haveActive) { mn = mx = s; haveActive = true; }
			if (s < mn) { mn = s; }
			if (s > mx) { mx = s; }
		}
		OA_LOG_INFO(OaLogComponent::ML,
			"dsgen3danim: active channel std — min=%.3g max=%.3g | %zu/%zu "
			"neutralized (dead, scale=1)",
			static_cast<double>(mn), static_cast<double>(mx),
			dead, Std_.Size());
	}
	return OaStatus::Ok();
}

bool OaDsGen3dAnim::LoadInternal_() {
	// Datasets are combined ".usd" stages (multiple SkelAnimation prims under one
	// Skeleton — see LoadUsd_). The legacy binary ".oads" format has been removed;
	// ReadUsdaMulti rejects anything that isn't a UsdSkel stage.
	return LoadUsd_();
}

bool OaDsGen3dAnim::LoadUsd_() {
	auto clipsR = OaUsd::ReadUsdaMulti(Path_);
	if (clipsR.IsError()) {
		OA_LOG_ERROR(OaLogComponent::ML, "dsgen3danim: usd load failed: %s",
			clipsR.GetStatus().ToString().c_str());
		return false;
	}
	const OaVec<OaUsdNamedClip>& named = clipsR.GetValue();
	if (named.Empty()) { return false; }

	const OaSkeleton& sk = OaSkMetaHuman();
	PoseDim_ = sk.PoseDim();
	if (PoseDim_ <= 0) { return false; }
	Fps_ = named[0].Clip.Fps;

	auto splitOf = [](const OaString& s) -> OaDsSplit {
		const std::string v = s.StdStr();
		if (v == "val")  { return OaDsSplit::Val; }
		if (v == "test") { return OaDsSplit::Test; }
		return OaDsSplit::Train;
	};
	// Strip a leading "[MF]T[NOU]_" body-variant prefix when present (legacy MTN
	// naming); otherwise classify on the full name.
	auto contentOf = [](const OaString& name) -> OaString {
		const std::string s = name.StdStr();
		if (s.size() > 4 && (s[0] == 'M' || s[0] == 'F') && s[1] == 'T' && s[3] == '_') {
			return OaString(s.substr(4).c_str());
		}
		return name;
	};

	Clips_.Clear();
	Metas_.Clear();
	Clips_.Reserve(named.Size());
	Metas_.Reserve(named.Size());

	for (OaUsize i = 0; i < named.Size(); ++i) {
		const OaUsdNamedClip& nc = named[i];
		auto packed = OaPosePack::Pack(nc.Clip, sk);
		if (packed.IsError()) {
			OA_LOG_WARN(OaLogComponent::ML, "dsgen3danim: skip clip '%s' (pack: %s)",
				nc.Name.c_str(), packed.GetStatus().ToString().c_str());
			continue;
		}
		OaPoseClip clip = packed.GetValue();
		if (static_cast<OaI32>(clip.PoseDim) != PoseDim_) {
			OA_LOG_WARN(OaLogComponent::ML, "dsgen3danim: skip clip '%s' (posedim %u != %d)",
				nc.Name.c_str(), clip.PoseDim, PoseDim_);
			continue;
		}

		const std::string stem = nc.Name.StdStr();
		OaDsClipMeta m;
		m.Name      = nc.Name;
		m.Content   = contentOf(nc.Name);
		m.Character = (!stem.empty() && stem[0] == 'M') ? 0
		            : (!stem.empty() && stem[0] == 'F') ? 1 : 255;
		m.BodyType  = (stem.size() > 2 && stem[2] == 'N') ? 0
		            : (stem.size() > 2 && stem[2] == 'O') ? 1
		            : (stem.size() > 2 && stem[2] == 'U') ? 2 : 255;
		m.Category  = OaDsCategoryOf(m.Content);
		m.Split     = splitOf(nc.Split);
		m.Frames    = clip.FrameCount;

		Clips_.PushBack(std::move(clip));
		Metas_.PushBack(std::move(m));
	}

	if (Clips_.Empty()) {
		OA_LOG_ERROR(OaLogComponent::ML, "dsgen3danim: usd dataset has no packable clips: %s",
			Path_.String().c_str());
		return false;
	}

	// Derive normalization at load time (train-split per-channel mean/std + the
	// dead-channel rule). RootTranslationDelta_ is off here, so ModelFeature_
	// returns the raw absolute feature — identical to the stats the binary .oads
	// baked at pack time.
	RecomputeStats_();

	OA_LOG_INFO(OaLogComponent::ML,
		"dsgen3danim: loaded %s (usd) — %zu clips · posedim=%d · fps=%.3g",
		Path_.String().c_str(), Clips_.Size(), PoseDim_, static_cast<double>(Fps_));
	return true;
}

void OaDsGen3dAnim::SetRootTranslationDelta(bool InEnabled) {
	if (RootTranslationDelta_ == InEnabled) { return; }
	RootTranslationDelta_ = InEnabled;
	if (Ok()) {
		RecomputeStats_();
	}
}

OaF32 OaDsGen3dAnim::ModelFeature_(const OaPoseClip& InClip, OaI32 InFrame, OaI32 InChannel) const {
	const OaUsize D = static_cast<OaUsize>(PoseDim_);
	const OaUsize idx = static_cast<OaUsize>(InFrame) * D + static_cast<OaUsize>(InChannel);
	const OaF32 v = InClip.Samples[idx];
	if (!RootTranslationDelta_ || InChannel < 0 || InChannel >= 3) {
		return v;
	}
	if (InFrame <= 0) {
		return 0.0f;
	}
	const OaUsize prev = static_cast<OaUsize>(InFrame - 1) * D + static_cast<OaUsize>(InChannel);
	return v - InClip.Samples[prev];
}

void OaDsGen3dAnim::RecomputeStats_() {
	const OaUsize D = static_cast<OaUsize>(PoseDim_);
	Mean_.Resize(D);
	Std_.Resize(D);
	if (D == 0 || Clips_.Empty()) { return; }

	OaVec<double> sum, sumsq;
	sum.Resize(D);
	sumsq.Resize(D);
	for (OaUsize i = 0; i < D; ++i) {
		sum[i] = 0.0;
		sumsq[i] = 0.0;
	}

	OaU64 trainFrames = 0;
	for (OaUsize ci = 0; ci < Clips_.Size(); ++ci) {
		if (Metas_[ci].Split != OaDsSplit::Train) { continue; }
		const OaPoseClip& clip = Clips_[ci];
		for (OaU32 f = 0; f < clip.FrameCount; ++f) {
			for (OaI32 c = 0; c < PoseDim_; ++c) {
				const OaUsize ch = static_cast<OaUsize>(c);
				const double v = static_cast<double>(ModelFeature_(clip, static_cast<OaI32>(f), c));
				sum[ch] += v;
				sumsq[ch] += v * v;
			}
		}
		trainFrames += clip.FrameCount;
	}

	const double inv = trainFrames ? 1.0 / static_cast<double>(trainFrames) : 0.0;
	for (OaUsize c = 0; c < D; ++c) {
		const double m = sum[c] * inv;
		const double var = sumsq[c] * inv - m * m;
		Mean_[c] = static_cast<OaF32>(m);
		Std_[c]  = static_cast<OaF32>(std::sqrt(var > 0.0 ? var : 0.0));
		// Dead-channel rule: a channel whose std is below kStdFloor is effectively
		// constant in this set and carries no learnable signal. Dividing by a tiny
		// floor would amplify pure numerical jitter by ~1/floor (×10^4) — blowing up
		// model inputs AND making its target an irreducible noise term that floors
		// the loss. Use scale=1 (pass-through): since x≈mean, the normalized value
		// is ≈0 — neutral, not amplified. (Same rule as sklearn StandardScaler.)
		if (Std_[c] < kStdFloor) { Std_[c] = 1.0f; }
	}

	OA_LOG_INFO(OaLogComponent::ML, "dsgen3danim: model root translation = %s",
		RootTranslationDelta_ ? "delta" : "absolute");
}

void OaDsGen3dAnim::BuildIndices(OaI32 InContextLen, OaI64 InSeed) {
	ContextLen_ = InContextLen;
	
	// Seed all per-split RNGs with derived seeds for reproducibility
	TrainRng_.seed(static_cast<OaU64>(InSeed));
	ValRng_.seed(static_cast<OaU64>(InSeed + 1000));
	TestRng_.seed(static_cast<OaU64>(InSeed + 2000));
	
	Windows_.Clear();
	for (auto& splitWindows : SplitWindows_) { splitWindows.Clear(); }
	for (auto& cursor : SplitCursor_) { cursor = 0; }
	for (OaUsize ci = 0; ci < Clips_.Size(); ++ci) {
		const OaI32 frames = static_cast<OaI32>(Clips_[ci].FrameCount);
		if (frames < InContextLen + 1) { continue; }
		for (OaI32 start = 0; start <= frames - InContextLen - 1; ++start) {
			const OaUsize windowIdx = Windows_.Size();
			Windows_.PushBack(Window{ static_cast<OaI32>(ci), start });
			SplitWindows_[SplitIndex(Metas_[ci].Split)].PushBack(windowIdx);
		}
	}
	// Build shuffled indices
	Indices_.Resize(Windows_.Size());
	for (OaUsize i = 0; i < Indices_.Size(); ++i) { Indices_[i] = static_cast<OaI64>(i); }
	// Use TrainRng for initial shuffle (affects all splits equally for now)
	std::shuffle(Indices_.begin(), Indices_.end(), TrainRng_);
	std::shuffle(SplitWindows_[0].begin(), SplitWindows_[0].end(), TrainRng_);
	std::shuffle(SplitWindows_[1].begin(), SplitWindows_[1].end(), ValRng_);
	std::shuffle(SplitWindows_[2].begin(), SplitWindows_[2].end(), TestRng_);
}

OaI64 OaDsGen3dAnim::Size() const {
	return static_cast<OaI64>(Windows_.Size());
}

OaMatrix OaDsGen3dAnim::GetItem(OaI64 InIndex) const {
	if (InIndex < 0 || static_cast<OaUsize>(InIndex) >= Windows_.Size()) { return OaMatrix(); }
	const Window& w = Windows_[static_cast<OaUsize>(InIndex)];
	const OaPoseClip& clip = Clips_[static_cast<OaUsize>(w.Clip)];
	// Return flat [Ctx * PoseDim] for now; reshape caller-side if needed
	const OaUsize n = static_cast<OaUsize>(ContextLen_) * PoseDim_;
	OaVec<float> xdata(n);
	for (OaI32 t = 0; t < ContextLen_; ++t) {
		for (OaI32 c = 0; c < PoseDim_; ++c) {
			const OaUsize ch = static_cast<OaUsize>(c);
			const OaUsize i = static_cast<OaUsize>(t) * static_cast<OaUsize>(PoseDim_) + ch;
			xdata[i] = (ModelFeature_(clip, w.Start + t, c) - Mean_[ch]) / Std_[ch];
		}
	}
	return OaFnMatrix::FromBytes(
		OaSpan<const OaU8>(reinterpret_cast<const OaU8*>(xdata.Data()), n * sizeof(float)),
		OaMatrixShape{ContextLen_, PoseDim_}, OaScalarType::Float32);
}

OaDataset::Sample OaDsGen3dAnim::GetSample(OaI64 InIndex) const {
	// X = context window, Y = next frame (single target)
	if (InIndex < 0 || static_cast<OaUsize>(InIndex) >= Windows_.Size()) { return Sample(); }
	const Window& w = Windows_[static_cast<OaUsize>(InIndex)];
	const OaPoseClip& clip = Clips_[static_cast<OaUsize>(w.Clip)];

	const OaUsize n = static_cast<OaUsize>(ContextLen_) * PoseDim_;
	OaVec<float> xdata(n);
	for (OaI32 t = 0; t < ContextLen_; ++t) {
		for (OaI32 c = 0; c < PoseDim_; ++c) {
			const OaUsize ch = static_cast<OaUsize>(c);
			const OaUsize i = static_cast<OaUsize>(t) * static_cast<OaUsize>(PoseDim_) + ch;
			xdata[i] = (ModelFeature_(clip, w.Start + t, c) - Mean_[ch]) / Std_[ch];
		}
	}
	OaMatrix x = OaFnMatrix::FromBytes(
		OaSpan<const OaU8>(reinterpret_cast<const OaU8*>(xdata.Data()), n * sizeof(float)),
		OaMatrixShape{ContextLen_, PoseDim_}, OaScalarType::Float32);

	// Y: next frame (single pose vector)
	OaVec<float> ydata(PoseDim_);
	for (OaI32 c = 0; c < PoseDim_; ++c) {
		const OaUsize ch = static_cast<OaUsize>(c);
		ydata[ch] = (ModelFeature_(clip, w.Start + ContextLen_, c) - Mean_[ch]) / Std_[ch];
	}
	OaMatrix y = OaFnMatrix::FromBytes(
		OaSpan<const OaU8>(reinterpret_cast<const OaU8*>(ydata.Data()), PoseDim_ * sizeof(float)),
		OaMatrixShape{PoseDim_}, OaScalarType::Float32);

	return Sample(x, y);
}

OaUsize OaDsGen3dAnim::WindowCount(OaDsSplit InSplit) const noexcept {
	return SplitWindows_[SplitIndex(InSplit)].Size();
}

bool OaDsGen3dAnim::RestrictSplitToClip(OaDsSplit InSplit, const OaString& InClipName) {
	const OaI32 clipIdx = FindClipByName(InClipName);
	if (clipIdx < 0) { return false; }

	const OaUsize splitIdx = SplitIndex(InSplit);
	OaVec<OaUsize> filtered;
	for (OaUsize windowIdx : SplitWindows_[splitIdx]) {
		if (Windows_[windowIdx].Clip == clipIdx) {
			filtered.PushBack(windowIdx);
		}
	}
	if (filtered.Empty()) { return false; }

	SplitWindows_[splitIdx] = std::move(filtered);
	SplitCursor_[splitIdx] = 0;
	return true;
}

OaUsize OaDsGen3dAnim::SplitClipCount(OaDsSplit InSplit) const noexcept {
	OaUsize n = 0;
	for (const auto& m : Metas_) {
		if (m.Split == InSplit) { ++n; }
	}
	return n;
}

void OaDsGen3dAnim::NextBatch(OaDsSplit InSplit, OaI32 InBatch, OaMatrix& OutX, OaMatrix& OutY) {
	const OaI64 rowLen = static_cast<OaI64>(ContextLen_) * PoseDim_;
	const OaI64 total  = static_cast<OaI64>(InBatch) * rowLen;
	OaVec<float> xdata(static_cast<OaUsize>(total));
	OaVec<float> ydata(static_cast<OaUsize>(total));

	const OaUsize splitIdx = SplitIndex(InSplit);
	OaVec<OaUsize>& splitWindows = SplitWindows_[splitIdx];
	if (splitWindows.Empty()) { OutX = OaMatrix(); OutY = OaMatrix(); return; }

	// Iterate shuffled windows without replacement inside an epoch. This matches
	// the MNIST/DataLoader pattern and makes tiny overfit sets deterministic.
	auto& rng = (InSplit == OaDsSplit::Train) ? TrainRng_ :
	            (InSplit == OaDsSplit::Val)   ? ValRng_   : TestRng_;
	for (OaI32 b = 0; b < InBatch; ++b) {
		if (SplitCursor_[splitIdx] >= splitWindows.Size()) {
			SplitCursor_[splitIdx] = 0;
			std::shuffle(splitWindows.begin(), splitWindows.end(), rng);
		}
		const Window& wnd = Windows_[splitWindows[SplitCursor_[splitIdx]++]];
		const OaPoseClip& clip = Clips_[static_cast<OaUsize>(wnd.Clip)];
		float* px = &xdata[static_cast<OaUsize>(static_cast<OaI64>(b) * rowLen)];
		float* py = &ydata[static_cast<OaUsize>(static_cast<OaI64>(b) * rowLen)];
		for (OaI64 t = 0; t < ContextLen_; ++t) {
			float* dx = px + t * PoseDim_;
			float* dy = py + t * PoseDim_;
			for (OaI32 c = 0; c < PoseDim_; ++c) {
				const float m = Mean_[static_cast<OaUsize>(c)];
				const float s = Std_[static_cast<OaUsize>(c)];
				dx[c] = (ModelFeature_(clip, static_cast<OaI32>(wnd.Start + t), c) - m) / s;
				dy[c] = (ModelFeature_(clip, static_cast<OaI32>(wnd.Start + t + 1), c) - m) / s;
			}
		}
	}

	OutX = OaFnMatrix::Empty(OaMatrixShape{InBatch, ContextLen_, PoseDim_}, OaScalarType::Float32);
	OutY = OaFnMatrix::Empty(OaMatrixShape{InBatch, ContextLen_, PoseDim_}, OaScalarType::Float32);
	std::memcpy(OutX.DataAs<float>(), xdata.Data(), static_cast<size_t>(total) * sizeof(float));
	std::memcpy(OutY.DataAs<float>(), ydata.Data(), static_cast<size_t>(total) * sizeof(float));
}

OaI32 OaDsGen3dAnim::FindClipByName(const OaString& InName) const {
	for (OaUsize i = 0; i < Metas_.Size(); ++i) {
		if (Metas_[i].Name.StdStr() == InName.StdStr()) { return static_cast<OaI32>(i); }
	}
	return -1;
}

bool OaDsGen3dAnim::SeedRaw(OaI32 InClipIdx, OaI32 InContext, OaVec<OaF32>& OutRaw) const {
	if (InClipIdx < 0 || static_cast<OaUsize>(InClipIdx) >= Clips_.Size()) { return false; }
	const OaPoseClip& clip = Clips_[static_cast<OaUsize>(InClipIdx)];
	if (static_cast<OaI32>(clip.FrameCount) < InContext || PoseDim_ <= 0) { return false; }
	const OaUsize n = static_cast<OaUsize>(InContext) * static_cast<OaUsize>(PoseDim_);
	OutRaw.Resize(n);
	std::memcpy(OutRaw.Data(), clip.Samples.Data(), n * sizeof(OaF32));
	return true;
}

bool OaDsGen3dAnim::ClipRaw(OaI32 InClipIdx, OaVec<OaF32>& OutRaw, OaU32& OutFrames) const {
	if (InClipIdx < 0 || static_cast<OaUsize>(InClipIdx) >= Clips_.Size()) { return false; }
	const OaPoseClip& clip = Clips_[static_cast<OaUsize>(InClipIdx)];
	OutFrames = clip.FrameCount;
	OutRaw.Resize(clip.Samples.Size());
	std::memcpy(OutRaw.Data(), clip.Samples.Data(), clip.Samples.Size() * sizeof(OaF32));
	return true;
}

bool OaDsGen3dAnim::SeedModelRaw(OaI32 InClipIdx, OaI32 InContext, OaVec<OaF32>& OutRaw) const {
	if (InClipIdx < 0 || static_cast<OaUsize>(InClipIdx) >= Clips_.Size()) { return false; }
	const OaPoseClip& clip = Clips_[static_cast<OaUsize>(InClipIdx)];
	if (static_cast<OaI32>(clip.FrameCount) < InContext || PoseDim_ <= 0) { return false; }
	const OaUsize n = static_cast<OaUsize>(InContext) * static_cast<OaUsize>(PoseDim_);
	OutRaw.Resize(n);
	for (OaI32 f = 0; f < InContext; ++f) {
		for (OaI32 c = 0; c < PoseDim_; ++c) {
			OutRaw[static_cast<OaUsize>(f) * static_cast<OaUsize>(PoseDim_) + static_cast<OaUsize>(c)] =
				ModelFeature_(clip, f, c);
		}
	}
	return true;
}

bool OaDsGen3dAnim::ClipModelRaw(OaI32 InClipIdx, OaVec<OaF32>& OutRaw, OaU32& OutFrames) const {
	if (InClipIdx < 0 || static_cast<OaUsize>(InClipIdx) >= Clips_.Size()) { return false; }
	const OaPoseClip& clip = Clips_[static_cast<OaUsize>(InClipIdx)];
	OutFrames = clip.FrameCount;
	OutRaw.Resize(clip.Samples.Size());
	for (OaU32 f = 0; f < clip.FrameCount; ++f) {
		for (OaI32 c = 0; c < PoseDim_; ++c) {
			OutRaw[static_cast<OaUsize>(f) * static_cast<OaUsize>(PoseDim_) + static_cast<OaUsize>(c)] =
				ModelFeature_(clip, static_cast<OaI32>(f), c);
		}
	}
	return true;
}
