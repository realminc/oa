// oa::DsGen3dAnim — 3D animation pose dataset implementation.

#include <data/dsGen3dAnim.h>

#include <oa/core/log.h>
#include <oa/core/fnMatrix.h>
#include <oa/core/std/memory.h>
#include <oa/core/std/scalarMath.h>
#include <oa/core/std/utility.h>

#include <anim/usd.h>
#include <anim/posePack.h>
#include <rig/skeleton.h>
#include <core/streamText.h>

namespace {

constexpr oa::F32 kStdFloor = 1e-4f;

char lowerAscii(char inValue) {
	return inValue >= 'A' && inValue <= 'Z'
		? static_cast<char>(inValue + ('a' - 'A'))
		: inValue;
}

const char* const kCategoryNames[] = {
	"idle", "walk", "backpack", "briefcase", "cup", "phone", "purse", "react", "other"
};

oa::Usize splitIndex(oa::DsSplit inSplit) {
	const oa::Usize idx = static_cast<oa::Usize>(inSplit);
	return idx < 3 ? idx : 0;
}

} // namespace

oa::U8 oa::dsCategoryOf(const oa::String& inContent) {
	oa::String c = inContent;
	for (oa::Usize i = 0; i < c.size(); ++i) {
		c[i] = lowerAscii(c[i]);
	}
	auto has = [&](const char* k) { return c.find(k) != oa::String::Npos; };
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

const char* oa::dsCategoryName(oa::U8 inCategory) {
	const oa::Usize n = sizeof(kCategoryNames) / sizeof(kCategoryNames[0]);
	return inCategory < n ? kCategoryNames[inCategory] : "?";
}

// ─────────────────────────────────────────────────────────────────────────────

oa::DsGen3dAnim::DsGen3dAnim(const oa::Path& inPath) : path_(inPath) {
	(void)loadInternal_();
}

oa::Status oa::DsGen3dAnim::load(const oa::Path& inPath) {
	path_ = inPath;
	clips_.clear();
	metas_.clear();
	mean_.clear();
	std_.clear();
	windows_.clear();
	indices_.clear();
	for (auto& splitWindows : splitWindows_) { splitWindows.clear(); }
	for (auto& cursor : splitCursor_) { cursor = 0; }
	contextLen_ = 0;
	if (!loadInternal_()) {
		return oa::Status::error("oa::DsGen3dAnim::load failed");
	}
	if (rootTranslationDelta_) {
		recomputeStats_();
	}

	// Conditioning diagnostic: channels whose std falls below kStdFloor are
	// near-constant in this dataset and are neutralized (scale=1 → normalized ≈0)
	// rather than amplified. A high count means most of the pose is static in this
	// clip (expected for a single locomotion clip — fixed fingers, locked joints).
	if (!std_.empty()) {
		// Dead channels were set to the neutralizing scale of exactly 1.0 above; all
		// other entries are the genuine per-channel std (active signal).
		oa::Usize dead = 0;
		oa::F32 mn = 0.0f, mx = 0.0f;
		bool haveActive = false;
		for (oa::Usize i = 0; i < std_.size(); ++i) {
			const oa::F32 s = std_[i];
			if (s == 1.0f) { ++dead; continue; }
			if (!haveActive) { mn = mx = s; haveActive = true; }
			if (s < mn) { mn = s; }
			if (s > mx) { mx = s; }
		}
		OaLogInfo(oa::LogComponent::Data,
			"dsgen3danim: active channel std — min=%.3g max=%.3g | %zu/%zu "
			"neutralized (dead, scale=1)",
			static_cast<double>(mn), static_cast<double>(mx),
			dead, std_.size());
	}
	return oa::Status::ok();
}

bool oa::DsGen3dAnim::loadInternal_() {
	// Datasets are combined ".usd" stages (multiple SkelAnimation prims under one
	// Skeleton — see loadUsd_). The legacy binary ".oads" format has been removed;
	// ReadUsdaMulti rejects anything that isn't a UsdSkel stage.
	return loadUsd_();
}

bool oa::DsGen3dAnim::loadUsd_() {
	auto clipsR = oa::Usd::readUsdaMulti(path_);
	if (clipsR.isError()) {
		OaLogError(oa::LogComponent::Data, "dsgen3danim: usd load failed: %s",
			clipsR.getStatus().toString().cStr());
		return false;
	}
	const oa::Vector<oa::UsdNamedClip>& named = clipsR.getValue();
	if (named.empty()) { return false; }

	const oa::Skeleton& sk = oa::skMetaHuman();
	poseDim_ = sk.poseDim();
	if (poseDim_ <= 0) { return false; }
	fps_ = named[0].clip.fps;

	auto splitOf = [](const oa::String& s) -> oa::DsSplit {
		const std::string v = oa::sdk::toStdString(s);
		if (v == "val")  { return oa::DsSplit::Val; }
		if (v == "test") { return oa::DsSplit::Test; }
		return oa::DsSplit::Train;
	};
	// Strip a leading "[MF]T[NOU]_" body-variant prefix when present (legacy MTN
	// naming); otherwise classify on the full name.
	auto contentOf = [](const oa::String& name) -> oa::String {
		const std::string s = oa::sdk::toStdString(name);
		if (s.size() > 4 && (s[0] == 'M' || s[0] == 'F') && s[1] == 'T' && s[3] == '_') {
			return oa::String(s.substr(4).c_str());
		}
		return name;
	};

	clips_.clear();
	metas_.clear();
	clips_.reserve(named.size());
	metas_.reserve(named.size());

	for (oa::Usize i = 0; i < named.size(); ++i) {
		const oa::UsdNamedClip& nc = named[i];
		auto packed = oa::PosePack::pack(nc.clip, sk);
		if (packed.isError()) {
			OaLogWarn(oa::LogComponent::Data, "dsgen3danim: skip clip '%s' (pack: %s)",
				nc.name.cStr(), packed.getStatus().toString().cStr());
			continue;
		}
		oa::PoseClip clip = packed.getValue();
		if (static_cast<oa::I32>(clip.poseDim) != poseDim_) {
			OaLogWarn(oa::LogComponent::Data, "dsgen3danim: skip clip '%s' (posedim %u != %d)",
				nc.name.cStr(), clip.poseDim, poseDim_);
			continue;
		}

		const std::string stem = oa::sdk::toStdString(nc.name);
		oa::DsClipMeta m;
		m.name      = nc.name;
		m.content   = contentOf(nc.name);
		m.character = (!stem.empty() && stem[0] == 'M') ? 0
		            : (!stem.empty() && stem[0] == 'F') ? 1 : 255;
		m.bodyType  = (stem.size() > 2 && stem[2] == 'N') ? 0
		            : (stem.size() > 2 && stem[2] == 'O') ? 1
		            : (stem.size() > 2 && stem[2] == 'U') ? 2 : 255;
		m.category  = oa::dsCategoryOf(m.content);
		m.split     = splitOf(nc.split);
		m.frames    = clip.frameCount;

		clips_.pushBack(oa::move(clip));
		metas_.pushBack(oa::move(m));
	}

	if (clips_.empty()) {
		OaLogError(oa::LogComponent::Data, "dsgen3danim: usd dataset has no packable clips: %s",
			path_.string().cStr());
		return false;
	}

	// Derive normalization at load time (train-split per-channel mean/std + the
	// dead-channel rule). rootTranslationDelta_ is off here, so modelFeature_
	// returns the raw absolute feature — identical to the stats the binary .oads
	// baked at pack time.
	recomputeStats_();

	OaLogInfo(oa::LogComponent::Data,
		"dsgen3danim: loaded %s (usd) — %zu clips · posedim=%d · fps=%.3g",
		path_.string().cStr(), clips_.size(), poseDim_, static_cast<double>(fps_));
	return true;
}

void oa::DsGen3dAnim::setRootTranslationDelta(bool inEnabled) {
	if (rootTranslationDelta_ == inEnabled) { return; }
	rootTranslationDelta_ = inEnabled;
	if (ok()) {
		recomputeStats_();
	}
}

oa::F32 oa::DsGen3dAnim::modelFeature_(const oa::PoseClip& inClip, oa::I32 inFrame, oa::I32 inChannel) const {
	const oa::Usize D = static_cast<oa::Usize>(poseDim_);
	const oa::Usize idx = static_cast<oa::Usize>(inFrame) * D + static_cast<oa::Usize>(inChannel);
	const oa::F32 v = inClip.samples[idx];
	if (!rootTranslationDelta_ || inChannel < 0 || inChannel >= 3) {
		return v;
	}
	if (inFrame <= 0) {
		return 0.0f;
	}
	const oa::Usize prev = static_cast<oa::Usize>(inFrame - 1) * D + static_cast<oa::Usize>(inChannel);
	return v - inClip.samples[prev];
}

void oa::DsGen3dAnim::recomputeStats_() {
	const oa::Usize D = static_cast<oa::Usize>(poseDim_);
	mean_.resize(D);
	std_.resize(D);
	if (D == 0 || clips_.empty()) { return; }

	oa::Vector<double> sum, sumsq;
	sum.resize(D);
	sumsq.resize(D);
	for (oa::Usize i = 0; i < D; ++i) {
		sum[i] = 0.0;
		sumsq[i] = 0.0;
	}

	oa::U64 trainFrames = 0;
	for (oa::Usize ci = 0; ci < clips_.size(); ++ci) {
		if (metas_[ci].split != oa::DsSplit::Train) { continue; }
		const oa::PoseClip& clip = clips_[ci];
		for (oa::U32 f = 0; f < clip.frameCount; ++f) {
			for (oa::I32 c = 0; c < poseDim_; ++c) {
				const oa::Usize ch = static_cast<oa::Usize>(c);
				const double v = static_cast<double>(modelFeature_(clip, static_cast<oa::I32>(f), c));
				sum[ch] += v;
				sumsq[ch] += v * v;
			}
		}
		trainFrames += clip.frameCount;
	}

	const double inv = trainFrames ? 1.0 / static_cast<double>(trainFrames) : 0.0;
	for (oa::Usize c = 0; c < D; ++c) {
		const double m = sum[c] * inv;
		const double var = sumsq[c] * inv - m * m;
		mean_[c] = static_cast<oa::F32>(m);
		std_[c]  = static_cast<oa::F32>(oa::sqrt(var > 0.0 ? var : 0.0));
		// Dead-channel rule: a channel whose std is below kStdFloor is effectively
		// constant in this set and carries no learnable signal. Dividing by a tiny
		// floor would amplify pure numerical jitter by ~1/floor (×10^4) — blowing up
		// model inputs AND making its target an irreducible noise term that floors
		// the loss. Use scale=1 (pass-through): since x≈mean, the normalized value
		// is ≈0 — neutral, not amplified. (Same rule as sklearn StandardScaler.)
		if (std_[c] < kStdFloor) { std_[c] = 1.0f; }
	}

	OaLogInfo(oa::LogComponent::Data, "dsgen3danim: model root translation = %s",
		rootTranslationDelta_ ? "delta" : "absolute");
}

void oa::DsGen3dAnim::buildIndices(oa::I32 inContextLen, oa::I64 inSeed) {
	contextLen_ = inContextLen;
	
	// seed all per-split RNGs with derived seeds for reproducibility
	trainRng_.seed(static_cast<oa::U64>(inSeed));
	valRng_.seed(static_cast<oa::U64>(inSeed + 1000));
	testRng_.seed(static_cast<oa::U64>(inSeed + 2000));
	
	windows_.clear();
	for (auto& splitWindows : splitWindows_) { splitWindows.clear(); }
	for (auto& cursor : splitCursor_) { cursor = 0; }
	for (oa::Usize ci = 0; ci < clips_.size(); ++ci) {
		const oa::I32 frames = static_cast<oa::I32>(clips_[ci].frameCount);
		if (frames < inContextLen + 1) { continue; }
		for (oa::I32 start = 0; start <= frames - inContextLen - 1; ++start) {
			const oa::Usize windowIdx = windows_.size();
			windows_.pushBack(Window{ static_cast<oa::I32>(ci), start });
			splitWindows_[splitIndex(metas_[ci].split)].pushBack(windowIdx);
		}
	}
	// Build shuffled indices
	indices_.resize(windows_.size());
	for (oa::Usize i = 0; i < indices_.size(); ++i) { indices_[i] = static_cast<oa::I64>(i); }
	// Use TrainRng for initial shuffle (affects all splits equally for now)
	trainRng_.shuffle(indices_.data(), indices_.size());
	trainRng_.shuffle(splitWindows_[0].data(), splitWindows_[0].size());
	valRng_.shuffle(splitWindows_[1].data(), splitWindows_[1].size());
	testRng_.shuffle(splitWindows_[2].data(), splitWindows_[2].size());
}

oa::I64 oa::DsGen3dAnim::size() const {
	return static_cast<oa::I64>(windows_.size());
}

oa::Matrix oa::DsGen3dAnim::getItem(oa::I64 inIndex) const {
	if (inIndex < 0 || static_cast<oa::Usize>(inIndex) >= windows_.size()) { return oa::Matrix(); }
	const Window& w = windows_[static_cast<oa::Usize>(inIndex)];
	const oa::PoseClip& clip = clips_[static_cast<oa::Usize>(w.clip)];
	// Return flat [ctx * poseDim] for now; reshape caller-side if needed
	const oa::Usize n = static_cast<oa::Usize>(contextLen_) * poseDim_;
	oa::Vector<float> xdata(n);
	for (oa::I32 t = 0; t < contextLen_; ++t) {
		for (oa::I32 c = 0; c < poseDim_; ++c) {
			const oa::Usize ch = static_cast<oa::Usize>(c);
			const oa::Usize i = static_cast<oa::Usize>(t) * static_cast<oa::Usize>(poseDim_) + ch;
			xdata[i] = (modelFeature_(clip, w.start + t, c) - mean_[ch]) / std_[ch];
		}
	}
	return oa::FnMatrix::fromBytes(
		oa::Span<const oa::U8>(reinterpret_cast<const oa::U8*>(xdata.data()), n * sizeof(float)),
		oa::MatrixShape{contextLen_, poseDim_}, oa::ScalarType::Float32);
}

oa::Dataset::Sample oa::DsGen3dAnim::getSample(oa::I64 inIndex) const {
	// X = context window, Y = next frame (single target)
	if (inIndex < 0 || static_cast<oa::Usize>(inIndex) >= windows_.size()) { return Sample(); }
	const Window& w = windows_[static_cast<oa::Usize>(inIndex)];
	const oa::PoseClip& clip = clips_[static_cast<oa::Usize>(w.clip)];

	const oa::Usize n = static_cast<oa::Usize>(contextLen_) * poseDim_;
	oa::Vector<float> xdata(n);
	for (oa::I32 t = 0; t < contextLen_; ++t) {
		for (oa::I32 c = 0; c < poseDim_; ++c) {
			const oa::Usize ch = static_cast<oa::Usize>(c);
			const oa::Usize i = static_cast<oa::Usize>(t) * static_cast<oa::Usize>(poseDim_) + ch;
			xdata[i] = (modelFeature_(clip, w.start + t, c) - mean_[ch]) / std_[ch];
		}
	}
	oa::Matrix x = oa::FnMatrix::fromBytes(
		oa::Span<const oa::U8>(reinterpret_cast<const oa::U8*>(xdata.data()), n * sizeof(float)),
		oa::MatrixShape{contextLen_, poseDim_}, oa::ScalarType::Float32);

	// Y: next frame (single pose vector)
	oa::Vector<float> ydata(poseDim_);
	for (oa::I32 c = 0; c < poseDim_; ++c) {
		const oa::Usize ch = static_cast<oa::Usize>(c);
		ydata[ch] = (modelFeature_(clip, w.start + contextLen_, c) - mean_[ch]) / std_[ch];
	}
	oa::Matrix y = oa::FnMatrix::fromBytes(
		oa::Span<const oa::U8>(reinterpret_cast<const oa::U8*>(ydata.data()), poseDim_ * sizeof(float)),
		oa::MatrixShape{poseDim_}, oa::ScalarType::Float32);

	return Sample(x, y);
}

oa::Usize oa::DsGen3dAnim::windowCount(oa::DsSplit inSplit) const noexcept {
	return splitWindows_[splitIndex(inSplit)].size();
}

bool oa::DsGen3dAnim::restrictSplitToClip(oa::DsSplit inSplit, const oa::String& inClipName) {
	const oa::I32 clipIdx = findClipByName(inClipName);
	if (clipIdx < 0) { return false; }

	const oa::Usize splitIdx = splitIndex(inSplit);
	oa::Vector<oa::Usize> filtered;
	for (oa::Usize windowIdx : splitWindows_[splitIdx]) {
		if (windows_[windowIdx].clip == clipIdx) {
			filtered.pushBack(windowIdx);
		}
	}
	if (filtered.empty()) { return false; }

	splitWindows_[splitIdx] = oa::move(filtered);
	splitCursor_[splitIdx] = 0;
	return true;
}

oa::Usize oa::DsGen3dAnim::splitClipCount(oa::DsSplit inSplit) const noexcept {
	oa::Usize n = 0;
	for (const auto& m : metas_) {
		if (m.split == inSplit) { ++n; }
	}
	return n;
}

void oa::DsGen3dAnim::nextBatch(oa::DsSplit inSplit, oa::I32 inBatch, oa::Matrix& outX, oa::Matrix& outY) {
	const oa::I64 rowLen = static_cast<oa::I64>(contextLen_) * poseDim_;
	const oa::I64 total  = static_cast<oa::I64>(inBatch) * rowLen;
	oa::Vector<float> xdata(static_cast<oa::Usize>(total));
	oa::Vector<float> ydata(static_cast<oa::Usize>(total));

	const oa::Usize splitIdx = splitIndex(inSplit);
	oa::Vector<oa::Usize>& splitWindows = splitWindows_[splitIdx];
	if (splitWindows.empty()) { outX = oa::Matrix(); outY = oa::Matrix(); return; }

	// Iterate shuffled windows without replacement inside an epoch. This matches
	// the MNIST/DataLoader pattern and makes tiny overfit sets deterministic.
	auto& rng = (inSplit == oa::DsSplit::Train) ? trainRng_ :
	            (inSplit == oa::DsSplit::Val)   ? valRng_   : testRng_;
	for (oa::I32 b = 0; b < inBatch; ++b) {
		if (splitCursor_[splitIdx] >= splitWindows.size()) {
			splitCursor_[splitIdx] = 0;
			rng.shuffle(splitWindows.data(), splitWindows.size());
		}
		const Window& wnd = windows_[splitWindows[splitCursor_[splitIdx]++]];
		const oa::PoseClip& clip = clips_[static_cast<oa::Usize>(wnd.clip)];
		float* px = &xdata[static_cast<oa::Usize>(static_cast<oa::I64>(b) * rowLen)];
		float* py = &ydata[static_cast<oa::Usize>(static_cast<oa::I64>(b) * rowLen)];
		for (oa::I64 t = 0; t < contextLen_; ++t) {
			float* dx = px + t * poseDim_;
			float* dy = py + t * poseDim_;
			for (oa::I32 c = 0; c < poseDim_; ++c) {
				const float m = mean_[static_cast<oa::Usize>(c)];
				const float s = std_[static_cast<oa::Usize>(c)];
				dx[c] = (modelFeature_(clip, static_cast<oa::I32>(wnd.start + t), c) - m) / s;
				dy[c] = (modelFeature_(clip, static_cast<oa::I32>(wnd.start + t + 1), c) - m) / s;
			}
		}
	}

	outX = oa::FnMatrix::empty(oa::MatrixShape{inBatch, contextLen_, poseDim_}, oa::ScalarType::Float32);
	outY = oa::FnMatrix::empty(oa::MatrixShape{inBatch, contextLen_, poseDim_}, oa::ScalarType::Float32);
	oa::memcpy(outX.dataAs<float>(), xdata.data(), static_cast<oa::Usize>(total) * sizeof(float));
	oa::memcpy(outY.dataAs<float>(), ydata.data(), static_cast<oa::Usize>(total) * sizeof(float));
}

oa::I32 oa::DsGen3dAnim::findClipByName(const oa::String& inName) const {
	for (oa::Usize i = 0; i < metas_.size(); ++i) {
		if (metas_[i].name == inName) { return static_cast<oa::I32>(i); }
	}
	return -1;
}

bool oa::DsGen3dAnim::seedRaw(oa::I32 inClipIdx, oa::I32 inContext, oa::Vector<oa::F32>& outRaw) const {
	if (inClipIdx < 0 || static_cast<oa::Usize>(inClipIdx) >= clips_.size()) { return false; }
	const oa::PoseClip& clip = clips_[static_cast<oa::Usize>(inClipIdx)];
	if (static_cast<oa::I32>(clip.frameCount) < inContext || poseDim_ <= 0) { return false; }
	const oa::Usize n = static_cast<oa::Usize>(inContext) * static_cast<oa::Usize>(poseDim_);
	outRaw.resize(n);
	oa::memcpy(outRaw.data(), clip.samples.data(), n * sizeof(oa::F32));
	return true;
}

bool oa::DsGen3dAnim::clipRaw(oa::I32 inClipIdx, oa::Vector<oa::F32>& outRaw, oa::U32& outFrames) const {
	if (inClipIdx < 0 || static_cast<oa::Usize>(inClipIdx) >= clips_.size()) { return false; }
	const oa::PoseClip& clip = clips_[static_cast<oa::Usize>(inClipIdx)];
	outFrames = clip.frameCount;
	outRaw.resize(clip.samples.size());
	oa::memcpy(outRaw.data(), clip.samples.data(), clip.samples.size() * sizeof(oa::F32));
	return true;
}

bool oa::DsGen3dAnim::seedModelRaw(oa::I32 inClipIdx, oa::I32 inContext, oa::Vector<oa::F32>& outRaw) const {
	if (inClipIdx < 0 || static_cast<oa::Usize>(inClipIdx) >= clips_.size()) { return false; }
	const oa::PoseClip& clip = clips_[static_cast<oa::Usize>(inClipIdx)];
	if (static_cast<oa::I32>(clip.frameCount) < inContext || poseDim_ <= 0) { return false; }
	const oa::Usize n = static_cast<oa::Usize>(inContext) * static_cast<oa::Usize>(poseDim_);
	outRaw.resize(n);
	for (oa::I32 f = 0; f < inContext; ++f) {
		for (oa::I32 c = 0; c < poseDim_; ++c) {
			outRaw[static_cast<oa::Usize>(f) * static_cast<oa::Usize>(poseDim_) + static_cast<oa::Usize>(c)] =
				modelFeature_(clip, f, c);
		}
	}
	return true;
}

bool oa::DsGen3dAnim::clipModelRaw(oa::I32 inClipIdx, oa::Vector<oa::F32>& outRaw, oa::U32& outFrames) const {
	if (inClipIdx < 0 || static_cast<oa::Usize>(inClipIdx) >= clips_.size()) { return false; }
	const oa::PoseClip& clip = clips_[static_cast<oa::Usize>(inClipIdx)];
	outFrames = clip.frameCount;
	outRaw.resize(clip.samples.size());
	for (oa::U32 f = 0; f < clip.frameCount; ++f) {
		for (oa::I32 c = 0; c < poseDim_; ++c) {
			outRaw[static_cast<oa::Usize>(f) * static_cast<oa::Usize>(poseDim_) + static_cast<oa::Usize>(c)] =
				modelFeature_(clip, static_cast<oa::I32>(f), c);
		}
	}
	return true;
}
