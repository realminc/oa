// dsgen3danimctl — Gen3dAnim dataset control tool (unified data pipeline).
//
// One binary, GitHub-styled subcommands, for the whole Gen3dAnim data path:
// ingest retargeted MetaHuman USD clips, combine them into a trainable `.usd`
// dataset, inspect it, and bake it to a C++ header. All CPU-only (no vulkan) —
// the training/generation runtime stays in traingen3danim / gen3danim.
//
// pipeline order:  clean → strip → pack → info → bake
//
//   clean  Ingest a retargeted MetaHuman USD clip onto the clean canonical base
//          (strip Unreal IK/weapon/corrective junk, collapse hinges, Y-up→spec).
//   strip  Losslessly delete junk joint prims (IK/weapon/COM) from source clips
//          IN PLACE, keeping the kept joints' transforms bit-exact (source hygiene;
//          pack strips anyway, so this is optional). Batch via --dir.
//   pack   Combine source UsdSkel clips into ONE human-readable `.usd` dataset
//          (single SkelRoot/Skeleton + N SkelAnimation prims). split by content.
//   info   Inspect a combined `.usd` dataset: config, per-clip split/frames/
//          category, window counts, normalization stat ranges.
//   bake   Bake a combined `.usd` dataset into a self-contained C++ header (raw
//          model-space frames + per-channel mean/std + per-clip metadata).
//
// usage:
//   dsgen3danimctl clean  --in clip.usd [--save clip.3danim] [--usda c.usda] [--fbx c.fbx]
//   dsgen3danimctl pack   --in <usdRoot> --out walkset.usd
//   dsgen3danimctl pack   --clips list.txt --out walkset.usd --val "Walk_Left" --test "Idle"
//   dsgen3danimctl info   --dataset walkset.usd [--context 32]
//   dsgen3danimctl bake   --dataset walkset.usd --out Walk.gen.h --ns walkFwdClip [--root-delta]
//
// Implements the canonical ALM dataset pipeline contract.

#include <oa/core/cli.h>
#include <oa/core/filesystem.h>

#include <anim/fbxWriter.h>
#include <anim/poseClip.h>
#include <core/streamText.h>
#include <anim/posePack.h>
#include <anim/usd.h>
#include <data/dsGen3dAnim.h>
#include <rig/skeleton.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <set>
#include <string>
#include <utility>

// ============================================================================
// Shared helpers
// ============================================================================

// Leaf bone name of a UsdSkel joint path ("root/pelvis/spine_01" -> "spine_01").
static oa::StringView leafName(oa::StringView inPath) {
	oa::Usize slash = oa::StringView::Npos;
	for (oa::Usize i = 0; i < inPath.size(); ++i) {
		if (inPath[i] == '/') { slash = i; }
	}
	return slash == oa::StringView::Npos ? inPath : inPath.subStr(slash + 1);
}

static void ensureParent(const oa::Path& p) {
	if (auto parent = p.parentPath(); !parent.empty()) {
		(void)oa::Filesystem::createDirectories(parent);
	}
}

// ============================================================================
// clean — ingest a retargeted MetaHuman USD clip into the clean canonical form
// ============================================================================

static int cmdClean(const oa::String& inIn, const oa::String& inSave,
                    const oa::String& inUsda, const oa::String& inFbx) {
	if (!oa::Filesystem::isFile(oa::Path(inIn))) {
		std::printf("clean: input not found: %s\n", inIn.cStr());
		return 1;
	}

	auto read = oa::Usd::readUsda(oa::Path(inIn));
	if (!read.isOk()) {
		std::printf("clean: read failed: %s\n", read.getStatus().toString().cStr());
		return 1;
	}
	const oa::UsdSkelClip& usd = *read;
	const oa::Skeleton& sk = oa::skMetaHuman();

	// Classify the clip's joints against the clean base.
	oa::I32 kept = 0, dropped = 0;
	oa::String droppedNames;
	for (oa::I32 u = 0; u < usd.jointCount(); ++u) {
		const oa::StringView leaf = leafName(usd.jointPaths[static_cast<oa::Usize>(u)]);
		if (sk.indexOf(leaf) >= 0) {
			++kept;
		} else {
			++dropped;
			if (dropped <= 16) { droppedNames += oa::String(leaf); droppedNames += " "; }
		}
	}
	oa::I32 baseMissing = 0;
	for (oa::I32 s = 0; s < sk.jointCount(); ++s) {
		bool found = false;
		for (oa::I32 u = 0; u < usd.jointCount() && !found; ++u) {
			found = (leafName(usd.jointPaths[static_cast<oa::Usize>(u)]) == sk.joints[static_cast<oa::Usize>(s)].name);
		}
		if (!found) { ++baseMissing; }
	}

	std::printf("clip:    %s\n", inIn.cStr());
	std::printf("  up-axis: %s   frames: %u   fps: %.3g\n",
		usd.upAxis == 1 ? "Y" : "Z", usd.frameCount, usd.fps);
	std::printf("  usd joints: %d   -> base kept: %d   junk dropped: %d   base absent: %d\n",
		usd.jointCount(), kept, dropped, baseMissing);
	std::printf("  dropped: %s\n", droppedNames.cStr());

	auto packed = oa::PosePack::pack(usd, sk);
	if (!packed.isOk()) {
		std::printf("clean: pack failed: %s\n", packed.getStatus().toString().cStr());
		return 1;
	}
	const oa::PoseClip& clip = *packed;
	std::printf("  packed: %u frames x %u channels (compact poseDim)  = %llu floats\n",
		clip.frameCount, clip.poseDim,
		static_cast<unsigned long long>(clip.samples.size()));
	const oa::I32 uniform = 9 + 6 * (sk.jointCount() - 1) + static_cast<oa::I32>(sk.contactJoints.size());
	std::printf("  vs uniform-6D would be %d channels (%.0f%% smaller)\n",
		uniform, 100.0 * (1.0 - static_cast<double>(clip.poseDim) / uniform));

	if (!inSave.empty()) {
		oa::Path p(inSave); ensureParent(p);
		if (auto st = clip.write3dAnim(p); !st.isOk()) {
			std::printf("  .3danim write failed: %s\n", st.toString().cStr());
		} else {
			std::printf("  wrote %s\n", inSave.cStr());
		}
	}
	if (!inUsda.empty() || !inFbx.empty()) {
		auto back = oa::PosePack::unpack(clip, sk);
		if (!back.isOk()) {
			std::printf("  unpack failed: %s\n", back.getStatus().toString().cStr());
			return 1;
		}
		if (!inUsda.empty()) {
			oa::Path p(inUsda); ensureParent(p);
			if (auto st = oa::Usd::writeUsda(p, *back, "rig"); !st.isOk()) {
				std::printf("  .usda write failed: %s\n", st.toString().cStr());
			} else {
				std::printf("  wrote %s\n", inUsda.cStr());
			}
		}
		if (!inFbx.empty()) {
			oa::Path p(inFbx); ensureParent(p);
			if (auto st = oa::Fbx::writeFbx(p, *back); !st.isOk()) {
				std::printf("  .fbx write failed: %s\n", st.toString().cStr());
			} else {
				std::printf("  wrote %s\n", inFbx.cStr());
			}
		}
	}
	return 0;
}

// ============================================================================
// pack — combine source UsdSkel clips into one `.usd` dataset
// ============================================================================

// Drop every joint that isn't in the clean base skeleton (Unreal IK joints,
// weapons, center_of_mass, interaction, correctives) — lossless for the kept base
// joints. The same leaf-name match oa::PosePack::Pack uses at load (junk is ignored
// there too); doing it here keeps the dataset .usd file itself clean + viewable.
static oa::UsdSkelClip stripToBase(const oa::UsdSkelClip& inClip, const oa::Skeleton& inSkel, oa::I32& outDropped) {
	const oa::I32 nIn = inClip.jointCount();
	oa::Vec<oa::I32> keep;
	for (oa::I32 u = 0; u < nIn; ++u) {
		if (inSkel.indexOf(leafName(inClip.jointPaths[static_cast<oa::Usize>(u)])) >= 0) {
			keep.pushBack(u);
		}
	}
	outDropped = nIn - static_cast<oa::I32>(keep.size());

	oa::UsdSkelClip out;
	out.frameCount = inClip.frameCount;
	out.fps        = inClip.fps;
	out.upAxis     = inClip.upAxis;
	const oa::Usize nOut = keep.size();
	out.jointPaths.reserve(nOut);
	for (oa::I32 u : keep) { out.jointPaths.pushBack(inClip.jointPaths[static_cast<oa::Usize>(u)]); }
	if (inClip.bindTransforms.size() == static_cast<oa::Usize>(nIn)) {
		out.bindTransforms.reserve(nOut);
		for (oa::I32 u : keep) { out.bindTransforms.pushBack(inClip.bindTransforms[static_cast<oa::Usize>(u)]); }
	}
	if (inClip.restTransforms.size() == static_cast<oa::Usize>(nIn)) {
		out.restTransforms.reserve(nOut);
		for (oa::I32 u : keep) { out.restTransforms.pushBack(inClip.restTransforms[static_cast<oa::Usize>(u)]); }
	}
	out.translations.reserve(static_cast<oa::Usize>(out.frameCount) * nOut);
	out.rotations.reserve(static_cast<oa::Usize>(out.frameCount) * nOut);
	for (oa::U32 f = 0; f < inClip.frameCount; ++f) {
		const oa::Usize base = static_cast<oa::Usize>(f) * static_cast<oa::Usize>(nIn);
		for (oa::I32 u : keep) {
			out.translations.pushBack(inClip.translations[base + static_cast<oa::Usize>(u)]);
			out.rotations.pushBack(inClip.rotations[base + static_cast<oa::Usize>(u)]);
		}
	}
	return out;
}

static std::set<std::string> parseList(const oa::String& inCsv) {
	std::set<std::string> out;
	std::string cur;
	for (oa::Usize i = 0; i < inCsv.size(); ++i) {
		const char ch = inCsv[i];
		if (ch == ',') { if (!cur.empty()) { out.insert(cur); cur.clear(); } }
		else if (ch != ' ') { cur.push_back(ch); }
	}
	if (!cur.empty()) { out.insert(cur); }
	return out;
}

// "…/MM_Unarmed_Walk_Fwd.usd" → stem "MM_Unarmed_Walk_Fwd".
static std::string stem(const oa::String& inPath) {
	std::string p = oa::sdk::toStdString(inPath);
	std::size_t slash = p.find_last_of("/\\");
	std::string base = (slash == std::string::npos) ? p : p.substr(slash + 1);
	std::size_t dot = base.find_last_of('.');
	return (dot == std::string::npos) ? base : base.substr(0, dot);
}

// "MTN_N_Idle_E" → content "N_Idle_E" (strip a [MF]T[NOU]_ body-variant prefix);
// names without that prefix (e.g. "MM_Unarmed_Walk_Fwd") pass through unchanged.
static std::string contentOf(const std::string& inStem) {
	if (inStem.size() > 4 && (inStem[0] == 'M' || inStem[0] == 'F') &&
	    inStem[1] == 'T' && inStem[3] == '_') {
		return inStem.substr(4);
	}
	return inStem;
}

// Read a --clips list file into source paths (skips blanks and '#' comments).
static oa::Vec<oa::String> readClipList(const oa::String& inPath) {
	oa::Vec<oa::String> out;
	auto text = oa::Filesystem::readText(oa::Path(inPath));
	if (!text.isOk()) { return out; }
	std::string s = oa::sdk::toStdString(*text);
	std::string line;
	auto flush = [&]() {
		std::size_t a = line.find_first_not_of(" \t\r\n");
		std::size_t b = line.find_last_not_of(" \t\r\n");
		if (a != std::string::npos && line[a] != '#') {
			out.pushBack(oa::String(line.substr(a, b - a + 1).c_str()));
		}
		line.clear();
	};
	for (char c : s) { if (c == '\n') { flush(); } else { line.push_back(c); } }
	flush();
	return out;
}

static int cmdPack(const oa::String& inIn, const oa::String& inClips, const oa::String& inOut,
                   const oa::String& inVal, const oa::String& inTest, bool inRaw) {
	// Gather source .usd paths from --clips list or --in folder walk.
	oa::Vec<oa::String> sources;
	if (!inClips.empty()) {
		sources = readClipList(inClips);
		if (sources.empty()) {
			std::printf("pack: empty/unreadable clip list: %s\n", inClips.cStr());
			return 1;
		}
	} else if (!inIn.empty()) {
		auto listing = oa::Filesystem::listAll(oa::Path(inIn), /*recursive=*/true);
		if (!listing.isOk()) {
			std::printf("pack: cannot list %s: %s\n", inIn.cStr(),
				listing.getStatus().toString().cStr());
			return 1;
		}
		for (const oa::Path& p : *listing) {
			const std::string path = oa::sdk::toStdString(p.string());
			if (path.size() >= 4 && path.substr(path.size() - 4) == ".usd") {
				sources.pushBack(p.string());
			}
		}
	} else {
		std::printf("pack: provide --in <dir> or --clips <list.txt>\n");
		return 1;
	}

	const std::set<std::string> valSet  = parseList(inVal);
	const std::set<std::string> testSet = parseList(inTest);
	const oa::Skeleton& sk = oa::skMetaHuman();

	oa::Vec<oa::UsdNamedClip> clips;
	oa::I32 nTrain = 0, nVal = 0, nTest = 0, nSkip = 0, nDropTotal = 0;
	for (const oa::String& src : sources) {
		auto read = oa::Usd::readUsda(oa::Path(src));
		if (!read.isOk()) {
			std::printf("  skip %s (read: %s)\n", src.cStr(), read.getStatus().toString().cStr());
			++nSkip; continue;
		}
		const std::string clipStem = stem(src);
		const std::string content = contentOf(clipStem);

		oa::UsdNamedClip nc;
		nc.name = oa::String(clipStem.c_str());
		if (inRaw) {
			nc.clip = oa::move(read.getValue());
		} else {
			oa::I32 dropped = 0;
			nc.clip = stripToBase(read.getValue(), sk, dropped);
			nDropTotal += dropped;
		}
		if      (valSet.count(content))  { nc.split = oa::String("val");  ++nVal; }
		else if (testSet.count(content)) { nc.split = oa::String("test"); ++nTest; }
		else                             { nc.split = oa::String("train"); ++nTrain; }
		clips.pushBack(oa::move(nc));
	}

	if (clips.empty()) {
		std::printf("pack: no usable .usd clips from input\n");
		return 1;
	}

	std::printf("combined %llu clips (train %d / val %d / test %d, skipped %d) · joints %s\n",
		static_cast<unsigned long long>(clips.size()), nTrain, nVal, nTest, nSkip,
		inRaw ? "raw (junk kept)" : "stripped to clean base");
	if (!inRaw) {
		std::printf("  junk joints dropped: %d total\n", nDropTotal);
	}

	if (auto st = oa::Usd::writeUsdaMulti(oa::Path(inOut),
			oa::Span<const oa::UsdNamedClip>(clips.data(), clips.size())); !st.isOk()) {
		std::printf("pack: write failed: %s\n", st.toString().cStr());
		return 1;
	}
	std::printf("wrote %s\n", inOut.cStr());
	return 0;
}

// ============================================================================
// strip — losslessly remove junk joints from source clips, in place
// ============================================================================
//
// Unlike `clean` (which round-trips through the lossy compact 272-channel pack
// and so rewrites the kept joints too), `strip` keeps every base joint's original
// per-frame trans+quat BIT-EXACT and only deletes the junk joint prims (Unreal
// IK / weapon / interaction / center_of_mass). Use it to make the SOURCE dataset
// clips themselves junk-free; the packed dataset is already stripped either way,
// so this is purely for source hygiene / DCC review, not a training requirement.
static int cmdStrip(const oa::String& inIn, const oa::String& inDir,
                    const oa::String& inExclude, bool inDryRun) {
	const oa::Skeleton& sk = oa::skMetaHuman();
	const std::set<std::string> excl = parseList(inExclude);

	// Gather targets: a single --in file, or every .usd under --dir (recursive),
	// minus any path containing an --exclude substring (AimOffset by default).
	oa::Vec<oa::String> files;
	if (!inDir.empty()) {
		auto listing = oa::Filesystem::listAll(oa::Path(inDir), /*recursive=*/true);
		if (!listing.isOk()) {
			std::printf("strip: cannot list %s: %s\n", inDir.cStr(),
				listing.getStatus().toString().cStr());
			return 1;
		}
		for (const oa::Path& p : *listing) {
			const std::string path = oa::sdk::toStdString(p.string());
			if (path.size() < 4 || path.substr(path.size() - 4) != ".usd") { continue; }
			bool skip = false;
			for (const std::string& e : excl) { if (path.find(e) != std::string::npos) { skip = true; break; } }
			if (skip) { continue; }
			files.pushBack(p.string());
		}
	} else if (!inIn.empty()) {
		files.pushBack(inIn);
	} else {
		std::printf("strip: provide --in <file> or --dir <root>\n");
		return 1;
	}
	if (files.empty()) { std::printf("strip: no .usd files to process\n"); return 1; }

	oa::I32 nDone = 0, nSkip = 0, nDrop = 0, nIncomplete = 0;
	for (const oa::String& src : files) {
		auto read = oa::Usd::readUsda(oa::Path(src));
		if (!read.isOk()) {
			std::printf("  skip %s (read: %s)\n", src.cStr(), read.getStatus().toString().cStr());
			++nSkip; continue;
		}
		oa::I32 dropped = 0;
		oa::UsdSkelClip stripped = stripToBase(read.getValue(), sk, dropped);
		const oa::I32 kept = stripped.jointCount();
		const bool incomplete = (kept != sk.jointCount());
		if (incomplete) { ++nIncomplete; }
		nDrop += dropped;
		if (inDryRun) {
			std::printf("  [dry] kept %d  drop %d%s  %s\n", kept, dropped,
				incomplete ? "  (INCOMPLETE base)" : "", src.cStr());
			++nDone; continue;
		}
		if (auto st = oa::Usd::writeUsda(oa::Path(src), stripped, "rig"); !st.isOk()) {
			std::printf("  FAIL %s (write: %s)\n", src.cStr(), st.toString().cStr());
			++nSkip; continue;
		}
		++nDone;
	}
	std::printf("strip: %s %d files · junk dropped %d total · incomplete-base %d · skipped %d\n",
		inDryRun ? "would process" : "rewrote", nDone, nDrop, nIncomplete, nSkip);
	return 0;
}

// ============================================================================
// info — inspect a combined `.usd` dataset
// ============================================================================

static int cmdInfo(const oa::String& inPath, oa::I32 inContext) {
	oa::DsGen3dAnim ds;
	if (auto st = ds.load(oa::Path(inPath)); !st.isOk()) {
		std::printf("info: load failed: %s\n", st.toString().cStr());
		return 1;
	}
	ds.buildIndices(inContext);

	std::printf("\n  Dataset: %s\n", inPath.cStr());
	std::printf("  poseDim: %d   fps: %.3g   Clips: %zu\n",
		ds.poseDim(), static_cast<double>(ds.fps()), ds.clipCount());
	std::printf("  split clips:   train=%zu  val=%zu  test=%zu\n",
		ds.splitClipCount(oa::DsSplit::Train),
		ds.splitClipCount(oa::DsSplit::Val),
		ds.splitClipCount(oa::DsSplit::Test));
	std::printf("  Windows@ctx%d:  train=%zu  val=%zu  test=%zu\n", inContext,
		ds.windowCount(oa::DsSplit::Train),
		ds.windowCount(oa::DsSplit::Val),
		ds.windowCount(oa::DsSplit::Test));

	// Normalization sanity: the ranges should be finite (no NaN/Inf), std >= 0.
	const oa::Vec<oa::F32>& mean = ds.mean();
	const oa::Vec<oa::F32>& sd   = ds.std();
	if (!mean.empty() && !sd.empty()) {
		oa::F32 mmin = mean[0], mmax = mean[0], smin = sd[0], smax = sd[0];
		for (oa::Usize i = 0; i < mean.size(); ++i) {
			mmin = std::min(mmin, mean[i]); mmax = std::max(mmax, mean[i]);
			smin = std::min(smin, sd[i]);   smax = std::max(smax, sd[i]);
		}
		std::printf("  Mean range:    [%.4g, %.4g]\n", static_cast<double>(mmin), static_cast<double>(mmax));
		std::printf("  std range:     [%.4g, %.4g]\n", static_cast<double>(smin), static_cast<double>(smax));
	}

	std::printf("\n  %-5s %-34s %-7s %-7s %s\n", "idx", "Name", "split", "frames", "category");
	const oa::Vec<oa::DsClipMeta>& metas = ds.metas();
	for (oa::Usize i = 0; i < metas.size(); ++i) {
		const oa::DsClipMeta& m = metas[i];
		const char* sp = (m.split == oa::DsSplit::Train) ? "train"
		               : (m.split == oa::DsSplit::Val)   ? "val" : "test";
		std::printf("  %-5zu %-34s %-7s %-7u %s\n",
			i, m.name.cStr(), sp, m.frames, oa::dsCategoryName(m.category));
	}
	std::printf("\n");
	return 0;
}

// ============================================================================
// bake — bake a combined `.usd` dataset into a C++ header
// ============================================================================

// Emit a valid C++ float literal. "%.7g" alone drops the decimal point for whole
// numbers ("30" → invalid "30f"), so append ".0" when there's no '.'/'e'/'E'.
static void emitFloat(FILE* f, double v) {
	char buf[64];
	std::snprintf(buf, sizeof(buf), "%.7g", v);
	const bool isFloatLit = std::strpbrk(buf, ".eE") != nullptr;
	std::fprintf(f, "%s%sf,", buf, isFloatLit ? "" : ".0");
}

static int cmdBake(const oa::String& inPath, const oa::String& inOut,
                   const oa::String& inNs, bool inRootDelta) {
	oa::DsGen3dAnim ds;
	if (auto st = ds.load(oa::Path(inPath)); !st.isOk()) {
		std::printf("bake: load failed: %s\n", st.toString().cStr());
		return 1;
	}
	// root as per-frame delta: recomputes Mean/std and makes clipModelRaw emit
	// displacement for channels 0-2. Small, well-conditioned root → stable rollout.
	ds.setRootTranslationDelta(inRootDelta);

	const oa::I32 poseDim = ds.poseDim();
	const oa::Usize clipCount = ds.clipCount();
	const oa::Vec<oa::DsClipMeta>& metas = ds.metas();
	const oa::Vec<oa::F32>& mean = ds.mean();
	const oa::Vec<oa::F32>& sd   = ds.std();

	// Gather every clip's raw model-space frames (un-standardized — the tutorial
	// standardizes on the fly with the baked Mean/std, exactly like nextBatch).
	oa::Vec<oa::Vec<oa::F32>> clipData;
	oa::Vec<oa::U32>        clipFrames;
	oa::U32 frameTotal = 0;
	for (oa::Usize i = 0; i < clipCount; ++i) {
		oa::Vec<oa::F32> raw; oa::U32 frames = 0;
		if (!ds.clipModelRaw(static_cast<oa::I32>(i), raw, frames) || frames == 0) {
			std::printf("bake: clip %zu has no frames\n", i);
			return 1;
		}
		frameTotal += frames;
		clipData.pushBack(oa::move(raw));
		clipFrames.pushBack(frames);
	}

	FILE* f = std::fopen(inOut.cStr(), "wb");
	if (!f) {
		std::printf("bake: cannot open %s for write\n", inOut.cStr());
		return 1;
	}

	std::fprintf(f, "#pragma once\n");
	std::fprintf(f, "// AUTO-GENERATED by dsgen3danimctl bake — DO NOT EDIT.\n");
	std::fprintf(f, "// source: %s\n", inPath.cStr());
	std::fprintf(f, "// Real MetaHuman ChannelSpec motion (poseDim=%d) baked for TutorialGen3dAnim.\n\n",
		poseDim);
	std::fprintf(f, "namespace %s {\n\n", inNs.cStr());

	std::fprintf(f, "inline constexpr int   PoseDim    = %d;\n", poseDim);
	{
		// scalar float literal (semicolon, not the array comma EmitFloat appends).
		char fpsBuf[64];
		std::snprintf(fpsBuf, sizeof(fpsBuf), "%.7g", static_cast<double>(ds.fps()));
		const bool isLit = std::strpbrk(fpsBuf, ".eE") != nullptr;
		std::fprintf(f, "inline constexpr float Fps        = %s%sf;\n", fpsBuf, isLit ? "" : ".0");
	}
	std::fprintf(f, "inline constexpr int   ClipCount  = %zu;\n", clipCount);
	std::fprintf(f, "inline constexpr int   FrameTotal = %u;\n", frameTotal);
	// When true, channels 0-2 (root xyz) are per-frame displacement, not absolute
	// position. The consumer must cumulatively integrate them to recover world pos.
	std::fprintf(f, "inline constexpr bool  RootTranslationDelta = %s;\n\n",
		inRootDelta ? "true" : "false");

	// Per-clip metadata.
	std::fprintf(f, "inline constexpr int ClipFrames[ClipCount] = { ");
	for (oa::Usize i = 0; i < clipCount; ++i) std::fprintf(f, "%u%s", clipFrames[i], i + 1 < clipCount ? ", " : " ");
	std::fprintf(f, "};\n");

	std::fprintf(f, "inline constexpr int ClipOffset[ClipCount] = { ");
	{ oa::U32 off = 0; for (oa::Usize i = 0; i < clipCount; ++i) { std::fprintf(f, "%u%s", off, i + 1 < clipCount ? ", " : " "); off += clipFrames[i]; } }
	std::fprintf(f, "};  // frame offset into Poses\n");

	std::fprintf(f, "inline constexpr int ClipSplit[ClipCount]  = { ");
	for (oa::Usize i = 0; i < clipCount; ++i) std::fprintf(f, "%d%s", static_cast<int>(metas[i].split), i + 1 < clipCount ? ", " : " ");
	std::fprintf(f, "};  // 0=train 1=val 2=test\n");

	std::fprintf(f, "inline constexpr const char* ClipNames[ClipCount] = { ");
	for (oa::Usize i = 0; i < clipCount; ++i) std::fprintf(f, "\"%s\"%s", metas[i].name.cStr(), i + 1 < clipCount ? ", " : " ");
	std::fprintf(f, "};\n\n");

	// Per-channel normalization stats.
	std::fprintf(f, "inline constexpr float Mean[PoseDim] = {\n");
	for (oa::I32 ch = 0; ch < poseDim; ++ch) { emitFloat(f, static_cast<double>(mean[static_cast<oa::Usize>(ch)])); if ((ch % 8 == 7)) std::fprintf(f, "\n"); }
	std::fprintf(f, "};\n");
	std::fprintf(f, "inline constexpr float Std[PoseDim] = {\n");
	for (oa::I32 ch = 0; ch < poseDim; ++ch) { emitFloat(f, static_cast<double>(sd[static_cast<oa::Usize>(ch)])); if ((ch % 8 == 7)) std::fprintf(f, "\n"); }
	std::fprintf(f, "};\n\n");

	// Raw model-space frames, clip-major: Poses[(ClipOffset[c]+f)*poseDim + ch].
	// `inline const` (not constexpr) keeps the large initializer fast to compile.
	std::fprintf(f, "// raw (un-standardized) model-space pose frames, frame-major.\n");
	std::fprintf(f, "inline const float Poses[FrameTotal * PoseDim] = {\n");
	for (oa::Usize i = 0; i < clipCount; ++i) {
		const oa::Vec<oa::F32>& raw = clipData[i];
		std::fprintf(f, "// --- clip %zu: %s (%u frames) ---\n", i, metas[i].name.cStr(), clipFrames[i]);
		for (oa::Usize k = 0; k < raw.size(); ++k) {
			emitFloat(f, static_cast<double>(raw[k]));
			if (k % static_cast<oa::Usize>(poseDim) == static_cast<oa::Usize>(poseDim) - 1) std::fprintf(f, "\n");
		}
	}
	std::fprintf(f, "};\n\n");
	std::fprintf(f, "}  // namespace %s\n", inNs.cStr());

	std::fclose(f);

	std::printf("bake: wrote %s\n  poseDim=%d fps=%.3g Clips=%zu FrameTotal=%u  (%.1f KB of floats)\n",
		inOut.cStr(), poseDim, static_cast<double>(ds.fps()), clipCount, frameTotal,
		static_cast<double>(frameTotal) * poseDim * 4.0 / 1024.0);
	return 0;
}

// ============================================================================
// CLI
// ============================================================================

struct DsCtlConfig {
	// clean
	oa::String cleanIn;
	oa::String cleanSave;
	oa::String cleanUsda;
	oa::String cleanFbx;
	// pack
	oa::String packIn;
	oa::String packClips;
	oa::String packOut;
	oa::String packVal;
	oa::String packTest;
	bool     packRaw = false;
	// strip
	oa::String stripIn;
	oa::String stripDir;
	oa::String stripExclude = "AimOffset";
	bool     stripDryRun  = false;
	// info
	oa::String infoPath;
	oa::I32    infoContext = 32;
	// bake
	oa::String bakePath;
	oa::String bakeOut;
	oa::String bakeNs = "walkFwdClip";
	bool     bakeRootDelta = false;
};

class DsCtlCli : public oa::Cli<DsCtlConfig> {
public:
	DsCtlCli() : oa::Cli<DsCtlConfig>(
		"dsgen3danimctl", "Gen3dAnim dataset control tool (clean/pack/info/bake)") {
		setEpilog(
			"pipeline: clean -> pack -> info -> bake\n"
			"\n"
			"Examples:\n"
			"  dsgen3danimctl clean --in clip.usd --usda clean.usda\n"
			"  dsgen3danimctl pack  --in usdRoot/ --out walkset.usd\n"
			"  dsgen3danimctl pack  --clips list.txt --out walkset.usd --val \"Walk_Left\" --test \"Idle\"\n"
			"  dsgen3danimctl info  --dataset walkset.usd --context 32\n"
			"  dsgen3danimctl bake  --dataset walkset.usd --out Walk.gen.h --ns walkFwdClip --root-delta\n"
		);

		auto* clean = addSubcommand("clean", "Ingest a retargeted MetaHuman USD clip into the clean canonical form");
		clean->addOption("--in,-i", cfg_.cleanIn,   "input .usda clip (retargeted MetaHuman)")->required();
		clean->addOption("--save",  cfg_.cleanSave, "Write canonical .3danim");
		clean->addOption("--usda",  cfg_.cleanUsda, "Write round-tripped .usda (clean base only)");
		clean->addOption("--fbx",   cfg_.cleanFbx,  "Write FBX (DCC review)");

		auto* pack = addSubcommand("pack", "Combine source UsdSkel clips into one .usd dataset");
		pack->addOption("--in,-i",  cfg_.packIn,    "root dir of source .usd clips (recursive)");
		pack->addOption("--clips",  cfg_.packClips, "list file: one source .usd path per line (alt to --in)");
		pack->addOption("--out,-o", cfg_.packOut,   "output combined .usd dataset")->required();
		pack->addOption("--val",    cfg_.packVal,   "Comma-separated content names for the val split");
		pack->addOption("--test",   cfg_.packTest,  "Comma-separated content names for the test split");
		pack->addFlag("--raw",      cfg_.packRaw,   "Keep source junk joints (default: strip to the clean 64-joint base)");

		auto* strip = addSubcommand("strip", "Losslessly remove junk joints (IK/weapon/COM) from source .usd clips, in place");
		strip->addOption("--in,-i",   cfg_.stripIn,      "Single input .usd to strip in place");
		strip->addOption("--dir",     cfg_.stripDir,     "root dir: strip every .usd in place (recursive)");
		strip->addOption("--exclude", cfg_.stripExclude, "Comma-separated path substrings to skip (default AimOffset)");
		strip->addFlag("--dry-run",   cfg_.stripDryRun,  "Report what would change without writing");

		auto* info = addSubcommand("info", "Inspect a combined .usd dataset");
		info->addOption("--dataset,-d", cfg_.infoPath,    "Combined .usd dataset path")->required();
		info->addOption("--context",    cfg_.infoContext, "context length used for window counts (default 32)");

		auto* bake = addSubcommand("bake", "Bake a combined .usd dataset into a C++ header");
		bake->addOption("--dataset,-d", cfg_.bakePath, "Combined .usd dataset path")->required();
		bake->addOption("--out,-o",     cfg_.bakeOut,  "output header path")->required();
		bake->addOption("--ns",         cfg_.bakeNs,   "header namespace (default walkFwdClip)");
		bake->addFlag("--root-delta",   cfg_.bakeRootDelta,
			"Bake root translation xyz as per-frame displacement (stable autoregressive root)");

		requireSubcommand(1, 1);
	}
};

int main(int argc, char** argv) {
	DsCtlCli cli;
	if (!cli.parse(argc, argv)) { return cli.helpRequested() ? 0 : 1; }

	const DsCtlConfig& c = cli.getConfig();
	const auto cmd = cli.getSubcommand();

	if (cmd == "clean") return cmdClean(c.cleanIn, c.cleanSave, c.cleanUsda, c.cleanFbx);
	if (cmd == "pack")  return cmdPack(c.packIn, c.packClips, c.packOut, c.packVal, c.packTest, c.packRaw);
	if (cmd == "strip") return cmdStrip(c.stripIn, c.stripDir, c.stripExclude, c.stripDryRun);
	if (cmd == "info")  return cmdInfo(c.infoPath, c.infoContext);
	if (cmd == "bake")  return cmdBake(c.bakePath, c.bakeOut, c.bakeNs, c.bakeRootDelta);

	std::printf("dsgen3danimctl: unknown command '%s'\n", cmd.cStr());
	return 1;
}
