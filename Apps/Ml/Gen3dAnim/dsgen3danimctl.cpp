// dsgen3danimctl — Gen3dAnim dataset control tool (unified data pipeline).
//
// One binary, GitHub-styled subcommands, for the whole Gen3dAnim data path:
// ingest retargeted MetaHuman USD clips, combine them into a trainable `.usd`
// dataset, inspect it, and bake it to a C++ header. All CPU-only (no Vulkan) —
// the training/generation runtime stays in traingen3danim / gen3danim.
//
// Pipeline order:  clean → strip → pack → info → bake
//
//   clean  Ingest a retargeted MetaHuman USD clip onto the clean canonical base
//          (strip Unreal IK/weapon/corrective junk, collapse hinges, Y-up→spec).
//   strip  Losslessly delete junk joint prims (IK/weapon/COM) from source clips
//          IN PLACE, keeping the kept joints' transforms bit-exact (source hygiene;
//          pack strips anyway, so this is optional). Batch via --dir.
//   pack   Combine source UsdSkel clips into ONE human-readable `.usd` dataset
//          (single SkelRoot/Skeleton + N SkelAnimation prims). Split by content.
//   info   Inspect a combined `.usd` dataset: config, per-clip split/frames/
//          category, window counts, normalization stat ranges.
//   bake   Bake a combined `.usd` dataset into a self-contained C++ header (raw
//          model-space frames + per-channel mean/std + per-clip metadata).
//
// Usage:
//   dsgen3danimctl clean  --in clip.usd [--save clip.3danim] [--usda c.usda] [--fbx c.fbx]
//   dsgen3danimctl pack   --in <usdRoot> --out walkset.usd
//   dsgen3danimctl pack   --clips list.txt --out walkset.usd --val "Walk_Left" --test "Idle"
//   dsgen3danimctl info   --dataset walkset.usd [--context 32]
//   dsgen3danimctl bake   --dataset walkset.usd --out Walk.gen.h --ns OaWalkFwdClip [--root-delta]
//
// Implements the canonical ALM dataset pipeline contract.

#include <Oa/Core/Cli.h>
#include <Oa/Core/Filesystem.h>

#include <Anim/FbxWriter.h>
#include <Anim/PoseClip.h>
#include <Anim/PosePack.h>
#include <Anim/Usd.h>
#include <Data/DsGen3dAnim.h>
#include <Rig/Skeleton.h>

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
static OaStringView LeafName(OaStringView InPath) {
	OaUsize slash = OaStringView::npos;
	for (OaUsize i = 0; i < InPath.Size(); ++i) {
		if (InPath[i] == '/') { slash = i; }
	}
	return slash == OaStringView::npos ? InPath : InPath.SubStr(slash + 1);
}

static void EnsureParent(const OaPath& p) {
	if (auto parent = p.ParentPath(); !parent.Empty()) {
		(void)OaFilesystem::CreateDirectories(parent);
	}
}

// ============================================================================
// clean — ingest a retargeted MetaHuman USD clip into the clean canonical form
// ============================================================================

static int CmdClean(const OaString& InIn, const OaString& InSave,
                    const OaString& InUsda, const OaString& InFbx) {
	if (!OaFilesystem::IsFile(OaPath(InIn))) {
		std::printf("clean: input not found: %s\n", InIn.c_str());
		return 1;
	}

	auto read = OaUsd::ReadUsda(OaPath(InIn));
	if (!read.IsOk()) {
		std::printf("clean: read failed: %s\n", read.GetStatus().ToString().c_str());
		return 1;
	}
	const OaUsdSkelClip& usd = *read;
	const OaSkeleton& sk = OaSkMetaHuman();

	// Classify the clip's joints against the clean base.
	OaI32 kept = 0, dropped = 0;
	OaString droppedNames;
	for (OaI32 u = 0; u < usd.JointCount(); ++u) {
		const OaStringView leaf = LeafName(usd.JointPaths[static_cast<OaUsize>(u)]);
		if (sk.IndexOf(leaf) >= 0) {
			++kept;
		} else {
			++dropped;
			if (dropped <= 16) { droppedNames += OaString(leaf); droppedNames += " "; }
		}
	}
	OaI32 baseMissing = 0;
	for (OaI32 s = 0; s < sk.JointCount(); ++s) {
		bool found = false;
		for (OaI32 u = 0; u < usd.JointCount() && !found; ++u) {
			found = (LeafName(usd.JointPaths[static_cast<OaUsize>(u)]) == sk.Joints[static_cast<OaUsize>(s)].Name);
		}
		if (!found) { ++baseMissing; }
	}

	std::printf("clip:    %s\n", InIn.c_str());
	std::printf("  up-axis: %s   frames: %u   fps: %.3g\n",
		usd.UpAxis == 1 ? "Y" : "Z", usd.FrameCount, usd.Fps);
	std::printf("  usd joints: %d   -> base kept: %d   junk dropped: %d   base absent: %d\n",
		usd.JointCount(), kept, dropped, baseMissing);
	std::printf("  dropped: %s\n", droppedNames.c_str());

	auto packed = OaPosePack::Pack(usd, sk);
	if (!packed.IsOk()) {
		std::printf("clean: pack failed: %s\n", packed.GetStatus().ToString().c_str());
		return 1;
	}
	const OaPoseClip& clip = *packed;
	std::printf("  packed: %u frames x %u channels (compact PoseDim)  = %llu floats\n",
		clip.FrameCount, clip.PoseDim,
		static_cast<unsigned long long>(clip.Samples.Size()));
	const OaI32 uniform = 9 + 6 * (sk.JointCount() - 1) + static_cast<OaI32>(sk.ContactJoints.Size());
	std::printf("  vs uniform-6D would be %d channels (%.0f%% smaller)\n",
		uniform, 100.0 * (1.0 - static_cast<double>(clip.PoseDim) / uniform));

	if (!InSave.empty()) {
		OaPath p(InSave); EnsureParent(p);
		if (auto st = clip.Write3dAnim(p); !st.IsOk()) {
			std::printf("  .3danim write failed: %s\n", st.ToString().c_str());
		} else {
			std::printf("  wrote %s\n", InSave.c_str());
		}
	}
	if (!InUsda.empty() || !InFbx.empty()) {
		auto back = OaPosePack::Unpack(clip, sk);
		if (!back.IsOk()) {
			std::printf("  unpack failed: %s\n", back.GetStatus().ToString().c_str());
			return 1;
		}
		if (!InUsda.empty()) {
			OaPath p(InUsda); EnsureParent(p);
			if (auto st = OaUsd::WriteUsda(p, *back, "rig"); !st.IsOk()) {
				std::printf("  .usda write failed: %s\n", st.ToString().c_str());
			} else {
				std::printf("  wrote %s\n", InUsda.c_str());
			}
		}
		if (!InFbx.empty()) {
			OaPath p(InFbx); EnsureParent(p);
			if (auto st = OaFbx::WriteFbx(p, *back); !st.IsOk()) {
				std::printf("  .fbx write failed: %s\n", st.ToString().c_str());
			} else {
				std::printf("  wrote %s\n", InFbx.c_str());
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
// joints. The same leaf-name match OaPosePack::Pack uses at load (junk is ignored
// there too); doing it here keeps the dataset .usd file itself clean + viewable.
static OaUsdSkelClip StripToBase(const OaUsdSkelClip& InClip, const OaSkeleton& InSkel, OaI32& OutDropped) {
	const OaI32 nIn = InClip.JointCount();
	OaVec<OaI32> keep;
	for (OaI32 u = 0; u < nIn; ++u) {
		if (InSkel.IndexOf(LeafName(InClip.JointPaths[static_cast<OaUsize>(u)])) >= 0) {
			keep.PushBack(u);
		}
	}
	OutDropped = nIn - static_cast<OaI32>(keep.Size());

	OaUsdSkelClip out;
	out.FrameCount = InClip.FrameCount;
	out.Fps        = InClip.Fps;
	out.UpAxis     = InClip.UpAxis;
	const OaUsize nOut = keep.Size();
	out.JointPaths.Reserve(nOut);
	for (OaI32 u : keep) { out.JointPaths.PushBack(InClip.JointPaths[static_cast<OaUsize>(u)]); }
	if (InClip.BindTransforms.Size() == static_cast<OaUsize>(nIn)) {
		out.BindTransforms.Reserve(nOut);
		for (OaI32 u : keep) { out.BindTransforms.PushBack(InClip.BindTransforms[static_cast<OaUsize>(u)]); }
	}
	if (InClip.RestTransforms.Size() == static_cast<OaUsize>(nIn)) {
		out.RestTransforms.Reserve(nOut);
		for (OaI32 u : keep) { out.RestTransforms.PushBack(InClip.RestTransforms[static_cast<OaUsize>(u)]); }
	}
	out.Translations.Reserve(static_cast<OaUsize>(out.FrameCount) * nOut);
	out.Rotations.Reserve(static_cast<OaUsize>(out.FrameCount) * nOut);
	for (OaU32 f = 0; f < InClip.FrameCount; ++f) {
		const OaUsize base = static_cast<OaUsize>(f) * static_cast<OaUsize>(nIn);
		for (OaI32 u : keep) {
			out.Translations.PushBack(InClip.Translations[base + static_cast<OaUsize>(u)]);
			out.Rotations.PushBack(InClip.Rotations[base + static_cast<OaUsize>(u)]);
		}
	}
	return out;
}

static std::set<std::string> ParseList(const OaString& InCsv) {
	std::set<std::string> out;
	std::string cur;
	for (OaUsize i = 0; i < InCsv.Size(); ++i) {
		const char ch = InCsv[i];
		if (ch == ',') { if (!cur.empty()) { out.insert(cur); cur.clear(); } }
		else if (ch != ' ') { cur.push_back(ch); }
	}
	if (!cur.empty()) { out.insert(cur); }
	return out;
}

// "…/MM_Unarmed_Walk_Fwd.usd" → stem "MM_Unarmed_Walk_Fwd".
static std::string Stem(const OaString& InPath) {
	std::string p = InPath.StdStr();
	std::size_t slash = p.find_last_of("/\\");
	std::string base = (slash == std::string::npos) ? p : p.substr(slash + 1);
	std::size_t dot = base.find_last_of('.');
	return (dot == std::string::npos) ? base : base.substr(0, dot);
}

// "MTN_N_Idle_E" → content "N_Idle_E" (strip a [MF]T[NOU]_ body-variant prefix);
// names without that prefix (e.g. "MM_Unarmed_Walk_Fwd") pass through unchanged.
static std::string ContentOf(const std::string& InStem) {
	if (InStem.size() > 4 && (InStem[0] == 'M' || InStem[0] == 'F') &&
	    InStem[1] == 'T' && InStem[3] == '_') {
		return InStem.substr(4);
	}
	return InStem;
}

// Read a --clips list file into source paths (skips blanks and '#' comments).
static OaVec<OaString> ReadClipList(const OaString& InPath) {
	OaVec<OaString> out;
	auto text = OaFilesystem::ReadText(OaPath(InPath));
	if (!text.IsOk()) { return out; }
	std::string s = text->StdStr();
	std::string line;
	auto flush = [&]() {
		std::size_t a = line.find_first_not_of(" \t\r\n");
		std::size_t b = line.find_last_not_of(" \t\r\n");
		if (a != std::string::npos && line[a] != '#') {
			out.PushBack(OaString(line.substr(a, b - a + 1).c_str()));
		}
		line.clear();
	};
	for (char c : s) { if (c == '\n') { flush(); } else { line.push_back(c); } }
	flush();
	return out;
}

static int CmdPack(const OaString& InIn, const OaString& InClips, const OaString& InOut,
                   const OaString& InVal, const OaString& InTest, bool InRaw) {
	// Gather source .usd paths from --clips list or --in folder walk.
	OaVec<OaString> sources;
	if (!InClips.empty()) {
		sources = ReadClipList(InClips);
		if (sources.Empty()) {
			std::printf("pack: empty/unreadable clip list: %s\n", InClips.c_str());
			return 1;
		}
	} else if (!InIn.empty()) {
		auto listing = OaFilesystem::ListAll(OaPath(InIn), /*recursive=*/true);
		if (!listing.IsOk()) {
			std::printf("pack: cannot list %s: %s\n", InIn.c_str(),
				listing.GetStatus().ToString().c_str());
			return 1;
		}
		for (const OaPath& p : *listing) {
			const std::string path = p.String().StdStr();
			if (path.size() >= 4 && path.substr(path.size() - 4) == ".usd") {
				sources.PushBack(p.String());
			}
		}
	} else {
		std::printf("pack: provide --in <dir> or --clips <list.txt>\n");
		return 1;
	}

	const std::set<std::string> valSet  = ParseList(InVal);
	const std::set<std::string> testSet = ParseList(InTest);
	const OaSkeleton& sk = OaSkMetaHuman();

	OaVec<OaUsdNamedClip> clips;
	OaI32 nTrain = 0, nVal = 0, nTest = 0, nSkip = 0, nDropTotal = 0;
	for (const OaString& src : sources) {
		auto read = OaUsd::ReadUsda(OaPath(src));
		if (!read.IsOk()) {
			std::printf("  skip %s (read: %s)\n", src.c_str(), read.GetStatus().ToString().c_str());
			++nSkip; continue;
		}
		const std::string stem    = Stem(src);
		const std::string content = ContentOf(stem);

		OaUsdNamedClip nc;
		nc.Name = OaString(stem.c_str());
		if (InRaw) {
			nc.Clip = std::move(read.GetValue());
		} else {
			OaI32 dropped = 0;
			nc.Clip = StripToBase(read.GetValue(), sk, dropped);
			nDropTotal += dropped;
		}
		if      (valSet.count(content))  { nc.Split = OaString("val");  ++nVal; }
		else if (testSet.count(content)) { nc.Split = OaString("test"); ++nTest; }
		else                             { nc.Split = OaString("train"); ++nTrain; }
		clips.PushBack(std::move(nc));
	}

	if (clips.Empty()) {
		std::printf("pack: no usable .usd clips from input\n");
		return 1;
	}

	std::printf("combined %llu clips (train %d / val %d / test %d, skipped %d) · joints %s\n",
		static_cast<unsigned long long>(clips.Size()), nTrain, nVal, nTest, nSkip,
		InRaw ? "raw (junk kept)" : "stripped to clean base");
	if (!InRaw) {
		std::printf("  junk joints dropped: %d total\n", nDropTotal);
	}

	if (auto st = OaUsd::WriteUsdaMulti(OaPath(InOut),
			OaSpan<const OaUsdNamedClip>(clips.Data(), clips.Size())); !st.IsOk()) {
		std::printf("pack: write failed: %s\n", st.ToString().c_str());
		return 1;
	}
	std::printf("wrote %s\n", InOut.c_str());
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
static int CmdStrip(const OaString& InIn, const OaString& InDir,
                    const OaString& InExclude, bool InDryRun) {
	const OaSkeleton& sk = OaSkMetaHuman();
	const std::set<std::string> excl = ParseList(InExclude);

	// Gather targets: a single --in file, or every .usd under --dir (recursive),
	// minus any path containing an --exclude substring (AimOffset by default).
	OaVec<OaString> files;
	if (!InDir.empty()) {
		auto listing = OaFilesystem::ListAll(OaPath(InDir), /*recursive=*/true);
		if (!listing.IsOk()) {
			std::printf("strip: cannot list %s: %s\n", InDir.c_str(),
				listing.GetStatus().ToString().c_str());
			return 1;
		}
		for (const OaPath& p : *listing) {
			const std::string path = p.String().StdStr();
			if (path.size() < 4 || path.substr(path.size() - 4) != ".usd") { continue; }
			bool skip = false;
			for (const std::string& e : excl) { if (path.find(e) != std::string::npos) { skip = true; break; } }
			if (skip) { continue; }
			files.PushBack(p.String());
		}
	} else if (!InIn.empty()) {
		files.PushBack(InIn);
	} else {
		std::printf("strip: provide --in <file> or --dir <root>\n");
		return 1;
	}
	if (files.Empty()) { std::printf("strip: no .usd files to process\n"); return 1; }

	OaI32 nDone = 0, nSkip = 0, nDrop = 0, nIncomplete = 0;
	for (const OaString& src : files) {
		auto read = OaUsd::ReadUsda(OaPath(src));
		if (!read.IsOk()) {
			std::printf("  skip %s (read: %s)\n", src.c_str(), read.GetStatus().ToString().c_str());
			++nSkip; continue;
		}
		OaI32 dropped = 0;
		OaUsdSkelClip stripped = StripToBase(read.GetValue(), sk, dropped);
		const OaI32 kept = stripped.JointCount();
		const bool incomplete = (kept != sk.JointCount());
		if (incomplete) { ++nIncomplete; }
		nDrop += dropped;
		if (InDryRun) {
			std::printf("  [dry] kept %d  drop %d%s  %s\n", kept, dropped,
				incomplete ? "  (INCOMPLETE base)" : "", src.c_str());
			++nDone; continue;
		}
		if (auto st = OaUsd::WriteUsda(OaPath(src), stripped, "rig"); !st.IsOk()) {
			std::printf("  FAIL %s (write: %s)\n", src.c_str(), st.ToString().c_str());
			++nSkip; continue;
		}
		++nDone;
	}
	std::printf("strip: %s %d files · junk dropped %d total · incomplete-base %d · skipped %d\n",
		InDryRun ? "would process" : "rewrote", nDone, nDrop, nIncomplete, nSkip);
	return 0;
}

// ============================================================================
// info — inspect a combined `.usd` dataset
// ============================================================================

static int CmdInfo(const OaString& InPath, OaI32 InContext) {
	OaDsGen3dAnim ds;
	if (auto st = ds.Load(OaPath(InPath)); !st.IsOk()) {
		std::printf("info: load failed: %s\n", st.ToString().c_str());
		return 1;
	}
	ds.PrepareWindows(InContext);

	std::printf("\n  Dataset: %s\n", InPath.c_str());
	std::printf("  PoseDim: %d   Fps: %.3g   Clips: %zu\n",
		ds.PoseDim(), static_cast<double>(ds.Fps()), ds.ClipCount());
	std::printf("  Split clips:   train=%zu  val=%zu  test=%zu\n",
		ds.SplitClipCount(OaDsSplit::Train),
		ds.SplitClipCount(OaDsSplit::Val),
		ds.SplitClipCount(OaDsSplit::Test));
	std::printf("  Windows@ctx%d:  train=%zu  val=%zu  test=%zu\n", InContext,
		ds.WindowCount(OaDsSplit::Train),
		ds.WindowCount(OaDsSplit::Val),
		ds.WindowCount(OaDsSplit::Test));

	// Normalization sanity: the ranges should be finite (no NaN/Inf), std >= 0.
	const OaVec<OaF32>& mean = ds.Mean();
	const OaVec<OaF32>& sd   = ds.Std();
	if (!mean.Empty() && !sd.Empty()) {
		OaF32 mmin = mean[0], mmax = mean[0], smin = sd[0], smax = sd[0];
		for (OaUsize i = 0; i < mean.Size(); ++i) {
			mmin = std::min(mmin, mean[i]); mmax = std::max(mmax, mean[i]);
			smin = std::min(smin, sd[i]);   smax = std::max(smax, sd[i]);
		}
		std::printf("  Mean range:    [%.4g, %.4g]\n", static_cast<double>(mmin), static_cast<double>(mmax));
		std::printf("  Std range:     [%.4g, %.4g]\n", static_cast<double>(smin), static_cast<double>(smax));
	}

	std::printf("\n  %-5s %-34s %-7s %-7s %s\n", "Idx", "Name", "Split", "Frames", "Category");
	const OaVec<OaDsClipMeta>& metas = ds.Metas();
	for (OaUsize i = 0; i < metas.Size(); ++i) {
		const OaDsClipMeta& m = metas[i];
		const char* sp = (m.Split == OaDsSplit::Train) ? "train"
		               : (m.Split == OaDsSplit::Val)   ? "val" : "test";
		std::printf("  %-5zu %-34s %-7s %-7u %s\n",
			i, m.Name.c_str(), sp, m.Frames, OaDsCategoryName(m.Category));
	}
	std::printf("\n");
	return 0;
}

// ============================================================================
// bake — bake a combined `.usd` dataset into a C++ header
// ============================================================================

// Emit a valid C++ float literal. "%.7g" alone drops the decimal point for whole
// numbers ("30" → invalid "30f"), so append ".0" when there's no '.'/'e'/'E'.
static void EmitFloat(FILE* f, double v) {
	char buf[64];
	std::snprintf(buf, sizeof(buf), "%.7g", v);
	const bool isFloatLit = std::strpbrk(buf, ".eE") != nullptr;
	std::fprintf(f, "%s%sf,", buf, isFloatLit ? "" : ".0");
}

static int CmdBake(const OaString& InPath, const OaString& InOut,
                   const OaString& InNs, bool InRootDelta) {
	OaDsGen3dAnim ds;
	if (auto st = ds.Load(OaPath(InPath)); !st.IsOk()) {
		std::printf("bake: load failed: %s\n", st.ToString().c_str());
		return 1;
	}
	// Root as per-frame delta: recomputes Mean/Std and makes ClipModelRaw emit
	// displacement for channels 0-2. Small, well-conditioned root → stable rollout.
	ds.SetRootTranslationDelta(InRootDelta);

	const OaI32 poseDim = ds.PoseDim();
	const OaUsize clipCount = ds.ClipCount();
	const OaVec<OaDsClipMeta>& metas = ds.Metas();
	const OaVec<OaF32>& mean = ds.Mean();
	const OaVec<OaF32>& sd   = ds.Std();

	// Gather every clip's raw model-space frames (un-standardized — the tutorial
	// standardizes on the fly with the baked Mean/Std, exactly like NextBatch).
	OaVec<OaVec<OaF32>> clipData;
	OaVec<OaU32>        clipFrames;
	OaU32 frameTotal = 0;
	for (OaUsize i = 0; i < clipCount; ++i) {
		OaVec<OaF32> raw; OaU32 frames = 0;
		if (!ds.ClipModelRaw(static_cast<OaI32>(i), raw, frames) || frames == 0) {
			std::printf("bake: clip %zu has no frames\n", i);
			return 1;
		}
		frameTotal += frames;
		clipData.PushBack(std::move(raw));
		clipFrames.PushBack(frames);
	}

	FILE* f = std::fopen(InOut.c_str(), "wb");
	if (!f) {
		std::printf("bake: cannot open %s for write\n", InOut.c_str());
		return 1;
	}

	std::fprintf(f, "#pragma once\n");
	std::fprintf(f, "// AUTO-GENERATED by dsgen3danimctl bake — DO NOT EDIT.\n");
	std::fprintf(f, "// Source: %s\n", InPath.c_str());
	std::fprintf(f, "// Real MetaHuman ChannelSpec motion (PoseDim=%d) baked for TutorialGen3dAnim.\n\n",
		poseDim);
	std::fprintf(f, "namespace %s {\n\n", InNs.c_str());

	std::fprintf(f, "inline constexpr int   PoseDim    = %d;\n", poseDim);
	{
		// Scalar float literal (semicolon, not the array comma EmitFloat appends).
		char fpsBuf[64];
		std::snprintf(fpsBuf, sizeof(fpsBuf), "%.7g", static_cast<double>(ds.Fps()));
		const bool isLit = std::strpbrk(fpsBuf, ".eE") != nullptr;
		std::fprintf(f, "inline constexpr float Fps        = %s%sf;\n", fpsBuf, isLit ? "" : ".0");
	}
	std::fprintf(f, "inline constexpr int   ClipCount  = %zu;\n", clipCount);
	std::fprintf(f, "inline constexpr int   FrameTotal = %u;\n", frameTotal);
	// When true, channels 0-2 (root xyz) are per-frame displacement, not absolute
	// position. The consumer must cumulatively integrate them to recover world pos.
	std::fprintf(f, "inline constexpr bool  RootTranslationDelta = %s;\n\n",
		InRootDelta ? "true" : "false");

	// Per-clip metadata.
	std::fprintf(f, "inline constexpr int ClipFrames[ClipCount] = { ");
	for (OaUsize i = 0; i < clipCount; ++i) std::fprintf(f, "%u%s", clipFrames[i], i + 1 < clipCount ? ", " : " ");
	std::fprintf(f, "};\n");

	std::fprintf(f, "inline constexpr int ClipOffset[ClipCount] = { ");
	{ OaU32 off = 0; for (OaUsize i = 0; i < clipCount; ++i) { std::fprintf(f, "%u%s", off, i + 1 < clipCount ? ", " : " "); off += clipFrames[i]; } }
	std::fprintf(f, "};  // frame offset into Poses\n");

	std::fprintf(f, "inline constexpr int ClipSplit[ClipCount]  = { ");
	for (OaUsize i = 0; i < clipCount; ++i) std::fprintf(f, "%d%s", static_cast<int>(metas[i].Split), i + 1 < clipCount ? ", " : " ");
	std::fprintf(f, "};  // 0=train 1=val 2=test\n");

	std::fprintf(f, "inline constexpr const char* ClipNames[ClipCount] = { ");
	for (OaUsize i = 0; i < clipCount; ++i) std::fprintf(f, "\"%s\"%s", metas[i].Name.c_str(), i + 1 < clipCount ? ", " : " ");
	std::fprintf(f, "};\n\n");

	// Per-channel normalization stats.
	std::fprintf(f, "inline constexpr float Mean[PoseDim] = {\n");
	for (OaI32 ch = 0; ch < poseDim; ++ch) { EmitFloat(f, static_cast<double>(mean[static_cast<OaUsize>(ch)])); if ((ch % 8 == 7)) std::fprintf(f, "\n"); }
	std::fprintf(f, "};\n");
	std::fprintf(f, "inline constexpr float Std[PoseDim] = {\n");
	for (OaI32 ch = 0; ch < poseDim; ++ch) { EmitFloat(f, static_cast<double>(sd[static_cast<OaUsize>(ch)])); if ((ch % 8 == 7)) std::fprintf(f, "\n"); }
	std::fprintf(f, "};\n\n");

	// Raw model-space frames, clip-major: Poses[(ClipOffset[c]+f)*PoseDim + ch].
	// `inline const` (not constexpr) keeps the large initializer fast to compile.
	std::fprintf(f, "// Raw (un-standardized) model-space pose frames, frame-major.\n");
	std::fprintf(f, "inline const float Poses[FrameTotal * PoseDim] = {\n");
	for (OaUsize i = 0; i < clipCount; ++i) {
		const OaVec<OaF32>& raw = clipData[i];
		std::fprintf(f, "// --- clip %zu: %s (%u frames) ---\n", i, metas[i].Name.c_str(), clipFrames[i]);
		for (OaUsize k = 0; k < raw.Size(); ++k) {
			EmitFloat(f, static_cast<double>(raw[k]));
			if (k % static_cast<OaUsize>(poseDim) == static_cast<OaUsize>(poseDim) - 1) std::fprintf(f, "\n");
		}
	}
	std::fprintf(f, "};\n\n");
	std::fprintf(f, "}  // namespace %s\n", InNs.c_str());

	std::fclose(f);

	std::printf("bake: wrote %s\n  PoseDim=%d Fps=%.3g Clips=%zu FrameTotal=%u  (%.1f KB of floats)\n",
		InOut.c_str(), poseDim, static_cast<double>(ds.Fps()), clipCount, frameTotal,
		static_cast<double>(frameTotal) * poseDim * 4.0 / 1024.0);
	return 0;
}

// ============================================================================
// CLI
// ============================================================================

struct DsCtlConfig {
	// clean
	OaString CleanIn;
	OaString CleanSave;
	OaString CleanUsda;
	OaString CleanFbx;
	// pack
	OaString PackIn;
	OaString PackClips;
	OaString PackOut;
	OaString PackVal;
	OaString PackTest;
	bool     PackRaw = false;
	// strip
	OaString StripIn;
	OaString StripDir;
	OaString StripExclude = "AimOffset";
	bool     StripDryRun  = false;
	// info
	OaString InfoPath;
	OaI32    InfoContext = 32;
	// bake
	OaString BakePath;
	OaString BakeOut;
	OaString BakeNs = "OaWalkFwdClip";
	bool     BakeRootDelta = false;
};

class DsCtlCli : public OaCli<DsCtlConfig> {
public:
	DsCtlCli() : OaCli("dsgen3danimctl", "Gen3dAnim dataset control tool (clean/pack/info/bake)") {
		SetEpilog(
			"Pipeline: clean -> pack -> info -> bake\n"
			"\n"
			"Examples:\n"
			"  dsgen3danimctl clean --in clip.usd --usda clean.usda\n"
			"  dsgen3danimctl pack  --in usdRoot/ --out walkset.usd\n"
			"  dsgen3danimctl pack  --clips list.txt --out walkset.usd --val \"Walk_Left\" --test \"Idle\"\n"
			"  dsgen3danimctl info  --dataset walkset.usd --context 32\n"
			"  dsgen3danimctl bake  --dataset walkset.usd --out Walk.gen.h --ns OaWalkFwdClip --root-delta\n"
		);

		auto* clean = AddSubcommand("clean", "Ingest a retargeted MetaHuman USD clip into the clean canonical form");
		clean->add_option("--in,-i", Cfg_.CleanIn,   "Input .usda clip (retargeted MetaHuman)")->required();
		clean->add_option("--save",  Cfg_.CleanSave, "Write canonical .3danim");
		clean->add_option("--usda",  Cfg_.CleanUsda, "Write round-tripped .usda (clean base only)");
		clean->add_option("--fbx",   Cfg_.CleanFbx,  "Write FBX (DCC review)");

		auto* pack = AddSubcommand("pack", "Combine source UsdSkel clips into one .usd dataset");
		pack->add_option("--in,-i",  Cfg_.PackIn,    "Root dir of source .usd clips (recursive)");
		pack->add_option("--clips",  Cfg_.PackClips, "List file: one source .usd path per line (alt to --in)");
		pack->add_option("--out,-o", Cfg_.PackOut,   "Output combined .usd dataset")->required();
		pack->add_option("--val",    Cfg_.PackVal,   "Comma-separated content names for the val split");
		pack->add_option("--test",   Cfg_.PackTest,  "Comma-separated content names for the test split");
		pack->add_flag("--raw",      Cfg_.PackRaw,   "Keep source junk joints (default: strip to the clean 64-joint base)");

		auto* strip = AddSubcommand("strip", "Losslessly remove junk joints (IK/weapon/COM) from source .usd clips, in place");
		strip->add_option("--in,-i",   Cfg_.StripIn,      "Single input .usd to strip in place");
		strip->add_option("--dir",     Cfg_.StripDir,     "Root dir: strip every .usd in place (recursive)");
		strip->add_option("--exclude", Cfg_.StripExclude, "Comma-separated path substrings to skip (default AimOffset)");
		strip->add_flag("--dry-run",   Cfg_.StripDryRun,  "Report what would change without writing");

		auto* info = AddSubcommand("info", "Inspect a combined .usd dataset");
		info->add_option("--dataset,-d", Cfg_.InfoPath,    "Combined .usd dataset path")->required();
		info->add_option("--context",    Cfg_.InfoContext, "Context length used for window counts (default 32)");

		auto* bake = AddSubcommand("bake", "Bake a combined .usd dataset into a C++ header");
		bake->add_option("--dataset,-d", Cfg_.BakePath, "Combined .usd dataset path")->required();
		bake->add_option("--out,-o",     Cfg_.BakeOut,  "Output header path")->required();
		bake->add_option("--ns",         Cfg_.BakeNs,   "Header namespace (default OaWalkFwdClip)");
		bake->add_flag("--root-delta",   Cfg_.BakeRootDelta,
			"Bake root translation xyz as per-frame displacement (stable autoregressive root)");

		RequireSubcommand(1, 1);
	}
};

int main(int argc, char** argv) {
	DsCtlCli cli;
	if (!cli.Parse(argc, argv)) { return 1; }

	const DsCtlConfig& c = cli.GetConfig();
	const auto cmd = cli.GetSubcommand();

	if (cmd == "clean") return CmdClean(c.CleanIn, c.CleanSave, c.CleanUsda, c.CleanFbx);
	if (cmd == "pack")  return CmdPack(c.PackIn, c.PackClips, c.PackOut, c.PackVal, c.PackTest, c.PackRaw);
	if (cmd == "strip") return CmdStrip(c.StripIn, c.StripDir, c.StripExclude, c.StripDryRun);
	if (cmd == "info")  return CmdInfo(c.InfoPath, c.InfoContext);
	if (cmd == "bake")  return CmdBake(c.BakePath, c.BakeOut, c.BakeNs, c.BakeRootDelta);

	std::printf("dsgen3danimctl: unknown command '%s'\n", cmd.c_str());
	return 1;
}
