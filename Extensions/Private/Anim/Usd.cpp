#include <Anim/Usd.h>

#include <cctype>
#include <cstdlib>
#include <set>
#include <sstream>
#include <string>
#include <utility>

bool OaUsdSkelClip::IsValid() const noexcept {
	const OaI32 n = JointCount();
	if (n == 0 || FrameCount == 0 || Fps <= 0.0f) {
		return false;
	}
	const OaUsize expect = static_cast<OaUsize>(FrameCount) * static_cast<OaUsize>(n);
	if (Translations.Size() != expect || Rotations.Size() != expect) {
		return false;
	}
	// Bind/rest are optional, but if present must be one per joint.
	if (BindTransforms.Size() != 0 && BindTransforms.Size() != static_cast<OaUsize>(n)) {
		return false;
	}
	if (RestTransforms.Size() != 0 && RestTransforms.Size() != static_cast<OaUsize>(n)) {
		return false;
	}
	return true;
}

// ── Writer ───────────────────────────────────────────────────────────────────

namespace {

OaString Mat4Usd(const VlmMat4& m) {
	std::ostringstream s;
	s.precision(9);
	s << "( ";
	for (int r = 0; r < 4; ++r) {
		s << "(" << m.M[r][0] << ", " << m.M[r][1] << ", " << m.M[r][2] << ", " << m.M[r][3] << ")";
		s << (r < 3 ? ", " : " ");
	}
	s << ")";
	return OaString(s.str());
}

// Comma-separated quoted joint token list shared by Skeleton + SkelAnimation.
// USD convention: the root joint should be named "root", not the empty string.
OaString JointsToken(const OaUsdSkelClip& InClip) {
	const OaI32 n = InClip.JointCount();
	std::ostringstream joints;
	for (OaI32 j = 0; j < n; ++j) {
		const OaString& path = InClip.JointPaths[static_cast<OaUsize>(j)];
		const char* jointName = (j == 0 && path.Empty()) ? "root" : path.CStr();
		joints << (j ? ", " : "") << "\"" << jointName << "\"";
	}
	return OaString(joints.str());
}

// USD prim names must be valid identifiers: [A-Za-z_][A-Za-z0-9_]*.
OaString SanitizePrimName(const OaString& InName) {
	std::string s = InName.StdStr();
	if (s.empty()) { return OaString("Anim"); }
	std::string out;
	out.reserve(s.size());
	for (char c : s) {
		out.push_back((std::isalnum(static_cast<unsigned char>(c)) || c == '_') ? c : '_');
	}
	if (std::isdigit(static_cast<unsigned char>(out[0]))) { out.insert(out.begin(), '_'); }
	return OaString(out);
}

// Emit one `def SkelAnimation "<animName>" { ... }` prim at indent `ind`. When
// `split`/`source` are non-null an `customData.oa` dictionary is written so the
// dataset partition + source name round-trip.
void EmitSkelAnimation(std::ostream& o, const char* ind, const OaString& InAnimName,
                       const OaUsdSkelClip& clip, const OaString& jointsTok,
                       const OaString* split, const OaString* source) {
	const OaI32 n = clip.JointCount();

	o << "\n" << ind << "def SkelAnimation \"" << InAnimName << "\"";
	if (split != nullptr || source != nullptr) {
		o << " (\n";
		o << ind << "    customData = {\n";
		o << ind << "        dictionary oa = {\n";
		if (split  != nullptr) { o << ind << "            string split = \""  << split->CStr()  << "\"\n"; }
		if (source != nullptr) { o << ind << "            string source = \"" << source->CStr() << "\"\n"; }
		o << ind << "        }\n";
		o << ind << "    }\n";
		o << ind << ")";
	}
	o << "\n" << ind << "{\n";
	o << ind << "    uniform token[] joints = [" << jointsTok << "]\n";

	// Static rest rotations (frame 0 convention) + per-frame timeSamples.
	o << ind << "    quatf[] rotations = [";
	for (OaI32 j = 0; j < n; ++j) {
		const VlmQuat& q = clip.Rotations[static_cast<OaUsize>(j)];
		o << (j ? ", " : "") << "(" << q.W << ", " << q.X << ", " << q.Y << ", " << q.Z << ")";
	}
	o << "]\n";
	o << ind << "    quatf[] rotations.timeSamples = {\n";
	for (OaU32 f = 0; f < clip.FrameCount; ++f) {
		o << ind << "        " << f << ": [";
		for (OaI32 j = 0; j < n; ++j) {
			const VlmQuat& q = clip.Rotations[static_cast<OaUsize>(f) * n + j];
			o << (j ? ", " : "") << "(" << q.W << ", " << q.X << ", " << q.Y << ", " << q.Z << ")";
		}
		o << "],\n";
	}
	o << ind << "    }\n";

	o << ind << "    float3[] translations = [";
	for (OaI32 j = 0; j < n; ++j) {
		const VlmVec3& t = clip.Translations[static_cast<OaUsize>(j)];
		o << (j ? ", " : "") << "(" << t.X << ", " << t.Y << ", " << t.Z << ")";
	}
	o << "]\n";
	o << ind << "    float3[] translations.timeSamples = {\n";
	for (OaU32 f = 0; f < clip.FrameCount; ++f) {
		o << ind << "        " << f << ": [";
		for (OaI32 j = 0; j < n; ++j) {
			const VlmVec3& t = clip.Translations[static_cast<OaUsize>(f) * n + j];
			o << (j ? ", " : "") << "(" << t.X << ", " << t.Y << ", " << t.Z << ")";
		}
		o << "],\n";
	}
	o << ind << "    }\n";

	o << ind << "    half3[] scales = [";
	for (OaI32 j = 0; j < n; ++j) {
		o << (j ? ", " : "") << "(1, 1, 1)";
	}
	o << "]\n";
	o << ind << "}\n";
}

// Shared stage emitter for both single- and multi-clip writers. Skeleton joints +
// bind/rest come from the first clip; one SkelAnimation prim is emitted per clip.
OaStatus WriteStage(const OaPath& InPath, OaSpan<const OaUsdNamedClip> InClips,
                    OaStringView InDefaultPrim, bool InEmitCustomData) {
	if (InClips.Empty()) {
		return OaStatus::InvalidArgument("OaUsd::WriteStage: no clips");
	}
	for (OaUsize i = 0; i < InClips.Size(); ++i) {
		if (!InClips[i].Clip.IsValid()) {
			return OaStatus::InvalidArgument("OaUsd::WriteStage: invalid clip");
		}
	}

	const OaUsdSkelClip& c0 = InClips[0].Clip;
	const OaI32 n = c0.JointCount();
	const OaString prim(InDefaultPrim);
	const char* skelPrim = "root";

	// Resolve unique, valid prim names up-front (the Skeleton's animationSource rel
	// must reference the first one).
	OaVec<OaString> animNames;
	animNames.Reserve(InClips.Size());
	{
		std::set<std::string> used;
		for (OaUsize i = 0; i < InClips.Size(); ++i) {
			OaString base = InClips[i].Name.Empty() ? OaString("Animation")
			                                        : SanitizePrimName(InClips[i].Name);
			std::string cand = base.StdStr();
			OaI32 suffix = 1;
			while (used.count(cand) != 0) {
				cand = base.StdStr() + "_" + std::to_string(suffix++);
			}
			used.insert(cand);
			animNames.PushBack(OaString(cand));
		}
	}

	OaU32 maxFrames = 0;
	for (OaUsize i = 0; i < InClips.Size(); ++i) {
		if (InClips[i].Clip.FrameCount > maxFrames) { maxFrames = InClips[i].Clip.FrameCount; }
	}

	std::ostringstream o;
	o.precision(9);

	// Stage metadata.
	o << "#usda 1.0\n(\n";
	o << "    defaultPrim = \"" << prim << "\"\n";
	o << "    endTimeCode = " << (maxFrames > 0 ? maxFrames - 1 : 0) << "\n";
	o << "    framesPerSecond = " << c0.Fps << "\n";
	o << "    metersPerUnit = 0.01\n";
	o << "    startTimeCode = 0\n";
	o << "    timeCodesPerSecond = " << c0.Fps << "\n";
	o << "    upAxis = \"" << (c0.UpAxis == 1 ? 'Y' : 'Z') << "\"\n";
	o << "    doc = \"OA Gen3dAnim squashed-USD clip\"\n";
	o << ")\n\n";

	const OaString jointsTok = JointsToken(c0);

	o << "def SkelRoot \"" << prim << "\" (\n";
	o << "    kind = \"assembly\"\n";
	o << ")\n{\n";

	// Skeleton nested under rig/root, matching Maya/UE squashed-USD exports.
	o << "    def Skeleton \"" << skelPrim << "\" (\n";
	o << "        prepend apiSchemas = [\"SkelBindingAPI\"]\n";
	o << "        customData = {\n";
	o << "            dictionary Maya = {\n";
	o << "                bool generated = 1\n";
	o << "            }\n";
	o << "        }\n";
	o << "    )\n    {\n";
	o << "        uniform token[] joints = [" << jointsTok << "]\n";
	if (c0.BindTransforms.Size() == static_cast<OaUsize>(n)) {
		o << "        uniform matrix4d[] bindTransforms = [";
		for (OaI32 j = 0; j < n; ++j) {
			o << (j ? ", " : "") << Mat4Usd(c0.BindTransforms[static_cast<OaUsize>(j)]);
		}
		o << "]\n";
	}
	if (c0.RestTransforms.Size() == static_cast<OaUsize>(n)) {
		o << "        uniform matrix4d[] restTransforms = [";
		for (OaI32 j = 0; j < n; ++j) {
			o << (j ? ", " : "") << Mat4Usd(c0.RestTransforms[static_cast<OaUsize>(j)]);
		}
		o << "]\n";
	}
	o << "        rel skel:animationSource = </" << prim << "/" << skelPrim << "/"
	  << animNames[0] << ">\n";

	// One SkelAnimation prim per clip, nested inside the Skeleton (UE layout).
	for (OaUsize i = 0; i < InClips.Size(); ++i) {
		const OaUsdNamedClip& nc = InClips[i];
		const OaString* split  = InEmitCustomData ? &nc.Split : nullptr;
		const OaString* source = InEmitCustomData ? &nc.Name  : nullptr;
		EmitSkelAnimation(o, "        ", animNames[i], nc.Clip, JointsToken(nc.Clip), split, source);
	}

	o << "    }\n";  // close Skeleton
	o << "}\n";      // close SkelRoot

	return OaFileIo::WriteText(InPath, OaString(o.str()));
}

} // namespace

OaStatus OaUsd::WriteUsda(const OaPath& InPath, const OaUsdSkelClip& InClip, OaStringView InDefaultPrim) {
	if (!InClip.IsValid()) {
		return OaStatus::InvalidArgument("OaUsd::WriteUsda: invalid clip");
	}
	OaUsdNamedClip nc;
	nc.Name = OaString("Animation");
	nc.Clip = InClip;
	const OaUsdNamedClip clips[1] = { nc };
	// Single-clip: no customData (keeps byte-for-byte compat with prior output).
	return WriteStage(InPath, OaSpan<const OaUsdNamedClip>(clips, 1), InDefaultPrim, /*customData=*/false);
}

OaStatus OaUsd::WriteUsdaMulti(const OaPath& InPath, OaSpan<const OaUsdNamedClip> InClips,
                               OaStringView InDefaultPrim) {
	return WriteStage(InPath, InClips, InDefaultPrim, /*customData=*/true);
}

// ── Reader ───────────────────────────────────────────────────────────────────

namespace {

// Find the substring inside the first open/close pair at or after `from`,
// honouring quoted strings (so brackets inside quotes are ignored). Returns the
// [innerBegin, innerEnd) bounds into `t`; false if no balanced pair is found.
bool Balanced(const std::string& t, OaUsize from, char open, char close,
              OaUsize& OutBegin, OaUsize& OutEnd) {
	const OaUsize npos = std::string::npos;
	OaUsize openPos = t.find(open, from);
	if (openPos == npos) {
		return false;
	}
	int depth = 0;
	bool inStr = false;
	for (OaUsize i = openPos; i < t.size(); ++i) {
		const char c = t[i];
		if (c == '"') {
			inStr = !inStr;
			continue;
		}
		if (inStr) {
			continue;
		}
		if (c == open) {
			++depth;
		} else if (c == close) {
			if (--depth == 0) {
				OutBegin = openPos + 1;
				OutEnd = i;
				return true;
			}
		}
	}
	return false;
}

// All numeric literals in [p, end), in order.
OaVec<OaF32> ParseFloats(const char* p, const char* end) {
	OaVec<OaF32> out;
	while (p < end) {
		const char c = *p;
		const bool numStart = (c >= '0' && c <= '9') || c == '-' || c == '+' || c == '.';
		if (!numStart) {
			++p;
			continue;
		}
		char* fin = nullptr;
		const double v = std::strtod(p, &fin);
		if (fin == p) {
			++p;
			continue;
		}
		out.PushBack(static_cast<OaF32>(v));
		p = fin;
	}
	return out;
}

// Locate the attribute then the next bracket array and parse all its floats.
bool FloatArrayAfter(const std::string& t, const char* attr, OaVec<OaF32>& Out) {
	const OaUsize npos = std::string::npos;
	const OaUsize key = t.find(attr);
	if (key == npos) {
		return false;
	}
	OaUsize b = 0, e = 0;
	if (!Balanced(t, key, '[', ']', b, e)) {
		return false;
	}
	Out = ParseFloats(t.data() + b, t.data() + e);
	return true;
}

// First quoted string after `key`, e.g. customData `string split = "train"`.
bool StringValueAfter(const std::string& t, const char* key, OaString& Out) {
	const OaUsize npos = std::string::npos;
	const OaUsize k = t.find(key);
	if (k == npos) { return false; }
	const OaUsize q1 = t.find('"', k);
	if (q1 == npos) { return false; }
	const OaUsize q2 = t.find('"', q1 + 1);
	if (q2 == npos) { return false; }
	Out = OaString(t.substr(q1 + 1, q2 - q1 - 1));
	return true;
}

// All `{...}` numeric literals after `attr` (a timeSamples block).
bool RawTimeSamples(const std::string& t, const char* attr, OaVec<OaF32>& Out) {
	const OaUsize npos = std::string::npos;
	const OaUsize key = t.find(attr);
	if (key == npos) { return false; }
	OaUsize b = 0, e = 0;
	if (!Balanced(t, key, '{', '}', b, e)) { return false; }
	Out = ParseFloats(t.data() + b, t.data() + e);
	return true;
}

// Each frame block is: frameKey then n component-tuples (translations: 1 + 3n
// floats per frame; rotations: 1 + 4n). Returns each frame's integer key and the
// key-stripped values, so rotation/translation tracks can be aligned by frame
// index — UE exports occasionally drop or add a sample on one track, leaving the
// two unequal/sparse (see the keyed alignment in ParseAnimClipFromText).
bool DeinterleaveKeyed(const OaVec<OaF32>& raw, int comps, OaI32 n,
                       OaVec<OaI32>& keys, OaVec<OaF32>& out) {
	const OaUsize perFrame = 1 + static_cast<OaUsize>(comps) * static_cast<OaUsize>(n);
	if (raw.Size() == 0 || raw.Size() % perFrame != 0) {
		return false;
	}
	const OaU32 frames = static_cast<OaU32>(raw.Size() / perFrame);
	keys.Reserve(frames);
	out.Reserve(static_cast<OaUsize>(frames) * comps * n);
	for (OaU32 f = 0; f < frames; ++f) {
		const OaUsize key = static_cast<OaUsize>(f) * perFrame;
		keys.PushBack(static_cast<OaI32>(raw[key]));
		for (OaI32 c = 0; c < comps * n; ++c) {
			out.PushBack(raw[key + 1 + static_cast<OaUsize>(c)]);
		}
	}
	return true;
}

// flat row-major 16-per-joint floats → per-joint matrices.
void MatricesFrom(const OaVec<OaF32>& flat, OaI32 n, OaVec<VlmMat4>& out) {
	if (flat.Size() != static_cast<OaUsize>(n) * 16) {
		return;
	}
	out.Resize(static_cast<OaUsize>(n));
	for (OaI32 j = 0; j < n; ++j) {
		VlmMat4 m{};
		for (int r = 0; r < 4; ++r) {
			for (int c = 0; c < 4; ++c) {
				m.M[r][c] = flat[static_cast<OaUsize>(j) * 16 + r * 4 + c];
			}
		}
		out[static_cast<OaUsize>(j)] = m;
	}
}

// Parse the joints + per-frame translations/rotations from a SkelAnimation text
// region `t` (the whole stage for a single-anim file, or one prim body for a
// multi-anim stage). Does NOT fill bind/rest — the caller applies the shared
// Skeleton transforms.
OaStatus ParseAnimClipFromText(const std::string& t, OaF32 InFps, OaI32 InUpAxis, OaUsdSkelClip& clip) {
	const OaUsize npos = std::string::npos;
	clip.Fps = InFps;
	clip.UpAxis = InUpAxis;

	// joints: first "joints =" array of quoted paths. Anchored on the '=' so the
	// data '[' is matched, not the '[' inside the "token[]" type tag.
	{
		const OaUsize key = t.find("joints =");
		if (key == npos) {
			return OaStatus::InvalidArgument("OaUsd: no joints array");
		}
		OaUsize b = 0, e = 0;
		if (!Balanced(t, key, '[', ']', b, e)) {
			return OaStatus::InvalidArgument("OaUsd: malformed joints array");
		}
		OaUsize i = b;
		while (i < e) {
			if (t[i] == '"') {
				const OaUsize q = t.find('"', i + 1);
				if (q == npos || q > e) {
					break;
				}
				clip.JointPaths.PushBack(OaString(t.substr(i + 1, q - i - 1)));
				i = q + 1;
			} else {
				++i;
			}
		}
	}
	const OaI32 n = clip.JointCount();
	if (n == 0) {
		return OaStatus::InvalidArgument("OaUsd: empty joints array");
	}

	// Rotation + translation tracks, each either time-sampled (animation) or a
	// single static array (a UE aim-offset / pose asset with no timeSamples → one
	// frame). UE exports occasionally leave the two tracks UNEQUAL or SPARSE (a
	// sample dropped/added on one), so we don't assume equal dense frames: each
	// track keeps its own frame KEYS and the clip timeline is the union of both.
	// A missing sample holds the previous value (step) — exact when a track is
	// merely shorter, harmless for a one-off dropped key.
	OaVec<OaI32> rotKeys, transKeys;
	OaVec<OaF32> rotVals, transVals;   // rotVals: frames*(4n); transVals: frames*(3n)
	bool rotStatic = false, transStatic = false;

	if (OaVec<OaF32> raw; RawTimeSamples(t, "rotations.timeSamples", raw)) {
		if (!DeinterleaveKeyed(raw, 4, n, rotKeys, rotVals)) {
			return OaStatus::InvalidArgument("OaUsd: rotations count mismatch");
		}
	} else if (FloatArrayAfter(t, "rotations =", rotVals)
	           && rotVals.Size() == static_cast<OaUsize>(n) * 4) {
		rotKeys.PushBack(0); rotStatic = true;   // single static pose
	} else {
		return OaStatus::InvalidArgument("OaUsd: no rotations (timeSamples or static)");
	}

	if (OaVec<OaF32> raw; RawTimeSamples(t, "translations.timeSamples", raw)) {
		if (!DeinterleaveKeyed(raw, 3, n, transKeys, transVals)) {
			return OaStatus::InvalidArgument("OaUsd: translations count mismatch");
		}
	} else if (FloatArrayAfter(t, "translations =", transVals)
	           && transVals.Size() == static_cast<OaUsize>(n) * 3) {
		transKeys.PushBack(0); transStatic = true;   // single static pose → broadcast
	} else {
		return OaStatus::InvalidArgument("OaUsd: no translations (timeSamples or static)");
	}

	// Clip timeline = union of the animated tracks' keys (a static track broadcasts
	// and contributes no keys of its own). Two static tracks → a single-frame pose.
	OaVec<OaI32> timeline;
	{
		std::set<OaI32> keySet;
		if (!rotStatic)   { for (OaI32 k : rotKeys)   { keySet.insert(k); } }
		if (!transStatic) { for (OaI32 k : transKeys) { keySet.insert(k); } }
		if (keySet.empty()) { keySet.insert(0); }
		for (OaI32 k : keySet) { timeline.PushBack(k); }   // std::set iterates ascending
	}

	// Index of the last key <= q (hold-previous); keys are ascending, clamp to 0.
	auto holdIdx = [](const OaVec<OaI32>& keys, OaI32 q) -> OaUsize {
		OaUsize idx = 0;
		for (OaUsize i = 0; i < keys.Size(); ++i) {
			if (keys[i] <= q) { idx = i; } else { break; }
		}
		return idx;
	};

	const OaU32 frames = static_cast<OaU32>(timeline.Size());
	clip.FrameCount = frames;
	clip.Rotations.Resize(static_cast<OaUsize>(frames) * n);
	clip.Translations.Resize(static_cast<OaUsize>(frames) * n);
	for (OaU32 f = 0; f < frames; ++f) {
		const OaI32 q = timeline[f];
		const OaUsize rB = (rotStatic   ? 0 : holdIdx(rotKeys, q))   * static_cast<OaUsize>(n) * 4;
		const OaUsize tB = (transStatic ? 0 : holdIdx(transKeys, q)) * static_cast<OaUsize>(n) * 3;
		for (OaI32 j = 0; j < n; ++j) {
			const OaUsize o  = static_cast<OaUsize>(f) * n + j;
			const OaUsize r  = rB + static_cast<OaUsize>(j) * 4;
			const OaUsize tr = tB + static_cast<OaUsize>(j) * 3;
			// USD quatf order (w, x, y, z) -> VlmQuat (x, y, z, w).
			clip.Rotations[o]    = { rotVals[r + 1], rotVals[r + 2], rotVals[r + 3], rotVals[r + 0] };
			clip.Translations[o] = { transVals[tr + 0], transVals[tr + 1], transVals[tr + 2] };
		}
	}
	return OaStatus::Ok();
}

// Stage-level fps + upAxis (defaults applied).
void ParseStageMeta(const std::string& text, OaF32& OutFps, OaI32& OutUpAxis) {
	const OaUsize npos = std::string::npos;
	OutFps = 30.0f;
	OutUpAxis = 2;
	if (OaUsize k = text.find("timeCodesPerSecond"); k != npos) {
		if (OaUsize eq = text.find('=', k); eq != npos) {
			OutFps = static_cast<OaF32>(std::strtod(text.data() + eq + 1, nullptr));
		}
	}
	if (OutFps <= 0.0f) { OutFps = 30.0f; }
	if (OaUsize k = text.find("upAxis"); k != npos) {
		if (OaUsize q = text.find('"', k); q != npos && q < k + 40) {
			OutUpAxis = (text[q + 1] == 'Y' || text[q + 1] == 'y') ? 1 : 2;
		}
	}
}

} // namespace

OaResult<OaUsdSkelClip> OaUsd::ReadUsda(const OaPath& InPath) {
	auto textResult = OaFileIo::ReadText(InPath);
	if (!textResult.IsOk()) {
		return textResult.GetStatus();
	}
	const std::string text = textResult->StdStr();

	OaF32 fps = 30.0f;
	OaI32 upAxis = 2;
	ParseStageMeta(text, fps, upAxis);

	OaUsdSkelClip clip;
	if (auto st = ParseAnimClipFromText(text, fps, upAxis, clip); st.IsError()) {
		return st;
	}
	const OaI32 n = clip.JointCount();

	// Optional bind/rest transforms (Skeleton-level): 16 floats per joint.
	if (OaVec<OaF32> f; FloatArrayAfter(text, "bindTransforms", f)) {
		MatricesFrom(f, n, clip.BindTransforms);
	}
	if (OaVec<OaF32> f; FloatArrayAfter(text, "restTransforms", f)) {
		MatricesFrom(f, n, clip.RestTransforms);
	}

	if (!clip.IsValid()) {
		return OaStatus::InvalidArgument("OaUsd::ReadUsda: assembled clip invalid");
	}
	return clip;
}

OaResult<OaVec<OaUsdNamedClip>> OaUsd::ReadUsdaMulti(const OaPath& InPath) {
	auto textResult = OaFileIo::ReadText(InPath);
	if (!textResult.IsOk()) {
		return textResult.GetStatus();
	}
	const std::string text = textResult->StdStr();
	const OaUsize npos = std::string::npos;

	OaF32 fps = 30.0f;
	OaI32 upAxis = 2;
	ParseStageMeta(text, fps, upAxis);

	// Shared Skeleton bind/rest (first occurrence — applied to every clip).
	OaVec<OaF32> bindFlat, restFlat;
	const bool hasBind = FloatArrayAfter(text, "bindTransforms", bindFlat);
	const bool hasRest = FloatArrayAfter(text, "restTransforms", restFlat);

	OaVec<OaUsdNamedClip> out;

	OaUsize pos = 0;
	while ((pos = text.find("def SkelAnimation", pos)) != npos) {
		// Prim name (first quoted token after the keyword).
		const OaUsize q1 = text.find('"', pos);
		const OaUsize q2 = (q1 == npos) ? npos : text.find('"', q1 + 1);
		OaString primName = (q1 != npos && q2 != npos)
			? OaString(text.substr(q1 + 1, q2 - q1 - 1)) : OaString("Animation");

		// Step past an optional metadata paren block `( ... )` (which itself contains
		// `{ }` for customData) so the prim BODY brace is matched, not customData.
		OaUsize headerEnd = (q2 == npos) ? pos : q2 + 1;
		while (headerEnd < text.size() && std::isspace(static_cast<unsigned char>(text[headerEnd]))) {
			++headerEnd;
		}
		OaString split  = OaString("train");
		OaString source = primName;
		if (headerEnd < text.size() && text[headerEnd] == '(') {
			OaUsize pb = 0, pe = 0;
			if (Balanced(text, headerEnd, '(', ')', pb, pe)) {
				const std::string header = text.substr(pb, pe - pb);
				StringValueAfter(header, "split", split);
				StringValueAfter(header, "source", source);
				headerEnd = pe + 1;
			}
		}

		// Prim body.
		OaUsize bb = 0, be = 0;
		if (!Balanced(text, headerEnd, '{', '}', bb, be)) {
			pos = (q2 == npos) ? pos + 17 : q2 + 1;
			continue;
		}
		const std::string block = text.substr(bb, be - bb);

		OaUsdNamedClip nc;
		nc.Name = source;
		nc.Split = split;
		if (ParseAnimClipFromText(block, fps, upAxis, nc.Clip).IsOk()) {
			const OaI32 n = nc.Clip.JointCount();
			if (hasBind) { MatricesFrom(bindFlat, n, nc.Clip.BindTransforms); }
			if (hasRest) { MatricesFrom(restFlat, n, nc.Clip.RestTransforms); }
			if (nc.Clip.IsValid()) {
				out.PushBack(std::move(nc));
			}
		}
		pos = be;
	}

	if (out.Empty()) {
		return OaStatus::InvalidArgument("OaUsd::ReadUsdaMulti: no valid SkelAnimation prims");
	}
	return out;
}
